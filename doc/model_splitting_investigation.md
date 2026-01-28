# Model Splitting Investigation: Reducing Compiler Pressure from Large VA Models

## Executive Summary

Large Verilog-A models like PSP103VA (782 parameters, ~96K IR statements) cause severe compilation pressure due to nonlinear complexity in Julia/LLVM compilation. This document investigates OpenVAF's approach to model splitting and proposes a conservative, simple method to split Cadnip.jl's stamp functions into static setup code and dynamic evaluation code.

**Recommendation**: A two-phase approach that separates **parameter-dependent computation** (runs once per device setup) from **voltage-dependent computation** (runs every Newton iteration), implemented through function decomposition in codegen.

## Part 1: OpenVAF's Code Generation Architecture

### Key Files Analyzed
- `openvaf/osdi/src/lib.rs` - Compilation orchestration
- `openvaf/osdi/src/setup.rs` - Static setup code generation
- `openvaf/osdi/src/eval.rs` - Dynamic evaluation code generation
- `openvaf/sim_back/src/init.rs` - Operating point independence analysis
- `openvaf/sim_back/src/context.rs` - Taint propagation for dependency analysis

### OpenVAF's Function Separation

OpenVAF generates **4 separate functions** per model, compiled in parallel:

| Function | Purpose | When Called | Depends On |
|----------|---------|-------------|------------|
| `access` | Access functions | Once per model | Nothing |
| `setup_model` | Model parameter initialization | Once per model | Parameters only |
| `setup_instance` | Instance initialization | Once per instance | Parameters + per-instance params |
| `eval` | Evaluation at operating point | Every Newton iteration | Parameters + voltages/currents |

### The Core Insight: Operating Point Dependence

OpenVAF's key innovation is **taint propagation** to identify operating point (OP) dependent code:

```rust
// From context.rs - identifies OP-dependent parameters
for (param, &val) in self.intern.params.iter() {
    if !dfg.value_dead(val) && param.op_dependent() {
        self.op_dependent_vals.push(val)
    }
}

// Propagate taint: any instruction using OP-dependent value is also OP-dependent
propagate_taint(&self.func, &self.dom_tree, &self.cfg,
                self.op_dependent_vals.iter().copied(),
                &mut self.op_dependent_insts)
```

**OP-dependent values** include:
- Node voltages (`V(p,n)`)
- Branch currents (`I(br)`)
- Time (`$abstime`)
- Temperature (when it varies)

**OP-independent values** include:
- Device parameters (`dev.R`, `dev.Is`, etc.)
- Model parameters
- Computed constants from parameters

### How `setup_instance` Works (init.rs)

The key logic in `Initialization::new()`:

```rust
fn split_block(&mut self, bb: Block) {
    for inst in block_instructions {
        if self.op_dependent_insts.contains(inst) {
            // Don't copy to init function
            // Only copy terminators for control flow
        } else {
            self.copy_instruction(inst, bb)  // Copy to init function
        }
    }
}
```

Values computed in init but needed in eval are stored in a **cache**:
- `cached_vals: IndexMap<Value, CacheSlot>` - maps values to cache slots
- eval function loads these as parameters

## Part 2: Cadnip.jl's Current Architecture

### Current stamp! Function Structure

From `vasim.jl`, the generated `stamp!` function contains:

```julia
function stamp!(dev::DeviceName, ctx::AnyMNAContext, nodes...;
                t::Real=0.0, x::AbstractVector=ZERO_VECTOR, ...)

    # Phase 1: Parameter extraction [STATIC]
    param1 = undefault(dev.param1)
    param2 = undefault(dev.param2)
    # ... (can be 782 parameters for PSP103VA)

    # Phase 2: Local variable initialization [MOSTLY STATIC]
    local_var_1 = 0.0
    local_var_2 = compute_from_params(param1, param2)
    # ... (hundreds of intermediate variables)

    # Phase 3: Internal node allocation [STATIC]
    _node_GP = alloc_internal_node!(ctx, :PSP103VA_GP, instance)
    # ...

    # Phase 4: Voltage extraction [DYNAMIC]
    V_1 = x[_node_p]
    V_2 = x[_node_n]
    # ...

    # Phase 5: Dual creation for AD [DYNAMIC]
    p = Dual{JacobianTag}(V_1, 1.0, 0.0, ...)
    n = Dual{JacobianTag}(V_2, 0.0, 1.0, ...)

    # Phase 6: Contribution evaluation [DYNAMIC - VA analog block]
    # This is the translated VA analog block - thousands of statements
    I_ds = complex_function_of_voltages_and_params(p, n, ...)

    # Phase 7: Stamp into matrices [DYNAMIC]
    stamp_G!(ctx, p, n, dI_dV)
    stamp_b!(ctx, p, -Ieq)
    # ... (unrolled for each branch)
end
```

### Why 90K+ Statements?

For PSP103VA:
1. **Parameter extraction**: 782 parameters × ~2 statements = ~1,500 statements
2. **Local variable computation**: Hundreds of model equations = ~30,000+ statements
3. **VA analog block**: Complex physics = ~40,000+ statements
4. **Unrolled stamping**: Per-branch stamping × many branches = ~20,000+ statements

### Existing Separation: MNAContext vs DirectStampContext

Cadnip.jl already has a two-context pattern:

| Context | Purpose | Allocation |
|---------|---------|------------|
| `MNAContext` | Structure discovery (COO indices) | Creates vectors |
| `DirectStampContext` | Value-only updates | Zero allocation |

But **both paths execute the same giant stamp! function**. The separation helps runtime, not compile time.

## Part 3: Analysis of Splitting Opportunities

### What's Actually Static vs Dynamic in Cadnip.jl?

**Truly Static (parameter-only, no x dependence)**:
- Parameter extraction from device struct
- Computed constants from parameters (e.g., `Vt = kT/q`)
- Model scaling factors
- Geometry-derived quantities

**Dynamic (depends on solution vector x)**:
- Voltage extraction from x
- VA analog block evaluation
- Jacobian extraction
- Matrix stamping

### Key Observation: The Split Point is Voltage Extraction

```julia
# Everything BEFORE this is static:
V_1 = x[_node_p]  # <-- SPLIT POINT

# Everything AFTER this is dynamic
```

## Part 4: Proposed Splitting Approach

### Recommended: Function Decomposition in Codegen

**Principle**: Split the generated stamp! into multiple `@noinline` functions at the codegen level.

```julia
# Generated code structure (simplified)

# STATIC: Parameter-to-model computation
@noinline function _stamp_model_setup!(dev::DeviceName, ctx::AnyMNAContext, ...)
    # Extract parameters
    param1 = undefault(dev.param1)
    # ...

    # Compute parameter-derived constants
    Vt = kT_q * temperature
    Is_eff = param_Is * (1 + param_dIs * (temperature - 300))
    # ... (all parameter-only computation)

    # Return computed values as NamedTuple for passing to eval
    return (Vt=Vt, Is_eff=Is_eff, ...)
end

# DYNAMIC: Voltage-dependent evaluation
@noinline function _stamp_eval!(model::NamedTuple, ctx::AnyMNAContext, nodes...; x, t)
    # Extract voltages
    V_1 = x[nodes[1]]
    # ...

    # Create duals
    p = Dual{JacobianTag}(V_1, ...)

    # Evaluate contributions using precomputed model values
    I_ds = model.Is_eff * (exp((p - n) / model.Vt) - 1)
    # ...

    # Stamp matrices
    stamp_G!(ctx, ...)
end

# Main entry point - thin coordinator
function stamp!(dev::DeviceName, ctx::AnyMNAContext, nodes...; x, t, ...)
    model = _stamp_model_setup!(dev, ctx, ...)
    _stamp_eval!(model, ctx, nodes...; x=x, t=t)
end
```

### Why This Works

1. **Compiler sees smaller functions**: Each function is ~30-40K statements instead of 90K
2. **Nonlinear complexity reduction**: Compilation complexity often scales as O(n²) or worse
3. **@noinline boundaries**: Prevents inlining that would recreate the giant function
4. **Minimal architectural change**: Same stamp! signature, just internal decomposition
5. **Cache-friendly**: Model values computed once, reused across Newton iterations

### Implementation in vasim.jl

Modify `generate_mna_stamp_method_nterm()` to:

1. **Analyze parameter dependence**: Walk the AST to identify which local variables depend only on parameters
2. **Split local_var_init**: Separate into parameter-only and voltage-dependent
3. **Generate two functions**: `_stamp_model_setup!` and `_stamp_eval!`
4. **Create coordinator**: Main stamp! calls both

### Conservative First Step

Before full AST analysis, a simpler approach:

```julia
# Split at a fixed point: parameter extraction vs everything else

@noinline function _extract_params(dev::DeviceName)
    # Just parameter extraction
    return (
        param1 = undefault(dev.param1),
        param2 = undefault(dev.param2),
        # ...
    )
end

function stamp!(dev::DeviceName, ctx::AnyMNAContext, nodes...; x, t, ...)
    params = _extract_params(dev)
    # ... rest of stamp! using params.param1, params.param2, etc.
end
```

This alone might reduce IR size by 20-30% and improve inlining decisions.

## Part 5: Alternative Approaches Considered

### Alternative A: Cache Model Computation Results

Like OpenVAF's init function, compute parameter-dependent values once and cache:

```julia
struct CachedModel
    Vt::Float64
    Is_eff::Float64
    # ... all parameter-derived values
end

mutable struct DeviceState
    cached::Union{Nothing, CachedModel}
end
```

**Pros**: Maximum sharing across iterations
**Cons**: Requires state management, complicates device struct

### Alternative B: Generated Function Specialization

Use `@generated` functions to specialize on device type:

```julia
@generated function stamp!(dev::T, ctx, nodes...) where T <: VAModel
    # Generate type-specific code at compile time
    # Split based on static analysis
end
```

**Pros**: Compile-time optimization
**Cons**: Complex metaprogramming, harder to debug

### Alternative C: Macro-Based Code Splitting

Use macros to mark split points in VA translation:

```julia
function stamp!(dev, ctx, nodes...; x, t)
    @static_section begin
        # Parameter extraction
    end

    @dynamic_section begin
        # Voltage-dependent computation
    end
end
```

**Pros**: Explicit boundaries
**Cons**: Macros harder to maintain, doesn't reduce actual code size

## Part 6: Recommended Implementation Plan

### Phase 1: Baseline Measurement (1-2 days)
- Measure compilation time for PSP103VA stamp! with current code
- Count IR statements using `code_llvm(stamp!, ...)` or similar
- Profile where compilation time is spent

### Phase 2: Simple Parameter Extraction Split (2-3 days)
- Modify `generate_mna_stamp_method_nterm()` in vasim.jl
- Generate `_extract_params(dev)` as separate @noinline function
- Return NamedTuple of parameter values
- Measure compilation improvement

### Phase 3: Local Variable Classification (3-5 days)
- Analyze local_var_decls to classify as parameter-only or voltage-dependent
- Split local_var_init into two blocks
- Generate `_stamp_model_setup!` with parameter-only computations
- Measure further improvement

### Phase 4: Full Model/Eval Split (5-7 days)
- Complete AST analysis for parameter dependence
- Generate full two-function split
- Test with PSP103VA, BSIM4, and other large models
- Validate simulation results match unsplit version

### Phase 5: Documentation and Tests (2-3 days)
- Document the splitting architecture
- Add regression tests
- Update codegen documentation

## Appendix: OpenVAF vs Cadnip.jl Design Philosophy

| Aspect | OpenVAF | Cadnip.jl |
|--------|---------|-----------|
| Language | Rust → LLVM IR | Julia → Julia IR → LLVM |
| Split Timing | At MIR level (before LLVM) | At codegen (Julia AST) |
| Dependence Analysis | Full dataflow/taint propagation | (Proposed) AST-level classification |
| Cache Storage | Explicit cache slots | (Proposed) NamedTuple return |
| Runtime Dispatch | None (static compilation) | Type dispatch (MNAContext vs Direct) |

The key difference: OpenVAF can analyze at IR level because it controls the whole pipeline. Cadnip.jl operates at Julia AST level, so the split must happen during code generation.

## Part 7: Addressing Practical Concerns

### Problem 1: Mixed Statements

Consider `I_ds = Is * (exp(V_gs / Vt) - 1)` where `Is` and `Vt` are parameters, `V_gs` is voltage.

**OpenVAF's solution**:
- Setup: `cache[slot_Is] = Is; cache[slot_Vt] = Vt`
- Eval: `Is = cache[slot_Is]; Vt = cache[slot_Vt]; I_ds = Is * (exp(V_gs / Vt) - 1)`

The computation happens in eval, but pre-computed values are loaded from cache.

### Problem 2: isdefault() Handling

`isdefault(param)` is purely parameter-dependent:
- Setup: `cache[slot_is_default] = Float64(isdefault(dev.param))`
- Eval: `if cache[slot_is_default] != 0.0 then ...`

Booleans are stored as Float64 (0.0/1.0) in the cache.

### Problem 3: Local Variable Classification

This is the hard problem. Variables can be:

| Category | Example | Handling |
|----------|---------|----------|
| Param-only | `Vt = k*T/q` | Store in cache during setup |
| Voltage-only | `V_gs = V_g - V_s` | Local to eval only |
| Mixed assignment | `x = param; if cond then x = V*2` | Needs full analysis |
| Loop variables | `for i = 1:N` | Local to containing function |

**OpenVAF's solution**: Full dataflow taint propagation. Any value that transitively depends on an OP-dependent value is marked OP-dependent.

**For Cadnip.jl**, implementing full taint propagation is complex. Simpler alternatives:

1. **Conservative: Assume most things are voltage-dependent**
   - Only extract raw parameters to cache
   - All intermediate computations stay in eval
   - Simple but misses optimization opportunities

2. **Heuristic: Classify by AST position**
   - Code before first `V(...)` reference → potentially static
   - Code after first `V(...)` reference → definitely dynamic
   - Imperfect but catches common patterns

### Empirical Analysis: Code Before First V()

Analyzed all VADistillerModels to check if the "code before first V()" heuristic is viable:

| Model | Static Lines | Notes |
|-------|-------------|-------|
| bsim4v8.va | 1364 | 20% of analog block |
| bsim3v3.va | 369 | |
| vdmos.va | 142 | |
| capacitor.va | 119 | |
| resistor.va | 117 | |
| bjt.va | 100 | |
| mos1-6.va | 78-83 | |
| jfet1-2.va | 41-45 | |
| mes1.va | 21 | Smallest |

**PSP103** (not VADistiller): ~1000 lines before first V() (48% of analog block).
PSP103 has explicitly named blocks: `initial_model`, `initial_instance`, `evaluateblock`.

**Key finding**: All models compute parameter-derived values BEFORE accessing voltages.
This pattern is universal because:
1. More efficient (compute once, use many times)
2. Matches conceptual flow (setup → evaluate → stamp)
3. VADistiller and manual writers follow this convention

**Risk case**: If someone wrote `vbe = V(b,e);` on line 1, followed by thousands of
parameter-only lines, the heuristic would fail. But this pattern doesn't exist in
practice - all real models do parameter processing first.

### Problem 4: The NamedTuple Size Issue

A 1000+ field NamedTuple creates its own compilation problems. **OpenVAF's solution**:

```rust
// Cache is flat memory indexed by integer slot
cache_slots: TiMap<CacheSlot, (ClassId, u32), Type>

// In setup:
store_cache_slot(instance, slot_idx, value)

// In eval:
value = load_cache_slot(instance, slot_idx)
```

**For Cadnip.jl**, use a `Vector{Float64}` workspace:

```julia
# Constants generated at codegen time
const WS_Vt = 1
const WS_Is_eff = 2
const WS_SIZE = 500

@noinline function _stamp_setup!(ws::Vector{Float64}, dev, ctx, ...)
    ws[WS_Vt] = k * spec.temp / q
    ws[WS_Is_eff] = undefault(dev.Is) * scaling_factor
end

@noinline function _stamp_eval!(ws::Vector{Float64}, ctx, nodes, x, t)
    Vt = ws[WS_Vt]
    Is_eff = ws[WS_Is_eff]
    # ... use Vt, Is_eff with voltages
end
```

This avoids NamedTuple overhead entirely.

## Part 8: Revised Recommendation - Pragmatic Chunking

Given the complexity of semantic analysis, a more practical approach:

### Option A: Semantic Split (Complex but Optimal)

Full OpenVAF-style separation requires:
1. AST analysis to identify V(...) and I(...) references
2. Dependency tracking to propagate "OP-dependent" flag
3. Separate generation of setup vs eval code
4. Cache slot allocation for shared values

**Effort**: 2-3 weeks for robust implementation

### Option B: Mechanical Chunking with Mutable State Struct

Split the generated code at **top-level statement boundaries**, using a mutable struct for all local variables:

```julia
# Generated per-device workspace struct
mutable struct PSP103VA_State
    # All local variables as fields
    Vt::Float64
    Is_eff::Float64
    V_gs::Float64
    I_ds::Float64
    # ... hundreds of fields for large models

    function PSP103VA_State()
        s = new()
        # Initialize all fields to zero
        s.Vt = 0.0
        s.Is_eff = 0.0
        # ...
        return s
    end
end

# Chunk functions - @noinline prevents SROA
@noinline function _chunk1!(s::PSP103VA_State, dev, ctx, nodes, x, t, spec)
    # Parameter extraction
    s.param1 = undefault(dev.param1)
    s.Vt = k * spec.temp / q
    s.Is_eff = s.param1 * scaling_factor
    # ... first chunk of code, all variable access via s.varname
end

@noinline function _chunk2!(s::PSP103VA_State, ctx, nodes, x, t)
    # Voltage extraction + more computation
    s.V_gs = x[nodes[1]] - x[nodes[2]]
    s.I_ds = s.Is_eff * (exp(s.V_gs / s.Vt) - 1)
    # ... uses s.Vt, s.Is_eff from chunk1
end

@noinline function _chunk3!(s::PSP103VA_State, ctx, nodes)
    # Stamping
    stamp_G!(ctx, nodes[1], nodes[2], s.dI_dV)
    stamp_b!(ctx, nodes[1], -s.Ieq)
    # ...
end

function stamp!(dev::PSP103VA, ctx, nodes...; x, t, spec)
    s = PSP103VA_State()
    _chunk1!(s, dev, ctx, nodes, x, t, spec)
    _chunk2!(s, ctx, nodes, x, t)
    _chunk3!(s, ctx, nodes)
end
```

**Why mutable struct instead of Vector{Float64}?**
- Named access (`s.Vt`) vs indexed (`ws[42]`) - more readable/debuggable
- Type-stable - Julia knows all field types at compile time
- No bounds checking overhead
- Same IR generation as local variables (field load/store)

**Why @noinline prevents SROA issues:**
- SROA only happens when functions are inlined
- With @noinline, struct is passed as opaque pointer
- Each chunk compiles independently, no 90K statement explosion
- Struct field access is simple load/store, not scalar replacement

**Key rules for safe split points**:
- Never split inside a conditional (`if`/`else`) block
- Never split inside a loop (`for`/`while`)
- Only split between top-level statements in the analog block

**Codegen changes in vasim.jl**:
1. Collect all local variable names and types
2. Generate `mutable struct DeviceName_State` with all fields
3. Transform `varname = expr` → `s.varname = expr` throughout
4. Transform `varname` references → `s.varname`
5. Split at top-level statement boundaries
6. Generate @noinline chunk functions

**Effort**: 1 week for basic implementation

### Option C: Named Block Detection (Best for PSP-style models)

PSP103 already has semantically meaningful named blocks:

```
analog begin
    begin : initial_model      // Pure parameter processing (no V())
    end
    begin : initial_instance   // Instance param processing (no V())
    end
    begin : evaluateblock      // Main evaluation
        begin : evaluateStatic     // DC calculations with V()
        begin : evaluateDynamic    // AC/transient calculations
        begin : loadStatic         // Static stamping
        begin : loadDynamic        // Dynamic stamping
        begin : noise              // Noise model
        begin : OPinfo             // Operating point info
    end
end
```

**VerilogAParser already exposes this:**
```julia
# AnalogSeqBlock structure
struct AnalogSeqBlock
    kwbegin::EXPR{Keyword}
    decl::Maybe{EXPR{AnalogBlockDecl}}  # Contains the block name!
    stmts::EXPRList{Any}
    kwend::EXPRErr{Keyword}
end

struct AnalogBlockDecl
    colon::EXPR{Notation}
    id::EXPR{Identifier}      # Block name like "initial_model"
    decls::EXPRList{...}
end

# Access block name:
if asb.decl !== nothing
    block_name = String(asb.decl.id)  # "initial_model", "evaluateblock", etc.
end
```

**Implementation approach:**
1. Walk the AST looking for named `AnalogSeqBlock` nodes
2. Map known block names to function roles:
   - `initial_model` → `_setup_model!(s, dev, spec)`
   - `initial_instance` → `_setup_instance!(s, dev, ctx, nodes, spec)`
   - `evaluateStatic`, `evaluateDynamic` → `_evaluate!(s, ctx, x, t)`
   - `loadStatic`, `loadDynamic` → `_stamp!(s, ctx, nodes)`
3. Generate separate @noinline functions for each named block
4. For unnamed/unknown blocks, use mechanical chunking as fallback

**Benefits:**
- Respects model author's semantic intent
- Natural boundaries (no risk of mid-computation splits)
- Works great for well-structured models like PSP
- Falls back to mechanical chunking for VADistiller models without named blocks

**Effort**: 1-2 weeks (depends on naming conventions to support)

### Option D: Simple Named Block Compilation (Recommended)

**Key insight**: PSP103 is the main model that blows up, and it already has named blocks.
Rather than complex analysis, just compile each named `begin : blockname` as a separate function.

**Implementation in vasim.jl:**

```julia
# When translating AnalogSeqBlock with a name:
function (to_julia::MNAScope)(asb::VANode{AnalogSeqBlock})
    if asb.decl !== nothing
        # Named block - generate as separate @noinline function
        block_name = Symbol(String(asb.decl.id))
        func_name = Symbol("_block_", block_name, "!")

        # Collect the block's statements
        body = Expr(:block)
        for stmt in asb.stmts
            push!(body.args, to_julia(stmt))
        end

        # Register this block for later function generation
        push!(to_julia.named_blocks, (name=func_name, body=body))

        # Return a call to the generated function
        return :($func_name(s, ctx, nodes, x, t, spec))
    else
        # Unnamed block - inline as before
        ret = Expr(:block)
        for stmt in asb.stmts
            push!(ret.args, to_julia(stmt))
        end
        return ret
    end
end

# Generated code structure:
mutable struct PSP103VA_State
    # All local variables
    EPSSI::Float64
    SWGEO_i::Float64
    # ... etc
end

@noinline function _block_initial_model!(s, dev, ctx, nodes, x, t, spec)
    s.EPSSI = 8.854e-12 * 11.7
    s.SWGEO_i = floor(...)
    # ... all initial_model code
end

@noinline function _block_initial_instance!(s, dev, ctx, nodes, x, t, spec)
    # ... all initial_instance code, uses s.EPSSI etc from previous block
end

@noinline function _block_evaluateblock!(s, dev, ctx, nodes, x, t, spec)
    # Contains nested blocks - could further split or inline
    _block_evaluateStatic!(s, ctx, nodes, x, t, spec)
    _block_evaluateDynamic!(s, ctx, nodes, x, t, spec)
    _block_loadStatic!(s, ctx, nodes)
    # ...
end

function stamp!(dev::PSP103VA, ctx, nodes...; x, t, spec)
    s = PSP103VA_State()
    _block_initial_model!(s, dev, ctx, nodes, x, t, spec)
    _block_initial_instance!(s, dev, ctx, nodes, x, t, spec)
    _block_evaluateblock!(s, dev, ctx, nodes, x, t, spec)
end
```

**Variable access transformation:**
- All local variable assignments: `varname = expr` → `s.varname = expr`
- All local variable reads: `varname` → `s.varname`
- Parameters stay as-is (read from `dev.param`)
- Node voltages stay as-is (read from `x[node_idx]`)

**Why this works:**
1. Each named block becomes a separate compilation unit
2. @noinline prevents inlining back into a giant function
3. State struct passed by reference - no copying overhead
4. Smaller models (VADistiller output) don't have named blocks → unchanged behavior

**Effort**: ~3-5 days
- Modify `MNAScope` to track named blocks
- Generate state struct with all local variables
- Transform variable access to struct field access
- Generate @noinline wrapper functions

## Part 9: Implementation Details for Mechanical Chunking

### Workspace Variable Assignment

At codegen time in `vasim.jl`:

```julia
# Build mapping: variable_name -> workspace_index
workspace_mapping = Dict{Symbol, Int}()
ws_index = 1

# Assign indices to all variables that need persistence
for var in all_local_variables
    workspace_mapping[var] = ws_index
    ws_index += 1
end

# Also assign indices for any expression results that cross chunk boundaries
```

### Identifying Safe Split Points

```julia
function find_safe_split_points(statements::Vector{Expr}, target_chunks::Int)
    total = length(statements)
    chunk_size = div(total, target_chunks)

    split_points = Int[]
    current_pos = chunk_size

    while current_pos < total
        # Find nearest safe point (not inside control structure)
        safe_pos = find_nearest_safe_point(statements, current_pos)
        push!(split_points, safe_pos)
        current_pos = safe_pos + chunk_size
    end

    return split_points
end

function find_nearest_safe_point(statements, target_pos)
    # Walk backward from target to find top-level statement
    # (not inside if/for/while)
    for pos in target_pos:-1:1
        if is_top_level_statement(statements, pos)
            return pos
        end
    end
    return 1
end
```

### Variable Liveness Across Chunks

To minimize workspace size, track which variables are live at each split point:

```julia
function compute_live_variables(statements, split_points)
    # For each chunk, compute:
    # - Variables defined in this chunk
    # - Variables used in later chunks (must persist)

    live_at_split = [Set{Symbol}() for _ in split_points]

    for (i, split_pos) in enumerate(split_points)
        # Variables defined before split_pos and used after
        for stmt in statements[1:split_pos]
            defined = extract_defined_vars(stmt)
            for var in defined
                if is_used_after(var, statements, split_pos)
                    push!(live_at_split[i], var)
                end
            end
        end
    end

    return live_at_split
end
```

## Conclusion

### Revised Recommendation

**Start with mechanical chunking** (Option B/C) because:
1. Immediate benefit with minimal analysis
2. No risk of incorrect semantic classification
3. Easy to verify correctness (same results as unsplit)
4. Foundation for future semantic optimization

### Implementation Path

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Chunk detection | Find safe split points in generated AST |
| 1 | Workspace setup | Generate workspace allocation + variable mapping |
| 2 | Chunk generation | Generate @noinline chunk functions |
| 2 | Integration | Wire chunks together in stamp! |
| 3 | Testing | Verify PSP103VA, BSIM4 compile and run correctly |
| 3 | Measurement | Compare compilation time before/after |

### Success Criteria

- PSP103VA compilation time reduced by 50%+
- No change in simulation results (bit-exact)
- Workspace overhead < 5% of runtime
- Code remains readable and debuggable

## Part 10: Implementation Status and Findings

### What Was Implemented

The following changes were made to `src/vasim.jl`:

1. **MNAScope Extended** (lines 459-479):
   - Added `use_state_struct::Bool` field
   - Added `named_blocks::Vector{NamedTuple{...}}` field
   - Added `is_local_variable()` helper function

2. **Variable Access Transformation** (when enabled):
   - `IdentifierPrimary` handler: `varname` → `s.varname` for local variables
   - `AnalogVariableAssignment` handler: `varname = expr` → `s.varname = expr`

3. **Named Block Detection**:
   - `AnalogSeqBlock` handler: detects `begin : blockname` and registers for separate generation
   - Returns function call to `_block_name!(s, dev, ctx, ...)` instead of inlining

4. **Block Function Generation** (lines 2111-2140):
   - Generates @noinline functions for each named block
   - Destructures node voltages from NamedTuple for V() access

5. **State Struct Generation** (lines 2078-2109):
   - Generates `mutable struct DeviceName_State` with all local variables
   - Inner constructor initializes all fields to type-appropriate defaults

### Critical Issue Discovered: Dual Type Mismatch

**The Problem**: Variables that receive voltage-dependent values (from V(), I() calls) get
ForwardDiff.Dual values, but the state struct has typed fields (Float64, Int, String).

```julia
# Generated state struct
mutable struct sp_resistor_State
    v_rhs::Float64  # Typed as Float64
end

# In stamp! body
s.v_rhs = V(p,n)  # V() returns Dual{JacobianTag, Float64, 2}
# ERROR: no method matching Float64(::ForwardDiff.Dual{...})
```

**OpenVAF's Solution**: They only put parameter-derived values in the cache, never voltage-
dependent values. The eval function computes voltage-dependent values locally.

### Current Status: State Struct Mode Disabled

To avoid breaking existing functionality, state struct mode is **disabled by default**:
- `use_state_struct = false` in MNAScope constructor
- All tests pass with this setting
- Named blocks are inlined as before

### Options to Enable State Struct Mode

1. **Option A: Any-typed fields** (simple but loses type stability)
   ```julia
   mutable struct DeviceName_State
       v_rhs::Any  # Can hold Float64 or Dual
   end
   ```

2. **Option B: Parameterized struct** (preserves type stability)
   ```julia
   mutable struct DeviceName_State{T}
       v_rhs::T
   end
   ```
   The struct type would be specialized based on whether duals are used.

3. **Option C: Separate storage for op-dependent variables** (like OpenVAF)
   - State struct only holds parameter-derived values
   - Voltage-dependent variables stay as local variables
   - Requires classifying variables by op-dependence

4. **Option D: Don't store voltage-dependent variables in struct**
   - Let block functions return computed values
   - Or use different mechanism for variable passing
   - More complex code generation

### Recommended Next Steps

1. **For immediate use**: Keep state struct disabled. Current code works correctly.

2. **For PSP103 specifically**: Try Option B (parameterized struct) since PSP103's named
   blocks have natural semantics:
   - `initial_model`, `initial_instance`: No voltage access, can use Float64 struct
   - `evaluateblock`: Has voltage access, struct needs to handle duals

3. **For general solution**: Implement Option C (OpenVAF-style classification)
   - Classify variables by op-dependence during AST analysis
   - Only static variables go in struct
   - More work but correct semantic separation

### Test Results

With state struct mode disabled:
- `test/mna/core.jl`: ✓ 356/356 tests pass
- `test/mna/va.jl`: ✓ 49/49 tests pass
- `test/mna/vadistiller.jl`: ✓ 42/42 tests pass
- `test/mna/vadistiller_integration.jl`: Times out (expected, PSP103 still monolithic)
