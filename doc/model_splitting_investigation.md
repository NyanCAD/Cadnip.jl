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

## Conclusion

The recommended approach is **function decomposition at codegen time**, splitting stamp! into:
1. `_extract_params()` - parameter extraction only
2. `_stamp_model_setup!()` - parameter-derived computation
3. `_stamp_eval!()` - voltage-dependent evaluation

This provides:
- **Immediate benefit**: Smaller individual functions
- **Conservative change**: Same external API
- **Reliable**: No complex dataflow analysis needed
- **Testable**: Can verify equivalence with unsplit version

The approach mirrors OpenVAF's setup/eval split conceptually while fitting Julia's compilation model.
