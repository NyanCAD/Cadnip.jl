# Path to Zero-Allocation Verilog-A Models

## Executive Summary

The current VA integration (`vasim.jl`) uses ForwardDiff to automatically extract Jacobians from Verilog-A contribution statements. While elegant, this approach allocates ~2KB per Newton iteration due to Dual number creation and COO stamping.

This document outlines a path to zero-allocation VA model evaluation while preserving correctness and GPU compatibility.

## Current Architecture Analysis

### How VA Models Work Now

```
VA Source → VerilogAParser → AST → make_mna_device() → stamp!(dev, ctx, nodes...; x, t, spec)
                                                              ↓
                                        [Each Newton iteration]
                                              ↓
                              1. Extract parameters from dev struct
                              2. Allocate internal nodes (idempotent)
                              3. Create Dual{JacobianTag} for each node voltage
                              4. Evaluate analog block with duals
                              5. Wrap ddt() values in Dual{ContributionTag}
                              6. Extract partials → stamp G, C, b
```

### Allocation Sources

| Source | Location | Bytes/call | Cause |
|--------|----------|------------|-------|
| Dual creation | vasim.jl:1584-1590 | ~128 | `Dual{JacobianTag}(V, partials...)` |
| Partials tuple | vasim.jl:1588 | ~64 | `ntuple(...)` for partials |
| `va_ddt()` | contrib.jl:96-114 | ~32 | `Dual{ContributionTag}(...)` |
| COO push! | context.jl | ~48 | First call only, then idempotent |
| Intermediate allocations | varies | ~200+ | Expressions creating temporaries |

**Total: ~500+ bytes per contribution evaluation, ~2KB per stamp! call for complex devices**

### Generated stamp! Method Structure

From `generate_mna_stamp_method_nterm()` (vasim.jl:1729-1771):

```julia
function stamp!(dev::DeviceName, ctx, node1, node2, ...; x, t, spec)
    # 1. Parameter extraction (no allocation)
    param1 = undefault(dev.param1)

    # 2. Internal node allocation (idempotent)
    _node_internal = alloc_internal_node!(ctx, :name)

    # 3. Voltage extraction (no allocation)
    V_1 = x[node1]

    # 4. Dual creation (ALLOCATES!)
    node1 = Dual{JacobianTag}(V_1, (1.0, 0.0, ...))  # ← Problem

    # 5. Evaluate contributions (may allocate via va_ddt)
    I_branch = expr_using_duals  # ← Problem if uses ddt()

    # 6. Extract Jacobian (type dispatch)
    if I_branch isa Dual{ContributionTag}  # ← Runtime check
        ...
    end

    # 7. Stamp into COO (first call allocates, subsequent don't)
    stamp_G!(ctx, p, k, dI_dVk)
end
```

## Zero-Allocation Architecture

### Design Principles

1. **Separate compilation from evaluation**: Discover structure once, evaluate many times
2. **Static typing**: Use compile-time known sizes to avoid dynamic dispatch
3. **Pre-computed indices**: Store COO indices, not Dual-based discovery
4. **Symbolic derivatives where possible**: Many VA contributions have analytic Jacobians

### Type Hierarchy

```julia
# For CPU simulation (sparse matrices)
struct VACompiledDevice{F,P,S}
    # Stamp structure (discovered at first call, frozen after)
    stamp_pattern::S            # COO indices for G, C, b

    # Evaluation function (generated, specialized)
    eval_fn::F                  # (params, spec, x, t) -> (G_vals, C_vals, b_vals)

    # Device data
    params::P
end

# For GPU ensemble (StaticArrays)
struct VAStaticDevice{N,T,K,F,P}
    # Pre-computed stamp pattern as tuples
    G_indices::NTuple{K_G, Tuple{Int,Int}}
    C_indices::NTuple{K_C, Tuple{Int,Int}}
    b_indices::NTuple{K_b, Int}

    # Specialized evaluation function
    eval_fn::F                  # (params, x, t) -> values tuple

    # Device parameters (as tuple for type stability)
    params::P
end
```

### Evaluation Without Duals

Instead of runtime AD, generate specialized evaluation code at compile time:

```julia
# Current (allocating):
function eval_contribution(x)
    V1 = Dual{JacobianTag}(x[1], 1.0, 0.0)  # Allocates
    V2 = Dual{JacobianTag}(x[2], 0.0, 1.0)  # Allocates
    I = V1 / R + C * va_ddt(V1 - V2)        # More allocations
    # ... extract from duals
end

# Zero-allocation:
@generated function eval_contribution_static(x::SVector{2,T}, params) where T
    quote
        V1, V2 = x[1], x[2]
        Vpn = V1 - V2

        # Resistive: I_R = Vpn / R
        I_R = Vpn / params.R
        dI_R_dV1 = 1 / params.R
        dI_R_dV2 = -1 / params.R

        # Reactive: I_C = C * ddt(Vpn) → stamps C into C matrix
        # Value contribution is 0 (capacitor current at DC)
        # Jacobian contribution is C
        dq_dV1 = params.C
        dq_dV2 = -params.C

        # Return all values (no Dual allocation)
        (I = I_R,
         dI_dV = (dI_R_dV1, dI_R_dV2),
         q = zero(T),
         dq_dV = (dq_dV1, dq_dV2))
    end
end
```

### Stamp Pattern Discovery

Run the model once to discover the COO pattern, then freeze it:

```julia
struct StampPattern
    # G matrix stamps: (row, col, evaluator_index)
    G_stamps::Vector{Tuple{Int,Int,Int}}

    # C matrix stamps: (row, col, evaluator_index)
    C_stamps::Vector{Tuple{Int,Int,Int}}

    # b vector stamps: (row, evaluator_index)
    b_stamps::Vector{Tuple{Int,Int}}
end

function compile_va_device(dev, ctx, nodes...; x)
    # Run stamp! once to discover pattern
    stamp!(dev, ctx, nodes...; x=x)

    # Extract pattern from ctx's COO arrays
    pattern = StampPattern(...)

    # Generate specialized evaluation function
    eval_fn = generate_eval_function(dev, pattern)

    return VACompiledDevice(pattern, eval_fn, dev_params)
end
```

## Implementation Phases

### Phase 1: Pattern Extraction (No code changes to vasim.jl)

Create infrastructure to extract stamp patterns from existing stamp! calls:

```julia
# New file: src/mna/va_compiled.jl

struct VAStampPattern
    G_entries::Vector{Tuple{Int,Int}}  # (row, col) pairs
    C_entries::Vector{Tuple{Int,Int}}
    b_entries::Vector{Int}             # row indices
end

function extract_stamp_pattern(builder, params, spec, n)
    ctx = builder(params, spec; x=zeros(n))
    return VAStampPattern(
        [(i, j) for (i, j) in zip(ctx.G_I, ctx.G_J)],
        [(i, j) for (i, j) in zip(ctx.C_I, ctx.C_J)],
        collect(ctx.b_I)
    )
end
```

### Phase 2: Value-Only Evaluation

Add a second pass that only evaluates values (not structure):

```julia
function evaluate_stamps!(G_vals, C_vals, b_vals, dev, pattern, x, t, spec)
    # Evaluate contribution expressions
    # Write values to pre-sized arrays
    # No COO structure modification
end
```

### Phase 3: Symbolic Jacobian for Common Patterns

Many VA contributions have simple analytic Jacobians:

| Contribution | Jacobian |
|--------------|----------|
| `V/R` | `dI/dV = 1/R` |
| `C*ddt(V)` | `dq/dV = C` |
| `Is*(exp(V/Vt)-1)` | `dI/dV = Is/Vt * exp(V/Vt)` |
| `gm*V` | `dI/dV = gm` |

Generate specialized code for these patterns in `make_mna_device()`:

```julia
# In vasim.jl, detect simple patterns
function extract_symbolic_jacobian(expr, node_syms)
    # Pattern match common forms
    if is_linear_in_voltage(expr)
        return generate_linear_jacobian(expr, node_syms)
    elseif is_exp_form(expr)
        return generate_exp_jacobian(expr, node_syms)
    else
        # Fall back to AD for complex expressions
        return nothing
    end
end
```

### Phase 4: StaticArrays Integration

Create zero-allocation versions for GPU ensemble:

```julia
struct VAStaticResistor{T}
    R::T
end

# Generated at compile time, no duals
@inline function va_eval_resistor(dev::VAStaticResistor{T},
                                   u::SVector{N,T}, t) where {N,T}
    V1, V2 = u[1], u[2]
    I = (V1 - V2) / dev.R
    dI_dV1 = one(T) / dev.R
    dI_dV2 = -one(T) / dev.R
    return (I=I, dI=(dI_dV1, dI_dV2), q=zero(T), dq=(zero(T), zero(T)))
end
```

### Phase 5: BSIM4 and Complex Models

For complex models (BSIM4, PSP, etc.), use hybrid approach:

1. **Model-specific optimizations**: Pre-compute expensive terms once per op
2. **Caching**: Store intermediate values that don't change per iteration
3. **Selective AD**: Only use duals for genuinely nonlinear sub-expressions

```julia
struct BSIM4Compiled{T}
    # Static parameters
    params::BSIM4Params{T}

    # Cached values (recomputed when bias changes significantly)
    cache::BSIM4Cache{T}

    # Stamp pattern (fixed after first call)
    pattern::VAStampPattern
end

function evaluate_bsim4!(vals, dev::BSIM4Compiled, x, t)
    # Check if cache needs refresh
    if needs_refresh(dev.cache, x)
        refresh_cache!(dev.cache, x, dev.params)
    end

    # Evaluate using cached values (mostly scalar ops)
    Ids, gm, gds, ... = compute_ids(dev.cache, x)

    # Pack into output arrays
    # No Dual allocation needed
end
```

## Migration Path

### Step 1: Add `compile_va_circuit()` (Parallel to existing API)

```julia
# New API for compiled circuits
compiled = compile_va_circuit(builder, params, spec, n)

# Returns CompiledVACircuit that can be used with:
# - StaticCircuit for GPU ensemble
# - PrecompiledCircuit for CPU with reduced allocation
```

### Step 2: Optimize Common Device Types

Priority order:
1. **Resistor, Capacitor, Inductor** - Trivial analytic Jacobians
2. **Diode** - Simple exponential pattern
3. **MOSFET Level 1** - Quadratic I-V, well-known Jacobian
4. **BSIM4** - Complex but high-value target

### Step 3: Integrate with Existing Tests

```julia
# Verify compiled versions match original
function test_compiled_equivalence(va_code)
    original = load_va_model(va_code)
    compiled = compile_va_model(va_code)

    for x in test_points
        r_orig = evaluate_original(original, x)
        r_comp = evaluate_compiled(compiled, x)
        @test r_orig ≈ r_comp
    end
end
```

## Expected Results

| Metric | Current | After Phase 1-2 | After Phase 3-4 |
|--------|---------|-----------------|-----------------|
| Bytes/iteration (simple) | ~500 | ~100 | 0 |
| Bytes/iteration (BSIM4) | ~5000 | ~500 | ~50 (cache only) |
| GPU ensemble support | ❌ | ❌ | ✅ |
| StaticArrays support | ❌ | ❌ | ✅ |

## Files to Modify/Create

| File | Changes |
|------|---------|
| `src/mna/va_compiled.jl` | **NEW**: VACompiledDevice, pattern extraction |
| `src/mna/va_static.jl` | **NEW**: VAStaticDevice for GPU |
| `src/vasim.jl` | Add symbolic Jacobian extraction |
| `src/mna/compiled.jl` | Integration with StaticCircuit |
| `test/mna/va_compiled.jl` | **NEW**: Zero-allocation tests |

## Open Questions

1. **How to handle ddx()?** The `ddx()` VA function computes partial derivatives - may need special handling.

2. **Conditional contributions**: VA `if` blocks that add/remove contributions require careful pattern handling.

3. **Internal node aliasing**: Short-circuit detection currently modifies structure at runtime.

4. **Time-dependent sources**: PWL, SIN sources change `b` vector - may need separate handling.

## Conclusion

Zero-allocation VA models are achievable through:
1. Separating structure discovery from value evaluation
2. Using compile-time code generation for common patterns
3. Pre-computing symbolic Jacobians where possible
4. Integrating with StaticArrays for GPU support

The migration can be done incrementally, starting with simple devices and progressing to complex models like BSIM4.
