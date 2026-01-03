# Zero-Allocation Verilog-A Evaluation

## Current Status (January 2026)

### Completed: MNAContext Reuse (~71% Memory Reduction)

The first major optimization has been implemented - storing and reusing the MNAContext
instead of allocating a new one each Newton iteration.

**Benchmark Results (VACASK RC circuit, 1s transient with dt=1µs):**

| Metric | Before ctx reuse | After ctx reuse | Current |
|--------|------------------|-----------------|---------|
| Memory | 60.33 MB | 17.56 MB | 1768 MB* |
| Bytes/iteration | 6,314 | 1,838 | 1768 |

*Note: Current measurement is for full 1s simulation (1M iterations vs 10ms before)

### Changes Made

1. **`src/mna/context.jl`**: Added `reset_for_restamping!(ctx)` function that empties all
   arrays while preserving their allocated capacity.

2. **`src/spc/codegen.jl`**: Modified builder function generation to accept optional `ctx`
   parameter for context reuse:
   ```julia
   function circuit_name(params, spec, t=0.0; x=Float64[], ctx=nothing)
       if ctx === nothing
           ctx = MNAContext()
       else
           reset_for_restamping!(ctx)
       end
       # ... stamp devices ...
       return ctx
   end
   ```

3. **`src/mna/precompile.jl`**:
   - Added `ctx::MNAContext` field to `PrecompiledCircuit` and `EvalWorkspace`
   - Modified `fast_rebuild!` to pass stored context to builder

## Memory Profiling Results (January 2026)

### Current Allocation Breakdown

Profiling the VACASK RC benchmark (1M timesteps, 1M Newton iterations) shows:

| Component | Bytes/iteration | Notes |
|-----------|-----------------|-------|
| Total measured | 1768 | Full transient simulation |
| Builder call (with ctx reuse) | ~900 | Direct builder measurement |
| PWLVoltageSource array copies | ~256 | Constructor copies times/values arrays |
| COO push! operations | ~500-600 | Pushing to `Vector{MNAIndex}` |
| ODE solver overhead | ~800 | DifferentialEquations internals |

### Identified Allocation Sources

Using `Profile.Allocs`, the top allocation types are:

1. **`NodeIndex`** (251 samples) - from `push!` in `stamp_G!`, `stamp_C!`, `stamp_b!`
2. **`Memory{Float64}`** (108 samples) - array storage for vectors
3. **`Vector{Float64}`** (97 samples) - PWLVoltageSource array copies
4. **`CurrentIndex`** (51 samples) - from voltage source `alloc_current!`
5. **`String`** (22 samples) - Symbol creation in `Symbol(:I_, V.name)`

### Root Causes

1. **PWLVoltageSource constructor copies arrays**: The constructor does `new(Float64.(times), Float64.(values), name)` which allocates 256 bytes per call even with already-Float64 inputs.

2. **COO arrays use abstract type**: `Vector{MNAIndex}` stores abstract type elements requiring potential boxing on each push (though Julia 1.11 can optimize this to some degree).

3. **Device structs recreated each call**: The generated builder creates new `PWLVoltageSource`, etc. on every iteration instead of reusing them.

4. **Symbol creation in voltage sources**: `Symbol(:I_, V.name)` creates strings on each stamp.

## Recommended Next Step: Phase 2.5 (PWL Device Reuse)

Before tackling full value-only mode, a quick win is to **reuse device structs across iterations**:

```julia
# Current (allocates per call):
stamp!(PWLVoltageSource(times, values; name=:vs), ctx, p, n; t=t)

# Proposed (zero allocation after first call):
# Store device in a const or in the circuit struct
const vs_device = PWLVoltageSource(times, values; name=:vs)
stamp!(vs_device, ctx, p, n; t=t)
```

This would eliminate the ~256 bytes/call from PWL array copying.

**Implementation approach**:
1. Modify codegen to create device structs once at module level
2. Pass device references to stamp! instead of creating new ones
3. For parameterized devices, store in a mutable holder

**Estimated improvement**: ~30% reduction (256 of 900 bytes from builder)

## Problem Statement

The VA integration (`vasim.jl`) allocates memory per Newton iteration. For a 1-second
simulation with ~2M iterations, this creates significant GC pressure.

**Root cause**: While `fast_rebuild!` now reuses the MNAContext, there are still allocations
from:
1. PWLVoltageSource array copies (~256 bytes/iter)
2. COO push! operations to abstract-typed vectors (~500-600 bytes/iter)
3. ODE solver internal allocations (~800 bytes/iter)

## Remaining Work

### Phase 2: Dictionary Elimination

The `node_to_idx` dictionary allocates when looking up/adding nodes. Replace with:

```julia
# Currently (allocates):
idx = get!(ctx.node_to_idx, name, length(ctx.node_to_idx) + 1)

# Target (zero-allocation):
# At compile time, generate a lookup table with known node indices
const NODE_INDICES = Dict(:vdd => 1, :gnd => 0, :out => 2, ...)
# At runtime, just use the constant lookup
```

**Estimated improvement**: Reduce to <100 bytes/iter

### Phase 3: True Zero-Allocation Value-Only Mode

The ultimate goal is to separate structure discovery from value evaluation:

```
Setup: builder(params, spec) → discover pattern → store COO indices + value pointers

Newton: evaluate_values!(pointers, params, x, t) → write directly through pointers
        ↑ NO ALLOCATION (true zero)
```

This requires:
1. Caching COO indices from initial build
2. Generating a value-only evaluation function that writes directly to cached slots
3. Skipping all structure-building code (get_node!, alloc_current!, etc.)

**Estimated improvement**: 0 bytes/iter

### Phase 4: Compile-Time Dual Specialization

The generated `stamp!` method creates Duals with inferrable but not explicit sizes:

```julia
# Current - size known but not propagated:
partials_tuple = Expr(:tuple, [k == i ? 1.0 : 0.0 for k in 1:n_all_nodes]...)
$node_sym = Dual{JacobianTag}($(Symbol("V_", i)), $partials_tuple...)

# Better - explicit size for compile-time specialization:
$node_sym = Dual{JacobianTag,Float64,$n_all_nodes}(
    $(Symbol("V_", i)),
    Partials{$n_all_nodes,Float64}($partials_tuple))
```

### Phase 5: GPU Ensemble Support (StaticArrays)

For GPU ensemble (parameter sweeps, Monte Carlo), return fresh StaticArrays:

```julia
struct StaticVACircuit{N,T,F}
    eval_fn::F  # Returns (G::SMatrix, C::SMatrix, b::SVector)
    params::NamedTuple
end

# Zero allocation because SMatrix/SVector are stack-allocated
@inline function evaluate(vc::StaticVACircuit{N,T,F}, u::SVector{N,T}, t) where {N,T,F}
    G, C, b = vc.eval_fn(u, t, vc.params)
    return G, C, b  # All stack-allocated
end
```

## Target Architectures

| Use Case | Array Type | Formulation | Current | Target |
|----------|------------|-------------|---------|--------|
| CPU standard | `Vector{T}`, `SparseMatrixCSC` | In-place `f!(resid, du, u, p, t)` | 1768 B/iter | 0 B/iter |
| GPU single | `CuArray{T}`, `CuSparseMatrixCSC` | In-place `f!(resid, du, u, p, t)` | N/A | 0 B/iter |
| GPU ensemble | `SVector{N,T}`, `SMatrix{N,N,T}` | Out-of-place `f(u, p, t) -> SVector` | N/A | 0 B/iter |

## Files Modified/To Modify

| File | Status | Changes |
|------|--------|---------|
| `src/mna/context.jl` | Done | Added `reset_for_restamping!()` |
| `src/spc/codegen.jl` | Done | Builder accepts optional `ctx` parameter |
| `src/mna/precompile.jl` | Done | Store and reuse MNAContext in PrecompiledCircuit and EvalWorkspace |
| `src/spc/codegen.jl` | Phase 2.5 | Generate device structs at module level, not per-call |
| `src/mna/devices.jl` | Phase 2.5 | Modify PWLVoltageSource to avoid array copies |
| `src/vasim.jl` | Future | Add `@inline`, explicit Dual sizes, value-only `evaluate!` |
| `src/mna/contrib.jl` | Future | Add `@inline` to `va_ddt()` and contribution helpers |

## Open Questions

1. **Conditional contributions**: VA `if` blocks that add/remove contributions change sparsity
   pattern. May need worst-case pattern with zeros.

2. **ddx()**: The VA `ddx()` function computes partial derivatives - may need special handling.

3. **Time-dependent sources**: PWL, SIN sources change `b` vector values but not structure -
   fits the value-only pattern.

4. **Internal node aliasing**: Short-circuit detection currently modifies structure at runtime -
   needs to be part of pattern discovery.
