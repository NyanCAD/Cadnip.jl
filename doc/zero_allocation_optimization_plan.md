# Zero-Allocation Optimization Plan for GPU Compatibility

## Current State Analysis

### Allocation Profile (from benchmark: `vacask/rc/cedarsim`)

| Component | Allocation | Status |
|-----------|------------|--------|
| `fast_rebuild!(ws, u, t)` | 24 bytes | **Near-zero** |
| Builder call (ValueOnlyContext) | 4360 bytes | Compilation overhead |
| `update_sparse_from_coo!` | 816 bytes | Compilation overhead |
| `reset_value_only!` | 0 bytes | **Zero** |

The 24 bytes per iteration is very close to the goal of true zero-allocation.

### Completed Optimizations (This PR)

1. **PWL/PULSE SVector codegen** - Time-dependent sources now generate SVector for
   constant parameters, eliminating device construction allocations.

2. **Positional builder arguments** - Added `@inline` positional argument versions
   of builder functions to bypass keyword argument overhead.

3. **Source of remaining 24 bytes** - Profile.Allocs identified the source:
   - `UnitRange{Int64}` (32 bytes) from `1:n_G` loop ranges
   - `Tuple{Int64, Int64}` from iterator state
   - These are from the loops in `fast_rebuild!` copying COO values

### Root Causes of Remaining Allocations

1. **PWL Source Array Creation** (Primary Issue)
   - Generated code creates runtime `Vector`:
   ```julia
   # CURRENT (allocates per device construction)
   stamp!(PWLVoltageSource([0.0, 1e-6, ...], [0.0, 0.0, ...]; name=:vs), ...)
   ```
   - Should use `SVector` for constant parameters:
   ```julia
   # TARGET (zero-allocation)
   stamp!(PWLVoltageSource(SVector{6}(0.0, 1e-6, ...), SVector{6}(0.0, 0.0, ...); name=:vs), ...)
   ```

2. **Dynamic Function Call Overhead**
   - Time parameter `t` boxing when passed through Union types
   - Keyword argument processing overhead

3. **Type Instability in Source Evaluation**
   - `get_source_value(V, t, mode)` may have type instability for different source types

## Architecture Comparison: OpenVAF vs CedarSim

### OpenVAF's Zero-Allocation Approach

1. **Direct Pointers to Matrix Entries**
   ```rust
   // Matrix entries are direct pointers, no lookup during eval
   ptr_resist.set(sim.jacobian_resist[i].get());
   ```

2. **Pre-allocated State Vectors**
   ```rust
   sim.state_1.resize(num_states, 0.0);
   sim.state_2.resize(num_states, 0.0);
   swap(&mut sim.state_1, &mut sim.state_2);  // Zero-copy iteration
   ```

3. **Separated Resist/React Components**
   - Resistive (G) and Reactive (C) components in separate arrays
   - Direct accumulation without intermediate structures

### CedarSim's Current Approach

1. **COO-to-CSC Mapping**
   - Pre-computed mapping from COO indices to sparse matrix positions
   - `update_sparse_from_coo!` updates nzval in-place

2. **ValueOnlyContext**
   - Replaces push! with indexed writes
   - Pre-sized arrays based on initial structure discovery

3. **Workspace Pattern**
   - `CompiledStructure` (immutable, shared)
   - `EvalWorkspace` (mutable, per-thread)

## Optimization Plan

### Phase 1: Fix PWL/PULSE Codegen (Immediate Win)

**Problem**: The PWL codegen at `src/spc/codegen.jl:1895-1920` creates runtime arrays.

**Solution**: Use SVector like PULSE does when parameters are constant.

```julia
# src/spc/codegen.jl - PWL codegen fix
if fname == :pwl
    vals = [cg_expr!(state, v) for v in tran_source.values]

    # Check if all values are constant
    all_constant = all(x -> x isa Number, vals)

    if all_constant && length(vals) >= 2
        # ZERO-ALLOCATION PATH: Use SVector
        n_points = div(length(vals), 2)
        times_sv = :(SVector{$n_points,Float64}($(Float64.(vals[1:2:end])...)))
        values_sv = :(SVector{$n_points,Float64}($(Float64.(vals[2:2:end])...)))

        return quote
            stamp!(PWLVoltageSource($times_sv, $values_sv; name=...), ctx, $p, $n; ...)
        end
    else
        # FALLBACK: Dynamic arrays for variable parameters
        ...
    end
end
```

**Impact**: Eliminates device construction allocation per iteration.

### Phase 1.5: Fix Loop Iterator Allocations (Remaining 24 bytes)

**Problem**: The loops copying COO values allocate iterator state:
```julia
@inbounds for k in 1:n_G
    ws.G_V[k] = vctx.G_V[k]
end
```
The `1:n_G` creates a `UnitRange{Int64}` and iteration allocates `Tuple{Int64, Int64}`.

**Solution**: Use explicit while loops or `@simd`:
```julia
# Option A: While loop (guaranteed no allocation)
k = 1
while k <= n_G
    @inbounds ws.G_V[k] = vctx.G_V[k]
    k += 1
end

# Option B: Use copyto! (may be optimized by Julia)
copyto!(ws.G_V, 1, vctx.G_V, 1, n_G)

# Option C: Use @simd which forces scalar loop
@inbounds @simd for k in 1:n_G
    ws.G_V[k] = vctx.G_V[k]
end
```

**Impact**: Should eliminate the remaining 24 bytes per iteration.

### Phase 2: Type-Stable Source Evaluation

**Problem**: `get_source_value(V, t, mode)` dispatches on multiple source types.

**Solution**: Inline evaluation in stamp! methods or use @generated functions.

```julia
# Option A: Inline evaluation (preferred for small sources)
@inline function stamp!(V::PWLVoltageSource{SVector{N,Float64},SVector{N,Float64}},
                        ctx::ValueOnlyContext, p::Int, n::Int;
                        t::Float64=0.0, _sim_mode_::Symbol=:dcop) where N
    # Inline PWL lookup - no allocation
    v = _pwl_lookup_inline(V.times, V.values, t)
    stamp_b!(ctx, I_idx, v)
    ...
end

# Option B: Specialize for common sizes
for N in (2, 4, 6, 8, 10)
    @eval @inline function _pwl_lookup_inline(
        times::SVector{$N,Float64}, values::SVector{$N,Float64}, t::Float64)
        # Binary search and interpolation
        ...
    end
end
```

### Phase 3: Eliminate Time Boxing

**Problem**: `t::Real` in function signatures can cause boxing for ForwardDiff.Dual.

**Solution**: Type-parameterize time or use `@inline` with concrete types.

```julia
# Current (may box)
function fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)

# Target (concrete types, no boxing)
@inline function fast_rebuild!(ws::EvalWorkspace{T}, u::Vector{T}, t::T) where T<:Real
```

### Phase 4: Direct Matrix Stamping (OpenVAF-Style)

**Problem**: COO-to-CSC mapping adds indirection.

**Solution**: Store direct pointers/indices to sparse matrix entries.

```julia
struct DirectStampContext{T}
    # Direct pointers to G matrix entries (no lookup)
    G_entries::Vector{Ptr{T}}  # Or indices for safety
    C_entries::Vector{Ptr{T}}
    b::Vector{T}

    # Position counters (reset per iteration)
    G_pos::Int
    C_pos::Int
end

# Zero-allocation stamping
@inline function stamp_G!(ctx::DirectStampContext{T}, val::T) where T
    pos = ctx.G_pos
    unsafe_store!(ctx.G_entries[pos], val)  # Direct write
    ctx.G_pos = pos + 1
end
```

This eliminates the `update_sparse_from_coo!` step entirely.

### Phase 5: GPU-Compatible Architecture

**Requirements for GPU execution**:
1. No dynamic allocation (achieved via above phases)
2. No dynamic dispatch (type-stable functions)
3. Fixed-size data structures (SVector, Tuple)
4. No exceptions (use sentinel values)

**GPU Kernel Structure**:
```julia
# GPU-compatible device evaluation
@inline function evaluate_device_kernel!(
    G_vals::CuDeviceVector{Float64},
    C_vals::CuDeviceVector{Float64},
    b_vals::CuDeviceVector{Float64},
    device_params::DeviceParams,  # Pre-baked struct
    node_indices::NTuple{N,Int32},
    x::CuDeviceVector{Float64},
    t::Float64
) where N
    # All operations are type-stable and allocation-free
    ...
end
```

**Required Changes**:
1. Device parameters must be `isbitstype`
2. Circuit topology encoded as fixed-size tuples
3. Separate kernels for structure discovery vs. value update

## Implementation Roadmap

| Step | Description | Effort | Impact |
|------|-------------|--------|--------|
| 1 | Fix PWL codegen to use SVector | Low | High |
| 2 | Add PULSE detection for SVector path | Low | High |
| 3 | Type-parameterize EvalWorkspace | Medium | Medium |
| 4 | Inline PWL lookup for common sizes | Medium | Medium |
| 5 | Direct matrix stamping (DirectStampContext) | High | High |
| 6 | GPU kernel extraction | Very High | Transformative |

## Verification Strategy

1. **Allocation Benchmarks**: Run `@allocated` tests for each path
2. **JET.jl Analysis**: Check for type instabilities
3. **Cthulhu.jl**: Verify inlining and specialization
4. **GPU Validation**: Test with CUDA.jl on simple circuits

## Expected Outcomes

After Phase 1-2:
- **0 bytes** allocation per `fast_rebuild!` iteration
- ~2x faster per-iteration evaluation

After Phase 5:
- GPU-compatible circuit evaluation
- Potential for 100-1000x speedup on large circuits with parallel device evaluation
