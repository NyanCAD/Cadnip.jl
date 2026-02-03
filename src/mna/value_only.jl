#==============================================================================#
# DirectStampContext: Zero-Copy Stamping to Sparse Matrices
#
# This module provides optimized circuit evaluation by stamping directly
# to sparse matrix nzval arrays using precomputed COO-to-nzval mapping.
#
# Key insight: COO indices (G_I, G_J, C_I, C_J) are CONSTANT after first build.
# Only values (G_V, C_V, b) change. DirectStampContext exploits this by:
#   - Storing node_to_idx reference for fast lookups (no new allocations)
#   - Tracking write positions (counter, not push!)
#   - Writing DIRECTLY to sparse matrix nzval via precomputed mapping
#
# Data flow (single write, no intermediate copy):
#   stamp → sparse.nzval[precomputed_idx]
#==============================================================================#

export DirectStampContext, create_direct_stamp_context, reset_direct_stamp!

"""
    DirectStampContext

Zero-copy context that stamps directly to sparse matrix nzval arrays.

No intermediate G_V, C_V arrays - values go straight to sparse storage.
Uses precomputed COO-to-nzval mapping for O(1) position lookup.

# Fields
- `node_to_idx::Dict{Symbol,Int}`: Node name to matrix index
- `n_nodes::Int`, `n_currents::Int`: Circuit dimensions
- `G_nzval::Vector{Float64}`: Direct reference to G sparse matrix nonzeros
- `C_nzval::Vector{Float64}`: Direct reference to C sparse matrix nonzeros
- `G_mapping::Vector{Int}`: COO position → G.nzval index
- `C_mapping::Vector{Int}`: COO position → C.nzval index
- `b::Vector{Float64}`: RHS vector (direct reference)
- `b_V::Vector{Float64}`: Deferred b stamp values
- `b_resolved::Vector{Int}`: Deferred stamp positions in b
- `G_pos::Int`, `C_pos::Int`, etc.: Mutable counters

This eliminates all intermediate copies:
  stamp → sparse.nzval[idx] directly
"""
mutable struct DirectStampContext
    # Node lookups (immutable reference)
    node_to_idx::Dict{Symbol,Int}
    n_nodes::Int
    n_currents::Int
    current_names::Vector{Symbol}  # For get_current_idx lookups (CCVS/CCCS)

    # Direct references to sparse matrix storage (these are the actual nzval arrays)
    G_nzval::Vector{Float64}
    C_nzval::Vector{Float64}

    # Precomputed COO-to-nzval mapping
    G_mapping::Vector{Int}
    C_mapping::Vector{Int}

    # RHS vector and deferred stamps
    b::Vector{Float64}
    b_V::Vector{Float64}
    b_resolved::Vector{Int}

    # Mutable counters (reset each iteration)
    G_pos::Int
    C_pos::Int
    b_deferred_pos::Int
    current_pos::Int
    charge_pos::Int

    # Expected sizes for bounds checking
    n_G::Int
    n_C::Int
    n_b_deferred::Int

    # Charge detection cache
    charge_is_vdep::Vector{Bool}
    charge_detection_pos::Int

    # Internal node indices for counter-based allocation (avoids Symbol interpolation)
    internal_node_indices::Vector{Int}
    internal_node_pos::Int

    # Warning flags for stamp overflow (when more stamps than detected)
    warned_G_overflow::Bool
    warned_C_overflow::Bool
    warned_b_overflow::Bool
end

"""
    create_direct_stamp_context(ctx::MNAContext, ...) -> DirectStampContext

Create a DirectStampContext from a completed MNAContext.

The context holds direct references to sparse matrix nzval arrays,
enabling zero-copy stamping.
"""
function create_direct_stamp_context(ctx::MNAContext, G_nzval::Vector{Float64},
                                     C_nzval::Vector{Float64}, b::Vector{Float64},
                                     G_mapping::Vector{Int}, C_mapping::Vector{Int},
                                     b_resolved::Vector{Int})
    n_G = length(ctx.G_V)
    n_C = length(ctx.C_V)
    n_b_deferred = length(ctx.b_V)

    # Collect internal node indices in allocation order for counter-based access
    # This avoids Symbol interpolation overhead during restamping
    internal_node_indices = findall(ctx.internal_node_flags)

    DirectStampContext(
        ctx.node_to_idx,
        ctx.n_nodes,
        ctx.n_currents,
        ctx.current_names,
        G_nzval,
        C_nzval,
        G_mapping,
        C_mapping,
        b,
        Vector{Float64}(undef, n_b_deferred),
        b_resolved,
        1, 1, 1, 1, 1,  # positions
        n_G, n_C, n_b_deferred,
        copy(ctx.charge_is_vdep),
        1,
        internal_node_indices,
        1,  # internal_node_pos
        false, false, false  # warning flags (G, C, b)
    )
end

"""
    get_current_idx(ctx::DirectStampContext, name::Symbol) -> CurrentIndex

Get the index of a current variable by name in DirectStampContext (for CCVS/CCCS restamping).
"""
function get_current_idx(ctx::DirectStampContext, name::Symbol)::CurrentIndex
    idx = findfirst(==(name), ctx.current_names)
    idx === nothing && error("Current variable $name not found in DirectStampContext")
    return CurrentIndex(idx)
end

get_current_idx(ctx::DirectStampContext, name::String) = get_current_idx(ctx, Symbol(name))

"""
    resolve_index(ctx::DirectStampContext, idx::MNAIndex) -> Int

Convert an MNA index (NodeIndex, CurrentIndex, ChargeIndex) to system row/column.
This mirrors the MNAContext version for DirectStampContext compatibility.
"""
@inline resolve_index(ctx::DirectStampContext, ::GroundIndex)::Int = 0
@inline resolve_index(ctx::DirectStampContext, idx::NodeIndex)::Int = idx.idx
@inline resolve_index(ctx::DirectStampContext, idx::CurrentIndex)::Int = ctx.n_nodes + idx.k
# ChargeIndex for DirectStampContext (charges come after currents)
@inline function resolve_index(ctx::DirectStampContext, idx::ChargeIndex)::Int
    return ctx.n_nodes + ctx.n_currents + idx.k
end

"""
    reset_direct_stamp!(dctx::DirectStampContext)

Reset counters and zero sparse matrix values for a new iteration.
"""
@inline function reset_direct_stamp!(dctx::DirectStampContext)
    # Reset counters
    dctx.G_pos = 1
    dctx.C_pos = 1
    dctx.b_deferred_pos = 1
    dctx.current_pos = 1
    dctx.charge_pos = 1
    dctx.charge_detection_pos = 1
    dctx.internal_node_pos = 1

    # Reset overflow warning flags (warn once per solve, not once per iteration)
    # Don't reset these here - we want to warn only once per solve
    # dctx.warned_G_overflow = false
    # dctx.warned_C_overflow = false

    # Zero sparse matrices and b vector
    fill!(dctx.G_nzval, 0.0)
    fill!(dctx.C_nzval, 0.0)
    fill!(dctx.b, 0.0)

    return nothing
end

#==============================================================================#
# DirectStampContext Stamp Methods - Direct to Sparse
#==============================================================================#

@inline function get_node!(dctx::DirectStampContext, name::Symbol)::Int
    (name === :gnd || name === Symbol("0") || name === Symbol("gnd!")) && return 0
    return dctx.node_to_idx[name]
end

@inline get_node!(dctx::DirectStampContext, name::String) = get_node!(dctx, Symbol(name))
@inline get_node!(dctx::DirectStampContext, idx::Int) = idx

@inline function alloc_internal_node!(dctx::DirectStampContext, name::Symbol)::Int
    # Counter-based access - ignores name to avoid dict lookup overhead
    # The name is still passed from the call site (unavoidable with current codegen)
    # but we don't use it. This eliminates the dict lookup cost.
    pos = dctx.internal_node_pos
    dctx.internal_node_pos = pos + 1
    return dctx.internal_node_indices[pos]
end

# Component-based version: avoids Symbol interpolation at call site
# For DirectStampContext, we use counter-based access and ignore the components
@inline function alloc_internal_node!(dctx::DirectStampContext, base_name::Symbol, instance_name::Symbol)::Int
    pos = dctx.internal_node_pos
    dctx.internal_node_pos = pos + 1
    return dctx.internal_node_indices[pos]
end

@inline alloc_internal_node!(dctx::DirectStampContext, name::String) = alloc_internal_node!(dctx, Symbol(name))

@inline function alloc_current!(dctx::DirectStampContext, name::Symbol)::CurrentIndex
    pos = dctx.current_pos
    dctx.current_pos = pos + 1
    return CurrentIndex(pos)
end

@inline alloc_current!(dctx::DirectStampContext, name::String) = alloc_current!(dctx, Symbol(name))

@inline function alloc_charge!(dctx::DirectStampContext, name::Symbol, p::Int, n::Int)::ChargeIndex
    pos = dctx.charge_pos
    dctx.charge_pos = pos + 1
    return ChargeIndex(pos)
end

@inline alloc_charge!(dctx::DirectStampContext, name::String, p::Int, n::Int) = alloc_charge!(dctx, Symbol(name), p, n)

# Component-based APIs: avoid Symbol construction at call site
# For DirectStampContext, names are IGNORED - uses counter-based access
# These exist so generated code can pass components without allocating Symbol

"""
    alloc_current!(dctx::DirectStampContext, prefix::Symbol, name::Symbol) -> CurrentIndex

Component-based current allocation that avoids Symbol interpolation at call site.
For DirectStampContext, both arguments are ignored - uses counter-based access.
For MNAContext, this builds: Symbol(prefix, name).

# Example
```julia
alloc_current!(ctx, :I_, :Vs)  # Zero allocation for DirectStampContext
```
"""
@inline function alloc_current!(dctx::DirectStampContext, prefix::Symbol, name::Symbol)::CurrentIndex
    pos = dctx.current_pos
    dctx.current_pos = pos + 1
    return CurrentIndex(pos)
end

"""
    alloc_current!(dctx::DirectStampContext, prefix::Symbol, name::Symbol, suffix::Symbol) -> CurrentIndex

Three-argument component-based current allocation for names like Symbol(:I_, :H1, :_in).
For DirectStampContext, all arguments are ignored - uses counter-based access.
For MNAContext, this builds: Symbol(prefix, name, suffix).
"""
@inline function alloc_current!(dctx::DirectStampContext, prefix::Symbol, name::Symbol, suffix::Symbol)::CurrentIndex
    pos = dctx.current_pos
    dctx.current_pos = pos + 1
    return CurrentIndex(pos)
end

"""
    alloc_charge!(dctx::DirectStampContext, prefix::Symbol, name::Symbol, p::Int, n::Int) -> ChargeIndex

Component-based charge allocation that avoids Symbol interpolation at call site.
For DirectStampContext, names are ignored - uses counter-based access.
"""
@inline function alloc_charge!(dctx::DirectStampContext, prefix::Symbol, name::Symbol, p::Int, n::Int)::ChargeIndex
    pos = dctx.charge_pos
    dctx.charge_pos = pos + 1
    return ChargeIndex(pos)
end

"""
    stamp_G!(dctx::DirectStampContext, i, j, val)

Stamp G matrix value DIRECTLY to sparse nzval using precomputed mapping.
No intermediate array - single memory write.
"""
@inline function stamp_G!(dctx::DirectStampContext, i, j, val)
    iszero(i) && return nothing
    iszero(j) && return nothing

    pos = dctx.G_pos
    dctx.G_pos = pos + 1

    # Bounds check - if we get more stamps than discovered, skip the extra ones
    # This can happen when PSP103VA has conditional code paths not taken during detection
    if pos > length(dctx.G_mapping)
        # Warn once per context
        if !dctx.warned_G_overflow
            @warn "DirectStampContext: more G stamps than detected (pos=$pos, expected=$(length(dctx.G_mapping))). Extra stamps ignored."
            dctx.warned_G_overflow = true
        end
        return nothing
    end

    # Direct write to sparse matrix nzval
    nz_idx = dctx.G_mapping[pos]
    if nz_idx > 0
        v = extract_value(val)
        dctx.G_nzval[nz_idx] += v
    end

    return nothing
end

"""
    stamp_C!(dctx::DirectStampContext, i, j, val)

Stamp C matrix value DIRECTLY to sparse nzval using precomputed mapping.
"""
@inline function stamp_C!(dctx::DirectStampContext, i, j, val)
    iszero(i) && return nothing
    iszero(j) && return nothing

    pos = dctx.C_pos
    dctx.C_pos = pos + 1

    # Bounds check - if we get more stamps than discovered, skip the extra ones
    if pos > length(dctx.C_mapping)
        if !dctx.warned_C_overflow
            @warn "DirectStampContext: more C stamps than detected (pos=$pos, expected=$(length(dctx.C_mapping))). Extra stamps ignored."
            dctx.warned_C_overflow = true
        end
        return nothing
    end

    nz_idx = dctx.C_mapping[pos]
    if nz_idx > 0
        v = extract_value(val)
        dctx.C_nzval[nz_idx] += v
    end

    return nothing
end

"""
    stamp_b!(dctx::DirectStampContext, i, val)

Stamp b vector value. All stamps are deferred and applied via pre-resolved indices.
This provides a consistent interface matching G and C stamping.
"""
@inline function stamp_b!(dctx::DirectStampContext, i, val)
    iszero(i) && return nothing
    v = extract_value(val)

    # All b stamps are deferred - applied via pre-resolved indices after stamping
    pos = dctx.b_deferred_pos
    dctx.b_deferred_pos = pos + 1

    # Bounds check
    if pos > length(dctx.b_V)
        if !dctx.warned_b_overflow
            @warn "DirectStampContext: more b stamps than detected (pos=$pos, expected=$(length(dctx.b_V))). Extra stamps ignored."
            dctx.warned_b_overflow = true
        end
        return nothing
    end

    dctx.b_V[pos] = v
    return nothing
end

"""
    stamp_b_ac!(dctx::DirectStampContext, i, val)

No-op for DirectStampContext. AC analysis uses MNAContext, not DirectStampContext.
DirectStampContext is optimized for transient restamping where AC stamps are not needed.
"""
@inline function stamp_b_ac!(dctx::DirectStampContext, i, val)
    # AC stamps are not used in transient analysis (DirectStampContext's purpose)
    return nothing
end

"""
    stamp_conductance!(dctx::DirectStampContext, p, n, G)

Stamp conductance pattern for 2-terminal element.
"""
@inline function stamp_conductance!(dctx::DirectStampContext, p::Int, n::Int, G)
    stamp_G!(dctx, p, p,  G)
    stamp_G!(dctx, p, n, -G)
    stamp_G!(dctx, n, p, -G)
    stamp_G!(dctx, n, n,  G)
    return nothing
end

"""
    stamp_capacitance!(dctx::DirectStampContext, p, n, C)

Stamp capacitance pattern for 2-terminal element.
"""
@inline function stamp_capacitance!(dctx::DirectStampContext, p::Int, n::Int, C)
    stamp_C!(dctx, p, p,  C)
    stamp_C!(dctx, p, n, -C)
    stamp_C!(dctx, n, p, -C)
    stamp_C!(dctx, n, n,  C)
    return nothing
end

#==============================================================================#
# Hoisted Stamping Primitives (for voltage-dependent conditionals)
#
# These functions separate allocation from value assignment, allowing
# allocations to be hoisted outside conditionals. Since allocations are
# hoisted, they execute in the same fixed order during both detection
# and runtime, so the positional counter works correctly.
#==============================================================================#

"""
    get_G_idx!(dctx::DirectStampContext, i, j) -> Int

Get the nzval index for G[i,j] using positional counter.
Since allocations are hoisted, the counter order matches detection order.

Returns 0 for ground indices or if position exceeds detected count.
"""
@inline function get_G_idx!(dctx::DirectStampContext, i, j)::Int
    iszero(i) && return 0
    iszero(j) && return 0
    pos = dctx.G_pos
    dctx.G_pos = pos + 1
    if pos > length(dctx.G_mapping)
        if !dctx.warned_G_overflow
            @warn "DirectStampContext: more G allocations than detected (pos=$pos, expected=$(length(dctx.G_mapping)))"
            dctx.warned_G_overflow = true
        end
        return 0
    end
    return dctx.G_mapping[pos]
end

"""
    stamp_G_at_idx!(dctx::DirectStampContext, idx::Int, val)

Stamp directly to G.nzval[idx] using pre-calculated index.
If idx <= 0 (e.g., from ground or overflow), the stamp is skipped.
"""
@inline function stamp_G_at_idx!(dctx::DirectStampContext, idx::Int, val)
    idx <= 0 && return nothing
    dctx.G_nzval[idx] += extract_value(val)
    return nothing
end

"""
    get_C_idx!(dctx::DirectStampContext, i, j) -> Int

Get the nzval index for C[i,j] using positional counter.
Since allocations are hoisted, the counter order matches detection order.

Returns 0 for ground indices or if position exceeds detected count.
"""
@inline function get_C_idx!(dctx::DirectStampContext, i, j)::Int
    iszero(i) && return 0
    iszero(j) && return 0
    pos = dctx.C_pos
    dctx.C_pos = pos + 1
    if pos > length(dctx.C_mapping)
        if !dctx.warned_C_overflow
            @warn "DirectStampContext: more C allocations than detected (pos=$pos, expected=$(length(dctx.C_mapping)))"
            dctx.warned_C_overflow = true
        end
        return 0
    end
    return dctx.C_mapping[pos]
end

"""
    stamp_C_at_idx!(dctx::DirectStampContext, idx::Int, val)

Stamp directly to C.nzval[idx] using pre-calculated index.
If idx <= 0 (e.g., from ground or overflow), the stamp is skipped.
"""
@inline function stamp_C_at_idx!(dctx::DirectStampContext, idx::Int, val)
    idx <= 0 && return nothing
    dctx.C_nzval[idx] += extract_value(val)
    return nothing
end

"""
    get_b_idx!(dctx::DirectStampContext, i) -> Int

Get the deferred b index using positional counter.
Since allocations are hoisted, the counter order matches detection order.

Returns 0 for ground index or if position exceeds detected count.
"""
@inline function get_b_idx!(dctx::DirectStampContext, i)::Int
    iszero(i) && return 0
    pos = dctx.b_deferred_pos
    dctx.b_deferred_pos = pos + 1
    if pos > length(dctx.b_V)
        if !dctx.warned_b_overflow
            @warn "DirectStampContext: more b allocations than detected (pos=$pos, expected=$(length(dctx.b_V)))"
            dctx.warned_b_overflow = true
        end
        return 0
    end
    return pos
end

"""
    stamp_b_at_idx!(dctx::DirectStampContext, idx::Int, val)

Stamp to deferred b value at pre-calculated index.
If idx <= 0 (e.g., from ground or overflow), the stamp is skipped.

Note: Uses assignment (=) not accumulation (+=) because b_V is not zeroed
in reset_direct_stamp! (unlike G_nzval/C_nzval which are zeroed).
Each b_V position represents a distinct deferred stamp, not a sum.
"""
@inline function stamp_b_at_idx!(dctx::DirectStampContext, idx::Int, val)
    idx <= 0 && return nothing
    dctx.b_V[idx] = extract_value(val)
    return nothing
end

"""
    stamp_voltage_contribution!(dctx::DirectStampContext, p::Int, n::Int, v_fn, x::AbstractVector, current_name::Symbol)

Zero-allocation voltage contribution stamping for DirectStampContext.
"""
@inline function stamp_voltage_contribution!(
    dctx::DirectStampContext,
    p::Int, n::Int,
    v_fn,
    x::AbstractVector,
    current_name::Symbol
)
    # Allocate current variable (counter-based, zero allocation)
    I_idx = alloc_current!(dctx, current_name)

    # Get voltage value
    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]
    Vpn = Vp - Vn
    v_val = v_fn(Vpn)

    # Stamp voltage source pattern
    stamp_G!(dctx, p, I_idx, 1.0)
    stamp_G!(dctx, n, I_idx, -1.0)
    stamp_G!(dctx, I_idx, p, 1.0)
    stamp_G!(dctx, I_idx, n, -1.0)
    stamp_b!(dctx, I_idx, v_val)

    return I_idx
end

"""
    stamp_voltage_contribution!(dctx::DirectStampContext, p::Int, n::Int, v_fn, x::AbstractVector, prefix::Symbol, name::Symbol)

Component-based API for zero allocation. Takes prefix and name separately
to avoid allocating a new Symbol during circuit rebuild.
"""
@inline function stamp_voltage_contribution!(
    dctx::DirectStampContext,
    p::Int, n::Int,
    v_fn,
    x::AbstractVector,
    prefix::Symbol, name::Symbol
)
    # Allocate current variable (counter-based, zero allocation)
    I_idx = alloc_current!(dctx, prefix, name)

    # Get voltage value
    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]
    Vpn = Vp - Vn
    v_val = v_fn(Vpn)

    # Stamp voltage source pattern
    stamp_G!(dctx, p, I_idx, 1.0)
    stamp_G!(dctx, n, I_idx, -1.0)
    stamp_G!(dctx, I_idx, p, 1.0)
    stamp_G!(dctx, I_idx, n, -1.0)
    stamp_b!(dctx, I_idx, v_val)

    return I_idx
end

@inline reset_for_restamping!(dctx::DirectStampContext) = reset_direct_stamp!(dctx)

#==============================================================================#
# AnyMNAContext: Union Type for Context Dispatch
#
# This allows device stamp! methods to work with either MNAContext
# (structure discovery) or DirectStampContext (fast restamping).
#==============================================================================#

"""
    AnyMNAContext

Union type alias for MNAContext or DirectStampContext.

Device stamp! methods use this to accept either context type, allowing
the same code to work for both initial structure discovery (MNAContext)
and zero-copy restamping (DirectStampContext).
"""
const AnyMNAContext = Union{MNAContext, DirectStampContext}
export AnyMNAContext

# Alias for backward compatibility
const AnyStampContext = AnyMNAContext
export AnyStampContext
