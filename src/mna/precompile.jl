#==============================================================================#
# MNA Optimization: Precompiled Circuit Evaluation
#
# This module provides optimized circuit evaluation by separating structure
# discovery (once) from value updates (every iteration).
#
# Key concepts:
# - CompiledStructure: Immutable circuit structure (sparsity pattern, mappings)
# - EvalWorkspace: Mutable workspace for per-iteration values
# - COO→CSC mapping: Maps COO indices to sparse matrix nonzero positions
# - Fast stamping: Devices write to preallocated COO storage
#
# Inspired by OpenVAF's OSDI implementation which uses direct pointers to
# matrix entries for maximum performance.
#
# Architecture (zero-allocation iteration):
#   CompiledStructure (immutable, shared)
#       ↓
#   EvalWorkspace (mutable, per-thread)
#       ↓
#   fast_rebuild!(ws, u, t) → no allocation!
#==============================================================================#

using SparseArrays
using LinearAlgebra
using ForwardDiff: Dual, value
using StaticArrays: SVector, MVector

# Extract real value from ForwardDiff.Dual (for tgrad compatibility)
# Needed for time-dependent sources with Rosenbrock solvers
real_time(t::Real) = Float64(t)
real_time(t::Dual) = Float64(value(t))

export CompiledStructure, EvalWorkspace, compile_structure, create_workspace
export fast_residual!, fast_jacobian!

#==============================================================================#
# Sparse Matrix Utilities for Zero-Allocation Jacobian
#==============================================================================#

"""
    _pad_to_pattern(M::SparseMatrixCSC, pattern::SparseMatrixCSC) -> SparseMatrixCSC

Expand sparse matrix M to have the sparsity pattern of `pattern`.
Entries in `pattern` not in `M` are zero.
"""
function _pad_to_pattern(M::SparseMatrixCSC{Tv}, pattern::SparseMatrixCSC) where Tv
    result = similar(pattern, Tv)
    fill!(nonzeros(result), zero(Tv))
    I, J, V = findnz(M)
    result[CartesianIndex.(I, J)] .= V
    return result
end

#==============================================================================#
# CompiledStructure: Immutable Circuit Structure
#==============================================================================#

"""
    CompiledStructure{F,P,S,M}

Immutable compiled circuit structure with parameterized matrix type.

This separation enables:
1. **Compiler optimization**: Fields can be constant-propagated
2. **Thread safety**: Can be shared across threads (each has own EvalWorkspace)
3. **Clear semantics**: Immutable parts shared, mutable parts per-evaluation

Type parameter `M` controls matrix storage:
- `SparseMatrixCSC{Float64,Int}`: sparse storage (default, large circuits)
- `Matrix{Float64}`: dense storage (small circuits, zero-alloc OrdinaryDiffEq)

Use `compile(circuit; dense=true)` to get dense matrices.
"""
struct CompiledStructure{F,P,S,M<:AbstractMatrix{Float64}}
    # Builder and parameters
    builder::F
    params::P
    spec::S

    # System dimensions (fixed)
    n::Int
    n_nodes::Int
    n_currents::Int

    # Node/current names for solution interpretation
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}

    # Fixed stamp position mappings
    # For sparse: COO position → nzval index
    # For dense: COO position → linear index (i + (j-1)*n)
    G_coo_to_idx::Vector{Int}
    C_coo_to_idx::Vector{Int}
    G_n_coo::Int
    C_n_coo::Int

    # Original COO (i,j) indices for rebuilding dense matrices
    # Only populated when M is Matrix{Float64}
    G_I::Vector{Int}
    G_J::Vector{Int}
    C_I::Vector{Int}
    C_J::Vector{Int}

    # Matrices (sparse or dense depending on M)
    G::M
    C::M

    # Resolved indices for deferred b stamps (CurrentIndex/ChargeIndex → actual positions)
    # Pre-computed during compilation for zero-allocation value-only mode
    b_deferred_resolved::Vector{Int}
    n_b_deferred::Int

    # Precomputed G diagonal indices for voltage nodes (for gshunt application)
    # For sparse: index into G.nzval
    # For dense: linear index (i + (i-1)*n)
    G_diag_idx::Vector{Int}
end

"""
    system_size(cs::CompiledStructure) -> Int

Return the system size (number of unknowns).
"""
system_size(cs::CompiledStructure) = cs.n

"""
    _get_storage(A::AbstractMatrix) -> AbstractVector

Get the underlying storage vector for direct stamping.
- For SparseMatrixCSC: returns nzval (the nonzero values array)
- For Matrix: returns vec(A) (column-major linear view)
"""
_get_storage(A::SparseMatrixCSC) = nonzeros(A)
_get_storage(A::Matrix) = vec(A)

#==============================================================================#
# EvalWorkspace: Zero-Copy Stamping Workspace
#
# Uses DirectStampContext which stamps directly to sparse matrix nzval,
# eliminating ALL intermediate arrays. Optimal for all circuit sizes.
#
# Data flow:
#   builder stamps → DirectStampContext → G.nzval/C.nzval (single write)
#
# No vctx.G_V, no intermediate copying between arrays.
#==============================================================================#

"""
    EvalWorkspace{T,CS}

Immutable evaluation workspace with zero-copy DirectStampContext.

Stamps go directly to matrix storage - no intermediate storage.
Works with both sparse (nzval) and dense matrices.

# Fields
- `structure::CS`: Compiled circuit structure (sparse or dense)
- `dctx::DirectStampContext`: Direct stamping context
- `resid_tmp::Vector{T}`: Working storage for residual computation
"""
struct EvalWorkspace{T,CS<:CompiledStructure}
    structure::CS
    dctx::DirectStampContext
    resid_tmp::Vector{T}
end

"""
    create_workspace(cs::CompiledStructure; ctx=nothing) -> EvalWorkspace

Create a workspace that stamps directly to matrix storage.

This is the single recommended API - it works optimally for all circuit sizes:
- No intermediate G_V, C_V arrays
- Stamps go straight to matrix storage (sparse nzval or dense matrix)
- Deferred b stamps resolved using precomputed mapping

# Arguments
- `cs`: Compiled structure to create workspace for
- `ctx`: Optional MNAContext to use (for voltage-dependent capacitor detection)

If `ctx` is provided, it will be used for the DirectStampContext (including its
detection cache). This is important for voltage-dependent capacitor detection:
if ZERO_VECTOR is used to build the context, reactive branches like ddt(Q(V))
may return scalars instead of Duals, causing incorrect detection cache.
"""
function create_workspace(cs::CompiledStructure{F,P,S,M}; ctx::Union{MNAContext, Nothing}=nothing) where {F,P,S,M}
    # Use provided context or rebuild (fallback for backward compatibility)
    if ctx === nothing
        ctx = cs.builder(cs.params, cs.spec, 0.0; x=ZERO_VECTOR)
    end

    # Create b vector
    b = zeros(Float64, cs.n)

    # Get storage view - works for both sparse (nzval) and dense (vec)
    G_storage = _get_storage(cs.G)
    C_storage = _get_storage(cs.C)

    # Create DirectStampContext with references to storage
    dctx = create_direct_stamp_context(
        ctx,
        G_storage,
        C_storage,
        b,
        cs.G_coo_to_idx,
        cs.C_coo_to_idx,
        cs.b_deferred_resolved
    )

    EvalWorkspace{Float64,typeof(cs)}(
        cs,
        dctx,
        zeros(Float64, cs.n)
    )
end

"""
    system_size(ws::EvalWorkspace) -> Int

Return the system size from the workspace's structure.
"""
system_size(ws::EvalWorkspace) = system_size(ws.structure)


#==============================================================================#
# COO to CSC Mapping
#==============================================================================#

"""
    compute_coo_to_nz_mapping(I, J, S::SparseMatrixCSC) -> Vector{Int}

Compute mapping from COO indices to positions in `nonzeros(S)`.

For each COO entry `(I[k], J[k])`, finds its index in `nonzeros(S)`.
This handles duplicate entries by finding the correct position where
the value should be accumulated.

# Algorithm
For each COO entry at position k:
1. Get column j = J[k]
2. Search column j's nonzero range for row i = I[k]
3. Store the nonzero index

This is O(nnz * avg_col_nnz) but only done once during compilation.
"""
function compute_coo_to_nz_mapping(I::Vector{Int}, J::Vector{Int}, S::SparseMatrixCSC)
    n_coo = length(I)
    mapping = zeros(Int, n_coo)

    rowval = rowvals(S)
    colptr = S.colptr

    for k in 1:n_coo
        i, j = I[k], J[k]

        # Skip ground entries (shouldn't happen, but be safe)
        if i == 0 || j == 0
            continue
        end

        # Search column j for row i
        for idx in colptr[j]:(colptr[j+1]-1)
            if rowval[idx] == i
                mapping[k] = idx
                break
            end
        end

        # Sanity check: mapping should be found
        if mapping[k] == 0 && i != 0 && j != 0
            error("COO entry ($i, $j) not found in sparse matrix at k=$k")
        end
    end

    return mapping
end

#==============================================================================#
# Circuit Compilation
#==============================================================================#

"""
    compile_structure(builder, params, spec; dense=false) -> CompiledStructure

Compile a circuit builder into an immutable CompiledStructure.

This performs structure discovery by calling the builder once,
then creates the matrices and stamp mappings.

# Arguments
- `builder`: Circuit builder function with signature:
    `(params, spec, t::Real=0.0; x=ZERO_VECTOR) -> MNAContext`
  Time is passed explicitly for zero-allocation iteration.
- `params`: Circuit parameters (NamedTuple)
- `spec`: Simulation specification (MNASpec)
- `ctx`: Optional pre-built context with detection cache. If provided,
  this context is used instead of building fresh.
- `dense`: If true, use dense matrices instead of sparse. This enables
  zero-allocation Jacobian operations with OrdinaryDiffEq.

# Returns
An immutable `CompiledStructure{F,P,S,M}` where `M` is `Matrix{Float64}` if
`dense=true`, otherwise `SparseMatrixCSC{Float64,Int}`.
"""
function compile_structure(builder::F, params::P, spec::S;
                          ctx::Union{MNAContext, Nothing}=nothing,
                          dense::Bool=false) where {F,P,S}
    # Use provided context or build fresh
    if ctx === nothing
        ctx0 = builder(params, spec, 0.0; x=ZERO_VECTOR)
    else
        ctx0 = ctx
    end
    n = system_size(ctx0)

    if n == 0
        # Empty circuit - return minimal structure
        M = dense ? Matrix{Float64} : SparseMatrixCSC{Float64,Int}
        G_empty = dense ? zeros(0, 0) : spzeros(0, 0)
        C_empty = dense ? zeros(0, 0) : spzeros(0, 0)
        return CompiledStructure{F,P,S,M}(
            builder, params, spec,
            0, 0, 0,
            Symbol[], Symbol[],
            Int[], Int[], 0, 0,
            Int[], Int[], Int[], Int[],  # Empty COO indices
            G_empty, C_empty,
            Int[], 0,
            Int[]
        )
    end

    # Resolve typed indices to actual matrix positions
    G_I_resolved = Int[resolve_index(ctx0, i) for i in ctx0.G_I]
    G_J_resolved = Int[resolve_index(ctx0, j) for j in ctx0.G_J]
    C_I_resolved = Int[resolve_index(ctx0, i) for i in ctx0.C_I]
    C_J_resolved = Int[resolve_index(ctx0, j) for j in ctx0.C_J]

    n_G = length(ctx0.G_I)
    n_C = length(ctx0.C_I)

    # Resolve b stamp indices
    n_b_deferred = length(ctx0.b_I)
    b_deferred_resolved = Vector{Int}(undef, n_b_deferred)
    for k in 1:n_b_deferred
        idx_typed = ctx0.b_I[k]
        b_deferred_resolved[k] = if idx_typed isa NodeIndex
            idx_typed.idx
        elseif idx_typed isa CurrentIndex
            ctx0.n_nodes + idx_typed.k
        elseif idx_typed isa ChargeIndex
            ctx0.n_nodes + ctx0.n_currents + idx_typed.k
        else
            0  # GroundIndex - skip
        end
    end

    if dense
        # Dense matrix compilation
        G = zeros(n, n)
        C = zeros(n, n)

        # Apply initial values
        for k in 1:n_G
            i, j = G_I_resolved[k], G_J_resolved[k]
            if i > 0 && j > 0
                G[i, j] += ctx0.G_V[k]
            end
        end
        for k in 1:n_C
            i, j = C_I_resolved[k], C_J_resolved[k]
            if i > 0 && j > 0
                C[i, j] += ctx0.C_V[k]
            end
        end

        # Compute linear indices for stamping: idx = i + (j-1)*n
        G_coo_to_idx = Int[i > 0 && j > 0 ? i + (j-1)*n : 0
                          for (i, j) in zip(G_I_resolved, G_J_resolved)]
        C_coo_to_idx = Int[i > 0 && j > 0 ? i + (j-1)*n : 0
                          for (i, j) in zip(C_I_resolved, C_J_resolved)]

        # Diagonal indices for gshunt
        G_diag_idx = Int[i + (i-1)*n for i in 1:ctx0.n_nodes]

        return CompiledStructure{F,P,S,Matrix{Float64}}(
            builder, params, spec,
            n, ctx0.n_nodes, ctx0.n_currents,
            copy(ctx0.node_names), copy(ctx0.current_names),
            G_coo_to_idx, C_coo_to_idx,
            n_G, n_C,
            G_I_resolved, G_J_resolved, C_I_resolved, C_J_resolved,
            G, C,
            b_deferred_resolved, n_b_deferred,
            G_diag_idx
        )
    else
        # Sparse matrix compilation (original path)
        G_raw = sparse(G_I_resolved, G_J_resolved, ctx0.G_V, n, n)
        C_raw = sparse(C_I_resolved, C_J_resolved, ctx0.C_V, n, n)

        # Create UNIFIED sparsity pattern for zero-allocation J = G + gamma*C
        jac_pattern = sparse(
            vcat(G_I_resolved, C_I_resolved),
            vcat(G_J_resolved, C_J_resolved),
            ones(length(G_I_resolved) + length(C_I_resolved)), n, n)

        # Pad G and C to match the unified pattern
        G = _pad_to_pattern(G_raw, jac_pattern)
        C = _pad_to_pattern(C_raw, jac_pattern)

        # COO→nzval mappings for sparse matrices
        G_coo_to_idx = compute_coo_to_nz_mapping(G_I_resolved, G_J_resolved, G)
        C_coo_to_idx = compute_coo_to_nz_mapping(C_I_resolved, C_J_resolved, C)

        # Diagonal nzval indices for gshunt
        G_diag_idx = _compute_diag_nz_indices(G, ctx0.n_nodes)

        return CompiledStructure{F,P,S,SparseMatrixCSC{Float64,Int}}(
            builder, params, spec,
            n, ctx0.n_nodes, ctx0.n_currents,
            copy(ctx0.node_names), copy(ctx0.current_names),
            G_coo_to_idx, C_coo_to_idx,
            n_G, n_C,
            Int[], Int[], Int[], Int[],  # Empty COO indices for sparse
            G, C,
            b_deferred_resolved, n_b_deferred,
            G_diag_idx
        )
    end
end

"""
    _compute_diag_nz_indices(A::SparseMatrixCSC, n_diag::Int) -> Vector{Int}

Compute nzval indices for diagonal elements A[i,i] for i in 1:n_diag.
Returns vector where result[i] = nzval index for A[i,i], or 0 if not present.
"""
function _compute_diag_nz_indices(A::SparseMatrixCSC, n_diag::Int)
    diag_nz = zeros(Int, n_diag)
    colptr = A.colptr
    rowval = A.rowval

    for col in 1:n_diag
        # Find diagonal element in this column (row == col)
        for nz_idx in colptr[col]:(colptr[col+1]-1)
            if rowval[nz_idx] == col
                diag_nz[col] = nz_idx
                break
            end
        end
    end

    return diag_nz
end


#==============================================================================#
# EvalWorkspace Fast Evaluation (Zero-Allocation Path)
#==============================================================================#

"""
    fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)

Zero-copy rebuild using DirectStampContext.

Stamps go directly to sparse matrix nzval - no intermediate arrays.
"""
function fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)
    fast_rebuild!(ws, ws.structure, u, t)
end

"""
    fast_rebuild!(ws::EvalWorkspace, cs::CompiledStructure, u::AbstractVector, t::Real)

Zero-copy rebuild using DirectStampContext with an explicit CompiledStructure.

This variant allows passing a different CompiledStructure (e.g., with modified spec
for dcop mode) while still using the same workspace for stamping.
"""
function fast_rebuild!(ws::EvalWorkspace, cs::CompiledStructure, u::AbstractVector, t::Real)
    dctx = ws.dctx

    # Reset counters and zero matrices
    reset_direct_stamp!(dctx)

    # Builder stamps directly to matrix storage via DirectStampContext
    cs.builder(cs.params, cs.spec, real_time(t); x=u, ctx=dctx)

    # Apply deferred b stamps
    n_deferred = cs.n_b_deferred
    for k in 1:n_deferred
        idx = dctx.b_resolved[k]
        if idx > 0
            dctx.b[idx] += dctx.b_V[k]
        end
    end

    # Apply srcFact scaling to b vector (for source stepping homotopy)
    srcFact = cs.spec.srcFact
    if srcFact < 1.0
        dctx.b .*= srcFact
    end

    # Apply gshunt to voltage node diagonals (for GMIN stepping / floating node stabilization)
    gshunt = cs.spec.gshunt
    if gshunt != 0.0
        G_storage = dctx.G_nzval  # Works for both sparse nzval and dense vec()
        G_diag_idx = cs.G_diag_idx
        @inbounds for i in 1:cs.n_nodes
            idx = G_diag_idx[i]
            if idx > 0
                G_storage[idx] += gshunt
            end
        end
    end

    return nothing
end

"""
    fast_residual!(resid, du, u, ws::EvalWorkspace, t)

Fast DAE residual evaluation using EvalWorkspace.

Computes: F(du, u, t) = C*du + G*u - b = 0
"""
function fast_residual!(resid::AbstractVector, du::AbstractVector,
                        u::AbstractVector, ws::EvalWorkspace, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # F(du, u) = C*du + G*u - b = 0
    mul!(resid, cs.C, du)
    mul!(resid, cs.G, u, 1.0, 1.0)
    resid .-= ws.dctx.b

    return nothing
end

"""
    fast_jacobian!(J, du, u, ws::EvalWorkspace, gamma, t)

Fast DAE Jacobian computation using EvalWorkspace: J = G + gamma*C

Dispatches based on the structure's matrix type:
- For sparse structures: zero-allocation nzval operations
- For dense structures: zero-allocation element-wise operations
"""
function fast_jacobian!(J::SparseMatrixCSC, du::AbstractVector,
                        u::AbstractVector,
                        ws::EvalWorkspace{T,CS},
                        gamma::Real, t::Real) where {T,CS<:CompiledStructure{<:Any,<:Any,<:Any,SparseMatrixCSC{Float64,Int}}}
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # J = G + gamma*C via direct nzval operations (zero allocation)
    # This works because G and C have been padded to the same sparsity pattern
    J_nz = nonzeros(J)
    G_nz = nonzeros(cs.G)
    C_nz = nonzeros(cs.C)
    @inbounds for i in eachindex(J_nz, G_nz, C_nz)
        J_nz[i] = G_nz[i] + gamma * C_nz[i]
    end

    return nothing
end

# Dense structure with dense J matrix (zero-allocation)
function fast_jacobian!(J::Matrix, du::AbstractVector,
                        u::AbstractVector,
                        ws::EvalWorkspace{T,CS},
                        gamma::Real, t::Real) where {T,CS<:CompiledStructure{<:Any,<:Any,<:Any,Matrix{Float64}}}
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # J = G + gamma*C via direct element access (zero allocation)
    G = cs.G
    C = cs.C
    @inbounds for i in eachindex(J, G, C)
        J[i] = G[i] + gamma * C[i]
    end

    return nothing
end

# Sparse structure with dense J matrix (conversion required - allocates)
function fast_jacobian!(J::Matrix, du::AbstractVector,
                        u::AbstractVector,
                        ws::EvalWorkspace{T,CS},
                        gamma::Real, t::Real) where {T,CS<:CompiledStructure{<:Any,<:Any,<:Any,SparseMatrixCSC{Float64,Int}}}
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # Convert sparse to dense (allocates - use dense compilation to avoid this)
    copyto!(J, Matrix(cs.G))
    J .+= gamma .* Matrix(cs.C)

    return nothing
end


"""
    b_vector(ws::EvalWorkspace) -> Vector{Float64}

Get the b vector from an EvalWorkspace.
"""
b_vector(ws::EvalWorkspace) = ws.dctx.b

#==============================================================================#
# OrdinaryDiffEq Integration Helpers
#==============================================================================#

export blind_step!

"""
    blind_step!(integrator)

Zero-allocation step for OrdinaryDiffEq integrators.

This wrapper discards the return value from `step!()`, which allows Julia's
compiler to optimize away the 16-byte ReturnCode allocation.

# Example
```julia
# Instead of:
step!(integrator)  # 16 bytes/call

# Use:
MNA.blind_step!(integrator)  # 0 bytes/call
```

Access the solution state via `integrator.u` after stepping.
"""
function blind_step!(integrator)
    step!(integrator)
    return nothing
end
