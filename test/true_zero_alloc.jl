#!/usr/bin/env julia
#==============================================================================#
# True Zero-Allocation Investigation
#
# Goal: Find and eliminate ALL allocations in step!
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CedarSim
using CedarSim.MNA
using Printf
using LinearAlgebra
using SparseArrays
using LinearSolve

println("=" ^ 70)
println("True Zero-Allocation Investigation")
println("=" ^ 70)

#==============================================================================#
# Zero-allocation circuit builder
#==============================================================================#

function build_rc_zero_alloc(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # Voltage source with component-based API (zero allocation)
    I_idx = MNA.alloc_current!(ctx, :I, :Vs)
    MNA.stamp_G!(ctx, vcc, I_idx,  1.0)
    MNA.stamp_G!(ctx, I_idx, vcc,  1.0)
    MNA.stamp_b!(ctx, I_idx, params.Vcc)

    MNA.stamp!(Resistor(params.R), ctx, vcc, out)
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    return ctx
end

#==============================================================================#
# Setup
#==============================================================================#

circuit = MNACircuit(build_rc_zero_alloc; Vcc=5.0, R=1000.0, C=1e-6)
ws = MNA.compile(circuit)
cs = ws.structure
n = MNA.system_size(cs)

println("\nSystem size: $n")

#==============================================================================#
# Test MNA core operations (should be truly zero)
#==============================================================================#

println("\n--- MNA Core Operations ---")

u = zeros(n)
du = zeros(n)
resid = zeros(n)
J = copy(cs.G)
t = 0.001
gamma = 1.0

# Extensive warmup
for _ in 1:1000
    MNA.fast_rebuild!(ws, u, t)
    MNA.fast_residual!(resid, du, u, ws, t)
    MNA.fast_jacobian!(J, du, u, ws, gamma, t)
end
GC.gc()

# Test with many iterations
N = 10000

allocs_rebuild = @allocated for _ in 1:N
    MNA.fast_rebuild!(ws, u, t)
end
@printf("fast_rebuild!:  %d bytes total, %.3f bytes/call\n", allocs_rebuild, allocs_rebuild/N)

allocs_residual = @allocated for _ in 1:N
    MNA.fast_residual!(resid, du, u, ws, t)
end
@printf("fast_residual!: %d bytes total, %.3f bytes/call\n", allocs_residual, allocs_residual/N)

allocs_jacobian = @allocated for _ in 1:N
    MNA.fast_jacobian!(J, du, u, ws, gamma, t)
end
@printf("fast_jacobian!: %d bytes total, %.3f bytes/call\n", allocs_jacobian, allocs_jacobian/N)

#==============================================================================#
# Test LinearSolve operations
#==============================================================================#

println("\n--- LinearSolve Operations ---")

# Pre-allocate
A = copy(cs.G)
b_rhs = zeros(n)
u_sol = zeros(n)

# Initialize with DC
MNA.fast_rebuild!(ws, u, 0.0)
u .= cs.G \ ws.dctx.b

dt = 1e-8
inv_dt = 1.0 / dt

# Form A = G + C/dt
A_nz = nonzeros(A)
G_nz = nonzeros(cs.G)
C_nz = nonzeros(cs.C)
@inbounds for i in eachindex(A_nz)
    A_nz[i] = G_nz[i] + inv_dt * C_nz[i]
end

# Create LinearSolve cache
linprob = LinearProblem(A, b_rhs)
lincache = init(linprob, KLUFactorization())

# Warmup
for _ in 1:100
    mul!(b_rhs, cs.C, u)
    b_rhs .*= inv_dt
    b_rhs .+= ws.dctx.b
    lincache.A = A
    sol = solve!(lincache)
    copyto!(u, sol.u)
end
GC.gc()

# Test solve! alone
allocs_solve = @allocated for _ in 1:N
    sol = solve!(lincache)
end
@printf("\nsolve!(lincache): %d bytes total, %.3f bytes/call\n", allocs_solve, allocs_solve/N)

# Test lincache.A assignment
allocs_assign = @allocated for _ in 1:N
    lincache.A = A
end
@printf("lincache.A = A:   %d bytes total, %.3f bytes/call\n", allocs_assign, allocs_assign/N)

# Test copyto! from solution
sol = solve!(lincache)
allocs_copy = @allocated for _ in 1:N
    copyto!(u, sol.u)
end
@printf("copyto!(u, sol.u): %d bytes total, %.3f bytes/call\n", allocs_copy, allocs_copy/N)

#==============================================================================#
# Test pure sparse LU without LinearSolve wrapper
#==============================================================================#

println("\n--- Pure SparseArrays LU ---")

using SparseArrays: lu, lu!

# Create LU factorization
A_lu = lu(A)

# Test lu! (in-place refactorization)
for _ in 1:100
    lu!(A_lu, A)
end
GC.gc()

allocs_lu = @allocated for _ in 1:N
    lu!(A_lu, A)
end
@printf("\nlu!(A_lu, A):   %d bytes total, %.3f bytes/call\n", allocs_lu, allocs_lu/N)

# Test ldiv! (in-place solve)
for _ in 1:100
    ldiv!(u_sol, A_lu, b_rhs)
end
GC.gc()

allocs_ldiv = @allocated for _ in 1:N
    ldiv!(u_sol, A_lu, b_rhs)
end
@printf("ldiv!(u, A_lu, b): %d bytes total, %.3f bytes/call\n", allocs_ldiv, allocs_ldiv/N)

#==============================================================================#
# Complete step with pure sparse operations
#==============================================================================#

println("\n--- Complete Step with Pure Sparse Operations ---")

function pure_sparse_step!(u, ws, cs, A, A_nz, G_nz, C_nz, A_lu, b_work, inv_dt, t)
    # Rebuild circuit
    MNA.fast_rebuild!(ws, u, t)

    # Form A = G + C/dt
    @inbounds for i in eachindex(A_nz)
        A_nz[i] = G_nz[i] + inv_dt * C_nz[i]
    end

    # Refactorize
    lu!(A_lu, A)

    # Form rhs = C/dt * u + b
    mul!(b_work, cs.C, u)
    @inbounds for i in eachindex(b_work)
        b_work[i] = b_work[i] * inv_dt + ws.dctx.b[i]
    end

    # Solve in-place
    ldiv!(u, A_lu, b_work)

    return nothing
end

b_work = zeros(n)
A_lu = lu(A)

# Warmup
for _ in 1:1000
    pure_sparse_step!(u, ws, cs, A, A_nz, G_nz, C_nz, A_lu, b_work, inv_dt, t)
end
GC.gc()

allocs_step = @allocated for _ in 1:N
    pure_sparse_step!(u, ws, cs, A, A_nz, G_nz, C_nz, A_lu, b_work, inv_dt, t)
end
@printf("\npure_sparse_step!: %d bytes total, %.3f bytes/call\n", allocs_step, allocs_step/N)

#==============================================================================#
# Try with pre-extracted nzval refs to avoid getproperty
#==============================================================================#

println("\n--- Step with Pre-extracted References ---")

# Pre-extract everything
ws_dctx_b = ws.dctx.b
cs_C = cs.C

function preextract_step!(u, ws, cs, A, A_nz, G_nz, C_nz, A_lu, b_work, inv_dt, t, ws_dctx_b, cs_C)
    # Rebuild circuit
    MNA.fast_rebuild!(ws, u, t)

    # Form A = G + C/dt
    @inbounds for i in eachindex(A_nz)
        A_nz[i] = G_nz[i] + inv_dt * C_nz[i]
    end

    # Refactorize
    lu!(A_lu, A)

    # Form rhs = C/dt * u + b
    mul!(b_work, cs_C, u)
    @inbounds for i in eachindex(b_work)
        b_work[i] = b_work[i] * inv_dt + ws_dctx_b[i]
    end

    # Solve in-place
    ldiv!(u, A_lu, b_work)

    return nothing
end

# Warmup
for _ in 1:1000
    preextract_step!(u, ws, cs, A, A_nz, G_nz, C_nz, A_lu, b_work, inv_dt, t, ws_dctx_b, cs_C)
end
GC.gc()

allocs_preextract = @allocated for _ in 1:N
    preextract_step!(u, ws, cs, A, A_nz, G_nz, C_nz, A_lu, b_work, inv_dt, t, ws_dctx_b, cs_C)
end
@printf("\npreextract_step!: %d bytes total, %.3f bytes/call\n", allocs_preextract, allocs_preextract/N)

#==============================================================================#
# Try KLU directly via SuiteSparse
#==============================================================================#

println("\n--- Direct KLU Usage ---")

# KLU is accessed through SparseArrays.UMFPACK or we need to use the wrapper
# Let's check if there's a way to use KLU with truly zero allocations

# Actually, let's check what lu! returns
println("\nChecking lu! return type:")
result = lu!(A_lu, A)
println("  Returns: $(typeof(result))")
println("  Same object? $(result === A_lu)")

#==============================================================================#
# Summary
#==============================================================================#

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

println("\nMNA Core (per call):")
@printf("  fast_rebuild!:  %.3f bytes\n", allocs_rebuild/N)
@printf("  fast_residual!: %.3f bytes\n", allocs_residual/N)
@printf("  fast_jacobian!: %.3f bytes\n", allocs_jacobian/N)

println("\nLinear Solve (per call):")
@printf("  lu!:   %.3f bytes\n", allocs_lu/N)
@printf("  ldiv!: %.3f bytes\n", allocs_ldiv/N)

println("\nComplete Step (per call):")
@printf("  pure_sparse_step!: %.3f bytes\n", allocs_step/N)
@printf("  preextract_step!:  %.3f bytes\n", allocs_preextract/N)

if allocs_preextract == 0
    println("\n✓ TRUE ZERO ALLOCATION ACHIEVED!")
else
    println("\n✗ Still allocating - need to investigate further")
end

println()
