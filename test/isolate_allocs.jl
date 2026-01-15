#!/usr/bin/env julia
#==============================================================================#
# Isolate Allocations - Determine if per-call or background noise
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CedarSim
using CedarSim.MNA
using Printf
using LinearAlgebra
using SparseArrays

println("=" ^ 70)
println("Isolating Allocations")
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

u = zeros(n)
du = zeros(n)
resid = zeros(n)
J = copy(cs.G)
t = 0.001
gamma = 1.0

println("\nSystem size: $n")

#==============================================================================#
# Test: Check if allocations scale with iteration count
#==============================================================================#

println("\n--- Testing if allocations scale with iterations ---")

# Warmup
for _ in 1:1000
    MNA.fast_rebuild!(ws, u, t)
end
GC.gc()
sleep(0.1)  # Let GC settle

# Test with different iteration counts
for N in [1, 10, 100, 1000, 10000]
    GC.gc()
    sleep(0.01)

    allocs = @allocated for _ in 1:N
        MNA.fast_rebuild!(ws, u, t)
    end

    @printf("N=%5d: total=%8d bytes, per_call=%.3f bytes\n", N, allocs, allocs/N)
end

#==============================================================================#
# Test: Multiple runs to check consistency
#==============================================================================#

println("\n--- Multiple runs for consistency check ---")

N = 1000
for run in 1:5
    GC.gc()
    sleep(0.01)

    allocs = @allocated for _ in 1:N
        MNA.fast_rebuild!(ws, u, t)
    end

    @printf("Run %d: total=%8d bytes, per_call=%.3f bytes\n", run, allocs, allocs/N)
end

#==============================================================================#
# Test: Single call allocation
#==============================================================================#

println("\n--- Single call allocation check ---")

for _ in 1:100
    MNA.fast_rebuild!(ws, u, t)
end
GC.gc()
sleep(0.1)

for i in 1:10
    allocs = @allocated MNA.fast_rebuild!(ws, u, t)
    @printf("Single call %d: %d bytes\n", i, allocs)
end

#==============================================================================#
# Test: Check with GC disabled
#==============================================================================#

println("\n--- With GC disabled ---")

GC.gc()
GC.enable(false)

try
    for i in 1:5
        allocs = @allocated for _ in 1:1000
            MNA.fast_rebuild!(ws, u, t)
        end
        @printf("Run %d (GC disabled): total=%8d bytes, per_call=%.3f bytes\n", i, allocs, allocs/1000)
    end
finally
    GC.enable(true)
end

#==============================================================================#
# Test: Sparse LU operations
#==============================================================================#

println("\n--- Sparse LU operations ---")

using SparseArrays: lu, lu!

A = copy(cs.G)
A_lu = lu(A)
b_work = zeros(n)
u_sol = zeros(n)

# Warmup
for _ in 1:100
    lu!(A_lu, A)
    ldiv!(u_sol, A_lu, b_work)
end
GC.gc()
sleep(0.1)

println("\nlu! single calls:")
for i in 1:5
    allocs = @allocated lu!(A_lu, A)
    @printf("  Call %d: %d bytes\n", i, allocs)
end

println("\nldiv! single calls:")
for i in 1:5
    allocs = @allocated ldiv!(u_sol, A_lu, b_work)
    @printf("  Call %d: %d bytes\n", i, allocs)
end

#==============================================================================#
# Test: Dense LU for comparison
#==============================================================================#

println("\n--- Dense LU for comparison ---")

A_dense = Matrix(A)
A_lu_dense = lu(A_dense)
b_dense = zeros(n)
u_dense = zeros(n)

# Warmup
for _ in 1:100
    lu!(A_lu_dense, A_dense)
    ldiv!(u_dense, A_lu_dense, b_dense)
end
GC.gc()
sleep(0.1)

println("\nDense lu! single calls:")
for i in 1:5
    allocs = @allocated lu!(A_lu_dense, A_dense)
    @printf("  Call %d: %d bytes\n", i, allocs)
end

println("\nDense ldiv! single calls:")
for i in 1:5
    allocs = @allocated ldiv!(u_dense, A_lu_dense, b_dense)
    @printf("  Call %d: %d bytes\n", i, allocs)
end

#==============================================================================#
# Test: Complete zero-alloc step with dense LU
#==============================================================================#

println("\n--- Complete step with dense LU ---")

A_dense = Matrix(cs.G)
G_dense = Matrix(cs.G)
C_dense = Matrix(cs.C)
A_lu_dense = lu(A_dense)
b_dense = zeros(n)
u_dense = zeros(n)
dt = 1e-8
inv_dt = 1.0 / dt

# Initialize
MNA.fast_rebuild!(ws, u_dense, 0.0)
u_dense .= G_dense \ ws.dctx.b

function dense_step!(u, ws, G, C, A, A_lu, b, inv_dt, t)
    MNA.fast_rebuild!(ws, u, t)

    # A = G + C/dt
    @inbounds for i in eachindex(A)
        A[i] = G[i] + inv_dt * C[i]
    end

    # Refactorize
    lu!(A_lu, A)

    # b = C/dt * u + ws.dctx.b
    mul!(b, C, u)
    @inbounds for i in eachindex(b)
        b[i] = b[i] * inv_dt + ws.dctx.b[i]
    end

    # Solve
    ldiv!(u, A_lu, b)

    return nothing
end

# Warmup
for _ in 1:1000
    dense_step!(u_dense, ws, G_dense, C_dense, A_dense, A_lu_dense, b_dense, inv_dt, t)
end
GC.gc()
sleep(0.1)

println("\nDense step single calls:")
for i in 1:10
    allocs = @allocated dense_step!(u_dense, ws, G_dense, C_dense, A_dense, A_lu_dense, b_dense, inv_dt, t)
    @printf("  Call %d: %d bytes\n", i, allocs)
end

println("\nDense step with GC disabled:")
GC.gc()
GC.enable(false)
try
    for i in 1:5
        allocs = @allocated for _ in 1:1000
            dense_step!(u_dense, ws, G_dense, C_dense, A_dense, A_lu_dense, b_dense, inv_dt, t)
        end
        @printf("  Run %d: total=%8d bytes, per_call=%.3f bytes\n", i, allocs, allocs/1000)
    end
finally
    GC.enable(true)
end

println("\n" * "=" ^ 70)
println("Done")
println("=" ^ 70)
