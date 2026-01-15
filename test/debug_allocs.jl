#!/usr/bin/env julia
#==============================================================================#
# Debug 24-byte allocations in MNA fast path
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CedarSim
using CedarSim.MNA
using Printf
using LinearAlgebra
using SparseArrays

println("=" ^ 70)
println("Debugging 24-byte allocations")
println("=" ^ 70)

#==============================================================================#
# Simple RC Circuit Builder
#==============================================================================#

function build_simple_rc(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    MNA.stamp!(VoltageSource(params.Vcc; name=:Vs), ctx, vcc, 0)
    MNA.stamp!(Resistor(params.R), ctx, vcc, out)
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    return ctx
end

#==============================================================================#
# Setup
#==============================================================================#

circuit = MNACircuit(build_simple_rc; Vcc=5.0, R=1000.0, C=1e-6)
ws = MNA.compile(circuit)
cs = ws.structure
dctx = ws.dctx

n = MNA.system_size(cs)
u = zeros(n)
du = zeros(n)
resid = zeros(n)
t = 0.001
gamma = 1.0
J = copy(cs.G)

println("\nSystem size: $n")

#==============================================================================#
# Test allocation measurement
#==============================================================================#

function measure_allocations(f::Function; warmup=10, iters=1000)
    for _ in 1:warmup; f(); end
    GC.gc()
    allocs = @allocated for _ in 1:iters; f(); end
    return allocs / iters
end

#==============================================================================#
# Detailed breakdown of fast_rebuild! - test via fast_rebuild! itself
#==============================================================================#

println("\n--- Testing fast_rebuild! with @allocated ---")

# Single call to verify it works
MNA.fast_rebuild!(ws, u, t)

# First, verify the behavior
println("\n@allocated for a single call:")
GC.gc()
allocs_single = @allocated MNA.fast_rebuild!(ws, u, t)
println("  Single call: $allocs_single bytes")

# Now test with multiple iterations
allocs_rebuild = measure_allocations() do
    MNA.fast_rebuild!(ws, u, t)
end
@printf("  Averaged: %.1f bytes/call\n", allocs_rebuild)

#==============================================================================#
# Test components individually using the correct workflow
#==============================================================================#

println("\n--- Testing MNA internal operations ---")

# Test mul! operations (these should be allocation-free)
mul!(resid, cs.C, du)
mul!(resid, cs.G, u, 1.0, 1.0)

allocs_mul1 = measure_allocations() do
    mul!(resid, cs.C, du)
end
@printf("\nmul!(resid, C, du):      %.1f bytes/call\n", allocs_mul1)

allocs_mul2 = measure_allocations() do
    mul!(resid, cs.G, u, 1.0, 1.0)
end
@printf("mul!(resid, G, u, α, β): %.1f bytes/call\n", allocs_mul2)

# Test vector operations
b_vec = ws.dctx.b
allocs_sub = measure_allocations() do
    resid .-= b_vec
end
@printf("resid .-= b:            %.1f bytes/call\n", allocs_sub)

# Test nzval loop (for jacobian)
J_nz = nonzeros(J)
G_nz = nonzeros(cs.G)
C_nz = nonzeros(cs.C)

allocs_nzloop = measure_allocations() do
    @inbounds for i in eachindex(J_nz, G_nz, C_nz)
        J_nz[i] = G_nz[i] + gamma * C_nz[i]
    end
end
@printf("nzval loop:             %.1f bytes/call\n", allocs_nzloop)

#==============================================================================#
# Full fast_* functions
#==============================================================================#

println("\n--- Testing full fast_* functions ---")

allocs_rebuild_full = measure_allocations() do
    MNA.fast_rebuild!(ws, u, t)
end
@printf("\nfast_rebuild!:   %.1f bytes/call\n", allocs_rebuild_full)

allocs_residual = measure_allocations() do
    MNA.fast_residual!(resid, du, u, ws, t)
end
@printf("fast_residual!:  %.1f bytes/call\n", allocs_residual)

allocs_jacobian = measure_allocations() do
    MNA.fast_jacobian!(J, du, u, ws, gamma, t)
end
@printf("fast_jacobian!:  %.1f bytes/call\n", allocs_jacobian)

#==============================================================================#
# Test with @time to see more details
#==============================================================================#

println("\n--- @time for 10000 calls ---")

GC.gc()
@time begin
    for _ in 1:10000
        MNA.fast_rebuild!(ws, u, t)
    end
end

GC.gc()
@time begin
    for _ in 1:10000
        MNA.fast_residual!(resid, du, u, ws, t)
    end
end

GC.gc()
@time begin
    for _ in 1:10000
        MNA.fast_jacobian!(J, du, u, ws, gamma, t)
    end
end

#==============================================================================#
# Check if allocation is from type instability
#==============================================================================#

println("\n--- Testing type stability ---")

using Test
@testset "Type stability" begin
    @test @inferred(MNA.real_time(1.0)) isa Float64
end

# Test if the allocation comes from inside the builder
println("\n--- Testing builder call via indirect means ---")

# Since we can't call the builder directly multiple times without reset issues,
# test if the allocation comes from the fast_rebuild! wrapping layer

# Test with explicit structure
allocs_rebuild_explict = measure_allocations() do
    MNA.fast_rebuild!(ws, cs, u, t)
end
@printf("\nfast_rebuild!(ws, cs, u, t): %.1f bytes/call\n", allocs_rebuild_explict)

#==============================================================================#
# Summary
#==============================================================================#

println("\n" * "=" ^ 70)
println("ALLOCATION SUMMARY")
println("=" ^ 70)

if allocs_rebuild_full > 0
    println("\nfast_rebuild! allocates $(allocs_rebuild_full) bytes per call")
    println("This allocation likely comes from inside the builder function.")
    println("\nPossible sources:")
    println("  - Symbol creation/interpolation")
    println("  - Type instability in device stamp functions")
    println("  - Struct boxing")
else
    println("\n✓ fast_rebuild! is allocation-free!")
end

println()
