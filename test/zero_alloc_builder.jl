#!/usr/bin/env julia
#==============================================================================#
# Test zero-allocation circuit builder
#
# This test verifies that we can achieve zero allocations by avoiding
# Symbol interpolation in the builder function.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CedarSim
using CedarSim.MNA
using Printf
using LinearAlgebra
using SparseArrays

println("=" ^ 70)
println("Zero-Allocation Circuit Builder Test")
println("=" ^ 70)

#==============================================================================#
# Standard RC Circuit Builder (has Symbol allocation)
#==============================================================================#

function build_rc_standard(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # This allocates due to Symbol(:I_, :Vs) interpolation
    MNA.stamp!(VoltageSource(params.Vcc; name=:Vs), ctx, vcc, 0)
    MNA.stamp!(Resistor(params.R), ctx, vcc, out)
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    return ctx
end

#==============================================================================#
# Zero-allocation RC Circuit Builder
#
# Avoid the VoltageSource stamp! which creates a Symbol for the current name.
# Instead, manually stamp the voltage source equations.
#==============================================================================#

function build_rc_zero_alloc(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # Voltage source: manual stamping with component-based current allocation
    # alloc_current!(ctx, base_name, instance_name) avoids Symbol creation for DirectStampContext
    I_idx = MNA.alloc_current!(ctx, :I, :Vs)

    # KCL: current flows from vcc through source to ground (node 0)
    MNA.stamp_G!(ctx, vcc, I_idx,  1.0)
    # n=0 (ground) would normally be stamp_G!(ctx, 0, I_idx, -1.0) but ground stamps are no-ops

    # Voltage equation: V(vcc) - V(gnd) = Vcc
    MNA.stamp_G!(ctx, I_idx, vcc,  1.0)
    MNA.stamp_b!(ctx, I_idx, params.Vcc)

    # Resistor: standard stamp
    MNA.stamp!(Resistor(params.R), ctx, vcc, out)

    # Capacitor: standard stamp
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    return ctx
end

#==============================================================================#
# Even more minimal: Pure resistor-capacitor (no voltage source)
#
# Uses initial conditions instead of voltage source for DC.
#==============================================================================#

function build_rc_minimal(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # Resistor between vcc and out
    MNA.stamp!(Resistor(params.R), ctx, vcc, out)

    # Capacitor between out and ground
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    # Ground vcc node to some potential via large resistor (for DC stability)
    MNA.stamp!(Resistor(1e12), ctx, vcc, 0)

    return ctx
end

#==============================================================================#
# Test allocation helper
#==============================================================================#

function measure_allocations(f::Function; warmup=10, iters=1000)
    for _ in 1:warmup; f(); end
    GC.gc()
    allocs = @allocated for _ in 1:iters; f(); end
    return allocs / iters
end

#==============================================================================#
# Compare standard vs zero-alloc builders
#==============================================================================#

println("\n--- Comparing builders ---")

# Standard builder
circuit_std = MNACircuit(build_rc_standard; Vcc=5.0, R=1000.0, C=1e-6)
ws_std = MNA.compile(circuit_std)

n = MNA.system_size(ws_std)
u = zeros(n)
t = 0.001

allocs_std = measure_allocations() do
    MNA.fast_rebuild!(ws_std, u, t)
end
@printf("\nStandard builder:    %.1f bytes/call\n", allocs_std)

# Zero-alloc builder (with component-based alloc_current!)
circuit_za = MNACircuit(build_rc_zero_alloc; Vcc=5.0, R=1000.0, C=1e-6)
ws_za = MNA.compile(circuit_za)

n_za = MNA.system_size(ws_za)
u_za = zeros(n_za)

allocs_za = measure_allocations() do
    MNA.fast_rebuild!(ws_za, u_za, t)
end
@printf("Zero-alloc builder:  %.1f bytes/call\n", allocs_za)

# Minimal builder (no voltage source)
circuit_min = MNACircuit(build_rc_minimal; Vcc=5.0, R=1000.0, C=1e-6)
ws_min = MNA.compile(circuit_min)

n_min = MNA.system_size(ws_min)
u_min = zeros(n_min)

allocs_min = measure_allocations() do
    MNA.fast_rebuild!(ws_min, u_min, t)
end
@printf("Minimal builder:     %.1f bytes/call\n", allocs_min)

#==============================================================================#
# Now test full simulation step with zero-alloc builder
#==============================================================================#

println("\n--- Testing full simulation step ---")

using LinearSolve

# Use the zero-alloc circuit
cs = ws_za.structure
dctx = ws_za.dctx

# Pre-allocate everything
J = copy(cs.G)  # Jacobian matrix
resid = zeros(n_za)
du = zeros(n_za)
b_work = zeros(n_za)
dt = 1e-8

# Form system matrix A = G + C/dt
A = copy(cs.G)
inv_dt = 1.0 / dt
A_nz = nonzeros(A)
G_nz = nonzeros(cs.G)
C_nz = nonzeros(cs.C)
@inbounds for i in eachindex(A_nz)
    A_nz[i] = G_nz[i] + inv_dt * C_nz[i]
end

# Create LinearSolve cache with KLU
linprob = LinearProblem(A, b_work)
lincache = init(linprob, KLUFactorization())

# Initialize u with DC solution
MNA.fast_rebuild!(ws_za, u_za, 0.0)
u_za .= cs.G \ dctx.b

#==============================================================================#
# Zero-allocation Euler step function
#==============================================================================#

function zero_alloc_euler_step!(u, ws, cs, A, lincache, dt, t)
    # Rebuild circuit (should be zero-alloc if builder is zero-alloc)
    MNA.fast_rebuild!(ws, u, t)

    # Update A = G + C/dt in-place
    inv_dt = 1.0 / dt
    A_nz = nonzeros(A)
    G_nz = nonzeros(cs.G)
    C_nz = nonzeros(cs.C)
    @inbounds for i in eachindex(A_nz)
        A_nz[i] = G_nz[i] + inv_dt * C_nz[i]
    end

    # Form rhs = C/dt * u + b
    b = lincache.b
    mul!(b, cs.C, u)
    b .*= inv_dt
    b .+= ws.dctx.b

    # Update matrix and solve
    lincache.A = A
    sol = solve!(lincache)
    copyto!(u, sol.u)

    return nothing
end

# Warmup
for _ in 1:10
    zero_alloc_euler_step!(u_za, ws_za, cs, A, lincache, dt, t)
end
GC.gc()

allocs_step = measure_allocations(warmup=10, iters=100) do
    zero_alloc_euler_step!(u_za, ws_za, cs, A, lincache, dt, t)
end
@printf("\nZero-alloc Euler step: %.1f bytes/call\n", allocs_step)

#==============================================================================#
# Summary
#==============================================================================#

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

println("\nBuilder allocations:")
@printf("  Standard (Symbol interp): %.1f bytes/call\n", allocs_std)
@printf("  Zero-alloc (component):   %.1f bytes/call\n", allocs_za)
@printf("  Minimal (no V source):    %.1f bytes/call\n", allocs_min)

println("\nFull step allocation:")
@printf("  Euler step: %.1f bytes/call\n", allocs_step)

if allocs_za == 0
    println("\n✓ Zero-allocation circuit rebuild achieved!")
else
    println("\n✗ Still allocating $(allocs_za) bytes per rebuild")
end

if allocs_step == 0
    println("✓ Zero-allocation simulation step achieved!")
else
    println("✗ Still allocating $(allocs_step) bytes per step")
end

println()
