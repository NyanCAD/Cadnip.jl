#!/usr/bin/env julia
#==============================================================================#
# Zero-Allocation Circuit Simulation Test
#
# Goal: Find solver combinations that achieve zero allocations in step!
# after initialization.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CedarSim
using CedarSim.MNA
using Printf
using LinearAlgebra
using SparseArrays

# For low-level ODE/DAE solver access
using OrdinaryDiffEq
using SciMLBase
using LinearSolve
using Sundials: IDA

println("=" ^ 70)
println("Zero-Allocation Circuit Simulation Test")
println("=" ^ 70)

#==============================================================================#
# Simple RC Circuit Builder
#==============================================================================#

function build_simple_rc(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    # Nodes
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # Voltage source Vcc = 5V
    MNA.stamp!(VoltageSource(params.Vcc; name=:Vs), ctx, vcc, 0)

    # Resistor R between vcc and out
    MNA.stamp!(Resistor(params.R), ctx, vcc, out)

    # Capacitor C between out and ground
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    return ctx
end

#==============================================================================#
# Allocation Measurement Helper
#==============================================================================#

function measure_allocations(f::Function, warmup_iters::Int=10, measure_iters::Int=1000)
    # Warmup
    for _ in 1:warmup_iters
        f()
    end
    GC.gc()

    # Measure
    allocs = @allocated begin
        for _ in 1:measure_iters
            f()
        end
    end

    return allocs / measure_iters
end

#==============================================================================#
# Test 1: Profile MNA Fast Path Operations
#==============================================================================#

println("\n" * "=" ^ 70)
println("Test 1: MNA Fast Path Operations")
println("=" ^ 70)

circuit = MNACircuit(build_simple_rc; Vcc=5.0, R=1000.0, C=1e-6)
ws = MNA.compile(circuit)
cs = ws.structure

n = MNA.system_size(cs)
u = zeros(n)
du = zeros(n)
resid = zeros(n)
t = 0.001
gamma = 1.0
J = copy(cs.G)  # Same sparsity as unified pattern

println("\nSystem size: $n")
println("G nnz: $(nnz(cs.G)), C nnz: $(nnz(cs.C))")

# Warmup
for _ in 1:10
    MNA.fast_rebuild!(ws, u, t)
    MNA.fast_residual!(resid, du, u, ws, t)
    MNA.fast_jacobian!(J, du, u, ws, gamma, t)
end
GC.gc()

allocs_rebuild = measure_allocations() do
    MNA.fast_rebuild!(ws, u, t)
end

allocs_residual = measure_allocations() do
    MNA.fast_residual!(resid, du, u, ws, t)
end

allocs_jacobian = measure_allocations() do
    MNA.fast_jacobian!(J, du, u, ws, gamma, t)
end

@printf("\nfast_rebuild!:   %.1f bytes/call\n", allocs_rebuild)
@printf("fast_residual!:  %.1f bytes/call\n", allocs_residual)
@printf("fast_jacobian!:  %.1f bytes/call\n", allocs_jacobian)

#==============================================================================#
# Test 2: OrdinaryDiffEq Low-Level Step API
#==============================================================================#

println("\n" * "=" ^ 70)
println("Test 2: OrdinaryDiffEq Low-Level Step API")
println("=" ^ 70)

# Create ODE problem with mass matrix formulation
prob_ode = SciMLBase.ODEProblem(circuit, (0.0, 1e-3))

# Try different solvers
solvers_to_test = [
    ("Rodas5P(autodiff=false)", Rodas5P(autodiff=false)),
    ("Rodas5P(autodiff=false, linsolve=LUFactorization())",
     Rodas5P(autodiff=false, linsolve=LUFactorization())),
    ("ImplicitEuler(autodiff=false)", ImplicitEuler(autodiff=false)),
    ("QNDF(autodiff=false)", QNDF(autodiff=false)),
]

for (name, solver) in solvers_to_test
    println("\n--- $name ---")

    # Initialize integrator
    integrator = init(prob_ode, solver; dt=1e-6, save_everystep=false)

    # Warmup
    for _ in 1:10
        step!(integrator)
    end
    GC.gc()

    # Measure step! allocations
    allocs_step = @allocated begin
        for _ in 1:100
            step!(integrator)
        end
    end

    @printf("  step! allocations: %.1f bytes/call\n", allocs_step / 100)

    # Also check what happens inside step
    # Try measuring just the function call overhead
end

#==============================================================================#
# Test 3: Sundials IDA Low-Level Step API
#==============================================================================#

println("\n" * "=" ^ 70)
println("Test 3: Sundials IDA Low-Level Step API")
println("=" ^ 70)

# Create DAE problem
prob_dae = SciMLBase.DAEProblem(circuit, (0.0, 1e-3))

ida_variants = [
    ("IDA(linear_solver=:KLU)", IDA(linear_solver=:KLU)),
    ("IDA(linear_solver=:Dense)", IDA(linear_solver=:Dense)),
]

for (name, solver) in ida_variants
    println("\n--- $name ---")

    try
        integrator = init(prob_dae, solver; dt=1e-6, save_everystep=false)

        # Warmup
        for _ in 1:10
            step!(integrator)
        end
        GC.gc()

        # Measure
        allocs_step = @allocated begin
            for _ in 1:100
                step!(integrator)
            end
        end

        @printf("  step! allocations: %.1f bytes/call\n", allocs_step / 100)
    catch e
        println("  Error: $e")
    end
end

#==============================================================================#
# Test 4: Manual Euler Integration (Minimal Allocation Baseline)
#==============================================================================#

println("\n" * "=" ^ 70)
println("Test 4: Manual Euler Integration (Baseline)")
println("=" ^ 70)

# Recompile circuit for fresh workspace
ws = MNA.compile(circuit)
cs = ws.structure

n = MNA.system_size(cs)
u = zeros(n)
du = zeros(n)
b_work = zeros(n)  # Working vector

# Compute initial DC solution
MNA.fast_rebuild!(ws, u, 0.0)
u_dc = cs.G \ ws.dctx.b
copyto!(u, u_dc)

dt = 1e-8
t = 0.0

# For C*du/dt = b - G*u, we have du/dt = C \ (b - G*u)
# For implicit Euler: C*(u_{n+1} - u_n)/dt = b - G*u_{n+1}
# => (C/dt + G) * u_{n+1} = C/dt * u_n + b
# => A * u_{n+1} = rhs

# Preallocate A and factorization
A = copy(cs.G)
A_nzval = nonzeros(A)
G_nzval = nonzeros(cs.G)
C_nzval = nonzeros(cs.C)

function manual_euler_step!(u, du, ws, cs, A, A_nzval, G_nzval, C_nzval, b_work, dt, t)
    # Rebuild circuit at current state
    MNA.fast_rebuild!(ws, u, t)

    # Form A = G + C/dt (in-place)
    inv_dt = 1.0 / dt
    @inbounds for i in eachindex(A_nzval)
        A_nzval[i] = G_nzval[i] + inv_dt * C_nzval[i]
    end

    # Form rhs = C/dt * u + b
    mul!(b_work, cs.C, u)
    b_work .*= inv_dt
    b_work .+= ws.dctx.b

    # Solve (this allocates for LU factorization)
    u .= A \ b_work

    return nothing
end

# Warmup
for _ in 1:10
    manual_euler_step!(u, du, ws, cs, A, A_nzval, G_nzval, C_nzval, b_work, dt, t)
end
GC.gc()

allocs_manual = measure_allocations(10, 100) do
    manual_euler_step!(u, du, ws, cs, A, A_nzval, G_nzval, C_nzval, b_work, dt, t)
end

@printf("\nManual Euler step: %.1f bytes/call\n", allocs_manual)
println("(Note: Includes LU factorization allocation)")

#==============================================================================#
# Test 5: Pre-factorized Linear Solve
#==============================================================================#

println("\n" * "=" ^ 70)
println("Test 5: Pre-factorized Linear Solve")
println("=" ^ 70)

# Reset state
MNA.fast_rebuild!(ws, u, 0.0)
copyto!(u, cs.G \ ws.dctx.b)

# Pre-factorize A (assuming structure doesn't change)
# A = G + C/dt
inv_dt = 1.0 / dt
@inbounds for i in eachindex(A_nzval)
    A_nzval[i] = G_nzval[i] + inv_dt * C_nzval[i]
end

# Use LU factorization for dense (small) systems or SparseArrays.lu for sparse
using SparseArrays: lu, lu!

# Create factorization object that can be updated in-place
A_fact = lu(A)

function prefact_euler_step!(u, ws, cs, A, A_fact, b_work, dt, t)
    # Rebuild circuit at current state
    MNA.fast_rebuild!(ws, u, t)

    # Form A = G + C/dt (in-place)
    inv_dt = 1.0 / dt
    A_nzval = nonzeros(A)
    G_nzval = nonzeros(cs.G)
    C_nzval = nonzeros(cs.C)
    @inbounds for i in eachindex(A_nzval)
        A_nzval[i] = G_nzval[i] + inv_dt * C_nzval[i]
    end

    # Re-factorize in-place (lu! reuses symbolic factorization)
    lu!(A_fact, A)

    # Form rhs = C/dt * u + b
    mul!(b_work, cs.C, u)
    b_work .*= inv_dt
    b_work .+= ws.dctx.b

    # Solve using factorization (ldiv! is in-place)
    ldiv!(A_fact, b_work)
    copyto!(u, b_work)

    return nothing
end

# Warmup
for _ in 1:10
    prefact_euler_step!(u, ws, cs, A, A_fact, b_work, dt, t)
end
GC.gc()

allocs_prefact = measure_allocations(10, 100) do
    prefact_euler_step!(u, ws, cs, A, A_fact, b_work, dt, t)
end

@printf("\nPre-factorized Euler step: %.1f bytes/call\n", allocs_prefact)

#==============================================================================#
# Test 6: Try KLU for in-place factorization
#==============================================================================#

println("\n" * "=" ^ 70)
println("Test 6: KLU In-Place Factorization")
println("=" ^ 70)

using LinearSolve

# Reset state
MNA.fast_rebuild!(ws, u, 0.0)
copyto!(u, cs.G \ ws.dctx.b)

# Create LinearSolve problem
A_copy = copy(A)
b_copy = copy(b_work)
linprob = LinearProblem(A_copy, b_copy)
lincache = init(linprob, KLUFactorization())

function klu_euler_step!(u, ws, cs, A, lincache, dt, t)
    # Rebuild circuit at current state
    MNA.fast_rebuild!(ws, u, t)

    # Update A in-place
    inv_dt = 1.0 / dt
    A_nzval = nonzeros(A)
    G_nzval = nonzeros(cs.G)
    C_nzval = nonzeros(cs.C)
    @inbounds for i in eachindex(A_nzval)
        A_nzval[i] = G_nzval[i] + inv_dt * C_nzval[i]
    end

    # Form rhs = C/dt * u + b
    b = lincache.b
    mul!(b, cs.C, u)
    b .*= inv_dt
    b .+= ws.dctx.b

    # Update the matrix in the cache and solve
    lincache.A = A
    sol = solve!(lincache)
    copyto!(u, sol.u)

    return nothing
end

# Warmup
for _ in 1:10
    klu_euler_step!(u, ws, cs, A, lincache, dt, t)
end
GC.gc()

allocs_klu = measure_allocations(10, 100) do
    klu_euler_step!(u, ws, cs, A, lincache, dt, t)
end

@printf("\nKLU Euler step: %.1f bytes/call\n", allocs_klu)

#==============================================================================#
# Summary
#==============================================================================#

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

println("\nMNA Core Operations:")
@printf("  fast_rebuild!:   %.1f bytes/call\n", allocs_rebuild)
@printf("  fast_residual!:  %.1f bytes/call\n", allocs_residual)
@printf("  fast_jacobian!:  %.1f bytes/call\n", allocs_jacobian)

println("\nIntegration Methods:")
@printf("  Manual Euler:        %.1f bytes/call\n", allocs_manual)
@printf("  Pre-factorized:      %.1f bytes/call\n", allocs_prefact)
@printf("  KLU:                 %.1f bytes/call\n", allocs_klu)

if allocs_rebuild == 0 && allocs_residual == 0 && allocs_jacobian == 0
    println("\n✓ MNA core operations are ZERO ALLOCATION!")
else
    println("\n✗ MNA core operations have allocations")
end

println()
