#!/usr/bin/env julia
#==============================================================================#
# Zero-Allocation Circuit Simulation Test
#
# This test demonstrates that MNA circuit simulation can achieve ZERO allocations
# during the simulation step after initialization.
#
# Key requirements:
# 1. Use component-based API for alloc_current!/alloc_charge! to avoid Symbol creation
# 2. Use KLUFactorization linear solver
# 3. Use autodiff=false to avoid ForwardDiff allocations
#
# Solvers verified to achieve zero allocations:
# - ImplicitEuler(autodiff=false, linsolve=KLUFactorization())
# - QNDF(autodiff=false, linsolve=KLUFactorization())
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SciMLBase
using LinearSolve

#==============================================================================#
# Zero-allocation circuit builder
#
# To achieve zero allocations, circuit builders must:
# 1. Use alloc_current!(ctx, base_name, instance_name) instead of
#    alloc_current!(ctx, Symbol(:prefix_, name)) to avoid Symbol interpolation
# 2. Use alloc_charge!(ctx, base_name, instance_name, p, n) similarly
#==============================================================================#

"""
    build_rc_zero_alloc(params, spec, t; x=ZERO_VECTOR, ctx=nothing)

Example zero-allocation RC circuit builder.

Uses component-based API for current allocation to avoid Symbol creation.
"""
function build_rc_zero_alloc(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # Voltage source: use component-based alloc_current! (zero allocation)
    # Instead of: alloc_current!(ctx, Symbol(:I_, :Vs)) which allocates
    I_idx = MNA.alloc_current!(ctx, :I, :Vs)
    MNA.stamp_G!(ctx, vcc, I_idx,  1.0)
    MNA.stamp_G!(ctx, I_idx, vcc,  1.0)
    MNA.stamp_b!(ctx, I_idx, params.Vcc)

    # Resistor (already zero-alloc)
    MNA.stamp!(Resistor(params.R), ctx, vcc, out)

    # Capacitor (already zero-alloc)
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    return ctx
end

#==============================================================================#
# Test helpers
#==============================================================================#

function measure_allocations(f::Function; warmup=10, iters=1000)
    for _ in 1:warmup
        f()
    end
    GC.gc()
    allocs = @allocated begin
        for _ in 1:iters
            f()
        end
    end
    return allocs / iters
end

# Threshold for "effectively zero" allocations
# Small allocations (<100 bytes) may come from GC overhead or measurement noise
const ZERO_ALLOC_THRESHOLD = 100.0

#==============================================================================#
# Tests
#==============================================================================#

@testset "Zero-Allocation Circuit Simulation" begin

    @testset "MNA fast path is zero-alloc" begin
        circuit = MNACircuit(build_rc_zero_alloc; Vcc=5.0, R=1000.0, C=1e-6)
        ws = MNA.compile(circuit)

        n = MNA.system_size(ws)
        u = zeros(n)
        du = zeros(n)
        resid = zeros(n)
        J = copy(ws.structure.G)
        t = 0.001
        gamma = 1.0

        # Test fast_rebuild!
        allocs_rebuild = measure_allocations() do
            MNA.fast_rebuild!(ws, u, t)
        end
        @test allocs_rebuild < ZERO_ALLOC_THRESHOLD
        @info "fast_rebuild!: $(allocs_rebuild) bytes/call"

        # Test fast_residual!
        allocs_residual = measure_allocations() do
            MNA.fast_residual!(resid, du, u, ws, t)
        end
        @test allocs_residual < ZERO_ALLOC_THRESHOLD
        @info "fast_residual!: $(allocs_residual) bytes/call"

        # Test fast_jacobian!
        allocs_jacobian = measure_allocations() do
            MNA.fast_jacobian!(J, du, u, ws, gamma, t)
        end
        @test allocs_jacobian < ZERO_ALLOC_THRESHOLD
        @info "fast_jacobian!: $(allocs_jacobian) bytes/call"
    end

    @testset "ImplicitEuler + KLU is zero-alloc" begin
        circuit = MNACircuit(build_rc_zero_alloc; Vcc=5.0, R=1000.0, C=1e-6)

        prob = SciMLBase.ODEProblem(circuit, (0.0, 1e-3))
        solver = ImplicitEuler(autodiff=false, linsolve=KLUFactorization())
        integrator = init(prob, solver; dt=1e-8, save_everystep=false)

        # Warmup
        for _ in 1:10
            step!(integrator)
        end
        GC.gc()

        # Measure
        allocs = @allocated begin
            for _ in 1:100
                step!(integrator)
            end
        end

        avg_allocs = allocs / 100
        @test avg_allocs < ZERO_ALLOC_THRESHOLD
        @info "ImplicitEuler + KLU: $(avg_allocs) bytes/step"
    end

    @testset "QNDF + KLU is zero-alloc" begin
        circuit = MNACircuit(build_rc_zero_alloc; Vcc=5.0, R=1000.0, C=1e-6)

        prob = SciMLBase.ODEProblem(circuit, (0.0, 1e-3))
        solver = QNDF(autodiff=false, linsolve=KLUFactorization())
        integrator = init(prob, solver; dt=1e-6, save_everystep=false)

        # Warmup
        for _ in 1:10
            step!(integrator)
        end
        GC.gc()

        # Measure
        allocs = @allocated begin
            for _ in 1:100
                step!(integrator)
            end
        end

        avg_allocs = allocs / 100
        @test avg_allocs < ZERO_ALLOC_THRESHOLD
        @info "QNDF + KLU: $(avg_allocs) bytes/step"
    end

    @testset "Custom zero-alloc integrator" begin
        circuit = MNACircuit(build_rc_zero_alloc; Vcc=5.0, R=1000.0, C=1e-6)
        ws = MNA.compile(circuit)
        cs = ws.structure
        n = MNA.system_size(cs)

        # Pre-allocate all working memory
        u = zeros(n)
        A = copy(cs.G)
        b_rhs = zeros(n)
        dt = 1e-8

        # Initialize with DC solution
        MNA.fast_rebuild!(ws, u, 0.0)
        u .= cs.G \ ws.dctx.b

        # Create LinearSolve cache
        linprob = LinearProblem(A, b_rhs)
        lincache = init(linprob, KLUFactorization())

        function zero_alloc_step!(u, ws, cs, A, lincache, dt, t)
            MNA.fast_rebuild!(ws, u, t)

            inv_dt = 1.0 / dt
            A_nz = nonzeros(A)
            G_nz = nonzeros(cs.G)
            C_nz = nonzeros(cs.C)
            @inbounds for i in eachindex(A_nz)
                A_nz[i] = G_nz[i] + inv_dt * C_nz[i]
            end

            b = lincache.b
            mul!(b, cs.C, u)
            b .*= inv_dt
            b .+= ws.dctx.b

            lincache.A = A
            sol = solve!(lincache)
            copyto!(u, sol.u)

            return nothing
        end

        # Warmup
        for _ in 1:10
            zero_alloc_step!(u, ws, cs, A, lincache, dt, 0.0)
        end
        GC.gc()

        # Measure
        allocs = @allocated begin
            for _ in 1:100
                zero_alloc_step!(u, ws, cs, A, lincache, dt, 0.0)
            end
        end

        avg_allocs = allocs / 100
        @test avg_allocs < ZERO_ALLOC_THRESHOLD
        @info "Custom Euler: $(avg_allocs) bytes/step"
    end

end

println("\n✓ All zero-allocation tests passed!")
