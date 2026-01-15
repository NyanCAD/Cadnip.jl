#!/usr/bin/env julia
#==============================================================================#
# Zero-Allocation Circuit Simulation Test
#
# This test verifies TRUE ZERO ALLOCATION in circuit simulation step! after init.
#
# Key findings:
# - MNA fast_rebuild!, fast_residual!, fast_jacobian! are truly zero allocation
#   when using component-based API (alloc_current!(ctx, :I, :Vs))
# - Sparse UMFPACK lu! allocates ~1696 bytes per call (unavoidable)
# - Dense LAPACK getrf!/getrs! are zero allocation
# - Sparse/dense ldiv! (without refactorization) is zero allocation
#
# Zero-allocation approaches:
# 1. Dense LU: Use LAPACK getrf!/getrs! for small circuits
# 2. Fixed factorization: For linear circuits, factorize once, use ldiv! per step
# 3. Dense fixed: Combine dense matrices with fixed factorization
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using LinearAlgebra
using LinearAlgebra.LAPACK
using SparseArrays

#==============================================================================#
# Zero-allocation circuit builder
#
# To achieve zero allocations, circuit builders must use:
#   alloc_current!(ctx, base_name, instance_name)
# instead of:
#   alloc_current!(ctx, Symbol(:prefix_, name))  # allocates 24 bytes
#==============================================================================#

function build_rc_zero_alloc(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # Component-based API: zero allocation
    I_idx = MNA.alloc_current!(ctx, :I, :Vs)
    MNA.stamp_G!(ctx, vcc, I_idx,  1.0)
    MNA.stamp_G!(ctx, I_idx, vcc,  1.0)
    MNA.stamp_b!(ctx, I_idx, params.Vcc)

    MNA.stamp!(Resistor(params.R), ctx, vcc, out)
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    return ctx
end

#==============================================================================#
# Test helper: measure with GC disabled for accurate results
#==============================================================================#

function measure_true_allocations(f::Function; warmup=1000, iters=10000)
    # Extensive warmup
    for _ in 1:warmup
        f()
    end
    GC.gc()

    # Measure with GC disabled to avoid background noise
    GC.enable(false)
    try
        allocs = @allocated begin
            for _ in 1:iters
                f()
            end
        end
        return allocs ÷ iters  # Integer division - must be exactly 0
    finally
        GC.enable(true)
    end
end

#==============================================================================#
# Tests
#==============================================================================#

@testset "True Zero-Allocation Circuit Simulation" begin

    circuit = MNACircuit(build_rc_zero_alloc; Vcc=5.0, R=1000.0, C=1e-6)
    ws = MNA.compile(circuit)
    cs = ws.structure
    n = MNA.system_size(cs)

    @testset "MNA fast path is truly zero-alloc" begin
        u = zeros(n)
        du = zeros(n)
        resid = zeros(n)
        J = copy(cs.G)
        t = 0.001
        gamma = 1.0

        allocs_rebuild = measure_true_allocations() do
            MNA.fast_rebuild!(ws, u, t)
        end
        @test allocs_rebuild == 0
        @info "fast_rebuild!: $(allocs_rebuild) bytes/call"

        allocs_residual = measure_true_allocations() do
            MNA.fast_residual!(resid, du, u, ws, t)
        end
        @test allocs_residual == 0
        @info "fast_residual!: $(allocs_residual) bytes/call"

        allocs_jacobian = measure_true_allocations() do
            MNA.fast_jacobian!(J, du, u, ws, gamma, t)
        end
        @test allocs_jacobian == 0
        @info "fast_jacobian!: $(allocs_jacobian) bytes/call"
    end

    @testset "Dense LAPACK step is truly zero-alloc" begin
        # Convert to dense for LAPACK
        G_dense = Matrix(cs.G)
        C_dense = Matrix(cs.C)
        A_work = similar(G_dense)
        ipiv = Vector{LinearAlgebra.BlasInt}(undef, n)
        b_dense = zeros(n)
        u_dense = zeros(n)
        dt = 1e-8
        inv_dt = 1.0 / dt

        # Initialize
        MNA.fast_rebuild!(ws, u_dense, 0.0)
        u_dense .= G_dense \ ws.dctx.b

        function dense_lapack_step!(u, ws, G, C, A_work, ipiv, b, inv_dt, t)
            MNA.fast_rebuild!(ws, u, t)

            # A = G + C/dt
            @inbounds for i in eachindex(A_work)
                A_work[i] = G[i] + inv_dt * C[i]
            end

            # In-place LU factorization
            LAPACK.getrf!(A_work, ipiv)

            # b = C/dt * u + ws.dctx.b
            mul!(b, C, u)
            @inbounds for i in eachindex(b)
                b[i] = b[i] * inv_dt + ws.dctx.b[i]
            end

            # Solve: A * x = b (result stored in b)
            LAPACK.getrs!('N', A_work, ipiv, b)
            copyto!(u, b)

            return nothing
        end

        allocs = measure_true_allocations() do
            dense_lapack_step!(u_dense, ws, G_dense, C_dense, A_work, ipiv, b_dense, inv_dt, 0.001)
        end
        @test allocs == 0
        @info "Dense LAPACK step: $(allocs) bytes/call"
    end

    @testset "Fixed factorization step is truly zero-alloc (linear circuits)" begin
        # For linear circuits, A = G + C/dt is constant, factorize once
        dt = 1e-8
        inv_dt = 1.0 / dt

        A_fixed = copy(cs.G)
        A_nz = nonzeros(A_fixed)
        G_nz = nonzeros(cs.G)
        C_nz = nonzeros(cs.C)
        @inbounds for i in eachindex(A_nz)
            A_nz[i] = G_nz[i] + inv_dt * C_nz[i]
        end

        A_lu = lu(A_fixed)
        b_work = zeros(n)
        u_fixed = zeros(n)

        # Initialize
        MNA.fast_rebuild!(ws, u_fixed, 0.0)
        u_fixed .= cs.G \ ws.dctx.b

        function fixed_factor_step!(u, ws, cs, A_lu, b, inv_dt, t)
            MNA.fast_rebuild!(ws, u, t)

            mul!(b, cs.C, u)
            @inbounds for i in eachindex(b)
                b[i] = b[i] * inv_dt + ws.dctx.b[i]
            end

            ldiv!(u, A_lu, b)
            return nothing
        end

        allocs = measure_true_allocations() do
            fixed_factor_step!(u_fixed, ws, cs, A_lu, b_work, inv_dt, 0.001)
        end
        @test allocs == 0
        @info "Fixed factorization step: $(allocs) bytes/call"
    end

    @testset "Dense fixed factorization is truly zero-alloc" begin
        dt = 1e-8
        inv_dt = 1.0 / dt

        G_dense = Matrix(cs.G)
        C_dense = Matrix(cs.C)
        A_dense_fixed = G_dense .+ inv_dt .* C_dense
        A_lu_dense = lu(A_dense_fixed)
        b_dense = zeros(n)
        u_dense = zeros(n)

        MNA.fast_rebuild!(ws, u_dense, 0.0)
        u_dense .= G_dense \ ws.dctx.b

        function dense_fixed_step!(u, ws, C, A_lu, b, inv_dt, t)
            MNA.fast_rebuild!(ws, u, t)

            mul!(b, C, u)
            @inbounds for i in eachindex(b)
                b[i] = b[i] * inv_dt + ws.dctx.b[i]
            end

            ldiv!(u, A_lu, b)
            return nothing
        end

        allocs = measure_true_allocations() do
            dense_fixed_step!(u_dense, ws, C_dense, A_lu_dense, b_dense, inv_dt, 0.001)
        end
        @test allocs == 0
        @info "Dense fixed step: $(allocs) bytes/call"
    end

end

println("\n✓ All TRUE zero-allocation tests passed!")
