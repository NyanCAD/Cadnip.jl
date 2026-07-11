#==============================================================================#
# PCNR (Predictor/Corrector Newton-Raphson) Limiting Tests
#
# See doc/pcnr_plan.md. Covers:
# - pnjlim: the pure PN-junction limiter function (ported from DEVpnjlim)
# - limit!: the $limit-shaped runtime primitive (vold read, w recording,
#   linear tracking row) that both native devices and future VA codegen use
# - MNAContext plumbing: alloc_limit!, LimitIndex, resolve_index
# - Diode(limit=true): the augmented [nodes; currents; charges; limits]
#   system, and fixed-point invariance vs. Diode(limit=false)
# - the internal PCNR Newton loop (_dc_pcnr_newton) on rectifier/chain circuits
# - a transient smoke test to confirm the augmented system doesn't break IDA
#==============================================================================#

using Test
using SparseArrays
using SciMLBase: ReturnCode

using Cadnip.MNA: MNAContext, get_node!, resolve_index, system_size
using Cadnip.MNA: stamp!, assemble!, reset_for_restamping!
using Cadnip.MNA: Resistor, Capacitor, VoltageSource, Diode
using Cadnip.MNA: alloc_limit!, get_limit_idx, has_limit, LimitIndex
using Cadnip.MNA: pnjlim, limit!, record_limit_w!
using Cadnip.MNA: ZERO_VECTOR
using Cadnip.MNA: MNASpec, MNACircuit, solve_dc
using Cadnip.MNA: SinWave, CedarPCNRCorrect
import Cadnip.MNA as MNA

using Cadnip
using Cadnip: dc!, tran!, pcnr_fbdf  # explicit import to avoid Julia 1.12 conflict
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEq.OrdinaryDiffEqCore: COEFFICIENT_MULTISTEP, DIRK
using Cadnip.MNA.ADTypes: AutoFiniteDiff

#==============================================================================#
# Shared builders for the structural/DC tests below (top-level, not nested in
# a @testset, so they're visible from every testset that needs them).
#
# All nodes are allocated via get_node! before any device is stamped, per the
# offset-stability discipline for MNAContext builds.
#==============================================================================#

function rectifier_lim(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end
    vin = get_node!(ctx, :vin)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(5.0; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, out)
    stamp!(Diode(Is=1e-14, Vt=0.026, limit=true, name=:D1), ctx, out, 0; x=x)
    return ctx
end

function rectifier_nolim(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end
    vin = get_node!(ctx, :vin)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(5.0; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, out)
    stamp!(Diode(Is=1e-14, Vt=0.026, limit=false, name=:D1), ctx, out, 0; x=x)
    return ctx
end

# 50V through 1k into three series diodes: vin -[1k]- n1 -D1- n2 -D2- n3 -D3- gnd
function chain_lim(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end
    vin = get_node!(ctx, :vin)
    n1 = get_node!(ctx, :n1)
    n2 = get_node!(ctx, :n2)
    n3 = get_node!(ctx, :n3)

    stamp!(VoltageSource(50.0; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, n1)
    stamp!(Diode(Is=1e-14, Vt=0.026, limit=true, name=:D1), ctx, n1, n2; x=x)
    stamp!(Diode(Is=1e-14, Vt=0.026, limit=true, name=:D2), ctx, n2, n3; x=x)
    stamp!(Diode(Is=1e-14, Vt=0.026, limit=true, name=:D3), ctx, n3, 0; x=x)
    return ctx
end

function chain_nolim(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end
    vin = get_node!(ctx, :vin)
    n1 = get_node!(ctx, :n1)
    n2 = get_node!(ctx, :n2)
    n3 = get_node!(ctx, :n3)

    stamp!(VoltageSource(50.0; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, n1)
    stamp!(Diode(Is=1e-14, Vt=0.026, limit=false, name=:D1), ctx, n1, n2; x=x)
    stamp!(Diode(Is=1e-14, Vt=0.026, limit=false, name=:D2), ctx, n2, n3; x=x)
    stamp!(Diode(Is=1e-14, Vt=0.026, limit=false, name=:D3), ctx, n3, 0; x=x)
    return ctx
end

@testset "PCNR limiting" begin

    #==========================================================================#
    # pnjlim: pure function
    #==========================================================================#

    @testset "pnjlim" begin
        # Below vcrit and small step: pass-through
        @test pnjlim(0.5, 0.49, 0.026, 0.7) == (0.5, false)

        # Fixed point: pnjlim(v, v, vt, vcrit) == (v, false) for v <= vcrit
        vt, vcrit = 0.026, 0.6588
        for v in (-0.3, 0.0, 0.3, 0.6588)
            vlim, limited = pnjlim(v, v, vt, vcrit)
            @test vlim ≈ v
            @test limited == false
        end

        # Forward-jump compression (vold > 0 branch): proposed jump gets
        # compressed logarithmically, landing strictly between vold and vnew.
        vold, vnew = 0.6, 5.0
        vlim, limited = pnjlim(vnew, vold, vt, vcrit)
        @test limited == true
        @test vold < vlim < vnew
        @test vlim < 1.0

        # vold <= 0 branch
        vlim2, limited2 = pnjlim(5.0, 0.0, 0.026, 0.66)
        @test limited2 == true
        @test vlim2 ≈ 0.026 * log(5.0 / 0.026)

        # Negative clamp branch: vnew < arg = -vold - 1 (vold > 0)
        vlim3, limited3 = pnjlim(-10.0, 0.5, 0.026, 0.66)
        @test (vlim3, limited3) == (-1.5, true)

        # Negative, no clamp needed: vnew > arg = 2*vold - 1 (vold <= 0)
        vlim4, limited4 = pnjlim(-0.5, 0.0, 0.026, 0.66)
        @test (vlim4, limited4) == (-0.5, false)
    end

    #==========================================================================#
    # _diode_iv: linear extension above exponent 80 (limited path only)
    #==========================================================================#

    @testset "_diode_iv linear extension" begin
        Is, nVt = 1e-14, 0.026

        # Continuous across the clamp boundary
        I_lo, G_lo = MNA._diode_iv(Is, nVt, 80.0 * nVt - 1e-9)
        I_hi, G_hi = MNA._diode_iv(Is, nVt, 80.0 * nVt + 1e-9)
        @test isapprox(I_lo, I_hi; rtol=1e-6)
        @test isapprox(G_lo, G_hi; rtol=1e-6)

        # Deep in the extension: finite, linear in v with constant slope
        I1, G1 = MNA._diode_iv(Is, nVt, 10.0)
        I2, G2 = MNA._diode_iv(Is, nVt, 11.0)
        @test isfinite(I1) && isfinite(I2)
        @test G1 == G2
        @test isapprox(I2 - I1, G1 * 1.0; rtol=1e-12)

        # Below the threshold: exact classic exponential
        I3, G3 = MNA._diode_iv(Is, nVt, 0.7)
        @test I3 ≈ Is * (exp(0.7 / nVt) - 1.0)
        @test G3 ≈ Is / nVt * exp(0.7 / nVt)
    end

    #==========================================================================#
    # limit!: the $limit-shaped runtime primitive
    #==========================================================================#

    @testset "limit! API" begin
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        c = get_node!(ctx, :c)

        # Previous-iterate state: x = [Va, Vc, x_lim]; vold = x[3] = 0.6
        x = [0.2, 0.0, 0.6]
        w = limit!(ctx, :vdlim, :D1, a, c, 0.2, x, (vn, vo) -> (vn + vo) / 2)

        @test w ≈ 0.4                    # fn(vnew=0.2, vold=0.6)
        @test ctx.n_limits == 1
        @test has_limit(ctx, :D1_vdlim)
        @test ctx.limit_w[1] ≈ 0.4       # recorded corrector target

        # Linear tracking row g_lim = x_lim - (Va - Vc) = 0
        sys = assemble!(ctx)
        li = system_size(sys)
        @test sys.G[li, li] == 1.0
        @test sys.G[li, a] == -1.0
        @test sys.G[li, c] == 1.0

        # Multi-site pattern (VA OldGet/NewSet): later records win
        record_limit_w!(ctx, MNA.get_limit_idx(ctx, :D1_vdlim), 0.55)
        @test ctx.limit_w[1] == 0.55

        # init kwarg seeds the PCNR solve
        ctx2 = MNAContext()
        get_node!(ctx2, :a)
        get_node!(ctx2, :c)
        limit!(ctx2, :vdlim, :D2, 1, 2, 0.0, ZERO_VECTOR, (vn, vo) -> vn; init=0.7)
        @test ctx2.limit_init[1] == 0.7
    end

    #==========================================================================#
    # MNAContext plumbing
    #==========================================================================#

    @testset "context plumbing" begin
        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)
        @test n1 == 1
        @test n2 == 2

        lidx = alloc_limit!(ctx, :D1_vdlim, 1, 2; init=0.66)

        @test ctx.n_limits == 1
        @test system_size(ctx) == 3  # 2 nodes + 1 limit variable
        @test resolve_index(ctx, lidx) == 3
        @test get_limit_idx(ctx, :D1_vdlim) == LimitIndex(1)
        @test has_limit(ctx, :D1_vdlim)
        @test ctx.limit_branches[1] == (1, 2)
        @test ctx.limit_init[1] == 0.66
        @test ctx.limit_w[1] == 0.66  # recording buffer seeded with init

        MNA.reset_for_restamping!(ctx)
        @test ctx.n_limits == 0
        @test isempty(ctx.limit_names)
    end

    #==========================================================================#
    # Matrix structure: augmented system layout
    #==========================================================================#

    @testset "matrix structure" begin
        ctx = rectifier_lim((;), MNASpec(mode=:dcop))
        sys = assemble!(ctx)

        @test sys.n_limits == 1
        @test :D1_vdlim in sys.limit_names
        @test system_size(sys) == sys.n_nodes + sys.n_currents + 1

        out_idx = get_node!(ctx, :out)  # vin=1, out=2
        li = system_size(sys)           # only limit variable, last index

        # Linear limiting row: g_lim = x_lim - (V_out - V_gnd) = 0
        @test sys.G[li, li] == 1.0
        @test sys.G[li, out_idx] == -1.0

        # Conductances land at the usual node positions (AD through the
        # limiter); the lim column carries no device entry. At the
        # ZERO_VECTOR build point the diode adds only Is/nVt ≈ 3.8e-13.
        @test sys.G[out_idx, li] == 0.0
        @test isapprox(sys.G[out_idx, out_idx], 1 / 1000.0; rtol=1e-6)
    end

    #==========================================================================#
    # DC fixed point invariance: limited and unlimited solves agree
    #==========================================================================#

    @testset "DC fixed point invariance" begin
        sol_lim = solve_dc(rectifier_lim, (;), MNASpec(mode=:dcop))
        sol_nolim = solve_dc(rectifier_nolim, (;), MNASpec(mode=:dcop))

        @test sol_lim[:out] ≈ sol_nolim[:out] atol=1e-6
        @test 0.55 < sol_lim[:out] < 0.75

        # Diode equation consistency (I through R == I through D)
        Is, Vt = 1e-14, 0.026
        I_R = (5.0 - sol_lim[:out]) / 1000.0
        I_D = Is * (exp(sol_lim[:out] / Vt) - 1.0)
        @test isapprox(I_R, I_D; rtol=1e-2)

        # The limit variable equals the branch voltage at convergence (branch
        # is out -> gnd, so branch voltage == V(out)), and is addressable by
        # name on DCSolution just like on transient solutions.
        @test sol_lim[:D1_vdlim] ≈ sol_lim[:out] atol=1e-6
    end

    #==========================================================================#
    # Stiff series chain: 50V through 1k into three series diodes
    #==========================================================================#

    @testset "stiff series chain" begin
        spec = MNASpec(mode=:dcop)

        ctx_lim = chain_lim((;), spec)
        u_lim, ok_lim = MNA.dc_solve_with_ctx(chain_lim, (;), spec, ctx_lim)
        @test ok_lim

        n1i = get_node!(ctx_lim, :n1)
        n2i = get_node!(ctx_lim, :n2)
        n3i = get_node!(ctx_lim, :n3)

        Vd1 = u_lim[n1i] - u_lim[n2i]
        Vd2 = u_lim[n2i] - u_lim[n3i]
        Vd3 = u_lim[n3i]  # n3 to gnd

        @test isapprox(Vd1, Vd2; rtol=1e-2)
        @test isapprox(Vd2, Vd3; rtol=1e-2)
        @test 0.6 < Vd3 < 0.85
        @test 3 * 0.6 < u_lim[n1i] < 3 * 0.85

        Is, Vt = 1e-14, 0.026
        I_R = (50.0 - u_lim[n1i]) / 1000.0
        I_D3 = Is * (exp(Vd3 / Vt) - 1.0)
        @test isapprox(I_R, I_D3; rtol=1e-2)

        # Compare against the unlimited solve, but only if it also converges.
        ctx_nolim = chain_nolim((;), spec)
        u_nolim, ok_nolim = MNA.dc_solve_with_ctx(chain_nolim, (;), spec, ctx_nolim)
        if ok_nolim
            @test u_lim[n1i] ≈ u_nolim[n1i] atol=1e-6
            @test u_lim[n2i] ≈ u_nolim[n2i] atol=1e-6
            @test u_lim[n3i] ≈ u_nolim[n3i] atol=1e-6
        end
    end

    #==========================================================================#
    # Internal PCNR Newton loop: convergence on both circuits
    #==========================================================================#

    @testset "PCNR iteration count" begin
        spec = MNASpec(mode=:dcop)

        ctx = MNA._detect_structure(rectifier_lim, (;), spec)
        cs = MNA.compile_structure(rectifier_lim, (;), spec; ctx=ctx)
        ws = MNA.create_workspace(cs; ctx=ctx)
        u, ok, iters = MNA._dc_pcnr_newton(cs, ws, zeros(system_size(ctx)); abstol=1e-10, maxiters=100)
        @test ok == true
        @test cs.n_limits == 1
        # Regression guard for the headline result: with initjct seeding and
        # evaluation-anchored companions this converges in ~7 iterations;
        # the unseeded crawl was 17 (see doc/pcnr_plan.md "Measured").
        @test iters <= 10

        ctx2 = MNA._detect_structure(chain_lim, (;), spec)
        cs2 = MNA.compile_structure(chain_lim, (;), spec; ctx=ctx2)
        ws2 = MNA.create_workspace(cs2; ctx=ctx2)
        u2, ok2, iters2 = MNA._dc_pcnr_newton(cs2, ws2, zeros(system_size(ctx2)); abstol=1e-10, maxiters=100)
        @test ok2 == true
        @test cs2.n_limits == 3
        @test iters2 <= 10

        # Zero-allocation guarantee for the limited stamping hot path
        # (mirrors test/mna/audio_integration.jl's measure_allocations pin).
        rebuild() = MNA.fast_rebuild!(ws2, cs2, u2, 0.0)
        rebuild()  # warm up
        @test (@allocated rebuild()) == 0
    end

    #==========================================================================#
    # Transient smoke test: augmented system must not break IDA
    #==========================================================================#

    @testset "transient smoke" begin
        function rectifier_cap(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vin = get_node!(ctx, :vin)
            out = get_node!(ctx, :out)

            stamp!(VoltageSource(5.0; name=:V1), ctx, vin, 0)
            stamp!(Resistor(1000.0), ctx, vin, out)
            stamp!(Diode(Is=1e-14, Vt=0.026, limit=true, name=:D1), ctx, out, 0; x=x)
            stamp!(Capacitor(1e-9), ctx, out, 0)
            return ctx
        end

        circuit = MNACircuit(rectifier_cap)
        sol_dc = dc!(circuit)
        sol_tran = tran!(circuit, (0.0, 1e-6))

        @test sol_tran.retcode == ReturnCode.Success
        @test sol_tran[:out][end] ≈ sol_dc[:out] atol=1e-3
    end

    #==========================================================================#
    # In-step transient limiting via NonlinearSolveAlg (pcnr_fbdf)
    #
    # These exercise CedarPCNR as the per-stage nonlinear solver of FBDF (see
    # src/mna/pcnr_nlsolve.jl): the SPICE predict/correct loop now runs inside
    # every timestep, not only at DC init.
    #==========================================================================#

    # A half-wave rectifier whose series diode switches on/off each cycle - the
    # in-step-limiting case. Native Diode so the test stays off the heavy VA
    # precompile path. `spec.mode`/`t` are threaded so the SIN source advances.
    function halfwave_rect(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing, limit::Bool=true)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)
        out = get_node!(ctx, :out)
        stamp!(VoltageSource(0.0; tran=SinWave(0.0, 10.0, 50.0), name=:V1), ctx, vin, 0, t, spec.mode)
        stamp!(Diode(Is=1e-14, Vt=0.026, limit=limit, name=:D1), ctx, vin, out; x=x)
        stamp!(Capacitor(1e-6), ctx, out, 0)
        stamp!(Resistor(10_000.0), ctx, out, 0)
        return ctx
    end

    @testset "corrector affine maps (params-tuple canary)" begin
        # Build a workspace for a single-limit circuit and drive CedarPCNRCorrect
        # with a hand-built stage params tuple, checking both the FBDF (identity)
        # and DIRK (u = tmp + γ·z) affine maps, gated on the active flag.
        spec = MNASpec(mode=:dcop)
        ctx = MNA._detect_structure(rectifier_lim, (;), spec)
        cs = MNA.compile_structure(rectifier_lim, (;), spec; ctx=ctx)
        ws = MNA.create_workspace(cs; ctx=ctx)
        @test cs.n_limits == 1
        n = system_size(cs)
        lim0 = n - 1

        # The bridge's ODE stage params (11-tuple). Only tmp(1), γ(3),
        # method(8), ws(9) are read by the corrector; the rest are placeholders.
        mktuple(method, γ, tmp) = (tmp, zeros(n), γ, 1.0, 0.0, zeros(n), 1.0, method, ws, 1e-6, nothing)
        correct! = CedarPCNRCorrect()

        # Active branch: FBDF identity map copies limit_w straight in.
        ws.dctx.limit_w[1] = 0.42
        ws.dctx.limit_active[1] = true
        u = zeros(n); u[lim0 + 1] = 99.0
        correct!(u, copy(u), mktuple(COEFFICIENT_MULTISTEP, 0.5, zeros(n)))
        @test u[lim0 + 1] == 0.42

        # Inert branch: value left untouched (Newton's g_lim solution stands).
        ws.dctx.limit_active[1] = false
        u = zeros(n); u[lim0 + 1] = 99.0
        correct!(u, copy(u), mktuple(COEFFICIENT_MULTISTEP, 0.5, zeros(n)))
        @test u[lim0 + 1] == 99.0

        # Active branch, DIRK map: z_lim = (w - tmp_lim) / γ.
        ws.dctx.limit_active[1] = true
        tmp = zeros(n); tmp[lim0 + 1] = 0.1; γ = 0.25
        u = zeros(n); u[lim0 + 1] = 99.0
        correct!(u, copy(u), mktuple(DIRK, γ, tmp))
        @test u[lim0 + 1] ≈ (0.42 - 0.1) / γ
    end

    @testset "in-step limiting reduces NR iterations" begin
        tspan = (0.0, 0.04)  # two 50Hz cycles
        kw = (; abstol=1e-9, reltol=1e-6, dense=false)

        sol_ida  = tran!(MNACircuit(halfwave_rect), tspan; kw...)
        sol_fbdf = tran!(MNACircuit(halfwave_rect), tspan; solver=FBDF(autodiff=AutoFiniteDiff()), kw...)
        sol_pcnr = tran!(MNACircuit(halfwave_rect), tspan; solver=pcnr_fbdf(), kw...)

        @test sol_ida.retcode  == ReturnCode.Success
        @test sol_fbdf.retcode == ReturnCode.Success
        @test sol_pcnr.retcode == ReturnCode.Success

        # In-step limiting fires: PCNR converges each stage in fewer Newton
        # iterations than plain FBDF (NLNewton) on this switching circuit.
        @test sol_pcnr.stats.nnonliniter < sol_fbdf.stats.nnonliniter

        # Same trajectory as the IDA reference (name-based access at the load
        # node), at least as accurate as plain FBDF.
        ts = range(1e-3, 0.039; length=25)
        err_pcnr = maximum(abs(sol_pcnr(t; idxs=2) - sol_ida(t; idxs=2)) for t in ts)
        @test err_pcnr < 0.02
        @test sol_pcnr[:out][end] ≈ sol_ida[:out][end] atol=1e-2
    end

    @testset "inert on unlimited circuits (n_limits == 0)" begin
        # Pure RC: no limiting variables, so pcnr_fbdf's corrector early-returns
        # and the result must track plain FBDF exactly.
        function rc(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vin = get_node!(ctx, :vin)
            out = get_node!(ctx, :out)
            stamp!(VoltageSource(1.0; name=:V1), ctx, vin, 0)
            stamp!(Resistor(1000.0), ctx, vin, out)
            stamp!(Capacitor(1e-6), ctx, out, 0)
            return ctx
        end
        kw = (; abstol=1e-10, reltol=1e-8, dense=false)
        sol_f = tran!(MNACircuit(rc), (0.0, 5e-3); solver=FBDF(autodiff=AutoFiniteDiff()), kw...)
        sol_p = tran!(MNACircuit(rc), (0.0, 5e-3); solver=pcnr_fbdf(), kw...)
        @test sol_p.retcode == ReturnCode.Success
        @test sol_p[:out][end] ≈ sol_f[:out][end] rtol=1e-6
        @test sol_p[:out][end] ≈ 1.0 atol=1e-3  # cap charged to source
    end

    @testset "sweep compatibility" begin
        # CircuitSweep + pcnr_fbdf: the augmented system size must flow through
        # the per-point rebuild without special handling.
        cs = Cadnip.CircuitSweep(halfwave_rect, Cadnip.Sweep(dummy=[1.0, 2.0]))
        results = collect(tran!(cs, (0.0, 0.02); solver=pcnr_fbdf(),
                                abstol=1e-9, reltol=1e-6, dense=false))
        @test length(results) == 2
        for (_, sol) in results
            @test sol.retcode == ReturnCode.Success
        end
    end

end
