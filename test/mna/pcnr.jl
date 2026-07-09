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
import Cadnip.MNA as MNA

using Cadnip
using Cadnip: dc!, tran!  # explicit import to avoid Julia 1.12 conflict

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
        # is out -> gnd, so branch voltage == V(out)). DCSolution's name-based
        # getindex only walks node_names/current_names (not limit_names), so
        # index the limit variable positionally via a freshly built context.
        ctx = rectifier_lim((;), MNASpec(mode=:dcop))
        li = system_size(ctx)
        @test sol_lim.x[li] ≈ sol_lim[:out] atol=1e-6
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
        u, ok = MNA._dc_pcnr_newton(cs, ws, zeros(system_size(ctx)); abstol=1e-10, maxiters=100)
        @test ok == true
        @test cs.n_limits == 1

        ctx2 = MNA._detect_structure(chain_lim, (;), spec)
        cs2 = MNA.compile_structure(chain_lim, (;), spec; ctx=ctx2)
        ws2 = MNA.create_workspace(cs2; ctx=ctx2)
        u2, ok2 = MNA._dc_pcnr_newton(cs2, ws2, zeros(system_size(ctx2)); abstol=1e-10, maxiters=100)
        @test ok2 == true
        @test cs2.n_limits == 3
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

end
