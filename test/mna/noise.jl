#==============================================================================#
# Noise-source channel (N0 groundwork — doc/noise_analysis_design.md)
#
# These are low-level stamping-mechanics tests: they assert that noise sources
# get registered on the deferred MNAContext channel during structure discovery,
# that the PSD helper shapes them correctly, and — crucially — that registration
# is invisible to DC/transient numerics (the value path is byte-identical and the
# transient hot-path DirectStampContext carries no noise machinery at all).
#==============================================================================#

using Test
using Cadnip
using Cadnip.MNA
using Cadnip.MNA: MNAContext, get_node!, stamp!, Resistor, Diode, VoltageSource, resolve_index
using Cadnip.MNA: reset_for_restamping!, num_noise_sources, noise_sources, noise_psd
using Cadnip.MNA: stamp_noise!, register_thermal_noise!, register_shot_noise!
using Cadnip.MNA: solve_dc, MNASpec
using Cadnip.MNA: THERMAL, SHOT, WHITE, FLICKER, NoiseSource
using Cadnip.MNA: NodeIndex, GroundIndex
using Cadnip.MNA: K_BOLTZMANN, Q_ELEMENTARY
using Cadnip.SpectreEnvironment

@testset "noise-source channel (N0)" begin

    @testset "resistor thermal noise registration" begin
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        b = get_node!(ctx, :b)
        stamp!(Resistor(1000.0; name=:R1), ctx, a, b)

        @test num_noise_sources(ctx) == 1
        src = noise_sources(ctx)[1]
        @test src.kind === THERMAL
        @test src.name === :R1
        @test src.a ≈ 1e-3          # conductance G = 1/R
        @test resolve_index(ctx, src.p) == a
        @test resolve_index(ctx, src.n) == b

        # PSD = 4·k·T·G, white (frequency-independent) at 27 °C
        T = 27.0
        expected = 4 * K_BOLTZMANN * (T + 273.15) * 1e-3
        @test noise_psd(src, T, 1e3) ≈ expected
        @test noise_psd(src, T, 1e6) ≈ expected   # white: no frequency dependence
    end

    @testset "diode shot noise registration at bias" begin
        # A forward-biased junction registers shot noise 2q·|I| with a = |I_D|
        # evaluated at the operating point the channel is built at. Hand-stamping
        # at a known bias vector is a low-level stamping-mechanics test.
        Is, Vt = 1e-14, 0.026
        Vbias = 0.6
        I0 = Is * (exp(Vbias / Vt) - 1.0)   # DC junction current at the bias

        ctx = MNAContext()
        a = get_node!(ctx, :a)              # index 1
        stamp!(Diode(Is=Is, Vt=Vt, limit=false, name=:D1), ctx, a, 0; x=[Vbias])

        @test num_noise_sources(ctx) == 1
        src = noise_sources(ctx)[1]
        @test src.kind === SHOT
        @test src.name === :D1
        @test src.a ≈ I0                    # a = |I_D| at the bias point
        @test resolve_index(ctx, src.p) == a

        # PSD = 2·q·|I|, white (frequency-independent)
        expected = 2 * Q_ELEMENTARY * I0
        @test noise_psd(src, 27.0, 1e3) ≈ expected
        @test noise_psd(src, 27.0, 1e9) ≈ expected
    end

    @testset "diode shot noise uses current magnitude under reverse bias" begin
        # Reverse bias: I_D saturates to -Is, so |I| = Is and the source is tiny
        # but non-degenerate (magnitude keeps it physical).
        Is, Vt = 1e-14, 0.026
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        stamp!(Diode(Is=Is, Vt=Vt, limit=false, name=:D1), ctx, a, 0; x=[-1.0])

        src = noise_sources(ctx)[1]
        @test src.a ≈ Is                    # |Is·(exp(-38)-1)| ≈ Is
        @test noise_psd(src, 27.0, 1e3) ≈ 2 * Q_ELEMENTARY * Is
    end

    @testset "multiple sources accumulate; rebuild does not duplicate" begin
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        b = get_node!(ctx, :b)
        stamp!(Resistor(1000.0; name=:R1), ctx, a, b)
        stamp!(Resistor(2000.0; name=:R2), ctx, b, 0)
        @test num_noise_sources(ctx) == 2

        # reset_for_restamping! empties the channel (recomputed every build),
        # so a rebuild re-registers rather than doubling up.
        reset_for_restamping!(ctx)
        @test num_noise_sources(ctx) == 0
        a2 = get_node!(ctx, :a)
        b2 = get_node!(ctx, :b)
        stamp!(Resistor(1000.0; name=:R1), ctx, a2, b2)
        stamp!(Resistor(2000.0; name=:R2), ctx, b2, 0)
        @test num_noise_sources(ctx) == 2
    end

    @testset "PSD shapes per kind" begin
        p = NodeIndex(1)
        g = GroundIndex()
        # white_noise(pwr) -> pwr (flat)
        w = NoiseSource(p, g, WHITE, 2.5e-18, 0.0, :nw)
        @test noise_psd(w, 27.0, 1.0) ≈ 2.5e-18
        @test noise_psd(w, 27.0, 1e9) ≈ 2.5e-18

        # flicker_noise(pwr, exp) -> pwr / f^exp
        fl = NoiseSource(p, g, FLICKER, 1e-18, 1.0, :nf)
        @test noise_psd(fl, 27.0, 10.0) ≈ 1e-19
        @test noise_psd(fl, 27.0, 100.0) ≈ 1e-20

        # shot noise 2·q·I
        sh = NoiseSource(p, g, SHOT, 1e-3, 0.0, :ns)
        @test noise_psd(sh, 27.0, 1e3) ≈ 2 * Q_ELEMENTARY * 1e-3
    end

    @testset "degenerate source (both terminals ground) is skipped" begin
        ctx = MNAContext()
        register_thermal_noise!(ctx, 0, 0, 1e-3; name=:Rgnd)
        @test num_noise_sources(ctx) == 0
    end

    @testset "numerics unchanged: DC divider solves exactly" begin
        # Registering thermal noise must not perturb G/C/b. A resistor divider
        # still solves to its analytical operating point.
        circuit = MNACircuit(sp"""
        V1 vcc 0 DC 5
        R1 vcc out 1k
        R2 out 0 1k
        """i)
        sol = dc!(circuit)
        @test sol[:out] ≈ 2.5
    end

    @testset "numerics unchanged: builtin diode rectifier solves" begin
        # A full Newton solve calls the Diode stamp (and its shot-noise
        # registration) on every iteration; the operating point must be
        # unperturbed. This is a stamping-mechanics check on the builtin Diode,
        # so it drives the stamp directly rather than through a .model card.
        function rect(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            ctx === nothing ? (ctx = MNAContext()) : reset_for_restamping!(ctx)
            vin = get_node!(ctx, :vin)
            out = get_node!(ctx, :out)
            stamp!(VoltageSource(5.0; name=:V1), ctx, vin, 0)
            stamp!(Resistor(1000.0; name=:R1), ctx, vin, out)
            stamp!(Diode(Is=1e-14, Vt=0.026, name=:D1), ctx, out, 0; x=x)
            return ctx
        end
        sol = solve_dc(rect, (;), MNASpec())
        # 5 V through 1k into a diode clamps `out` around a forward drop.
        @test 0.4 < sol[:out] < 0.8
        # And the diode's shot-noise source is registered at that bias.
        ctx = rect((;), MNASpec(); x=sol.x)
        shot = filter(s -> s.kind === SHOT, noise_sources(ctx))
        @test length(shot) == 1
        @test shot[1].a > 0                # |I_D| at the clamped operating point
    end

    @testset "transient hot path unaffected (DirectStampContext no-op)" begin
        # An RC low-pass driven through the high-level API exercises the
        # zero-allocation DirectStampContext restamping path, where the resistor
        # stamp calls register_thermal_noise! — a no-op there. If the no-op were
        # missing this would error; the solve reaching steady state confirms it.
        circuit = MNACircuit(sp"""
        V1 in 0 DC 1
        R1 in out 1k
        C1 out 0 1u
        """i)
        sol = tran!(circuit, (0.0, 20e-3))
        @test sol[:out][end] ≈ 1.0 atol=1e-2
    end
end
