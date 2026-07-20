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
using Cadnip.MNA: MNAContext, get_node!, stamp!, Resistor, resolve_index
using Cadnip.MNA: reset_for_restamping!, num_noise_sources, noise_sources, noise_psd
using Cadnip.MNA: stamp_noise!, register_thermal_noise!
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
