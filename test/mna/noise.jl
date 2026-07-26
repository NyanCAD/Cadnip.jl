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
using Cadnip.MNA: MNAContext, get_node!, stamp!, Resistor, Capacitor, Diode, SimpleMOSFET, VoltageSource, resolve_index
using Cadnip.MNA: reset_for_restamping!, num_noise_sources, noise_sources, noise_psd
using Cadnip.MNA: stamp_noise!, register_thermal_noise!, register_shot_noise!
using Cadnip.MNA: register_channel_thermal_noise!, register_white_noise!, register_flicker_noise!
using Cadnip.MNA: noise_enabled, noise_source_name, DiodeWithCap
using Cadnip.MNA: solve_dc, MNASpec, MNACircuit
using Cadnip.MNA: THERMAL, SHOT, WHITE, FLICKER, NoiseSource
using Cadnip.MNA: NodeIndex, GroundIndex
using Cadnip.MNA: K_BOLTZMANN, Q_ELEMENTARY
using Cadnip.SpectreEnvironment

# A Verilog-A device carrying both noise kinds, one of them behind a runtime
# conditional. The two contribution-codegen paths a `white_noise` /
# `flicker_noise` call can sit on (unconditional branch stamping and inline
# stamping inside an `if`) are exercised by the same module.
va"""
module VANoisyRes(p, n);
    parameter real r = 1000.0;
    parameter real pwr = 4.0e-21;
    parameter real kfl = 1.0e-20;
    parameter integer noisy = 1;
    inout p, n;
    electrical p, n;
    analog begin
        I(p,n) <+ V(p,n)/r;
        I(p,n) <+ white_noise(pwr, "thermal");
        if (noisy > 0) begin
            I(p,n) <+ flicker_noise(kfl, 1.0, "flicker");
        end
    end
endmodule
"""

# Divider of two VA noisy resistors — used to check that the noise sources are
# invisible to the DC/transient value path.
function va_noisy_divider(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    ctx === nothing ? (ctx = MNAContext()) : reset_for_restamping!(ctx)
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)
    stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
    stamp!(VANoisyRes(), ctx, vcc, out; _mna_instance_=:x1, _mna_x_=x)
    stamp!(VANoisyRes(), ctx, out, 0; _mna_instance_=:x2, _mna_x_=x)
    stamp!(Capacitor(1e-9; name=:C1), ctx, out, 0)
    return ctx
end

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

    @testset "MOSFET channel thermal noise registration in saturation" begin
        # A MOSFET biased in saturation registers channel thermal noise
        # 4kT·(2/3)·gm between drain and source, with a = (2/3)·gm evaluated at
        # the operating-point transconductance. Hand-stamping at a known bias is a
        # low-level stamping-mechanics test.
        Vth, K = 0.5, 1e-3
        Vg, Vd = 1.5, 2.0              # Vgs = 1.5 > Vth, Vds = 2.0 > Vgs - Vth = 1.0
        gm = K * (Vg - Vth)           # square-law saturation gm = K·(Vgs - Vth)

        ctx = MNAContext()
        d = get_node!(ctx, :d)        # index 1
        g = get_node!(ctx, :g)        # index 2
        stamp!(SimpleMOSFET(Vth=Vth, K=K, lambda=0.0, name=:M1), ctx, d, g, 0; x=[Vd, Vg])

        @test num_noise_sources(ctx) == 1
        src = noise_sources(ctx)[1]
        @test src.kind === THERMAL
        @test src.name === :M1
        @test src.a ≈ (2/3) * gm      # effective channel noise conductance
        @test resolve_index(ctx, src.p) == d
        @test resolve_index(ctx, src.n) == 0

        # PSD = 4·k·T·(2/3)·gm, white
        T = 27.0
        expected = 4 * K_BOLTZMANN * (T + 273.15) * (2/3) * gm
        @test noise_psd(src, T, 1e3) ≈ expected
        @test noise_psd(src, T, 1e9) ≈ expected   # white: no frequency dependence
    end

    @testset "MOSFET in cutoff registers no channel noise" begin
        # Vgs <= Vth: the channel carries no current (gm == 0), so there is no
        # channel thermal noise to register.
        ctx = MNAContext()
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)
        stamp!(SimpleMOSFET(Vth=0.5, K=1e-3, name=:M1), ctx, d, g, 0; x=[1.0, 0.2])
        @test num_noise_sources(ctx) == 0
    end

    @testset "register_channel_thermal_noise! honors gamma" begin
        # The effective noise conductance scales with the excess-noise factor γ.
        ctx = MNAContext()
        d = get_node!(ctx, :d)
        s = get_node!(ctx, :s)
        register_channel_thermal_noise!(ctx, d, s, 2e-3; gamma=1.0, name=:Mg)
        src = noise_sources(ctx)[1]
        @test src.kind === THERMAL
        @test src.a ≈ 2e-3            # γ = 1 ⇒ a = gm
    end

    @testset "diode flicker noise registration at bias" begin
        # A junction with KF > 0 registers flicker noise KF·|I|^AF / f^FFE at the
        # operating-point current, alongside its (always-on) shot source. Hand-
        # stamping at a known bias is a low-level stamping-mechanics test.
        Is, Vt = 1e-14, 0.026
        KF, AF, FFE = 1e-15, 1.0, 1.0
        Vbias = 0.6
        I0 = Is * (exp(Vbias / Vt) - 1.0)

        ctx = MNAContext()
        a = get_node!(ctx, :a)
        stamp!(Diode(Is=Is, Vt=Vt, KF=KF, AF=AF, FFE=FFE, limit=false, name=:D1),
               ctx, a, 0; x=[Vbias])

        # Both shot and flicker sources are registered for this device.
        @test num_noise_sources(ctx) == 2
        flick = only(filter(s -> s.kind === FLICKER, noise_sources(ctx)))
        @test flick.name === :D1
        @test flick.a ≈ KF * abs(I0)^AF     # coefficient = KF·|I|^AF
        @test flick.b ≈ FFE
        @test resolve_index(ctx, flick.p) == a

        # PSD = KF·|I|^AF / f^FFE — rolls off as 1/f for FFE = 1.
        S(f) = KF * abs(I0)^AF / f^FFE
        @test noise_psd(flick, 27.0, 10.0) ≈ S(10.0)
        @test noise_psd(flick, 27.0, 1e3) ≈ S(1e3)
        @test noise_psd(flick, 27.0, 10.0) ≈ 100 * noise_psd(flick, 27.0, 1e3)
    end

    @testset "diode flicker is off by default (KF = 0)" begin
        # Without a KF card only shot noise registers — the flicker path is inert,
        # so the default builtin diode's noise footprint is unchanged.
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        stamp!(Diode(Is=1e-14, Vt=0.026, limit=false, name=:D1), ctx, a, 0; x=[0.6])
        @test isempty(filter(s -> s.kind === FLICKER, noise_sources(ctx)))
        @test num_noise_sources(ctx) == 1        # shot only
    end

    @testset "DiodeWithCap flicker noise registration" begin
        Is, Vt = 1e-14, 0.026
        KF, AF = 2e-16, 1.0
        Vbias = 0.5
        I0 = Is * (exp(Vbias / Vt) - 1.0)

        ctx = MNAContext()
        a = get_node!(ctx, :a)
        stamp!(DiodeWithCap(Is=Is, Vt=Vt, KF=KF, AF=AF, name=:D1), ctx, a, 0; x=[Vbias])

        flick = only(filter(s -> s.kind === FLICKER, noise_sources(ctx)))
        @test flick.a ≈ KF * abs(I0)^AF
        @test flick.b ≈ 1.0
    end

    @testset "MOSFET flicker noise registration in conduction" begin
        # A conducting MOSFET with KF > 0 registers drain→source flicker noise
        # KF·|Ids|^AF / f^FFE at the operating-point drain current, alongside its
        # channel thermal source.
        Vth, K = 0.5, 1e-3
        KF, AF, FFE = 1e-24, 2.0, 1.0
        Vg, Vd = 1.5, 2.0                     # saturation: Vgs=1.0>0, Vds=2.0>Vgs-Vth
        Ids = K / 2 * (Vg - Vth)^2            # square-law saturation current (lambda=0)

        ctx = MNAContext()
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)
        stamp!(SimpleMOSFET(Vth=Vth, K=K, lambda=0.0, KF=KF, AF=AF, FFE=FFE, name=:M1),
               ctx, d, g, 0; x=[Vd, Vg])

        @test num_noise_sources(ctx) == 2     # channel thermal + flicker
        flick = only(filter(s -> s.kind === FLICKER, noise_sources(ctx)))
        @test flick.name === :M1
        @test flick.a ≈ KF * abs(Ids)^AF
        @test flick.b ≈ FFE
        @test resolve_index(ctx, flick.p) == d
        @test resolve_index(ctx, flick.n) == 0
    end

    @testset "MOSFET in cutoff registers no flicker noise" begin
        # Ids == 0 in cutoff ⇒ no flicker source even with KF > 0.
        ctx = MNAContext()
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)
        stamp!(SimpleMOSFET(Vth=0.5, K=1e-3, KF=1e-24, name=:M1), ctx, d, g, 0; x=[1.0, 0.2])
        @test num_noise_sources(ctx) == 0
    end

    @testset "builtin flicker uses the shared register_flicker_noise! path" begin
        # The builtin diode/MOSFET stamps register through the same
        # `register_flicker_noise!(ctx, p, n, pwr, exp)` entry point the Verilog-A
        # `flicker_noise(pwr, exp)` lowering uses — no parallel construction.
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)
        register_flicker_noise!(ctx, p, n, 1e-18, 1.2; name=:Nf)
        src = only(noise_sources(ctx))
        @test src.kind === FLICKER
        @test src.a ≈ 1e-18
        @test src.b ≈ 1.2
        @test noise_psd(src, 27.0, 100.0) ≈ 1e-18 / 100.0^1.2
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

    #==========================================================================#
    # Verilog-A noise sources (N1)
    #
    # `white_noise`/`flicker_noise` lower to a value of 0.0 (unchanged
    # DC/transient numerics) plus a registration on the context's noise channel,
    # injected between the enclosing contribution's branch nodes.
    #==========================================================================#

    @testset "noise_source_name composes instance and model label" begin
        @test noise_source_name(:q1, :rb) === :q1_rb
        @test noise_source_name(:q1, Symbol("")) === :q1
        @test noise_source_name(Symbol(""), :rb) === :rb
        @test noise_source_name(Symbol(""), Symbol("")) === :va
    end

    @testset "VA white_noise/flicker_noise registration" begin
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        b = get_node!(ctx, :b)
        stamp!(VANoisyRes(), ctx, a, b; _mna_instance_=:x1)

        srcs = noise_sources(ctx)
        @test length(srcs) == 2

        w = only(filter(s -> s.kind === WHITE, srcs))
        @test w.name === :x1_thermal        # instance + the model's label string
        @test w.a ≈ 4.0e-21                 # white_noise(pwr) ⇒ S = pwr
        @test resolve_index(ctx, w.p) == a  # injected on the contribution's branch
        @test resolve_index(ctx, w.n) == b

        f = only(filter(s -> s.kind === FLICKER, srcs))
        @test f.name === :x1_flicker
        @test f.a ≈ 1.0e-20                 # flicker_noise(pwr, exp) ⇒ S = pwr/f^exp
        @test f.b ≈ 1.0
        @test noise_psd(f, 27.0, 100.0) ≈ 1.0e-22
    end

    @testset "VA noise follows conditionals and scales with \$mfactor" begin
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        # noisy=0 switches off the flicker contribution, so only the
        # unconditional white source registers; m=4 parallel devices carry four
        # times the (independent, hence additive) noise power.
        stamp!(VANoisyRes(noisy=0), ctx, a, 0; _mna_instance_=:x1, _mna_mfactor_=4.0)

        srcs = noise_sources(ctx)
        @test length(srcs) == 1
        @test srcs[1].kind === WHITE
        @test srcs[1].a ≈ 4 * 4.0e-21
    end

    @testset "VA noise is invisible to the value path" begin
        # The device solves as a plain 1 kΩ resistor — the noise contributions
        # evaluate to 0 A exactly as they did before the noise channel existed.
        # The transient additionally drives DirectStampContext restamping, where
        # `noise_enabled` is false and the whole registration branch (the noise
        # power expressions included) folds away.
        @test noise_enabled(MNAContext())
        circuit = MNACircuit(va_noisy_divider)
        @test dc!(circuit)[:out] ≈ 2.5 rtol=1e-6
        sol = tran!(circuit, (0.0, 1e-5))
        @test sol[:out][end] ≈ 2.5 rtol=1e-3
    end
end
