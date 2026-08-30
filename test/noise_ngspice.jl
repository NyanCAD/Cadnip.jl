#==============================================================================#
# Small-signal noise analysis (noise!) — validation against ngspice.
#
# `test/noise.jl` checks `noise!` against closed-form results (4kT·R through an
# RC pole, kT/C). That covers the machinery but not the *device* noise models,
# which is where a simulator actually earns trust: the shot/flicker split of a
# junction, the channel thermal noise of a MOSFET, the five sources SPICE3 gives
# a BJT. This file pins those against ngspice.
#
# The decks in `test/ngspice/noise/` are read by both simulators. Each carries
# its own `.noise` and `.print noise` cards, so the reference is regenerated with
#
#     ngspice -b test/ngspice/noise/<deck>.cir
#
# and Cadnip loads the same file (sema ignores the analysis and output cards).
# There is no second copy of the netlist to drift out of step.
#
# ngspice reports noise as a spectral *density* in V/√Hz; `noise!` reports a PSD
# in V²/Hz, so the comparisons are against `sqrt.(ns[:onoise])`.
#
# Reference generated with ngspice-42 at the default TEMP = TNOM = 27 °C. The
# tolerance is 1e-4 throughout: the two agree to ~1e-7 relative, and the floor is
# set by the Boltzmann constant (ngspice carries the pre-2019 CODATA value, worth
# 2e-5 in a thermal PSD) rather than by anything either simulator does.
#==============================================================================#

using Test
using Cadnip
using Cadnip.MNA
using Cadnip.SpectreEnvironment
using VADistillerModels   # sp_diode / sp_mos1 / sp_bjt behind the `d`/`nmos`/`npn` tiers

const NGSPICE_NOISE_DECKS = joinpath(@__DIR__, "ngspice", "noise")

# The decks have to be loaded at module top level: `Base.include(mod,
# SpiceFile(path))` defines the builder as a top-level function, which is what
# keeps the `@testset` bodies below clear of world age (see CLAUDE.md,
# "File-First Circuit Loading").
Base.include(@__MODULE__, SpiceFile(joinpath(NGSPICE_NOISE_DECKS, "rc_lowpass.cir")))
Base.include(@__MODULE__, SpiceFile(joinpath(NGSPICE_NOISE_DECKS, "diode.cir")))
Base.include(@__MODULE__, SpiceFile(joinpath(NGSPICE_NOISE_DECKS, "mos1.cir")))
Base.include(@__MODULE__, SpiceFile(joinpath(NGSPICE_NOISE_DECKS, "mos1_flicker.cir")))
Base.include(@__MODULE__, SpiceFile(joinpath(NGSPICE_NOISE_DECKS, "bjt.cir")))

# The `dec 2 1 1e6` grid every deck asks for, spelled out so the reference rows
# below line up with it positionally.
const NGSPICE_FREQS = [1.0, 3.162278, 10.0, 31.62278, 100.0, 316.2278, 1e3,
                       3162.278, 1e4, 31622.78, 1e5, 316227.8, 1e6]

const NGSPICE_RTOL = 1e-4

"""
    ngspice_noise(ns) -> (onoise, inoise)

The two quantities ngspice's `.noise` prints, in the units it prints them:
spectral densities in V/√Hz, from `noise!`'s V²/Hz PSDs.
"""
ngspice_noise(ns) = (sqrt.(ns[:onoise]), sqrt.(ns[:inoise]))

"""
    contributions_sum(ns) -> Vector{Float64}

Incoherent sum of the per-source contributions. Equal to the total output PSD by
construction; checked per circuit so a source that is registered but dropped from
the decomposition shows up.
"""
contributions_sum(ns) = sum(values(ns.contributions))

#------------------------------------------------------------------------------#
# Reference tables: [onoise_spectrum inoise_spectrum] in V/√Hz, row-aligned to
# NGSPICE_FREQS, straight from `ngspice -b <deck>.cir`.
#------------------------------------------------------------------------------#

const RC_REF = [
    1.287481e-08  1.287481e-08
    1.287481e-08  1.287481e-08
    1.287480e-08  1.287481e-08
    1.287478e-08  1.287481e-08
    1.287455e-08  1.287481e-08
    1.287227e-08  1.287481e-08
    1.284947e-08  1.287481e-08
    1.262795e-08  1.287481e-08
    1.090152e-08  1.287481e-08
    5.788057e-09  1.287481e-08
    2.023620e-09  1.287481e-08
    6.471598e-10  1.287481e-08
    2.048830e-10  1.287481e-08
]

const DIODE_REF = [
    1.230539e-07  2.089608e-05
    6.920070e-08  1.175114e-05
    3.891873e-08  6.608885e-06
    2.189328e-08  3.717751e-06
    1.232512e-08  2.092959e-06
    6.955099e-09  1.181063e-06
    3.953822e-09  6.714083e-07
    2.297649e-09  3.901694e-07
    1.416023e-09  2.404583e-07
    9.847683e-10  1.672259e-07
    8.014746e-10  1.361004e-07
    7.340479e-10  1.246505e-07
    7.113970e-10  1.208041e-07
]

# Thermal only (`kf` absent), so the curve is white across the band.
const MOS1_REF = [
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
    1.092414e-08  2.706065e-09
]

# The same stage with `kf=1e-30 af=1`, which puts the flicker corner above the
# band: ngspice's curve is 1/f all the way to 1 MHz.
const MOS1_FLICKER_REF = [
    5.761168e-06  1.427124e-06
    3.239755e-06  8.025336e-07
    1.821871e-06  4.513034e-07
    1.024553e-06  2.537964e-07
    5.762193e-07  1.427377e-07
    3.241578e-07  8.029852e-08
    1.825110e-07  4.521058e-08
    1.030302e-07  2.552206e-08
    5.863813e-08  1.452550e-08
    3.418956e-08  8.469243e-09
    2.124255e-08  5.262084e-09
    1.497651e-08  3.709894e-09
    1.235021e-08  3.059324e-09
]

const BJT_REF = [
    2.231740e-05  2.354841e-07
    1.255051e-05  1.324279e-07
    7.058590e-06  7.447937e-08
    3.970971e-06  4.190006e-08
    2.235943e-06  2.359276e-08
    1.262510e-06  1.332150e-08
    7.190383e-07  7.586999e-09
    4.200776e-07  4.432487e-09
    2.622470e-07  2.767124e-09
    1.863288e-07  1.966065e-09
    1.547554e-07  1.632916e-09
    1.433309e-07  1.512369e-09
    1.395236e-07  1.472196e-09
]

@testset "noise! vs ngspice" begin

    @testset "RC low-pass: resistor thermal noise through the pole" begin
        # The one source in the circuit is R1's thermal noise, so this is the
        # machinery under test with no device model in the way: the shaping by
        # the RC pole (output) and its exact cancellation against the same pole
        # in the gain (input-referred, flat at 4kT·R).
        ns = noise!(MNACircuit(rc_lowpass), :out; freqs=NGSPICE_FREQS, input=:V1)
        onoise, inoise = ngspice_noise(ns)

        @test collect(keys(ns.contributions)) == [:r1]
        @test isapprox(onoise, RC_REF[:, 1]; rtol=NGSPICE_RTOL)
        @test isapprox(inoise, RC_REF[:, 2]; rtol=NGSPICE_RTOL)
        @test isapprox(contributions_sum(ns), ns[:onoise]; rtol=1e-12)
    end

    @testset "diode: junction shot noise plus flicker" begin
        # Forward-biased at ~0.44 mA through a 10k load. `kf=1e-14 af=1` puts the
        # flicker corner near 30 kHz, so the low decades exercise the 1/f source
        # and the high ones the shot/thermal floor.
        ns = noise!(MNACircuit(diode), :out; freqs=NGSPICE_FREQS, input=:V1)
        onoise, inoise = ngspice_noise(ns)

        @test isapprox(onoise, DIODE_REF[:, 1]; rtol=NGSPICE_RTOL)
        @test isapprox(inoise, DIODE_REF[:, 2]; rtol=NGSPICE_RTOL)
        @test isapprox(contributions_sum(ns), ns[:onoise]; rtol=1e-12)
    end

    @testset "level-1 MOSFET: channel thermal noise" begin
        # Common-source stage in saturation (Id ≈ 1.03 mA, gm ≈ 2.06 mS). The
        # `tox` on the model card is what gives the device a gate charge, and a
        # gate charge is what used to shift its limit variables out from under
        # the linearization — the operating point came out at Vgs = 1.5 V with
        # half the real gm, and this white level with it (`limit_state_index`).
        circuit = MNACircuit(mos1)
        op = dc!(circuit)
        @test isapprox(op[:m1_vgs], 2.0; rtol=1e-9)          # not 1.5
        @test isapprox(op[:m1_gm], 2.058823e-3; rtol=1e-6)   # ngspice @m1[gm]
        @test isapprox(op[:m1_i_d], 1.029412e-3; rtol=1e-6)  # ngspice @m1[id]

        ns = noise!(circuit, :d; freqs=NGSPICE_FREQS, input=:Vg)
        onoise, inoise = ngspice_noise(ns)

        @test isapprox(onoise, MOS1_REF[:, 1]; rtol=NGSPICE_RTOL)
        @test isapprox(inoise, MOS1_REF[:, 2]; rtol=NGSPICE_RTOL)
        @test isapprox(contributions_sum(ns), ns[:onoise]; rtol=1e-12)
    end

    @testset "level-1 MOSFET: flicker noise (broken)" begin
        # `sp_mos1` registers a flicker source and then leaves its power at zero:
        # the PSD does not respond to `kf` (1e-30 and 1e-20 give the same curve)
        # or to `nlev`, so the deck's `kf=1e-30 af=1` produces the same white
        # floor as the deck without it, while ngspice's curve is 1/f across the
        # whole band. The `flicker_psd` in `models/VADistillerModels.jl/va/mos1.va`
        # is assigned inside a `case (nlev)`; nothing else in the model's noise
        # block is conditional, and every other device's flicker term is right,
        # which is what points at the lowering of that statement. See
        # `doc/noise_ngspice_validation.md` — it is why there is a second deck.
        #
        # Only a `@test_broken`: asserting the zero would turn the fix into a
        # failure. This flips to an "Unexpected Pass" the moment the flicker
        # source carries its power, which is the announcement wanted here.
        ns = noise!(MNACircuit(mos1_flicker), :d; freqs=NGSPICE_FREQS, input=:Vg)
        onoise, _ = ngspice_noise(ns)

        @test_broken isapprox(onoise, MOS1_FLICKER_REF[:, 1]; rtol=NGSPICE_RTOL)
    end

    @testset "BJT: the whole SPICE3 noise model" begin
        # Common-emitter stage. The model carries seven sources — thermal noise
        # of rb/re/rc, base and collector shot noise, base flicker noise, and the
        # load resistor — so this is the deck where the decomposition itself is
        # under test.
        ns = noise!(MNACircuit(bjt), :c; freqs=NGSPICE_FREQS, input=:Vb)
        onoise, inoise = ngspice_noise(ns)

        @test issetequal(keys(ns.contributions),
                         [:q1_flicker, :q1_ib, :q1_ic, :q1_rb, :q1_rc, :q1_re, :rc])
        @test isapprox(onoise, BJT_REF[:, 1]; rtol=NGSPICE_RTOL)
        @test isapprox(inoise, BJT_REF[:, 2]; rtol=NGSPICE_RTOL)
        @test isapprox(contributions_sum(ns), ns[:onoise]; rtol=1e-12)
    end

end
