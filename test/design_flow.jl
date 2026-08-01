#==============================================================================#
# Design flow: an NMOS common-source amplifier, hand derivation → simulation
#
# This is a UX test as much as a numerical one. It walks the steps a designer
# takes on a new stage, in order, through the high-level API only:
#
#   1. size the device and pick the bias by hand (square law)
#   2. `dc!`   — check the operating point lands where the hand math says
#   3. `dc!(CircuitSweep(...))` — DC transfer curve, small-signal gain as its slope
#   4. `ac!`   — midband gain and the load-pole bandwidth
#   5. `tran!` — the same gain on a real waveform, unclipped
#   6. `noise!` — output and input-referred noise
#
# Every step reads a `.param` the design is parameterized on, so the whole file
# also pins the parameterization contract: a `.param` a designer sweeps has to
# reach the device or source it feeds. When it does not, this reads as a
# perfectly flat transfer curve rather than as an error (see `test/params.jl`).
#
# Hand derivation (Shichman-Hodges, level 1):
#   K    = kp·W/L      = 100 µA/V² · 20      = 2 mA/V²
#   VOV  = VGS − VTO   = 1.1472 − 0.7        = 447.2 mV
#   ID   = ½·K·VOV²                          = 200 µA
#   gm   = K·VOV                             = 894.4 µS
#   VD   = VDD − ID·RD = 5 − 200µ·10k        = 3.0 V      (saturation: VD > VOV ✓)
#   Av   = −gm·RD                            = −8.94 V/V  (19.0 dB)
#   f₋₃dB ≈ 1/(2π·RD·CL) = 1/(2π·10k·1p)     = 15.9 MHz
#
# λ = 0.01 V⁻¹ shifts the simulated values a few percent off the λ=0 hand
# numbers (ro = 1/(λ·ID) = 500 kΩ loads RD, ID rises with VDS); the tolerances
# below are sized for that, not for sloppiness.
#==============================================================================#

module design_flow_tests

using Test
using Cadnip
using SciMLBase
using VADistillerModels                       # .model nch nmos level=1 → MOS1
using Cadnip.MNA: MNACircuit, nameat, alter
using Cadnip: dc!, ac!, tran!, noise!, acdec, magnitude_db, total_noise
using Cadnip: CircuitSweep, Sweep

# Hand-derived design targets
const KP    = 100e-6
const WL    = 20.0
const K     = KP * WL
const VTO   = 0.7
const VBIAS = 1.1472
const VOV   = VBIAS - VTO
const ID    = 0.5 * K * VOV^2                 # 200 µA
const GM    = K * VOV                         # 894 µS
const RD    = 10e3
const CL    = 1e-12
const AV    = -GM * RD                        # −8.94 V/V

const cs_amp = sp"""
.model nch nmos level=1 vto=0.7 kp=100u lambda=0.01
.param vbias=1.1472
.param rd=10k
.param vsup=5
.param vac=5m
.param freq=1meg
Vdd vdd 0 DC vsup
Vin gate 0 DC vbias AC 1 SIN vbias vac freq
M1 drain gate 0 0 nch w=20u l=1u
Rd vdd drain rd
CL drain 0 1p
"""i

@testset "CS amplifier design flow" begin

    circuit = MNACircuit(cs_amp)

    #==========================================================================#
    # Step 2: operating point. Is the bias where the hand math put it, and is
    # the device actually in saturation?
    #==========================================================================#
    @testset "operating point" begin
        op = dc!(circuit)

        @test op[:gate] ≈ VBIAS
        @test op[:vdd] ≈ 5.0

        # Drain current, reported by the device itself — no inferring it from a
        # branch that happens to carry it (doc/operating_point_info.md).
        id = op[:i_m1_d]
        @test isapprox(id, ID; rtol=0.05)
        # It is the supply current here, because Rd carries all of it; that
        # identity is a check on the report, not the way to obtain it.
        @test isapprox(id, -op[:I_vdd]; rtol=1e-6)
        @test isapprox(id, op[:i_rd_p]; rtol=1e-6)
        # A DC gate draws nothing, and the four terminals conserve charge.
        @test op[:i_m1_g] ≈ 0.0 atol=1e-9
        @test op[:i_m1_d] + op[:i_m1_g] + op[:i_m1_s] + op[:i_m1_b] ≈ 0.0 atol=1e-9
        @test isapprox(op[:drain], 5.0 - id * RD; rtol=1e-6)   # KVL on the load
        @test isapprox(op[:drain], 3.0; rtol=0.05)

        # Saturation headroom: VDS > VOV, with room for the output swing.
        @test op[:drain] > VOV

        # The bias is a parameter, so re-biasing is a re-parameterization.
        hotter = dc!(alter(circuit; vbias=VBIAS + 0.05))
        @test hotter[:drain] < op[:drain]          # more gate drive → more ID
    end

    #==========================================================================#
    # Step 3: DC transfer curve. The small-signal gain is its slope at the bias
    # — the first independent check on Av, and the check that says how much
    # input swing stays in the linear region.
    #==========================================================================#
    @testset "DC transfer curve" begin
        vbiases = 1.05:0.05:1.30
        sweep = CircuitSweep(cs_amp, Sweep(vbias = vbiases); vbias=VBIAS)
        result = dc!(sweep)

        vin = [p.vbias for (p, _) in result]
        vout = [sol[:drain] for (_, sol) in result]

        @test vin == collect(vbiases)
        # A swept bias has to move the operating point.
        @test issorted(vout; rev=true)
        @test vout[end] < vout[1] - 1.0

        # Slope at the design bias, centered difference across it.
        i = findfirst(≈(1.15), vin)
        slope = (vout[i+1] - vout[i-1]) / (vin[i+1] - vin[i-1])
        @test isapprox(slope, AV; rtol=0.10)
    end

    #==========================================================================#
    # Step 4: AC. Midband gain and the pole the load capacitor sets.
    #==========================================================================#
    @testset "AC response" begin
        freqs = acdec(10, 1e3, 1e10)
        ac = ac!(circuit, freqs)
        db = magnitude_db(ac, :drain)

        # Midband gain, against the hand number in dB.
        @test isapprox(db[1], 20log10(abs(AV)); atol=0.5)

        # −3 dB corner within a decade-step of 1/(2π·RD·CL).
        i3 = findfirst(<(db[1] - 3), db)
        @test i3 !== nothing
        @test isapprox(freqs[i3], 1 / (2π * RD * CL); rtol=0.35)

        # Rolls off past the pole (single-pole load: −20 dB/decade).
        @test db[end] < db[1] - 40
    end

    #==========================================================================#
    # Step 5: transient. The same gain on a waveform, and the swing stays out
    # of the rails and out of triode.
    #==========================================================================#
    @testset "transient gain" begin
        sol = tran!(circuit, (0.0, 5e-6))
        @test SciMLBase.successful_retcode(sol)

        ts = range(3e-6, 5e-6; length=200)       # after the initial transient
        vo = [nameat(sol, :drain, t) for t in ts]
        vi = [nameat(sol, :gate, t) for t in ts]

        gain = (maximum(vo) - minimum(vo)) / (maximum(vi) - minimum(vi))
        @test isapprox(gain, abs(AV); rtol=0.10)

        # Unclipped: the output stays inside the supply and above VOV.
        @test minimum(vo) > VOV
        @test maximum(vo) < 5.0

        # Driving harder must swing harder — the source amplitude is a `.param`.
        loud = tran!(alter(circuit; vac=10e-3), (0.0, 5e-6))
        vo_loud = [nameat(loud, :drain, t) for t in ts]
        @test isapprox((maximum(vo_loud) - minimum(vo_loud)) /
                       (maximum(vo) - minimum(vo)), 2.0; rtol=0.10)
    end

    #==========================================================================#
    # Step 6: noise. Channel thermal noise of M1 plus Johnson noise of Rd,
    # referred back to the input through the gain computed above.
    #==========================================================================#
    @testset "noise" begin
        ns = noise!(circuit, :drain; freqs=acdec(10, 1e3, 1e9), input=:Vin)

        @test total_noise(ns) > 0                # V rms at the drain
        @test total_noise(ns; referred=:input) > 0

        # In the midband the input-referred PSD is the output PSD over |Av|².
        @test isapprox(ns[:onoise][1] / ns[:inoise][1], AV^2; rtol=0.05)

        # Both noise sources of this stage show up by name: the load resistor's
        # Johnson noise, and M1's channel thermal noise (`m1_id`) from the model.
        @test haskey(ns.contributions, :rd)
        @test haskey(ns.contributions, :m1_id)

        # 4kT·(2/3)·gm of M1 against 4kT/RD: with gm·RD ≈ 9, the transistor is
        # the dominant noise source of the stage.
        @test sum(ns.contributions[:m1_id]) > sum(ns.contributions[:rd])
    end
end

end # module design_flow_tests
