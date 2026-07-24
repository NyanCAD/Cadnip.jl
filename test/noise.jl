#==============================================================================#
# Small-signal noise analysis (noise!) — behavioral tests.
#
# These drive the high-level API on SPICE netlists and check the output-noise
# PSD against textbook analytical results:
#   - a resistor divider: white output noise 4kT·(R1‖R2);
#   - an RC low-pass: resistor thermal noise 4kTR shaped by the RC pole,
#     S_vout(f) = 4kTR / (1 + (2πfRC)²), whose band integral is the famous kT/C.
#==============================================================================#

using Test
using Cadnip
using Cadnip.MNA
using Cadnip.SpectreEnvironment
using Cadnip.MNA: K_BOLTZMANN

# Boltzmann·T at the default 27 °C operating temperature.
const _kT = K_BOLTZMANN * (27.0 + 273.15)

@testset "noise! analysis" begin

    @testset "resistor divider: white output noise 4kT·(R1‖R2)" begin
        # V1 is DC-only, so for noise it is an AC short: node `in` sits at AC
        # ground. Output node `out` then sees R1 to ground and R2 to ground,
        # i.e. the two thermal current sources both see Z = R1‖R2.
        circuit = MNACircuit(sp"""
        V1 in 0 DC 0
        R1 in out 1k
        R2 out 0 1k
        """i)

        ns = noise!(circuit, :out; freqs=[1.0, 1e3, 1e6])

        Rpar = 500.0                       # 1k ‖ 1k
        expected = 4 * _kT * Rpar          # V²/Hz, white
        @test all(≈(expected; rtol=1e-6), ns[:onoise])

        # Two equal sources → each contributes half, and they sum to the total.
        # (SPICE device names are normalized to lower case.)
        @test ns[:r1] ≈ ns[:r2]
        @test ns[:r1] .+ ns[:r2] ≈ ns[:onoise]
    end

    @testset "RC low-pass: thermal noise shaped by the RC pole" begin
        R = 1e3
        C = 1e-6
        circuit = MNACircuit(sp"""
        V1 in 0 DC 0
        R1 in out 1k
        C1 out 0 1u
        """i)

        freqs = acdec(10, 1.0, 1e7)
        ns = noise!(circuit, :out; freqs=freqs)

        # S_vout(f) = 4kTR / (1 + (2πfRC)²)
        expected = [4 * _kT * R / (1 + (2π * f * R * C)^2) for f in freqs]
        @test ns[:onoise] ≈ expected rtol=1e-6

        # Low-frequency plateau is the bare resistor thermal noise 4kTR.
        @test ns[:onoise][1] ≈ 4 * _kT * R rtol=1e-3

        # Single source: its contribution is the whole output noise.
        @test ns[:r1] ≈ ns[:onoise]
    end

    @testset "kT/C: band-integrated RC noise" begin
        # Integrating S_vout over all frequency gives the classic kT/C result.
        # Use a dense linear grid well past the pole (f_c ≈ 159 kHz) so the
        # trapezoidal integral captures the tail.
        C = 1e-6
        circuit = MNACircuit(sp"""
        V1 in 0 DC 0
        R1 in out 1k
        C1 out 0 1u
        """i)

        freqs = collect(range(0.0, 5e6; length=200_001))
        ns = noise!(circuit, :out; freqs=freqs)
        vrms2 = total_noise(ns)^2
        @test vrms2 ≈ _kT / C rtol=2e-2
    end

    @testset "input-referred: resistor divider" begin
        # Gain from the input source V1 (drives node `in`) to `out` is the
        # divider ratio R2/(R1+R2) = 0.5, flat in frequency. Input-referring the
        # white output noise divides by that gain²: S_in = 4kT·(R1‖R2)/0.25.
        circuit = MNACircuit(sp"""
        V1 in 0 DC 0
        R1 in out 1k
        R2 out 0 1k
        """i)

        ns = noise!(circuit, :out; freqs=[1.0, 1e3, 1e6], input=:V1)

        @test all(≈(0.5; rtol=1e-6), real.(ns.gain))
        @test all(≈(0.0; atol=1e-9), imag.(ns.gain))

        Rpar = 500.0
        expected_in = 4 * _kT * Rpar / 0.25     # = 4kT·2000
        @test all(≈(expected_in; rtol=1e-6), ns[:inoise])
        @test ns[:inoise] ≈ ns[:onoise] ./ abs2.(ns.gain)
    end

    @testset "input-referred: RC low-pass flattens to 4kTR" begin
        # The input→output gain 1/(1+jωRC) shapes the output noise by exactly the
        # same pole that shapes the thermal noise, so input-referred noise is the
        # bare resistor thermal noise 4kTR — flat across the whole band.
        R = 1e3
        circuit = MNACircuit(sp"""
        V1 in 0 DC 0
        R1 in out 1k
        C1 out 0 1u
        """i)

        freqs = acdec(10, 1.0, 1e7)
        ns = noise!(circuit, :out; freqs=freqs, input=:V1)

        @test all(≈(4 * _kT * R; rtol=1e-6), ns[:inoise])
        @test total_noise(ns; referred=:input)^2 ≈
              4 * _kT * R * (freqs[end] - freqs[1]) rtol=1e-6
    end

    @testset "errors and edge cases" begin
        circuit = MNACircuit(sp"""
        V1 in 0 DC 0
        R1 in out 1k
        R2 out 0 1k
        """i)

        @test_throws ArgumentError noise!(circuit, :out; freqs=Float64[])
        ns = noise!(circuit, :out; freqs=[1e3])
        @test_throws ErrorException ns[:nonexistent_source]
        @test onoise(ns) === ns[:onoise]

        # No input requested: input-referred access and integration both error.
        @test_throws ErrorException ns[:inoise]
        @test_throws ErrorException total_noise(ns; referred=:input)
        @test_throws ArgumentError total_noise(ns; referred=:bogus)

        # An input that is not an independent voltage source is rejected.
        @test_throws ErrorException noise!(circuit, :out; freqs=[1e3], input=:R1)
    end
end
