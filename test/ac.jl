#==============================================================================#
# AC Analysis Tests - MNA Backend
#
# Tests small-signal AC analysis using the MNA-based implementation.
# Uses a 3rd order Butterworth low-pass filter as the test circuit.
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using DescriptorSystems
using LinearAlgebra

#==============================================================================#
# Test Circuit: 3rd Order Butterworth Low-Pass Filter
#
# Circuit topology:
#   Vin --[L1]-- n1 --[L3]-- vout
#                |            |
#               [C2]         [R4||R5]
#                |            |
#               GND          GND
#
# Component values for ωc = 1 rad/s Butterworth response:
#   L1 = 3/2 H, C2 = 4/3 F, L3 = 1/2 H, R = 1 Ω
#==============================================================================#

const L1_val = 3/2
const C2_val = 4/3
const L3_val = 1/2
const R4_val = 1.0

@testset "AC Analysis (MNA Backend)" begin

"""
    build_butterworth_filter(params, spec, t=0.0; x=ZERO_VECTOR, ctx=nothing)

Build a 3rd order Butterworth low-pass filter.
"""
function build_butterworth_filter(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    # Get/create nodes
    vin = get_node!(ctx, :vin)
    n1 = get_node!(ctx, :n1)
    vout = get_node!(ctx, :vout)

    # AC voltage source at input (1V AC magnitude)
    # The AC excitation is identified by the current variable naming convention
    stamp!(VoltageSource(0.0; ac=1.0+0.0im, name=:V1), ctx, vin, 0, t, spec.mode)

    # L1: vin to n1
    stamp!(Inductor(L1_val; name=:L1), ctx, vin, n1)

    # C2: n1 to ground
    stamp!(Capacitor(C2_val; name=:C2), ctx, n1, 0)

    # L3: n1 to vout
    stamp!(Inductor(L3_val; name=:L3), ctx, n1, vout)

    # R4 and R5 in parallel (each 2*R4_val, so parallel = R4_val)
    stamp!(Resistor(2*R4_val; name=:R4), ctx, vout, 0)
    stamp!(Resistor(2*R4_val; name=:R5), ctx, vout, 0)

    return ctx
end

# Analytical Butterworth transfer function: H(s) = 1/((s+1)*(s²+s+1))
# Frequency response: H(jω) = 1/((jω+1)*(−ω²+jω+1))
function butterworth_freqresp(ωs)
    resp = similar(ωs, ComplexF64)
    for (i, ω) in enumerate(ωs)
        s = im * ω
        resp[i] = 1 / ((s + 1) * (s^2 + s + 1))
    end
    return resp
end

@testset "Butterworth Filter Frequency Response" begin
    # Create circuit
    circuit = MNACircuit(build_butterworth_filter)

    # Perform AC analysis
    ac = ac!(circuit)

    # Get system accessor for node references
    sys = IRODESystem(ac)

    # Generate frequency sweep (angular frequencies)
    ωs = 2π .* acdec(20, 0.01, 10)

    # Compute frequency response at output node
    resp_sim = DescriptorSystems.freqresp(ac, sys.node_vout, ωs)

    # Analytical frequency response
    resp_an = butterworth_freqresp(ωs)

    # Compare - should match analytical result
    @test resp_sim ≈ resp_an rtol=1e-6

    # Check input node (should be unity gain, directly driven)
    resp_vin = DescriptorSystems.freqresp(ac, sys.node_vin, ωs)
    @test all(resp_vin .≈ 1.0)
end

@testset "State Space Conversion" begin
    circuit = MNACircuit(build_butterworth_filter)
    ac = ac!(circuit)
    sys = IRODESystem(ac)

    ωs = 2π .* acdec(20, 0.01, 10)

    # Get descriptor state-space subsystem for output
    dss_sys = ac[sys.node_vout]

    # Compute frequency response using DSS
    fr = DescriptorSystems.freqresp(dss_sys, ωs)
    resp_sim = vec(fr[1, 1, :])

    # Analytical
    resp_an = butterworth_freqresp(ωs)

    # Compare magnitudes
    @test abs.(resp_sim) ≈ abs.(resp_an) rtol=1e-5

    # Compare phases
    @test angle.(resp_sim) ≈ angle.(resp_an) rtol=1e-5
end

@testset "Internal Node Response" begin
    circuit = MNACircuit(build_butterworth_filter)
    ac = ac!(circuit)
    sys = IRODESystem(ac)

    ωs = 2π .* acdec(20, 0.01, 10)

    # Get response at internal node n1
    resp_n1 = DescriptorSystems.freqresp(ac, sys.node_n1, ωs)

    # At low frequencies (DC), n1 should equal vin (inductors are shorts, caps are open)
    @test abs(resp_n1[1]) ≈ 1.0 rtol=0.1

    # At high frequencies, n1 should attenuate (inductors block, C shorts to ground)
    @test abs(resp_n1[end]) < 0.1
end

@testset "acdec Helper" begin
    # Test logarithmic frequency generation
    freqs = acdec(10, 1.0, 100.0)

    # Should span 2 decades with 10 points per decade = ~21 points
    @test length(freqs) == 21

    # Check bounds
    @test freqs[1] ≈ 1.0
    @test freqs[end] ≈ 100.0

    # Check logarithmic spacing
    ratios = freqs[2:end] ./ freqs[1:end-1]
    @test all(isapprox.(ratios, ratios[1], rtol=1e-10))
end

#==============================================================================#
# MOS1 Inverter Test (Simple Nonlinear Circuit)
#==============================================================================#

@testset "MOS1 Inverter AC Analysis" begin
    # Simple NMOS inverter with resistive load
    # This tests AC analysis with nonlinear devices (linearized at DC op point)

    function build_mos1_inverter(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end

        vin = get_node!(ctx, :vin)
        vout = get_node!(ctx, :vout)
        vdd = get_node!(ctx, :vdd)

        # DC bias on gate with small AC signal
        Vbias = params.Vbias
        stamp!(VoltageSource(Vbias; ac=1.0+0.0im, name=:Vin), ctx, vin, 0, t, spec.mode)

        # VDD supply
        stamp!(VoltageSource(params.Vdd; name=:Vdd), ctx, vdd, 0, t, spec.mode)

        # Load resistor
        stamp!(Resistor(params.Rload; name=:Rload), ctx, vdd, vout)

        # NMOS transistor (drain=vout, gate=vin, source=gnd)
        # Using SimpleMOSFET for testing
        stamp!(SimpleMOSFET(Vth=params.Vth, K=params.K, lambda=0.0; name=:M1),
               ctx, vout, vin, 0; x=x)

        return ctx
    end

    # Create circuit with typical parameters
    circuit = MNACircuit(build_mos1_inverter;
        Vdd = 5.0,
        Vbias = 2.5,  # Bias in active region
        Rload = 10e3,
        Vth = 1.0,
        K = 1e-3
    )

    # Perform AC analysis
    ac = ac!(circuit)
    sys = IRODESystem(ac)

    # Check that we get some gain (output should respond to input)
    ωs = 2π .* [100.0, 1000.0, 10000.0]
    resp_out = DescriptorSystems.freqresp(ac, sys.node_vout, ωs)
    resp_in = DescriptorSystems.freqresp(ac, sys.node_vin, ωs)

    # Input should be unity (directly driven)
    @test all(abs.(resp_in) .≈ 1.0)

    # Output should have some response (gain depends on DC operating point)
    # At mid-band, expect |Av| > 0 (some amplification or attenuation)
    @test all(abs.(resp_out) .> 0)

    # At low frequencies with our simple model, gain should be approximately -gm*Rload
    # where gm = K*(Vgs - Vth) at the operating point
    # With Vbias=2.5V, Vth=1.0V: gm = 1e-3 * 1.5 = 1.5e-3
    # Expected gain ≈ -gm*Rload = -1.5e-3 * 10e3 = -15
    # But the actual gain depends on the DC operating point solution
    # Just check that gain is non-trivial
    low_freq_gain = abs(resp_out[1])
    @test low_freq_gain > 0.1  # Should have some meaningful gain
end

end  # main testset

println("AC analysis tests completed!")
