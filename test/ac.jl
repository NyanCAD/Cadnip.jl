using CedarSim
using CedarSim.MNA
using Test
using DescriptorSystems
using LinearAlgebra

# Third order low pass filter, butterworth, with ω_c = 1
# L1=1.5H, C2=1.333F, L3=0.5H, R=1Ω
circuit = sp"""
*Third order low pass filter
V1 vin 0 AC 1
L1 vin n1 1.5
C2 n1 0 1.333333
L3 n1 vout 0.5
R4 vout 0 2
R5 vout 0 2
"""

circ = MNACircuit(circuit)
ac = ac!(circ; reltol=1e-6, abstol=1e-6)

ωs = 2π .* acdec(20, 0.01, 10)
sys = IRODESystem(ac)
resp_sim = DescriptorSystems.freqresp(ac, sys.node_vout, ωs)

# Analytical: H(s) = 1/((s+1)*(s²+s+1))
function butterworth_tf(ωs)
    resp = similar(ωs, ComplexF64)
    for (i, ω) in enumerate(ωs)
        s = im * ω
        resp[i] = 1 / ((s + 1) * (s^2 + s + 1))
    end
    return resp
end
resp_an = butterworth_tf(ωs)

@test resp_sim ≈ resp_an rtol=1e-4

@test all(DescriptorSystems.freqresp(ac, sys.node_vin, ωs) .≈ 1.0)

# State space conversion test
dss_sys = ac[sys.node_vout]
fr = DescriptorSystems.freqresp(dss_sys, ωs)
@test vec(fr[1, 1, :]) ≈ resp_an rtol=1e-4

println("AC tests passed!")
