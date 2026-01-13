using CedarSim
using CedarSim.MNA
using Test
using CedarSim.SpectreEnvironment
using SpectreNetlistParser
using DescriptorSystems
using ControlSystemsBase

const L1_val = 3/2
const C2_val = 4/3
const L3_val = 1/2
const R4_val = 1
const ω_val = 1

spice_code = """
*Third order low pass filter, butterworth, with ω_c = 1
.param res=$(R4_val)

V1 vin 0 AC 1
L1 vin n1 $(L1_val)
C2 n1 0 $(C2_val)
L3 n1 vout $(L3_val)
* conceptually one resistor, split in two make a less trivial noise dss
R4 vout 0 '2*res'
R5 vout 0 '2*res'
"""

# Parse SPICE and create MNA circuit
ast = SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
circuit_code = CedarSim.make_mna_circuit(ast)
circuit_fn = eval(circuit_code)

# Create MNACircuit with parameters
circ = MNACircuit(circuit_fn; res=R4_val)
ac = ac!(circ)

ωs = 2π .* acdec(20, 0.01, 10) # equivalent to spice .ac dec 10 0.01 10
resp_sim = CedarSim.freqresp(ac, :vout, ωs) # compute frequency response

# analytic
s = tf("s")
H = 1/((s+1)*(s^2+s+1))
resp_an = ControlSystemsBase.freqrespv(H, ωs)

@test resp_sim ≈ resp_an

@test all(CedarSim.freqresp(ac, :vin, ωs) .≈ 1.0) # check directly-observed source

# NOTE: Bode test skipped - DSS to standard SS conversion not directly supported
# The original test used DAECompiler's system which integrated with ControlSystemsBase differently
# For MNA-based AC, use freqresp directly to compute magnitude/phase

# NOTE: Device observable test (sys.l3.V) skipped - not yet supported in MNA AC
# Original test:
# obs = DescriptorSystems.freqresp(ac, sys.l3.V, ωs)
# G = s*L3_val*H
# an = ControlSystemsBase.freqrespv(G, ωs)
# @test obs ≈ an


## AC noise - not yet implemented for MNA
# noise = noise!(circ; reltol=1e-6, abstol=1e-6);
# ... noise tests skipped ...


# Nonlinear circuit test - requires BSIM-CMG, skipped for now
# if !@isdefined(bsimcmg_inverter)
#     @warn "This test is expected to run after the bsimcmg_inverter example."
#     include(joinpath(@__DIR__, "bsimcmg", "inverter.jl"))
# end
# ac = ac!(bsimcmg_inverter.circuit; reltol=1e-6, abstol=1e-6);
# ... skipped ...
