using CedarSim
using CedarSim.MNA
using Test
using CedarSim.SpectreEnvironment
using SpectreNetlistParser
using DescriptorSystems
using ControlSystemsBase
using RobustAndOptimalControl  # provides ss(::DescriptorStateSpace) for bode plots
using VADistillerModels  # for sp_mos1 nonlinear AC test

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

# Convert to ControlSystems state space, compute bode
# RobustAndOptimalControl provides ss(::DescriptorStateSpace) conversion
mag_sim, phase_sim, w_sim = bode(ss(ac[:vout]), ωs)
mag_an, phase_an, w_an = bode(H, ωs)

@test mag_sim ≈ mag_an
@test phase_sim ≈ phase_an
@test w_sim ≈ w_an

#==============================================================================#
# Skipped Tests - Functionality Not Yet Implemented in MNA AC
#==============================================================================#

# LIMITATION 1: Device observable access
# --------------------------------------
# Cannot observe internal device variables like inductor voltage (sys.l3.V).
# MNA tracks flat node/current names without hierarchical device scope.
# Original test verified: voltage across L3 = s*L3*H(s) where H is transfer function.
#
# Original test code:
# obs = DescriptorSystems.freqresp(ac, sys.l3.V, ωs)
# G = s*L3_val*H
# an = ControlSystemsBase.freqrespv(G, ωs)
# @test obs ≈ an

# LIMITATION 2: Noise analysis
# ----------------------------
# noise!() is not implemented. Would require:
# - Thermal noise (4kTR) for resistors
# - Shot noise (2qI) for semiconductor junctions
# - Flicker noise (K/f) for MOSFETs
# - Correlation matrix assembly and output noise spectral density calculation
#
# Original test code:
# noise = noise!(circ; reltol=1e-6, abstol=1e-6)
# ... noise transfer function and output noise density tests ...

#==============================================================================#
# Nonlinear Circuit AC Test - CMOS Inverter with sp_mos1 vs ngspice
#==============================================================================#

# CMOS inverter - compare CedarSim AC analysis with ngspice reference
# Using MOS1 model with gate capacitances for frequency-dependent behavior

# ngspice reference data (generated with ngspice-43):
# Circuit: CMOS inverter with nmos/pmos level=1, vto=±0.7, kp=100u/50u, lambda=0.01
#          cgso=cgdo=1f, w=1u/2u, l=1u, Cload=10f, Vdd=3.3V, Vin=1.65V DC + 1V AC
# Format: [frequency, real(vout), imag(vout)]
const ngspice_inverter = [
    1.000000e+03  -2.14000e+02   1.489857e-02
    1.584893e+03  -2.14000e+02   2.361264e-02
    2.511886e+03  -2.14000e+02   3.742352e-02
    3.981072e+03  -2.14000e+02   5.931228e-02
    6.309573e+03  -2.13999e+02   9.400361e-02
    1.000000e+04  -2.13999e+02   1.489856e-01
    1.584893e+04  -2.13999e+02   2.361262e-01
    2.511886e+04  -2.13999e+02   3.742341e-01
    3.981072e+04  -2.13998e+02   5.931183e-01
    6.309573e+04  -2.13995e+02   9.400182e-01
    1.000000e+05  -2.13989e+02   1.489785e+00
    1.584893e+05  -2.13973e+02   2.360977e+00
    2.511886e+05  -2.13934e+02   3.741208e+00
    3.981072e+05  -2.13835e+02   5.926675e+00
    6.309573e+05  -2.13587e+02   9.382259e+00
    1.000000e+06  -2.12967e+02   1.482671e+01
    1.584893e+06  -2.11425e+02   2.332862e+01
    2.511886e+06  -2.07649e+02   3.631300e+01
    3.981072e+06  -1.98733e+02   5.508106e+01
    6.309573e+06  -1.79386e+02   7.879875e+01
    1.000000e+07  -1.44138e+02   1.003481e+02
    1.584893e+07  -9.65055e+01   1.064839e+02
    2.511886e+07  -5.27328e+01   9.221740e+01
    3.981072e+07  -2.46492e+01   6.831786e+01
    6.309573e+07  -1.05440e+01   4.631670e+01
    1.000000e+08  -4.32594e+00   3.011701e+01
]

# CedarSim AC analysis with equivalent circuit
const inverter_cedar = CedarSim.parse_spice_to_mna("""
* CMOS Inverter for AC analysis (CedarSim)
Vdd vdd 0 DC 3.3
Vin vin 0 DC 1.65 AC 1
XMP vout vin vdd vdd sp_mos1 type=-1 vto=-0.7 kp=50e-6 lambda=0.01 cgso=1e-15 cgdo=1e-15 w=2e-6 l=1e-6
XMN vout vin 0 0 sp_mos1 type=1 vto=0.7 kp=100e-6 lambda=0.01 cgso=1e-15 cgdo=1e-15 w=1e-6 l=1e-6
Cload vout 0 10f
.END
"""; circuit_name=:cmos_inverter, imported_hdl_modules=[sp_mos1_module])
eval(inverter_cedar)

inverter_circ = MNACircuit(cmos_inverter)
inverter_ac = ac!(inverter_circ)

# Compute CedarSim response at ngspice frequencies
ωs_inv = 2π .* ngspice_inverter[:, 1]
resp_cedar = CedarSim.freqresp(inverter_ac, :vout, ωs_inv)

# ngspice reference as complex numbers
resp_ngspice = Complex.(ngspice_inverter[:, 2], ngspice_inverter[:, 3])

# Compare magnitude
@test isapprox(abs.(resp_cedar), abs.(resp_ngspice); rtol=0.05)

# Compare phase (handle wrap-around at ±π)
phase_diff = abs.(angle.(resp_cedar) .- angle.(resp_ngspice))
phase_diff = min.(phase_diff, 2π .- phase_diff)
@test all(phase_diff .< 0.1)  # ~5.7 degrees tolerance
