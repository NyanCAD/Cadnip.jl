using GLMakie
using Cadnip
using Cadnip.SpectreEnvironment
using OrdinaryDiffEq
using StructArrays

spice_code = """
*Third order low pass filter, butterworth, with ω_c = 1

.param L1_val=3/2
+ C2_val=4/3
+ L3_val=1/2
+ R4_val=1
+ f_val=1

V1 vin 0 SIN (0, 1, 'f_val')
L1 vin n1 'L1_val'
C2 n1 0 'C2_val'
L3 n1 vout 'L3_val'
R4 vout 0 'R4_val'
"""

circuit_code = Cadnip.make_spectre_circuit(
    Cadnip.NyanSpectreNetlistParser.SPICENetlistParser.SPICENetlistCSTParser.parse(spice_code);
);
circuit = eval(circuit_code)

sim = ParamSim(circuit; l1_val=3/2, c2_val=4/3, l3_val=1/2, r4_val=1.0, f_val=1.0)
sys = CircuitIRODESystem(sim)
prob = ODEProblem(sys, nothing, (0.0, 100.0), sim)
sol = solve(prob, FBDF(autodiff=false); initializealg=CedarDCOp())

Cadnip.explore(sol)