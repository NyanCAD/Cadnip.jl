using Cadnip
using Cadnip.SpectreEnvironment
using Test

ckt = """
subckt myres pos neg
    parameters r=1k
    r1 (pos neg) resistor r=r
ends myres

x1 (vcc 0) myres r=2k
v1 (vcc 0) vsource dc=1
"""

circuit_code = Cadnip.make_spectre_circuit(
    Cadnip.NyanSpectreNetlistParser.parse(ckt),
);
fn = eval(circuit_code);

# test aliasmap
map = Cadnip.aliasmap(fn)
ref = Dict{Cadnip.DScope, Cadnip.DScope}(
    Cadnip.DScope(Cadnip.DScope(Cadnip.DScope(), :x1), :node_pos) => Cadnip.DScope(Cadnip.DScope(), :node_vcc),
    Cadnip.DScope(Cadnip.DScope(Cadnip.DScope(), :x1), :node_neg) => Cadnip.DScope(Cadnip.DScope(), :node_0),
)
@test map == ref

# test alias in circuit
include("common.jl")
sys, sol = solve_circuit(fn)

@test all(isapprox.(sol[sys.x1.node_pos], sol[sys.node_vcc]))
@test all(isapprox.(sol[sys.x1.node_neg], sol[sys.node_0]))
