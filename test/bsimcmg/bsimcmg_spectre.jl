module bsimcmg_spectre

using Cadnip
using Cadnip.NyanVerilogAParser
using OrdinaryDiffEq
using UnicodePlots
using Sundials
using NyanSpectreNetlistParser
using Cadnip.SpectreEnvironment
using Test
using SciMLBase

# Use pre-parsed BSIM-CMG model from CMCModels package
# The exported type is `bsimcmg` (matches VA module name)
using CMCModels: bsimcmg

sa = NyanSpectreNetlistParser.parsefile(joinpath(dirname(pathof(NyanSpectreNetlistParser)), "../test/examples/7nm_TT.scs"));
eval(Cadnip.make_spectre_netlist(sa))

sa2 = NyanSpectreNetlistParser.parsefile(joinpath(@__DIR__, "asap7_inv.scs"));
code = Cadnip.make_spectre_circuit(sa2)
circuit = eval(code)
# circuit()

sys = CircuitIRODESystem(circuit)

prob = DAEProblem(sys, nothing, nothing, (0.0, 1e-7); initializealg=CedarDCOp())
integ = init(prob, IDA(); abstol=1e-8)
sol = solve(prob, IDA())

# We expect the out node to be positive after dc init. It is possible for this
# node to go negative, but that requires capacitive effects, which are off in
# DC. A naive initialization of the DAE would give a non-DC solution.
@test integ[sys.node_Vout] > 0.
@test sol.retcode == SciMLBase.ReturnCode.Success

end # module bsimcmg_spectre
