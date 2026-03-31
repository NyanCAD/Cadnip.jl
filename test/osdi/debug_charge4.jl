#!/usr/bin/env julia
using Cadnip, Cadnip.MNA, Cadnip.OsdiLoader
using Sundials: IDA
using OrdinaryDiffEq: Rodas5P, ABDF2
using LinearSolve: KLUFactorization

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")
const spice = """* single OSDI diode
.model d1 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
vs a 0 sin 0.0 5 1000
R1 a anode 1k
N1 anode 0 d1
.end
"""
code = Cadnip.parse_spice_to_mna(spice; circuit_name=:ckt, osdi_files=[DIODE_OSDI])
eval(code)

for (name, solver) in [
    ("Rodas5P", Rodas5P(linsolve=KLUFactorization())),
    ("ABDF2", ABDF2(linsolve=KLUFactorization())),
    ("IDA(KLU)", IDA(linear_solver=:KLU, max_error_test_failures=20)),
]
    c = MNACircuitFromSetup((p) -> Base.invokelatest(ckt, p), (;), MNASpec())
    MNA.assemble!(c)
    println("$name:")
    sol = tran!(c, (0.0, 0.001); dtmax=1e-5, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
    println("  Steps=$(length(sol.t)) retcode=$(sol.retcode) t_end=$(sol.t[end])")
end
