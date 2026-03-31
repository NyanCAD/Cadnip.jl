#!/usr/bin/env julia
# Test OSDI graetz with different solvers/paths
using Cadnip
using Cadnip.MNA
using Cadnip.OsdiLoader
using Sundials: IDA
using OrdinaryDiffEq: Rodas5P, ABDF2, ImplicitEuler
using LinearSolve: KLUFactorization

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

const spice_osdi = """* OSDI diode graetz
.model d1n4007 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
vs inp inn 0 sin 0.0 20 50.0
N1 inp outp d1n4007
N2 outn inp d1n4007
N3 inn outp d1n4007
N4 outn inn d1n4007
cl outp outn 100u
rl outp outn 1k
rgnd1 inn 0 1meg
rgnd2 outn 0 1meg
.end
"""
osdi_code = Cadnip.parse_spice_to_mna(spice_osdi; circuit_name=:osdi_ckt,
                                          osdi_files=[DIODE_OSDI])
eval(osdi_code)

tspan = (0.0, 0.001)

for (name, solver) in [
    ("ABDF2(KLU)", ABDF2(linsolve=KLUFactorization())),
    ("Rodas5P(KLU)", Rodas5P(linsolve=KLUFactorization())),
    ("ImplicitEuler(KLU)", ImplicitEuler(linsolve=KLUFactorization())),
    ("IDA(KLU)", IDA(linear_solver=:KLU, max_error_test_failures=20, max_nonlinear_iters=10)),
]
    gc = MNACircuitFromSetup((p) -> Base.invokelatest(osdi_ckt, p), (;), MNASpec())
    MNA.assemble!(gc)
    println("$name:")
    sol = tran!(gc, tspan; dtmax=1e-5, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
    println("  Steps: $(length(sol.t)), retcode=$(sol.retcode), t_end=$(sol.t[end])")
    if length(sol.t) > 1
        println("  NR iters: $(sol.stats.nnonliniter), rejected: $(sol.stats.nreject)")
    end
end
