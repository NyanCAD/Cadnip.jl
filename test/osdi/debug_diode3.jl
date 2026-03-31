#!/usr/bin/env julia
#==============================================================================#
# Debug: Compare OSDI vs VA transient residuals at matching operating point
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Cadnip.OsdiLoader
using VADistillerModels
using LinearAlgebra
using SparseArrays

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

# Full graetz circuit
const spice_va = """* VA diode graetz
vs inp inn 0 sin 0.0 20 50.0
xd1 inp outp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd2 outn inp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd3 inn outp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd4 outn inn sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
cl outp outn 100u
rl outp outn 1k
rgnd1 inn 0 1meg
rgnd2 outn 0 1meg
.end
"""

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

println("=== Building circuits ===")
va_code = Cadnip.parse_spice_to_mna(spice_va; circuit_name=:va_ckt,
                                       imported_hdl_modules=[sp_diode_module])
eval(va_code)

osdi_code = Cadnip.parse_spice_to_mna(spice_osdi; circuit_name=:osdi_ckt,
                                          osdi_files=[DIODE_OSDI])
eval(osdi_code)

# Try ODE solver (mass matrix formulation) instead of DAE
println("\n=== ODE (Rodas5P) solver ===")
using OrdinaryDiffEq: Rodas5P
using LinearSolve: KLUFactorization

solver = Rodas5P(linsolve=KLUFactorization())

va_circuit = MNACircuit(va_ckt)
MNA.assemble!(va_circuit)
println("VA tran (Rodas5P)...")
va_tran = tran!(va_circuit, (0.0, 1.0); dtmax=1e-6, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=10_000_000, dense=false)
println("  $(length(va_tran.t)) steps, retcode=$(va_tran.retcode), t_end=$(va_tran.t[end])")

osdi_circuit = MNACircuitFromSetup((params) -> Base.invokelatest(osdi_ckt, params), (;), MNASpec())
MNA.assemble!(osdi_circuit)
println("OSDI tran (Rodas5P)...")
osdi_tran = tran!(osdi_circuit, (0.0, 1.0); dtmax=1e-6, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=10_000_000, dense=false)
println("  $(length(osdi_tran.t)) steps, retcode=$(osdi_tran.retcode), t_end=$(osdi_tran.t[end])")
