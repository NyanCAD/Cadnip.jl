#!/usr/bin/env julia
#==============================================================================#
# Debug script: Compare OSDI diode vs VA-compiled diode
# Stamps both at the same operating point and compares G, C, b
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Cadnip.OsdiLoader
using VADistillerModels
using LinearAlgebra

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

# --- Non-OSDI (VA compiled) diode ---
# Parse SPICE to get the non-OSDI version
const spice_va = """* VA diode test
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

const spice_osdi = """* OSDI diode test
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

println("=== VA Diode (non-OSDI) ===")
va_code = Cadnip.parse_spice_to_mna(spice_va; circuit_name=:va_circuit,
                                       imported_hdl_modules=[sp_diode_module])
eval(va_code)
va_ckt = MNACircuit(va_circuit)
va_sol = dc!(va_ckt)
println("DC solution:")
for name in va_sol.node_names
    println("  $name = $(voltage(va_sol, name))")
end

println("\n=== OSDI Diode ===")
osdi_code = Cadnip.parse_spice_to_mna(spice_osdi; circuit_name=:osdi_circuit,
                                          osdi_files=[DIODE_OSDI])
eval(osdi_code)
# OSDI returns a setup function
osdi_setup = Base.invokelatest(osdi_circuit, (;))
osdi_ckt = MNACircuitFromSetup((params) -> Base.invokelatest(osdi_circuit, params), (;), MNASpec())
MNA.assemble!(osdi_ckt)
osdi_sol = dc!(osdi_ckt)
println("DC solution:")
for name in osdi_sol.node_names
    println("  $name = $(voltage(osdi_sol, name))")
end

println("\n=== Transient comparison ===")
println("VA tran...")
va_ckt2 = MNACircuit(va_circuit)
MNA.assemble!(va_ckt2)
va_tran = tran!(va_ckt2, (0.0, 0.01); dtmax=1e-5, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
println("  VA: $(length(va_tran.t)) timepoints, retcode=$(va_tran.retcode), t_end=$(va_tran.t[end])")

println("OSDI tran...")
osdi_ckt2 = MNACircuitFromSetup((params) -> Base.invokelatest(osdi_circuit, params), (;), MNASpec())
MNA.assemble!(osdi_ckt2)
osdi_tran = tran!(osdi_ckt2, (0.0, 0.01); dtmax=1e-5, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
println("  OSDI: $(length(osdi_tran.t)) timepoints, retcode=$(osdi_tran.retcode), t_end=$(osdi_tran.t[end])")
