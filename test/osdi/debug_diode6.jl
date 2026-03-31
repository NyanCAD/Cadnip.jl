#!/usr/bin/env julia
# Test OSDI diode specifically with IDA
using Cadnip
using Cadnip.MNA
using Cadnip.OsdiLoader
using VADistillerModels
using Sundials: IDA

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

const spice_va = """* VA diode
vs a 0 sin 0.0 5 1000
R1 a anode 1k
xd1 anode 0 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
.end
"""
const spice_osdi = """* OSDI diode
.model d1 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
vs a 0 sin 0.0 5 1000
R1 a anode 1k
N1 anode 0 d1
.end
"""

va_code = Cadnip.parse_spice_to_mna(spice_va; circuit_name=:va_ckt,
                                       imported_hdl_modules=[sp_diode_module])
eval(va_code)
osdi_code = Cadnip.parse_spice_to_mna(spice_osdi; circuit_name=:osdi_ckt,
                                          osdi_files=[DIODE_OSDI])
eval(osdi_code)

solver = IDA(max_nonlinear_iters=100, max_error_test_failures=20)

println("=== VA IDA ===")
va_circuit = MNACircuit(va_ckt)
MNA.assemble!(va_circuit)
va_sol = tran!(va_circuit, (0.0, 0.01); dtmax=1e-5, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
println("Steps: $(length(va_sol.t)), retcode=$(va_sol.retcode), t_end=$(va_sol.t[end])")

println("\n=== OSDI IDA ===")
osdi_circuit = MNACircuitFromSetup((p) -> Base.invokelatest(osdi_ckt, p), (;), MNASpec())
MNA.assemble!(osdi_circuit)
osdi_sol = tran!(osdi_circuit, (0.0, 0.01); dtmax=1e-5, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
println("Steps: $(length(osdi_sol.t)), retcode=$(osdi_sol.retcode), t_end=$(osdi_sol.t[end])")

# Also try full graetz with IDA
println("\n=== Full graetz OSDI with IDA ===")
const spice_graetz_osdi = """* OSDI diode graetz
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
osdi_graetz_code = Cadnip.parse_spice_to_mna(spice_graetz_osdi; circuit_name=:osdi_graetz,
                                                 osdi_files=[DIODE_OSDI])
eval(osdi_graetz_code)

# Try with progressively shorter simulations
for t_end in [0.001, 0.005, 0.01, 0.05]
    println("\nGraetz IDA t_end=$t_end, dtmax=1e-5:")
    gc = MNACircuitFromSetup((p) -> Base.invokelatest(osdi_graetz, p), (;), MNASpec())
    MNA.assemble!(gc)
    sol = tran!(gc, (0.0, t_end); dtmax=1e-5, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=1000000, dense=false)
    println("  Steps: $(length(sol.t)), retcode=$(sol.retcode), t_end=$(sol.t[end])")
    if sol.retcode != :Success && length(sol.t) > 1
        println("  Failed at t=$(sol.t[end]), stats: nNR=$(sol.stats.nnonliniter), nreject=$(sol.stats.nreject)")
    end
end

# Try graetz OSDI with IDA and dtmax=1e-6
println("\nGraetz IDA t_end=0.001, dtmax=1e-6:")
gc2 = MNACircuitFromSetup((p) -> Base.invokelatest(osdi_graetz, p), (;), MNASpec())
MNA.assemble!(gc2)
sol2 = tran!(gc2, (0.0, 0.001); dtmax=1e-6, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
println("  Steps: $(length(sol2.t)), retcode=$(sol2.retcode), t_end=$(sol2.t[end])")
