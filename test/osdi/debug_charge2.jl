#!/usr/bin/env julia
using Cadnip
using Cadnip.MNA
using Cadnip.MNA: build_with_detection, compile_structure, create_workspace
using Cadnip.OsdiLoader
using Sundials: IDA

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

const spice = """* single OSDI diode
.model d1 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
V1 a 0 DC 5
R1 a anode 1k
N1 anode 0 d1
.end
"""
code = Cadnip.parse_spice_to_mna(spice; circuit_name=:ckt, osdi_files=[DIODE_OSDI])
eval(code)

# Build and inspect
circuit = MNACircuitFromSetup((p) -> Base.invokelatest(ckt, p), (;), MNASpec())
MNA.assemble!(circuit)

# DC first
sol = dc!(circuit)
println("DC solution:")
for name in sol.node_names
    println("  $name = $(voltage(sol, name))")
end
println("System size: ", length(sol.x))
println("Node names: ", sol.node_names)

# Now try tran
println("\nTran with IDA:")
circuit2 = MNACircuitFromSetup((p) -> Base.invokelatest(ckt, p), (;), MNASpec())
MNA.assemble!(circuit2)
solver = IDA(max_nonlinear_iters=100, max_error_test_failures=20)
sol2 = tran!(circuit2, (0.0, 0.001); dtmax=1e-5, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=10000, dense=false)
println("Steps: $(length(sol2.t)), retcode=$(sol2.retcode), t_end=$(sol2.t[end])")
