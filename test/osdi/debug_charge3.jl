#!/usr/bin/env julia
using Cadnip
using Cadnip.MNA
using Cadnip.MNA: build_with_detection, compile_structure, create_workspace, fast_rebuild!
using Cadnip.OsdiLoader
using LinearAlgebra, SparseArrays

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")
const spice = """* single OSDI diode DC
.model d1 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
V1 a 0 DC 5
R1 a anode 1k
N1 anode 0 d1
.end
"""
code = Cadnip.parse_spice_to_mna(spice; circuit_name=:ckt, osdi_files=[DIODE_OSDI])
eval(code)

circuit = MNACircuitFromSetup((p) -> Base.invokelatest(ckt, p), (;), MNASpec())

# Build and inspect matrices at zero
ctx = build_with_detection(circuit)
sys = assemble!(ctx)
println("System size: ", MNA.system_size(sys))
println("Node names: ", sys.node_names)
println("Charge names: ", sys.charge_names)
println("n_charges: ", sys.n_charges)

println("\nG matrix:")
display(Matrix(sys.G))
println("\nC matrix:")
display(Matrix(sys.C))
println("\nb vector:")
display(sys.b)

# Now compile and test fast_rebuild
cs = compile_structure(circuit.builder, circuit.params, circuit.spec; ctx=ctx)
ws = create_workspace(cs; ctx=ctx)

# Rebuild at zero
u0 = zeros(MNA.system_size(sys))
fast_rebuild!(ws, u0, 0.0)
println("\nAfter rebuild at u=0:")
println("G nzval: ", cs.G.nzval)
println("C nzval: ", cs.C.nzval)
println("b: ", ws.dctx.b)

# Check residual at u=0, du=0
resid = cs.C * zeros(length(u0)) + cs.G * u0 - ws.dctx.b
println("\nResidual at u=0, du=0: ", resid)

# DC solve
sol = dc!(circuit)
println("\nDC solution:")
for name in sol.node_names
    println("  $name = $(voltage(sol, name))")
end
println("Full x: ", sol.x)
