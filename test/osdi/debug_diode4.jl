#!/usr/bin/env julia
#==============================================================================#
# Debug: Direct G/C/b comparison at non-trivial operating point via restamping
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Cadnip.MNA: compile_structure, create_workspace, build_with_detection, EvalWorkspace, fast_rebuild!
using Cadnip.OsdiLoader
using VADistillerModels
using LinearAlgebra
using SparseArrays

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

# Simple single-diode circuit for clean comparison
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

println("=== Building VA circuit ===")
va_code = Cadnip.parse_spice_to_mna(spice_va; circuit_name=:va_ckt,
                                       imported_hdl_modules=[sp_diode_module])
eval(va_code)
va_circuit = MNACircuit(va_ckt)
va_ctx = build_with_detection(va_circuit)
va_cs = compile_structure(va_circuit.builder, va_circuit.params, va_circuit.spec; ctx=va_ctx)
va_ws = create_workspace(va_cs; ctx=va_ctx)

println("=== Building OSDI circuit ===")
osdi_code = Cadnip.parse_spice_to_mna(spice_osdi; circuit_name=:osdi_ckt,
                                          osdi_files=[DIODE_OSDI])
eval(osdi_code)
osdi_circuit = MNACircuitFromSetup((params) -> Base.invokelatest(osdi_ckt, params), (;), MNASpec())
osdi_ctx = build_with_detection(osdi_circuit)
osdi_cs = compile_structure(osdi_circuit.builder, osdi_circuit.params, osdi_circuit.spec; ctx=osdi_ctx)
osdi_ws = create_workspace(osdi_cs; ctx=osdi_ctx)

println("\nVA nodes: ", va_ctx.node_names, " (size ", MNA.system_size(va_ctx), ")")
println("OSDI nodes: ", osdi_ctx.node_names, " (size ", MNA.system_size(osdi_ctx), ")")

# First, do DC to get a realistic operating point
println("\n=== DC operating point ===")
va_dc = dc!(MNACircuit(va_ckt))
osdi_dc = dc!(MNACircuitFromSetup((p) -> Base.invokelatest(osdi_ckt, p), (;), MNASpec()))
println("VA DC: ", [voltage(va_dc, n) for n in va_dc.node_names])
println("OSDI DC: ", [voltage(osdi_dc, n) for n in osdi_dc.node_names])

# Use a non-trivial operating point (like forward biased diode at t=0.0001)
# VA system is 4-dim (a, anode, a_int, I_vs)
# OSDI system is 5-dim (a, anode, a_int, implicit_eq_0, I_vs)
# At t=0.0001, Vs = sin(2π*1000*0.0001)*5 ≈ 3.09V

t_test = 0.0002
println("\n=== Rebuild at t=$t_test ===")

# Use realistic x vectors (from DC, but slightly perturbed)
va_x = copy(va_dc.x)
va_x[1] = 2.0  # V_a
va_x[2] = 0.6  # V_anode
va_x[3] = 0.59  # V_a_int

osdi_x = copy(osdi_dc.x)
osdi_x[1] = 2.0  # V_a
osdi_x[2] = 0.6  # V_anode
osdi_x[3] = 0.59  # V_a_int (n1_a_int)
osdi_x[4] = 0.0   # V_implicit_eq_0

println("VA x = ", va_x)
println("OSDI x = ", osdi_x)

# Rebuild with the test operating point
fast_rebuild!(va_ws, va_x, t_test)
fast_rebuild!(osdi_ws, osdi_x, t_test)

println("\nVA G at operating point:")
display(Matrix(va_cs.G))
println("\nOSDI G at operating point:")
display(Matrix(osdi_cs.G))

println("\nVA C:")
display(Matrix(va_cs.C))
println("\nOSDI C:")
display(Matrix(osdi_cs.C))

println("\nVA b:")
display(va_ws.dctx.b)
println("\nOSDI b:")
display(osdi_ws.dctx.b)

# Compute residual at this point (assuming du=0 for simplicity)
va_du = zeros(length(va_x))
va_resid = va_cs.C * va_du + va_cs.G * va_x - va_ws.dctx.b
println("\nVA residual (du=0): ", va_resid)

osdi_du = zeros(length(osdi_x))
osdi_resid = osdi_cs.C * osdi_du + osdi_cs.G * osdi_x - osdi_ws.dctx.b
println("OSDI residual (du=0): ", osdi_resid)

# Compare the common nodes
println("\n=== Node-by-node comparison ===")
println("Row 1 (a):     VA resid=$(va_resid[1]),  OSDI resid=$(osdi_resid[1])")
println("Row 2 (anode): VA resid=$(va_resid[2]),  OSDI resid=$(osdi_resid[2])")
println("Row 3 (a_int): VA resid=$(va_resid[3]),  OSDI resid=$(osdi_resid[3])")
if length(osdi_resid) >= 4
    println("Row 4 (impl):  OSDI resid=$(osdi_resid[4])")
end
println("Row I (I_vs):  VA resid=$(va_resid[end]), OSDI resid=$(osdi_resid[end])")

# Also check what happens when we rebuild multiple times (counter alignment)
println("\n=== Rebuild consistency (3 rebuilds) ===")
for iter in 1:3
    fast_rebuild!(osdi_ws, osdi_x, t_test + iter * 1e-6)
    b_copy = copy(osdi_ws.dctx.b)
    G_vals = Vector(osdi_cs.G.nzval)
    println("Iter $iter: b=$b_copy")
    println("         G_nzval=$(G_vals)")
end
