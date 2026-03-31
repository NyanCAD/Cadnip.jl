#!/usr/bin/env julia
#==============================================================================#
# Debug script: Compare OSDI diode vs VA-compiled diode - matrix comparison
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Cadnip.OsdiLoader
using VADistillerModels
using LinearAlgebra
using SparseArrays
using Sundials: IDA

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

# --- Single diode: simple circuit for clean comparison ---
# VA version
const spice_va = """* VA diode
V1 a 0 DC 5
R1 a anode 1k
xd1 anode 0 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
.end
"""
const spice_osdi = """* OSDI diode
.model d1 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
V1 a 0 DC 5
R1 a anode 1k
N1 anode 0 d1
.end
"""

# VA circuit
println("=== Building VA circuit ===")
va_code = Cadnip.parse_spice_to_mna(spice_va; circuit_name=:va_ckt,
                                       imported_hdl_modules=[sp_diode_module])
eval(va_code)

# Build context at zero operating point
va_ctx = va_ckt((;), MNASpec())
va_sys = assemble!(va_ctx)
println("VA nodes: ", va_ctx.node_names, " (", va_ctx.n_nodes, " nodes, ", va_ctx.n_currents, " currents)")
println("VA G entries: ", length(va_ctx.G_V))
println("VA C entries: ", length(va_ctx.C_V))
println("VA b entries: ", length(va_ctx.b_V))

# OSDI circuit
println("\n=== Building OSDI circuit ===")
osdi_code = Cadnip.parse_spice_to_mna(spice_osdi; circuit_name=:osdi_ckt,
                                          osdi_files=[DIODE_OSDI])
eval(osdi_code)
osdi_setup = Base.invokelatest(osdi_ckt, (;))
osdi_ctx = Base.invokelatest(osdi_setup, (;), MNASpec())
osdi_sys = assemble!(osdi_ctx)
println("OSDI nodes: ", osdi_ctx.node_names, " (", osdi_ctx.n_nodes, " nodes, ", osdi_ctx.n_currents, " currents)")
println("OSDI G entries: ", length(osdi_ctx.G_V))
println("OSDI C entries: ", length(osdi_ctx.C_V))
println("OSDI b entries: ", length(osdi_ctx.b_V))

# Compare matrices
println("\n=== G matrix comparison ===")
println("VA G ($(size(va_sys.G))):")
display(Matrix(va_sys.G))
println("\nOSDI G ($(size(osdi_sys.G))):")
display(Matrix(osdi_sys.G))

println("\n=== C matrix comparison ===")
println("VA C ($(size(va_sys.C))):")
display(Matrix(va_sys.C))
println("\nOSDI C ($(size(osdi_sys.C))):")
display(Matrix(osdi_sys.C))

println("\n=== b vector comparison ===")
println("VA b: ", va_sys.b)
println("OSDI b: ", osdi_sys.b)

# DC solve
println("\n=== DC comparison ===")
va_circuit = MNACircuit(va_ckt)
va_sol = dc!(va_circuit)
println("VA DC:")
for name in va_sol.node_names
    println("  $name = $(voltage(va_sol, name))")
end

osdi_circuit = MNACircuitFromSetup((params) -> Base.invokelatest(osdi_ckt, params), (;), MNASpec())
MNA.assemble!(osdi_circuit)
osdi_sol = dc!(osdi_circuit)
println("OSDI DC:")
for name in osdi_sol.node_names
    println("  $name = $(voltage(osdi_sol, name))")
end

# Now rebuild at the converged DC operating point and compare G/C/b
println("\n=== Rebuild at DC operating point ===")
va_x = va_sol.u
osdi_x = osdi_sol.u
println("VA solution vector: ", va_x)
println("OSDI solution vector: ", osdi_x)

# Rebuild VA at operating point
va_ctx2 = va_ckt((;), MNASpec(); x=va_x)
va_sys2 = assemble!(va_ctx2)
println("\nVA G at DC op:")
display(Matrix(va_sys2.G))
println("\nVA C at DC op:")
display(Matrix(va_sys2.C))
println("\nVA b at DC op:")
display(va_sys2.b)

# Rebuild OSDI at operating point
osdi_setup2 = Base.invokelatest(osdi_ckt, (;))
osdi_ctx2 = Base.invokelatest(osdi_setup2, (;), MNASpec(); x=osdi_x)
osdi_sys2 = assemble!(osdi_ctx2)
println("\nOSDI G at DC op:")
display(Matrix(osdi_sys2.G))
println("\nOSDI C at DC op:")
display(Matrix(osdi_sys2.C))
println("\nOSDI b at DC op:")
display(osdi_sys2.b)

# Check OSDI diode's internal structure
println("\n=== OSDI device info ===")
f = osdi_load(DIODE_OSDI)
dev = f.devices[1]
println("Device: ", dev.name)
println("Nodes ($(dev.num_nodes)): ", [n.name for n in dev.nodes])
println("Terminals: ", dev.num_terminals)
println("States: ", dev.num_states)
println("Jacobian entries ($(length(dev.jacobian_entries))):")
for (i, entry) in enumerate(dev.jacobian_entries)
    n1 = dev.nodes[entry.nodes.node_1 + 1].name
    n2 = dev.nodes[entry.nodes.node_2 + 1].name
    has_resist = (entry.flags & JACOBIAN_ENTRY_RESIST) != 0
    has_resist_const = (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0
    has_react = (entry.flags & JACOBIAN_ENTRY_REACT) != 0
    has_react_const = (entry.flags & JACOBIAN_ENTRY_REACT_CONST) != 0
    react_off = entry.react_ptr_off == typemax(UInt32) ? "none" : string(entry.react_ptr_off)
    println("  [$i] ($n1, $n2) resist=$(has_resist) resist_const=$(has_resist_const) react=$(has_react) react_const=$(has_react_const) react_off=$react_off")
end
println("Collapsible pairs:")
for (i, pair) in enumerate(dev.collapsible)
    n1 = dev.nodes[pair.node_1 + 1].name
    n2 = pair.node_2 == typemax(UInt32) ? "GND" : dev.nodes[pair.node_2 + 1].name
    println("  [$i] $(n1) -> $(n2)")
end
