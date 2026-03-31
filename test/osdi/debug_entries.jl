#!/usr/bin/env julia
using Cadnip.OsdiLoader
using Cadnip.OsdiLoader: JACOBIAN_ENTRY_RESIST, JACOBIAN_ENTRY_RESIST_CONST, JACOBIAN_ENTRY_REACT, JACOBIAN_ENTRY_REACT_CONST

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

f = osdi_load(DIODE_OSDI)
dev = f.devices[1]

println("Device: ", dev.name)
println("Nodes ($(dev.num_nodes)):")
for (i, n) in enumerate(dev.nodes)
    terminal = i <= dev.num_terminals ? " [TERMINAL]" : " [INTERNAL]"
    println("  [$i] $(n.name)$terminal")
end
println("Num terminals: ", dev.num_terminals)
println("Num states: ", dev.num_states)
println("Num resistive entries: ", dev.num_resistive_entries)
println("Num reactive entries: ", dev.num_reactive_entries)

println("\nJacobian entries ($(length(dev.jacobian_entries))):")
for (i, entry) in enumerate(dev.jacobian_entries)
    n1 = dev.nodes[entry.nodes.node_1 + 1].name
    n2 = dev.nodes[entry.nodes.node_2 + 1].name
    flags = entry.flags
    parts = String[]
    (flags & JACOBIAN_ENTRY_RESIST) != 0 && push!(parts, "RESIST")
    (flags & JACOBIAN_ENTRY_RESIST_CONST) != 0 && push!(parts, "RESIST_CONST")
    (flags & JACOBIAN_ENTRY_REACT) != 0 && push!(parts, "REACT")
    (flags & JACOBIAN_ENTRY_REACT_CONST) != 0 && push!(parts, "REACT_CONST")
    react_off = entry.react_ptr_off == typemax(UInt32) ? "none" : "$(entry.react_ptr_off)"
    println("  [$i] ($n1, $n2) flags=[$(join(parts, "|"))] react_ptr_off=$react_off")
end

println("\nCollapsible pairs ($(length(dev.collapsible))):")
for (i, pair) in enumerate(dev.collapsible)
    n1 = dev.nodes[pair.node_1 + 1].name
    n2 = pair.node_2 == typemax(UInt32) ? "GND" : dev.nodes[pair.node_2 + 1].name
    println("  [$i] $(n1) -> $(n2)")
end
