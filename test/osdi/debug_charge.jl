#!/usr/bin/env julia
using Cadnip
using Cadnip.MNA
using Cadnip.MNA: CHARGE_SCALE
using Cadnip.OsdiLoader
using Cadnip.OsdiLoader: osdi_eval!, eval_flags_for_mode, make_bucketed_prev_solve
using Cadnip.OsdiLoader: detect_implicit_nodes, compute_coupling_map
using Cadnip.OsdiLoader: JACOBIAN_ENTRY_RESIST, JACOBIAN_ENTRY_REACT

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

f = osdi_load(DIODE_OSDI)
dev = f.devices[1]

model = OsdiModel(dev)
setup_model!(model)
inst = OsdiInstance(model)
set_param!(inst, "is", 76.9e-12)
set_param!(inst, "rs", 42.0e-3)
set_param!(inst, "cjo", 26.5e-12)
set_param!(inst, "m", 0.333)
set_param!(inst, "n", 1.45)
setup_instance!(inst)

# Map nodes: a=1, c=2 (terminals), a_int=3, implicit_eq=4
inst.node_mapping[1] = 1  # anode
inst.node_mapping[2] = 2  # cathode
inst.node_mapping[3] = 3  # a_int
inst.node_mapping[4] = 4  # implicit_eq (virtual)

inst.implicit_nodes = detect_implicit_nodes(dev)
println("Implicit nodes: ", inst.implicit_nodes)

using Cadnip.OsdiLoader: write_node_mapping!, write_state_idx!
write_node_mapping!(inst)
write_state_idx!(inst)

# Eval at an operating point: V_anode=0.6, V_cathode=0, V_a_int=0.59
x = [0.0, 0.6, 0.0, 0.59, 0.0]  # bucketed: [bucket, V1, V2, V3, V4]
flags = eval_flags_for_mode(:tran)
osdi_eval!(inst, flags, 0.0, x; mode=:tran)

# Get reactive Jacobian (dQ/dV)
n_react = dev.num_reactive_entries
println("Num reactive entries: ", n_react)
jacobian_react = Vector{Float64}(undef, n_react)
GC.@preserve inst jacobian_react begin
    ccall(dev.fn_write_jacobian_array_react, Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
        pointer(inst.blob), pointer(inst.model.blob), pointer(jacobian_react))
end
println("Reactive Jacobian: ", jacobian_react)

# Get reactive residual (Q values)
react_residual = zeros(Float64, 5)
GC.@preserve inst react_residual begin
    ccall(dev.fn_load_residual_react, Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
        pointer(inst.blob), pointer(inst.model.blob), pointer(react_residual))
end
println("Reactive residual (all nodes): ", react_residual)
println("  bucket[0]: ", react_residual[1])
println("  node 1 (anode): ", react_residual[2])
println("  node 2 (cathode): ", react_residual[3])
println("  node 3 (a_int): ", react_residual[4])
println("  node 4 (implicit): ", react_residual[5])

# Get resist Jacobian for coupling map
n_resist = dev.num_resistive_entries
jacobian_resist = Vector{Float64}(undef, n_resist)
GC.@preserve inst jacobian_resist begin
    ccall(dev.fn_write_jacobian_array_resist, Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
        pointer(inst.blob), pointer(inst.model.blob), pointer(jacobian_resist))
end
inst.coupling_map = compute_coupling_map(dev, inst.implicit_nodes, jacobian_resist)
println("\nCoupling map: ", inst.coupling_map)

# Print what the charge formulation would produce
for impl_node in inst.implicit_nodes
    couplings = inst.coupling_map[impl_node]
    Q = react_residual[inst.node_mapping[impl_node] + 1]
    println("\nImplicit node $impl_node:")
    println("  Q(V) = ", Q)
    println("  CHARGE_SCALE * Q = ", CHARGE_SCALE * Q)
    println("  Couplings: ", couplings)

    # Show dQ/dV for each reactive entry in impl row
    react_k = 0
    for entry in dev.jacobian_entries
        entry.react_ptr_off == typemax(UInt32) && continue
        has_react = (entry.flags & JACOBIAN_ENTRY_REACT) != 0
        if has_react
            react_k += 1
        end
        row_node = Int(entry.nodes.node_1) + 1
        row_node == impl_node || continue
        col_node = Int(entry.nodes.node_2) + 1
        dQ_dV = has_react ? jacobian_react[react_k] : 0.0
        V_col = x[inst.node_mapping[col_node] + 1]
        println("  dQ/dV_$(dev.nodes[col_node].name) = $dQ_dV, V = $V_col")
    end
end
