#!/usr/bin/env julia
#==============================================================================#
# Show generated circuit code
#==============================================================================#

using CedarSim
using CedarSim.MNA

const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Show the generated code
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)

println("=" ^ 80)
println("Generated circuit code:")
println("=" ^ 80)
println(circuit_code)
println("=" ^ 80)
