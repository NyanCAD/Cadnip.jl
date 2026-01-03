#!/usr/bin/env julia
#==============================================================================#
# Check for union types with @code_warntype
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!
using InteractiveUtils

const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

spec = MNASpec()
cs = compile_structure(rc_circuit, NamedTuple(), spec)
ws = create_workspace(cs)
u = zeros(Float64, cs.n)
t = Float64(0.5)

# Warmup
for _ in 1:100
    fast_rebuild!(ws, u, t)
end

println("=" ^ 80)
println("@code_warntype for fast_rebuild!")
println("=" ^ 80)
println()

@code_warntype fast_rebuild!(ws, u, t)
