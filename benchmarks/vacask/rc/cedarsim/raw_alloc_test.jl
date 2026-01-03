#!/usr/bin/env julia
#==============================================================================#
# Raw allocation test (no BenchmarkTools)
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!
using StaticArrays

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

# Create circuit and compile structure
spec = MNASpec()
cs = compile_structure(rc_circuit, NamedTuple(), spec)
ws = create_workspace(cs)
u = zeros(Float64, cs.n)
t = Float64(0.5)

println("Warming up (1000 iterations)...")
for _ in 1:1000
    fast_rebuild!(ws, u, t)
end

# Force GC
GC.gc(true)

# Multiple raw @allocated measurements
println("\nRaw @allocated measurements (10 runs):")
for i in 1:10
    a = @allocated fast_rebuild!(ws, u, t)
    println("  Run $i: $a bytes")
end

# Force GC and run more
GC.gc(true)
println("\nAfter GC (10 more runs):")
for i in 1:10
    a = @allocated fast_rebuild!(ws, u, t)
    println("  Run $i: $a bytes")
end

# Try with function barrier to isolate
function run_fast_rebuild_isolated(ws, u, t)
    fast_rebuild!(ws, u, t)
    return nothing
end

# Warmup the barrier function
for _ in 1:100
    run_fast_rebuild_isolated(ws, u, t)
end

println("\nWith function barrier (10 runs):")
for i in 1:10
    a = @allocated run_fast_rebuild_isolated(ws, u, t)
    println("  Run $i: $a bytes")
end
