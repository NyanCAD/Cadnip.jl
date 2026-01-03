#!/usr/bin/env julia
#==============================================================================#
# Find exact source of 24 bytes allocation using Profile.Allocs
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!
using StaticArrays
using Profile

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

println("Warming up...")
for _ in 1:1000
    fast_rebuild!(ws, u, t)
end

# Try to use Profile.Allocs (Julia 1.8+)
if isdefined(Profile, :Allocs)
    using Profile.Allocs
    println("Using Profile.Allocs...")

    # Clear and profile
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 begin
        for _ in 1:100
            fast_rebuild!(ws, u, t)
        end
    end

    results = Profile.Allocs.fetch()
    println("\nAllocation results (sorted by size):")
    allocations = sort(collect(results.allocs), by=x->x.size, rev=true)
    for (i, a) in enumerate(allocations[1:min(20, length(allocations))])
        println("  $i. size=$(a.size) type=$(a.type)")
        # Print stack trace if available
        if !isempty(a.stacktrace)
            for (j, frame) in enumerate(a.stacktrace[1:min(3, length(a.stacktrace))])
                println("     $j. $(frame.func) at $(frame.file):$(frame.line)")
            end
        end
    end
else
    println("Profile.Allocs not available in this Julia version")
    println("Using @allocated multiple times instead...")

    # Run multiple times and check consistency
    allocs = [(@allocated fast_rebuild!(ws, u, t)) for _ in 1:10]
    println("Allocations from 10 runs: $allocs")
    println("All same: $(all(==(allocs[1]), allocs))")
end
