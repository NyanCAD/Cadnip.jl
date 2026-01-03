#!/usr/bin/env julia
#==============================================================================#
# Inspect optimized code for fast_rebuild!
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!, reset_value_only!
using CedarSim.MNA: update_sparse_from_coo!
using InteractiveUtils

# Load circuit
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
println("@code_typed for fast_rebuild!")
println("=" ^ 80)
println()

# Get the typed code
typed = @code_typed fast_rebuild!(ws, u, t)
println(typed)

println()
println("=" ^ 80)
println("Looking for allocation sites in the IR...")
println("=" ^ 80)

# Convert to string and search for allocation-related instructions
ir_str = string(typed)

# Look for patterns that indicate allocations
patterns = [
    r"jl_gc",
    r"jl_alloc",
    r"Array",
    r"new_from_type",
    r"jl_box",
    r"invoke.*alloc",
    r"Core\.Intrinsics",
]

for p in patterns
    matches = collect(eachmatch(p, ir_str))
    if !isempty(matches)
        println("\nPattern '$p' found $(length(matches)) times")
    end
end

println()
println("=" ^ 80)
println("@code_typed for builder call (positional)")
println("=" ^ 80)
println()

vctx = ws.vctx
reset_value_only!(vctx)

# Check the builder signature
println("Builder type: ", typeof(cs.builder))
println("Positional call signature: builder(params, spec, t, u, vctx)")
println()

# Try to get typed code for the builder
try
    typed_builder = @code_typed cs.builder(cs.params, cs.spec, t, u, vctx)
    println(typed_builder)
catch e
    println("Error getting typed code for builder: ", e)
end

println()
println("=" ^ 80)
println("@code_typed for update_sparse_from_coo!")
println("=" ^ 80)
println()

n_G = cs.G_n_coo
typed_update = @code_typed update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, n_G)
println(typed_update)
