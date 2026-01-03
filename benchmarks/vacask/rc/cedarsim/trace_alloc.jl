#!/usr/bin/env julia
#==============================================================================#
# Trace exact allocation source in fast_rebuild!
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!
using CedarSim.MNA: reset_value_only!, ValueOnlyContext, MNAContext
using CedarSim.MNA: update_sparse_from_coo!
using StaticArrays
using BenchmarkTools

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
vctx = ws.vctx

println("=" ^ 80)
println("Tracing exact allocation source")
println("=" ^ 80)

# Warmup
for _ in 1:100
    fast_rebuild!(ws, u, t)
end

# Test the positional builder call (what fast_rebuild! uses internally)
println("\n1. Testing positional builder call:")
reset_value_only!(vctx)
for _ in 1:10
    reset_value_only!(vctx)
    rc_circuit(cs.params, cs.spec, t, u, vctx)
end
reset_value_only!(vctx)
alloc_pos = @allocated rc_circuit(cs.params, cs.spec, t, u, vctx)
println("   Positional builder call: $alloc_pos bytes")

# Test keyword builder call
println("\n2. Testing keyword builder call:")
reset_value_only!(vctx)
for _ in 1:10
    reset_value_only!(vctx)
    rc_circuit(cs.params, cs.spec, t; x=u, ctx=vctx)
end
reset_value_only!(vctx)
alloc_kw = @allocated rc_circuit(cs.params, cs.spec, t; x=u, ctx=vctx)
println("   Keyword builder call: $alloc_kw bytes")

# Test fast_rebuild! directly
println("\n3. Testing fast_rebuild!:")
for _ in 1:10
    fast_rebuild!(ws, u, t)
end
alloc_rebuild = @allocated fast_rebuild!(ws, u, t)
println("   fast_rebuild!: $alloc_rebuild bytes")

# Test individual steps of fast_rebuild!
println("\n4. Component breakdown inside fast_rebuild! path:")

alloc_reset = @allocated reset_value_only!(vctx)
println("   reset_value_only!: $alloc_reset bytes")

reset_value_only!(vctx)
alloc_builder = @allocated cs.builder(cs.params, cs.spec, ws.time, u, vctx)
println("   cs.builder (positional): $alloc_builder bytes")

n_G = cs.G_n_coo
n_C = cs.C_n_coo

alloc_copy_G = @allocated begin
    @inbounds for k in 1:n_G
        ws.G_V[k] = vctx.G_V[k]
    end
end
println("   Copy G_V: $alloc_copy_G bytes")

alloc_copy_C = @allocated begin
    @inbounds for k in 1:n_C
        ws.C_V[k] = vctx.C_V[k]
    end
end
println("   Copy C_V: $alloc_copy_C bytes")

alloc_fill_b = @allocated fill!(ws.b, 0.0)
println("   fill!(b): $alloc_fill_b bytes")

n_b = length(vctx.b)
alloc_copy_b = @allocated begin
    @inbounds for i in 1:n_b
        ws.b[i] = vctx.b[i]
    end
end
println("   Copy b: $alloc_copy_b bytes")

alloc_update_G = @allocated update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, n_G)
println("   update_sparse_from_coo! G: $alloc_update_G bytes")

alloc_update_C = @allocated update_sparse_from_coo!(cs.C, ws.C_V, cs.C_coo_to_nz, n_C)
println("   update_sparse_from_coo! C: $alloc_update_C bytes")

# Benchmark with BenchmarkTools
println("\n" * "=" ^ 80)
println("Benchmark:")
display(@benchmark fast_rebuild!($ws, $u, $t))
