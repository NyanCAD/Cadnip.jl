#!/usr/bin/env julia
#==============================================================================#
# Test if Dict lookup is source of 24 bytes allocation
#==============================================================================#

# Test Dict lookup allocation behavior

d = Dict{Symbol,Int}(:a => 1, :b => 2, Symbol("1") => 3, Symbol("2") => 4)

# Warmup
for _ in 1:10000
    d[:a]
    d[Symbol("1")]
end

GC.gc(true)

println("Dict lookup allocations:")

# Test symbolic key lookup
println("\nSymbol key (interned, e.g., :a):")
for i in 1:5
    a = @allocated d[:a]
    println("  call $i: $a bytes")
end

# Test dynamically created Symbol lookup
println("\nSymbol(\"1\") key (may not be interned):")
for i in 1:5
    a = @allocated d[Symbol("1")]
    println("  call $i: $a bytes")
end

# Test with cached symbol
const s1 = Symbol("1")
println("\nCached Symbol(\"1\"):")
for _ in 1:10000
    d[s1]
end
GC.gc(true)
for i in 1:5
    a = @allocated d[s1]
    println("  call $i: $a bytes")
end

# Now test with actual circuit code
println("\n" * "=" ^ 60)
println("Testing with actual circuit code")
println("=" ^ 60)

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!, reset_value_only!

const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

spec = MNASpec()
cs = compile_structure(rc_circuit, NamedTuple(), spec)
ws = create_workspace(cs)
vctx = ws.vctx
u = zeros(Float64, cs.n)
t = Float64(0.5)

# Warmup
for _ in 1:5000
    fast_rebuild!(ws, u, t)
end

println("\nValueOnlyContext node_to_idx Dict:")
println("  Keys: ", keys(vctx.node_to_idx))

# Test the Dict lookups that happen in the builder
println("\nDict lookups from vctx.node_to_idx:")
for _ in 1:1000
    vctx.node_to_idx[Symbol("1")]
    vctx.node_to_idx[Symbol("2")]
end
GC.gc(true)
for i in 1:5
    a = @allocated begin
        vctx.node_to_idx[Symbol("1")]
        vctx.node_to_idx[Symbol("2")]
    end
    println("  call $i: $a bytes (two lookups)")
end

# Check with pre-resolved indices (what the code should be doing)
println("\nCompare to fast_rebuild! (which uses Dict lookups internally):")
GC.gc(true)
for i in 1:5
    a = @allocated fast_rebuild!(ws, u, t)
    println("  call $i: $a bytes")
end
