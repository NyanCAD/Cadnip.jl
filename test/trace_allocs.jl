#!/usr/bin/env julia
#==============================================================================#
# Trace 24-byte allocations using allocation profiler
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CedarSim
using CedarSim.MNA
using Profile
using Profile.Allocs

println("=" ^ 70)
println("Tracing 24-byte allocations")
println("=" ^ 70)

#==============================================================================#
# Simple RC Circuit Builder
#==============================================================================#

function build_simple_rc(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    MNA.stamp!(VoltageSource(params.Vcc; name=:Vs), ctx, vcc, 0)
    MNA.stamp!(Resistor(params.R), ctx, vcc, out)
    MNA.stamp!(Capacitor(params.C), ctx, out, 0)

    return ctx
end

#==============================================================================#
# Setup
#==============================================================================#

circuit = MNACircuit(build_simple_rc; Vcc=5.0, R=1000.0, C=1e-6)
ws = MNA.compile(circuit)
cs = ws.structure
dctx = ws.dctx

n = MNA.system_size(cs)
u = zeros(n)

println("\nSystem size: $n")

#==============================================================================#
# Warmup
#==============================================================================#

for _ in 1:100
    MNA.fast_rebuild!(ws, u, 0.001)
end
GC.gc()

#==============================================================================#
# Allocation profiling
#==============================================================================#

println("\n--- Starting allocation profiler ---")

# Profile with sampling to see where allocations come from
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 begin
    for _ in 1:1000
        MNA.fast_rebuild!(ws, u, 0.001)
    end
end

# Get the allocation results
results = Profile.Allocs.fetch()

println("\nAllocation profile summary:")
println("  Total allocations: $(length(results.allocs))")

# Group by type and size
type_counts = Dict{Any, Int}()
type_sizes = Dict{Any, Int}()

for alloc in results.allocs
    t = alloc.type
    type_counts[t] = get(type_counts, t, 0) + 1
    type_sizes[t] = get(type_sizes, t, 0) + alloc.size
end

println("\nAllocations by type:")
sorted_types = sort(collect(type_counts), by=x->x[2], rev=true)
for (t, count) in sorted_types[1:min(10, length(sorted_types))]
    total_size = type_sizes[t]
    avg_size = total_size ÷ count
    println("  $t: $count allocations ($total_size bytes total, avg $avg_size bytes)")
end

#==============================================================================#
# More detailed trace using @allocated line by line
#==============================================================================#

println("\n--- Line-by-line allocation test ---")

# Test reset_direct_stamp! in isolation
println("\nreset_direct_stamp! alone:")
GC.gc()
allocs1 = @allocated for _ in 1:1000
    MNA.reset_direct_stamp!(dctx)
end
println("  $allocs1 bytes for 1000 calls ($(allocs1/1000) per call)")

# Now test the builder call alone (but we need reset in between)
println("\nBuilder call (with reset):")
GC.gc()
allocs2 = @allocated for _ in 1:1000
    MNA.reset_direct_stamp!(dctx)
    cs.builder(cs.params, cs.spec, MNA.real_time(0.001); x=u, ctx=dctx)
end
allocs_builder_only = allocs2 - allocs1
println("  Builder alone: $(allocs_builder_only/1000) bytes per call")

# Test deferred b application alone
println("\nDeferred b stamps:")
MNA.reset_direct_stamp!(dctx)
cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
n_deferred = cs.n_b_deferred
GC.gc()
allocs3 = @allocated for _ in 1:1000
    for k in 1:n_deferred
        idx = dctx.b_resolved[k]
        if idx > 0
            dctx.b[idx] += dctx.b_V[k]
        end
    end
end
println("  $allocs3 bytes for 1000 calls ($(allocs3/1000) per call)")

#==============================================================================#
# Check srcFact/gshunt paths
#==============================================================================#

println("\n--- Testing srcFact/gshunt paths ---")

# Check if srcFact scaling allocates
spec_normal = cs.spec
spec_srcfact = MNA.with_srcfact(spec_normal, 0.5)

println("\nWith srcFact=1.0 (default):")
GC.gc()
allocs_nofact = @allocated for _ in 1:1000
    MNA.fast_rebuild!(ws, u, 0.001)
end
println("  $(allocs_nofact/1000) bytes per call")

# Temporarily modify spec to test srcFact path
# We can't easily modify cs.spec since CompiledStructure is immutable
# But we can check if the allocation is from that path by looking at the code

#==============================================================================#
# Check if it's the 'nothing' return
#==============================================================================#

println("\n--- Testing return value ---")

# Test if fast_rebuild! returning nothing allocates
function test_nothing_return()
    return nothing
end

GC.gc()
allocs_nothing = @allocated for _ in 1:1000
    test_nothing_return()
end
println("\nReturning nothing: $(allocs_nothing/1000) bytes per call")

#==============================================================================#
# Direct test of builder function
#==============================================================================#

println("\n--- Testing builder function directly ---")

# Check if it's the builder call that allocates
function bench_builder(builder, params, spec, t, x, dctx)
    builder(params, spec, t; x=x, ctx=dctx)
    return nothing
end

MNA.reset_direct_stamp!(dctx)
GC.gc()
allocs_builder = @allocated for _ in 1:1000
    MNA.reset_direct_stamp!(dctx)
    bench_builder(cs.builder, cs.params, cs.spec, 0.001, u, dctx)
end
println("\nBuilder via wrapper: $(allocs_builder/1000) bytes per call")

# Check if it's the stamp! functions allocating
MNA.reset_direct_stamp!(dctx)
vcc_idx = MNA.get_node!(dctx, :vcc)
out_idx = MNA.get_node!(dctx, :out)

vsrc = VoltageSource(5.0; name=:Vs)
res = Resistor(1000.0)
cap = Capacitor(1e-6)

println("\nIndividual stamp! calls:")

# VoltageSource stamp
MNA.reset_direct_stamp!(dctx)
GC.gc()
allocs_v = @allocated for _ in 1:1000
    MNA.reset_direct_stamp!(dctx)
    MNA.stamp!(vsrc, dctx, vcc_idx, 0)
end
println("  VoltageSource: $(allocs_v/1000 - allocs1/1000) bytes (excluding reset)")

# Resistor stamp
MNA.reset_direct_stamp!(dctx)
GC.gc()
allocs_r = @allocated for _ in 1:1000
    MNA.reset_direct_stamp!(dctx)
    MNA.stamp!(res, dctx, vcc_idx, out_idx)
end
println("  Resistor: $(allocs_r/1000 - allocs1/1000) bytes (excluding reset)")

# Capacitor stamp
MNA.reset_direct_stamp!(dctx)
GC.gc()
allocs_c = @allocated for _ in 1:1000
    MNA.reset_direct_stamp!(dctx)
    MNA.stamp!(cap, dctx, out_idx, 0)
end
println("  Capacitor: $(allocs_c/1000 - allocs1/1000) bytes (excluding reset)")

println("\n" * "=" ^ 70)
println("Trace complete")
println("=" ^ 70)
