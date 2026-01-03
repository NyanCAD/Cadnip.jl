#!/usr/bin/env julia
#==============================================================================#
# Component-level allocation test
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!, reset_value_only!
using CedarSim.MNA: update_sparse_from_coo!

# Load and parse the SPICE netlist
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

# Heavy warmup
for _ in 1:5000
    fast_rebuild!(ws, u, t)
end

println("=" ^ 60)
println("Single-call @allocated (after heavy warmup)")
println("=" ^ 60)

GC.gc(true)

# Single calls to fast_rebuild!
println("\nfast_rebuild! single calls:")
for i in 1:5
    a = @allocated fast_rebuild!(ws, u, t)
    println("  call $i: $a bytes")
end

# Isolate update_sparse_from_coo!
n_G = cs.G_n_coo
n_C = cs.C_n_coo

# Warmup update_sparse_from_coo! specifically
for _ in 1:5000
    update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, n_G)
end
GC.gc(true)

println("\nupdate_sparse_from_coo! single calls:")
for i in 1:5
    a = @allocated update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, n_G)
    println("  call $i: $a bytes")
end

# Isolate builder
for _ in 1:5000
    reset_value_only!(vctx)
    cs.builder(cs.params, cs.spec, t, u, vctx)
end
GC.gc(true)

println("\nbuilder single calls:")
for i in 1:5
    reset_value_only!(vctx)
    a = @allocated cs.builder(cs.params, cs.spec, t, u, vctx)
    println("  call $i: $a bytes")
end

# Test isolating only the copy loops
println("\nCopy loops only (no builder, no update_sparse):")

function test_copy_only(ws, vctx, cs)
    n_G = cs.G_n_coo
    n_C = cs.C_n_coo
    n_b = length(vctx.b)

    k = 1
    @inbounds while k <= n_G
        ws.G_V[k] = vctx.G_V[k]
        k += 1
    end
    k = 1
    @inbounds while k <= n_C
        ws.C_V[k] = vctx.C_V[k]
        k += 1
    end
    fill!(ws.b, 0.0)
    i = 1
    @inbounds while i <= n_b
        ws.b[i] = vctx.b[i]
        i += 1
    end
    return nothing
end

# Warmup
for _ in 1:5000
    test_copy_only(ws, vctx, cs)
end
GC.gc(true)

for i in 1:5
    a = @allocated test_copy_only(ws, vctx, cs)
    println("  call $i: $a bytes")
end
