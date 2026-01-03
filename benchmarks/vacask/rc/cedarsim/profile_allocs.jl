#!/usr/bin/env julia
#==============================================================================#
# Memory Allocation Profiling for VACASK RC Circuit
#
# This script profiles memory allocations to identify remaining sources
# of allocations in the circuit evaluation path.
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!, update_sparse_from_coo!
using CedarSim.MNA: reset_value_only!, CompiledStructure, EvalWorkspace
using BenchmarkTools

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

println("Generated circuit code:")
println("=" ^ 80)
println(circuit_code)
println("=" ^ 80)
println()

# Create circuit and compile structure
circuit = MNACircuit(rc_circuit)
spec = MNASpec()

# Compile the structure and create workspace
cs = compile_structure(rc_circuit, NamedTuple(), spec)
ws = create_workspace(cs)
u = zeros(Float64, cs.n)
t = 0.5  # some mid-simulation time

println("Circuit info:")
println("  n_nodes: ", cs.n_nodes)
println("  n_currents: ", cs.n_currents)
println("  G_n_coo: ", cs.G_n_coo)
println("  C_n_coo: ", cs.C_n_coo)
println("  supports_value_only_mode: ", ws.supports_value_only_mode)
println("  supports_ctx_reuse: ", ws.supports_ctx_reuse)
println()

# Warmup
for _ in 1:100
    fast_rebuild!(ws, u, t)
end

# Benchmark fast_rebuild!
println("Benchmarking fast_rebuild!:")
bench = @benchmark fast_rebuild!($ws, $u, $t)
display(bench)
println()

# Use @allocated for profiling
println("\n" * "=" ^ 80)
println("Allocation Profile (using @allocated):")
println("=" ^ 80)

# Reset and measure
for _ in 1:10
    fast_rebuild!(ws, u, t)
end

alloc = @allocated fast_rebuild!(ws, u, t)
println("Bytes allocated per fast_rebuild!: ", alloc)

# Profile individual components
println("\nComponent breakdown:")

# Test the builder call directly
if ws.supports_value_only_mode
    vctx = ws.vctx
    reset_value_only!(vctx)
    alloc_builder = @allocated cs.builder(cs.params, cs.spec, Float64(t); x=u, ctx=vctx)
    println("  Builder call (ValueOnlyContext): ", alloc_builder, " bytes")

    # Test reset_value_only!
    alloc_reset = @allocated reset_value_only!(vctx)
    println("  reset_value_only!: ", alloc_reset, " bytes")
else
    ctx = ws.ctx
    MNA.reset_for_restamping!(ctx)
    alloc_builder = @allocated cs.builder(cs.params, cs.spec, Float64(t); x=u, ctx=ctx)
    println("  Builder call (MNAContext): ", alloc_builder, " bytes")
end

# Test sparse matrix update
alloc_update_G = @allocated update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, cs.G_n_coo)
alloc_update_C = @allocated update_sparse_from_coo!(cs.C, ws.C_V, cs.C_coo_to_nz, cs.C_n_coo)
println("  update_sparse_from_coo! (G): ", alloc_update_G, " bytes")
println("  update_sparse_from_coo! (C): ", alloc_update_C, " bytes")

println("\n" * "=" ^ 80)
println("Type inspection:")
println("=" ^ 80)
println("  builder type: ", typeof(cs.builder))
println("  params type: ", typeof(cs.params))
println("  spec type: ", typeof(cs.spec))
println()

println("\nDone profiling.")
