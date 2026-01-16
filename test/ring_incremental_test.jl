#!/usr/bin/env julia
#==============================================================================#
# Incremental Ring Oscillator Test - PSP103 + CedarTranOp Homotopy
#
# Step-by-step test:
# 1. Parse and compile circuit
# 2. DC operating point with different init algorithms
# 3. Short transient (1ns)
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarDCOp, CedarTranOp, CedarUICOp
using OrdinaryDiffEq: Rodas5P
using Printf
using Logging

# Enable debug logging
global_logger(ConsoleLogger(stderr, Logging.Debug))

println("="^60)
println("Step 1: Loading PSP103 model...")
println("="^60)
using PSPModels
println("  PSP103 loaded.")

println("\n" * "="^60)
println("Step 2: Parsing SPICE netlist...")
println("="^60)
const spice_file = joinpath(@__DIR__, "..", "benchmarks", "vacask", "ring", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)
println("  Parsed.")

println("\n" * "="^60)
println("Step 3: Assembling circuit...")
println("="^60)
circuit = MNACircuit(ring_circuit)
data = MNA.assemble!(circuit)
println("  Assembled. $(size(data.G, 1)) unknowns.")

println("\n" * "="^60)
println("Step 4: DC operating point with CedarDCOp...")
println("="^60)
try
    dc_result = dc!(circuit)
    println("  DC retcode: $(dc_result.retcode)")
    println("  Max voltage: $(maximum(abs.(dc_result.u)))")
catch e
    println("  DC failed: $e")
end

println("\n" * "="^60)
println("Step 5: DC operating point with CedarTranOp...")
println("="^60)
circuit2 = MNACircuit(ring_circuit)
MNA.assemble!(circuit2)
try
    # Use tran! with very short span to test TranOp initialization
    sol = tran!(circuit2, (0.0, 1e-12); solver=Rodas5P(),
                initializealg=CedarTranOp(), maxiters=10000, dense=false)
    println("  TranOp init: $(sol.retcode)")
    println("  Initial state max: $(maximum(abs.(sol.u[1])))")
catch e
    println("  TranOp failed: $e")
    showerror(stdout, e, catch_backtrace())
end

println("\n" * "="^60)
println("Step 6: Short transient (1ns) with CedarTranOp...")
println("="^60)
circuit3 = MNACircuit(ring_circuit)
MNA.assemble!(circuit3)
try
    sol = tran!(circuit3, (0.0, 1e-9); dtmax=0.1e-9, solver=Rodas5P(),
                initializealg=CedarTranOp(), maxiters=50000, dense=false)
    println("  Status: $(sol.retcode)")
    println("  Timepoints: $(length(sol.t))")
    println("  Final time: $(sol.t[end])")
    if sol.stats !== nothing
        println("  NR iters: $(sol.stats.nnonliniter)")
    end
catch e
    println("  Transient failed: $e")
    showerror(stdout, e, catch_backtrace())
end

println("\n" * "="^60)
println("Step 7: Short transient with CedarUICOp (for oscillators)...")
println("="^60)
circuit4 = MNACircuit(ring_circuit)
MNA.assemble!(circuit4)
try
    sol = tran!(circuit4, (0.0, 1e-9); dtmax=0.1e-9, solver=Rodas5P(),
                initializealg=CedarUICOp(warmup_steps=20), maxiters=50000, dense=false)
    println("  Status: $(sol.retcode)")
    println("  Timepoints: $(length(sol.t))")
    println("  Final time: $(sol.t[end])")
    if sol.stats !== nothing
        println("  NR iters: $(sol.stats.nnonliniter)")
    end
catch e
    println("  UICOp transient failed: $e")
    showerror(stdout, e, catch_backtrace())
end

println("\n" * "="^60)
println("Done!")
println("="^60)
