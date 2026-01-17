#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Ring Oscillator with PSP103 MOSFETs
#
# 9-stage ring oscillator using PSP103 MOSFET model.
#
# Winning configuration (discovered via CI testing):
# - Solver: FBDF (BDF method, standard for stiff circuits)
# - Init: CedarTranOp (homotopy: GMIN stepping → source stepping)
# - dtmax: 0.01ns (10ps max step to capture fast switching dynamics)
#
# Usage: julia runme.jl
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp
using OrdinaryDiffEq: FBDF
using BenchmarkTools
using Printf

# Import pre-parsed PSP103 model from PSPModels package
using PSPModels

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
# Pass PSP103VA_module so the SPICE parser knows about our VA device
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    circuit = MNACircuit(ring_circuit)
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(; tspan=(0.0, 1e-6), dtmax=0.01e-9, maxiters=10_000_000)
    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Winning configuration:
    # - FBDF: BDF method, standard for stiff circuit simulation
    # - CedarTranOp: Homotopy initialization (GMIN stepping → source stepping)
    # - dtmax=0.01ns: Small enough to capture fast ring oscillator dynamics
    solver = FBDF(autodiff=false)
    init = CedarTranOp()

    println("Ring Oscillator Benchmark (PSP103)")
    println("="^50)
    println("  Solver:  FBDF")
    println("  Init:    CedarTranOp (homotopy)")
    println("  dtmax:   $(dtmax*1e9) ns")
    println("  tspan:   $(tspan[2]*1e6) us")
    println()

    # Run once to verify it works and get statistics
    println("Running transient analysis...")
    sol = tran!(circuit, tspan; dtmax=dtmax, solver=solver,
                initializealg=init, maxiters=maxiters, dense=false)

    println("\n=== Results ===")
    println("  Status:     $(sol.retcode)")
    @printf("  Timepoints: %d\n", length(sol.t))
    @printf("  Final time: %.3e s\n", sol.t[end])
    if sol.stats !== nothing && sol.stats.nnonliniter > 0
        @printf("  NR iters:   %d\n", sol.stats.nnonliniter)
        @printf("  Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    end

    if sol.retcode == :Success
        # Benchmark the actual simulation
        println("\nBenchmarking (3 samples)...")
        circuit = setup_simulation()
        bench = @benchmark tran!($circuit, $tspan; dtmax=$dtmax, solver=$solver,
                                 initializealg=$init, maxiters=$maxiters, dense=false) samples=3 evals=1 seconds=300
        display(bench)
        println()
        return bench, sol
    else
        println("\nSimulation failed - skipping benchmark")
        return nothing, sol
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
