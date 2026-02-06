#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Ring Oscillator with PSP103 MOSFETs
#
# 9-stage ring oscillator using PSP103 MOSFET model.
# Uses VACASKModels' precompiled nmos/pmos builders for fast startup.
#
# Usage: julia runme.jl
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp
using OrdinaryDiffEq: FBDF
using BenchmarkTools
using Printf

# Import precompiled PSP103 builders (nmos_mna_builder, pmos_mna_builder)
using VACASKModels

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
# VACASKModels provides precompiled nmos_mna_builder/pmos_mna_builder
# which are resolved via imported_hdl_modules for the exposed subcircuit references
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[VACASKModels])
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

function run_benchmark(solver; tspan=(0.0, 100e-9), dtmax=0.01e-9, maxiters=10_000_000)
    # Setup the simulation outside the timed region
    circuit = setup_simulation()
    solver_name = nameof(typeof(solver))
    init = CedarTranOp()

    println("Ring Oscillator Benchmark (PSP103)")
    println("="^50)
    println("  Solver:  $solver_name")
    println("  Init:    CedarTranOp (homotopy)")
    println("  dtmax:   $(dtmax*1e9) ns")
    println("  tspan:   $(tspan[2]*1e9) ns")
    println()

    # Run once to verify it works and get statistics
    println("Running transient analysis...")
    sol = tran!(circuit, tspan; dtmax=dtmax, solver=solver,
                initializealg=init, maxiters=maxiters, dense=false)

    println("\n=== Results ===")
    println("  Status:     $(sol.retcode)")
    @printf("  Timepoints: %d\n", length(sol.t))
    @printf("  Final time: %.3e s\n", sol.t[end])
    if sol.stats !== nothing && hasproperty(sol.stats, :nnonliniter) && sol.stats.nnonliniter > 0
        @printf("  NR iters:   %d\n", sol.stats.nnonliniter)
        @printf("  Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    end

    if sol.retcode == :Success || sol.retcode == SciMLBase.ReturnCode.Success
        # Benchmark the actual simulation
        println("\nBenchmarking (3 samples)...")
        circuit = setup_simulation()
        bench = @benchmark tran!($circuit, $tspan; dtmax=$dtmax, solver=$solver,
                                 initializealg=$init, maxiters=$maxiters, dense=false) samples=3 evals=1 seconds=600
        display(bench)
        println()
        return bench, sol
    else
        println("\nSimulation did not converge - skipping benchmark")
        return nothing, sol
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using SciMLBase
    solver = FBDF(autodiff=false)
    run_benchmark(solver)
end
