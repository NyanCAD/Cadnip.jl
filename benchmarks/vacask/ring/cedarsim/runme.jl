#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Ring Oscillator with PSP103 MOSFETs (OSDI)
#
# 9-stage ring oscillator using PSP103 MOSFET model via OSDI.
# Uses the ngspice models.inc for PSP103 model parameters.
# Circuit matches ngspice/VACASK reference (no load caps, 10uA pulse).
#
# Usage: julia runme.jl
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp
using CedarSim.OsdiLoader
using OrdinaryDiffEq: FBDF
using SciMLBase
using BenchmarkTools
using Printf

# Path to precompiled OSDI PSP103
const PSP103_OSDI = joinpath(@__DIR__, "..", "..", "..", "..", "test", "osdi", "psp103.osdi")

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")

# Parse SPICE file with OSDI device — returns a setup function
const circuit_code = parse_spice_file_to_mna(spice_file; circuit_name=:ring_circuit,
                                              osdi_files=[PSP103_OSDI])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    wrapped_setup = (params) -> Base.invokelatest(ring_circuit, params)
    circuit = MNACircuitFromSetup(wrapped_setup, (;), MNASpec())
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(solver; tspan=(0.0, 1e-6), dtmax=0.05e-9, maxiters=100_000_000)
    # Setup the simulation outside the timed region
    circuit = setup_simulation()
    solver_name = nameof(typeof(solver))
    init = CedarTranOp()

    println("Ring Oscillator Benchmark (PSP103 OSDI)")
    println("="^50)
    println("  Solver:  $solver_name")
    println("  Init:    CedarTranOp (homotopy)")
    println("  dtmax:   $(dtmax*1e9) ns")
    println("  tspan:   $(tspan[2]*1e6) μs")
    println()

    # Run once to verify it works and get statistics
    # force_dtmin + relaxed tolerances needed for no-cap circuit
    # (PSP103 internal caps ~1fF cause sub-ps switching transitions)
    println("Running transient analysis...")
    sol = tran!(circuit, tspan; dtmax=dtmax, solver=solver,
                initializealg=init, maxiters=maxiters, dense=false,
                force_dtmin=true, abstol=1e-4, reltol=1e-2,
                unstable_check=(dt,u,p,t)->false)

    println("\n=== Results ===")
    println("  Status:     $(sol.retcode)")
    @printf("  Timepoints: %d\n", length(sol.t))
    @printf("  Final time: %.3e s\n", sol.t[end])
    if sol.stats !== nothing && hasproperty(sol.stats, :nnonliniter) && sol.stats.nnonliniter > 0
        @printf("  NR iters:   %d\n", sol.stats.nnonliniter)
        @printf("  Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    end

    if sol.retcode == SciMLBase.ReturnCode.Success
        # Benchmark the actual simulation
        println("\nBenchmarking (3 samples)...")
        circuit = setup_simulation()
        bench = @benchmark tran!($circuit, $tspan; dtmax=$dtmax, solver=$solver,
                                 initializealg=$init, maxiters=$maxiters, dense=false,
                                 force_dtmin=true, abstol=1e-4, reltol=1e-2,
                                 unstable_check=(dt,u,p,t)->false) samples=3 evals=1 seconds=1800
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
    solver = FBDF(autodiff=false)
    run_benchmark(solver)
end
