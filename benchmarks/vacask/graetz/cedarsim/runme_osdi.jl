#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Graetz Bridge (Full-wave Rectifier) — OSDI variant
#
# Same circuit as runme.jl but uses OSDI diode instead of VA-compiled diode.
# Allows comparing VA vs OSDI performance for the same circuit topology.
#
# Usage: julia runme_osdi.jl [solver]
#   solver: IDA (default), FBDF, or Rodas5P
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.OsdiLoader
using Sundials: IDA
using OrdinaryDiffEq: FBDF, Rodas5P
using BenchmarkTools
using Printf

# Path to precompiled OSDI diode
const DIODE_OSDI = joinpath(@__DIR__, "..", "..", "..", "..", "test", "osdi", "diode.osdi")

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme_osdi.sp")

# Parse SPICE file with OSDI device — returns a setup function
const circuit_code = parse_spice_file_to_mna(spice_file; circuit_name=:graetz_osdi_circuit,
                                              osdi_files=[DIODE_OSDI])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    wrapped_setup = (params) -> Base.invokelatest(graetz_osdi_circuit, params)
    circuit = MNACircuitFromSetup(wrapped_setup, (;), MNASpec())
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(solver; dt=1e-6, maxiters=10_000_000)
    tspan = (0.0, 1.0)  # 1 second simulation (~1M timepoints with dt=1us)
    solver_name = nameof(typeof(solver))

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with $solver_name (dtmax=$dt)...")
    bench = @benchmark tran!($circuit, $tspan; dtmax=$dt, solver=$solver, abstol=1e-3, reltol=1e-3, maxiters=$maxiters, dense=false) samples=3 evals=1 seconds=300

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; dtmax=dt, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=maxiters, dense=false)

    println("\n=== Results ($solver_name, OSDI) ===")
    @printf("Timepoints:  %d\n", length(sol.t))
    @printf("Expected:    ~%d\n", round(Int, (tspan[2] - tspan[1]) / dt) + 1)
    @printf("NR iters:    %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:   %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    solver_name = length(ARGS) >= 1 ? ARGS[1] : "IDA"
    solver = if solver_name == "IDA"
        IDA(max_nonlinear_iters=100, max_error_test_failures=20)
    elseif solver_name == "FBDF"
        FBDF()
    elseif solver_name == "Rodas5P"
        Rodas5P()
    else
        error("Unknown solver: $solver_name. Use IDA, FBDF, or Rodas5P")
    end
    run_benchmark(solver)
end
