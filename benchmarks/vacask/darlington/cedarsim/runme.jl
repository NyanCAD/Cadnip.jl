#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Darlington Pair Switch
#
# Two cascaded NPN BJTs (sp_bjt, Gummel-Poon) switched between cutoff and
# saturation by a 500kHz pulse train. Each sp_bjt carries three limited
# junctions (vbe, vbc, vsub), making this the multi-branch junction-limiting
# stress case for in-step PCNR - the transient companion of the DC-only
# darlington circuit in benchmarks/pcnr/dc_newton_iterations.jl.
#
# Benchmark target: ~500k timepoints (dtmax=2ns over 1ms, 500 switch cycles)
#
# Usage: julia runme.jl [solver]
#   solver: IDA (default), FBDF, PCNR (= FBDF with in-step PCNR limiting)
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Sundials: IDA
using OrdinaryDiffEqBDF: FBDF
using BenchmarkTools
using Printf

# Import pre-parsed BJT model from VADistillerModels package
using VADistillerModels

# File-first load: VADistillerModels registers sp_bjt for `.model ... npn`
# via ModelRegistry (Tier 1).
const spice_file = joinpath(@__DIR__, "runme.sp")
Base.include(@__MODULE__, SpiceFile(spice_file; name=:darlington_circuit))

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    circuit = MNACircuit(darlington_circuit)
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(solver; dt=2e-9, maxiters=10_000_000)
    tspan = (0.0, 1e-3)  # 1ms simulation (~500k timepoints with dt=2ns)
    solver_name = nameof(typeof(solver))

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with $solver_name (dtmax=$dt)...")
    bench = @benchmark tran!($circuit, $tspan; dtmax=$dt, solver=$solver, abstol=1e-3, reltol=1e-3, maxiters=$maxiters, dense=false) samples=3 evals=1 seconds=300

    # Also run once to get solution statistics
    circuit = setup_simulation()
    Cadnip.reset_pcnr_activations!()
    sol = tran!(circuit, tspan; dtmax=dt, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=maxiters, dense=false)

    println("\n=== Results ($solver_name) ===")
    @printf("Timepoints:  %d\n", length(sol.t))
    @printf("Expected:    ~%d\n", round(Int, (tspan[2] - tspan[1]) / dt) + 1)
    @printf("NR iters:    %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:   %.2f\n", sol.stats.nnonliniter / length(sol.t))
    @printf("PCNR limiter activations: %d\n", Cadnip.pcnr_activations())
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    solver_name = length(ARGS) >= 1 ? ARGS[1] : "IDA"
    solver = if solver_name == "IDA"
        IDA(linear_solver=:KLU, max_error_test_failures=20)
    elseif solver_name == "FBDF"
        FBDF()
    elseif solver_name == "PCNR"
        pcnr_fbdf()
    else
        error("Unknown solver: $solver_name. Use IDA, FBDF, or PCNR")
    end
    run_benchmark(solver)
end
