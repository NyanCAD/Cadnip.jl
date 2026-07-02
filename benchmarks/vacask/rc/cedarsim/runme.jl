#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: RC Circuit
#
# RC circuit excited by a pulse train.
# This is a simple linear circuit - no rejected timepoints.
#
# Benchmark target: ~1 million timepoints, ~2 million iterations
#
# Usage: julia runme.jl [solver]
#   solver: IDA (default), FBDF, or Rodas5P
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Sundials: IDA
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqRosenbrock: Rodas5P
using BenchmarkTools
using Printf

# Load and parse the SPICE netlist from file.
# File-first load: defines `rc_circuit(params, spec, ...)` at top level.
const spice_file = joinpath(@__DIR__, "runme.sp")
Base.include(@__MODULE__, SpiceFile(spice_file; name=:rc_circuit))

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    circuit = MNACircuit(rc_circuit)
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(circuit)
    return circuit
end

# vs 1 0 dc 0 pulse 0 1 1u 1u 1u 1m 2m -> delay=1u rise=1u width=1m fall=1u period=2m
# `dtmax` (1e-6) equals the pulse rise/fall time, so an adaptive controller with
# no knowledge of the discontinuity can get stuck trying to land a step exactly
# on the edge (observed: FBDF/DFBDF/Trapezoid/TRBDF2 all eventually fail this way
# on the full 1s/500-period run, each at a different period). Passing `tstops`
# for every edge - delay start, rise end, width end, fall end, each period -
# removes the ambiguity and gives every solver its best shot, matching what
# `wpd/run_wpd.jl`'s `case_tstops` already does for the same circuit and what a
# real SPICE engine does internally (breakpoints from source edges).
#
# TODO: hardcoding breakpoints here is a stopgap. The real fix is for Cadnip's
# transient driver to derive `tstops` automatically from PULSE/SIN source
# parameters (and VA `$bound_step`-style hints) so callers never have to do
# this by hand - see the `wpd` duplication of this same logic as evidence it's
# needed in more than one place already.
function pulse_tstops(tspan; delay=1e-6, rise=1e-6, width=1e-3, fall=1e-6, period=2e-3)
    bps = Float64[]
    k = 0
    while true
        base = delay + k * period
        base > tspan[2] && break
        for off in (0.0, rise, rise + width, rise + width + fall)
            e = base + off
            tspan[1] <= e <= tspan[2] && push!(bps, e)
        end
        k += 1
    end
    return bps
end

function run_benchmark(solver; dt=1e-6, maxiters=10_000_000)
    tspan = (0.0, 1.0)  # 1 second simulation
    solver_name = nameof(typeof(solver))
    tstops = pulse_tstops(tspan)

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with $solver_name (dtmax=$dt)...")
    bench = @benchmark tran!($circuit, $tspan; dtmax=$dt, solver=$solver, maxiters=$maxiters, dense=false, tstops=$tstops) samples=3 evals=1 seconds=300

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; dtmax=dt, solver=solver, maxiters=maxiters, dense=false, tstops=tstops)

    println("\n=== Results ($solver_name) ===")
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
        IDA(max_error_test_failures=20)
    elseif solver_name == "FBDF"
        FBDF()
    elseif solver_name == "Rodas5P"
        Rodas5P()
    else
        error("Unknown solver: $solver_name. Use IDA, FBDF, or Rodas5P")
    end
    run_benchmark(solver)
end
