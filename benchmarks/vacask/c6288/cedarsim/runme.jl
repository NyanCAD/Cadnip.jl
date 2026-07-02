#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: C6288 16x16 Multiplier
#
# A 16x16 bit multiplier circuit using PSP103 MOSFETs (212228 variables).
#
# STATUS: Uses CedarDCOp initialization (DC solve with GMIN/source stepping)
#
# Notes:
# ngspice's 'uic' skips DC operating point analysis and starts transient
# integration directly from u=0, relying on its own per-step Newton retry
# with timestep halving to work through the harsh startup (vdd hard-on at
# t=0). SciML's DAE initialization framework requires a consistent t=0
# state up front rather than tolerating repeated per-step failures, so the
# naive translation of 'uic' -- CedarUICOp's fixed-dt pseudo-transient
# relaxation -- cannot take even a single non-degenerate step here (Newton
# fails identically at every dt, including with unlimited shrinking room).
# CedarDCOp's existing GMIN-stepping/source-stepping homotopy chain (the
# same machinery vacask/ngspice use to find a DC operating point for
# awkward circuits) does find a usable t=0 state, and the transient
# proceeds normally from there. See doc/c6288_bottleneck_findings.md.
#
# Usage: julia runme.jl [solver]
#   solver: IDA, FBDF, or Rodas5P (default)
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Cadnip.MNA: CedarDCOp
using Sundials: IDA
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqRosenbrock: Rodas5P
using ADTypes: AutoFiniteDiff
using LinearSolve: KLUFactorization
using BenchmarkTools
using Printf

# Import pre-parsed PSP103 model from PSPModels package
using PSPModels
println("PSP103VA loaded from PSPModels package")

# Parse SPICE file, inject PSP103VA module as Tier-2 scope so `.model` cards
# referring to PSP103VA resolve. Codegen runs at top level (no world-age tax).
const spice_file = joinpath(@__DIR__, "runme.sp")
let ast = Cadnip.NyanSpectreNetlistParser.parsefile(spice_file; start_lang=:spice, implicit_title=true),
    sema_result = Cadnip.sema(ast; imported_hdl_modules=[PSP103VA_module])
    eval(Cadnip._make_mna_circuit_with_sema(sema_result; circuit_name=:c6288_circuit))
end

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
"""
function setup_simulation()
    circuit = MNACircuit(c6288_circuit)
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(solver; reltol=1e-3, maxiters=10_000_000)
    tspan = (0.0, 2e-9)  # 2ns simulation (same as ngspice)
    solver_name = nameof(typeof(solver))

    # Setup the simulation outside the timed region
    circuit = setup_simulation()
    n = MNA.system_size(circuit)
    println("Circuit size: $n variables")

    # Use CedarDCOp for initialization: the GMIN/source-stepping fallback
    # chain in _dc_solve_with_fallbacks finds a usable operating point for
    # this circuit where CedarUICOp's pseudo-transient warmup cannot take a
    # single step. See doc/c6288_bottleneck_findings.md.
    init = CedarDCOp()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with $solver_name (reltol=$reltol)...")
    bench = @benchmark tran!($circuit, $tspan; solver=$solver, reltol=$reltol, maxiters=$maxiters, initializealg=$init, dense=false) samples=3 evals=1 seconds=300

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; solver=solver, reltol=reltol, maxiters=maxiters, initializealg=init, dense=false)

    println("\n=== Results ($solver_name) ===")
    @printf("Timepoints: %d\n", length(sol.t))
    @printf("NR iters:   %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    solver_name = length(ARGS) >= 1 ? ARGS[1] : "Rodas5P"
    solver = if solver_name == "IDA"
        # linear_solver=:KLU is required at this scale: Sundials' own default
        # dense solver allocates an n x n matrix (n=212228 -> ~360GB), which
        # doesn't raise a catchable OutOfMemoryError -- it segfaults inside
        # SUNMatZero_Dense.
        IDA(linear_solver=:KLU, max_nonlinear_iters=100, max_error_test_failures=20)
    elseif solver_name == "FBDF"
        # autodiff=AutoFiniteDiff() is required: FBDF's own default resolves
        # to AutoSparse{AutoForwardDiff, KnownJacobianSparsityDetector,
        # GreedyColoringAlgorithm}, which ignores the analytic jac= we supply
        # and rebuilds/colors its own Jacobian -- whose combination with the
        # mass matrix in jacobian2W! OOMs at this scale. Matches
        # SOLVER_FBDF_RING in benchmarks/vacask/run_benchmarks.jl.
        FBDF(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
    elseif solver_name == "Rodas5P"
        Rodas5P()
    else
        error("Unknown solver: $solver_name. Use IDA, FBDF, or Rodas5P")
    end
    run_benchmark(solver)
end
