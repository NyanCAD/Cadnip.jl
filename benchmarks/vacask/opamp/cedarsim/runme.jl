#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: 5-Transistor OTA (CMOS Differential Pair)
#
# Accuracy-constrained benchmark: measures work-precision tradeoff rather
# than raw throughput at fixed dtmax. The OTA step response has rich dynamics
# (slew rate limiting + exponential settling) that exercise the adaptive
# error estimator.
#
# Methodology:
# 1. Compute reference solution at tight tolerances (ESDIRK54I8L2SA at 1e-9)
# 2. Run each solver at multiple tolerance levels
# 3. Measure actual error (max abs diff from reference at sample points)
# 4. Report: error vs function evaluations, error vs wall time
#
# Key findings from solver survey:
# - ESDIRK54I8L2SA and ESDIRK436L2SA2 are the most robust (work 1e-3 to 1e-9)
# - KenCarp4/5/47 work well to ~1e-7, go unstable at tighter tolerances
# - IDA (Sundials) works to 1e-8
# - Rodas4P/5P go unstable at 1e-7
# - FBDF/QNDF/ABDF2/Rodas3 via ODEProblem get stuck at 3 steps (mass matrix issue)
#
# Usage:
#   julia --project=benchmarks benchmarks/vacask/opamp/cedarsim/runme.jl
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp, MNACircuit, MNASolutionAccessor
using CedarSim.MNA: voltage, assemble!
using OrdinaryDiffEq
using Sundials: IDA
using SciMLBase
using LinearSolve: KLUFactorization
using BenchmarkTools
using Printf
using Statistics

# Import precompiled PSP103 builders
using VACASKModels

# Load and parse the SPICE netlist
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:opamp_circuit,
                                         imported_hdl_modules=[VACASKModels])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
"""
function setup_simulation()
    circuit = MNACircuit(opamp_circuit)
    MNA.assemble!(circuit)
    return circuit
end

# Simulation parameters
const TSPAN = (0.0, 2e-6)  # 2us - captures step + settling
const N_SAMPLE = 500        # Points for error measurement

# Sample times for accuracy comparison - dense around the step edge
const SAMPLE_TIMES = let
    # Uniform spacing over full range, plus extra points around step
    t_uniform = range(TSPAN[1], TSPAN[2]; length=N_SAMPLE÷2)
    # Dense around step edge (100ns +/- 50ns)
    t_step = range(50e-9, 300e-9; length=N_SAMPLE÷2)
    sort(unique(vcat(collect(t_uniform), collect(t_step))))
end

"""
    compute_reference(; abstol=1e-9, reltol=1e-9)

Compute a high-accuracy reference solution using ESDIRK54I8L2SA (the most
robust solver for this circuit) at tight tolerances.
"""
function compute_reference(; abstol=1e-9, reltol=1e-9)
    circuit = setup_simulation()
    solver = ESDIRK54I8L2SA(linsolve=KLUFactorization(), autodiff=false)
    println("Computing reference solution (ESDIRK54I8L2SA, abstol=$abstol, reltol=$reltol)...")
    sol = tran!(circuit, TSPAN; solver=solver,
                abstol=abstol, reltol=reltol, dense=true, maxiters=10_000_000)
    @assert sol.retcode == ReturnCode.Success "Reference solution failed: $(sol.retcode)"
    @printf("  Reference: %d timepoints, final t=%.3e\n", length(sol.t), sol.t[end])
    return sol
end

"""
    extract_output_waveform(sol, circuit, times)

Extract the output voltage at the given sample times.
"""
function extract_output_waveform(sol, circuit, times)
    sys = assemble!(circuit)
    acc = MNASolutionAccessor(sol, sys)
    return [voltage(acc, :out, t) for t in times]
end

"""
    measure_error(sol, circuit, ref_waveform, times)

Compute the maximum absolute error of the solution vs reference at sample times.
Also returns L2 (RMS) error.
"""
function measure_error(sol, circuit, ref_waveform, times)
    waveform = extract_output_waveform(sol, circuit, times)
    abs_errors = abs.(waveform .- ref_waveform)
    max_err = maximum(abs_errors)
    rms_err = sqrt(mean(abs_errors.^2))
    return max_err, rms_err
end

"""
    run_solver_accuracy(solver_fn, solver_name, ref_waveform; tolerances=...)

Run a solver at multiple tolerance levels, measuring accuracy and cost.
Returns a vector of named tuples with metrics per tolerance.
"""
function run_solver_accuracy(solver_fn, solver_name, ref_waveform;
                             tolerances=10.0 .^ (-3:-1:-8))
    results = []
    init = CedarTranOp()

    for tol in tolerances
        circuit = setup_simulation()
        solver = solver_fn()

        print("  $solver_name @ tol=$(@sprintf("%.0e", tol))... ")
        try
            t_start = time_ns()
            sol = tran!(circuit, TSPAN; solver=solver,
                        abstol=tol, reltol=tol,
                        initializealg=init, dense=true,
                        maxiters=10_000_000)
            t_elapsed = (time_ns() - t_start) / 1e9

            if sol.retcode != ReturnCode.Success
                @printf("FAILED (%s)\n", sol.retcode)
                push!(results, (tol=tol, max_err=NaN, rms_err=NaN,
                                nf=0, nsteps=0, time_s=t_elapsed,
                                rejected=0, retcode=string(sol.retcode)))
                continue
            end

            max_err, rms_err = measure_error(sol, circuit, ref_waveform, SAMPLE_TIMES)
            nf = hasproperty(sol.stats, :nf) ? sol.stats.nf : 0
            nsteps = length(sol.t)
            rejected = hasproperty(sol.stats, :nreject) ? sol.stats.nreject : 0

            @printf("err=%.2e  steps=%d  rejected=%d  nf=%d  time=%.3fs\n",
                    max_err, nsteps, rejected, nf, t_elapsed)

            push!(results, (tol=tol, max_err=max_err, rms_err=rms_err,
                            nf=nf, nsteps=nsteps, time_s=t_elapsed,
                            rejected=rejected, retcode="Success"))
        catch e
            @printf("ERROR: %s\n", sprint(showerror, e; context=:limit=>true))
            push!(results, (tol=tol, max_err=NaN, rms_err=NaN,
                            nf=0, nsteps=0, time_s=NaN,
                            rejected=0, retcode="Error"))
        end
    end
    return results
end

#==============================================================================#
# Solver Definitions
#==============================================================================#

# DAE solver (Sundials)
solver_ida() = IDA(linear_solver=:KLU, max_error_test_failures=20, max_nonlinear_iters=10)

# Rosenbrock family (ODE with mass matrix)
solver_rodas4p() = Rodas4P(linsolve=KLUFactorization())
solver_rodas5p() = Rodas5P(linsolve=KLUFactorization())

# ESDIRK / SDIRK family (FSAL methods, ODE with mass matrix)
solver_kencarp3() = KenCarp3(linsolve=KLUFactorization(), autodiff=false)
solver_kencarp4() = KenCarp4(linsolve=KLUFactorization(), autodiff=false)
solver_kencarp5() = KenCarp5(linsolve=KLUFactorization(), autodiff=false)
solver_kencarp47() = KenCarp47(linsolve=KLUFactorization(), autodiff=false)
solver_kvaerno5() = Kvaerno5(linsolve=KLUFactorization(), autodiff=false)
solver_esdirk54i8l2sa() = ESDIRK54I8L2SA(linsolve=KLUFactorization(), autodiff=false)
solver_esdirk436l2sa2() = ESDIRK436L2SA2(linsolve=KLUFactorization(), autodiff=false)

# Solvers to benchmark (name, constructor, category)
# Note: FBDF/QNDF/ABDF2/Rodas3 are excluded - they get stuck at 3 steps
# due to a mass matrix handling issue with this circuit's ODE formulation.
const ALL_SOLVERS = [
    # DAE solver (baseline)
    ("IDA",              solver_ida,              "BDF (DAE)"),
    # Rosenbrock methods (ODE with mass matrix)
    ("Rodas4P",          solver_rodas4p,          "Rosenbrock"),
    ("Rodas5P",          solver_rodas5p,          "Rosenbrock"),
    # ESDIRK methods (FSAL, ODE with mass matrix)
    ("KenCarp3",         solver_kencarp3,         "ESDIRK"),
    ("KenCarp4",         solver_kencarp4,         "ESDIRK"),
    ("KenCarp5",         solver_kencarp5,         "ESDIRK"),
    ("KenCarp47",        solver_kencarp47,        "ESDIRK"),
    ("Kvaerno5",         solver_kvaerno5,         "ESDIRK"),
    ("ESDIRK54I8L2SA",   solver_esdirk54i8l2sa,   "ESDIRK"),
    ("ESDIRK436L2SA2",   solver_esdirk436l2sa2,   "ESDIRK"),
]

#==============================================================================#
# Legacy interface for run_benchmarks.jl compatibility
#==============================================================================#

function run_benchmark(solver; tspan=TSPAN, maxiters=10_000_000)
    circuit = setup_simulation()
    solver_name = nameof(typeof(solver))
    init = CedarTranOp()

    println("OTA Benchmark (PSP103)")
    println("="^50)
    println("  Solver:  $solver_name")
    println("  tspan:   $(tspan[2]*1e6) us")
    println()

    println("Running transient analysis...")
    sol = tran!(circuit, tspan; solver=solver, initializealg=init,
                maxiters=maxiters, dense=false,
                abstol=1e-6, reltol=1e-4)

    println("\n=== Results ===")
    println("  Status:     $(sol.retcode)")
    @printf("  Timepoints: %d\n", length(sol.t))
    @printf("  Final time: %.3e s\n", sol.t[end])
    if sol.stats !== nothing && hasproperty(sol.stats, :nnonliniter) && sol.stats.nnonliniter > 0
        @printf("  NR iters:   %d\n", sol.stats.nnonliniter)
        @printf("  Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    end

    if sol.retcode == ReturnCode.Success
        println("\nBenchmarking (3 samples)...")
        circuit = setup_simulation()
        bench = @benchmark tran!($circuit, $tspan; solver=$solver,
                                 initializealg=$init, maxiters=$maxiters, dense=false,
                                 abstol=1e-6, reltol=1e-4) samples=3 evals=1 seconds=1800
        display(bench)
        println()
        return bench, sol
    else
        println("\nSimulation did not converge - skipping benchmark")
        return nothing, sol
    end
end

#==============================================================================#
# Accuracy Benchmark Runner
#==============================================================================#

function generate_accuracy_markdown(all_results)
    io = IOBuffer()

    println(io, "# OTA Accuracy Benchmark Results")
    println(io)
    println(io, "5-Transistor OTA (PSP103 MOSFETs) step response accuracy benchmark.")
    println(io, "Measures actual error vs reference solution at $(length(SAMPLE_TIMES)) sample points.")
    println(io, "Reference: ESDIRK54I8L2SA at abstol/reltol=1e-9.")
    println(io)

    # Work-precision table grouped by category
    for category in unique(r[2] for r in all_results)
        cat_results = filter(r -> r[2] == category, all_results)
        isempty(cat_results) && continue

        println(io, "## $category Methods")
        println(io)
        println(io, "| Solver | Tol | Max Error | RMS Error | Steps | Rejected | nf | Time (s) |")
        println(io, "|--------|-----|-----------|-----------|-------|----------|----|----------|")

        for (solver_name, _, results) in cat_results
            for r in results
                if isnan(r.max_err)
                    println(io, "| $solver_name | $(@sprintf("%.0e", r.tol)) | $(r.retcode) | - | $(r.nsteps) | - | - | $(@sprintf("%.3f", r.time_s)) |")
                else
                    println(io, "| $solver_name | $(@sprintf("%.0e", r.tol)) | $(@sprintf("%.2e", r.max_err)) | $(@sprintf("%.2e", r.rms_err)) | $(r.nsteps) | $(r.rejected) | $(r.nf) | $(@sprintf("%.3f", r.time_s)) |")
                end
            end
        end
        println(io)
    end

    # Best solver at each tolerance
    println(io, "## Best Solver per Tolerance (by max error, among successful runs)")
    println(io)
    println(io, "| Tol | Best Solver | Max Error | Steps | nf |")
    println(io, "|-----|-------------|-----------|-------|----|")

    all_tols = sort(unique(r.tol for (_, _, results) in all_results for r in results))
    for tol in all_tols
        best_name = ""
        best_err = Inf
        best_steps = 0
        best_nf = 0
        for (solver_name, _, results) in all_results
            for r in results
                if r.tol == tol && !isnan(r.max_err) && r.max_err < best_err
                    best_name = solver_name
                    best_err = r.max_err
                    best_steps = r.nsteps
                    best_nf = r.nf
                end
            end
        end
        if !isempty(best_name)
            println(io, "| $(@sprintf("%.0e", tol)) | $best_name | $(@sprintf("%.2e", best_err)) | $best_steps | $best_nf |")
        end
    end
    println(io)

    # Most efficient solver at each tolerance (fewest nf with reasonable accuracy)
    # Only count runs where max_err < 10x the tolerance (i.e., solver is actually working)
    println(io, "## Most Efficient Solver per Tolerance (fewest nf, requiring max_err < 10*tol)")
    println(io)
    println(io, "| Tol | Best Solver | nf | Max Error | Steps | Time (s) |")
    println(io, "|-----|-------------|----|-----------|-------|----------|")

    for tol in all_tols
        best_name = ""
        best_nf = typemax(Int)
        best_err = NaN
        best_steps = 0
        best_time = NaN
        err_threshold = 10 * tol  # Must achieve at least order-of-magnitude accuracy
        for (solver_name, _, results) in all_results
            for r in results
                if r.tol == tol && !isnan(r.max_err) && r.max_err < err_threshold && r.nf > 0 && r.nf < best_nf
                    best_name = solver_name
                    best_nf = r.nf
                    best_err = r.max_err
                    best_steps = r.nsteps
                    best_time = r.time_s
                end
            end
        end
        if !isempty(best_name)
            println(io, "| $(@sprintf("%.0e", tol)) | $best_name | $best_nf | $(@sprintf("%.2e", best_err)) | $best_steps | $(@sprintf("%.3f", best_time)) |")
        end
    end
    println(io)

    return String(take!(io))
end

function run_accuracy_benchmark(; solvers=ALL_SOLVERS,
                                  tolerances=10.0 .^ (-3:-1:-8))
    println("=" ^ 60)
    println("OTA Accuracy Benchmark (PSP103 MOSFETs)")
    println("=" ^ 60)
    println()

    # Step 1: Reference solution
    ref_sol = compute_reference()
    ref_circuit = setup_simulation()
    ref_waveform = extract_output_waveform(ref_sol, ref_circuit, SAMPLE_TIMES)
    println()

    # Step 2: Run each solver at multiple tolerances
    all_results = []
    for (solver_name, solver_fn, category) in solvers
        println("--- $solver_name ($category) ---")
        results = run_solver_accuracy(solver_fn, solver_name, ref_waveform;
                                       tolerances=tolerances)
        push!(all_results, (solver_name, category, results))
        println()
    end

    # Step 3: Generate report
    markdown = generate_accuracy_markdown(all_results)
    return all_results, markdown
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    all_results, markdown = run_accuracy_benchmark()

    if length(ARGS) >= 1
        output_file = ARGS[1]
        open(output_file, "w") do f
            write(f, markdown)
        end
        println("Report written to: $output_file")
    else
        println()
        println(markdown)
    end
end
