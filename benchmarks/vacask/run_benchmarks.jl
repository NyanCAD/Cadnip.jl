#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark Runner
#
# Runs all VACASK benchmarks and outputs results as markdown suitable for
# GitHub Actions Job Summaries.
#
# Usage:
#   julia --project=benchmarks benchmarks/vacask/run_benchmarks.jl [output_file]
#
# If output_file is provided, markdown is written there.
# Otherwise, it's written to stdout.
#==============================================================================#

using Pkg
Pkg.instantiate()

using Printf
using Statistics
using BenchmarkTools
using SciMLBase: ReturnCode
using Sundials: IDA
using OrdinaryDiffEqBDF: QNDF, QBDF, FBDF
using OrdinaryDiffEqRosenbrock: Rodas3
using ADTypes: AutoFiniteDiff
using LinearSolve: KLUFactorization

const BENCHMARK_DIR = @__DIR__

# Solver definitions, confirmed by full-scale @benchmark runs (not just a
# pass/fail screen) - all use KLU sparse solver (3-4x faster than dense LU
# for these circuits).
#
# ABDF2 was the previous pick here (fastest raw wall-clock) but is deliberately
# excluded now: on the RC circuit it silently returns a solution with rel-L2
# error 1.6e-3 against the exact analytic solution, vs 7e-12-3e-8 for every
# other solver at the SAME abstol=1e-10/reltol=1e-8 - about 5-6 orders of
# magnitude worse, most likely because `smooth_est=true` (its default) damps
# the embedded error estimate enough to make its accept/reject decision too
# lenient. It isn't even fast on the nonlinear cases either (18s on graetz vs
# IDA's 4.7s, 7.9s on mul vs IDA's 2.2s), so nothing is traded away by
# dropping it.
const SOLVER_IDA = ("IDA", () -> IDA(linear_solver=:KLU, max_error_test_failures=20))
const SOLVER_QNDF = ("QNDF", () -> QNDF(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff()))
const SOLVER_QBDF = ("QBDF", () -> QBDF(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff()))
const SOLVER_FBDF = ("FBDF", () -> FBDF(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff()))
const SOLVER_RODAS3 = ("Rodas3", () -> Rodas3(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff()))

# Ring oscillator: FBDF is 4x faster than IDA for PSP103 ring oscillator
const SOLVER_FBDF_RING = ("FBDF", () -> FBDF(autodiff=AutoFiniteDiff()))

# Per-benchmark solver configurations: top 3 by real median @benchmark time
# among solvers that both (a) reach the end and (b) respect the requested
# tolerance, at full production scale - see the accuracy-vs-throughput
# rationale above for why that's not the same as "top 3 by raw wall clock".
#
# RC Circuit (linear). QNDF (0.83s) < IDA (1.83s) < Rodas3 (2.04s), all with
# tight rel-L2 error against the exact analytic solution (<3e-8, Rodas3
# <1e-11). ImplicitEuler/DImplicitEuler stay excluded per the existing
# rationale (heavy rejection near every pulse edge). FBDF/DFBDF/Trapezoid/
# TRBDF2 are also excluded: even with `tstops` at every pulse edge (see
# rc/cedarsim/runme.jl's `pulse_tstops`) they still fail to reach t=1,
# grinding to `maxiters` or underflowing `dt` partway through the
# 500-period sweep.
const SOLVERS_RC = [SOLVER_QNDF, SOLVER_IDA, SOLVER_RODAS3]
# Graetz/Mul (nonlinear diodes): IDA and FBDF are the fastest solvers that
# are robust on BOTH circuits (Graetz: IDA 4.4s, FBDF 6.2s. Mul: IDA 2.4s,
# FBDF 3.3s - all timed on Julia 1.12, matching CI). Plain QNDF was tried
# here too but is EXCLUDED: it looked fine in local Julia 1.11 testing but
# failed for real in CI on Graetz Bridge (`Unstable`, dt underflow at
# t=0.27 - exactly a diode current zero-crossing - after only 272658 of the
# expected ~1000001 steps). That reproduces deterministically on Julia 1.12
# with the SAME resolved package versions that succeeded on 1.11, so it's a
# genuine solver/compiler-codegen interaction, not stale local testing or a
# different package resolution.
#
# Root cause: QNDF's default `kappa` correction coefficients (the
# Shampine-Reichelt stability/accuracy correction for the quasi-constant-step
# formulation) interact badly with the derivative kink at the diode
# zero-crossing. Zeroing `kappa` removes the failure entirely - which is
# exactly what the `QBDF` algorithm is (`QNDF` with `kappa` all zeros).
# `QNDF2` (a distinct fixed-2nd-order implementation, not just "QNDF capped
# at order 2" - that still fails) also survives, but neither tuning
# `extrapolant` nor just lowering `max_order` helps, and `QNDF1` fails even
# earlier than the default - so this isn't a generic "be more conservative"
# fix, it's specifically the kappa term. QBDF fills the third slot: robust
# on both circuits (validated on Julia 1.12), and 5x faster than Rodas3 on
# graetz (8.9s vs 46.1s) and 8x faster on mul (2.7s vs 22.8s, also Julia
# 1.12 timings). Rodas5P/RadauIIA5 are also robust (0 rejects) but slower
# still; every SDIRK/ESDIRK method tried (Trapezoid, TRBDF2,
# KenCarp3/4/5/47/58, Kvaerno3/4/5, SDIRK2, Cash4, Hairer4/42) either fails
# outright or is markedly slower.
const SOLVERS_NONLINEAR = [SOLVER_IDA, SOLVER_FBDF, SOLVER_QBDF]
# Ring Oscillator (PSP103 MOSFETs): FBDF with force_dtmin for no-cap circuit
const SOLVERS_RING = [SOLVER_FBDF_RING]

# Results storage
struct BenchmarkResult
    name::String
    solver::String
    status::Symbol  # :success, :failed, :skipped
    median_time::Float64  # seconds
    min_time::Float64
    max_time::Float64
    memory::Float64  # MB
    allocs::Int
    timepoints::Int
    rejected::Int
    error_msg::String
end

BenchmarkResult(name, solver, status, error_msg="") = BenchmarkResult(name, solver, status, NaN, NaN, NaN, NaN, 0, 0, 0, error_msg)

# Label used for rows sourced from the real VACASK simulator (run_vacask.sh).
const VACASK_REF_SOLVER = "VACASK"

#==============================================================================#
# VACASK reference numbers
#
# run_vacask.sh runs the real VACASK binary via the upstream benchmark.py and
# writes a TSV (benchmark, time_s, timepoints, rejected, iterations). If the
# path is provided via $VACASK_REFERENCE_TSV we fold those rows into the same
# comparison tables for an apples-to-apples, same-machine comparison.
#==============================================================================#
function load_vacask_reference()
    refs = BenchmarkResult[]
    path = get(ENV, "VACASK_REFERENCE_TSV", "")
    (isempty(path) || !isfile(path)) && return refs
    for line in eachline(path)
        isempty(strip(line)) && continue
        cols = split(line, '\t')
        length(cols) < 5 && continue
        name = String(cols[1])
        t = tryparse(Float64, cols[2])
        t === nothing && continue
        tp = something(tryparse(Int, cols[3]), 0)
        rej = something(tryparse(Int, cols[4]), 0)
        push!(refs, BenchmarkResult(name, VACASK_REF_SOLVER, :success,
                                    t, t, t, NaN, 0, tp, rej, ""))
    end
    return refs
end

function format_time(seconds::Float64)
    if isnan(seconds)
        return "-"
    elseif seconds < 1e-3
        return @sprintf("%.2f µs", seconds * 1e6)
    elseif seconds < 1
        return @sprintf("%.2f ms", seconds * 1e3)
    else
        return @sprintf("%.2f s", seconds)
    end
end

function format_memory(mb::Float64)
    if isnan(mb)
        return "-"
    elseif mb < 1
        return @sprintf("%.1f KB", mb * 1024)
    elseif mb < 1024
        return @sprintf("%.1f MB", mb)
    else
        return @sprintf("%.2f GB", mb / 1024)
    end
end

#==============================================================================#
# Generate Markdown Report
#==============================================================================#
function generate_markdown(results::Vector{BenchmarkResult})
    io = IOBuffer()

    println(io, "# VACASK Benchmark Results")
    println(io)
    println(io, "Benchmarks run on Julia $(VERSION)")
    println(io)
    if any(r -> r.solver == VACASK_REF_SOLVER, results)
        println(io, "Rows labelled **$(VACASK_REF_SOLVER)** are the real VACASK simulator measured on the same machine (see `run_vacask.sh`), for an apples-to-apples comparison.")
        println(io)
    end

    # Summary table with all solvers
    println(io, "## Summary")
    println(io)
    println(io, "| Benchmark | Solver | Status | Median Time | Timepoints | Rejected | Memory |")
    println(io, "|-----------|--------|--------|-------------|------------|----------|--------|")

    for r in results
        status_emoji = r.status == :success ? "✅" : r.status == :skipped ? "⏭️" : "❌"
        rejected_str = r.rejected >= 0 ? string(r.rejected) : "-"
        println(io, "| $(r.name) | $(r.solver) | $(status_emoji) | $(format_time(r.median_time)) | $(r.timepoints > 0 ? r.timepoints : "-") | $(rejected_str) | $(format_memory(r.memory)) |")
    end
    println(io)

    # Detailed results grouped by benchmark
    println(io, "## Detailed Results")
    println(io)

    # Group results by benchmark name
    benchmarks = unique(r.name for r in results)
    for bench_name in benchmarks
        println(io, "### $(bench_name)")
        println(io)

        bench_results = filter(r -> r.name == bench_name, results)
        successful = filter(r -> r.status == :success, bench_results)

        if !isempty(successful)
            println(io, "| Solver | Median | Min | Max | Rejected | Memory |")
            println(io, "|--------|--------|-----|-----|----------|--------|")
            for r in successful
                println(io, "| $(r.solver) | $(format_time(r.median_time)) | $(format_time(r.min_time)) | $(format_time(r.max_time)) | $(r.rejected) | $(format_memory(r.memory)) |")
            end
            println(io)

            # Show fastest Cadnip solver (excluding the VACASK reference row)
            cadnip_successful = filter(r -> r.solver != VACASK_REF_SOLVER, successful)
            if !isempty(cadnip_successful)
                fastest = argmin(r -> r.median_time, cadnip_successful)
                note = ""
                ref = filter(r -> r.solver == VACASK_REF_SOLVER, successful)
                if !isempty(ref)
                    ratio = ref[1].median_time / fastest.median_time
                    note = ratio >= 1 ? " — $(round(ratio, digits=2))× faster than VACASK" :
                                        " — $(round(1/ratio, digits=2))× slower than VACASK"
                end
                println(io, "> 🏆 Fastest Cadnip solver: **$(fastest.solver)** ($(format_time(fastest.median_time)))$(note)")
                println(io)
            end
        end

        # Show failures
        failed = filter(r -> r.status == :failed, bench_results)
        for r in failed
            println(io, "> ❌ $(r.solver) failed: $(r.error_msg)")
        end
        println(io)
    end

    return String(take!(io))
end

#==============================================================================#
# Run individual benchmarks
#==============================================================================#

function run_benchmark_with_solver(name, script_path, solver_name, solver_fn)
    println("Running $name with $solver_name...")
    try
        # Include the benchmark script (only once per script)
        if !isdefined(Main, Symbol("__included_$(hash(script_path))"))
            include(script_path)
            Core.eval(Main, Expr(:(=), Symbol("__included_$(hash(script_path))"), true))
        end

        solver = solver_fn()
        bench, sol = Base.invokelatest(run_benchmark, solver)
        if bench === nothing || sol === nothing
            return BenchmarkResult(name, solver_name, :failed, "Benchmark returned nothing")
        end

        # Check if simulation reached the end time
        tspan_end = sol.prob.tspan[2]
        reached_end = isapprox(sol.t[end], tspan_end; rtol=1e-6)

        # Warn about non-success retcode but don't fail if we reached the end
        warning = ""
        if sol.retcode != ReturnCode.Success
            warning = " ($(sol.retcode))"
            @warn "$name with $solver_name: $(sol.retcode)"
        end

        if !reached_end
            return BenchmarkResult(name, solver_name, :failed,
                "Stopped at t=$(sol.t[end]), expected $(tspan_end)")
        end

        # Extract rejected steps from solver stats
        rejected = hasproperty(sol.stats, :nreject) ? sol.stats.nreject : 0

        return BenchmarkResult(
            name, solver_name, :success,
            median(bench.times) / 1e9, minimum(bench.times) / 1e9, maximum(bench.times) / 1e9,
            bench.memory / 1e6, bench.allocs, length(sol.t), rejected, warning
        )
    catch e
        return BenchmarkResult(name, solver_name, :failed, sprint(showerror, e))
    end
end

function run_benchmark_all_solvers(name, script_path, solvers)
    results = BenchmarkResult[]
    for (solver_name, solver_fn) in solvers
        push!(results, run_benchmark_with_solver(name, script_path, solver_name, solver_fn))
    end
    return results
end

#==============================================================================#
# Main
#==============================================================================#
function main()
    println("=" ^ 60)
    println("VACASK Benchmark Suite")
    println("=" ^ 60)
    println()

    results = BenchmarkResult[]

    # VACASK reference numbers (real simulator, same machine) if available.
    vacask_refs = load_vacask_reference()
    if !isempty(vacask_refs)
        println("Including VACASK reference numbers for: ",
                join((r.name for r in vacask_refs), ", "))
        println()
    end

    # Append a benchmark's Cadnip solver results, immediately followed by the
    # matching VACASK reference row so the tables compare side by side.
    function add_benchmark!(name, script, solvers)
        append!(results, run_benchmark_all_solvers(name, script, solvers))
        for ref in vacask_refs
            ref.name == name && push!(results, ref)
        end
    end

    # RC Circuit - linear circuit
    add_benchmark!("RC Circuit",
        joinpath(BENCHMARK_DIR, "rc", "cedarsim", "runme.jl"),
        SOLVERS_RC)

    # Graetz Bridge - nonlinear (diodes), needs robust DAE solver
    add_benchmark!("Graetz Bridge",
        joinpath(BENCHMARK_DIR, "graetz", "cedarsim", "runme.jl"),
        SOLVERS_NONLINEAR)

    # Voltage Multiplier - nonlinear (diodes), needs robust DAE solver
    add_benchmark!("Voltage Multiplier",
        joinpath(BENCHMARK_DIR, "mul", "cedarsim", "runme.jl"),
        SOLVERS_NONLINEAR)

    # Ring Oscillator - IDA only (needs tuned settings for oscillation)
    add_benchmark!("Ring Oscillator (PSP103)",
        joinpath(BENCHMARK_DIR, "ring", "cedarsim", "runme.jl"),
        SOLVERS_RING)

    # # C6288 Multiplier - all solvers
    # append!(results, run_benchmark_all_solvers(
    #     "C6288 Multiplier",
    #     joinpath(BENCHMARK_DIR, "c6288", "cedarsim", "runme.jl")
    # ))

    println()
    println("=" ^ 60)
    println("Generating report...")
    println("=" ^ 60)

    markdown = generate_markdown(results)

    # Write to file or stdout
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

    # Return success - skipped benchmarks don't cause failure
    return 0
end

exit(main())
