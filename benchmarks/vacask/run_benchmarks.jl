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
using OrdinaryDiffEqBDF: QNDF, FBDF
using OrdinaryDiffEqRosenbrock: Rodas3
using ADTypes: AutoFiniteDiff
using LinearSolve: KLUFactorization

const BENCHMARK_DIR = @__DIR__

include(joinpath(BENCHMARK_DIR, "report_utils.jl"))

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
# TRBDF2 are also excluded: even with tstops at every pulse edge (tran!'s
# auto_tstops derives these automatically from the PULSE source now - see
# src/mna/breakpoints.jl) they still fail to reach t=1, grinding to
# `maxiters` or underflowing `dt` partway through the 500-period sweep.
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
# Root cause of the QNDF failure: its default `kappa` correction coefficients
# (the Shampine-Reichelt stability/accuracy correction for the
# quasi-constant-step formulation) interact badly with the derivative kink at
# the diode zero-crossing. Zeroing `kappa` removes the failure - which is
# exactly what the `QBDF` algorithm is (`QNDF` with `kappa` all zeros) - so
# QBDF was the third slot for a while.
#
# The third slot is now Rodas3 (a low-order Rosenbrock method), NOT QBDF.
# QBDF was dropped because it regressed and no longer reaches the end on
# graetz:
#
#   - 2026-07-06 onward, QBDF aborts `Unstable` on graetz with dt forced
#     below Float64 eps at t=0.312 (a diode zero-crossing, same failure mode
#     QNDF has), after ~313099 of the expected ~1000001 steps. Reproduced
#     locally on Julia 1.12 at the throughput settings (dtmax=1e-6,
#     reltol=abstol=1e-3) and in CI on main.
#   - The pin `OrdinaryDiffEqBDF = "=2.2.2"` (added 2026-07-10) does NOT fix
#     it: the latest main CI run resolves OrdinaryDiffEqBDF v2.2.2 exactly
#     and QBDF still aborts at t=0.312. The regression is not in
#     OrdinaryDiffEqBDF at all - it lives in the transitive solver stack that
#     floats regardless of the BDF pin (OrdinaryDiffEqCore v4.6.0,
#     OrdinaryDiffEqDifferentiation v3.2.3, OrdinaryDiffEqNonlinearSolve
#     v2.1.0 in that run). Pinning only BDF was treating the wrong package;
#     the pin has been removed since it holds BDF back for no working reason.
#   - QBDF still passes on mul (it's only graetz's diode kink that trips it),
#     but a solver that no longer reaches the end on one of the two circuits
#     it's listed for can't stay.
#
# Rodas3 (3rd-order Rosenbrock, the same low-order method already used for the
# RC case) replaces it. Rosenbrock is the family the work-precision benchmark
# uses on these diode circuits (see wpd/run_wpd.jl / wpd/README.md "Solver
# survey"), but *low* order is the right pick for THIS benchmark specifically:
# throughput forces a tiny fixed dtmax, so the local truncation error is
# dt-limited, not order-limited, and a high-order method just pays for stages
# it can't use. Measured on Julia 1.12 at the throughput settings, every
# Rosenbrock order reaches t_end with retcode Success on BOTH graetz and mul
# at the SAME accuracy (rel-L2 = 2.87e-6 vs IDA on graetz for Rodas3, Rodas5P
# and ROS3P alike), but runtime tracks order monotonically - graetz (to
# t=0.4): Rodas3 5.5s < Rodas4 7.1s < Rodas5P 8.5s; mul (to 0.2ms): Rodas3
# 2.7s < Rodas4 3.6s < Rodas5P 4.3s. So Rodas3 is the fastest robust
# Rosenbrock here (ROS3P is a touch faster still, ~4.2s/2.6s, but Rodas3 is
# already imported for the RC case and equally accurate, so there's no reason
# to add a second Rosenbrock name). The wpd survey excludes Rodas5P from mul
# because it hangs there under *adaptive* stepping (mul's 100kHz
# cascaded-diode switching is far stiffer); that caveat does NOT apply here -
# the fixed dtmax=1e-9 caps the step and keeps every Rosenbrock order stable.
# RadauIIA5, the other wpd graetz candidate, is currently broken upstream on
# this stack (`UndefVarError: FastConvergence not defined in
# OrdinaryDiffEqFIRK`), so it's not an option. Every SDIRK/ESDIRK method
# previously tried (Trapezoid, TRBDF2, KenCarp3/4/5/47/58, Kvaerno3/4/5,
# SDIRK2, Cash4, Hairer4/42) either fails outright on graetz or is markedly
# slower.
const SOLVERS_NONLINEAR = [SOLVER_IDA, SOLVER_FBDF, SOLVER_RODAS3]
# Ring Oscillator (PSP103 MOSFETs): FBDF with force_dtmin for no-cap circuit
const SOLVERS_RING = [SOLVER_FBDF_RING]

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

    # C6288 Multiplier deliberately NOT run here: its generated builder
    # (2419 inlined subckt calls) hits LLVM RAGreedy's super-linear blowup
    # at default -O2, so it needs its own `julia -O0` process. This process
    # runs everything else at default optimization, so folding c6288 into
    # this loop would reintroduce that hang. See the separate "Run Cadnip
    # c6288 benchmark" CI step (.github/workflows/benchmark.yml) and
    # doc/c6288_bottleneck_findings.md.

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
