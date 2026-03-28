#!/usr/bin/env julia
#==============================================================================#
# OTA Accuracy Benchmark Runner
#
# Work-precision benchmark for a 5-transistor CMOS OTA (PSP103 MOSFETs).
# Compares BDF (IDA), Rosenbrock, and ESDIRK solver families on an accuracy-
# constrained problem: step response of a differential pair.
#
# Unlike the speed benchmarks (which fix dtmax and measure throughput),
# this benchmark varies tolerance and measures actual accuracy vs a
# reference solution, producing work-precision diagrams.
#
# Usage:
#   julia --project=benchmarks benchmarks/vacask/run_accuracy_benchmarks.jl [output.md]
#
# Solvers compared:
#   BDF (DAE):  IDA (Sundials, baseline)
#   Rosenbrock: Rodas4P, Rodas5P
#   ESDIRK:     KenCarp3, KenCarp4, KenCarp5, KenCarp47, Kvaerno5,
#               ESDIRK54I8L2SA, ESDIRK436L2SA2
#==============================================================================#

using Pkg
Pkg.instantiate()

using Printf
using Statistics

# Include the OTA benchmark (loads circuit, defines all solvers)
const BENCHMARK_DIR = @__DIR__
include(joinpath(BENCHMARK_DIR, "opamp", "cedarsim", "runme.jl"))

function main()
    # Tolerance sweep: 1e-3 to 1e-8
    # (1e-9 excluded from sweep since it's the reference tolerance)
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    all_results, markdown = run_accuracy_benchmark(;
        solvers=ALL_SOLVERS,
        tolerances=tolerances
    )

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

    return 0
end

exit(main())
