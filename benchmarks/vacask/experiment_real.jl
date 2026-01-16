#!/usr/bin/env julia
#==============================================================================#
# VACASK Real Benchmark Solver Experiment
#
# Tests different solver configurations on the actual benchmark circuits.
#
# Usage:
#   julia --project=benchmarks benchmarks/vacask/experiment_real.jl
#==============================================================================#

using Pkg
Pkg.instantiate()

using Printf
using Statistics
using BenchmarkTools
using SciMLBase: ReturnCode
using Sundials: IDA
using OrdinaryDiffEq: FBDF, ImplicitEuler
using LinearSolve: KLUFactorization, LUFactorization

using CedarSim
using CedarSim.MNA
using VADistillerModels

const BENCHMARK_DIR = joinpath(@__DIR__)

#==============================================================================#
# Load actual benchmark circuits
#==============================================================================#

# RC Circuit
const rc_spice = read(joinpath(BENCHMARK_DIR, "rc/cedarsim/runme.sp"), String)
const rc_code = parse_spice_to_mna(rc_spice; circuit_name=:rc_circuit)
eval(rc_code)

# Graetz Bridge
const graetz_spice = read(joinpath(BENCHMARK_DIR, "graetz/cedarsim/runme.sp"), String)
const graetz_code = parse_spice_to_mna(graetz_spice; circuit_name=:graetz_circuit,
                                       imported_hdl_modules=[sp_diode_module])
eval(graetz_code)

# Voltage Multiplier
const mul_spice = read(joinpath(BENCHMARK_DIR, "mul/cedarsim/runme.sp"), String)
const mul_code = parse_spice_to_mna(mul_spice; circuit_name=:mul_circuit,
                                    imported_hdl_modules=[sp_diode_module])
eval(mul_code)

#==============================================================================#
# Solver configurations
#==============================================================================#

const SOLVERS = [
    ("IDA-KLU", () -> IDA(linear_solver=:KLU, max_error_test_failures=20, max_nonlinear_iters=10)),
    ("IDA-Dense", () -> IDA(linear_solver=:Dense, max_error_test_failures=20, max_nonlinear_iters=10)),
    ("FBDF-KLU", () -> FBDF(linsolve=KLUFactorization())),
    ("FBDF-LU", () -> FBDF(linsolve=LUFactorization())),
    ("ImplicitEuler-KLU", () -> ImplicitEuler(linsolve=KLUFactorization())),
    ("ImplicitEuler-LU", () -> ImplicitEuler(linsolve=LUFactorization())),
]

#==============================================================================#
# Test function
#==============================================================================#

function run_test(circuit_name, builder_fn, solver_name, solver_fn;
                  dt, tend, abstol=1e-10, reltol=1e-8, maxiters=10_000_000, samples=3)
    print("  $solver_name... ")
    tspan = (0.0, tend)

    try
        # Warmup
        circuit = MNACircuit(builder_fn)
        MNA.assemble!(circuit)
        solver = solver_fn()
        sol = tran!(circuit, tspan; dtmax=dt, solver=solver, abstol=abstol, reltol=reltol,
                    maxiters=maxiters, dense=false)

        if sol.t[end] < 0.95 * tend
            println("FAILED (stopped at t=$(sol.t[end]))")
            return nothing
        end
        println("OK ($(length(sol.t)) pts)")

        # Benchmark
        bench = @benchmark begin
            circuit = MNACircuit($builder_fn)
            MNA.assemble!(circuit)
            tran!(circuit, $tspan; dtmax=$dt, solver=$(solver_fn()), abstol=$abstol,
                  reltol=$reltol, maxiters=$maxiters, dense=false)
        end samples=samples evals=1 seconds=300

        # Final stats
        circuit = MNACircuit(builder_fn)
        MNA.assemble!(circuit)
        sol = tran!(circuit, tspan; dtmax=dt, solver=solver_fn(), abstol=abstol,
                    reltol=reltol, maxiters=maxiters, dense=false)

        return (
            name=solver_name,
            time=median(bench.times) / 1e9,
            min_time=minimum(bench.times) / 1e9,
            max_time=maximum(bench.times) / 1e9,
            allocs=bench.allocs,
            memory=bench.memory / 1e6,
            points=length(sol.t),
            iters=sol.stats.nnonliniter,
            rejected=hasproperty(sol.stats, :nreject) ? sol.stats.nreject : 0
        )
    catch e
        println("ERROR: ", sprint(showerror, e))
        return nothing
    end
end

function run_circuit_tests(name, builder_fn; dt, tend, abstol=1e-10, reltol=1e-8)
    println("\n=== $name ===")
    println("Config: dt=$(dt), tend=$(tend), abstol=$(abstol), reltol=$(reltol)")

    results = []
    for (solver_name, solver_fn) in SOLVERS
        result = run_test(name, builder_fn, solver_name, solver_fn;
                         dt=dt, tend=tend, abstol=abstol, reltol=reltol)
        if result !== nothing
            push!(results, result)
        end
    end

    if isempty(results)
        println("All tests failed!")
        return results
    end

    sort!(results, by=r->r.time)
    println("\n| Solver | Time | Allocs | Memory | Points | Rejected |")
    println("|--------|------|--------|--------|--------|----------|")
    for r in results
        @printf("| %-16s | %7.2f ms | %8d | %6.2f MB | %6d | %4d |\n",
                r.name, r.time*1000, r.allocs, r.memory, r.points, r.rejected)
    end

    fastest = results[1]
    baseline = filter(r -> r.name == "IDA-KLU", results)
    if !isempty(baseline)
        speedup = baseline[1].time / fastest.time
        println("\n🏆 Fastest: $(fastest.name) ($(round(fastest.time*1000, digits=2)) ms)")
        println("   Speedup vs IDA-KLU: $(round(speedup, digits=2))x")
    end

    return results
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    println("=" ^ 70)
    println("VACASK Real Benchmark Solver Experiment")
    println("=" ^ 70)

    all_results = Dict{String, Vector}()

    # RC Circuit - matches actual benchmark config
    all_results["RC"] = run_circuit_tests("RC Circuit", rc_circuit;
                                          dt=1e-6, tend=1.0)

    # Graetz Bridge - matches actual benchmark config
    all_results["Graetz"] = run_circuit_tests("Graetz Bridge", graetz_circuit;
                                              dt=1e-6, tend=1.0, abstol=1e-3, reltol=1e-3)

    # Voltage Multiplier - matches actual benchmark config
    all_results["Mul"] = run_circuit_tests("Voltage Multiplier", mul_circuit;
                                           dt=1e-9, tend=0.5e-3, abstol=1e-3, reltol=1e-3)

    # Summary
    println("\n" * "=" ^ 70)
    println("OVERALL SUMMARY")
    println("=" ^ 70)

    println("\nBest solver per circuit:")
    for (circuit, results) in all_results
        if !isempty(results)
            best = results[1]  # Already sorted
            baseline = filter(r -> r.name == "IDA-KLU", results)
            speedup_str = !isempty(baseline) ? " ($(round(baseline[1].time / best.time, digits=2))x vs IDA-KLU)" : ""
            println("  $circuit: $(best.name)$speedup_str")
        end
    end

    # Memory comparison
    println("\nMemory usage (IDA-Dense vs IDA-KLU):")
    for (circuit, results) in all_results
        dense = filter(r -> r.name == "IDA-Dense", results)
        sparse = filter(r -> r.name == "IDA-KLU", results)
        if !isempty(dense) && !isempty(sparse)
            ratio = dense[1].memory / sparse[1].memory
            println("  $circuit: $(round(ratio, digits=2))x ($(round(dense[1].memory, digits=2)) MB vs $(round(sparse[1].memory, digits=2)) MB)")
        end
    end
end

main()
