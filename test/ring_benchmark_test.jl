#!/usr/bin/env julia
#==============================================================================#
# Test Ring Benchmark with CedarTranOp Homotopy Modes
#
# Tests different initialization algorithms and homotopy configurations
# for the 9-stage ring oscillator with PSP103 MOSFETs.
#
# This is a manual test - PSP103 compilation takes ~150s.
# Run with: julia --project=test test/ring_benchmark_test.jl
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarDCOp, CedarTranOp, CedarUICOp
using OrdinaryDiffEq: Rodas5P
using Printf
using Logging
using Test

# Enable debug logging to see homotopy stepping progress
global_logger(ConsoleLogger(stderr, Logging.Debug))

# Import pre-parsed PSP103 model from PSPModels package
using PSPModels

# Load and parse the SPICE netlist
const spice_file = joinpath(@__DIR__, "..", "benchmarks", "vacask", "ring", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

function setup_circuit()
    circuit = MNACircuit(ring_circuit)
    MNA.assemble!(circuit)
    return circuit
end

function test_init(name::String, init; solver=Rodas5P(), tspan=(0.0, 50e-9), dtmax=0.1e-9)
    println("\n" * "="^60)
    println("Testing: $name")
    println("="^60)

    circuit = setup_circuit()

    try
        # Short transient simulation to test initialization
        sol = tran!(circuit, tspan; dtmax=dtmax, solver=solver, initializealg=init,
                    maxiters=100_000, dense=false)

        println("  Status:     $(sol.retcode)")
        println("  Timepoints: $(length(sol.t))")
        println("  Final time: $(sol.t[end])")

        if sol.stats !== nothing
            println("  NR iters:   $(sol.stats.nnonliniter)")
            println("  Iter/step:  $(@sprintf("%.2f", sol.stats.nnonliniter / max(1, length(sol.t))))")
        end

        # Check if solution looks reasonable (not NaN or Inf)
        if any(isnan, sol.u[end]) || any(isinf, sol.u[end])
            println("  WARNING: Solution contains NaN or Inf!")
            return :diverged
        end

        return sol.retcode
    catch e
        println("  ERROR: $e")
        showerror(stdout, e, catch_backtrace())
        println()
        return :error
    end
end

function main()
    println("Ring Oscillator Benchmark - CedarTranOp Homotopy Test")
    println("="^60)
    println("Using 9-stage ring oscillator with PSP103 model")

    results = Dict{String, Symbol}()

    # Test 1: CedarDCOp with default settings (use_stepping=true)
    results["CedarDCOp (default)"] = test_init(
        "CedarDCOp (default, use_stepping=true)",
        CedarDCOp()
    )

    # Test 2: CedarTranOp with default settings (use_stepping=true)
    results["CedarTranOp (default)"] = test_init(
        "CedarTranOp (default, use_stepping=true)",
        CedarTranOp()
    )

    # Test 3: CedarTranOp with Shampine collocation
    results["CedarTranOp (shampine)"] = test_init(
        "CedarTranOp (use_shampine=true)",
        CedarTranOp(use_shampine=true)
    )

    # Test 4: CedarTranOp with stepping disabled
    results["CedarTranOp (no stepping)"] = test_init(
        "CedarTranOp (use_stepping=false)",
        CedarTranOp(use_stepping=false)
    )

    # Test 5: CedarTranOp with both shampine and stepping
    results["CedarTranOp (shampine+stepping)"] = test_init(
        "CedarTranOp (use_shampine=true, use_stepping=true)",
        CedarTranOp(use_shampine=true, use_stepping=true)
    )

    # Test 6: CedarUICOp (pseudo-transient for oscillators)
    results["CedarUICOp (default)"] = test_init(
        "CedarUICOp (default)",
        CedarUICOp()
    )

    # Test 7: CedarUICOp with more warmup steps
    results["CedarUICOp (warmup=50)"] = test_init(
        "CedarUICOp (warmup_steps=50)",
        CedarUICOp(warmup_steps=50)
    )

    # Test 8: CedarUICOp with Shampine
    results["CedarUICOp (shampine)"] = test_init(
        "CedarUICOp (use_shampine=true)",
        CedarUICOp(use_shampine=true)
    )

    # Summary
    println("\n" * "="^60)
    println("Summary")
    println("="^60)
    for (name, result) in sort(collect(results), by=first)
        status = result == :Success ? "PASS" : "FAIL ($result)"
        println("  $name: $status")
    end

    successes = count(r -> r == :Success, values(results))
    println("\nPassed: $successes / $(length(results))")

    # Return test pass/fail for CI
    @testset "Ring oscillator CedarTranOp homotopy modes" begin
        for (name, result) in results
            @test result == :Success
        end
    end

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
