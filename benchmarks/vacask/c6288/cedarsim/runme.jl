#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: C6288 16x16 Multiplier
#
# A 16x16 bit multiplier circuit using PSP103 MOSFETs.
#
# Benchmark target: High complexity digital circuit (154k variables)
#
# STATUS: NOT WORKING - requires fixes to sparse Jacobian handling
#
# Known Issues (January 2026):
# 1. DC operating point solve fails with SingularException - this is expected
#    for digital circuits that don't have a valid DC solution. ngspice handles
#    this with 'uic' (use initial conditions) to skip DC and start from zeros.
#
# 2. When using KLU sparse solver (required - dense would need 23GB for 154k^2),
#    Sundials reports "Sparsity Pattern in receiving SUNMatrix doesn't match
#    sending SparseMatrix". This is due to a mismatch between:
#    - The jac_prototype (G+C pattern) we provide to IDA
#    - The actual Jacobian matrix filled in by fast_jacobian!
#
# 3. The root cause appears to be in how Sundials.jl handles sparse matrix
#    copying between Julia SparseMatrixCSC and SUNMatrix. When G and C have
#    different sparsity patterns, naive addition/broadcasting doesn't preserve
#    the expected structure.
#
# Potential solutions to investigate:
# - Precompute index mappings from G.nzval/C.nzval indices to J.nzval indices
# - Ensure jac_prototype uses structural union (abs.(G) .+ abs.(C)) to prevent
#   numerical cancellation
# - Check if ShampineCollocationInit's internal FD Jacobian causes issues
#
# See: https://sciml.ai/news/2025/09/17/sundials_v5_release/ for IDA options
#==============================================================================#

using CedarSim
using CedarSim.MNA
using Sundials
using BenchmarkTools
using Printf
using VerilogAParser

# Load the PSP103 model
const psp103_path = joinpath(@__DIR__, "..", "..", "..", "..", "test", "vadistiller", "models", "psp103v4", "psp103.va")

if isfile(psp103_path)
    println("Loading PSP103 from: ", psp103_path)
    va = VerilogAParser.parsefile(psp103_path)
    if !va.ps.errored
        Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
        println("PSP103VA_module loaded successfully")
    else
        error("Failed to parse PSP103 VA model")
    end
else
    error("PSP103 VA model not found at $psp103_path")
end

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")

# Parse SPICE file to code, then evaluate to get the builder function
const circuit_code = parse_spice_file_to_mna(spice_file; circuit_name=:c6288_circuit,
                                              imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
"""
function setup_simulation()
    circuit = MNACircuit(c6288_circuit)
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(; reltol=1e-3)
    tspan = (0.0, 2e-9)  # 2ns simulation (same as ngspice)

    # Use Sundials IDA with KLU sparse solver
    # Note: Dense solver would require 154k^2 * 8 bytes = 189GB memory!
    # TODO: This currently fails with sparsity pattern mismatch errors
    solver = IDA(linear_solver=:KLU, max_nonlinear_iters=100, max_error_test_failures=20)

    # Setup the simulation outside the timed region
    circuit = setup_simulation()
    n = MNA.system_size(circuit)
    println("Circuit size: $n variables")

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with IDA (reltol=$reltol)...")
    bench = @benchmark tran!($circuit, $tspan; solver=$solver, reltol=$reltol) samples=6 evals=1 seconds=600

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; solver=solver, reltol=reltol)

    println("\n=== Results ===")
    @printf("Timepoints: %d\n", length(sol.t))
    @printf("NR iters:   %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
