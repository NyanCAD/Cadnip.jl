#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Ring Oscillator with PSP103VA
#
# 5-stage ring oscillator using PSP103VA MOSFET model.
#
# IMPORTANT: PSP103VA is a complex model (~700 parameters) that requires
# significant compilation and simulation time:
# - Model loading: ~2-3 minutes in sandbox, ~30s native
# - Circuit compilation: ~3-4 minutes in sandbox, ~1min native
# - Simulation (100ns): >10 minutes in sandbox, ~30s native with IDA
#
# This benchmark successfully demonstrates:
# - Inline VA model loading (workaround for sandbox precompilation issues)
# - Ring oscillator circuit construction
# - Transient simulation initialization
#
# For production benchmarking, run in native environment (not sandbox/CI).
#
# Usage: julia --project=. runme.jl
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp
using OrdinaryDiffEq: FBDF
using Sundials: IDA
using BenchmarkTools
using Printf
using VerilogAParser

# Parse PSP103VA model from VA file directly
const psp103_va_path = joinpath(dirname(@__DIR__), "..", "..", "..", "models", "PSPModels.jl", "va", "psp103.va")

println("Loading PSP103VA...")
println("This may take several minutes in sandbox environments...")
const psp103_va = VerilogAParser.parsefile(psp103_va_path)

println("Generating MNA module...")
const psp_module_expr = CedarSim.make_mna_module(psp103_va)

println("Evaluating module (this is the slow part)...")
eval(psp_module_expr)
println("PSP103VA loaded successfully!")

# Create registration module
module PSPModelsInline
    using ..CedarSim
    using ..PSP103VA_module
    export PSP103VA
    const PSP103VA = PSP103VA_module.PSP103VA

    import ..CedarSim.ModelRegistry: getmodel
    getmodel(::Val{:psp103va}, ::Nothing, ::Nothing, ::Type{<:CedarSim.ModelRegistry.AbstractSimulator}) = PSP103VA
end

# Simplified ring oscillator with minimal PSP103VA parameters
# Use direct PSP103VA instances with uppercase parameters (VA models use uppercase)
const spice_code = """
5 stage ring oscillator

* Inverter using direct PSP103VA instances with load capacitance
.subckt inverter in out vdd vss w=10u l=1u pfact=2
  xp out in vdd vdd PSP103VA TYPE=-1 W={w*pfact} L={l}
  xn out in vss vss PSP103VA TYPE=1 W={w} L={l}
  cload out 0 10f
.ends

* Startup pulse
i0 0 1 dc 0 pulse 0 100u 0.1n 0.1n 0.5n 10n

* 5-stage ring (odd number for oscillation)
xu1 1 2 vdd 0 inverter w=10u l=1u
xu2 2 3 vdd 0 inverter w=10u l=1u
xu3 3 4 vdd 0 inverter w=10u l=1u
xu4 4 5 vdd 0 inverter w=10u l=1u
xu5 5 1 vdd 0 inverter w=10u l=1u

vdd vdd 0 1.2

.end
"""

println("\nParsing SPICE netlist...")
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSPModelsInline])
println("Evaluating circuit...")
eval(circuit_code)

function setup_simulation()
    circuit = MNACircuit(ring_circuit)
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(solver=nothing; tspan=(0.0, 100e-9), dtmax=0.1e-9, maxiters=10_000_000)
    circuit = setup_simulation()

    if solver === nothing
        # Use IDA for more robust handling of oscillators
        solver = IDA(linear_solver=:KLU, max_error_test_failures=20)
    end
    init = CedarTranOp()

    println("\nRing Oscillator Benchmark (PSP103VA - 5 stages)")
    println("="^50)
    println("  Solver:  $(typeof(solver).name.name)")
    println("  Init:    CedarTranOp")
    println("  dtmax:   $(dtmax*1e9) ns")
    println("  tspan:   $(tspan[2]*1e9) ns")
    println()

    println("Running transient analysis...")
    sol = tran!(circuit, tspan; dtmax=dtmax, solver=solver,
                initializealg=init, maxiters=maxiters, dense=false)

    println("\n=== Results ===")
    println("  Status:     $(sol.retcode)")
    @printf("  Timepoints: %d\n", length(sol.t))
    @printf("  Final time: %.3e s\n", sol.t[end])

    if sol.retcode == :Success
        println("\nBenchmarking...")
        circuit = setup_simulation()
        bench = @benchmark tran!($circuit, $tspan; dtmax=$dtmax, solver=$solver,
                                 initializealg=$init, maxiters=$maxiters, dense=false) samples=3 evals=1 seconds=300
        display(bench)
        println()
        return bench, sol
    else
        return nothing, sol
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
