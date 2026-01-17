#!/usr/bin/env julia
#==============================================================================#
# Ring Oscillator Test - CedarTranOp Homotopy Variations
#
# Focus on CedarTranOp (which successfully initializes) with different:
# - Solvers (Rodas5P, FBDF, IDA)
# - Tolerances
# - force_dtmin settings
# - Linear solvers
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp
using OrdinaryDiffEq: Rodas5P, FBDF, ImplicitEuler
using Sundials: IDA
using Printf
using Logging

# Enable debug logging
global_logger(ConsoleLogger(stderr, Logging.Debug))

println("="^60)
println("Step 1: Loading PSP103 model...")
println("="^60)
using PSPModels
println("  PSP103 loaded.")

println("\n" * "="^60)
println("Step 2: Parsing SPICE netlist...")
println("="^60)
const spice_file = joinpath(@__DIR__, "..", "benchmarks", "vacask", "ring", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)
println("  Parsed.")

println("\n" * "="^60)
println("Step 3: Assembling circuit...")
println("="^60)
circuit = MNACircuit(ring_circuit)
data = MNA.assemble!(circuit)
println("  Assembled. $(size(data.G, 1)) unknowns.")

# Test function
function test_tran(name, circuit_factory; solver, tspan=(0.0, 1e-9), kwargs...)
    println("\n" * "="^60)
    println("$name")
    println("="^60)

    circ = circuit_factory()
    MNA.assemble!(circ)

    try
        sol = tran!(circ, tspan; solver=solver, initializealg=CedarTranOp(),
                    dense=false, kwargs...)
        println("  Status:     $(sol.retcode)")
        println("  Timepoints: $(length(sol.t))")
        println("  Final time: $(@sprintf("%.3e", sol.t[end])) / $(@sprintf("%.3e", tspan[2]))")
        if sol.stats !== nothing && sol.stats.nnonliniter > 0
            println("  NR iters:   $(sol.stats.nnonliniter)")
            println("  Iter/step:  $(@sprintf("%.1f", sol.stats.nnonliniter / max(1, length(sol.t))))")
        end
        return sol.retcode
    catch e
        println("  FAILED: $(typeof(e))")
        println("  $e")
        return :error
    end
end

circuit_factory() = MNACircuit(ring_circuit)

# ============================================================================
# Rodas5P variations
# ============================================================================

test_tran("Rodas5P (baseline)", circuit_factory;
    solver=Rodas5P(), dtmax=0.1e-9, maxiters=100000)

test_tran("Rodas5P + tighter tol", circuit_factory;
    solver=Rodas5P(), dtmax=0.1e-9, maxiters=100000,
    abstol=1e-12, reltol=1e-10)

test_tran("Rodas5P + smaller dtmax", circuit_factory;
    solver=Rodas5P(), dtmax=0.01e-9, maxiters=200000)

test_tran("Rodas5P + force_dtmin", circuit_factory;
    solver=Rodas5P(), dtmax=0.1e-9, maxiters=100000,
    force_dtmin=true)

# ============================================================================
# FBDF (BDF method - standard for circuits)
# ============================================================================

test_tran("FBDF (baseline)", circuit_factory;
    solver=FBDF(autodiff=false), dtmax=0.1e-9, maxiters=100000)

test_tran("FBDF + force_dtmin", circuit_factory;
    solver=FBDF(autodiff=false), dtmax=0.1e-9, maxiters=100000,
    force_dtmin=true)

test_tran("FBDF + tighter tol", circuit_factory;
    solver=FBDF(autodiff=false), dtmax=0.1e-9, maxiters=100000,
    abstol=1e-12, reltol=1e-10)

# ============================================================================
# IDA (native DAE solver)
# ============================================================================

test_tran("IDA (baseline)", circuit_factory;
    solver=IDA(linear_solver=:KLU), dtmax=0.1e-9, maxiters=100000)

test_tran("IDA + force_dtmin", circuit_factory;
    solver=IDA(linear_solver=:KLU), dtmax=0.1e-9, maxiters=100000,
    force_dtmin=true)

# ============================================================================
# ImplicitEuler with force_dtmin (simple but robust)
# ============================================================================

test_tran("ImplicitEuler + force_dtmin", circuit_factory;
    solver=ImplicitEuler(autodiff=false), dt=0.01e-9, dtmax=0.1e-9,
    maxiters=100000, force_dtmin=true)

test_tran("ImplicitEuler + force_dtmin + tight tol", circuit_factory;
    solver=ImplicitEuler(autodiff=false), dt=0.01e-9, dtmax=0.1e-9,
    maxiters=100000, force_dtmin=true, abstol=1e-12, reltol=1e-10)

println("\n" * "="^60)
println("Done!")
println("="^60)
