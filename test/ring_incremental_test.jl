#!/usr/bin/env julia
#==============================================================================#
# Ring Oscillator Test - CedarTranOp Focused
#
# Most promising configurations based on prior results:
# - Rodas5P got 34k timepoints before unstable
# - force_dtmin may help push through
# - IDA is standard for circuits
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp
using OrdinaryDiffEq: Rodas5P, FBDF
using Sundials: IDA
using Printf
using Logging

global_logger(ConsoleLogger(stderr, Logging.Debug))

println("="^60)
println("Loading PSP103 and parsing circuit...")
println("="^60)
using PSPModels

const spice_file = joinpath(@__DIR__, "..", "benchmarks", "vacask", "ring", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

circuit = MNACircuit(ring_circuit)
data = MNA.assemble!(circuit)
println("  Assembled. $(size(data.G, 1)) unknowns.")

function test_tran(name; solver, tspan=(0.0, 1e-9), kwargs...)
    println("\n" * "="^60)
    println("$name")
    println("="^60)

    circ = MNACircuit(ring_circuit)
    MNA.assemble!(circ)

    try
        sol = tran!(circ, tspan; solver=solver, initializealg=CedarTranOp(),
                    dense=false, kwargs...)
        println("  Status:     $(sol.retcode)")
        println("  Timepoints: $(length(sol.t))")
        println("  Final time: $(@sprintf("%.3e", sol.t[end])) / $(@sprintf("%.3e", tspan[2]))")
        pct = 100 * sol.t[end] / tspan[2]
        println("  Progress:   $(@sprintf("%.1f", pct))%")
        if sol.stats !== nothing && sol.stats.nnonliniter > 0
            println("  NR iters:   $(sol.stats.nnonliniter)")
        end
        return sol.retcode, sol.t[end]
    catch e
        println("  FAILED: $e")
        return :error, 0.0
    end
end

# Test 1: Rodas5P + force_dtmin (baseline was unstable, try forcing through)
test_tran("Rodas5P + force_dtmin";
    solver=Rodas5P(), dtmax=0.1e-9, maxiters=100000, force_dtmin=true)

# Test 2: FBDF + force_dtmin (BDF is standard for stiff circuits)
test_tran("FBDF + force_dtmin";
    solver=FBDF(autodiff=false), dtmax=0.1e-9, maxiters=100000, force_dtmin=true)

# Test 3: IDA (native DAE solver with KLU sparse)
test_tran("IDA";
    solver=IDA(linear_solver=:KLU), dtmax=0.1e-9, maxiters=100000)

println("\n" * "="^60)
println("Done!")
println("="^60)
