#!/usr/bin/env julia
#==============================================================================#
# Ring Oscillator Test - FBDF Focus
#
# FBDF got source stepping to succeed (149 pts, 0.25ns, 4.6 iter/step)
# Try variations to push further
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp
using OrdinaryDiffEq: FBDF
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
            println("  Iter/step:  $(@sprintf("%.1f", sol.stats.nnonliniter / max(1, length(sol.t))))")
        end
        return sol.retcode, sol.t[end]
    catch e
        println("  FAILED: $e")
        return :error, 0.0
    end
end

# FBDF baseline (got 149 pts, 0.25ns before)
test_tran("FBDF baseline";
    solver=FBDF(autodiff=false), dtmax=0.1e-9, maxiters=100000)

# FBDF + force_dtmin (force through unstable points)
test_tran("FBDF + force_dtmin";
    solver=FBDF(autodiff=false), dtmax=0.1e-9, maxiters=100000, force_dtmin=true)

# FBDF + smaller dtmax (catch fast dynamics)
test_tran("FBDF + dtmax=0.01ns";
    solver=FBDF(autodiff=false), dtmax=0.01e-9, maxiters=200000)

# FBDF + looser tolerances (less strict convergence)
test_tran("FBDF + loose tol";
    solver=FBDF(autodiff=false), dtmax=0.1e-9, maxiters=100000,
    abstol=1e-6, reltol=1e-4)

println("\n" * "="^60)
println("Done!")
println("="^60)
