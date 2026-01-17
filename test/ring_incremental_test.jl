#!/usr/bin/env julia
#==============================================================================#
# Ring Oscillator Test - Winning Configuration
#
# FBDF + CedarTranOp + dtmax=0.01ns successfully simulates the PSP103 ring oscillator.
#
# See doc/ring_oscillator_investigation.md for full investigation details.
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp
using OrdinaryDiffEq: FBDF
using Printf
using Test

println("="^60)
println("Ring Oscillator Test - PSP103 + CedarTranOp")
println("="^60)

println("\nLoading PSP103 model...")
using PSPModels

const spice_file = joinpath(@__DIR__, "..", "benchmarks", "vacask", "ring", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

circuit = MNACircuit(ring_circuit)
data = MNA.assemble!(circuit)
println("Assembled: $(size(data.G, 1)) unknowns")

println("\nRunning transient (FBDF + CedarTranOp + dtmax=0.01ns)...")
sol = tran!(circuit, (0.0, 1e-9);
    solver=FBDF(autodiff=false),
    dtmax=0.01e-9,
    initializealg=CedarTranOp(),
    maxiters=100000,
    dense=false)

println("\n=== Results ===")
println("  Status:     $(sol.retcode)")
@printf("  Timepoints: %d\n", length(sol.t))
@printf("  Final time: %.3e / %.3e\n", sol.t[end], 1e-9)
pct = 100 * sol.t[end] / 1e-9
@printf("  Progress:   %.1f%%\n", pct)
if sol.stats !== nothing && sol.stats.nnonliniter > 0
    @printf("  NR iters:   %d\n", sol.stats.nnonliniter)
    @printf("  Iter/step:  %.1f\n", sol.stats.nnonliniter / length(sol.t))
end

@testset "Ring oscillator transient" begin
    @test sol.retcode == :Success
    @test sol.t[end] >= 1e-9 * 0.99  # Allow small tolerance
end

println("\nDone!")
