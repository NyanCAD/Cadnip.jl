#!/usr/bin/env julia
#==============================================================================#
# Compare CedarSim vs ngspice ring oscillator waveforms
#
# Runs CedarSim on the same circuit as ngspice (no load caps, 10uA pulse),
# loads ngspice reference data, and generates a comparison plot.
#
# Setup (extra deps not in benchmarks/Project.toml):
#   julia --project=benchmarks -e 'using Pkg; Pkg.add(["CairoMakie", "DelimitedFiles"])'
#
# Generate ngspice reference data (requires ngspice + psp103v4.osdi):
#   cd benchmarks/vacask/ring/ngspice && ngspice runme_export.sim
#
# Run:
#   julia --project=benchmarks benchmarks/vacask/ring/cedarsim/compare_ngspice.jl
#
# Output: benchmarks/vacask/ring/cedarsim/ring_oscillator_waveforms.png
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarTranOp, MNASolutionAccessor, voltage
using OrdinaryDiffEq: FBDF
using SciMLBase
using Printf
using VACASKModels
using CairoMakie
using DelimitedFiles

# Parse circuit (same netlist as ngspice — no load caps, 10uA pulse)
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_compare,
                                         imported_hdl_modules=[VACASKModels])
eval(circuit_code)

println("Running 1μs transient (matching ngspice settings)...")
circuit = MNACircuit(ring_compare)
sys = MNA.assemble!(circuit)

t0 = time()
sol = tran!(circuit, (0.0, 1e-6);
    solver=FBDF(autodiff=false), dtmax=0.05e-9,
    initializealg=CedarTranOp(), maxiters=100_000_000, dense=false,
    force_dtmin=true, abstol=1e-4, reltol=1e-2,
    unstable_check=(dt,u,p,t)->false)
elapsed = time() - t0

@printf("CedarSim: %s, %d pts, %.1f s, %d NR iters\n",
    sol.retcode, length(sol.t), elapsed,
    sol.stats !== nothing ? sol.stats.nnonliniter : 0)

# Extract CedarSim waveforms
acc = MNASolutionAccessor(sol, sys)
n_samples = 10000
times_cs = range(0.0, 1e-6; length=n_samples)
println("Extracting CedarSim waveforms...")

V_cs = Dict{Int, Vector{Float64}}()
for node_num in 1:9
    node_sym = Symbol(string(node_num))
    V_cs[node_num] = [voltage(acc, node_sym, t) for t in times_cs]
end

# Load ngspice data
println("Loading ngspice data...")
ngspice_file = joinpath(@__DIR__, "..", "ngspice", "ring_ngspice.csv")
raw = readdlm(ngspice_file)
t_ng = raw[:, 1]
V_ng = Dict{Int, Vector{Float64}}()
for i in 1:9
    V_ng[i] = raw[:, 2*i]  # voltage columns at 2,4,6,8,10,12,14,16,18
end
@printf("ngspice: %d timepoints, t=[%.3e, %.3e]\n", length(t_ng), t_ng[1], t_ng[end])

# Plot comparison
println("Generating comparison plots...")
t_cs_ns = collect(times_cs) .* 1e9
t_ng_ns = t_ng .* 1e9

fig = Figure(size=(1400, 1000))

# Panel 1: Full 1μs comparison of node 1
ax1 = Axis(fig[1, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
           title="Node 1: CedarSim vs ngspice (Full 1μs)")
lines!(ax1, t_ng_ns, V_ng[1]; color=:blue, linewidth=1.5, label="ngspice")
lines!(ax1, t_cs_ns, V_cs[1]; color=:red, linewidth=1.0, label="CedarSim")
axislegend(ax1; position=:rt)
ylims!(ax1, -0.2, 1.4)

# Panel 2: Zoom into startup (0-50ns)
ax2 = Axis(fig[2, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
           title="Node 1: Startup Detail (0–50 ns)")
mask_ng = t_ng_ns .<= 50
mask_cs = t_cs_ns .<= 50
lines!(ax2, t_ng_ns[mask_ng], V_ng[1][mask_ng]; color=:blue, linewidth=1.5, label="ngspice")
lines!(ax2, t_cs_ns[mask_cs], V_cs[1][mask_cs]; color=:red, linewidth=1.0, label="CedarSim")
axislegend(ax2; position=:rt)
ylims!(ax2, -0.2, 1.4)

# Panel 3: Steady-state zoom (900-1000ns)
ax3 = Axis(fig[3, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
           title="Node 1: Steady-State Detail (900–1000 ns)")
mask_ng3 = t_ng_ns .>= 900 .&& t_ng_ns .<= 1000
mask_cs3 = t_cs_ns .>= 900 .&& t_cs_ns .<= 1000
lines!(ax3, t_ng_ns[mask_ng3], V_ng[1][mask_ng3]; color=:blue, linewidth=1.5, label="ngspice")
lines!(ax3, t_cs_ns[mask_cs3], V_cs[1][mask_cs3]; color=:red, linewidth=1.0, label="CedarSim")
axislegend(ax3; position=:rt)
ylims!(ax3, -0.2, 1.4)

# Panel 4: All nodes at steady state
ax4 = Axis(fig[4, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
           title="All 9 Nodes — CedarSim (900–1000 ns)")
colors = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :brown, :gray]
for i in 1:9
    lines!(ax4, t_cs_ns[mask_cs3], V_cs[i][mask_cs3]; color=colors[i], linewidth=1.0, label="Node $i")
end
axislegend(ax4; position=:rt, nbanks=3)
ylims!(ax4, -0.2, 1.4)

outfile = joinpath(@__DIR__, "ring_oscillator_waveforms.png")
save(outfile, fig; px_per_unit=2)
println("Saved: $outfile")

# Print comparison stats
println("\n=== Comparison Summary ===")
@printf("ngspice:  1μs in 1.5s, %d pts, 80580 NR iters, trap method\n", length(t_ng))
@printf("CedarSim: 1μs in %.1fs, %d pts, %d NR iters, FBDF\n",
    elapsed, length(sol.t),
    sol.stats !== nothing ? sol.stats.nnonliniter : 0)

println("\nDone!")
