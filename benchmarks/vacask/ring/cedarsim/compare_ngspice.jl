#!/usr/bin/env julia
#==============================================================================#
# Compare CedarSim vs ngspice ring oscillator waveforms
#
# Plots adjacent node voltages and VDD supply current (shoot-through indicator)
# to diagnose MNA accuracy vs ngspice/VACASK reference data.
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
#         benchmarks/vacask/ring/cedarsim/ring_oscillator_currents.png
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

# Helper
_mean(x) = sum(x) / length(x)

# Parse circuit (same netlist as ngspice — no load caps, 10uA pulse)
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_compare,
                                         imported_hdl_modules=[VACASKModels])
eval(circuit_code)

println("Running 1us transient (matching ngspice settings)...")
circuit = MNACircuit(ring_compare)
sys = MNA.assemble!(circuit)

# Print available variable names for debugging
println("\nMNA system variables:")
println("  Nodes ($(sys.n_nodes)): ", join(string.(sys.node_names[1:min(20, end)]), ", "),
        sys.n_nodes > 20 ? "..." : "")
println("  Currents ($(sys.n_currents)): ", join(string.(sys.current_names), ", "))
println("  Charges ($(sys.n_charges)): $(sys.n_charges) state variables")
println()

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

# Extract CedarSim waveforms at uniform sample points (for voltage comparison)
acc = MNASolutionAccessor(sol, sys)
n_samples = 10000
times_cs = range(0.0, 1e-6; length=n_samples)
println("Extracting CedarSim waveforms...")

V_cs = Dict{Int, Vector{Float64}}()
for node_num in 1:9
    node_sym = Symbol(string(node_num))
    V_cs[node_num] = [voltage(acc, node_sym, t) for t in times_cs]
end

# Extract VDD supply current at raw solver time points (preserves transient features).
# The branch current through the VDD voltage source shows total shoot-through:
# when both NMOS and PMOS conduct during switching, I(VDD) spikes.
println("Extracting VDD supply current...")
I_vdd_name = :I_vdd
has_ivdd = I_vdd_name in sys.current_names
I_vdd_raw = Float64[]
t_raw = collect(sol.t)
if has_ivdd
    I_vdd_raw = acc[I_vdd_name]
    @printf("  I(VDD): %d raw points, range [%.3e, %.3e] A\n",
        length(I_vdd_raw), minimum(I_vdd_raw), maximum(I_vdd_raw))
else
    println("  WARNING: I_vdd not found in current variables")
    println("  Available: ", join(string.(sys.current_names), ", "))
end

# Also extract VDD current at uniform points for smoother comparison plots
I_vdd_uniform = Float64[]
if has_ivdd
    ivdd_idx = findfirst(==(I_vdd_name), sys.current_names)
    curr_sys_idx = sys.n_nodes + ivdd_idx
    I_vdd_uniform = [sol(t)[curr_sys_idx] for t in times_cs]
end

# Load ngspice data
println("Loading ngspice data...")
ngspice_file = joinpath(@__DIR__, "..", "ngspice", "ring_ngspice.csv")
has_ngspice = isfile(ngspice_file)
t_ng = Float64[]
V_ng = Dict{Int, Vector{Float64}}()
I_vdd_ng = Float64[]
has_ng_current = false

if has_ngspice
    raw = readdlm(ngspice_file)
    t_ng = raw[:, 1]
    ncols = size(raw, 2)
    for i in 1:9
        col = 2*i
        if col <= ncols
            V_ng[i] = raw[:, col]
        end
    end
    @printf("ngspice: %d timepoints, t=[%.3e, %.3e]\n", length(t_ng), t_ng[1], t_ng[end])

    # Check for current data (column 20 = I(VDD) if exported)
    if ncols >= 20
        I_vdd_ng = raw[:, 20]
        has_ng_current = true
        @printf("ngspice I(VDD): range [%.3e, %.3e] A\n", minimum(I_vdd_ng), maximum(I_vdd_ng))
    else
        println("ngspice: no current data (re-run ngspice with runme_export.sim to include I(VDD))")
    end
else
    println("ngspice data not found at $ngspice_file")
    println("To generate: cd benchmarks/vacask/ring/ngspice && ngspice runme_export.sim")
end

#==============================================================================#
# Plot 1: Original waveform comparison (voltages)
#==============================================================================#
println("\nGenerating waveform comparison plot...")
t_cs_ns = collect(times_cs) .* 1e9
t_ng_ns = has_ngspice ? t_ng .* 1e9 : Float64[]

fig = Figure(size=(1400, 1000))

# Panel 1: Full 1us comparison of node 1
ax1 = Axis(fig[1, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
           title="Node 1: CedarSim vs ngspice (Full 1us)")
if has_ngspice
    lines!(ax1, t_ng_ns, V_ng[1]; color=:blue, linewidth=1.5, label="ngspice")
end
lines!(ax1, t_cs_ns, V_cs[1]; color=:red, linewidth=1.0, label="CedarSim")
axislegend(ax1; position=:rt)
ylims!(ax1, -0.2, 1.4)

# Panel 2: Zoom into startup (0-50ns)
ax2 = Axis(fig[2, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
           title="Node 1: Startup Detail (0-50 ns)")
mask_cs = t_cs_ns .<= 50
if has_ngspice
    mask_ng = t_ng_ns .<= 50
    lines!(ax2, t_ng_ns[mask_ng], V_ng[1][mask_ng]; color=:blue, linewidth=1.5, label="ngspice")
end
lines!(ax2, t_cs_ns[mask_cs], V_cs[1][mask_cs]; color=:red, linewidth=1.0, label="CedarSim")
axislegend(ax2; position=:rt)
ylims!(ax2, -0.2, 1.4)

# Panel 3: Steady-state zoom (900-1000ns)
ax3 = Axis(fig[3, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
           title="Node 1: Steady-State Detail (900-1000 ns)")
mask_cs3 = t_cs_ns .>= 900 .&& t_cs_ns .<= 1000
if has_ngspice
    mask_ng3 = t_ng_ns .>= 900 .&& t_ng_ns .<= 1000
    lines!(ax3, t_ng_ns[mask_ng3], V_ng[1][mask_ng3]; color=:blue, linewidth=1.5, label="ngspice")
end
lines!(ax3, t_cs_ns[mask_cs3], V_cs[1][mask_cs3]; color=:red, linewidth=1.0, label="CedarSim")
axislegend(ax3; position=:rt)
ylims!(ax3, -0.2, 1.4)

# Panel 4: All nodes at steady state
ax4 = Axis(fig[4, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
           title="All 9 Nodes - CedarSim (900-1000 ns)")
colors = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :brown, :gray]
for i in 1:9
    lines!(ax4, t_cs_ns[mask_cs3], V_cs[i][mask_cs3]; color=colors[i], linewidth=1.0, label="Node $i")
end
axislegend(ax4; position=:rt, nbanks=3)
ylims!(ax4, -0.2, 1.4)

outfile = joinpath(@__DIR__, "ring_oscillator_waveforms.png")
save(outfile, fig; px_per_unit=2)
println("Saved: $outfile")

#==============================================================================#
# Plot 2: Adjacent nodes and device currents (shoot-through analysis)
#
# This plot is designed to diagnose MNA accuracy by showing:
# - Two adjacent inverter outputs (V(1) = xu1 input, V(2) = xu1 output)
# - VDD supply current I(VDD) which spikes during shoot-through
# - Zoomed transition to show edge shape and shoot-through bump
#
# Context from jax-spice investigation (Jan 2025):
# - Shoot-through causes a bump on rising/falling edges visible in ngspice/VACASK
# - If gate-drain capacitance or current is missing, the bump is absent
# - Blocky current waveforms indicate insufficient time resolution
# - Oscillation frequency depends on timestep if reactive terms are wrong
#==============================================================================#
println("\nGenerating current analysis plot...")

t_raw_ns = t_raw .* 1e9

fig2 = Figure(size=(1600, 1200))

# --- Left column: Steady-state overview (show ~6 oscillation periods) ---
# Pick a window that shows several complete cycles
ss_start, ss_end = 900.0, 960.0  # ns

# Panel [1,1]: Adjacent node voltages V(1) and V(2)
ax_v = Axis(fig2[1, 1]; xlabel="Time (ns)", ylabel="Voltage (V)",
            title="Adjacent Nodes: V(1) input, V(2) output of inverter xu1")
mask_ss = t_cs_ns .>= ss_start .&& t_cs_ns .<= ss_end
lines!(ax_v, t_cs_ns[mask_ss], V_cs[1][mask_ss]; color=:blue, linewidth=1.5, label="V(1) - xu1 input")
lines!(ax_v, t_cs_ns[mask_ss], V_cs[2][mask_ss]; color=:red, linewidth=1.5, label="V(2) - xu1 output")
if has_ngspice
    mask_ng_ss = t_ng_ns .>= ss_start .&& t_ng_ns .<= ss_end
    lines!(ax_v, t_ng_ns[mask_ng_ss], V_ng[1][mask_ng_ss]; color=:blue, linewidth=0.8,
           linestyle=:dash, label="ngspice V(1)")
    lines!(ax_v, t_ng_ns[mask_ng_ss], V_ng[2][mask_ng_ss]; color=:red, linewidth=0.8,
           linestyle=:dash, label="ngspice V(2)")
end
axislegend(ax_v; position=:rt, nbanks=2)
ylims!(ax_v, -0.2, 1.4)

# Panel [2,1]: VDD supply current (shoot-through indicator)
ax_i = Axis(fig2[2, 1]; xlabel="Time (ns)", ylabel="Current (mA)",
            title="I(VDD) Supply Current - shoot-through spikes at transitions")
if has_ivdd
    # Use raw solver time points for current (preserves transient features)
    mask_raw_ss = t_raw_ns .>= ss_start .&& t_raw_ns .<= ss_end
    lines!(ax_i, t_raw_ns[mask_raw_ss], I_vdd_raw[mask_raw_ss] .* 1e3;
           color=:darkgreen, linewidth=1.0, label="CedarSim I(VDD) [raw dt]")
    # Also show uniform-sampled version to reveal interpolation smoothing
    lines!(ax_i, t_cs_ns[mask_ss], I_vdd_uniform[mask_ss] .* 1e3;
           color=:green, linewidth=0.8, linestyle=:dash, label="CedarSim I(VDD) [interp]")
end
if has_ng_current
    mask_ng_ss_i = t_ng_ns .>= ss_start .&& t_ng_ns .<= ss_end
    lines!(ax_i, t_ng_ns[mask_ng_ss_i], I_vdd_ng[mask_ng_ss_i] .* 1e3;
           color=:purple, linewidth=1.0, label="ngspice I(VDD)")
end
axislegend(ax_i; position=:rt, nbanks=2)

# --- Right column: Zoomed single transition ---
# Find a rising edge on V(2) in steady state to zoom into
# V(2) rises when V(1) falls (inverter), so look for V(1) falling edge
v2_ss = V_cs[2][mask_ss]
t_ss = t_cs_ns[mask_ss]
# Find crossings of V(2) through 0.6V (mid-rail) going up
crossings = Int[]
for k in 2:length(v2_ss)
    if v2_ss[k-1] < 0.6 && v2_ss[k] >= 0.6
        push!(crossings, k)
    end
end

if length(crossings) >= 2
    # Pick a crossing in the middle of the window for the zoom
    cross_idx = crossings[div(length(crossings), 2)]
    t_cross = t_ss[cross_idx]
    zoom_half = 1.5  # ns half-window
    zoom_start, zoom_end = t_cross - zoom_half, t_cross + zoom_half
else
    # Fallback: zoom into middle of window
    zoom_start, zoom_end = 925.0, 928.0
end

# Panel [1,2]: Zoomed voltage transition
ax_vz = Axis(fig2[1, 2]; xlabel="Time (ns)", ylabel="Voltage (V)",
             title="Zoomed Transition: V(2) rising edge")
mask_zoom = t_cs_ns .>= zoom_start .&& t_cs_ns .<= zoom_end
lines!(ax_vz, t_cs_ns[mask_zoom], V_cs[1][mask_zoom]; color=:blue, linewidth=2.0, label="V(1)")
lines!(ax_vz, t_cs_ns[mask_zoom], V_cs[2][mask_zoom]; color=:red, linewidth=2.0, label="V(2)")
if has_ngspice
    mask_ng_zoom = t_ng_ns .>= zoom_start .&& t_ng_ns .<= zoom_end
    if any(mask_ng_zoom)
        lines!(ax_vz, t_ng_ns[mask_ng_zoom], V_ng[1][mask_ng_zoom]; color=:blue, linewidth=1.0,
               linestyle=:dash, label="ngspice V(1)")
        lines!(ax_vz, t_ng_ns[mask_ng_zoom], V_ng[2][mask_ng_zoom]; color=:red, linewidth=1.0,
               linestyle=:dash, label="ngspice V(2)")
    end
end
axislegend(ax_vz; position=:rt)
ylims!(ax_vz, -0.2, 1.4)

# Panel [2,2]: Zoomed current during transition
ax_iz = Axis(fig2[2, 2]; xlabel="Time (ns)", ylabel="Current (mA)",
             title="Zoomed I(VDD) During Transition")
if has_ivdd
    mask_raw_zoom = t_raw_ns .>= zoom_start .&& t_raw_ns .<= zoom_end
    if any(mask_raw_zoom)
        # Plot with markers to show actual solver time points
        scatter!(ax_iz, t_raw_ns[mask_raw_zoom], I_vdd_raw[mask_raw_zoom] .* 1e3;
                 color=:darkgreen, markersize=4, label="CedarSim [solver pts]")
        lines!(ax_iz, t_raw_ns[mask_raw_zoom], I_vdd_raw[mask_raw_zoom] .* 1e3;
               color=:darkgreen, linewidth=1.0)
    end
    lines!(ax_iz, t_cs_ns[mask_zoom], I_vdd_uniform[mask_zoom] .* 1e3;
           color=:green, linewidth=0.8, linestyle=:dash, label="CedarSim [interp]")
end
if has_ng_current
    mask_ng_zoom_i = t_ng_ns .>= zoom_start .&& t_ng_ns .<= zoom_end
    if any(mask_ng_zoom_i)
        lines!(ax_iz, t_ng_ns[mask_ng_zoom_i], I_vdd_ng[mask_ng_zoom_i] .* 1e3;
               color=:purple, linewidth=1.5, label="ngspice I(VDD)")
    end
end
axislegend(ax_iz; position=:rt)

# --- Bottom row: Solver timestep analysis ---
# Plot dt vs time to see if timestep is adapting properly near transitions
ax_dt = Axis(fig2[3, 1:2]; xlabel="Time (ns)", ylabel="Timestep (ps)",
             title="Solver Timestep (should refine near transitions, coarsen on rails)")
dt_raw = diff(t_raw) .* 1e12  # Convert to picoseconds
t_dt = t_raw_ns[1:end-1]
mask_dt_ss = t_dt .>= ss_start .&& t_dt .<= ss_end
lines!(ax_dt, t_dt[mask_dt_ss], dt_raw[mask_dt_ss]; color=:black, linewidth=0.8)
# Add V(2) scaled to show correlation with timestep changes
ax_dt_v = Axis(fig2[3, 1:2]; ylabel="V(2) (V)", yaxisposition=:right,
               yticklabelcolor=:red, ylabelcolor=:red)
lines!(ax_dt_v, t_cs_ns[mask_ss], V_cs[2][mask_ss]; color=(:red, 0.3), linewidth=1.0)
ylims!(ax_dt_v, -0.2, 1.4)
hidespines!(ax_dt_v)
hidexdecorations!(ax_dt_v)

outfile2 = joinpath(@__DIR__, "ring_oscillator_currents.png")
save(outfile2, fig2; px_per_unit=2)
println("Saved: $outfile2")

#==============================================================================#
# Print comparison stats
#==============================================================================#
println("\n=== Comparison Summary ===")
if has_ngspice
    @printf("ngspice:  1us, %d pts, trap method\n", length(t_ng))
end
@printf("CedarSim: 1us in %.1fs, %d pts, %d NR iters, FBDF\n",
    elapsed, length(sol.t),
    sol.stats !== nothing ? sol.stats.nnonliniter : 0)

# Measure oscillation period from V(1) zero crossings
v1_ss = V_cs[1][mask_cs3]
t1_ss = t_cs_ns[mask_cs3]
rising_crossings = Float64[]
for k in 2:length(v1_ss)
    if v1_ss[k-1] < 0.6 && v1_ss[k] >= 0.6
        # Linear interpolation for more precise crossing time
        frac = (0.6 - v1_ss[k-1]) / (v1_ss[k] - v1_ss[k-1])
        push!(rising_crossings, t1_ss[k-1] + frac * (t1_ss[k] - t1_ss[k-1]))
    end
end
if length(rising_crossings) >= 2
    periods = diff(rising_crossings)
    avg_period = sum(periods) / length(periods)
    freq_ghz = 1.0 / avg_period  # ns -> GHz
    @printf("\nCedarSim oscillation: period=%.3f ns, freq=%.3f GHz\n", avg_period, freq_ghz)
end

if has_ivdd
    # Current statistics
    mask_raw_full_ss = t_raw_ns .>= 800 .&& t_raw_ns .<= 1000
    ivdd_ss = I_vdd_raw[mask_raw_full_ss]
    @printf("\nI(VDD) steady-state stats:\n")
    @printf("  Mean: %.4f mA\n", _mean(ivdd_ss) * 1e3)
    @printf("  Peak: %.4f mA (shoot-through spikes)\n", maximum(abs.(ivdd_ss)) * 1e3)
    @printf("  Solver time points in [800,1000]ns: %d\n", sum(mask_raw_full_ss))
end

println("\nDone!")
