#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Ring Oscillator with PSP103 MOSFETs
#
# 9-stage ring oscillator using PSP103 MOSFET model.
#
# Benchmark target: ~20,000-30,000 timepoints (approximately matching VACASK)
# VACASK reference: 26,066 timepoints, 81,875 iterations, 1.18s (trapezoidal)
# Ngspice reference: 20,556 timepoints, 80,018 iterations, 1.60s
#
# Ring oscillators have no stable DC operating point, so we bypass DC analysis
# by providing explicit initial conditions. The ring nodes are set with
# alternating high/low voltages to break symmetry and kick-start oscillation.
# PSP103 internal nodes (SI, DI, BP, BI, BS, BD) are initialized based on
# transistor type (PMOS tied to vdd, NMOS tied to ground).
#
# Solver options:
# - IDA (DAE): Uses explicit Jacobian, reaches ~0.6μs before convergence failure
# - Rodas5P (ODE): Completes full 1μs but slower (~15 min for full simulation)
#
# The IDA convergence failure occurs around t=0.589μs when the timestep becomes
# extremely small (h ~ 1e-84). Rodas5P avoids this by using a mass matrix
# formulation instead of the full DAE.
#==============================================================================#

using CedarSim
using CedarSim.MNA
using Sundials
using OrdinaryDiffEq
using SciMLBase
using BenchmarkTools
using Printf
using VerilogAParser
using DiffEqBase: BrownFullBasicInit, NoInit

# Load the PSP103 model
const psp103_path = joinpath(@__DIR__, "..", "..", "..", "..", "test", "vadistiller", "models", "psp103v4", "psp103.va")

# Parse and eval the PSP103 model
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
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
# Pass PSP103VA_module so the SPICE parser knows about our VA device
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

"""
    get_circuit_info()

Get circuit size and node names without triggering DC solve.
Returns (n, node_names, current_names) where n is total system size.
"""
function get_circuit_info()
    circuit = MNACircuit(ring_circuit)
    spec = MNASpec(temp=circuit.spec.temp, mode=:tran, time=0.0)
    ctx = circuit.builder(circuit.params, spec, 0.0; x=MNA.ZERO_VECTOR)
    n = MNA.system_size(ctx)
    return n, ctx.node_names, ctx.current_names
end

"""
    get_initial_conditions(n, node_names, current_names)

Create initial conditions for ring oscillator that bypass DC analysis.
- vdd = 1.2V (supply voltage)
- Ring nodes: alternating 0.9V/0.3V to break symmetry and kick-start oscillation
- PMOS internal nodes (SI, BS, BP, BI): tied to vdd (1.2V)
- NMOS internal nodes (SI, BS, BP, BI): tied to ground (0V)
- Other internal nodes and currents: mid-rail (0.6V) or zero
"""
function get_initial_conditions(n, node_names, current_names)
    u0 = fill(0.6, n)  # Default to mid-rail
    du0 = zeros(n)

    for (i, name) in enumerate(node_names)
        sname = string(name)
        if name == :vdd
            u0[i] = 1.2  # Supply voltage
        elseif occursin(r"^[1-9]$", sname)
            # Ring stage outputs: alternate high/low to break symmetry
            ring_idx = parse(Int, sname)
            u0[i] = iseven(ring_idx) ? 0.9 : 0.3
        elseif occursin("_xmp_", sname)  # PMOS internal nodes
            if endswith(sname, "_SI") || endswith(sname, "_BS") ||
               endswith(sname, "_BP") || endswith(sname, "_BI")
                u0[i] = 1.2  # Tied to vdd
            end
        elseif occursin("_xmn_", sname)  # NMOS internal nodes
            if endswith(sname, "_SI") || endswith(sname, "_BS") ||
               endswith(sname, "_BP") || endswith(sname, "_BI")
                u0[i] = 0.0  # Tied to ground
            end
        end
    end

    # Set all current variables to zero
    n_nodes = length(node_names)
    for i in 1:length(current_names)
        u0[n_nodes + i] = 0.0
    end

    return u0, du0
end

"""
    run_benchmark(; solver=:ida, dtmax=0.05e-9, reltol=1e-2, abstol=1e-6, tspan_us=1.0)

Run the ring oscillator benchmark with the specified solver.

# Solver options:
- `:ida` - Sundials IDA (DAE solver with explicit Jacobian). Fast but hits
  convergence failure around t=0.589μs due to numerical instability.
- `:rodas5p` - OrdinaryDiffEq Rodas5P (ODE solver with mass matrix). Completes
  full simulation but slower (~15 min for 1μs).

# Arguments
- `solver`: `:ida` or `:rodas5p`
- `dtmax`: Maximum timestep (default 0.05ns to match VACASK)
- `reltol`: Relative tolerance (default 1e-2)
- `abstol`: Absolute tolerance (default 1e-6)
- `tspan_us`: Simulation duration in microseconds (default 1.0)
"""
function run_benchmark(; solver=:ida, dtmax=0.05e-9, reltol=1e-2, abstol=1e-6, tspan_us=1.0)
    tspan = (0.0, tspan_us * 1e-6)

    # Get circuit size and create initial conditions (bypasses DC solve)
    n, node_names, current_names = get_circuit_info()
    u0, du0 = get_initial_conditions(n, node_names, current_names)

    # Create circuit
    circuit = MNACircuit(ring_circuit)

    if solver == :ida
        # Use Sundials IDA (DAE solver with explicit Jacobian)
        # Higher iteration limits help with convergence on this stiff oscillator
        ida_solver = IDA(max_nonlinear_iters=200, max_error_test_failures=50, max_convergence_failures=100)
        prob = SciMLBase.DAEProblem(circuit, tspan; u0=u0, du0=du0)
        init = BrownFullBasicInit(abstol=1e-2)

        println("\nBenchmarking transient analysis with IDA (DAE)...")
        println("  dtmax=$dtmax, reltol=$reltol, abstol=$abstol")
        println("Running simulation...")

        t1 = time()
        sol = SciMLBase.solve(prob, ida_solver; dtmax=dtmax, reltol=reltol, abstol=abstol, initializealg=init)
        t2 = time()

    elseif solver == :rodas5p
        # Use OrdinaryDiffEq Rodas5P (ODE solver with mass matrix)
        # Slower but completes where IDA fails
        prob = SciMLBase.ODEProblem(circuit, tspan; u0=u0)

        println("\nBenchmarking transient analysis with Rodas5P (ODE)...")
        println("  dtmax=$dtmax, reltol=$reltol, abstol=$abstol")
        println("Running simulation (this may take ~15 minutes for 1μs)...")

        t1 = time()
        sol = SciMLBase.solve(prob, Rodas5P(); dtmax=dtmax, reltol=reltol, abstol=abstol,
                              initializealg=NoInit(), maxiters=10000000)
        t2 = time()
    else
        error("Unknown solver: $solver. Use :ida or :rodas5p")
    end

    # VACASK reference: 26,066 timepoints, 81,875 iterations
    # Ngspice reference: 20,556 timepoints, 80,018 iterations
    println("\n=== Results ===")
    completed = sol.t[end] >= 0.99 * tspan[2]
    println("Completed: $(completed ? "YES ✓" : "NO ✗") (final t = $(round(sol.t[end]*1e6, digits=3))μs)")
    @printf("Timepoints: %d (VACASK: 26,066, Ngspice: 20,556)\n", length(sol.t))
    if hasproperty(sol.stats, :nnonliniter)
        @printf("NR iters:   %d (VACASK: 81,875, Ngspice: 80,018)\n", sol.stats.nnonliniter)
        @printf("Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    end
    @printf("Wall time:  %.1fs\n", t2-t1)
    @printf("retcode:    %s\n", sol.retcode)
    println()

    return sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
