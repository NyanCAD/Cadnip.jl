#!/usr/bin/env julia
#==============================================================================#
# VACASK Stiffness Benchmark
#
# Measures stiffness metrics for each VACASK benchmark circuit:
# - Condition number of the Jacobian (G + γC where γ = 1/dt)
# - Stiffness ratio (max eigenvalue / min eigenvalue)
# - State variable magnitude ranges
#
# The condition number κ(J) indicates how sensitive the solution is to
# perturbations. High condition numbers indicate stiff systems.
#
# Usage:
#   julia --project=. benchmarks/vacask/stiffness_benchmark.jl
#==============================================================================#

using Pkg
Pkg.instantiate()

using Printf
using LinearAlgebra
using SparseArrays
using Statistics

using CedarSim
using CedarSim.MNA
using VerilogAParser

const BENCHMARK_DIR = @__DIR__

#==============================================================================#
# Stiffness Metrics
#==============================================================================#

"""
    StiffnessMetrics

Container for stiffness analysis results.
"""
struct StiffnessMetrics
    name::String
    n_nodes::Int
    n_currents::Int
    n_charges::Int
    system_size::Int

    # Condition numbers at different timesteps
    cond_dt_1ns::Float64      # γ = 1e9 (dt = 1ns)
    cond_dt_1us::Float64      # γ = 1e6 (dt = 1μs)
    cond_dt_1ms::Float64      # γ = 1e3 (dt = 1ms)

    # Eigenvalue analysis
    max_eigenvalue::Float64
    min_eigenvalue::Float64
    stiffness_ratio::Float64

    # State variable magnitude analysis
    voltage_range::Tuple{Float64, Float64}
    current_range::Tuple{Float64, Float64}
    charge_range::Tuple{Float64, Float64}

    status::Symbol  # :success, :failed
    error_msg::String
end

StiffnessMetrics(name::String, status::Symbol, error_msg::String) =
    StiffnessMetrics(name, 0, 0, 0, 0, NaN, NaN, NaN, NaN, NaN, NaN,
                     (NaN, NaN), (NaN, NaN), (NaN, NaN), status, error_msg)

"""
    compute_condition_number(G, C, gamma) -> Float64

Compute condition number of the Jacobian J = G + γC.
"""
function compute_condition_number(G::SparseMatrixCSC, C::SparseMatrixCSC, gamma::Float64)
    J = Matrix(G + gamma * C)
    n = size(J, 1)

    # Add small regularization to avoid singular matrix
    for i in 1:n
        if abs(J[i, i]) < 1e-30
            J[i, i] += 1e-15
        end
    end

    return cond(J)
end

"""
    compute_stiffness_ratio(G, C, gamma) -> (max_eig, min_eig, ratio)

Compute stiffness ratio from eigenvalue analysis of the Jacobian.
"""
function compute_stiffness_ratio(G::SparseMatrixCSC, C::SparseMatrixCSC, gamma::Float64)
    J = Matrix(G + gamma * C)
    n = size(J, 1)

    # Add regularization
    for i in 1:n
        if abs(J[i, i]) < 1e-30
            J[i, i] += 1e-15
        end
    end

    eigs = eigvals(J)
    abs_eigs = abs.(eigs)
    nonzero_eigs = filter(e -> e > 1e-30, abs_eigs)

    if isempty(nonzero_eigs)
        return (NaN, NaN, NaN)
    end

    max_eig = maximum(nonzero_eigs)
    min_eig = minimum(nonzero_eigs)
    ratio = max_eig / min_eig

    return (max_eig, min_eig, ratio)
end

"""
    analyze_state_magnitudes(x, n_nodes, n_currents, n_charges)

Analyze the magnitude range of different state variable types.
"""
function analyze_state_magnitudes(x::Vector{Float64}, n_nodes::Int, n_currents::Int, n_charges::Int)
    n = length(x)

    # Voltage states (1:n_nodes)
    if n_nodes > 0 && n >= n_nodes
        voltages = x[1:n_nodes]
        nonzero_v = filter(v -> abs(v) > 1e-30, voltages)
        if !isempty(nonzero_v)
            voltage_range = (minimum(abs.(nonzero_v)), maximum(abs.(nonzero_v)))
        else
            voltage_range = (0.0, 0.0)
        end
    else
        voltage_range = (NaN, NaN)
    end

    # Current states (n_nodes+1:n_nodes+n_currents)
    if n_currents > 0 && n >= n_nodes + n_currents
        currents = x[n_nodes+1:n_nodes+n_currents]
        nonzero_i = filter(i -> abs(i) > 1e-30, currents)
        if !isempty(nonzero_i)
            current_range = (minimum(abs.(nonzero_i)), maximum(abs.(nonzero_i)))
        else
            current_range = (0.0, 0.0)
        end
    else
        current_range = (NaN, NaN)
    end

    # Charge states (n_nodes+n_currents+1:end)
    if n_charges > 0 && n >= n_nodes + n_currents + n_charges
        charges = x[n_nodes+n_currents+1:n_nodes+n_currents+n_charges]
        nonzero_q = filter(q -> abs(q) > 1e-30, charges)
        if !isempty(nonzero_q)
            charge_range = (minimum(abs.(nonzero_q)), maximum(abs.(nonzero_q)))
        else
            charge_range = (0.0, 0.0)
        end
    else
        charge_range = (NaN, NaN)
    end

    return (voltage_range, current_range, charge_range)
end

"""
    analyze_circuit_stiffness(name, G, C, n_nodes, n_currents, n_charges, u0) -> StiffnessMetrics

Analyze stiffness metrics from precomputed matrices.
"""
function analyze_circuit_stiffness(name::String, G::SparseMatrixCSC, C::SparseMatrixCSC,
                                   n_nodes::Int, n_currents::Int, n_charges::Int, u0::Vector{Float64})
    try
        n = size(G, 1)

        # Compute condition numbers at different timesteps
        cond_1ns = compute_condition_number(G, C, 1e9)
        cond_1us = compute_condition_number(G, C, 1e6)
        cond_1ms = compute_condition_number(G, C, 1e3)

        # Compute stiffness ratio (using typical transient γ = 1e6)
        max_eig, min_eig, ratio = compute_stiffness_ratio(G, C, 1e6)

        # Analyze state magnitudes
        v_range, i_range, q_range = analyze_state_magnitudes(u0, n_nodes, n_currents, n_charges)

        return StiffnessMetrics(
            name, n_nodes, n_currents, n_charges, n,
            cond_1ns, cond_1us, cond_1ms,
            max_eig, min_eig, ratio,
            v_range, i_range, q_range,
            :success, ""
        )
    catch e
        return StiffnessMetrics(name, :failed, sprint(showerror, e))
    end
end

#==============================================================================#
# Report Generation
#==============================================================================#

function format_scientific(x::Float64)
    if isnan(x) || isinf(x)
        return "-"
    end
    return @sprintf("%.2e", x)
end

function format_range(r::Tuple{Float64, Float64})
    if isnan(r[1]) || isnan(r[2])
        return "-"
    end
    return "[$(format_scientific(r[1])), $(format_scientific(r[2]))]"
end

function generate_report(results::Vector{StiffnessMetrics})
    io = IOBuffer()

    println(io, "# VACASK Stiffness Analysis Report")
    println(io)
    println(io, "Analyzed on Julia $(VERSION)")
    println(io)

    # Summary table
    println(io, "## Condition Number Summary")
    println(io)
    println(io, "The condition number κ(J) of the Jacobian J = G + γC measures system stiffness.")
    println(io, "Higher values indicate more difficult numerical integration.")
    println(io)
    println(io, "| Benchmark | Size | κ(dt=1ns) | κ(dt=1μs) | κ(dt=1ms) | Stiffness Ratio |")
    println(io, "|-----------|------|-----------|-----------|-----------|-----------------|")

    for r in results
        if r.status == :success
            println(io, "| $(r.name) | $(r.system_size) | $(format_scientific(r.cond_dt_1ns)) | $(format_scientific(r.cond_dt_1us)) | $(format_scientific(r.cond_dt_1ms)) | $(format_scientific(r.stiffness_ratio)) |")
        else
            println(io, "| $(r.name) | - | Failed | Failed | Failed | - |")
        end
    end
    println(io)

    # Detailed results
    println(io, "## Detailed Results")
    println(io)

    for r in results
        println(io, "### $(r.name)")
        println(io)

        if r.status == :success
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            println(io, "| System Size | $(r.system_size) |")
            println(io, "| Voltage Nodes | $(r.n_nodes) |")
            println(io, "| Current Variables | $(r.n_currents) |")
            println(io, "| Charge Variables | $(r.n_charges) |")
            println(io, "| κ(J) at dt=1ns | $(format_scientific(r.cond_dt_1ns)) |")
            println(io, "| κ(J) at dt=1μs | $(format_scientific(r.cond_dt_1us)) |")
            println(io, "| κ(J) at dt=1ms | $(format_scientific(r.cond_dt_1ms)) |")
            println(io, "| Max Eigenvalue | $(format_scientific(r.max_eigenvalue)) |")
            println(io, "| Min Eigenvalue | $(format_scientific(r.min_eigenvalue)) |")
            println(io, "| Stiffness Ratio | $(format_scientific(r.stiffness_ratio)) |")
            println(io, "| Voltage Range | $(format_range(r.voltage_range)) |")
            println(io, "| Current Range | $(format_range(r.current_range)) |")
            println(io, "| Charge Range | $(format_range(r.charge_range)) |")
        else
            println(io, "> ❌ Analysis failed: $(r.error_msg)")
        end
        println(io)
    end

    # Interpretation
    println(io, "## Interpretation")
    println(io)
    println(io, "- **Condition Number**: κ(J) > 10^6 indicates a stiff system")
    println(io, "- **Stiffness Ratio**: λ_max/λ_min > 10^6 requires implicit solvers")
    println(io, "- **State Magnitude Spread**: Large differences between voltage/current/charge")
    println(io, "  magnitudes contribute to poor conditioning")
    println(io)
    println(io, "### Recommendations")
    println(io)
    println(io, "If the system is stiff due to charge state magnitude differences:")
    println(io, "1. **Charge Scaling**: Scale charge variables to match voltage/current magnitudes")
    println(io, "2. **State Normalization**: Use scaled state variables in the DAE formulation")
    println(io, "3. **Preconditioner**: Apply diagonal scaling preconditioner")
    println(io)

    return String(take!(io))
end

#==============================================================================#
# Load and Analyze Circuits
#==============================================================================#

function analyze_rc_circuit()
    println("Analyzing RC circuit...")

    spice_file = joinpath(BENCHMARK_DIR, "rc", "cedarsim", "runme.sp")
    spice_code = read(spice_file, String)

    circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit_stiff)
    eval(circuit_code)

    circuit = Base.invokelatest(MNACircuit, rc_circuit_stiff)
    u0, du0 = Base.invokelatest(MNA.compute_initial_conditions, circuit)

    # Get matrices from compiled structure
    cs = Base.invokelatest(MNA.compile_structure, circuit.builder, circuit.params, circuit.spec)

    return analyze_circuit_stiffness(
        "RC Circuit", cs.G, cs.C, cs.n_nodes, cs.n_currents, 0, u0
    )
end

function analyze_graetz_circuit()
    println("Analyzing Graetz Bridge circuit...")

    # Load diode model
    diode_va_path = joinpath(BENCHMARK_DIR, "..", "..", "test", "vadistiller", "models", "diode.va")
    va = VerilogAParser.parsefile(diode_va_path)
    eval(CedarSim.make_mna_module(va))

    spice_file = joinpath(BENCHMARK_DIR, "graetz", "cedarsim", "runme.sp")
    spice_code = read(spice_file, String)

    circuit_code = parse_spice_to_mna(spice_code; circuit_name=:graetz_circuit_stiff,
                                       imported_hdl_modules=[sp_diode_module])
    eval(circuit_code)

    circuit = Base.invokelatest(MNACircuit, graetz_circuit_stiff)
    u0, du0 = Base.invokelatest(MNA.compute_initial_conditions, circuit)

    cs = Base.invokelatest(MNA.compile_structure, circuit.builder, circuit.params, circuit.spec)

    return analyze_circuit_stiffness(
        "Graetz Bridge", cs.G, cs.C, cs.n_nodes, cs.n_currents, 0, u0
    )
end

function analyze_mul_circuit()
    println("Analyzing Voltage Multiplier circuit...")

    # Load diode model
    diode_va_path = joinpath(BENCHMARK_DIR, "..", "..", "test", "vadistiller", "models", "diode.va")
    va = VerilogAParser.parsefile(diode_va_path)
    eval(CedarSim.make_mna_module(va))

    spice_file = joinpath(BENCHMARK_DIR, "mul", "cedarsim", "runme.sp")
    spice_code = read(spice_file, String)

    circuit_code = parse_spice_to_mna(spice_code; circuit_name=:mul_circuit_stiff,
                                       imported_hdl_modules=[sp_diode_module])
    eval(circuit_code)

    circuit = Base.invokelatest(MNACircuit, mul_circuit_stiff)
    u0, du0 = Base.invokelatest(MNA.compute_initial_conditions, circuit)

    cs = Base.invokelatest(MNA.compile_structure, circuit.builder, circuit.params, circuit.spec)

    return analyze_circuit_stiffness(
        "Voltage Multiplier", cs.G, cs.C, cs.n_nodes, cs.n_currents, 0, u0
    )
end

function analyze_ring_circuit()
    println("Analyzing Ring Oscillator circuit...")

    # Load PSP model
    psp103_path = joinpath(BENCHMARK_DIR, "..", "..", "test", "vadistiller", "models", "psp103v4", "psp103.va")
    va = VerilogAParser.parsefile(psp103_path)
    eval(CedarSim.make_mna_module(va))

    spice_file = joinpath(BENCHMARK_DIR, "ring", "cedarsim", "runme.sp")
    spice_code = read(spice_file, String)

    circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit_stiff,
                                       imported_hdl_modules=[PSP103VA_module])
    eval(circuit_code)

    circuit = Base.invokelatest(MNACircuit, ring_circuit_stiff)
    u0, du0 = Base.invokelatest(MNA.compute_initial_conditions, circuit)

    cs = Base.invokelatest(MNA.compile_structure, circuit.builder, circuit.params, circuit.spec)

    # Ring oscillator likely has charge states from MOSFET capacitances
    n_charges = length(u0) - cs.n_nodes - cs.n_currents

    return analyze_circuit_stiffness(
        "Ring Oscillator", cs.G, cs.C, cs.n_nodes, cs.n_currents, max(0, n_charges), u0
    )
end

function analyze_c6288_circuit()
    println("Analyzing C6288 Multiplier circuit...")

    # Load PSP model
    psp103_path = joinpath(BENCHMARK_DIR, "..", "..", "test", "vadistiller", "models", "psp103v4", "psp103.va")
    va = VerilogAParser.parsefile(psp103_path)
    eval(CedarSim.make_mna_module(va))

    spice_file = joinpath(BENCHMARK_DIR, "c6288", "cedarsim", "runme.sp")
    spice_dir = dirname(spice_file)

    # Change to SPICE directory for includes to resolve
    orig_dir = pwd()
    cd(spice_dir)
    try
        spice_code = read(spice_file, String)

        circuit_code = parse_spice_to_mna(spice_code; circuit_name=:c6288_circuit_stiff,
                                           imported_hdl_modules=[PSP103VA_module])
        eval(circuit_code)
    finally
        cd(orig_dir)
    end

    circuit = Base.invokelatest(MNACircuit, c6288_circuit_stiff)
    u0, du0 = Base.invokelatest(MNA.compute_initial_conditions, circuit)

    cs = Base.invokelatest(MNA.compile_structure, circuit.builder, circuit.params, circuit.spec)

    n_charges = length(u0) - cs.n_nodes - cs.n_currents

    return analyze_circuit_stiffness(
        "C6288 Multiplier", cs.G, cs.C, cs.n_nodes, cs.n_currents, max(0, n_charges), u0
    )
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    println("=" ^ 60)
    println("VACASK Stiffness Analysis")
    println("=" ^ 60)
    println()

    results = StiffnessMetrics[]

    # RC Circuit
    try
        push!(results, analyze_rc_circuit())
    catch e
        push!(results, StiffnessMetrics("RC Circuit", :failed, sprint(showerror, e)))
    end

    # Graetz Bridge
    try
        push!(results, analyze_graetz_circuit())
    catch e
        push!(results, StiffnessMetrics("Graetz Bridge", :failed, sprint(showerror, e)))
    end

    # Voltage Multiplier
    try
        push!(results, analyze_mul_circuit())
    catch e
        push!(results, StiffnessMetrics("Voltage Multiplier", :failed, sprint(showerror, e)))
    end

    # Ring Oscillator
    try
        push!(results, analyze_ring_circuit())
    catch e
        push!(results, StiffnessMetrics("Ring Oscillator", :failed, sprint(showerror, e)))
    end

    # C6288 Multiplier (skip if PSP model takes too long)
    try
        push!(results, analyze_c6288_circuit())
    catch e
        push!(results, StiffnessMetrics("C6288 Multiplier", :failed, sprint(showerror, e)))
    end

    println()
    println("=" ^ 60)
    println("Generating stiffness report...")
    println("=" ^ 60)

    report = generate_report(results)

    # Write to file
    output_file = joinpath(BENCHMARK_DIR, "stiffness_report.md")
    open(output_file, "w") do f
        write(f, report)
    end
    println("Report written to: $output_file")

    # Also print to stdout
    println()
    println(report)

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
