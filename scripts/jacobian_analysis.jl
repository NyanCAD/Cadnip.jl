#!/usr/bin/env julia
#==============================================================================#
# Jacobian Analysis Script for MNA Circuits
#
# Analyzes the condition number and identifies problematic states
# in the MNA Jacobian matrix.
#
# Since the VA models have precompilation issues, we use built-in MNA devices
# to demonstrate the analysis methodology.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using CedarSim.MNA
using CedarSim.MNA: MNACircuit, MNASpec, build_with_detection, assemble!, system_size
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!
using CedarSim.MNA: MNAContext, get_node!, stamp!, ZERO_VECTOR, reset_for_restamping!
using CedarSim.MNA: VoltageSource, Resistor, Capacitor, Diode
using LinearAlgebra
using SparseArrays
using Printf

println("="^70)
println("Jacobian Analysis: Simple Circuit with Diode")
println("="^70)
println()

# Build a simple circuit with nonlinear devices
# This is a basic RC circuit with a diode to create a nonlinear Jacobian
function simple_nonlinear_circuit(params, spec, t::Real=0.0; x=ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end

    # Get nodes
    vcc = get_node!(ctx, :vcc)
    node1 = get_node!(ctx, :node1)
    node2 = get_node!(ctx, :node2)
    node3 = get_node!(ctx, :node3)

    # Get operating point voltages for nonlinear devices
    v1 = length(x) >= 2 ? x[2] : 0.0  # node1 voltage
    v2 = length(x) >= 3 ? x[3] : 0.0  # node2 voltage
    v3 = length(x) >= 4 ? x[4] : 0.0  # node3 voltage

    # Voltage source
    stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)

    # Resistors
    stamp!(Resistor(params.R1), ctx, vcc, node1)
    stamp!(Resistor(params.R2), ctx, node1, node2)
    stamp!(Resistor(params.R3), ctx, node2, node3)
    stamp!(Resistor(params.R4), ctx, node3, 0)

    # Capacitors - small values typical of circuits
    stamp!(Capacitor(params.C1), ctx, node1, 0)  # 10 fF
    stamp!(Capacitor(params.C2), ctx, node2, 0)  # 10 fF
    stamp!(Capacitor(params.C3), ctx, node3, 0)  # 10 fF

    # Diodes (nonlinear devices)
    stamp!(Diode(Is=params.Is, Vt=params.Vt), ctx, node1, 0; x=x)
    stamp!(Diode(Is=params.Is, Vt=params.Vt), ctx, node2, 0; x=x)

    return ctx
end

# Circuit parameters
params = (
    Vcc = 3.3,        # Supply voltage
    R1 = 1000.0,      # 1 kΩ
    R2 = 1000.0,      # 1 kΩ
    R3 = 1000.0,      # 1 kΩ
    R4 = 1000.0,      # 1 kΩ
    C1 = 10e-15,      # 10 fF (typical gate capacitance)
    C2 = 10e-15,      # 10 fF
    C3 = 10e-15,      # 10 fF
    Is = 1e-14,       # Diode saturation current
    Vt = 0.026,       # Thermal voltage
)

# Create circuit with DC operating point mode
spec = MNASpec(mode=:dcop)
circuit = MNACircuit(simple_nonlinear_circuit, params, spec)

println("Building circuit with structure detection...")
ctx = build_with_detection(circuit)
sys = assemble!(ctx)

n = system_size(sys)
println("\n=== System Size ===")
println("  Total unknowns: $n")
println("  Node voltages:  $(sys.n_nodes)")
println("  Branch currents: $(length(sys.current_names))")

# Get matrices
G = sys.G
C = sys.C
b = sys.b

println("\n=== Matrix Statistics ===")
println("  G matrix: $(nnz(G)) nonzeros")
println("  C matrix: $(nnz(C)) nonzeros ($(count(x -> abs(x) > 1e-30, nonzeros(C))) actual nonzeros)")

# Check for zero diagonal entries
G_diag = diag(G)
zero_diag_count = count(x -> abs(x) < 1e-15, G_diag)
println("  G diagonal zeros: $zero_diag_count")

# Compute condition number
println("\n=== Condition Number Analysis ===")

# Dense conversion for analysis
G_dense = Matrix(G)
C_dense = Matrix(C)

# Check if G is singular
try
    lu_G = lu(G_dense; check=false)
    if issuccess(lu_G)
        println("  G matrix: LU factorization succeeded")
        cond_G = cond(G_dense)
        @printf("  Condition number of G: %.2e\n", cond_G)
    else
        println("  G matrix: SINGULAR (LU factorization failed)")

        # Try with GMIN regularization
        gmin = 1e-12
        G_reg = G_dense + gmin * I
        cond_G_reg = cond(G_reg)
        @printf("  Condition number with GMIN=1e-12: %.2e\n", cond_G_reg)
    end
catch e
    println("  Error during factorization: $e")
end

# Analyze row/column norms to identify problematic states
println("\n=== Row/Column Scaling Analysis ===")

row_norms = [norm(G_dense[i, :]) for i in 1:n]
col_norms = [norm(G_dense[:, j]) for j in 1:n]

# Find extreme rows
sorted_row_idx = sortperm(row_norms)
println("\n  Smallest row norms (most singular):")
for i in 1:min(5, n)
    idx = sorted_row_idx[i]
    name = idx <= sys.n_nodes ? string(sys.node_names[idx]) : sys.current_names[idx - sys.n_nodes]
    @printf("    Row %3d (%-20s): %.2e\n", idx, name, row_norms[idx])
end

println("\n  Largest row norms:")
for i in max(1, n-4):n
    idx = sorted_row_idx[i]
    name = idx <= sys.n_nodes ? string(sys.node_names[idx]) : sys.current_names[idx - sys.n_nodes]
    @printf("    Row %3d (%-20s): %.2e\n", idx, name, row_norms[idx])
end

# Analyze C matrix for charge states
println("\n=== Charge State Analysis (C matrix) ===")
C_row_norms = [norm(C_dense[i, :]) for i in 1:n]
nonzero_C_rows = findall(x -> x > 1e-30, C_row_norms)

println("  Variables with C row entries (differential states):")
for idx in nonzero_C_rows
    name = idx <= sys.n_nodes ? string(sys.node_names[idx]) : sys.current_names[idx - sys.n_nodes]
    c_mag = C_row_norms[idx]
    g_mag = row_norms[idx]
    ratio = c_mag > 0 ? g_mag / c_mag : Inf
    @printf("    Row %3d (%-20s): |C|=%.2e, |G|=%.2e, |G|/|C|=%.2e\n", idx, name, c_mag, g_mag, ratio)
end

# Check for charge scaling issues
println("\n=== Scaling Issues ===")

# Look for rows where G and C have very different magnitudes
scaling_issues = []
for i in 1:n
    if C_row_norms[i] > 1e-30 && row_norms[i] > 1e-30
        ratio = row_norms[i] / C_row_norms[i]
        if ratio > 1e6 || ratio < 1e-6
            name = i <= sys.n_nodes ? string(sys.node_names[i]) : sys.current_names[i - sys.n_nodes]
            push!(scaling_issues, (i, name, row_norms[i], C_row_norms[i], ratio))
        end
    end
end

if isempty(scaling_issues)
    println("  No severe scaling issues found (all |G|/|C| ratios within 1e-6 to 1e6)")
else
    println("  Rows with scaling issues (|G|/|C| outside 1e-6 to 1e6):")
    for (idx, name, g_norm, c_norm, ratio) in scaling_issues
        @printf("    Row %3d (%-20s): |G|=%.2e, |C|=%.2e, ratio=%.2e\n", idx, name, g_norm, c_norm, ratio)
    end
end

# Analyze the source vector
println("\n=== Source Vector Analysis ===")
b_nonzero = findall(x -> abs(x) > 1e-15, b)
println("  Nonzero entries in b vector: $(length(b_nonzero))")
for idx in b_nonzero
    name = idx <= sys.n_nodes ? string(sys.node_names[idx]) : sys.current_names[idx - sys.n_nodes]
    @printf("    b[%3d] (%-20s): %.4e\n", idx, name, b[idx])
end

# Try to do a DC solve and analyze residual
println("\n=== DC Solve Attempt ===")
using NonlinearSolve

# Compile for fast evaluation
cs = compile_structure(circuit.builder, circuit.params, MNASpec(mode=:dcop); ctx=ctx)
ws = create_workspace(cs; ctx=ctx)

# Try linear solve first (at u=0)
fast_rebuild!(ws, zeros(n), 0.0)
G_lin = cs.G
b_lin = ws.dctx.b

try
    u_linear = G_lin \ b_lin
    resid_linear = G_lin * u_linear - b_lin
    @printf("  Linear solve (at u=0):\n")
    @printf("    Max |u|: %.4e\n", maximum(abs.(u_linear)))
    @printf("    Max |residual|: %.4e\n", maximum(abs.(resid_linear)))

    # Show key voltages
    vdd_idx = findfirst(==(:vdd), sys.node_names)
    if vdd_idx !== nothing
        @printf("    V(vdd): %.4f V\n", u_linear[vdd_idx])
    end
catch e
    println("  Linear solve failed: $e")
end

# Now try Newton solve
println("\n  Attempting Newton solve...")

function residual!(F, u, p)
    fast_rebuild!(ws, cs, u, 0.0)
    mul!(F, cs.G, u)
    F .-= ws.dctx.b
    return nothing
end

function jacobian!(J, u, p)
    fast_rebuild!(ws, cs, u, 0.0)
    copyto!(J, cs.G)
    return nothing
end

try
    nlfunc = NonlinearFunction(residual!; jac=jacobian!, jac_prototype=cs.G)
    nlprob = NonlinearProblem(nlfunc, zeros(n))

    # Try LevenbergMarquardt (has GMIN-like regularization)
    sol = solve(nlprob, LevenbergMarquardt(); abstol=1e-8, maxiters=200)

    @printf("  Newton solve result: %s\n", sol.retcode)
    @printf("  Max |u|: %.4e\n", maximum(abs.(sol.u)))

    # Compute final residual
    resid = zeros(n)
    residual!(resid, sol.u, nothing)
    @printf("  Max |residual|: %.4e\n", maximum(abs.(resid)))

    # Show key voltages
    vdd_idx = findfirst(==(:vdd), sys.node_names)
    if vdd_idx !== nothing
        @printf("  V(vdd): %.4f V\n", sol.u[vdd_idx])
    end

    # Analyze Jacobian at solution point
    println("\n=== Jacobian at Solution Point ===")
    fast_rebuild!(ws, cs, sol.u, 0.0)
    G_sol = Matrix(cs.G)

    cond_G_sol = cond(G_sol)
    @printf("  Condition number at solution: %.2e\n", cond_G_sol)

    # SVD analysis
    svd_G = svd(G_sol)
    println("\n  Singular values:")
    @printf("    Max σ: %.2e\n", svd_G.S[1])
    @printf("    Min σ: %.2e\n", svd_G.S[end])
    @printf("    Ratio: %.2e\n", svd_G.S[1] / svd_G.S[end])

    # Show smallest singular values and their corresponding states
    println("\n  Smallest singular values and associated states:")
    for i in max(1, n-4):n
        idx = sortperm(svd_G.S)[i-max(0, n-5)]
        σ = svd_G.S[idx]
        # Right singular vector shows which unknowns are associated
        v_max_idx = argmax(abs.(svd_G.V[:, idx]))
        name = v_max_idx <= sys.n_nodes ? string(sys.node_names[v_max_idx]) : sys.current_names[v_max_idx - sys.n_nodes]
        @printf("    σ[%d] = %.2e (associated with: %s)\n", idx, σ, name)
    end

catch e
    println("  Newton solve failed: $e")
    showerror(stdout, e, catch_backtrace())
end

println("\n" * "="^70)
println("Analysis complete")
println("="^70)
