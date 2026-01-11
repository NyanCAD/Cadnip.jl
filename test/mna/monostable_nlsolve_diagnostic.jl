#==============================================================================#
# Detailed diagnostic for sp_bjt monostable initialization failure
#
# Traces what happens inside the nonlinear solver to understand why
# RobustMultiNewton/LevenbergMarquardt/PseudoTransient are failing.
#
# Run with: julia --project=test test/mna/monostable_nlsolve_diagnostic.jl
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNACircuit, MNASolutionAccessor, MNASpec, MNAContext
using CedarSim.MNA: voltage, assemble!, CedarDCOp, CedarRobustNLSolve
using CedarSim.MNA: build_with_detection, compile_structure, create_workspace, fast_rebuild!
using VerilogAParser
using SciMLBase
using SciMLBase: solve
using CedarSim: tran!, parse_spice_to_mna
using LinearAlgebra
using SparseArrays

# NonlinearSolve is a dependency of CedarSim but not re-exported
# Use invokelatest to access from Base.loaded_modules or just skip the individual algorithm tests

#==============================================================================#
# Load sp_bjt model
#==============================================================================#

println("Loading sp_bjt model...")
const bjt_path = joinpath(@__DIR__, "..", "..", "models", "VADistillerModels.jl", "va", "bjt.va")
const bjt_va = VerilogAParser.parsefile(bjt_path)
if bjt_va.ps.errored
    error("Failed to parse bjt.va")
end
Core.eval(@__MODULE__, CedarSim.make_mna_module(bjt_va))

#==============================================================================#
# Load SPICE netlist
#==============================================================================#

const spice_path = joinpath(@__DIR__, "..", "vadistiller", "circuits", "monostable_multivibrator.spice")
const spice_code = read(spice_path, String)

const monostable_code = parse_spice_to_mna(spice_code;
    circuit_name=:monostable_multivibrator,
    imported_hdl_modules=[sp_bjt_module])
eval(monostable_code)

#==============================================================================#
# Build circuit and get workspace
#==============================================================================#

println("\n" * "="^60)
println("Building circuit...")
println("="^60)

circuit = MNACircuit(monostable_multivibrator)
ctx = build_with_detection(circuit)
sys = assemble!(ctx)

n = MNA.system_size(sys)
println("System size: $n")

spec = MNASpec(mode=:dcop)
cs = compile_structure(circuit.builder, circuit.params, spec; ctx=ctx)
ws = create_workspace(cs; ctx=ctx)

#==============================================================================#
# Test the residual and Jacobian at different points
#==============================================================================#

println("\n" * "="^60)
println("Residual and Jacobian Analysis")
println("="^60)

function analyze_point(name, u)
    println("\n--- $name ---")
    println("u range: $(extrema(u))")

    # Rebuild at this point
    fast_rebuild!(ws, cs, u, 0.0)

    # Get G and b
    G = cs.G
    b = ws.dctx.b

    # Compute residual: F = G*u - b
    resid = G * u - b

    println("G diagonal range: $(extrema(diag(G)))")
    println("b range: $(extrema(b))")
    println("residual norm: $(norm(resid))")
    println("residual range: $(extrema(resid))")

    # Check for NaN/Inf in G
    g_vals = nonzeros(G)
    nan_count = count(isnan, g_vals)
    inf_count = count(isinf, g_vals)
    if nan_count > 0 || inf_count > 0
        println("WARNING: G has $nan_count NaN and $inf_count Inf values!")
    end

    # Check for NaN/Inf in b
    nan_b = count(isnan, b)
    inf_b = count(isinf, b)
    if nan_b > 0 || inf_b > 0
        println("WARNING: b has $nan_b NaN and $inf_b Inf values!")
    end

    # Check condition number
    try
        G_dense = Matrix(G)
        if !any(isnan, G_dense) && !any(isinf, G_dense)
            cond_num = cond(G_dense)
            println("G condition number: $cond_num")
            if cond_num > 1e15
                println("  WARNING: Severely ill-conditioned!")
            end
        end
    catch e
        println("Could not compute condition number: $e")
    end

    # Try to solve linear system
    println("\nLinear solve test (G\\b):")
    try
        lu_fact = lu(G; check=false)
        if issuccess(lu_fact)
            x_lin = lu_fact \ b
            println("  Linear solve succeeded")
            println("  Solution range: $(extrema(x_lin))")
            if any(isnan, x_lin) || any(isinf, x_lin)
                println("  WARNING: Solution contains NaN/Inf!")
            end
        else
            println("  Linear solve FAILED - matrix singular")
        end
    catch e
        println("  Linear solve error: $e")
    end

    return resid, G, b
end

# Test at zeros
analyze_point("u = zeros", zeros(n))

# Test at small positive values
analyze_point("u = 0.1", fill(0.1, n))

# Test at values that respect BJT physics (Vbe ~ 0.7V)
u_bjt = zeros(n)
# Set node voltages to reasonable BJT operating point
for (i, name) in enumerate(sys.node_names)
    if occursin("base", String(name))
        u_bjt[i] = 0.7  # Vbe ~ 0.7V
    elseif occursin("coll", String(name))
        u_bjt[i] = 2.5  # Moderate collector voltage
    elseif occursin("vcc", String(name))
        u_bjt[i] = 5.0
    elseif occursin("xf", String(name))
        u_bjt[i] = 0.0  # Internal phase nodes
    elseif occursin("c_int", String(name)) || occursin("sub_con", String(name))
        u_bjt[i] = 2.5  # Internal BJT nodes
    end
end
analyze_point("u = BJT-like operating point", u_bjt)

#==============================================================================#
# Manual Newton iteration with logging
#==============================================================================#

println("\n" * "="^60)
println("Manual Newton Iteration (with logging)")
println("="^60)

abstol = 1e-10
maxiters = 20
u_newton = zeros(n)

for iter in 1:maxiters
    global u_newton
    # Rebuild at current point
    fast_rebuild!(ws, cs, u_newton, 0.0)
    G = cs.G
    b = ws.dctx.b

    # Compute residual
    resid = G * u_newton - b
    resid_norm = norm(resid)

    println("\nIteration $iter:")
    println("  ||F|| = $resid_norm")
    println("  u range: $(extrema(u_newton))")

    if resid_norm < abstol
        println("  CONVERGED!")
        break
    end

    if any(isnan, resid) || any(isinf, resid)
        println("  FAILED: Residual contains NaN/Inf")
        break
    end

    # Try Newton step: solve G * du = -resid
    try
        lu_fact = lu(G; check=false)
        if !issuccess(lu_fact)
            println("  FAILED: Jacobian is singular")
            break
        end

        du = lu_fact \ (-resid)
        du_norm = norm(du)
        println("  ||du|| = $du_norm")

        if any(isnan, du) || any(isinf, du)
            println("  FAILED: Newton step contains NaN/Inf")
            break
        end

        # Line search / damping
        alpha = 1.0
        u_new = u_newton + alpha * du
        fast_rebuild!(ws, cs, u_new, 0.0)
        resid_new = cs.G * u_new - ws.dctx.b
        resid_new_norm = norm(resid_new)

        # Simple backtracking
        for _ in 1:10
            if resid_new_norm < resid_norm || alpha < 1e-8
                break
            end
            alpha *= 0.5
            u_new = u_newton + alpha * du
            fast_rebuild!(ws, cs, u_new, 0.0)
            resid_new = cs.G * u_new - ws.dctx.b
            resid_new_norm = norm(resid_new)
            println("  Backtracking: alpha=$alpha, ||F_new||=$resid_new_norm")
        end

        if any(isnan, u_new) || any(isinf, u_new)
            println("  FAILED: New u contains NaN/Inf after step")
            break
        end

        u_newton = u_new

    catch e
        println("  FAILED: $e")
        break
    end
end

#==============================================================================#
# Try CedarRobustNLSolve via MNA._dc_newton_compiled
#==============================================================================#

println("\n" * "="^60)
println("Testing MNA._dc_newton_compiled with CedarRobustNLSolve")
println("="^60)

u0 = zeros(n)
try
    u_sol, converged = MNA._dc_newton_compiled(cs, ws, u0;
                                                abstol=1e-10, maxiters=100,
                                                nlsolve=CedarRobustNLSolve())
    println("Converged: $converged")
    println("Solution range: $(extrema(u_sol))")

    # Always verify residual
    fast_rebuild!(ws, cs, u_sol, 0.0)
    resid = cs.G * u_sol - ws.dctx.b
    println("Final residual norm: $(norm(resid))")
    println("Final residual range: $(extrema(resid))")

    # Show node values
    println("\nNode voltages at solution:")
    for (i, name) in enumerate(sys.node_names)
        println("  V($name) = $(u_sol[i])")
    end

    # Check which nodes have large residual
    println("\nResidual breakdown:")
    for i in 1:n
        if abs(resid[i]) > 1e-6
            if i <= length(sys.node_names)
                println("  Node $(sys.node_names[i]): residual = $(resid[i])")
            else
                println("  Current $(i - length(sys.node_names)): residual = $(resid[i])")
            end
        end
    end

catch e
    println("ERROR: $e")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

#==============================================================================#
# Try with relaxed tolerances
#==============================================================================#

println("\n" * "="^60)
println("Testing with various tolerances")
println("="^60)

for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    u_sol, converged = MNA._dc_newton_compiled(cs, ws, zeros(n);
                                                abstol=tol, maxiters=100,
                                                nlsolve=CedarRobustNLSolve())
    println("abstol=$tol: Converged=$converged")
end

#==============================================================================#
# Check if the "failed" solution is actually usable
#==============================================================================#

println("\n" * "="^60)
println("Testing if DC solution enables transient simulation")
println("="^60)

using OrdinaryDiffEq: Rodas5P

try
    circuit2 = MNACircuit(monostable_multivibrator)

    # Try with abstol=1e-2 for DC solve (should converge)
    sol = CedarSim.tran!(circuit2, (0.0, 1e-3);
                          solver=Rodas5P(),
                          initializealg=CedarDCOp(abstol=1e-2),
                          abstol=1e-6, reltol=1e-4)

    println("Transient with CedarDCOp(abstol=1e-2): $(sol.retcode)")

    if sol.retcode == SciMLBase.ReturnCode.Success
        sys2 = assemble!(circuit2)
        acc = MNA.MNASolutionAccessor(sol, sys2)
        v_q1 = MNA.voltage(acc, :q1_coll, 0.5e-3)
        v_q2 = MNA.voltage(acc, :q2_coll, 0.5e-3)
        println("  V(q1_coll) @ 0.5ms = $v_q1")
        println("  V(q2_coll) @ 0.5ms = $v_q2")
    end
catch e
    println("ERROR: $e")
    println(stacktrace())
end

println("\n" * "="^60)
println("Diagnostic Complete")
println("="^60)
