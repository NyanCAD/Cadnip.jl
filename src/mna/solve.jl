#==============================================================================#
# MNA Phase 1: Analysis Solvers
#
# This module provides analysis functions for the assembled MNA system:
# - DC Analysis: Steady-state solution (G*x = b)
# - AC Analysis: Small-signal frequency response ((G + jωC)*x = b)
# - Transient: ODEProblem formulation for DifferentialEquations.jl
#
# The solvers work with the MNASystem assembled from MNAContext.
#==============================================================================#

using LinearAlgebra
using SparseArrays

export DCSolution, ACSolution
export solve_dc, solve_dc!, solve_ac
export make_ode_problem, make_ode_function
export make_dae_problem, make_dae_function
export voltage, current, magnitude_db, phase_deg

#==============================================================================#
# Solution Types
#==============================================================================#

"""
    DCSolution

Result of DC operating point analysis.

# Fields
- `x::Vector{Float64}`: Solution vector [V₁, V₂, ..., I₁, I₂, ...]
- `node_names::Vector{Symbol}`: Node names for interpretation
- `current_names::Vector{Symbol}`: Current variable names
- `n_nodes::Int`: Number of voltage nodes
"""
struct DCSolution
    x::Vector{Float64}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
end

"""
    DCSolution(sys::MNASystem, x::Vector{Float64})

Create a DC solution from a system and solution vector.
"""
DCSolution(sys::MNASystem, x::Vector{Float64}) =
    DCSolution(x, sys.node_names, sys.current_names, sys.n_nodes)

# Accessors
Base.getindex(sol::DCSolution, i::Int) = sol.x[i]
Base.length(sol::DCSolution) = length(sol.x)

"""
    voltage(sol::DCSolution, name::Symbol) -> Float64

Get the voltage at a node by name.
"""
function voltage(sol::DCSolution, name::Symbol)
    (name === :gnd || name === Symbol("0")) && return 0.0
    idx = findfirst(==(name), sol.node_names)
    idx === nothing && error("Unknown node: $name")
    return sol.x[idx]
end

"""
    voltage(sol::DCSolution, idx::Int) -> Float64

Get the voltage at a node by index (0 = ground).
"""
function voltage(sol::DCSolution, idx::Int)
    idx == 0 && return 0.0
    return sol.x[idx]
end

"""
    current(sol::DCSolution, name::Symbol) -> Float64

Get a current variable by name.
"""
function current(sol::DCSolution, name::Symbol)
    idx = findfirst(==(name), sol.current_names)
    idx === nothing && error("Unknown current: $name")
    return sol.x[sol.n_nodes + idx]
end

function Base.show(io::IO, sol::DCSolution)
    print(io, "DCSolution(")
    for (i, name) in enumerate(sol.node_names)
        i > 1 && print(io, ", ")
        @printf(io, "%s=%.4g", name, sol.x[i])
    end
    for (i, name) in enumerate(sol.current_names)
        print(io, ", ")
        @printf(io, "%s=%.4g", name, sol.x[sol.n_nodes + i])
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", sol::DCSolution)
    println(io, "DC Solution:")
    println(io, "  Node Voltages:")
    for (i, name) in enumerate(sol.node_names)
        @printf(io, "    V(%s) = %.6g V\n", name, sol.x[i])
    end
    if !isempty(sol.current_names)
        println(io, "  Branch Currents:")
        for (i, name) in enumerate(sol.current_names)
            @printf(io, "    %s = %.6g A\n", name, sol.x[sol.n_nodes + i])
        end
    end
end

#==============================================================================#
# AC Solution
#==============================================================================#

"""
    ACSolution

Result of AC small-signal analysis.

# Fields
- `freqs::Vector{Float64}`: Frequency points (Hz)
- `x::Vector{Vector{ComplexF64}}`: Solution at each frequency
- `node_names::Vector{Symbol}`: Node names
- `current_names::Vector{Symbol}`: Current variable names
- `n_nodes::Int`: Number of voltage nodes
"""
struct ACSolution
    freqs::Vector{Float64}
    x::Vector{Vector{ComplexF64}}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
end

"""
    voltage(sol::ACSolution, name::Symbol) -> Vector{ComplexF64}

Get the complex voltage at a node across all frequencies.
"""
function voltage(sol::ACSolution, name::Symbol)
    (name === :gnd || name === Symbol("0")) && return zeros(ComplexF64, length(sol.freqs))
    idx = findfirst(==(name), sol.node_names)
    idx === nothing && error("Unknown node: $name")
    return [x[idx] for x in sol.x]
end

"""
    voltage(sol::ACSolution, name::Symbol, freq_idx::Int) -> ComplexF64

Get the complex voltage at a specific frequency index.
"""
function voltage(sol::ACSolution, name::Symbol, freq_idx::Int)
    (name === :gnd || name === Symbol("0")) && return 0.0 + 0.0im
    idx = findfirst(==(name), sol.node_names)
    idx === nothing && error("Unknown node: $name")
    return sol.x[freq_idx][idx]
end

"""
    magnitude_db(sol::ACSolution, name::Symbol) -> Vector{Float64}

Get the voltage magnitude in dB at a node.
"""
magnitude_db(sol::ACSolution, name::Symbol) = 20 .* log10.(abs.(voltage(sol, name)))

"""
    phase_deg(sol::ACSolution, name::Symbol) -> Vector{Float64}

Get the voltage phase in degrees at a node.
"""
phase_deg(sol::ACSolution, name::Symbol) = rad2deg.(angle.(voltage(sol, name)))

function Base.show(io::IO, sol::ACSolution)
    print(io, "ACSolution($(length(sol.freqs)) frequencies, ")
    print(io, "$(length(sol.node_names)) nodes)")
end

#==============================================================================#
# DC Analysis
#==============================================================================#

"""
    solve_dc(sys::MNASystem) -> DCSolution

Solve for DC operating point: G*x = b

For circuits with capacitors/inductors, this finds the steady-state
where all derivatives are zero (C*dx/dt = 0).
"""
function solve_dc(sys::MNASystem)
    n = system_size(sys)
    n == 0 && return DCSolution(Float64[], Symbol[], Symbol[], 0)

    # Solve G*x = b
    # Use \ which automatically selects appropriate solver
    x = sys.G \ sys.b

    return DCSolution(sys, x)
end

"""
    solve_dc!(x::Vector{Float64}, sys::MNASystem)

Solve DC operating point into pre-allocated vector x.
"""
function solve_dc!(x::Vector{Float64}, sys::MNASystem)
    n = system_size(sys)
    n == 0 && return x
    length(x) >= n || resize!(x, n)

    # Solve in-place using ldiv! if possible
    copyto!(view(x, 1:n), sys.b)
    F = lu(sys.G)
    ldiv!(F, view(x, 1:n))

    return x
end

"""
    solve_dc(ctx::MNAContext) -> DCSolution

Convenience function: assemble and solve in one step.
"""
function solve_dc(ctx::MNAContext)
    sys = assemble!(ctx)
    return solve_dc(sys)
end

#==============================================================================#
# AC Analysis
#==============================================================================#

"""
    solve_ac(sys::MNASystem, freqs::AbstractVector{<:Real}) -> ACSolution

Solve AC small-signal analysis at given frequencies.

For each frequency f, solves: (G + j*2π*f*C) * x = b

This linearizes around the DC operating point, so DC analysis
should be performed first if the circuit contains nonlinear elements.
"""
function solve_ac(sys::MNASystem, freqs::AbstractVector{<:Real})
    n = system_size(sys)
    nf = length(freqs)

    results = Vector{Vector{ComplexF64}}(undef, nf)
    b_complex = complex.(sys.b)

    for (i, f) in enumerate(freqs)
        omega = 2π * f
        # Form Y = G + jωC
        Y = sys.G + (im * omega) * sys.C
        # Solve Y*x = b
        results[i] = Y \ b_complex
    end

    return ACSolution(collect(Float64, freqs), results,
                      sys.node_names, sys.current_names, sys.n_nodes)
end

"""
    solve_ac(sys::MNASystem; fstart, fstop, points_per_decade) -> ACSolution

Solve AC analysis with logarithmically spaced frequencies.
"""
function solve_ac(sys::MNASystem; fstart::Real, fstop::Real, points_per_decade::Int=10)
    # Generate log-spaced frequencies
    decades = log10(fstop / fstart)
    n_points = max(2, round(Int, decades * points_per_decade) + 1)
    freqs = 10 .^ range(log10(fstart), log10(fstop), length=n_points)
    return solve_ac(sys, freqs)
end

#==============================================================================#
# Transient Analysis: ODEProblem Formulation
#==============================================================================#

"""
    make_ode_function(sys::MNASystem) -> ODEFunction

Create an ODEFunction for use with DifferentialEquations.jl.

The MNA system G*x + C*dx/dt = b is converted to the form:
    C * dx/dt = b - G*x

This returns an ODEFunction with mass_matrix = C.

# Notes
- For singular C (algebraic constraints), use a DAE solver
- For constant C with nonzero diagonal, use implicit ODE solver
"""
function make_ode_function(sys::MNASystem)
    G = sys.G
    C = sys.C
    b = sys.b
    n = system_size(sys)

    # Check if C has any structure
    has_dynamics = nnz(C) > 0

    if !has_dynamics
        # Pure algebraic system - warn and return constant function
        @warn "No capacitors/inductors - system is purely algebraic. Consider using solve_dc instead."
    end

    # RHS function: C * du/dt = b - G*u
    # We provide f!(du, u, p, t) where du represents C * dx/dt
    function rhs!(du, u, p, t)
        # du = b - G*u
        mul!(du, G, u)
        du .*= -1
        du .+= b
        return nothing
    end

    # Jacobian: d(rhs)/du = -G
    function jac!(J, u, p, t)
        copyto!(J, -G)
        return nothing
    end

    # Create ODEFunction with mass matrix
    # Using SciMLBase/DiffEqBase types
    return (
        rhs! = rhs!,
        jac! = jac!,
        mass_matrix = C,
        jac_prototype = -G  # Sparsity pattern
    )
end

"""
    make_ode_problem(sys::MNASystem, tspan::Tuple{Real,Real};
                     u0::Union{Nothing,Vector{Float64}}=nothing) -> NamedTuple

Create an ODEProblem-like structure for transient analysis.

Returns a NamedTuple with fields needed to construct an ODEProblem:
- `f`: RHS function
- `u0`: Initial condition
- `tspan`: Time span
- `mass_matrix`: Mass matrix C
- `jac`: Jacobian function
- `jac_prototype`: Sparsity pattern for Jacobian

# Usage with OrdinaryDiffEq
```julia
using OrdinaryDiffEq

prob_data = make_ode_problem(sys, (0.0, 1e-3))
f = ODEFunction(prob_data.f;
                mass_matrix = prob_data.mass_matrix,
                jac = prob_data.jac,
                jac_prototype = prob_data.jac_prototype)
prob = ODEProblem(f, prob_data.u0, prob_data.tspan)
sol = solve(prob, Rodas5())
```

# Arguments
- `sys::MNASystem`: The assembled MNA system
- `tspan`: Time span (tstart, tstop)
- `u0`: Initial condition (default: DC solution)
"""
function make_ode_problem(sys::MNASystem, tspan::Tuple{Real,Real};
                          u0::Union{Nothing,Vector{Float64}}=nothing)
    n = system_size(sys)

    # Default initial condition: DC solution
    if u0 === nothing
        dc_sol = solve_dc(sys)
        u0 = dc_sol.x
    end

    # Get ODE function components
    ode_funcs = make_ode_function(sys)

    return (
        f = ode_funcs.rhs!,
        u0 = u0,
        tspan = Float64.(tspan),
        mass_matrix = ode_funcs.mass_matrix,
        jac = ode_funcs.jac!,
        jac_prototype = ode_funcs.jac_prototype,
        sys = sys  # Keep reference for solution interpretation
    )
end

"""
    make_dae_function(sys::MNASystem) -> NamedTuple

Create a DAE function for use with DAE solvers (e.g., Sundials IDA).

The MNA system G*x + C*dx/dt = b is converted to implicit DAE form:
    F(du, u, p, t) = C*du + G*u - b = 0

This is useful when C is singular (has zero rows for algebraic equations).

# Returns
NamedTuple with:
- `f!`: Residual function F!(resid, du, u, p, t)
- `jac_du!`: Jacobian w.r.t. du (= C)
- `jac_u!`: Jacobian w.r.t. u (= G)
- `differential_vars`: Boolean vector indicating differential variables

# Usage with Sundials
```julia
using Sundials

dae_data = make_dae_function(sys)
prob = DAEProblem(dae_data.f!, dae_data.du0, dae_data.u0, tspan;
                  differential_vars = dae_data.differential_vars)
sol = solve(prob, IDA())
```
"""
function make_dae_function(sys::MNASystem)
    G = sys.G
    C = sys.C
    b = sys.b
    n = system_size(sys)

    # DAE residual: F = C*du + G*u - b = 0
    function dae_residual!(resid, du, u, p, t)
        # resid = C*du + G*u - b
        mul!(resid, C, du)        # resid = C*du
        mul!(resid, G, u, 1.0, 1.0)  # resid += G*u
        resid .-= b               # resid -= b
        return nothing
    end

    # Jacobian w.r.t. du: dF/d(du) = C
    function jac_du!(J, du, u, p, gamma, t)
        copyto!(J, C)
        return nothing
    end

    # Jacobian w.r.t. u: dF/du = G
    function jac_u!(J, du, u, p, gamma, t)
        copyto!(J, G)
        return nothing
    end

    # Determine which variables are differential (have nonzero C row)
    # A variable is differential if its corresponding row in C has nonzeros
    differential_vars = zeros(Bool, n)
    for j in 1:n
        for k in nzrange(C, j)
            i = rowvals(C)[k]
            differential_vars[i] = true
        end
    end

    return (
        f! = dae_residual!,
        jac_du! = jac_du!,
        jac_u! = jac_u!,
        differential_vars = differential_vars,
        C = C,
        G = G,
        b = b
    )
end

"""
    make_dae_problem(sys::MNASystem, tspan::Tuple{Real,Real};
                     u0::Union{Nothing,Vector{Float64}}=nothing) -> NamedTuple

Create a DAEProblem-like structure for transient analysis with DAE solvers.

# Arguments
- `sys::MNASystem`: The assembled MNA system
- `tspan`: Time span (tstart, tstop)
- `u0`: Initial condition (default: DC solution)

# Returns
NamedTuple with fields for DAEProblem construction.
"""
function make_dae_problem(sys::MNASystem, tspan::Tuple{Real,Real};
                          u0::Union{Nothing,Vector{Float64}}=nothing)
    n = system_size(sys)

    # Default initial condition: DC solution
    if u0 === nothing
        dc_sol = solve_dc(sys)
        u0 = dc_sol.x
    end

    # Get DAE function components
    dae_funcs = make_dae_function(sys)

    # Initial du from the DAE: C*du = b - G*u => du = C \ (b - G*u)
    # For consistent initialization, du should satisfy F(du, u, 0) = 0
    rhs = sys.b - sys.G * u0

    # Compute initial du (only for differential variables)
    # C is often singular (zero rows for algebraic equations)
    du0 = zeros(n)
    C = sys.C
    diff_vars = dae_funcs.differential_vars

    if any(diff_vars)
        # For each differential variable, compute its initial derivative
        # from the corresponding row of C*du = rhs
        C_dense = Matrix(C)
        for i in 1:n
            if diff_vars[i]
                # Find the diagonal element of C for this variable
                c_ii = C_dense[i, i]
                if abs(c_ii) > 1e-15
                    # Simple case: diagonal C element, du[i] = rhs[i] / c_ii
                    du0[i] = rhs[i] / c_ii
                else
                    # Off-diagonal case: solve row by row (simplified)
                    # For MNA, C is usually diagonal or block-diagonal
                    row_sum = sum(abs.(C_dense[i, :]))
                    if row_sum > 1e-15
                        # Use pseudoinverse for this row
                        du0[i] = rhs[i] / row_sum
                    end
                end
            end
            # Algebraic variables (diff_vars[i] == false) keep du0[i] = 0
        end
    end

    return (
        f! = dae_funcs.f!,
        u0 = u0,
        du0 = du0,
        tspan = Float64.(tspan),
        differential_vars = dae_funcs.differential_vars,
        jac_du! = dae_funcs.jac_du!,
        jac_u! = dae_funcs.jac_u!,
        sys = sys
    )
end

#==============================================================================#
# Utility Functions
#==============================================================================#

"""
    check_singular(sys::MNASystem) -> Bool

Check if the G matrix is singular (no DC solution possible).
Returns true if singular.
"""
function check_singular(sys::MNASystem)
    n = system_size(sys)
    n == 0 && return false
    try
        F = lu(sys.G; check=false)
        return !issuccess(F)
    catch
        return true
    end
end

"""
    condition_number(sys::MNASystem) -> Float64

Compute the condition number of the G matrix.
Large values indicate ill-conditioning.
"""
function condition_number(sys::MNASystem)
    n = system_size(sys)
    n == 0 && return 1.0
    # For sparse matrices, compute via SVD of small systems or estimate
    if n <= 100
        return cond(Matrix(sys.G))
    else
        # For large systems, estimate using iterative methods
        # (simplified: just return norm ratio)
        return norm(sys.G, 1) * norm(inv(Matrix(sys.G)), 1)
    end
end
