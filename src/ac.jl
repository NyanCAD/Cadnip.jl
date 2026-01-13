#==============================================================================#
# AC Small-Signal Analysis
#
# Performs frequency-domain analysis by linearizing the circuit around
# a DC operating point and computing transfer functions.
#
# Key concepts:
# - DC operating point is found via Newton iteration
# - Linearized G (conductance) and C (capacitance) matrices at operating point
# - AC excitation vector b_ac from sources with AC specification
# - DescriptorStateSpace system: E·dx = A·x + B·u, y = C·x
#
# For MNA: C·dx/dt + G·x = b
# DSS form: E·dx = A·x + B·u  where E=C, A=-G, B=b_ac
#==============================================================================#

using DescriptorSystems
using LinearAlgebra

export ac!, acdec, freqresp

abstract type FreqSol end

"""
    ACSol <: FreqSol

AC small-signal solution for MNA circuits.

Contains the linearized descriptor state-space system and DC operating point.

# Fields
- `dss`: DescriptorStateSpace system (E·dx = A·x + B·u, y = C·x)
- `dc_x`: DC operating point solution vector
- `node_names`: Names of voltage nodes
- `current_names`: Names of current variables
- `n_nodes`: Number of voltage nodes

# Usage
```julia
ac_sol = ac!(circuit)
ωs = 2π .* acdec(20, 0.01, 10)  # frequencies in rad/s
resp = freqresp(ac_sol, :vout, ωs)  # frequency response at node :vout
```
"""
struct ACSol <: FreqSol
    dss::DescriptorStateSpace{Float64, Matrix{Float64}}
    dc_x::Vector{Float64}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
end

"""
    ac!(circuit::MNA.MNACircuit; gmin=1e-12) -> ACSol

Perform AC small-signal analysis on an MNA circuit.

# Algorithm
1. Solve DC operating point via Newton iteration
2. Extract linearized G, C matrices at operating point
3. Assemble AC excitation vector b_ac from sources
4. Build DescriptorStateSpace: E=C, A=-G, B=b_ac, C=I, D=0

# Arguments
- `circuit::MNACircuit`: Circuit with builder function and parameters
- `gmin`: Minimum conductance for numerical stability

# Returns
`ACSol` containing the linearized DSS system and DC solution.

# Example
```julia
circuit = MNACircuit(build_filter; R=1000.0, C=1e-6)
ac_sol = ac!(circuit)
ωs = 2π .* acdec(20, 0.01, 10)
resp = freqresp(ac_sol, :vout, ωs)
```
"""
function ac!(circuit::MNA.MNACircuit; gmin=1e-12)
    # Build circuit with structure discovery
    ctx = MNA.MNAContext()
    circuit.builder(circuit.params, circuit.spec, 0.0; x=MNA.ZERO_VECTOR, ctx=ctx)

    # Solve DC operating point using the MNA solve_dc
    dc_sol = MNA.solve_dc(circuit)
    dc_x = dc_sol.x

    # Rebuild at DC solution to get linearized matrices
    MNA.reset_for_restamping!(ctx)
    circuit.builder(circuit.params, circuit.spec, 0.0; x=dc_x, ctx=ctx)

    # Assemble matrices
    G = Matrix(MNA.assemble_G(ctx; gmin=gmin))
    C = Matrix(MNA.assemble_C(ctx))
    b_ac = MNA.get_rhs_ac(ctx)

    n = MNA.system_size(ctx)

    # Build DescriptorStateSpace
    # MNA: C·dx/dt + G·x = b
    # Standard DSS: E·dx/dt = A·x + B·u
    # Rewrite: C·dx/dt = -G·x + b_ac
    # So: E = C, A = -G, B = b_ac (as column vector)
    #
    # Transfer function: H(jω) = C·(jωE - A)⁻¹·B = I·(jωC + G)⁻¹·b_ac
    # This gives voltage/current at each node for AC excitation

    # Handle case where there's no AC excitation
    if all(iszero, b_ac)
        @warn "No AC sources found in circuit"
        B = zeros(n, 1)
    else
        B = reshape(real.(b_ac), n, 1)
    end

    # Handle complex AC excitation (with phase)
    # The DSS library expects real matrices, so include both real and imag parts
    has_complex_ac = any(x -> imag(x) != 0, b_ac)
    if has_complex_ac
        B = hcat(real.(b_ac), imag.(b_ac))
    end

    # Identity output matrix (observe all states)
    C_out = Matrix{Float64}(I, n, n)
    D = zeros(n, size(B, 2))

    dss_sys = dss(-G, C, B, C_out, D)

    return ACSol(dss_sys, dc_x, copy(ctx.node_names), copy(ctx.current_names), ctx.n_nodes)
end

"""
    freqresp(ac::ACSol, node::Symbol, ωs::AbstractVector{<:Real}) -> Vector{ComplexF64}

Compute frequency response at specified node across frequencies.

# Arguments
- `ac`: AC solution from `ac!()`
- `node`: Node name (Symbol) to observe
- `ωs`: Angular frequencies in rad/s

# Returns
Vector of complex frequency response values.

# Example
```julia
ac_sol = ac!(circuit)
ωs = 2π .* acdec(20, 1, 1e6)  # 1 Hz to 1 MHz
resp = freqresp(ac_sol, :vout, ωs)
mag_dB = 20 .* log10.(abs.(resp))
phase_deg = angle.(resp) .* 180 ./ π
```
"""
function freqresp(ac::ACSol, node::Symbol, ωs::AbstractVector{<:Real})
    idx = _get_node_index(ac, node)
    idx == 0 && error("Cannot compute frequency response at ground (node 0)")

    # Get DSS data
    A, E, B, C_out, D = dssdata(ac.dss)
    n = size(A, 1)
    n_inputs = size(B, 2)

    # Compute frequency response for this output
    result = Vector{ComplexF64}(undef, length(ωs))

    for (i, ω) in enumerate(ωs)
        # H(jω) = C·(jωE - A)⁻¹·B + D
        # For single output at index idx: H_idx = e_idx' · (jωE - A)⁻¹ · B
        jωE_minus_A = (im * ω) .* E .- A

        # Solve (jωE - A) * x = B for x
        x = jωE_minus_A \ B

        # Output at node idx
        if n_inputs == 1
            result[i] = x[idx, 1]
        else
            # Complex excitation: result = x[:,1] + im*x[:,2]
            result[i] = x[idx, 1] + im * x[idx, 2]
        end
    end

    return result
end

# DescriptorSystems.freqresp integration
function DescriptorSystems.freqresp(ac::ACSol, node::Symbol, ωs::AbstractVector{<:Real})
    return freqresp(ac, node, ωs)
end

"""
    _get_node_index(ac::ACSol, node::Symbol) -> Int

Get the system index for a node by name. Returns 0 for ground.
"""
function _get_node_index(ac::ACSol, node::Symbol)
    (node === :gnd || node === Symbol("0")) && return 0
    idx = findfirst(==(node), ac.node_names)
    idx === nothing && error("Unknown node: $node. Available: $(ac.node_names)")
    return idx
end

"""
    _get_current_index(ac::ACSol, name::Symbol) -> Int

Get the system index for a current variable by name.
"""
function _get_current_index(ac::ACSol, name::Symbol)
    idx = findfirst(==(name), ac.current_names)
    idx === nothing && error("Unknown current: $name. Available: $(ac.current_names)")
    return ac.n_nodes + idx
end

"""
    Base.getindex(ac::ACSol, node::Symbol) -> DescriptorStateSpace

Get the descriptor state-space subsystem for observing a specific node.

This returns a SISO (single-input single-output) system from AC excitation
to the specified node voltage.

# Example
```julia
ac_sol = ac!(circuit)
dss_vout = ac_sol[:vout]
# Can use with ControlSystemsBase for bode plots, etc.
```
"""
function Base.getindex(ac::ACSol, node::Symbol)
    idx = _get_node_index(ac, node)
    idx == 0 && error("Cannot create subsystem for ground")

    A, E, B, C_out, D = dssdata(ac.dss)
    n = size(A, 1)

    # Single-row C matrix for this output
    C_row = zeros(1, n)
    C_row[1, idx] = 1.0

    D_row = zeros(1, size(B, 2))

    return dss(A, E, B, C_row, D_row)
end

"""
    acdec(nd, fstart, fstop)

Generate a logarithmically spaced frequency vector from `fstart` to `fstop`
with `nd` points per decade.  Equivalent to the SPICE command:

    .ac dec nd fstart fstop

Return value is a vector in hertz per second.
"""
function acdec(nd, fstart, fstop)
    fstart = log10(fstart)
    fstop = log10(fstop)
    points = Int(ceil((fstop-fstart)*nd))+1
    return exp10.(range(fstart, stop=fstop, length=points))
end
