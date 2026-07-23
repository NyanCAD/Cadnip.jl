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
#
# Observable access:
# ------------------
# Any system variable is accessible by name — top-level nodes, branch currents,
# and hierarchical subcircuit nodes (which flatten into the name table, e.g.
# `:x1_out`). A `NodeRef` from `scope(...)` also indexes an `ACSol`.
#      - Node voltages: `ac[:vout]` (over the grid) or `freqresp(ac, :vout, ωs)`
#      - Branch currents: `ac[:I_V1]` or `freqresp(ac, :I_V1, ωs)`
#
# The one thing that is NOT a system variable is a voltage *across* a device
# wired between two named nodes (an inductor's `V(l3)`); that is a derived
# quantity, taken as a node difference:
#      `freqresp(ac, :n1, ωs) - freqresp(ac, :n2, ωs)`
#
# Noise analysis: `noise!()` (see src/noise.jl) reuses this same
# rebuild-at-DC-operating-point linearization, adjoint-solving `(jωC+G)` per
# frequency to accumulate the output-noise PSD from the sources on the context's
# noise channel. See doc/noise_analysis_design.md.
#
# Bode plots: Use RobustAndOptimalControl to convert DescriptorStateSpace to
# standard StateSpace: `bode(ss(subsystem(ac, :vout)), ωs)`
#==============================================================================#

using DescriptorSystems
using LinearAlgebra

export ac!, acdec, freqresp, magnitude_db, phase_deg, subsystem

abstract type FreqSol end

"""
    ACSol <: FreqSol

AC small-signal solution for MNA circuits.

Contains the linearized descriptor state-space system and DC operating point,
plus the optional Hz frequency grid the analysis was requested over.

# Fields
- `dss`: DescriptorStateSpace system (E·dx = A·x + B·u, y = C·x)
- `dc_x`: DC operating point solution vector
- `node_names`: Names of voltage nodes
- `current_names`: Names of current variables
- `n_nodes`: Number of voltage nodes
- `freqs`: Frequency grid in **hertz** (SPICE `.ac` convention); empty when the
  analysis was run without one

# Usage
```julia
# With a frequency grid: name-based access returns the SPICE-native response.
ac_sol = ac!(circuit, acdec(20, 0.01, 10))   # freqs in Hz
resp = ac_sol[:vout]                          # complex response over the grid

# Without a grid: evaluate at chosen angular frequencies via freqresp.
ac_sol = ac!(circuit)
ωs = 2π .* acdec(20, 0.01, 10)                # rad/s (ControlSystems contract)
resp = freqresp(ac_sol, :vout, ωs)
```
"""
struct ACSol <: FreqSol
    dss::DescriptorStateSpace{Float64, Matrix{Float64}}
    dc_x::Vector{Float64}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
    freqs::Vector{Float64}
end

"""
    ac!(circuit::MNA.MNACircuit, freqs=Float64[]; gmin=1e-12) -> ACSol

Perform AC small-signal analysis on an MNA circuit.

# Algorithm
1. Solve DC operating point via Newton iteration
2. Extract linearized G, C matrices at operating point
3. Assemble AC excitation vector b_ac from sources
4. Build DescriptorStateSpace: E=C, A=-G, B=b_ac, C=I, D=0

# Arguments
- `circuit::MNACircuit`: Circuit with builder function and parameters
- `freqs`: Frequency grid in **hertz** (SPICE `.ac` convention, e.g.
  `acdec(20, 1, 1e6)`). When supplied, name-based access `ac[:name]` and the
  two-argument `magnitude_db(ac, :name)` / `phase_deg(ac, :name)` return the
  response over this grid. Optional — omit it and evaluate at arbitrary angular
  frequencies via [`freqresp`](@ref) instead.
- `gmin`: Minimum conductance for numerical stability

# Returns
`ACSol` containing the linearized DSS system, DC solution, and frequency grid.

# Example
```julia
circuit = MNACircuit(build_filter; R=1000.0, C=1e-6)
ac_sol = ac!(circuit, acdec(20, 0.01, 10))   # freqs in Hz
resp = ac_sol[:vout]                          # response over the grid
```
"""
function ac!(circuit::MNA.MNACircuit, freqs::AbstractVector{<:Real}=Float64[]; gmin=1e-12)
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
    G = Matrix(MNA.assemble_G(ctx; gshunt=gmin))
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

    return ACSol(dss_sys, dc_x, copy(ctx.node_names), copy(ctx.current_names),
                 ctx.n_nodes, collect(Float64, freqs))
end

"""
    freqresp(ac::ACSol, name::Symbol, ωs::AbstractVector{<:Real}) -> Vector{ComplexF64}

Compute frequency response at specified node or current across frequencies.

# Arguments
- `ac`: AC solution from `ac!()`
- `name`: Node or current name (Symbol) to observe (e.g., `:vout`, `:I_V1`)
- `ωs`: Angular frequencies in rad/s

# Returns
Vector of complex frequency response values.

# Example
```julia
ac_sol = ac!(circuit)
ωs = 2π .* acdec(20, 1, 1e6)  # 1 Hz to 1 MHz
resp = freqresp(ac_sol, :vout, ωs)      # Node voltage
resp_i = freqresp(ac_sol, :I_V1, ωs)    # Voltage source current
mag_dB = 20 .* log10.(abs.(resp))
phase_deg = angle.(resp) .* 180 ./ π
```
"""
function freqresp(ac::ACSol, name::Symbol, ωs::AbstractVector{<:Real})
    idx = _get_index(ac, name)
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
function DescriptorSystems.freqresp(ac::ACSol, name::Symbol, ωs::AbstractVector{<:Real})
    return freqresp(ac, name, ωs)
end

"""
    magnitude_db(ac::ACSol, name::Symbol, freqs::AbstractVector{<:Real}) -> Vector{Float64}

Magnitude in dB (`20·log₁₀|H|`) of node/current `name` across `freqs`, which are
given in **hertz** (as returned by [`acdec`](@ref)), matching SPICE `.ac`
conventions, so you no longer need to convert to rad/s by hand. See also the
two-argument [`magnitude_db(::ACSol, ::Symbol)`](@ref) which reads the grid
stored on `ac`.

```julia
ac = ac!(circuit)
f  = acdec(20, 1, 1e6)            # 1 Hz … 1 MHz, in Hz
mag = magnitude_db(ac, :vout, f)  # dB, no 2π conversion needed
```
"""
function MNA.magnitude_db(ac::ACSol, name::Symbol, freqs::AbstractVector{<:Real})
    return 20 .* log10.(abs.(freqresp(ac, name, 2π .* freqs)))
end

"""
    phase_deg(ac::ACSol, name::Symbol, freqs::AbstractVector{<:Real}) -> Vector{Float64}

Phase in degrees of node/current `name` across `freqs` (in **hertz**). Companion
to [`magnitude_db(::ACSol, ::Symbol, ::AbstractVector)`](@ref); see it for units.
"""
function MNA.phase_deg(ac::ACSol, name::Symbol, freqs::AbstractVector{<:Real})
    return rad2deg.(angle.(freqresp(ac, name, 2π .* freqs)))
end

"""
    magnitude_db(ac::ACSol, name::Symbol) -> Vector{Float64}

Magnitude in dB of node/current `name` over the frequency grid the analysis was
built with (`ac!(circuit, freqs)`); errors if `ac` carries no grid.
"""
function MNA.magnitude_db(ac::ACSol, name::Symbol)
    return 20 .* log10.(abs.(ac[name]))
end

"""
    phase_deg(ac::ACSol, name::Symbol) -> Vector{Float64}

Phase in degrees of node/current `name` over the frequency grid the analysis was
built with (`ac!(circuit, freqs)`). Errors if `ac` carries no grid.
"""
function MNA.phase_deg(ac::ACSol, name::Symbol)
    return rad2deg.(angle.(ac[name]))
end

"""
    _get_node_index(ac::ACSol, node::Symbol) -> Int

Get the system index for a node by name. Returns 0 for ground.
Returns `nothing` if not found (doesn't error).
"""
function _get_node_index(ac::ACSol, node::Symbol)
    (node === :gnd || node === Symbol("0")) && return 0
    idx = findfirst(==(node), ac.node_names)
    return idx  # nothing if not found
end

"""
    _get_current_index(ac::ACSol, name::Symbol) -> Int

Get the system index for a current variable by name.
Returns `nothing` if not found (doesn't error).
"""
function _get_current_index(ac::ACSol, name::Symbol)
    idx = findfirst(==(name), ac.current_names)
    idx === nothing && return nothing
    return ac.n_nodes + idx
end

"""
    _get_index(ac::ACSol, name::Symbol) -> Int

Get the system index for a node or current variable by name.
Tries node lookup first, then current lookup.
Returns 0 for ground, errors if not found.
"""
function _get_index(ac::ACSol, name::Symbol)
    # Try node first
    idx = _get_node_index(ac, name)
    idx !== nothing && return idx

    # Try current
    idx = _get_current_index(ac, name)
    idx !== nothing && return idx

    # Not found
    error("Unknown node or current: $name. Available nodes: $(ac.node_names), currents: $(ac.current_names)")
end

"""
    ac[name::Symbol] -> Vector{ComplexF64}

Name-based access to an AC solution: the complex response at node voltage or
branch current `name` over the frequency grid the analysis was built with
(`ac!(circuit, freqs)`). This is the SPICE-native readout and follows the same
`sol[:name]` convention as the DC and transient solutions, so all three index
alike.

Hierarchical subcircuit nodes flatten into the name table (`:x1_out`), so they
resolve here too; a `NodeRef` from `scope(...)` works as the index as well.

For the descriptor-state-space (ControlSystems) object of an output, use
[`subsystem`](@ref) instead. For evaluation at arbitrary angular frequencies
(without a stored grid), use [`freqresp`](@ref).

# Example
```julia
ac = ac!(circuit, acdec(20, 1, 1e6))   # freqs in Hz
resp = ac[:vout]                        # complex response over the grid
iv1  = ac[:I_V1]                        # source-current response
```
"""
function Base.getindex(ac::ACSol, name::Symbol)
    if isempty(ac.freqs)
        error("ACSol has no frequency grid: `ac[:$name]` needs one. Call " *
              "`ac!(circuit, freqs)` with a Hz grid (e.g. `acdec(20, 1, 1e6)`), " *
              "or use `freqresp(ac, :$name, ωs)` with angular frequencies.")
    end
    return freqresp(ac, name, 2π .* ac.freqs)
end

# Hierarchical access: a NodeRef (from `scope`) resolves to its flat name,
# mirroring the DCSolution / transient-solution NodeRef indexing.
Base.getindex(ac::ACSol, ref::MNA.NodeRef) = ac[MNA._flat_name(ref)]
freqresp(ac::ACSol, ref::MNA.NodeRef, ωs::AbstractVector{<:Real}) =
    freqresp(ac, MNA._flat_name(ref), ωs)

"""
    subsystem(ac::ACSol, name::Symbol) -> DescriptorStateSpace

Descriptor-state-space subsystem for observing a specific node voltage or branch
current: a SISO (single-input single-output) system from AC excitation to the
named output. Use it for the ControlSystems / DescriptorSystems interop —
`ss`, `bode`, poles/zeros.

# Example
```julia
ac_sol = ac!(circuit)
dss_vout = subsystem(ac_sol, :vout)     # node voltage
dss_iv1  = subsystem(ac_sol, :I_V1)     # voltage-source current
bode(ss(subsystem(ac_sol, :vout)), ωs)  # via ControlSystemsBase
```
"""
function subsystem(ac::ACSol, name::Symbol)
    idx = _get_index(ac, name)
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

Return value is a vector of frequencies in hertz (Hz). Pass it straight to
[`magnitude_db`](@ref)/[`phase_deg`](@ref); for [`freqresp`](@ref) (which takes
angular frequency in rad/s) convert first with `2π .* acdec(...)`.
"""
function acdec(nd, fstart, fstop)
    fstart = log10(fstart)
    fstop = log10(fstop)
    points = Int(ceil((fstop-fstart)*nd))+1
    return exp10.(range(fstart, stop=fstop, length=points))
end
