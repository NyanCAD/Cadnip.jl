#==============================================================================#
# Small-signal noise analysis
#
# Noise analysis reuses the AC linearization at the DC operating point (see
# `src/ac.jl` and `doc/noise_analysis_design.md`). Each registered noise source
# (the deferred channel on `MNAContext` populated during structure discovery) is
# a small-signal current source injected between two branch nodes, with a
# one-sided power spectral density `S_k(f)` set by its `NoiseKind`.
#
# For an output `y` (a node voltage or branch current) the transfer function
# from source `k` to `y` is `H_k(jω)`. Since the sources are uncorrelated, the
# output noise PSD is the incoherent sum
#
#     S_y(f) = Σ_k |H_k(jω)|² · S_k(f),   ω = 2πf.
#
# The transfer functions all share one factorization per frequency: solving the
# adjoint system `(jωC + G)ᵀ x_adj = e_y` once yields `H_k = x_adjᵀ b_k` for
# *every* source `k` at O(1) each, where `b_k = e_{p_k} − e_{n_k}` is the unit
# current injection of source `k`. This is the classic SPICE `.noise` inner loop
# (design doc §"The dual approach, reconciled").
#
# The analysis is source-agnostic: it consumes whatever sits on the noise
# channel, so every future source (diode/BJT shot, VA `white_noise`/
# `flicker_noise`) flows through unchanged once registered.
#==============================================================================#

using LinearAlgebra

export noise!, NoiseSol, onoise, total_noise

"""
    NoiseSol

Output of [`noise!`](@ref): the output-referred noise of a circuit over a Hz
frequency grid, decomposed into per-source contributions.

# Fields
- `freqs`: frequency grid in **hertz** (SPICE `.noise` convention)
- `output`: the observed output (node voltage or branch current) name
- `onoise`: total output noise PSD over the grid, in V²/Hz (voltage output) or
  A²/Hz (current output)
- `contributions`: per-source output-noise PSD, keyed by the originating device
  name; the values sum to `onoise` (incoherently)
- `temp`: temperature (Celsius) the PSDs were evaluated at
- `input`: the input source name the noise is referred back to, or `nothing`
  when no `input=` was requested
- `gain`: the input→output transfer function `H(jω)` over the grid, empty when
  no input was requested
- `inoise`: input-referred noise PSD `onoise / |gain|²`, in V²/Hz (voltage
  input source) or A²/Hz (current input source); empty when no input was
  requested

Index it like the other solutions: `ns[:onoise]` is the total output PSD,
`ns[:inoise]` is the input-referred PSD (when an `input=` was given), and
`ns[name]` is a single device's contribution. [`total_noise`](@ref) integrates
the band to an RMS value.
"""
struct NoiseSol
    freqs::Vector{Float64}
    output::Symbol
    onoise::Vector{Float64}
    contributions::Dict{Symbol,Vector{Float64}}
    temp::Float64
    input::Union{Symbol,Nothing}
    gain::Vector{ComplexF64}
    inoise::Vector{Float64}
end

"""
    noise!(circuit::MNA.MNACircuit, output; freqs, input=nothing, gmin=1e-12) -> NoiseSol

Small-signal noise analysis of `circuit`, computing the output-referred noise
spectral density at `output` (a node-voltage or branch-current name) over the Hz
grid `freqs`.

# Algorithm
1. Solve the DC operating point (Newton), then rebuild the linearized `G`/`C` at
   that point — exactly as [`ac!`](@ref) does.
2. Collect the noise sources registered on the context during that rebuild
   (resistor thermal `4kT·G`, and any device/VA source that registers itself).
3. Per frequency, one adjoint solve `(jωC + G)ᵀ x_adj = e_output` gives the
   transfer `H_k` from every source at O(1) each; accumulate the incoherent sum
   `S_out(f) = Σ_k |H_k|² S_k(f)` and keep each source's contribution. The same
   adjoint yields the input→output gain `H(jω)` for input-referral at no extra
   solve.

# Arguments
- `output`: node or current name to observe (e.g. `:vout`, `:I_V1`).
- `freqs`: frequency grid in **hertz**, e.g. `acdec(20, 1, 1e6)` (required).
- `input`: an independent **voltage-source** name (e.g. `:V1`) to refer the noise
  back to. When given, the result carries the input→output gain `H(jω)` and the
  input-referred PSD `inoise = onoise / |H|²` (V²/Hz for a voltage input),
  mirroring SPICE `.noise v(out) V1`. The gain is the unit-voltage transfer of
  the source, independent of any `AC` card on it. Defaults to `nothing`
  (output-referred only).
- `gmin`: shunt conductance added to `G` for numerical stability.

# Returns
[`NoiseSol`](@ref) — index `ns[:onoise]` for the total output PSD, `ns[:inoise]`
for the input-referred PSD (when `input=` was given), `ns[:r1]` for one source's
contribution (device names are normalized to lower case), and
[`total_noise`](@ref)`(ns)` for the band-integrated RMS.

# Example
```julia
circuit = MNACircuit(sp\"\"\"
V1 in 0 DC 0
R1 in out 1k
C1 out 0 1u
\"\"\"i)
ns = noise!(circuit, :out; freqs=acdec(10, 1, 1e6), input=:V1)
ns[:onoise]              # output voltage noise PSD (V²/Hz) over the grid
ns[:inoise]              # input-referred noise PSD (V²/Hz)
total_noise(ns)         # integrated RMS output noise voltage
total_noise(ns; referred=:input)   # integrated RMS input-referred noise
```
"""
function noise!(circuit::MNA.MNACircuit, output::Symbol;
                freqs::AbstractVector{<:Real}, input::Union{Symbol,Nothing}=nothing,
                gmin=1e-12)
    isempty(freqs) && throw(ArgumentError(
        "noise!(circuit, output; freqs=...) needs a non-empty Hz grid " *
        "(e.g. acdec(20, 1, 1e6))"))

    # Structure discovery, then DC operating point.
    ctx = MNA.MNAContext()
    circuit.builder(circuit.params, circuit.spec, 0.0; x=MNA.ZERO_VECTOR, ctx=ctx)
    dc_sol = MNA.solve_dc(circuit)

    # Rebuild the linearization (and noise channel) at the operating point.
    MNA.reset_for_restamping!(ctx)
    circuit.builder(circuit.params, circuit.spec, 0.0; x=dc_sol.x, ctx=ctx)

    G = Matrix(MNA.assemble_G(ctx; gshunt=gmin))
    C = Matrix(MNA.assemble_C(ctx))
    n = MNA.system_size(ctx)

    out_idx = _noise_output_index(ctx, output)

    srcs = MNA.noise_sources(ctx)
    isempty(srcs) && @warn "noise!: circuit has no registered noise sources; \
                            output noise is zero"

    # Input-referral: the unit-excitation column of the named input source. The
    # input→output gain is then `H = x_adjᵀ b_in`, sharing the noise adjoint.
    in_idx = input === nothing ? 0 : _noise_input_index(ctx, input)

    temp_c = circuit.spec.temp
    e_out = zeros(ComplexF64, n)
    e_out[out_idx] = 1.0

    fs = collect(Float64, freqs)
    onoise = zeros(Float64, length(fs))
    contributions = Dict{Symbol,Vector{Float64}}(
        s.name => zeros(Float64, length(fs)) for s in srcs)
    gain = input === nothing ? ComplexF64[] : Vector{ComplexF64}(undef, length(fs))
    inoise = input === nothing ? Float64[] : Vector{Float64}(undef, length(fs))

    for (fi, f) in enumerate(fs)
        ω = 2π * f
        # Adjoint solve: (jωC + G)ᵀ x_adj = e_out.
        x_adj = (transpose(im * ω .* C .+ G)) \ e_out
        for s in srcs
            p = MNA.resolve_index(ctx, s.p)
            q = MNA.resolve_index(ctx, s.n)
            # H_k = x_adjᵀ (e_p − e_n); ground (index 0) contributes nothing.
            Hk = (p == 0 ? zero(ComplexF64) : x_adj[p]) -
                 (q == 0 ? zero(ComplexF64) : x_adj[q])
            Sk = MNA.noise_psd(s, temp_c, f)
            contrib = abs2(Hk) * Sk
            onoise[fi] += contrib
            contributions[s.name][fi] += contrib
        end
        if input !== nothing
            # Same adjoint gives the input→output gain: H = x_adjᵀ b_in, where
            # b_in = e_{in_idx} is the unit-voltage injection of the source.
            H = x_adj[in_idx]
            gain[fi] = H
            inoise[fi] = iszero(H) ? Inf : onoise[fi] / abs2(H)
        end
    end

    return NoiseSol(fs, output, onoise, contributions, temp_c, input, gain, inoise)
end

"""
    _noise_output_index(ctx, name::Symbol) -> Int

System index of node-voltage or branch-current `name` in `ctx`. Errors for
ground or unknown names.
"""
function _noise_output_index(ctx::MNA.MNAContext, name::Symbol)
    (name === :gnd || name === Symbol("0")) &&
        error("noise!: output cannot be ground (node 0)")
    ni = findfirst(==(name), ctx.node_names)
    ni === nothing || return ni
    ci = findfirst(==(name), ctx.current_names)
    ci === nothing || return ctx.n_nodes + ci
    error("noise!: unknown output $name. Available nodes: $(ctx.node_names), " *
          "currents: $(ctx.current_names)")
end

"""
    _noise_input_index(ctx, input::Symbol) -> Int

System index of the unit-voltage excitation column for the independent
voltage source named `input`. A voltage source `V1` carries its branch equation
in the current-variable row `:I_V1`, so injecting a unit there is exactly a
1 V small-signal drive; the input→output transfer read from the adjoint is then
the AC voltage gain of the source. Errors if `input` is not a voltage source.
"""
function _noise_input_index(ctx::MNA.MNAContext, input::Symbol)
    # SPICE device names normalize to lower case, so accept `:V1` or `:v1`.
    for cand in (Symbol(:I_, input), Symbol(:I_, lowercase(String(input))))
        ci = findfirst(==(cand), ctx.current_names)
        ci === nothing || return ctx.n_nodes + ci
    end
    error(
        "noise!: input source $input is not an independent voltage source " *
        "(no current variable I_$input). Input-referred noise refers back to a " *
        "voltage source; available current variables: $(ctx.current_names)")
end

"""
    onoise(ns::NoiseSol) -> Vector{Float64}

Total output-noise PSD over the frequency grid (alias for `ns[:onoise]`).
"""
onoise(ns::NoiseSol) = ns.onoise

"""
    ns[name::Symbol] -> Vector{Float64}

Name-based access to a noise solution, mirroring `sol[:name]` for DC/AC/tran.
`ns[:onoise]` is the total output-noise PSD; `ns[:inoise]` is the input-referred
PSD (only when `noise!` was called with `input=`); any other symbol is looked up
as a noise-source (device) name and returns that source's contribution to the
output PSD.
"""
function Base.getindex(ns::NoiseSol, name::Symbol)
    name === :onoise && return ns.onoise
    if name === :inoise
        ns.input === nothing && error(
            "NoiseSol: no input-referred noise — call " *
            "`noise!(circuit, output; freqs, input=:V1)` to refer the noise " *
            "back to an input source.")
        return ns.inoise
    end
    haskey(ns.contributions, name) && return ns.contributions[name]
    error("NoiseSol: unknown key :$name. Use :onoise for the total, :inoise for \
           the input-referred noise, or a source name from \
           $(sort(collect(keys(ns.contributions)))).")
end

"""
    total_noise(ns::NoiseSol; referred=:output) -> Float64

Band-integrated RMS noise: `sqrt(∫ S(f) df)` over the analysis grid, by
trapezoidal integration. With `referred=:output` (default) it integrates the
output PSD `ns[:onoise]` (units V for a voltage output, A for a current output);
with `referred=:input` it integrates the input-referred PSD `ns[:inoise]` (units
of the input source), which requires `noise!` to have been called with `input=`.
For a meaningful integral supply a dense grid spanning the band of interest.
"""
function total_noise(ns::NoiseSol; referred::Symbol=:output)
    psd = referred === :output ? ns[:onoise] :
          referred === :input  ? ns[:inoise] :
          throw(ArgumentError("total_noise: referred must be :output or :input, got :$referred"))
    fs = ns.freqs
    length(fs) < 2 && return sqrt(length(fs) == 1 ? psd[1] : 0.0)
    acc = 0.0
    for i in 1:length(fs)-1
        acc += 0.5 * (psd[i] + psd[i+1]) * (fs[i+1] - fs[i])
    end
    return sqrt(acc)
end
