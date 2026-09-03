#==============================================================================#
# DC sensitivity analysis
#
# How much does one operating-point reading move when one parameter moves?
# SPICE spells the question `.sens v(out)`; here it is `sens!(circuit, :out)`.
#
# The operating point is the root of the MNA residual
#
#     F(x, p) = G(x, p)·x − b(x, p) = 0
#
# — the same residual Newton drives to zero in `_dc_newton_compiled`, with `G`
# the linearization the companion models stamp, which is also its Jacobian.
# Differentiating the root with respect to a parameter `p` (the implicit
# function theorem) gives
#
#     ∂F/∂x · dx/dp + ∂F/∂p = 0   ⟹   dx/dp = −G⁻¹ · ∂F/∂p.
#
# One output `y = eᵀx` needs no `dx/dp` at all: transposing moves the single
# linear solve off the parameter loop,
#
#     dy/dp = −eᵀ G⁻¹ ∂F/∂p = −λᵀ ∂F/∂p,   where   Gᵀλ = e.
#
# So the cost is *one* factorization for the output, and per parameter only the
# cheap part: `∂F/∂p`, a residual evaluated at the frozen operating point with
# the parameter nudged. That is the same adjoint trick `noise!` uses to get
# every noise source's transfer function out of one solve per frequency, and the
# reason a hundred-parameter sensitivity costs about what a hundred residual
# stampings cost, not a hundred DC solves.
#
# `∂F/∂p` is taken by central differences over a *rebuild*, not by AD: the
# parameter enters through the generated builder, and the context it stamps into
# is `Float64`-typed, so a dual number cannot flow through it. The differencing
# is on the residual, not on the solve — no Newton iteration is repeated, no
# convergence tolerance leaks into the derivative — which is what keeps it
# accurate. Measured against a brute-force re-solve of the whole operating point
# it agrees to ~1e-9 relative; see `doc/sensitivity_design.md`.
#
# What can be differentiated is what the netlist *declares*: `.param` values, at
# the top level or inside a subcircuit, and subcircuit instance parameters —
# the same names `alter` and a sweep axis address, discovered the same way
# (`observed_params`, src/param_overrides.jl). A device instance parameter
# (`r1=(r=2k,)`) is not reachable, here or anywhere else yet; parameterize the
# netlist with a `.param` and sweep that.
#==============================================================================#

using LinearAlgebra
using SparseArrays
using Printf

export sens!, SensSolution, normalized_sensitivity

"""
    SensSolution

Output of [`sens!`](@ref): the DC sensitivity of one operating-point reading to
each of a circuit's parameters.

# Fields
- `output`: the observed node voltage or branch current
- `names`: parameter selectors, spelled as [`alter`](@ref) spells them — `:rs`
  at the top level, `Symbol("x1.rv")` inside instance `X1`
- `values`: the value each parameter was differentiated at (the override in
  effect, else the netlist default)
- `sens`: `∂output/∂param`, in output units per parameter unit
- `op`: the [`DCSolution`](@ref) it linearized at

Index it by parameter name for the absolute sensitivity, and read the
per-percent column with [`normalized_sensitivity`](@ref):

```julia
s = sens!(circuit, :out)
s[:rs]                        # V per Ω
normalized_sensitivity(s, :rs)  # V per 1% change in rs
```
"""
struct SensSolution
    output::Symbol
    names::Vector{Symbol}
    values::Vector{Float64}
    sens::Vector{Float64}
    op::MNA.DCSolution
end

"""
    sens!(circuit::MNA.MNACircuit, output::Symbol; params=nothing, step=cbrt(eps())) -> SensSolution

DC sensitivity of `output` — a node voltage or branch current at the operating
point — to the circuit's parameters, the analysis SPICE spells `.sens v(out)`.

# Algorithm
1. Solve the DC operating point, then rebuild the linearization `G` there.
2. Solve the adjoint system `Gᵀλ = e_output` **once**.
3. Per parameter, re-stamp the residual at that frozen operating point with the
   parameter nudged either way, and read `dy/dp = −λᵀ ∂F/∂p` off the adjoint.

Step 3 costs a stamping, not a DC solve, so the whole analysis is one Newton
solve plus one factorization however many parameters it covers.

# Arguments
- `output`: node or current name to observe (`:vout`, `:I_V1`), the same
  namespace `sol[:name]` indexes.
- `params`: which parameters to differentiate against. Defaults to *every*
  parameter the circuit declares — the netlist's `.param` cards at the top
  level, each subcircuit's own parameters, and the instance parameters on its
  `X` lines. Pass an iterable of selectors (`[:rs, Symbol("x1.rv")]`) to pick
  some of those, or a `name => value` mapping (`(R1 = 1e3, R2 = 2e3)`) to state
  the parameters and the values they sit at outright — which is how a
  hand-written builder, whose parameters nothing can enumerate, gets covered.
- `step`: relative finite-difference step for `∂F/∂p`; the default
  `cbrt(eps())` is the usual optimum for a central difference.
- `gmin`: shunt conductance added to the adjoint matrix, as [`ac!`](@ref) and
  [`noise!`](@ref) add it to theirs — it is what keeps a circuit with a floating
  island (whose `G` is singular, even though its operating point solves) from
  having no adjoint at all. At the default `1e-12` it perturbs a sensitivity far
  below the finite-difference noise.

# Returns
[`SensSolution`](@ref) — `s[:rs]` for one absolute sensitivity,
[`normalized_sensitivity`](@ref)`(s)` for the per-percent column SPICE prints
alongside it, `pairs(s)` to walk them all.

# Example
```julia
circuit = MNACircuit(sp\"\"\"
.param rtop=1k
.param rbot=1k
V1 vcc 0 DC 5
R1 vcc out 'rtop'
R2 out 0 'rbot'
\"\"\"i)
s = sens!(circuit, :out)
s[:rtop]                          # -1.25e-3 V/Ω
normalized_sensitivity(s, :rtop)  # -12.5 mV per 1% of rtop
```

Only a *DC* sensitivity: it differentiates the operating point, so a capacitor
or inductor value shows up as zero unless it moves the DC solution.
"""
function sens!(circuit::MNA.MNACircuit, output::Symbol;
               params=nothing, step::Real=cbrt(eps(Float64)), gmin::Real=1e-12)
    step > 0 || throw(ArgumentError("sens!: `step` must be positive, got $step"))

    # The operating point, in the mode `dc!` uses: transient sources at their DC
    # values, not their t=0 waveform.
    dcc = MNA.with_mode(circuit, :dcop)
    op = MNA.solve_dc(dcc)
    op.converged || @warn "sens!: DC operating point did not converge; " *
                          "the sensitivities are taken at whatever point Newton stopped on"
    x = op.x
    isempty(x) && throw(ArgumentError("sens!: circuit has no unknowns to observe"))

    # One scratch context, reused for the base linearization and every perturbed
    # residual: allocating nodes once keeps the parameter loop to pure stamping.
    ctx = MNA.MNAContext()
    dcc.builder(dcc.params, dcc.spec, 0.0; x=MNA.ZERO_VECTOR, ctx=ctx)
    MNA.reset_for_restamping!(ctx)
    dcc.builder(dcc.params, dcc.spec, 0.0; x=x, ctx=ctx)

    n = MNA.system_size(ctx)
    G = MNA.assemble_G(ctx; gshunt=Float64(gmin))
    out_idx = _sens_output_index(ctx, output)

    selectors = _sens_parameters(dcc, params)

    # The adjoint: one solve, reused by every parameter.
    e = zeros(Float64, n)
    e[out_idx] = 1.0
    λ = _adjoint(G, e, output)

    names = Symbol[]
    at = Float64[]
    sens = Float64[]
    for (selector, value) in selectors
        # Relative step, except for a parameter sitting at exactly zero, where a
        # relative step is zero and there is nothing to scale against.
        h = iszero(value) ? Float64(step) : step * abs(value)
        Fp = _residual(ctx, dcc, selector, value + h, x, n)
        Fm = _residual(ctx, dcc, selector, value - h, x, n)
        # dF/dp by central difference, then the adjoint contraction.
        dydp = 0.0
        @inbounds for i in 1:n
            dydp -= λ[i] * (Fp[i] - Fm[i])
        end
        push!(names, selector)
        push!(at, value)
        push!(sens, dydp / (2h))
    end

    return SensSolution(output, names, at, sens, op)
end

# System row of the observed output, with `sens!`'s own diagnostics around the
# shared lookup.
function _sens_output_index(ctx::MNA.MNAContext, name::Symbol)
    idx = MNA.state_index(ctx, name)
    idx === 0 && error("sens!: output cannot be ground (node 0) — its voltage " *
                       "is zero by definition, so every sensitivity would be too")
    idx === nothing &&
        error("sens!: unknown output $name. Available nodes: $(ctx.node_names), " *
              "currents: $(ctx.current_names)")
    return idx
end

# Solve `Gᵀλ = e`. A singular `G` here is the same floating-node/degenerate
# circuit that would have made the operating point itself meaningless, so say so
# rather than handing back a `SingularException` from three layers down.
function _adjoint(G::AbstractMatrix, e::Vector{Float64}, output::Symbol)
    try
        return transpose(G) \ e
    catch err
        err isa Union{SingularException,LinearAlgebra.LAPACKException} || rethrow()
        error("sens!: the linearized circuit is singular at the operating " *
              "point even with the gmin shunt, so the sensitivity of $output " *
              "is not defined. A loop of voltage sources or a shorted source " *
              "is the usual cause — a floating node the shunt already covers, " *
              "and a larger `gmin=` covers a worse-conditioned one.")
    end
end

# The residual `F(x*, p) = G·x* − b` of the circuit with one parameter moved,
# evaluated at the *unperturbed* operating point. `F` is zero there for the
# unperturbed parameters, which is what makes the difference of two of these a
# clean `∂F/∂p`.
function _residual(ctx::MNA.MNAContext, circuit::MNA.MNACircuit, selector::Symbol,
                   value::Float64, x::Vector{Float64}, n::Int)
    perturbed = MNA.alter(circuit; (selector => value,)...)
    MNA.reset_for_restamping!(ctx)
    perturbed.builder(perturbed.params, perturbed.spec, 0.0; x=x, ctx=ctx)
    MNA.system_size(ctx) == n || error(
        "sens!: parameter `$selector` changes the size of the system " *
        "($(MNA.system_size(ctx)) unknowns, was $n), so there is no single " *
        "operating point to differentiate. A parameter that selects a `.if` " *
        "branch or an instance count is a discrete choice, not a derivative; " *
        "sweep it instead.")
    return MNA.assemble_G(ctx) * x .- MNA.get_rhs(ctx)
end

#==============================================================================#
# Which parameters
#
# The names come from the same observation `alter` validates overrides against
# (`observed_params`), so what is differentiable is exactly what is overridable,
# and the observed defaults give the *value* to nudge — unless the circuit
# carries an override for that name, which is then the value in effect.
#==============================================================================#

# `[(selector, value)]` for the requested parameters, or for every declared one.
#
# `params` names parameters three ways, in rising order of how much the caller
# has to know: `nothing` takes every declared one, a list of selectors picks
# some of them (values still read off the circuit), and a `name => value`
# mapping states the values outright — which is the only form a hand-written
# builder can use, since nothing can observe what it declares.
function _sens_parameters(circuit::MNA.MNACircuit, requested)
    if requested isa NamedTuple
        return Tuple{Symbol,Float64}[(k, Float64(getproperty(requested, k)))
                                     for k in keys(requested)]
    end
    if requested !== nothing
        # Collected once: `requested` may be a generator, and it is walked twice
        # below (once to classify it, once to read it).
        requested = collect(requested)
        if !isempty(requested) && all(p -> p isa Pair, requested)
            return Tuple{Symbol,Float64}[(Symbol(k), Float64(v)) for (k, v) in requested]
        end
    end

    tree = observed_params(circuit.builder)
    tree === nothing && throw(ArgumentError(
        "sens!: nothing declares this circuit's parameters, so `sens!` cannot " *
        "find them or the values they sit at. A netlist parameterized with " *
        "`.param` cards (or a `.subckt` with parameters) is what it reads; a " *
        "hand-written builder that takes `params` as a plain NamedTuple is " *
        "opaque to that, so state the parameters and their values yourself: " *
        "`params = (R1 = 1e3, R2 = 2e3)`."))

    overrides = canonicalize_params(circuit.params)
    declared = Tuple{Symbol,Float64}[]
    _collect_params!(declared, tree, overrides, Symbol[])
    sort!(declared; by=first)

    if requested === nothing
        isempty(declared) && throw(ArgumentError(
            "sens!: this circuit declares no parameters, so there is nothing " *
            "to differentiate against. Give the netlist a `.param` for what " *
            "you want the sensitivity to (`.param rd=10k`, then `Rd vdd " *
            "drain rd`) — a device value written inline is not a parameter of " *
            "anything."))
        return declared
    end

    known = Dict(declared)
    out = Tuple{Symbol,Float64}[]
    for sel in requested
        selector = Symbol(sel)
        haskey(known, selector) || throw(ArgumentError(
            "sens!: unknown parameter `$selector`. This circuit declares: " *
            (isempty(declared) ? "no parameters at all." :
             string(join(first.(declared), ", "), "."))))
        push!(out, (selector, known[selector]))
    end
    return out
end

# Walk the observed tree — this scope's parameters under `:params`, one child
# per instantiated subcircuit — flattening to the dotted selectors `alter`
# understands.
function _collect_params!(out::Vector{Tuple{Symbol,Float64}}, tree::Dict,
                          overrides::NamedTuple, path::Vector{Symbol})
    own = get(tree, :params, nothing)
    if own !== nothing
        for (name, default) in own
            value = _effective_value(overrides, path, name, default)
            # A non-real parameter (a string-valued `.param`, say) has no
            # derivative; skip it rather than failing the whole analysis.
            value isa Real || continue
            push!(out, (_selector(path, name), Float64(value)))
        end
    end
    for (name, child) in tree
        name === :params && continue
        child isa ParamObserver || continue
        _collect_params!(out, getfield(child, :params), overrides, [path; name])
    end
    return out
end

_selector(path::Vector{Symbol}, name::Symbol) =
    isempty(path) ? name : Symbol(join(path, "."), ".", name)

# The value in effect for one parameter: the circuit's own override where it
# carries one, the netlist default otherwise. `overrides` is canonical, so a
# scope's parameters are under `params` and its children are the other fields.
function _effective_value(overrides::NamedTuple, path::Vector{Symbol},
                          name::Symbol, default)
    scope = overrides
    for step in path
        scope = get(scope, step, nothing)
        scope isa NamedTuple || return default
    end
    own = get(scope, :params, nothing)
    (own isa NamedTuple && haskey(own, name)) ? getproperty(own, name) : default
end

#==============================================================================#
# Readout
#==============================================================================#

"""
    s[name::Symbol] -> Float64

Absolute DC sensitivity `∂output/∂name`, in output units per parameter unit
(volts per ohm, volts per volt, amps per ohm...). Follows the same `sol[:name]`
convention as the DC, AC and transient solutions.
"""
function Base.getindex(s::SensSolution, name::Symbol)
    i = findfirst(==(name), s.names)
    i === nothing && error(
        "sens!: no sensitivity for `$name` in this result. It covers: " *
        (isempty(s.names) ? "no parameters." : string(join(s.names, ", "), ".")))
    return s.sens[i]
end

Base.keys(s::SensSolution) = copy(s.names)
Base.values(s::SensSolution) = copy(s.sens)
Base.pairs(s::SensSolution) = [n => v for (n, v) in zip(s.names, s.sens)]
Base.haskey(s::SensSolution, name::Symbol) = findfirst(==(name), s.names) !== nothing
Base.get(s::SensSolution, name::Symbol, default) = haskey(s, name) ? s[name] : default
Base.length(s::SensSolution) = length(s.names)
Base.iterate(s::SensSolution, state=1) =
    state > length(s) ? nothing : ((s.names[state] => s.sens[state]), state + 1)
Base.eltype(::Type{SensSolution}) = Pair{Symbol,Float64}

"""
    normalized_sensitivity(s::SensSolution) -> Vector{Pair{Symbol,Float64}}
    normalized_sensitivity(s::SensSolution, name::Symbol) -> Float64

Sensitivity per *percent* of the parameter — `∂output/∂p · p/100` — which is
SPICE's `NORMALIZED SENSITIVITY` column and the one worth comparing across
parameters: it puts a 1 kΩ resistor and a 5 V supply on the same footing, where
the absolute column has them in incomparable units.

A parameter sitting at zero normalizes to zero, since a percentage of zero is.

```julia
s = sens!(circuit, :out)
sort!(normalized_sensitivity(s); by=p -> -abs(p.second))   # what to tighten first
```
"""
normalized_sensitivity(s::SensSolution) =
    [n => d * v / 100 for (n, d, v) in zip(s.names, s.sens, s.values)]

function normalized_sensitivity(s::SensSolution, name::Symbol)
    i = findfirst(==(name), s.names)
    i === nothing && error(
        "sens!: no sensitivity for `$name` in this result. It covers: " *
        (isempty(s.names) ? "no parameters." : string(join(s.names, ", "), ".")))
    return s.sens[i] * s.values[i] / 100
end

function Base.show(io::IO, ::MIME"text/plain", s::SensSolution)
    println(io, "SensSolution: ∂", s.output, " / ∂p at the DC operating point",
            s.op.converged ? "" : " (NOT CONVERGED)")
    if isempty(s.names)
        print(io, "  (no parameters)")
        return nothing
    end
    w = max(9, maximum(length ∘ String, s.names))
    println(io, "  ", rpad("parameter", w), "  ", lpad("value", 12), "  ",
            lpad("∂out/∂p", 12), "  ", lpad("per %", 12))
    for (n, val, d) in zip(s.names, s.values, s.sens)
        println(io, "  ", rpad(String(n), w), "  ",
                lpad(_sens_fmt(val), 12), "  ",
                lpad(_sens_fmt(d), 12), "  ",
                lpad(_sens_fmt(d * val / 100), 12))
    end
    return nothing
end

# Six significant digits: enough to read a sensitivity off, few enough that the
# finite-difference noise in the last few bits does not crowd the table.
_sens_fmt(v::Float64) = iszero(v) ? "0" : Printf.@sprintf("%.6g", v)
