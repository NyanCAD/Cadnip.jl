#==============================================================================#
# Overrides that name nothing
#
# An undeclared override is inert: `alter(circuit; vbais=1.2)` runs, and every
# point of the sweep comes back with the netlist default — a typo reads as a
# parameter with no effect, and a dead sweep axis reads as a flat curve rather
# than an error.
#
# The names a circuit declares are already discoverable: `ParamObserver` is an
# `AbstractParamLens` that records, instead of overriding, so building once with
# one in place of `ParamLens` yields the whole tree — this scope's parameters
# under `:params`, each instantiated subcircuit under its instance name,
# recursively. Dispatch keeps that entirely off the hot path: the builder calls
# the lens generically, the transient path still gets the `@generated`
# `ParamLens` that folds away.
#
# One observation per builder, memoized, so `alter` — which rebuilds an
# `MNACircuit` per sweep point — pays for it once.
#==============================================================================#

const _OBSERVED_PARAMS = IdDict{Any,Any}()
const _OBSERVED_LOCK = ReentrantLock()

"""
    observed_params(builder) -> Dict{Symbol,Any} | nothing

The tree of names `builder` declares, discovered by building it once with a
`ParamObserver`. `nothing` when the builder cannot be observed, or declares
nothing at all — either way there is nothing to check an override against.

Memoized on the builder, because `alter` reconstructs an `MNACircuit` at every
sweep point and the answer cannot change between them.
"""
function observed_params(@nospecialize(builder))
    @lock _OBSERVED_LOCK get!(_OBSERVED_PARAMS, builder) do
        tree = try
            👀 = ParamObserver()
            # `invokelatest`: `MNACircuit(path)` evals its builder inside a
            # function body, so the builder is newer than our world.
            Base.invokelatest(builder, 👀, MNA.MNASpec())
            getfield(👀, :params)
        catch
            # A builder that reads `params` as a plain NamedTuple (any
            # hand-written one) throws on the observer. It knows what its own
            # parameters mean; we don't, so we check nothing.
            nothing
        end
        # An empty tree means the builder never consulted the lens — a
        # hand-written builder, or a netlist with no `.param` and no subcircuit
        # instance, which has no knob to typo in the first place.
        (tree === nothing || isempty(tree)) ? nothing : tree
    end
end

function MNA.check_override_names(@nospecialize(builder), params::NamedTuple)
    isempty(params) && return nothing
    declared = observed_params(builder)
    declared === nothing && return nothing
    _check_scope(declared, params, Symbol[])
    return nothing
end

# Walks the override tuple in the *compact* shape the user writes it in, under
# the same rule the lens reads it by: a leaf is a parameter of this scope, a
# group is a child, and an explicit `params = (...)` names parameters even when
# the name collides with an instance.
#
# The observation ran on the netlist's own defaults, so a scope reached only
# through a `.if` the overrides themselves would select is not in the tree. That
# is the same blind spot `alter` has had all along, and it fails safe: such a
# name reports as unknown rather than silently doing nothing.
function _check_scope(obs::Dict, nt::NamedTuple, path::Vector{Symbol})
    own = get(obs, :params, nothing)
    for name in keys(nt)
        v = getfield(nt, name)
        if isa(v, AbstractParamLens)
            # A lens passed as a value is a hand-written builder injecting its
            # own lens (`params.lens(; R=…)`). Observing such a builder mints a
            # phantom child scope for that name, so there is nothing to check
            # it against — and a lens addresses whatever the builder does with
            # it, not a name we could validate anyway.
            continue
        elseif name === :params && isa(v, NamedTuple)
            for pname in keys(v)
                _check_param(obs, own, pname, path)
            end
        elseif isa(v, NamedTuple)
            _check_child(obs, own, name, v, path)
        else
            _check_param(obs, own, name, path)
        end
    end
    return nothing
end

_declares(own, name) = own !== nothing && haskey(own, name)
_instantiates(obs, name) = name !== :params && haskey(obs, name)

function _check_param(obs::Dict, own, name::Symbol, path::Vector{Symbol})
    _declares(own, name) && return nothing
    if _instantiates(obs, name)
        _override_error(path, name,
            "`$name` names a subcircuit instance, not a parameter",
            "Write `$(_spell(path, name)) = (inner_param = value,)` to set a parameter inside it")
    end
    _unknown_error(own, path, name)
end

function _check_child(obs::Dict, own, name::Symbol, sub::NamedTuple, path::Vector{Symbol})
    if _instantiates(obs, name)
        return _check_scope(getfield(obs[name], :params), sub, [path; name])
    elseif _declares(own, name)
        _override_error(path, name,
            "`$name` is a parameter of this scope, not an instance",
            "Write `$(_spell(path, name)) = value`")
    end
    _unknown_error(own, path, name)
end

_spell(path::Vector{Symbol}, name::Symbol) =
    isempty(path) ? string(name) : string(join(path, "."), ".", name)

_scope_desc(path::Vector{Symbol}) =
    isempty(path) ? "the top level" : "instance `$(join(path, "."))`"

_override_error(path, name, what, hint) =
    throw(ArgumentError(string(
        "parameter override `", _spell(path, name), "` — ", what, ". ", hint, ".")))

function _unknown_error(own, path, name)
    declared = own === nothing ? Symbol[] : sort!(collect(keys(own)))
    known = isempty(declared) ? "It declares no parameters at all." :
                                string("It declares: ", join(declared, ", "), ".")
    throw(ArgumentError(string(
        "unknown parameter override `", _spell(path, name), "` — ",
        _scope_desc(path), " declares no parameter `", name, "`. ", known)))
end
