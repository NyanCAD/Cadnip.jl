#==============================================================================#
# Declared parameter names
#
# An override that names nothing is inert: `alter(circuit; vbais=1.2)` runs, and
# every point of the sweep comes back with the netlist default. Codegen already
# knows every name the netlist declares, so it emits that as a table beside the
# builder and the override tuple is checked against it — no builder pass, no
# circuit construction (`MNACircuit` is a plain struct that `alter` rebuilds per
# sweep point, so validation has to stay free of both).
#
# One table per *scope* — the top level and each `.subckt` — not per instance,
# so the tables are sized by the netlist text rather than by the flattened
# design.
#==============================================================================#

"""
    ParamScope(params, children, devices)

The names one generated builder's scope declares:

- `params`: its own `.param` / formal parameter names — the leaves an override
  may set.
- `children`: instance name => the builder of the subcircuit it instantiates —
  the groups an override may descend into. A `nothing` builder means the
  subcircuit came from somewhere this table cannot see, so the subtree is
  accepted unchecked.
- `devices`: device instance names. Known to the netlist, but not addressable by
  an override — see `doc/parameter_overrides.md` §1.

`declared_params(builder)` returns the scope of a generated builder, and
`nothing` for a hand-written one — which declares nothing, so nothing is
checked.
"""
struct ParamScope
    params::Set{Symbol}
    children::Dict{Symbol,Any}
    devices::Set{Symbol}
end

ParamScope(params::AbstractVector, children::AbstractVector, devices::AbstractVector) =
    ParamScope(Set{Symbol}(params), Dict{Symbol,Any}(children), Set{Symbol}(devices))

"""
    DeclaredParams()

Marker argument: a generated builder answers `builder(DeclaredParams())` with
its `ParamScope`.

The table rides on the builder as an extra *method of the builder itself*
rather than as a method of `declared_params`, because generated code routinely
lands in a local scope — `@testset begin circuit = sp"..." end` — where adding
a method to a function defined elsewhere is a syntax error, while adding one to
the function being defined right there is not.
"""
struct DeclaredParams end

"""
    declared_params(builder) -> ParamScope | nothing

The parameter scope a generated circuit builder declares, or `nothing` for a
builder that carries no table — a hand-written one, above all — which switches
override checking off.
"""
function declared_params(@nospecialize(builder))
    # `invokelatest`, because `MNACircuit(path)` evals its builder inside a
    # function body: the table method is newer than the world we are called in.
    # This is a once-per-construction check, never a solve-loop cost.
    Base.invokelatest(applicable, builder, DeclaredParams()) || return nothing
    scope = Base.invokelatest(builder, DeclaredParams())
    return isa(scope, ParamScope) ? scope : nothing
end

"""
    check_override_names(builder, params)

Throw an `ArgumentError` if `params` names a parameter, an instance, or a path
that `builder`'s netlist does not declare. A silent no-op when the builder has
no declared-name table (hand-written builders) or `params` is not a NamedTuple.
"""
check_override_names(@nospecialize(builder), @nospecialize(params)) = nothing

function check_override_names(@nospecialize(builder), params::NamedTuple)
    isempty(params) && return nothing
    scope = declared_params(builder)
    scope === nothing && return nothing
    _check_scope(scope, params, Symbol[])
    return nothing
end

# Walks the override tuple in the *compact* shape the user writes it in, under
# the same rule the lens reads it by: a leaf is a parameter of this scope, a
# group is a child, and an explicit `params = (...)` names parameters even when
# the name collides with an instance.
function _check_scope(scope::ParamScope, nt::NamedTuple, path::Vector{Symbol})
    for name in keys(nt)
        v = getfield(nt, name)
        if name === :params && isa(v, NamedTuple)
            for pname in keys(v)
                _check_param(scope, pname, path)
            end
        elseif isa(v, NamedTuple)
            _check_child(scope, name, v, path)
        else
            _check_param(scope, name, path)
        end
    end
    return nothing
end

function _check_param(scope::ParamScope, name::Symbol, path::Vector{Symbol})
    name in scope.params && return nothing
    if haskey(scope.children, name)
        _override_error(path, name,
            "`$name` names a subcircuit instance, not a parameter",
            "Write `$(_spell(path, name)) = (inner_param = value,)` to set a parameter inside it")
    elseif name in scope.devices
        _device_error(path, name)
    else
        _unknown_error(scope, path, name)
    end
end

function _check_child(scope::ParamScope, name::Symbol, sub::NamedTuple, path::Vector{Symbol})
    if haskey(scope.children, name)
        child = scope.children[name]
        child === nothing && return nothing   # opaque subcircuit: accept unchecked
        cscope = declared_params(child)
        cscope === nothing && return nothing
        return _check_scope(cscope, sub, [path; name])
    elseif name in scope.devices
        _device_error(path, name)
    elseif name in scope.params
        _override_error(path, name,
            "`$name` is a parameter of this scope, not an instance",
            "Write `$(_spell(path, name)) = value`")
    else
        _unknown_error(scope, path, name)
    end
end

_spell(path::Vector{Symbol}, name::Symbol) =
    isempty(path) ? string(name) : string(join(path, "."), ".", name)

_scope_desc(path::Vector{Symbol}) =
    isempty(path) ? "the top level" : "instance `$(join(path, "."))`"

function _override_error(path, name, what, hint)
    throw(ArgumentError(string(
        "parameter override `", _spell(path, name), "` — ", what, ". ", hint, ".")))
end

function _device_error(path, name)
    throw(ArgumentError(string(
        "parameter override `", _spell(path, name), "` — `", name,
        "` names a device instance, and device instance parameters are not ",
        "overridable. Give the netlist a `.param` and reference it from the ",
        "device line instead.")))
end

function _unknown_error(scope::ParamScope, path, name)
    declared = sort!(collect(scope.params))
    known = isempty(declared) ? "It declares no parameters at all." :
                                string("It declares: ", join(declared, ", "), ".")
    throw(ArgumentError(string(
        "unknown parameter override `", _spell(path, name), "` — ",
        _scope_desc(path), " declares no parameter `", name, "`. ", known)))
end
