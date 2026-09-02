#==============================================================================#
# Netlist introspection — "what is `r1`?"
#
# A deck is a namespace, and SPICE puts five kinds of thing in it: instances,
# parameters, models, subcircuits and nets. One name can be several of them at
# once (`.param x1` next to `X1`), which is exactly why asking is worth
# something. `netlist(circuit).r1` answers which, and prints the card that
# defines it, with the file and line it came from.
#
#     julia> nl = netlist(circuit)
#     SPICE netlist "voltage divider"
#       instances (3): v1, r1, r2
#       parameters (1): res
#       nets (3): 0, out, vcc
#
#     julia> nl.r1
#     Instance r1 (Resistor) — vcc, out
#     ╭ divider.sp:4
#     │ R1 vcc out {res}
#     ╰
#
# The index is built once, at codegen time, out of the `SemaResult`, and emitted
# as a `const` in the deck's own module. It holds nothing but `Symbol`s, `Int`s
# and `String`s copied out of the source, so it does not keep the CST alive after
# the deck is compiled.
#==============================================================================#

"""
    Card

One netlist line that defines a name, as written, with where it was written.

A name can have more than one card: `.if`/`.else` branches each define the
instance, and a later `.param` redefines an earlier one.
"""
struct Card
    file::Symbol
    line::Int
    text::String
end

"""
    CircRef

What a name in a netlist refers to. One of `InstanceRef`, `ParamRef`,
`ModelRef`, `SubcktRef`, `NetRef`, or `AmbiguousRef` when a name is several of
those at once.
"""
abstract type CircRef end

"""
    InstanceRef

A device or subcircuit instance. `device` is the netlist card that produced it
(`:Resistor`, `:SubcktCall`, `:MOSFET`, ...) and `nets` are the nodes it
connects, in card order.
"""
struct InstanceRef <: CircRef
    name::Symbol
    device::Symbol
    nets::Vector{Symbol}
    cards::Vector{Card}
end

"""
    ParamRef

A `.param` (or Spectre `parameters`) declaration. This is the kind of name
`alter`, `MNACircuit(...; name=value)` and a sweep axis address.
"""
struct ParamRef <: CircRef
    name::Symbol
    cards::Vector{Card}
end

"""
    ModelRef

A `.model` card. `device` is the `GlobalRef` the two-tier model resolution
picked for it, or `nothing` when nothing claimed the card.
"""
struct ModelRef <: CircRef
    name::Symbol
    device::Union{GlobalRef, Nothing}
    cards::Vector{Card}
end

"""
    NetRef

A node. Nets are not declared, so a `NetRef` carries no card — it carries the
instances that connect to it, which is what a node *is* in a netlist.
"""
struct NetRef <: CircRef
    name::Symbol
    instances::Vector{Symbol}
end

using SymbolicIndexingInterface
SymbolicIndexingInterface.symbolic_type(::NetRef) = ScalarSymbolic()
SymbolicIndexingInterface.symbolic_type(::Type{<:NetRef}) = ScalarSymbolic()

"""
    NetlistIndex

Every name a deck (or one `.subckt` of it) declares, and what each one refers
to. Reached with `netlist(circuit)`.

Names are addressed with `.` or `[]` — `nl.r1`, `nl[:r1]` — case-insensitively,
as SPICE resolves them. `keys`, `haskey`, `get` and `propertynames` enumerate.
The struct's own fields are deliberately *not* reachable through `.`: a netlist
is free to name a device `title`, and the name it wrote wins.
"""
struct NetlistIndex
    title::Union{Nothing, String}
    lang::Symbol
    entries::OrderedDict{Symbol, CircRef}
end

"""
    SubcktRef

A `.subckt` definition. `ports` are its formal terminals, and `index` is the
subcircuit's own namespace: `netlist(c).divider[:r1]` reaches inside it.
"""
struct SubcktRef <: CircRef
    name::Symbol
    ports::Vector{Symbol}
    cards::Vector{Card}
    index::NetlistIndex
end

"""
    AmbiguousRef

A name that is more than one kind of thing. `.param x1` next to an `X1`
instance line is the case the parameter lens has its own rule for; this is how
you see it.
"""
struct AmbiguousRef <: CircRef
    name::Symbol
    refs::Vector{CircRef}
end

#==============================================================================#
# Building the index from sema
#==============================================================================#

# `SNode{SP.Resistor}` -> `:Resistor`, and `SNode{SP.ControlledSource{:V,:V}}`
# -> `:ControlledSource`. The card kind is what a reader recognises; the type
# parameters are codegen's business.
_card_kind(node::SNode) = Base.typename(typeof(node).parameters[1]).name

# `header_only` is for a card that spans a body — a `.subckt` node's text is the
# whole subcircuit, and the line that names it is the one worth printing. What
# is inside is the ref's own nested index, not its card.
function _card(node::SNode; header_only::Bool=false)
    lnn = Base.LineNumberNode(node)
    file = something(lnn.file, Symbol("unknown"))
    text = strip(String(node))
    header_only && (text = first(split(text, '\n')))
    return Card(file, lnn.line, text)
end

# Sema stores every definition of a name as `position => MaybeConditional`, in
# source order. All of them are worth showing: an instance defined in both arms
# of an `.if` has two cards and no single "the" definition.
_defs(entry) = [d[2].val for d in entry]

function _instance_nets(node::SNode)
    applicable(sema_nets, node) || return Symbol[]
    return Symbol[LSymbol(n) for n in sema_nets(node)]
end

# A name already in the index is not an error — SPICE lets one name be a
# parameter and an instance at once — so the second arrival folds both into an
# `AmbiguousRef` rather than overwriting.
function _add!(entries::OrderedDict{Symbol, CircRef}, ref::CircRef)
    name = ref.name
    prev = get(entries, name, nothing)
    if prev === nothing
        entries[name] = ref
    elseif prev isa AmbiguousRef
        push!(prev.refs, ref)
    else
        entries[name] = AmbiguousRef(name, CircRef[prev, ref])
    end
    return entries
end

_lang(kind::CircuitKind) = kind === SPICECircuit ? :spice :
                           kind === Mixed ? :mixed : :spectre

# A deck's first line is its title. Where it was written as a comment — the
# usual `* my circuit` — the star is punctuation, not part of the name, and an
# *implicit* title keeps it in the `line` child.
function _title(sema::SemaResult)
    sema.title === nothing && return nothing
    text = strip(lstrip(strip(String(sema.title.line)), '*'))
    return isempty(text) ? nothing : String(text)
end

"""
    netlist_index(sema::SemaResult; lang) -> NetlistIndex

Summarise one scope — a deck or a `.subckt` body — into the names it declares.
Recurses into subcircuits, so the whole hierarchy is reachable from the top.

`lang` is the deck's, and is inherited by every scope inside it: only the deck's
own root carries the statements `circuit_kind` reads.
"""
function netlist_index(sema::SemaResult; lang::Symbol=_lang(sema.kind))
    entries = OrderedDict{Symbol, CircRef}()

    # One pass over the instances gives both directions: what each one connects
    # to, and — inverted here rather than re-derived per net below — which
    # instances land on each node.
    connections = Dict{Symbol, Vector{Symbol}}()
    for (name, entry) in sema.instances
        isempty(entry) && continue
        nodes = _defs(entry)
        nets = _instance_nets(last(nodes))
        for net in nets
            push!(get!(() -> Symbol[], connections, net), name)
        end
        _add!(entries, InstanceRef(name, _card_kind(last(nodes)), nets, map(_card, nodes)))
    end

    for (name, entry) in sema.params
        isempty(entry) && continue
        _add!(entries, ParamRef(name, map(_card, _defs(entry))))
    end

    for (name, entry) in sema.models
        isempty(entry) && continue
        defs = _defs(entry)          # each is a `Pair{SNode, GlobalRef}`
        resolved = last(defs).second
        _add!(entries, ModelRef(name, resolved isa GlobalRef ? resolved : nothing,
                                Card[_card(d.first) for d in defs]))
    end

    for (name, entry) in sema.subckts
        isempty(entry) && continue
        semas = _defs(entry)
        _add!(entries, SubcktRef(name, extract_subcircuit_ports(last(semas)),
                                 Card[_card(s.ast; header_only=true) for s in semas],
                                 netlist_index(last(semas); lang)))
    end

    # Nets last: an instance or parameter of the same name is the more specific
    # answer, and folding the net in after it keeps that reading order.
    for name in keys(sema.nets)
        _add!(entries, NetRef(name, get(connections, name, Symbol[])))
    end

    return NetlistIndex(_title(sema), lang, entries)
end

#==============================================================================#
# Reaching the index from a circuit
#
# Codegen binds the index under `NETLIST_INDEX_BINDING` in the module it emits
# the builder into, so the builder function is the handle: `parentmodule` finds
# the module, and the binding is read in the latest world because an earlier
# `Core.eval` created it in a world newer than our caller's.
#==============================================================================#

const NETLIST_INDEX_BINDING = Symbol("#netlist_index#")

"""
    netlist(circuit) -> NetlistIndex

What the netlist behind `circuit` declares: its instances, parameters, models,
subcircuits and nets, each with the card that defines it.

`circuit` is an `MNACircuit` or a generated builder function.

```julia
julia> nl = netlist(MNACircuit("divider.sp"));

julia> nl.r1
Instance r1 (Resistor) — vcc, out
╭ divider.sp:4
│ R1 vcc out 'res'
╰
```

Names are addressed with `.` or `[]`, case-insensitively as SPICE resolves them;
`keys`, `haskey`, `get` and `propertynames` enumerate. A `SubcktRef` is indexable
in turn, so `nl.divider[:r1]` reaches inside the subcircuit.

Throws for a hand-written builder: only a netlist has cards to point at.
"""
function netlist end

netlist(circuit::MNA.MNACircuit) = netlist(circuit.builder)

function netlist(@nospecialize(builder))
    mod = parentmodule(builder)
    if !latest_isdefined(mod, NETLIST_INDEX_BINDING)
        throw(ArgumentError(
            "no netlist behind $(builder): `netlist` reads the index a generated " *
            "builder carries, and a hand-written builder has no cards to point at."))
    end
    return latest_global(mod, NETLIST_INDEX_BINDING)::NetlistIndex
end

export netlist

#==============================================================================#
# Lookup
#==============================================================================#

_lname(name::Symbol) = Symbol(lowercase(String(name)))

function _lookup(idx::NetlistIndex, name::Symbol)
    entries = getfield(idx, :entries)
    ref = get(entries, _lname(name), nothing)
    ref === nothing && throw(ArgumentError(
        "netlist declares no `$name`. It declares $(length(entries)) name" *
        (length(entries) == 1 ? "" : "s") * "; `keys` lists them."))
    return ref
end

Base.getproperty(idx::NetlistIndex, name::Symbol) = _lookup(idx, name)
Base.getindex(idx::NetlistIndex, name::Symbol) = _lookup(idx, name)
Base.propertynames(idx::NetlistIndex) = collect(keys(getfield(idx, :entries)))
Base.keys(idx::NetlistIndex) = keys(getfield(idx, :entries))
Base.values(idx::NetlistIndex) = values(getfield(idx, :entries))
Base.haskey(idx::NetlistIndex, name::Symbol) = haskey(getfield(idx, :entries), _lname(name))
Base.get(idx::NetlistIndex, name::Symbol, default) = get(getfield(idx, :entries), _lname(name), default)
Base.length(idx::NetlistIndex) = length(getfield(idx, :entries))

# A subcircuit is a namespace too, reached through the ref that names it.
Base.getindex(s::SubcktRef, name::Symbol) = s.index[name]
Base.keys(s::SubcktRef) = keys(s.index)
Base.haskey(s::SubcktRef, name::Symbol) = haskey(s.index, name)
Base.get(s::SubcktRef, name::Symbol, default) = get(s.index, name, default)

#==============================================================================#
# Printing
#==============================================================================#

_kindname(::InstanceRef) = "Instance"
_kindname(::ParamRef) = "Parameter"
_kindname(::ModelRef) = "Model"
_kindname(::SubcktRef) = "Subcircuit"
_kindname(::NetRef) = "Net"
_kindname(::AmbiguousRef) = "Ambiguous"

Base.show(io::IO, r::CircRef) = print(io, _kindname(r), " ", r.name)

function _show_cards(io::IO, cards::Vector{Card})
    for (i, card) in enumerate(cards)
        i > 1 && println(io)
        printstyled(io, "╭ ", card.file, ":", card.line; color=:light_black)
        println(io)
        for line in split(card.text, '\n')
            printstyled(io, "│ "; color=:light_black)
            println(io, rstrip(line))
        end
        printstyled(io, "╰"; color=:light_black)
    end
    if length(cards) > 1
        println(io)
        printstyled(io, "($(length(cards)) definitions — the name is declared in more than one branch)";
                   color=:light_black)
    end
end

function _show_header(io::IO, r::CircRef)
    printstyled(io, _kindname(r), " "; color=:light_black)
    printstyled(io, r.name; bold=true)
end

function Base.show(io::IO, ::MIME"text/plain", r::InstanceRef)
    _show_header(io, r)
    print(io, " (", r.device, ")")
    isempty(r.nets) || print(io, " — ", join(r.nets, ", "))
    println(io)
    _show_cards(io, r.cards)
end

function Base.show(io::IO, ::MIME"text/plain", r::ParamRef)
    _show_header(io, r)
    println(io)
    _show_cards(io, r.cards)
end

function Base.show(io::IO, ::MIME"text/plain", r::ModelRef)
    _show_header(io, r)
    print(io, r.device === nothing ? " (unresolved)" : " → $(r.device)")
    println(io)
    _show_cards(io, r.cards)
end

function Base.show(io::IO, ::MIME"text/plain", r::SubcktRef)
    _show_header(io, r)
    isempty(r.ports) || print(io, " — ", join(r.ports, ", "))
    println(io)
    _show_cards(io, r.cards)
    n = length(r.index)
    print(io, "\n", n, " name", n == 1 ? "" : "s", " inside")
end

function Base.show(io::IO, ::MIME"text/plain", r::NetRef)
    _show_header(io, r)
    println(io)
    if isempty(r.instances)
        printstyled(io, "unconnected"; color=:light_black)
    else
        print(io, length(r.instances), " connection",
              length(r.instances) == 1 ? "" : "s", ": ", join(r.instances, ", "))
    end
end

function Base.show(io::IO, ::MIME"text/plain", r::AmbiguousRef)
    _show_header(io, r)
    print(io, " — ", join((_kindname(x) for x in r.refs), ", "), " share this name")
    for sub in r.refs
        println(io)
        show(io, MIME"text/plain"(), sub)
    end
end

# Long name lists are noise; the first few say what kind of deck this is.
function _print_names(io::IO, label::AbstractString, names::Vector{Symbol}; limit::Int=8)
    isempty(names) && return
    print(io, "\n  ", label, " (", length(names), "): ")
    print(io, join(view(names, 1:min(limit, length(names))), ", "))
    length(names) > limit && printstyled(io, ", … "; color=:light_black)
end

function _names_of_kind(idx::NetlistIndex, ::Type{T}) where {T <: CircRef}
    names = Symbol[]
    for (name, ref) in getfield(idx, :entries)
        if ref isa T || (ref isa AmbiguousRef && any(x -> x isa T, ref.refs))
            push!(names, name)
        end
    end
    return names
end

function Base.show(io::IO, ::MIME"text/plain", idx::NetlistIndex)
    lang = getfield(idx, :lang)
    print(io, lang === :spice ? "SPICE netlist" :
              lang === :mixed ? "SPICE/Spectre netlist" : "Spectre netlist")
    title = getfield(idx, :title)
    if title !== nothing
        print(io, " ")
        printstyled(io, '"', title, '"'; bold=true)
    end
    _print_names(io, "instances", _names_of_kind(idx, InstanceRef))
    _print_names(io, "parameters", _names_of_kind(idx, ParamRef))
    _print_names(io, "models", _names_of_kind(idx, ModelRef))
    _print_names(io, "subcircuits", _names_of_kind(idx, SubcktRef))
    _print_names(io, "nets", _names_of_kind(idx, NetRef))
end

Base.show(io::IO, idx::NetlistIndex) = print(io, "NetlistIndex(", length(idx), " names)")
