# MNA Netlist Parser
# Parses SPICE/Spectre netlists and builds MNA circuits

using SpectreNetlistParser
using SpectreNetlistParser: SPICENetlistParser
using SpectreNetlistParser.SPICENetlistParser: SPICENetlistCSTParser
using AbstractTrees

export parse_spice, parse_spice_file, MNANetlist, simulate_netlist, @spice_str

const SP = SPICENetlistCSTParser
const SNode = SpectreNetlistParser.RedTree.Node

"""
    MNANetlist

Holds parsed netlist information ready for MNA circuit construction.
"""
struct MNANetlist
    title::String
    elements::Vector{NamedTuple}
    subcircuits::Dict{Symbol, Any}
    models::Dict{Symbol, NamedTuple}
end

"""
    parse_spice(netlist::String) -> MNANetlist

Parse a SPICE netlist string and return an MNANetlist.
"""
function parse_spice(netlist::String)
    # Write to temp file (parser requires file)
    tmpfile = tempname() * ".sp"
    try
        open(tmpfile, "w") do f
            write(f, netlist)
        end
        return parse_spice_file(tmpfile)
    finally
        rm(tmpfile, force=true)
    end
end

"""
    parse_spice_file(filename::String) -> MNANetlist

Parse a SPICE netlist file.
"""
function parse_spice_file(filename::String)
    ast = SP.parsefile(filename)

    if ast.ps.errored
        error("Parse error in SPICE netlist: $filename")
    end

    elements = NamedTuple[]
    subcircuits = Dict{Symbol, Any}()
    models = Dict{Symbol, NamedTuple}()
    title = ""

    # Process each child statement
    for child in AbstractTrees.children(ast)
        T = typeof(child).parameters[1]

        if T == SP.Title
            title = String(child)
        elseif T == SP.Resistor
            push!(elements, parse_resistor(child))
        elseif T == SP.Capacitor
            push!(elements, parse_capacitor(child))
        elseif T == SP.Inductor
            push!(elements, parse_inductor(child))
        elseif T == SP.Voltage
            push!(elements, parse_voltage(child))
        elseif T == SP.Current
            push!(elements, parse_current(child))
        elseif T == SP.Diode
            push!(elements, parse_diode(child))
        elseif T <: SP.ControlledSource
            # ControlledSource{:V, :V} = VCVS, ControlledSource{:V, :C} = VCCS
            if T == SP.ControlledSource{:V, :V}
                push!(elements, parse_vcvs(child))
            elseif T == SP.ControlledSource{:V, :C}
                push!(elements, parse_vccs(child))
            end
        elseif T == SP.Model
            name, model = parse_model(child)
            models[name] = model
        elseif T == SP.Subckt
            name, subckt = parse_subcircuit(child)
            subcircuits[name] = subckt
        end
    end

    return MNANetlist(title, elements, subcircuits, models)
end

# Get property value safely
function get_prop(node, prop::Symbol)
    try
        return getproperty(node, prop)
    catch
        return nothing
    end
end

# Convert node to symbol
function node_symbol(n)
    s = lowercase(String(n))
    s == "0" && return :gnd
    return Symbol(s)
end

# Parse numeric value with SPICE suffixes
function parse_value(val)
    val === nothing && return 0.0
    val isa Number && return Float64(val)

    s = String(val)

    # SPICE metric suffixes
    suffixes = [
        ("meg", 1e6), ("MEG", 1e6),
        ("t", 1e12), ("T", 1e12),
        ("g", 1e9), ("G", 1e9),
        ("k", 1e3), ("K", 1e3),
        ("m", 1e-3),
        ("u", 1e-6), ("μ", 1e-6),
        ("n", 1e-9),
        ("p", 1e-12),
        ("f", 1e-15),
        ("a", 1e-18)
    ]

    # Try to parse with suffix
    for (suffix, mult) in suffixes
        if endswith(lowercase(s), lowercase(suffix))
            num_part = s[1:end-length(suffix)]
            v = tryparse(Float64, num_part)
            v !== nothing && return v * mult
        end
    end

    # Plain number
    v = tryparse(Float64, s)
    return v !== nothing ? v : 0.0
end

# Extract value from NumberLiteral child
function extract_value(node)
    for c in AbstractTrees.children(node)
        T = typeof(c).parameters[1]
        if T == SP.NumberLiteral
            return parse_value(String(c))
        end
    end
    return 0.0
end

function parse_resistor(node)
    name = Symbol(lowercase(String(node.name)))
    n1 = node_symbol(node.pos)
    n2 = node_symbol(node.neg)
    r = extract_value(node)
    return (type='R', name=name, nodes=[n1, n2], params=(r=r,))
end

function parse_capacitor(node)
    name = Symbol(lowercase(String(node.name)))
    n1 = node_symbol(node.pos)
    n2 = node_symbol(node.neg)
    c = extract_value(node)
    return (type='C', name=name, nodes=[n1, n2], params=(c=c,))
end

function parse_inductor(node)
    name = Symbol(lowercase(String(node.name)))
    n1 = node_symbol(node.pos)
    n2 = node_symbol(node.neg)
    l = extract_value(node)
    return (type='L', name=name, nodes=[n1, n2], params=(l=l,))
end

# Extract DC value from children (DCSource child)
function extract_dc_value(node)
    for c in AbstractTrees.children(node)
        T = typeof(c).parameters[1]
        if T == SP.DCSource
            return parse_value(String(c))
        end
    end
    return 0.0
end

function parse_voltage(node)
    name = Symbol(lowercase(String(node.name)))
    n1 = node_symbol(node.pos)
    n2 = node_symbol(node.neg)
    dc = extract_dc_value(node)
    return (type='V', name=name, nodes=[n1, n2], params=(dc=dc,))
end

function parse_current(node)
    name = Symbol(lowercase(String(node.name)))
    n1 = node_symbol(node.pos)
    n2 = node_symbol(node.neg)
    dc = extract_dc_value(node)
    return (type='I', name=name, nodes=[n1, n2], params=(dc=dc,))
end

function parse_diode(node)
    name = Symbol(lowercase(String(node.name)))
    n1 = node_symbol(node.pos)
    n2 = node_symbol(node.neg)

    model = nothing
    try
        model = Symbol(lowercase(String(node.model)))
    catch
    end

    return (type='D', name=name, nodes=[n1, n2], params=(model=model,))
end

# Parse VoltageControl/CurrentControl child to get control nodes and gain
function parse_control_child(node)
    for c in AbstractTrees.children(node)
        T = typeof(c).parameters[1]
        if T == SP.VoltageControl || T == SP.CurrentControl
            children = collect(AbstractTrees.children(c))
            if length(children) >= 3
                nc1 = node_symbol(children[1])
                nc2 = node_symbol(children[2])
                gain = parse_value(String(children[3]))
                return (nc1, nc2, gain)
            end
        end
    end
    return (:gnd, :gnd, 1.0)
end

function parse_vcvs(node)
    name = Symbol(lowercase(String(node.name)))
    n1 = node_symbol(node.pos)
    n2 = node_symbol(node.neg)
    nc1, nc2, gain = parse_control_child(node)
    return (type='E', name=name, nodes=[n1, n2, nc1, nc2], params=(gain=gain,))
end

function parse_vccs(node)
    name = Symbol(lowercase(String(node.name)))
    n1 = node_symbol(node.pos)
    n2 = node_symbol(node.neg)
    nc1, nc2, gain = parse_control_child(node)
    return (type='G', name=name, nodes=[n1, n2, nc1, nc2], params=(gain=gain,))
end

function parse_model(node)
    name = Symbol(lowercase(String(node.name)))

    # Find model type (Identifier child after name) and parameters
    mtype = :unknown
    params = Dict{Symbol, Float64}()

    for c in AbstractTrees.children(node)
        T = typeof(c).parameters[1]
        if T == SP.Identifier
            mtype = Symbol(lowercase(String(c)))
        elseif T == SP.Parameter
            # Parameter is "name=value" format
            pstr = String(c)
            if occursin("=", pstr)
                parts = split(pstr, "=", limit=2)
                pname = Symbol(lowercase(strip(parts[1])))
                pval = parse_value(strip(parts[2]))
                params[pname] = pval
            end
        end
    end

    return name, (name=name, type=mtype, params=NamedTuple(params))
end

function parse_subcircuit(node)
    name = Symbol(lowercase(String(node.name)))
    ports = Symbol[]
    for n in node.subckt_nodes
        push!(ports, node_symbol(n))
    end

    elements = NamedTuple[]
    for child in AbstractTrees.children(node)
        T = typeof(child).parameters[1]
        if T == SP.Resistor
            push!(elements, parse_resistor(child))
        elseif T == SP.Capacitor
            push!(elements, parse_capacitor(child))
        elseif T == SP.Inductor
            push!(elements, parse_inductor(child))
        elseif T == SP.Voltage
            push!(elements, parse_voltage(child))
        elseif T == SP.Current
            push!(elements, parse_current(child))
        end
    end

    return name, (name=name, ports=ports, elements=elements)
end

"""
    build_circuit(netlist::MNANetlist; temp=27.0, gmin=1e-12) -> MNACircuit

Build an MNACircuit from a parsed netlist.
"""
function build_circuit(netlist::MNANetlist; temp=27.0, gmin=1e-12)
    circuit = MNACircuit(temp=temp, gmin=gmin)

    # Map :gnd to the actual ground node (index 0)
    # This way all references to :gnd use the real ground
    circuit.nets[:gnd] = circuit.ground

    for elem in netlist.elements
        add_element!(circuit, elem, netlist.models)
    end

    return circuit
end

# Add element to circuit based on type
function add_element!(circuit::MNACircuit, elem::NamedTuple, models::Dict)
    t = elem.type
    nodes = elem.nodes
    params = elem.params
    name = elem.name

    if t == 'R'
        r = get(params, :r, 1000.0)
        r > 0 && resistor!(circuit, nodes[1], nodes[2], r; name=name)

    elseif t == 'C'
        c = get(params, :c, 1e-12)
        c > 0 && capacitor!(circuit, nodes[1], nodes[2], c; name=name)

    elseif t == 'L'
        l = get(params, :l, 1e-9)
        l > 0 && inductor!(circuit, nodes[1], nodes[2], l; name=name)

    elseif t == 'V'
        dc = get(params, :dc, 0.0)
        vsource!(circuit, nodes[1], nodes[2]; dc=dc, name=name)

    elseif t == 'I'
        dc = get(params, :dc, 0.0)
        isource!(circuit, nodes[1], nodes[2]; dc=dc, name=name)

    elseif t == 'D'
        model_name = get(params, :model, nothing)
        diode_params = Dict{Symbol, Any}()
        if model_name !== nothing && haskey(models, model_name)
            model = models[model_name]
            for (k, v) in pairs(model.params)
                diode_params[k] = v
            end
        end
        diode!(circuit, nodes[1], nodes[2]; diode_params..., name=name)

    elseif t == 'E'
        gain = get(params, :gain, 1.0)
        if length(nodes) >= 4
            vcvs!(circuit, nodes[1], nodes[2], nodes[3], nodes[4], gain; name=name)
        end

    elseif t == 'G'
        gain = get(params, :gain, 1.0)
        if length(nodes) >= 4
            vccs!(circuit, nodes[1], nodes[2], nodes[3], nodes[4], gain; name=name)
        end
    end
end

"""
    simulate_netlist(netlist_str::String; analysis=:dc, kwargs...)

Parse and simulate a SPICE netlist string.

# Example
```julia
result = simulate_netlist(\"\"\"
* Simple voltage divider
V1 vcc 0 5
R1 vcc out 1k
R2 out 0 1k
.end
\"\"\")
```
"""
function simulate_netlist(netlist_str::String; analysis=:dc, temp=27.0, gmin=1e-12, kwargs...)
    netlist = parse_spice(netlist_str)
    circuit = build_circuit(netlist; temp=temp, gmin=gmin)

    if analysis == :dc
        return dc!(circuit; kwargs...)
    elseif analysis == :tran
        tspan = get(kwargs, :tspan, (0.0, 1e-3))
        return tran!(circuit, tspan; kwargs...)
    else
        error("Unsupported analysis type: $analysis")
    end
end

"""
    @spice_str(netlist) -> DCResult

String macro for inline SPICE netlists.

# Example
```julia
result = spice\"\"\"
V1 vcc 0 5
R1 vcc out 1k
R2 out 0 1k
\"\"\"
```
"""
macro spice_str(netlist)
    quote
        simulate_netlist($netlist)
    end
end
