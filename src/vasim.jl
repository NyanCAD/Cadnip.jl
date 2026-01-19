using VerilogAParser
using AbstractTrees
using AbstractTrees: parent, nodevalue
using VerilogAParser.VerilogACSTParser:
    ContributionStatement, AnalogSeqBlock, AnalogBlock,
    InOutDeclaration, NetDeclaration, ParameterDeclaration, AliasParameterDeclaration,
    VerilogModule, Literal, BinaryExpression, BPFC,
    IdentifierPrimary, @case, BranchDeclaration,
    AnalogFunctionDeclaration,
    IntRealDeclaration, IntRealVarDecl, AnalogStatement,
    AnalogConditionalBlock, AnalogVariableAssignment, AnalogProceduralAssignment,
    Parens, AnalogIf, AnalogFor, AnalogWhile, AnalogRepeat, UnaryOp, Function,
    AnalogSystemTaskEnable, StringLiteral,
    CaseStatement, FunctionCall, FunctionCallStatement, TernaryExpr,
    FloatLiteral, ChunkTree, virtrange,
    filerange, LineNumbers, compute_line,
    SystemIdentifier, Node, Identifier, IdentifierConcatItem,
    IdentifierPart, Attributes
using VerilogAParser.VerilogATokenize:
    Kind, INPUT, OUTPUT, INOUT, REAL, INTEGER, STRING, is_scale_factor
using Combinatorics
using ForwardDiff
using ForwardDiff: Dual

const VAT = VerilogAParser.VerilogATokenize

const VANode = VerilogAParser.VerilogACSTParser.Node

#==============================================================================#
# Taint Tracking for Code Splitting
#
# Classifies expressions and statements as "static" (parameter/temperature
# dependent only) or "dynamic" (voltage/current dependent).
#
# Static code can be moved to init_device!() and computed once.
# Dynamic code must stay in stamp!() and run every Newton iteration.
#==============================================================================#

"""
    CodegenTaint

Tracks which variables and statements are static vs dynamic during codegen.

Static = depends only on parameters, temperature, constants
Dynamic = depends on voltages (V()), currents (I()), or solution vector (x)

Conservative rule: if uncertain, classify as dynamic (safe but less optimal).
"""
mutable struct CodegenTaint
    # Variables that are known static (param-only)
    static_vars::Set{Symbol}
    # Variables that are known dynamic (voltage-dependent)
    dynamic_vars::Set{Symbol}
    # Statements classified as static (can go in init_device!)
    static_stmts::Vector{Any}
    # Statements classified as dynamic (must stay in stamp!)
    dynamic_stmts::Vector{Any}
    # Track which static vars are actually used (for cache struct)
    used_static_vars::Set{Symbol}
end

function CodegenTaint()
    CodegenTaint(Set{Symbol}(), Set{Symbol}(), Any[], Any[], Set{Symbol}())
end

"""
    is_static_expr(taint::CodegenTaint, expr) -> Bool

Check if an expression is static (doesn't depend on voltages/currents).
Conservative: returns false if uncertain.
"""
function is_static_expr(taint::CodegenTaint, expr)
    # Nothing is static (e.g., suppressed $warning calls)
    if expr === nothing
        return true
    end
    if expr isa Number || expr isa String
        return true
    end
    if expr isa QuoteNode
        return true
    end
    # Type values (like Float64, Int) are static - used for type conversion
    if expr isa DataType
        return true
    end
    if expr isa Symbol
        # Known dynamic symbols
        if expr in taint.dynamic_vars
            return false
        end
        # Known static symbols (params, temp)
        if expr in taint.static_vars
            return true
        end
        # Special MNA symbols that are dynamic
        if expr in (:_mna_x_, :_mna_t_, :_mna_mode_, :ctx)
            return false
        end
        # Special MNA symbols that are static
        if expr in (:_mna_spec_, :dev)
            return true
        end
        # Type names are static (used in type conversions)
        if expr in (:Float64, :Int, :Int64, :Int32, :Bool, :String, :Real, :Integer)
            return true
        end
        # VA system variables that are static (temperature-dependent, not voltage-dependent)
        # Note: VA parser may produce symbols like \$temperature (with backslash) or $temperature
        expr_str = string(expr)
        debug = get(ENV, "CEDARSIM_DEBUG_TAINT", "") == "1"
        is_va_sysvar = startswith(expr_str, "\$") || startswith(expr_str, "\\\$") || startswith(expr_str, "var\"")
        if debug && occursin("temp", expr_str)
            @info "Checking temperature symbol" expr expr_str is_va_sysvar
        end
        if is_va_sysvar
            # $temperature, $vt, $simparam, etc. are static (temperature-dependent)
            # $realtime, $abstime are dynamic (time-dependent)
            if occursin("temperature", expr_str) || occursin("vt", expr_str)
                debug && @info "Recognized as static VA system var" expr_str
                return true
            end
            if occursin("simparam", expr_str) || occursin("param_given", expr_str)
                return true
            end
            # Time-related functions are dynamic
            if occursin("realtime", expr_str) || occursin("abstime", expr_str)
                return false
            end
        end
        # Unknown - assume dynamic to be safe
        return false
    end
    if expr isa Expr
        if expr.head == :call
            fname = expr.args[1]
            # V() and I() are always dynamic - they access voltages/currents
            if fname == :V || fname == :I
                return false
            end
            # va_ddt is dynamic (involves time derivative of state)
            if fname == :va_ddt
                return false
            end
            # Dual creation is dynamic
            if fname == :Dual || (fname isa Expr && fname.head == :curly && fname.args[1] == :Dual)
                return false
            end
            # MNA stamping functions are dynamic
            if fname in (:stamp_G!, :stamp_C!, :stamp_b!, :stamp_current_contribution!)
                return false
            end
            # Allocation functions are dynamic (need ctx)
            if fname in (:alloc_internal_node!, :alloc_current!, :alloc_charge!)
                return false
            end
            # Type constructors and conversion functions are static
            if fname == :zero || fname == :one || fname == :Float64 || fname == :Int
                return true
            end
            # Check if it's a module-qualified function
            debug = get(ENV, "CEDARSIM_DEBUG_TAINT", "") == "1"
            if debug
                @info "is_static_expr call check" fname fname_type=typeof(fname) is_expr=(fname isa Expr) head=(fname isa Expr ? fname.head : nothing)
            end
            # Handle resolved function objects (when macro has already resolved the function)
            if fname isa Function
                fname_mod = parentmodule(fname)
                fname_str = string(fname_mod) * "." * string(nameof(fname))
                debug && @info "Resolved function" fname_str
                # CedarSim.MNA functions are dynamic
                if occursin("MNA", fname_str)
                    return false
                end
                # VerilogAEnvironment functions are static (type conversion, math functions)
                if occursin("VerilogAEnvironment", fname_str)
                    if debug
                        @info "VerilogAEnvironment resolved function call" fname_str args=expr.args[2:end]
                        for (i, arg) in enumerate(expr.args[2:end])
                            @info "  checking arg $i" arg result=is_static_expr(taint, arg)
                        end
                    end
                    return all(is_static_expr(taint, arg) for arg in expr.args[2:end])
                end
                # Base math functions are static (*, +, -, /, ^, etc.)
                if fname_mod == Base || fname_mod == Core
                    if debug
                        @info "Base/Core resolved function call" fname_str args=expr.args[2:end]
                        for (i, arg) in enumerate(expr.args[2:end])
                            @info "  checking arg $i" arg result=is_static_expr(taint, arg)
                        end
                    end
                    return all(is_static_expr(taint, arg) for arg in expr.args[2:end])
                end
            end
            if fname isa Expr && fname.head == :.
                # Extract the full path as a string for easier matching
                fname_str = string(fname)
                debug && @info "Qualified function name" fname_str
                # CedarSim.MNA functions are dynamic
                if startswith(fname_str, "CedarSim.MNA")
                    return false
                end
                # VerilogAEnvironment functions are static (type conversion, math functions)
                if occursin("VerilogAEnvironment", fname_str)
                    # These are static: vaconvert, etc. - check arguments
                    args_to_check = expr.args[2:end]
                    if get(ENV, "CEDARSIM_DEBUG_TAINT", "") == "1"
                        @info "VerilogAEnvironment call" fname_str args=args_to_check
                        for (i, arg) in enumerate(args_to_check)
                            @info "  checking arg $i" arg result=is_static_expr(taint, arg)
                        end
                    end
                    return all(is_static_expr(taint, arg) for arg in args_to_check)
                end
            end
            # Other calls: static if all args are static
            return all(is_static_expr(taint, arg) for arg in expr.args[2:end])
        elseif expr.head == :.
            # Field access (e.g., dev.R, _mna_spec_.temp)
            obj = expr.args[1]
            if obj == :dev || obj == :_mna_spec_
                return true  # Parameter/spec access is static
            end
            return is_static_expr(taint, obj)
        elseif expr.head == :if || expr.head == :elseif
            # Conditional: static only if condition AND all branches are static
            return all(is_static_expr(taint, arg) for arg in expr.args)
        elseif expr.head == :block
            # Block: static if all statements are static
            return all(is_static_expr(taint, arg) for arg in expr.args if !(arg isa LineNumberNode))
        elseif expr.head == :(=) || expr.head == :(+=) || expr.head == :(-=) || expr.head == :(*=) || expr.head == :(/=)
            # Assignment: static if RHS is static (LHS is just a symbol)
            return is_static_expr(taint, expr.args[2])
        elseif expr.head == :let
            # Let binding: check the body
            return is_static_expr(taint, expr.args[end])
        elseif expr.head == :tuple || expr.head == :vect
            return all(is_static_expr(taint, arg) for arg in expr.args)
        elseif expr.head == :ref
            # Array indexing: check index and array
            return all(is_static_expr(taint, arg) for arg in expr.args)
        elseif expr.head == :comparison
            return all(is_static_expr(taint, arg) for arg in expr.args[1:2:end])
        elseif expr.head == :&&  || expr.head == :||
            return all(is_static_expr(taint, arg) for arg in expr.args)
        elseif expr.head == :curly
            # Type parameters - static
            return true
        elseif expr.head == :(::)
            # Type annotation - check value if present
            if length(expr.args) == 2
                return is_static_expr(taint, expr.args[1])
            end
            return true
        end
        # Default: check all subexpressions
        return all(is_static_expr(taint, arg) for arg in expr.args if !(arg isa LineNumberNode))
    end
    # Unknown type - assume dynamic to be safe
    return false
end

"""
    classify_assignment!(taint::CodegenTaint, lhs::Symbol, rhs, stmt)

Classify an assignment statement as static or dynamic, updating taint tracking.
"""
function classify_assignment!(taint::CodegenTaint, lhs::Symbol, rhs, stmt)
    is_static = is_static_expr(taint, rhs)
    if get(ENV, "CEDARSIM_DEBUG_TAINT", "") == "1"
        @info "classify_assignment!" lhs rhs is_static
    end
    if is_static
        push!(taint.static_vars, lhs)
        push!(taint.static_stmts, stmt)
    else
        push!(taint.dynamic_vars, lhs)
        push!(taint.dynamic_stmts, stmt)
    end
end

"""
    extract_symbols(expr) -> Set{Symbol}

Extract all symbols referenced in an expression.
"""
function extract_symbols(expr, symbols=Set{Symbol}())
    if expr isa Symbol
        push!(symbols, expr)
    elseif expr isa Expr
        for arg in expr.args
            extract_symbols(arg, symbols)
        end
    end
    return symbols
end

"""
    extract_assigned_symbols(expr) -> Set{Symbol}

Extract all symbols that are ASSIGNED TO in an expression.
This walks the expression tree looking for assignment statements.
"""
function extract_assigned_symbols(expr, symbols=Set{Symbol}())
    if expr isa Expr
        if expr.head in (:(=), :(+=), :(-=), :(*=), :(/=))
            lhs = expr.args[1]
            if lhs isa Symbol
                push!(symbols, lhs)
            elseif lhs isa Expr && lhs.head == :(::) && lhs.args[1] isa Symbol
                push!(symbols, lhs.args[1])
            end
        end
        # Recursively check all subexpressions
        for arg in expr.args
            extract_assigned_symbols(arg, symbols)
        end
    end
    return symbols
end

"""
    generate_cache_struct(device_name::Symbol, static_vars::Dict{Symbol, Type})

Generate a mutable cache struct to store static computation results.

# Example output:
```julia
mutable struct MyDeviceCache
    var1::Float64
    var2::Float64
    _initialized::Bool
    MyDeviceCache() = new(0.0, 0.0, false)
end
```
"""
function generate_cache_struct(device_name::Symbol, static_var_types::Dict{Symbol, Type})
    cache_name = Symbol(device_name, "Cache")

    # Build field declarations
    fields = Any[]
    init_values = Any[]

    for (name, T) in sort(collect(static_var_types), by=first)  # Sort for deterministic order
        push!(fields, Expr(:(::), name, T))
        # Default initialization value based on type
        default_val = T === String ? "" : T === Int ? 0 : 0.0
        push!(init_values, default_val)
    end

    # Add _initialized flag
    push!(fields, Expr(:(::), :_initialized, :Bool))
    push!(init_values, false)

    # Build struct with inner constructor
    struct_body = Expr(:block, fields...)
    push!(struct_body.args, :($(cache_name)() = new($(init_values...))))

    return Expr(:struct, true,  # mutable
                cache_name,
                struct_body)
end

"""
    generate_init_device_method(device_name, cache_name, params_to_locals, static_stmts, static_var_names)

Generate the init_device! method that computes static values.

# Example output:
```julia
function CedarSim.MNA.init_device!(cache::MyDeviceCache, dev::MyDevice, spec::MNASpec)
    if cache._initialized
        return cache
    end
    # Extract parameters
    R = undefault(dev.R)
    # Compute static values
    R_scaled = R * 1000.0
    # Store in cache
    cache.R_scaled = R_scaled
    cache._initialized = true
    return cache
end
```
"""
function generate_init_device_method(device_name::Symbol, cache_name::Symbol,
                                      params_to_locals::Vector, local_var_decls::Vector,
                                      static_stmts::Vector,
                                      static_var_names::Set{Symbol},
                                      function_defs::Vector=Any[])
    # Build the function body
    body = Expr(:block)

    # Early return if already initialized
    push!(body.args, quote
        if cache._initialized
            return cache
        end
    end)

    # Extract parameters from device
    for stmt in params_to_locals
        push!(body.args, stmt)
    end

    # Include analog function definitions (needed for static code that calls them)
    for fdef in function_defs
        push!(body.args, fdef)
    end

    # Declare local variables (needed before static statements can assign to them)
    # Convert `local name::T = init_expr` to simple `name = init_expr` for init_device scope
    for decl in local_var_decls
        if decl.head == :local
            inner = decl.args[1]
            if inner isa Expr && inner.head == :(=)
                lhs = inner.args[1]
                rhs = inner.args[2]
                if lhs isa Expr && lhs.head == :(::)
                    name = lhs.args[1]
                    # Only include if the variable is used in static statements
                    # This avoids declaring variables that are only used dynamically
                    push!(body.args, :($name = $rhs))
                end
            end
        end
    end

    # Execute static statements
    for stmt in static_stmts
        push!(body.args, stmt)
    end

    # Store results in cache
    for var in sort(collect(static_var_names))  # Sort for deterministic order
        push!(body.args, :(cache.$var = $var))
    end

    # Mark as initialized and return
    push!(body.args, :(cache._initialized = true))
    push!(body.args, :(return cache))

    # Debug: print generated init_device code
    if get(ENV, "CEDARSIM_DEBUG_INIT_DEVICE", "") == "1"
        @info "Generated init_device! for $device_name" body=body
    end

    # Build the function
    return quote
        function CedarSim.MNA.init_device!(cache::$cache_name, dev::$device_name,
                                           _mna_spec_::CedarSim.MNA.MNASpec)
            $body
        end
    end
end

"""
    generate_make_cache_method(device_name, cache_name)

Generate a method to create a new cache instance for a device.
"""
function generate_make_cache_method(device_name::Symbol, cache_name::Symbol)
    quote
        CedarSim.MNA.make_cache(::Type{$device_name}) = $cache_name()
        CedarSim.MNA.make_cache(::$device_name) = $cache_name()
    end
end

function eisa(e::VANode{S}, T::Type) where {S}
    S <: T
end
formof(e::VANode{S}) where {S} = S

@enum BranchKind CURRENT VOLTAGE

struct VAFunction
    arg_order::Vector{Symbol}
    inout_decls::Dict{Symbol, Symbol}
end

# Scale factor mapping for Verilog-A literals
const sf_mapping = Dict(
    'T' => 1e12,
    'G' => 1e9,
    'M' => 1e6,
    'K' => 1e3,
    'k' => 1e3,
    'm' => 1e-3,
    'u' => 1e-6,
    'n' => 1e-9,
    'p' => 1e-12,
    'f' => 1e-15,
    'a' => 1e-18,
)

function assemble_id_string(id)
    if isa(id, Node{SystemIdentifier})
        return String(id)
    elseif isa(id, Node{Identifier})
        return join(assemble_id_string(c) for c in children(id))
    elseif isa(id, Node{IdentifierPart})
        s = String(id)
        id.escaped && (s = 2[2:end])
        return s
    elseif isa(id, Node{IdentifierConcatItem})
        return assemble_id_string(id.id)
    else
        error(typeof(id))
    end
end

kw_to_T(kw::Kind) = kw === REAL ? Float64 : kw === STRING ? String : Int

function pins(vm::VANode{VerilogModule})
    plist = vm.port_list
    plist === nothing && return []
    pins = Symbol[]
    mapreduce(vcat, plist.ports) do port_decl
        Symbol(port_decl.item)
    end
end

using Base.Meta
using Core.Compiler: SlotNumber

function find_ddx!(ddx_order::Vector{Symbol}, va::VANode)
    for stmt in AbstractTrees.PreOrderDFS(va)
        if stmt isa VANode{FunctionCall} && Symbol(stmt.id) == :ddx
            item = stmt.args[2].item
            @assert formof(item) == FunctionCall
            @assert Symbol(item.id) == :V
            for arg in item.args
                name = Symbol(arg.item)
                !in(name, ddx_order) && push!(ddx_order, Symbol(arg.item))
            end
        end
    end
end


#==============================================================================#
# MNA Device Generation
#
# Generates stamp! methods for Verilog-A devices.
# Uses s-dual approach for automatic resist/react separation.
#==============================================================================#

"""
    make_mna_device(vm::VANode{VerilogModule})

Generate MNA-compatible Julia code for a Verilog-A module.

Generates `stamp!` methods that work with MNAContext directly.

# Generated Code Structure
```julia
@kwdef struct DeviceName <: VAModel
    param1::DefaultOr{Float64} = default1
    ...
end

function MNA.stamp!(dev::DeviceName, ctx::MNAContext, p::Int, n::Int;
                    t::Real=0.0, mode::Symbol=:dcop)
    # Parameter extraction
    param1 = undefault(dev.param1)

    # Contribution function (captures parameters)
    function contrib(Vpn)
        # VA analog block translated to Julia
        # ddt(x) becomes va_ddt(x)
        Vpn / param1  # Example: resistor
    end

    # Stamp contribution (uses AD for Jacobian)
    stamp_current_contribution!(ctx, p, n, contrib, zeros(max(p, n)))
end
```
"""
function make_mna_device(vm::VANode{VerilogModule})
    ps = pins(vm)
    modname = String(vm.id)
    symname = Symbol(modname)

    # Collect struct fields and parameters
    struct_fields = Any[]
    parameter_names = Set{Symbol}()
    param_defaults = Dict{Symbol, Any}()  # Store parameter default expressions
    param_types = Dict{Symbol, Type}()    # Store parameter types (Float64, Int, String)

    # Find ddx order for derivatives
    ddx_order = Vector{Symbol}()
    find_ddx!(ddx_order, vm)

    # Create scope for translating parameter defaults (undefault_ids=true)
    to_julia_defaults = MNAScope(Set{Symbol}(), Vector{Symbol}(), 0,
        Vector{Pair{Symbol}}(), Set{Pair{Symbol}}(),
        Dict{Symbol, Union{Type{Int}, Type{Float64}, Type{String}}}(),
        Dict{Symbol, VAFunction}(), true, ddx_order,
        Dict{Symbol, Pair{Symbol,Symbol}}())

    internal_nodes = Vector{Symbol}()
    var_types = Dict{Symbol, Union{Type{Int}, Type{Float64}, Type{String}}}()
    var_inits = Dict{Symbol, Any}()  # Store variable initialization expressions
    aliases = Dict{Symbol, Symbol}()

    # Pre-pass: collect parameters and nodes
    for child in vm.items
        item = child.item
        @case formof(item) begin
            InOutDeclaration => nothing
            NetDeclaration => begin
                for net in item.net_names
                    id = Symbol(assemble_id_string(net.item))
                    if !(id in ps)
                        push!(internal_nodes, id)
                    end
                end
            end
            ParameterDeclaration => begin
                for param in item.params
                    param = param.item
                    pT = Float64
                    if item.ptype !== nothing
                        pT = kw_to_T(item.ptype.kw)
                    end
                    paramname = String(assemble_id_string(param.id))
                    paramsym = Symbol(paramname)
                    push!(parameter_names, paramsym)
                    param_types[paramsym] = pT  # Store parameter type
                    # Extract default value from param.default_expr
                    if param.default_expr !== nothing
                        # Parse the default expression using the scope
                        param_defaults[paramsym] = to_julia_defaults(param.default_expr)
                    else
                        # Type-appropriate fallback defaults
                        param_defaults[paramsym] = pT === String ? "" : 0.0
                    end
                    # Use simplest possible field - just type annotation, no default
                    # @kwdef will use the type's default constructor
                    field_expr = Expr(:(::), paramsym, :(CedarSim.DefaultOr{$pT}))
                    push!(struct_fields, field_expr)
                    var_types[Symbol(paramname)] = pT
                end
            end
            AliasParameterDeclaration => begin
                param = item
                paramsym = Symbol(assemble_id_string(param.id))
                targetsym = Symbol(assemble_id_string(param.value))
                push!(parameter_names, paramsym)
                aliases[paramsym] = targetsym
            end
            IntRealDeclaration => begin
                T = kw_to_T(item.kw.kw)
                for ident in item.idents
                    # ident.item is IntRealVarDecl with id, eq, init fields
                    vardecl = ident.item
                    name = Symbol(assemble_id_string(vardecl.id))
                    var_types[name] = T
                    # Capture initialization expression if present
                    if vardecl.init !== nothing
                        var_inits[name] = to_julia_defaults(vardecl.init)
                    end
                end
            end
        end
    end

    # Second pre-pass: collect named branches (branch declarations like "branch (pos, neg) br;" or "branch (node) br;")
    named_branches = Dict{Symbol, Pair{Symbol,Symbol}}()
    for child in vm.items
        item = child.item
        @case formof(item) begin
            BranchDeclaration => begin
                # Extract references (nodes in parentheses)
                refs = Symbol[]
                for ref in item.references
                    push!(refs, Symbol(assemble_id_string(ref.item)))
                end
                # Support both two-node branches (pos, neg) and single-node branches (node) to ground
                @assert length(refs) in (1, 2) "Branch declaration must have 1 or 2 nodes, got $(length(refs))"
                if length(refs) == 2
                    pos_node, neg_node = refs[1], refs[2]
                else
                    # Single-node branch: branch (node) name; - connected to ground
                    pos_node, neg_node = refs[1], Symbol("0")
                end

                # Extract branch identifiers
                for bid in item.ids
                    branch_name = Symbol(assemble_id_string(bid.item.id))
                    named_branches[branch_name] = pos_node => neg_node
                end
            end
            _ => nothing
        end
    end

    # Build scope for code generation
    node_order = [ps; internal_nodes; Symbol("0")]
    to_julia_mna = MNAScope(parameter_names, node_order, length(internal_nodes),
        collect(map(x->Pair(x...), combinations(node_order, 2))),
        Set{Pair{Symbol}}(),
        var_types,
        Dict{Symbol, VAFunction}(), false,
        ddx_order,
        named_branches)

    # Generate analog block code
    analog_body = Expr(:block)
    contributions = Any[]
    function_defs = Any[]
    analog_block_ast = nothing  # Store for short circuit detection

    for child in vm.items
        item = child.item
        @case formof(item) begin
            InOutDeclaration => nothing
            IntRealDeclaration => nothing
            NetDeclaration => nothing
            BranchDeclaration => nothing
            ParameterDeclaration => nothing
            AliasParameterDeclaration => nothing
            AnalogFunctionDeclaration => begin
                push!(function_defs, to_julia_mna(item))
            end
            AnalogBlock => begin
                # Collect contributions from analog block
                mna_collect_contributions!(contributions, to_julia_mna, item.stmt)
                analog_block_ast = item.stmt  # Store for short circuit detection
            end
            _ => nothing
        end
    end

    # Collect variable declarations from inside analog blocks (named blocks like "begin : evaluateblock")
    # These are not at the module's top level but need to be initialized in local_var_init
    if analog_block_ast !== nothing
        collect_nested_var_decls!(var_types, var_inits, analog_block_ast)
    end

    # Detect short circuits (V(internal, external) <+ 0) for node aliasing
    short_circuits = if analog_block_ast !== nothing && !isempty(internal_nodes)
        detect_short_circuits(analog_block_ast, to_julia_mna, internal_nodes)
    else
        Dict{Symbol, NamedTuple{(:external, :condition), Tuple{Symbol, Any}}}()
    end

    # Generate parameter extraction
    params_to_locals = map(collect(parameter_names)) do id
        :($id = $(undefault)(dev.$id))
    end

    # Generate variable declarations for non-parameter local vars
    # Use initialization expression if provided, otherwise default to type-appropriate zero value
    local_var_decls = Any[]
    for (name, T) in var_types
        if !(name in parameter_names)
            # Type-appropriate default: zero for numeric, "" for String
            default_init = T === String ? "" : :(zero($T))
            init_expr = get(var_inits, name, default_init)
            push!(local_var_decls, :(local $name::$T = $init_expr))
        end
    end

    #===========================================================================#
    # Code Splitting: Classify statements as static vs dynamic
    #
    # Static code (parameter/temperature dependent) goes in init_device!()
    # Dynamic code (voltage dependent) stays in stamp!()
    #===========================================================================#

    taint = CodegenTaint()

    # Mark parameters as static
    for p in parameter_names
        push!(taint.static_vars, p)
    end
    # Mark spec-related as static
    push!(taint.static_vars, :_mna_spec_)
    push!(taint.static_vars, :dev)

    # Classify local variable initializations
    for decl in local_var_decls
        if decl.head == :local
            inner = decl.args[1]
            if inner isa Expr && inner.head == :(=)
                lhs = inner.args[1]
                rhs = inner.args[2]
                if lhs isa Expr && lhs.head == :(::)
                    name = lhs.args[1]
                    var_type = lhs.args[2]
                    # Create a simple assignment for classification
                    simple_stmt = :($name = $rhs)
                    classify_assignment!(taint, name, rhs, simple_stmt)
                elseif lhs isa Symbol
                    classify_assignment!(taint, lhs, rhs, inner)
                end
            end
        end
    end

    # Classify contributions - assignments and conditionals
    for c in contributions
        if c.kind == :assignment
            expr = c.expr
            if expr isa Expr && expr.head in (:(=), :(+=), :(-=), :(*=), :(/=))
                lhs = expr.args[1]
                rhs = expr.args[2]
                if lhs isa Symbol
                    classify_assignment!(taint, lhs, rhs, expr)
                else
                    # Complex LHS (e.g., array indexing) - assume dynamic
                    push!(taint.dynamic_stmts, expr)
                end
            else
                # Not a simple assignment - check if static
                if is_static_expr(taint, expr)
                    push!(taint.static_stmts, expr)
                else
                    push!(taint.dynamic_stmts, expr)
                end
            end
        elseif c.kind == :conditional
            # Conditionals: if any branch has V/I access, entire block is dynamic
            if is_static_expr(taint, c.expr)
                push!(taint.static_stmts, c.expr)
                # Also mark variables assigned inside static conditionals as static
                for sym in extract_assigned_symbols(c.expr)
                    push!(taint.static_vars, sym)
                end
            else
                push!(taint.dynamic_stmts, c.expr)
                # IMPORTANT: Mark variables assigned inside dynamic conditionals as dynamic
                # This ensures they get proper initialization and aren't loaded from cache
                for sym in extract_assigned_symbols(c.expr)
                    push!(taint.dynamic_vars, sym)
                end
            end
        end
        # Current/voltage contributions are always dynamic (handled in stamp_code)
    end

    # Identify which static vars are actually used in dynamic code
    # (these need to be in the cache and loaded at stamp! start)
    # IMPORTANT: Exclude variables that are also assigned dynamically (in dynamic_vars)
    # because they will be computed in stamp! and shouldn't be loaded from cache
    for stmt in taint.dynamic_stmts
        syms = extract_symbols(stmt)
        for sym in syms
            if sym in taint.static_vars && !(sym in taint.dynamic_vars)
                push!(taint.used_static_vars, sym)
            end
        end
    end

    # Also check current contribution expressions for static var usage
    for c in contributions
        if c.kind == :current || c.kind == :voltage
            syms = extract_symbols(c.expr)
            for sym in syms
                if sym in taint.static_vars && !(sym in taint.dynamic_vars)
                    push!(taint.used_static_vars, sym)
                end
            end
        end
    end

    # CRITICAL: Statements that assign to dynamic vars must be in dynamic_stmts,
    # even if the RHS is static. This ensures the assignment runs in stamp!().
    # Also track the static vars used in such assignments for caching.
    stmts_to_move = Expr[]
    for stmt in taint.static_stmts
        if stmt isa Expr && stmt.head in (:(=), :(+=), :(-=), :(*=), :(/=))
            lhs = stmt.args[1]
            rhs = stmt.args[2]
            # If this statement assigns to a variable that's now in dynamic_vars,
            # move it to dynamic_stmts and track static vars from RHS
            lhs_sym = lhs isa Symbol ? lhs : (lhs isa Expr && lhs.head == :(::) ? lhs.args[1] : nothing)
            if lhs_sym !== nothing && lhs_sym in taint.dynamic_vars
                push!(stmts_to_move, stmt)
                # Extract static vars from the RHS - they're needed for dynamic computation
                syms = extract_symbols(rhs)
                for sym in syms
                    if sym in taint.static_vars && !(sym in taint.dynamic_vars)
                        push!(taint.used_static_vars, sym)
                    end
                end
            end
        end
    end
    # Move statements from static to dynamic
    for stmt in stmts_to_move
        filter!(s -> s !== stmt, taint.static_stmts)
        push!(taint.dynamic_stmts, stmt)
    end

    # Build static variable types dict (for cache struct generation)
    static_var_types = Dict{Symbol, Type}()
    for var in taint.used_static_vars
        if var in parameter_names
            # Skip raw parameters - they're already in the device struct
            continue
        end
        # Skip function arguments (dev, _mna_spec_) - they're already available
        if var in (:dev, :_mna_spec_)
            continue
        end
        T = get(var_types, var, Float64)
        static_var_types[var] = T
    end

    # Debug output
    if get(ENV, "CEDARSIM_DEBUG_TAINT", "") == "1"
        @info "Code splitting taint analysis for $symname" static_vars=taint.static_vars dynamic_vars=taint.dynamic_vars used_static_vars=taint.used_static_vars static_stmts=length(taint.static_stmts) dynamic_stmts=length(taint.dynamic_stmts) static_var_types=static_var_types

        # Debug specific variables of interest
        for var in [:DIOtSatCur, :DIOtSatSWCur, :csat, :csatsw, :cdb, :gdb, :load_cd, :cdeq]
            in_static = var in taint.static_vars
            in_dynamic = var in taint.dynamic_vars
            in_used = var in taint.used_static_vars
            in_types = haskey(static_var_types, var)
            @info "Variable $var: static=$in_static dynamic=$in_dynamic used=$in_used in_cache=$in_types"
        end
    end

    # Generate cache struct (only if there are static vars to cache)
    cache_name = Symbol(symname, "Cache")
    has_cache = !isempty(static_var_types)

    # Generate stamp method using unified n-terminal approach
    # (works for any number of terminals, including 2)
    port_args = ps
    stamp_method = generate_mna_stamp_method_nterm(
        symname, ps, port_args, internal_nodes, params_to_locals, local_var_decls,
        function_defs, contributions, to_julia_mna, short_circuits,
        taint, cache_name, has_cache)

    # Build struct and constructor directly without @kwdef to avoid macro hygiene issues
    # that rename field symbols in baremodule contexts

    # Filter out alias parameters from struct fields (aliases don't need storage)
    real_params = filter(p -> !haskey(aliases, p), parameter_names)

    # 1. Build plain struct definition (only real parameters, not aliases)
    struct_body = Expr(:block)
    for paramsym in real_params
        pT = get(param_types, paramsym, Float64)
        push!(struct_body.args, Expr(:(::), paramsym, :(CedarSim.DefaultOr{$pT})))
    end
    struct_def = Expr(:struct, false,
        Expr(:<:, symname, :(VerilogAEnvironment.VAModel)),
        struct_body)

    # 2. Build keyword constructor that mimics @kwdef
    # Constructor accepts both real params and aliases
    # Aliases forward their value to the target parameter
    if !isempty(real_params) || !isempty(aliases)
        # Build keyword parameter list with defaults
        kw_params = Expr(:parameters)
        call_args = Any[]

        # Add real parameters
        for paramsym in real_params
            # Each parameter: paramsym = mkdefault(default_value)
            # Use the actual default value from the VA parameter declaration
            pT = get(param_types, paramsym, Float64)
            fallback_default = pT === String ? "" : 0.0
            default_val = get(param_defaults, paramsym, fallback_default)
            push!(kw_params.args, Expr(:kw, paramsym, :(CedarSim.mkdefault($default_val))))
        end

        # Add alias parameters (they default to nothing, meaning "use target's value")
        for (alias_sym, _) in aliases
            push!(kw_params.args, Expr(:kw, alias_sym, :nothing))
        end

        # Build call args: for each real param, check if an alias was provided
        for paramsym in real_params
            # Find aliases that target this parameter
            targeting_aliases = [a for (a, t) in aliases if t == paramsym]
            if isempty(targeting_aliases)
                # No aliases, just use the parameter directly
                push!(call_args, Expr(:call, :(VerilogAEnvironment.vaconvert),
                    :(CedarSim.notdefault(fieldtype($symname, $(QuoteNode(paramsym))))),
                    paramsym))
            else
                # Has aliases - use alias value if provided, otherwise use parameter
                # Build: something(alias1, alias2, ..., paramsym) where something picks first non-nothing
                alias_expr = paramsym
                for alias_sym in reverse(targeting_aliases)
                    alias_expr = :($alias_sym !== nothing ? $alias_sym : $alias_expr)
                end
                push!(call_args, Expr(:call, :(VerilogAEnvironment.vaconvert),
                    :(CedarSim.notdefault(fieldtype($symname, $(QuoteNode(paramsym))))),
                    alias_expr))
            end
        end

        # Build constructor function
        constructor = Expr(:function,
            Expr(:call, symname, kw_params),
            Expr(:call, symname, call_args...))
    else
        constructor = nothing
    end

    # 3. Build getproperty override to support alias access
    # Base.getproperty(dev::TypeName, s::Symbol) = s == :alias ? getfield(dev, :target) : getfield(dev, s)
    getproperty_override = nothing
    if !isempty(aliases)
        # Build the if-elseif chain from inside out (rightmost first)
        # Start with the final else clause
        alias_list = collect(aliases)
        alias_checks = :(getfield(dev, s))  # Default: direct field access

        # Build chain from last alias to first
        for i in length(alias_list):-1:1
            (alias_sym, target_sym) = alias_list[i]
            check = :(s == $(QuoteNode(alias_sym)))
            result = :(getfield(dev, $(QuoteNode(target_sym))))
            alias_checks = Expr(:if, check, result, alias_checks)
        end

        getproperty_override = :(function Base.getproperty(dev::$symname, s::Symbol)
            $alias_checks
        end)
    end

    result_args = Any[struct_def]
    if constructor !== nothing
        push!(result_args, constructor)
    end
    if getproperty_override !== nothing
        push!(result_args, getproperty_override)
    end

    # Add cache-related methods if there are static vars to cache
    if has_cache
        # Generate cache struct
        cache_struct = generate_cache_struct(symname, static_var_types)
        push!(result_args, cache_struct)

        # Filter out parameters and function arguments from used_static_vars
        # Only local variables go in cache
        cached_local_vars = Set{Symbol}()
        for var in taint.used_static_vars
            if var in parameter_names
                continue
            end
            # Skip function arguments - they're already available
            if var in (:dev, :_mna_spec_)
                continue
            end
            push!(cached_local_vars, var)
        end

        # Generate init_device! method
        init_method = generate_init_device_method(
            symname, cache_name, params_to_locals, local_var_decls,
            taint.static_stmts, cached_local_vars, function_defs)
        push!(result_args, init_method)

        # Generate make_cache method
        make_cache_method = generate_make_cache_method(symname, cache_name)
        push!(result_args, make_cache_method)
    end

    push!(result_args, stamp_method)

    Expr(:toplevel, result_args...)
end

"""
MNA-specific scope that translates VA constructs for MNA stamping.
"""
struct MNAScope
    parameters::Set{Symbol}
    node_order::Vector{Symbol}
    ninternal_nodes::Int
    branch_order::Vector{Pair{Symbol}}
    used_branches::Set{Pair{Symbol}}
    var_types::Dict{Symbol, Union{Type{Int}, Type{Float64}, Type{String}}}
    all_functions::Dict{Symbol, VAFunction}
    undefault_ids::Bool
    ddx_order::Vector{Symbol}
    named_branches::Dict{Symbol, Pair{Symbol,Symbol}}  # Maps branch name -> (pos, neg) nodes
end

# Literal parsing methods
function (::MNAScope)(cs::VANode{Literal})
    Meta.parse(String(cs))
end

function (::MNAScope)(cs::VANode{FloatLiteral})
    txt = String(cs)
    sf = nothing
    if is_scale_factor(txt[end])
        sf = txt[end]
        txt = txt[1:end-1]
    end
    ret = Base.parse(Float64, txt)
    if sf !== nothing
        ret *= sf_mapping[sf]
    end
    return ret
end

function (scope::MNAScope)(ip::VANode{IdentifierPrimary})
    # Check if this is a system identifier that needs special translation
    # (e.g., $temperature, $vt, $simparam)
    if isa(ip.id, VANode{SystemIdentifier})
        return scope(ip.id)  # Delegate to SystemIdentifier handler
    end
    id = Symbol(assemble_id_string(ip.id))
    if scope.undefault_ids
        id = Expr(:call, undefault, id)
    end
    id
end

function (to_julia::MNAScope)(cs::VANode{BinaryExpression})
    op = Symbol(cs.op)
    if op == :(||)
        # Use va_or which handles mixed Bool/Float64/Dual types
        return Expr(:call, :va_or, to_julia(cs.lhs), to_julia(cs.rhs))
    elseif op == :(&&)
        # Use va_and which handles mixed Bool/Float64/Dual types
        return Expr(:call, :va_and, to_julia(cs.lhs), to_julia(cs.rhs))
    elseif op == Symbol("**")
        # Power operator (**) in Verilog-A: use `pow` from VerilogAEnvironment
        # which uses NaNMath.pow and handles ForwardDiff duals correctly
        return Expr(:call, :pow, to_julia(cs.lhs), to_julia(cs.rhs))
    elseif op == :^
        # XOR operator (^) in Verilog-A: bitwise XOR for integers
        # Note: ^ is NOT power in Verilog-A! Power is **
        # We pass through to the VA environment's ^ which is Base.:(⊻)
        return Expr(:call, :^, to_julia(cs.lhs), to_julia(cs.rhs))
    else
        return Expr(:call, op, to_julia(cs.lhs), to_julia(cs.rhs))
    end
end

function (to_julia::MNAScope)(stmt::VANode{UnaryOp})
    return Expr(:call, Symbol(stmt.op), to_julia(stmt.operand))
end

function (to_julia::MNAScope)(stmt::VANode{Parens})
    return to_julia(stmt.inner)
end

function (to_julia::MNAScope)(cs::VANode{TernaryExpr})
    # Convert condition to bool (VA integers are truthy)
    cond_bool = :(!(iszero($(to_julia(cs.condition)))))
    return Expr(:if, cond_bool, to_julia(cs.ifcase), to_julia(cs.elsecase))
end

function (to_julia::MNAScope)(stmt::VANode{FunctionCall})
    fname = Symbol(stmt.id)

    if fname == :V
        # Voltage access - return variable name that will be replaced with Vpn
        @assert length(stmt.args) in (1, 2)
        id1 = Symbol(stmt.args[1].item)

        # Check if this is a named branch (e.g., V(br) where br is a branch)
        if length(stmt.args) == 1 && haskey(to_julia.named_branches, id1)
            # Named branch: V(br) -> V_pos - V_neg
            branch_nodes = to_julia.named_branches[id1]
            pos_node, neg_node = branch_nodes.first, branch_nodes.second
            push!(to_julia.used_branches, pos_node => neg_node)
            if neg_node == Symbol("0")
                return pos_node
            else
                return :($pos_node - $neg_node)
            end
        end

        id2 = length(stmt.args) > 1 ? Symbol(stmt.args[2].item) : Symbol("0")
        push!(to_julia.used_branches, id1 => id2)

        if id2 == Symbol("0")
            return id1
        else
            return :($id1 - $id2)
        end
    elseif fname == :I
        # Current access
        @assert length(stmt.args) in (1, 2)

        if length(stmt.args) == 1
            # I(br) - current through named branch
            branch_name = Symbol(stmt.args[1].item)
            if haskey(to_julia.named_branches, branch_name)
                # Named branch: I(br) returns the branch current variable
                # The current variable is accessed via a special symbol that will be
                # replaced with the actual current index in the generated code
                branch_nodes = to_julia.named_branches[branch_name]
                push!(to_julia.used_branches, branch_nodes.first => branch_nodes.second)
                # Return a symbol that represents the branch current
                # This will be provided as a variable in the generated stamp function
                return Symbol("_I_branch_", branch_name)
            else
                return :(error("I() with single argument requires a named branch"))
            end
        else
            # I(a, b) - not directly supported in contribution-based stamping
            return :(error("I(a,b) probe not supported in MNA contribution"))
        end
    elseif fname == :ddt
        # Time derivative - use va_ddt
        return Expr(:call, :va_ddt, to_julia(stmt.args[1].item))
    elseif fname == :ddx
        # Partial derivative - ddx(expr, V(a,b)) returns ∂expr/∂V(a,b)
        # For n-terminal MNA devices, duals are indexed by node_order (port positions)
        item = stmt.args[2].item
        @assert formof(item) == FunctionCall
        @assert Symbol(item.id) == :V
        if length(item.args) == 1
            probe = Symbol(item.args[1].item)
            # Use node_order for partial index (duals indexed by port position)
            id_idx = findfirst(==(probe), to_julia.node_order)
            return :(let x = $(to_julia(stmt.args[1].item))
                isa(x, Dual) ? @inbounds(ForwardDiff.partials(x, $id_idx)) : 0.0
            end)
        else
            probe1 = Symbol(item.args[1].item)
            id1_idx = findfirst(==(probe1), to_julia.node_order)
            probe2 = Symbol(item.args[2].item)
            id2_idx = findfirst(==(probe2), to_julia.node_order)
            # ∂expr/∂V(a,b) = (∂expr/∂V_a - ∂expr/∂V_b) / 2
            # This works because V(a,b) = V_a - V_b, so:
            # ∂expr/∂V_a = ∂expr/∂V(a,b) and ∂expr/∂V_b = -∂expr/∂V(a,b)
            return :(let x = $(to_julia(stmt.args[1].item)),
                        dx1 = isa(x, Dual) ? @inbounds(ForwardDiff.partials(x, $id1_idx)) : 0.0,
                        dx2 = isa(x, Dual) ? @inbounds(ForwardDiff.partials(x, $id2_idx)) : 0.0
                (dx1 - dx2) / 2
            end)
        end
    elseif fname == Symbol("\$temperature")
        return :(_mna_spec_.temp + 273.15)  # Convert to Kelvin
    elseif fname == Symbol("\$vt")
        return :((_mna_spec_.temp + 273.15) * 8.617333262e-5)  # kT/q
    elseif fname == Symbol("\$param_given")
        # Check if a parameter was explicitly specified (not using default)
        id = Symbol(stmt.args[1].item)
        return Expr(:call, :!, Expr(:call, CedarSim.isdefault,
            Expr(:., :dev, QuoteNode(id))))
    elseif fname == Symbol("\$simparam")
        # Simulator parameter access - $simparam("name") or $simparam("name", default)
        if stmt.args[1].item isa VANode{StringLiteral}
            param_str = String(stmt.args[1].item)[2:end-1]  # Strip quotes
        else
            param_str = String(stmt.args[1].item)
        end
        param_sym = Symbol(param_str)
        if length(stmt.args) == 1
            # No default - error if not found
            return :(hasproperty(_mna_spec_, $(QuoteNode(param_sym))) ?
                     getproperty(_mna_spec_, $(QuoteNode(param_sym))) :
                     error("Unknown simparam: " * $param_str))
        else
            # With default value
            default_expr = to_julia(stmt.args[2].item)
            return :(hasproperty(_mna_spec_, $(QuoteNode(param_sym))) ?
                     getproperty(_mna_spec_, $(QuoteNode(param_sym))) :
                     $default_expr)
        end
    elseif fname == :analysis
        # Analysis type check - returns true if current analysis matches the string
        # Mapping:
        #   "dc" or "static" -> _mna_spec_.mode == :dcop
        #   "tran" or "transient" -> _mna_spec_.mode == :tran
        #   "ac" -> _mna_spec_.mode == :ac
        #   "nodeset" -> false (not supported)
        @assert length(stmt.args) == 1 "analysis() takes exactly one argument"
        analysis_str = to_julia(stmt.args[1].item)
        if analysis_str isa String
            analysis_sym = analysis_str
        else
            # If it's not a constant string, evaluate at runtime
            return :(
                let atype = $analysis_str
                    if atype == "dc" || atype == "static"
                        _mna_spec_.mode == :dcop
                    elseif atype == "tran" || atype == "transient"
                        _mna_spec_.mode == :tran
                    elseif atype == "ac"
                        _mna_spec_.mode == :ac
                    else
                        false
                    end
                end
            )
        end
        # For constant strings, generate simpler code
        if analysis_sym == "dc" || analysis_sym == "static"
            return :(_mna_spec_.mode == :dcop)
        elseif analysis_sym == "tran" || analysis_sym == "transient"
            return :(_mna_spec_.mode == :tran)
        elseif analysis_sym == "ac"
            return :(_mna_spec_.mode == :ac)
        else
            return false
        end
    elseif fname == Symbol("\$limit")
        # $limit(voltage, limiter_fn, ...) - voltage limiting for Newton convergence
        # In our MNA implementation, we can optionally apply limiting or just return the voltage
        # For now, we simply return the voltage value without limiting
        # This allows the model to run, though convergence may be slower
        voltage_expr = to_julia(stmt.args[1].item)
        return voltage_expr
    end

    # Noise functions - return 0 in MNA (noise not simulated in DC/transient)
    if fname in (:white_noise, :flicker_noise)
        return 0.0
    end

    # Check for VA-defined function
    vaf = get(to_julia.all_functions, fname, nothing)
    if vaf !== nothing
        # Call to a Verilog-A defined function - handle output/inout parameters
        args = map(x -> to_julia(x.item), stmt.args)
        in_args = Any[]
        out_args = Any[]

        if length(args) != length(vaf.arg_order)
            return Expr(:call, error, "Wrong number of arguments to function $fname ($args, $(vaf.arg_order))")
        end

        for (arg, vaarg) in zip(args, vaf.arg_order)
            kind = vaf.inout_decls[vaarg]
            if kind == :output || kind == :inout
                isa(arg, Symbol) || return Expr(:call, error, "Output argument $vaarg to function $fname must be a symbol. Got $arg.")
                push!(out_args, arg)
            end
            if kind == :input || kind == :inout
                push!(in_args, arg)
            end
        end
        ret = Expr(:call, fname, in_args...)
        if length(out_args) != 0
            s = gensym()
            ret = @nolines quote
                ($s, ($(out_args...),)) = $ret
                $s
            end
        end
        return ret
    end

    # Default: pass through function call
    # Strip $ prefix for system functions (e.g., $pow -> pow, $ln -> ln)
    fname_str = String(fname)
    if startswith(fname_str, "\$")
        fname = Symbol(fname_str[2:end])
    end
    return Expr(:call, fname, map(x -> to_julia(x.item), stmt.args)...)
end

function (to_julia::MNAScope)(stmt::VANode{AnalogVariableAssignment})
    assignee = Symbol(stmt.lvalue)
    varT = get(to_julia.var_types, assignee, Float64)

    eq = stmt.eq.op
    op = eq == VAT.EQ ? :(=) :
         eq == VAT.PLUS_EQ ? :(+=) :
         eq == VAT.MINUS_EQ ? :(-=) :
         eq == VAT.STAR_EQ ? :(*=) :
         eq == VAT.SLASH_EQ ? :(/=) :
         :(=)

    return Expr(op, assignee,
        Expr(:call, VerilogAEnvironment.vaconvert, varT, to_julia(stmt.rvalue)))
end

function (to_julia::MNAScope)(stmt::VANode{AnalogProceduralAssignment})
    return to_julia(stmt.assign)
end
function (to_julia::MNAScope)(stmt::VANode{FunctionCallStatement})
    return to_julia(stmt.call)
end

function (to_julia::MNAScope)(asb::VANode{AnalogSeqBlock})
    ret = Expr(:block)
    for stmt in asb.stmts
        push!(ret.args, to_julia(stmt))
    end
    ret
end

(to_julia::MNAScope)(stmt::VANode{AnalogStatement}) = to_julia(stmt.stmt)

function (to_julia::MNAScope)(stmt::VANode{AnalogConditionalBlock})
    aif = stmt.aif
    function if_body_to_julia(ifstmt)
        if formof(ifstmt) == AnalogSeqBlock
            return to_julia(ifstmt)
        else
            return Expr(:block, to_julia(ifstmt))
        end
    end

    # Convert VA condition to boolean - in VA, integers are truthy (non-zero = true)
    function va_condition_to_bool(cond_expr)
        # Wrap in !iszero() to convert any numeric type to Bool
        :(!(iszero($(to_julia(cond_expr)))))
    end

    ifex = ex = Expr(:if, va_condition_to_bool(aif.condition), if_body_to_julia(aif.stmt))
    for case in stmt.elsecases
        if formof(case.stmt) == AnalogIf
            elif = case.stmt
            newex = Expr(:elseif, va_condition_to_bool(elif.condition), if_body_to_julia(elif.stmt))
            push!(ex.args, newex)
            ex = newex
        else
            push!(ex.args, if_body_to_julia(case.stmt))
        end
    end
    ifex
end

# Handle system identifiers like $mfactor
function (to_julia::MNAScope)(ip::VANode{SystemIdentifier})
    id = Symbol(ip)
    if id == Symbol("\$mfactor")
        # Device multiplicity - default to 1.0
        return :(hasproperty(_mna_spec_, :mfactor) ? _mna_spec_.mfactor : 1.0)
    elseif id == Symbol("\$temperature")
        return :(_mna_spec_.temp + 273.15)
    else
        # For other system identifiers, return as function call
        return Expr(:call, id)
    end
end

# Handle analog system task enable (e.g., $warning, $strobe)
function (to_julia::MNAScope)(stmt::VANode{AnalogSystemTaskEnable})
    if formof(stmt.task) == FunctionCall
        fc = stmt.task
        fname = Symbol(fc.id)
        args = map(x -> to_julia(x.item), fc.args)
        if fname == Symbol("\$warning")
            # Warnings are suppressed in MNA simulation
            return nothing
        elseif fname == Symbol("\$strobe")
            return nothing
        elseif fname == Symbol("\$error")
            return Expr(:call, :error, args...)
        elseif fname == Symbol("\$discontinuity")
            # Discontinuity markers are no-ops in MNA
            return nothing
        else
            # Default: treat as regular function call (strip $ prefix)
            fname_str = String(fname)
            if startswith(fname_str, "\$")
                fname = Symbol(fname_str[2:end])
            end
            return Expr(:call, fname, args...)
        end
    else
        return nothing
    end
end

# Handle string literals
function (to_julia::MNAScope)(stmt::VANode{StringLiteral})
    return String(stmt)[2:end-1]  # Strip quotes
end

# Handle analog for loops
function (to_julia::MNAScope)(stmt::VANode{AnalogFor})
    body = to_julia(stmt.stmt)
    push!(body.args, to_julia(stmt.update_stmt))
    # Convert VA condition to boolean
    cond = :(!(iszero($(to_julia(stmt.cond_expr)))))
    while_expr = Expr(:while, cond, body)
    Expr(:block, to_julia(stmt.init_stmt), while_expr)
end

# Handle analog while loops
function (to_julia::MNAScope)(stmt::VANode{AnalogWhile})
    body = to_julia(stmt.stmt)
    # Convert VA condition to boolean
    cond = :(!(iszero($(to_julia(stmt.cond_expr)))))
    Expr(:while, cond, body)
end

# Handle analog repeat loops
function (to_julia::MNAScope)(stmt::VANode{AnalogRepeat})
    body = to_julia(stmt.stmt)
    Expr(:for, :(_ = 1:$(stmt.num_repeat)), body)
end

# Handle contribution statements inside conditionals
# When a contribution is inside an if-block, we generate inline stamping code
function (to_julia::MNAScope)(cs::VANode{ContributionStatement})
    bpfc = cs.lvalue
    kind_sym = Symbol(bpfc.id)
    kind = kind_sym == :I ? :current : kind_sym == :V ? :voltage : :unknown

    refs = map(bpfc.references) do ref
        Symbol(assemble_id_string(ref.item))
    end

    p_sym = refs[1]
    n_sym = length(refs) > 1 ? refs[2] : Symbol("0")
    push!(to_julia.used_branches, p_sym => n_sym)

    # Get the node variable names (using the convention from generate_mna_stamp_method_nterm)
    # Handle ground node (Symbol("0")) specially - it maps to integer 0
    p_idx = p_sym == Symbol("0") ? nothing : findfirst(==(p_sym), to_julia.node_order)
    n_idx = n_sym == Symbol("0") ? nothing : findfirst(==(n_sym), to_julia.node_order)
    # Ground node uses literal 0; other nodes use their parameter symbol
    p_node = p_idx === nothing ? 0 : Symbol("_node_", p_sym)
    n_node = n_idx === nothing ? 0 : Symbol("_node_", n_sym)

    # For voltage contributions (V(a,b) <+ expr), we need proper MNA stamping
    # with a branch current variable to carry DC current (essential for short circuits)
    if kind == :voltage
        # Voltage contribution: V(p,n) <+ value means we enforce V_p - V_n = value
        # This requires a branch current variable for proper DC current flow
        expr = to_julia(cs.assign_expr)
        # Create a unique name for this voltage contribution's current variable
        # OPTIMIZATION: Use component-based API to avoid Symbol interpolation at call site
        # For DirectStampContext, both names are ignored (counter-based access)
        # For MNAContext, the full name is built from components
        I_alloc_base_name = QuoteNode(Symbol("I_V_", p_sym, "_", n_sym))
        return quote
            # Voltage contribution V($p_sym, $n_sym) <+ $expr
            # Skip if nodes are aliased (short circuit optimization)
            if $p_node != $n_node
                # Allocate branch current (idempotent - returns existing index if already allocated)
                let I_var = CedarSim.MNA.alloc_current!(ctx, $I_alloc_base_name, _mna_instance_)
                    v_contrib_raw = $expr
                    v_val = v_contrib_raw isa ForwardDiff.Dual ? ForwardDiff.value(v_contrib_raw) : Float64(v_contrib_raw)

                    # Stamp proper MNA voltage source:
                    # - KCL at p: current I flows out → G[p, I] = 1
                    # - KCL at n: current I flows in → G[n, I] = -1
                    # - Voltage constraint: V_p - V_n = v_val → G[I, p] = 1, G[I, n] = -1, b[I] = v_val
                    if $p_node != 0
                        CedarSim.MNA.stamp_G!(ctx, $p_node, I_var, 1.0)
                        CedarSim.MNA.stamp_G!(ctx, I_var, $p_node, 1.0)
                    end
                    if $n_node != 0
                        CedarSim.MNA.stamp_G!(ctx, $n_node, I_var, -1.0)
                        CedarSim.MNA.stamp_G!(ctx, I_var, $n_node, -1.0)
                    end
                    CedarSim.MNA.stamp_b!(ctx, I_var, v_val)
                end
            end
        end
    end

    # Current contribution: generate inline contribution stamping with full Jacobian
    # This is used when contributions are inside conditionals
    expr = to_julia(cs.assign_expr)
    # node_order is [ports; internal_nodes; ground] - exclude ground from Jacobian
    # Duals are only created for non-ground nodes (ports + internal)
    all_node_syms = to_julia.node_order
    n_nonground = length(all_node_syms) - 1  # Exclude ground at end

    # Build the Jacobian extraction code - extract from the INNER JacobianTag Dual
    # When I_branch is Dual{ContributionTag, Dual{JacobianTag}}, we need value(I_branch) first
    jac_extract_code = Any[]
    for k in 1:n_nonground
        dI_dVk = Symbol("dI_dV", k)
        push!(jac_extract_code, :($dI_dVk = I_resist isa ForwardDiff.Dual ? ForwardDiff.partials(I_resist, $k) : 0.0))
    end

    # Build Jacobian stamping code
    jac_stamp_code = Any[]
    rhs_terms = Any[:I_val]
    for k in 1:n_nonground
        node_sym = all_node_syms[k]
        k_node = Symbol("_node_", node_sym)
        dI_dVk = Symbol("dI_dV", k)
        V_k = Symbol("V_", k)

        # Stamp Jacobian into G matrix
        push!(jac_stamp_code, quote
            if $p_node != 0 && $k_node != 0
                CedarSim.MNA.stamp_G!(ctx, $p_node, $k_node, $dI_dVk)
            end
            if $n_node != 0 && $k_node != 0
                CedarSim.MNA.stamp_G!(ctx, $n_node, $k_node, -$dI_dVk)
            end
        end)

        # Build RHS terms: I_val - sum(dI/dVk * Vk)
        push!(rhs_terms, :(- $dI_dVk * $V_k))
    end

    rhs_expr = Expr(:call, :+, rhs_terms...)

    # For contributions inside conditionals, we evaluate and stamp inline with full Jacobian
    # Handle nested Duals: ContributionTag wraps JacobianTag when ddt() is used
    return quote
        # Contribution I($p_sym, $n_sym) <+ ...
        let I_branch = $expr
            # Extract value and the resistive Dual (for Jacobian extraction)
            # ContributionTag wraps the JacobianTag Dual - we need to unwrap it
            local I_resist
            if I_branch isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                # Has ddt: ContributionTag outer, JacobianTag inner
                I_resist = ForwardDiff.value(I_branch)  # Inner Dual{JacobianTag}
                I_val = ForwardDiff.value(I_resist)
            elseif I_branch isa ForwardDiff.Dual
                # Pure resistive: just voltage dual (JacobianTag)
                I_resist = I_branch
                I_val = ForwardDiff.value(I_branch)
            else
                # Scalar result
                I_resist = I_branch
                I_val = Float64(I_branch)
            end

            # Extract Jacobian partials from the resistive Dual (JacobianTag)
            $(jac_extract_code...)

            # Stamp Jacobian into G matrix
            $(jac_stamp_code...)

            # Stamp RHS using Newton companion model
            let Ieq = $rhs_expr
                if $p_node != 0
                    CedarSim.MNA.stamp_b!(ctx, $p_node, -Ieq)
                end
                if $n_node != 0
                    CedarSim.MNA.stamp_b!(ctx, $n_node, Ieq)
                end
            end
        end
    end
end

# Handle case statements
function (to_julia::MNAScope)(stmt::VANode{CaseStatement})
    s = gensym()
    first = true
    expr = nothing
    default_case = nothing
    for case in stmt.cases
        if isa(case.conds, Node)
            # Default case
            default_case = to_julia(case.item)
        else
            conds = map(cond -> :($s == $(to_julia(cond.item))), case.conds)
            cond = length(conds) == 1 ? conds[1] : Expr(:(||), conds...)
            ex = Expr(first ? :if : :elseif, cond, to_julia(case.item))
            if first
                expr = ex
                first = false
            else
                push!(expr.args, ex)
            end
        end
    end
    if expr === nothing
        return default_case !== nothing ? default_case : Expr(:block)
    end
    default_case !== nothing && push!(expr.args, default_case)
    Expr(:block, :($s = $(to_julia(stmt.switch))), expr)
end

function (to_julia::MNAScope)(fd::VANode{AnalogFunctionDeclaration})
    # Similar to Scope version but uses MNAScope for body
    type_decls = Dict{Symbol, Any}()
    inout_decls = Dict{Symbol, Symbol}()
    fname = Symbol(fd.id)
    var_types = Dict{Symbol, Union{Type{Int}, Type{Float64}}}()
    rt = fd.fty === nothing ? Real : kw_to_T(fd.fty.kw)
    var_types[fname] = rt
    arg_order = Symbol[]

    for decl in fd.items
        item = decl.item
        @case formof(item) begin
            InOutDeclaration => begin
                kind = item.kw.kw === INPUT ? :input :
                       item.kw.kw === OUTPUT ? :output :
                       :inout
                for name in item.portnames
                    ns = Symbol(name.item)
                    inout_decls[ns] = kind
                    push!(arg_order, ns)
                end
            end
            IntRealDeclaration => begin
                T = kw_to_T(item.kw.kw)
                for ident in item.idents
                    # ident.item is IntRealVarDecl with id, eq, init fields
                    vardecl = ident.item
                    name = Symbol(assemble_id_string(vardecl.id))
                    var_types[name] = T
                end
            end
        end
    end

    to_julia_internal = MNAScope(to_julia.parameters, to_julia.node_order,
        to_julia.ninternal_nodes, to_julia.branch_order, to_julia.used_branches, var_types,
        to_julia.all_functions, to_julia.undefault_ids, to_julia.ddx_order,
        to_julia.named_branches)

    in_args = [k for k in arg_order if inout_decls[k] in (:input, :inout)]
    out_args = [k for k in arg_order if inout_decls[k] in (:output, :inout)]
    rt_decl = length(out_args) == 0 ? fname : :(($fname, ($(out_args...),)))

    to_julia.all_functions[fname] = VAFunction(arg_order, inout_decls)

    localize_vars = Any[]
    for var in keys(var_types)
        var in arg_order && continue
        push!(localize_vars, :(local $var))
    end

    return @nolines quote
        @inline function $fname($(in_args...))
            $(localize_vars...)
            local $fname = VerilogAEnvironment.vaconvert($rt, 0)
            $(to_julia_internal(fd.stmt))
            return $rt_decl
        end
    end
end

"""
Collect contribution statements and regular statements from analog block for MNA stamping.
"""
function mna_collect_contributions!(contributions, to_julia::MNAScope, stmt)
    if stmt isa VANode{ContributionStatement}
        push!(contributions, mna_translate_contribution(to_julia, stmt))
    elseif stmt isa VANode{AnalogSeqBlock}
        for s in stmt.stmts
            mna_collect_contributions!(contributions, to_julia, s)
        end
    elseif stmt isa VANode{AnalogStatement}
        mna_collect_contributions!(contributions, to_julia, stmt.stmt)
    elseif stmt isa VANode{AnalogConditionalBlock}
        # For conditional blocks, we need to handle them specially
        # For now, add the whole translated block
        push!(contributions, (kind=:conditional, expr=to_julia(stmt)))
    elseif stmt isa VANode{AnalogVariableAssignment}
        # Regular assignments (e.g., cdrain = R*V(g,s)**2)
        push!(contributions, (kind=:assignment, expr=to_julia(stmt)))
    elseif stmt isa VANode{AnalogProceduralAssignment}
        # Procedural assignments in analog block (e.g., cdrain = R*V(g,s)**2;)
        push!(contributions, (kind=:assignment, expr=to_julia(stmt)))
    end
end

"""
Collect variable declarations from nested analog blocks.

Verilog-A allows variable declarations inside named blocks (e.g., `begin : evaluateblock`).
These declarations are not at the module's top level, so we need to walk the AST to find them.
This ensures all local variables are initialized in local_var_init before they're used.
"""
function collect_nested_var_decls!(var_types::Dict{Symbol, Union{Type{Int}, Type{Float64}, Type{String}}},
                                    var_inits::Dict{Symbol, Any}, stmt; depth=0)
    form = formof(stmt)

    if form == IntRealDeclaration
        # Found a variable declaration - add to var_types
        T = kw_to_T(stmt.kw.kw)
        for ident in stmt.idents
            vardecl = ident.item
            name = Symbol(assemble_id_string(vardecl.id))
            # Only add if not already present (top-level declarations take precedence)
            if !haskey(var_types, name)
                var_types[name] = T
                if vardecl.init !== nothing
                    var_inits[name] = to_julia_defaults(vardecl.init)
                end
            end
        end
    elseif form == AnalogSeqBlock
        # Named or unnamed sequential block - check for variable declarations
        # For named blocks (begin : blockname), declarations are in decl.decls
        if hasproperty(stmt, :decl) && stmt.decl !== nothing
            block_decl = stmt.decl
            if hasproperty(block_decl, :decls)
                for decl_item in block_decl.decls
                    # Each decl_item is an AnalogDeclarationItem with an .item field
                    inner_decl = hasproperty(decl_item, :item) ? decl_item.item : decl_item
                    if formof(inner_decl) == IntRealDeclaration
                        # Found a variable declaration - add to var_types
                        T = kw_to_T(inner_decl.kw.kw)
                        for ident in inner_decl.idents
                            vardecl = ident.item
                            name = Symbol(assemble_id_string(vardecl.id))
                            # Only add if not already present (top-level declarations take precedence)
                            if !haskey(var_types, name)
                                var_types[name] = T
                                if vardecl.init !== nothing
                                    var_inits[name] = to_julia_defaults(vardecl.init)
                                end
                            end
                        end
                    end
                end
            end
        end
        # Also recurse into statements (they may contain nested blocks)
        if hasproperty(stmt, :stmts)
            for s in stmt.stmts
                # s might be a node wrapper or the actual statement
                inner = hasproperty(s, :item) ? s.item : s
                collect_nested_var_decls!(var_types, var_inits, inner; depth=depth+1)
            end
        end
    elseif form == AnalogStatement
        collect_nested_var_decls!(var_types, var_inits, stmt.stmt; depth=depth+1)
    elseif form == AnalogIf
        # Single if statement - recurse into body
        if hasproperty(stmt, :stmt)
            collect_nested_var_decls!(var_types, var_inits, stmt.stmt; depth=depth+1)
        end
    elseif form == AnalogConditionalBlock
        # Recurse into if/else branches
        # AnalogConditionalBlock has aif (the if) and elsecases (list of ElseCase)
        aif = stmt.aif
        if hasproperty(aif, :stmt)
            collect_nested_var_decls!(var_types, var_inits, aif.stmt; depth=depth+1)
        end
        if hasproperty(stmt, :elsecases)
            for elsecase in stmt.elsecases
                if hasproperty(elsecase, :stmt)
                    collect_nested_var_decls!(var_types, var_inits, elsecase.stmt; depth=depth+1)
                end
            end
        end
    elseif form == AnalogFor || form == AnalogWhile || form == AnalogRepeat
        # Recurse into loop bodies - all these use .stmt for the body
        if hasproperty(stmt, :stmt)
            collect_nested_var_decls!(var_types, var_inits, stmt.stmt; depth=depth+1)
        end
    elseif form == CaseStatement
        # Recurse into case items
        if hasproperty(stmt, :items)
            for item in stmt.items
                inner = hasproperty(item, :item) ? item.item : item
                if hasproperty(inner, :stmt)
                    collect_nested_var_decls!(var_types, var_inits, inner.stmt; depth=depth+1)
                end
            end
        end
    end
    # Other statement types don't contain variable declarations
end

"""
Detect short circuit patterns from VA conditionals.

Scans the VA AST for patterns like:
    if (rs==0) V(a_int, a) <+ 0;

Returns a Dict mapping internal_node => (external_node, condition_expr) for each short circuit.
These can be used for node aliasing to reduce system size.
"""
function detect_short_circuits(analog_block, to_julia::MNAScope, internal_nodes::Vector{Symbol})
    internal_set = Set(internal_nodes)
    short_circuits = Dict{Symbol, NamedTuple{(:external, :condition), Tuple{Symbol, Any}}}()

    # Helper to check if expression is constant zero
    function is_zero_expr(expr)
        # Handle different node types
        try
            form = formof(expr)
            if form == Literal
                # Try to get the value - it might be in .v or we might need to parse String(expr)
                if hasproperty(expr, :v)
                    return expr.v == 0
                else
                    # Fallback: try to parse the string representation
                    return String(expr) == "0"
                end
            elseif form == FloatLiteral
                return parse(Float64, String(expr)) == 0.0
            end
        catch
            # Not a recognized form, continue checking
        end
        # Check if it's wrapped in something
        if hasproperty(expr, :item)
            return is_zero_expr(expr.item)
        end
        return false
    end

    # Scan a conditional block for V(internal, external) <+ 0 patterns
    function scan_conditional(aif_stmt, condition_expr)
        # Recursively scan statements for contribution statements
        function scan_stmts(stmts_or_stmt)
            stmts = if formof(stmts_or_stmt) == AnalogSeqBlock
                stmts_or_stmt.stmts
            else
                [stmts_or_stmt]
            end

            for stmt in stmts
                unwrapped = formof(stmt) == AnalogStatement ? stmt.stmt : stmt
                form = formof(unwrapped)

                if form == ContributionStatement
                    cs = unwrapped
                    bpfc = cs.lvalue
                    kind_sym = Symbol(bpfc.id)

                    if kind_sym == :V && is_zero_expr(cs.assign_expr)
                        refs = map(x -> Symbol(assemble_id_string(x.item)), bpfc.references)
                        if length(refs) == 2
                            p, n = refs[1], refs[2]
                            # Check if one is internal and one is external (port)
                            if p in internal_set && !(n in internal_set)
                                short_circuits[p] = (external=n, condition=condition_expr)
                            elseif n in internal_set && !(p in internal_set)
                                short_circuits[n] = (external=p, condition=condition_expr)
                            end
                        end
                    end
                elseif form == AnalogSeqBlock
                    # Recurse into nested sequence blocks
                    scan_stmts(unwrapped)
                end
            end
        end

        scan_stmts(aif_stmt.stmt)
    end

    # Walk the analog block looking for conditionals
    function walk(stmt)
        form = formof(stmt)
        if form == AnalogSeqBlock
            for s in stmt.stmts
                walk(s)
            end
        elseif form == AnalogStatement
            walk(stmt.stmt)
        elseif form == AnalogConditionalBlock
            # Scan the if-branch for short circuits
            aif = stmt.aif
            condition_expr = to_julia(aif.condition)
            scan_conditional(aif, condition_expr)
            # Also check else-if branches
            for case in stmt.elsecases
                if formof(case.stmt) == AnalogIf
                    walk(case.stmt)
                end
            end
        end
    end

    walk(analog_block)
    return short_circuits
end

"""
Translate a contribution statement for MNA.
"""
function mna_translate_contribution(to_julia::MNAScope, cs::VANode{ContributionStatement})
    bpfc = cs.lvalue
    kind_sym = Symbol(bpfc.id)
    kind = kind_sym == :I ? :current : kind_sym == :V ? :voltage : :unknown

    refs = map(bpfc.references) do ref
        Symbol(assemble_id_string(ref.item))
    end

    if length(refs) == 1
        node = refs[1]
        # Check if this is a named branch (e.g., V(br) or I(br) where br is a branch)
        if haskey(to_julia.named_branches, node)
            branch_nodes = to_julia.named_branches[node]
            pos_node, neg_node = branch_nodes.first, branch_nodes.second
            push!(to_julia.used_branches, pos_node => neg_node)
            # Return with branch_name to indicate this is a named branch contribution
            return (kind=kind, p=pos_node, n=neg_node, expr=to_julia(cs.assign_expr),
                    branch_name=node, is_branch=true)
        else
            push!(to_julia.used_branches, node => Symbol("0"))
            return (kind=kind, p=node, n=Symbol("0"), expr=to_julia(cs.assign_expr),
                    branch_name=nothing, is_branch=false)
        end
    elseif length(refs) == 2
        (id1, id2) = refs
        push!(to_julia.used_branches, id1 => id2)
        return (kind=kind, p=id1, n=id2, expr=to_julia(cs.assign_expr),
                branch_name=nothing, is_branch=false)
    end

    return (kind=:unknown, expr=:(error("Invalid contribution")),
            branch_name=nothing, is_branch=false)
end

"""
Generate stamp! method for n-terminal device (potentially with internal nodes).

For n-terminal devices with internal nodes, we use a vector-valued dual approach:
1. Allocate internal nodes using alloc_internal_node! (done once per context)
   - If a short circuit is detected (V(internal, external) <+ 0), alias instead
2. Create duals with partials for each node voltage (terminals + internal)
3. Evaluate the contribution expression
4. Extract ∂I/∂V_k for each node k and stamp into G matrix

# Arguments
- `symname`: Module/device name symbol
- `ps`: Port/terminal symbols (e.g., [:p, :n])
- `port_args`: Port argument symbols for the stamp method
- `internal_nodes`: Internal node symbols declared in the module
- `params_to_locals`: Parameter extraction expressions
- `local_var_decls`: Local variable declarations
- `function_defs`: VA function definitions
- `contributions`: Branch contribution tuples
- `to_julia`: MNAScope for code translation
- `short_circuits`: Dict mapping internal_node => (external, condition) for node aliasing
- `taint`: CodegenTaint for static/dynamic classification (optional)
- `cache_name`: Name of the cache struct (optional)
- `has_cache`: Whether to generate cache-aware stamp! (optional)
"""
function generate_mna_stamp_method_nterm(symname, ps, port_args, internal_nodes, params_to_locals,
                                          local_var_decls, function_defs, contributions,
                                          to_julia, short_circuits=Dict{Symbol, NamedTuple}(),
                                          taint=nothing, cache_name=nothing, has_cache=false)
    n_ports = length(port_args)
    n_internal = length(internal_nodes)
    n_all_nodes = n_ports + n_internal

    # Create unique node parameter names (prefixed to avoid conflict with voltage vars)
    # Terminal nodes come from function arguments
    node_params = [Symbol("_node_", p) for p in port_args]
    # Internal node indices will be allocated at runtime
    internal_node_params = [Symbol("_node_", n) for n in internal_nodes]
    # All node symbols for dual creation (terminals + internal)
    all_node_syms = [port_args; internal_nodes]
    all_node_params = [node_params; internal_node_params]

    # Split into two blocks:
    # 1. local_var_init: Variable initialization (before internal_node_alloc)
    #    - Uses Float64 for scalar initialization to avoid Dual issues with short-circuit conditions
    #    - Cached variables are excluded (loaded from cache instead)
    # 2. contrib_eval: Dynamic contribution computation expressions (after dual_creation)
    #    - May update variables with Dual-compatible values
    #    - Static assignments are excluded when has_cache (computed in init_device!)
    local_var_init = Expr(:block)
    contrib_eval = Expr(:block)

    # Cache loading block (only used when has_cache is true)
    cache_load = Expr(:block)

    # For n-terminal devices:
    # 1. Don't use `local` - variables need to be visible in outer scope for stamp_code
    # 2. Initialize with Float64 first (for short-circuit condition evaluation)
    # 3. Integer variables stay as Int (for control flow - booleans, counters)
    # 4. String variables stay as String (for parameter comparisons)
    #
    # When has_cache is true:
    # - Static variables are loaded from cache (computed in init_device!)
    # - Only dynamic variables are initialized here
    first_port = port_args[1]

    # Build set of cached static var names for quick lookup
    cached_static_vars = has_cache && taint !== nothing ? taint.used_static_vars : Set{Symbol}()

    for decl in local_var_decls
        # Convert `local name::T = init_expr` appropriately
        if decl.head == :local
            inner = decl.args[1]
            if inner isa Expr && inner.head == :(=)
                lhs = inner.args[1]
                rhs = inner.args[2]
                if lhs isa Expr && lhs.head == :(::)
                    name = lhs.args[1]
                    var_type = lhs.args[2]  # Type annotation (Int, Float64, or String)

                    # Cached variables: load from cache, skip initialization
                    if has_cache && name in cached_static_vars
                        push!(cache_load.args, :($name = _cache_.$name))
                        continue
                    end

                    # Non-cached variables: initialize normally
                    is_integer_type = var_type == :Int || var_type == Int
                    is_string_type = var_type == :String || var_type == String

                    if is_integer_type
                        push!(local_var_init.args, :($name = $rhs))
                    elseif is_string_type
                        push!(local_var_init.args, :($name = $rhs))
                    elseif rhs isa Expr && rhs.head == :call && rhs.args[1] == :zero
                        push!(local_var_init.args, :($name = 0.0))
                    else
                        push!(local_var_init.args, :($name = Float64($rhs)))
                    end
                else
                    # If no type annotation, still strip `local`
                    if lhs isa Symbol && has_cache && lhs in cached_static_vars
                        push!(cache_load.args, :($lhs = _cache_.$lhs))
                    else
                        push!(local_var_init.args, inner)
                    end
                end
            elseif inner isa Expr && inner.head == :(::)
                # Just type annotation, no initialization
                name = inner.args[1]
                var_type = inner.args[2]

                # Cached variables: load from cache, skip initialization
                if has_cache && name in cached_static_vars
                    push!(cache_load.args, :($name = _cache_.$name))
                    continue
                end

                # Non-cached variables: initialize with default
                is_integer_type = var_type == :Int || var_type == Int
                is_string_type = var_type == :String || var_type == String
                if is_integer_type
                    push!(local_var_init.args, :($name = 0))
                elseif is_string_type
                    push!(local_var_init.args, :($name = ""))
                else
                    push!(local_var_init.args, :($name = 0.0))
                end
            else
                # Plain assignment inside local
                if inner isa Symbol && has_cache && inner in cached_static_vars
                    push!(cache_load.args, :($inner = _cache_.$inner))
                else
                    push!(local_var_init.args, inner)
                end
            end
        else
            push!(local_var_init.args, decl)
        end
    end

    # Collect current contributions by branch, and voltage contributions for named branches
    branch_contribs = Dict{Tuple{Symbol,Symbol}, Vector{Any}}()
    voltage_branch_contribs = Dict{Symbol, NamedTuple}()  # branch_name -> (p, n, exprs)
    # Two-node voltage contributions V(p,n) <+ expr (not named branches) - needs branch current
    twonode_voltage_contribs = Dict{Tuple{Symbol,Symbol}, Vector{Any}}()  # (p,n) -> exprs

    for c in contributions
        if c.kind == :current
            branch = (c.p, c.n)
            if !haskey(branch_contribs, branch)
                branch_contribs[branch] = Any[]
            end
            push!(branch_contribs[branch], c.expr)
        elseif c.kind == :voltage && hasproperty(c, :is_branch) && c.is_branch
            # Voltage contribution to a named branch (e.g., V(br) <+ expr for inductor)
            branch_name = c.branch_name
            if !haskey(voltage_branch_contribs, branch_name)
                voltage_branch_contribs[branch_name] = (p=c.p, n=c.n, exprs=Any[])
            end
            push!(voltage_branch_contribs[branch_name].exprs, c.expr)
        elseif c.kind == :voltage && (!hasproperty(c, :is_branch) || !c.is_branch)
            # Two-node voltage contribution V(p,n) <+ expr (not a named branch)
            # Collect these to be stamped with proper branch current variables later
            # This is essential for short circuits (V(p,n) <+ 0) to carry DC current
            branch = (c.p, c.n)
            if !haskey(twonode_voltage_contribs, branch)
                twonode_voltage_contribs[branch] = Any[]
            end
            push!(twonode_voltage_contribs[branch], c.expr)
        elseif c.kind == :conditional
            # Skip static conditionals when has_cache (computed in init_device!)
            if has_cache && taint !== nothing && c.expr in taint.static_stmts
                continue
            end
            push!(contrib_eval.args, c.expr)
        elseif c.kind == :assignment
            # Skip static assignments when has_cache (computed in init_device!)
            if has_cache && taint !== nothing && c.expr in taint.static_stmts
                continue
            end
            # Regular dynamic assignment (e.g., cdrain = R*V(g,s)**2)
            push!(contrib_eval.args, c.expr)
        end
    end

    if get(ENV, "CEDARSIM_DEBUG_STAMP", "") == "1"
        println("DEBUG contrib_eval for $symname: $(length(contrib_eval.args)) statements")
    end

    # Allocate current variables for named branches with voltage contributions
    # OPTIMIZATION: Use component-based API to avoid Symbol interpolation at call site
    # For DirectStampContext, both names are ignored (counter-based access)
    # For MNAContext, the full name is built from components
    branch_current_alloc = Expr(:block)
    branch_current_vars = Dict{Symbol, Symbol}()  # branch_name -> current_var_name
    for (branch_name, _) in voltage_branch_contribs
        I_var = Symbol("_I_branch_", branch_name, "_idx")
        # Base name is device_type + _I_ + branch_name (compile-time constant)
        alloc_base_name = QuoteNode(Symbol(symname, "_I_", branch_name))
        push!(branch_current_alloc.args,
            :($I_var = CedarSim.MNA.alloc_current!(ctx, $alloc_base_name, _mna_instance_)))
        branch_current_vars[branch_name] = I_var
    end

    # Allocate current variables for two-node voltage contributions (e.g., V(a,b) <+ 0)
    # These need branch currents to carry DC current through short circuits
    # OPTIMIZATION: Use component-based API to avoid Symbol interpolation at call site
    twonode_voltage_vars = Dict{Tuple{Symbol,Symbol}, Symbol}()  # (p,n) -> current_var_name
    for ((p_sym, n_sym), _) in twonode_voltage_contribs
        I_var = Symbol("_I_V_", p_sym, "_", n_sym, "_idx")
        # Base name is device_type + _I_V_ + p_sym + _ + n_sym (compile-time constant)
        alloc_base_name = QuoteNode(Symbol(symname, "_I_V_", p_sym, "_", n_sym))
        push!(branch_current_alloc.args,
            :($I_var = CedarSim.MNA.alloc_current!(ctx, $alloc_base_name, _mna_instance_)))
        twonode_voltage_vars[(p_sym, n_sym)] = I_var
    end

    # Generate voltage-dependent charge detection block (runs with plain Float64 values)
    # Detection must run BEFORE dual_creation to avoid capturing JacobianTag duals
    # Results are cached in ctx.charge_is_vdep so stamp_code only checks the cache
    detection_block = Expr(:block)

    # Generate stamping code for each unique branch - UNROLL loops at codegen time
    # Now handles both terminal nodes and internal nodes
    stamp_code = Expr(:block)
    for ((p_sym, n_sym), exprs) in branch_contribs
        # Look up node indices in combined list (terminals + internal)
        # Handle ground node (Symbol("0")) specially - it maps to integer 0
        p_idx = p_sym == Symbol("0") ? nothing : findfirst(==(p_sym), all_node_syms)
        n_idx = n_sym == Symbol("0") ? nothing : findfirst(==(n_sym), all_node_syms)
        # Ground node uses literal 0; other nodes use their parameter symbol
        p_node = p_idx === nothing ? 0 : all_node_params[p_idx]
        n_node = n_idx === nothing ? 0 : all_node_params[n_idx]

        # Sum all contributions to this branch
        sum_expr = length(exprs) == 1 ? exprs[1] : Expr(:call, :+, exprs...)

        # Generate unrolled stamping code
        # Result structure depends on whether va_ddt() was called:
        #
        # 1. Pure resistive (no ddt): Dual{JacobianTag}(I, dI/dV₁, ..., dI/dVₙ)
        # 2. Has ddt (pure or mixed): Dual{ContributionTag}(resist_dual, react_dual)
        #    where resist_dual = Dual{JacobianTag}(I, dI/dV...) and react_dual = Dual{JacobianTag}(q, dq/dV...)
        # 3. Scalar: constant value (e.g., Ids=0 in cutoff)
        #
        # Tag ordering (JacobianTag ≺ ContributionTag) guarantees ContributionTag is always
        # the outer wrapper when present. No need to check for reversed nesting.
        #
        # IMPORTANT: has_reactive is set based on TYPE, not value. This ensures
        # consistent COO structure for precompilation regardless of operating point.
        # Generate charge variable base name for this branch (used if voltage-dependent)
        # OPTIMIZATION: Use component-based API to avoid Symbol interpolation at call site
        # For DirectStampContext, both names are ignored (counter-based access)
        # For MNAContext, the full name is built from components
        charge_base_name = QuoteNode(Symbol(symname, "_Q_", p_sym, "_", n_sym))

        # NOTE: Detection now happens inline during stamp evaluation (not in detection_block)
        # by comparing capacitance values across multiple runs with different random operating points.
        # See detect_or_cached! in contrib.jl for the new approach.

        branch_stamp = quote
            # Evaluate the branch current
            I_branch = $sum_expr

            if I_branch isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                # Has ddt: ContributionTag wraps JacobianTag duals
                I_resist = ForwardDiff.value(I_branch)       # Dual{JacobianTag} for I and ∂I/∂V
                I_react = ForwardDiff.partials(I_branch, 1)  # Dual{JacobianTag} for q and ∂q/∂V

                # Extract resistive values (JacobianTag partials are plain floats)
                I_val = ForwardDiff.value(I_resist)
                $([:($(Symbol("dI_dV", k)) = ForwardDiff.partials(I_resist, $k)) for k in 1:n_all_nodes]...)

                # Extract charge value and capacitances
                q_val = ForwardDiff.value(I_react)
                $([:($(Symbol("dq_dV", k)) = ForwardDiff.partials(I_react, $k)) for k in 1:n_all_nodes]...)

                has_reactive = true

            elseif I_branch isa ForwardDiff.Dual
                # Pure resistive: just JacobianTag dual, no ContributionTag
                I_val = ForwardDiff.value(I_branch)
                $([:($(Symbol("dI_dV", k)) = ForwardDiff.partials(I_branch, $k)) for k in 1:n_all_nodes]...)
                q_val = 0.0
                $([:($(Symbol("dq_dV", k)) = 0.0) for k in 1:n_all_nodes]...)
                has_reactive = false

            else
                # Scalar result (constant contribution)
                I_val = Float64(I_branch)
                $([:($(Symbol("dI_dV", k)) = 0.0) for k in 1:n_all_nodes]...)
                q_val = 0.0
                $([:($(Symbol("dq_dV", k)) = 0.0) for k in 1:n_all_nodes]...)
                has_reactive = false
            end
        end

        # Stamp resistive Jacobians into G matrix
        # MNA sign convention: I(p,n) flows from p to n
        # G[p,k] = +dI/dVk (current leaving p)
        # G[n,k] = -dI/dVk (current entering n)
        for k in 1:n_all_nodes
            k_node = all_node_params[k]
            push!(branch_stamp.args, quote
                if $p_node != 0 && $k_node != 0
                    CedarSim.MNA.stamp_G!(ctx, $p_node, $k_node, $(Symbol("dI_dV", k)))
                end
                if $n_node != 0 && $k_node != 0
                    CedarSim.MNA.stamp_G!(ctx, $n_node, $k_node, -$(Symbol("dI_dV", k)))
                end
            end)
        end

        # Stamp reactive Jacobians (capacitances) into C matrix OR use charge formulation
        # Same sign convention as G matrix
        # Only stamp if device has reactive components (determined by TYPE, not value)
        # This ensures consistent COO structure for precompilation
        #
        # For voltage-dependent capacitors, we use the charge formulation to achieve
        # a constant mass matrix:
        # - Allocate a charge state variable q
        # - Stamp constraint: q = Q(V) as algebraic equation
        # - Stamp KCL coupling: dq/dt appears in node equations
        #
        # For linear capacitors, we use standard C matrix stamping (no extra variables).
        #
        # Detection works by comparing Q/V ratios across multiple runs with different
        # random operating points. If the ratio varies, the charge is voltage-dependent.
        #
        # Key insight: dq_dV from ForwardDiff doesn't capture intermediate value dependencies
        # (because intermediates are computed before JacobianTag duals exist), but the actual
        # Q value DOES change correctly when x changes. So we compare Q/V ratios.

        # Compute V_branch = V_p - V_n for detection
        V_p_expr = p_idx === nothing ? 0.0 : Symbol("V_", p_idx)
        V_n_expr = n_idx === nothing ? 0.0 : Symbol("V_", n_idx)
        v_branch_expr = if p_idx === nothing && n_idx === nothing
            0.0
        elseif p_idx === nothing
            :(-$V_n_expr)
        elseif n_idx === nothing
            V_p_expr
        else
            :($V_p_expr - $V_n_expr)
        end

        push!(branch_stamp.args, quote
            if has_reactive
                # Detect voltage-dependence by comparing Q/V ratio across runs
                # q_val is the charge value, V_branch is the branch voltage
                _V_branch = $v_branch_expr
                _is_voltage_dependent = CedarSim.MNA.detect_or_cached!(ctx, $charge_base_name, _mna_instance_, _V_branch, q_val)
                if _is_voltage_dependent
                    # Voltage-dependent charge: use charge formulation for constant mass matrix
                    # Allocate charge variable (or get existing one)
                    # NOTE: Charge state is stored as q_scaled = q * CHARGE_SCALE for conditioning
                    _q_idx = CedarSim.MNA.alloc_charge!(ctx, $charge_base_name, _mna_instance_, $p_node, $n_node)

                    # --- Mass matrix (constant entries with CHARGE_SCALE!) ---
                    # KCL coupling: I = dq/dt = (1/CHARGE_SCALE) * d(q_scaled)/dt
                    # The (1/CHARGE_SCALE) factor converts scaled charge derivative to current
                    if $p_node != 0
                        CedarSim.MNA.stamp_C!(ctx, $p_node, _q_idx, 1.0 / CedarSim.MNA.CHARGE_SCALE)
                    end
                    if $n_node != 0
                        CedarSim.MNA.stamp_C!(ctx, $n_node, _q_idx, -1.0 / CedarSim.MNA.CHARGE_SCALE)
                    end

                    # --- Constraint Jacobian (in G matrix, scaled) ---
                    # Scaled constraint: F_scaled = q_scaled - CHARGE_SCALE*Q(V) = 0
                    # ∂F_scaled/∂q_scaled = 1 (keeps diagonal well-conditioned)
                    CedarSim.MNA.stamp_G!(ctx, _q_idx, _q_idx, 1.0)

                    # ∂F_scaled/∂V_k = -CHARGE_SCALE * ∂Q/∂V_k for each node k
                    $([quote
                        if $(all_node_params[k]) != 0
                            CedarSim.MNA.stamp_G!(ctx, _q_idx, $(all_node_params[k]), -CedarSim.MNA.CHARGE_SCALE * $(Symbol("dq_dV", k)))
                        end
                    end for k in 1:n_all_nodes]...)

                    # --- Constraint RHS (Newton companion, scaled) ---
                    # For scaled constraint F_scaled = q_scaled - CHARGE_SCALE*Q(V):
                    # b = CHARGE_SCALE * (Q(V₀) - Σ(dQ/dV_k * V_k))
                    _b_constraint = q_val  # Q(V₀)
                    $([quote
                        _b_constraint -= $(Symbol("dq_dV", k)) * $(Symbol("V_", k))  # - dQ/dV_k * V_k
                    end for k in 1:n_all_nodes]...)
                    CedarSim.MNA.stamp_b!(ctx, _q_idx, CedarSim.MNA.CHARGE_SCALE * _b_constraint)
                else
                    # Linear capacitor: use standard C matrix stamping
                    $([quote
                        if $p_node != 0 && $(all_node_params[k]) != 0
                            CedarSim.MNA.stamp_C!(ctx, $p_node, $(all_node_params[k]), $(Symbol("dq_dV", k)))
                        end
                        if $n_node != 0 && $(all_node_params[k]) != 0
                            CedarSim.MNA.stamp_C!(ctx, $n_node, $(all_node_params[k]), -$(Symbol("dq_dV", k)))
                        end
                    end for k in 1:n_all_nodes]...)
                end
            end
        end)

        # Stamp RHS: Ieq = I_val - sum(dI/dVk * Vk)
        # MNA sign convention: b[p] -= Ieq, b[n] += Ieq
        # (matches stamp_contribution! in contrib.jl)
        ieq_terms = Any[:I_val]
        for k in 1:n_all_nodes
            push!(ieq_terms, :(- $(Symbol("dI_dV", k)) * $(Symbol("V_", k))))
        end
        ieq_expr = Expr(:call, :+, ieq_terms...)

        # RHS stamping using Newton companion model
        # Must match the 2-term convention in contrib.jl:
        #   stamp_b!(ctx, p, -b_companion)
        #   stamp_b!(ctx, n, +b_companion)
        # where b_companion = I_val - dI/dVp*Vp - dI/dVn*Vn = Ieq
        push!(branch_stamp.args, quote
            let Ieq = $ieq_expr
                if $p_node != 0
                    CedarSim.MNA.stamp_b!(ctx, $p_node, -Ieq)
                end
                if $n_node != 0
                    CedarSim.MNA.stamp_b!(ctx, $n_node, Ieq)
                end
            end
        end)

        push!(stamp_code.args, branch_stamp)
    end

    # Generate internal node allocation code (runs once per stamp! call)
    # This allocates matrix/vector entries for internal nodes
    # If a short circuit is detected, alias to external node instead of allocating
    # Internal node names need to be unique per instance
    # We use _mna_instance_ as a prefix to create names like "xu1_xm_PSP103VA_GP"
    #
    # OPTIMIZATION: Use component-based alloc_internal_node!(ctx, base_name, instance_name)
    # For DirectStampContext, this avoids Symbol interpolation entirely (counter-based access)
    # For MNAContext, the function builds the full name from components
    internal_node_alloc = Expr(:block)
    for (i, (int_sym, int_param)) in enumerate(zip(internal_nodes, internal_node_params))
        # Base name is the device type + internal node name (compile-time constant)
        base_name = QuoteNode(Symbol(symname, "_", int_sym))

        if haskey(short_circuits, int_sym)
            # This internal node can be aliased to an external node when condition is true
            sc = short_circuits[int_sym]
            ext_sym = sc.external
            condition = sc.condition

            # Find the external node's parameter symbol
            ext_idx = findfirst(==(ext_sym), port_args)
            if ext_idx !== nothing
                ext_param = node_params[ext_idx]
                # Conditional allocation: alias if short circuit, else allocate
                push!(internal_node_alloc.args,
                    :($int_param = if !(iszero($condition))
                        $ext_param  # Alias to external node
                    else
                        CedarSim.MNA.alloc_internal_node!(ctx, $base_name, _mna_instance_)
                    end))
            else
                # External node not found in ports, fall back to regular allocation
                push!(internal_node_alloc.args,
                    :($int_param = CedarSim.MNA.alloc_internal_node!(ctx, $base_name, _mna_instance_)))
            end
        else
            # Regular allocation (no short circuit detected)
            push!(internal_node_alloc.args,
                :($int_param = CedarSim.MNA.alloc_internal_node!(ctx, $base_name, _mna_instance_)))
        end
    end

    # Add GMIN (minimum conductance) to ground for internal nodes
    # This prevents floating nodes that can cause singular matrix issues
    # Especially important for noise-related internal nodes that have no DC path
    # GMIN value: 1e-12 S (1 pS) - standard SPICE minimum conductance
    gmin_stamp = Expr(:block)
    for int_param in internal_node_params
        push!(gmin_stamp.args, quote
            if $int_param != 0
                CedarSim.MNA.stamp_G!(ctx, $int_param, $int_param, 1e-12)
            end
        end)
    end

    # Generate voltage extraction for all nodes (terminals + internal)
    # V_1..V_n_ports are for terminal nodes
    # V_(n_ports+1)..V_n_all_nodes are for internal nodes
    voltage_extraction = Expr(:block)
    # Terminal nodes (from function arguments)
    # Ground nodes (index 0) always return 0.0, others index into the solution vector
    for i in 1:n_ports
        np = node_params[i]
        push!(voltage_extraction.args,
            :($(Symbol("V_", i)) = $np == 0 ? 0.0 : _mna_x_[$np]))
    end
    # Internal nodes (from alloc_internal_node!)
    # Note: Internal nodes can be aliased to terminals (including ground) via short-circuit detection
    for i in 1:n_internal
        idx = n_ports + i
        inp = internal_node_params[i]
        push!(voltage_extraction.args,
            :($(Symbol("V_", idx)) = $inp == 0 ? 0.0 : _mna_x_[$inp]))
    end

    # Generate Float64 assignment for all nodes (for detection phase)
    # This runs BEFORE dual_creation so detection lambdas capture plain floats
    float_node_assignment = Expr(:block)
    for i in 1:n_all_nodes
        node_sym = all_node_syms[i]
        push!(float_node_assignment.args,
            :($node_sym = $(Symbol("V_", i))))
    end

    # Generate dual creation for all nodes (terminals + internal)
    # Each node gets a dual with identity partials: ∂V_i/∂V_k = δ_ik
    #
    # We use single-layer JacobianTag duals for ddx() support.
    # Voltage-dependent capacitor detection runs BEFORE this block with plain Float64
    # values, so detection lambdas never capture JacobianTag duals.
    dual_creation = Expr(:block)
    for i in 1:n_all_nodes
        node_sym = all_node_syms[i]
        # Create JacobianTag dual with identity partials for ddx() support
        partials = Expr(:tuple, [k == i ? 1.0 : 0.0 for k in 1:n_all_nodes]...)
        push!(dual_creation.args,
            :($node_sym = Dual{CedarSim.MNA.JacobianTag}($(Symbol("V_", i)), $partials...)))
    end

    # Generate branch current extraction for named branches
    # ZeroVector returns 0.0 for any index, eliminating bounds checks
    # NOTE: We use begin/end instead of let to avoid local scope issues
    # (the variable must be accessible in contrib_eval and voltage_stamp_code)
    branch_current_extraction = Expr(:block)
    for (branch_name, I_var) in branch_current_vars
        I_sym = Symbol("_I_branch_", branch_name)
        push!(branch_current_extraction.args,
            :(begin
                _i_idx = CedarSim.MNA.resolve_index(ctx, $I_var)
                $I_sym = _mna_x_[_i_idx]
            end))
    end

    # Initialize branch current variables for named branches that don't have voltage contributions
    # These are typically noise branches (e.g., I(NOII) <+ white_noise(...)) that are probed
    # In DC/transient analysis, noise sources contribute 0 current
    for (branch_name, _) in to_julia.named_branches
        if !haskey(branch_current_vars, branch_name)
            I_sym = Symbol("_I_branch_", branch_name)
            push!(branch_current_extraction.args, :($I_sym = 0.0))
        end
    end

    # Generate voltage contribution stamping for named branches
    voltage_stamp_code = Expr(:block)
    for (branch_name, vc) in voltage_branch_contribs
        I_var = branch_current_vars[branch_name]
        p_sym, n_sym = vc.p, vc.n
        exprs = vc.exprs

        # Look up node indices
        p_idx = p_sym == Symbol("0") ? nothing : findfirst(==(p_sym), all_node_syms)
        n_idx = n_sym == Symbol("0") ? nothing : findfirst(==(n_sym), all_node_syms)
        p_node = p_idx === nothing ? 0 : all_node_params[p_idx]
        n_node = n_idx === nothing ? 0 : all_node_params[n_idx]

        # Sum all voltage contributions
        sum_expr = length(exprs) == 1 ? exprs[1] : Expr(:call, :+, exprs...)

        # Generate stamping code for voltage contribution
        # V(br) <+ expr means V_p - V_n = expr
        # With current variable I_br:
        # - KCL: G[p, I_br] = 1, G[n, I_br] = -1 (current flows from p to n)
        # - Voltage constraint: G[I_br, p] = 1, G[I_br, n] = -1, b[I_br] = expr
        #
        # For inductor: V = L*ddt(I) uses va_ddt which creates Dual{ContributionTag}
        # - value = resistive part
        # - partials(1) = reactive part (stamps into C[I_br, I_br])

        v_stamp = quote
            # Evaluate voltage contribution
            V_contrib = $sum_expr

            # Stamp KCL: current I flows from p to n
            if $p_node != 0
                CedarSim.MNA.stamp_G!(ctx, $p_node, $I_var, 1.0)
            end
            if $n_node != 0
                CedarSim.MNA.stamp_G!(ctx, $n_node, $I_var, -1.0)
            end

            # Voltage constraint: V_p - V_n = V_contrib
            CedarSim.MNA.stamp_G!(ctx, $I_var, $p_node, 1.0)
            CedarSim.MNA.stamp_G!(ctx, $I_var, $n_node, -1.0)

            # Handle reactive (ddt) contributions
            if V_contrib isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                # Contains ddt() terms
                V_resist = ForwardDiff.value(V_contrib)
                V_react = ForwardDiff.partials(V_contrib, 1)

                # V_resist is the resistive voltage part (e.g., R*I)
                # V_react is the reactive coefficient (e.g., L from L*ddt(I) = L*s*I)
                V_resist_val = V_resist isa ForwardDiff.Dual ? ForwardDiff.value(V_resist) : Float64(V_resist)
                V_react_val = V_react isa ForwardDiff.Dual ? ForwardDiff.value(V_react) : Float64(V_react)

                # Stamp RHS with resistive voltage
                CedarSim.MNA.stamp_b!(ctx, $I_var, V_resist_val)

                # Stamp C matrix for reactive part: V = L*dI/dt
                # Voltage equation: V_p - V_n - L*dI/dt = 0
                # This stamps -L into C[I_var, I_var]
                # Always stamp - we're inside ContributionTag branch so device has ddt()
                CedarSim.MNA.stamp_C!(ctx, $I_var, $I_var, -V_react_val)
            else
                # Pure resistive voltage
                V_val = V_contrib isa ForwardDiff.Dual ? ForwardDiff.value(V_contrib) : Float64(V_contrib)
                CedarSim.MNA.stamp_b!(ctx, $I_var, V_val)
            end
        end

        push!(voltage_stamp_code.args, v_stamp)
    end

    # Generate stamping code for two-node voltage contributions V(p,n) <+ expr
    # These require branch current variables to carry DC current (especially for short circuits)
    twonode_voltage_stamp_code = Expr(:block)
    for ((p_sym, n_sym), exprs) in twonode_voltage_contribs
        I_var = twonode_voltage_vars[(p_sym, n_sym)]

        # Look up node indices
        p_idx = p_sym == Symbol("0") ? nothing : findfirst(==(p_sym), all_node_syms)
        n_idx = n_sym == Symbol("0") ? nothing : findfirst(==(n_sym), all_node_syms)
        p_node = p_idx === nothing ? 0 : all_node_params[p_idx]
        n_node = n_idx === nothing ? 0 : all_node_params[n_idx]

        # Sum all voltage contributions
        sum_expr = length(exprs) == 1 ? exprs[1] : Expr(:call, :+, exprs...)

        # Generate stamping code for two-node voltage contribution
        # V(p,n) <+ expr means V_p - V_n = expr
        # With current variable I:
        # - KCL at p: current I flows out of p → G[p, I] = 1
        # - KCL at n: current I flows into n → G[n, I] = -1
        # - Voltage constraint: V_p - V_n = expr → G[I, p] = 1, G[I, n] = -1, b[I] = expr
        # Skip if nodes are aliased (short circuit optimization)
        v_stamp = quote
            # Skip if nodes are aliased (p and n point to same index)
            if $p_node != $n_node
                # Evaluate voltage contribution
                V_contrib = $sum_expr

                # Extract scalar value from dual if needed
                V_val = V_contrib isa ForwardDiff.Dual ? ForwardDiff.value(V_contrib) : Float64(V_contrib)

                # Stamp KCL: current I flows from p to n
                if $p_node != 0
                    CedarSim.MNA.stamp_G!(ctx, $p_node, $I_var, 1.0)
                end
                if $n_node != 0
                    CedarSim.MNA.stamp_G!(ctx, $n_node, $I_var, -1.0)
                end

                # Voltage constraint: V_p - V_n = V_val
                if $p_node != 0
                    CedarSim.MNA.stamp_G!(ctx, $I_var, $p_node, 1.0)
                end
                if $n_node != 0
                    CedarSim.MNA.stamp_G!(ctx, $I_var, $n_node, -1.0)
                end
                CedarSim.MNA.stamp_b!(ctx, $I_var, V_val)
            end
        end

        push!(twonode_voltage_stamp_code.args, v_stamp)
    end

    # Build the stamp method
    # Terminal nodes come from function parameters; internal nodes are allocated dynamically
    # NOTE: Using _mna_*_ prefixes to avoid conflicts with VA parameter/variable names
    # (e.g., PSP103 has 'x', some models have 't', 'mode' is a common parameter name)
    # NOTE: ctx accepts AnyMNAContext (MNAContext or DirectStampContext) for zero-allocation mode
    # NOTE: @noinline prevents LLVM SROA from blowing up when this gets compiled
    # into a circuit function. Without it, large VA models (782-field PSP103VA) cause OOM.
    #
    # When has_cache is true, stamp! accepts an optional _cache_ parameter.
    # When _cache_ is provided and initialized, static computations are loaded from it.
    # When _cache_ is nothing, static computations are done inline (backward compat).

    # Common function body (after variable initialization)
    common_body = quote
        # Allocate internal nodes (idempotent - returns existing index if already allocated)
        $internal_node_alloc

        # Add GMIN to ground for internal nodes (prevents floating nodes)
        $gmin_stamp

        # Allocate current variables for named branches with voltage contributions
        $branch_current_alloc

        # Get operating point voltages (Float64) - used for RHS linearization
        $voltage_extraction

        # Get operating point currents for named branches
        $branch_current_extraction

        # Assign Float64 voltages to node symbols (for detection phase)
        # This runs BEFORE dual_creation so detection lambdas capture plain floats
        $float_node_assignment

        # Run voltage-dependent charge detection (with plain Float64 values)
        # Results are cached in ctx.charge_is_vdep for later use in stamp_code
        $detection_block

        # Reset detection counter for stamp_code phase (DirectStampContext uses counter-based access)
        CedarSim.MNA.reset_detection_counter!(ctx)

        # Create duals with partials for each node voltage (terminals + internal)
        # dual[i] = Dual(V_i, (k==1 ? 1 : 0), (k==2 ? 1 : 0), ...)
        # This overwrites node symbols (p, n, etc.) with JacobianTag duals
        $dual_creation

        # Evaluate contribution expressions with duals
        $contrib_eval

        # Stamp current contributions
        $stamp_code

        # Stamp voltage contributions for named branches (e.g., inductor V = L*dI/dt)
        $voltage_stamp_code

        # Stamp two-node voltage contributions (e.g., V(a,b) <+ 0 for short circuit)
        $twonode_voltage_stamp_code

        return nothing
    end

    # Generate stamp! with mandatory cache parameter
    # Cache type is the generated cache struct, or Nothing for models without cache
    cache_type = has_cache ? cache_name : :Nothing

    if has_cache
        quote
            # Main stamp! with mandatory cache - this is the small, optimized version
            # Static code has been moved to init_device!(), only cache loading remains
            Base.@noinline function CedarSim.MNA.stamp!(dev::$symname, ctx::CedarSim.MNA.AnyMNAContext,
                                         $([:($np::Int) for np in node_params]...),
                                         _cache_::$cache_type;
                                         _mna_t_::Real=0.0, _mna_mode_::Symbol=:dcop, _mna_x_::AbstractVector=CedarSim.MNA.ZERO_VECTOR,
                                         _mna_spec_::CedarSim.MNA.MNASpec=CedarSim.MNA.MNASpec(),
                                         _mna_instance_::Symbol=Symbol(""))
                # Convert empty vectors to ZERO_VECTOR for safe indexing
                _mna_x_ = isempty(_mna_x_) ? CedarSim.MNA.ZERO_VECTOR : _mna_x_
                $(params_to_locals...)
                $(function_defs...)

                # Initialize local variables (non-cached ones only)
                $local_var_init

                # Load cached static values (cache is required and must be initialized)
                $cache_load

                $common_body
            end
        end
    else
        quote
            # stamp! with nothing cache for models without static vars
            Base.@noinline function CedarSim.MNA.stamp!(dev::$symname, ctx::CedarSim.MNA.AnyMNAContext,
                                         $([:($np::Int) for np in node_params]...),
                                         _cache_::$cache_type;
                                         _mna_t_::Real=0.0, _mna_mode_::Symbol=:dcop, _mna_x_::AbstractVector=CedarSim.MNA.ZERO_VECTOR,
                                         _mna_spec_::CedarSim.MNA.MNASpec=CedarSim.MNA.MNASpec(),
                                         _mna_instance_::Symbol=Symbol(""))
                # Convert empty vectors to ZERO_VECTOR for safe indexing
                _mna_x_ = isempty(_mna_x_) ? CedarSim.MNA.ZERO_VECTOR : _mna_x_
                $(params_to_locals...)
                $(function_defs...)

                # Initialize local variables
                $local_var_init

                $common_body
            end
        end
    end
end

"""
    make_mna_module(va::VANode)

Generate an MNA-compatible module from a parsed Verilog-A file.
"""
function make_mna_module(va::VANode)
    vamod = va.stmts[end]
    s = Symbol(String(vamod.id), "_module")
    typename = Symbol(vamod.id)

    # Get the device definition (returns Expr(:toplevel, struct_def, constructor, stamp_method))
    device_expr = CedarSim.make_mna_device(vamod)

    Expr(:toplevel, :(baremodule $s
        using Base: AbstractVector, Real, Symbol, Float64, Int, String, isempty, max, zeros, zero, length
        using Base: hasproperty, getproperty, getfield, error, !==, iszero, abs
        import Base  # For getproperty override in aliasparam
        import ..CedarSim
        using ..CedarSim.VerilogAEnvironment
        using ..CedarSim.MNA: va_ddt, stamp_current_contribution!, MNAContext, MNASpec, alloc_internal_node!, alloc_current!, ZERO_VECTOR
        using ..CedarSim.MNA: AnyMNAContext, get_is_vdep, reset_detection_counter!  # For DirectStampContext support
        using ForwardDiff: Dual, value, partials
        import ForwardDiff
        export $typename
        $(device_expr.args...)
    end), :(using .$s))
end

struct VAFile
    file::String
end
Base.String(vaf::VAFile) = vaf.file
Base.abspath(file::VAFile) = VAFile(Base.abspath(file.file))
Base.isfile(file::VAFile) = Base.isfile(file.file)
Base.isabspath(file::VAFile) = Base.isabspath(file.file)
Base.findfirst(str::String, file::VAFile) = Base.findfirst(str, file.file)
Base.joinpath(str::String, file::VAFile) = VAFile(Base.joinpath(str, file.file))
Base.normpath(file::VAFile) = VAFile(Base.normpath(file.file))
export VAFile, @va_str

function parse_and_eval_vafile(mod::Module, file::VAFile)
    va = VerilogAParser.parsefile(file.file)
    if va.ps.errored
        cedarthrow(LoadError(file.file, 0, VAParseError(va)))
    else
        Core.eval(mod, make_mna_module(va))
    end
    return va.ps.srcfiles
end

function Base.include(mod::Module, file::VAFile)
    parse_and_eval_vafile(mod, file)
    return nothing
end

macro va_str(str)
    va = VerilogAParser.parse(IOBuffer(str))
    if va.ps.errored
        cedarthrow(LoadError("va_str", 0, VAParseError(va)))
    else
        # Use runtime eval to handle module definitions which must be at top level.
        # QuoteNode prevents any hygiene transformations on the AST.
        expr = make_mna_module(va)
        :(Core.eval($__module__, $(QuoteNode(expr))))
    end
end

struct VAParseError
    va
end

Base.show(io::IO, vap::VAParseError) = VerilogAParser.VerilogACSTParser.visit_errors(vap.va; io)

#==============================================================================#
# VA Device Package Loading for Precompilation
#==============================================================================#

"""
    load_mna_va_module(into::Module, file::String)
    load_mna_va_module(file::String)

Load a Verilog-A file and generate MNA device module(s).

When `into` module is provided, evaluates the module directly into that module
and returns the created module. This is the preferred usage for device packages
as it enables precompilation.

When called without `into`, returns the expression for manual evaluation.

# Arguments
- `into`: Target module to define the device module in (for precompilation)
- `file`: Path to the Verilog-A file

# Alternative: VAFile + Base.include

The existing pattern also works and is equivalent:
```julia
using RelocatableFolders
const device_va = @path joinpath(@__DIR__, "device.va")
Base.include(@__MODULE__, VAFile(device_va))
```

This is what packages like BSIM4.jl use. The difference is that `load_mna_va_module`
returns the created module for convenience.

# Example (Device package usage - enables precompilation)
```julia
module BSIM4
    using CedarSim
    const bsim4_module = CedarSim.load_mna_va_module(@__MODULE__,
        joinpath(@__DIR__, "bsim4.va"))
    using .bsim4_module: bsim4
    export bsim4
end
```

# Example (manual eval)
```julia
expr = CedarSim.load_mna_va_module("bsim4.va")
eval(expr)
using .bsim4_module: bsim4
```

# Generated Module Structure

For a VA file containing `module bsim4(d, g, s, b); ... endmodule`, generates:
- A submodule named `bsim4_module`
- Exports the device type `bsim4`
- Device has `stamp!(dev::bsim4, ctx, d, g, s, b; ...)` method
"""
function load_mna_va_module end

# Version that evals into target module (preferred for precompilation)
function load_mna_va_module(into::Module, file::String)
    # Parse the VA file
    va = VerilogAParser.parsefile(file)
    if va.ps.errored
        throw(LoadError(file, 0, VAParseError(va)))
    end

    # Generate the module expression
    expr = make_mna_module(va)

    # expr is (:toplevel, module_def, using_stmt)
    # We need to eval both parts
    @assert expr.head == :toplevel
    module_def = expr.args[1]
    using_stmt = expr.args[2]

    # Eval the module definition
    Core.eval(into, module_def)

    # Also eval the using statement to bring the module into scope
    Core.eval(into, using_stmt)

    # Get the created module name from the module definition
    # module_def is :(baremodule modname ... end)
    mod_name = module_def.args[2]  # The module name symbol

    # Return the created module
    return getfield(into, mod_name)
end

# Version that returns expression for manual eval
function load_mna_va_module(file::String)
    # Parse the VA file
    va = VerilogAParser.parsefile(file)
    if va.ps.errored
        throw(LoadError(file, 0, VAParseError(va)))
    end

    # Return the expression for manual evaluation
    return make_mna_module(va)
end

"""
    load_mna_va_modules(into::Module, file::String)

Load all Verilog-A modules from a file.

Similar to `load_mna_va_module` but handles files with multiple modules,
returning a NamedTuple mapping module names to the created Julia modules.

# Example
```julia
module MyDevices
    using CedarSim
    const devices = CedarSim.load_mna_va_modules(@__MODULE__,
        joinpath(@__DIR__, "devices.va"))
    # devices.resistor_module, devices.capacitor_module, etc.
end
```
"""
function load_mna_va_modules(into::Module, file::String)
    # Parse the VA file
    va = VerilogAParser.parsefile(file)
    if va.ps.errored
        throw(LoadError(file, 0, VAParseError(va)))
    end

    # Collect all VerilogModule nodes
    result_modules = Dict{Symbol, Module}()

    for stmt in va.stmts
        if stmt isa VANode{VerilogModule}
            vamod = stmt
            typename = Symbol(vamod.id)
            mod_name = Symbol(String(vamod.id), "_module")

            # Generate device expression
            device_expr = make_mna_device(vamod)

            # Create module expression
            module_expr = :(baremodule $mod_name
                using Base: AbstractVector, Real, Symbol, Float64, Int, String, isempty, max, zeros, zero, length
                using Base: hasproperty, getproperty, getfield, error, !==, iszero, abs
                import Base
                import ..CedarSim
                using ..CedarSim.VerilogAEnvironment
                using ..CedarSim.MNA: va_ddt, stamp_current_contribution!, MNAContext, MNASpec, alloc_internal_node!, alloc_current!, ZERO_VECTOR
                using ForwardDiff: Dual, value, partials
                import ForwardDiff
                export $typename
                $(device_expr.args...)
            end)

            # Eval into target module
            Core.eval(into, module_expr)
            Core.eval(into, :(using .$mod_name))

            result_modules[mod_name] = getfield(into, mod_name)
        end
    end

    return NamedTuple(result_modules)
end
