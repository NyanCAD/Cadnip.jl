# =============================================================================
# Verilog-A Model Definition Extraction
# =============================================================================
#
# This module provides functionality to extract model definitions and their
# parameters from Verilog-A source files. It supports case-insensitive lookup
# while preserving original case and parameter order.

using NyanVerilogAParser
using NyanVerilogAParser.VerilogACSTParser: VerilogModule, ParameterDeclaration, Node

const VANode = NyanVerilogAParser.VerilogACSTParser.Node

# =============================================================================
# Data Structures
# =============================================================================

"""
    ModelParameter

Represents a parameter in a Verilog-A model with its type and default value.

Fields:
- `name::String`: Parameter name with original case preserved
- `ptype::String`: Parameter type (e.g., "real", "integer")
- `default_value::Union{String, SNode}`: Default value - String (from VA model) or SNode (from SPICE override)
"""
struct ModelParameter
    name::String
    ptype::String
    default_value::Union{String, SNode}
end

"""
    ModelDefinition

Represents a Verilog-A model (module) with its parameters.

Fields:
- `name::String`: Model name with original case preserved
- `parameters::Vector{ModelParameter}`: Parameters in declaration order
- `param_lookup::Dict{Symbol, Int}`: Case-insensitive lookup (lowercase → index)
- `source_file::Union{String, Nothing}`: Source VA file basename (e.g., "sp_resistor.va")
"""
struct ModelDefinition
    name::String
    parameters::Vector{ModelParameter}
    param_lookup::Dict{Symbol, Int}
    source_file::Union{String, Nothing}
end

"""
    ModelDefinition(name::String, parameters::Vector{ModelParameter}, source_file::Union{String, Nothing}=nothing)

Constructor that automatically builds the case-insensitive parameter lookup.
"""
function ModelDefinition(name::String, parameters::Vector{ModelParameter}, source_file::Union{String, Nothing}=nothing)
    param_lookup = Dict{Symbol, Int}()
    for (i, param) in enumerate(parameters)
        param_lookup[Symbol(lowercase(param.name))] = i
    end
    ModelDefinition(name, parameters, param_lookup, source_file)
end

"""
    ModelDatabase

Collection of model definitions with case-insensitive lookup.

Fields:
- `models::Vector{ModelDefinition}`: All models in declaration order
- `model_lookup::Dict{Symbol, Int}`: Case-insensitive lookup (lowercase → index)
"""
struct ModelDatabase
    models::Vector{ModelDefinition}
    model_lookup::Dict{Symbol, Int}
end

"""
    ModelDatabase(models::Vector{ModelDefinition})

Constructor that automatically builds the case-insensitive model lookup.
"""
function ModelDatabase(models::Vector{ModelDefinition})
    model_lookup = Dict{Symbol, Int}()
    for (i, model) in enumerate(models)
        model_lookup[Symbol(lowercase(model.name))] = i
    end
    ModelDatabase(models, model_lookup)
end

# =============================================================================
# Model Extraction
# =============================================================================

"""
    extract_module_parameters(vm::VANode{VerilogModule}) -> Vector{ModelParameter}

Extract parameters from a Verilog-A module in declaration order.
"""
function extract_module_parameters(vm::VANode{VerilogModule})
    parameters = ModelParameter[]

    for child in vm.items
        item = child.item
        # Check if this is a ParameterDeclaration
        if isa(item, Node{ParameterDeclaration})
            # Extract parameter type (defaults to "real" if not specified)
            param_type = if item.ptype !== nothing
                String(item.ptype)
            else
                "real"
            end

            # Extract each parameter in this declaration
            for param in item.params
                param = param.item
                param_name = String(param.id)
                default_val = String(param.default_expr)
                push!(parameters, ModelParameter(param_name, param_type, default_val))
            end
        end
    end

    return parameters
end

"""
    extract_model_definitions(filepath::String) -> ModelDatabase

Parse a Verilog-A file and extract all model definitions with their parameters.

Returns a `ModelDatabase` with case-insensitive lookup while preserving original
case and parameter declaration order.

# Arguments
- `filepath::String`: Path to Verilog-A file

# Returns
- `ModelDatabase`: Extracted model definitions

# Example
```julia
db = extract_model_definitions("models.va")
model = get_model(db, "BasicVAResistor")  # Case-insensitive
println(model.parameters[1].name)  # Original case: "R"
```
"""
function extract_model_definitions(filepath::String)
    # Parse the Verilog-A file
    va = NyanVerilogAParser.parsefile(filepath)

    if va.ps.errored
        println(stderr, "Parse errors in Verilog-A file: $filepath")
        NyanVerilogAParser.VerilogACSTParser.visit_errors(va; io=stderr)
    end

    models = ModelDefinition[]

    # Store just the basename for use in `include directives
    source_basename = basename(filepath)

    # Walk through top-level statements looking for modules
    for stmt in va.stmts
        if isa(stmt, Node{VerilogModule})
            module_name = String(stmt.id)
            parameters = extract_module_parameters(stmt)

            # Create model definition with source file tracking
            model = ModelDefinition(module_name, parameters, source_basename)
            push!(models, model)
        end
    end

    return ModelDatabase(models)
end

"""
    merge_model_databases(dbs::Vector{ModelDatabase}) -> ModelDatabase

Merge multiple model databases into a single database.

If multiple definitions exist for the same model name (case-insensitive),
the last definition wins.

# Arguments
- `dbs::Vector{ModelDatabase}`: Vector of model databases to merge

# Returns
- `ModelDatabase`: Merged database
"""
function merge_model_databases(dbs::Vector{ModelDatabase})
    if isempty(dbs)
        return ModelDatabase(ModelDefinition[])
    end

    if length(dbs) == 1
        return dbs[1]
    end

    # Collect all models, using dict to handle duplicate names
    model_dict = Dict{Symbol, ModelDefinition}()
    model_order = Symbol[]  # Track order of first appearance

    for db in dbs
        for model in db.models
            key = Symbol(lowercase(model.name))
            if !haskey(model_dict, key)
                push!(model_order, key)
            end
            model_dict[key] = model
        end
    end

    # Rebuild models vector in order
    models = ModelDefinition[]
    for key in model_order
        push!(models, model_dict[key])
    end

    return ModelDatabase(models)
end

"""
    get_model(db::ModelDatabase, name::String) -> Union{ModelDefinition, Nothing}

Get a model definition by name (case-insensitive lookup).

# Arguments
- `db::ModelDatabase`: Model database
- `name::String`: Model name (case-insensitive)

# Returns
- `ModelDefinition`: Model definition if found
- `Nothing`: If model not found
"""
function get_model(db::ModelDatabase, name::String)
    key = Symbol(lowercase(name))
    idx = get(db.model_lookup, key, nothing)
    return idx === nothing ? nothing : db.models[idx]
end

"""
    get_param(model::ModelDefinition, name::String) -> Union{ModelParameter, Nothing}

Get a parameter from a model definition by name (case-insensitive lookup).

# Arguments
- `model::ModelDefinition`: Model definition
- `name::String`: Parameter name (case-insensitive)

# Returns
- `ModelParameter`: Parameter if found
- `Nothing`: If parameter not found
"""
function get_param(model::ModelDefinition, name::String)
    key = Symbol(lowercase(name))
    idx = get(model.param_lookup, key, nothing)
    return idx === nothing ? nothing : model.parameters[idx]
end

"""
    get_param_name(model::ModelDefinition, name::String) -> Union{String, Nothing}

Get the original-case parameter name from a model definition (case-insensitive lookup).

# Arguments
- `model::ModelDefinition`: Model definition
- `name::String`: Parameter name (case-insensitive)

# Returns
- `String`: Original-case parameter name if found
- `Nothing`: If parameter not found
"""
function get_param_name(model::ModelDefinition, name::String)
    param = get_param(model, name)
    return param === nothing ? nothing : param.name
end

"""
    add_model!(db::ModelDatabase, model::ModelDefinition, lookup_name::String)

Add a model to the database with a specific lookup name (case-insensitive).
Mutates the database in place.

# Arguments
- `db::ModelDatabase`: Database to modify
- `model::ModelDefinition`: Model to add
- `lookup_name::String`: Name to use for lookup (will be lowercased)
"""
function add_model!(db::ModelDatabase, model::ModelDefinition, lookup_name::String)
    push!(db.models, model)
    db.model_lookup[Symbol(lowercase(lookup_name))] = length(db.models)
    return db
end

# Export public API
export ModelParameter, ModelDefinition, ModelDatabase
export extract_model_definitions, merge_model_databases
export get_model, get_param, get_param_name, add_model!
