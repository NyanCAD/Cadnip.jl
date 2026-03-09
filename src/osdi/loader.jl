#==============================================================================#
# OSDI Shared Library Loader
#
# Opens .osdi files (compiled by OpenVAF), validates version, and parses
# OsdiDescriptor metadata into Julia-friendly types.
#==============================================================================#

import Libdl

"""
    OsdiParamInfo

Parsed parameter/opvar metadata from an OSDI descriptor.
"""
struct OsdiParamInfo
    name::String
    aliases::Vector{String}
    osdi_id::UInt32           # index into param_opvar array
    type::Type                # Float64, Int32, or String
    kind::Symbol              # :model, :instance, :opvar
    len::UInt32               # 0 = scalar, >0 = array length
end

"""
    OsdiNodeInfo

Parsed node metadata.
"""
struct OsdiNodeInfo
    name::String
    units::String
    resist_residual_off::UInt32
    react_residual_off::UInt32
    is_flow::Bool
end

"""
    OsdiDeviceType

Julia wrapper around one OsdiDescriptor from a loaded .osdi file.
Contains all metadata needed to create instances and stamp into MNA.
"""
struct OsdiDeviceType
    lib::Ptr{Nothing}              # dlopen handle (prevents unload)
    descriptor_ptr::Ptr{CDescriptor}
    name::String
    num_nodes::Int
    num_terminals::Int
    nodes::Vector{OsdiNodeInfo}
    params::Vector{OsdiParamInfo}
    param_by_name::Dict{String, OsdiParamInfo}
    jacobian_entries::Vector{CJacobianEntry}
    collapsible::Vector{CNodePair}  # node pairs that may be collapsed
    num_resistive_entries::Int
    num_reactive_entries::Int
    num_states::Int
    instance_size::Int
    model_size::Int
    node_mapping_offset::UInt32
    jacobian_ptr_resist_offset::UInt32
    state_idx_off::UInt32
    collapsed_offset::UInt32
    bound_step_offset::UInt32
    # Function pointers
    fn_access::Ptr{Cvoid}
    fn_setup_model::Ptr{Cvoid}
    fn_setup_instance::Ptr{Cvoid}
    fn_eval::Ptr{Cvoid}
    fn_load_residual_resist::Ptr{Cvoid}
    fn_load_residual_react::Ptr{Cvoid}
    fn_load_limit_rhs_resist::Ptr{Cvoid}
    fn_load_spice_rhs_dc::Ptr{Cvoid}
    fn_load_jacobian_resist::Ptr{Cvoid}
    fn_load_jacobian_react::Ptr{Cvoid}
    fn_write_jacobian_array_resist::Ptr{Cvoid}
    fn_write_jacobian_array_react::Ptr{Cvoid}
end

"""
    OsdiFile

Represents a loaded .osdi shared library with one or more device descriptors.
"""
struct OsdiFile
    path::String
    lib::Ptr{Nothing}
    devices::Vector{OsdiDeviceType}
end

function _read_cstring(ptr::Ptr{UInt8})
    ptr == C_NULL && return ""
    return unsafe_string(ptr)
end

function _parse_param_type(flags::UInt32)
    ty = flags & PARA_TY_MASK
    ty == PARA_TY_REAL && return Float64
    ty == PARA_TY_INT  && return Int32
    ty == PARA_TY_STR  && return String
    error("Unknown OSDI parameter type: $ty")
end

function _parse_param_kind(flags::UInt32)
    kind = flags & PARA_KIND_MASK
    kind == PARA_KIND_MODEL && return :model
    kind == PARA_KIND_INST  && return :instance
    kind == PARA_KIND_OPVAR && return :opvar
    error("Unknown OSDI parameter kind: $kind")
end

function _parse_params(desc::CDescriptor)
    # num_instance_params counts how many of num_params are instance params (they overlap)
    n = Int(desc.num_params) + Int(desc.num_opvars)
    params = Vector{OsdiParamInfo}(undef, n)
    param_by_name = Dict{String, OsdiParamInfo}()

    for i in 1:n
        cparam = unsafe_load(desc.param_opvar, i)
        # Read primary name and aliases
        num_names = Int(cparam.num_alias) + 1
        names = String[]
        for j in 1:num_names
            name_ptr = unsafe_load(cparam.name, j)
            push!(names, _read_cstring(name_ptr))
        end
        primary = names[1]
        aliases = names[2:end]

        info = OsdiParamInfo(
            primary, aliases,
            UInt32(i - 1),  # 0-based id
            _parse_param_type(cparam.flags),
            _parse_param_kind(cparam.flags),
            cparam.len
        )
        params[i] = info
        param_by_name[primary] = info
        for a in aliases
            param_by_name[a] = info
        end
    end
    return params, param_by_name
end

function _parse_nodes(desc::CDescriptor)
    n = Int(desc.num_nodes)
    nodes = Vector{OsdiNodeInfo}(undef, n)
    for i in 1:n
        cnode = unsafe_load(desc.nodes, i)
        nodes[i] = OsdiNodeInfo(
            _read_cstring(cnode.name),
            _read_cstring(cnode.units),
            cnode.resist_residual_off,
            cnode.react_residual_off,
            cnode.is_flow
        )
    end
    return nodes
end

function _parse_jacobian_entries(desc::CDescriptor)
    n = Int(desc.num_jacobian_entries)
    entries = Vector{CJacobianEntry}(undef, n)
    for i in 1:n
        entries[i] = unsafe_load(desc.jacobian_entries, i)
    end
    return entries
end

"""
    OsdiDeviceType(lib, desc_ptr)

Parse an OsdiDescriptor from a loaded shared library into a Julia-friendly type.
"""
function _parse_collapsible(desc::CDescriptor)
    n = Int(desc.num_collapsible)
    pairs = Vector{CNodePair}(undef, n)
    for i in 1:n
        pairs[i] = unsafe_load(desc.collapsible, i)
    end
    return pairs
end

function OsdiDeviceType(lib::Ptr{Nothing}, desc_ptr::Ptr{CDescriptor})
    desc = unsafe_load(desc_ptr)
    params, param_by_name = _parse_params(desc)
    nodes = _parse_nodes(desc)
    jacobian_entries = _parse_jacobian_entries(desc)
    collapsible = _parse_collapsible(desc)

    OsdiDeviceType(
        lib, desc_ptr,
        _read_cstring(desc.name),
        Int(desc.num_nodes),
        Int(desc.num_terminals),
        nodes, params, param_by_name,
        jacobian_entries,
        collapsible,
        Int(desc.num_resistive_jacobian_entries),
        Int(desc.num_reactive_jacobian_entries),
        Int(desc.num_states),
        Int(desc.instance_size),
        Int(desc.model_size),
        desc.node_mapping_offset,
        desc.jacobian_ptr_resist_offset,
        desc.state_idx_off,
        desc.collapsed_offset,
        desc.bound_step_offset,
        # Function pointers
        desc.fn_access,
        desc.fn_setup_model,
        desc.fn_setup_instance,
        desc.fn_eval,
        desc.fn_load_residual_resist,
        desc.fn_load_residual_react,
        desc.fn_load_limit_rhs_resist,
        desc.fn_load_spice_rhs_dc,
        desc.fn_load_jacobian_resist,
        desc.fn_load_jacobian_react,
        desc.fn_write_jacobian_array_resist,
        desc.fn_write_jacobian_array_react,
    )
end

"""
    osdi_load(path::String) -> OsdiFile

Load an .osdi shared library, validate its OSDI version, and parse all
device descriptors.

# Example
```julia
f = osdi_load("resistor.osdi")
f.devices[1].name  # "resistor"
```
"""
function osdi_load(path::String)
    lib = Libdl.dlopen(path)

    major_ptr = Libdl.dlsym(lib, :OSDI_VERSION_MAJOR; throw_error=true)
    minor_ptr = Libdl.dlsym(lib, :OSDI_VERSION_MINOR; throw_error=true)
    major = unsafe_load(Ptr{UInt32}(major_ptr))
    minor = unsafe_load(Ptr{UInt32}(minor_ptr))
    if major != 0 || minor != 4
        Libdl.dlclose(lib)
        error("Unsupported OSDI version $major.$minor (expected 0.4)")
    end

    num_ptr = Libdl.dlsym(lib, :OSDI_NUM_DESCRIPTORS; throw_error=true)
    num = unsafe_load(Ptr{UInt32}(num_ptr))

    desc_arr_ptr = Libdl.dlsym(lib, :OSDI_DESCRIPTORS; throw_error=true)
    # OSDI_DESCRIPTORS is an array of CDescriptor (not pointer to pointer)
    base_ptr = Ptr{CDescriptor}(desc_arr_ptr)

    devices = OsdiDeviceType[]
    for i in 1:Int(num)
        desc_ptr = base_ptr + (i - 1) * sizeof(CDescriptor)
        push!(devices, OsdiDeviceType(lib, desc_ptr))
    end

    OsdiFile(path, lib, devices)
end
