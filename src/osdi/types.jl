#==============================================================================#
# OSDI C Type Mirrors and Constants
#
# Julia structs matching osdi_0_4.h layout for unsafe_load from shared libraries.
# These must match the C ABI exactly — field order, sizes, and alignment.
#==============================================================================#

# Parameter type flags
const PARA_TY_MASK   = UInt32(3)
const PARA_TY_REAL   = UInt32(0)
const PARA_TY_INT    = UInt32(1)
const PARA_TY_STR    = UInt32(2)
const PARA_KIND_MASK  = UInt32(3) << 30
const PARA_KIND_MODEL = UInt32(0) << 30
const PARA_KIND_INST  = UInt32(1) << 30
const PARA_KIND_OPVAR = UInt32(2) << 30

# Access flags
const ACCESS_FLAG_READ     = UInt32(0)
const ACCESS_FLAG_SET      = UInt32(1)
const ACCESS_FLAG_INSTANCE = UInt32(4)

# Jacobian entry flags
const JACOBIAN_ENTRY_RESIST_CONST = UInt32(1)
const JACOBIAN_ENTRY_REACT_CONST  = UInt32(2)
const JACOBIAN_ENTRY_RESIST       = UInt32(4)
const JACOBIAN_ENTRY_REACT        = UInt32(8)

# Eval flags (sim_info.flags)
const CALC_RESIST_RESIDUAL  = UInt32(1)
const CALC_REACT_RESIDUAL   = UInt32(2)
const CALC_RESIST_JACOBIAN  = UInt32(4)
const CALC_REACT_JACOBIAN   = UInt32(8)
const CALC_NOISE            = UInt32(16)
const CALC_OP               = UInt32(32)
const CALC_RESIST_LIM_RHS   = UInt32(64)
const CALC_REACT_LIM_RHS    = UInt32(128)
const ENABLE_LIM            = UInt32(256)
const INIT_LIM              = UInt32(512)
const ANALYSIS_NOISE        = UInt32(1024)
const ANALYSIS_DC           = UInt32(2048)
const ANALYSIS_AC           = UInt32(4096)
const ANALYSIS_TRAN         = UInt32(8192)
const ANALYSIS_IC           = UInt32(16384)
const ANALYSIS_STATIC       = UInt32(32768)
const ANALYSIS_NODESET      = UInt32(65536)

# Eval return flags
const EVAL_RET_FLAG_LIM    = UInt32(1)
const EVAL_RET_FLAG_FATAL  = UInt32(2)
const EVAL_RET_FLAG_FINISH = UInt32(4)
const EVAL_RET_FLAG_STOP   = UInt32(8)

# Init error codes
const INIT_ERR_OUT_OF_BOUNDS = UInt32(1)

#==============================================================================#
# C Struct Mirrors
#
# These must match osdi_0_4.h layout exactly. On 64-bit systems:
# - Pointers are 8 bytes
# - uint32_t is 4 bytes
# - double is 8 bytes
# - bool is 1 byte (but may have padding after it)
#==============================================================================#

struct CNodePair
    node_1::UInt32
    node_2::UInt32
end

struct CJacobianEntry
    nodes::CNodePair
    react_ptr_off::UInt32
    flags::UInt32
end

# OsdiNode has a trailing bool with padding — we need to match the C layout.
# On most ABIs, the struct is padded to pointer alignment (8 bytes) after is_flow.
struct CNode
    name::Ptr{UInt8}
    units::Ptr{UInt8}
    residual_units::Ptr{UInt8}
    resist_residual_off::UInt32
    react_residual_off::UInt32
    resist_limit_rhs_off::UInt32
    react_limit_rhs_off::UInt32
    is_flow::Bool
end

struct CParamOpvar
    name::Ptr{Ptr{UInt8}}      # char** (primary name + aliases)
    num_alias::UInt32
    description::Ptr{UInt8}
    units::Ptr{UInt8}
    flags::UInt32
    len::UInt32
end

struct CNoiseSource
    name::Ptr{UInt8}
    nodes::CNodePair
end

struct CNatureRef
    ref_type::UInt32
    index::UInt32
end

struct CSimParas
    names::Ptr{Ptr{UInt8}}
    vals::Ptr{Float64}
    names_str::Ptr{Ptr{UInt8}}
    vals_str::Ptr{Ptr{UInt8}}
end

struct CSimInfo
    paras::CSimParas
    abstime::Float64
    prev_solve::Ptr{Float64}
    prev_state::Ptr{Float64}
    next_state::Ptr{Float64}
    flags::UInt32
end

struct CInitError
    code::UInt32
    parameter_id::UInt32  # union payload
end

struct CInitInfo
    flags::UInt32
    num_errors::UInt32
    errors::Ptr{CInitError}
end

# The full OsdiDescriptor — must match field order in osdi_0_4.h exactly.
# This is a large struct with mixed pointers, uint32s, and function pointers.
struct CDescriptor
    name::Ptr{UInt8}

    num_nodes::UInt32
    num_terminals::UInt32
    nodes::Ptr{CNode}

    num_jacobian_entries::UInt32
    jacobian_entries::Ptr{CJacobianEntry}

    num_collapsible::UInt32
    collapsible::Ptr{CNodePair}
    collapsed_offset::UInt32

    noise_sources::Ptr{CNoiseSource}
    num_noise_src::UInt32

    num_params::UInt32
    num_instance_params::UInt32
    num_opvars::UInt32
    param_opvar::Ptr{CParamOpvar}

    node_mapping_offset::UInt32
    jacobian_ptr_resist_offset::UInt32

    num_states::UInt32
    state_idx_off::UInt32

    bound_step_offset::UInt32

    instance_size::UInt32
    model_size::UInt32

    fn_access::Ptr{Cvoid}
    fn_setup_model::Ptr{Cvoid}
    fn_setup_instance::Ptr{Cvoid}
    fn_eval::Ptr{Cvoid}
    fn_load_noise::Ptr{Cvoid}
    fn_load_residual_resist::Ptr{Cvoid}
    fn_load_residual_react::Ptr{Cvoid}
    fn_load_limit_rhs_resist::Ptr{Cvoid}
    fn_load_limit_rhs_react::Ptr{Cvoid}
    fn_load_spice_rhs_dc::Ptr{Cvoid}
    fn_load_spice_rhs_tran::Ptr{Cvoid}
    fn_load_jacobian_resist::Ptr{Cvoid}
    fn_load_jacobian_react::Ptr{Cvoid}
    fn_load_jacobian_tran::Ptr{Cvoid}
    fn_given_flag_model::Ptr{Cvoid}
    fn_given_flag_instance::Ptr{Cvoid}

    num_resistive_jacobian_entries::UInt32
    num_reactive_jacobian_entries::UInt32

    fn_write_jacobian_array_resist::Ptr{Cvoid}
    fn_write_jacobian_array_react::Ptr{Cvoid}

    num_inputs::UInt32
    inputs::Ptr{CNodePair}

    fn_load_jacobian_with_offset_resist::Ptr{Cvoid}
    fn_load_jacobian_with_offset_react::Ptr{Cvoid}

    unknown_nature::Ptr{CNatureRef}
    residual_nature::Ptr{CNatureRef}

    noise_source_type::Ptr{UInt32}
    fn_load_noise_params::Ptr{Cvoid}
end
