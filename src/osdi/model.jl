#==============================================================================#
# OSDI Model and Instance
#
# Memory management for OSDI model/instance blobs, parameter access,
# and limit state management for Newton iteration.
#==============================================================================#

"""
    OsdiLimitState

Double-buffered state management for \$limit() constructs.

During Newton iteration, eval reads from prev_state and writes to next_state.
After each eval call, the buffers are rotated (swap prev/next).

For Rosenbrock (no Newton loop), both point to the same buffer, making
\$limit() a no-op pass-through.
"""
mutable struct OsdiLimitState
    state_a::Vector{Float64}
    state_b::Vector{Float64}
    a_is_prev::Bool
    converged::Vector{Float64}   # snapshot for timestep rejection rollback
    iteration::Int               # counts evals within a Newton solve
    last_t::Float64              # detect new timestep in IDA path
end

function OsdiLimitState(num_states::Int)
    OsdiLimitState(
        zeros(Float64, num_states),
        zeros(Float64, num_states),
        true,
        zeros(Float64, num_states),
        0,
        NaN,
    )
end

"""Get current prev/next state pointers for eval."""
function get_state_ptrs(ls::OsdiLimitState)
    if ls.a_is_prev
        return (ls.state_a, ls.state_b)
    else
        return (ls.state_b, ls.state_a)
    end
end

"""Rotate state buffers after eval (swap prev ↔ next)."""
function rotate!(ls::OsdiLimitState)
    ls.a_is_prev = !ls.a_is_prev
    return nothing
end

"""Reset limit state from converged snapshot (for new timestep or rejection)."""
function reset_from_converged!(ls::OsdiLimitState)
    copyto!(ls.state_a, ls.converged)
    ls.a_is_prev = true
    ls.iteration = 0
    return nothing
end

"""Promote current prev_state to converged (on Newton/timestep convergence)."""
function promote_converged!(ls::OsdiLimitState)
    prev, _ = get_state_ptrs(ls)
    copyto!(ls.converged, prev)
    return nothing
end

"""
    OsdiModel

Wraps a model-level blob for an OSDI device type.
Multiple instances can share one model (same model parameters, different instance params).
"""
mutable struct OsdiModel
    device::OsdiDeviceType
    blob::Vector{UInt8}
    initialized::Bool
end

function OsdiModel(device::OsdiDeviceType)
    blob = zeros(UInt8, device.model_size)
    OsdiModel(device, blob, false)
end

"""
    OsdiInstance

Wraps an instance-level blob for an OSDI device.
This is the object that gets stamped into the MNA system.
"""
mutable struct OsdiInstance
    model::OsdiModel
    blob::Vector{UInt8}
    # Scratch arrays for load_residual (reused every eval)
    residual_resist::Vector{Float64}
    residual_react::Vector{Float64}
    # Limit state ($limit support)
    limit_state::OsdiLimitState
    # References to bound sparse matrix storage (prevent GC of nzval arrays)
    bound_G_nzval::Vector{Float64}
    bound_C_nzval::Vector{Float64}
    # Scratch cell for ground-node Jacobian writes (absorbs writes that go nowhere)
    ground_scratch::Vector{Float64}
    # Cached node mapping (OSDI local node → MNA global unknown index)
    node_mapping::Vector{Int}
    initialized::Bool
end

function OsdiInstance(model::OsdiModel)
    dev = model.device
    blob = zeros(UInt8, dev.instance_size)
    OsdiInstance(
        model, blob,
        zeros(Float64, dev.num_nodes),
        zeros(Float64, dev.num_nodes),
        OsdiLimitState(dev.num_states),
        Float64[], Float64[],  # bound nzval refs (set during bind)
        Float64[0.0],           # ground_scratch: absorbs ground-node Jacobian writes
        zeros(Int, dev.num_nodes),
        false,
    )
end

#==============================================================================#
# Parameter Access
#==============================================================================#

"""
    set_param!(blob, model_blob, device, info, val)

Set a parameter value in a model or instance blob via the OSDI access function.
"""
function set_param!(blob::Vector{UInt8}, model_blob::Vector{UInt8},
                    device::OsdiDeviceType, info::OsdiParamInfo, val)
    flags = ACCESS_FLAG_SET
    if info.kind == :instance
        flags |= ACCESS_FLAG_INSTANCE
    end
    GC.@preserve blob model_blob begin
        ptr = ccall(device.fn_access, Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{Cvoid}, UInt32, UInt32),
            pointer(blob), pointer(model_blob), info.osdi_id, flags)
        if info.type == Float64
            unsafe_store!(Ptr{Float64}(ptr), Float64(val))
        elseif info.type == Int32
            unsafe_store!(Ptr{Int32}(ptr), Int32(val))
        elseif info.type == String
            # String parameters need special handling — skip for now
            error("String parameter setting not yet implemented")
        end
    end
    return nothing
end

"""
    set_param!(model::OsdiModel, name::String, val)

Set a model parameter by name.
"""
function set_param!(model::OsdiModel, name::String, val)
    info = model.device.param_by_name[name]
    set_param!(model.blob, model.blob, model.device, info, val)
end

"""
    set_param!(inst::OsdiInstance, name::String, val)

Set an instance parameter by name.
"""
function set_param!(inst::OsdiInstance, name::String, val)
    info = inst.model.device.param_by_name[name]
    set_param!(inst.blob, inst.model.blob, inst.model.device, info, val)
end

#==============================================================================#
# Sim Params ($simparam support)
#
# OSDI's simparam_opt() iterates a NULL-terminated names array and looks up
# corresponding vals. We provide standard SPICE simulator parameters.
#==============================================================================#

# NULL-terminated name arrays (matching VACASK's simParamNames/simStrParamNames)
const SIM_PARAM_NAMES = [
    "iniLim", "gmin", "gdev", "tnom", "minr", "scale",
    "iteration", "simulatorVersion", "simulatorSubversion",
    "sourceScaleFactor", "reltol", "vntol", "abstol", "chgtol", "fluxtol",
    # additional params some models use
    "epsmin",
]
const SIM_STR_PARAM_NAMES = ["analysis_name", "analysis_type", "cwd"]

# Pre-allocate C string pointer arrays (NULL-terminated)
const _sim_param_name_ptrs = let
    ptrs = Vector{Ptr{UInt8}}(undef, length(SIM_PARAM_NAMES) + 1)
    for (i, name) in enumerate(SIM_PARAM_NAMES)
        ptrs[i] = pointer(name)
    end
    ptrs[end] = Ptr{UInt8}(0)  # NULL terminator
    ptrs
end

const _sim_str_param_name_ptrs = let
    ptrs = Vector{Ptr{UInt8}}(undef, length(SIM_STR_PARAM_NAMES) + 1)
    for (i, name) in enumerate(SIM_STR_PARAM_NAMES)
        ptrs[i] = pointer(name)
    end
    ptrs[end] = Ptr{UInt8}(0)  # NULL terminator
    ptrs
end

# Default string values
const _sim_str_default_dc = "dc"
const _sim_str_default_tran = "tran"
const _sim_str_default_cwd = "."

"""
    make_sim_paras(; gmin=1e-12, tnom=27.0, temperature=300.15, mode=:dcop)

Build a CSimParas struct with standard SPICE simulator parameters.
Returns (sim_paras_ref, vals, str_vals, str_ptrs) — caller must GC.@preserve all.
"""
function make_sim_paras(; gmin::Float64=1e-12, tnom::Float64=27.0,
                        mode::Symbol=:dcop, iteration::Int=0,
                        init_lim::Bool=false)
    vals = Float64[
        init_lim ? 1.0 : 0.0,  # iniLim (1 on first Newton iter, 0 otherwise)
        gmin,   # gmin
        0.0,    # gdev
        tnom,   # tnom (°C)
        1e-15,  # minr
        1.0,    # scale
        Float64(iteration),  # iteration
        1.0,    # simulatorVersion
        0.0,    # simulatorSubversion
        1.0,    # sourceScaleFactor
        1e-3,   # reltol
        1e-6,   # vntol
        1e-12,  # abstol
        1e-14,  # chgtol
        1e-14,  # fluxtol
        1e-28,  # epsmin
    ]

    analysis_name = mode == :tran ? _sim_str_default_tran : _sim_str_default_dc
    str_ptrs = Ptr{UInt8}[
        pointer(analysis_name),
        pointer(analysis_name),  # analysis_type = analysis_name
        pointer(_sim_str_default_cwd),
    ]

    sp = Ref(CSimParas(
        pointer(_sim_param_name_ptrs),
        pointer(vals),
        pointer(_sim_str_param_name_ptrs),
        pointer(str_ptrs),
    ))
    return (sp, vals, str_ptrs)
end

#==============================================================================#
# Setup
#==============================================================================#

"""
    setup_model!(model::OsdiModel; gmin=1e-12, tnom=27.0)

Call the OSDI setup_model function to initialize derived model parameters.
"""
function setup_model!(model::OsdiModel; gmin::Float64=1e-12, tnom::Float64=27.0)
    dev = model.device
    sp, vals, str_ptrs = make_sim_paras(; gmin, tnom)
    init_info = Ref(CInitInfo(UInt32(0), UInt32(0), C_NULL))
    GC.@preserve model sp vals str_ptrs _sim_param_name_ptrs _sim_str_param_name_ptrs init_info begin
        ccall(dev.fn_setup_model, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{CSimParas}, Ptr{CInitInfo}),
            C_NULL, pointer(model.blob), sp, init_info)
    end
    if init_info[].num_errors > 0
        @warn "OSDI setup_model reported $(init_info[].num_errors) error(s)"
    end
    model.initialized = true
    return nothing
end

"""
    setup_instance!(inst::OsdiInstance; temperature=300.15, gmin=1e-12, tnom=27.0)

Call the OSDI setup_instance function to initialize the instance.
"""
function setup_instance!(inst::OsdiInstance; temperature::Float64=300.15,
                         gmin::Float64=1e-12, tnom::Float64=27.0)
    dev = inst.model.device
    sp, vals, str_ptrs = make_sim_paras(; gmin, tnom)
    init_info = Ref(CInitInfo(UInt32(0), UInt32(0), C_NULL))
    GC.@preserve inst sp vals str_ptrs _sim_param_name_ptrs _sim_str_param_name_ptrs init_info begin
        ccall(dev.fn_setup_instance, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Float64, UInt32, Ptr{CSimParas}, Ptr{CInitInfo}),
            C_NULL, pointer(inst.blob), pointer(inst.model.blob),
            temperature, UInt32(dev.num_terminals), sp, init_info)
    end
    if init_info[].num_errors > 0
        @warn "OSDI setup_instance reported $(init_info[].num_errors) error(s)"
    end
    inst.initialized = true
    return nothing
end

#==============================================================================#
# Blob Manipulation Helpers
#==============================================================================#

"""Write the node mapping array into the instance blob."""
function write_node_mapping!(inst::OsdiInstance)
    dev = inst.model.device
    base = dev.node_mapping_offset
    GC.@preserve inst begin
        for i in 1:dev.num_nodes
            unsafe_store!(Ptr{UInt32}(pointer(inst.blob) + base + (i-1)*4),
                          UInt32(inst.node_mapping[i]))
        end
    end
    return nothing
end

"""Write state_idx values into the instance blob (identity mapping: state i → index i-1)."""
function write_state_idx!(inst::OsdiInstance)
    dev = inst.model.device
    dev.num_states == 0 && return nothing
    base = dev.state_idx_off
    GC.@preserve inst begin
        for i in 1:dev.num_states
            unsafe_store!(Ptr{UInt32}(pointer(inst.blob) + base + (i-1)*4),
                          UInt32(i - 1))  # 0-based index into state arrays
        end
    end
    return nothing
end

"""
    apply_node_collapse!(inst::OsdiInstance)

Read the collapsed boolean array from the instance blob (set by setup_instance)
and update node_mapping so collapsed nodes share their partner's MNA index.

Uses union-find to handle transitive collapses correctly (e.g., BP→BI, BS→BI,
B→BI all share one index). For each collapse group, if a member already has an
MNA index assigned (e.g., a terminal), all members get that index.
"""
function apply_node_collapse!(inst::OsdiInstance)
    dev = inst.model.device
    isempty(dev.collapsible) && return nothing
    base = dev.collapsed_offset

    # Union-find
    uf_parent = collect(1:dev.num_nodes)
    function uf_find(x::Int)
        while uf_parent[x] != x; uf_parent[x] = uf_parent[uf_parent[x]]; x = uf_parent[x]; end; x
    end

    ground_collapsed = Set{Int}()
    GC.@preserve inst begin
        for (i, pair) in enumerate(dev.collapsible)
            is_collapsed = unsafe_load(Ptr{Bool}(pointer(inst.blob) + base + (i-1)))
            if is_collapsed
                a = Int(pair.node_1) + 1
                if pair.node_2 == typemax(UInt32)
                    push!(ground_collapsed, a)
                else
                    b = Int(pair.node_2) + 1
                    ra, rb = uf_find(a), uf_find(b)
                    ra != rb && (uf_parent[ra] = rb)
                end
            end
        end
    end

    # Build group → MNA index map from already-assigned nodes
    group_mna = Dict{Int,Int}()
    for node in ground_collapsed
        group_mna[uf_find(node)] = 0
    end
    for i in 1:dev.num_nodes
        root = uf_find(i)
        if inst.node_mapping[i] != 0 && !haskey(group_mna, root)
            group_mna[root] = inst.node_mapping[i]
        end
    end

    # Apply: any node whose group has a resolved index gets that index
    for i in 1:dev.num_nodes
        root = uf_find(i)
        if haskey(group_mna, root)
            inst.node_mapping[i] = group_mna[root]
        elseif i in ground_collapsed
            inst.node_mapping[i] = 0
        end
    end
    return nothing
end
