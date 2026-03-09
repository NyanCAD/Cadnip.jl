#==============================================================================#
# OSDI MNA Stamping
#
# OSDI arrays are indexed by node_mapping values which are 1-based MNA indices.
# Since OSDI treats these as 0-based C indices, all arrays passed to OSDI
# functions must have a "bucket" element prepended at position 0.
# Julia array[1] = C bucket, array[k+1] = MNA node k.
#
# Discovery phase (MNAContext): eval + write_jacobian_array + stamp_G!/stamp_C!
#   Records COO positions and registers a setup hook to bind Jacobian pointers
#   after sparse matrices are built.
#
# Restamping phase (DirectStampContext): eval + load_jacobian + load_spice_rhs_dc
#   Pointers are already bound into the blob by the setup hook, so OSDI writes
#   directly into sparse nzval with zero intermediate copies.
#   The SPICE RHS function computes the Newton companion model (J*x - residual)
#   which is what the MNA solver's G*u = b formulation requires.
#==============================================================================#

using ..MNA: MNAContext, DirectStampContext, AnyMNAContext
using ..MNA: stamp_G!, stamp_C!, stamp_b!
using ..MNA: alloc_internal_node!

#==============================================================================#
# Entry Count Helpers
#
# write_jacobian_array_resist only fills entries with JACOBIAN_ENTRY_RESIST flag.
# But entries with only JACOBIAN_ENTRY_RESIST_CONST also have bound pointers
# and need G matrix slots. Same logic applies for reactive entries.
#==============================================================================#

"""Count non-ground G matrix stamps (RESIST + RESIST_CONST, both nodes non-zero)."""
function count_G_entries(dev::OsdiDeviceType, node_mapping::Vector{Int})
    n = 0
    for entry in dev.jacobian_entries
        if (entry.flags & JACOBIAN_ENTRY_RESIST) != 0 ||
           (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0
            row = node_mapping[entry.nodes.node_1 + 1]
            col = node_mapping[entry.nodes.node_2 + 1]
            if row != 0 && col != 0
                n += 1
            end
        end
    end
    return n
end

"""Count non-ground C matrix stamps (entries with reactive pointers, both nodes non-zero)."""
function count_C_entries(dev::OsdiDeviceType, node_mapping::Vector{Int})
    n = 0
    for entry in dev.jacobian_entries
        if entry.react_ptr_off != typemax(UInt32)
            row = node_mapping[entry.nodes.node_1 + 1]
            col = node_mapping[entry.nodes.node_2 + 1]
            if row != 0 && col != 0
                n += 1
            end
        end
    end
    return n
end

#==============================================================================#
# Bucketed Array Helpers
#
# OSDI uses node_mapping values as C array indices. Our node_mapping stores
# Julia 1-based MNA indices, so C index k accesses Julia array element k+1.
# We prepend a bucket at Julia index 1 so C index 0 → bucket, C index k → MNA node k.
#==============================================================================#

"""
    make_bucketed_prev_solve(mna_x, max_idx)

Create a zero-prepended copy of the MNA solution vector for OSDI consumption.
C index 0 = bucket (0.0), C index k = mna_x[k] for k ≥ 1.
"""
function make_bucketed_prev_solve(mna_x::AbstractVector, max_idx::Int)
    n = max(length(mna_x), max_idx)
    buf = zeros(Float64, n + 1)  # +1 for bucket
    for i in 1:length(mna_x)
        buf[i + 1] = mna_x[i]
    end
    return buf
end

#==============================================================================#
# SimInfo Construction
#==============================================================================#

function make_sim_info(flags::UInt32, t::Real, prev_solve::AbstractVector,
                       prev_state::Vector{Float64}, next_state::Vector{Float64},
                       sim_paras_ref)
    Ref(CSimInfo(
        sim_paras_ref[],
        Float64(t),
        isempty(prev_solve) ? Ptr{Float64}(C_NULL) : pointer(prev_solve),
        isempty(prev_state) ? Ptr{Float64}(C_NULL) : pointer(prev_state),
        isempty(next_state) ? Ptr{Float64}(C_NULL) : pointer(next_state),
        flags,
    ))
end

function eval_flags_for_mode(mode::Symbol)
    base = CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN
    if mode == :dcop || mode == :dc
        return base | ANALYSIS_DC
    elseif mode == :tran
        return base | CALC_REACT_RESIDUAL | CALC_REACT_JACOBIAN | ANALYSIS_TRAN
    elseif mode == :ac
        return base | CALC_REACT_JACOBIAN | ANALYSIS_AC
    else
        return base | ANALYSIS_DC
    end
end

#==============================================================================#
# OSDI Eval + Load
#==============================================================================#

"""
    osdi_eval!(inst, flags, t, prev_solve_bucketed)

Evaluate the OSDI device. `prev_solve_bucketed` must be a bucketed array
(bucket at index 1, MNA values at indices 2..n+1).
"""
function osdi_eval!(inst::OsdiInstance, flags::UInt32, t::Real,
                    prev_solve_bucketed::AbstractVector; mode::Symbol=:dcop,
                    iteration::Int=0)
    dev = inst.model.device
    ls = inst.limit_state
    prev, next = get_state_ptrs(ls)

    init_lim = (flags & INIT_LIM) != 0
    sp, sp_vals, sp_strs = make_sim_paras(; mode, iteration, init_lim)
    sim_info = make_sim_info(flags, t, prev_solve_bucketed, prev, next, sp)

    ret = GC.@preserve inst sim_info sp_vals sp_strs _sim_param_name_ptrs _sim_str_param_name_ptrs begin
        ccall(dev.fn_eval, UInt32,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{CSimInfo}),
            C_NULL, pointer(inst.blob), pointer(inst.model.blob), sim_info)
    end
    return ret
end

#==============================================================================#
# Jacobian Pointer Binding
#
# Resolves COO positions captured during discovery into nzval pointers and
# writes them into the instance blob. Called by the setup hook after
# compile_structure builds the sparse matrices.
#==============================================================================#

"""
    bind_jacobian_pointers!(inst, G_nzval, G_coo_to_idx, G_coo_start,
                            C_nzval, C_coo_to_idx, C_coo_start)

Write Ptr{Float64} values into the instance blob pointing at sparse matrix
nzval entries. `G_coo_start` is the COO position before the first stamp_G!
call during discovery — OSDI resist entry k maps to COO index `G_coo_start + k`.
"""
function bind_jacobian_pointers!(inst::OsdiInstance,
        G_nzval::AbstractVector{Float64}, G_coo_to_idx::Vector{Int}, G_coo_start::Int,
        C_nzval::AbstractVector{Float64}, C_coo_to_idx::Vector{Int}, C_coo_start::Int)
    dev = inst.model.device

    # Keep references to prevent GC collection of nzval arrays
    inst.bound_G_nzval = G_nzval
    inst.bound_C_nzval = C_nzval

    # Write resistive Jacobian pointers
    # The resist pointer array has one slot per ALL Jacobian entries (indexed by entry position).
    # load_jacobian_resist writes through ptr[entry_idx] for each entry with a resistive result.
    # coo_k tracks only non-ground entries that were stamped to G (have COO positions).
    resist_base = dev.jacobian_ptr_resist_offset
    scratch_ptr = pointer(inst.ground_scratch)
    GC.@preserve inst G_nzval begin
        coo_k = 0
        for (entry_idx, entry) in enumerate(dev.jacobian_entries)
            has_resist = (entry.flags & JACOBIAN_ENTRY_RESIST) != 0 ||
                         (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0
            row = inst.node_mapping[entry.nodes.node_1 + 1]
            col = inst.node_mapping[entry.nodes.node_2 + 1]
            if has_resist && row != 0 && col != 0
                # Non-ground resist entry → map to nzval position
                coo_k += 1
                coo_idx = G_coo_start + coo_k
                nzval_idx = G_coo_to_idx[coo_idx]
                ptr = nzval_idx > 0 ? pointer(G_nzval, nzval_idx) : scratch_ptr
            else
                # No resist flag, or ground entry → scratch buffer
                ptr = scratch_ptr
            end
            unsafe_store!(Ptr{Ptr{Float64}}(pointer(inst.blob) + resist_base +
                          (entry_idx - 1) * sizeof(Ptr{Float64})), ptr)
        end
    end

    # Write reactive Jacobian pointers (stored individually at react_ptr_off in blob)
    GC.@preserve inst C_nzval begin
        coo_k = 0
        for entry in dev.jacobian_entries
            entry.react_ptr_off == typemax(UInt32) && continue
            row = inst.node_mapping[entry.nodes.node_1 + 1]
            col = inst.node_mapping[entry.nodes.node_2 + 1]
            if row != 0 && col != 0
                coo_k += 1
                coo_idx = C_coo_start + coo_k
                nzval_idx = C_coo_to_idx[coo_idx]
                ptr = nzval_idx > 0 ? pointer(C_nzval, nzval_idx) : scratch_ptr
            else
                ptr = scratch_ptr
            end
            unsafe_store!(Ptr{Ptr{Float64}}(pointer(inst.blob) + entry.react_ptr_off), ptr)
        end
    end
end

#==============================================================================#
# Discovery Phase: stamp! for MNAContext
#==============================================================================#

function MNA.stamp!(inst::OsdiInstance, ctx::MNAContext, terminals::Int...;
                    _mna_x_::AbstractVector=Float64[],
                    _mna_spec_=MNA.MNASpec(), t::Real=0.0,
                    instance_name::Symbol=Symbol(""))
    dev = inst.model.device

    # 1. Map terminals
    for i in 1:dev.num_terminals
        inst.node_mapping[i] = terminals[i]
    end

    # 2. Read collapsed flags (set by setup_instance) and build collapse map
    # collapsed_target[node_idx] = target node_idx if collapsed, 0 otherwise
    # When a pair is collapsed, BOTH the source voltage node AND its implicit
    # equation node are eliminated. Implicit equation nodes are the last
    # num_collapsible nodes in the OSDI descriptor.
    collapsed_target = zeros(Int, dev.num_nodes)
    if !isempty(dev.collapsible)
        base = dev.collapsed_offset
        # Implicit equation node for pair i is at 0-based index:
        # num_nodes - num_collapsible + (i-1)
        n_collapsible = length(dev.collapsible)
        GC.@preserve inst begin
            for (i, pair) in enumerate(dev.collapsible)
                is_collapsed = unsafe_load(Ptr{Bool}(pointer(inst.blob) + base + (i-1)))
                if is_collapsed
                    # Collapse the source voltage node
                    src = Int(pair.node_1) + 1  # 0-based → 1-based
                    if pair.node_2 == typemax(UInt32)
                        collapsed_target[src] = -1  # collapse to ground
                    else
                        collapsed_target[src] = Int(pair.node_2) + 1
                    end
                    # Also collapse the associated implicit equation node to ground
                    impl_idx = dev.num_nodes - n_collapsible + i  # 1-based
                    collapsed_target[impl_idx] = -1  # eliminate
                end
            end
        end
    end

    # 3. Allocate internal nodes, skipping collapsed ones
    for i in (dev.num_terminals+1):dev.num_nodes
        if collapsed_target[i] != 0
            # Collapsed: map to target's MNA index
            target = collapsed_target[i]
            if target == -1
                inst.node_mapping[i] = 0  # ground
            else
                inst.node_mapping[i] = inst.node_mapping[target]
            end
        else
            inst.node_mapping[i] = alloc_internal_node!(ctx,
                Symbol(dev.nodes[i].name), instance_name)
        end
    end

    # 4. Write node_mapping and state indices into instance blob
    write_node_mapping!(inst)
    write_state_idx!(inst)

    # 3. Eval with bucketed prev_solve
    mode = hasproperty(_mna_spec_, :mode) ? _mna_spec_.mode : :dcop
    flags = eval_flags_for_mode(mode)
    max_idx = maximum(inst.node_mapping)
    prev_solve = make_bucketed_prev_solve(_mna_x_, max_idx)
    osdi_eval!(inst, flags, t, prev_solve; mode)

    # 4. Write Jacobian to flat arrays and stamp into COO
    # write_jacobian_array_resist only fills entries with JACOBIAN_ENTRY_RESIST flag.
    # Entries with only JACOBIAN_ENTRY_RESIST_CONST still need G matrix slots
    # (they have bound pointers and are written by load_jacobian_resist during restamping).
    n_resist = dev.num_resistive_entries
    jacobian_resist = Vector{Float64}(undef, n_resist)
    GC.@preserve inst jacobian_resist begin
        ccall(dev.fn_write_jacobian_array_resist, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
            pointer(inst.blob), pointer(inst.model.blob), pointer(jacobian_resist))
    end

    # 5. Stamp G — capture COO start position for pointer binding later
    # resist_array_k tracks write_jacobian_array index (RESIST entries only)
    G_coo_start = length(ctx.G_V)
    resist_array_k = 0
    for entry in dev.jacobian_entries
        has_resist = (entry.flags & JACOBIAN_ENTRY_RESIST) != 0
        has_resist_const = (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0
        (has_resist || has_resist_const) || continue
        row = inst.node_mapping[entry.nodes.node_1 + 1]
        col = inst.node_mapping[entry.nodes.node_2 + 1]
        if has_resist
            resist_array_k += 1
            stamp_G!(ctx, row, col, jacobian_resist[resist_array_k])
        else
            # RESIST_CONST-only: value loaded via pointer during restamping
            stamp_G!(ctx, row, col, 0.0)
        end
    end

    # 6. Stamp b vector — use load_spice_rhs_dc for correct Newton companion model
    # (J*x - residual, which gives 0 for linear devices at zero operating point)
    # OSDI accumulates into rhs_buf[node_mapping[i]] so collapsed nodes (same MNA index)
    # are already summed. Stamp each unique MNA index only once.
    rhs_buf = zeros(Float64, max_idx + 1)  # bucketed
    GC.@preserve inst rhs_buf prev_solve begin
        ccall(dev.fn_load_spice_rhs_dc, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
            pointer(inst.blob), pointer(inst.model.blob),
            pointer(rhs_buf), pointer(prev_solve))
    end
    stamped = Set{Int}()
    for i in 1:dev.num_nodes
        mna_row = inst.node_mapping[i]
        mna_row == 0 && continue
        mna_row in stamped && continue
        push!(stamped, mna_row)
        stamp_b!(ctx, mna_row, rhs_buf[mna_row + 1])  # +1 for Julia bucket offset
    end

    # 7. Stamp C — capture COO start position
    # Reactive entries: react_ptr_off != typemax(UInt32) determines which entries
    # have C matrix slots. write_jacobian_array_react fills REACT-flag entries.
    C_coo_start = length(ctx.C_V)
    n_react = dev.num_reactive_entries
    if n_react > 0
        jacobian_react = Vector{Float64}(undef, n_react)
        GC.@preserve inst jacobian_react begin
            ccall(dev.fn_write_jacobian_array_react, Cvoid,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
                pointer(inst.blob), pointer(inst.model.blob), pointer(jacobian_react))
        end

        react_array_k = 0
        for entry in dev.jacobian_entries
            entry.react_ptr_off == typemax(UInt32) && continue
            row = inst.node_mapping[entry.nodes.node_1 + 1]
            col = inst.node_mapping[entry.nodes.node_2 + 1]
            has_react = (entry.flags & JACOBIAN_ENTRY_REACT) != 0
            if has_react
                react_array_k += 1
                stamp_C!(ctx, row, col, jacobian_react[react_array_k])
            else
                stamp_C!(ctx, row, col, 0.0)
            end
        end
    end

    # 8. Register setup hook — called by create_workspace after sparse matrices exist
    push!(ctx.setup_hooks,
        (G_nzval, G_coo_to_idx, C_nzval, C_coo_to_idx) ->
            bind_jacobian_pointers!(inst,
                G_nzval, G_coo_to_idx, G_coo_start,
                C_nzval, C_coo_to_idx, C_coo_start))

    return nothing
end

#==============================================================================#
# Restamping Phase: stamp! for DirectStampContext
#
# Pointers are already bound by the setup hook. OSDI load functions write
# directly through them into sparse nzval — zero intermediate copies.
# The b vector uses load_spice_rhs_dc for Newton companion model.
#==============================================================================#

function MNA.stamp!(inst::OsdiInstance, ctx::DirectStampContext, terminals::Int...;
                    _mna_x_::AbstractVector=Float64[],
                    _mna_spec_=MNA.MNASpec(), t::Real=0.0,
                    instance_name::Symbol=Symbol(""))
    dev = inst.model.device
    ls = inst.limit_state

    # Limit state management — use solver's newton_iter counter for per-solve tracking.
    # For DC: newton_iter resets to 0 at each Newton solve start (by _dc_newton_compiled),
    #         fast_rebuild! increments it before calling builder.
    # For transient: time changes trigger reset_from_converged!
    current_t = Float64(t)
    if current_t != ls.last_t
        reset_from_converged!(ls)
        ls.last_t = current_t
    end
    ls.iteration += 1

    # Build eval flags
    mode = hasproperty(_mna_spec_, :mode) ? _mna_spec_.mode : :dcop
    flags = eval_flags_for_mode(mode)
    limiting_enabled = dev.num_states > 0
    if limiting_enabled
        flags |= ENABLE_LIM | CALC_RESIST_LIM_RHS
        # INIT_LIM on the first fast_rebuild! of each Newton solve
        if ctx.newton_iter <= 1
            flags |= INIT_LIM
        end
    end

    # Create bucketed prev_solve for OSDI
    max_idx = maximum(inst.node_mapping)
    prev_solve = make_bucketed_prev_solve(_mna_x_, max_idx)

    # Eval
    eval_ret = osdi_eval!(inst, flags, t, prev_solve; mode, iteration=ls.iteration)

    # Rotate limit state after eval
    if limiting_enabled
        rotate!(ls)
    end

    # Check if limiting was applied by the device
    limiting_applied = (eval_ret & EVAL_RET_FLAG_LIM) != 0

    # Load Jacobian directly through bound pointers → writes into G/C nzval
    GC.@preserve inst begin
        ccall(dev.fn_load_jacobian_resist, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}),
            pointer(inst.blob), pointer(inst.model.blob))
    end
    # Advance G_pos by non-ground G stamp count to keep counter aligned
    n_G_stamps = count_G_entries(dev, inst.node_mapping)
    ctx.G_pos += n_G_stamps

    n_C_stamps = count_C_entries(dev, inst.node_mapping)
    if n_C_stamps > 0
        GC.@preserve inst begin
            ccall(dev.fn_load_jacobian_react, Cvoid,
                (Ptr{Cvoid}, Ptr{Cvoid}, Float64),
                pointer(inst.blob), pointer(inst.model.blob), 1.0)
        end
        ctx.C_pos += n_C_stamps
    end

    # Load SPICE RHS (Newton companion model: J*x - residual)
    # load_spice_rhs_dc computes: rhs = J(xl) * x_actual - f(xl)
    # When limiting is applied, we need: rhs = J(xl) * xl - f(xl)
    # The correction is: load_limit_rhs_resist = J(xl) * (xl - x_actual)
    # So: rhs_correct = load_spice_rhs_dc + load_limit_rhs_resist
    rhs_buf = zeros(Float64, length(prev_solve))  # same size as bucketed prev_solve
    GC.@preserve inst rhs_buf prev_solve begin
        ccall(dev.fn_load_spice_rhs_dc, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
            pointer(inst.blob), pointer(inst.model.blob),
            pointer(rhs_buf), pointer(prev_solve))
    end

    # Add limit RHS correction when limiting was applied
    if limiting_applied
        lim_rhs_buf = zeros(Float64, length(prev_solve))
        GC.@preserve inst lim_rhs_buf begin
            ccall(dev.fn_load_limit_rhs_resist, Cvoid,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
                pointer(inst.blob), pointer(inst.model.blob),
                pointer(lim_rhs_buf))
        end
        rhs_buf .-= lim_rhs_buf
    end

    # Stamp companion current into b vector (each unique MNA index only once)
    stamped = Set{Int}()
    for i in 1:dev.num_nodes
        idx = inst.node_mapping[i]
        idx == 0 && continue
        idx in stamped && continue
        push!(stamped, idx)
        stamp_b!(ctx, idx, rhs_buf[idx + 1])  # +1 for Julia bucket offset
    end

    return nothing
end
