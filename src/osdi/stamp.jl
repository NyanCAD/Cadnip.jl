#==============================================================================#
# OSDI MNA Stamping
#
# Implicit equation elimination:
# OpenVAF creates "implicit_equation_*" nodes for ddt() contributions.
# These create auxiliary MNA unknowns with off-diagonal C entries that cause
# severe numerical stiffness. We eliminate them by redistributing their
# reactive entries to the physical nodes they couple to through G.
#
# For a diode with implicit_equation_0:
#   Original: G[a_int,impl]=+1, G[c,impl]=-1, G[impl,impl]=-1,
#             C[impl,a_int]=Cj, C[impl,c]=-Cj
#   After elimination: C[a_int,a_int]+=Cj, C[a_int,c]-=Cj,
#                       C[c,a_int]-=Cj, C[c,c]+=Cj
# This matches the direct capacitance stamp used by VA-compiled models.
#==============================================================================#

using ..MNA: MNAContext, DirectStampContext, AnyMNAContext
using ..MNA: stamp_G!, stamp_C!, stamp_b!
using ..MNA: alloc_internal_node!

#==============================================================================#
# Implicit Node Detection
#==============================================================================#

"""Check if an OSDI node is an implicit equation node (created by OpenVAF for ddt())."""
function is_implicit_node(dev::OsdiDeviceType, node_1based::Int)
    node_1based > dev.num_nodes && return false
    name = dev.nodes[node_1based].name
    return startswith(name, "implicit_equation")
end

"""
    detect_implicit_nodes(dev) -> Set{Int}

Return 1-based indices of implicit equation nodes.
"""
function detect_implicit_nodes(dev::OsdiDeviceType)
    impl = Set{Int}()
    for i in (dev.num_terminals+1):dev.num_nodes
        is_implicit_node(dev, i) && push!(impl, i)
    end
    return impl
end

"""
    compute_coupling_map(dev, implicit_nodes, jacobian_resist) -> Dict

For each implicit node, find the physical nodes that couple to it through G entries
(entries where col == implicit_node with RESIST or RESIST_CONST flags), and extract
the coupling coefficient from the resist array.
"""
function compute_coupling_map(dev::OsdiDeviceType, implicit_nodes::Set{Int},
                              jacobian_resist::Vector{Float64})
    coupling = Dict{Int, Vector{Tuple{Int, Float64}}}()
    for impl in implicit_nodes
        coupling[impl] = Tuple{Int, Float64}[]
    end

    resist_k = 0
    for entry in dev.jacobian_entries
        has_resist = (entry.flags & JACOBIAN_ENTRY_RESIST) != 0
        has_resist_const = (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0
        (has_resist || has_resist_const) || continue

        row = Int(entry.nodes.node_1) + 1  # 1-based OSDI node index
        col = Int(entry.nodes.node_2) + 1

        if has_resist
            resist_k += 1
        end

        # Check if this entry couples a physical node (row) to an implicit node (col)
        if col in implicit_nodes && !(row in implicit_nodes)
            # The coupling coefficient is the G[row, impl] value
            coeff = has_resist ? jacobian_resist[resist_k] : 0.0
            # For RESIST_CONST entries without RESIST, the value comes from the pointer.
            # At zero operating point, RESIST entries give us the correct coupling.
            # For pure RESIST_CONST (value loaded via pointer), we need to handle specially.
            # However, for the implicit equation pattern, the coupling entries (like entry 9
            # for the diode) have the RESIST flag, so the value IS in the resist array.
            if coeff != 0.0
                push!(coupling[col], (row, coeff))
            end
        end
    end
    return coupling
end

#==============================================================================#
# Bucketed Array Helpers
#==============================================================================#

"""
    make_bucketed_prev_solve(mna_x, max_idx)

Create a zero-prepended copy of the MNA solution vector for OSDI consumption.
C index 0 = bucket (0.0), C index k = mna_x[k] for k >= 1.
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
#==============================================================================#

function bind_jacobian_pointers!(inst::OsdiInstance,
        G_nzval::AbstractVector{Float64}, G_coo_to_idx::Vector{Int}, G_coo_start::Int,
        C_nzval::AbstractVector{Float64}, C_coo_to_idx::Vector{Int}, C_coo_start::Int)
    dev = inst.model.device

    inst.bound_G_nzval = G_nzval
    inst.bound_C_nzval = C_nzval

    resist_base = dev.jacobian_ptr_resist_offset
    scratch_ptr = pointer(inst.ground_scratch)
    GC.@preserve inst G_nzval begin
        coo_k = 0
        for (entry_idx, entry) in enumerate(dev.jacobian_entries)
            has_resist = (entry.flags & JACOBIAN_ENTRY_RESIST) != 0 ||
                         (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0
            row_node = Int(entry.nodes.node_1) + 1
            col_node = Int(entry.nodes.node_2) + 1
            row = inst.node_mapping[row_node]
            col = inst.node_mapping[col_node]
            is_impl = row_node in inst.implicit_nodes || col_node in inst.implicit_nodes
            if has_resist && !is_impl && row != 0 && col != 0
                # Non-ground, non-implicit resist entry → map to nzval position
                coo_k += 1
                coo_idx = G_coo_start + coo_k
                nzval_idx = G_coo_to_idx[coo_idx]
                ptr = nzval_idx > 0 ? pointer(G_nzval, nzval_idx) : scratch_ptr
            else
                ptr = scratch_ptr
            end
            unsafe_store!(Ptr{Ptr{Float64}}(pointer(inst.blob) + resist_base +
                          (entry_idx - 1) * sizeof(Ptr{Float64})), ptr)
        end
    end

    # Reactive pointers: only bind for non-implicit entries
    # Implicit entries use write_jacobian_array_react + manual redistribution
    GC.@preserve inst C_nzval begin
        coo_k = 0
        for entry in dev.jacobian_entries
            entry.react_ptr_off == typemax(UInt32) && continue
            row_node = Int(entry.nodes.node_1) + 1
            row = inst.node_mapping[row_node]
            col = inst.node_mapping[entry.nodes.node_2 + 1]
            if !(row_node in inst.implicit_nodes) && row != 0 && col != 0
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
# Helper: stamp redistributed C entries from implicit node reactive Jacobian
#==============================================================================#

"""
    stamp_redistributed_C!(ctx, dev, inst, jacobian_react)

For each REACT entry in the implicit equation row, redistribute the C value
to physical nodes using the pre-computed coupling coefficients.
Non-implicit REACT entries are stamped normally.
"""
function stamp_redistributed_C!(ctx, dev::OsdiDeviceType, inst::OsdiInstance,
                                jacobian_react::Vector{Float64})
    react_k = 0
    for entry in dev.jacobian_entries
        entry.react_ptr_off == typemax(UInt32) && continue
        row_node = Int(entry.nodes.node_1) + 1  # 1-based OSDI node
        col_node = Int(entry.nodes.node_2) + 1
        has_react = (entry.flags & JACOBIAN_ENTRY_REACT) != 0

        if has_react
            react_k += 1
        end

        col = inst.node_mapping[col_node]

        if row_node in inst.implicit_nodes
            # Redistribute this implicit row's C entry to coupled physical nodes
            C_val = has_react ? jacobian_react[react_k] : 0.0
            couplings = get(inst.coupling_map, row_node, Tuple{Int, Float64}[])
            for (phys_node, coeff) in couplings
                phys_mna = inst.node_mapping[phys_node]
                stamp_C!(ctx, phys_mna, col, coeff * C_val)
            end
        else
            # Normal (non-implicit) C entry
            row = inst.node_mapping[row_node]
            if has_react
                stamp_C!(ctx, row, col, jacobian_react[react_k])
            else
                stamp_C!(ctx, row, col, 0.0)
            end
        end
    end
end

#==============================================================================#
# Discovery Phase: stamp! for MNAContext
#==============================================================================#

function MNA.stamp!(inst::OsdiInstance, ctx::MNAContext, terminals::Int...;
                    _mna_x_::AbstractVector=Float64[],
                    _mna_spec_=MNA.MNASpec(), t::Real=0.0,
                    _mna_instance_::Symbol=Symbol(""))
    dev = inst.model.device

    # 1. Map terminals
    for i in 1:dev.num_terminals
        inst.node_mapping[i] = terminals[i]
    end

    # 2. Detect implicit equation nodes
    inst.implicit_nodes = detect_implicit_nodes(dev)

    # 3. Read collapsed flags and group nodes using union-find
    uf_parent = collect(1:dev.num_nodes)
    uf_find = let p = uf_parent
        function find(x::Int)
            while p[x] != x; p[x] = p[p[x]]; x = p[x]; end; x
        end
    end
    ground_collapsed = Set{Int}()
    if !isempty(dev.collapsible)
        base = dev.collapsed_offset
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
    end

    # 4. Allocate internal nodes, respecting collapse groups
    # Implicit equation nodes get a "virtual" MNA index (beyond real system size)
    # so OSDI can index arrays correctly, but they don't create real MNA unknowns.
    group_mna = Dict{Int,Int}()
    for node in ground_collapsed
        group_mna[uf_find(node)] = 0
    end
    for i in 1:dev.num_terminals
        root = uf_find(i)
        if !haskey(group_mna, root)
            group_mna[root] = inst.node_mapping[i]
        end
    end
    # Track next virtual index for implicit nodes
    # We'll assign virtual indices after knowing max real index
    virtual_nodes = Int[]
    for i in (dev.num_terminals+1):dev.num_nodes
        root = uf_find(i)
        if haskey(group_mna, root)
            inst.node_mapping[i] = group_mna[root]
        elseif i in ground_collapsed
            inst.node_mapping[i] = 0
        elseif i in inst.implicit_nodes
            # Defer — assign virtual index later
            push!(virtual_nodes, i)
        else
            idx = alloc_internal_node!(ctx, Symbol(dev.nodes[i].name), _mna_instance_)
            inst.node_mapping[i] = idx
            group_mna[root] = idx
        end
    end
    # Assign virtual indices for implicit nodes (beyond real MNA space)
    # These exist only in bucketed arrays passed to OSDI, not in the MNA system
    max_real = maximum(inst.node_mapping; init=0)
    virtual_base = max_real + 1
    for (k, i) in enumerate(virtual_nodes)
        inst.node_mapping[i] = virtual_base + k - 1
    end

    # 5. Write node_mapping and state indices into instance blob
    write_node_mapping!(inst)
    write_state_idx!(inst)

    # 6. Eval
    mode = hasproperty(_mna_spec_, :mode) ? _mna_spec_.mode : :dcop
    flags = eval_flags_for_mode(mode)
    max_idx = maximum(inst.node_mapping)
    prev_solve = make_bucketed_prev_solve(_mna_x_, max_idx)
    osdi_eval!(inst, flags, t, prev_solve; mode)

    # 7. Write resistive Jacobian to flat array
    n_resist = dev.num_resistive_entries
    jacobian_resist = Vector{Float64}(undef, n_resist)
    GC.@preserve inst jacobian_resist begin
        ccall(dev.fn_write_jacobian_array_resist, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
            pointer(inst.blob), pointer(inst.model.blob), pointer(jacobian_resist))
    end

    # 8. Compute coupling map from resist array (before stamping)
    inst.coupling_map = compute_coupling_map(dev, inst.implicit_nodes, jacobian_resist)

    # 9. Stamp G — skip entries involving implicit nodes
    G_coo_start = length(ctx.G_V)
    resist_array_k = 0
    for entry in dev.jacobian_entries
        has_resist = (entry.flags & JACOBIAN_ENTRY_RESIST) != 0
        has_resist_const = (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0
        (has_resist || has_resist_const) || continue

        row_node = Int(entry.nodes.node_1) + 1
        col_node = Int(entry.nodes.node_2) + 1

        if has_resist
            resist_array_k += 1
        end

        # Skip entries involving implicit nodes
        if row_node in inst.implicit_nodes || col_node in inst.implicit_nodes
            continue
        end

        row = inst.node_mapping[row_node]
        col = inst.node_mapping[col_node]
        if has_resist
            stamp_G!(ctx, row, col, jacobian_resist[resist_array_k])
        else
            stamp_G!(ctx, row, col, 0.0)
        end
    end

    # 10. Stamp b — skip implicit nodes, redistribute their RHS
    rhs_buf = zeros(Float64, max_idx + 1)
    GC.@preserve inst rhs_buf prev_solve begin
        ccall(dev.fn_load_spice_rhs_dc, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
            pointer(inst.blob), pointer(inst.model.blob),
            pointer(rhs_buf), pointer(prev_solve))
    end
    stamped = Set{Int}()
    for i in 1:dev.num_nodes
        i in inst.implicit_nodes && continue
        mna_row = inst.node_mapping[i]
        mna_row == 0 && continue
        mna_row in stamped && continue
        push!(stamped, mna_row)
        stamp_b!(ctx, mna_row, rhs_buf[mna_row + 1])
    end

    # 11. Stamp C — redistribute implicit rows to physical nodes
    C_coo_start = length(ctx.C_V)
    n_react = dev.num_reactive_entries
    jacobian_react = n_react > 0 ? Vector{Float64}(undef, n_react) : Float64[]
    if n_react > 0
        GC.@preserve inst jacobian_react begin
            ccall(dev.fn_write_jacobian_array_react, Cvoid,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
                pointer(inst.blob), pointer(inst.model.blob), pointer(jacobian_react))
        end
    end
    stamp_redistributed_C!(ctx, dev, inst, jacobian_react)

    # 12. Register setup hook
    push!(ctx.setup_hooks,
        (G_nzval, G_coo_to_idx, C_nzval, C_coo_to_idx) ->
            bind_jacobian_pointers!(inst,
                G_nzval, G_coo_to_idx, G_coo_start,
                C_nzval, C_coo_to_idx, C_coo_start))

    return nothing
end

#==============================================================================#
# Restamping Phase: stamp! for DirectStampContext
#==============================================================================#

function MNA.stamp!(inst::OsdiInstance, ctx::DirectStampContext, terminals::Int...;
                    _mna_x_::AbstractVector=Float64[],
                    _mna_spec_=MNA.MNASpec(), t::Real=0.0,
                    _mna_instance_::Symbol=Symbol(""))
    dev = inst.model.device
    ls = inst.limit_state

    # Limit state management
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
        if ls.iteration <= 1
            flags |= INIT_LIM
        end
    end

    # Create bucketed prev_solve (virtual indices get 0.0 by default)
    max_idx = maximum(inst.node_mapping)
    prev_solve = make_bucketed_prev_solve(_mna_x_, max_idx)

    # Eval
    osdi_eval!(inst, flags, t, prev_solve; mode, iteration=ls.iteration)

    # Rotate limit state
    if limiting_enabled
        rotate!(ls)
    end

    # Load G via bound pointers (implicit entries go to scratch)
    GC.@preserve inst begin
        ccall(dev.fn_load_jacobian_resist, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}),
            pointer(inst.blob), pointer(inst.model.blob))
    end
    # Advance G_pos by non-implicit, non-ground G stamp count
    n_G_stamps = 0
    for entry in dev.jacobian_entries
        has_resist = (entry.flags & JACOBIAN_ENTRY_RESIST) != 0 ||
                     (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0
        has_resist || continue
        row_node = Int(entry.nodes.node_1) + 1
        col_node = Int(entry.nodes.node_2) + 1
        (row_node in inst.implicit_nodes || col_node in inst.implicit_nodes) && continue
        row = inst.node_mapping[row_node]
        col = inst.node_mapping[col_node]
        (row != 0 && col != 0) && (n_G_stamps += 1)
    end
    ctx.G_pos += n_G_stamps

    # Load C via write_jacobian_array + manual redistribution
    n_react = dev.num_reactive_entries
    jacobian_react = n_react > 0 ? Vector{Float64}(undef, n_react) : Float64[]
    if n_react > 0
        GC.@preserve inst jacobian_react begin
            ccall(dev.fn_write_jacobian_array_react, Cvoid,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
                pointer(inst.blob), pointer(inst.model.blob), pointer(jacobian_react))
        end
    end
    stamp_redistributed_C!(ctx, dev, inst, jacobian_react)

    # Load SPICE RHS
    rhs_buf = zeros(Float64, length(prev_solve))
    GC.@preserve inst rhs_buf prev_solve begin
        ccall(dev.fn_load_spice_rhs_dc, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
            pointer(inst.blob), pointer(inst.model.blob),
            pointer(rhs_buf), pointer(prev_solve))
    end

    # Stamp b — skip implicit nodes
    stamped = Set{Int}()
    for i in 1:dev.num_nodes
        i in inst.implicit_nodes && continue
        idx = inst.node_mapping[i]
        idx == 0 && continue
        idx in stamped && continue
        push!(stamped, idx)
        stamp_b!(ctx, idx, rhs_buf[idx + 1])
    end

    return nothing
end
