# Code Tour: MNA Pipeline from SPICE to Zero-Allocation Simulation

This document traces the complete path from a SPICE netlist through code generation
to zero-allocation Newton iteration. The focus is on understanding how the builder
function, context structs, and stamping functions work together.

## Table of Contents

1. [Overview](#overview)
2. [Entry Point: Running a Benchmark](#entry-point-running-a-benchmark)
3. [Code Generation](#code-generation)
   - [SPICE → Builder Function](#spice--builder-function)
   - [Verilog-A → Device Type + stamp! Method](#verilog-a--device-type--stamp-method)
4. [Context Types](#context-types)
   - [MNAContext (Structure Discovery)](#mnacontext-structure-discovery)
   - [ValueOnlyContext (Zero Allocation)](#valueonlycontext-zero-allocation)
5. [The Three Rebuild Paths](#the-three-rebuild-paths)
6. [Value-Only Mode Deep Dive](#value-only-mode-deep-dive)
7. [Complete Data Flow](#complete-data-flow)

---

## Overview

The MNA (Modified Nodal Analysis) pipeline transforms circuit descriptions into
efficient simulation code. The key insight is separating **structure discovery**
(which matrix entries exist) from **value updates** (what values those entries have).

```
SPICE Netlist + VA Models
         │
         ▼
    Code Generation
         │
         ├─► Builder Function (circuit topology)
         └─► stamp! Methods (device equations)
                    │
                    ▼
         ┌─────────────────────────────────┐
         │   First Build (MNAContext)      │
         │   - Discover sparsity pattern   │
         │   - Create sparse matrices G, C │
         │   - Compute COO→CSC mapping     │
         └─────────────────────────────────┘
                    │
                    ▼
         ┌─────────────────────────────────┐
         │   Value-Only Rebuilds           │
         │   - Same code, different type   │
         │   - Counter-based writes        │
         │   - Zero allocations            │
         └─────────────────────────────────┘
```

---

## Entry Point: Running a Benchmark

**File:** `benchmarks/vacask/graetz/cedarsim/runme.jl`

```julia
# 1. Load Verilog-A diode model → generates sp_diode type + stamp! method
va = VerilogAParser.parsefile(diode_va_path)
Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

# 2. Parse SPICE netlist → generates builder function
circuit_code = parse_spice_to_mna(spice_code; circuit_name=:graetz_circuit,
                                   imported_hdl_modules=[sp_diode_module])
eval(circuit_code)

# 3. Create circuit wrapper
circuit = MNACircuit(graetz_circuit)

# 4. Run transient analysis
sol = tran!(circuit, (0.0, 1.0); dt=1e-6)
```

The SPICE netlist (`runme.sp`):
```spice
vs inp inn 0 sin 0.0 20 50.0
xd1 inp outp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd2 outn inp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd3 inn outp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd4 outn inn sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
cl outp outn 100u
rl outp outn 1k
```

---

## Code Generation

### SPICE → Builder Function

**File:** `src/spc/codegen.jl` - `make_mna_circuit()`

The SPICE parser generates a **builder function** that constructs the circuit
topology by stamping devices into a context.

**Generated code structure:**

```julia
function graetz_circuit(params, spec::MNASpec, t::Real=0.0;
                        x::AbstractVector=ZERO_VECTOR,
                        ctx::Union{MNAContext, ValueOnlyContext, Nothing}=nothing)
    # ─────────────────────────────────────────────────────────────────────
    # Context handling: supports both MNAContext and ValueOnlyContext
    # ─────────────────────────────────────────────────────────────────────
    if ctx === nothing
        ctx = MNAContext()              # First build: allocate new
    else
        reset_for_restamping!(ctx)      # Subsequent: reuse existing
    end

    # ─────────────────────────────────────────────────────────────────────
    # Node allocation (get_node! dispatches based on ctx type)
    # ─────────────────────────────────────────────────────────────────────
    inp = get_node!(ctx, :inp)
    inn = get_node!(ctx, :inn)
    outp = get_node!(ctx, :outp)
    outn = get_node!(ctx, :outn)

    # ─────────────────────────────────────────────────────────────────────
    # Device stamps (stamp! dispatches based on ctx type)
    # ─────────────────────────────────────────────────────────────────────

    # Sinusoidal voltage source
    stamp!(SinVoltageSource(0.0, 20.0, 50.0; name=:vs), ctx, inp, inn;
           t=t, _sim_mode_=spec.mode)

    # VA diode instances
    let dev = sp_diode(is=76.9e-12, rs=42e-3, cjo=26.5e-12, m=0.333, n=1.45)
        stamp!(dev, ctx, inp, outp;
               _mna_t_=t, _mna_mode_=spec.mode, _mna_x_=x, _mna_spec_=spec,
               _mna_instance_=:xd1)
    end
    # ... more diodes ...

    # Passive components
    stamp!(Capacitor(100e-6; name=:cl), ctx, outp, outn)
    stamp!(Resistor(1000.0; name=:rl), ctx, outp, outn)

    return ctx
end
```

**Key design decisions:**

1. **Union type for ctx**: `Union{MNAContext, ValueOnlyContext, Nothing}` allows
   the same generated code to work with both context types via dispatch.

2. **Explicit time parameter**: `t` is passed explicitly (not via closure) to
   enable JIT optimization and avoid boxing issues.

3. **Solution vector `x`**: Passed to nonlinear devices so they can compute
   I-V characteristics at the current operating point.

### Verilog-A → Device Type + stamp! Method

**File:** `src/vasim.jl` - `make_mna_device()`, `generate_mna_stamp_method_nterm()`

VA code generation creates:
1. A **struct** to hold device parameters
2. A **stamp! method** that computes contributions and stamps matrices

**Generated device struct:**

```julia
struct sp_diode <: VAModel
    is::DefaultOr{Float64}    # Saturation current
    rs::DefaultOr{Float64}    # Series resistance
    cjo::DefaultOr{Float64}   # Junction capacitance
    n::DefaultOr{Float64}     # Emission coefficient
    # ... more parameters ...
end

# Constructor with defaults
sp_diode(; is=mkdefault(1e-14), rs=mkdefault(0.0), ...) = ...
```

**Generated stamp! method (simplified):**

```julia
function stamp!(dev::sp_diode, ctx::AnyMNAContext, _node_a::Int, _node_c::Int;
                _mna_t_=0.0, _mna_mode_=:tran, _mna_x_=Float64[],
                _mna_spec_=MNASpec(), _mna_instance_=Symbol(""))

    # Extract parameters
    is = undefault(dev.is)
    rs = undefault(dev.rs)

    # Allocate internal node (for series resistance)
    _node_a_int = alloc_internal_node!(ctx, Symbol(_mna_instance_, "_a_int"))

    # Get voltages from solution vector
    V_a = _mna_x_[_node_a]
    V_c = _mna_x_[_node_c]
    V_a_int = _mna_x_[_node_a_int]

    # ═══════════════════════════════════════════════════════════════════
    # Branch: I(a, a_int) <+ V(a, a_int) / rs  (series resistance)
    # ═══════════════════════════════════════════════════════════════════

    # Wrap voltages in ForwardDiff Duals to extract derivatives
    V_1_dual = Dual{JacobianTag}(V_a, (1.0, 0.0, 0.0))      # ∂/∂V_a
    V_2_dual = Dual{JacobianTag}(V_c, (0.0, 1.0, 0.0))      # ∂/∂V_c
    V_3_dual = Dual{JacobianTag}(V_a_int, (0.0, 0.0, 1.0))  # ∂/∂V_a_int

    # Compute branch current (Dual propagates derivatives)
    I_branch = (V_1_dual - V_3_dual) / rs

    # Extract value and Jacobian entries
    I_val = value(I_branch)
    dI_dV1 = partials(I_branch, 1)   # ∂I/∂V_a = 1/rs
    dI_dV2 = partials(I_branch, 2)   # ∂I/∂V_c = 0
    dI_dV3 = partials(I_branch, 3)   # ∂I/∂V_a_int = -1/rs

    # Stamp G matrix (Jacobian)
    # These calls dispatch to different methods for MNAContext vs ValueOnlyContext!
    stamp_G!(ctx, _node_a, _node_a, dI_dV1)
    stamp_G!(ctx, _node_a, _node_a_int, dI_dV3)
    stamp_G!(ctx, _node_a_int, _node_a, -dI_dV1)
    stamp_G!(ctx, _node_a_int, _node_a_int, -dI_dV3)

    # Stamp RHS (Newton companion model)
    Ieq = I_val - dI_dV1*V_a - dI_dV2*V_c - dI_dV3*V_a_int
    stamp_b!(ctx, _node_a, -Ieq)
    stamp_b!(ctx, _node_a_int, +Ieq)

    # ═══════════════════════════════════════════════════════════════════
    # Branch: I(a_int, c) <+ is * (exp(V/Vt) - 1)  (diode junction)
    # ═══════════════════════════════════════════════════════════════════
    # ... similar stamping for exponential I-V ...

    return nothing
end
```

**The `AnyMNAContext` type alias:**

```julia
const AnyMNAContext = Union{MNAContext, ValueOnlyContext}
```

This allows the same `stamp!` method to work with either context type. The
individual `stamp_G!`, `stamp_C!`, `stamp_b!` calls dispatch to different
implementations based on the actual type.

---

## Context Types

### MNAContext (Structure Discovery)

**File:** `src/mna/context.jl`

Used during the first build to discover circuit structure.

```julia
mutable struct MNAContext
    # Node tracking
    node_names::Vector{Symbol}
    node_to_idx::Dict{Symbol,Int}
    n_nodes::Int

    # Current variables (for voltage sources, inductors)
    current_names::Vector{Symbol}
    n_currents::Int

    # COO format - GROWS during stamping via push!
    G_I::Vector{MNAIndex}   # Row indices
    G_J::Vector{MNAIndex}   # Column indices
    G_V::Vector{Float64}    # Values

    C_I::Vector{MNAIndex}
    C_J::Vector{MNAIndex}
    C_V::Vector{Float64}

    b::Vector{Float64}      # RHS vector
    # ...
end
```

**Stamping methods use `push!`:**

```julia
function stamp_G!(ctx::MNAContext, i, j, val)
    iszero(i) && return nothing
    iszero(j) && return nothing

    push!(ctx.G_I, _to_typed(i))    # ← Allocates when capacity exceeded
    push!(ctx.G_J, _to_typed(j))    # ← Allocates when capacity exceeded
    push!(ctx.G_V, extract_value(val))  # ← Allocates when capacity exceeded

    return nothing
end
```

### ValueOnlyContext (Zero Allocation)

**File:** `src/mna/value_only.jl`

Used during Newton iterations. Only stores values, not indices.

```julia
mutable struct ValueOnlyContext{T}
    # Reference to node mapping (from first build)
    node_to_idx::Dict{Symbol,Int}
    n_nodes::Int
    n_currents::Int

    # Pre-sized value arrays (no indices - they're fixed!)
    G_V::Vector{T}
    C_V::Vector{T}
    b::Vector{T}

    # Write positions (counters, not push!)
    G_pos::Int
    C_pos::Int

    # Expected sizes (for assertions)
    n_G::Int
    n_C::Int
end
```

**Stamping methods use counter-based writes:**

```julia
function stamp_G!(vctx::ValueOnlyContext{T}, i, j, val) where T
    iszero(i) && return nothing
    iszero(j) && return nothing

    v = extract_value(val)
    pos = vctx.G_pos

    @inbounds vctx.G_V[pos] = v    # Direct write - NO ALLOCATION
    vctx.G_pos = pos + 1           # Just increment counter

    # Note: No G_I, G_J writes - indices are FIXED from first build!
    return nothing
end
```

**Key insight:** The COO indices (G_I, G_J) are always the same after the first
build. Only the values change. ValueOnlyContext exploits this by:
1. Not storing indices at all
2. Writing values directly to pre-sized arrays
3. Using counters instead of push!

---

## The Three Rebuild Paths

When the solver needs to recompute matrices, `fast_rebuild!` chooses among
three paths based on what the builder supports:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        fast_rebuild!(ws, u, t)                              │
│                                                                             │
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌───────────────────┐  │
│  │  VALUE-ONLY MODE    │   │  CONTEXT REUSE MODE │   │   FALLBACK MODE   │  │
│  │  (Zero Allocation)  │   │  (Reduced Alloc)    │   │   (Full Alloc)    │  │
│  │                     │   │                     │   │                   │  │
│  │  ValueOnlyContext   │   │  MNAContext reused  │   │  New MNAContext   │  │
│  │  ~0 bytes/iter      │   │  ~100 bytes/iter    │   │  ~1-5 KB/iter     │  │
│  └─────────────────────┘   └─────────────────────┘   └───────────────────┘  │
│          ▲                          ▲                         ▲             │
│          │                          │                         │             │
│   supports_value_only    supports_ctx_reuse         neither supported       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**File:** `src/mna/precompile.jl` - `fast_rebuild!()`

```julia
function fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)
    cs = ws.structure
    ws.time = real_time(t)

    if ws.supports_value_only_mode
        # ═══════════════════════════════════════════════════════════════════
        # PATH 1: TRUE ZERO-ALLOCATION
        # ═══════════════════════════════════════════════════════════════════
        vctx = ws.vctx
        reset_value_only!(vctx)  # Just reset counters, zero b

        cs.builder(cs.params, cs.spec, ws.time; x=u, ctx=vctx)

        # Copy values (simple array operations, no allocation)
        @inbounds for k in 1:cs.G_n_coo
            ws.G_V[k] = vctx.G_V[k]
        end
        # ... similar for C_V, b ...

    elseif ws.supports_ctx_reuse
        # ═══════════════════════════════════════════════════════════════════
        # PATH 2: REDUCED ALLOCATION (reuse MNAContext)
        # ═══════════════════════════════════════════════════════════════════
        ctx = cs.builder(cs.params, cs.spec, ws.time; x=u, ctx=ws.ctx)
        _copy_ctx_to_workspace!(ws, ctx, cs)

    else
        # ═══════════════════════════════════════════════════════════════════
        # PATH 3: FALLBACK (new MNAContext each time)
        # ═══════════════════════════════════════════════════════════════════
        ctx = cs.builder(cs.params, cs.spec, ws.time; x=u)
        _copy_ctx_to_workspace!(ws, ctx, cs)
    end

    # Update sparse matrices in-place
    update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, cs.G_n_coo)
    update_sparse_from_coo!(cs.C, ws.C_V, cs.C_coo_to_nz, cs.C_n_coo)
end
```

---

## Value-Only Mode Deep Dive

### Compilation Phase (once, at simulation start)

```julia
# In DAEProblem(circuit, tspan):

# 1. Call builder to discover structure
ctx0 = builder(params, spec, 0.0; x=ZERO_VECTOR)
# ctx0.G_I = [1, 1, 2, 2, 3, ...]  (row indices)
# ctx0.G_J = [1, 2, 1, 2, 1, ...]  (column indices)
# ctx0.G_V = [0.001, -0.001, ...]  (values)

# 2. Build sparse matrices from COO format
G_I_resolved = [resolve_index(ctx0, i) for i in ctx0.G_I]
G_J_resolved = [resolve_index(ctx0, j) for j in ctx0.G_J]
G = sparse(G_I_resolved, G_J_resolved, ctx0.G_V, n, n)

# 3. Compute COO→CSC mapping
# For each COO entry, find its position in G.nzval
G_coo_to_nz = compute_coo_to_nz_mapping(G_I_resolved, G_J_resolved, G)

# 4. Create ValueOnlyContext from discovered structure
vctx = create_value_only_context(ctx0)
# vctx.G_V = Vector{Float64}(undef, length(ctx0.G_V))  ← Pre-sized!
# vctx.G_pos = 1  ← Counter starts at 1

# 5. Package into CompiledStructure + EvalWorkspace
cs = CompiledStructure(builder, params, spec, G_coo_to_nz, C_coo_to_nz, G, C, ...)
ws = EvalWorkspace(cs, vctx, ...)
```

### Iteration Phase (every Newton step)

```julia
# 1. Reset ValueOnlyContext
reset_value_only!(vctx)
# vctx.G_pos = 1
# vctx.C_pos = 1
# fill!(vctx.b, 0.0)

# 2. Call builder with ValueOnlyContext
builder(params, spec, t; x=u, ctx=vctx)
# All stamp_G!(vctx, ...) calls write to vctx.G_V[pos++]
# No allocations - just counter increments and array writes

# 3. Update sparse matrix in-place
update_sparse_from_coo!(G, vctx.G_V, G_coo_to_nz, n_G)
# G.nzval[coo_to_nz[k]] = G_V[k] for each k
```

### The update_sparse_from_coo! Function

```julia
function update_sparse_from_coo!(A::SparseMatrixCSC, V::Vector{Float64},
                                  coo_to_nz::Vector{Int}, n_coo::Int)
    nz = nonzeros(A)  # Pointer to A.nzval (no allocation)
    fill!(nz, 0.0)    # Zero out

    # Accumulate COO values into sparse positions
    @inbounds for k in 1:n_coo
        nz_idx = coo_to_nz[k]
        nz[nz_idx] += V[k]  # Accumulate (multiple COO entries may map to same position)
    end
end
```

### Type Dispatch Comparison

| Operation | MNAContext | ValueOnlyContext |
|-----------|------------|------------------|
| `stamp_G!(ctx, i, j, v)` | `push!(G_I, i); push!(G_J, j); push!(G_V, v)` | `G_V[pos++] = v` |
| `stamp_C!(ctx, i, j, v)` | `push!(C_I, i); push!(C_J, j); push!(C_V, v)` | `C_V[pos++] = v` |
| `stamp_b!(ctx, i, v)` | `b[i] += v` or deferred push | `b[i] += v` |
| `get_node!(ctx, name)` | Dict lookup, maybe allocate new | Dict lookup only |
| `alloc_current!(ctx, name)` | push + return CurrentIndex | counter++ only |

---

## Complete Data Flow

```
                              tran!(circuit, (0, 1.0))
                                        │
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │         compile_structure()               │
                    │                                           │
                    │  ctx0 = builder(params, spec, 0.0; x=[])  │
                    │  G = sparse(ctx0.G_I, ctx0.G_J, ctx0.G_V) │
                    │  G_coo_to_nz = compute_mapping(...)       │
                    │  vctx = create_value_only_context(ctx0)   │
                    └───────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │         DC solve for u0                   │
                    │         G * u0 = b                        │
                    └───────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │         DAEProblem(f, du0, u0, tspan, ws) │
                    │         - residual! uses fast_rebuild!    │
                    │         - ws (EvalWorkspace) passed as p  │
                    └───────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           IDA SOLVER LOOP                                     │
│                                                                               │
│   For each timestep:                                                          │
│     For each Newton iteration:                                                │
│       ┌─────────────────────────────────────────────────────────────────┐     │
│       │ residual!(F, du, u, ws, t)                                      │     │
│       │                                                                 │     │
│       │   fast_rebuild!(ws, u, t)                                       │     │
│       │     │                                                           │     │
│       │     ├─► reset_value_only!(vctx)                                 │     │
│       │     │     vctx.G_pos = 1; vctx.C_pos = 1; fill!(vctx.b, 0)     │     │
│       │     │                                                           │     │
│       │     ├─► builder(params, spec, t; x=u, ctx=vctx)                 │     │
│       │     │     │                                                     │     │
│       │     │     ├─► get_node!(vctx, :inp)  → lookup only              │     │
│       │     │     ├─► stamp!(Resistor, vctx, p, n)                      │     │
│       │     │     │     └─► stamp_G!(vctx, p, p, G)                     │     │
│       │     │     │           vctx.G_V[1] = G; vctx.G_pos = 2           │     │
│       │     │     │     └─► stamp_G!(vctx, p, n, -G)                    │     │
│       │     │     │           vctx.G_V[2] = -G; vctx.G_pos = 3          │     │
│       │     │     │     └─► ...                                         │     │
│       │     │     │                                                     │     │
│       │     │     └─► stamp!(sp_diode, vctx, a, c; x=u, ...)            │     │
│       │     │           └─► Compute I, dI/dV via ForwardDiff            │     │
│       │     │           └─► stamp_G!(vctx, ..., dI_dV)                  │     │
│       │     │           └─► stamp_b!(vctx, ..., Ieq)                    │     │
│       │     │                                                           │     │
│       │     └─► update_sparse_from_coo!(G, vctx.G_V, coo_to_nz)         │     │
│       │           G.nzval[coo_to_nz[k]] += vctx.G_V[k]                  │     │
│       │                                                                 │     │
│       │   F = C*du + G*u - b                                            │     │
│       └─────────────────────────────────────────────────────────────────┘     │
│       │                                                                       │
│       ▼                                                                       │
│       jacobian!(J, du, u, ws, γ, t)                                           │
│         J = G + γ*C                                                           │
│       │                                                                       │
│       ▼                                                                       │
│       Δu = J \ F                                                              │
│       u = u - Δu                                                              │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │              ODESolution                  │
                    │  sol.t = [0.0, 1e-6, 2e-6, ...]          │
                    │  sol.u = [state vectors]                  │
                    │  sol(t) = interpolated state              │
                    └───────────────────────────────────────────┘
```

---

## Performance Impact

For the VACASK Graetz benchmark (1M timepoints, ~2M Newton iterations):

| Mode | Allocations/iter | Total for 2M iters | Notes |
|------|------------------|-------------------|-------|
| Fallback | ~3 KB | ~6 GB | New MNAContext each time |
| Context Reuse | ~100 B | ~200 MB | Reuse ctx, but push! still allocates |
| Value-Only | ~0 B | ~0 | Counter-based writes to pre-sized arrays |

The value-only mode is essential for competitive performance with traditional
SPICE simulators. The key invariants that make it work:

1. **COO indices are constant** after first build - only values change
2. **Pre-sized arrays** eliminate all push! allocations
3. **Counter-based writes** replace dynamic array growth
4. **Type dispatch** routes to different implementations transparently

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/spc/codegen.jl` | SPICE → builder function codegen |
| `src/vasim.jl` | VA → device struct + stamp! codegen |
| `src/mna/context.jl` | MNAContext struct and stamp methods |
| `src/mna/value_only.jl` | ValueOnlyContext for zero-alloc mode |
| `src/mna/precompile.jl` | CompiledStructure, EvalWorkspace, fast_rebuild! |
| `src/mna/build.jl` | MNASystem assembly from context |
| `src/mna/solve.jl` | MNACircuit, DAEProblem/ODEProblem conversion |
| `src/mna/devices.jl` | Basic device stamp! methods (R, C, L, V, I) |
