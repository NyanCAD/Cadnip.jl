# Implementing PCNR for Circuit Simulation in Cadnip.jl

A Predictor/Corrector Newton-Raphson (PCNR) solver for circuit simulation, designed to integrate with Cadnip.jl's MNA backend and Julia's SciML ecosystem. This refined plan is based on analysis of NonlinearSolve.jl, OrdinaryDiffEqNonlinearSolve, ACME.jl, and Cadnip.jl's existing architecture.

## Algorithm Summary (from Reference Paper)

The PCNR method augments standard MNA with **limiting variables** for robust convergence:

```
State vector:     x = [x_MNA; x_lim]
Residual:         g(x) = [g_MNA; g_lim]
Jacobian blocks:  [J_MNA/MNA  J_MNA/lim]
                  [J_lim/MNA  J_lim/lim]
```

**Key insight:** When limiting equations are `g_lim = v_D - V_junction` (where V_junction is a node voltage difference), then `J_lim/lim = I` (identity), making Schur complement trivial.

### PCNR Flow

```
Start → Initialize → Predict → Correct → Converged? → Return
              ↑                    |
              └────── No ──────────┘
```

**Initialize:** `x_0 = [x_0,MNA; x_0,lim]` — independent initialization by device type

**Predict (Schur Complement):**
```
(i)   Δx_MNA = -((J_MNA/MNA - J_MNA/lim · J_lim/MNA)⁻¹ · (g_MNA - J_MNA/lim · g_lim))|_{x_i}
(ii)  Δx_lim = -(g_lim|_{x_i} + J_lim/MNA|_{x_i} · Δx_MNA)
(iii) x_{i+1} = x_i + [Δx_MNA; Δx_lim]
```

**Correct:** `x_{i+1,lim} = refine(x_i, x_{i+1})` — device-specific limiting

---

## Part 1: Integration with Cadnip.jl's MNA Architecture

### 1.1 Current Cadnip.jl Structure (Code References)

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **MNAContext** | `src/mna/context.jl` | 109-238 | COO-format matrix accumulator |
| **DirectStampContext** | `src/mna/value_only.jl` | - | Zero-allocation restamping |
| **CompiledStructure** | `src/mna/precompile.jl` | 86-119 | Immutable circuit structure |
| **EvalWorkspace** | `src/mna/precompile.jl` | 141-199 | Per-iteration mutable state |
| **fast_rebuild!** | `src/mna/precompile.jl` | 200-280 | Zero-allocation matrix update |
| **_dc_newton_compiled** | `src/mna/solve.jl` | 319-356 | Current Newton solver |
| **CedarRobustNLSolve** | `src/mna/solve.jl` | 273-305 | Multi-algorithm fallback |

### 1.2 Extending MNAContext for Limiting Variables

Add limiting variable allocation to `MNAContext`:

```julia
# src/mna/context.jl — extend existing struct
@kwdef mutable struct MNAContext
    # ... existing fields from lines 109-238 ...

    # NEW: Limiting variable support
    n_lim::Int = 0                           # Number of limiting variables
    lim_names::Vector{Symbol} = Symbol[]     # e.g., :D1_vd, :Q1_vbe
    lim_device_idx::Vector{Int} = Int[]      # Maps lim_var → device registry index
    lim_junction_p::Vector{Int} = Int[]      # Positive node of junction
    lim_junction_n::Vector{Int} = Int[]      # Negative node of junction
end

"""
    alloc_lim!(ctx, name, device_idx, p, n) -> lim_idx

Allocate a limiting variable for a junction between nodes p and n.
The limiting equation is: g_lim = v_lim - (V_p - V_n) = 0
"""
function alloc_lim!(ctx::MNAContext, name::Symbol, device_idx::Int, p::Int, n::Int)
    ctx.n_lim += 1
    push!(ctx.lim_names, name)
    push!(ctx.lim_device_idx, device_idx)
    push!(ctx.lim_junction_p, p)
    push!(ctx.lim_junction_n, n)
    return ctx.n_lim  # 1-based index into x_lim partition
end
```

### 1.3 System Size with Limiting Variables

```julia
# Total system: x = [x_MNA; x_lim]
function system_size(ctx::MNAContext)
    n_mna = ctx.n_nodes + ctx.n_currents + ctx.n_charges
    return n_mna + ctx.n_lim
end

function mna_size(ctx::MNAContext)
    return ctx.n_nodes + ctx.n_currents + ctx.n_charges
end

function lim_offset(ctx::MNAContext)
    return mna_size(ctx)  # Limiting vars start after MNA vars
end
```

### 1.4 Device Stamping Pattern for PCNR

Each nonlinear device stamps both MNA and limiting equations:

```julia
abstract type LimitableDevice end

"""
Device must implement:
- stamp_mna!(device, ctx, x_mna, x_lim) → stamps G, C, b
- stamp_lim!(device, ctx, x_mna, x_lim) → returns (g_lim, J_lim_mna)
- refine(device, v_old, v_new, vt) → limited voltage
- vcrit(device) → critical voltage for limiting
"""

struct DiodeDevice <: LimitableDevice
    Is::Float64
    n::Float64
    Vt::Float64  # Thermal voltage ≈ 26mV
    p::Int       # Positive node
    n_node::Int  # Negative node
    lim_idx::Int # Index into x_lim
end

function stamp_mna!(d::DiodeDevice, ctx, x_mna, x_lim)
    # Get limited junction voltage from x_lim
    v_lim = x_lim[d.lim_idx]

    # Diode model at limited operating point
    I_d = d.Is * (exp(v_lim / (d.n * d.Vt)) - 1)
    G_d = d.Is / (d.n * d.Vt) * exp(v_lim / (d.n * d.Vt))

    # Stamp linearized companion model into MNA
    # Current source: I_eq = I_d - G_d * v_lim
    stamp_conductance!(ctx, d.p, d.n_node, G_d)
    stamp_b!(ctx, d.p, -(I_d - G_d * v_lim))
    stamp_b!(ctx, d.n_node, I_d - G_d * v_lim)

    # J_MNA/lim: derivative of MNA equations w.r.t. v_lim
    # ∂I_stamp/∂v_lim = G_d (stamps coupling to limiting var)
end

function stamp_lim!(d::DiodeDevice, ctx, x_mna, x_lim)
    # g_lim = v_lim - (V_p - V_n)
    v_lim = x_lim[d.lim_idx]
    V_p = d.p == 0 ? 0.0 : x_mna[d.p]
    V_n = d.n_node == 0 ? 0.0 : x_mna[d.n_node]

    g_lim = v_lim - (V_p - V_n)

    # J_lim/MNA: [-1, +1] at positions [p, n]
    # J_lim/lim: [1] (identity for this limiting var)

    return g_lim
end
```

---

## Part 2: NonlinearSolve.jl Integration

### 2.1 NonlinearSolve.jl Iterator Interface (Code References)

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **solve!** | `lib/NonlinearSolveBase/src/solve.jl` | 295-305 | Main loop with `step!` |
| **step!** | `lib/NonlinearSolveBase/src/solve.jl` | 603-626 | Single Newton iteration |
| **init** | `lib/NonlinearSolveBase/src/solve.jl` | 195-237 | Cache initialization |
| **NewtonRaphson** | `lib/NonlinearSolveFirstOrder/src/raphson.jl` | 1-43 | Algorithm definition |
| **NewtonDescent** | `lib/NonlinearSolveBase/src/descent/newton.jl` | 1-139 | Descent direction |
| **AbstractDescentDirection** | `lib/NonlinearSolveBase/src/abstract_types.jl` | 39-90 | Extension interface |

### 2.2 Custom PCNR Descent Direction

Implement PCNR as a custom `AbstractDescentDirection`:

```julia
# src/mna/pcnr.jl

using NonlinearSolveBase: AbstractDescentDirection, AbstractDescentCache
using NonlinearSolveBase: InternalAPI, DescentResult

"""
    PCNRDescent

Custom descent direction implementing Schur complement predictor step.
"""
@kwdef @concrete struct PCNRDescent <: AbstractDescentDirection
    linsolve = nothing          # Linear solver for Schur complement
    device_registry::DeviceRegistry
end

@concrete mutable struct PCNRDescentCache <: AbstractDescentCache
    δu_mna::Vector{Float64}     # MNA step direction
    δu_lim::Vector{Float64}     # Limiting step direction
    δu::Vector{Float64}         # Combined [δu_mna; δu_lim]

    # Schur complement components
    S::SparseMatrixCSC{Float64,Int}  # J_MNA/MNA - J_MNA/lim * J_lim/MNA
    g_mna_hat::Vector{Float64}       # g_MNA - J_MNA/lim * g_lim

    # Block matrices (views into full Jacobian)
    J_mna_mna::SubArray            # J[1:n_mna, 1:n_mna]
    J_mna_lim::SubArray            # J[1:n_mna, n_mna+1:end]
    J_lim_mna::SubArray            # J[n_mna+1:end, 1:n_mna]
    # J_lim_lim is identity, not stored

    # Linear solver cache
    lincache

    # Device registry
    registry::DeviceRegistry
    n_mna::Int
    n_lim::Int
end

function InternalAPI.init(
    prob, alg::PCNRDescent, J, fu, u;
    stats, shared = nothing, pre_inverted = Val(false),
    linsolve_kwargs = (;), abstol = nothing, reltol = nothing,
    timer = get_timer_output(), kwargs...
)
    n_mna = alg.device_registry.n_mna
    n_lim = alg.device_registry.n_lim
    n_total = n_mna + n_lim

    @assert length(u) == n_total "State vector size mismatch"

    # Allocate step directions
    δu_mna = similar(u, n_mna)
    δu_lim = similar(u, n_lim)
    δu = similar(u)

    # Schur complement (same sparsity as J_MNA/MNA)
    S = copy(J[1:n_mna, 1:n_mna])
    g_mna_hat = similar(fu, n_mna)

    # Block views
    J_mna_mna = @view J[1:n_mna, 1:n_mna]
    J_mna_lim = @view J[1:n_mna, n_mna+1:end]
    J_lim_mna = @view J[n_mna+1:end, 1:n_mna]

    # Linear solver for Schur complement
    linprob = LinearProblem(S, g_mna_hat)
    lincache = init(linprob, alg.linsolve; abstol, reltol, linsolve_kwargs...)

    return PCNRDescentCache(
        δu_mna, δu_lim, δu,
        S, g_mna_hat,
        J_mna_mna, J_mna_lim, J_lim_mna,
        lincache,
        alg.device_registry,
        n_mna, n_lim
    )
end

function InternalAPI.solve!(
    cache::PCNRDescentCache, J, fu, u, idx::Val = Val(1);
    new_jacobian = true, kwargs...
)
    (; δu_mna, δu_lim, δu, S, g_mna_hat, lincache, n_mna, n_lim) = cache

    # Extract residual blocks
    g_mna = @view fu[1:n_mna]
    g_lim = @view fu[n_mna+1:end]

    # --- PREDICTOR: Schur Complement Solve ---

    # Since J_lim/lim = I, the Schur complement simplifies:
    # S = J_MNA/MNA - J_MNA/lim * I⁻¹ * J_lim/MNA
    #   = J_MNA/MNA - J_MNA/lim * J_lim/MNA

    J_mna_mna = @view J[1:n_mna, 1:n_mna]
    J_mna_lim = @view J[1:n_mna, n_mna+1:end]
    J_lim_mna = @view J[n_mna+1:end, 1:n_mna]

    # Compute Schur complement (sparse matrix operations)
    # S = J_MNA/MNA - J_MNA/lim * J_lim/MNA
    copyto!(S, J_mna_mna)

    # For identity J_lim/lim, J_lim/MNA has specific structure:
    # Row i of J_lim/MNA has -1 at col p_i, +1 at col n_i
    # This makes J_MNA/lim * J_lim/MNA sparse with known pattern
    for i in 1:n_lim
        device = cache.registry.devices[i]
        p, n = device.lim_junction_p, device.lim_junction_n

        # J_MNA/lim[:, i] * J_lim/MNA[i, :] = J_MNA/lim[:, i] * [-1 at p, +1 at n]
        # Subtract this outer product from S
        for row in 1:n_mna
            if p > 0
                S[row, p] += J_mna_lim[row, i]  # -(-1) = +1
            end
            if n > 0
                S[row, n] -= J_mna_lim[row, i]  # -(+1) = -1
            end
        end
    end

    # Compute modified RHS: g_MNA_hat = g_MNA - J_MNA/lim * g_lim
    # Since J_lim/lim = I: g_MNA_hat = g_MNA - J_MNA/lim * g_lim
    copyto!(g_mna_hat, g_mna)
    mul!(g_mna_hat, J_mna_lim, g_lim, -1.0, 1.0)

    # Solve: S * Δx_MNA = -g_MNA_hat
    g_mna_hat .*= -1
    lincache.b = g_mna_hat
    lincache.A = S
    sol = solve!(lincache)
    copyto!(δu_mna, sol.u)

    # Back-substitute for limiting: Δx_lim = -(g_lim + J_lim/MNA * Δx_MNA)
    # Since J_lim/lim = I, no inversion needed
    mul!(δu_lim, J_lim_mna, δu_mna)
    δu_lim .+= g_lim
    δu_lim .*= -1

    # Assemble full step
    δu[1:n_mna] .= δu_mna
    δu[n_mna+1:end] .= δu_lim

    return DescentResult(; δu, success = Val(true), linsolve_success = Val(true))
end
```

### 2.3 PCNR Solver with Corrector Phase

Wrap the iterator interface with device-specific correction:

```julia
"""
    PCNRSolver

PCNR solver that wraps NonlinearSolve.jl's NewtonRaphson with:
1. Custom PCNRDescent for Schur complement predictor
2. Post-step corrector phase with device limiting
"""
struct PCNRSolver{R<:DeviceRegistry} <: NonlinearSolve.AbstractNonlinearSolveAlgorithm
    registry::R
    max_iter::Int
    abstol::Float64
    linsolve::Any
end

function PCNRSolver(registry; max_iter=100, abstol=1e-10, linsolve=nothing)
    linsolve = linsolve === nothing ? LinearSolve.KLUFactorization() : linsolve
    return PCNRSolver(registry, max_iter, abstol, linsolve)
end

function CommonSolve.solve(prob::NonlinearProblem, alg::PCNRSolver; kwargs...)
    # Build inner Newton solver with PCNR descent
    descent = PCNRDescent(; linsolve=alg.linsolve, device_registry=alg.registry)
    inner_alg = GeneralizedFirstOrderAlgorithm(;
        descent = descent,
        linesearch = nothing,  # PCNR uses corrector instead
        trustregion = nothing,
    )

    # Initialize solver cache
    cache = init(prob, inner_alg; abstol=alg.abstol, kwargs...)

    u_prev = copy(cache.u)

    for iter in 1:alg.max_iter
        copyto!(u_prev, cache.u)

        # PREDICTOR: Schur complement Newton step
        step!(cache)

        # CORRECTOR: Device-specific limiting
        apply_corrector!(cache.u, u_prev, alg.registry)

        # Recompute residual after correction
        prob.f(cache.fu, cache.u, prob.p)

        # Check convergence
        if maximum(abs, cache.fu) < alg.abstol
            return SciMLBase.build_solution(
                prob, alg, cache.u, cache.fu;
                retcode = ReturnCode.Success
            )
        end
    end

    return SciMLBase.build_solution(
        prob, alg, cache.u, cache.fu;
        retcode = ReturnCode.MaxIters
    )
end

"""
Apply device-specific limiting to the solution.
"""
function apply_corrector!(u, u_prev, registry::DeviceRegistry)
    n_mna = registry.n_mna

    for (i, device) in enumerate(registry.devices)
        lim_idx = n_mna + i
        v_new = u[lim_idx]
        v_old = u_prev[lim_idx]

        # Device-specific limiting
        u[lim_idx] = refine(device, v_old, v_new)
    end
end
```

---

## Part 3: Device Limiting Functions

### 3.1 SPICE-Style PN Junction Limiting (pnjlim)

From SPICE3 source and reference paper:

```julia
"""
    pnjlim(v_new, v_old, vt, vcrit) -> v_limited

SPICE-style PN junction voltage limiting.

The critical voltage vcrit ≈ n*Vt*ln(n*Vt/(Is*√2)) ≈ 0.6V for silicon.
"""
function pnjlim(v_new::Real, v_old::Real, vt::Real, vcrit::Real)
    if v_new > vcrit && abs(v_new - v_old) > 2*vt
        if v_old > 0
            arg = (v_new - v_old) / vt
            if arg > 0
                return v_old + vt * (2 + log(arg - 2))
            else
                return v_old - vt * (2 + log(2 - arg))
            end
        else
            return vt * log(v_new / vt)
        end
    else
        # Large negative change - use log limiting
        if v_new < 0 && abs(v_new - v_old) > 2*vt
            return v_old - vt * log1p(-v_new/vt + 1)
        end
    end
    return v_new
end

"""
    vcrit(Is, n, vt) -> Float64

Compute critical voltage for PN junction limiting.
"""
function vcrit(Is::Real, n::Real, vt::Real)
    return n * vt * log(n * vt / (sqrt(2.0) * Is))
end
```

### 3.2 Device-Specific Refine Methods

```julia
# Diode limiting
function refine(d::DiodeDevice, v_old, v_new)
    vc = vcrit(d.Is, d.n, d.Vt)
    return pnjlim(v_new, v_old, d.Vt, vc)
end

# BJT limiting (two junctions)
function refine(q::BJTDevice, v_old::Tuple, v_new::Tuple)
    vbe_old, vbc_old = v_old
    vbe_new, vbc_new = v_new

    vc_be = vcrit(q.Is_be, q.n_be, q.Vt)
    vc_bc = vcrit(q.Is_bc, q.n_bc, q.Vt)

    vbe_lim = pnjlim(vbe_new, vbe_old, q.Vt, vc_be)
    vbc_lim = pnjlim(vbc_new, vbc_old, q.Vt, vc_bc)

    return (vbe_lim, vbc_lim)
end

# MOSFET limiting (typically less critical than BJT/diode)
function refine(m::MOSFETDevice, v_old, v_new)
    # MOSFET often uses simpler voltage clamping
    # or relies on the Schur complement alone
    δv = v_new - v_old
    max_step = 0.5  # Limit to 500mV per iteration

    if abs(δv) > max_step
        return v_old + sign(δv) * max_step
    end
    return v_new
end
```

---

## Part 4: OrdinaryDiffEq Integration for Transient Analysis

### 4.1 OrdinaryDiffEqNonlinearSolve Structure (Code References)

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **NLNewton** | `lib/OrdinaryDiffEqNonlinearSolve/src/type.jl` | 28-51 | Newton for implicit methods |
| **NonlinearSolveAlg** | `lib/OrdinaryDiffEqNonlinearSolve/src/type.jl` | 53-72 | External solver wrapper |
| **nlsolve!** | `lib/OrdinaryDiffEqNonlinearSolve/src/nlsolve.jl` | 1-172 | Main solve loop |
| **compute_step!** | `lib/OrdinaryDiffEqNonlinearSolve/src/newton.jl` | 192-322 | Newton step |
| **BrownFullBasicInit** | `lib/OrdinaryDiffEqNonlinearSolve/src/initialize_dae.jl` | 462-632 | DAE initialization |
| **ShampineCollocationInit** | `lib/OrdinaryDiffEqNonlinearSolve/src/initialize_dae.jl` | 99-458 | Collocation init |

### 4.2 PCNR as OrdinaryDiffEq Nonlinear Solver

```julia
"""
    NLPCNR

PCNR nonlinear solver for OrdinaryDiffEq implicit methods.
Compatible with `ImplicitEuler(nlsolve=NLPCNR(...))`.
"""
@kwdef struct NLPCNR{R<:DeviceRegistry} <: OrdinaryDiffEqCore.AbstractNLSolverAlgorithm
    κ::Rational{Int} = 1//100
    max_iter::Int = 10
    fast_convergence_cutoff::Rational{Int} = 1//5
    registry::R
end

# Build PCNR solver cache for OrdinaryDiffEq
function OrdinaryDiffEqNonlinearSolve.build_nlsolver(
    alg, nlalg::NLPCNR, u, uprev, p, t, dt, f, rate_prototype,
    uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits,
    γ, c, ::Val{iip}
) where {iip}
    # ... initialize PCNR-specific cache with device registry
end
```

### 4.3 Alternative: Using NonlinearSolveAlg Wrapper

For simpler integration, wrap PCNRSolver:

```julia
# In tran! function
function tran!(circuit::MNACircuit, tspan;
               solver=IDA(nlsolve=NonlinearSolveAlg(PCNRSolver(registry))),
               kwargs...)
    # ... existing transient setup from src/sweeps.jl:423-503
end
```

### 4.4 Integration with Cadnip.jl's CedarDCOp

Extend `CedarDCOp` (from `src/mna/dcop.jl:28-248`) to use PCNR:

```julia
struct CedarDCOpPCNR{R} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    maxiters::Int
    registry::R
end

function CedarDCOpPCNR(registry; abstol=1e-9, maxiters=500)
    return CedarDCOpPCNR(abstol, maxiters, registry)
end

function SciMLBase.initialize_dae!(integrator, initializealg::CedarDCOpPCNR, x...)
    # Use PCNRSolver instead of CedarRobustNLSolve
    solver = PCNRSolver(initializealg.registry;
                        abstol=initializealg.abstol,
                        max_iter=initializealg.maxiters)

    # ... rest similar to CedarDCOp implementation at src/mna/dcop.jl:145-248
end
```

---

## Part 5: Device Registry and Circuit Builder Integration

### 5.1 Device Registry

```julia
"""
Registry of all limitable devices in a circuit.
Populated during circuit structure discovery.
"""
struct DeviceRegistry
    devices::Vector{LimitableDevice}
    n_mna::Int
    n_lim::Int

    # Mapping from limiting variable index to device
    lim_to_device::Vector{Int}
end

function DeviceRegistry(ctx::MNAContext, devices::Vector{LimitableDevice})
    n_mna = mna_size(ctx)
    n_lim = ctx.n_lim
    lim_to_device = ctx.lim_device_idx

    return DeviceRegistry(devices, n_mna, n_lim, lim_to_device)
end
```

### 5.2 Extended Circuit Builder Pattern

Modify Cadnip.jl's builder pattern to support limiting variables:

```julia
# Example circuit with PCNR-enabled devices
function diode_rectifier(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    ctx = ctx === nothing ? MNAContext() : ctx

    # Standard MNA stamping
    vdd = get_node!(ctx, :vdd)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(1.0; name=:Vin), ctx, vdd, 0)
    stamp!(Resistor(1000.0; name=:R), ctx, vdd, out)

    # PCNR-enabled diode with limiting variable
    device_idx = 1  # First limitable device
    lim_idx = alloc_lim!(ctx, :D1_vd, device_idx, out, 0)

    # Get voltages from solution vector
    n_mna = mna_size(ctx)
    v_lim = isempty(x) ? 0.6 : x[n_mna + lim_idx]  # Default to ~Vf

    # Stamp diode at limited operating point
    diode = DiodeDevice(Is=1e-14, n=1.0, Vt=0.026,
                        p=out, n_node=0, lim_idx=lim_idx)
    stamp_mna!(diode, ctx, x, [v_lim])

    return ctx
end
```

### 5.3 Integration with Existing VA Device Infrastructure

Extend Verilog-A contribution stamping (from `src/mna/contrib.jl`):

```julia
# Detect if VA device needs limiting
function needs_limiting(va_device)
    # Devices with exponential I-V (diodes, BJTs) benefit from limiting
    # Detect via contribution analysis or device annotation
end

# Auto-generate limiting equations from VA model
function stamp_va_with_limiting!(device, ctx, x; enable_pcnr=true)
    if enable_pcnr && needs_limiting(device)
        # Allocate limiting variables for each junction
        for junction in device.junctions
            lim_idx = alloc_lim!(ctx, junction.name, device.idx,
                                 junction.p, junction.n)
            # ... stamp with limiting
        end
    else
        # Standard stamping without limiting
        stamp_va!(device, ctx, x)
    end
end
```

---

## Part 6: ACME.jl Comparison and Lessons

ACME.jl (`/home/user/ACME.jl`) provides valuable patterns:

### 6.1 Homotopy Continuation (Alternative to PCNR)

**File:** `/home/user/ACME.jl/src/solvers.jl:238-302`

ACME uses homotopy as backup when Newton fails:
- Interpolate: `p(λ) = (1-λ)*p_start + λ*p_target`
- Binary search for intermediate λ values
- Each point provides warm start for next

This could complement PCNR for extremely difficult convergence cases.

### 6.2 Solution Caching

**File:** `/home/user/ACME.jl/src/solvers.jl:304-405`

ACME caches solutions in k-d tree for fast nearest-neighbor lookup:
- Useful for parameter sweeps where adjacent points are similar
- Could accelerate PCNR initialization

### 6.3 Device Residual Pattern

**File:** `/home/user/ACME.jl/src/elements.jl:226-245`

Each device returns `(res, J)` tuple:
```julia
nonlinear_eq = @inline function(q)
    v, i = q
    ex = exp(v*(1 / (25e-3 * η)))
    res = @SVector [is * (ex - 1) - i]
    J = @SMatrix [is/(25e-3 * η) * ex -1]
    return (res, J)
end
```

This aligns well with PCNR's device-centric structure.

---

## Part 7: Implementation Roadmap

### Phase 1: Foundation (Modify Existing Code)

1. **Extend MNAContext** (`src/mna/context.jl`)
   - Add `n_lim`, `lim_names`, `lim_device_idx`, `lim_junction_p`, `lim_junction_n`
   - Implement `alloc_lim!`, `system_size`, `mna_size`, `lim_offset`

2. **Create DeviceRegistry** (`src/mna/pcnr.jl` - new file)
   - `LimitableDevice` abstract type
   - `DeviceRegistry` struct
   - `refine`, `vcrit`, `pnjlim` functions

3. **Update fast_rebuild!** (`src/mna/precompile.jl`)
   - Handle augmented state vector `[x_MNA; x_lim]`
   - Stamp limiting equations into residual

### Phase 2: PCNR Solver

4. **Implement PCNRDescent** (`src/mna/pcnr.jl`)
   - Schur complement computation
   - Integration with NonlinearSolve.jl

5. **Implement PCNRSolver** (`src/mna/pcnr.jl`)
   - Iterator wrapper with corrector phase
   - Device registry integration

6. **Add limiting to existing devices** (`src/mna/devices.jl`)
   - `DiodeDevice` with limiting
   - `BJTDevice` with BE/BC junction limiting

### Phase 3: Integration

7. **Extend tran!** (`src/sweeps.jl`)
   - Option to use PCNR for transient analysis
   - Auto-detect circuits that benefit from limiting

8. **Extend CedarDCOp** (`src/mna/dcop.jl`)
   - `CedarDCOpPCNR` variant

9. **VA integration** (`src/mna/contrib.jl`)
   - Auto-detect exponential devices
   - Generate limiting equations from VA models

### Phase 4: Testing and Validation

10. **Unit tests** (`test/mna/pcnr.jl` - new file)
    - Schur complement correctness
    - Limiting function accuracy
    - Convergence comparison with/without PCNR

11. **Integration tests**
    - Diode rectifier circuits
    - BJT amplifiers
    - CMOS inverters
    - Ring oscillators

12. **Benchmark**
    - Newton iteration count comparison
    - Wall-clock time for difficult circuits
    - Memory usage

---

## Appendix A: Jacobian Block Structure Details

For the example circuit in the reference (two diodes D1, D2):

```
x_MNA = [e1, e2, i]       (node voltages + source current)
x_lim = [v_D1, v_D2]      (limiting variables)

g_MNA = [i + Is1*(exp(v_D1/Vt)-1) + Is2*(exp(v_D2/Vt)-1),
         -Is1*(exp(v_D1/Vt)-1) - Is2*(exp(v_D2/Vt)-1) + e2/R,
         e1 - v_src]

g_lim = [v_D1 - (e1 - e2),
         v_D2 - (e1 - e2)]

J_MNA/MNA = [0,    0,    1;
             0,    1/R,  0;
             1,    0,    0]

J_MNA/lim = [Is1/Vt*exp(v_D1/Vt),  Is2/Vt*exp(v_D2/Vt);
             -Is1/Vt*exp(v_D1/Vt), -Is2/Vt*exp(v_D2/Vt);
             0,                     0]

J_lim/MNA = [-1,  1,  0;
             -1,  1,  0]

J_lim/lim = [1, 0;
             0, 1]  (identity!)
```

The Schur complement `S = J_MNA/MNA - J_MNA/lim * J_lim/MNA` is 3×3 (same as original MNA), not 5×5.

---

## Appendix B: Key File References

| File | Lines | Component |
|------|-------|-----------|
| **Cadnip.jl** | | |
| `src/mna/context.jl` | 109-238 | MNAContext (extend for limiting) |
| `src/mna/precompile.jl` | 86-119 | CompiledStructure |
| `src/mna/precompile.jl` | 200-280 | fast_rebuild! |
| `src/mna/solve.jl` | 273-305 | CedarRobustNLSolve |
| `src/mna/solve.jl` | 319-356 | _dc_newton_compiled |
| `src/mna/dcop.jl` | 28-248 | CedarDCOp |
| `src/mna/devices.jl` | 124-183 | Basic device types |
| `src/mna/contrib.jl` | 1-74 | VA contribution stamping |
| `src/sweeps.jl` | 423-503 | tran!/dc! API |
| **NonlinearSolve.jl** | | |
| `lib/NonlinearSolveBase/src/solve.jl` | 295-305 | Main solve loop |
| `lib/NonlinearSolveBase/src/solve.jl` | 603-626 | step! function |
| `lib/NonlinearSolveBase/src/abstract_types.jl` | 39-90 | AbstractDescentDirection |
| `lib/NonlinearSolveBase/src/descent/newton.jl` | 1-139 | NewtonDescent |
| **OrdinaryDiffEq.jl** | | |
| `lib/OrdinaryDiffEqNonlinearSolve/src/type.jl` | 28-51 | NLNewton |
| `lib/OrdinaryDiffEqNonlinearSolve/src/type.jl` | 53-72 | NonlinearSolveAlg |
| `lib/OrdinaryDiffEqNonlinearSolve/src/nlsolve.jl` | 1-172 | nlsolve! |
| `lib/OrdinaryDiffEqNonlinearSolve/src/initialize_dae.jl` | 462-632 | BrownFullBasicInit |
| **ACME.jl** | | |
| `src/solvers.jl` | 151-236 | SimpleSolver (Newton) |
| `src/solvers.jl` | 238-302 | HomotopySolver |
| `src/elements.jl` | 226-245 | Diode implementation |
