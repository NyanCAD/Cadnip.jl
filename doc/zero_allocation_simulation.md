# Zero-Allocation Circuit Simulation

This document describes how to achieve TRUE zero-allocation circuit simulation in Cadnip.jl for real-time applications.

## Summary

The MNA (Modified Nodal Analysis) engine achieves **true zero allocation** in the simulation step after initialization:

- `fast_rebuild!`: 0 bytes - Rebuilds circuit matrices with new parameter/time values
- `fast_residual!`: 0 bytes - Computes F(x) residual for Newton iteration
- `fast_jacobian!`: 0 bytes - Computes Jacobian matrix for Newton iteration
- ForwardDiff AD: 0 bytes - Automatic differentiation for nonlinear devices (VA models)
- Dense LAPACK: 0 bytes - `getrf!`/`getrs!` for LU factorization and solve

The only unavoidable allocation is sparse UMFPACK `lu!` (~1696 bytes per call), which can be avoided by using dense matrices or fixed factorization.

## Key Patterns

### 1. Component-Based Current Allocation

**Allocating (24 bytes per call):**
```julia
I_idx = alloc_current!(ctx, Symbol(:I_, name))  # Creates new Symbol
```

**Zero-allocation:**
```julia
I_idx = alloc_current!(ctx, :I_, name)  # Component-based API
```

The component-based API passes prefix and name as separate Symbols, avoiding runtime Symbol construction.

### 2. Solver Selection

| Solver | Allocation | Use Case |
|--------|------------|----------|
| Dense LAPACK (`getrf!`/`getrs!`) | 0 bytes | Small circuits (<50 nodes) |
| Fixed factorization (`ldiv!`) | 0 bytes | Linear circuits |
| Sparse UMFPACK (`lu!`) | ~1696 bytes | Large circuits (unavoidable) |

### 3. Circuit Builder Pattern

```julia
function zero_alloc_builder(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    end

    # Get nodes (zero-alloc with Symbol literals)
    vin = get_node!(ctx, :vin)
    vout = get_node!(ctx, :vout)

    # Voltage source with component-based API
    I_vs = alloc_current!(ctx, :I_, :Vs)
    stamp_G!(ctx, vin, I_vs, 1.0)
    stamp_G!(ctx, I_vs, vin, 1.0)
    stamp_b!(ctx, I_vs, params.Vin)

    # Linear devices (already zero-alloc)
    stamp!(Resistor(params.R), ctx, vin, vout)
    stamp!(Capacitor(params.C), ctx, vout, 0)

    # Nonlinear VA devices (zero-alloc via ForwardDiff)
    stamp!(npnbjt(), ctx, base, emitter, collector; _mna_x_=x)

    return ctx
end
```

## Measurement Methodology

**Correct approach (measures actual allocations):**
```julia
GC.enable(false)
allocs = @allocated begin
    for _ in 1:N
        step!(...)
    end
end
GC.enable(true)
```

**Incorrect approach (includes loop overhead in some environments):**
```julia
# DON'T DO THIS - may show false allocations
allocs = @allocated for _ in 1:N
    step!(...)
end
```

## Implementation Details

### DirectStampContext

For zero-allocation rebuilding, use `DirectStampContext` which uses counter-based allocation:

```julia
# Initial structure discovery with MNAContext
ctx = MNAContext()
builder(params, spec, 0.0; ctx=ctx)

# Create DirectStampContext for fast rebuilding
dctx = create_direct_stamp_context(ctx, G_nzval, C_nzval, b)

# Zero-allocation rebuild
builder(params, spec, t; ctx=dctx, x=u)
```

### EvalWorkspace

The `EvalWorkspace` wraps both contexts and provides the fast path:

```julia
# Standard compilation (sparse Jacobian zero-alloc, dense allocates)
ws = MNA.compile(circuit)

# Compilation with dense caches (ALL operations zero-alloc)
ws = MNA.compile(circuit; dense=true)

# Zero-allocation operations
MNA.fast_rebuild!(ws, u, t)
MNA.fast_residual!(resid, du, u, ws, t)
MNA.fast_jacobian!(J, du, u, ws, gamma, t)  # Zero-alloc if dense=true
```

When `dense=true`, the workspace pre-allocates dense versions of G and C matrices
that are updated during `fast_rebuild!`. This enables zero-allocation dense
Jacobian computation for use with OrdinaryDiffEq's ImplicitEuler solver.

## OrdinaryDiffEq Integration

**Finding**: OrdinaryDiffEq's `step!()` can achieve near-zero allocation (16 bytes/call) with proper configuration:

| Configuration | Allocation | Notes |
|--------------|------------|-------|
| `ImplicitEuler (default) + dense=true` | **16 bytes/call** | Best config for small circuits |
| `ImplicitEuler + KLUFactorization` (sparse) | ~1700 bytes/call | UMFPACK allocation unavoidable |
| `ImplicitEuler + LUFactorization` (explicit) | ~624 bytes/call | LinearSolve wrapper overhead |
| Manual backward Euler | **0 bytes/call** | True zero-alloc |

### Achieving 16 bytes/call with OrdinaryDiffEq

The key is to:
1. Use `compile(circuit; dense=true)` to enable zero-allocation dense Jacobian
2. Pass workspace through `p` parameter (not closure capture)
3. Use `ImplicitEuler(autodiff=false)` with default linear solver
4. Disable all saving and callbacks

```julia
using CedarSim, CedarSim.MNA, OrdinaryDiffEq

# Compile with dense caches for zero-allocation Jacobian
ws = MNA.compile(circuit; dense=true)
n = length(ws.dctx.b)
du_work = zeros(n)

# Use a struct to pass workspace (avoids closure boxing)
struct WorkspaceParams{W, D}
    ws::W
    du_work::D
end
params = WorkspaceParams(ws, du_work)

# ODE functions access workspace through p parameter
function ode_f!(du, u, p, t)
    MNA.fast_rebuild!(p.ws, u, t)
    MNA.fast_residual!(du, p.du_work, u, p.ws, t)
    return nothing
end

function ode_jac!(J, u, p, t)
    MNA.fast_rebuild!(p.ws, u, t)
    MNA.fast_jacobian!(J, p.du_work, u, p.ws, 1.0, t)
    return nothing
end

# Create ODE problem with workspace as p
jac_proto = zeros(n, n)
odef = ODEFunction(ode_f!; jac=ode_jac!, jac_prototype=jac_proto)
prob = ODEProblem(odef, u0, tspan, params)

# Create integrator with minimal overhead
integrator = init(prob, ImplicitEuler(autodiff=false),
    adaptive=false,
    dt=1e-5,
    callback=nothing,
    save_on=false,
    dense=false,
    maxiters=10_000_000
)

# Real-time loop (16 bytes/call)
while running
    step!(integrator)
end
```

**For true zero-allocation (0 bytes)**, use the manual stepper pattern below.

## Manual Transient Stepper (Zero-Allocation)

```julia
using LinearAlgebra
using LinearAlgebra.LAPACK

# Setup (allocates - done once)
G_dense = Matrix(ws.structure.G)
C_dense = Matrix(ws.structure.C)
A_work = similar(G_dense)
ipiv = Vector{LinearAlgebra.BlasInt}(undef, n)
b_work = zeros(n)
inv_dt = 1.0 / dt

# Step function (zero allocation)
function backward_euler_step!(u, ws, G, C, A_work, ipiv, b, inv_dt, t)
    # Rebuild circuit matrices (0 bytes)
    MNA.fast_rebuild!(ws, u, t)

    # Form A = G + C/dt (0 bytes)
    @inbounds for i in eachindex(A_work)
        A_work[i] = G[i] + inv_dt * C[i]
    end

    # LU factorize (0 bytes with LAPACK)
    LAPACK.getrf!(A_work, ipiv)

    # Form RHS: b = C*u/dt + sources (0 bytes)
    mul!(b, C, u)
    @inbounds for i in eachindex(b)
        b[i] = b[i] * inv_dt + ws.dctx.b[i]
    end

    # Solve (0 bytes with LAPACK)
    LAPACK.getrs!('N', A_work, ipiv, b)
    copyto!(u, b)
end

# Real-time loop
while running
    backward_euler_step!(u, ws, G_dense, C_dense, A_work, ipiv, b_work, inv_dt, t)
    t += dt
end
```

## Files

- `src/mna/context.jl`: Component-based `alloc_current!` APIs
- `src/mna/value_only.jl`: `DirectStampContext` and `stamp_voltage_contribution!`
- `src/mna/contrib.jl`: Zero-allocation ForwardDiff contribution stamping
- `test/mna/zero_alloc.jl`: Comprehensive zero-allocation tests
