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

### EvalWorkspace and Parameterized Matrix Types

The `CompiledStructure{F,P,S,M}` is parameterized on matrix type `M`:
- `SparseMatrixCSC{Float64,Int}`: sparse storage (default, for large circuits)
- `Matrix{Float64}`: dense storage (for small circuits, enables zero-alloc OrdinaryDiffEq)

```julia
# Standard compilation with sparse matrices
ws = MNA.compile(circuit)          # M = SparseMatrixCSC

# Compilation with dense matrices (ALL operations zero-alloc)
ws = MNA.compile(circuit; dense=true)  # M = Matrix{Float64}

# Zero-allocation operations
MNA.fast_rebuild!(ws, u, t)
MNA.fast_residual!(resid, du, u, ws, t)
MNA.fast_jacobian!(J, du, u, ws, gamma, t)  # Zero-alloc with matching J type
```

With `dense=true`, the G and C matrices are stored as dense `Matrix{Float64}` directly
in the `CompiledStructure`. This enables zero-allocation `fast_jacobian!` with dense
output matrices, required for optimal OrdinaryDiffEq integration.

## OrdinaryDiffEq Integration

**Finding**: OrdinaryDiffEq can achieve **true zero allocation (0 bytes/call)** with the right approach.

### Requirements for Fixed Timestep (adaptive=false)

| Requirement | Reason |
|------------|--------|
| **Dense matrices** (`dense=true`) | Sparse UMFPACK `lu!` allocates ~1696 bytes/call |
| **`blind_step!()` wrapper** | `step!()` returns ReturnCode causing 16 bytes boxing |
| **`autodiff=false`** | Use explicit Jacobian from MNA |

All solvers below achieve zero allocation with fixed timestep (`adaptive=false`).

### Adaptive Timestep (adaptive=true)

**Key finding**: Some solvers achieve **zero allocation even with adaptive stepping**!

| Solver | Fixed Timestep | Adaptive Timestep | Notes |
|--------|----------------|-------------------|-------|
| `QNDF` | 0 bytes | **0 bytes** | Recommended for stiff circuits |
| `Rodas5P` | 0 bytes | **0 bytes** | Recommended for moderate stiffness |
| `Rosenbrock23` | 0 bytes | **0 bytes** | Robust fallback |
| `ImplicitEuler` | 0 bytes | **0 bytes** | Simple, very stable |
| `FBDF` | 0 bytes | **~56 bytes** | Allocates in order control |

**Why FBDF allocates with adaptive stepping:**
FBDF uses Lagrange interpolation for variable order control, which allocates in:
- `calc_Lagrange_interp!` (bdf_utils.jl:157) - ~45 bytes for view/slice operations
- `choose_order!` (controllers.jl:178) - ~11 bytes for intermediate arrays

**Recommendation:** Use `QNDF` instead of `FBDF` for zero-allocation adaptive BDF stepping.
QNDF uses backward differences with κ-correction which avoids these allocations.

### Compatible Solvers

| Solver | Order | Type | Adaptive Alloc |
|--------|-------|------|----------------|
| `QNDF` | 1-5 | Variable-order quasi-constant BDF | **0 bytes** |
| `Rodas5P`, `Rodas5` | 5 | Rosenbrock | **0 bytes** |
| `Rodas4P`, `Rodas4` | 4 | Rosenbrock | **0 bytes** |
| `Rosenbrock23` | 2-3 | Rosenbrock | **0 bytes** |
| `ImplicitEuler` | 1 | SDIRK | **0 bytes** |
| `ImplicitMidpoint` | 2 | IRK | **0 bytes** |
| `Trapezoid` | 2 | SDIRK | **0 bytes** |
| `FBDF` | 1-5 | Variable-order BDF | ~56 bytes |

**Not compatible** (don't support mass matrices): `TRBDF2`, `KenCarp4`

### Example: Adaptive Timestep (Zero Allocation)

```julia
using CedarSim, CedarSim.MNA, OrdinaryDiffEq

circuit = MNACircuit(builder; params...)
prob = ODEProblem(circuit, (0.0, 1e-3); dense=true)

# QNDF with adaptive timestep (still 0 bytes/step!)
integrator = init(prob, QNDF(autodiff=false);
    adaptive=true,  # Adaptive stepping enabled
    dt=1e-9,        # Initial timestep
    save_on=false, dense=false,
    maxiters=10_000_000,
    initializealg=MNA.CedarTranOp())

# Real-time loop with automatic step size control (0 bytes/call)
while running
    MNA.blind_step!(integrator)
    # integrator.dt contains current adaptive timestep
    # Access state: integrator.u
end
```

For maximum control or the simplest zero-allocation approach without ODE solvers, use the manual stepper below.

## Manual Transient Stepper (Zero-Allocation)

For the simplest zero-allocation transient simulation without any ODE solver overhead:

```julia
using LinearAlgebra
using LinearAlgebra.LAPACK

# Compile with dense matrices
ws = MNA.compile(circuit; dense=true)
n = MNA.system_size(ws)

# Get dense matrices directly (no conversion needed with dense=true)
G = ws.structure.G  # Already Matrix{Float64}
C = ws.structure.C  # Already Matrix{Float64}

# Allocate working arrays once
A_work = similar(G)
ipiv = Vector{LinearAlgebra.BlasInt}(undef, n)
b_work = zeros(n)
u = zeros(n)  # State vector
dt = 1e-6
inv_dt = 1.0 / dt

# Initialize with DC solution
MNA.fast_rebuild!(ws, u, 0.0)
u .= G \ ws.dctx.b

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

# Real-time loop (TRUE 0 bytes/call)
t = 0.0
while running
    backward_euler_step!(u, ws, G, C, A_work, ipiv, b_work, inv_dt, t)
    t += dt
end
```

## Files

- `src/mna/context.jl`: Component-based `alloc_current!` APIs
- `src/mna/value_only.jl`: `DirectStampContext` and `stamp_voltage_contribution!`
- `src/mna/contrib.jl`: Zero-allocation ForwardDiff contribution stamping
- `test/mna/audio_integration.jl`: Zero-allocation OrdinaryDiffEq integration tests (QNDF with adaptive timestep)
