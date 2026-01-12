# Implementing PCNR for circuit simulation in Julia's SciML ecosystem

A Predictor/Corrector Newton-Raphson (PCNR) solver for circuit simulation can be cleanly architected in Julia using NonlinearSolve.jl's iterator interface as the primary extensibility point, combined with BlockArrays.jl for Schur complement reduction. The key insight is that NonlinearSolve.jl lacks built-in per-iteration callbacks, but its `init`/`step!` pattern provides complete control over the Newton iteration loop, enabling custom corrector phases and device-specific refinement callbacks.

## NonlinearSolve.jl provides the foundation through its iterator interface

The `NewtonRaphson` solver in NonlinearSolve.jl is implemented as a `GeneralizedFirstOrderAlgorithm` with pluggable descent methods and globalization strategies. For PCNR, the critical extensibility mechanism is the **iterator interface**:

```julia
cache = init(prob, NewtonRaphson())
while cache.retcode == SciMLBase.ReturnCode.Default
    step!(cache; recompute_jacobian = nothing)
    # PCNR corrector phase injects HERE
    apply_corrector!(cache.u, limiter_state)
end
```

The cache exposes `cache.u` (current solution), `cache.fu` (residual), and `cache.nsteps`. After `step!`, you can directly modify `cache.u` before the next iteration—this is where device limiting and corrector refinement occur. For deeper integration, NonlinearSolve.jl provides two abstract interfaces: `AbstractDescentDirection` for custom step computation, and `AbstractTrustRegionMethod` for globalization. A PCNR-style solver could implement a custom descent that wraps the Newton step with Schur complement reduction:

```julia
struct PCNRDescent <: NonlinearSolveBase.AbstractDescentDirection
    inner::NewtonDescent
    device_registry::DeviceRegistry
end
```

## DiffEq uses a parallel nonlinear solve infrastructure optimized for implicit stepping

OrdinaryDiffEq.jl does **not** directly call NonlinearSolve.jl for implicit methods—it uses `OrdinaryDiffEqNonlinearSolve`, a specialized sublibrary with `NLNewton`, `NLAnderson`, and `NLFunctional` algorithms. These are purpose-built for implicit ODE stepping, with quasi-Newton Jacobian reuse across time steps. The integration point is the `nlsolve` keyword:

```julia
ImplicitEuler(nlsolve = NLNewton(κ = 1//100, max_iter = 10))
```

For PCNR integration with transient simulation, you have two paths. First, implement PCNR within `OrdinaryDiffEqNonlinearSolve` by creating a new `NLPCNR` type that extends the internal nonlinear solver. Second, use `NonlinearSolveAlg` wrapper to pass a custom NonlinearSolve.jl algorithm:

```julia
ImplicitEuler(nlsolve = NonlinearSolveAlg(PCNRSolver(device_registry)))
```

The second approach is cleaner but loses some ODE-specific optimizations. For DAE initialization (finding consistent initial conditions), NonlinearSolve.jl is used directly via `BrownFullBasicInit` or `ShampineCollocationInit`—PCNR would naturally apply here since initialization often requires robust convergence aids for nonlinear devices.

## No CadNIP.jl exists, but CedarSim.jl and MOSLab.jl provide circuit simulation patterns

The closest Julia packages to MNA-based circuit simulation are **CedarSim.jl** (commercial, comprehensive SPICE-like simulator with Verilog-A support), **ACME.jl** (audio circuits using state-space formulation rather than MNA), and **MOSLab.jl** (MOSFET modeling with evidence of MNA stamping in `stampTest.jl`). None implement explicit SPICE-style limiting functions like `pnjlim`.

For MNA formulation, ModelingToolkit.jl offers the most idiomatic approach. Devices are defined as acausal components with symbolic equations:

```julia
@mtkmodel Diode begin
    @extend OnePort()
    @parameters begin Is = 1e-14; n = 1.0; Vt = 0.026 end
    @equations begin i ~ Is * (exp(v/(n*Vt)) - 1) end
end
```

For PCNR, each device would stamp **two** sets of equations: the standard MNA constitutive relation (g_MNA) plus limiting constraint equations (g_lim). A diode stamps its exponential I-V into the MNA block, plus an equation like `V_d - V_lim = 0` where `V_lim` is a limited auxiliary variable. The Jacobian has block structure:

| Block | Description |
|-------|-------------|
| **J_vv** | Standard MNA Jacobian (∂g_MNA/∂v) |
| **J_vl** | Coupling from limited quantities to MNA (∂g_MNA/∂l) |
| **J_lv** | Coupling from node voltages to limiting (∂g_lim/∂v) |
| **J_ll** | Limiting self-Jacobian—often diagonal or block-diagonal |

## Schur complement reduction exploits the diagonal structure of limiting equations

The PCNR linear solve uses Schur complement to avoid doubling the system size. Given the block system `[J_vv B; C D][Δv; Δl] = [r_v; r_l]`, where D = J_ll is typically diagonal (one limiting equation per junction), the reduced system is:

```
S = J_vv - B * D⁻¹ * C    # Schur complement (same size as original MNA!)
Δv = S⁻¹ * (r_v - B * D⁻¹ * r_l)
Δl = D⁻¹ * (r_l - C * Δv)
```

When D is diagonal—the common case where each limiting variable has an independent constraint—D⁻¹ is trivially `Diagonal(1 ./ diag(D))`, making the overhead O(m) rather than O(m³). BlockArrays.jl provides efficient block indexing via `view(M, Block(i,j))`, while LinearSolve.jl handles the reduced system solve with appropriate algorithm selection (KLU for sparse, RecursiveFactorization for dense systems under **500×500**):

```julia
function schur_solve_diagonal_D(A, B, C, d_diag::Vector, f, g)
    d_inv = 1.0 ./ d_diag
    D_inv_C = d_inv .* C              # Scale rows of C
    S = A - B * D_inv_C               # Schur complement
    f_hat = f - B * (d_inv .* g)      # Modified RHS
    
    x = S \ f_hat                      # Solve reduced system
    y = d_inv .* (g - C * x)          # Back-substitute
    return x, y
end
```

For block-diagonal D (when devices have multiple coupled limiting variables), pre-factorize each small block and apply in parallel:

```julia
struct BlockDiagD{T}
    factorizations::Vector{LU{T, Matrix{T}, Vector{Int}}}
    block_size::Int
end
```

## Device-specific refinement uses a registry pattern with the iterator interface

Since NonlinearSolve.jl lacks built-in per-iteration callbacks, the idiomatic approach is a **device registry** where each device registers a `refine!` function called after every `step!`:

```julia
abstract type CircuitDevice end
function refine!(device::CircuitDevice, u_proposed, u_prev) end

struct DeviceRegistry
    devices::Vector{CircuitDevice}
    lim_indices::Dict{CircuitDevice, UnitRange{Int}}  # Maps device → limiting variable indices
end

function refine_all!(registry::DeviceRegistry, u, u_prev)
    for device in registry.devices
        idx = registry.lim_indices[device]
        @views u[idx] .= refine!(device, u[idx], u_prev[idx])
    end
end
```

For DiffEq integration, this maps to the `CallbackSet` pattern where each device contributes a `DiscreteCallback`. However, since implicit stepping calls the nonlinear solver internally, the cleaner integration is to pass the device registry into a custom nonlinear solver via `NonlinearSolveAlg`.

SPICE-style limiting computes a critical voltage `V_CRIT = n*Vt*ln(n*Vt/(Is*√2))` ≈ **0.6V for silicon** and bounds voltage changes logarithmically when exceeded:

```julia
function pnjlim(v_new, v_old, vt, vcrit)
    if v_new > vcrit && abs(v_new - v_old) > 2*vt
        return v_old + vt * log1p((v_new - v_old) / vt)
    end
    return v_new
end
```

## Complete PCNR architecture for SciML integration

The recommended architecture combines these elements into a cohesive solver:

```julia
struct PCNRSolver{D<:DeviceRegistry, B<:BlockSolverCache}
    devices::D
    block_cache::B
    max_iter::Int
    tol::Float64
end

function solve!(solver::PCNRSolver, prob)
    # Initialize with inner Newton solver
    cache = init(prob, NewtonRaphson(linsolve = nothing))
    
    for iter in 1:solver.max_iter
        u_prev = copy(cache.u)
        
        # PREDICTOR: Schur complement Newton step
        J_blocks = compute_jacobian_blocks(cache.u, solver.devices)
        r_blocks = compute_residual_blocks(cache.u, solver.devices)
        Δu = schur_solve(solver.block_cache, J_blocks, r_blocks)
        cache.u .+= Δu
        
        # CORRECTOR: Device-specific refinement
        refine_all!(solver.devices, cache.u, u_prev)
        
        # Convergence check
        cache.fu .= compute_residual(cache.u)
        if norm(cache.fu) < solver.tol
            return SciMLBase.build_solution(prob, solver, cache.u, cache.fu; retcode=Success)
        end
    end
    return SciMLBase.build_solution(prob, solver, cache.u, cache.fu; retcode=MaxIters)
end
```

For transient simulation, wrap this as an `AbstractNLSolver` compatible with OrdinaryDiffEq's `nlsolve` interface, or use `NonlinearSolveAlg(PCNRSolver(...))` to pass it through the existing machinery. The device registry naturally extends—each new device type (diode, BJT, MOSFET) implements `stamp_mna!`, `stamp_limiting!`, and `refine!` methods, with the solver orchestrating the predictor-corrector iteration transparently.

## Key implementation recommendations

The path forward involves three main components. First, build the Schur complement solver using LinearSolve.jl with cached factorizations—exploit diagonal D structure for O(m) limiting overhead. Second, implement device models with dual stamping (MNA + limiting constraints) using a trait-based interface that ModelingToolkit could potentially auto-generate from symbolic device equations. Third, wrap the complete PCNR iteration in either a custom `AbstractDescentDirection` for pure NonlinearSolve.jl use, or a new `NLPCNR` type for tight OrdinaryDiffEq integration.

The existing SciML infrastructure provides all necessary primitives—the innovation is in the PCNR-specific device interface and the block-structured solve. No existing Julia package implements this pattern, making it a novel contribution to the ecosystem.