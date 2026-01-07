# DC Solver and Initialization Unification

This document analyzes the current fragmented DC solve and initialization APIs, identifies bugs/issues, and proposes a unified API.

---

## Part 1: Current State Analysis

### 1.1 DC Solve Functions

There are **4 different `solve_dc` overloads** with inconsistent behavior:

| Signature | Location | Nonlinear Support | Notes |
|-----------|----------|-------------------|-------|
| `solve_dc(sys::MNASystem)` | solve.jl:272 | ❌ Linear only | Direct `G \ b` |
| `solve_dc(ctx::MNAContext)` | solve.jl:306 | ❌ Linear only | Assemble + linear solve |
| `solve_dc(builder, params, spec)` | solve.jl:368 | ✅ Newton iteration | Full NonlinearSolve.jl |
| `solve_dc(circuit::MNACircuit)` | solve.jl:1663 | ❌ **BUG** | Calls linear version! |

#### Bug: `solve_dc(circuit::MNACircuit)` is broken

```julia
# solve.jl:1663-1667 - CURRENT (BROKEN)
function solve_dc(circuit::MNACircuit)
    ctx = build_with_detection(circuit)
    sys = assemble!(ctx)
    return solve_dc(sys)  # ← This is the LINEAR version!
end
```

The `dc!(circuit)` function in sweeps.jl calls `MNA.solve_dc(circuit)`, which chains to `solve_dc(sys)` - the **linear-only** path. For nonlinear circuits (diodes, MOSFETs, VA devices), this gives wrong results.

The Newton iteration version (`solve_dc(builder, params, spec)`) exists and works, but `solve_dc(circuit::MNACircuit)` doesn't use it.

### 1.2 Initialization Algorithms

#### CedarDCOp - Only works with IDA!

`CedarDCOp` (dcop.jl) only has an `initialize_dae!` method for `Sundials.IDAIntegrator`:

```julia
# dcop.jl:115 - The ONLY handler
function SciMLBase.initialize_dae!(integrator::Sundials.IDAIntegrator,
                                   alg::Union{CedarDCOp, CedarTranOp})
```

For other solvers:

| Solver | What happens with CedarDCOp |
|--------|----------------------------|
| IDA (Sundials) | ✅ Proper dcop mode switch + Newton solve |
| DFBDF, DABDF2 (OrdinaryDiffEq) | ❓ Falls back to default (undocumented) |
| Rodas5P, FBDF (ODE) | N/A (uses `NoInit()` instead) |

The docstring claims:
> "2. For OrdinaryDiffEq: Delegates to ShampineCollocationInit"

But there's **no implementation** for this! The claim in the docstring is false.

#### NoInit is deprecated

The ODE solver path uses `NoInit()`:

```julia
# sweeps.jl:532 - WRONG
function _tran_dispatch(..., solver::SciMLBase.AbstractODEAlgorithm;
                        initializealg=OrdinaryDiffEq.NoInit(), kwargs...)
```

`NoInit` is deprecated - the modern equivalent is `CheckInit`.

### 1.3 Initialization Entry Points

There are **multiple** paths to compute initial conditions:

| Entry Point | Location | When Used |
|-------------|----------|-----------|
| `compute_initial_conditions(circuit)` | solve.jl:1332 | `DAEProblem(circuit)`, `ODEProblem(circuit)` |
| `CedarDCOp` via `initialize_dae!` | dcop.jl:115 | `tran!` with IDA only |
| Linear solve in `DAEProblem` | solve.jl:1498 | Before CedarDCOp runs |
| `bootstrapped_nlsolve` | dcop.jl:69 | Inside CedarDCOp |

The relationship is confusing:
1. `DAEProblem(circuit)` calls `compute_initial_conditions` to get `u0`, `du0`
2. Then `tran!` passes `initializealg=CedarDCOp()` which RECOMPUTES initial conditions
3. But only for IDA - DFBDF uses whatever default exists

### 1.4 Summary of Issues

1. **`solve_dc(circuit::MNACircuit)` doesn't do Newton iteration** - major bug
2. **`CedarDCOp` only works with IDA** - undocumented limitation
3. **`NoInit()` is deprecated** - should use `CheckInit`
4. **Duplicate initialization paths** - `compute_initial_conditions` vs `CedarDCOp`
5. **No AC analysis on MNACircuit** - must use legacy DAECompiler path (`ac.jl`)
6. **Inconsistent API** - `dc!` vs `dc`, `solve_dc` vs various overloads

---

## Part 2: Proposed Unified API

### 2.1 Design Principles

1. **Single DC solve function** that handles both linear and nonlinear
2. **Single initialization algorithm** that works with all solvers
3. **No duplicate paths** - one way to do things
4. **Consistent naming** - `dc`, `tran`, `ac` (no `!` suffix)

### 2.2 Proposed DC Solve API

Remove all overloads except one primary entry point:

```julia
"""
    dc(circuit::MNACircuit; abstol=1e-10, maxiters=100) -> DCSolution

DC operating point analysis with automatic Newton iteration for nonlinear circuits.

For linear circuits, converges in one iteration. For nonlinear circuits (diodes,
MOSFETs, VA devices), uses Newton iteration via NonlinearSolve.jl.
"""
function dc(circuit::MNACircuit; abstol=1e-10, maxiters=100)
    # Use mode=:dcop to disable time-dependent sources
    dc_spec = with_mode(circuit.spec, :dcop)
    return solve_dc_newton(circuit.builder, circuit.params, dc_spec;
                           abstol=abstol, maxiters=maxiters)
end

# Internal: Newton iteration DC solve (consolidates solve_dc(builder, params, spec))
function solve_dc_newton(builder, params, spec; abstol=1e-10, maxiters=100)
    # ... existing Newton iteration logic from solve.jl:368-444 ...
end

# Keep low-level versions as internal only
# solve_dc(sys::MNASystem) - linear, internal use only
# solve_dc(ctx::MNAContext) - linear, internal use only
```

**Remove:**
- `dc!` (deprecated alias)
- `solve_dc(circuit::MNACircuit)` (replaced by `dc`)

### 2.3 Proposed Initialization API

Create a unified initialization algorithm that works with all solvers:

```julia
"""
    CedarInit <: DiffEqBase.DAEInitializationAlgorithm

Unified initialization algorithm for MNA circuits.

Works with all solver types:
- Sundials (IDA): dcop mode switch + Newton solve + IDADefaultInit
- OrdinaryDiffEq DAE (DFBDF, DABDF2): dcop mode switch + Newton solve + BrownFullBasicInit
- OrdinaryDiffEq ODE (Rodas5P): CheckInit (validates pre-computed u0)

# Arguments
- `abstol`: Tolerance for DC solve (default: 1e-10)
- `mode`: Initialization mode - `:dcop` (default) or `:tranop`
"""
struct CedarInit{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    mode::Symbol  # :dcop or :tranop
    nlsolve::NLSOLVE
end
CedarInit(; abstol=1e-10, mode=:dcop, nlsolve=RobustMultiNewton()) = CedarInit(abstol, mode, nlsolve)

# Handler for Sundials IDA
function SciMLBase.initialize_dae!(integrator::Sundials.IDAIntegrator, alg::CedarInit)
    # ... existing CedarDCOp logic ...
    SciMLBase.initialize_dae!(integrator, Sundials.DefaultInit())
end

# Handler for OrdinaryDiffEq DAE solvers
function SciMLBase.initialize_dae!(integrator::OrdinaryDiffEq.DAEIntegrator, alg::CedarInit)
    # Implement dcop mode switch + Newton solve
    # Then call BrownFullBasicInit
    _cedar_init_common!(integrator, alg)
    SciMLBase.initialize_dae!(integrator, BrownFullBasicInit())
end

# For ODE problems, just validate the pre-computed u0
# (u0 comes from compute_initial_conditions which already does Newton)
```

**Remove:**
- `CedarDCOp` (replaced by `CedarInit`)
- `CedarTranOp` (merged into `CedarInit(mode=:tranop)`)

### 2.4 Proposed Transient API

```julia
"""
    tran(circuit::MNACircuit, tspan; solver=IDA(), kwargs...) -> ODESolution

Transient analysis with unified initialization.

Automatically selects DAE or ODE formulation based on solver type.
Uses CedarInit for initialization (works with all solvers).
"""
function tran(circuit::MNACircuit, tspan::Tuple{<:Real,<:Real};
              solver=IDA(max_error_test_failures=20, max_nonlinear_iters=10),
              abstol=1e-10, reltol=1e-8,
              initializealg=CedarInit(),
              kwargs...)
    if solver isa SciMLBase.AbstractDAEAlgorithm
        prob = DAEProblem(circuit, tspan)
    else
        prob = ODEProblem(circuit, tspan)
        # For ODE, use CheckInit since u0 already computed
        initializealg = CheckInit()
    end
    return solve(prob, solver; abstol, reltol, initializealg, kwargs...)
end
```

**Remove:**
- `tran!` (replaced by `tran`)
- `_tran_dispatch` methods (consolidated into `tran`)

### 2.5 Proposed AC Analysis (New)

AC analysis currently only works with DAECompiler (`ac.jl`). Add MNA support:

```julia
"""
    ac(circuit::MNACircuit, freqs) -> ACSolution
    ac(circuit::MNACircuit; fstart, fstop, ppd=10) -> ACSolution

Small-signal AC analysis at specified frequencies.

First computes DC operating point (with Newton for nonlinear), then
linearizes and solves (G + jωC)x = b at each frequency.
"""
function ac(circuit::MNACircuit, freqs::AbstractVector{<:Real})
    # 1. DC operating point with Newton iteration
    dc_sol = dc(circuit)

    # 2. Build linearized system at operating point
    ac_spec = with_mode(circuit.spec, :ac)
    ctx = circuit.builder(circuit.params, ac_spec, 0.0; x=dc_sol.x)
    sys = assemble!(ctx)

    # 3. Solve at each frequency
    return solve_ac(sys, freqs)
end
```

### 2.6 API Migration Summary

| Old API | New API | Notes |
|---------|---------|-------|
| `dc!(circuit)` | `dc(circuit)` | Newton iteration by default |
| `solve_dc(circuit::MNACircuit)` | `dc(circuit)` | Consolidated |
| `solve_dc(builder, params, spec)` | internal | Keep as `solve_dc_newton` |
| `tran!(circuit, tspan)` | `tran(circuit, tspan)` | Unified init |
| `CedarDCOp()` | `CedarInit()` | Works with all solvers |
| `CedarTranOp()` | `CedarInit(mode=:tranop)` | Merged |
| `NoInit()` | `CheckInit()` | For ODE path |
| N/A | `ac(circuit, freqs)` | New MNA AC |

---

## Part 3: Implementation Plan

### Phase 1: Fix Critical Bugs

1. **Fix `solve_dc(circuit::MNACircuit)`** to use Newton iteration
2. **Replace `NoInit()` with `CheckInit()`** in ODE path

### Phase 2: Unify Initialization

1. Create `CedarInit` with handlers for:
   - `Sundials.IDAIntegrator`
   - `OrdinaryDiffEq.DAEIntegrator` (DFBDF, DABDF2, DImplicitEuler)
2. Remove duplicate initialization in `compute_initial_conditions` vs `CedarDCOp`

### Phase 3: Unify Analysis Functions

1. Create `dc()`, `tran()`, `ac()` as primary API
2. Deprecate `dc!()`, `tran!()`
3. Add `ac()` for MNACircuit

### Phase 4: Clean Up

1. Remove deprecated functions
2. Update all tests
3. Update documentation

---

## Appendix: Code Locations

| File | Functions | Purpose |
|------|-----------|---------|
| `src/mna/solve.jl:272-309` | `solve_dc(sys)`, `solve_dc(ctx)` | Linear DC solve |
| `src/mna/solve.jl:368-444` | `solve_dc(builder, params, spec)` | Newton DC solve |
| `src/mna/solve.jl:1663-1667` | `solve_dc(circuit)` | **BROKEN** |
| `src/mna/dcop.jl` | `CedarDCOp`, `CedarTranOp` | IDA-only initialization |
| `src/sweeps.jl:435-437` | `dc!(circuit)` | Wrapper for `solve_dc(circuit)` |
| `src/sweeps.jl:488-534` | `tran!`, `_tran_dispatch` | Transient analysis |
| `src/ac.jl` | `ac!`, `noise!` | Legacy DAECompiler AC |
