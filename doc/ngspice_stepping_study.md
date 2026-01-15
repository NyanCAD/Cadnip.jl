# ngspice GMIN Stepping and Source Stepping Study

## Overview

This document analyzes ngspice's convergence helper algorithms (GMIN stepping and source stepping) and compares them with Cadnip.jl's approach.

## Critical Distinction: gmin vs gshunt vs diagGmin

ngspice has THREE distinct GMIN-related parameters that serve different purposes:

### 1. CKTgmin (Device-Level GMIN)
- **Default: 1e-12**
- **Purpose:** Added to device model conductances to prevent zero-slope
- **Used in:** Device model code (diodes, MOSFETs, BJTs)
- **Example from BSIM4:**
```c
// In b4v6ld.c - diode junction conductance
gbs = SourceSatCurrent * evbs / Nvtms + ckt->CKTgmin;
cbs = SourceSatCurrent * (evbs - 1.0) + ckt->CKTgmin * vbs_jct;
```
- **Why:** Ensures devices always have some minimum conductance, preventing singular Jacobians when devices are off

### 2. CKTgshunt (Node-to-Ground Shunt)
- **Default: 0 (disabled)**
- **Purpose:** Explicit shunt resistor from each voltage node to ground
- **Set via:** `.options gshunt=1e-12`
- **Why:** Helps with floating nodes that have no DC path to ground

### 3. CKTdiagGmin (Stepping GMIN)
- **Default: starts at CKTgshunt (0), used during stepping**
- **Purpose:** Temporary conductance added to ALL matrix diagonals during GMIN stepping
- **Applied via:** `LoadGmin()` in sparse matrix code
- **Why:** The "sledgehammer" approach - makes the matrix more diagonally dominant

**Key insight:** The stepped `diagGmin` is **NOT** the same as device-level `gmin`. Stepping `diagGmin` from 1e-2 down to the target doesn't affect device models - it only affects the matrix. This is why ngspice also has `new_gmin()` which steps the actual device-level `CKTgmin`.

## ngspice Implementation (cktop.c)

ngspice uses a layered approach for DC operating point convergence:

### 1. Main Flow (`CKTop()`)

```
1. First, try direct Newton iteration (NIiter)
2. If failed and gmin stepping enabled:
   - If numGminSteps == 1: try dynamic_gmin(), then new_gmin()
   - Else: try spice3_gmin() (fixed steps)
3. If still failed and source stepping enabled:
   - If numSrcSteps == 1: try gillespie_src() (adaptive)
   - Else: try spice3_src() (fixed steps)
4. If still failed: try OPtran() (transient OP)
```

### 2. GMIN Stepping Variants

#### 2.1 Dynamic GMIN Stepping (Gillespie's Algorithm)

**Location:** `cktop.c:dynamic_gmin()`

Algorithm:
1. Start with large `CKTdiagGmin = 1e-2 / factor`
2. Add GMIN to **all matrix diagonals** via `LoadGmin()` in sparse matrix
3. Try Newton iteration
4. **Adaptive step size:**
   - If converged quickly (< 25% of maxiters): increase factor (bigger steps)
   - If converged slowly (> 75% of maxiters): decrease factor (smaller steps)
   - If failed: reduce step size dramatically (sqrt(sqrt(factor)))
5. **Save/restore state:** On failure, restore previous `RhsOld` and `CKTstate0`
6. Continue until `CKTdiagGmin <= gtarget` (typically gshunt or original gmin)
7. **Final validation:** Remove GMIN and do one more Newton to confirm solution

Key parameters:
- `CKTgminFactor`: Step factor (default: typically 10)
- `CKTnumGminSteps`: Number of steps (1 = dynamic, >1 = fixed)
- `CKTgshunt`: Target gmin value

#### 2.2 SPICE3 GMIN Stepping

**Location:** `cktop.c:spice3_gmin()`

Simpler fixed-step algorithm:
1. Start with `CKTdiagGmin = gmin * gminFactor^numGminSteps`
2. For each step: try Newton, divide gmin by factor
3. No backtracking or adaptive sizing

#### 2.3 True GMIN Stepping (new_gmin)

**Location:** `cktop.c:new_gmin()`

Similar to dynamic_gmin but steps `CKTgmin` (device model gmin) instead of `CKTdiagGmin` (matrix diagonal). This affects device model conductances directly rather than adding shunts to the matrix.

### 3. Source Stepping Variants

#### 3.1 Gillespie's Source Stepping (Adaptive)

**Location:** `cktop.c:gillespie_src()`

Algorithm:
1. Set `CKTsrcFact = 0` (all sources at zero)
2. Initialize all node voltages and states to zero
3. Try to converge with sources at zero
4. If zero-source fails: combine with 10-step gmin stepping
5. **Adaptive source ramping:**
   - Start with `raise = 0.001` (0.1% steps)
   - If converged quickly: `raise *= 1.5`
   - If converged slowly: `raise *= 0.5`
   - If failed: `raise /= 10` and restore previous state
6. Continue until `CKTsrcFact = 1.0`

How sources use `CKTsrcFact`:
```c
// In vsrcload.c
value = here->VSRCdcValue * ckt->CKTsrcFact;  // DC mode
// or
if (ckt->CKTmode & MODETRANOP)
    value *= ckt->CKTsrcFact;  // TranOP mode
```

#### 3.2 SPICE3 Source Stepping (Fixed)

**Location:** `cktop.c:spice3_src()`

Simple linear stepping:
```c
for (i = 0; i <= numSrcSteps; i++) {
    ckt->CKTsrcFact = (double)i / (double)numSrcSteps;
    converged = NIiter(...);
}
```

### 4. How GMIN is Applied to Matrix

**Location:** `spsmp.c:LoadGmin()`

```c
static void LoadGmin(SMPmatrix *eMatrix, double Gmin) {
    if (Gmin != 0.0) {
        Diag = Matrix->Diag;
        for (I = Matrix->Size; I > 0; I--) {
            if ((diag = Diag[I]) != NULL)
                diag->Real += Gmin;  // Add to ALL diagonals
        }
    }
}
```

This is called during `SMPreorder()` and `SMPluFac()` before matrix factorization.

**Important:** This adds gmin to ALL diagonals including branch current equations!
The code comment says: "use of this routine is not recommended. It is included here simply for compatibility with Spice3."

For a voltage source, the MNA branch equation is `Vp - Vn = Vs` (diagonal = 0).
After LoadGmin: `Vp - Vn + gmin*Ibr = Vs` - physically meaningless but provides regularization.

### Voltage-Only vs Full-Diagonal Stepping

| Approach | What it does | Physical meaning |
|----------|--------------|------------------|
| **Full diagonal** (ngspice LoadGmin) | Adds to ALL diagonals | Regularization hack, not physical |
| **Voltage node only** | Adds to voltage diagonals (1:n_nodes) | Physical: shunt resistor to ground |
| **Device gmin** | In device equations | Physical: minimum conductance |

For Cadnip.jl, voltage-node-only gshunt is more physically correct. The ngspice full-diagonal approach is a sledgehammer that "works" but adds spurious terms to branch current equations.

---

## Cadnip.jl Implementation

### 1. Approach: Poly-Algorithm Nonlinear Solver

Cadnip.jl uses NonlinearSolve.jl's poly-algorithm approach instead of explicit stepping:

**Location:** `src/mna/solve.jl:CedarRobustNLSolve()`

```julia
function CedarRobustNLSolve()
    rmn = RobustMultiNewton()
    algs = (rmn.algs..., LevenbergMarquardt(), PseudoTransient())
    return NonlinearSolvePolyAlgorithm(algs)
end
```

This combines:
1. **RobustMultiNewton** (6 trust region/Newton variants)
2. **LevenbergMarquardt** - Provides GMIN-like regularization via damping:
   - Solves `(J'J + λI)Δx = -J'f` where λ acts like gmin
3. **PseudoTransient** - Continuation method similar to transient OP

### 2. Static GMIN Support

Cadnip.jl supports static GMIN via `MNASpec.gmin`:

**Location:** `src/mna/build.jl:assemble_G()`

```julia
function assemble_G(ctx::MNAContext; gmin::Float64=0.0)
    if gmin > 0
        # Add GMIN from each voltage node to ground
        gmin_I = collect(1:ctx.n_nodes)
        gmin_J = collect(1:ctx.n_nodes)
        gmin_V = fill(gmin, ctx.n_nodes)
        # ... combine with device stamps
    end
end
```

### 3. Alternative Initialization Methods

**Location:** `src/mna/dcop.jl`

For difficult circuits, Cadnip.jl provides:

1. **CedarDCOp** - Standard DC solve with robust nonlinear solver
2. **CedarTranOp** - DC solve in tranop mode
3. **CedarUICOp** - Pseudo-transient relaxation for oscillators:
   - Takes fixed implicit Euler steps instead of Newton
   - Useful for circuits with no stable DC operating point

---

## Comparison

| Feature | ngspice | Cadnip.jl |
|---------|---------|-----------|
| **GMIN Stepping** | Explicit stepping algorithms | LevenbergMarquardt damping |
| **Source Stepping** | Scale sources 0→1 | Not implemented (uses continuation) |
| **Adaptive Step Size** | Yes (Gillespie variants) | Built into LM/trust region |
| **State Save/Restore** | Manual (OldRhsOld arrays) | Handled by NonlinearSolve |
| **Matrix GMIN** | Added to all diagonals | Only voltage nodes |
| **Fallback Chain** | gmin→source→transientOP | polyalgorithm + CedarUICOp |

### Advantages of ngspice Approach

1. **Explicit control** over stepping behavior
2. **Source stepping** provides physical intuition (ramp up supplies)
3. **Well-tested** on decades of SPICE circuits
4. **Predictable** behavior for debugging

### Advantages of Cadnip.jl Approach

1. **Simpler code** - delegates to NonlinearSolve.jl
2. **Modern algorithms** - trust region, LM, pseudo-transient in one package
3. **Automatic fallback** via poly-algorithm
4. **No need for srcFact** infrastructure in device models

---

## Issues with Current Cadnip.jl Implementation

### Conflated GMIN Concepts

Cadnip.jl currently has a single `MNASpec.gmin` that's used for:
1. Device-level gmin in VA models (via `$simparam("gmin")`)
2. Matrix diagonal shunts in `assemble_G(ctx; gmin=...)`

This conflates two different concepts:
- **Device gmin** should be small (1e-12) and always present
- **Matrix gshunt** is optional and used for floating nodes

### Missing Features

1. **No device-level gmin by default** - VA models use `$simparam("gmin")` but it's not automatically applied to built-in devices like Diode
2. **No source stepping** - LM provides some regularization but doesn't scale sources 0→1
3. **No explicit GMIN stepping** - Relies on LM damping which works differently
4. **Matrix gmin adds to wrong locations** - `assemble_G()` only adds to voltage nodes (1:n_nodes), but ngspice's `LoadGmin()` adds to ALL diagonals including current variables

### Recommended Changes

1. **Separate gmin and gshunt** in MNASpec:
   ```julia
   struct MNASpec
       gmin::Float64 = 1e-12    # Device-level (used in device models)
       gshunt::Float64 = 0.0    # Node-to-ground shunt (optional)
   end
   ```

2. **Always apply gmin in device models** - Add `+ spec.gmin` to junction conductances in built-in Diode, BJT, etc.

3. **Apply gshunt correctly** - Only to voltage node diagonals, not current variables

4. **Consider adding srcFact** - For difficult circuits, source stepping can be more robust than LM

---

## Implementation Notes for Adding ngspice-Style Stepping

### Prerequisites: Proper gshunt Implementation

Before implementing GMIN stepping, Cadnip.jl needs a proper gshunt:

```julia
# In MNASpec:
struct MNASpec
    gmin::Float64 = 1e-12    # Device-level (added to device conductances)
    gshunt::Float64 = 0.0    # Node-to-ground shunt (added to G matrix diagonals)
end

# In assemble_G or fast_rebuild!:
# Add gshunt ONLY to voltage node diagonals (1:n_nodes), NOT to current variables
for i in 1:ctx.n_nodes
    G[i, i] += spec.gshunt
end
```

Key distinctions:
- **gmin** (1e-12): Passed to device models, used in `g_diode = Is/Vt * exp(...) + gmin`
- **gshunt** (0): Added to voltage node diagonals to create a path to ground
- During stepping, only **gshunt** is varied; **gmin** stays constant

### GMIN Stepping Implementation

```julia
"""
    gmin_stepping(builder, params, spec, ctx;
                  maxiters=100, gmin_factor=10.0, num_steps=10)

GMIN stepping for DC convergence. Steps gshunt from large value down to spec.gshunt.
"""
function gmin_stepping(builder, params, spec, ctx;
                       maxiters=100, gmin_factor=10.0, num_steps=10)
    n = system_size(ctx)
    u = zeros(n)
    u_saved = zeros(n)

    # Start with large gshunt
    gtarget = max(spec.gmin, spec.gshunt)
    gshunt_current = gtarget
    for _ in 1:num_steps
        gshunt_current *= gmin_factor
    end

    # Step down to target
    for step in 1:num_steps+1
        spec_step = MNASpec(spec; gshunt=gshunt_current)

        # Try Newton at this gshunt level
        converged = false
        try
            u, converged = _dc_newton_compiled(builder, params, spec_step, ctx, u;
                                                abstol=1e-10, maxiters=maxiters)
        catch
        end

        if !converged
            # Backtrack: restore previous state and try smaller step
            u .= u_saved
            gmin_factor = sqrt(gmin_factor)  # Smaller steps
            gshunt_current = gshunt_saved / gmin_factor
        else
            # Save state for potential backtrack
            u_saved .= u
            gshunt_saved = gshunt_current
            gshunt_current /= gmin_factor
        end

        if gshunt_current <= gtarget
            break
        end
    end

    # Final validation: solve with target gshunt (usually 0)
    spec_final = MNASpec(spec; gshunt=spec.gshunt)
    u, converged = _dc_newton_compiled(builder, params, spec_final, ctx, u;
                                        abstol=1e-10, maxiters=maxiters)

    return u, converged
end
```

### Source Stepping Implementation

```julia
"""
    source_stepping(builder, params, spec, ctx;
                    maxiters=100, num_steps=10)

Source stepping: ramp all independent sources from 0 to full value.
Requires srcFact support in source stamps.
"""
function source_stepping(builder, params, spec, ctx;
                         maxiters=100, raise=0.001)
    n = system_size(ctx)
    u = zeros(n)
    u_saved = zeros(n)

    srcFact = 0.0
    ConvFact = 0.0

    while ConvFact < 1.0
        spec_step = MNASpec(spec; srcFact=srcFact)

        converged = false
        iters = 0
        try
            u, converged, iters = _dc_newton_compiled_with_iters(
                builder, params, spec_step, ctx, u; maxiters=maxiters)
        catch
        end

        if converged
            ConvFact = srcFact
            u_saved .= u

            # Adaptive step size based on iteration count
            if iters <= maxiters ÷ 4
                raise *= 1.5
            elseif iters > 3 * maxiters ÷ 4
                raise *= 0.5
            end

            srcFact = ConvFact + raise
            srcFact = min(srcFact, 1.0)
        else
            # Backtrack
            if srcFact - ConvFact < 1e-8
                break  # Can't make progress
            end
            raise /= 10
            srcFact = ConvFact
            u .= u_saved
        end
    end

    return u, ConvFact == 1.0
end
```

### Key Implementation Details

1. **gshunt must be separate from gmin** - Device models should always use `spec.gmin` (typically 1e-12), while `spec.gshunt` controls the matrix diagonal shunts

2. **gshunt applies only to voltage nodes** - Don't add gshunt to current variable rows (branch currents in voltage sources, inductors, etc.)

3. **srcFact requires source modification** - Each independent source must multiply its value by srcFact:
   ```julia
   function stamp!(src::VoltageSource, ctx, p, n)
       v = src.v * ctx.spec.srcFact  # Scale by srcFact
       # ... rest of stamping
   end
   ```

4. **State save/restore is essential** - Both stepping algorithms need to backtrack on failure

5. **Final validation without helpers** - After stepping completes, do one more solve with gshunt=0 and srcFact=1 to confirm the solution works for the actual circuit

---

## Integration with CedarDCOp

The stepping algorithms should be integrated into the existing `CedarDCOp` initialization framework rather than as standalone functions. This keeps all DC initialization logic in one place.

### Proposed CedarDCOp Enhancement

```julia
"""
    CedarDCOp with homotopy support

Fields:
- `abstol`: Newton convergence tolerance
- `maxiters`: Max Newton iterations per step
- `nlsolve`: Base nonlinear solver (fallback)
- `use_shampine`: Use ShampineCollocationInit after DC solve
- `gmin_stepping`: Enable GMIN stepping homotopy
- `source_stepping`: Enable source stepping homotopy
- `gmin_factor`: Factor for GMIN stepping (default: 10)
- `num_gmin_steps`: Number of GMIN steps (default: 10)
"""
struct CedarDCOp{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    maxiters::Int
    nlsolve::NLSOLVE
    use_shampine::Bool
    # Homotopy options
    gmin_stepping::Bool
    source_stepping::Bool
    gmin_factor::Float64
    num_gmin_steps::Int
end

CedarDCOp(; abstol=1e-9, maxiters=500, nlsolve=CedarRobustNLSolve(),
          use_shampine=false, gmin_stepping=false, source_stepping=false,
          gmin_factor=10.0, num_gmin_steps=10) =
    CedarDCOp(abstol, maxiters, nlsolve, use_shampine,
              gmin_stepping, source_stepping, gmin_factor, num_gmin_steps)
```

### Homotopy Integration Flow

```julia
function SciMLBase.initialize_dae!(integrator, alg::CedarDCOp)
    ws = integrator.sol.prob.p::EvalWorkspace
    u0 = zeros(length(integrator.u))

    converged = false

    # 1. First try direct Newton (like ngspice NIiter)
    u0, converged = _dc_newton_compiled(ws, u0; abstol=alg.abstol, maxiters=alg.maxiters)

    if !converged && alg.gmin_stepping
        # 2. Try GMIN stepping (homotopy on gshunt)
        u0, converged = _gmin_stepping(ws, u0;
            abstol=alg.abstol, maxiters=alg.maxiters,
            factor=alg.gmin_factor, num_steps=alg.num_gmin_steps)
    end

    if !converged && alg.source_stepping
        # 3. Try source stepping (homotopy on srcFact)
        u0, converged = _source_stepping(ws, u0;
            abstol=alg.abstol, maxiters=alg.maxiters)
    end

    if !converged
        # 4. Fall back to poly-algorithm (LM, PseudoTransient)
        u0, converged = _dc_newton_compiled(ws, u0;
            abstol=alg.abstol, maxiters=alg.maxiters,
            nlsolve=alg.nlsolve)
    end

    integrator.u .= u0
    # ... rest of initialization
end
```

### Required Infrastructure Changes

1. **Add gshunt to MNASpec** (separate from gmin):
   ```julia
   struct MNASpec{T<:Real}
       temp::Float64 = 27.0
       mode::Symbol = :tran
       time::T = 0.0
       gmin::Float64 = 1e-12      # Device-level (used in device models)
       gshunt::Float64 = 0.0      # Node-to-ground shunt (for stepping)
       srcFact::Float64 = 1.0     # Source scaling factor (for stepping)
       # ... other fields
   end
   ```

2. **Modify fast_rebuild! to apply gshunt**:
   ```julia
   function fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)
       # ... existing stamping code ...

       # Apply gshunt to voltage node diagonals
       gshunt = ws.structure.spec.gshunt
       if gshunt > 0
           G = ws.structure.G
           for i in 1:ws.structure.n_nodes
               G[i, i] += gshunt
           end
       end
   end
   ```

3. **Modify source stamps to use srcFact**:
   ```julia
   function stamp!(src::VoltageSource, ctx::MNAContext, p::Int, n::Int)
       v = src.v * ctx.spec.srcFact  # Scale by srcFact
       # ... existing stamping
   end
   ```

4. **Add with_gshunt and with_srcfact helpers**:
   ```julia
   with_gshunt(spec::MNASpec, gshunt::Real) = MNASpec(spec; gshunt=Float64(gshunt))
   with_srcfact(spec::MNASpec, srcFact::Real) = MNASpec(spec; srcFact=Float64(srcFact))
   ```

### Why Integrate with CedarDCOp?

1. **Single entry point** - All DC initialization goes through CedarDCOp
2. **Consistent state management** - EvalWorkspace already tracks everything needed
3. **Composable** - Can combine gmin stepping, source stepping, and poly-algorithm fallback
4. **Debugging** - Single place to add logging/diagnostics for convergence issues
5. **User control** - Users can enable/disable homotopy methods via kwargs
