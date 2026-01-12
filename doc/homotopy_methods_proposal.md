# Homotopy Methods and Convergence Improvement Proposal

## Executive Summary

This document analyzes homotopy methods used in ngspice and VACASK, compares them with Cadnip.jl's current implementation, and proposes improvements for solving challenging circuits like ring oscillators and astable multivibrators.

## 0. Key Questions and Answers

### Q1: Is the `gmin_stamp` in vasim.jl a hack that should be removed?

**Context**: In `vasim.jl:1789-1794`, GMIN (1e-12 S) is stamped from internal VA nodes to ground:

```julia
# vasim.jl:1791-1794
if $int_param != 0
    CedarSim.MNA.stamp_G!(ctx, $int_param, $int_param, 1e-12)
end
```

**Analysis**: This is actually correct SPICE behavior for *internal* nodes:
- Internal nodes (created by the VA model, not user-visible) often have no DC path
- Example: Excess phase modeling nodes, noise correlation nodes
- Without GMIN, these create singular Jacobians

**Recommendation**: Keep it, but consider making it spec-controlled:
```julia
CedarSim.MNA.stamp_G!(ctx, $int_param, $int_param, _mna_spec_.gmin)
```

This is different from the **device-level GMIN** ngspice adds to junction conductances. That should be done in the device model itself via `$simparam("gmin")`.

### Q2: Does our GMIN/source stepping have advantages over SciML methods?

**Short answer**: They solve different problems at different levels.

| Level | ngspice/SPICE | SciML/Cadnip.jl | Equivalent? |
|-------|---------------|-----------------|-------------|
| **Device-level GMIN** | `gd += gmin` in device code | `$simparam("gmin")` in VA models | Yes, VA models can access gmin |
| **Matrix-level GMIN** | `CKTdiagGmin` added to diagonal | `assemble_G(...; gmin=1e-12)` | Yes, but unused currently |
| **GMIN stepping** | Ramp GMIN from 0.01→1e-12 | `LevenbergMarquardt` damping | Similar effect, different mechanism |
| **Source stepping** | Ramp srcFact from 0→1 | `PseudoTransient` continuation | Similar but not identical |

**Key insight**: LevenbergMarquardt's `(J + λI)⁻¹` regularization and ngspice's GMIN stepping `(J + gmin·I)⁻¹` are mathematically similar. LM actually provides more sophisticated adaptive damping.

**Advantages of SciML approach**:
1. **Automatic adaptation**: LM adjusts damping based on trust region, not fixed geometric progression
2. **Multiple algorithm fallback**: RobustMultiNewton tries 6 variants before LM and PseudoTransient
3. **No circuit modification**: Doesn't require rebuilding circuit with different gmin values

**Advantages of SPICE approach**:
1. **Physics-based**: Device-level GMIN maintains physical interpretation (leakage current)
2. **Separate control**: Can tune GMIN stepping independently of Newton damping
3. **Proven reliability**: 40+ years of production use

**Recommendation**: Leverage SciML solvers, but add device-level GMIN support for models that expect it (via `$simparam("gmin")`).

### Q3: Where should pnjlim be implemented?

**ngspice implementation**: `DEVpnjlim()` is a **C helper function** in `devsup.c` called by device load functions:

```c
// bjtload.c example
vbe = DEVpnjlim(vbe, *(ckt->CKTstate0 + here->BJTvbe),
                vt, here->BJTtVcrit, &Check);
```

This limits voltage **before** evaluating the exponential - it's a Newton step limiter, not a math function.

**Verilog-A has `$limit`**: The `$limit` system function is the VA equivalent:
```verilog
Vd = $limit(V(anode, cathode), "pnjlim", Vt, Vcrit);
```

**Cadnip.jl current state**:
- `$limit` is parsed but **not implemented** (returns voltage unchanged in `vasim.jl:662-668`)
- `limexp` is a reserved keyword but **not implemented**

**Implementation levels**:

| Function | Level | Where to implement |
|----------|-------|-------------------|
| `DEVpnjlim()` | SPICE device code | N/A - we use VA not SPICE models |
| `$limit(V, "pnjlim", ...)` | Verilog-A | `vasim.jl` - implement properly |
| `limexp(x)` | Verilog-A | `va_env.jl` - simple function |
| Backtracking linesearch | SciML solver | Already have via NonlinearSolve.jl |

**Recommendation**:

1. **Implement `limexp()` in `va_env.jl`** - trivial:
   ```julia
   limexp(x) = x < 90.0 ? exp(x) : exp(90.0) * (1.0 + x - 90.0)
   ```

2. **Implement `$limit()` properly in `vasim.jl`** - medium effort:
   ```julia
   # For "pnjlim" limiter:
   function va_pnjlim(vnew, vold, vt, vcrit)
       if vnew > vcrit && abs(vnew - vold) > 2*vt
           if vold > 0
               arg = (vnew - vold) / vt
               vnew = arg > 0 ? vold + vt*(2 + log(arg - 2)) : vold - vt*(2 + log(2 - arg))
           else
               vnew = vt * log(vnew / vt)
           end
       end
       return vnew
   end
   ```

3. **Let SciML handle global limiting** - The backtracking linesearch in NonlinearSolve already provides step limiting; `$limit` is for per-variable control within device models.

**Not needed for SciML**: Contributing pnjlim to SciML is not appropriate - it's a circuit-domain heuristic, not a general numerical technique. SciML's trust regions and line searches serve the same purpose more generally.

### Q4: Should the `gmin` kwarg in `assemble_G` be connected to `MNASpec.gmin`?

**Short answer**: No - ngspice uses separate `gmin` and `gshunt` parameters for exactly this reason.

**ngspice terminology** (from `cktdefs.h` and `cktop.c`):
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CKTgmin` | 1e-12 | Device-level: `gd += gmin` inside device models |
| `CKTgshunt` | 0 | Matrix diagonal: permanent shunt to ground on all nodes |
| `CKTdiagGmin` | (homotopy) | Temporary: starts at 0.01, steps down to `MAX(gmin, gshunt)` |

**GSHUNT** (`assemble_G(...; gmin=x)` - should rename to `gshunt`):
```julia
# Changes the problem being solved:
F(x) = (G + gshunt·I) · x - b
# Every node has a shunt resistor to ground
# The equilibrium point is DIFFERENT from the original circuit
```

**LevenbergMarquardt damping** (in NonlinearSolve):
```julia
# Keeps the original problem:
F(x) = G · x - b
# Only the Newton step calculation changes:
Δx = (J'J + λI)⁻¹ J' F   # λ regularizes Jacobian inversion only
# If it converges, you get the TRUE circuit equilibrium
```

**Critical distinction**:
| Aspect | GSHUNT (matrix diagonal) | LM Damping |
|--------|--------------------------|------------|
| Changes residual | Yes | No |
| Changes equilibrium | Yes (pulls nodes to 0V) | No |
| Oscillators (no DC solution) | Creates fake equilibrium | Can't help - no solution exists |
| Singular Jacobian | Regularizes problem | Regularizes inversion |

**Why this matters for oscillators**: Ring oscillators have no DC equilibrium - LM can iterate forever because there's no solution to converge to. GSHUNT creates a fake (but stable) equilibrium by adding shunt resistors.

**Recommendation**:
1. **Rename `assemble_G`'s `gmin` kwarg to `gshunt`** to match ngspice terminology
2. **Keep `MNASpec.gmin`** for device-level GMIN (accessed via `$simparam("gmin")`)
3. **Add `MNASpec.gshunt`** for matrix diagonal (default 0, like ngspice):
   ```julia
   Base.@kwdef struct MNASpec{T<:Real}
       gmin::Float64 = 1e-12       # Device-level GMIN (for VA models)
       gshunt::Float64 = 0.0       # Matrix diagonal shunt (ngspice: gshunt)
       # ...
   end
   ```
4. **Use gshunt for oscillator initialization** when no DC solution exists

## 1. ngspice Default Tolerances and Homotopy Methods

### 1.1 Default Tolerance Values (from `cktntask.c`)

| Parameter | ngspice Default | Cadnip.jl Current | Notes |
|-----------|-----------------|-------------------|-------|
| `abstol` | 1e-12 | 1e-12 | Current tolerance (Amperes) |
| `reltol` | 1e-3 | 1e-3 | Relative tolerance |
| `vntol` | 1e-6 | 1e-6 | Voltage tolerance (Volts) |
| `gmin` | 1e-12 | 1e-12 | Device-level minimum conductance (Siemens) |
| `gshunt` | 0 | N/A | Matrix diagonal shunt conductance (Siemens) |
| `chgtol` | 1e-14 | N/A | Charge tolerance |
| `trtol` | 7 | N/A | Truncation error factor |
| `gminFactor` | 10 | N/A | GMIN stepping factor |
| `pivotAbsTol` | 1e-13 | N/A | Pivot absolute tolerance |
| `pivotRelTol` | 1e-3 | N/A | Pivot relative tolerance |
| `lteReltol` | 1e-3 | N/A | Local truncation error relative |
| `lteAbstol` | 1e-6 | N/A | Local truncation error absolute |

### 1.2 ngspice Homotopy Methods (from `cktop.c`)

ngspice implements a **sequential fallback chain** in `CKTop()`:

```
1. Direct Newton (NIiter)
   ↓ (if failed)
2. GMIN Stepping (diagGmin: 0.01 → MAX(gmin, gshunt))
   ↓ (if failed)
3. Source Stepping (srcFact: 0 → 1)
```

**Key clarification**: There is no separate "gshunt stepping". The `gshunt` parameter is simply the **target floor** for GMIN stepping. When `gshunt > 0`, GMIN stepping ramps `diagGmin` down to `gshunt` instead of to 0.

**Defaults** (from `cktntask.c`):
- `numGminSteps = 1` → use `dynamic_gmin` (adaptive)
- `numSrcSteps = 1` → use `gillespie_src` (adaptive)
- `gmin = 1e-12` (device-level)
- `gshunt = 0` (matrix diagonal target)

#### 1.2.1 GMIN Stepping (diagGmin homotopy)

This is a **matrix diagonal regularization homotopy** — it does NOT change device-level gmin.

**Where diagGmin is applied** (from `spsmp.c:LoadGmin`):
```c
// Called before matrix factorization in SMPluFac() and SMPreorder()
if (Gmin != 0.0) {
    for (I = Matrix->Size; I > 0; I--) {
        if ((diag = Diag[I]) != NULL)
            diag->Real += Gmin;  // Add diagGmin to matrix diagonal
    }
}
```

**Three separate things**:
| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `CKTgmin` | 1e-12 | Device-level: `gd += gmin` in device load functions |
| `CKTdiagGmin` | 0 | Matrix diagonal: added during LU factorization |
| `CKTgshunt` | 0 | Target floor for diagGmin after stepping completes |

**During GMIN stepping**: `diagGmin` ramps from 0.01 → MAX(gmin, gshunt)
**After stepping**: `diagGmin = gshunt` (default 0, so no matrix diagonal shunt)
**Always**: Native devices use `gmin` (1e-12) in their load functions

- **Homotopy parameter**: `diagGmin` ∈ [0.01, MAX(gmin, gshunt)]
- **Effect**: Adds `diagGmin` to every diagonal element of G matrix via `LoadGmin()`
- **Physics**: Equivalent to shunt resistor `1/diagGmin` from each node to ground

Two variants:
- `dynamic_gmin`: Adaptive stepping with factor adjustment based on iteration count
- `spice3_gmin`: Fixed geometric progression (divide by `gminFactor` each step)

```
Algorithm (dynamic_gmin):
1. Start with diagGmin = 0.01 S (100Ω shunt to ground)
2. Solve at current diagGmin level
3. If converged:
   - Fast convergence (iters < maxiters/4): increase factor
   - Slow convergence (iters > 3*maxiters/4): decrease factor
   - diagGmin = diagGmin / factor
4. If failed:
   - Reduce stepping factor: factor = sqrt(sqrt(factor))
   - Restore previous solution, try smaller step
5. Continue until diagGmin reaches MAX(gmin, gshunt)
6. Final solve at target diagGmin
```

#### 1.2.2 Source Stepping (srcFact homotopy)

This is a **parameter continuation homotopy** — it scales the RHS (b vector).

**Where srcFact is applied** (from device load functions):
```c
// isrcload.c:53 - current sources
value = here->ISRCdcValue * ckt->CKTsrcFact;

// vsrcload.c - voltage sources
value = here->VSRCdcValue * ckt->CKTsrcFact;
```

- **Homotopy parameter**: `srcFact` ∈ [0, 1]
- **Effect**: All source values multiplied by `srcFact` (scales b vector contributions)
- **Physics**: Gradually "turns on" the circuit from zero-source state

Two variants:
- `gillespie_src`: Adaptive stepping based on convergence speed
- `spice3_src`: Fixed stepping

```
Algorithm (gillespie_src):
1. Set srcFact = 0 (all sources at 0V/0A)
2. Solve with zero sources (may use GMIN stepping too if this fails)
3. Gradually increase srcFact toward 1.0:
   - Fast convergence: raise *= 1.5
   - Slow convergence: raise *= 0.5
   - Failed: reduce raise by 10x, restore previous solution
4. Continue until srcFact = 1.0 (full source values)
```

#### 1.2.3 Comparison of Homotopy Methods

| Aspect | GMIN Stepping | Source Stepping | PseudoTransient |
|--------|---------------|-----------------|-----------------|
| Homotopy parameter | diagGmin ∈ [0.01, 1e-12] | srcFact ∈ [0, 1] | pseudo-time Δt |
| What changes | Matrix diagonal (shunts) | RHS (source values) | Adds C·dx/dt term |
| Initial state | All nodes shunted to 0V | All sources = 0 | Initial guess |
| Path to solution | Remove shunts gradually | Turn on sources gradually | Time evolution |
| For oscillators | Creates fake equilibrium | srcFact=0 gives trivial x=0 | May oscillate |
| Equilibrium found | Shifted by residual gshunt | True (if srcFact reaches 1) | True (if steady state exists) |

### 1.3 GMIN Application in Devices

**Key insight**: In ngspice, GMIN is added directly to the device conductance, not just the diagonal:

```c
// dioload.c lines 282-287
gd = gd + ckt->CKTgmin;
cd = cd + ckt->CKTgmin*vd;
```

This is the **companion model** approach:
- Device current: `I = I_device(V) + gmin * V`
- Device conductance: `G = G_device(V) + gmin`

**vs. Diagonal GMIN**: Adding GMIN only to the matrix diagonal is a cruder regularization that doesn't follow the physics as closely.

## 2. VACASK Approach

From the benchmarks/vacask README, VACASK is a full Verilog-A simulator with:

- Residual tolerance check (option `nr_residualcheck`)
- Continuation bypass (option `nr_contbypass`) - skips evaluation in first iteration when continuing from nearby solution
- Inactive element bypass (option `nr_bypass`)

VACASK focuses on performance optimization rather than convergence robustness, assuming converged points from continuation.

## 3. Current Cadnip.jl Implementation Analysis

### 3.1 Current Solver Stack (`solve.jl`, `dcop.jl`)

```julia
CedarRobustNLSolve() = NonlinearSolvePolyAlgorithm((
    RobustMultiNewton.algs...,  # 6 trust region variants
    LevenbergMarquardt(),       # Jacobian regularization (NOT same as GMIN stepping)
    PseudoTransient()           # Time-like continuation (NOT same as source stepping)
))
```

**Mapping to ngspice** (with important distinctions):

| ngspice Method | Cadnip.jl | Equivalent? | Key Difference |
|----------------|-----------|-------------|----------------|
| Newton-Raphson | `NewtonRaphson()` | ✓ Yes | Same algorithm |
| Junction limiting (pnjlim) | `BackTracking()` linesearch | ~ Partial | Backtracking is global, pnjlim is per-device |
| **GMIN stepping** | `LevenbergMarquardt()` | **✗ No** | LM regularizes Jacobian only, doesn't change residual |
| **Source stepping** | `PseudoTransient()` | **✗ No** | Different homotopy parameters (see §1.2.3) |
| diagGmin homotopy | Not implemented | - | Need explicit GMIN stepping |
| srcFact homotopy | Not implemented | - | Need source scaling parameter |

**Critical distinction** (see Q4 above):
- **LevenbergMarquardt**: Step = `(J'J + λI)⁻¹ J' F` — regularizes *inversion*, residual unchanged
- **GMIN stepping**: Residual = `(G + diagGmin·I)·x - b` — changes the *problem* being solved

For oscillators with no DC equilibrium, LM cannot help because there's no solution to converge to.
GMIN stepping creates a fake equilibrium by adding shunt resistors.

### 3.2 GSHUNT in Cadnip.jl (`build.jl`)

Currently, matrix diagonal shunt is available but **not connected** to MNASpec:

```julia
function assemble_G(ctx::MNAContext; gmin::Float64=0.0)  # Should rename to gshunt
    if gmin > 0
        # Add shunt from each voltage node to ground (ngspice calls this gshunt)
        gmin_I = collect(1:ctx.n_nodes)
        gmin_J = collect(1:ctx.n_nodes)
        gmin_V = fill(gmin, ctx.n_nodes)
        ...
    end
end
```

**Two different things** (ngspice terminology):
1. **gmin** (device-level): Added inside device models to junction conductance: `gd += gmin`
   - Available via `$simparam("gmin")` in VA models
   - Correct for physics (models leakage current)

2. **gshunt** (matrix diagonal): Added to G matrix diagonal during assembly
   - Currently the `gmin` kwarg in `assemble_G` (should rename)
   - Used for homotopy stepping and oscillator initialization
   - Not connected to MNASpec yet

### 3.3 Charge Formulation vs Companion Models

**ngspice approach** (Companion Model):
- Capacitors: `I = C * (V - V_prev) / dt`
- Charge: Implicit in time-stepping
- Mass matrix: Not applicable (fully integrated into G)

**Cadnip.jl approach** (Charge States):
- For voltage-dependent capacitors: `q` is an explicit state variable
- Constraint: `q = Q(V)` enforced via Newton
- Mass matrix: Constant (entries of 1 for charge states)

**Implications**:
1. Cadnip.jl's charge formulation creates additional algebraic constraints
2. These constraints can increase problem stiffness
3. The condition number may be worse due to mixed scaling between voltages and charges

## 4. Identified Issues

### 4.1 Ring Oscillator Challenges

From `doc/ring_oscillator_investigation.md`:
- System size: 371 unknowns, 2846 G nonzeros, 137 zero diagonal entries
- C rank: 72/371 (highly rank deficient) - only 72 differential variables, 299 algebraic
- Condition number: ~6.57e18 even with GMIN
- No stable DC equilibrium (oscillators don't have one)

### 4.2 Astable Multivibrator Challenges

From test files:
- BJT internal nodes (excess phase modeling) create exponential overflow
- `exp(Vbe/Vt)` terms blow up when internal nodes start at 0V
- Symmetric DC point causes both BJTs to saturate (metastable)

### 4.3 Comparison of Failure Modes

**DC Initialization Issues**:

| Issue | ngspice Handling | Cadnip.jl Current |
|-------|------------------|-------------------|
| Singular Jacobian | GMIN to devices + diagGmin | GMIN to gmin_stamp only |
| No DC equilibrium | diagGmin stepping + source stepping | PseudoTransient (not equivalent) |
| Metastable states | NODESET hints | Not implemented |

**Runtime (Transient) Issues**:

| Issue | ngspice Handling | Cadnip.jl Current |
|-------|------------------|-------------------|
| Exponential overflow | Per-iteration limiting (pnjlim/fetlim) | **Not implemented** |
| Large Newton steps | Per-device voltage limiting | Backtracking linesearch (global only) |
| Non-convergence | Timestep cut by 8x | DiffEq adaptive stepping |

**Key gap**: ngspice's per-iteration limiting is device-aware (limits Vbe, Vbc, Vds separately).
Our backtracking linesearch is global and doesn't know which voltages are junction voltages.

## 5. Proposed Improvements

### 5.1 Implement True GMIN Stepping (Priority: High)

Add device-level GMIN similar to ngspice:

```julia
# In VA stamp code generation or device models
function stamp_diode!(ctx, p, n, Is, Vt; gmin=ctx.spec.gmin)
    Vd = voltage(p) - voltage(n)
    Id = Is * (exp(clamp(Vd/Vt, -40, 40)) - 1)
    Gd = Is/Vt * exp(clamp(Vd/Vt, -40, 40))

    # Add GMIN to device conductance (ngspice style)
    Id_total = Id + gmin * Vd
    Gd_total = Gd + gmin

    stamp_current!(ctx, p, n, Id_total, Gd_total)
end
```

**Algorithm**: Adaptive GMIN stepping

```julia
function gmin_stepping(solve_fn, u0;
                       gmin_start=1e-2,
                       gmin_target=1e-12,
                       factor=10.0)
    gmin = gmin_start
    u = u0

    while gmin > gmin_target
        result = solve_fn(u, gmin)
        if result.converged
            u = result.u
            if result.iterations < max_iters/4
                factor = min(factor * 1.5, 10.0)  # Speed up
            elseif result.iterations > 3*max_iters/4
                factor = max(sqrt(factor), 1.1)    # Slow down
            end
            gmin = max(gmin / factor, gmin_target)
        else
            factor = sqrt(sqrt(factor))  # Reduce step
            if factor < 1.00005
                return (converged=false, u=u)
            end
        end
    end

    # Final solve without homotopy
    return solve_fn(u, gmin_target)
end
```

### 5.2 Implement Source Stepping (Priority: High)

Add `srcFact` parameter to circuit builders:

```julia
struct MNASpec{T<:Real}
    ...
    srcFact::Float64 = 1.0  # Source scaling factor for source stepping
end

# In voltage source stamping
function stamp!(vs::VoltageSource, ctx, p, n)
    v = vs.value * ctx.spec.srcFact  # Scale by srcFact
    stamp_voltage_source!(ctx, p, n, v, vs.name)
end
```

**Algorithm**: Adaptive source stepping (Gillespie style)

```julia
function source_stepping(solve_fn, u0; raise=0.001)
    srcFact = 0.0
    u = u0

    # First solve with sources at 0
    result = solve_fn(u, srcFact)
    if !result.converged
        # Try GMIN stepping first
        result = gmin_stepping(u -> solve_fn(u, 0.0), u0)
        if !result.converged
            return (converged=false, u=u)
        end
        u = result.u
    end

    while srcFact < 1.0
        srcFact = min(srcFact + raise, 1.0)
        result = solve_fn(u, srcFact)

        if result.converged
            u = result.u
            if result.iterations < max_iters/4
                raise *= 1.5  # Speed up
            elseif result.iterations > 3*max_iters/4
                raise *= 0.5  # Slow down
            end
        else
            raise /= 10
            if (srcFact - last_converged) < 1e-8
                break
            end
        end
    end

    return (converged = srcFact >= 1.0, u=u, srcFact=srcFact)
end
```

### 5.3 Improve Charge State Scaling (Priority: High)

**Why scaling matters**: MNA with charge states creates ill-conditioned systems because
residuals span many orders of magnitude:

| Equation Type | Residual Units | Typical Magnitude |
|---------------|----------------|-------------------|
| KCL (node) | Amperes | 1e-3 to 1 A |
| Voltage source | Volts | ~1 V |
| Charge constraint `q = Q(V)` | Coulombs | ~1e-12 C |
| If `V = dV/dt` appears | V/s | ~1e9 V/s (for ns timescales!) |

**VADistiller experience**: During MOS1 model compilation, the equation `v_x = dv_y/dt`
appeared when the compiler restructured charge conservation equations. With dt ~ 1 ns,
dV/dt ~ 1 GV/s, causing residuals 12+ orders of magnitude larger than current residuals.
Scaling resolved the convergence problems.

**Two scaling issues**:

1. **Charge state magnitude**: q ~ 1e-12 C vs V ~ 1 V
   - Solution: Use `q_scaled = q / CHARGE_SCALE` where CHARGE_SCALE ~ 1e-12

2. **Time derivative magnitude**: dV/dt ~ V/dt ~ 1V / 1ns = 1e9 V/s
   - Solution: Scale time-related equations or use proper companion model formulation

**Proposal**: Scale charge states to similar magnitude as voltages

```julia
# During stamp_charge_state!
# Instead of q in Coulombs, use q_scaled = q / CHARGE_SCALE
const CHARGE_SCALE = 1e-12  # Scale factor to bring charges to ~O(1)

function stamp_charge_state!(ctx, p, n, q_fn, x, charge_name)
    q_idx = alloc_charge!(ctx, charge_name, p, n)

    # State variable: q_scaled = q / CHARGE_SCALE (dimensionless, ~O(1))
    # Differential: d(q_scaled)/dt = I / CHARGE_SCALE
    # Constraint: q_scaled = Q(V) / CHARGE_SCALE

    # KCL: I = dq/dt = CHARGE_SCALE * d(q_scaled)/dt
    # Stamps into C matrix with scale factor
    if p != 0
        stamp_C!(ctx, p, q_idx, CHARGE_SCALE)  # I enters KCL at node p
    end
    if n != 0
        stamp_C!(ctx, n, q_idx, -CHARGE_SCALE)  # I leaves KCL at node n
    end

    # Constraint row: q_scaled - Q(V)/CHARGE_SCALE = 0
    # This residual is now O(1), not O(1e-12)
    stamp_G!(ctx, q_idx, q_idx, 1.0)
    stamp_G!(ctx, q_idx, p, -result.dq_dVp / CHARGE_SCALE)
    stamp_G!(ctx, q_idx, n, -result.dq_dVn / CHARGE_SCALE)

    b_constraint = (-result.q + result.dq_dVp * Vp + result.dq_dVn * Vn) / CHARGE_SCALE
    stamp_b!(ctx, q_idx, b_constraint)
end
```

**Alternative: Equation scaling** (as used in VADistiller)
Instead of scaling state variables, scale entire rows of the Jacobian:
```julia
# After assembling J, scale rows by equation type
for eq in charge_equations
    J[eq, :] ./= CHARGE_SCALE
    b[eq] /= CHARGE_SCALE
end
```
This is mathematically equivalent but may be simpler to implement.

#### 5.3.1 Jacobian Analysis: Ring Oscillator with MOS1

Running Jacobian analysis on the 3-stage ring oscillator with sp_mos1 MOSFETs:

**System structure** (5 unknowns: 4 node voltages + 1 branch current):
- G matrix condition number: **3.63e+11** (extremely ill-conditioned!)
- Min singular value: **2.76e-12** (nearly singular)
- C matrix has 16 nonzeros (MOS junction capacitances + load caps)

**Row-by-row analysis**:

| Node | |G| (row norm) | |C| (row norm) | |G|/|C| | Problem |
|------|---------------|---------------|--------|---------|
| in1  | 5.09e-06 | 1.00e-14 | 5.09e+08 | Low G |
| out1 | **4.12e-12** | 1.00e-14 | 4.12e+02 | **G ≈ gmin** |
| vdd  | 1.00e+00 | 0 | ∞ | Voltage source |
| out2 | **1.83e-11** | 1.00e-14 | 1.83e+03 | **G ≈ gmin** |
| I_vdd| 1.00e+00 | 0 | ∞ | Branch current |

**Critical insight**: The rows for out1 and out2 have G values ~1e-11 to 1e-12.
This is because at the initial operating point (V=0), all MOSFETs are in cutoff
and their conductances are essentially just gmin (1e-12 S).

**Why the ring oscillator is hard**:
1. **No stable DC point**: Ring oscillator is inherently unstable - it has no equilibrium
2. **Cutoff MOSFETs**: At x=0, all MOSFETs are off, so KCL equations only have gmin terms
3. **Feedback loop**: The output of each inverter feeds the next, creating circular dependency
4. **Near-singular Jacobian**: Without GMIN stepping, Newton iteration fails immediately

**This explains the need for GMIN stepping**:
- At u=0, the Jacobian is nearly singular (κ = 3.6e11)
- Adding diagGmin = 0.01 S would change condition number to ~100
- As GMIN ramps down, the MOSFETs gradually turn on and provide conductance

**Comparison with simple RC circuit**:
- Simple RC with diodes: κ = 1.7e3 (well-conditioned)
- Ring oscillator with MOSFETs: κ = 3.6e11 (ill-conditioned)

The difference is the MOSFET in cutoff vs the diode which always has some conductance.

**Recommendation**: GMIN stepping is essential for DC convergence. Source stepping
alone won't help because the problem is the Jacobian singularity, not the source values.

### 5.4 Runtime Convergence: Per-Iteration Voltage Limiting (Priority: High)

**This is critical for transient simulation, not just DC init.**

ngspice calls limiting functions **every Newton iteration** in device load functions:

```c
// bjtload.c:384-391 - called EVERY Newton iteration
vbe = DEVpnjlim(vbe, *(ckt->CKTstate0 + here->BJTvbe), vt, vcrit, &icheck);
vbc = DEVpnjlim(vbc, *(ckt->CKTstate0 + here->BJTvbc), vt, vcrit, &ichk1);
```

The `vold` comes from the **state vector** (previous iteration), not initial guess.

**Three limiting functions** (from `devsup.c`):

| Function | What it limits | Used by |
|----------|----------------|---------|
| `DEVpnjlim()` | PN junction voltage | Diodes, BJTs |
| `DEVfetlim()` | FET gate voltage | MOSFETs |
| `DEVlimvds()` | Drain-source voltage | MOSFETs |

**Why this matters for ring oscillator**: During transient, large Newton steps can push internal BJT/diode voltages to extreme values, causing `exp(V/Vt)` overflow. Limiting prevents this.

**Implementation approach for Cadnip.jl**:

1. **Implement `$limit()` in vasim.jl** - VA models can use `$limit(V, "pnjlim", Vt, Vcrit)`
2. **Store previous iteration voltages** - Need state vector for `vold`
3. **Implement `limexp()` in va_env.jl** - Limited exponential function

```julia
# limexp: limited exponential to prevent overflow
limexp(x) = x < 80.0 ? exp(x) : exp(80.0) * (1.0 + x - 80.0)

# pnjlim: limit PN junction voltage change per iteration
function pnjlim(vnew, vold, vt, vcrit)
    if vnew > vcrit && abs(vnew - vold) > 2 * vt
        if vold > 0
            arg = (vnew - vold) / vt
            vnew = arg > 0 ? vold + vt * (2 + log(arg - 2)) : vold - vt * (2 + log(2 - arg))
        else
            vnew = vt * log(vnew / vt)
        end
    elseif vnew < 0
        arg = vold > 0 ? -vold - 1 : 2 * vold - 1
        vnew = max(vnew, arg)
    end
    return vnew
end
```

**Also needed: timestep control on non-convergence** (from `dctran.c:802`):
```c
ckt->CKTdelta = ckt->CKTdelta / 8;  // Cut timestep by 8x on non-convergence
```
DifferentialEquations.jl handles this automatically via adaptive stepping.

#### 5.4.1 SciML Integration Challenges

**The problem**: ngspice's limiting works because device load functions are called every Newton
iteration and have access to `CKTstate0` (previous iteration's values). In SciML, the residual
function `f!(F, u, p)` is a black box - there are no hooks into individual Newton iterations.

**Potential approaches** (in order of practicality):

1. **`limexp()` only (simplest)** - Doesn't need previous state, prevents overflow:
   ```julia
   limexp(x) = x < 80 ? exp(x) : exp(80) * (1 + x - 80)
   ```
   Benefit: Trivial to implement in `va_env.jl`. Limitation: Doesn't provide same convergence
   benefits as pnjlim (which limits voltage change, not just exponential value).

2. **Iterator interface for DC only** - Use `init()`/`step!()` for manual Newton control:
   ```julia
   cache = init(prob, NewtonRaphson())
   u_prev = copy(cache.u)
   for i in 1:maxiters
       step!(cache)
       apply_limiting!(cache.u, u_prev)
       u_prev .= cache.u
   end
   ```
   Works for DC operating point but doesn't integrate with OrdinaryDiffEq's implicit timesteppers.

3. **`step_limiter!` callback** - Available on BDF methods (`FBDF`, `ABDF2`):
   ```julia
   FBDF(; step_limiter! = (u, integrator, p, t) -> apply_limiting!(u))
   ```
   But this is called *after* Newton converges at each timestep, not between Newton iterations.
   Still useful for catching runaway solutions early.

4. **Custom nonlinear solver** - Wrap NonlinearSolve.jl with limiting:
   ```julia
   struct LimitingNewton{ALG} <: AbstractNonlinearSolveAlgorithm
       alg::ALG
       junction_indices::Vector{Int}  # Which u indices are junction voltages
   end
   ```
   Would need to implement the full solver interface, using iterator interface internally.

5. **Reformulate equations** - Instead of `I = Is*(exp(V/Vt) - 1)`, use implicit form:
   ```julia
   # Clamp voltage in residual evaluation
   V_safe = clamp(V, -40*Vt, 40*Vt)
   I = Is*(exp(V_safe/Vt) - 1)
   ```
   Simple but changes the mathematical problem slightly.

**Recommended approach**:
1. Implement `limexp()` immediately (trivial, helps with overflow)
2. Use `step_limiter!` for post-Newton limiting on BDF methods
3. Consider opening a SciML issue requesting Newton iteration hooks for circuit simulation
4. Long-term: Custom nonlinear solver wrapper with proper limiting

### 5.5 Implement NODESET Support (Priority: Low)

Add initial guess hints for specific nodes:

```julia
# SPICE: .NODESET V(node)=value
function set_nodeset!(circuit, node::Symbol, value::Float64)
    push!(circuit.nodesets, (node, value))
end

# In DC solve, use nodesets as initial guess
function apply_nodesets!(u0, circuit, sys)
    for (node, value) in circuit.nodesets
        idx = get_node_index(sys, node)
        u0[idx] = value
    end
end
```

### 5.6 Enhanced CedarDCOp Algorithm (Priority: High)

Combine all homotopy methods into a robust initialization:

```julia
struct CedarDCOp{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    maxiters::Int
    nlsolve::NLSOLVE
    use_shampine::Bool
    # New options:
    use_gmin_stepping::Bool
    use_source_stepping::Bool
    gmin_initial::Float64
    gmin_factor::Float64
end

function initialize_dae!(integrator, alg::CedarDCOp)
    # 1. Try direct Newton
    result = newton_solve(...)
    if result.converged
        return result
    end

    # 2. Try GMIN stepping
    if alg.use_gmin_stepping
        result = gmin_stepping(...)
        if result.converged
            return result
        end
    end

    # 3. Try source stepping
    if alg.use_source_stepping
        result = source_stepping(...)
        if result.converged
            return result
        end
    end

    # 4. Fallback to PseudoTransient
    result = pseudo_transient(...)
    return result
end
```

## 6. Implementation Plan

### Phase 1: Core Homotopy Methods (2-3 weeks effort)

1. **Add `srcFact` to MNASpec** and propagate through all source stamping
2. **Implement `gmin_stepping()`** algorithm with adaptive factor
3. **Implement `source_stepping()`** algorithm with GMIN fallback
4. **Integrate into CedarDCOp** as fallback chain

### Phase 2: Device-Level Improvements (1-2 weeks effort)

1. **Add device-level GMIN** to diode, BJT, MOSFET models
2. **Implement `pnjlim()`** junction voltage limiting
3. **Add limiting to VA code generation** for exp() terms

### Phase 3: Scaling and Conditioning (1 week effort)

1. **Implement charge scaling** in `stamp_charge_state!()`
2. **Add condition number monitoring** for debugging
3. **Test on ring oscillator and astable multivibrator**

### Phase 4: Polish and Testing (1 week effort)

1. **Add NODESET support** for initial guess hints
2. **Create comprehensive test suite** for convergence
3. **Document all new options** in API docs

## 7. Recommended Testing Benchmarks

1. **Ring Oscillator** (9-stage CMOS) - No DC equilibrium
2. **Astable Multivibrator** (BJT) - Metastable DC point
3. **Monostable Multivibrator** (BJT) - Stable DC, but needs proper init
4. **Full-wave Rectifier** - Strong diode nonlinearity
5. **CMOS Inverter Chain** - Digital switching
6. **OpAmp with feedback** - High gain, sensitive to init

## 8. Summary of LevenbergMarquardt vs ngspice GMIN

Current `LevenbergMarquardt(damping_initial=1.0)` provides similar regularization to GMIN stepping:
- Both add a diagonal term to make the Jacobian invertible
- LM damping: `(J + λI)⁻¹`
- GMIN: `(J + gmin*I)⁻¹` (approximately, when applied to diagonal)

**Key difference**: ngspice applies GMIN to device conductance (physics-based), while LM applies damping uniformly (numerical technique). For better conditioning and physical accuracy, device-level GMIN is preferred.

## 9. Conclusion

The main gaps between Cadnip.jl and ngspice are:

1. **Device-level GMIN** - ngspice adds gmin to device conductance, Cadnip.jl only to matrix diagonal
2. **GMIN stepping** - ngspice has adaptive GMIN ramping, Cadnip.jl relies on LM damping
3. **Source stepping** - ngspice ramps sources from 0, Cadnip.jl uses PseudoTransient (similar but different)
4. **Junction limiting** - ngspice limits voltage changes, Cadnip.jl uses backtracking

The LevenbergMarquardt finding from the ring oscillator investigation shows that regularization-based approaches can work. The proposal builds on this success while adding the full ngspice-style homotopy chain for maximum robustness.

Implementing these improvements should significantly improve convergence for challenging circuits while maintaining the performance advantages of the MNA/SciML architecture.
