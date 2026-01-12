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

**Short answer**: No - these are different things with different purposes.

**Matrix-level GMIN** (`assemble_G(...; gmin=x)`):
```julia
# Changes the problem being solved:
F(x) = (G + gmin·I) · x - b
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
| Aspect | Matrix GMIN | LM Damping |
|--------|-------------|------------|
| Changes residual | Yes | No |
| Changes equilibrium | Yes (pulls nodes to 0V) | No |
| Oscillators (no DC solution) | Creates fake equilibrium | Can't help - no solution exists |
| Singular Jacobian | Regularizes problem | Regularizes inversion |

**Why this matters for oscillators**: Ring oscillators have no DC equilibrium - LM can iterate forever because there's no solution to converge to. Matrix GMIN creates a fake (but stable) equilibrium by adding shunt resistors.

**Recommendation**:
1. **Keep `MNASpec.gmin`** for device-level GMIN (accessed via `$simparam("gmin")`) - this is physics-correct
2. **Add a separate `matrix_gmin` parameter** for creating initial conditions:
   ```julia
   Base.@kwdef struct MNASpec{T<:Real}
       gmin::Float64 = 1e-12       # Device-level GMIN (for VA models)
       matrix_gmin::Float64 = 0.0  # Matrix diagonal GMIN (for oscillator init)
       # ...
   end
   ```
3. **Use matrix_gmin only for `:dcop` mode** when finding initial conditions for oscillators

## 1. ngspice Default Tolerances and Homotopy Methods

### 1.1 Default Tolerance Values (from `cktntask.c`)

| Parameter | ngspice Default | Cadnip.jl Current | Notes |
|-----------|-----------------|-------------------|-------|
| `abstol` | 1e-12 | 1e-12 | Current tolerance (Amperes) |
| `reltol` | 1e-3 | 1e-3 | Relative tolerance |
| `vntol` | 1e-6 | 1e-6 | Voltage tolerance (Volts) |
| `gmin` | 1e-12 | 1e-12 | Minimum conductance (Siemens) |
| `chgtol` | 1e-14 | N/A | Charge tolerance |
| `trtol` | 7 | N/A | Truncation error factor |
| `gminFactor` | 10 | N/A | GMIN stepping factor |
| `pivotAbsTol` | 1e-13 | N/A | Pivot absolute tolerance |
| `pivotRelTol` | 1e-3 | N/A | Pivot relative tolerance |
| `lteReltol` | 1e-3 | N/A | Local truncation error relative |
| `lteAbstol` | 1e-6 | N/A | Local truncation error absolute |

### 1.2 ngspice Homotopy Methods (from `cktop.c`)

ngspice implements a fallback chain:

1. **Direct Newton** (`NIiter`) - Standard Newton-Raphson
2. **GMIN Stepping** - Two variants:
   - `dynamic_gmin`: Adaptive stepping with factor adjustment
   - `spice3_gmin`: Fixed-step geometric progression
3. **Source Stepping** - Two variants:
   - `gillespie_src`: Adaptive source ramping
   - `spice3_src`: Fixed-step source scaling

#### 1.2.1 Dynamic GMIN Stepping Algorithm

```
1. Start with GMIN = 0.01 S (100Ω to ground on all nodes)
2. Solve at current GMIN level
3. If converged:
   - If iteration count < maxiters/4: increase factor (faster stepping)
   - If iteration count > 3*maxiters/4: decrease factor (more careful)
   - GMIN = GMIN / factor
4. If failed:
   - Reduce stepping factor: factor = sqrt(sqrt(factor))
   - Restore previous solution
   - Try again with smaller step
5. Continue until GMIN reaches target (gmin or gshunt)
6. Final solve without GMIN homotopy
```

#### 1.2.2 Source Stepping Algorithm

```
1. Set srcFact = 0 (all sources at 0V/0A)
2. Solve with zero sources (often uses GMIN stepping too)
3. Gradually increase srcFact toward 1.0:
   - If converged quickly: increase step size (raise *= 1.5)
   - If converged slowly: decrease step size (raise *= 0.5)
   - If failed: reduce step size, restore previous solution
4. Continue until srcFact = 1.0 (full source values)
```

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
    LevenbergMarquardt(),       # GMIN-like regularization
    PseudoTransient()           # Continuation method
))
```

**Mapping to ngspice**:

| ngspice Method | Cadnip.jl Equivalent | Notes |
|----------------|----------------------|-------|
| Newton-Raphson | `NewtonRaphson()` | Part of RobustMultiNewton |
| Junction limiting | `BackTracking()` linesearch | Similar damping effect |
| GMIN stepping | `LevenbergMarquardt(damping_initial=1.0)` | Regularizes Jacobian |
| Source stepping | `PseudoTransient()` | Similar homotopy approach |
| Dynamic GMIN | Not implemented | Need adaptive GMIN |

### 3.2 GMIN in Cadnip.jl (`build.jl`)

Currently, GMIN is added only to the **diagonal** of G matrix during assembly:

```julia
function assemble_G(ctx::MNAContext; gmin::Float64=0.0)
    if gmin > 0
        # Add GMIN from each voltage node to ground
        gmin_I = collect(1:ctx.n_nodes)
        gmin_J = collect(1:ctx.n_nodes)
        gmin_V = fill(gmin, ctx.n_nodes)
        ...
    end
end
```

This is fundamentally different from ngspice's approach of adding GMIN to the **device conductance**.

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

| Issue | ngspice Handling | Cadnip.jl Current |
|-------|------------------|-------------------|
| Singular Jacobian | GMIN to devices + diagonal | GMIN diagonal only |
| No DC equilibrium | Source stepping to ramp | PseudoTransient, CedarUICOp warmup |
| Exponential overflow | Junction voltage limiting | Backtracking linesearch |
| Metastable states | NODESET hints | Not implemented |

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

### 5.3 Improve Charge State Scaling (Priority: Medium)

The charge formulation may create ill-conditioned systems because:
- Voltages: O(1-10) V
- Charges: O(1e-12 - 1e-15) C (femtocoulombs)

**Proposal**: Scale charge states to similar magnitude as voltages

```julia
# During stamp_charge_state!
# Instead of q in Coulombs, use q_scaled = q / charge_scale
const CHARGE_SCALE = 1e-12  # Scale factor to bring charges to ~O(1)

function stamp_charge_state!(ctx, p, n, q_fn, x, charge_name)
    q_idx = alloc_charge!(ctx, charge_name, p, n)

    # Store unscaled charge function, but scale the state variable
    # q_scaled = q / CHARGE_SCALE
    # dq/dt = I  =>  d(q_scaled)/dt = I / CHARGE_SCALE

    # KCL: I = dq/dt enters node equations
    # Scale: q_idx column gets scaled by 1/CHARGE_SCALE
    if p != 0
        stamp_C!(ctx, p, q_idx, 1.0 / CHARGE_SCALE)
    end
    if n != 0
        stamp_C!(ctx, n, q_idx, -1.0 / CHARGE_SCALE)
    end

    # Constraint: q_scaled = Q(V) / CHARGE_SCALE
    # Jacobian entries also scaled
    stamp_G!(ctx, q_idx, q_idx, 1.0)
    stamp_G!(ctx, q_idx, p, -result.dq_dVp / CHARGE_SCALE)
    stamp_G!(ctx, q_idx, n, -result.dq_dVn / CHARGE_SCALE)

    b_constraint = (-result.q + result.dq_dVp * Vp + result.dq_dVn * Vn) / CHARGE_SCALE
    stamp_b!(ctx, q_idx, b_constraint)
end
```

### 5.4 Junction Voltage Limiting (Priority: Medium)

ngspice uses `DEVpnjlim()` to limit voltage changes across PN junctions to prevent exponential overflow:

```julia
function pnjlim(vnew, vold, vt, vcrit)
    # Limit voltage change to ~2-3 thermal voltages per iteration
    if vnew > vcrit && abs(vnew - vold) > 2 * vt
        if vold > 0
            arg = (vnew - vold) / vt
            if arg > 0
                vnew = vold + vt * (2 + log(arg - 2))
            else
                vnew = vold - vt * (2 + log(2 - arg))
            end
        else
            vnew = vt * log(vnew / vt)
        end
    end
    return vnew, vnew != vold
end
```

This should be integrated into the Backtracking linesearch or as a separate step.

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
