# State Scaling Implementation Plan

## Problem Statement

Charge state variables are typically O(1e-12) to O(1e-15) Coulombs, while voltages are O(1) and currents O(1e-3 to 1e-6). This magnitude mismatch creates poorly conditioned Jacobian matrices, leading to:
- Slower convergence
- Higher stiffness
- Potential numerical instability

## Proposed Solution: Charge State Scaling

**Core idea:** Scale charge variables by a constant factor so they're O(1) like voltages.

```
q_scaled = q * CHARGE_SCALE    where CHARGE_SCALE ≈ 1e12
```

The solver works with `q_scaled` internally. When we need actual charge, divide by scale.

## Implementation

### 1. Define Scale Constant

```julia
# src/mna/contrib.jl or src/mna/context.jl
const CHARGE_SCALE = 1e12  # Coulombs → "scaled Coulombs" (~pC)
```

### 2. Modify `stamp_charge_state!`

Current stamping for constraint `F = q - Q(V) = 0`:
```julia
# G matrix: ∂F/∂q = 1, ∂F/∂V = -∂Q/∂V
stamp_G!(ctx, q_idx, q_idx, 1.0)
stamp_G!(ctx, q_idx, p, -result.dq_dVp)
stamp_G!(ctx, q_idx, n, -result.dq_dVn)

# C matrix: KCL coupling
stamp_C!(ctx, p, q_idx, 1.0)   # I = dq/dt
stamp_C!(ctx, n, q_idx, -1.0)

# RHS
b_constraint = -result.q + result.dq_dVp * Vp + result.dq_dVn * Vn
stamp_b!(ctx, q_idx, b_constraint)
```

**With scaling** (constraint becomes `F = q_scaled/SCALE - Q(V) = 0`):
```julia
# G matrix: ∂F/∂q_scaled = 1/SCALE
stamp_G!(ctx, q_idx, q_idx, 1.0 / CHARGE_SCALE)
stamp_G!(ctx, q_idx, p, -result.dq_dVp)  # unchanged
stamp_G!(ctx, q_idx, n, -result.dq_dVn)  # unchanged

# C matrix: I = d(q_scaled/SCALE)/dt = (1/SCALE) * dq_scaled/dt
stamp_C!(ctx, p, q_idx, 1.0 / CHARGE_SCALE)
stamp_C!(ctx, n, q_idx, -1.0 / CHARGE_SCALE)

# RHS: scale Q(V) to match
b_constraint = -result.q + result.dq_dVp * Vp + result.dq_dVn * Vn
stamp_b!(ctx, q_idx, b_constraint)  # Q is still in real units here
```

**Alternative formulation** (multiply constraint by SCALE for better conditioning):
```julia
# Scaled constraint: F_scaled = q_scaled - SCALE*Q(V) = 0
# This keeps diagonal entry = 1.0 (well-conditioned)

stamp_G!(ctx, q_idx, q_idx, 1.0)  # ∂F_scaled/∂q_scaled = 1
stamp_G!(ctx, q_idx, p, -CHARGE_SCALE * result.dq_dVp)
stamp_G!(ctx, q_idx, n, -CHARGE_SCALE * result.dq_dVn)

# C matrix: still (1/SCALE) because I = (1/SCALE)*dq_scaled/dt
stamp_C!(ctx, p, q_idx, 1.0 / CHARGE_SCALE)
stamp_C!(ctx, n, q_idx, -1.0 / CHARGE_SCALE)

# RHS: scaled
b_constraint = -CHARGE_SCALE * result.q + CHARGE_SCALE * (result.dq_dVp * Vp + result.dq_dVn * Vn)
stamp_b!(ctx, q_idx, b_constraint)
```

### 3. Update Initial Conditions

When setting up DC initial conditions or transient IC:
```julia
# x[q_idx] should be scaled
x[q_idx] = Q_initial * CHARGE_SCALE
```

### 4. Output Conversion

When reporting charge values to user:
```julia
q_actual = x[q_idx] / CHARGE_SCALE
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/mna/contrib.jl` | Add `CHARGE_SCALE`, modify `stamp_charge_state!` |
| `src/mna/solve.jl` | Scale initial charge values, unscale in output |
| `src/mna/context.jl` | Possibly store scale factor per charge variable |

## Expected Benefits

1. **Better conditioning**: All state variables O(1)
2. **Faster convergence**: Newton iterations converge in fewer steps
3. **Lower stiffness**: Reduced condition number of Jacobian
4. **Compatible with SciML**: No solver internals need changing

## Testing Plan

### Test 1: Condition Number Comparison

Create a test circuit with voltage-dependent capacitor and measure condition number:

```julia
# test/mna/charge_scaling.jl
using LinearAlgebra
using Test

@testset "Charge scaling reduces condition number" begin
    # Build a simple circuit with voltage-dependent capacitor
    # (e.g., varactor diode model)

    function build_test_circuit(; charge_scale=1.0)
        # ... circuit with C(V) = C0 * (1 + V/V0)
    end

    # Build with and without scaling
    sys_unscaled = build_test_circuit(charge_scale=1.0)
    sys_scaled = build_test_circuit(charge_scale=CHARGE_SCALE)

    # Assemble Jacobian at operating point
    G_unscaled, C_unscaled = assemble(sys_unscaled)
    G_scaled, C_scaled = assemble(sys_scaled)

    # Compute condition numbers
    κ_unscaled = cond(Matrix(G_unscaled))
    κ_scaled = cond(Matrix(G_scaled))

    @test κ_scaled < κ_unscaled
    @test κ_scaled / κ_unscaled < 1e-6  # Expect ~1e12 improvement

    println("Condition number improvement: $(κ_unscaled / κ_scaled)x")
end
```

### Test 2: Solution Accuracy

Verify scaling doesn't change the physics:

```julia
@testset "Charge scaling preserves solution accuracy" begin
    # Same circuit, same operating point
    sol_unscaled = dc!(circuit_unscaled)
    sol_scaled = dc!(circuit_scaled)

    # Voltages should match exactly
    @test sol_unscaled.V ≈ sol_scaled.V rtol=1e-12

    # Charges should match after unscaling
    q_unscaled = sol_unscaled.q
    q_scaled = sol_scaled.q_scaled / CHARGE_SCALE
    @test q_unscaled ≈ q_scaled rtol=1e-12
end
```

### Test 3: Transient Accuracy

Run transient and compare waveforms:

```julia
@testset "Charge scaling preserves transient accuracy" begin
    tspan = (0.0, 1e-6)

    sol_unscaled = tran!(circuit_unscaled, tspan)
    sol_scaled = tran!(circuit_scaled, tspan)

    # Compare voltage at all timepoints
    for t in range(tspan..., 100)
        V_unscaled = sol_unscaled(t)
        V_scaled = sol_scaled(t)
        @test V_unscaled ≈ V_scaled rtol=1e-9
    end
end
```

### Test 4: Convergence Speed (Optional)

Measure Newton iterations if using custom solver:

```julia
@testset "Charge scaling improves convergence" begin
    # Count iterations to converge
    iters_unscaled = dc_with_stats!(circuit_unscaled).iterations
    iters_scaled = dc_with_stats!(circuit_scaled).iterations

    @test iters_scaled <= iters_unscaled
end
```

### Running the Tests

```bash
~/.juliaup/bin/julia --project=test test/mna/charge_scaling.jl
```

## Notes

- This is purely a numerical conditioning technique
- The physics is unchanged; just the representation
- Could make scale configurable per-variable if needed
- Similar approach could apply to flux (inductors) if needed
