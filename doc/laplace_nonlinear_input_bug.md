# Bug: Incorrect DC for laplace_nd with nonlinear input

## Observed behavior

PhotoDetector model: `I(ele_out) <+ laplace_nd(-responsivity * pow(OptE(opt_in[0]), 2), {1}, {1, tau})`

With `OptE(opt_in[0]) = 1.0V`, expected `V(ele_out) = 1.0V`, actual `V(ele_out) = -3.0V`.

Linear inputs (e.g., `laplace_nd(V(inp), ...)`) produce correct results.

## Verified facts

1. The state-space conversion is correct: `va_laplace_nd_dss((1,), (1, tau))` gives DC gain = 1 (verified: `-C * inv(A) * B + D = 1.0`).

2. The G matrix at the DC solution has:
   - `G[state, state] = 1.88e11` (from `-A`, correct)
   - `G[state, opt_in_0] = -2.0` (from Jacobian of `B * (-pow(V,2))` at V=1)
   - `G[ele_out, state] = 1.88e11` (from `C_out`, correct)
   - `G[ele_out, ele_out] = 1.0` (from 1Ω resistor, correct)

3. The b vector at the DC solution has `b[state] = 1.0`.

4. The residual `G * x_dc - b` is zero — the Newton solver has converged.

5. The state equation row gives: `1.88e11 * V(state) + (-2) * 1.0 = 1.0`, so `V(state) = 3/1.88e11`.

6. The output equation gives: `1.88e11 * V(state) + 1.0 * V(ele_out) = 0`, so `V(ele_out) = -3.0`.

7. The correct answer should be `V(state) = 1/1.88e11` (from `(-A)*x = B*u = B*(-1) = -1`, giving `x = 1/1.88e11`), producing `V(ele_out) = -1.0`.

## What we know about the cause

The `extra_stamps_b` code stamps both a Jacobian into G and a Newton companion value into b for the state equation's input coupling (`b[si] = B * u(V)`). This follows the same pattern as regular contribution stamps.

For linear inputs, the Jacobian is constant and the companion b equals the raw value, so the system is correct.

For nonlinear inputs like `pow(V, 2)`, the Jacobian `∂(B*u)/∂V = B * 2V` and the companion `b = B*u(V0) - Jac*V0` differ from the raw value `B*u(V0)`. The Newton solver converges in one step to a solution that satisfies `G*x = b` but does not satisfy the original nonlinear equation.

The factor of 3 appears because: `b_companion = f(V0) - J*V0 = -1 - (-2)*1 = 1`, and `G*x = (-A)*x_s + J*V = (-A)*x_s - 2`, giving `(-A)*x_s = 3` instead of `(-A)*x_s = 1`.

## Affected models

- PhotoDetector (uses `pow(OptE, 2)` as laplace input)
- Any model using a nonlinear expression as `laplace_nd` input

## Not affected

- Linear laplace inputs (`V(node)`, `gm * V(node)`)
- AC analysis (linearized, Jacobian coupling is correct)
- Transient analysis (solver re-evaluates at each timestep)
