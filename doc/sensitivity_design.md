# DC sensitivity analysis

`sens!(circuit, output)` answers "how much does this operating-point reading move
when that parameter moves" — SPICE's `.sens v(out)`, and the first of
`doc/FINDINGS.rst`'s gap table to be closed from the differentiability side
rather than the netlist side.

Status: implemented in `src/sensitivity.jl`, tested in `test/sensitivity.jl`.

## The quantity

The operating point is the root of the MNA residual

```
F(x, p) = G(x, p)·x − b(x, p) = 0
```

— the residual `_dc_newton_compiled` drives to zero, with `G` the linearization
the companion models stamp, which for correct companion stamping is also
`∂F/∂x`. Differentiating the root with respect to a parameter (the implicit
function theorem) gives

```
∂F/∂x · dx/dp + ∂F/∂p = 0     ⟹     dx/dp = −G⁻¹ · ∂F/∂p
```

and for one scalar output `y = eᵀx`, transposing moves the linear solve off the
parameter loop entirely:

```
dy/dp = −eᵀ G⁻¹ ∂F/∂p = −λᵀ ∂F/∂p ,     Gᵀ λ = e .
```

That is the same adjoint trick `noise!` uses to get every source's transfer
function out of one solve per frequency (`doc/noise_analysis_design.md`). One
factorization covers every parameter; each parameter costs only `∂F/∂p`.

## Why finite differences, and why they are accurate here

`∂F/∂p` is taken by central differences over a *rebuild*: re-stamp the circuit
with the parameter nudged, at the operating point held frozen, and difference the
two residuals.

Not by AD, because the parameter enters through the generated builder and the
context it stamps into is `Float64`-typed — `ctx.G_V`, `ctx.b_V` are
`Vector{Float64}` — so a dual number cannot flow from a `.param` value into the
matrix. (The duals Cadnip *does* use run the other way: `x` → the Jacobian, which
is why `G` is exact and the adjoint is not itself an approximation.)

The differencing is on the **residual**, not on the solve. Nothing re-runs
Newton, so no convergence tolerance and no solver path enters the derivative;
what is differenced is a stamping, which is a smooth function of the parameter
evaluated to machine precision. The usual finite-difference objection — "you are
differencing two noisy solves" — does not apply.

Step: `cbrt(eps)` relative, the standard optimum for a central difference
(truncation `h²` against cancellation `eps/h`).

## Measured

Against a brute-force re-solve — nudge the parameter, solve the *whole* operating
point again, difference the output — which is the definition of the quantity
computed the expensive way.

Resistive divider (`.param rtop=1k`, `.param rbot=1k`, 5 V), `V(out)`, where the
closed form is `−5·rbot/(rtop+rbot)²`:

```
                 sens!            analytic
∂out/∂rtop  -1.2500000000570e-3  -1.25e-3
∂out/∂rbot   1.2500000000481e-3   1.25e-3
```

Nonlinear — a 1 kΩ + VA diode rectifier with a subcircuit hanging across the
supply — where nothing is a rearrangement of a linear formula:

```
             sens!            brute-force re-solve     rel. difference
rs     -3.718212822069e-5   -3.718212826520e-5           1.2e-9
vin     8.585742096628e-3    8.585742095057e-3           1.8e-10
x1.rv  -0.0 (exactly)        0.0                         —
```

A level-1 NMOS common-source stage (`test/design_flow.jl`'s `cs_amp`), `V(drain)`:

```
parameter    value      ∂out/∂p      per %      vs. re-solve
freq         1e+06      0            0          —
rd           10000     -2.01834e-4  -0.0201834  9.7e-9
vac          0.005      0            0          —
vbias        1.1472    -9.02655     -0.103553   9.9e-9
vsup         5          0.980393     0.0490197  1.0e-8
```

`∂drain/∂vbias = −9.03` is the stage's small-signal voltage gain, against the
hand derivation's `−gm·RD = −8.94` — a DC sensitivity *is* the gain when the
parameter is the input. `freq` and `vac` come back exactly zero: they parameterize
the `SIN` waveform, which the `:dcop` mode never evaluates.

Cost, same circuit, 5 parameters: 1.1 ms and 3.7 k allocations, against 0.07 s
for a single `dc!`. One Newton solve and one factorization, whatever the
parameter count.

## What is differentiable

Exactly what is overridable: the names `observed_params` (`src/param_overrides.jl`)
finds by building once with a `ParamObserver` — top-level `.param` cards, each
subcircuit's own parameters, and the instance parameters on an `X` line,
addressed with the dotted selectors `alter` and a sweep axis already use
(`x1.rv`). The observation also supplies the *value* to nudge, unless the circuit
carries an override for that name, which then wins — so `MNACircuit(deck;
rtop=3e3)` differentiates at 3 kΩ, not at the netlist's default.

A device instance parameter (`r1=(r=2k,)`) is not reachable — the same gap
`doc/parameter_overrides.md` §1 records for `alter`; parameterize the netlist with
a `.param` instead. When this lands (PR "A device line is a scope"), `sens!`
inherits it with no change: it enumerates whatever the observer reports.

A hand-written builder is opaque to the observer, so it names its parameters and
their values itself: `sens!(c, :out; params=(R1=1e3, R2=2e3))`.

## Design notes

**One scratch context.** The base linearization and every perturbed residual
stamp into the same `MNAContext`, reset between builds. Node allocation happens
once; the parameter loop is pure stamping.

**gmin on the adjoint.** `G` is assembled with a `gshunt` (default `1e-12`), as
`ac!` and `noise!` assemble theirs. A circuit with a floating island has a
singular `G` even though its operating point solves — zeros satisfy the residual,
so `_dc_newton_compiled` returns at once — and without the shunt there is no
adjoint at all. At `1e-12` the perturbation to a sensitivity is far below the
finite-difference noise. A singularity the shunt cannot fix (a voltage-source
loop) is reported as such rather than as a `SingularException` from three layers
down.

**A parameter that changes the system size** — one selecting an `.if` branch,
say — is a discrete choice, not a derivative. The perturbed rebuild checks the
size and says so instead of returning a number computed from mismatched vectors.

**`state_index`.** The name→row lookup `noise!` had inlined is now
`MNA.state_index` (`src/mna/solve.jl`), returning `0` for ground and `nothing`
for unknown, with each analysis wrapping it in its own error message.

## Not done

- **AC sensitivity.** `∂|H(jω)|/∂p` needs the same adjoint against `(jωC + G)`,
  complex, and a convention for magnitude/phase. The machinery is all there;
  the API question (a grid of sensitivities per parameter) is the work.
- **`.sens` the card.** Following `doc/noise_analysis_design.md` N3 — Julia is
  the simulation API, a deck does not drive an analysis — the card is not wired
  to anything. It parses; that is all it has to do.
- **Sensitivity of a transient trajectory**, which is what the retired
  DAECompiler-era `test/sensitivity.jl` demonstrated with
  `ODEForwardSensitivityProblem`. That test file is now this analysis's test
  file; the trajectory version wants the MNA `ODEProblem` to carry its
  parameters as a SciML `p`, which it does not today.
