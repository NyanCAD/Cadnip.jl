# PCNR: Predictor/Corrector Newton-Raphson for Cadnip

*Reviewed and refined implementation plan. Supersedes the earlier research
artifact (`compass_artifact_wf-c99b8a86...md`), whose code references and
device-stamping example were incorrect — see "Corrections to the previous
revision" below. Scope: implement PCNR-style limiting **inside Cadnip
first**, with a deferred plan for upstreaming the solver-side pieces into
NonlinearSolve.jl / OrdinaryDiffEq.jl.*

## Reference

Karthik V. Aadithya, Eric R. Keiter, Ting Mei (Sandia National Laboratories),
**"Predictor/Corrector Newton-Raphson (PCNR): A Simple, Flexible, Scalable,
Modular, and Consistent Replacement for Limiting in Circuit Simulation"**,
SAND2018-5689C; published in *Scientific Computing in Electrical Engineering*
(Springer, 2020), [doi:10.1007/978-3-030-44101-2_19](https://doi.org/10.1007/978-3-030-44101-2_19);
open access at [OSTI 1523781](https://www.osti.gov/servlets/purl/1523781).

### Algorithm summary (verified against the paper)

SPICE-style limiting keeps per-device state (the junction voltage the device
*actually used* last iteration, `vold`) and evaluates the model at an
intermediate point `pnjlim(vnew, vold)` instead of at the requested solution.
The paper's critique: this makes `g` and its Jacobian functions of the
*history* of `x`, is inconsistent when two devices limit the same branch, and
forces every device to carry iteration bookkeeping.

PCNR fixes this with three ideas:

1. **Every limited quantity becomes an unknown.** `x = [x_MNA; x_lim]`,
   `g(x) = [g_MNA; g_lim]` with one limiting equation per limited quantity:
   `g_lim,k = v_lim,k − (V_p − V_n) = 0`. Devices evaluate at `v_lim,k`, so
   evaluation is a pure function of `x` again. Each device *owns* the
   variables it limits, so two diodes across the same node pair get two
   independent limiting variables — no clashes.

2. **Predict, then correct.** The predictor is a plain Newton step on the
   augmented system. The corrector then replaces the limiting components:
   `x_{i+1,lim} = refine(x_i, x_{i+1})` — the simulator explicitly invokes
   each device's limiting function (e.g. `pnjlim`) on the variables that
   device owns.

3. **Schur complement keeps the linear solve MNA-sized.** With the equation
   form above, `J_lim/lim = I` identically, so:

   ```
   (i)   Δx_MNA = -((J_MNA/MNA - J_MNA/lim · J_lim/MNA)⁻¹ · (g_MNA - J_MNA/lim · g_lim))
   (ii)  Δx_lim = -(g_lim + J_lim/MNA · Δx_MNA)
   (iii) x_{i+1} = x_i + [Δx_MNA; Δx_lim]
   ```

   Note (worked example, Fig. 2 of the paper): in this formulation the diode
   conductance appears **only in the `J_MNA/lim` columns** — `J_MNA/MNA`
   carries *no* entries for the limited device, because the device current
   depends on `v_lim`, not on the node voltages. The Schur complement
   `S = J_MNA/MNA − J_MNA/lim·J_lim/MNA` puts the conductances back at the
   familiar node-node positions, reproducing exactly the matrix a SPICE-style
   simulator stamps today. The paper also notes PCNR works for DAEs
   (transient) and never takes more iterations than traditional limiting.

## Why this matters for Cadnip specifically

Two facts make PCNR unusually well-suited — arguably *necessary* — for
Cadnip's architecture, and one fact makes it urgent:

1. **Cadnip builders are stateless.** The whole MNA backend re-stamps the
   circuit from `(params, spec, t; x)` on every evaluation
   (`fast_rebuild!`, `src/mna/precompile.jl:470-526`). There is no per-device
   object that survives between Newton iterations, so SPICE's hidden
   `vold` state has nowhere to live. PCNR's answer — put the limiting state
   *in the solution vector* — is exactly the pattern Cadnip already uses for
   charge unknowns (`alloc_charge!`, `src/mna/context.jl:556`; see
   `doc/voltage_dependent_capacitors.md`). Limiting variables are the same
   trick applied to Newton state instead of dynamical state.

2. **`$limit` is currently a silent no-op.** `src/vasim.jl:1192-1198` returns
   the raw probe voltage, and `$discontinuity` is a no-op
   (`src/vasim.jl:2057-2059`). Meanwhile **every VADistiller model in the
   repo ships complete SPICE limiting logic in Verilog-A** and calls
   `$limit` to reach the state the simulator is supposed to maintain:

   | model | `$limit` sites | limited branches |
   |---|---|---|
   | `diode.va` (726, 741) | 2 | vd |
   | `bjt.va` (1040-1067) | 6 | vbe, vbc, vsub |
   | `mos1/2/3/6/9, bsim3v3, vdmos` | 8 each | vgs, vds, vbs, vbd |
   | `bsim4v8.va` (5992+) | 18 | 9 junction/terminal voltages |

   The limiter implementations (`DEVpnjlim`, `DEVfetlim`, ...) are already
   present as VA analog functions (e.g. `diode.va:403-440`) — we do **not**
   need to write `pnjlim` ourselves, and the earlier revision's hand-derived
   pseudo-code (which was garbled) is unnecessary. Implementing PCNR in
   Cadnip *is* implementing `$limit`.

3. **The convergence pain is real and documented.** With limiting disabled,
   the only guard is the `limexp` clamp (`src/va_env.jl:36`). The robust-DC
   machinery (`CedarRobustNLSolve` polyalgorithm, gmin/source stepping
   fallbacks, `src/mna/solve.jl:330-657`) does the work limiting was meant to
   do, at the cost of LM/PseudoTransient fallbacks and stepping. See
   `doc/dc_newton_termination.md` (diode-chain slack), the BJT monostable
   story in `CedarShampineNLSolve`'s docstring (`src/mna/solve.jl:344-393`),
   and `doc/ring_oscillator_investigation.md`.

## `$limit` semantics (design decision)

The VADistiller models use a two-site pattern that pins down the semantics we
must provide. From `diode.va`:

```verilog
DIOvoltage = $limit(V(a_int,c), DEVlimitOldGet);            // read state
// ... load_vd = DEVpnjlim(V(a_int,c), DIOvoltage, vte, vcrit) inline ...
load_vd = $limit(V(a_int,c), DEVlimitNewSet, load_vd, limited);  // write state
```

where `DEVlimitOldGet(vnew, vold) = vold` and
`DEVlimitNewSet(vnew, vold, new_value, limited) = new_value` (plus
`$discontinuity(-1)` when limiting was applied). Per the Verilog-AMS LRM, the
simulator prepends `(vnew, vold)` to the user arguments, where `vold` is the
value the model used on the previous iteration.

For the OldGet/NewSet pair to work, **limit state must be keyed per
(instance, probe branch), shared across `$limit` call sites on that branch**:
the OldGet site reads the state without changing it, the NewSet site stores
the limited voltage the model actually used. (Per-callsite state would freeze
the OldGet slot at its initial value.) In PCNR terms: **one limiting variable
per (instance, limited branch)** — which is precisely the paper's "each
device owns the solution variables it limits".

The state slot for a branch is updated to the value returned by the *last*
`$limit` site executed on that branch in an evaluation pass.

## v1 design: paper-pure formulation with an explicit corrector (implemented)

> **Design correction (learned during implementation).** An earlier revision
> of this plan proposed a "fused" v1: keep the models' inline limiter calls,
> stamp `g_lim = x_lim − w(V)` as a linearized row, and let plain Newton do
> the correcting — no custom solver loop. **That cannot work.** Any limit
> equation solved *simultaneously* with the MNA rows makes `x_lim` track the
> new branch voltage to first order, so at every stamping pass
> `vold ≈ vnew` and the limiter never fires (trace the 5V rectifier from
> zeros: iteration 1 sets `x_lim = V_out ≈ 5`, iteration 2 sees
> `pnjlim(5, 5) = 5` and evaluates `exp(5/0.026)`). The nonlinear corrector
> applied *between* iterations is not an optimization — it is the mechanism.
> This is presumably exactly why the paper has an explicit correct phase.

The implemented design is the paper's, with the corrector living in a small
Cadnip-owned Newton loop:

### Formulation — one API, shared with the future `$limit` codegen

Augment the state: `x = [V₁..Vₙ, I₁..Iₘ, q₁..qₖ, v_lim,1..v_lim,L]`
(`LimitIndex`, `alloc_limit!` — mirrors of `ChargeIndex`/`alloc_charge!`).

All limiting goes through **one runtime primitive**, shaped 1:1 like
Verilog-A's `$limit(V(p,n), fn, args...)` so the native devices exercise
exactly the entry point Phase 2 codegen will lower to (no second system):

```julia
w = limit!(ctx, base_name, instance, p, n, vnew, x, fn, args...)
```

which, for the branch's limiting variable `ℓ`:

- allocates `ℓ` (positionally stable, unconditional);
- reads `vold = x[ℓ]` as a plain Float64 — the voltage the device evaluated
  at last iteration;
- computes `w = fn(vnew, vold, args...)`. `vnew` may be a `JacobianTag`
  dual, so AD carries `∂w/∂vnew` into the device's conductances — which
  therefore land in the **node-node block** as usual (the `x_lim` column
  stays empty apart from the row diagonal);
- **records** `extract_value(w)` into a per-iteration workspace buffer
  (`limit_w`) — this is the corrector's target;
- stamps the **linear** tracking row `g_lim = x[ℓ] − (V_p − V_n) = 0`:
  `G[ℓ,ℓ] = 1`, `G[ℓ,p] = −1`, `G[ℓ,n] = +1` — so `J_lim/lim = I`.

The device then evaluates its currents at `w`. Multi-site `$limit` branches
(VADistiller's OldGet/NewSet pattern) lower to the same primitives —
allocate once per branch, record at each site, last record wins.

### The PCNR loop (`_dc_pcnr_newton`, src/mna/solve.jl)

Runs as tier 0 of `_dc_solve_with_fallbacks`, only when `cs.n_limits > 0`:

1. **Initialize**: seed `x_lim` slots from `limit_init` (allocation-time
   `init=` values; default 0).
2. **Predict**: plain Newton step on the augmented system
   (`δ = G \ F`; `G` is the companion Jacobian, same machinery as the
   standard path).
3. **Correct**: `x_lim,k ← limit_w[k]` — copy the recorded limited voltage
   each device *actually evaluated at* during this iteration's stamping, so
   it becomes `vold` for the next. The simulator needs no knowledge of any
   device's limiter function — the correction is a copy, which is what makes
   the same mechanism serve both native devices and VA models whose limiter
   lives inline (`DEVpnjlim`).
4. **Converge**: `‖F‖ < abstol`. The `g_lim` rows are part of `F`, so
   convergence implies `x_lim = V_branch`, hence `w = fn(V, V) = V` (no
   active limiting) — SPICE's "no limiting applied this iteration"
   condition falls out automatically, as does the `$discontinuity(-1)`
   iterate-again signal.

On failure it falls through to the unchanged standard chain
(`CedarRobustNLSolve` → gmin stepping → source stepping).

> Historical note: the first Phase-1 pilot used the paper-verbatim shape
> instead — device evaluates *at* `x[ℓ]`, conductance stamped into the
> `x_lim` column, simulator-side `PNJunctionLimit` refine specs. It worked
> (same iteration counts), but its corrector required the simulator to know
> each device's limiter, which VADistiller models can't provide — their
> limiter is inline VA. It was replaced by the recorded-`w` scheme above to
> keep one system; `pnjlim` survives as the limiter function itself.

### Scope: DC and transient initialization. In-step transient: built, and its activation condition found structurally unsatisfiable.

The corrector runs in `_dc_pcnr_newton`, i.e. for `dc!` and for transient
*initialization* (`CedarDCOp`/`CedarTranOp` route through
`_dc_solve_with_fallbacks`). SPICE also applies pnjlim on every NR iteration
inside every timestep (`vold` seeded from the previous timepoint); Cadnip
does not, and — unlike the DC case — this is not a missing feature. It was
built and benchmarked (2026-07).

**What was actually shown, precisely.** This is not a claim that limiting
partway through a transient step would provide no benefit if it fired — that
question was never tested and is not what the argument below addresses. What
was shown is narrower and mechanical: the *classical* limiting mechanism —
compare the newly proposed branch voltage against a `vold` carried from
somewhere else, and only pass through unmodified when they already agree —
never has a chance to distinguish anything during a warm-started step,
because both quantities it's comparing are forced to the same value before
the comparison ever happens (see below). The corrector's *detection
condition* is structurally unsatisfiable there, not observed-and-found-
useless. During stepping the limit variables ride along inertly as a
consequence of that; the paragraphs below explain why that specific inertness
cannot be engineered away by tuning the corrector, and why it says nothing
about the discontinuity-reinitialization case ("Open question" below), where
the same detection condition is not forced.

**What was tried.** `OrdinaryDiffEqNonlinearSolve.NonlinearSolveAlg` lets an
implicit stepper delegate its stage solve to any NonlinearSolve.jl algorithm,
so the PCNR predict/correct loop was packaged as an
`AbstractNonlinearSolveAlgorithm` and driven as
`FBDF(nlsolve=NonlinearSolveAlg(PCNRSolver(...)))` — zero OrdinaryDiffEq
changes, with the corrector's `u = tmp + γ·z` affine remap for DIRK stages
(FBDF's own stage variable needed no remap, being `u(t+dt)` directly) and an
active-branch skip so an inert corrector never lags the accepted state.

**Why it doesn't fire — mechanism, checked against source, not just asserted.**
The recorded-`w` formulation makes the limiter's `vold` a genuine DAE state
(`x_lim`, tied to the branch voltage by the linear row
`g_lim = x_lim − V_branch = 0`). The claim below is *not* "the predictor
happens to track `V_branch` accurately" — accuracy is irrelevant to it. It's a
structural equality, and it was verified directly against FBDF's predictor
code (`OrdinaryDiffEqBDF/src/bdf_utils.jl`, `_eval_lagrange_oop` /
`_eval_lagrange_iip!`, the "evaluate Lagrange interpolant through `u_history`
at Θ=1" step in `perform_step!`): the predicted state is
`Σⱼ Lⱼ(θ)·u_history[j]`, where `Lⱼ` is a *scalar* Lagrange weight — a function
only of past time abscissas, never of which state component it multiplies —
applied identically, componentwise, to every past accepted state vector.
Since `x_lim` and `V_branch` are numerically *identical* at every past
accepted timepoint (that is what "accepted" means: `g_lim` was solved to
residual tolerance), the same linear functional applied to pointwise-identical
inputs produces pointwise-identical outputs:
`predicted(x_lim) == predicted(V_branch)`, exactly (up to roundoff), no matter
how sharp or discontinuous the true `V_branch(t)` actually is — a bad
prediction is still the *same* bad prediction for both components. So
`vold ≈ vnew` before the stage solve's first Newton iteration even runs, and
`pnjlim` passes through, independent of step size. (This componentwise-linear
structure is standard Nordsieck/polynomial-predictor theory for any linear
multistep DAE integrator, and Sundials IDA's C Newton loop is documented to
use the same shape of predictor — but that claim was verified against source
only for FBDF here, not re-derived from Sundials' C code.) Classical SPICE
limiting escapes this because its `vold` is the previous *Newton iteration's*
value within a single step, never an inter-step predictor's extrapolation.

This mechanism argument was then checked empirically, not just derived (a
corrector-activation counter, instrumented across the switching half-wave
rectifier, the Graetz bridge, a BJT chain, and — crucially — an adaptive
work-precision-style sweep with no forced step size at reltol 1e-3 down to
1e-6, so large steps were available): the corrector adopted zero limit values
in every run. Not "rarely" — never. Big adaptive steps do not help, because
the predictor keeps pace with them too. The two lines of evidence agree: the
source-level argument explains *why* the counter reads zero, not just that it
does.

**Cost, not just no-benefit.** The wrapped stage solve was also measured
slower than plain `FBDF` on every benchmark circuit (Graetz, the diode
multiplier, a BJT chain), even after switching the wrapper to a
modified-Newton (factor-once-per-step) strategy to minimize the tax — because
a dormant corrector still forces the per-step machinery of a second
`NonlinearProblem`/cache through `NonlinearSolveAlg`, with no compensating
win. So this is not "harmless but unused": shipping it would be a net
regression for zero functional benefit.

**Conclusion.** *Warm-started* in-step limiting — a normal predictor-then-
correct step, which is what every step of a driven transient is once the
integrator is past its first point — has its activation condition forced
unsatisfiable by construction, for the reason above, under any
predictor/corrector ODE/DAE integrator that extrapolates the limiting state
the same way it extrapolates everything else. That is a statement about the
mechanism never getting to fire, not a statement about limiting being useless
there if it could; it covers every step tested here (see "Open question"
below for the one class of step that was *not* tested). The DC/init corrector is unaffected and remains the
proven, working place for PCNR in this codebase: there is no predictor there,
so `vold` genuinely lags. Sundials IDA (today's `tran!` default) could never
have used the in-step mechanism anyway — its Newton loop is C — so no
solver-selection tradeoff was ever on the table.

### Open question: discontinuity-triggered reinitialization (not investigated)

The inertness argument above rests on one premise: the predictor step is a
*linear extrapolation from history that was already consistent* (`x_lim`
equal to `V_branch` at every point in the window). That premise could fail at
a genuine reinitialization — a point where the integrator throws away its
history and re-solves for consistent algebraic variables via Newton, rather
than extrapolating. A Newton re-solve is not a linear functional of past
history, so the "identical inputs ⇒ identical outputs" argument does not
apply to it, and such a re-solve is structurally much closer to DC init
(where PCNR demonstrably helps) than to warm-started stepping (where it
demonstrably doesn't).

**This was not tested, and — checked against source while writing this up —
is not even exercised by Cadnip today.** `tran!`'s breakpoint mechanism
(`src/mna/breakpoints.jl`, fed into `d_discontinuities`) only nudges `t` by
one ULP past the source edge and forces an FSAL/Jacobian refresh
(`OrdinaryDiffEqCore`'s `update_fsal!`/`shift_past_discontinuity!`); it never
touches `integrator.derivative_discontinuity` or calls `initialize_dae!`. So
every breakpoint hit in the benchmarks above was still a normal warm predictor
step, just a forced one — consistent with, not an exception to, the zero-
activation result.

OrdinaryDiffEqBDF does expose a real hook for the other case:
`derivative_discontinuity!(integrator, true)` (settable from a
`DiscreteCallback`) makes `reinitFBDF!`
(`OrdinaryDiffEqBDF/src/bdf_utils.jl`) wipe FBDF's history and restart at
order 1, and — per `OrdinaryDiffEqBDF`'s own `dae_derivative_discontinuity_tests.jl`
— pairing it with a DAE initialization algorithm that actually re-solves
algebraic variables (`BrownFullBasicInit`) triggers a genuine
`initialize_dae!` Newton solve on the next step. Two concrete obstacles before
this is worth building, not just noting:

1. Nothing in Cadnip wires a callback to call `derivative_discontinuity!` at
   breakpoints today — this would be new plumbing, not a flag flip.
2. The specific init algorithm the OrdinaryDiffEqBDF test uses to trigger the
   Newton re-solve, `BrownFullBasicInit`, is already known broken for
   Cadnip's mass-matrix ODE formulation — see `src/mna/dcop.jl`'s `CedarDCOp`
   docstring, which is why `tran!` uses `ShampineCollocationInit` instead.
   Whether `ShampineCollocationInit`'s Newton loop breaks the same
   history-consistency premise, and whether PCNR's corrector could be wired
   into it at all, is unresearched.

Not pursued in this branch. Flagged here because it is a genuinely different
mechanism from the one that was built and found inert above — not covered by
that negative result — and worth its own investigation before either
building it or ruling it out.

### Why no Schur factorization is needed (and what that costs)

The paper's third key insight — Schur-eliminating the limit variables so
linear solves stay MNA-sized — exists because in its formulation the device
conductances live in the `J_MNA/lim` columns, coupling the blocks both ways
(and L is not small: ~9 per BSIM4 instance, comparable to its node count).
The recorded-`w` formulation reads `vold = x[ℓ]` as a plain Float64, so
`∂I/∂x_lim` is deliberately dropped and those columns are *empty*: the
augmented matrix is block lower-triangular `[[A, 0], [∓1, I]]`, its Schur
complement is trivially `A`, and KLU's BTF preprocessing exploits the
structure automatically — the extra unknowns cost O(L) memory plus a
back-substitution. This is not free lunch: it is bought with the
vold-as-per-iteration-constant approximation, i.e. exactly SPICE's classical
Jacobian, and the corrector overwrites the lim slots anyway. Benchmark cost
of the approximation: ~1 iteration vs the paper-pure pilot.

For the upstream flagship (paper-pure, refine functions as parameters):
full-solve of the augmented system is the correct baseline — fill-reducing
orderings pivot the lim variables early (3-entry rows, unit diagonal) and
implicitly perform the paper's elimination. The explicit `PCNRDescent`
Schur descent is a performance option for contexts that must preserve the
exact n×n legacy pattern (existing symbolic factorizations,
preconditioners) or dense solves, not a correctness requirement.

### Why every other solver still works, unchanged

For solvers with no corrector phase (transient IDA/FBDF/Rodas, the
NonlinearSolve fallback chain), the augmented system reduces to the
original one: the device conductances are in the node-node block as always,
the `x_lim` columns are empty except for their row diagonals (block
lower-triangular Jacobian), and the linear `g_lim` rows are solved exactly
by every Newton step, so `x_lim` tracks `V_branch` and
`w = fn(V, V) = V` — the limiter is inert and the MNA iterates are the
unlimited ones. Same trajectory, same fixed point, `L` extra rows of
bookkeeping. Nothing regresses; the corrector is pure upside where it runs.

### Cost

One row/col per limited junction (diode: 1, BSIM4: 9), ~5 extra nnz per
junction, no dual widening (device evaluation at a scalar `x[ℓ]`), one
`Vector{Any}` of refine specs consulted only in the corrector (L iterations
of dynamic dispatch per Newton step — noise).

### Transient, AC, initialization

- **Transient**: limiting variables are algebraic (no `C` entries), so
  `detect_differential_vars` (computed from `C` rows,
  `src/mna/solve.jl:1675-1712`) marks them automatically; IDA and the mass-matrix ODE solvers need no
  changes. Between timesteps, the integrator's own predictor extrapolates
  `x_lim` along with everything else — limiting is then per-NR-iteration
  within each step, as in SPICE. Nothing in OrdinaryDiffEqNonlinearSolve or
  Sundials needs to be touched for v1.
- **AC** (`mode=:ac`): linearization at the DC OP; `g_lim` rows are
  well-conditioned and inert.
- **Initialization**: `u0 = zeros` keeps `x_lim = 0 = V_pn` — consistent.
  SPICE-style `vcrit` seeding of junction voltages is a Phase-2 nicety (an
  init-value argument on the allocation call, used to seed `u0` in
  `dc_solve_with_ctx`).
- **Tolerances**: `g_lim` rows are voltage-type; the per-class abstol
  NamedTuple (`(vntol=..., iabstol=..., chgtol=...)`, `src/sweeps.jl:521`,
  `MNA.state_abstol` at `src/mna/build.jl:262-271`) gains a limit-variable
  class mapped to `vntol`.

## Corrections to the previous revision

For anyone diffing against the old research artifact:

1. **Stamping example was wrong.** Its §1.4 `stamp_mna!` stamped the diode
   conductance node-to-node *and* claimed it as `J_MNA/lim` — contradicting
   the paper's own worked example (and double-counting). In the paper-pure
   formulation the conductances live only in the `J_MNA/lim` columns.
2. **`DeviceRegistry` does not fit Cadnip.** There are no persistent device
   objects to register; limiting metadata belongs in the context
   (`MNAContext`/`DirectStampContext`), mirroring charge variables.
3. **`pnjlim` pseudo-code was garbled** (mixed branches, bogus `log1p` form).
   Canonical implementations already exist in-repo as VA analog functions.
4. **`IDA(nlsolve=NonlinearSolveAlg(...))` is not a thing.** IDA is a C
   solver; `NonlinearSolveAlg`/`NLNewton` are OrdinaryDiffEq-only (verified
   in `OrdinaryDiffEqNonlinearSolve/src/type.jl`).
5. **Stale line numbers throughout** — corrected reference table below.
6. Its `PCNRSolver` bypassed `_dc_solve_with_fallbacks`; any custom loop must
   slot into that chain, not replace it.

## Implementation plan (all inside Cadnip)

### Phase 1: context plumbing + native `Diode` pilot — **implemented**

End-to-end limiting on the handwritten `Diode` before touching codegen.
As built:

1. `src/mna/context.jl`: `LimitIndex <: MNAIndex`; `resolve_index` maps it to
   `n_nodes + n_currents + n_charges + k`; `system_size`,
   `reset_for_restamping!`, `clear!`, `show` extended;
   `alloc_limit!(ctx, name, p, n; refine=spec)` storing `limit_names`,
   `limit_branches`, `limit_specs` (mirror of `alloc_charge!`, including the
   component-based name variants).
2. `src/mna/value_only.jl`: `limit_pos` counter + `n_charges` field on
   `DirectStampContext` (needed to resolve `LimitIndex`); counter-based
   `alloc_limit!`.
3. `src/mna/precompile.jl`: `LimitIndex` branch in the deferred-b resolution;
   `CompiledStructure` carries `n_limits` + `limit_specs`.
4. `src/mna/build.jl`: `MNAData.limit_names`/`n_limits`; SII lookup so
   `sol[:D1_vdlim]` works; `state_abstol` maps limit variables to `vntol`.
5. `src/mna/devices.jl`: `pnjlim` (verbatim port of `DEVpnjlim` from
   `diode.va`, value + `limited` flag), `PNJunctionLimit` refine spec,
   `refine_limit`, `limit_initial_value`; `Diode` gained `limit::Bool=true`
   and stamps the paper-pure form (evaluation at `x[ℓ]`, conductance in the
   lim column, linear `g_lim` row). Exponent clamped at 80 with linear
   extension (limexp-style) so wild intermediate iterates stay finite.
6. `src/mna/solve.jl`: `_dc_pcnr_newton` + tier-0 wiring in
   `_dc_solve_with_fallbacks`.
7. `test/mna/pcnr.jl`: pnjlim unit tests, plumbing, matrix structure,
   fixed-point invariance, stiff-chain convergence, transient smoke.

Measured (cold start from zeros, abstol=1e-10): 5V/1k rectifier and the
50V/1k 3-diode series chain both converge in **7 iterations**; limited and
unlimited solutions agree to ~1e-15. For reference, plain Newton takes 65+
on these and the best trust-region methods 15-27 (see `benchmarks/pcnr/`).

Getting to 7 required two mechanisms beyond the bare recorded-`w` corrector,
both lifted from how real SPICE stacks behave (without them the counts were
17-18 — an honest reproduction of ngspice's *cold-start crawl*, where the
evaluation voltage climbs in ~2·vt steps while the node waits at the source
voltage):

1. **initjct** (ngspice `MODEINITJCT`): on the first stamping of a cold
   start, `limit!` bypasses the limiter and evaluates at the variable's
   `init` seed (vcrit for junctions) — signalled by an `initjct` flag on the
   stamping context, armed by `_dc_pcnr_newton` for exactly one pass.
   Seeding `x_lim` alone is useless: `pnjlim(vnew=0, vold=vcrit) = 0`
   because pnjlim trusts a small `vnew`; the bypass is what ngspice's
   MODEINITJCT branch does in every device's C load function.

   *Scoping (design-reviewed):* only the solver may set the flag, because
   initjct encodes solver knowledge-state ("no iterate history yet"), not
   circuit state. Value heuristics ("bypass when `vnew == 0`" or
   `vnew < vcrit`) are provably wrong: those are legitimate operating points
   (reverse-biased junctions, the graetz zero-crossing diodes, unbiased
   junctions), and a value-triggered bypass would re-fire every iteration
   for any junction whose true solution rests there, resetting `x_lim` to
   vcrit forever — non-convergent by construction. Transient needs no flag
   interaction at all: warm starts have `vnew ≈ vold`, which makes pnjlim a
   no-op on its own; `x_lim` carries the previous timestep's value through
   the DAE state, playing the role of ngspice's MODEINITPRED vold. Placement
   note (corrected 2026-07): an earlier revision suggested moving the signal
   to `MNASpec` and exposing it as `$simparam("initjct", 0)` — but no model
   reads any "initjct" simparam. What the VADistiller models actually read is
   `$simparam("iniLim", -1)` inside `initialize_limiting()` (all 13 limited
   models, e.g. `diode.va:370`), so Phase 2 keeps the flag on the stamping
   context and special-cases `iniLim` in `$simparam` codegen to read it (see
   Phase 2 below). No spec-modified `CompiledStructure` pass is needed.

   *Convergence-tail detail:* the recorded-`w` corrector leaves a one-step
   lag in the `g_lim` rows (`x_lim = V` from the previous iterate), so on
   convergence the loop adopts the *current* stamping's `limit_w` lag-free
   and re-verifies — otherwise fast solves hand transient initialization a
   state whose residual (~|ΔV_last|) fails CheckInit's tighter tolerance.
   No extra state is needed: reaching the check implies the limiter was
   inert on that stamping, and pnjlim's pass-through returns `vnew`
   bit-exactly, so `limit_w` already equals the branch voltages.
2. **Evaluation-anchored companions** (ngspice convention ≡ OSDI/OpenVAF
   `lim_rhs`): when the device evaluates at `w ≠ V_probe`, its companion
   must be linearized around `w` — `I ≈ I(w) + G_d·(V − w)`, full `G_d` at
   the node positions — not around the probe. Probe-anchoring (what the AD
   contribution path does naturally) injects `I(w)` as a phantom current at
   the probe voltage; with a vcrit seed that reads as "18mA at 0V" and
   drives the node *negative*. Chain-rule AD anchoring is also consistent
   but keeps the damped `∂w/∂vnew` in the Jacobian, which is exactly what
   produces the 17-iteration crawl. This is why OSDI has `lim_rhs`; Phase 2
   codegen must implement the same correction
   (`Δb = Σ ∂I/∂w_k · (V_probe,k − w_k)` for each limit-replaced voltage).

The earlier paper-pure pilot's 5-6 iterations are thus fully explained and
recovered: its advantage was never the corrector-on-proposal placement, it
was that evaluating *at* `x_lim` made the vcrit seed effective and the
anchoring trivially consistent. Same behavior now lives inside the unified
`limit!` API.

### Phase 2: `$limit` codegen in vasim.jl — refined plan (researched 2026-07)

**The runtime side is already built and proven.** The `limit!` primitive
(see "Formulation" above) is exactly what `$limit(V(p,n), fn, args...)`
lowers to, and the native `Diode` already goes through it — so Phase 2 is
*pure codegen*: no new solver or context mechanisms, and (verified) no
changes to `contrib.jl`/`context.jl`/`value_only.jl`/`solve.jl` at all,
because generated VA `stamp!` methods inline all of their own stamping
(dual extraction `src/vasim.jl:2907-2948`, Ieq companion `:3055-3078`) and
never route through `stamp_contribution!`.

Trace check (rectifier from zeros): iter 0 stamps at `w₀ = pnjlim(0,0) = 0`;
predict pushes `V → 5` but correct resets `x_lim ← w₀ = 0`; iter 1 computes
`w₁ = pnjlim(5, 0) = vt·ln(5/vt) ≈ 0.14` — the limiter fires exactly as in
ngspice.

#### Verified facts that reshape the earlier sketch

1. **initjct seeding is deferred (measured: it must wait for Option 2).**
   All 13 limited models ship `initialize_limiting()` *with* the MODEINITJCT
   vcrit-seed branch emitted (e.g. `diode.va:367-381` and the seed at
   `:738-740`), gated on `$simparam("iniLim", -1)`, so no model patch or
   upstream ask is needed to *reach* the seed. But the seed as the models
   write it is `load_vd = DIOtVcrit` — a plain constant assignment that
   **drops the AD partials**. Under Option 1 (below), the device then stamps
   *zero* conductance on the seeded pass, which is singular for
   series-junction topologies (the 3-diode chain floats its internal nodes).
   Native `limit!` avoids this because its bypass seeds while *keeping*
   passthrough partials (`vnew − value(vnew) + init`, `devices.jl:1206`);
   reproducing that for VA models needs the w-anchored dual the site builds
   in Option 2, not the model's own constant seed. So the Option 1 impl
   leaves `$simparam("iniLim")` returning **0** (special-cased in the
   `$simparam` handler at `src/vasim.jl:1135-1154`, seed off) and takes the
   honest cold-start limiter crawl instead. Wiring `iniLim` to `ctx.initjct`
   becomes correct once Option 2's passthrough-dual return is in place.
   The LRM string form `$limit(V(p,n), "pnjlim", vte, vcrit)` (simulator-
   owned limiter, covered by `limit!`'s own initjct bypass) stays future
   work — no in-repo model uses it.

2. **Simparam inventory** (grep over all 13 models; every call carries a
   default, so unmatched names never error):

   | simparam (default) | uses | status |
   |---|---|---|
   | `tnom`, `gmin`, `reltol`, `vntol`, `abstol` | 64 | resolve against existing `MNASpec` fields |
   | `scale` (1), `defw`/`defl` (1e-4), `defas`/`defad` (0), `epsmin` | 62 | defaults correct (ngspice defaults) |
   | `oldlimit` (0) | 12 | default 0 = ngspice `CKTfixLimit` off = modern fetlim+limvds path |
   | `iteration` (10) | 13 | only read in the `initialize_limiting()` heuristic; dead while `iniLim` returns 0 |
   | **`iniLim` (−1)** | 13 | special-cased to **0** for now (seed off); wire to `ctx.initjct` in Option 2 (above) |
   | `sourceScaleFactor` (1.0) | 1 | vdmos thermal branch; ngspice's srcFact — optionally map to `_mna_spec_.srcFact` |

3. **Per-branch state, per-site lowering, no OldGet/NewSet classification.**
   State (limit variable, `vold`, `g_lim` row) is keyed per (instance,
   probe branch) — allocated hoisted/unconditionally, one slot per branch.
   Every `$limit` site on that branch lowers uniformly: call its limiter fn
   as `fn(vnew_dual, vold, user_args...)` via the existing VA-function
   machinery (`:1206-1237`), `record_limit_w!(extract_value(result))`
   (last record wins — OldGet's `vold` record is overwritten by NewSet's),
   return per the dual-semantics choice below. Site counts are OldGet/NewSet
   pairs everywhere: diode 2/1, bjt 6/3, mos*/bsim3v3/vdmos 8/4, jfet 4/2,
   bsim4v8 18/9.

#### Jacobian dual semantics — the option space

The design hinge is what dual a `$limit` site returns, because its partials
become the stamped conductance. The inline `DEVpnjlim` output `w_raw`
carries chain-ruled partials `∂w/∂V` (≈1 inactive, ≪1 when compressing).
Coherent (G, RHS-anchor) combinations:

- **Option 1 — chain-ruled (damped) G + probe anchor.** Return `w_raw`
  as-is. `G = dI/dw·∂w/∂V` is the true composite derivative, so the
  existing probe-anchored `Ieq` (`:3058-3062`) is already consistent.
  ~3-line change, no dual widening, no Δb. Cost: hard limiting collapses
  the stamped conductance → the measured 17-18-iteration cold-start crawl
  (see the "Measured" notes above). Correct, just slower.
- **Option 2 — passthrough (undamped) G + w anchor** (ngspice/OSDI
  `lim_rhs`). Return a fresh dual `Dual(w, ±1 at the probe-node slots)`:
  `G = dI/dw` full-strength (SPICE's `gd`), i.e. linearization
  `I ≈ I(w) + G·(V − w)`. The probe-anchored companion is then wrong by a
  phantom current `G·(V_probe − w)`, so the RHS needs
  `Δb = Σⱼ (∂I/∂wⱼ)·(V_probe,ⱼ − wⱼ)`. Getting `∂I/∂wⱼ` requires one
  extra dual slot per site (widen the `JacobianTag` duals from
  `n_all_nodes` to `W = n_all_nodes + S`; site j's return carries `+1` at
  slot `n_all_nodes+j`, read only for Δb, never stamped into G) — the node
  partial is the *sum* of direct-V dependence (series-R path, gmin terms)
  and through-w dependence, and Δb needs the through-w part isolated. This
  dual shape mirrors the initjct bypass already inside `limit!`
  (`devices.jl:1206`). Cross-check against the native diode:
  `Ieq = I(w) − G·V + G·(V − w) = I(w) − G·w` — exactly
  `stamp_limited_companion!`.
- **Option 3 — chain-ruled G + w anchor**: what native `Diode` does today
  (generic `pnjlim` under AD + `stamp_limited_companion!`); impure but
  benign — damping only bites far from the solution, and with the initjct
  seed the limiter is barely active after the first pass; 7 iterations
  measured on the pilot.
- Undamped G + probe anchor is the invalid fourth combination (the phantom
  "18 mA at 0 V" failure in the Measured notes).

**Sequencing: land Option 1 first** (verifies VA limiting end-to-end with
the PCNR corrector at minimal complexity), **then upgrade to Option 2**
(dual widening + Δb) with `benchmarks/pcnr` iteration counts as referee.
The two remain a one-line A/B at the site-return emission — which the
"damped Jacobian" risk item below explicitly asks to keep.

#### Effort and steps

**Effort assessment**: this is the most delicate file in the repo
(positional-counter discipline, dual handling, per-branch site grouping
across a 3.7k-line codegen; BSIM4 has 18 sites/9 branches to validate).
Multi-session, careful work — not delegable to a quick agent pass. No type
piracy anywhere: it's all Cadnip-owned codegen and context plumbing.

7. `src/vasim.jl` changes. `MNAScope` (~`:551-575`) gains `limit_branches`
   (ordered unique probe branches), `limit_sites` (one per site in lowering
   order) and a `cond_depth` counter (incremented around conditional/loop/
   case body lowering). The `$limit` handler (replace `:1192-1198`) errors
   at codegen time for non-`V(p,n)` probes, string-form limiters, unknown
   fns, or a site under a runtime conditional (`cond_depth > 0` would
   desync positional counters); otherwise find-or-append the branch, push
   the site, and emit the uniform lowering above (reusing the VA-function
   arg-marshalling at `:1206-1237`). `$simparam` gains the `iniLim`
   special case. Assembly (`generate_mna_stamp_method_nterm`): hoisted
   limit preamble spliced between `$internal_node_alloc` (`:3426`) and
   `$instance_stamp_calls` (`:3429`) — per branch: `alloc_limit!(...;
   init=0.0)` (the vcrit seed comes from the model's own iniLim branch,
   unlike native `Diode`'s `init=vcrit`), the `vold = _mna_x_[li]` read,
   and the 3-entry `g_lim` row. For Option 2 additionally: per-site
   partials constants, widened `dual_creation` (`:3176-3183`), `dIdW/dqdW`
   extraction in the three `branch_stamp` type branches (`:2907-2948`),
   and Δb terms in the Ieq (`:3058-3062`), charge-constraint
   (`:3036-3040`, before the `CHARGE_SCALE` multiply) and two-node voltage
   (`:3325-3357`) companions. A module with no `$limit` sites generates
   byte-identical code to today.
8. Integration tests. `test/mna/pcnr.jl` additions: sp_diode structure
   detection (5 random-x passes), anchoring unit test against
   hand-computed `pnjlim` (crafted `x` with `V_probe = 5`,
   `x_lim = vold = 0.6`), initjct seed test (`limit_w ≈ vcrit`, no phantom
   current), rectifier/chain3 convergence counts, transient smoke.
   `test/mna/vadistiller.jl` / `test/mna/vadistiller_integration.jl`
   regression — audit size-sensitive assertions (system sizes grow by
   n_limits for every limited model). Validation ladder: diode → bjt
   (`test/mna/astable_bjt_test.jl`, 3 limit vars/instance) → mos1 (fetlim
   cross-branch: vds limited using vgs — fine, the `g_lim` row stays
   3-entry, cross couplings ride the node conductances via AD); bsim4v8
   compile+DC smoke only (W = n_all_nodes+18 is the compile-pressure worst
   case; the `noinline` escape hatch at `:3412` exists).
9. Benchmarks: `benchmarks/pcnr/dc_newton_iterations.jl` gains VA twins of
   its four circuits (stamping `sp_diode(is=76.9e-12, n=1.45)` directly,
   pattern per `test/mna/vadistiller_integration.jl:211-221`) alongside
   the native rows — native `limit=false` remains the unaugmented baseline
   VA cannot express; the NonlinearSolve methods run on the VA circuits'
   augmented system (limiter provably inert under correctorless solvers,
   see "Why every other solver still works"). `benchmarks/vacask` remains
   the acceptance gate: NR iteration counts and wall-clock vs. VACASK
   (which runs the same models *with* limiting active), wpd plots for
   `mul` — limiting should cut iterations and/or fallback invocations
   without accuracy regressions.

### Phase 3 (optional, evidence-driven): explicit corrector + full coupling

Only if v1 convergence is insufficient on real cases:

10. Full `vold` coupling (option (b)): extra dual slot per limit variable,
    real 4-block Jacobian, solved either as one sparse factorization or via
    the paper's Schur elimination inside a Cadnip-owned Newton loop
    (`_dc_pcnr_newton`, slotted as the *first* tier of
    `_dc_solve_with_fallbacks` before the polyalgorithm).
11. Explicit corrector phase + per-device `x_lim` initialization (vcrit
    seeding), giving the paper's "never more iterations than limiting"
    guarantee.

## Deferred upstreaming plan

Nothing above requires forking or patching dependencies. Strategy: PCNR is a
published algorithm (Sandia/Springer), which SciML historically receives
well — but the corrector requires a cross-cutting API change (a post-step
hook in the solve loop), and most of PCNR's value lives in the device
coupling (limiting functions, `$limit` state, stamping), which is
Cadnip-specific and stays local. So: **implement locally first, open an
upstream API discussion early** (an RFC issue describing the corrector hook,
citing the paper and the working Cadnip implementation), and only PR the
generic pieces once the hook API is agreed. The generic pieces are small —
a Schur-complement descent over an identity block and a `refine`-style
iterate callback; everything else is not upstreamable by nature.

Dependency status (verified 2026-07 by fresh resolution on Julia 1.12):
Manifests are gitignored and CI resolves fresh, and the resolver already
picks **NonlinearSolve v4.20.2** (NonlinearSolveBase v2.33.0,
NonlinearSolveFirstOrder v2.1.3, SciMLBase v3.33, OrdinaryDiffEq v7.1,
DiffEqBase v7.6, Sundials v6.2). The old `NonlinearSolve = "3, 4"` compat was
therefore never CI-tested on the 3.x branch; it has been narrowed to `"4"`.
The descent API (`AbstractDescentDirection`, `InternalAPI.init/solve!`,
`DescentResult`) is confirmed present at the resolved NonlinearSolveBase
version, so nothing blocks prototyping `PCNRDescent` against the shipped
packages. Once v1/v2 evidence exists in Cadnip, the reusable pieces map
upstream as follows:

| piece | target | shape | status/blockers |
|---|---|---|---|
| Post-step correction hook + PCNR algorithm | **NonlinearSolveBase / NonlinearSolveFirstOrder** | The single required API is a `correct!(u_proposed, u_prev, p)` hook run between the Newton update and the next residual evaluation. The pilot-vs-SPICE variant choice is *invisible to the solver* — it lives inside the callback (algorithm-owned refine closures vs a copy from a problem-recorded buffer). Ship the published algorithm as the hook's flagship instantiation: `PCNR(limit_idxs, refine_fns; inits)` per Aadithya et al., documented and tested against the paper's diode example — the citable, self-contained shape that's easiest to land. Cadnip consumes the same hook with its recorded-`w` callback (production models can't expose refine functions; their limiter is inline — which is itself the argument that the hook, not the named algorithm, is the right API boundary). | Notably, the recorded-`w` formulation **shrank** the upstream footprint: its Jacobian's lim columns are empty (block lower-triangular), so plain `NewtonDescent` is already optimal and the previously-planned `PCNRDescent` Schur descent is only relevant to the paper-pure variant — keep it as an optional companion to the flagship, not a requirement. Compat already narrowed to `NonlinearSolve = "4"`. **Termination-mode caveat (2026-07, found while evaluating whether `_dc_pcnr_newton` could migrate to a hypothetical upstream hook once it exists):** `correct!` mutates `u_proposed` only *after* the Newton update, so a termination check against the residual `fu` computed at the *start* of that same iteration is systematically one correction behind — exactly the "x_lim lags one iterate behind" lag `_dc_pcnr_newton`'s settle step (§ "Getting to 7") works around. That settle step is therefore not a Cadnip-specific wart; it's the generic price of pairing a post-update corrector with residual-based termination, and any hook consumer using `AbsNormTerminationMode`-style checks will hit it too. An increment-based termination check (comparing successive *corrected* iterates, `‖u_proposed − u_prev‖`) sidesteps it — this is, in effect, what the abandoned in-step bridge did (`ndz = ‖z − nlcache.u‖` in `OrdinaryDiffEqNonlinearSolve`'s stage loop), and no settle-equivalent was needed there. So: `_dc_pcnr_newton` is not a drop-in replacement for the upstream algorithm the moment it lands — either the upstream `PCNR` algorithm ships with (or documents needing) increment-based termination, or a Cadnip caller wraps `solve!` with its own settle-and-reverify pass, same as today. No pressure to actually do this migration — `_dc_pcnr_newton` works, is tuned to ~7 iterations, and is fully tested; this is a note for whoever attempts it. |
| Corrector inside *warm-started* implicit ODE steps | *(dropped)* | Not pursued upstream — tried locally via `AbstractNonlinearSolveAlgorithm` + `NonlinearSolveAlg` and found structurally inert for every step tested (see "Scope" section): the predictor's Lagrange-interpolant extrapolation (verified against FBDF source) is a componentwise-identical linear functional of past history, and `x_lim`/`V_branch` are numerically identical throughout that history, so the predicted values are forced equal too — `vold ≈ vnew` before the stage solve's first iteration, corrector never fires, independent of step size. Not a `NonlinearSolveAlg`-specific limitation — any predictor-based in-step wiring hits the same wall, *for a warm-started step*. | N/A — no further work planned for the warm-started case. The untested adjacent case (a genuine discontinuity-triggered DAE reinitialization, which is not a linear-history extrapolation and so isn't covered by this argument) is flagged as an open question in "Scope", not ruled out — see "Open question: discontinuity-triggered reinitialization". |
| Limiting in IDA | **Sundials.jl** | not possible — IDA's Newton loop is C and unhookable | v1's formulation (limiting fused into the residual/Jacobian) is the *only* PCNR variant that works under IDA. This is a strong argument for keeping the v1 formulation permanently, even after upstream hooks exist. |
| `$limit` semantics & experience | **VADistiller / VACASK** (Bürmen, codeberg) | no model changes needed — the models drive the design; report the per-branch state-keying convention and any model bugs found | coordination only |
| `$limit` parsing | **NyanVerilogAParser.jl** | already parses (vasim receives the args today) | none expected |

Stays in Cadnip permanently: the MNA limiting-variable plumbing, the vasim
codegen, `pnjlim` for native devices, tests/benchmarks.

## Risks and open questions

- **Conditional `$limit` sites** would desynchronize DirectStampContext's
  positional counters; enforce hoisting (error at codegen time if a site is
  under a runtime conditional). All in-repo models are unconditional.
- **Cross-branch limiters** (`DEVfetlim` limits vds using vgs): fine — the
  limiter runs inline with AD, so `∂w/∂V` picks up all node couplings; the
  `g_lim` row just gains entries for every node the limiter reads.
- **Structure detection**: the 5-pass random-`x` detection
  (`_detect_structure`, `src/mna/solve.jl:697-720`) must produce identical
  stamp counts regardless of which limiter branch executes — guaranteed by
  hoisted allocation + branch-free stamping of the `g_lim` row.
- **Damped Jacobian in deep limiting**: option (a)'s node conductances carry
  the factor `∂w/∂vnew` (≤1, small when limiting hard). If this ever yields
  near-singular rows, fall back to stamping `G(w)` un-damped with the
  matching companion shift (ngspice's convention) — a one-line change in the
  dual handling, worth an A/B test in Phase 1.
- **Residual norms shift**: extra rows change `norm(F)`; the DC solve's
  scalar `abstol` and the per-class abstol work
  (`doc/dc_newton_termination.md`) should treat limit rows as voltage-class
  from day one.
- **`alter`/sweep caching**: limit variables change `system_size`; anything
  caching sizes per builder (CircuitSweep) just sees the new size at
  `compile_structure` time — no special handling expected, but verify in
  `test/sweep.jl`.

## Corrected code reference table

| Component | File:lines (verified 2026-07) |
|---|---|
| `MNAContext` struct | `src/mna/context.jl:156-217` |
| `alloc_charge!` (pattern to mirror) | `src/mna/context.jl:556-575` |
| `resolve_index` | `src/mna/context.jl:393-396` |
| Hoisted stamping (`get_G_idx!` etc.) | `src/mna/context.jl:800-875` |
| `DirectStampContext` | `src/mna/value_only.jl:42-87` |
| `CompiledStructure` | `src/mna/precompile.jl:75-118` |
| `compile_structure` | `src/mna/precompile.jl:306-432` |
| `fast_rebuild!` | `src/mna/precompile.jl:470-526` |
| `_dc_newton_compiled` | `src/mna/solve.jl:409-445` |
| `CedarRobustNLSolve` | `src/mna/solve.jl:330-339` |
| gmin/source stepping fallbacks | `src/mna/solve.jl:467-657` |
| `MNASpec` (gmin, gshunt, srcFact, vntol...) | `src/mna/solve.jl:56-69` |
| `CedarDCOp` + `initialize_dae!` | `src/mna/dcop.jl:74-82, 160-262` |
| Native `Diode` stamp | `src/mna/devices.jl:1158-1219` |
| `stamp_contribution!` (companion pattern) | `src/mna/contrib.jl:438-475` |
| `evaluate_contribution` (dual extraction) | `src/mna/contrib.jl:507-572` |
| VA node dual creation (identity partials) | `src/vasim.jl:3176-3183` |
| `$limit` no-op | `src/vasim.jl:1192-1198` |
| `$discontinuity` no-op | `src/vasim.jl:2057-2059` |
| `limexp` clamp | `src/va_env.jl:36` |
| `dc!` / `tran!` / per-class abstol | `src/sweeps.jl:435, 554-616, 521-611` |
| `DEVpnjlim` + `$limit` sites (VA) | `models/VADistillerModels.jl/va/diode.va:383-440, 726-741` |
| BJT / BSIM4 limit sites | `bjt.va:1040-1067`, `bsim4v8.va:5992+` |
