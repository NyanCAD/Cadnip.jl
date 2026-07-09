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

## v1 design: inline limiting with state in `x` (recommended)

Rather than the paper-pure flow (devices evaluate at `v_lim`, separate
corrector phase, Schur solve), v1 keeps the models' own inline limiter calls
and only moves the state into `x`. This turns out to *be* PCNR with a
one-sided coupling approximation — and it requires **zero changes to any
solver**: DC (`CedarRobustNLSolve` and the stepping fallbacks), transient
(IDA, FBDF, Rodas5P), sweeps and `alter` all see just a slightly larger
system.

### Formulation

Augment the state: `x = [V₁..Vₙ, I₁..Iₘ, q₁..qₖ, v_lim,1..v_lim,L]`.

During stamping at `x`, for limit site(s) on branch `(p, n)` with limiting
variable index `ℓ`:

- `vnew` = `V_p − V_n` as today's `JacobianTag` dual (node partials only);
- `vold` = `x[ℓ]` read as **plain Float64** (no dual — see "Jacobian
  treatment" below);
- the model computes `w = limfn(vnew, vold, args...)` inline (its own
  `DEVpnjlim`), and evaluates all currents/charges at `w`. AD flows the
  node partials through `limfn`, so `J_MNA/MNA` keeps its usual node-anchored
  conductance entries (evaluated at the limited point, damped by
  `∂w/∂vnew`) — **the sparsity pattern of the MNA block is unchanged**;
- the simulator stamps one extra row, the limiting equation
  `g_lim,ℓ = x[ℓ] − w(x_MNA) = 0`, in Newton-companion form (exactly the
  pattern of `stamp_contribution!`, `src/mna/contrib.jl:438-475`):
  `G[ℓ,ℓ] = 1`, `G[ℓ,p] = −∂w/∂V_p`, `G[ℓ,n] = −∂w/∂V_n`,
  `b[ℓ] = G_row·x₀ − g_lim(x₀)`.

### Why this is PCNR

With `vold` un-dualed, `J_MNA/lim = 0` and `J_lim/lim = I`, so the augmented
Jacobian is block lower-triangular:

```
J = [ J_MNA/MNA      0 ]        Δx_MNA solved from J_MNA/MNA alone  (predictor,
    [ J_lim/MNA      I ]        Schur complement is trivially J_MNA/MNA)
                                Δx_lim = -(g_lim + J_lim/MNA·Δx_MNA)  (paper's step (ii))
```

The linear solve stays MNA-sized *by construction* — the paper's Schur
elimination is free here. And the corrector is fused into the `g_lim` row:
Newton's update gives `x_lim,i+1 = w_i + ∂w/∂V·ΔV`, i.e. a first-order
estimate of `refine(x_i, x_{i+1})` — the limited voltage at the new iterate.
The fixed point is unchanged: at convergence `x[ℓ] = w`, and since
`pnjlim(v, v) = v`, `w = vnew`, so the solution solves the original
unlimited system exactly. The `limited` flag / `$discontinuity(-1)`
"iterate again" signal is subsumed by convergence checking: while limiting is
active, `g_lim ≠ 0` keeps the residual norm up automatically.

In practice the solver sees a *full* sparse system of size `n_MNA + L` (KLU
factors the block-triangular structure cheaply; nnz grows by ~4 per limited
branch, one row/col per junction — BSIM4: 9, diode: 1). No dual widening: the
generated per-module dual width stays `n_all_nodes`, because `vold` is a
Float64. Cost is negligible.

### Jacobian treatment options (for the record)

| option | `vold` | coupling | notes |
|---|---|---|---|
| **(a) v1, chosen** | Float64 | `J_MNA/lim = 0`, block triangular | free Schur; matches SPICE's "vold is a constant this iteration" |
| (b) full AD | extra dual slot per lim var | full 4-block Jacobian | exact Newton on the smoothed system; needs dual width `n_nodes + n_lim` (BSIM4: +9) and a real Schur solve or full factorization; try only if (a)'s convergence disappoints |
| (c) paper-pure | evaluate at `v_lim` itself, explicit corrector | conductances *move* to `J_MNA/lim` columns | without Schur, `J_MNA/MNA` loses junction conductances (zero diagonals on diode-only nodes); with Schur, reproduces (a)'s matrix plus corrector flexibility. This is the upstreaming shape, not the v1 shape. |

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

### Phase 1: context plumbing + native `Diode` pilot

Goal: end-to-end limiting on the handwritten `Diode`
(`src/mna/devices.jl:1158-1219`) before touching codegen.

1. `src/mna/context.jl`:
   - `LimitIndex <: MNAIndex` (like `ChargeIndex`); `resolve_index` maps it to
     `n_nodes + n_currents + n_charges + k`; extend `system_size`,
     `reset_for_restamping!`, `clear!`, `show`.
   - `alloc_limit!(ctx, name::Symbol, p::Int, n::Int) -> LimitIndex` storing
     `limit_names` and `limit_branches` (mirror of `alloc_charge!`,
     including the component-based name variants for DirectStampContext
     compatibility).
2. `src/mna/value_only.jl`: positional `limit_pos` counter on
   `DirectStampContext` (mirror of `charge_pos`); `alloc_limit!` returns the
   pre-resolved index.
3. `src/mna/contrib.jl` (or a small new `src/mna/limit.jl`):
   `stamp_limit_row!(ctx, ℓ, p, n, w_dual, x)` emitting the companion-form
   `g_lim` row described above.
4. `src/mna/build.jl` / SII: limit names appear in solution indexing
   (`sol[:D1_vdlim]`); `state_abstol` gains the limit class.
5. Opt-in limiting for `Diode`: read `vold`, apply a Julia `pnjlim`
   (port of `DEVpnjlim` — one small function, unit-testable), stamp the row.
6. `test/mna/pcnr.jl`: fixed-point invariance (solution identical with/without
   limiting on converging circuits), iteration-count reduction on a stiff
   diode chain (the `doc/dc_newton_termination.md` 3-diode case and the
   VACASK `mul` 50 V chain), Jacobian block structure checks.

### Phase 2: `$limit` codegen in vasim.jl

7. `src/vasim.jl` `$limit` handler (`:1192`):
   - group call sites by probe branch within the module; first site allocates
     (hoisted, unconditional — same rule as hoisted `get_G_idx!` stamping, so
     DirectStampContext counters stay synchronized; all VADistiller sites are
     unconditional today, but enforce/document the constraint);
   - emit `vold = _mna_x_[resolve(ℓ)]` (Float64, pre-dual read) and
     `w = fn(vnew, vold, args...)` via the existing VA analog-function call
     machinery (`all_functions`, handles the `inout limited` flag already);
   - after the analog block, stamp the `g_lim` row from the **last** site's
     returned value per branch (NewSet site for VADistiller models).
8. Integration tests: `test/mna/vadistiller.jl` additions — diode rectifier,
   the astable BJT case (`test/mna/astable_bjt_test.jl`), MOS sweeps; verify
   `dc!`/`tran!` results unchanged on already-converging circuits and
   fallback usage (gmin/source stepping) reduced on the difficult ones.
9. Benchmarks: `benchmarks/vacask` suite — NR iteration counts and wall-clock
   vs. VACASK (which runs the same models *with* limiting active), wpd plots
   for `mul`. This is the acceptance gate: limiting should cut iterations
   and/or fallback invocations without accuracy regressions.

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

Nothing above requires forking or patching dependencies. Once v1/v2 evidence
exists in Cadnip, the reusable solver pieces map upstream as follows:

| piece | target | shape | status/blockers |
|---|---|---|---|
| PCNR descent (Schur predictor, option (c)) | **NonlinearSolveBase / NonlinearSolveFirstOrder** | `PCNRDescent <: AbstractDescentDirection` with `InternalAPI.init(prob, alg, J, fu, u; ...)` / `InternalAPI.solve!(cache, J, fu, u, idx; ...) -> DescentResult` (API verified against master) | Cadnip's compat allows NonlinearSolve 3 *and* 4; the descent API is 4.x-only — pin `NonlinearSolve = "4"` first. The corrector needs a **post-descent hook** in `GeneralizedFirstOrderAlgorithm` that doesn't exist yet — open an upstream issue proposing a `correct!(u, u_prev, p)` callback so limiting composes with TrustRegion/LineSearch instead of replacing them. |
| Corrector inside implicit ODE steps | **OrdinaryDiffEqNonlinearSolve** | either a post-iteration hook in `NLNewton`'s `compute_step!`, or an `NLPCNR <: AbstractNLSolverAlgorithm` (verified: `NLNewton`, `NonlinearSolveAlg` exist; no hook today; the `relax` field is the nearest precedent) | Only needed for the *explicit-corrector* variant; v1's fused formulation needs nothing here. |
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
