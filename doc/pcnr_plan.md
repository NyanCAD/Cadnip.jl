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

### Formulation

Augment the state: `x = [V₁..Vₙ, I₁..Iₘ, q₁..qₖ, v_lim,1..v_lim,L]`
(`LimitIndex`, `alloc_limit!` — mirrors of `ChargeIndex`/`alloc_charge!`).

For a limited device on branch `(p, n)` with limiting variable `ℓ`:

- the device evaluates its currents **at `w = x[ℓ]`** (not at `V_p − V_n`),
  and stamps its conductance into the **`x_lim` column**:
  `G[p,ℓ] += G_d`, `G[n,ℓ] −= G_d`, with the matching companion
  `b[p] −= I(w) − G_d·w`, `b[n] += ...`. The diode contributes *nothing* to
  the node-node block — exactly the paper's Jacobian (Fig. 2);
- the simulator stamps the **linear** limiting row
  `g_lim = x[ℓ] − (V_p − V_n) = 0`:
  `G[ℓ,ℓ] = 1`, `G[ℓ,p] = −1`, `G[ℓ,n] = +1`, `b[ℓ] = 0` — so
  `J_lim/lim = I` identically;
- the refine spec (e.g. `PNJunctionLimit(vt, vcrit)`) is recorded at
  allocation time (`alloc_limit!(...; refine=spec)`) and carried through
  `MNAContext.limit_specs` → `CompiledStructure.limit_specs`.

### The PCNR loop (`_dc_pcnr_newton`, src/mna/solve.jl)

Runs as tier 0 of `_dc_solve_with_fallbacks`, only when `cs.n_limits > 0`:

1. **Initialize**: seed `x_lim` slots with `limit_initial_value(spec)`
   (vcrit for junctions — SPICE's junction seeding) when starting from zeros.
2. **Predict**: plain Newton step on the augmented system
   (`δ = G \ F`; `G` is the companion Jacobian, same machinery as the
   standard path).
3. **Correct**: `x_lim,k ← refine_limit(spec_k, proposed_k, previous_k)` —
   i.e. `pnjlim(vnew, vold, vt, vcrit)` — applied by the *simulator*,
   per the paper.
4. **Converge**: `‖F‖ < abstol`. The `g_lim` rows are part of `F`, so
   convergence implies `x_lim = V_branch` (no active limiting) — SPICE's
   "no limiting applied this iteration" condition falls out automatically,
   as does the `$discontinuity(-1)` iterate-again signal.

On failure it falls through to the unchanged standard chain
(`CedarRobustNLSolve` → gmin stepping → source stepping).

### Why every other solver still works, unchanged

For solvers with no corrector phase (transient IDA/FBDF/Rodas, the
NonlinearSolve fallback chain), plain Newton on the augmented system is
**step-for-step equivalent** to Newton on the original system: the Schur
complement of the linear `g_lim` rows is
`S = J_MNA/MNA − J_MNA/lim·J_lim/MNA`, which puts each `G_d` back at its
familiar node-node positions, and `g_lim = 0` along the whole trajectory
(it starts at 0 from zeros/DC-consistent states and the linear rows are
solved exactly). So: same iterates, same fixed point, ~`L` extra rows of
bookkeeping. Nothing regresses; the corrector is pure upside where it runs.
The augmented matrix is nonsingular exactly when the Schur complement (=
today's matrix) is, even for diode-only nodes whose node-node diagonal is
empty in the augmented form.

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

Measured on first bring-up (cold start from zeros, abstol=1e-10):
5V/1k rectifier converges in **5 iterations**, 50V/1k 3-diode series chain
in **6 iterations**; limited and unlimited solutions agree to ~1e-15.

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
| PCNR descent (Schur predictor, option (c)) | **NonlinearSolveBase / NonlinearSolveFirstOrder** | `PCNRDescent <: AbstractDescentDirection` with `InternalAPI.init(prob, alg, J, fu, u; ...)` / `InternalAPI.solve!(cache, J, fu, u, idx; ...) -> DescentResult` (API verified at the resolved NonlinearSolveBase v2.33.0 and on master) | compat narrowed to `NonlinearSolve = "4"` (no-op for resolution — see above). The corrector needs a **post-descent hook** in `GeneralizedFirstOrderAlgorithm` that doesn't exist yet — that's the major-API-change part: prototype the hook locally, open the upstream RFC proposing a `correct!(u, u_prev, p)` callback so limiting composes with TrustRegion/LineSearch instead of replacing them, and PR `PCNRDescent` only once the hook shape is agreed. |
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
