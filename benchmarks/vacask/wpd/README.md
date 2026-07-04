# Work-Precision Diagram (WPD) benchmark: Cadnip vs VACASK

The other benchmarks under `benchmarks/vacask/` measure **throughput**: they pin a
tiny fixed `dtmax` and relaxed tolerances and time the result. That bypasses the
adaptive local-truncation-error control entirely, so it rewards whichever solver has
the cheapest per-step cost and says nothing about *accuracy per unit of work*.

This benchmark is the complement. Each solver runs **adaptively** across a sweep of
tolerances, and we plot **error vs. runtime** (log-log) — a work-precision diagram,
following the [SciML `WorkPrecisionSet`](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/)
methodology. A higher-order BDF/Rosenbrock method that takes big accurate steps
should sit *below* a simple low-order method at tight tolerance; that separation is
exactly what the throughput benchmark hides.

## Running

```bash
# one command: Cadnip sweep + VACASK sweep + report with inline ASCII plots
julia --project=benchmarks benchmarks/vacask/wpd/run_wpd.jl          # all cases
julia --project=benchmarks benchmarks/vacask/wpd/run_wpd.jl rc graetz # a subset
```

It writes `out/wpd_results.md` — per-case tables plus inline **ASCII work-precision
diagrams** (UnicodePlots), so it renders directly in a terminal or a GitHub Actions
job summary (no image hosting needed). It also writes higher-quality **PNG/SVG plots**
(Plots.jl/GR, real color and vector curves) to `out/plots/<case>.{png,svg}` — these are
*not* embedded in the summary (Markdown there can't reference local files), only
uploaded in the CI artifact, so the job summary stays ASCII while the download gets
proper images of the same data. The CI `work-precision` job
(`.github/workflows/benchmark.yml`) runs this and publishes the report to the job
summary, and both the report and the PNG/SVG plots to the artifact.

VACASK is located via `$VACASK_COMMAND` or the cache populated by
`benchmarks/vacask/fetch_vacask.sh` (also used by `run_vacask.sh`); run that once, or
let CI do it. Without VACASK, the Cadnip curves and analytic goldens still work.

## Circuits

Driven, dissipative circuits with a stable forced response, so error degrades
gracefully with tolerance. Autonomous/digital circuits (ring oscillator, digital
multiplier) are deliberately avoided: their phase error accumulates without bound, so
the WPD curve saturates at O(1) and "falls off a cliff".

| Case     | What                                       | Golden |
|----------|--------------------------------------------|--------|
| `filter` | 3rd-order Butterworth LC ladder (linear)   | analytic (exact) |
| `rc`     | RC network driven by a pulse train (linear)| analytic (exact) |
| `graetz` | Graetz bridge full-wave rectifier (diodes) | VACASK (tight) |
| `mul`    | Diode voltage multiplier (stiff, diodes)   | Cadnip IDA (tight) |

`filter` is a smooth drive; `rc` adds sharp source edges (Cadnip gets `tstops` at the
pulse breakpoints, as SPICE engines break at source breakpoints internally); `graetz`
and `mul` add the diode nonlinearity, `mul` being markedly stiffer.

## Precision / golden reference

Error is the relative L2 (RMS) norm of the output node (SciML `error_estimate=:l2`),
evaluated at **each run's own output timepoints** against a *dense* reference
(analytic: 200k pts; VACASK/Cadnip-tight: dense). Measuring at native points matters:
a high-order solver takes large steps and emits few points, so interpolating *its*
output onto a fixed grid would penalise it for the interpolation rather than its real
accuracy — interpolating the dense reference onto the run's points is fair.

The golden is **pinned per case** in `config.json`
(`golden: analytic|vacask|cadnip|self`), chosen by what actually converges —
`run_wpd.jl` uses exactly that and errors if it fails (no silent substitution):

- `filter`, `rc`: **exact analytic** closed form (also avoids any "VACASK vs its own
  golden" ambiguity).
- `graetz`: tight VACASK run (fine `maxstep`); it completes and is cross-checked.
  (A tight Cadnip golden was tried too, for consistency with `mul`'s `self` mode
  below, but Cadnip's own IDA fallback ladder doesn't converge on this circuit at
  any tolerance from `1e-7` to `1e-11` - so `graetz` stays on the single VACASK
  golden, which has always worked fine here.)
- `mul`: **`self`** — Cadnip's curves are scored against a tight Cadnip IDA
  golden, VACASK's curve against its own tight VACASK golden, *not* a shared
  cross-simulator reference. See "Findings about VACASK" below for why: scoring
  VACASK against Cadnip's golden here previously produced a misleading "VACASK
  can't get more accurate than 1.3e-4" reading that turned out to be almost
  entirely the two simulators' independently-converged answers disagreeing with
  *each other*, not either one failing to converge.

**VACASK runs at its best, not its default — and "best" is picked per case, not
assumed.** VACASK defaults to trapezoidal (2nd order); the benchmark's global
default is `tran_method="gear" tran_maxord=5` (variable-order Gear/BDF),
configurable via `vacask_tran_method` / `vacask_tran_maxord` in `config.json`.
`rc` overrides this down to `tran_maxord=4` (see "Findings about VACASK" below
— order 5 has a confirmed controller bug on that specific circuit, reported
upstream as [issue #83](https://codeberg.org/arpadbuermen/VACASK/issues/83));
`filter`/`graetz`/`mul` keep the order-5 default, which is fine or best there.

Every plot also carries a second, fixed **`VACASK gear2`** series (2nd-order
Gear/BDF) alongside the case's own best-order curve, regardless of what that
case's default/override picks — per the maintainer, circuit simulators
historically stick to order ≤ 2 for A-stability, so it's worth seeing that
tradeoff directly rather than only ever plotting whichever order a case
happens to use. Both series are scored against the same golden.

## Solver families per case

Not a blanket linear/nonlinear split — viability was checked empirically per case
(`run_wpd.jl`'s `SOLVERS` dict), and a run that bails out early (retcode other than
`Success`, or one that doesn't reach `t1`) is excluded rather than plotted as a false
data point:

| Case     | Solvers                        |
|----------|---------------------------------|
| `filter` | IDA, FBDF, Rodas6P, Kvaerno5 (5th-order L-stable ESDIRK), RadauIIA5 (5th-order FIRK) |
| `rc`     | IDA, FBDF, Rodas5P, Kvaerno5, RadauIIA5 |
| `graetz` | IDA, FBDF, Rodas5P, RadauIIA5 |
| `mul`    | IDA, FBDF, KenCarp4 (4th-order ESDIRK), Rodas6P |

- **Kvaerno3/Kvaerno5 stall on both diode circuits.** Tried as a higher-order
  alternative to backward Euler; both get stuck in the diode's stiff turn-on
  transient on `graetz` and `mul` (thousands of steps without leaving `t≈0`), so no
  SDIRK method is used on the diode cases — only on the linear `filter`/`rc`, where
  Kvaerno5 works well and converges cleanly.
- **The best single Rosenbrock variant is genuinely case-dependent — no
  strict order.** Head-to-head at matching tolerances:
  - `rc`: Rodas5P is more accurate at all 4 tolerances tested (e.g.
    `reltol=1e-9`: 3.18e-7 vs 3.59e-7), though Rodas6P takes fewer steps
    each time (282 vs 363 at `1e-9`) — an accuracy-per-step tradeoff, not a
    clean win. Rodas5P used.
  - `filter`: Rodas6P *strictly* dominates — lower error **and** fewer
    steps at every tolerance (e.g. `reltol=1e-3`: 1.1e-4 error/80 steps vs
    5.6e-4 error/88 steps). Rodas6P used instead of Rodas5P.
  - `graetz`: they cross over — Rodas5P wins at the loosest tolerance
    (`1e-3`: 7.0e-6 vs 9.4e-6), Rodas6P wins at medium (`1e-5`: 7.1e-7 vs
    10.0e-7), they're tied by `1e-7`. Rodas5P kept since the crossover
    favors it at the more commonly-used loose end, and it's already the
    established choice there.
  - `mul`: Rodas6P is both more accurate and reaches one more tolerance
    point than Rodas5P (which only clears the loosest point at all).
    Rodas6P used instead of Rodas5P.

  This tracks with theory, not just noise: a 6th-order method's larger
  error constant only pays off once the step size is small enough for the
  asymptotic convergence order to dominate — at loose tolerances a
  lower-order method can have the smaller *practical* error. Each case
  keeps whichever one wins there, never both (one Rosenbrock representative
  per case, per the "don't clutter the plot with a whole family" rule).
- **RadauIIA5 matches the chosen Rodas variant's accuracy on the linear cases
  and on `graetz` at loose/medium tolerance.** It's the only new addition
  that's competitive wherever Rodas5P/Rodas6P is (all are A/L-stable,
  high-order, constant-mass-matrix friendly), and unlike the Rodas family it
  also tolerates a general (non-diagonal) mass matrix in principle — but
  empirically it still goes `:Unstable` past `reltol≈1e-5` on `graetz` and
  everywhere on `mul`, so it's added to `filter`/`rc`/`graetz` but not `mul`.
- **RadauIIA5 has a one-tolerance outlier on `rc`, reproduced and localized.**
  At `reltol=1e-7` (only) its error jumps to 5.5e-3 — four orders of magnitude
  worse than its neighbors (`1e-6`: 4.6e-7, `1e-8`: 4.1e-7) despite `retcode`
  reporting `Success`. Reproduced standalone (no VACASK involved) and traced to
  a single bad step: `t=1.0020ms → 1.0030ms`, the pulse's falling edge, which
  is bounded on both sides by explicit `tstops` exactly 1us apart (the fall
  duration). It's the *only* step in that run with a recorded LTE rejection
  (`rejects=1`); every other tolerance has zero rejects. The true solution
  barely moves over that 1us window (τ=1ms), but the accepted step reports a
  drop ~100x too large. Looks like an `OrdinaryDiffEqFIRK` step-size-selection
  edge case with a forced step sandwiched between two closely-spaced `tstops`,
  not a Cadnip stamping bug — not chased further here since it's upstream of
  Cadnip and affects one point out of 35 RadauIIA5 runs across all cases.
- **KenCarp4 and Rodas6P are the two solvers that get *any* correct points on
  `mul` beyond IDA/FBDF.** Both fail outright on `graetz` (`:Unstable`/
  `MaxIters` at every tolerance), but on `mul` KenCarp4 succeeds at the two
  loosest tolerances (`1e-3`, `1e-5`) with accuracy on par with IDA, and
  Rodas6P matches that same two-point coverage with its own accuracy
  profile. Rodas5P/Rodas4P/Rodas4P2/Rodas5Pr, by contrast, only ever reach
  the single loosest tolerance point on `mul` — Rodas6P is a genuine (if
  narrow) improvement over the rest of its own family specifically on this
  circuit, not just noise.
- **Rodas6P on `mul` doesn't fail cleanly below `reltol=1e-5` — it degrades
  catastrophically instead, and that only showed up in real CI, not the
  exploration sweep.** The exploration script bounded `maxiters` to 50,000
  to keep the survey itself tractable, so a solver grinding through the
  diode's stiffest switching region would hit that cap and report a clean
  `MaxIters` failure in a couple of seconds — that's what made Rodas6P look
  like it simply stopped working past `1e-5` on `mul`. With production's
  real `maxiters=50,000,000`, it instead actually grinds: `reltol=1e-6`
  *succeeded* but took 451s and 6.4M steps (vs. KenCarp4's 2.2s/43,775
  steps at the same tolerance on the same run), and `reltol=1e-7` never
  finished at all, running until CI's 60-minute job timeout killed it. The
  `SOLVERS` entry for `mul` sets `min_reltol=1e-5` for Rodas6P specifically
  to avoid this — a bounded `maxiters` in a quick survey is a useful filter
  for "does this basically work" but isn't a substitute for sweeping the
  full tolerance range at the real budget before trusting a solver's
  behavior past where the survey stopped looking.
- **IDA and FBDF are robust everywhere**, though FBDF also loses individual
  tolerance points to `:Unstable` on the diode circuits at times.

## Solver survey

A broader one-off sweep (12 candidates × 4 tolerances × all 4 cases, using a
Cadnip-tight-IDA reference in place of VACASK where the real VACASK binary
wasn't available) was run to decide what belongs in the `SOLVERS` dict above
and what's outright unsuitable for Cadnip's MNA formulation (singular/
non-diagonal mass matrix, index-1 semi-explicit DAE — see
`doc/mna_design.md` and `doc/Sciml charge formulation.md`). Findings beyond
what's already folded into the table above:

- **Only Rosenbrock23/Rosenbrock32 require a diagonal mass matrix — every
  other Rosenbrock variant in `OrdinaryDiffEqRosenbrock` is fine.** Checked
  directly against the package source
  (`OrdinaryDiffEqRosenbrock/src/alg_utils.jl`):
  `only_diagonal_mass_matrix(alg::Union{Rosenbrock23, Rosenbrock32}) = true`
  is the *only* override in the file: the ~30 other exports (Rodas3, Rodas4,
  Rodas4P, Rodas4P2, Rodas5, Rodas5P, Rodas5Pe, Rodas5Pr, Rodas6P, ROS3P,
  RosShamp4, GRK4T/GRK4A, ROS34PW*, etc.) all accept general mass matrices —
  confirmed empirically here too, since Rodas4P2/Rodas5Pr/Rodas6P (tested as
  a follow-up alongside Rodas5P on `graetz`/`mul`) ran without error on
  every case. The ArgumentError is specific to the two low-order W-methods,
  not a general Rosenbrock limitation. Rodas4P2/Rodas5Pr tracked Rodas5P
  closely wherever tested and added nothing over it, so weren't added to
  `SOLVERS`; Rodas5P vs Rodas6P themselves turned out *not* to be
  interchangeable — see "The best single Rosenbrock variant is genuinely
  case-dependent" above, where a later head-to-head on all 4 cases (not
  just `graetz`/`mul`) found Rodas6P actually wins outright on `filter`.
  Rosenbrock23/32 themselves are still a hard discard for MNA circuits —
  it's architectural (W-method internals assume diagonal `M`), not a
  tolerance/tuning issue.
- **True (implicit-first-stage) SDIRK does not beat ESDIRK on the diode
  turn-on transient — if anything it's worse.** PLECS's solver docs note it
  defaults to (E)SDIRK when a circuit needs MNA, and that "SDIRK is
  typically more stable" than ESDIRK, which reads like implicit-first-stage
  should handle a hard diode turn-on better than ESDIRK's explicit
  Euler-like first stage. Tested directly: `SDIRK2` (A-B-L stable, 2nd
  order), `Cash4` and `Hairer4`/`Hairer42` (all A-L stable, 4th order) — the
  genuine fully-implicit SDIRK methods in `OrdinaryDiffEqSDIRK`, as opposed
  to Kvaerno/KenCarp/TRBDF2 which are ESDIRK (Kennedy-Carpenter/Kvaerno
  tableaus have an explicit first stage by construction). All four go
  `:Unstable` at *every* tolerance on both `graetz` and `mul` — worse than
  KenCarp4's partial success on `mul`. Whatever's driving the diode-turn-on
  failures for the ESDIRK family in Cadnip's formulation isn't specific to
  the explicit first stage; true SDIRK isn't a fix here. Not added anywhere.
- **ABDF2 converges but is quietly wrong.** Across every case it reaches
  `t1` with retcode `Success`, but its relative-L2 error against the golden
  sits at 10%-1400% *regardless of how tight the tolerance is set* (e.g.
  `rc`: 0.16-8.3 at `reltol` from `1e-9` to `1e-3`; `mul`: 0.09-0.31) —
  its embedded error estimator isn't controlling accuracy for this
  formulation. It's still used in the *throughput* benchmark
  (`run_benchmarks.jl`), which is fine since that benchmark only times
  raw step cost and never checks output values, but it should not be
  trusted for anything accuracy-sensitive and isn't a WPD candidate.
- **DFBDF (the Julia-native, GPU-friendly BDF for `DAEProblem`) only works
  on the linear cases.** It converges cleanly on `filter`/`rc`, but on both
  `graetz` and `mul` it aborts with `dt forced below floating point epsilon`
  or outright `:Unstable` at every tolerance tried — consistent with the
  upstream note that DFBDF "still needs more optimization" (see
  `doc/Sciml charge formulation.md`). Not yet a viable IDA alternative for
  nonlinear circuits.
- **TRBDF2 and QNDF were tried and dropped as redundant, not unsuitable.**
  TRBDF2 converges on `filter`/`rc` (competitive with Kvaerno5 at loose
  tolerance) but fails everywhere on both diode circuits like Kvaerno5/
  KenCarp4's turn-on stall; QNDF tracks FBDF closely on every case (same
  family, same failure modes on the diode circuits) but never beats it.
  Neither adds a case where it's uniquely useful, so neither was added to
  keep the per-case solver count (and the 6-marker ASCII plot legend)
  legible.
- **Takeaway:** no single "best" solver — IDA (DAE, explicit analytic
  Jacobian) is the reliable floor everywhere; Rodas5P/RadauIIA5 (high-order,
  constant/general mass matrix) win on accuracy-per-step whenever the
  circuit's turn-on transient lets them get started at all; and the
  stiffest, fastest-switching circuit (`mul`) narrows the viable set down
  to IDA, FBDF, and (partially) KenCarp4 — every ODE-mass-matrix solver
  with real stage-derivative coupling (Rosenbrock, SDIRK/ESDIRK, FIRK)
  either stalls in the diode's stiff turn-on or goes unstable once the
  100kHz switching kicks in.

## Findings about VACASK

Reported upstream as [VACASK issue #83](https://codeberg.org/arpadbuermen/VACASK/issues/83);
the items below reflect the maintainer's diagnosis, not just ours.

- **`rc`'s plateau is a `tran_maxord=5` controller bug, not a breakpoint/edge
  problem.** Earlier revisions of this doc guessed the pulse-train `rc` case's
  ~1.5-7% error plateau (flat regardless of `reltol`) came from VACASK stepping
  over the 1us pulse edges. That guess was wrong and has been retracted: dumping
  `tran1.raw` shows 5-6 accepted points inside every rise/fall window at every
  `reltol` tested — breakpoints are handled correctly. Isolated the real cause
  by sweeping only `tran_maxord` at fixed `reltol=1e-6` (netlist in the issue):
  orders 1-4 converge normally (points/error: 4449/6.5e-4, 891/1.0e-4,
  551/1.2e-4, 550/6.3e-5) but order 5 alone jumps three orders of magnitude
  worse (448/6.9e-2) while taking *fewer* steps — the step-size/LTE controller
  is accepting steps that are far too large specifically at order 5 on this
  circuit. It's circuit-specific, not a blanket "order 5 is broken": the same
  sweep on `filter` (smooth sine, no discontinuities) has order 5 as the
  *best* point (562 points/1.4e-4, beating every lower order), and on `graetz`
  (diode, nonlinear) order 5 tracks a tight order-2 reference to ~2e-5 with no
  anomaly. `config.json`'s `rc.vacask_override` now caps `tran_maxord` at 4
  instead of clamping `maxstep` (see below) — the maintainer is looking into
  the order-5 controller itself.
- **The old `maxstep=5us` workaround is gone — capping the order was the real
  fix.** The previous override forced `maxstep=5e-6` (a crutch: it bounds
  *every* step in the run, not just the ones near the bug, and floors accuracy
  at the gear method's own LTE on the ramp instead of letting `reltol` drive
  it). Capping `tran_maxord` at 4 instead removes the need for any `maxstep`
  bound at all — VACASK now gets genuine unbounded, `reltol`-driven adaptive
  stepping on `rc`: 1.5e-4 error at `reltol=1e-3` down to ~3.8e-5 at
  `reltol=1e-9`, monotonically improving, which the order-5 config never did
  at any `maxstep`.
- **On method order in general: VACASK's own design historically stays at
  order ≤ 2 for A-stability, so "higher order = more accurate" (this
  benchmark's original assumption for picking `tran_maxord=5` as "VACASK's
  best") doesn't hold unconditionally.** Per the maintainer: methods of order
  ≤ 2 (trapezoidal, `gear2`) are A-stable and never blow up for a stable
  circuit regardless of step size; stability regions shrink as order grows, so
  a higher-order method can need *smaller* steps on a stiff circuit to stay
  stable, the opposite of the "bigger steps, same accuracy" benefit
  higher order usually buys. That framing fits `graetz`/`filter` (order 5
  is fine or best there) better than a strict "5 is broken" reading, but it
  does mean this benchmark should keep picking `tran_maxord` per case by
  what actually converges, the same empirical policy already used for
  picking Cadnip's solver family per case, rather than assuming one order is
  "VACASK's best" everywhere.
- **`mul` aborts at small steps — confirmed to be VACASK's `nr_residualcheck`,
  not a step-resolution problem.** `mul`'s source is a smooth sine, so unlike
  `rc` there's no edge to resolve; VACASK hits "Timestep too small" below
  `reltol ≈ 1e-5` regardless of `tran_maxord` (checked 1 through 5, all abort
  identically after 2 accepted / 21 rejected points). Per the maintainer, this
  is a known, acknowledged sensitivity in the residual check: it "cannot
  establish a reference value for equations that have only one residual
  contribution," and the recommended remedy for any circuit that triggers it
  is to disable it — which is exactly what `mul`'s `vacask_override.extra_opts`
  already does (`nr_residualcheck=0 tran_lteratio=3.5 tran_itl=50`, carried
  over from the real throughput sim's own tuning). With that applied: no abort
  anywhere in the sweep.
- **The "1.3e-4 floor" this produced against Cadnip's golden was never a
  VACASK accuracy problem — it was scoring VACASK against the wrong
  reference.** With the fix above applied, checked VACASK's *self*-consistency
  at `reltol=1e-9` directly: comparing `tran_maxord=2/4/5`, `tran_lteratio`
  tightened/loosened, and `abstol` down to `1e-18` against each other all
  agree to ~1e-6–1e-7 - i.e. VACASK is converging cleanly, two full orders of
  magnitude tighter than the ~1.3e-4 gap it showed against Cadnip's golden.
  That gap is VACASK's and Cadnip's independently-converged answers
  disagreeing with *each other* by a small, roughly tolerance-independent
  amount — most likely the two simulators' separately-compiled diode OSDI/VA
  models not being bit-identical (both use nominally the same
  `is/rs/cjo/m/n`, but VACASK's bundled `spice/sn/diode.osdi` and Cadnip's
  VADistillerModels SPICE diode are different compiled sources) — not a
  tolerance-tuning gap on either side. `mul`'s `golden` is now `"self"`
  (`config.json`): Cadnip's curves score against a tight Cadnip golden,
  VACASK's curve against its own tight VACASK golden, and the two goldens'
  mutual disagreement is reported separately as the `xcheck` line
  (currently **1.03e-4**, matching the old "floor" almost exactly - strong
  confirmation this is exactly what was happening). With `self` scoring,
  VACASK's own curve now properly converges to **4.27e-6 at `reltol=1e-9`**,
  the same order of magnitude as Cadnip IDA's own **2.49e-6** there - not the
  "VACASK simply cannot reach this accuracy, a real engine-level gap" this
  doc previously claimed. That specific claim is retracted. Diffing the two
  diode `.va` sources to find the actual ~1e-4 discrepancy is a real, open
  item, not yet investigated here.
- **`filter`'s "lever-immune" floor turned out to be OUR benchmark's own bug:
  the initial-timestep hint was far too coarse.** Checked `tran_maxord`
  2/3/4/5 (identical 2.0-2.1e-5 at `reltol=1e-9`), `abstol=1e-15`,
  `tran_lteratio`, and `nr_residualcheck=0` on `filter` - none moved it, and
  the `relrefsol/relrefres/relreflte="pointlocal"/"local"` modes he
  described made it *worse* (immediate "Timestep too small" abort,
  consistent with his own caveat about establishing a reference near a
  zero initial condition, exactly `filter`'s starting point). What actually
  mattered, per his separate comment that "initial step selection lacks
  sophistication - designers should specify sensible values per circuit":
  the `analysis tran ... step=` argument. `wpd_common`/`run_wpd.jl` were
  passing `tspan/n_grid` there (`100/2000 = 0.05` for `filter`) - reasonable
  as an output-density hint, but that value **also sets VACASK's very first
  internal timestep**, and forcing a first step of `0.05` on a circuit whose
  natural period is `2π≈6.28` (and whose LC ladder is only lightly damped
  by `R4=1`) injects a small error that never damps out over the rest of
  the run, capping accuracy regardless of `reltol`. Confirmed directly:
  forcing `step` down through `1e-6, 1e-9, 1e-12` at `reltol=1e-9` all land
  within 0.01% of each other (~4.27e-7) - a **~50x** improvement over the
  `0.05` floor - while total point counts barely change (1720 vs 1725),
  showing it's specifically the *first* step's error that's fixed, not
  overall resolution. `rc`/`graetz`/`mul` were essentially unaffected by the
  same change (their `tspan/n_grid` was already fine enough relative to
  their own dynamics) - confirmed by re-running all four cases end-to-end.
  `run_vacask_once` now always requests `step=1e-12`, tspan-independent;
  `filter`'s VACASK curve genuinely converges now (`4.11e-7` at
  `reltol=1e-9`, no floor), and nothing else changed. This was never a
  VACASK bug - it's exactly what the maintainer's own comment predicted:
  a naive fixed initial-step choice in our own harness.
- **`rc`'s floor is real, open, and NOT the same mechanism.** Unlike
  `filter`, forcing `rc`'s initial `step` down to `1e-12` (from the same
  `tspan/n_grid` origin) made *no* difference at all (3.8330e-5 either way,
  full reltol sweep re-checked). `rc`'s floor also survives `tran_maxord`
  2/3/4 (3.2-3.8e-5 at `reltol=1e-9`), `abstol` down to `1e-18`, and
  `tran_lteratio` sweeps - every documented tuning knob was tried; none of
  them touch it. **A gmin/leakage-conductance origin is ruled out, not just
  untested:** `gmin` isn't exposed as a global option in this VACASK build
  (only `gshunt` and `homotopy_*gmin*`, which are DC-operating-point
  homotopy convergence aids, not something applied through transient -
  confirmed by binary symbol names like `OpNRSolver::loadShunts`);
  explicitly forcing `gshunt=0`/`1e-15` and
  `homotopy_startgmin=0 homotopy_maxgmin=0` left the error completely
  unchanged (3.8330e-5 in every variant, to 5 significant figures). More
  conclusively: a fixed leakage conductance anywhere would make the
  relative error scale with circuit impedance (a bigger `R` means the leak
  carries proportionally more current), so `rc`'s `R`/`C` were rescaled
  1Ω/1F down to 1MΩ/1nF (same `τ=RC=1ms`, same pulse) — error stayed at
  3.83-3.84e-5 across all 6 decades of impedance. No leakage-conductance
  mechanism, gmin or otherwise, produces that, and it isn't a startup-step
  artifact either (see above). Current best guess: `rc`'s pulse train hits
  a breakpoint (forced order-1 restart) roughly every 1ms, ~20 times over
  the run, so unlike `filter`'s one-time startup cost this could be a small
  error reintroduced at *every* breakpoint rather than damping away between
  them - not yet confirmed, still an open item.
- **`vntol` tied numerically to `reltol` is a simplification, flagged as such
  by the maintainer, but checked here not to bias these specific results.**
  Both VACASK sweeps (`run_vacask_once` in `wpd_common.jl`) pass `vntol=reltol`
  — conflating an absolute voltage tolerance with a dimensionless relative
  one, which the maintainer correctly points out "lacks universal
  applicability across circuits." Checked directly: re-running `rc` at
  `tran_maxord=4` with `vntol` fixed at `1e-6` (decoupled from `reltol`)
  instead of tied to it gives the same error to 3 significant figures at
  every `reltol` tested (e.g. `reltol=1e-9`: 3.833e-5 tied vs. 3.833e-5
  fixed) — because every case's node voltages sit in a similar O(0.1-50V)
  band where the two happen to coincide numerically. Left as-is since it
  doesn't change the numbers, but it's not a generally sound pattern to reuse
  for a circuit with a very different voltage scale.

## History

The `graetz`/`mul` (diode) and full pulse-train `rc` cases were previously blocked by
two Cadnip bugs — the SPICE diode didn't conduct and the PULSE source didn't repeat —
now fixed on `main` (#197, #196). This benchmark surfaced both by checking output
*values*, which the throughput benchmarks never do.

## Files
- `run_wpd.jl` — single entry point (Cadnip sweep, VACASK sweep, error, ASCII + PNG/SVG
  plots, markdown report).
- `wpd_common.jl` — shared helpers (config, CSV/raw IO, interpolation, error metric,
  VACASK discovery).
- `config.json` — sweep, per-case `tspan`/`output`/`golden`, VACASK integration order.
- `filter.sp` — the linear filter netlist (the others reuse `../<case>/cedarsim/`).
- `out/` — generated report + intermediate CSVs + `plots/*.{png,svg}` (git-ignored).
