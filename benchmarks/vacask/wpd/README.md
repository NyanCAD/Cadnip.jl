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

The golden is **pinned per case** in `config.json` (`golden: analytic|vacask|cadnip`),
chosen by what actually converges — `run_wpd.jl` uses exactly that and errors if it
fails (no silent substitution):

- `filter`, `rc`: **exact analytic** closed form (also avoids any "VACASK vs its own
  golden" ambiguity).
- `graetz`: tight VACASK run (fine `maxstep`); it completes and is cross-checked.
- `mul`: **Cadnip IDA** tight run — VACASK's multiplier golden aborts (see below).

**VACASK runs at its best, not its default:** VACASK defaults to trapezoidal (2nd
order); the benchmark sets `tran_method="gear" tran_maxord=5` (variable-order
Gear/BDF), configurable via `vacask_tran_method` / `vacask_tran_maxord` in
`config.json`.

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

- **`mul` aborts at small steps.** VACASK hits "Timestep too small" on the voltage
  multiplier below `reltol ≈ 1e-5` (its fine-`maxstep` golden aborts too), so `mul`
  uses a Cadnip golden and VACASK contributes only its loose-tolerance points — a
  direct illustration of Cadnip covering a range VACASK can't on a stiff diode network.
- **Adaptive stepping plateaus.** With `reltol` alone (no forced fine `maxstep`),
  VACASK's error stops improving past a point (e.g. the pulse-train `rc`): its LTE
  controller settles at a step count well short of the accuracy its fine-`maxstep`
  golden reaches. Cadnip's solvers keep converging.
- **Giving VACASK a fair, reasonable shot on `rc`/`mul` confirms the plateau/abort
  is a controller/tuning gap, not an engine ceiling — and a single global
  `maxstep` can't be "loose except at the edges".** VACASK's `analysis tran`
  only exposes one scalar `maxstep` applied to every step of the whole run —
  there's no separate breakpoints-only directive in anything this repo's
  `.sim` files use, so tightening it enough to catch a sharp source edge also
  caps every other step in the run, uniformly. `rc` and `mul` each get one
  extra "fair" VACASK sweep (`config.json`'s `vacask_probes`) built from that
  tradeoff plus (for `mul`) the real throughput sim's own NR/LTE tuning:
  - `rc`: `maxstep=1us` (the real throughput benchmark's `dtmax`) confirmed
    fixing the 1.5-7% unbounded plateau (~1000x error drop, to ~2-7e-5) — the
    unbounded run never resolves the 1us pulse edges without a step bound.
    Swept looser from there: `5us` (5x looser) gives the *same* accuracy at
    *5x lower* runtime (0.011s vs 0.051s) — real headroom, not a knife-edge
    value, so `5us` is what's plotted as "fair". Error still doesn't improve
    with tighter `reltol` at either step size, though: once `maxstep` binds,
    `reltol` stops driving step size, and the residual ~2-7e-5 is the gear
    method's local truncation error on the ramp itself, not a tolerance-driven
    quantity — no single global `maxstep` gets both "resolves the edge" and
    "`reltol` still matters everywhere else".
  - `mul`: unlike `rc`, its source is a smooth sine, not a pulse — there's no
    edge to resolve, so forcing `maxstep` here would just be a crutch masking
    whether `reltol`-driven stepping actually works, not a fair test of the
    engine. "fair" for `mul` leaves `maxstep` unbounded and only replicates
    the real throughput sim's control block (`mul/vacask/runme.sim`:
    `tran_method=gear2`, `nr_residualcheck=0`, `tran_lteratio=3.5`,
    `tran_itl=50` — none of which the plain `reltol` sweep ever tries). That
    alone is enough: no abort anywhere in the sweep, and unlike the
    maxstep-forced version tried first (which flatlined at ~1.0e-4 from the
    start, `reltol` no longer doing anything), error now genuinely tracks
    `reltol` — 4.1e-2 → 1.2e-2 → 2.1e-3 → 1.7e-4 — before flattening at
    `reltol≈1e-6` and staying at ~1.4e-4 through `1e-9`. So the abort really
    was the NR/LTE-tuning gap, not a step-resolution problem, confirming the
    `maxstep`-based fix was masking the wrong variable. Head-to-head with
    Cadnip IDA: within the accuracy band VACASK can actually reach, it's
    competitive-to-faster (VACASK's cheapest run at its own floor, `1e-6`:
    1.7e-4 error/0.073s, vs Cadnip's *loosest, cheapest* setting, `1e-3`:
    already-better 1.1e-4 error/0.126s — so even there Cadnip's floor point
    alone is both more accurate and not much slower). But VACASK has a hard
    accuracy ceiling around 1.4e-4 that ten more orders of `reltol` tightening
    (`1e-6` through `1e-9`, 0.073s → 0.789s) never breaks through, while
    Cadnip keeps converging to ~3e-6 — an accuracy range VACASK simply cannot
    reach on this circuit at any cost, a real engine-level gap rather than a
    benchmark-config artifact.

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
