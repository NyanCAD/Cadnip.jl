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
| `filter`, `rc` | IDA, FBDF, Rodas5P, Kvaerno5 (5th-order L-stable SDIRK), **RadauIIA5** (5th-order FIRK) |
| `graetz` | IDA, FBDF, Rodas5P, **RadauIIA5** |
| `mul`    | IDA, FBDF, **KenCarp4** (4th-order ESDIRK) |

- **Kvaerno3/Kvaerno5 stall on both diode circuits.** Tried as a higher-order
  alternative to backward Euler; both get stuck in the diode's stiff turn-on
  transient on `graetz` and `mul` (thousands of steps without leaving `t≈0`), so no
  SDIRK method is used on the diode cases — only on the linear `filter`/`rc`, where
  Kvaerno5 works well and converges cleanly.
- **Rodas5P works on `graetz` but not `mul`.** On `graetz` it gives the best
  accuracy-per-runtime of any Cadnip solver (converges to ~3e-8), degrading
  gracefully to `:Unstable` only at the tightest `reltol=1e-9` (excluded by the
  retcode filter). On `mul` — whose 100kHz cascaded-diode switching is far
  stiffer — it hangs even at the loosest `reltol=1e-3`, so it's excluded there.
- **RadauIIA5 matches Rodas5P's accuracy on the linear cases and on `graetz`
  at loose/medium tolerance.** It's the only new addition that's competitive
  wherever Rodas5P is (both are A/L-stable, high-order, constant-mass-matrix
  friendly), and unlike Rodas5P it also tolerates a general (non-diagonal)
  mass matrix in principle — but empirically it still goes `:Unstable` past
  `reltol≈1e-5` on `graetz` and everywhere on `mul`, so it's added to
  `filter`/`rc`/`graetz` but not `mul`.
- **KenCarp4 is the one ESDIRK method that gets *any* correct points on `mul`.**
  It fails outright on `graetz` (`:Unstable` at every tolerance), but on `mul`
  it succeeds at the two loosest tolerances (`1e-3`, `1e-5`) with accuracy on
  par with IDA before it too hits the 100kHz cascaded-diode switching wall —
  added to `mul` alone for that partial coverage.
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

- **Rosenbrock23 is architecturally incompatible, not just inefficient.**
  It threw `ArgumentError: Rosenbrock23 only works with diagonal mass
  matrices` immediately on every case — its W-method implementation assumes
  a diagonal `M`, which MNA's coupled-node capacitance matrix essentially
  never is. Discard it (and by the same low-order-W-method family,
  Rosenbrock32) for MNA circuits entirely; it's not a tolerance/tuning issue.
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
