# Work-Precision Diagram (WPD) benchmark: Cadnip vs VACASK

The other benchmarks under `benchmarks/vacask/` measure **throughput**: they pin a
tiny fixed `dtmax` and relaxed tolerances and time the result. That bypasses the
adaptive local-truncation-error control entirely, so it rewards whichever solver
has the cheapest per-step cost and tells you nothing about *accuracy per unit of
work*.

This benchmark is the complement. Each solver runs **adaptively** across a sweep of
tolerances, and we plot **error vs. runtime** (log-log) — a work-precision diagram,
following the [SciML `WorkPrecisionSet`](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/)
methodology. A higher-order BDF/Rosenbrock method that takes big accurate steps
should sit *below* a simple low-order method at tight tolerance; that separation is
exactly what the throughput benchmark hides.

It is pure Julia: Cadnip runs in-process, VACASK is shelled out to (its SPICE3 raw
output is parsed directly), and plotting is done with Plots.jl. No Python.

## Pipeline (three stages)

```bash
# (sandbox: use ~/.juliaup/bin/julia, Julia 1.11)

# 0. one-time: fetch the VACASK binary (downloads the pinned release into the cache)
CASES="" benchmarks/vacask/run_vacask.sh        # or set $VACASK_COMMAND yourself

# 1. Cadnip sweep -> out/cadnip_*.csv (waveforms) + out/cadnip_<case>.csv (timing)
julia --project=benchmarks benchmarks/vacask/wpd/cadnip_wpd.jl

# 2. VACASK sweep + golden -> out/vacask_*.csv, out/ref_<case>.csv
julia --project=benchmarks benchmarks/vacask/wpd/vacask_wpd.jl

# 3. errors + diagrams -> plots/<case>.png/.svg, out/wpd_results.md
julia --project=benchmarks benchmarks/vacask/wpd/plot_wpd.jl
```

Pass case names to any stage to restrict it, e.g. `... cadnip_wpd.jl filter`.

## Precision / golden reference

Per case, the golden reference is the most accurate one available:

- **Analytic** closed-form solution when the circuit has one (the linear filter).
  Exact, so it never floors the tightest runs.
- Otherwise a **tight VACASK run** (the more mature simulator), using a moderately
  tight `reltol` with a *fine* `maxstep` so the trajectory is accurate without
  tripping VACASK's min-timestep abort. When both an analytic and a VACASK golden
  exist they are cross-checked against each other (printed as `analytic vs
  VACASK-tight`).

Error is the relative L2 (RMS) norm of the output node (SciML `error_estimate =
:l2`), evaluated at **each run's own output timepoints** against the *dense*
reference (analytic: 200k pts; VACASK-tight: ~100k pts). Measuring at native points
matters: a high-order solver takes large steps and emits few points, so
interpolating *its* output onto a fixed grid would penalise it for the
interpolation rather than its real accuracy. Interpolating the dense reference onto
the run's points instead is accurate and fair.

**VACASK is run at its best, not its default.** VACASK defaults to trapezoidal
(2nd order); the benchmark sets `tran_method="gear" tran_maxord=5` so it uses its
variable-order Gear/BDF (up to 5th order) — configurable via `vacask_tran_method` /
`vacask_tran_maxord` in `config.json`.

## Circuits

Chosen to be **driven and dissipative** with a stable forced response, so error
degrades gracefully with tolerance. Autonomous/digital circuits (ring oscillator,
digital multiplier) are deliberately avoided: their phase error accumulates without
bound, so the WPD curve saturates at O(1) and "falls off a cliff".

| Case     | What                                     | Reference | Status |
|----------|------------------------------------------|-----------|--------|
| `filter` | 3rd-order Butterworth LC ladder (linear) | analytic  | active |
| `rc`     | RC single-pulse step response (linear)   | VACASK    | active |
| `graetz` | Graetz bridge full-wave rectifier        | VACASK    | disabled — see below |

`filter` is a smooth drive; `rc` adds two sharp source edges, so the adaptive
controller must catch the discontinuities (it is given `tstops` at the pulse edges,
mirroring how SPICE engines break at source breakpoints internally).

### Cadnip bugs this benchmark surfaced

Checking *output values* (which the throughput benchmarks never do) exposed two
latent Cadnip bugs:

1. **SPICE diode does not conduct.** Via the `.model … d` netlist path the diode is
   an open circuit (`5 V` through `1 kΩ` into a diode to ground stays at `5 V`; the
   graetz bridge output is identically `0`). Root cause: the SPICE path passes the
   instance default `area = 0` to `sp_diode`, which scales the current by area, so
   `sp_diode(area=0.0)` is open while `sp_diode()` / `sp_diode(area=1.0)` conduct.
   The direct `stamp!(sp_diode(), …)` API works and is tested; the netlist path is
   untested. This **disables `graetz`** (kept in `config.json`'s `_disabled_cases`).
   The diode voltage multiplier is also excluded — it additionally trips VACASK's
   own min-timestep abort below `reltol ~ 1e-5`.

2. **PULSE source does not repeat.** Cadnip's pulse fires once then stays at `val0`
   (the `period` parameter is ignored); VACASK repeats correctly. So `rc` is
   **windowed to a single pulse** (`tspan = [0, 2 ms]`), where Cadnip and VACASK
   agree. Widen once the pulse-repetition bug is fixed.

Both bugs surface only when comparing waveforms against a reference — exactly what
this work-precision benchmark does. Re-enable `graetz` by moving its entry from
`_disabled_cases` back into `cases` once the diode model is fixed.

## Outputs

- `out/cadnip_<case>_<solver>_<reltol>.csv` — Cadnip output waveform per run.
- `out/cadnip_<case>.csv` — per-run timing / accepted-steps / rejects / NR iters.
- `out/vacask_<case>_<reltol>.csv`, `out/vacask_<case>.csv` — VACASK waveforms/timing.
- `out/ref_<case>.csv` — VACASK tight golden on the grid; `out/analytic_<case>.csv`
  — closed-form waveform (linear cases).
- `out/wpd_results.md` — error/runtime table per circuit.
- `plots/<case>.png` / `.svg` — the work-precision diagrams.

`out/` and `plots/` are git-ignored (regenerated by the scripts).

## Configuration

`config.json` is the single source of truth shared by all three stages: the
tolerance sweep (`reltols`), the golden tolerances (`ref_reltol`/`ref_abstol`) and
its fine-maxstep factor (`ref_maxstep_factor`), the grid size (`n_grid`), and the
per-case `tspan` / `output` node(s).
