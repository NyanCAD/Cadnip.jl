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

## Circuits

Chosen to be **driven and dissipative** with a stable forced response, so error
degrades gracefully with tolerance. Autonomous/digital circuits (ring oscillator,
digital multiplier) are deliberately avoided: their phase error accumulates without
bound, so the WPD curve saturates at O(1) and "falls off a cliff".

| Case     | What                                   | Reference        |
|----------|----------------------------------------|------------------|
| `filter` | 3rd-order Butterworth LC ladder (linear) | closed-form analytic |
| `graetz` | Graetz bridge full-wave rectifier      | VACASK (tight)   |
| `mul`    | Diode voltage multiplier (stiffer)     | VACASK (tight)   |

The circuits mirror the existing `<case>/cedarsim/runme.sp` netlists so Cadnip and
VACASK simulate the *same* problem.

## Precision / golden reference

VACASK is the more mature simulator, so a **tight-tolerance VACASK run**
(`ref_reltol`/`ref_abstol` in `config.json`) is the golden reference. The output
node is sampled on a common time grid, and error is the relative L2 (RMS) norm of
every swept run against that single golden. For the linear filter the closed-form
solution is carried alongside as an independent check that the golden is itself
correct (printed as a `golden vs analytic` cross-check).

If the VACASK binary / InSpice-VACASK backend is unavailable, the harness degrades
gracefully: it falls back to the analytic golden (linear) or the tightest Cadnip
run (nonlinear) and plots Cadnip-only curves.

## Running

Two steps. First the Cadnip side (Julia), then the VACASK side + plots (Python).

```bash
# 1. Cadnip waveforms + timing (writes wpd/out/*.csv)
julia --project=benchmarks benchmarks/vacask/wpd/cadnip_wpd.jl
#    (sandbox: ~/.juliaup/bin/julia, Julia 1.11)

# 2. VACASK sweep + work-precision diagrams (writes wpd/plots/*.png and wpd/out/wpd_results.md)
pip install -r benchmarks/vacask/wpd/requirements.txt
python3 benchmarks/vacask/wpd/run_wpd.py
```

Run a single case by name, e.g. `... cadnip_wpd.jl graetz` and `... run_wpd.py graetz`.

### Making VACASK discoverable

`run_wpd.py` looks for the VACASK binary in this order:

1. `$VACASK_COMMAND` (and optionally `$OSDI_PATH` for the compiled `.osdi` models),
2. the cache populated by `../run_vacask.sh` (`~/.cache/cadnip-vacask/vacask/simulator/vacask`).

The simplest setup is to run `../run_vacask.sh` once (it downloads the pinned
VACASK release and its model libraries), then run `run_wpd.py`. Otherwise point
`VACASK_COMMAND`/`OSDI_PATH` at your own VACASK build.

## Outputs

- `out/cadnip_<case>_<solver>_<reltol>.csv` — Cadnip output waveform per run.
- `out/cadnip_<case>.csv` — per-run timing/steps/retcode summary.
- `out/analytic_<case>.csv` — closed-form waveform (linear cases).
- `out/wpd_results.md` — error/runtime table per circuit.
- `plots/<case>.png` / `.svg` — the work-precision diagrams.

## Configuration

`config.json` is the single source of truth shared by both scripts: the tolerance
sweep (`reltols`), the golden tolerances (`ref_reltol`/`ref_abstol`), the grid size
(`n_grid`), and per-case `tspan` / `output` node(s). Keep both scripts reading it so
the two simulators stay on the same grid and reference.
```
