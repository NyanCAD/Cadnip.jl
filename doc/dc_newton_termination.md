# Open item: DC Newton terminates on residual norm; weakly-conducting nodes can converge mV-to-tens-of-mV off

*Filed as a doc because GitHub issues are disabled for this repository. Found
while debugging the VACASK `mul` WPD-benchmark cross-check gap (the ~1e-4
"open item"); the dominant cause there was the internal-node gmin ground leak,
fixed in `src/vasim.jl` (the internal-node gmin anchor is removed entirely;
models keep their internal nodes conductive via junction gmin, rs paths, and
`V(int,ext) <+ 0` node collapse).
The termination-criterion problem below is real, separate, measured, and
should be resolved by the per-type abstol work rather than a bolt-on.*

## Problem

`_dc_newton_compiled` (shared by `dc!` and `CedarDCOp` transient init)
terminates Newton on a single **absolute residual norm**
(`solve(nlprob, nlsolve; abstol=1e-10)`; `CedarDCOp` defaults to `1e-9`).
On weakly-conducting nodes this accepts solutions whose voltages are still
far off, and the tolerance **cannot simply be tightened**, because
differently-scaled KCL rows have wildly different floating-point noise
floors.

## Measurements (VACASK `mul` circuit: diode chain at 50 V, junction conductance Is/(N·Vt) ≈ 2e-9 S)

Exact chain operating point is 50.000000 V (VACASK: 50.000000, ngspice:
50.000058). Cadnip's DC solve, after the gmin-leak fix:

| DC solve `abstol` | result |
|---|---|
| 1e-10 (default) | v(20) = 49.9998 V — 0.2 mV of termination slack |
| 1e-12 | same answer |
| 1e-15 | **solve fails outright, all fallbacks exhausted** |

On a synthetic 3-diode chain (same Is/N, no `$limit`, no charge states) the
slack at default tolerance measured **8 mV**.

Why tightening can't work: the residual vector mixes rows with ~100 S
conductances (the 0.01 Ω source loop at 50 V cancels ~5e3-scale currents,
leaving ≥1e-12 A of roundoff noise) and rows that need <1e-13 A resolution
(the 2e-9 S junctions). Both are *current-type* rows, so **a per-type
residual abstol alone cannot separate them** — the conditioning is per-row,
not per-type.

## What actually pins these nodes

Standard SPICE terminates NR on **per-node voltage updates**
(`|Δv| ≤ reltol·|v| + vntol`) plus per-device current checks — a
voltage-typed criterion on the *update*, not the residual. A prototype
full-Newton polish loop terminating on `max|δu| ≤ 1e-9 + 1e-12·max|u|`
(see this branch's history) pinned the synthetic chain to machine-exact
50 V where the residual criterion left 8 mV of slack. It was reverted in
favor of doing this properly in the per-type abstol work, because:

- on the real `mul` (VA diode with `$limit` and charge states) the extra
  Newton steps bounce at the ~mV scale: each rebuild updates the limiter
  state, making the residual iteration-history-dependent at the ~1e-8
  level, so the update never shrinks below that;
- a hardcoded voltage tolerance bolted onto the residual solve duplicates
  what a typed-tolerance design should own.

## Suggested resolution (fold into the per-type abstol PR)

- Newton termination for the DC/OP solve should include a
  **voltage-typed criterion on the update** (`vntol`-style, per unknown
  type: V for node voltages, A for branch currents, scaled charge for
  charge states), not only residual norms. `MNASpec` already carries
  `vntol` / `reltol` / `iabstol` fields that fit naturally.
- Mind the `$limit` interaction: the residual seen by the solver is
  history-dependent through the limiter state, so the criterion must
  tolerate an update-noise floor (converge when the update stops
  shrinking, not insist on an absolute target).
- Regression reference: the `mul` chain DC operating point should land
  ≤ ~1 mV from 50 V (ngspice: 58 µV; Cadnip today: ~0.2 mV — that slack
  *is* this issue). The stamping-level part is already covered by the
  "Internal-node gmin anchor must not leak to ground" test in
  `test/mna/vadistiller.jl`; a solver-level bound tighter than the
  current loose 2e-2 sanity check becomes possible once the update
  criterion exists.
