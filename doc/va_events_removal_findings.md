# VA Voltage-Dependent Event Detection: Built, Tested, Removed

## Summary

Cadnip briefly had a second discontinuity-handling mechanism alongside
`auto_tstops` (source-breakpoint-derived `tstops`, kept — see
`src/mna/breakpoints.jl`): `va_events`, which intercepted every `>`/`<`/`>=`/`<=`
comparison in a Verilog-A analog block, gave it a lexical "condition slot", and
used a `VectorContinuousCallback` to root-find the exact time each comparison
flipped sign, forcing the integrator to land precisely on region-switch
boundaries (e.g. cutoff/triode/saturation in a MOSFET model) instead of
stepping over them.

It was implemented completely (condition-slot allocation in `MNAContext`/
`DirectStampContext`/`CompiledStructure`, `va_cmp_*` runtime dispatch in
`vasim.jl`'s codegen, `condition_is_vdep` filtering to distinguish genuinely
voltage-dependent comparisons from parameter/constant ones via `ForwardDiff.Dual`
detection, and `va_event_callback`/`tran!` wiring), and tested against a
synthetic single-comparator circuit where it worked exactly as designed. It was
then removed in full after empirical testing against real vendored compact
models showed no case where it improved anything, and multiple cases where it
made simulation measurably worse. This document records why it was tried, what
was measured, and why it's gone — so the idea isn't re-attempted from scratch
without this evidence.

## What worked

A hand-built synthetic circuit (an ideal comparator reading a filtered node
voltage, `test/mna/va_events.jl`, since deleted) confirmed the mechanism itself
is correct: `condition_is_vdep` correctly identified the one genuinely
voltage-dependent comparison out of several, `va_event_callback` produced a
`VectorContinuousCallback` that root-found the crossing to tight tolerance, and
work was skipped entirely (`nothing` returned) when there were no
voltage-dependent slots to watch. This confirmed the architecture, dispatch,
and DAE/ODE/DDE solver wiring all functioned as designed. It just never
translated into a benefit on a real model.

## What failed: sp_mos1 ring oscillator

`test/mna/oscillator_test.jl`'s 3-stage sp_mos1 (Level-1 MOSFET) ring
oscillator was the first real-model test, and it is exactly the kind of
circuit `va_events` was built for: mos1.va branches on cutoff/linear/
saturation region based on node-voltage-derived locals, a literal `if`-based
discontinuity of the kind smooth/surface-potential models like PSP103
deliberately avoid.

Two things went wrong even on this favorable case:

1. **Slot count vs. genuinely-voltage-dependent count.** A "simple" Level-1
   model has 246 total intercepted comparisons per device evaluation, of
   which `condition_is_vdep` filtering found only 89 ever observed a `Dual`
   operand — the rest are parameter validation (`if (L <= 0)`), temperature
   clamps, and internal bookkeeping that never depend on the solved voltage.
   The filter works (it is a real, non-trivial reduction), but 89 remaining
   roots is still a lot to watch simultaneously via one
   `VectorContinuousCallback`.
2. **The remaining 89 are themselves too fast for per-comparison root-finding
   to help.** A free-running 3-stage ring switches all 6 transistors' regions
   on sub-nanosecond timescales. Watching 89 simultaneous roots, each
   requiring dense-output interpolation and a full circuit rebuild per
   candidate crossing, overwhelmed the adaptive step controller: the solver
   took **~240x more integration steps** across the same 20ns window, and
   the oscillation trajectory itself changed (collapsed) relative to
   `va_events=false` — measured at up to **426 events/ns**. This is not "it
   got slower but still correct" — it's "the mechanism designed to make
   discontinuity handling *more* faithful made the simulated waveform *less*
   faithful," because forcing precise stops at hundreds of closely-spaced
   roots per nanosecond fights the step controller instead of helping it.

This result already argued against the mechanism's practical value even on
the favorable case (a model with genuine literal-`if` region switches).

## Why PSP103 wasn't expected to fare better

The natural follow-up question was whether `va_events` could replace the ring
oscillator benchmark's `force_dtmin=true` + relaxed-tolerance workaround
(`benchmarks/vacask/ring/cedarsim/runme.jl`), since that workaround exists
specifically because the integrator keeps failing to converge through the
PSP103 ring's switching transitions — architecturally the kind of problem
event detection targets. Investigation (`doc/ring_oscillator_investigation.md`,
`models/PSPModels.jl/va/PSP103_SPCalculation.include`) found:

- PSP103 is a **surface-potential model deliberately designed to be smooth
  and continuous** across region boundaries — that's the entire point of the
  surface-potential formulation, as opposed to Level-1-style piecewise
  region equations. It has comparisons (`x_s < se05`, `Ds > ke05`, etc.), but
  they are overwhelmingly numerical safety guards inside its own internal
  surface-potential Newton loop, not physical discontinuities in the
  device's I-V/Q-V characteristic.
- `force_dtmin` fires **continuously across the whole simulated span**
  (sustained ~5.5 Newton iterations/step throughout a 1us run per
  `doc/ring_oscillator_investigation.md`'s bottleneck decomposition), not as
  a one-time cost at a handful of discrete switching instants. That pattern
  is consistent with genuine continuous numerical stiffness (PSP103's
  femtofarad-scale internal parasitics plus its own internal Newton loop,
  combined with the ring's total lack of a stable DC equilibrium to warm-
  start from), not with the integrator repeatedly missing a small number of
  crossing events.

Put together with the sp_mos1 result — a model *with* literal discontinuities
already showed no net benefit and a real regression — the reasoning applied
directly: if a model built with actual region-switch discontinuities didn't
benefit, a model deliberately built to be smooth by design was not going to
benefit either, and there was no reason to expect the outcome to differ.
Rather than run that experiment and get the predictable answer, the whole
mechanism was removed.

## Why compact models don't suit per-comparison event detection in general

Both real-model results point at the same root cause: a compact model's
analog block is dense with comparisons that have nothing to do with the
physical region boundaries the mechanism was designed to catch — parameter
validation, junction-diode and capacitance-model region checks, temperature
clamps, and (for surface-potential models) internal iterative-solve
bookkeeping. `condition_is_vdep` correctly filters out the parameter-only
ones, but the genuinely voltage-dependent remainder in any model complex
enough to be industrially relevant is still large, and in fast circuits those
comparisons toggle on timescales where forcing an exact-root landing for each
one is worse than letting the adaptive step controller handle it with its
normal error control. `auto_tstops` avoids this failure mode entirely because
it only reacts to *known, parameter-derived* source edges (PWL/PULSE/SIN
breakpoints computed once from the source definition, not from per-step
Dual-tainted state) — a fundamentally cheaper and more targeted problem than
watching arbitrary in-model comparisons at runtime.

## Decision

Removed in full: `src/mna/va_events.jl`, the condition-slot fields/allocation
in `MNAContext`/`DirectStampContext`/`CompiledStructure`, the `va_cmp_*`
comparison-interception codegen in `vasim.jl` (including the
`cond_slot_counter` scope field), `va_event_callback`/`_merge_callback` in
`src/mna/solve.jl`, and the `va_events`/`interp_points`/`callback` kwargs and
IDA-fallback logic in `tran!`/`_tran_dispatch` (`src/sweeps.jl`). All
associated tests (`test/mna/va_events.jl`, the sp_mos1 regression testset in
`test/mna/oscillator_test.jl`) were removed with it.

`auto_tstops` (source-breakpoint-derived `tstops`/`d_discontinuities`) is
unaffected and remains the sole discontinuity-handling mechanism: it captures
the class of discontinuity that's cheap to know in advance (source edges) and
leaves the class that would require expensive runtime detection (arbitrary
in-model region switches) to the integrator's normal adaptive step control,
which the evidence above shows handles it at least as well in practice.
