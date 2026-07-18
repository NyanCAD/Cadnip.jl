# AC Noise Analysis — design plan

Status: **not implemented** (`noise!` is a stub, `src/ac.jl` LIMITATION 2; the
VA `white_noise`/`flicker_noise` primitives return `0.0`, `src/va_env.jl`).
CedarSim had a working implementation — see commit
`5d5ea8d4e7e17fd06f775eddd11f15a4731a4210` for the original as a reference.

Goal: `.noise`-style small-signal noise analysis on top of the existing AC
linearization, returning output-referred (and input-referred) noise spectral
density vs frequency, plus a per-device noise budget.

## What "done" looks like

Two acceptance tests already exist and encode the target API and golden data:

- `test/inverter_noise.jl` — GF180 inverter, expects `nsol = noise!(circuit)` and
  compares output noise density (V/√Hz) against an ngspice reference table.
- `test/ac.jl` — commented-out block (search "LIMITATION 2") sketching a
  noise transfer function + output-density test on the Butterworth filter.

Wire both up as the regression tests once the feature lands.

## Physics: per-device noise sources

Each noisy element contributes one or more uncorrelated current-noise sources
between two nodes, described by a one-sided power spectral density (A²/Hz):

- **Thermal (resistor):** `S_I = 4kT/R`, white.
- **Shot (junction current `I`):** `S_I = 2qI`, white.
- **Flicker / 1/f (MOSFET drain, VA `flicker_noise`):** `S_I = KF·I^AF / f^EF`.
- **VA models** already declare their noise via `white_noise(pwr, name)` and
  `flicker_noise(pwr, exp, name)` in the analog block; `pwr` is the PSD in A²/Hz.
  `noiseparams` in `src/spectre.jl` already walks the `ϵ`-prefixed model fields —
  reuse/extend that for source discovery.

## Method: adjoint (transfer-impedance) accumulation

This is the standard SPICE approach and reuses the AC machinery in `src/ac.jl`.

1. Solve the DC operating point and build the linearized `Y(jω) = G + jωC`
   (already done for `ac!`).
2. For output node `o`, solve the **adjoint** system once per frequency:
   `Yᵀ z = e_o`. Then the transfer impedance from a unit current injected at
   source `k`'s node pair `(a,b)` to the output is `H_k = z_a − z_b`. One solve
   yields the transfer function from *every* source to the output.
3. Sources are uncorrelated ⇒ sum of squares:
   `S_out(f) = Σ_k |H_k(jω)|² · S_k(f)`.
4. **Input-referred:** `S_in(f) = S_out(f) / |A_v(jω)|²`, where `A_v` is the
   gain from the designated input source to the output (one more AC solve, or
   reuse the `ac!` DSS).
5. **Integrated RMS noise** over a band: `sqrt(∫ S_out df)`.

Correlated sources (e.g. induced-gate / channel thermal noise correlation) need
a per-device correlation matrix rather than sum-of-squares — out of scope for v1,
note it as a follow-up.

## Prerequisite groundwork (break it up)

1. **Noise-source registry in the MNA context.** A list of
   `(node_a, node_b, psd_fn(f, T, opnt))` entries, populated during stamping.
   Add `alloc_noise_source!`/`record_noise!` alongside the existing `alloc_*`
   primitives in `src/mna/context.jl`.
2. **Codegen wiring.** Make `white_noise`/`flicker_noise` (`src/va_env.jl`,
   `src/vasim.jl`) register a source instead of returning `0.0` when the spec
   mode is `:noise`; leave the DC/transient `0.0` behavior intact. Add thermal
   stamps for builtin `Resistor` and shot stamps for `Diode`.
3. **`noise!` driver.** Reuse `ac!`'s DC-op + linearization, then loop the
   adjoint solve over `acdec` frequencies. Return a `NoiseSolution` carrying
   `freqs` (Hz), output PSD, input-referred PSD, and per-device contributions
   (the budget makes this genuinely useful for designers — a UX win).
4. **API + units.** Frequencies in Hz (SPICE `.noise` convention, matching
   `acdec` and the new Hz-based `magnitude_db`/`phase_deg`). Suggested surface:
   `noise!(circuit; out=:vout, in=:V1, freqs=acdec(...))`.
5. **Tests.** Un-stub `test/inverter_noise.jl` and the `test/ac.jl` block.

## Notes

- Do this on top of, not beside, the AC path — the noise linearization *is* the
  AC linearization. This dovetails with the AC-result-type unification todo in
  `scratchpad.md`: a single AC solution object should also carry noise.
- Keep everything differentiable (ForwardDiff) so noise can feed optimization,
  consistent with the Ecosystem pillar.
