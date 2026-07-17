# Noise analysis design

Noise analysis is a CedarSim feature we still want to port (`doc/scratchpad.md`,
CedarSim pillar). This note records the intended approach so the design isn't
lost, and points at the current entry points.

## Status

Not implemented. The scaffolding that exists today:

- `src/ac.jl` — `noise!()` is scoped in the header comment but unimplemented.
  The AC path already builds a linearized descriptor state-space system
  (`E·dx = A·x + B·u, y = C·x`) at the DC operating point, which is exactly the
  linearization a noise analysis consumes.
- `src/va_env.jl` / `src/vasim.jl` — the Verilog-A `white_noise` /
  `flicker_noise` builtins are parsed (NyanVerilogAParser keywords) and stubbed
  to return `0.0`. These are the real per-device noise-source hooks; a noise
  analysis lights them up instead of zeroing them.

Builtin device noise (thermal `4kT/R`, shot `2qI`, flicker `KF·I^AF/f`) has no
home yet and would attach to the MNA device stamps.

## Two candidate formulations

### A. Classic adjoint-PSD (the SPICE method)

At each frequency: assemble a per-device noise-current PSD, solve the adjoint
(transpose) of the AC system for the transfer function from each noise source to
the output, then sum `|H|²·S` contributions (and cross-terms for correlated
sources). Well-trodden, matches ngspice/Xyce/vacask, easy to cross-check.

### B. AD-perturbation (the desirable approach)

Represent each noise source as a differentiable **perturbation input** (`ϵ`
terms) injected at a perturbation frequency (`ϵω`), and obtain the
source→output transfer functions by differentiating the simulator output with
respect to those inputs — reusing the same AD that already flows through the MNA
stamps. Instead of hand-writing an adjoint solve, the transfer functions fall
out of AD, and correlated / non-white / large-signal (cyclostationary) noise
compose naturally because the perturbations ride through the real device
equations rather than a separately-derived small-signal noise model.

**This is the approach we want.** It is the noise-shaped instance of Cadnip's
core bet — differentiability of the whole simulator (`doc/scratchpad.md`,
Ecosystem/SciML pillar) — and it distinguishes us from the classic simulators
rather than re-deriving them. Approach A is worth keeping in mind only as a
cross-check oracle for a handful of textbook circuits.

### Prior art in git history

CedarSim/DAECompiler already used the `ϵ`-perturbation representation: device
models carried `ϵ`-prefixed fields (noise-perturbation inputs) and `SimSpec`
carried an `ϵω` perturbation frequency. A `noiseparams` helper walked the
circuit builder with a `ParamObserver` mock to harvest the set of `ϵ` knobs
across the hierarchy. That enumeration code was removed in **b771716** as dead
(it cataloged knobs but computed no PSDs, matrices, or transfer functions, and
was welded to the old struct-field representation). Revive it from that commit
if the `ϵ`-field harvesting pattern is useful — but note that MNAContext's
structure-discovery pass already flattens the hierarchy during stamping, so the
natural place to register noise sources is that same pass, not a separate
`ParamObserver` walk.
