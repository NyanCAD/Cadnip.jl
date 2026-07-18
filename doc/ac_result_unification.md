# AC result-type unification — design plan

Status: **not started.** This records the design for cleaning up the AC-analysis
UX gaps discovered while adding the Hz-based `magnitude_db`/`phase_deg` helpers
(`src/ac.jl`). Small, self-contained follow-ups; the tracking checkboxes live in
`doc/scratchpad.md`.

## The problem: two parallel AC result types

We currently ship two AC solution types with different conventions, and the
high-level `ac!` only returns one of them:

- **`solve_ac` → `ACSolution`** (`src/mna/solve.jl`): SPICE-native. Frequencies
  in **Hz**; `sol[:name]` returns the complex response *trajectory*;
  `magnitude_db`/`phase_deg` give a Bode readout. The low-level path used by
  `test/mna/core.jl`.
- **`ac!` → `ACSol`** (`src/ac.jl`): a linearized `DescriptorStateSpace`.
  `freqresp`, `ss`, `bode`, `dss` here implement the **DescriptorSystems /
  ControlSystems** interfaces, so their convention is angular frequency in
  **rad/s** — that ecosystem's convention, not ours. `sol[:name]` returns a
  *SISO subsystem*, not a response vector. This is the path the high-level API
  and README expose.

The Hz-vs-rad/s split is not an accident: it falls out of `freqresp` being a
`DescriptorSystems.freqresp` method. Any unification has to **keep the
ControlSystems interop** (a headline selling point — see the Ecosystem pillar)
while giving SPICE users a Hz-first surface. Per CLAUDE.md we carry no parallel
implementations, so the end state is one type, not two.

## Sub-tasks

### 1. Unify on a single AC result type

Leaning toward keeping `ACSol` (DSS-backed, strictly more capable — you can get
`ss`/`bode`/poles/zeros/`freqresp` out of it) and deriving the SPICE-native
views from it, then retiring `solve_ac`/`ACSolution`. Concretely:

- Give `ACSol` the SPICE-native surface `ACSolution` has today: an Hz-based
  `sol[:name]` response trajectory and the `magnitude_db`/`phase_deg` Bode
  readout (the Hz methods added on top of `freqresp` are the first step).
- Port `test/mna/core.jl`'s `solve_ac` call sites to the unified type.
- Delete `ACSolution`, `solve_ac`, and the now-duplicated accessors.

### 2. Make `sol[:name]` return-type consistent

`ACSolution[:name]` returns a response vector; `ACSol[:name]` returns a SISO
subsystem. Same indexing syntax, different return type is a footgun. Pick one
meaning (a response vector reads as the natural SPICE answer; expose the
subsystem via an explicit accessor such as `subsystem(ac, :name)` or the
existing `ac[:name]` kept only if it clearly means "control-systems object").

### 3. Hz-first helpers, rad/s only where the contract demands it

Keep `freqresp` in rad/s (it *is* the ControlSystems contract) but route every
SPICE-facing helper — `magnitude_db`, `phase_deg`, `sol[:name]` trajectory
access, and a future `.ac` netlist card — through Hz so users never
hand-convert. Document the one rad/s boundary (`freqresp`) prominently.

### 4. Hierarchical device-observable access (`src/ac.jl` LIMITATION 1)

Only flat top-level node/current names are observable today; device-internal
variables via a hierarchical path (`sys.l3.V`) are not. Voltages across devices
wired between top-level nodes can already be had as node differences; the gap is
genuinely-internal nodes. Scope this against whatever the unified type exposes.

## Notes

- Noise analysis is tracked separately in `doc/noise_analysis_design.md`; a
  unified AC solution object is the natural place to also carry noise, so land
  this cleanup in a way that leaves room for that.
- Keep everything ForwardDiff-differentiable, consistent with the Ecosystem
  pillar (AC/Bode responses as differentiable objectives).
