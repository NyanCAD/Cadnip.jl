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

### 2. Make `sol[:name]` return-type consistent — **done**

`ACSolution[:name]` returned a response vector; `ACSol[:name]` returned a SISO
subsystem. Same indexing syntax, different return type was a footgun. Resolved:

- `ACSol[:name]` now returns a complex **response vector** over the analysis's
  Hz grid — the natural SPICE answer, matching `ACSolution[:name]`. To carry the
  grid, `ac!` gained an optional `freqs` argument: `ac!(circuit, acdec(...))`.
  Without a grid, `ac[:name]` errors and points at `freqresp`.
- The SISO descriptor subsystem moved to the explicit accessor
  `subsystem(ac, :name)` (control-systems object → `ss`/`bode`/poles/zeros).
- `magnitude_db(ac, :name)` / `phase_deg(ac, :name)` two-arg forms read the
  stored grid; the three-arg Hz forms remain.
- `ACSolution[:name]` also now resolves branch currents, not just node voltages,
  so both types index alike.

Still open for the full unification (sub-task 1): retiring `ACSolution`/`solve_ac`
onto a single type. The two `[:name]` surfaces now agree, so that retirement is a
mechanical follow-up rather than a behavior change.

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
