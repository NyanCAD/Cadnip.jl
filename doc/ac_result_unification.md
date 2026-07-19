# AC result-type unification — design plan

Status: **done.** Sub-tasks 1–4 landed: `ACSol` is the single AC result type,
its `sol[:name]` is a Hz response vector, the DSS interop moved to
`subsystem`/`freqresp`, and hierarchical node access works. This records the
design; the tracking checkboxes live in `doc/scratchpad.md`.

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

### 1. Unify on a single AC result type — **done**

Kept `ACSol` (DSS-backed, strictly more capable — `ss`/`bode`/poles/zeros/
`freqresp`) and derived the SPICE-native views from it, then retired
`solve_ac`/`ACSolution`. Concretely, as landed:

- `ACSol` carries the SPICE-native surface: an Hz `sol[:name]` response vector
  (over the grid passed to `ac!(circuit, freqs)`) and the two-argument
  `magnitude_db`/`phase_deg` Bode readout, all on top of `freqresp`.
- `test/mna/core.jl`'s `solve_ac` call sites moved to `ac!` (netlist-driven
  behavioral tests, per CLAUDE.md), and `test/common.jl` dropped the import.
- Deleted `ACSolution`, all `solve_ac` methods, and the duplicated accessors;
  `magnitude_db`/`phase_deg` are now bare generics in `solve.jl` whose only
  methods live on `ACSol`.

One correctness note surfaced during the port: the old low-level
`solve_ac(sys, freqs)` excited with the DC rhs `b`, so it treated a plain DC
source as an AC stimulus. `ac!` uses `b_ac` (explicit `AC` magnitudes) — the
correct SPICE semantics — so ported tests give their sources an `AC`/`ac=` spec.

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
Landed together with sub-task 1: with `ACSolution` retired there is only one
`[:name]` surface, and it is the response vector.

### 3. Hz-first helpers, rad/s only where the contract demands it — **done**

`freqresp` stays in rad/s (it *is* the ControlSystems contract) while every
SPICE-facing helper — `magnitude_db`, `phase_deg`, `sol[:name]` trajectory
access — routes through Hz so users never hand-convert. The one rad/s boundary
(`freqresp`) is documented in the README AC section and the `ac.jl` header.

### 4. Hierarchical device-observable access — **done (for what MNA models)**

Subcircuit-internal nodes flatten into the flat name table (`:x1_out`), so they
are observable by name on an `ACSol` exactly like top-level nodes, and a
`NodeRef` from `scope(...)` now indexes an `ACSol` too (parity with
`DCSolution`/transient). The only thing that is *not* a system variable is a
voltage across a device wired between two named nodes (an inductor's `V(l3)`) —
that is a derived quantity, taken as a node difference. So there is no remaining
hierarchical-access gap for anything MNA represents as a state; genuine
device-internal observables would need the device to expose them as nodes.

## Notes

- Noise analysis is tracked separately in `doc/noise_analysis_design.md`; a
  unified AC solution object is the natural place to also carry noise, so land
  this cleanup in a way that leaves room for that.
- Keep everything ForwardDiff-differentiable, consistent with the Ecosystem
  pillar (AC/Bode responses as differentiable objectives).
