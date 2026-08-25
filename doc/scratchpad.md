This file is for humans and agents to share ideas to work on and progress made.

Don't take big tasks head on, break them up in planning docs and prerequisite groundwork until they seem easy.

Spread your effort across different pillars.

**Keep entries to one short line.** This file exists to balance work across the
pillars, avoid duplicating effort, and track what is left to do — it is not a
design document. One sentence naming the task is enough; a design doc in `doc/`
and the pull request description do the heavy lifting, so link them instead of
summarising them here. If an entry needs a paragraph to explain, that paragraph
belongs in a design doc.

# Things to work on

## Documentation

A lot of the readme, design docs, and user docs are stale. Update them to reflect what's actually in the code, remove anything completely superseded, or add new undocumented functionality.

## Cleanup

There is both old CedarSim code and lingering duplication and warts.
Anything that makes the repo a more clean and lean codebase is good progress.
This may include larger refactors to simplify, deduplicated, and improve things.
At this stage we don't need any "backwards compatibility"

## CedarSim porting

The original CedarSim is at 5d5ea8d4e7e17fd06f775eddd11f15a4731a4210 and still has stuff that we don't.
Part of that is tests that we didn't port, part is inapplicable, but there are also some big features (noise etc) and extensions (makie etc).
Its user facing API also feels a bit more thoughtful rather than just grown.

## Production readiness

Adding more tests and benchmarks to see how we compare to ngspice, vacask, xyce.
Like we run the vacask benchmarks, we can explore the ngpsice and xyce test suites.
Or find other benchmarks online or even build our own well known circuits from the literature.
It would be good to have tests for the open source PDKs (atm sky130, gf180, ihp) as well as non-manufacturable ones such as asap7 and more.

## UX

Make sure the high level API is actually nice to use by working through mock designs, following the steps an experienced designer would take from hand derivations to post-layout simulations.
This can result in new examples and tests, gaps in our API, bugfixes, and more.

## Performance

Take one of our benchmarks where we're lagging behind the competition and do a deep dive on where we lose time.
Small tweaks can be done on the fly, for big tasks just note the findings below for another session to improve.
Or try to remove a workaround and see what would be needed to do without it.
Or explore how different solvers perform on different problems.

## Ecosystem

One of the big selling points of CedarSim and Cadnip is that we have access to the whole Julia ecosystem, and in particular SciML.
Basically anything that benefits from differentiability of the simulator.
There were ideas of design optimization, surrogates, inverse design, GPU acceleration, and more rigorous alternatives to Monte-Carlo.
Researching what is out there and trying it out is worthwile.

## Features

The most nebulous and least important at this stage: copying features from other simulators

# Progress

- [x] UX: enumerable DC operating point (`keys`/`values`/`pairs`/`haskey`/`get` on `DCSolution`)
- [x] AC source phase (`V1 ... AC mag phase`)
- [x] Combined AC+transient sources, and Spectre `vsource`/`isource` AC support
- [x] Cleanup: drop dead backward-compat aliases
- [x] Cleanup: drop dead `netlist_utils.jl` composition operators
- [x] Control/analysis dot-cards no longer crash sema
- [x] Cleanup: drop dead DAECompiler-era `aliasextract.jl` and its `net_alias` stub
- [x] Cleanup: drop superseded `stamp_reactive_with_detection!` API and the always-empty codegen `detection_block`
- [x] Port the Makie extension (`explore`) to the MNA backend, with a headless CairoMakie test
- [x] Cleanup: drop dead DAECompiler-era `noiseparams`/`modelfields` extraction and the unused `SimSpec.ϵω`
- [x] UX: Hz-based `magnitude_db`/`phase_deg` for `ac!` (`ACSol`)
- [x] AC UX: one Hz-first AC surface keeping the ControlSystems interop — retired `ACSolution`/`solve_ac` — design: `doc/ac_result_unification.md`
- [x] AC UX: `sol[:name]` return-type consistent across AC types; DSS subsystem moved to `subsystem(ac, :name)` — design: `doc/ac_result_unification.md`
- [x] AC UX: hierarchical device-observable access in AC (`ac[:x1_out]`, `NodeRef` indexing) — design: `doc/ac_result_unification.md`
- [x] Noise N0: deferred noise-source channel on `MNAContext` + resistor thermal noise — design: `doc/noise_analysis_design.md`
- [x] Noise N2: noise transfer functions via the AC linearization (`src/noise.jl`, `test/noise.jl`)
- [x] Noise N3 (partial): output PSD, per-source contributions, band-integrated `total_noise`
- [x] Noise N1: per-source PSD at the DC bias — diode shot, MOSFET channel thermal, VA `white_noise`/`flicker_noise`, builtin flicker
- [x] Noise N3 rest (input-referral): `noise!(…; input=:V1)`, `ns[:inoise]`, `total_noise(…; referred=:input)`
- [x] UX/design: netlist `.param` overrides reach the netlist — design: `doc/parameter_overrides.md`
- [x] UX/design: `test/design_flow.jl` walks an NMOS common-source stage op → DC → AC → transient → noise
- [ ] Finish the override design: reach raw device instance parameters (`r1=(r=2e3,)`) through the lens — design: `doc/parameter_overrides.md`
- [x] Diagnose unknown override names — design: `doc/parameter_overrides.md` §2
- [x] Bug: a deck declaring nothing (no `.param`, no subckt) observed as unobservable, so every override on it passed silently — `doc/FINDINGS.rst` 4
- [x] Codegen: a deck is a namespace — each loaded netlist gets its own module — design: `doc/parameter_overrides.md` §3
- [x] Sema/codegen: let a `.model` card read a `.param`, so a process corner is an ordinary sweep axis — design: `doc/parameter_overrides.md` §4
- [ ] UX/design follow-ups from the same walkthrough:
  - [x] Report device terminal currents in the operating point — design: `doc/operating_point_info.md`
  - [x] Report device small-signal parameters and region via Verilog-A operating-point variables — design: `doc/operating_point_info.md`
  - [x] `.dc` sweep with continuation (`dc!(::CircuitSweep)` warm-starts each point); transient sweeps still init cold
- [x] Documentation: README reviewed against a running Cadnip, building on #256 — install, world-age, string-macro and override-validation claims corrected
- [x] Documentation: user manual (`docs/src`) rewritten against the MNA backend, with a CI job that runs every `@example` in it
- [x] Documentation: the `doc/*.jmd` Weave set retired — MNA background and troubleshooting ported to `docs/src/background.md` / `docs/src/troubleshooting.md`, the DAECompiler/abstract-interpretation pair and the Weave build dropped
- [x] Codegen: one shared import list for the circuit and PDK paths — fixes a live `UndefVarError` for PDK subckts with E/G cards — design: `doc/codegen_unification.md` §1
- [x] Codegen: merge the duplicated `.model` lowering between the circuit and PDK paths — design: `doc/codegen_unification.md` §2
- [x] Codegen: clear the runtime warnings (world-age binding access, SciMLBase import) — design: `doc/codegen_unification.md` §3
- [x] Codegen: last world-age warning shape — an `isdefined(hdl_mod, …)` gate on a `.hdl` device-type lookup answered against a stale world with no warning at all (worse than the `getfield` reads already fixed); `latest_isdefined` in `src/util.jl` replaces all ten call sites — `test/basic.jl`, `test/mna/table_model.jl`, `test/mna/vadistiller.jl` now run clean under `--depwarn=error` — design: `doc/codegen_unification.md` §3
- [x] Bug: an F/H card inside a `.subckt` looks its sense source up unprefixed — fixed by scoping the sense name at the `get_current_idx` call sites
- [x] Codegen: let `sp"..."`/`spc"..."` expand inside a function body — design: `doc/codegen_unification.md` §4
- [x] Bug: a `.subckt` body reading a parent `.param` it has no local default for raised `UndefVarError` — the builder now binds `parent_params` as locals — design: `doc/parameter_overrides.md` §5
- [x] Noise N3 rest: dropped — Julia is the simulation API, a deck does not drive an analysis; `.noise` only had to stop failing to parse — design: `doc/noise_analysis_design.md` N3
- [x] Cleanup: delete the unreachable pre-MNA codegen path (`codegen!`, `SpCircuit`, `query.jl`, `generated.jl`) — design: `doc/codegen_unification.md` §5, `doc/FINDINGS.rst` 2 & 6
- [ ] Noise N4: validation against ngspice `.noise` through the high-level API
- [ ] Noise N5 (stretch): differentiable noise objectives + cyclostationary (PSS/PAC) noise
- [x] Bug: a netlist `.temp`/`.option temp` card dropped the rest of `spec` (gmin, tnom, tolerances) on rebind — the card still wins unconditionally (it's the deck's own control card, not a default to negotiate against a caller), fixed via `with_temp` — design/measurements: `doc/FINDINGS.rst` finding 1
- [x] `noise!` disagrees with the devices whenever a `.temp` card is present — the builder now records the temperature it stamped at on the context (`record_temp!`/`stamped_temp`) and `noise!` reads that, not `circuit.spec.temp`; circuit-level `with_temp`/`with_mode` stopped dropping the rest of the spec — `doc/FINDINGS.rst` finding 1
- [ ] UX/design: route `.temp` through `ParamLens` like a `.param`/`.subckt` default, so an explicit override is unambiguous (present-in-tree vs not) instead of magic-default guessing, and `circuit.spec`/`noise!` get a real resolved value for free — needs a collision-safe name (`.temp` and a user's own `.param temp=...` would share a namespace, unlike the `x1`/`X1` case) and crosses `MNASpec`'s current boundary from the `params`/lens tree — `doc/FINDINGS.rst` finding 1
- [x] Test brittleness: the `test/mna/vadistiller.jl` anchor-residual check asserted *exact* FP cancellation, below the floating-point floor of the quantity it measures — bounded against the ulp instead (28760af)
- [ ] Documentation: the README's world-age example is stricter than measured — `doc/FINDINGS.rst` 5
- [x] UX: `node_names(sol)` / `branch_names(sol)` — nothing classifies a name `keys(sol)` returns — `doc/FINDINGS.rst`
- [x] Cleanup: `test/basic.jl` carries twelve `#= =#` DAECompiler-era blocks — ported or dropped; the four still disabled now record the *measured* failure, not the 2022 guess
- [x] Features: subcircuit multiplicity — `m` on an `X` line / `.subckt` card scales every device inside and composes across nesting
- [ ] Bug: Spectre `type=pwl wave=[...]` — sema has no `sema_visit_ids!` for a `SpectreArray`, so the deck dies before codegen — disabled block in `test/basic.jl`
- [ ] Bug: alternate E/G forms (`vol=`/`cur=`) fail in the parser with `LString(::Nothing)` — disabled block in `test/basic.jl`
- [ ] Bug: an instance param reading another param of the same `X` line (`nrd='w/2'`) is built in the caller's scope, so `UndefVarError: w` — disabled block in `test/basic.jl`
- [x] Features: semiconductor resistor — a `.model … r` card resolves `rsh`/geometry/`tc1`/`tc2`, instance `tc` too; `docs/src/devices.md`, `test/basic.jl`
- [ ] CedarSim porting: model binning has a full runtime half in `src/spectre.jl` and no producer since §5 — blocks binned PDKs — design: `doc/codegen_unification.md` §5 "Worth porting"
- [ ] UX: generated code carries no netlist source positions, so a stamp error points at `codegen.jl` — design: `doc/codegen_unification.md` §5 "Worth porting"
- [ ] Bug: two simultaneously active conditional instances of one name stamp twice, silently — design: `doc/codegen_unification.md` §5 "Worth porting"
- [x] Features: every `.option` naming an `MNASpec` field reaches the builder (`gmin`, `tnom`, the `$simparam` tolerances), recorded as `stamped_spec(ctx)`; `.option scale` stays unimplemented but warns — `doc/FINDINGS.rst` 2
- [ ] Features: `.option scale` (geometry scale factor) — nothing consumes it; needs every device's geometric parameters to scale
- [ ] UX: netlist introspection (`circuit.r1` → kind + defining line), sketched by the deleted `SpRef` — design: `doc/codegen_unification.md` §5 "Worth porting"
