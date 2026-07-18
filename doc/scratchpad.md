This file is for humans and agents to share ideas to work on and progress made.

Don't take big tasks head on, break them up in planning docs and prerequisite groundwork until they seem easy.

Spread your effort across different pillars.

# Things to work on

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

### AC analysis UX (todo)

We currently ship **two parallel AC result types** with different conventions,
and the high-level `ac!` only returns one of them. Worth unifying, but it needs
a deliberate decision because they answer to two different ecosystems:

- `solve_ac` → `ACSolution` (`src/mna/solve.jl`): SPICE-native. Frequencies in
  **Hz**, `sol[:name]` returns the complex response trajectory,
  `magnitude_db`/`phase_deg` give a Bode readout. This is the low-level path
  used by `test/mna/core.jl`.
- `ac!` → `ACSol` (`src/ac.jl`): a linearized `DescriptorStateSpace`. `freqresp`,
  `ss`, `bode`, `dss` here implement the **DescriptorSystems / ControlSystems**
  interfaces, so their convention is angular frequency in **rad/s** — that
  ecosystem's convention, not ours. `sol[:name]` returns a *SISO subsystem*, not
  a response vector. This is the path the high-level API and README expose.

The Hz-vs-rad/s split is not an accident: it falls out of `freqresp` being a
`DescriptorSystems.freqresp` method. Any unification has to keep the
ControlSystems interop (that's a headline selling point — see the Ecosystem
pillar) while giving SPICE users a Hz-first surface.

- [ ] Pick a single AC result type. Leaning toward `ACSol` (DSS-backed, strictly
      more capable — you can get `ss`/`bode`/poles/zeros out of it) and deriving
      the SPICE-native views from it, then retiring `solve_ac`/`ACSolution`
      (per CLAUDE.md: no parallel implementations).
- [ ] Make `sol[:name]` consistent across whatever survives — `ACSolution[:name]`
      returns a response vector, `ACSol[:name]` returns a subsystem. Same
      indexing syntax, different return type is a footgun.
- [ ] Keep `freqresp` in rad/s (ControlSystems contract) but document the
      convention loudly, and route the SPICE-facing helpers (`magnitude_db`,
      `phase_deg`, and any future `sol[:name]` trajectory access) through Hz so
      users never hand-convert. The Hz methods added above are a first step.
- [ ] Noise analysis (`noise!`) is still a stub (`src/ac.jl` LIMITATION 2);
      CedarSim had it — see the CedarSim porting pillar.
- [ ] Hierarchical device-observable access in AC (`src/ac.jl` LIMITATION 1):
      only flat top-level node/current names are observable today.

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

- [x] AC source phase (`V1 ... AC mag phase`)
- [x] Combined AC+transient sources, and Spectre `vsource`/`isource` AC support
- [x] Cleanup: drop dead backward-compat aliases
- [x] Cleanup: drop dead `netlist_utils.jl` composition operators
- [x] Control/analysis dot-cards no longer crash sema
- [x] Cleanup: drop dead DAECompiler-era `aliasextract.jl` and its `net_alias` stub
- [x] Cleanup: drop superseded `stamp_reactive_with_detection!` API, its two legacy `detect_or_cached!` overloads, and the always-empty codegen `detection_block`
- [x] Port the Makie extension (`explore`) to the MNA backend and wire it into `[extensions]` with a headless CairoMakie test
- [x] UX: Hz-based `magnitude_db`/`phase_deg` for the high-level `ac!` (`ACSol`) result, fixed AC docs/README
