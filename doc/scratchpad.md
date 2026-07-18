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

- [ ] Unify the two AC result types — `solve_ac`→`ACSolution` (SPICE-native, Hz) and `ac!`→`ACSol` (a `DescriptorStateSpace` in rad/s for ControlSystems/DescriptorSystems interop) — into one that keeps the ControlSystems interop but presents a Hz-first surface
- [ ] Make `sol[:name]` return-type consistent across AC types (`ACSolution` gives a response vector, `ACSol` gives a SISO subsystem)
- [ ] Route the SPICE-facing AC helpers through Hz while keeping `freqresp` in rad/s per the ControlSystems contract
- [ ] AC noise analysis (`noise!`, `src/ac.jl` LIMITATION 2) — see `doc/noise_analysis_plan.md`
- [ ] Hierarchical device-observable access in AC (`src/ac.jl` LIMITATION 1)

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
