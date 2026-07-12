This file is for humans and agents to share ideas to work on and progress made.

Don't take big tasks head on, break them up in planning docs and prerequisite groundwork until they seem easy.

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

## Performance

Take one of our benchmarks where we're lagging behind the competition and do a deep dive on where we lose time.
Small tweaks can be done on the fly, for big tasks just note the findings below for another session to improve.
Or try to remove a workaround and see what would be needed to do without it.
Or explore how different solvers perform on different problems.

## Ecosystem

One of the big selling points of CedarSim and Cadnip is that we have access to the whole Julia ecosystem, and in particular SciML.
Basically anything that benefits from differentiability of the simulator.
There were ideas of design optimization, surrogates, inverse design, and more rigorous alternatives to Monte-Carlo.
Researching what is out there and trying it out is worthwile.

## Features

The most nebulous and least important at this stage: copying features from other simulators

# Progress

## AC source phase (`V1 ... AC mag phase`)

NyanSpectreNetlistParser's `ACSource` node only captured magnitude; phase was
parsed-and-discarded (`# TODO acphase` in forms.jl), so `ac.jl` had to
document phase as an unsupported limitation even though the MNA codegen
already had a `ac_phase` slot wired to `mag * exp(im * phase_deg * π/180)` —
it just always saw `nothing`. Added `acphase::Union{Nothing, EXPR}` to
`ACSource` and an optional trailing-expression parse (guarded by
`!is_kw`, so it doesn't eat a following `SIN(...)`/keyword), then threaded
it through `sema_visit_ids!` and both SPICE `cg_mna_instance!` call sites in
Cadnip. Added a `test/ac.jl` case asserting a 90 phase source resolves to
`+j`. The Spectre path already supported `acphase=` as a named param, so this
was SPICE-only.
