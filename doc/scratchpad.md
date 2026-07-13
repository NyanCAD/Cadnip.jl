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

## Combined AC+transient sources, and Spectre `vsource`/`isource` AC support

Two related gaps in `src/spc/codegen.jl`'s MNA `cg_mna_instance!`, both from
`ac.jl`'s documented limitations list:

1. **SPICE `V1 ... AC mag phase SIN(...)`** (or PWL/PULSE): the transient
   branches built `VoltageSource`/`CurrentSource` without ever passing the
   `ac=` kwarg, even though the constructor and `stamp!` (`src/mna/devices.jl`)
   have supported simultaneous `dc`/`ac`/`tran` fields all along — `ac!()`
   reads `.ac` into `b_ac` independently of whatever `.tran` closure is
   attached. So a source with both specs silently dropped the AC one. Fixed
   by threading `ac_expr` through every transient-source branch (PWL
   zero-alloc + fallback, SIN, PULSE, and the unknown-type fallback) for both
   voltage and current sources.

2. **Spectre `vsource`/`isource` had no AC support at all** in the MNA
   backend (`master == "vsource"/"isource"` in `cg_mna_instance!`) — despite
   the doc note above claiming otherwise (that turned out to refer to the old
   DAECompiler-era `cg_instance!`/`cg_spice_instance!` path, not the current
   MNA codegen; grepping for `"mag"`/`"phase"` param names outside test files
   turned up nothing). `ac!()` on a Spectre netlist using `vsource dc=... mag=...`
   would just get `ac=0+0im` always. Added `_cg_spectre_ac_expr` (mirrors the
   SPICE `mag * exp(im * phase_deg * π/180)` construction, reading Spectre's
   `mag=`/`phase=` params) and threaded it through both masters' DC/PWL/SIN
   (vsource) and DC/PWL (isource) branches.

Added `test/ac.jl` cases: a SPICE source with both `AC 1 90` and `SIN(0 1 1k)`
(checks `ac!` sees the phasor and `tran!` still follows the sine), plus
Spectre `vsource mag=1 phase=90` and `isource mag=1` cases. Verified against
`test/basic.jl`, `test/transients.jl`, `test/mna/core.jl` (356/356), and
`test/mna/subckt_scoping.jl` with no regressions.
