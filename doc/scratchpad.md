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

- [x] UX: enumerable DC operating point — `keys`/`values`/`pairs`/`haskey`/`get` on `DCSolution` so an operating point can be iterated, `Dict`-collected, and diffed without knowing node names up front (nodes + branch currents; string names delegate to the symbol lookup)
- [x] AC source phase (`V1 ... AC mag phase`)
- [x] Combined AC+transient sources, and Spectre `vsource`/`isource` AC support
- [x] Cleanup: drop dead backward-compat aliases
- [x] Cleanup: drop dead `netlist_utils.jl` composition operators
- [x] Control/analysis dot-cards no longer crash sema
- [x] Cleanup: drop dead DAECompiler-era `aliasextract.jl` and its `net_alias` stub
- [x] Cleanup: drop superseded `stamp_reactive_with_detection!` API, its two legacy `detect_or_cached!` overloads, and the always-empty codegen `detection_block`
- [x] Port the Makie extension (`explore`) to the MNA backend and wire it into `[extensions]` with a headless CairoMakie test
- [x] Cleanup: drop dead DAECompiler-era `noiseparams`/`modelfields` noise extraction and the unused `SimSpec.ϵω` field
- [x] UX: Hz-based `magnitude_db`/`phase_deg` for the high-level `ac!` (`ACSol`) result, fixed AC docs/README
- [x] AC UX: unify the two AC result types onto one Hz-first surface that keeps the ControlSystems interop — retired `ACSolution`/`solve_ac`; `ac!`/`ACSol` is the single AC path (DSS-backed `freqresp`/`ss`/`bode` + SPICE-native `sol[:name]`)
- [x] AC UX: make `sol[:name]` return-type consistent across AC types — `ACSol` now indexes to a complex response vector (matching DC/tran); the DSS subsystem moved to `subsystem(ac, :name)`; `ac!(circuit, freqs)` carries a Hz grid and 2-arg `magnitude_db`/`phase_deg`
- [x] AC UX: hierarchical device-observable access in AC — subcircuit nodes flatten into the name table so `ac[:x1_out]` resolves, and a `NodeRef` from `scope(...)` indexes an `ACSol` (parity with DC/tran); genuine device-across voltages remain node differences
- [x] Noise N0: deferred noise-source channel on `MNAContext`, no-op on `DirectStampContext` (zero transient cost); resistor Johnson–Nyquist thermal noise (`4kT·G`) is the first registered source, PSD helper covers thermal/shot/white/flicker — design: `doc/noise_analysis_design.md`. Still to wire for N1: builtin diode/BJT/MOSFET shot+flicker and the VA `white_noise`/`flicker_noise` codegen path (needs branch context at the contribution site).
- [x] Noise N2: noise transfer functions via the AC linearization — `noise!(circuit, output; freqs)` adjoint-solves `(jωC+G)ᵀ x_adj = e_out` per frequency, transfer `H_k = x_adj[p_k]-x_adj[n_k]` for every source at O(1) (`src/noise.jl`, `test/noise.jl`)
- [x] Noise N3 (partial): `noise!()` output PSD `Σ_k |H_k|² S_k`, per-source contributions (`ns[:onoise]`/`ns[:devname]`), band-integrated `total_noise` validated against RC `4kTR`-shaped noise and `kT/C`. Source-agnostic: every registered source (thermal, diode shot, MOSFET channel thermal) lights up for free
- [x] Noise N1: per-source PSD at the DC bias — thermal (done), and:
  - [x] builtin diode shot noise (`2q·|I|`) registered from the junction bias via `register_shot_noise!`
  - [x] builtin MOSFET channel thermal noise (`4kT·(2/3)·gm`) registered from the bias gm via `register_channel_thermal_noise!`
  - [x] VA `white_noise`/`flicker_noise` register on the noise channel — the contribution codegen binds the LHS branch into `_mna_noise_p_`/`_mna_noise_n_`, the call still evaluates to `0.0`, and an `if noise_enabled(ctx)` gate (constant-false on `DirectStampContext`) folds the whole registration — power expression included — out of the transient hot path. Sources are named `<instance>_<label>` (`:q1_rb`, `:q1_flicker`) and scale with `$mfactor`. This lights up **every** VADistiller model's SPICE3 noise model at once (resistor, diode, BJT, MOS1/2/3/6/9, JFET, MESFET, BSIM3/4) plus any PDK/user Verilog-A, so BJT shot+flicker and MOSFET flicker come in through the models rather than as builtin-stamp special cases
  - [x] builtin `Diode`/`DiodeWithCap`/`SimpleMOSFET` flicker (1/f) noise (`KF·|I|^AF / f^FFE`) — `KF`/`AF`/`FFE` cards (off by default) register through the **same** `register_flicker_noise!(ctx, p, n, pwr, exp)` entry point the VA `flicker_noise(pwr, exp)` lowering uses (no parallel construction), reading the flicker coefficient from the DC bias current; exercises the `FLICKER` kind end-to-end through `noise!` for the reference builtins (`test/mna/noise.jl`, `test/noise.jl`)
- [x] Noise N3 rest (input-referral): `noise!(circuit, output; freqs, input=:V1)` refers the output noise back to a voltage-source input via the same adjoint — the unit-voltage transfer `H(jω)=x_adjᵀ b_in` is read for free per frequency, `ns[:inoise] = onoise/|H|²`, `total_noise(ns; referred=:input)`. Validated: divider input-referred `4kT·2000` and RC input-referred flattening to the bare `4kTR` (`src/noise.jl`, `test/noise.jl`)
- [x] UX/design: netlist `.param` overrides actually reach the netlist. Walking a
  hand-designed NMOS common-source stage through the high-level API turned up a
  silent-wrong-answer class of bug: `MNACircuit(ckt; vbias=1.2)` and
  `Sweep(vbias=…)` resolved to the netlist default for every point, a subcircuit
  parameter the instance line spelled out was unreachable from an override
  entirely, and `alter` on a circuit built without that knob threw instead of
  introducing it. A swept design therefore read as a perfectly flat transfer
  curve — no error anywhere.
  The root cause was not a missing design but a disabled one: `ParamLens`'s
  constructor had `#nnt = canonicalize_params(nt)` commented out, so the lens
  only ever read the canonical `params=(…)` shape while users (and
  `ParamObserver`, which reports `compact_params`) wrote the compact one. Fixed
  by restoring canonicalization, which collapses to one rule — **a leaf is a
  parameter of the scope, a group is a child**, `params=` names a parameter
  explicitly when a name is both (`.param x1` next to `X1`). Plus codegen
  precedence lens > instance line > `.subckt` default, and `alter` inserting
  along the path so a swept axis needs no seeding. `alter` is also one function
  instead of two shadowing ones (`using Cadnip` gets the documented
  `alter(circuit; …)`). `test/design_flow.jl` walks the stage
  op → DC transfer curve → AC → transient → noise against the hand derivation, so
  the parameterization contract is pinned by a design rather than by a unit test.
  The two `test/mna/audio_integration.jl` sweeps were green *because* they were
  no-ops; they now assert the swept value reaches the source.
- [ ] Parameterization follow-ups, in the order they bite:
  - [ ] **Device instance parameters are not overridable** — `alter(c; var"r1.r"=2e3)`,
    `r1=(r=2e3,)` and `var"r1.params.r"` are all silently ignored, so W/L can only
    be swept if the netlist author wrapped the value in a `.param`. The
    netlist-*text* `alter(io, ast; r1=(r=4.0,))` honours exactly this spelling, and
    the commented-out CedarSim test at `test/basic.jl` ("device == param") shows
    device params were part of the lens tree by design (`i1=(dc=-1,)`,
    `rload=(r=2000.0,)`). Each device's codegen builds its params at a single site
    (MOSFET/BJT/Diode/OSDI share one `param_kwargs` list; R/C/L one value
    expression each), so it is ~6 hooks of the form
    `Base.getproperty(lens, :r1)(; r=1000.0, m=1)` — constant-folded to the
    literals under an empty lens, but in the restamp hot path, so gate it on the
    zero-alloc transient tests and the vacask benchmarks.
  - [ ] **`.model` cards cannot reference a `.param`** — `.param vt0=0.7` +
    `.model nch nmos vto=vt0` fails at *load* with `UndefVarError: vt0`, because
    model cards are emitted as module-level `const`s outside the builder
    (`const rmod = (rsh = rval,)`). This blocks PDK corner parameterization
    (`.model … vto='vto_nom+dvt'`). Model cards need to move inside the builder
    (or take a lens-resolved closure).
  - [ ] Override names are not validated: a typo'd knob is inert, and so is a
    device param (above). `ParamObserver` already walks the full tree, so a check
    at `MNACircuit` construction could say "`vinn` is not a parameter of this
    circuit; did you mean `vin`?" or "`x1` is an instance — write `x1=(rv=…)`".
  - [ ] Subcircuit builders are named after the `.subckt` (`divider` →
    `divider_mna_builder`), so two netlists loaded into the same module that both
    define a `.subckt divider` silently overwrite each other's builder — the
    second definition wins and the first netlist's instances then call it with
    the wrong keyword arguments. Hit while writing `test/params.jl`. The name
    needs the netlist's identity in it (the `sp"..."` gensym already has one).
- [ ] UX/design follow-ups found on the same walkthrough, none blocking:
  - [ ] No device-level operating point. "Is M1 in saturation, what is gm?" can
    only be answered by hand from node voltages — there is no `.op` report and no
    device current (only voltage-source branch currents are state variables). The
    noise channel is the proven pattern for this: a deferred op-info channel on
    `MNAContext`, no-op on `DirectStampContext`.
  - [ ] No `.dc` sweep analysis. `CircuitSweep` covers it, but each point solves
    from zeros — `dc_solve_with_ctx` has no `u0`, so there is no continuation
    (warm start from the previous point) the way SPICE `.dc` does it.
- [ ] Noise N3 rest: `.noise` netlist card driven through the high-level API
- [ ] Noise N4: validation against ngspice `.noise` through the high-level API
- [ ] Noise N5 (stretch): differentiable noise objectives + cyclostationary (PSS/PAC) noise
