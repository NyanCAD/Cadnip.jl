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
- [x] UX/design: netlist `.param` overrides reach the netlist — `MNACircuit(ckt; vbias=1.2)` and `Sweep(vbias=…)` used to resolve to the default at every point, so a swept design read as a flat transfer curve with no error. Restored `canonicalize_params` in `ParamLens` (one rule: a leaf is a parameter, a group is a child), codegen precedence lens > instance line > `.subckt` default, `alter` inserting along the path, and one `alter` instead of two shadowing ones — design: `doc/parameter_overrides.md`
- [x] UX/design: `test/design_flow.jl` walks a hand-sized NMOS common-source stage op → DC transfer curve → AC → transient → noise against the square-law derivation; the two `test/mna/audio_integration.jl` sweeps now assert the swept value reaches the source (they used to pass as no-ops)
- [ ] Finish the override design: reach raw device instance parameters (`r1=(r=2e3,)`, `m1=(w=…)`) through the lens. Named parameters land; device instance parameters were meant to be in the same tree (see the commented-out "device == param" test in `test/basic.jl`) and codegen never consults the lens at device sites — design: `doc/parameter_overrides.md`
- [x] Diagnose unknown override names — `ParamObserver` (the recording `AbstractParamLens`) already reports every name a circuit declares, so building once with it in place of `ParamLens` yields the whole tree and `src/param_overrides.jl` diffs the override tuple against it; dispatch keeps observation off the hot path and the result is memoized per builder, so `alter` pays once per sweep, not per point. `MNACircuit`/`alter`/sweeps all check, and the message names the fix ("`x1` is an instance, write `x1=(rv=…)`"). Two overrides that reached nothing fixed alongside: dotted selectors at construction (`MNACircuit(c; var"x1.r1val"=…)`) and `CircuitSweep`'s base circuit — design: `doc/parameter_overrides.md` §2
- [x] Codegen: a deck is a namespace — `Base.include(mod, SpiceFile(...))` used to eval generated code straight into the caller's module, so two netlists that each define `.subckt divider` overwrote each other's builder; identical positional signatures meant no error, just the second deck answering for the first. `SpiceFile`/`SpectreFile` load a *complete* deck (they parse with `implicit_title=true`), and a `.subckt` is deck-local in SPICE, so each deck now gets a module of its own and only the circuit builder is bound in the caller's module — the isolation VA files, PDKs, and `MNACircuit(path)` already had. Two `sp"..."` decks in one *local* scope still collide, which is ordinary Julia redefinition in a scope the author wrote (`src/spc/interface.jl`, `test/mna/subckt_scoping.jl`, `doc/parameter_overrides.md` §3)
- [ ] Sema/codegen: let a `.model` card read a `.param`. `.model nch nmos vto=vt0` fails at load with `UndefVarError` with no override in play, because model cards are hoisted out of the builder
- [ ] UX/design follow-ups from the same walkthrough:
  - [x] Report device terminal currents in the operating point, via an op-info channel on `MNAContext` — a deferred channel mirroring the noise one (nothing on `DirectStampContext`, `op_enabled(ctx)` constant-false there, so transient restamping is untouched). Builtin R/D/MOSFET/current-source register at their stamp; the Verilog-A stamp codegen accumulates one current per port over every branch that reaches it, following the `V(int,ext) <+ 0` collapse a model does when `rd=0` — so it lights up every VADistiller/PDK/user model at once. `dc!` reads the channel off the rebuild it already does at the converged point: `op[:i_m1_d]`, `terminal_currents(op)`, and they join `keys`/`values`/`pairs`/`haskey`/`show` — design: `doc/operating_point_info.md`
  - [ ] Report device small-signal parameters and region (gm, gds, triode/saturation), which needs Verilog-A operating-point variables in the VA front end
  - [x] `.dc` sweep with continuation: `dc_solve_with_ctx`/`solve_dc`/`dc!` take a `u0`, and `dc!(::CircuitSweep)` warm-starts each point from the previous *converged* one (`continuation=true` by default, as SPICE `.dc` does). `DCSolution` carries `converged`, which is what gates the hand-off; a guess of the wrong length (a point that changed the system size) is dropped, and the GMIN/source-stepping fallbacks still restart from zeros, so a bad guess costs iterations and never a solution. Transient sweeps still DC-init cold per point — `tran!(::CircuitSweep)` inits through `CedarDCOp`, which has no `u0` seam yet
- [ ] Noise N3 rest: `.noise` netlist card driven through the high-level API
- [ ] Noise N4: validation against ngspice `.noise` through the high-level API
- [ ] Noise N5 (stretch): differentiable noise objectives + cyclostationary (PSS/PAC) noise
