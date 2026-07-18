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
For noise, the intended AD-perturbation approach is written up in `doc/noise_analysis_design.md`.
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

# In flight

## Noise simulation

Bring back noise analysis (CedarSim port). Design + research:
`doc/noise_analysis_design.md`. Approach: noise sources as differentiable
perturbation inputs riding the existing AC descriptor-state-space machinery,
collected through a deferred `MNAContext`-only channel that is a no-op on the
transient hot path (`DirectStampContext`), so DC/transient cost stays at zero.

- [ ] **N0 — Groundwork: noise-source channel.** Add a deferred noise-source
  list to `MNAContext` (COO-style, mirroring `b_ac_I`/`b_ac_V`). Do *not* add
  it to `DirectStampContext`. Make the VA `white_noise`/`flicker_noise`
  builtins (and builtin thermal/shot) ctx-aware: register a source on
  `MNAContext`, no-op on `DirectStampContext`, always return `0.0` for the
  value path so DC/transient numerics are byte-identical. Assert zero extra
  allocations in a transient benchmark.
- [ ] **N1 — PSD models at the DC bias.** Evaluate per-source spectral density
  at the operating point: thermal `4kT·g`, shot `2qI`, flicker `KF·I^AF/f`, VA
  `white_noise(pwr)` → `pwr`, `flicker_noise(pwr,exp)` → `pwr/f^exp`. Bias comes
  from the DC solution already computed by the AC path.
- [ ] **N2 — Transfer functions via the AC system.** Reuse `ac!`'s linearized
  `(jωC + G)`. Per output+frequency, one adjoint solve
  `(jωC+G)ᵀ x_adj = e_out` gives the transfer from *every* source at O(1) each
  (`H_k = x_adjᵀ e_k`); reuse the factorization across sources. (Equivalent to
  adding each source as a `B` column and reading `freqresp` — the
  perturbation-input framing — but the adjoint is the efficient evaluation for
  a single output.)
- [ ] **N3 — `noise!()` analysis + output.** Output PSD `Σ_k |H_k(jω)|² S_k(ω)`,
  band integration for total noise, input-referred via the input transfer
  function. Netlist `.noise` card + name-based access, mirroring `ac!`/`ACSol`.
- [ ] **N4 — Tests + validation.** Netlist tests (thermal noise of an RC =
  `4kT·R` shaped by the RC pole; op-amp input-referred noise) cross-checked
  against ngspice `.noise`, driven through the high-level API.
- [ ] **N5 (stretch) — differentiable / large-signal.** Differentiate output
  noise w.r.t. design params (the SciML payoff), and scope cyclostationary
  (PSS/PAC-style) noise on a periodically-time-varying linearization. Design
  only for now.

# Done

- [x] AC source phase (`V1 ... AC mag phase`)
- [x] Combined AC+transient sources, and Spectre `vsource`/`isource` AC support
- [x] Cleanup: drop dead backward-compat aliases
- [x] Cleanup: drop dead `netlist_utils.jl` composition operators
- [x] Control/analysis dot-cards no longer crash sema
- [x] Cleanup: drop dead DAECompiler-era `aliasextract.jl` and its `net_alias` stub
- [x] Cleanup: drop superseded `stamp_reactive_with_detection!` API, its two legacy `detect_or_cached!` overloads, and the always-empty codegen `detection_block`
- [x] Cleanup: drop dead DAECompiler-era `noiseparams`/`modelfields` noise extraction and the unused `SimSpec.ϵω` field
