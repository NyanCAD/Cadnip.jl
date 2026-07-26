# Noise analysis design

Noise analysis is a CedarSim feature we still want to port (`doc/scratchpad.md`,
CedarSim pillar). This note records the intended approach and the research into
how it threads through the current MNA system **without slowing DC/transient**.

## Status

Single-operating-point linear noise analysis is implemented (`src/noise.jl`):
`noise!(circuit, output; freqs)` computes the output-noise PSD and per-source
contributions, and `input=:V1` refers the noise back to a voltage-source input.
Sources register from the builtin stamps (resistor thermal, diode shot+flicker,
MOSFET channel thermal+flicker) *and* from Verilog-A `white_noise`/`flicker_noise`,
which means every VADistiller model's full SPICE3 noise model — resistor, diode,
BJT, MOS1/2/3/6/9, JFET, MESFET, BSIM3/BSIM4 — is live. The builtin flicker
sources register through the very same `register_flicker_noise!` entry point the
VA lowering uses, so there is one noise code path, not two. The remaining work is
the `.noise` netlist card surface and the ngspice cross-validation (N4). The
scaffolding this builds on:

- `src/ac.jl` — the AC path builds a linearized descriptor state-space system
  (`E·dx = A·x + B·u, y = C·x`) at the DC operating point, which is exactly the
  linearization the noise analysis consumes; `noise!` reuses that
  rebuild-at-op-point pattern.
- `src/vasim.jl` — the Verilog-A `white_noise` / `flicker_noise` builtins are
  parsed (NyanVerilogAParser keywords) and lowered by `mna_noise_source`. These
  are the real per-device noise-source hooks.

Builtin device noise (thermal `4kT·g`, shot `2qI`, flicker `KF·|I|^AF/f^FFE`) is
attached to the MNA device stamps: the resistor registers thermal noise, the
diode shot+flicker, and the MOSFET channel-thermal+flicker, all at the DC bias
the noise channel is built at. The flicker registrations call the shared
`register_flicker_noise!(ctx, p, n, pwr, exp)` — the same one `mna_noise_source`
emits for VA `flicker_noise` — so builtins and Verilog-A converge on one path.

## How AC threads through MNA today (the pattern to reuse)

`ac!()` (`src/ac.jl`) does structure discovery once, solves the DC operating
point, rebuilds at that point, and reads the linearized matrices:

```
G = assemble_G(ctx; gshunt=gmin)   # resistive Jacobian at the op point
C = assemble_C(ctx)                # reactive Jacobian
b_ac = get_rhs_ac(ctx)             # AC excitation column(s)
dss(-G, C, B=b_ac, C_out=I, D=0)   # H(jω) = (jωC + G)⁻¹ · b_ac
```

The important detail for noise is **how the AC excitation is stored**:
`MNAContext` carries a *fully deferred* AC channel — `b_ac_I::Vector{MNAIndex}`,
`b_ac_V::Vector{ComplexF64}` (`context.jl:202`). AC sources append to it during
stamping; it is only materialized when `ac!` runs. DC and transient never touch
it.

Crucially, the **transient hot path uses a different context**:
`DirectStampContext` (`value_only.jl`) carries *only* `G_nzval`, `C_nzval`, and
`b` — there is no `b_ac` channel and no AC machinery in it at all. Restamping
during a transient solve writes straight to sparse `nzval` arrays via a
precomputed COO→nzval map, and never allocates. So the AC channel already costs
transient exactly nothing, because it does not exist in the hot-path context.

**This is the template for the noise channel.** A noise-source channel added to
`MNAContext` (structure-discovery only) and deliberately *absent* from
`DirectStampContext` inherits the same zero-transient-cost property by
construction.

## Threading noise sources through stamping

Add a deferred noise-source list to `MNAContext` alongside `b_ac_*` — enough to
reconstruct, per source: the branch nodes it injects into, and how to evaluate
its PSD at the bias point (source kind + params, or a closure).

Make the noise builtins **context-aware** rather than mode-branched:

- On `MNAContext`: `white_noise(ctx, pwr, name)` / `flicker_noise(ctx, pwr,
  exp, name)` (and builtin thermal/shot on R/diode stamps) append a source
  descriptor to the noise channel **and return `0.0`**.
- On `DirectStampContext`: no-op, return `0.0`.

Because the *value* returned is `0.0` in every context, the DC/transient
contribution (`I(NOII) <+ white_noise(...)`) is byte-identical to today — the
noise call only has a side effect during the one structure-discovery build on an
`MNAContext`. No `if mode === :noise` branch is threaded through the generated
builder, so the hot path has nothing to skip.

## The dual approach, reconciled

The two formulations are **not competitors at the linear-noise level — they are
the same transfer functions evaluated two ways.**

- **AD-perturbation (the framing we want).** Treat each noise source as a
  differentiable perturbation input `ϵ_k` injected at the source branch; the
  source→output transfer function is `∂y/∂ϵ_k`. For the linear AC system this
  is exactly the `freqresp` of an extra `B` column carrying a unit injection —
  i.e. reuse the existing DSS machinery, one input column per source.
- **Adjoint-PSD (the efficient evaluation for one output).** For a chosen
  output you don't want `∂y/∂ϵ_k` for each `k` via a separate forward solve;
  you want them all at once. One adjoint solve `(jωC+G)ᵀ x_adj = e_out` per
  frequency yields `H_k = x_adjᵀ e_k` for *every* source at O(1) each, reusing a
  single factorization. This is the classic SPICE `.noise` inner loop, and it is
  simply the cheap way to evaluate the same derivatives.

So the initial port evaluates the AD-perturbation transfer functions **via the
adjoint** on top of the existing AC linearization. The AD framing is what keeps
the door open to the parts that classic simulators can't do cheaply:

- differentiating the *output noise* w.r.t. design parameters (the SciML
  payoff — noise as one more differentiable objective for optimization);
- large-signal / cyclostationary noise, where the linearization is
  periodically time-varying (PSS/PAC-style) rather than a single DC-point AC
  system.

Both are N5 (stretch); N0–N4 deliberately stay on the tractable
single-operating-point linear `.noise` that the AC path already supports.

## Output quantities (N3)

At each frequency: output PSD `S_out(ω) = Σ_k |H_k(jω)|² S_k(ω)` (add cross
terms only when correlated sources are introduced). Total noise integrates
`S_out` over the band; input-referred noise divides by `|H_input(ω)|²` using the
input source's transfer function. Surface via a `NoiseSol` mirroring `ACSol`,
with name-based access, and a `.noise` netlist card driven through the
high-level API.

## Performance guardrails (the "don't blow up transient" contract)

1. The noise channel lives on `MNAContext` only; `DirectStampContext` gets no
   noise fields and no noise methods. Transient restamping is untouched.
2. Noise builtins return `0.0` in the value path, so DC/transient numerics are
   unchanged; registration is a structure-discovery-time side effect.
3. The adjoint solve reuses the AC factorization across all sources; cost scales
   with (#frequencies × #outputs), independent of #sources beyond a dot product.
4. N0 lands with a transient allocation/throughput benchmark asserting no
   regression before any PSD/solver work builds on top.

## Roadmap

- **N0 — Groundwork: noise-source channel. _(landed)_** A deferred noise-source
  channel lives on `MNAContext` as COO-style parallel vectors
  (`noise_p/noise_n/noise_kind/noise_a/noise_b/noise_names`), mirroring
  `b_ac_I`/`b_ac_V`, and is absent from `DirectStampContext` — `stamp_noise!` /
  `register_thermal_noise!` are no-ops there, so transient restamping is
  untouched. A `NoiseKind` enum (`THERMAL`/`SHOT`/`WHITE`/`FLICKER`) plus a
  `noise_psd(src, temp_c, f)` helper carry the spectral shapes. The resistor
  stamp registers Johnson–Nyquist thermal noise (`4kT·G`) as the first real
  source; the G/C/b value path is byte-identical, so DC/transient numerics are
  unchanged (`test/mna/noise.jl`).

  The builtin `Diode`/`DiodeWithCap` stamps now register shot noise (`2q·|I|`)
  via `register_shot_noise!`, reading the junction current `I0` at the operating
  point the channel is built at (`test/mna/noise.jl`). Same zero-cost contract:
  no-op on `DirectStampContext`, structure-discovery side effect on `MNAContext`,
  G/C/b value path byte-identical.

  The builtin `SimpleMOSFET` stamp now registers channel thermal noise
  (`4kT·(2/3)·gm`, drain→source) via `register_channel_thermal_noise!`, reading
  the operating-point transconductance `gm` the channel is built at and skipping
  cutoff (`gm == 0`). This reuses the `THERMAL` kind with an *effective* channel
  noise conductance `γ·gm` (`γ = 2/3` in saturation — the ngspice level-1/2/3
  `(8/3)·k·T·gm` shape). Same zero-cost contract: no-op on `DirectStampContext`,
  structure-discovery side effect on `MNAContext`, G/C/b value path byte-identical
  (`test/mna/noise.jl`).

  The Verilog-A `white_noise`/`flicker_noise` builtins now register too (see
  N1), and the builtin `Diode`/`DiodeWithCap`/`SimpleMOSFET` stamps carry
  `KF`/`AF`/`FFE` cards (off by default) whose flicker noise registers through
  the same `register_flicker_noise!` entry point as the Verilog-A path.
- **N1 — PSD models at the DC bias. _(landed)_** Per-source spectral density at
  the operating point: thermal `4kT·g`, shot `2qI`, VA `white_noise(pwr)` →
  `pwr`, `flicker_noise(pwr,exp)` → `pwr/f^exp`. Bias comes from the DC solution
  the AC path already computes.

  **The Verilog-A path.** `white_noise`/`flicker_noise` used to fold to a
  literal `0.0` at codegen. They still *evaluate* to `0.0` — the DC/transient
  value path is byte-identical — but now carry a registration side effect. The
  branch a source injects into is the LHS of the enclosing contribution
  (`I(b_int, e_int) <+ white_noise(2*q*Ib, "ib")`), which the isolated call
  expression cannot see, so the contribution codegen binds it into two locals
  (`_mna_noise_p_` / `_mna_noise_n_`) immediately before evaluating the RHS and
  the lowering reads them. Both contribution paths bind: unconditional branch
  stamping in `generate_mna_stamp_method_nterm`, and the inline stamping the
  `ContributionStatement` handler emits for contributions inside an `if` (which
  is where e.g. the VADistiller resistor's `if (noisy)` noise block lives).
  Potential contributions re-seed the binding to ground — the channel models
  current sources, so a noise term there has nowhere to land.

  The registration is wrapped in `if noise_enabled(ctx)`, a predicate that is
  `true` on `MNAContext` and `false` on `DirectStampContext`. Because it is a
  constant on a concrete context type, the *entire* branch — the noise-power
  expression included — is eliminated during transient restamping, so lighting
  up VA noise costs the hot path nothing (previously the power expressions were
  never emitted at all, so this preserves the status quo rather than trading it
  away). Sources are named `<instance>_<label>` from the model's own label
  string (`:q1_rb`, `:q1_flicker`), so a `NoiseSol` decomposes a device's noise
  per physical mechanism. `$mfactor` scales the PSD linearly: `m` parallel
  devices are `m` independent sources, and independent sources add in power.

  This lights up every VADistiller model's noise at once — resistor (thermal +
  flicker), diode (rs thermal, shot, flicker), BJT (rb/rc/re thermal, ic/ib
  shot, flicker), MOS1/2/3/6/9, JFET, MESFET, BSIM3/BSIM4 — plus any user or PDK
  Verilog-A that writes noise contributions (`test/noise.jl`,
  `test/mna/noise.jl`).

  **The builtin stamps.** The reference `Diode`/`DiodeWithCap`/`SimpleMOSFET`
  builtins have no Verilog-A behind them, so they register flicker noise
  (`KF·|I|^AF / f^FFE`, off unless a `KF`/`AF`/`FFE` card is given) directly from
  their stamps — but through the *same* `register_flicker_noise!(ctx, p, n, pwr,
  exp)` entry point the VA lowering emits (`pwr = KF·|I|^AF`, `exp = FFE`), not a
  parallel construction. The diode reads the junction current, the MOSFET the
  drain current, at the operating point the channel is built at — same zero-cost
  contract (`test/mna/noise.jl`, `test/noise.jl`).
- **N2 — Transfer functions via the AC system. _(landed)_** Reuse `ac!`'s
  linearized `(jωC + G)`; per output+frequency, one adjoint solve
  `(jωC+G)ᵀ x_adj = e_out` gives the transfer from every source at O(1) each
  (`H_k = x_adj[p_k] − x_adj[n_k]`), reusing the factorization across sources.
  Implemented in `noise!` (`src/noise.jl`).
- **N3 — `noise!()` analysis + output. _(landed, partial)_** `noise!(circuit,
  output; freqs)` returns a `NoiseSol` (mirroring `ACSol`) with the output PSD
  `S_out(f) = Σ_k |H_k(jω)|² S_k(f)`, per-source contributions
  (`ns[:onoise]` / `ns[:devname]`), and band-integrated `total_noise` (the RC
  case integrates to `kT/C`). The analysis is source-agnostic — it consumes
  whatever sits on the noise channel, so device/VA sources light up here for
  free once registered.

  **Input-referral _(landed)_.** `noise!(circuit, output; freqs, input=:V1)`
  refers the noise back to an independent voltage-source input, matching SPICE
  `.noise v(out) V1`. A voltage source `V1` carries its branch equation in the
  current-variable row `:I_V1`, so a unit injection there is a 1 V small-signal
  drive; the input→output gain is the same adjoint dot product used for the
  noise sources, `H(jω) = x_adjᵀ b_in = x_adj[idx(:I_V1)]`, read for free per
  frequency (no extra solve). The result then carries `ns.gain` (the complex
  transfer), `ns[:inoise] = onoise ./ |H|²` (V²/Hz for a voltage input), and
  `total_noise(ns; referred=:input)`. Validated against a resistor divider
  (input-referred `4kT·2000`) and an RC low-pass whose input-referred noise
  flattens to the bare `4kTR` because the gain pole cancels the noise-shaping
  pole (`test/noise.jl`).

  **Still open:** the `.noise` netlist card driven through the high-level API.
- **N4 — Tests + validation.** Netlist tests (thermal noise of an RC = `4kT·R`
  shaped by the RC pole; op-amp input-referred noise) cross-checked against
  ngspice `.noise`, driven through the high-level API.
- **N5 (stretch) — differentiable / large-signal.** Differentiate output noise
  w.r.t. design params (the SciML payoff), and scope cyclostationary (PSS/PAC)
  noise on a periodically-time-varying linearization. Design only for now.

## Prior art in git history

CedarSim/DAECompiler used an `ϵ`-perturbation representation: device models
carried `ϵ`-prefixed fields (noise-perturbation inputs) and `SimSpec` carried an
`ϵω` perturbation frequency. A `noiseparams` helper walked the builder with a
`ParamObserver` mock to harvest the set of `ϵ` knobs across the hierarchy. That
enumeration code was removed in **b771716** as dead — it cataloged knobs but
computed no PSDs, matrices, or transfer functions, and was welded to the old
struct-field representation. Revive it from that commit only if the `ϵ`-field
harvesting pattern proves useful; note that MNAContext's structure-discovery
pass already flattens the hierarchy during stamping, so the natural place to
register noise sources is that same pass, not a separate `ParamObserver` walk.
