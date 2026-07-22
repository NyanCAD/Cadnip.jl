# Noise analysis design

Noise analysis is a CedarSim feature we still want to port (`doc/scratchpad.md`,
CedarSim pillar). This note records the intended approach and the research into
how it threads through the current MNA system **without slowing DC/transient**.

## Status

Single-operating-point linear noise analysis is implemented (`src/noise.jl`):
`noise!(circuit, output; freqs)` computes the output-noise PSD and per-source
contributions over the thermal sources the N0 channel registers. The remaining
work is lighting up the other sources (device/VA shot & flicker) and the
input-referred / `.noise`-card surface. The scaffolding this builds on:

- `src/ac.jl` — the AC path builds a linearized descriptor state-space system
  (`E·dx = A·x + B·u, y = C·x`) at the DC operating point, which is exactly the
  linearization the noise analysis consumes; `noise!` reuses that
  rebuild-at-op-point pattern.
- `src/va_env.jl` / `src/vasim.jl` — the Verilog-A `white_noise` /
  `flicker_noise` builtins are parsed (NyanVerilogAParser keywords) and lowered
  to `0.0` (`vasim.jl:1297`, and named noise branches zeroed at
  `vasim.jl:3413`). These are the real per-device noise-source hooks; a noise
  analysis lights them up instead of zeroing them.

Builtin device noise (thermal `4kT·g`, shot `2qI`, flicker `KF·I^AF/f`) has no
home yet and would attach to the MNA device stamps.

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

  Still open within the "make every source ctx-aware" scope, deferred toward N1:
  builtin diode/BJT/MOSFET shot+flicker noise (need the branch current at the
  device stamp) and the VA `white_noise`/`flicker_noise` builtins — those fold
  to a literal `0.0` at codegen today (`vasim.jl`), and registering them needs
  the LHS branch context at the contribution site rather than at the isolated
  call expression.
- **N1 — PSD models at the DC bias.** Evaluate per-source spectral density at the
  operating point: thermal `4kT·g`, shot `2qI`, flicker `KF·I^AF/f`, VA
  `white_noise(pwr)` → `pwr`, `flicker_noise(pwr,exp)` → `pwr/f^exp`. Bias comes
  from the DC solution the AC path already computes.
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
  free once registered. **Still open:** input-referred noise (needs the input
  source→output transfer) and the `.noise` netlist card driven through the
  high-level API.
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
