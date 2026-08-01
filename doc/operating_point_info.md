# Operating-point info: device terminal currents

A designer reading an operating point wants the device's numbers, not the
system's: *what is M1's drain current, is it in saturation, where is the bias
sitting*. What `dc!` returned was the MNA solution vector — node voltages plus
the branch currents that happen to be unknowns (voltage sources, inductors) —
so the drain current had to be inferred from somewhere else. `test/design_flow.jl`
did exactly that, and said so:

```julia
# Drain current, read off the supply branch (Rd carries all of it).
id = -op[:I_vdd]
```

That works for a single-device stage and stops working the moment two devices
share a supply. This note records the channel that fixes it (scratchpad "UX/design
follow-ups": *Report device terminal currents in the operating point, via an
op-info channel on `MNAContext`*).

## Why it has to be a channel

The terminal currents are not recoverable from the solved system. KCL sums every
device at a node, so `G·x − b` at a node is the *total*, and the split between the
devices that meet there is gone by assembly time. `x` cannot answer "how much of
that came from M1" for the same reason.

But the split is right there while the device is stamping: the diode has just
computed `I0`, the MOSFET its `Ids`, and every Verilog-A `I(p,n) <+ …` branch its
current — the value they linearize around. All that is needed is somewhere to
put it.

This is exactly the shape of the noise channel (`doc/noise_analysis_design.md`),
and it reuses that design wholesale:

- a set of parallel vectors on `MNAContext`, fully deferred, materialized only
  when someone asks;
- **nothing at all on `DirectStampContext`** — the zero-allocation restamping
  context used by the transient hot path has no op channel, so the registrations
  are no-op method dispatches and `op_enabled(ctx)` is a compile-time `false`
  that eliminates the codegen'd accumulation entirely;
- one registration entry point that builtin stamps and Verilog-A codegen share,
  rather than two parallel mechanisms.

## The channel

```julia
opi_devices::Vector{Symbol}   # instance
opi_terminals::Vector{Symbol} # terminal
opi_p::Vector{MNAIndex}       # branch, for ohmic entries
opi_n::Vector{MNAIndex}
opi_a::Vector{Float64}        # current, or conductance for ohmic entries
opi_ohmic::Vector{Bool}
```

Instance and terminal are kept apart rather than stored as the composed
`:i_m1_d`, so registration is a pure push: `Symbol(:i_, device, :_, terminal)`
allocates, and stamping runs it once per terminal per device on every discovery
and detection pass. The name is composed once instead, at readout.

Two registrars write it:

- `register_terminal_current!(ctx, device, terminal, I)` — the current into
  `terminal`, evaluated by the device. Everything that sees the operating point
  uses this.
- `register_ohmic_terminal_current!(ctx, device, terminal, p, n, G)` — the
  current into `terminal` is `G·(V_p − V_n)`, evaluated later. This exists for
  the linear devices whose stamp signature never receives `x`: `stamp!(R::Resistor,
  ctx, p, n)` genuinely cannot compute its own current, and threading `x` into
  every linear stamp to make it uniform would be a much larger change for a
  value that is one multiply away at readout.

Sign convention: **positive is into the device**. A resistor's two ends are equal
and opposite, a MOSFET's drain and source currents sum to zero through the
channel, and the sign of `i_m1_d` is the sign of the drain current a designer
would draw on a schematic. Repeated names sum, so a terminal fed by several
branches reports their total.

`terminal_currents(ctx, x)` materializes the channel against a solution vector,
evaluating the ohmic entries and folding duplicates.

## Where the readout comes from

`solve_dc` already rebuilds the context at the converged point — it needs
`assemble!` there for the node names — so the op channel it fills on that pass
*is* the operating point's. No extra stamping pass, no extra solve:

```julia
reset_for_restamping!(ctx)
builder(params, spec, 0.0; x=u, ctx=ctx)
sys_final = assemble!(ctx)
return DCSolution(sys_final, u, converged;
                  terminal_currents=terminal_currents(ctx, u))
```

`DCSolution` carries them as `name => current` pairs and folds them into the
readout surface it already had:

```julia
op = dc!(circuit)
op[:i_m1_d]                    # drain current of M1
keys(op)                       # nodes, branch currents, then terminal currents
Dict(pairs(op))                # the whole operating point
terminal_currents(op)          # just the device currents, in stamp order
show(op)                       # prints a "Device Terminal Currents" section
```

## Verilog-A: one accumulator per port

The builtin stamps register directly. The Verilog-A path cannot, because a
model's branches are not its terminals: a MOSFET with series resistances
contributes `I(d,di)`, not `I(d,s)`, and the drain current arrives at the
terminal through that branch. So the generated stamp method accumulates.

`generate_mna_stamp_method_nterm` emits one `Float64` accumulator per port,
adds each branch's DC current to the ports it spans (`+I` at `p`, `−I` at `n`;
internal nodes are skipped, since their currents stay inside the device), and
registers the accumulators once at the end of the stamp. Potential contributions
(`V(p,n) <+ …`, including the `V(d,di) <+ 0` short circuit models use when a
series resistance is zero) carry their current in an MNA unknown instead, so
those read it off `x` — skipping them would silently report a zero drain current
on exactly the models that set `rd=0`.

Every one of those touches sits inside `if op_enabled(ctx)`. On
`DirectStampContext` that is a constant `false`, so the accumulators are dead and
the whole thing — sums, reads, registration — is eliminated. Transient restamping
is byte-identical to before, which is the same property the noise channel has.

The leverage is the same as the noise work's: this lights up *every* VADistiller
model at once (resistor, diode, BJT, MOS1/2/3/6/9, JFET, MESFET, BSIM3/4), plus
any PDK or user Verilog-A, without a per-model line of code.

## What is not covered yet

- **Small-signal parameters and region** (`gm`, `gds`, saturation/triode). These
  are the other half of the scratchpad item and need Verilog-A operating-point
  variables (`(* desc=… *)` `real` outputs) in the VA front end — a separate
  piece of work, with a wider channel (`Float64` observables keyed by name)
  that this one's shape generalizes to.
- **Controlled sources** (VCCS/CCCS/VCVS/CCVS) do not register. The
  voltage-output ones already expose their branch current as an MNA unknown
  (`I_e1`); the current-output ones would need a third entry kind (a
  transconductance across a *different* branch than the one it drives).
- **Capacitive terminal currents** are zero at DC and stay unregistered. A
  transient readout of them would be a different feature — the channel is a DC
  operating-point report, not a per-timestep probe.
