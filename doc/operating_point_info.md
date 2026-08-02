# Operating-point info: device terminal currents and variables

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

## Operating-point variables: `gm`, `gds`, `vdsat`

A terminal current says how the device is loaded; it does not say how the device
*behaves* there. The numbers a designer reaches for next — the transconductance,
the output conductance, the saturation voltage, the threshold the model actually
used — are not currents at all, and unlike currents they are not even
in-principle recoverable from the stamp: `gm` is a partial derivative the model
took internally and then folded into the Jacobian along with everything else.

They are, however, already *named* in the model. Verilog-A marks the variables a
simulator should report at the operating point with a descriptive attribute:

```verilog
(* desc = "Transconductance" *)      real gm;
(* desc = "Drain-Source conductance" *) real gds;
(* desc = "Saturation drain voltage" *) real vdsat;
```

Every VADistiller model carries these already — they are how the SPICE models
they were distilled from expose their `.op` output — and so do PDK and CMC
models. So the front end does not need a per-model list, an annotation pass, or
a naming convention of its own: **a module-level `real`/`integer` declaration
with a `desc` or `units` attribute is an operating-point variable, and a bare
one is an internal scratch variable.** That is the whole rule.

The channel is the terminal-current channel with the branch machinery removed,
because there is nothing to accumulate — the model has computed the value by the
time the stamp is done:

```julia
opv_devices::Vector{Symbol}   # instance
opv_names::Vector{Symbol}     # variable
opv_values::Vector{Float64}
```

`register_op_var!(ctx, device, var, value)` is a pure push; `op_vars(ctx)`
composes `m1_gm` at readout and needs no solution vector. A repeated name keeps
its **last** value rather than summing — a variable is the device's own scalar,
not a quantity split across branches, which is the one place the two channels
differ.

Codegen mirrors the terminal-current registration exactly: one
`register_op_var!` per declared variable at the end of the stamp, the whole
block inside `if op_enabled(ctx)`, so `DirectStampContext` folds it away and
transient restamping is untouched. Modules that declare no such variable emit no
block at all. String-typed variables are skipped — the channel is `Float64`.

This lights up MOS1/2/3/6/9 (`gm`, `gds`, `gmb`, `vdsat`, `von`, `vgs`, `vds`,
the junction capacitances), BSIM3/BSIM4 (`gm`, `gds`, `gmbs`, `vth`, `vdsat`),
the BJT (`gm`, `gpi`, `gmu`, `go`, `cpi`, `cmu`), the JFETs, MESFET, diode
(`gd`, `cd`) and resistor (`i`, `p`) at once, plus any PDK or user Verilog-A —
the same no-per-model-code leverage as the noise channel.

Region — "is M1 in saturation?" — falls out of the variables rather than being
its own report: `op[:m1_vds] > op[:m1_vdsat]` is the saturation test, in the
model's own numbers rather than in a hand-derived overdrive.

```julia
op = dc!(circuit)
op[:m1_gm]                     # transconductance, from the model
op[:m1_vds] > op[:m1_vdsat]    # saturation
op_vars(op)                    # every device variable, in stamp order
```

## What is not covered yet

- **Builtin device stamps register no operating-point variables.** The builtins
  (`SimpleMOSFET`, `Diode`) are reference stamps; the netlist path resolves
  `level=1` to VADistiller MOS1, which reports. If a builtin ever needs to, it
  calls the same `register_op_var!`.
- **Controlled sources** (VCCS/CCCS/VCVS/CCVS) do not register. The
  voltage-output ones already expose their branch current as an MNA unknown
  (`I_e1`); the current-output ones would need a third entry kind (a
  transconductance across a *different* branch than the one it drives).
- **Capacitive terminal currents** are zero at DC and stay unregistered. A
  transient readout of them would be a different feature — the channel is a DC
  operating-point report, not a per-timestep probe.
