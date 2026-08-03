# Analyses

Every analysis takes an `MNACircuit` and returns a solution object that is read
by name. `sol[:vout]` means the same thing in DC, transient, AC and noise — a
node voltage, a branch current, or a device observable — so a readout written
once survives a change of analysis.

```@example analyses
using Cadnip
using Cadnip.MNA: MNACircuit, nameat, terminal_currents, op_vars

lowpass = MNACircuit(sp"""
* loaded RC low-pass, driven for DC, AC and transient at once
V1 in 0 DC 1 AC 1 SIN(0 1 100k)
R1 in out 1k
R2 out 0 3k
C1 out 0 1n
""")
nothing # hide
```

## DC operating point — `dc!`

```@example analyses
op = dc!(lowpass)
op[:out], op[:I_v1], op.converged
```

Node voltages are named after the net, branch currents through a voltage source
are `:I_<source>`, and `:gnd` (or node `0`) is `0.0`. `sol.converged` says
whether Newton actually reached tolerance — worth checking before trusting a
number from a circuit that might not have solved.

### The operating point is enumerable

```@example analyses
sort(collect(keys(op)))
```

`keys`, `values`, `pairs`, `haskey` and `get` all work, so an operating point
can be collected into a `Dict`, diffed against another, or exported without
knowing the node names up front:

```@example analyses
Dict(pairs(op))
```

### Terminal currents

The solution vector cannot tell you a device's terminal current: KCL has
already summed every device that meets at a node. Devices therefore report their
own terminal currents as they stamp the converged point. Positive is *into* the
device.

```@example analyses
terminal_currents(op)
```

```julia
op[:i_m1_d]        # drain current of M1 — not inferred from a supply branch
op[:i_r1_p]        # current into R1's first terminal
```

Built-in devices report at their stamp, and so does every Verilog-A model: the
stamp codegen accumulates one current per port over every branch that reaches
it, so PDK and user models are covered without per-model configuration.

### Operating-point variables

Alongside them the models report the small-signal quantities they computed —
`gm`, `gds`, `vth`, `vdsat` — which no amount of reading the solution vector
recovers. A Verilog-A model contributes every variable it declares with a
`desc`/`units` attribute, which is how CMC and SPICE3-derived models already
spell their `.op` output:

```julia
op[:m1_gm]                     # transconductance of M1, from the model itself
op[:m1_vds] > op[:m1_vdsat]    # in saturation, in the model's own numbers
op_vars(op)                    # every device variable, as name => value
```

Both channels are gathered on the rebuild `dc!` already does at the converged
point, and are absent from the fast restamping path, so transient simulation
pays nothing for them.

## Transient — `tran!`

`tran!(circuit, tspan)` returns a SciML solution. It is dense in time: index it
with `nameat(sol, :name, t)`, or call it as `sol(t)` for the whole state.

```@example analyses
sol = tran!(lowpass, (0.0, 20e-6))
[nameat(sol, :out, t) for t in (0.0, 5e-6, 20e-6)]
```

The default solver is Sundials' `IDA` on the DAE form, configured for circuits
(KLU sparse linear solver, extra error-test and Newton retries). Any SciML DAE,
ODE or DDE algorithm can be passed instead, and `tran!` dispatches the problem
type accordingly:

```julia
sol = tran!(circuit, (0.0, 1e-6); solver=Rodas5P())       # ODE (mass matrix) path
sol = tran!(circuit, (0.0, 1e-6); abstol=1e-12, reltol=1e-9)
```

`abstol` also accepts a per-class NamedTuple, so node voltages, branch currents
and charge states each get their natural scale instead of one scalar being
dominated by the tiniest-unit variable:

```julia
sol = tran!(circuit, (0.0, 1e-6); abstol=(vntol=1e-6, iabstol=1e-12, chgtol=1e-14))
```

Breakpoints of `PWL`, `PULSE` and `SIN` sources are derived automatically and
handed to the integrator as `tstops`, so it lands on source edges instead of
discovering them through rejected steps (`auto_tstops=false` disables this).

Initialization follows SPICE: transient sources are evaluated at `t = 0` and a
DC steady state is solved there before stepping.

## AC small-signal — `ac!`

`ac!` linearizes about the DC operating point and returns an `ACSol` carrying
the descriptor state-space system and the Hz grid you asked for. Frequencies are
in hertz, as in SPICE `.ac`; `acdec(n, fstart, fstop)` builds a log grid of `n`
points per decade.

```@example analyses
f  = acdec(50, 1e3, 1e7)
ac = ac!(lowpass, f)

resp = ac[:out]                  # complex response over the grid
mag  = magnitude_db(ac, :out)    # dB
phs  = phase_deg(ac, :out)       # degrees

i3 = findfirst(<(mag[1] - 3), mag)
f[i3]                            # first grid point past 1/(2π·(R1∥R2)·C1) = 212 kHz
```

Without a grid, evaluate at arbitrary angular frequencies (rad/s — the
ControlSystems convention) instead:

```@example analyses
freqresp(ac!(lowpass), :out, 2π .* [1e3, 1e5])
```

For ControlSystems / DescriptorSystems interop, `subsystem(ac, :name)` returns
the SISO descriptor system, ready for `ss`, `bode`, poles and zeros:

```julia
using RobustAndOptimalControl                    # ss(::DescriptorStateSpace)
mag, phase, w = bode(ss(subsystem(ac, :out)), 2π .* f)
```

## Noise — `noise!`

`noise!(circuit, output; freqs)` computes the output-referred noise PSD at
`output` over a Hz grid, decomposed per source. It reuses the AC linearization:
one adjoint solve `(jωC + G)ᵀ x = e_out` per frequency gives the transfer from
*every* noise source at once, so the cost is one solve per frequency regardless
of how many sources the circuit has.

```@example analyses
ns = noise!(lowpass, :out; freqs=acdec(10, 1e3, 1e8))

ns[:onoise][1]                   # output PSD in V²/Hz at the first frequency
```

```@example analyses
total_noise(ns)                  # band-integrated, V rms
```

That total is the textbook `sqrt(kT/C)` of the load capacitor — the resistors set
both the noise density and the bandwidth, and the two cancel.

Sources register themselves during that rebuild, so each is named after the
device it came from and can be read on its own:

```@example analyses
collect(keys(ns.contributions))
```

The registered sources today are resistor Johnson–Nyquist noise, diode shot
noise, MOSFET channel thermal noise and flicker noise for the built-in devices,
plus every `white_noise`/`flicker_noise` a Verilog-A model declares — which
lights up the SPICE3 noise models of the VADistiller library (resistor, diode,
BJT, MOS1/2/3/6/9, JFET, MESFET, BSIM3/4) and any PDK or user model.

Naming a voltage source as the `input` refers the noise back to it. The
unit-voltage transfer comes out of the same adjoint solve, so this costs
nothing extra:

```@example analyses
ns = noise!(lowpass, :out; freqs=acdec(10, 1e3, 1e8), input=:V1)
ns[:inoise][1], total_noise(ns; referred=:input)
```

## Sweeping an analysis

`dc!` and `tran!` accept a `CircuitSweep` and return a `SweepResult` of
`(params, sol)` pairs — see [Parameters and sweeps](@ref).

```@example analyses
divider = sp"""
* divider with a swept top resistor
.param r=1k
V1 in 0 DC 1
R1 in out r
R2 out 0 1k
"""

for (p, sol) in dc!(CircuitSweep(divider, Sweep(r = [1e3, 2e3, 4e3])))
    println(p.r, " → ", sol[:out])
end
```

## Interactive exploration

With a Makie backend loaded, `Cadnip.explore(circuit, tspan)` opens a transient
plot with one logarithmic slider per scalar circuit parameter and re-simulates
live as you drag:

```julia
using GLMakie
Cadnip.explore(MNACircuit(lowpass_builder; R1=1e3), (0.0, 20e-6))
```
