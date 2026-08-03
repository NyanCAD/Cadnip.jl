# Cadnip.jl

**C**ircuit **A**nalysis & **D**ifferentiable **N**umerical **I**ntegration **P**rogram

Cadnip is an analog circuit simulator written in Julia, focused on simplicity,
maintainability, and robustness.

## Features

- Import of circuits as netlists, in several dialects of SPICE and Spectre
- Definition of new devices in Verilog-A
- DC, transient, AC small-signal, and noise analyses
- Full differentiability with respect to parameter values via ForwardDiff (for
  sensitivities, optimization, ML, etc.)
- Parameter sweeps with `CircuitSweep`
- Works with standard Julia releases (1.11+)

Cadnip is a fork of the now-inactive CedarSim that replaces the DAECompiler
backend with a straightforward Modified Nodal Analysis (MNA) implementation.
Internally a circuit is compiled to an ordinary Julia function that stamps the
MNA matrices, which is what makes it both fast and differentiable by the rest of
the Julia ecosystem.

## Installation

Cadnip is in the General registry:

```julia
using Pkg
Pkg.add("Cadnip")
```

Or clone and develop locally:

```bash
git clone https://github.com/NyanCAD/Cadnip.jl
cd Cadnip.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

A resistive voltage divider is the circuit-level "hello world". The `sp"..."`
string macro inlines a SPICE netlist:

```julia
using Cadnip
using Cadnip.MNA: MNACircuit

circuit = MNACircuit(sp"""
* Voltage divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
""")
sol = dc!(circuit)
println("Vout = ", sol[:out], " V")            # 2.5
```

**Mind the title line.** A SPICE deck's first line is its title, so `sp"..."`
treats it as a comment — that is what the `* Voltage divider` line is for. Drop
it and the `V1` card is swallowed as the title, which costs you the source
without any error:

```julia
julia> sol = dc!(MNACircuit(sp"""
       V1 vcc 0 DC 5
       R1 vcc out 1k
       R2 out 0 1k
       """))            # no title line: V1 is eaten
DC Solution:
  Node Voltages:
    V(out) = 0 V
    V(vcc) = 0 V
  Device Terminal Currents:
    i_r1_p = 0 A
    ...
```

There is no source left in the circuit, so everything sits at zero — note that
the `Branch Currents` section is missing entirely, because `V1` never made it in.

Add the `i` (inline) flag when a snippet has no title line of its own:

```julia
julia> sol = dc!(MNACircuit(sp"""
       V1 vcc 0 DC 5
       R1 vcc out 1k
       R2 out 0 1k
       """i))           # `i` = no title expected
DC Solution:
  Node Voltages:
    V(out) = 2.5 V
    V(vcc) = 5 V
  Branch Currents:
    I_v1 = -0.0025 A
  Device Terminal Currents:
    i_r1_p = 0.0025 A
    ...
```

Spectre syntax works the same way, via `spc"..."` (which has no title line, so
no flag):

```julia
circuit = MNACircuit(spc"""
v1 (vcc 0) vsource type=dc dc=5
r1 (vcc out) resistor r=1k
r2 (out 0) resistor r=1k
""")
println("Vout = ", dc!(circuit)[:out], " V")   # 2.5
```

For anything larger than a snippet, keep the netlist in a file. The extension
picks the language: `.scs` is Spectre, anything else is SPICE.

```julia
circuit = MNACircuit("amp.sp")                 # extension → .scs Spectre, else SPICE
sol = dc!(circuit)
println("Output voltage: ", sol[:out])
```

### Loading options

| Input                        | Loader                                            |
| ---------------------------- | ------------------------------------------------- |
| SPICE file                   | `MNACircuit("amp.sp")`                            |
| Spectre file                 | `MNACircuit("amp.scs")`                           |
| Top-level include in module  | `Base.include(@__MODULE__, SpiceFile("amp.sp"))`  |
| SPICE string                 | `sp"""..."""` or `MNACircuit(code; lang=:spice)`  |
| Spectre string               | `spc"""..."""` or `MNACircuit(code; lang=:spectre)` |
| Verilog-A string             | `va"""..."""`                                     |
| Already-compiled builder     | `MNACircuit(my_builder_fn; R=1e3)`                |
| PDK package                  | `.lib "jlpkg://MyPDK/..." typical` in the netlist |

**Runtime parsing defines a builder, so mind the world age.** `MNACircuit("path")`
and `MNACircuit(code; lang=...)` parse the netlist and `Base.eval` a builder
function for it. Julia freezes the world age of a top-level statement while it
runs, so the fresh builder can only be *called* from a **later** statement:

```julia
circuit = MNACircuit("amp.sp")   # statement 1: defines the builder
sol = dc!(circuit)               # statement 2: calls it — fine

dc!(MNACircuit("amp.sp"))        # ✗ same statement: MethodError, "method too new"
```

A function body freezes its world age at entry the same way, so building *and*
solving inside one call fails for the same reason. Bring the circuit into scope
at top level first and the function is free to build and solve as it likes:

```julia
Base.include(@__MODULE__, SpiceFile("amp.sp"))   # top level: defines `amp`

function run_sim()
    c = MNACircuit(amp; R1=1e3)                  # no eval, no world-age tax
    dc!(c)
end
```

The string macros expand at the call site rather than eval'ing, so they have no
world-age tax. But a netlist macro (`sp"..."`, `spc"..."`) expands to a block
carrying `using` statements, which Julia only allows at top level, so those two
cannot go inside a function body either:

```julia
function bad()
    dc!(MNACircuit(sp"""
    * divider
    V1 vcc 0 DC 5
    R1 vcc out 1k
    R2 out 0 1k
    """))
end
# ERROR: syntax: "using" expression not at top level
```

So a netlist — inline or from a file — is loaded at top level, and functions
take it from there.

### Parameters and sweeps

A netlist `.param` is the knob a design is parameterized on: bias point, device
size, source amplitude. Override one by name when the circuit is built, re-bind
it with `alter`, or make it a sweep axis — the same name in every case:

```julia
Base.include(@__MODULE__, SpiceFile("amp.sp"))  # .param vbias=1.1472, .param rd=10k

c = MNACircuit(amp; vbias=1.2)                  # override at construction
c = alter(c; rd=20e3)                           # re-bind (introduces the knob if absent)

for (p, sol) in dc!(CircuitSweep(amp, Sweep(vbias = 1.05:0.05:1.30)))
    @show p.vbias, sol[:drain]                  # DC transfer curve
end
```

Subcircuit parameters are keyed by instance name, and an override outranks the
value the instance line spells out (`X1 in out divider r1val=2k`):

```julia
c = MNACircuit(top; x1=(r1val=1e3,))            # or var"x1.r1val" as a sweep axis
```

Parameters and instances share a namespace, and the shape of the override tells
them apart: **a leaf is a parameter, a group is an instance**. So with a
`.param x1` next to an `X1` instance, `x1=2.0` sets the parameter and
`x1=(r1val=1e3,)` addresses the instance; `params=(x1=2.0,)` names the parameter
explicitly when you need both at once.

A `.model` card reads `.param`s too, so a process corner is an ordinary sweep
axis rather than a second netlist:

```spice
.param vt0=0.7
.model nch nmos level=1 vto=vt0 kp=100u
```

```julia
dc!(CircuitSweep(amp, Sweep(vt0 = [0.6, 0.7, 0.8])))
```

Override names are checked against what the netlist actually declares, so a typo
is an error at construction rather than a sweep that quietly returns the default
at every point:

```julia
julia> MNACircuit(amp; vbais=1.2)
ERROR: ArgumentError: unknown parameter override `vbais` — the top level
declares no parameter `vbais`. It declares: rd, vbias.
```

The same check runs for `alter` and for sweep axes. Two limits worth knowing:
device instance parameters are not reachable this way (`r1=(r=2e3,)` throws —
give the netlist a `.param` and use that instead), and a builder that declares
nothing at all — a hand-written builder, or a netlist with no `.param` and no
subcircuit instance — has no knob to typo, so nothing is checked for it.

### Analyses

```julia
sol = dc!(circuit)                             # DC operating point
sol = tran!(circuit, (0.0, 1e-3))              # Transient
ac  = ac!(circuit)                             # AC small-signal (linearized DSS)
ns  = noise!(circuit, :out; freqs=acdec(10, 1, 1e6))   # Noise
result = dc!(CircuitSweep(circuit, sweep))     # Parameter sweep
```

`dc!(cs::CircuitSweep)` returns a `SweepResult` that iterates `(params, sol)`
pairs. Solutions support name-based access via `sol[:node]` / `sol[:I_vsrc]`,
and `sol.converged` says whether Newton actually reached tolerance.

A DC sweep *continues*: each point starts Newton from the previous point's
solution rather than from zeros, like a SPICE `.dc` sweep. Adjacent points are a
small perturbation of each other, so this costs far fewer Newton iterations on
nonlinear circuits (a 40-junction ladder over 60 points: 877 iterations cold vs
477 continued). A point that fails to converge is never used as a starting
guess. Pass `continuation=false` for independent cold solves, and
`dc!(circuit; u0=sol.x)` to warm-start a single operating point by hand:

```julia
result = dc!(CircuitSweep(circuit, sweep))                     # continued (default)
result = dc!(CircuitSweep(circuit, sweep); continuation=false)  # independent points
```

A DC operating point also reports **device terminal currents** — the drain
current of a MOSFET, the current through a resistor — which the solution vector
itself cannot give you, because KCL has already summed every device that meets
at a node. Devices report them as they stamp the converged point, so builtin
devices and every Verilog-A model (VADistiller, PDK, your own) are covered
alike. Positive is *into* the device:

```julia
op = dc!(circuit)
op[:i_m1_d]               # drain current of M1 — not inferred from a supply branch
op[:i_r1_p]               # current into R1's first terminal
terminal_currents(op)     # every device terminal, as name => current
```

Alongside them it reports **device operating-point variables** — the
small-signal numbers the *models* compute (`gm`, `gds`, `vth`, `vdsat`), which
no amount of reading the solution vector can recover. A Verilog-A model
contributes every variable it declares with a `desc`/`units` attribute, which
is how the VADistiller and PDK models already spell their `.op` output, so this
needs nothing configured per model:

```julia
op[:m1_gm]                     # transconductance of M1, from the model itself
op[:m1_vds] > op[:m1_vdsat]    # in saturation, in the model's own numbers
op_vars(op)                    # every device variable, as name => value
```

And it is enumerable, so you can introspect or export the whole thing without
knowing the names up front:

```julia
sol = dc!(circuit)
keys(sol)                 # nodes, branch currents, terminal currents, device variables: [:in, :out, :I_V1, :i_r1_p, …]
Dict(pairs(sol))          # the whole operating point as name => value
get(sol, :out, NaN)       # non-throwing lookup (sol[:out] throws if absent)
haskey(sol, :out)         # true
```

`ac!(circuit, freqs)` returns an `ACSol` — a linearized descriptor state-space
system about the DC operating point, carrying the Hz frequency grid you asked
for. Name-based access is the SPICE-native readout (same `sol[:name]` meaning as
DC/transient), and Hz helpers need no manual 2π conversion:

```julia
f   = acdec(20, 1, 1e6)                         # 1 Hz … 1 MHz, in Hz (SPICE .ac dec)
ac  = ac!(circuit, f)
resp = ac[:vout]                                # complex response over the grid
mag  = magnitude_db(ac, :vout)                  # magnitude in dB
phs  = phase_deg(ac, :vout)                     # phase in degrees
```

`freqresp` evaluates at arbitrary angular frequencies (ω in rad/s — the
ControlSystems convention), and needs no stored grid:

```julia
ac = ac!(circuit)
H  = freqresp(ac, :vout, 2π .* f)               # raw complex response
```

For the ControlSystems / DescriptorSystems interop, `subsystem(ac, :name)`
returns the SISO descriptor system for `ss`, `bode`, poles/zeros:

```julia
using RobustAndOptimalControl                   # ss(::DescriptorStateSpace)
mag, phase, w = bode(ss(subsystem(ac, :vout)), 2π .* f)
```

### Two-tier model resolution

Device names resolve via two tiers:

- **Tier 1 (builtins).** R, C, L, D, level-dispatched MOSFETs/BJTs. Just
  `using VADistillerModels` / `using BSIM4` and `.model nmosfet nmos level=1`
  resolves automatically.
- **Tier 2 (netlist scope).** PDKs and custom VA devices via netlist directives:
  `.hdl "file.va"`, `.include "lib.sp"`, `.lib "lib.sp" section`, and
  `jlpkg://Package/path`. Most-recent include wins.

PDK authors expose content via `Cadnip.precompile_pdk(@__MODULE__, "pdk.spice")`
and `Cadnip.precompile_va(@__MODULE__, "device.va")` at package build time.

## Testing

Run the test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or run specific test groups:

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["mna"])'
```

## License

This package is available under the MIT license (see [LICENSE](LICENSE)). You may also use it under CERN-OHL-S v2 if that better suits your project.

Contributions are welcome! Please open an issue or pull request on GitHub.

## Related Projects

- [SpiceArmyKnife.jl](SpiceArmyKnife.jl/) - Tool for parsing and converting between netlist languages
- [NyanVerilogAParser.jl](NyanVerilogAParser.jl/) - Verilog-A parser
- [NyanSpectreNetlistParser.jl](NyanSpectreNetlistParser.jl/) - Spectre netlist parser
