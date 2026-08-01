# Cadnip.jl

**C**ircuit **A**nalysis & **D**ifferentiable **N**umerical **I**ntegration **P**rogram

Cadnip is an MNA-based analog circuit simulator written in Julia, focused on simplicity, maintainability, and robustness. It is a fork of CedarSim that replaces the DAECompiler backend with a straightforward Modified Nodal Analysis (MNA) implementation.

## Features

- Import of multi-dialect SPICE/Spectre netlists
- Import of Verilog-A models
- DC and transient analyses
- Full differentiability via ForwardDiff (for sensitivities, optimization, ML, etc.)
- Parameter sweeps with `CircuitSweep`
- Works with standard Julia releases (1.11+)

## Installation

Install from GitHub by first adding the subpackages, then the main package:

```julia
using Pkg
Pkg.add(url="https://github.com/NyanCAD/Cadnip.jl", subdir="NyanLexers.jl")
Pkg.add(url="https://github.com/NyanCAD/Cadnip.jl", subdir="NyanSpectreNetlistParser.jl")
Pkg.add(url="https://github.com/NyanCAD/Cadnip.jl", subdir="NyanVerilogAParser.jl")
Pkg.add(url="https://github.com/NyanCAD/Cadnip.jl")
```

Or clone and develop locally:

```bash
git clone https://github.com/NyanCAD/Cadnip.jl
cd Cadnip.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

```julia
using Cadnip
using Cadnip.MNA: MNACircuit

# --- File-first (production): load a netlist from disk ---
circuit = MNACircuit("amp.sp")                 # extension → .scs Spectre, else SPICE
sol = dc!(circuit)
println("Output voltage: ", sol[:out])

# --- Inline (tests, small samples): string macros ---
circuit = MNACircuit(sp"""
* Voltage divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
""")
sol = dc!(circuit)
println("Vout = ", sol[:out], " V")            # 2.5

# --- Spectre syntax via spc"..." ---
circuit = MNACircuit(spc"""
v1 (vcc 0) vsource type=dc dc=5
r1 (vcc out) resistor r=1k
r2 (out 0) resistor r=1k
""")
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

**Top-level only for runtime parsing.** `MNACircuit("path")` and
`MNACircuit(code; lang=...)` call `Base.eval` internally and must be used at
the REPL or module top level. Inside a function body, Julia freezes the
caller's world age at entry and the freshly-defined builder can't be
dispatched. For that case, bring the circuit into scope at top level first:

```julia
Base.include(@__MODULE__, SpiceFile("amp.sp"))   # top level: defines `amp`

function run_sim()
    c = MNACircuit(amp; R1=1e3)                  # no eval, no world-age tax
    dc!(c)
end
```

The string macros (`sp"..."`, `spc"..."`, `va"..."`) expand at the call site
and work transparently in both top-level and function-body contexts.

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

Two limits worth knowing: device instance parameters are not reachable this way
(`r1=(r=2e3,)` does nothing — give the netlist a `.param` and use that), and
override names are not validated against the netlist, so a typo silently leaves
the default in place.

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

And it is enumerable, so you can introspect or export the whole thing without
knowing the names up front:

```julia
sol = dc!(circuit)
keys(sol)                 # nodes, branch currents, terminal currents: [:in, :out, :I_V1, :i_r1_p, …]
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

This package is available under the MIT license (see LICENSE.MIT). You may also use it under CERN-OHL-S v2 if that better suits your project.

Contributions are welcome! Please open an issue or pull request on GitHub.

## Related Projects

- [SpiceArmyKnife.jl](SpiceArmyKnife.jl/) - Tool for parsing and converting between netlist languages
- [NyanVerilogAParser.jl](NyanVerilogAParser.jl/) - Verilog-A parser
- [NyanSpectreNetlistParser.jl](NyanSpectreNetlistParser.jl/) - Spectre netlist parser
