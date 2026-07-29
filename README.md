# Cadnip.jl

**C**ircuit **A**nalysis & **D**ifferentiable **N**umerical **I**ntegration **P**rogram

Cadnip is an analog circuit simulator written in Julia, focused on simplicity, maintainability, and robustness. 

## Features

- Import of circuits in the form of netlists written in multiple dialects of Spice or Spectre
- Definition of new devices using Verilog-A models
- DC and transient analyses
- Full differentiability with respect to parameter values via ForwardDiff (for sensitivities, optimization, ML, etc.)
- Parameter sweeps with `CircuitSweep`
- Works with standard Julia releases (1.11+)

Cadnip is a fork of now inactive CedarSim that replaces the DAECompiler backend with a Modified Nodal Analysis (MNA) implementation. Internally, Cadnip represents ciruits using Julia function. The result is high performance simulation that interacts well with Julia's optimization and execution capabilities.

## Installation

```
Is this complex installation necessary? If so, why? If not necessary for normal use, can we provide both kinds of install with a justification?
```
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

A simple voltage divider is the circuit-level "hello world!" equivalent. Here's how you can analyze that circuit with using the `sp` string macro to inline a Spice netlist

```julia
using Cadnip
using Cadnip.MNA: MNACircuit

# --- Inline (tests, small samples): string macros ---
circuit = MNACircuit(sp"""
* Voltage divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
""")
sol = dc!(circuit)
println("Vout = ", sol[:out], " V")            # 2.5
```
Note that `sp"..."` treats the first line as comment. If we don't have that `* Voltage divider` line, we get a confusing result:
```julia
julia> circuit = MNACircuit(sp"""
       V1 vcc 0 DC 5
       R1 vcc out 1k
       R2 out 0 1k
       """)
MNACircuit{var"###circuit#257", @NamedTuple{}, Cadnip.MNA.MNASpec{Float64}}(var"##circuit#257", NamedTuple(), Cadnip.MNA.MNASpec{Float64}(27.0, :tran, 0.0, 1.0e-12, 0.0, 1.0, 27.0, 1.0e-12, 0.001, 1.0e-6, 1.0e-12))

julia> sol = dc!(circuit)
DC Solution:
  Node Voltages:
    V(out) = 0 V
    V(vcc) = 0 V
```

You can use the inline form `sp"..."i` (note the `i` at the end) to avoid that:
```julia
julia> circuit = MNACircuit(sp"""
       V1 vcc 0 DC 5
       R1 vcc out 1k
       R2 out 0 1k
       """i)
MNACircuit{var"###circuit#258", @NamedTuple{}, Cadnip.MNA.MNASpec{Float64}}(var"##circuit#258", NamedTuple(), Cadnip.MNA.MNASpec{Float64}(27.0, :tran, 0.0, 1.0e-12, 0.0, 1.0, 27.0, 1.0e-12, 0.001, 1.0e-6, 1.0e-12))

julia> sol = dc!(circuit)
DC Solution:
  Node Voltages:
    V(out) = 2.5 V
    V(vcc) = 5 V
  Branch Currents:
    I_v1 = -0.0025 A
```

You can also use Spectre syntax instead with the `spc` string macro:

```julia
circuit = MNACircuit(spc"""
v1 (vcc 0) vsource type=dc dc=5
r1 (vcc out) resistor r=1k
r2 (out 0) resistor r=1k
""")
sol = dc!(circuit)
println("Vout = ", sol[:out], " V")            # 2.5
```

For larger netlists, loading from a file is easier. In Cadnip, the `.scs` extension on a file name indicates that the file contains a Spectre-formatted netlist while anything else indicates Spice syntax.

```julia
# --- File-first (production): load a netlist from disk ---
circuit = MNACircuit("amp.sp")                 # extension → .scs Spectre, else SPICE
sol = dc!(circuit)
println("Output voltage: ", sol[:out])
```

```
Quick question: would it useful to provide a convenience overload of dc! and trans!
so that the call to MNACircuit isn't necessary? Or is the world age thing the problem
that causes this to be necessary?
```

### Loading options

Circuits can be loaded in a number of ways

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

**Warning: limitation on runtime parsing**

Note that `MNACircuit("path")` calls `Base.eval` internally. That means that if you
write code that tries to analyze a circuit that you load from a file, you can get
a surprising error message as shown here:
```julia
julia> function foo()
          dc!(MNACircuit("divider.sp"))
       end
foo (generic function with 1 method)

julia> foo()
ERROR: MethodError: no method matching divider(::@NamedTuple{}, ::Cadnip.MNA.MNASpec{Float64}, ::Float64; x::Cadnip.MNA.ZeroVector)
The applicable method may be too new: running in world age 27017, while current world is 27025.
...
```
This happens because `eval` is called with the age of the world frozen at the time of the call to `foo`.
But `eval` advances the state of the world so when other methods get added, those methods are too recent
to call in the context of the original world because code compiled in two different ages may have conflicting
assumptions baked in.

A similarly surprising result can be had if you are parsing an inline Spice netlist:
```julia
julia> function foo()
              dc!(MNACircuit(sp"""
              V1 vcc 0 DC 5
       R1 vcc out 1k
       R2 out 0 1k
       """))
       end
ERROR: syntax: "using" expression not at top level
Stacktrace:
 [1] top-level scope
   @ REPL[33]:1
```

The solution in both cases is the same. Define the circuit at the REPL
or in the `Base` and pass it in.

```
julia> foo(MNACircuit("divider.sp"))
DC Solution:
  Node Voltages:
    V(out) = 2.5 V
    V(vcc) = 5 V
  Branch Currents:
    I_v1 = -0.0025 A


julia> foo(MNACircuit(sp"""
       * divider
       V1 vcc 0 DC 5
       R1 vcc out 1k
       R2 out 0 1k
              """))
DC Solution:
  Node Voltages:
    V(out) = 2.5 V
    V(vcc) = 5 V
  Branch Currents:
    I_v1 = -0.0025 A
```

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

A DC operating point is also enumerable, so you can introspect or export it
without knowing the node names up front:

```julia
sol = dc!(circuit)
keys(sol)                 # node voltages then branch currents, e.g. [:in, :out, :I_V1]
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
