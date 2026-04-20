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

### Analyses

```julia
sol = dc!(circuit)                             # DC operating point
sol = tran!(circuit, (0.0, 1e-3))              # Transient
sol = ac!(circuit, freqs)                      # AC small-signal
result = dc!(CircuitSweep(circuit, sweep))     # Parameter sweep
```

`dc!(cs::CircuitSweep)` returns a `SweepResult` that iterates `(params, sol)`
pairs. Solutions support name-based access via `sol[:node]` / `sol[:I_vsrc]`.

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
