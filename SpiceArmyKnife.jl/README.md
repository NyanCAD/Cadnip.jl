# SpiceArmyKnife.jl

Command-line tools for parsing, converting, and cataloging SPICE / Spectre /
Verilog-A netlists. Built on the [NyanSpectreNetlistParser](../NyanSpectreNetlistParser.jl)
and [NyanVerilogAParser](../NyanVerilogAParser.jl) parsers.

Ships two CLI apps:

- **`spak-convert`** — translate a netlist between SPICE/Spectre dialects, or
  compile a SPICE netlist to Verilog-A (targeting Gnucap or OpenVAF).
- **`spak-generate`** — crawl public SPICE model archives and assemble a
  unified JSON database (currently targets [NyanCAD](https://nyancad.github.io)).

## Conversion targets

| From   | To             | Notes                                        |
| ------ | -------------- | -------------------------------------------- |
| SPICE  | SPICE          | apply compatibility transforms between dialects |
| SPICE  | Verilog-A      | target Gnucap or OpenVAF                     |
| SPICE  | Spectre        | target VACASK                                |

Supported input/output simulators (via `--input-simulator` / `--output-simulator`):
`ngspice`, `pspice`, `hspice`, `spectre`, `vacask`, `openvaf`, `gnucap`.

## Installation

Install as a Julia app:

```julia
using Pkg
Pkg.add("SpiceArmyKnife")
```

Or install the CLI shims directly from the repo:

```
julia> ] app add "https://github.com/NyanCAD/Cadnip.jl:SpiceArmyKnife.jl"
```

## Usage

```bash
# Convert Cordell PSpice models to ngspice-compatible form (drops doc params)
spak-convert Cordell-Models.txt ngspice-models.sp \
    --input-simulator pspice --output-simulator ngspice

# Compile a SPICE netlist to Verilog-A for OpenVAF
spak-convert diode_divider.lib diode_divider.va \
    --input-simulator ngspice --output-simulator openvaf

# Convert SPICE to VACASK-ready Spectre
spak-convert combined_models/sky130.lib.spice vacask/combined_models/sky130.lib.spice \
    --input-simulator ngspice --output-simulator vacask
```

## License

MIT.
