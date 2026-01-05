# SpiceArmyKnife.jl

A tool for parsing and converting netlist languages.

SpiceArmyKnife.jl builds on the advanced CedarSim parsers for Spice, Spectre, and Verilog-A for various parsing and conversion needs.

## Parsing

SPAK.jl can parse large collections of spice models to generate a database file, currently targeting NyanCAD.

## Conversion

SPAK.jl can convert between various netlist languages and dialects.

Currently functionality:
- Spice -> spice, applying some compatibility transforms
- Spice -> Verilog-A, currently targeting Gnucap
- Spice -> Spectre, currently targeting VACASK

## Installation

- launch a julia repl with `julia`
- access the package shell with `]`
- install the app with `app add "https://github.com/NyanCAD/SpiceArmyKnife.jl:SpiceArmyKnife.jl"`

## Usage

```
spak-convert combined_models/sky130.lib.spice vacask/combined_models/sky130.lib.spice --input-simulator ngspice --output-simulator vacask
```
