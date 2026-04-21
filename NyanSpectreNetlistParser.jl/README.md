# NyanSpectreNetlistParser.jl

A full-fidelity parser for SPICE and Spectre netlists, producing a concrete
syntax tree (CST) that preserves every token, including whitespace and
comments. Handles multi-dialect SPICE (ngspice, PSpice, HSPICE) and Spectre,
with dialect switching mid-file.

Used as the front-end of [Cadnip.jl](https://github.com/NyanCAD/Cadnip.jl) and
[SpiceArmyKnife.jl](../SpiceArmyKnife.jl). Forked from CedarSim.

## Installation

```julia
using Pkg
Pkg.add("NyanSpectreNetlistParser")
```

## Usage

```julia
using NyanSpectreNetlistParser
using NyanSpectreNetlistParser.SPICENetlistParser

# Parse a SPICE file
ast = SPICENetlistParser.parsefile("amp.sp"; spice_dialect=:ngspice)

# Parse a Spectre file (may contain `simulator lang=spice` regions)
ast = NyanSpectreNetlistParser.parsefile("amp.scs")

# Parse a string
ast = SPICENetlistParser.parse("""
* divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
""")
```

The returned CST preserves source positions and trivia; walk it with
`AbstractTrees.children`, or use the `EXPR` / `Node` types for typed
pattern matching.

## SPICE / Spectre references

Spectre:

- [Spectre Reference](https://www.researchgate.net/profile/Bahram_Rashidi/post/How_can_I_calculate_switching_power_of_any_digital_circuit_using_CADENCE_SPECTRE_simulator/attachment/59d643f379197b807799f526/AS%3A446369983930370%401483434306014/download/spectre_reference.pdf)
- [UC Berkeley EE142 Spectre intro](https://inst.eecs.berkeley.edu/~ee142/fa04/pdf/spectre_start.pdf)

SPICE:

- [Stanford EE133 SPICE quick reference](https://web.stanford.edu/class/ee133/handouts/general/spice_ref.pdf)
- [HSPICE user guide (ch. 4)](https://cseweb.ucsd.edu/classes/wi10/cse241a/assign/hspice_sa.pdf)
- [SkyWater PDK sky130 model library](https://github.com/google/skywater-pdk-libs-sky130_fd_pr/blob/main/models/sky130.lib.spice)
- [Berkeley SPICE element reference](http://bwrcs.eecs.berkeley.edu/Classes/IcBook/SPICE/UserGuide/elements_fr.html)
- [PSpice reference guide](https://www.seas.upenn.edu/~jan/spice/PSpice_ReferenceguideOrCAD.pdf)

## License

MIT.
