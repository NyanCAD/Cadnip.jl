# NyanVerilogAParser.jl

A Verilog-A parser producing a concrete syntax tree that preserves every
token, comment, and whitespace span. Used as the Verilog-A front-end of
[Cadnip.jl](https://github.com/NyanCAD/Cadnip.jl) for compiling compact
device models (BSIM4, PSP, BSIM-CMG, photonic models, …) into stampable
MNA contributions. Forked from CedarSim.

## Installation

```julia
using Pkg
Pkg.add("NyanVerilogAParser")
```

## Usage

```julia
using NyanVerilogAParser

ast = NyanVerilogAParser.parsefile("bsim4.va")

# Or parse a string:
ast = NyanVerilogAParser.parse("""
module VAResistor(p, n);
    parameter real R = 1000.0;
    inout p, n;
    electrical p, n;
    analog I(p,n) <+ V(p,n)/R;
endmodule
""")
```

Walk the tree via `AbstractTrees.children`, or pattern-match on the `EXPR` /
`Node` types exposed by the module.

## License

MIT.
