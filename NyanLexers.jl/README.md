# NyanLexers.jl

Shared lexer primitives used by [NyanSpectreNetlistParser.jl](../NyanSpectreNetlistParser.jl)
and [NyanVerilogAParser.jl](../NyanVerilogAParser.jl). Provides a small state
machine (`Lexer`), token types, and utilities (`emit`, `accept`, `accept_batch`,
`peekchar`, `readchar`) that concrete SPICE/Spectre/Verilog-A tokenizers build on.

This package is not intended for standalone use — it's the common lexing layer
for the Nyan parser family. If you want to parse netlists, use one of the
parser packages directly, or the [Cadnip](https://github.com/NyanCAD/Cadnip.jl)
umbrella which ties them together with an analog circuit simulator.

## Installation

```julia
using Pkg
Pkg.add("NyanLexers")
```

## License

MIT.
