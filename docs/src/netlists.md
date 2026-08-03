# Loading circuits

Cadnip is a compiler as much as a simulator. A netlist is parsed, resolved
(`sema`), and turned into a Julia *builder function* — a function that stamps
the circuit's contributions into an MNA context. Everything downstream (`dc!`,
`tran!`, `ac!`, `noise!`, sweeps, differentiation) works on that function, so
the loading step is where a netlist stops being text and becomes code.

`MNACircuit` pairs a builder with a set of parameters. All the entry points
below produce one.

## From a file

The canonical form. The language is inferred from the extension: `.scs` is
Spectre, anything else is SPICE.

```julia
circuit = MNACircuit("amp.sp")      # SPICE
circuit = MNACircuit("amp.scs")     # Spectre
sol = dc!(circuit)
```

For performance-sensitive code, define the builder at module top level instead.
`Base.include` on a `SpiceFile` / `SpectreFile` evaluates the
generated code into your module, binding a function named after the file:

```julia
Base.include(@__MODULE__, SpiceFile("amp.sp"))   # defines `amp(params, spec, ...)`

c = MNACircuit(amp; R1=1e3)
sol = dc!(c)
```

!!! warning "Runtime parsing is top-level only"
    `MNACircuit(path)` and `MNACircuit(code; lang=...)` call `Base.eval` to
    install the freshly generated builder. Julia freezes a function's world age
    at entry, so calling either *inside a function body* and then simulating
    fails with a "method too new" error. At the REPL or module top level they
    are fine.

    Inside a function body, load the deck at top level (as above) and pass the
    builder function. The string macros expand at the call site and work in
    both contexts.

## From a string

`sp"..."` is SPICE, `spc"..."` is Spectre, `va"..."` is Verilog-A. They are
macros, so the netlist is compiled when the surrounding code is — there is no
world-age caveat and no parse cost at run time.

!!! note "The first line of a SPICE deck is its title"
    `sp"..."` parses a complete deck, and SPICE decks open with a title line, so
    the first line is *not* read as a circuit element. Give the deck a comment
    line (`* divider`) as below, or pass the `i` flag — `sp"..."i` — to treat
    the string as inline content with no title. Spectre has no title line, so
    `spc"..."` needs neither.

```@example netlists
using Cadnip
using Cadnip.MNA: MNACircuit

divider = MNACircuit(sp"""
* divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
""")

dc!(divider)[:out]
```

The same circuit in Spectre syntax:

```@example netlists
divider_scs = MNACircuit(spc"""
v1 (vcc 0) vsource type=dc dc=5
r1 (vcc out) resistor r=1k
r2 (out 0) resistor r=1k
""")

dc!(divider_scs)[:out]
```

A netlist string that only exists at run time (read from a database, generated
by a script) goes through the string form of `MNACircuit`, which parses and
evals on the spot — subject to the top-level rule above:

```julia
circuit = MNACircuit(read("amp.sp", String); lang=:spice, source_dir=@__DIR__)
```

`source_dir` is what relative `.include` / `.hdl` paths inside such a string
resolve against; a netlist loaded from a file resolves them against its own
directory.

| Input | Loader |
| ----- | ------ |
| SPICE file | `MNACircuit("amp.sp")` |
| Spectre file | `MNACircuit("amp.scs")` |
| Deck as a top-level definition | `Base.include(@__MODULE__, SpiceFile("amp.sp"))` |
| SPICE string | `sp"""..."""` or `MNACircuit(code; lang=:spice)` |
| Spectre string | `spc"""..."""` or `MNACircuit(code; lang=:spectre)` |
| Verilog-A string | `va"""..."""` |
| Already-compiled builder | `MNACircuit(my_builder; R=1e3)` |

## Subcircuits

`.subckt` definitions are instantiated with `X` lines, and their internal nodes
are reachable in the flattened name table with the instance name as a prefix:

```@example netlists
top = MNACircuit(sp"""
* divider instantiated as a subcircuit
.subckt divider a b
R1 a mid 1k
R2 mid b 1k
.ends
V1 vcc 0 DC 6
X1 vcc 0 divider
""")

op = dc!(top)
op[:x1_mid]
```

Instance parameters on the `X` line, and the `.param` defaults a `.subckt`
declares, are covered in [Parameters and sweeps](@ref).

## A deck is a namespace

A netlist file is a *deck*, and each loaded deck gets a Julia module of its own;
only the builder is bound in your module. Two decks that each define
`.subckt divider` therefore do not overwrite each other:

```julia
Base.include(@__MODULE__, SpiceFile("amp_a.sp"))   # its own .subckt divider
Base.include(@__MODULE__, SpiceFile("amp_b.sp"))   # a different .subckt divider

dc!(MNACircuit(amp_a))   # each answers for its own deck
dc!(MNACircuit(amp_b))
```

Two `sp"..."` strings in the same *local* scope are the exception: a module
cannot be defined in expression position, so that case is ordinary Julia
redefinition in a scope you wrote.

## Include directives and libraries

The usual SPICE directives pull in external content, and paths resolve relative
to the including file:

```spice
.include "models.sp"
.lib "corners.sp" typical
.hdl "mydevice.va"
```

A `jlpkg://Package/path` URL resolves inside an installed Julia package, which
is how a PDK shipped as a package is referenced:

```spice
.lib "jlpkg://MyPDK/models/corners.sp" typical
```

Anything a directive brings in lands in the netlist's scope, which is the second
of the two tiers device names resolve through — see
[Model cards and the two-tier lookup](@ref).
