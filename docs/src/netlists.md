# Loading circuits

Cadnip is a compiler as much as a simulator. A netlist is parsed, resolved
(`sema`), and turned into a Julia *builder function* — a function that stamps
the circuit's contributions into an MNA context. Everything downstream (`dc!`,
`tran!`, `ac!`, `noise!`, sweeps, differentiation) works on that function, so
the loading step is where a netlist stops being text and becomes code.

`MNACircuit` pairs a builder with a set of parameters. All the entry points
below produce one.

## From a string

The shortest way in. `sp"..."` is SPICE, `spc"..."` is Spectre, `va"..."` is
Verilog-A:

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

They are macros, so the netlist is compiled when the surrounding code is — no
parse cost at run time, and no world-age tax, because nothing is eval'd.

!!! note "The first line of a SPICE deck is its title"
    `sp"..."` parses a complete deck, and SPICE decks open with a title line, so
    the first line is *not* read as a circuit element. That is what the
    `* divider` line above is for. For a snippet with no title of its own, pass
    the `i` (inline) flag — `sp"..."i`.

    Getting this wrong is silent: drop the title and the `V1` card is eaten as
    one, leaving a circuit with no source that solves to all zeros and has no
    `I_v1` to read — see the transcript in the README.

The same circuit in Spectre syntax. Spectre has no title line, so `spc"..."`
needs no flag:

```@example netlists
divider_scs = MNACircuit(spc"""
v1 (vcc 0) vsource type=dc dc=5
r1 (vcc out) resistor r=1k
r2 (out 0) resistor r=1k
""")

dc!(divider_scs)[:out]
```

## From a file

For anything larger than a snippet, keep the netlist in a file. The language is
inferred from the extension: `.scs` is Spectre, anything else is SPICE.

```julia
circuit = MNACircuit("amp.sp")      # SPICE
circuit = MNACircuit("amp.scs")     # Spectre
sol = dc!(circuit)
```

For performance-sensitive code, define the builder at module top level instead.
`Base.include` on a `SpiceFile` / `SpectreFile` evaluates the generated code
into your module, binding a function named after the file:

```julia
Base.include(@__MODULE__, SpiceFile("amp.sp"))   # defines `amp(params, spec, ...)`

c = MNACircuit(amp; R1=1e3)
sol = dc!(c)
```

A netlist string that only exists at run time (read from a database, generated
by a script) goes through the string form of `MNACircuit`, which parses and
evals on the spot:

```julia
circuit = MNACircuit(read("amp.sp", String); lang=:spice, source_dir=@__DIR__)
```

`source_dir` is what relative `.include` / `.hdl` paths inside such a string
resolve against; a netlist loaded from a file resolves them against its own
directory.

## Loading happens at top level

Two separate rules, both of which come down to loading a netlist where Julia
can see the result.

!!! warning "Runtime parsing defines a builder, so mind the world age"
    `MNACircuit(path)` and `MNACircuit(code; lang=...)` parse the netlist and
    `Base.eval` a builder function for it. Julia freezes the world age of a
    top-level statement while it runs, so the fresh builder can only be
    *called* from a **later** statement:

    ```julia
    circuit = MNACircuit("amp.sp")   # statement 1: defines the builder
    sol = dc!(circuit)               # statement 2: calls it — fine

    dc!(MNACircuit("amp.sp"))        # ✗ same statement: MethodError, "method too new"
    ```

    A function body freezes its world age at entry the same way, so building
    *and* solving inside one call fails for the same reason. Load the deck at
    top level, and the function is free to build and solve as it likes:

    ```julia
    Base.include(@__MODULE__, SpiceFile("amp.sp"))   # top level: defines `amp`

    function run_sim()
        c = MNACircuit(amp; R1=1e3)                  # no eval, no world-age tax
        dc!(c)
    end
    ```

!!! note "A netlist macro is not"
    `sp"..."`, `spc"..."` and `va"..."` compile their deck when the macro
    *expands*, which happens before the enclosing code runs — so the builder
    always predates any call to it, and the macro leaves nothing at the call
    site but the builder. No world-age tax, and they go wherever an expression
    goes, a function body included:

    ```@example netlists
    function run_sim()
        dc!(MNACircuit(sp"""
        * divider
        V1 vcc 0 DC 5
        R1 vcc out 1k
        R2 out 0 1k
        """))[:out]
    end
    run_sim()
    ```

    Each expansion gets a deck module of its own, so two decks in one function
    that both define `.subckt divider` do not collide.

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

### Multiplicity

`m` is the one name on an instance line that is not an ordinary parameter. It is
the instance's *multiplicity* — how many identical copies of the device sit in
parallel — so it scales the device instead of being read by name: a resistance
and an inductance are divided by `m`, a capacitance and a Verilog-A model's
`$mfactor` are multiplied by it.

An `X` line carries it too, and there it applies to everything inside the
subcircuit, composing with the multiplicity of every enclosing instance:

```@example netlists
dc!(MNACircuit(sp"""
* three copies of a subcircuit that is itself four resistors wide
.subckt inner a b m=4
R1 a b 12k
.ends
.subckt outer a b
X1 a b inner
.ends
V1 vcc 0 DC 1
XO vcc out outer m=3
R2 out 0 1k
"""))[:out]
```

`inner` stamps `12k / 4`, the `m=3` on `XO` divides that again, and the divider
against `R2` reads `1k / (1k + 1k)`. An `m=` on an instance line *replaces* the
`.subckt` line's default rather than multiplying with it — the default is
simply what the instance line would have said.

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

## What is in this deck?

A netlist is text until it is loaded, and code afterwards — which is convenient
for the simulator and awkward for you, because "what is `r1`?" is a question
about the text. `netlist(circuit)` answers it: it is the deck's own namespace,
as sema resolved it, with the card that defines each name.

```@example netlists
nl = netlist(divider)
```

Names are addressed with `.` or `[]`, case-insensitively as SPICE resolves them,
and each one prints the line that defines it:

```@example netlists
nl.R1
```

There are five kinds of name in a deck — instances, parameters, models,
subcircuits and nets — and each answers with what it has: a net has no defining
card, so it reports its connections instead.

```@example netlists
nl.out
```

SPICE happily lets one name be several of these at once. That is not a corner
case — a `.param x1` next to an `X1` instance line is the collision
[Parameters and sweeps](@ref) has its own rule for — and the index is how you
see that it is there:

```@example netlists
collision = netlist(MNACircuit(sp"""
* a parameter and an instance sharing a name
.param x1=2.0
V1 vcc 0 DC 'x1'
X1 vcc 0 rdiv
.subckt rdiv a b
R3 a b 1k
.ends
"""))

collision.x1
```

A subcircuit is a namespace of its own, and the index nests the same way, so
`R3` is reachable through the subcircuit that declares it rather than from the
top level:

```@example netlists
collision.rdiv[:r3]
```

`netlist` needs a deck to read: it throws for a hand-written builder function,
which has no cards to point at.

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
