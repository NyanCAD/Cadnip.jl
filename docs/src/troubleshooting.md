# When a circuit does not solve

Circuits fail in a small number of characteristic ways, and which one you have
is usually decidable in a few lines. This page walks the pipeline in order —
does it parse, does it build the circuit you meant, does DC converge, does the
transient run — because a failure late in that list is very often a mistake
early in it.

## Is it the circuit you wrote?

Before blaming the solver, check that the deck compiled into the netlist you
think it did. The solution's own names are the cheapest way:

```@example trouble
using Cadnip
using Cadnip.MNA: MNACircuit
using VADistillerModels     # supplies the `d` model card used further down

divider = MNACircuit(sp"""
* divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
""")

op = dc!(divider)
node_names(op), branch_names(op)
```

A net you expected and do not see, or one you did not expect and do — `outp`
next to `out` — is a typo in the deck, and SPICE has no declarations to catch
it: any word in a terminal position is a net, and a misspelled one is simply a
new net.

!!! warning "The first line of a SPICE deck is its title"
    `sp"..."` parses a whole deck, and the first line of a deck is its title —
    not a card. Drop the title and the first *element* is eaten as one:

    ```@example trouble
    untitled = MNACircuit(sp"""
    V1 vcc 0 DC 5
    R1 vcc out 1k
    R2 out 0 1k
    """)

    node_names(dc!(untitled)), branch_names(dc!(untitled))
    ```

    The `V1` card is gone: there is no `I_v1` to read, and the circuit — now
    with no source in it — solves quietly to all zeros. Use the `i` (inline)
    flag, `sp"..."i`, for a snippet with no title of its own.

An override that reaches nothing is caught for you and throws rather than
running as a silent no-op, so a misspelled *parameter* is not in this category —
see [Names that reach nothing throw](@ref).

## Is everything connected?

**Do you have a ground node?** Is everything connected to it, through some path,
at DC? Even the bulk terminals of the transistors in the netlist someone else
gave you?

A net with no DC path to ground has no equation fixing its voltage, and the
matrix is singular. Cadnip's DC solve does not simply fail on that: `gmin`
stepping (below) adds a conductance from every node to ground, and if the
solution survives shrinking that conductance back to zero you get an answer. If
it does not, this is the first thing to check.

The usual culprits are a floating terminal, two nets that differ by a typo,
and a series capacitor with nothing to define the DC voltage on its far side.

## DC does not converge

Newton is the DC solver, and `sol.converged` is the honest report of whether it
got there. A non-converged solve still *returns* — its last iterate, with a
warning — so a number read out of `sol` without checking that flag may be
nothing at all:

```@example trouble
op.converged
```

Cadnip already tries hard before giving up. `dc!` runs an ngspice-style chain,
each tier starting where the previous one gave out:

1. **Newton with a robust polyalgorithm** (`CedarRobustNLSolve` — Newton with
   line search, Levenberg–Marquardt and pseudo-transient behind it). When the
   circuit declares limiting variables, a SPICE-style predictor/corrector
   Newton with junction limiting runs first.
2. **`gmin` stepping.** A shunt conductance from every voltage node to ground,
   starting at 1 mS, makes the matrix diagonally dominant and the problem easy.
   It is then stepped down by decades to zero, each step warm-started from the
   last. A step that fails is retried with a smaller factor rather than
   abandoned.
3. **Source stepping.** Every independent source is scaled by a factor ramped
   from 0 to 1. At 0 the circuit is dead and the solution is trivial; raising it
   in increments lets nonlinear devices turn on gradually.

So by the time you see the warning, three strategies have failed, and the
problem is usually the circuit rather than the solve. Four things help.

### Give it a starting point

`dc!(circuit; u0=x)` starts Newton from `x` instead of zeros. The solution of a
nearby circuit is an excellent guess, which is exactly what a swept `dc!` does
automatically — see [DC sweeps continue](@ref).

```@example trouble
warm = dc!(divider; u0=op.x)
warm.converged
```

You rarely have a whole vector, though — what you usually have is a hunch about
one or two nodes. Name them, which is what SPICE writes as `.nodeset`, and every
state you do not name starts at zero exactly as a cold start leaves it:

```@example trouble
hinted = dc!(divider; u0=(out = 2.4,))
hinted[:out], hinted.converged
```

A guess costs nothing when it is wrong — the stepping tiers above restart from
zeros regardless — and a name the circuit does not have throws rather than
quietly doing nothing. Where it earns its keep is a circuit with *more than one*
operating point: a latch, a Schmitt trigger, anything cross-coupled. Newton
returns the root nearest where it started, so the guess is how you say which
state you meant.

### Walk in from a circuit that does solve

Sweep a supply from a low value up to its nominal one, continuing each point
from the last. This is source stepping done by hand, with a step schedule you
control:

```@example trouble
ramp = sp"""
* diode-loaded divider, supply as a parameter
.param vdd=0.2
V1 vcc 0 DC vdd
R1 vcc out 1k
D1 out 0 dmod
.model dmod d is=1e-14 n=1.0
"""

for (p, sol) in dc!(CircuitSweep(ramp, Sweep(vdd = [0.2, 0.5, 1.0, 2.0, 5.0])))
    println(p.vdd, " V → out = ", sol[:out], "  (converged: ", sol.converged, ")")
end
```

### Raise `gmin`

`.option gmin=1e-9` (or `MNASpec(spec; gmin=1e-9)`) leaves a little conductance
across every junction. It is a deliberate small error in the answer, traded for
a much better-conditioned problem — see [Netlist options](@ref).

### Cut the circuit down

Delete blocks until it converges, then add them back one at a time. The block
that reintroduces the failure is the one to look at, and it is usually much
smaller than the full netlist.

## Transient trouble

A transient starts from a DC solve — the default `initializealg` is
`CedarTranOp`, which evaluates every source at `t = 0`, solves the operating
point there through the same fallback chain, and only then starts stepping. **If
the DC operating point does not converge, fix that first**: everything above
applies, and `dc!(circuit)` is a far faster way to iterate on it than a
transient that fails after the initialization.

Three knobs, roughly in the order worth trying:

**Tolerances.** `tran!` takes `abstol` and `reltol` directly. Loosening them is
the standard response to a solve that crawls or dies with a `dtmin` warning;
tightening them is how to check whether a suspicious waveform is a numerical
artifact.

```julia
sol = tran!(circuit, (0.0, 1e-6); abstol=1e-8, reltol=1e-6)    # looser, faster
```

One scalar `abstol` is a poor fit for a state vector holding volts, amps and
coulombs at once — the smallest-unit variable ends up setting the pace for
everything. A NamedTuple gives each class its own scale, as SPICE's `vntol` /
`abstol` / `chgtol` do:

```julia
sol = tran!(circuit, (0.0, 1e-6); abstol=(vntol=1e-6, iabstol=1e-12, chgtol=1e-14))
```

**Initialization.** For a circuit with no stable DC operating point — an
oscillator is the standard case — the initialization is the thing that fails,
and no tolerance will fix it. `CedarUICOp` skips the DC solve entirely and
relaxes the algebraic constraints with a few fixed implicit-Euler steps, which
is what SPICE's `uic` means:

```julia
sol = tran!(circuit, (0.0, 1e-6); initializealg=CedarUICOp())
sol = tran!(circuit, (0.0, 1e-6); initializealg=CedarUICOp(warmup_steps=50, use_shampine=true))
```

An oscillator that *does* have a DC operating point has the opposite problem: it
sits in it forever, because a perfectly symmetric solution is a valid one.
Nudge it with an initial condition or an asymmetric source rather than expecting
the integrator to find its own way out.

**The solver.** The default is Sundials' `IDA` on the DAE form with a KLU sparse
linear solver, which is a good default for stiff circuit problems. Any SciML
algorithm can be passed instead, and `tran!` builds the matching problem type:

```julia
sol = tran!(circuit, (0.0, 1e-6); solver=Rodas5P())     # ODE / mass-matrix path
sol = tran!(circuit, (0.0, 1e-6); solver=FBDF())
```

Circuits are stiff, so an explicit method will not help; the useful axis is
which implicit method, not implicit versus explicit.

Finally, **simplify the stimulus**. Replace a `PULSE` with a `SIN`, a `SIN` with
a `DC`, and see which change makes the problem go away. Source breakpoints are
handed to the integrator as `tstops` automatically, so a waveform with sharp
edges should not by itself cause trouble — if it does, that is worth knowing.

## World age errors

`MNACircuit("amp.sp")` and `MNACircuit(code_string)` evaluate the generated
builder on the spot. Julia freezes the world age of a function when it is
entered, so a builder defined *during* a call cannot be dispatched to from
inside that same call:

```julia
function run_sim()
    circuit = MNACircuit("amp.sp")   # defines a new method...
    dc!(circuit)                     # ...that this world cannot see
end
```

If you see a `MethodError` or a world-age complaint naming a freshly generated
function, this is it. The fix is to load the netlist at top level and pass the
builder into the function — or to use the string macros, which expand at compile
time and are not affected:

```julia
Base.include(@__MODULE__, SpiceFile("amp.sp"))   # top level, defines `amp`

function run_sim()
    dc!(MNACircuit(amp; R1=1e3))                 # no eval, no world-age tax
end
```

[Loading happens at top level](@ref) covers the rule and the exceptions.

## The answer is wrong rather than absent

A run that converges to the wrong number is the harder case. Two things are
worth ruling out before anything else.

### Sign conventions

A branch current through a voltage source is positive *into* its `+` terminal,
so a source that is delivering power reads negative, and a card with its nodes
the other way round puts the net *below* ground:

```@example trouble
flipped = MNACircuit(sp"""
* same 5 V, terminals swapped
V1 0 vcc DC 5
R1 vcc 0 2k
""")

op[:I_v1], dc!(flipped)[:vcc]     # 5 V over 2 kΩ, sourced; and −5 V
```

Device terminal currents use the same convention — positive is *into* the
device — so a two-terminal device's currents are equal and opposite:

```@example trouble
using Cadnip.MNA: terminal_currents
terminal_currents(op)
```

### A second solution

A nonlinear circuit can have more than one DC operating point, and Newton finds
the one nearest where it started. If a warm start or a sweep gives a different
answer from a cold solve, both may be real. Pass `continuation=false` to
`dc!(::CircuitSweep)` to solve every point cold when following a branch is what
you *don't* want.

Then check the operating point against what the devices themselves report.
`op_vars` gives the model's own small-signal numbers, which is a far more direct
statement about what a transistor is doing than any node voltage:

```julia
op[:m1_vds] > op[:m1_vdsat]    # is M1 actually in saturation?
op[:m1_gm]                     # from the model, not inferred
```

See [Operating-point variables](@ref).
