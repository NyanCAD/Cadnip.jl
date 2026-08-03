# Parameters and sweeps

A netlist `.param` is the knob a design is parameterized on: a bias voltage, a
device size, a source amplitude, a process corner. Cadnip treats those knobs as
the circuit's parameters — override one at construction, re-bind it with
`alter`, or make it a sweep axis, using the same name in every case.

## Overriding a `.param`

```@example params
using Cadnip
using Cadnip.MNA: MNACircuit

amp = sp"""
* parameterized divider
.param rtop=1k
.param rbot=1k
V1 vcc 0 DC 6
R1 vcc out rtop
R2 out 0 rbot
"""

c = MNACircuit(amp; rtop=2e3)      # override at construction
dc!(c)[:out]
```

`alter` returns a new circuit with a knob re-bound. It introduces the knob if
the circuit was not carrying it yet, so a `.param` need not be seeded at
construction time:

```@example params
c2 = alter(c; rbot=3e3)
dc!(c2)[:out]
```

Overrides reach every place the netlist uses the parameter: DC source values,
device values, `.model` card parameters, and the runtime-evaluated amplitudes
and frequencies of `SIN`/`PULSE` sources, at top level or inside a subcircuit.

## Hierarchy: a leaf is a parameter, a group is an instance

A scope — the top level, or a subcircuit instance — has parameters of its own
and children it instantiates, in two namespaces that can collide. One rule tells
them apart: **a leaf is a parameter of the scope, a group is a child.**

```@example params
top = sp"""
* divider as a subcircuit, with an instance parameter
.subckt divider a b r1val=1k
R1 a mid r1val
R2 mid b 1k
.ends
V1 vcc 0 DC 6
X1 vcc 0 divider r1val=2k
"""

c = MNACircuit(top; x1=(r1val=1e3,))    # a group → the instance X1
dc!(c)[:x1_mid]
```

An override outranks the instance line: `X1 ... r1val=2k` in the netlist still
leaves `r1val` reachable from `alter` and from a sweep axis. A dotted selector
addresses the same place and is what you use for a sweep axis name:

```@example params
dc!(alter(c; var"x1.r1val"=4e3))[:x1_mid]
```

When a name is *both* a parameter and an instance (a `.param x1` next to an `X1`
instance), the shape decides: `x1=2.0` sets the parameter, `x1=(r1val=…)`
addresses the instance, and `params=(x1=2.0,)` names the parameter explicitly
when you need both at once.

## Names that reach nothing throw

An override that names nothing used to be silent, which reads exactly like a
parameter with no effect — a swept design would come out as a flat curve rather
than an error. Names are now checked against what the netlist declares:

```@example params
try
    MNACircuit(amp; rtopp=2e3)          # typo
catch err
    println(sprint(showerror, err))
end
```

Two things are genuinely not reachable this way, and both throw at construction:

- **device instance parameters** (`r1=(r=2e3,)`, `m1=(w=…)`) — parameterize the
  netlist with a `.param` and override that instead;
- **a name no scope declares** — i.e. a typo, as above.

A hand-written builder function declares no names, so its parameters are not
checked.

## Corners are just parameters

A `.model` card can read a `.param`, so a process corner is an ordinary sweep
axis rather than a second netlist:

```spice
.param vt0=0.7
.model nch nmos level=1 vto=vt0 kp=100u
```

```julia
dc!(CircuitSweep(amp, Sweep(vt0 = [0.6, 0.7, 0.8])))
```

## Sweeps

Don't hand-roll a loop that rebuilds a circuit per value. Wrap the range in a
sweep, bind it to a builder with `CircuitSweep`, and run the analysis: `dc!` and
`tran!` return a `SweepResult` that iterates `(params, sol)` pairs, with
each parameter point aligned to its solution.

```@example params
sweep = CircuitSweep(amp, Sweep(rtop = [1e3, 2e3, 4e3]))

for (p, sol) in dc!(sweep)
    println(p.rtop, " Ω → ", sol[:out], " V")
end
```

Keyword arguments to `CircuitSweep` set the *base* value of everything the sweep
does not vary; a swept axis needs no seeding of its own.

```julia
cs = CircuitSweep(ce_amplifier, Sweep(vac = [0.5e-3, 1e-3, 2e-3]); freq=1e3)
for (params, sol) in tran!(cs, (0.0, 5e-3))
    # ...
end
```

Axes combine in three ways:

| Constructor | Meaning |
| ----------- | ------- |
| `ProductSweep(a=…, b=…)` | full grid: every `a` with every `b` |
| `TandemSweep(a=…, b=…)` | zipped: `a[i]` with `b[i]` |
| `SerialSweep(s1, s2)` | one sweep after the other |

`Sweep(a=…, b=…)` with several keywords is a product sweep. Hierarchical axes
use the same dotted selectors as `alter`:

```julia
CircuitSweep(top, Sweep(var"x1.r1val" = [1e3, 2e3]))
```

## DC sweeps continue

`dc!(::CircuitSweep)` warm-starts each point from the previous *converged*
operating point, the way a SPICE `.dc` sweep does. Adjacent points are usually a
small perturbation of each other, so the warm start lands inside Newton's
quadratic-convergence basin and costs markedly fewer iterations — on a
40-junction ladder over 60 points, 477 Newton iterations continued against 877
cold.

```julia
result = dc!(CircuitSweep(circuit, sweep))                      # continued (default)
result = dc!(CircuitSweep(circuit, sweep); continuation=false)  # independent points
```

A point that fails to converge is never used as a starting guess, a guess of the
wrong length (a point that changed the system size) is dropped, and the
GMIN/source-stepping fallbacks restart from zeros — so a bad guess costs
iterations, never a solution. Pass `continuation=false` when following a branch
is exactly what you don't want, e.g. on a circuit with multiple DC solutions.

A single operating point can be warm-started by hand with the same seam:

```julia
sol  = dc!(circuit)
warm = dc!(alter(circuit; rtop=1.01e3); u0=sol.x)
```

Transient sweeps DC-initialize each point independently.
