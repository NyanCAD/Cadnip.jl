# How a circuit becomes equations

Nothing on this page is needed to *use* Cadnip — [Loading circuits](@ref) is
where to start for that. It is here because the vocabulary the rest of the
manual uses (a *stamp*, the `G` and `C` matrices, a *branch current*, why `dc!`
can fail to converge at all) comes from one piece of numerical machinery, and
the machinery is easier to work with than to guess at.

## Lumped circuits

Electrical circuits are governed by Maxwell's equations, but unless you are
building antennas, a *lumped* model is enough: discrete elements connected by
ideal wires. Even non-ideal wires are handled this way — a transmission line
becomes a chain of discrete elements, and recovering that chain from a physical
layout is what parasitic extraction does.

Every element in the lumped model has a *characteristic equation* relating the
current through it to the voltage across it:

| element | characteristic equation |
|---|---|
| current source | ``I`` imposed; infinite resistance |
| voltage source | ``V`` imposed; zero resistance |
| resistor | ``V = IR`` |
| capacitor | ``I = C\,\mathrm{d}V/\mathrm{d}t`` |
| inductor | ``V = L\,\mathrm{d}I/\mathrm{d}t`` |

plus the four controlled sources (voltage- or current-controlled, sourcing a
voltage or a current), which is what the `E`, `F`, `G` and `H` cards are.

Everything else is built from these. A diode is a nonlinear resistor; a MOSFET
and a BJT are, to a first approximation, voltage-controlled current sources
whose characteristic equation is a few thousand lines of Verilog-A rather than
one line of algebra. A simple long-channel NMOS in saturation is
``I_D = \frac{\mu C_{ox}}{2}\frac{W}{L}(V_{GS}-V_{th})^2``; a modern compact
model is the same idea with several hundred parameters covering short-channel
effects, overlap capacitances, and temperature.

## The netlist

A netlist is literally a list of nets and components: topology, parameters, and
nothing about layout or geometry. In SPICE syntax the first character of a card
names the device *type*, the rest of the first word names the *instance*, then
come the nets its terminals connect to, then its parameters. A line starting
with `*` is a comment, a line starting with `.` is a directive, and the first
line of a deck is always its title.

```@example background
using Cadnip
using Cadnip.MNA: MNACircuit, MNAContext, assemble!

filter3 = sp"""
* third-order Butterworth low-pass
V1 vin 0 SIN(0 1 3.14)
L1 vin n1 1.5
C1 n1 0 1.333
L2 n1 vout 0.5
R1 vout 0 1
"""
nothing # hide
```

`V1` is a voltage source between `vin` and ground; `L1` is an inductor between
`vin` and `n1`; and so on. A pen-and-paper analysis of this filter would combine
series and parallel impedances (``Z = sL``, ``Z = 1/sC``) until a transfer
function falls out. That is not what a simulator does.

## Modified nodal analysis

What a simulator does is [modified nodal
analysis](https://en.wikipedia.org/wiki/Modified_nodal_analysis). Take one
unknown per net — its voltage, referred to ground, which is the reference node
and gets no unknown of its own — and write [Kirchhoff's current
law](https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws) at each: the
currents of every element meeting there sum to zero. For an element whose
current is a function of the node voltages (a resistor, a capacitor, a
current source) that is all you need.

The *modified* part covers the elements where it is not. A voltage source has no
current-in-terms-of-voltage form at all, and an inductor's is an integral. Each
of those gets an extra unknown — its branch current — and an extra row, its own
characteristic equation. That is why a solution names both node voltages and
branch currents:

```@example background
sol = dc!(MNACircuit(filter3))
node_names(sol), branch_names(sol)
```

Cadnip assembles the result into

```math
G\,x + C\,\dot{x} = b
```

where `x` stacks the node voltages and the branch unknowns, `G` collects
everything algebraic (conductances, and the ±1 entries that tie a voltage
source's branch current to its two nodes), `C` collects everything
differentiated in time (capacitances, inductances), and `b` collects the
independent sources. The three are exactly the fields of the assembled system:

```@example background
divider = sp"""
* resistive divider with one voltage source
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
"""

circuit = MNACircuit(divider)
sys = assemble!(circuit.builder(circuit.params, circuit.spec))

sys.node_names, sys.current_names
```

```@example background
Matrix(sys.G)
```

Read that against the names above. The leading block, indexed by the nodes, is
the conductance matrix of the two resistors: `1/1k + 1/1k` on `out`'s diagonal,
`-1/1k` coupling it to `vcc`. `V1` owns the last row and the last column. Its
column entry sits in `vcc`'s KCL row and says *this branch's current leaves
`vcc`* — which is why a source that is delivering power reads negative. Its row
is the source's own characteristic equation, `V(vcc) = b`, and the only thing in
`b`:

```@example background
sys.b
```

## Stamping

No part of that matrix was derived symbolically. Each device is handed the
context and the indices of its own nodes, and it *stamps* its contribution — a
few entries added into `G`, `C` and `b`. Nothing has global knowledge of the
circuit, which is why a Verilog-A model from a PDK drops in next to a built-in
resistor without either one knowing about the other, and why assembly costs one
pass over the devices however large the circuit gets.

Nonlinear devices stamp a *linearization* rather than their equation. At the
current guess for `x`, a diode contributes its small-signal conductance
``\partial I/\partial V`` to `G` and the leftover ``I - V\,\partial I/\partial V``
to `b` — the companion model of the textbooks. Solving that linear system gives
a better guess, and repeating is Newton's method. This is why the builder
function signature carries `x`: a device has to know the voltage it is being
linearized *at*.

```julia
function circuit(params, spec, t=0.0; x=Float64[], ctx=nothing)
```

## Where each analysis sits

Every analysis is a different thing done to `G x + C ẋ = b`.

- **DC** (`dc!`) drops the time derivative — ``G x = b`` — and Newton-iterates
  until the stamps stop moving. For a linear circuit that converges in one step.
  For anything else, [When a circuit does not solve](@ref) is the page about it.
- **Transient** (`tran!`) integrates the whole thing. `C` is singular whenever
  some node has no capacitance to ground — which is most circuits — so this is a
  differential-*algebraic* system, not an ODE, and the default solver is a DAE
  solver.
- **AC** (`ac!`) linearizes once at the DC operating point and solves
  ``(G + \mathrm{j}\omega C)\,x = b`` per frequency. The RC pole below falls
  where ``\omega C`` overtakes ``G``:

```@example background
rc = MNACircuit(sp"""
* RC low-pass, AC excitation
V1 in 0 DC 0 AC 1
R1 in out 1k
C1 out 0 1n
""")

f = acdec(10, 1e3, 1e7)
mag = magnitude_db(ac!(rc, f), :out)
f[findfirst(<(mag[1] - 3), mag)]     # first grid point past −3 dB; the pole is
                                     # 1/(2π·1k·1n) ≈ 159 kHz
```

- **Noise** (`noise!`) reuses that same linearization, transposed: one adjoint
  solve per frequency carries every noise source in the circuit to the output at
  once.

## Verilog-A and compact models

A compact model is not data the simulator interprets at run time — it is
compiled, the same way the netlist is. A Verilog-A module becomes a Julia struct
holding the module's parameters, plus a `stamp!` method generated from its
`analog` block that writes the model's contributions into the context exactly as
the built-in resistor above does. From the assembler's side there is no
difference between the two.

The derivatives that go into `G` and `C` are produced by that generated code
rather than entered separately, so a model's small-signal behaviour is whatever
its equations say it is — there is no second description to keep in sync. It is
also why `gm` and `vdsat` can be read out of an operating point: they are
variables the model itself computed on the way to its stamp. See
[Verilog-A models](@ref) for writing one.

## Further reading

- Sandia's [Xyce mathematical
  formulation](https://xyce.sandia.gov/files/xyce/Xyce_Math_Formulation.pdf)
  is the clearest write-up of how a production simulator assembles these
  matrices.
- `doc/code_tour_mna_pipeline.md` in the repository traces the implementation:
  the builder function, the context types, and how the sparsity pattern
  discovered on the first build lets every later rebuild run without allocating.
