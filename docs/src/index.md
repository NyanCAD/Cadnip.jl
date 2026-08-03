# Cadnip.jl

**C**ircuit **A**nalysis & **D**ifferentiable **N**umerical **I**ntegration **P**rogram

Cadnip is an analog circuit simulator written in Julia. It reads SPICE and
Spectre netlists and Verilog-A models, compiles them into Julia code, and solves
them with a Modified Nodal Analysis (MNA) engine on top of the SciML solver
stack. Because the whole simulator is ordinary Julia, a circuit is
differentiable, sweepable, and composable with the rest of the ecosystem.

## Installation

Cadnip is in the General registry:

```julia
using Pkg
Pkg.add("Cadnip")
```

Cadnip itself provides the passives, sources, and the analyses. Transistor and
diode model cards (`.model nch nmos level=1`, `.model dmod d`, …) resolve
through [model packages](@ref "Model cards and the two-tier lookup") — the ones
this repository ships live under `models/` (`VADistillerModels`, `PSPModels`,
`CMCModels`, …) and are loaded alongside Cadnip with `using`.

## A first circuit

A netlist is the input, and the same `sol[:name]` readout serves every analysis:

```@example index
using Cadnip
using Cadnip.MNA: MNACircuit, nameat

divider = MNACircuit(sp"""
* resistive divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
""")

sol = dc!(divider)
sol[:out]
```

!!! note "Two namespaces"
    `Cadnip` exports the analyses and the sweep API (`dc!`, `tran!`, `ac!`,
    `noise!`, `alter`, `Sweep`, `CircuitSweep`, `acdec`, …) and the netlist
    macros. The circuit and device layer lives in the `Cadnip.MNA` submodule, so
    `MNACircuit` and friends are imported from there — `using Cadnip.MNA:
    MNACircuit, nameat`, as above.

Something with a time constant, driven by a step:

```@example index
rc = MNACircuit(sp"""
* RC low-pass driven by a 0 → 1 V step at t = 1 µs
V1 in 0 PWL(0 0 1u 0 1.001u 1)
R1 in out 1k
C1 out 0 1n
""")

sol = tran!(rc, (0.0, 20e-6))
[nameat(sol, :out, t) for t in (0.0, 2e-6, 20e-6)]   # before the step, mid-rise, settled
```

and the same circuit in the frequency domain — the pole sits at
1/(2π·1 kΩ·1 nF) ≈ 159 kHz:

```@example index
lowpass = MNACircuit(sp"""
* RC low-pass, AC excitation
V1 in 0 DC 0 AC 1
R1 in out 1k
C1 out 0 1n
""")

ac = ac!(lowpass, acdec(10, 1e3, 1e7))
magnitude_db(ac, :out)[1:3]     # flat in the passband
```

## Where to go next

| Page | What it covers |
| ---- | -------------- |
| [Loading circuits](@ref) | netlist files, string macros, subcircuits, include/PDK directives |
| [Parameters and sweeps](@ref) | `.param` overrides, `alter`, sweep axes, continuation |
| [Analyses](@ref) | `dc!`, `tran!`, `ac!`, `noise!` and how to read their results |
| [Devices and models](@ref) | built-in devices, model cards, Verilog-A, custom devices |

Design notes and internals live in the `doc/` directory of the repository —
`doc/mna_architecture.md` and `doc/code_tour_mna_pipeline.md` are the entry
points for the compiler and solver pipeline.
