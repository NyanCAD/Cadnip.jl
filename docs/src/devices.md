# Devices and models

## Built-in devices

Cadnip stamps the primitives itself. These need no model package and no `.model`
card:

| SPICE line | Device |
| ---------- | ------ |
| `R1 a b 1k` | resistor |
| `C1 a b 1n` | capacitor |
| `L1 a b 1m` | inductor |
| `V1 a b DC 5 AC 1` | independent voltage source (DC and AC excitation) |
| `I1 a b DC 1m` | independent current source |
| `V1 a b PWL(0 0 1u 1)` / `PULSE(...)` / `SIN(...)` | time-dependent sources |
| `E/G/H/F` lines | voltage- and current-controlled sources (VCVS, VCCS, CCVS, CCCS) |
| `X1 a b sub` | subcircuit instance |

Sources take a DC value, an AC magnitude and phase, and a transient waveform on
the same line; each analysis reads the part that applies to it, so one netlist
serves `dc!`, `ac!` and `tran!` without editing.

```spice
V1 in 0 DC 1.2 AC 1 SIN(1.2 5m 1meg)
```

## Model cards and the two-tier lookup

Anything with a `.model` card — diodes, MOSFETs, BJTs, JFETs — resolves through
two tiers:

**Tier 1, the registry.** Level-dispatched built-in and library models. A model
package adds methods to `Cadnip.ModelRegistry.getmodel`, so loading the package
is all it takes for the matching `.model` card to resolve:

```@example devices
using Cadnip
using Cadnip.MNA: MNACircuit
using VADistillerModels        # registers the SPICE3 model library

amp = MNACircuit(sp"""
* NMOS common-source stage
.model nch nmos level=1 vto=0.7 kp=100u lambda=0.01
Vdd vdd 0 DC 5
Vin gate 0 DC 1.1472
M1 drain gate 0 0 nch w=20u l=1u
Rd vdd drain 10k
""")

op = dc!(amp)
op[:drain], op[:i_m1_d], op[:m1_gm]
```

`VADistillerModels` claims the SPICE3 model cards: `nmos`/`pmos` at levels 1, 2,
3, 6, 9, BSIM3v3 (levels 8/49) and BSIM4v8 (levels 14/54), `npn`/`pnp`
(Gummel–Poon), `njf`/`pjf` (levels 1/2), and `d`. The library carries more
models than it registers cards for (MESFET, passives), reachable by name from
the netlist scope. Other model packages in this repository register their own
cards — `PSPModels` (PSP103, JUNCAP200), `CMCModels`, `VACASKModels`,
`PhotonicModels`.

The `:r`, `:c` and `:l` cards resolve to Cadnip's own primitives. The `:d` card
deliberately does *not*: `MNA.Diode` is an incomplete reference implementation
(no `cjo`, `m`, `bv`, `tt`), so a diode model card needs a Tier-1 provider such
as `VADistillerModels` rather than silently getting a partial device.

**Tier 2, the netlist scope.** PDK-specific and custom Verilog-A devices come
from the netlist's own directives — `.hdl "foo.va"`, `.include "foo.sp"`,
`.lib "foo.sp" section`, and their `jlpkg://Package/path` forms. The most recent
include wins.

## Verilog-A models

Verilog-A is the supported way to add a device. Cadnip compiles the module into
the same stamping code its built-in devices use, so a VA device is a first-class
device: it takes part in DC, AC, transient and noise, contributes terminal
currents and operating-point variables, and is differentiable.

```@example devices
va"""
module VAResistor(p, n);
    parameter real R = 1000.0;
    inout p, n;
    electrical p, n;
    analog I(p,n) <+ V(p,n)/R;
endmodule
"""
nothing # hide
```

!!! warning "Disciplines are implicit"
    `electrical`, `V()` and `I()` are built into the parser. Do **not**
    `include "disciplines.vams"` — it triggers parser bugs.

!!! note "ForwardDiff must be loadable"
    The code generated for a Verilog-A module imports `ForwardDiff` (it is what
    carries the derivatives through the device equations), so `ForwardDiff` has
    to be resolvable in the environment that loads the model — add it to the
    project that contains your `va"..."` or `.hdl`-including netlist.

From a netlist, a `.hdl` directive brings the module into scope and the module
name becomes the device's model name:

```spice
.hdl "myresistor.va"
N1 a b varesistor r=2k
```

Two model features are picked up from ordinary Verilog-A declarations, with
nothing extra to configure:

- a module-level `real`/`integer` with a `desc`/`units` attribute is an
  **operating-point variable**, reported as `op[:m1_gm]` and friends:

  ```verilog
  (* desc = "Transconductance", units = "S" *) real gm;
  ```

- `white_noise(pwr)` and `flicker_noise(pwr, exp)` register **noise sources**
  named `<instance>_<label>`, which `noise!` then decomposes by name. The call
  itself evaluates to zero in the time domain, and the whole registration folds
  away on the transient path.

## PDKs as packages

A PDK author bakes netlist and Verilog-A content into a package at build time:

```julia
Cadnip.precompile_pdk(@__MODULE__, "pdk.spice")
Cadnip.precompile_va(@__MODULE__, "device.va")
```

Users then reference the baked content from their netlist, and pay no parse or
compile cost for it at run time:

```spice
.lib "jlpkg://MyPDK/models/corners.sp" typical
```

PDK modules are `baremodule`s so that SPICE names like `inv`, `log` and `exp`
cannot collide with Julia's; generated code spells out `Base.` for anything it
needs from `Base`.

## Custom devices in Julia

Below the netlist layer, a device is a type with a `stamp!` method that writes
its contributions into an `MNAContext`. This is Cadnip's internal API — it is
what the built-in devices and the Verilog-A codegen both target — and is
documented in the repository's `doc/mna_architecture.md` and
`doc/mna_ad_stamping.md`.

```@example devices
using Cadnip.MNA

function rc(params, spec, t=0.0; x=Float64[], ctx=MNAContext())
    reset_for_restamping!(ctx)
    src = get_node!(ctx, :src)
    out = get_node!(ctx, :out)
    stamp!(VoltageSource(params.vin; name=:V1), ctx, src, 0)
    stamp!(Resistor(params.R), ctx, src, out)
    stamp!(Capacitor(params.C), ctx, out, 0)
    return ctx
end

dc!(MNACircuit(rc; vin=1.0, R=1e3, C=1e-9))[:out]
```

A builder written this way is an ordinary `MNACircuit` builder: sweeps, `alter`
and every analysis work on it. What it does *not* get is the netlist's parameter
bookkeeping — it declares no parameter names, so overrides are not checked
against it, and it has to read `params.<name>` itself.

For a custom device meant to be used from a netlist, prefer Verilog-A: it gets
model-card parameter handling, noise, operating-point variables and terminal
currents for free.

## ModelingToolkit components

With ModelingToolkit loaded, `@declare_MSLConnector` wraps an MTK component with
electrical pins into a Cadnip device, so a model developed and validated in the
MTK ecosystem can be dropped into a SPICE circuit:

```@docs; canonical=false
@declare_MSLConnector
```
