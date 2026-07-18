# Full SPICE device support — design (milestone 2)

**Status:** design approved (brainstorming). Next: implementation plan (writing-plans).

## Goal

Give the Rust-parser → VACASK pipeline **full SPICE device breadth**. A `.cir`/SPICE-first
netlist (or an embedded `simulator lang=spice` region inside a Spectre file) must parse,
elaborate, and simulate in VACASK with no Julia runtime — covering passives, sources
(DC/AC/tran functions), diodes, MOSFETs, BJTs, controlled sources, and subcircuit calls.

Builds on milestone 1 (Spectre path: `netlist::Netlist` → `ParserTables` → elaborate → tran,
already shipped). This is the deferred "Task 7" from `doc/vacask_rust_parser_integration_plan.md`.

## Key decisions (from brainstorming)

1. **SPICE devices are NOT cast into Spectre `Instance`.** A `simulator lang=spice` region
   parses to an embedded `SPICENetlistSource`; it projects into a distinct, **SPICE-native**
   `SpiceBlock`. Mirrors Julia, which keeps `SP.*` forms separate from `SC.*` and maps them in
   codegen (`sema.jl:133` recurses into the SPICE-block handler at `:362`; `cg_instance!` has
   per-`SP.*` methods).
2. **The Rust projection stays purely structural.** `SpiceBlock` carries device *kind* +
   positional nodes + raw source text. **No** master-name or param-name rewriting on the Rust
   side — that is consumer-specific (VACASK's masters differ from circulax's).
3. **The C++ (VACASK) adapter does the SPICE→VACASK mapping** (kind→master, positional
   nodes/value→named PT params), mirroring Julia's `cg_instance!` and VACASK's device set.
4. **Full breadth on projection; adapter bounded by VACASK's device set.** VACASK has no master
   for Switch, behavioral `B`, JFET, or mutual-inductor `K` — the adapter **warns and skips**
   those (a visible diagnostic, never a silent drop). Everything VACASK supports is mapped.
5. **Verification:** per-device Rust projection unit tests + C++ adapter unit tests + a few
   end-to-end `.cir` sims diffed against **ngspice-43** (`/usr/local/bin/ngspice`).

## VACASK target device set (from `build/lib/vacask/mod` + `lib/devbuiltins.cpp`)

| SPICE | VACASK master | Notes |
|---|---|---|
| R / C / L | `resistor` / `capacitor` / `inductor` (OSDI) | positional value → `r`/`c`/`l`; or model ref |
| V / I | `vsource` / `isource` (builtin) | DC/AC/tran function → named params |
| D | `diode` (OSDI) + model card | |
| M | `bsim3v3` / `bsim4v8` / `bsimbulk106` / `psp103v4` (OSDI) | via `.model` level/version dispatch; terminals d g s b |
| Q | `vbic` / `vbic_1p3_4t` / `vbic_1p3_5t` (OSDI) | terminals c b e [s] |
| E / G / F / H | `vcvs` / `vccs` / `cccs` / `ccvs` (builtin) | control nodes / controlling source |
| X | master = subckt name | subcircuit call |
| Switch, B, JFET, K | — none — | **warn + skip** |

Source param convention (empirical, `test/resonator.sim` / `demo/api/rc.scs`):
`dc=<v>`, `mag=<v> phase=<v>` (AC), `type="pulse" val0= val1= delay= rise= fall= width= [period=]`,
`type="sine" ...`, `type="pwl" ...`. The adapter maps SPICE positional function args
(`PULSE(v0 v1 td tr tf pw per)`, `SIN(vo va freq td theta)`, `PWL(t0 v0 t1 v1 …)`, `EXP(...)`,
`SFFM(...)`) → these named params. Exact per-function arg tables are enumerated in the plan.

## SpiceBlock POD schema (cxx bridge additions to `crates/netlist-cxx/src/lib.rs`)

```
// C-style shared enum (cxx-compatible)
enum SpiceDeviceKind {
    Resistor, Capacitor, Inductor, VSource, ISource, Diode, Mosfet, Bjt, Jfet,
    SubcktCall, Vcvs, Vccs, Ccvs, Cccs, MutualInductor, Behavioral, Switch, Osdi
}

struct SpiceSource {          // populated for VSource/ISource
    dc: String,
    ac_mag: String, ac_phase: String,
    tran_kind: String,        // "" | pulse | sine | pwl | exp | sffm
    tran_args: Vec<String>,   // positional function args, verbatim
}

struct SpiceDevice {
    kind: SpiceDeviceKind,
    name: String,
    nodes: Vec<String>,       // canonical SPICE order per kind (R:[+,−], M:[d,g,s,b], Q:[c,b,e,(s)])
    value: String,            // R/C/L positional value (verbatim) or ""
    model: String,            // D/M/Q/OSDI model-card ref or ""
    params: Vec<Param>,       // trailing name=value params
    ctrl_nodes: Vec<String>,  // E/G control nodes; F/H controlling Vsource name in ctrl_value
    ctrl_value: String,
    source: SpiceSource,
}

struct SpiceModel { name: String, model_type: String /* d,nmos,pnp,… */, level: String, params: Vec<Param> }

struct SpiceSubckt {          // SPICE .subckt (recursive, SPICE-native)
    name: String, ports: Vec<String>, params: Vec<Param>,
    devices: Vec<SpiceDevice>, models: Vec<SpiceModel>, subckts: Vec<SpiceSubckt>,
}

struct SpiceBlock {
    params: Vec<Param>,           // .param
    models: Vec<SpiceModel>,      // .model
    subckts: Vec<SpiceSubckt>,    // .subckt
    devices: Vec<SpiceDevice>,
    includes: Vec<Include>,       // .include / .lib (resolved in C++, as in milestone 1)
}
```

`Netlist` and `Subckt` each gain `spice_blocks: Vec<SpiceBlock>`.

## Rust projection (`crates/netlist-cxx/src/lib.rs`)

- In `collect_scope`, add a `SyntaxKind::SPICENetlistSource` arm calling a new
  `project_spice_block(node) -> SpiceBlock`, pushed onto the enclosing scope's `spice_blocks`.
- `project_spice_block` walks the SPICE typed AST (`netlist_syntax::ast`) — `Resistor`/
  `Capacitor`/`Inductor` (`pos`/`neg`/`value`), `Voltage`/`Current` (DC/AC/Tran source funcs),
  `Diode`, `MOSFET` (d/g/s/b/model), `BipolarTransistor`, `SubcktCall`, `ControlledSource`
  (E/F/G/H with `VoltageControl`/`CurrentControl`), `MutualInductor`, `Switch`, `Behavioral`,
  `Model`, `Subckt` — populating `SpiceDevice` per kind. Structural only: no renaming.
- Node vectors use the SPICE canonical order for each kind (documented per kind).
- SPICE `.include`/`.lib` collected into `SpiceBlock.includes`.

## C++ adapter (`../VACASK/lib/netlistrs.cpp`)

- New `spiceBlockToTables(const netlist::SpiceBlock&, PTSubcircuitDefinition& into, ParserTables&, Parser&, Status&) -> bool`, called from `mergeNetlist` for each `spice_blocks` entry (top level and, if present, within subckts).
- Per-`kind` switch → emit `PTInstance(name, master, terminals).add(params)`:
  - passives: master + positional value → `r`/`c`/`l` param (or model ref → PTInstance master = model, no value param);
  - sources: `vsource`/`isource`; `spiceSourceToParams(SpiceSource)` builds the `dc=`/`mag=`/`type="…" …` string → `parseParameters`;
  - D/M/Q: master via model-type / level dispatch (`spiceModelMaster(model_type, level, version)`), model card emitted as `PTModel(name, master)`;
  - E/G/F/H: `vcvs`/`vccs`/`cccs`/`ccvs` with control nodes / controlling-source param;
  - X: PTInstance master = subckt name;
  - Switch/B/JFET/K: `Simulator::err()` warning + skip.
- `.model` → `PTModel`; SPICE `.subckt` → `PTSubcircuitDefinition` (recursive, same per-kind mapping).
- Params re-parsed through `Parser::parseParameters` (verbatim-text invariant preserved) — same as the Spectre path.
- `SpiceBlock.includes` fed into the existing C++ include resolver (milestone 1).

## Dispatch / includes

No new dispatch: the milestone-1 `.cir`→SPICE / `.scs`→Spectre file entry already produces a
top-level `SPICENetlistSource`, now projected as a top-level `SpiceBlock` and merged into the
default subdef. The C++ include resolver (recursive, cycle-guarded, bool error propagation) is
reused for SPICE `.include`.

## Verification

- **Rust unit tests:** one per `SpiceDeviceKind` — assert the projected `SpiceBlock` shape
  (kind, node order, value/model/params, source fields).
- **C++ adapter unit tests:** feed a `SpiceBlock` (or parse a small `.cir`), build tables, assert
  master/param mapping via `tab.dump` (e.g. `M1` → `bsim4v8`, terminals d g s b; `V1 PULSE(...)`
  → `type="pulse" val0=… val1=…`); assert the warn-and-skip diagnostic for a `.switch`/`B` device.
- **End-to-end vs ngspice-43:** representative `.cir` netlists, run through VACASK and ngspice,
  node-voltage/waveform diff within tolerance:
  1. RC + `PULSE` source (passives + source + tran) — also analytic τ=R·C check.
  2. Diode half-wave rectifier (`.model d`).
  3. MOSFET stage (e.g. resistor-load inverter/amplifier, `.model` + BSIM level).
  4. Subcircuit instantiation (`X` + `.subckt`) with a parameter override.
- CTests added alongside the milestone-1 ones (WORKING_DIRECTORY `/tmp`, model dir via the
  existing `VACASK_MOD_DIR` mechanism).

## Out of scope / deferred

- Switch, behavioral `B`, JFET, mutual-inductor `K` → warn + skip (no VACASK master).
- Verilog-A `.hdl`/`ahdl_include` device compilation (OpenVAF owns `.va`→`.osdi`).
- circulax/Python consumer of `SpiceBlock` (separate track; the SPICE-native shape is designed to
  serve it, but no Python adapter here).
- Deriving OSDI `PTLoad`s from the netlist automatically — driver/test still adds the loads it
  needs (as milestone 1).
- Most-recent-include-wins override semantics (carried over from milestone 1's deferrals).

## Critical files

- Rust: `netlist-parser-rs/crates/netlist-cxx/src/lib.rs` (projection), `crates/netlist-syntax/src/ast.rs` (SPICE typed AST — accessors already complete).
- C++: `../VACASK/lib/netlistrs.cpp` + `include/netlistrs.h` (adapter), `demo/api/` (E2E netlists + CTests).
- Reference (mapping semantics): `src/spc/sema.jl` (`sema!` for `SPICENetlistSource`, `spice_select_device`), `src/spc/codegen.jl` (`cg_instance!` per `SP.*`, source-function handling ~line 391-418, MOSFET level dispatch ~438-473).
- VACASK device set: `build/lib/vacask/mod/*.osdi`, `lib/devbuiltins.cpp`.
