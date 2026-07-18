# Full SPICE Device Support — Implementation Plan (milestone 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Project every SPICE device kind into a distinct SPICE-native `SpiceBlock` (Rust cxx bridge) and map it to VACASK `ParserTables` in the C++ adapter, so `.cir`/SPICE netlists parse → elaborate → simulate in VACASK.

**Architecture:** A `simulator lang=spice` region parses to an embedded `SPICENetlistSource`. The Rust projection (`crates/netlist-cxx/src/lib.rs`) walks it into `SpiceBlock` using the SPICE typed AST (`ast.rs`) — purely structural (kind + positional nodes + raw text). The VACASK C++ adapter (`lib/netlistrs.cpp`) maps each device kind → VACASK master + named PT params, mirroring Julia `cg_instance!`. Design: `doc/spice_support_design.md`.

**Tech stack:** Rust (cxx 1.0, rowan), C++20, CMake/Corrosion, VACASK, ngspice-43 (`/usr/local/bin/ngspice`) as E2E oracle.

## Global Constraints

- **Rust projection is purely structural.** `SpiceBlock` carries device *kind* + positional nodes + verbatim text. NO master-name or param-name rewriting on the Rust side.
- **All SPICE→VACASK mapping lives in the C++ adapter.** Kind→master, positional nodes/value→named PT params. Params re-parsed via `Parser::parseParameters` (verbatim-text invariant) — never hand-classified PV/PE, never a bespoke evaluator.
- **Never cast a SPICE device into the Spectre `Instance` struct.** SPICE devices flow only through `SpiceBlock`/`SpiceDevice`.
- **Full breadth on projection; adapter bounded by VACASK's device set.** VACASK masters: `resistor`/`capacitor`/`inductor`/`diode` (OSDI), `bsim3v3`/`bsim4v8`/`bsimbulk106`/`psp103v4` (MOSFET OSDI), `vbic`/`vbic_1p3_4t`/`vbic_1p3_5t` (BJT OSDI), `vsource`/`isource`/`vcvs`/`vccs`/`cccs`/`ccvs` (builtin). Kinds with NO VACASK master — **Switch, Behavioral (B), JFET, MutualInductor (K)** — must `Simulator::err()` **warn and skip** (visible, never silent).
- **Two repos:** Rust crate in `Cadnip.jl/netlist-parser-rs/` (branch `rust-parser-spike`); VACASK in `../VACASK/` (branch `rust-parser-integration`). Commit separately.
- **cxx header include is `netlist_cxx_bridge/lib.h`; `NAMESPACE`=`sim`; `Status` setter is `s.set(Status::Syntax|NotFound, msg)` (no `Status::Error`).** Params round-trip through `Parser::parseParameters`.
- Build/test: `cmake --build /home/pepijn/code/nyanodide/build.VACASK/Release --target demo_netlistrs`; CTests from that build dir with `ctest -R ... --output-on-failure`; run VACASK with CWD `/tmp`; model dir via existing `VACASK_MOD_DIR` mechanism.

## SPICE typed-AST accessors (pinned from `ast.rs`)

- `Resistor`/`Capacitor`/`Inductor` (macro `two_terminal_value`): `name()`, `pos()`, `neg()`, `value() -> Option<Expr>`, plus trailing `Parameter`s.
- `Voltage`/`Current` (macro `source_instance`): `name()`, `pos()`, `neg()`, `sources() -> impl Iterator<Item=SyntaxNode>` (DCSource/ACSource/TranSource).
  - `DCSource.value()`; `ACSource.magnitude()`, `.phase()`; `TranSource.function() -> Option<SyntaxToken>` (PULSE/SIN/PWL/…), `TranSource.values() -> impl Iterator<Item=Expr>`.
- `Diode`: `name()`, `pos()`, `neg()`, `model()`, `params()`.
- `MOSFET`: `name()`, `drain()`, `gate()`, `source()`, `bulk()`, `model()`, `params()`.
- `BipolarTransistor`: `name()`, `nodes()` (c, b, e, [s], model — positional), `params()`.
- `SubcktCall`: `name()`, `nodes()` (connection nodes + master last, positional), `params()`.
- `ControlledSource`: `name()`, `pos()`, `neg()`, `control() -> Option<SyntaxNode>` (VoltageControl/CurrentControl/PolyControl/TableControl).
  - `VoltageControl.control_nodes()`, `.value()`, `.params()`; `CurrentControl.vnam()`, `.value()`, `.params()`.
- `Model`: `name()`, `model_type() -> Option<SyntaxToken>`, `params()`.
- `Subckt`: `name() -> Option<SyntaxToken>`, `ports()`, `params()`, `body() -> impl Iterator<Item=SyntaxNode>`.
- `MutualInductor`, `Switch`, `Behavioral` (+ JFET via `fet_device`): expose `name()` + `nodes()`/positional accessors + `params()` (see `ast.rs`). Projected structurally; adapter warn-skips K/Switch/B/JFET.
- Helpers already in `lib.rs`: `tok_text`, `unquote`, `project_param` (Spectre), and for SPICE `Parameter` use `p.name()` + `p.value().map(|v| v.text())` (see `project_spice_param` pattern). `HierarchialNode` text via a small `hier_text(&HierarchialNode) -> String` helper (`.base()?.token()?.text()`), added in Task S1.

## Existing helper reference (SPICE Parameter/node text)

From an earlier reverted attempt, these shapes are known-good:
```rust
fn hier_text(n: &netlist_syntax::ast::HierarchialNode) -> String {
    n.base().and_then(|b| b.token()).map(|t| t.text().to_string()).unwrap_or_default()
}
fn project_spice_param(p: &netlist_syntax::ast::Parameter) -> ffi::Param {
    ffi::Param { name: tok_text(p.name()), value: p.value().map(|v| v.text().to_string()).unwrap_or_default() }
}
```

---

### Task 1: Rust — SpiceBlock schema + passive projection (R/C/L) [S1]

**Files:** Modify `Cadnip.jl/netlist-parser-rs/crates/netlist-cxx/src/lib.rs`. Test: same file `#[cfg(test)]`.

**Interfaces — Produces (the whole bridge schema; later tasks fill more device kinds):**
- Shared enum `SpiceDeviceKind` and structs `SpiceSource`, `SpiceDevice`, `SpiceModel`, `SpiceSubckt`, `SpiceBlock` in the `#[cxx::bridge]` mod.
- `Netlist` and `Subckt` each gain `spice_blocks: Vec<SpiceBlock>`.
- Free fns: `hier_text`, `project_spice_param`, `project_spice_block(node: SyntaxNode) -> SpiceBlock`.

- [ ] **Step 1: Add the bridge schema.** In the `#[cxx::bridge(namespace = "netlist")] mod ffi`, add (place enums/structs before `Netlist`):

```rust
    #[derive(Debug, Clone, Copy)]
    enum SpiceDeviceKind {
        Resistor, Capacitor, Inductor, VSource, ISource, Diode, Mosfet, Bjt,
        Jfet, SubcktCall, Vcvs, Vccs, Ccvs, Cccs, MutualInductor, Behavioral, Switch, Osdi,
    }
    struct SpiceSource {
        dc: String, ac_mag: String, ac_phase: String,
        tran_kind: String, tran_args: Vec<String>,
    }
    struct SpiceDevice {
        kind: SpiceDeviceKind,
        name: String,
        nodes: Vec<String>,
        value: String,
        model: String,
        params: Vec<Param>,
        ctrl_nodes: Vec<String>,
        ctrl_value: String,
        source: SpiceSource,
    }
    struct SpiceModel { name: String, model_type: String, level: String, params: Vec<Param> }
    struct SpiceSubckt {
        name: String, ports: Vec<String>, params: Vec<Param>,
        devices: Vec<SpiceDevice>, models: Vec<SpiceModel>, subckts: Vec<SpiceSubckt>,
    }
    struct SpiceBlock {
        params: Vec<Param>, models: Vec<SpiceModel>, subckts: Vec<SpiceSubckt>,
        devices: Vec<SpiceDevice>, includes: Vec<Include>,
    }
```
Add `spice_blocks: Vec<SpiceBlock>` to both `struct Netlist` and `struct Subckt`.

- [ ] **Step 2: Extend the Rust `Scope` and initialize the new fields.** Add `spice_blocks: Vec<ffi::SpiceBlock>` to the internal `Scope` struct; set `spice_blocks: scope.spice_blocks` in both the `Netlist` builder (`parse_netlist`) and `project_subckt`. A default/empty `SpiceSource { dc:"".into(), ac_mag:"".into(), ac_phase:"".into(), tran_kind:"".into(), tran_args: vec![] }` helper is handy.

- [ ] **Step 3: Write the failing test.**
```rust
#[test]
fn projects_spice_block_passives() {
    let nl = super::parse_netlist("* t\nR1 a b 1k\nC1 b 0 1u\nL1 a b 2n\n", true);
    assert!(nl.errors.is_empty());
    assert_eq!(nl.spice_blocks.len(), 1);
    let d = &nl.spice_blocks[0].devices;
    assert_eq!(d.len(), 3);
    assert_eq!(d[0].kind, super::ffi::SpiceDeviceKind::Resistor);
    assert_eq!(d[0].name, "R1");
    assert_eq!(d[0].nodes, vec!["a", "b"]);
    assert_eq!(d[0].value, "1k");
}
```

- [ ] **Step 4: Run to verify it fails** — `cd Cadnip.jl/netlist-parser-rs && cargo test -p netlist-cxx projects_spice_block_passives` → FAIL (field/function missing).

- [ ] **Step 5: Implement `hier_text`, `project_spice_param`, empty-source helper, and `project_spice_block` covering R/C/L.** In `collect_scope`, add:
```rust
SyntaxKind::SPICENetlistSource => {
    scope.spice_blocks.push(project_spice_block(stmt));
}
```
`project_spice_block` walks `node.children()`; for R/C/L (`ast::Resistor`/`Capacitor`/`Inductor`) build a `SpiceDevice` with the matching `SpiceDeviceKind`, `nodes = [hier_text(pos), hier_text(neg)]`, `value = value().map(|e| e.text()).unwrap_or_default()`, `params` from trailing `Parameter`s, empty source/model/ctrl fields. Push onto `block.devices`. (Other kinds handled in S1b–S1d; leave a `_ => {}` for now.)

- [ ] **Step 6: Run tests** — `cargo test -p netlist-cxx` → PASS (new + existing). Warnings pristine.

- [ ] **Step 7: Commit** — `git add netlist-parser-rs/crates/netlist-cxx/src/lib.rs && git commit -m "feat(netlist-cxx): SpiceBlock schema + passive (R/C/L) projection"`

---

### Task 2: Rust — SPICE source projection (V/I DC/AC/tran) [S1b]

**Files:** `crates/netlist-cxx/src/lib.rs` (extend `project_spice_block`). Test: same.

**Interfaces:** Consumes S1 schema. Produces: `project_source(&ast::Voltage/Current) -> SpiceDevice` populating `SpiceSource`.

- [ ] **Step 1: Failing test.**
```rust
#[test]
fn projects_spice_sources() {
    let nl = super::parse_netlist(
        "* t\nV1 1 0 DC 5 AC 1 PULSE(0 5 1m 1u 1u 4m 10m)\nI1 0 2 DC 1m\n", true);
    let d = &nl.spice_blocks[0].devices;
    let v = d.iter().find(|x| x.name == "V1").unwrap();
    assert_eq!(v.kind, super::ffi::SpiceDeviceKind::VSource);
    assert_eq!(v.nodes, vec!["1", "0"]);
    assert_eq!(v.source.dc, "5");
    assert_eq!(v.source.ac_mag, "1");
    assert_eq!(v.source.tran_kind.to_lowercase(), "pulse");
    assert_eq!(v.source.tran_args, vec!["0","5","1m","1u","1u","4m","10m"]);
}
```
(If the exact ngspice source syntax the parser accepts differs, adjust the input to what `parse_netlist` accepts with zero errors — verify with a scratch dump first.)

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** For `ast::Voltage`/`Current` → kind `VSource`/`ISource`, `nodes=[pos,neg]`. Build `SpiceSource` by iterating `sources()`: for a `DCSource` node → `dc = value().text()`; `ACSource` → `ac_mag`/`ac_phase` from `magnitude()`/`phase()`; `TranSource` → `tran_kind = tok_text(function())`, `tran_args = values().map(|e| e.text()).collect()`. Cast each `SyntaxNode` via `ast::DCSource::cast` etc.
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** — `feat(netlist-cxx): SPICE source (V/I DC/AC/tran) projection`.

---

### Task 3: Rust — SPICE semiconductor + model + subckt projection (D/M/Q, .model, .subckt, X) [S1c]

**Files:** `crates/netlist-cxx/src/lib.rs`. Test: same.

**Interfaces:** Consumes S1 schema. Produces: projection for `Diode`, `MOSFET`, `BipolarTransistor`, `SubcktCall`, and `project_spice_model`, `project_spice_subckt` (recursive → `SpiceSubckt`).

- [ ] **Step 1: Failing tests** (one netlist exercising each):
```rust
#[test]
fn projects_spice_semiconductors_and_subckt() {
    let src = "* t\n\
        D1 a k dmod\n\
        M1 d g s b nch w=1u l=0.1u\n\
        Q1 c b e qmod\n\
        X1 in out amp gain=2\n\
        .model dmod d is=1e-14\n\
        .subckt amp in out\n R1 in out 1k\n .ends\n";
    let nl = super::parse_netlist(src, true);
    let b = &nl.spice_blocks[0];
    let m = b.devices.iter().find(|x| x.name == "M1").unwrap();
    assert_eq!(m.kind, super::ffi::SpiceDeviceKind::Mosfet);
    assert_eq!(m.nodes, vec!["d","g","s","b"]);
    assert_eq!(m.model, "nch");
    let d = b.devices.iter().find(|x| x.name == "D1").unwrap();
    assert_eq!(d.model, "dmod");
    let x = b.devices.iter().find(|x| x.name == "X1").unwrap();
    assert_eq!(x.kind, super::ffi::SpiceDeviceKind::SubcktCall);
    assert_eq!(x.model, "amp");           // master (last positional node)
    assert_eq!(b.models.len(), 1);
    assert_eq!(b.models[0].name, "dmod");
    assert_eq!(b.models[0].model_type, "d");
    assert_eq!(b.subckts.len(), 1);
    assert_eq!(b.subckts[0].name, "amp");
}
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.**
  - `Diode` → kind `Diode`, `nodes=[pos,neg]`, `model=hier_text(model())`, `params`.
  - `MOSFET` → kind `Mosfet`, `nodes=[drain,gate,source,bulk]`, `model=hier_text(model())`, `params`.
  - `BipolarTransistor` → kind `Bjt`; collect `nodes()` into a Vec; the LAST positional node that isn't a connection is the model — mirror Julia `sema_nets(::BipolarTransistor)` (c,b,e[,s]) + model. Concretely: `let ns: Vec<_> = q.nodes().map(|n| hier_text(&n)).collect();` treat the final entry as `model` if there are >3 (i.e. 4th/5th could be substrate vs model — follow ngspice: `Q c b e [s] model`); set `nodes = ns[..ns.len()-1]`, `model = ns[last]`. Document the heuristic; a follow-up can refine substrate handling.
  - `SubcktCall` → kind `SubcktCall`; `nodes()` positional with master LAST: `model = last`, `nodes = rest`, `params`.
  - `.model` (`ast::Model`) → `block.models.push(SpiceModel { name: hier_text(name()), model_type: tok_text(model_type()), level: "" (extract from params if present), params })`.
  - `.subckt` (`ast::Subckt`) → `block.subckts.push(project_spice_subckt(&s))`; `project_spice_subckt` recurses over `body()` reusing the same per-kind device/model/subckt logic (factor the device-matching into a helper `project_spice_device(node) -> Option<SpiceDevice>` used by both the block and subckt walkers).
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** — `feat(netlist-cxx): SPICE semiconductor/model/subckt projection`.

---

### Task 4: Rust — SPICE controlled sources + remaining kinds + includes [S1d]

**Files:** `crates/netlist-cxx/src/lib.rs`. Test: same.

**Interfaces:** Consumes S1 schema. Produces: projection for `ControlledSource` (E/F/G/H), `MutualInductor`, `Switch`, `Behavioral`, JFET, OSDI devices, and SPICE `.include`/`.lib` → `SpiceBlock.includes`.

- [ ] **Step 1: Failing test.**
```rust
#[test]
fn projects_controlled_sources_and_include() {
    let src = "* t\n\
        E1 out 0 in 0 2.0\n\
        G1 o 0 a b 1e-3\n\
        F1 o 0 vsense 10\n\
        .include \"models.spice\"\n";
    let nl = super::parse_netlist(src, true);
    let b = &nl.spice_blocks[0];
    let e = b.devices.iter().find(|x| x.name == "E1").unwrap();
    assert_eq!(e.kind, super::ffi::SpiceDeviceKind::Vcvs);
    assert_eq!(e.nodes, vec!["out","0"]);
    assert_eq!(e.ctrl_nodes, vec!["in","0"]);
    assert_eq!(e.ctrl_value, "2.0");
    assert_eq!(b.includes.len(), 1);
    assert_eq!(b.includes[0].path, "models.spice");
}
```
(Kind for E/F/G/H: E=Vcvs, G=Vccs, F=Cccs, H=Ccvs. Determine which from the `ControlledSource`'s control node type + the SPICE prefix — see `ast.rs`/parser: voltage-controlled → VoltageControl (E/G), current-controlled → CurrentControl (F/H). If the AST doesn't distinguish E-vs-G / F-vs-H directly, derive from the device-name prefix letter via `name()[0]`. Verify against a scratch dump and set the kind accordingly.)

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.**
  - `ControlledSource` → `nodes=[pos,neg]`; inspect `control()`: `VoltageControl` → `ctrl_nodes = control_nodes()`, `ctrl_value = value().map(text)`; `CurrentControl` → `ctrl_nodes = [vnam()]` (controlling source name), `ctrl_value = value().map(text)`. Kind from name-prefix letter (E/F/G/H → Vcvs/Cccs/Vccs/Ccvs). (POLY/TABLE control → capture args into `params`/`ctrl_value` best-effort; note if deferred.)
  - `MutualInductor`/`Switch`/`Behavioral`/JFET → kind `MutualInductor`/`Switch`/`Behavioral`/`Jfet`; populate name/nodes/params structurally (no special handling — adapter warn-skips).
  - OSDI devices (N/Y) → kind `Osdi`, model + nodes + params.
  - SPICE `.include`/`.lib` nodes → `block.includes.push(Include { path: unquote(...), section: ... })` (reuse the Spectre `Include` projection shape).
- [ ] **Step 4: Run tests → PASS. Also add a `projects_full_breadth` smoke** parsing one netlist with every kind, asserting `devices.len()` and no `errors`.
- [ ] **Step 5: Commit** — `feat(netlist-cxx): SPICE controlled sources + remaining kinds + includes`.

---

### Task 5: VACASK adapter — passives + sources -> ParserTables + RC/PULSE E2E vs ngspice [S2]

**Files:** Modify `../VACASK/lib/netlistrs.cpp`, `include/netlistrs.h`. Create `../VACASK/demo/api/spice_rc.cir`. Modify `demo/api/CMakeLists.txt`. Create a compare harness `../VACASK/demo/api/ngspice_compare.sh` (or reuse an existing pattern).

**Interfaces:** Consumes `netlist::SpiceBlock` (S1). Produces: `bool spiceBlockToTables(const netlist::SpiceBlock&, PTSubcircuitDefinition& into, ParserTables&, Parser&, Status&)`, called from `mergeNetlist` for each `nl.spice_blocks` (and each `Subckt.spice_blocks`). Helper `std::string spiceSourceParams(const netlist::SpiceSource&)`.

- [ ] **Step 1: Failing test — build the tables + E2E.** Create `demo/api/spice_rc.cir`:
```
* RC low-pass, SPICE dialect
V1 1 0 DC 0 PULSE(0 5 1m 1u 1u 4m 10m)
R1 1 2 1k
C1 2 0 2u
.tran 1u 10m
.end
```
Add a CTest `demo_netlistrs_spice_rc` (COMMAND `demo_netlistrs .../spice_rc.cir <VACASK_MOD_DIR>`, WORKING_DIRECTORY /tmp).

- [ ] **Step 2: Run → FAIL** (adapter ignores `spice_blocks`; no `R1`/`C1`/`V1` instances → verify/elaborate error).

- [ ] **Step 3: Implement `spiceBlockToTables` for passives + sources; wire into `mergeNetlist`.**
```cpp
static std::string spiceSourceParams(const netlist::SpiceSource& s) {
    std::ostringstream os; bool first = true;
    auto add = [&](const std::string& kv){ if(!kv.empty()){ if(!first) os<<" "; os<<kv; first=false; } };
    if (!std::string(s.dc).empty())      add("dc=" + std::string(s.dc));
    if (!std::string(s.ac_mag).empty())  add("mag=" + std::string(s.ac_mag));
    if (!std::string(s.ac_phase).empty())add("phase=" + std::string(s.ac_phase));
    std::string tk = s.tran_kind;        // pulse/sine/pwl/...
    if (!tk.empty()) {
        std::transform(tk.begin(), tk.end(), tk.begin(), ::tolower);
        add("type=\"" + (tk=="sin"?std::string("sine"):tk) + "\"");
        // positional tran args -> VACASK named params, per function:
        static const std::map<std::string, std::vector<const char*>> ARGS = {
            {"pulse", {"val0","val1","delay","rise","fall","width","period"}},
            {"sine",  {"sinedc","ampl","freq","delay","damp"}},
            {"pwl",   {}},   // handled specially below (t/v pairs)
            {"exp",   {"val0","val1","td1","tau1","td2","tau2"}},
        };
        // map s.tran_args[i] -> ARGS[tk][i]; for pwl emit "wave=[t0 v0 t1 v1 ...]".
    }
    return os.str();
}
```
For each `SpiceDevice` (switch on `kind`):
- `Resistor`/`Capacitor`/`Inductor`: `PTInstance(name, "resistor"|"capacitor"|"inductor", terms)`; if `value` non-empty add `p.parseParameters(("r"|"c"|"l") + "=" + value)`; add any `params`. If `value` empty but `model` set → master = model (model-based passive).
- `VSource`/`ISource`: master `vsource`/`isource`; `p.parseParameters(spiceSourceParams(dev.source) + extra params)`.
Terminals: `nodeList(dev.nodes)` (reuse the existing helper). Params: reuse the existing `paramString(dev.params)` + `parseParameters`.
In `mergeNetlist`, after the existing Spectre handling, add: `for (const auto& sb : nl.spice_blocks) if(!spiceBlockToTables(sb, top, tab, p, s)) return false;` (and likewise inside subckt filling for `Subckt.spice_blocks`).

- [ ] **Step 4: Build + run the CTest** → dumps tables/hierarchy, `Analysis OK`, exit 0.

- [ ] **Step 5: E2E vs ngspice.** Write `ngspice_compare.sh <cir> <vacask_out_raw>`: run `ngspice -b -r ref.raw <cir>`, extract node 2 vs time, compare against VACASK's saved output within tol (e.g. 2% + 1mV). Assert RC τ=2ms. Wire a CTest `demo_netlistrs_spice_rc_ngspice` that runs the comparison (skip-and-warn if `ngspice` not found, so CI without ngspice still builds). Report the compared numbers.

- [ ] **Step 6: Commit** (VACASK) — `feat: SPICE passives+sources adapter + RC E2E vs ngspice`.

---

### Task 6: VACASK adapter — semiconductors (D/M/Q) + model dispatch + E2E [S3]

**Files:** `../VACASK/lib/netlistrs.cpp`. Create `demo/api/spice_diode.cir`, `demo/api/spice_mos.cir`. Modify `demo/api/CMakeLists.txt`.

**Interfaces:** Consumes S1c projections + S2 helpers. Produces: `spiceModelMaster(model_type, level, version) -> std::string`; per-kind mapping for Diode/Mosfet/Bjt in `spiceBlockToTables`.

- [ ] **Step 1: Failing tests.** `spice_diode.cir` — half-wave rectifier (`V PULSE`/`SIN` → `D` → `R` load, `.model d`). `spice_mos.cir` — resistor-load NMOS stage (`.model nch nmos level=…` matching an available BSIM). CTests for each.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.**
  - `spiceModelMaster`: map SPICE model_type + level/version → VACASK OSDI master. Diode `d`→`diode`. MOSFET `nmos`/`pmos` by level: 1/2/3→(nearest available, e.g. `bsim3v3` as fallback), 49/53→`bsim3v3`, 54→`bsim4v8`, 70/72→`bsimbulk106`, 103→`psp103v4`; type param (`nmos`/`pmos`) passed through as a model param. BJT `npn`/`pnp`→`vbic` (type as param). Emit a `Simulator::err()` warning + pick the closest master when an exact level isn't available; document the table.
  - Diode: `PTInstance(name, "diode", [pos,neg])` + params; model card `PTModel(model, "diode")` from the matching `SpiceModel`.
  - MOSFET: `PTInstance(name, spiceModelMaster(...), [drain,gate,source,bulk])` + params; `PTModel(model, master)` with the `.model` params.
  - BJT: `PTInstance(name, "vbic", [c,b,e,(s)])` + params; `PTModel(model, "vbic")`.
  - Model cards: build `PTModel`s from `block.models` mapped through `spiceModelMaster`; add before instances (VACASK resolves by name).
- [ ] **Step 4: Build + run CTests → `Analysis OK`.**
- [ ] **Step 5: E2E vs ngspice** for diode rectifier + MOS stage (node-voltage diff within tol; ngspice uses its own built-in device models, so compare qualitatively/within a looser tol and document expected divergence from model-card differences — assert diode clamps ~0.6-0.7V, MOS stage gain sign/DC operating point).
- [ ] **Step 6: Commit** (VACASK) — `feat: SPICE D/M/Q adapter + model dispatch + E2E`.

---

### Task 7: VACASK adapter — controlled sources + subckt (X/.subckt) + warn-skip + E2E [S4]

**Files:** `../VACASK/lib/netlistrs.cpp`. Create `demo/api/spice_subckt.cir`. Modify `demo/api/CMakeLists.txt`.

**Interfaces:** Consumes S1c/S1d projections. Produces: mapping for Vcvs/Vccs/Ccvs/Cccs, SubcktCall, recursive SPICE `.subckt` → `PTSubcircuitDefinition`; warn-skip for Switch/Behavioral/Jfet/MutualInductor.

- [ ] **Step 1: Failing test.** `spice_subckt.cir` — a `.subckt` (RC block) instantiated via `X` with a param override, driven by a source; `.tran`. CTest.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.**
  - Controlled sources: `E`→`vcvs` with control nodes as extra terminals + gain param; `G`→`vccs`; `F`→`cccs` (controlling source name); `H`→`ccvs`. Map `ctrl_nodes`/`ctrl_value` to VACASK's vcvs/etc. param/terminal convention (check `lib/devbuiltins.cpp` for the exact terminal/param names; adjust).
  - `SubcktCall` → `PTInstance(name, dev.model /*master=subckt name*/, dev.nodes)` + params.
  - SPICE `.subckt` (`SpiceSubckt`) → `PTSubcircuitDefinition(name, ports)`; recurse: fill its params/models/devices/subckts via the same per-kind mapping (factor a `spiceSubcktToDef`); `into.add(std::move(def))`.
  - Warn-skip: `Switch`/`Behavioral`/`Jfet`/`MutualInductor` → `Simulator::err() << "WARNING: SPICE device <name> (<kind>) has no VACASK equivalent; skipped\n";` continue.
- [ ] **Step 4: Build + run CTest → `Analysis OK`; assert the subckt override took effect** (via `cir.instanceParameter` like milestone-1 Task 6, require found).
- [ ] **Step 5: E2E vs ngspice** for the subckt netlist.
- [ ] **Step 6: Commit** (VACASK) — `feat: SPICE controlled sources + subckt adapter + warn-skip + E2E`.

---

## Self-Review

**Spec coverage:** SpiceBlock schema (S1) ✅; full-breadth projection R/C/L (S1), V/I (S1b), D/M/Q/model/subckt/X (S1c), E/F/G/H + K/switch/B/JFET/OSDI + includes (S1d) ✅; adapter passives+sources (S2), semiconductors+model dispatch (S3), controlled+subckt+warn-skip (S4) ✅; E2E vs ngspice for RC/diode/MOS/subckt ✅; warn-skip for the 4 unsupported kinds ✅ (S1d projects, S4 warn-skips).

**Placeholder scan:** Deliberate deferrals flagged for scratch-verification against the live parser (never silent): exact ngspice source syntax accepted (S1b Step 1), E/G vs F/H kind derivation (S1d Step 1), BJT substrate-vs-model node heuristic (S1c), MOSFET level→master table + vcvs terminal/param names (S3/S4 — verify against `devbuiltins.cpp`). Each names the file/scratch check. Exemplar code is complete for S1 and S2; later tasks give exact accessors + the specific mapping deltas.

**Type consistency:** `SpiceDeviceKind`/`SpiceDevice`/`SpiceSource`/`SpiceBlock` field names identical across S1→S4; `spiceBlockToTables`/`spiceSourceParams`/`spiceModelMaster` signatures consistent S2→S4; reuses milestone-1 helpers (`nodeList`, `paramString`, `parseParameters`, `mergeNetlist`).

## Verification (milestone-level)

`.cir` netlists for RC+PULSE, diode rectifier, MOSFET stage, and X-subckt each: VACASK `demo_netlistrs` → `Analysis OK`; then diffed against ngspice-43 on the same file within tolerance (documented per case). Per-`SpiceDeviceKind` Rust projection unit tests. Warn-skip diagnostic asserted for a Switch/`B` device.
