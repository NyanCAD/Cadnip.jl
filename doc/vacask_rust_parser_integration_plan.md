# VACASK ← Rust Netlist Parser Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed real SPICE/Spectre netlists into VACASK through the `netlist-cxx` Rust parser — parse → build `ParserTables` → `Circuit::elaborate` → run an analysis — with no Julia runtime.

**Architecture:** The `netlist-cxx` crate (Rust, [cxx](https://cxx.rs)) already exposes an eager, owned POD projection of the Spectre typed AST as `netlist::Netlist`. This plan (1) adds start-language dispatch to that crate so `.cir` files open in SPICE and `.scs` in Spectre (mirroring Julia's `parse_spectre_with(src, StartLang)`); (2) wires the crate into VACASK's CMake via **Corrosion** + `corrosion_add_cxxbridge`; (3) adds a C++ adapter (`netlistrs.{h,cpp}`) that walks `netlist::Netlist` into VACASK's `ParserTables` using the fluent `PT*` API, routing every parameter through `Parser::parseParameters` so VACASK itself classifies value-vs-expression; (4) resolves `.include`/`.lib`/section in the **C++ adapter** (clean separation: Rust parses a single in-memory source string; C++ owns the filesystem and the splicing); (5) proves it end-to-end with a driver that elaborates and runs a `tran`, comparing against a reference.

**Tech Stack:** Rust (cxx 1.0, rowan), C++20, CMake ≥ 3.18, Corrosion, `cxxbridge-cmd` CLI, VACASK (`simlib`/`sim`), Julia (reference oracle: Cadnip).

## Global Constraints

- **The Rust side does NO scope resolution, NO subckt flattening, NO parameter evaluation.** VACASK resolves all of that in `Circuit::elaborate()`. The parser owes only an un-flattened, symbolic structural view. (Verified in the scoping doc `~/.claude/plans/now-that-we-have-stateless-stallman.md` against `lib/circuit.cpp`/`hierdevice.cpp`/`rpneval.cpp`/`context.cpp`.)
- **Parameter/expression values cross the FFI as verbatim source text.** The C++ adapter re-parses them with VACASK's own `Parser::parseExpression`/`parseParameters` — never a bespoke Rust expression evaluator.
- **Include resolution and ALL filesystem access live in C++, not Rust.** The Rust crate only parses a single in-memory source string into a `Netlist`. The C++ adapter reads included files, re-parses each with `parse_netlist`, and splices the members at the include site (recursive, cycle-guarded, relative to the including file's directory). This keeps the Rust/C++ boundary clean.
- **File-entry dispatch by extension:** `.sim` → VACASK's own native flex/bison parser (`Parser::parseNetlistFile`, unchanged existing path); `.scs` → Rust parser starting in Spectre; **everything else → Rust parser starting in SPICE** (SPICE being the notoriously inconsistent catch-all). A `simulator lang=` line may still switch dialect mid-file within the Rust path.
- **No separate neutral-IR crate yet.** Transcribe `netlist::Netlist` → `ParserTables` directly in the adapter. Extract an IR only later when building the circulax elaborator (out of scope here).
- **Both dialects flow through one parser.** A Spectre netlist may switch to SPICE mid-file via `simulator lang=spice`; the only per-file difference is the *starting* language (see the dialect rule above).
- Two repos: the Rust crate lives in `Cadnip.jl/netlist-parser-rs/` (branch `rust-parser-spike`); VACASK lives in `../VACASK` (its own git repo — commit there separately, on a feature branch, per its CLAUDE.md).
- VACASK binary lives at `../build.VACASK/Release/simulator/vacask`; the demo/test executables build under the CMake build dir. Run VACASK with `/tmp` as CWD (VACASK CLAUDE.md).
- Prereq on PATH: `cargo` (`/usr/bin/cargo`) and `cxxbridge` (`~/.cargo/bin/cxxbridge`) — both confirmed present.

## File Structure

**Rust (`Cadnip.jl/netlist-parser-rs/`):**
- Modify `crates/netlist-cxx/src/lib.rs` — add a `start_spice: bool` parameter to the bridge entry (dispatch to `parse_spectre_with`). No filesystem, no include logic on this side.
- Reference only: `crates/netlist-syntax/src/lib.rs` (`parse_spectre_with`, `StartLang`).

**C++ (`../VACASK/`):**
- Create `include/netlistrs.h` — adapter API: `buildParserTables(...)` (from a source string) and `buildParserTablesFromFile(...)` (reads a file, resolves includes).
- Create `lib/netlistrs.cpp` — the walker + the C++ include resolver.
- Reference only (for the C++ include/section semantics): `Cadnip.jl/src/spc/sema.jl` (`sema_include!`, `get_section!`, most-recent-include-wins, self-include handling).
- Modify `lib/CMakeLists.txt` — add `netlistrs.cpp`/`netlistrs.h` to `simlib`; link the cxxbridge target.
- Modify top-level `CMakeLists.txt` — FetchContent Corrosion; `corrosion_import_crate` + `corrosion_add_cxxbridge`.
- Create `demo/api/demo_netlistrs.cpp` — end-to-end driver (mirrors `demo/api/demo1.cpp`).
- Modify `demo/api/CMakeLists.txt` — add the demo executable + a CTest.
- Create `demo/api/rc.scs` + `demo/api/rc_nested.scs` — test netlists.

## Interfaces (facts pinned from the source)

**Rust bridge (current, `crates/netlist-cxx/src/lib.rs`):**
```rust
fn parse_spectre_netlist(src: &str) -> Netlist;   // starts in Spectre
```
`netlist_syntax` exports: `parse_spectre(src) -> SyntaxNode`, `parse_spectre_with(src, StartLang) -> SyntaxNode`, `pub use spectre_parser::StartLang;` (`StartLang::Spice` / `StartLang::Spectre`). Root is always `SpectreNetlistSource`.

**`Netlist` POD schema (bridge structs):** `Netlist{ params, models, subckts, instances, analyses, saves, ics, globals, includes, ahdl_includes, errors }`; `Instance{ name, master, nodes: Vec<String>, params: Vec<Param> }`; `Model{ name, master, params }`; `Param{ name, value }` (value = verbatim text); `Subckt{ name, is_inline, ports, params, models, instances, subckts, conditionals }`; `Conditional{ clauses: Vec<CondClause> }`; `CondClause{ condition, instance }` (condition empty ⇒ trailing `else`); `Analysis{ name, analysis_type, nodes, params }`; `SaveItem{ signal, modifier }`; `IcItem{ node, value }`; `Include{ path, section }`; `ParseError{ start, end }`.

**VACASK PT API (from `include/parseroutput.h`, `include/parser.h`, template `demo/api/demo1.cpp`):**
- `ParserTables tab(title);` → fluent `.add(PTLoad&&)`, `.defaultGround()`, `.setDefaultSubDef(PTSubcircuitDefinition&&)`, `.addGlobal(PTParsedIdentifier)`, `.addCommand(PTAnalysis&&)`, `.verify(Status&) -> bool`, `.dump(int, ostream&)`.
- `Parser p(tab);` → `Rpn parseExpression(std::string)` (throws), `PTParameters parseParameters(std::string)` (throws) — the latter returns a `PTParameters` already split into value/expression buckets.
- `PTSubcircuitDefinition()` / `PTSubcircuitDefinition(Id name, PTIdentifierList&& terms)`; fluent `.add(PTParameters&&)`, `.add(PTModel&&)`, `.add(PTInstance&&)`, `.add(PTBlockSequence&&)`, `.add(PTSubcircuitDefinition&&)`.
- `PTModel(Id name, Id device)`; `.add(PTParameters&&)`.
- `PTInstance(Id name, Id master, PTIdentifierList&& terms)`; `.add(PTParameters&&)`. `PTIdentifierList` = `std::vector<PTParsedIdentifier>`; `PTParsedIdentifier(const char* name)`.
- `PTBlockSequence()`; `.add(Rpn&& cond, PTBlock&& block)`. `PTBlock()`; `.add(PTInstance&&)` / `.add(PTModel&&)`.
- `PTAnalysis(Id name, Id typeName)`; parameters via `.parameters().add(...)` or build a `PTParameters` and merge.
- `Id` constructs from `const char*` / `std::string` (interned identifier). `PV = PTParameterValue`, `PE = PTParameterExpression`.
- Downstream: `OpenvafCompiler comp; Circuit cir(tab, &comp, s); cir.elaborate({}, "__topdef__", "__topinst__", nullptr, s); auto a = Analysis::create(desc, cir, s); a->add(PTSave(...)); a->run(s);`

**Adapter param strategy (decided):** the adapter does **not** guess value-vs-expression. For each `PT*` that carries params, it builds one Spectre-style parameter string `name=value name2=value2 …` from the projected `Param`s and calls `p.parseParameters(str)`, then `.add(std::move(thatPTParameters))`. This delegates classification + expression compilation to VACASK, satisfying the "verbatim text, re-parsed by VACASK" constraint.

**Loads (`PTLoad` = OSDI module):** out of scope for the adapter in milestone 1. The **driver** adds the `PTLoad`s it needs (as `demo1.cpp` does), and the test netlists use those masters. Deriving loads from netlist directives is milestone 2.

---

### Task 1: Rust — start-language dispatch in the bridge

**Files:**
- Modify: `Cadnip.jl/netlist-parser-rs/crates/netlist-cxx/src/lib.rs`
- Test: same file (`#[cfg(test)] mod tests`)

**SCOPE — pure dispatch only.** This task adds ONLY the `start_spice` routing. Do **not** project SPICE devices. Do **not** touch `collect_scope`. A `simulator lang=spice` region parses into an embedded `SPICENetlistSource` subtree that `collect_scope` already leaves unprojected (`_ => {}`) — that is correct here; SPICE-block projection is a separate task (Task 7), which will embed a distinct `SpiceBlock` (mirroring Julia's `SP.*` form family) rather than casting SPICE devices into the Spectre `Instance` struct. Casting SPICE devices into `Instance` is explicitly forbidden.

**Interfaces:**
- Consumes: `netlist_syntax::{parse_spectre_with, StartLang}`.
- Produces: bridge fn `fn parse_netlist(src: &str, start_spice: bool) -> Netlist`. (Keep `parse_spectre_netlist` as a thin wrapper so existing Rust tests/PyO3/demo keep compiling.)

- [ ] **Step 1: Write the failing test** — add to the `tests` module in `crates/netlist-cxx/src/lib.rs`. This asserts the start-language flag actually routes to the right dialect, without relying on any (deferred) SPICE-device projection. The source is a SPICE block followed by a `simulator lang=spectre` switch and a trailing Spectre instance `r2` (which the *existing* Spectre projection handles). Empirically verified: starting in SPICE the whole file parses with 0 errors and `r2` projects; starting in Spectre the leading SPICE line is a parse error.

```rust
#[test]
fn start_language_dispatch() {
    // SPICE block, then switch to Spectre, then a Spectre instance.
    let src = "* title\nR1 a b 1k\nsimulator lang=spectre\nr2 (a b) resistor r=2k\n";

    // Start in SPICE: the leading SPICE block parses cleanly; after the
    // `simulator lang=spectre` switch, control returns to the Spectre driver
    // and the trailing Spectre instance r2 projects. (SPICE-device projection
    // of the leading block is deferred to Task 7 — not asserted here.)
    let spice = super::parse_netlist(src, /*start_spice=*/ true);
    assert!(spice.errors.is_empty(), "spice-start should parse cleanly");
    assert_eq!(spice.instances.len(), 1);
    assert_eq!(spice.instances[0].name, "r2");

    // Start in Spectre: the same leading SPICE line `R1 a b 1k` is invalid
    // Spectre and produces error node(s).
    let spectre = super::parse_netlist(src, /*start_spice=*/ false);
    assert!(!spectre.errors.is_empty(), "spectre-start should error on the SPICE line");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Cadnip.jl/netlist-parser-rs && cargo test -p netlist-cxx start_language_dispatch`
Expected: FAIL — `cannot find function 'parse_netlist' in module 'super'`.

- [ ] **Step 3: Write minimal implementation** — in the `#[cxx::bridge]` `extern "Rust"` block, replace the entry declaration:

```rust
    extern "Rust" {
        /// Parse netlist source and project it into a flat, owned `Netlist`.
        /// `start_spice` selects the starting dialect (`.cir` → true / SPICE,
        /// `.scs` → false / Spectre); a `simulator lang=` line may still switch
        /// mid-file.
        fn parse_netlist(src: &str, start_spice: bool) -> Netlist;
        /// Back-compat: parse starting in the Spectre dialect. MUST stay in the
        /// bridge — the cxx demo (`demo/smoke.cpp`) and other C++ consumers call
        /// `netlist::parse_spectre_netlist` through the generated shim.
        fn parse_spectre_netlist(src: &str) -> Netlist;
    }
```

Replace the free function `parse_spectre_netlist` body with a generalized core plus a wrapper:

```rust
pub fn parse_netlist(src: &str, start_spice: bool) -> ffi::Netlist {
    let lang = if start_spice { StartLang::Spice } else { StartLang::Spectre };
    let root = parse_spectre_with(src, lang);
    let errors = collect_errors(&root);
    let source = sast::SpectreNetlistSource::cast(root).expect("root is SpectreNetlistSource");
    let scope = collect_scope(source.statements());
    ffi::Netlist {
        params: scope.params,
        models: scope.models,
        subckts: scope.subckts,
        instances: scope.instances,
        analyses: scope.analyses,
        saves: scope.saves,
        ics: scope.ics,
        globals: scope.globals,
        includes: scope.includes,
        ahdl_includes: scope.ahdl_includes,
        errors,
    }
}

/// Back-compat wrapper: parse starting in Spectre.
pub fn parse_spectre_netlist(src: &str) -> ffi::Netlist {
    parse_netlist(src, false)
}
```

Update the `use` line at the top of the file to import the dispatch entry points:

```rust
use netlist_syntax::{parse_spectre_with, StartLang, SyntaxKind, SyntaxNode, SyntaxToken};
```

(Remove the now-unused `parse_spectre` import if the compiler warns.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Cadnip.jl/netlist-parser-rs && cargo test -p netlist-cxx`
Expected: PASS — all existing tests (`projects_top_level`, `projects_nested_subckt_and_conditional`, `collects_include_and_errors`) plus `start_language_dispatch`.

- [ ] **Step 5: Commit** (in `Cadnip.jl`)

```bash
git add netlist-parser-rs/crates/netlist-cxx/src/lib.rs
git commit -m "feat(netlist-cxx): start-language dispatch (parse_netlist)"
```

---

### Task 2: VACASK — Corrosion build wiring, link a trivial call

**SCOPE — bridge only, do NOT touch `simlib`.** This task proves the Rust parser links into a C++ executable through Corrosion. It links the trivial demo against ONLY `netlist_cxx_bridge` (not `simlib`), so the build stays fast (cargo builds the crate + cxxbridge codegen + one small .cpp — no VACASK library build). Wiring the bridge into `simlib` and registering `netlistrs.cpp` is **Task 3** (where those files exist); doing it here would break the `simlib` build (missing `netlistrs.cpp`) and force a full VACASK compile.

**Files:**
- Modify: `../VACASK/CMakeLists.txt` (add Corrosion + cxxbridge target)
- Create: `../VACASK/demo/api/demo_netlistrs.cpp` (minimal; expanded in Tasks 3-4)
- Modify: `../VACASK/demo/api/CMakeLists.txt`

**Interfaces:**
- Consumes: the cxx bridge header generated from `crates/netlist-cxx/src/lib.rs` — include path `netlist_cxx_bridge/lib.h`, C++ namespace `netlist`, symbol `netlist::parse_netlist(rust::Str, bool)` (both `parse_netlist` and `parse_spectre_netlist` are exported — Task 1 done).
- Produces: CMake target `netlist_cxx_bridge` (static lib + generated header include dir), later linked by `simlib` in Task 3.

**Environment note:** if VACASK's top-level `cmake` *configure* fails for a reason unrelated to Corrosion (missing Boost/KLU/flex/bison/Python/etc. that VACASK itself needs), report **BLOCKED** with the exact CMake error — do not try to patch VACASK's unrelated dependencies. The Corrosion/bridge portion is what this task owns.

- [ ] **Step 1: Add Corrosion + bridge target to `../VACASK/CMakeLists.txt`.** After `enable_testing()` near the top, add:

```cmake
# --- Rust netlist parser (netlist-cxx via Corrosion) ------------------------
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.1
)
FetchContent_MakeAvailable(Corrosion)

set(NETLIST_RS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Cadnip.jl/netlist-parser-rs")
corrosion_import_crate(MANIFEST_PATH "${NETLIST_RS_DIR}/crates/netlist-cxx/Cargo.toml")
corrosion_add_cxxbridge(netlist_cxx_bridge
    CRATE netlist_cxx
    FILES lib.rs
)
```

(`corrosion_add_cxxbridge` compiles the generated shim, links the `netlist_cxx` static lib, and exposes headers under `#include "netlist_cxx_bridge/lib.h"`.)

- [ ] **Step 2: Create a minimal driver to force a link.** `../VACASK/demo/api/demo_netlistrs.cpp`. NOTE: SPICE-device projection is deferred (Task 7), so a bare SPICE line would project 0 instances — assert on a **Spectre** netlist instead, which the current projection handles:

```cpp
#include "netlist_cxx_bridge/lib.h"
#include <iostream>

int main() {
    netlist::Netlist nl =
        netlist::parse_netlist(rust::Str("simulator lang=spectre\nr1 (a b) resistor r=1k\n"), false);
    std::cout << "instances: " << nl.instances.size() << "\n";
    return (nl.instances.size() == 1 && nl.errors.empty()) ? 0 : 1;
}
```

Add to `../VACASK/demo/api/CMakeLists.txt` (mirror the existing demo targets, but link ONLY the bridge — not `simlib`):

```cmake
add_executable(demo_netlistrs demo_netlistrs.cpp)
target_link_libraries(demo_netlistrs PRIVATE netlist_cxx_bridge)
```

- [ ] **Step 3: Configure + build + run.**

Run:
```bash
cmake -S ../VACASK -B ../build.VACASK/Release -DCMAKE_BUILD_TYPE=Release
cmake --build ../build.VACASK/Release --target demo_netlistrs
(cd /tmp && /home/pepijn/code/nyanodide/build.VACASK/Release/demo/api/demo_netlistrs)
```
Expected: configures (Corrosion fetched, `cargo build -p netlist-cxx` runs, cxxbridge codegen), the `demo_netlistrs` target compiles + links against the bridge only (no `simlib`/VACASK build), prints `instances: 1`, exit 0.

- [ ] **Step 4: Commit** (in `../VACASK`)

```bash
cd ../VACASK && git add CMakeLists.txt demo/api/demo_netlistrs.cpp demo/api/CMakeLists.txt && git commit -m "build: link Rust netlist-cxx parser via Corrosion" && cd -
```

---

### Task 3: VACASK — `netlist::Netlist` → `ParserTables` adapter

**Files:**
- Create: `../VACASK/include/netlistrs.h`
- Create: `../VACASK/lib/netlistrs.cpp`
- Modify: `../VACASK/lib/CMakeLists.txt` (register `netlistrs.cpp`/`.h` in `simlib`; link the bridge into `simlib`)
- Modify: `../VACASK/demo/api/CMakeLists.txt` (link `demo_netlistrs` against `simlib` too, now that it uses `ParserTables`)
- Test: extend `../VACASK/demo/api/demo_netlistrs.cpp` to assert on a built `ParserTables` (via `tab.verify` + `tab.dump`).

**Build wiring (Task 2 deferred this here — `netlistrs.cpp` now exists):**
- In `../VACASK/lib/CMakeLists.txt`, add `${CMAKE_CURRENT_SOURCE_DIR}/../include/netlistrs.h` and `netlistrs.cpp` to the `add_library(simlib STATIC ...)` source list, and after `target_include_directories(simlib PUBLIC ...)` add `target_link_libraries(simlib PUBLIC netlist_cxx_bridge)`.
- In `../VACASK/demo/api/CMakeLists.txt`, change the demo link line to `target_link_libraries(demo_netlistrs PRIVATE simlib netlist_cxx_bridge)`.
- This is the first time `simlib` (and thus much of VACASK) gets compiled — expect a longer build. If VACASK's own sources fail to compile for environment reasons unrelated to `netlistrs.cpp`, report BLOCKED with the exact error.

**Interfaces:**
- Consumes: `netlist::parse_netlist` (Task 1/2); `Parser`, `ParserTables`, `PT*` (VACASK).
- Produces:
  ```cpp
  namespace NAMESPACE {
      // Build tab's default toplevel subdef + subckt defs + analyses from source.
      // Returns false and sets s on parse error. Loads are the caller's responsibility.
      bool buildParserTables(const std::string& source, bool startSpice,
                             ParserTables& tab, Parser& p, Status& s);
  }
  ```

- [ ] **Step 1: Write the failing test** — replace `demo/api/demo_netlistrs.cpp`'s body with a `ParserTables`-building assertion:

```cpp
#include "netlistrs.h"
#include "parser.h"
#include <iostream>

using namespace sim;

int main() {
    const char* src =
        "simulator lang=spectre\n"
        "parameters c0=1u\n"
        "r1 (1 2) res r=1k\n"
        "c1 (2 0) cap c=2*c0\n"
        "model res resistor\n"
        "model cap capacitor\n";
    ParserTables tab("rs smoke");
    Parser p(tab);
    Status s;
    if (!buildParserTables(src, /*startSpice=*/false, tab, p, s)) {
        std::cerr << "adapter failed: " << s.message() << "\n";
        return 1;
    }
    if (!tab.verify(s)) { std::cerr << "verify failed: " << s.message() << "\n"; return 1; }
    tab.dump(0, std::cout);
    return 0;
}
```

(Adjust the `sim::` namespace token to whatever `NAMESPACE` expands to — check the top of `include/common.h`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build ../build.VACASK/Release --target demo_netlistrs`
Expected: FAIL — `netlistrs.h: No such file` / `buildParserTables not declared`.

- [ ] **Step 3: Write the header** `../VACASK/include/netlistrs.h`:

```cpp
#ifndef __NETLISTRS_DEFINED
#define __NETLISTRS_DEFINED

#include <string>
#include "parser.h"
#include "parseroutput.h"
#include "status.h"
#include "common.h"

namespace NAMESPACE {

// Parse `source` with the Rust netlist-cxx parser and transcribe the result
// into `tab`'s default toplevel subcircuit definition, nested subckt defs, and
// analyses. Parameters/expressions are re-parsed with `p` (parseParameters), so
// VACASK owns value-vs-expression classification. `startSpice` selects the
// starting dialect (true = SPICE, false = Spectre). PTLoads are NOT added here.
// Returns false and populates `s` if the parser reported error nodes.
bool buildParserTables(const std::string& source, bool startSpice,
                       ParserTables& tab, Parser& p, Status& s);

}

#endif
```

- [ ] **Step 4: Write the implementation** `../VACASK/lib/netlistrs.cpp`:

```cpp
#include "netlistrs.h"
#include "netlist_cxx_bridge/lib.h"
#include <sstream>

namespace NAMESPACE {

namespace {

std::string sv(const rust::String& s) { return std::string(s); }

// "name=value name2=value2 …" from projected params, or "" if none.
std::string paramString(const rust::Vec<netlist::Param>& params) {
    std::ostringstream os;
    bool first = true;
    for (const auto& p : params) {
        if (!first) os << " ";
        os << sv(p.name) << "=" << sv(p.value);
        first = false;
    }
    return os.str();
}

PTIdentifierList nodeList(const rust::Vec<rust::String>& nodes) {
    PTIdentifierList terms;
    for (const auto& n : nodes) terms.push_back(PTParsedIdentifier(sv(n).c_str()));
    return terms;
}

PTInstance makeInstance(const netlist::Instance& i, Parser& p) {
    PTInstance inst(Id(sv(i.name).c_str()), Id(sv(i.master).c_str()), nodeList(i.nodes));
    auto ps = paramString(i.params);
    if (!ps.empty()) inst.add(p.parseParameters(ps));
    return inst;
}

PTModel makeModel(const netlist::Model& m, Parser& p) {
    PTModel mod(Id(sv(m.name).c_str()), Id(sv(m.master).c_str()));
    auto ps = paramString(m.params);
    if (!ps.empty()) mod.add(p.parseParameters(ps));
    return mod;
}

// if/else-if/else over instances → one PTBlockSequence.
PTBlockSequence makeConditional(const netlist::Conditional& c, Parser& p) {
    PTBlockSequence seq;
    for (const auto& cl : c.clauses) {
        PTBlock block;
        block.add(makeInstance(cl.instance, p));
        // Trailing else has empty condition → true (1).
        std::string cond = cl.condition.size() ? sv(cl.condition) : std::string("1");
        seq.add(p.parseExpression(cond), std::move(block));
    }
    return seq;
}

// Fill a subdef (toplevel or subckt) from a projected scope's members.
template <typename Scope>
void fillSubDef(PTSubcircuitDefinition& def, const Scope& s, Parser& p) {
    auto sp = paramString(s.params);
    if (!sp.empty()) def.add(p.parseParameters(sp));
    for (const auto& m : s.models)        def.add(makeModel(m, p));
    for (const auto& i : s.instances)     def.add(makeInstance(i, p));
    for (const auto& c : s.conditionals)  def.add(makeConditional(c, p));
    for (const auto& sub : s.subckts) {
        PTSubcircuitDefinition child(Id(sv(sub.name).c_str()), nodeList(sub.ports));
        fillSubDef(child, sub, p);
        def.add(std::move(child));
    }
}

} // namespace

bool buildParserTables(const std::string& source, bool startSpice,
                       ParserTables& tab, Parser& p, Status& s) {
    netlist::Netlist nl = netlist::parse_netlist(rust::Str(source), startSpice);
    if (!nl.errors.empty()) {
        std::ostringstream os;
        os << "netlist parse error(s): " << nl.errors.size()
           << " (first at bytes [" << nl.errors[0].start << ", " << nl.errors[0].end << "))";
        s.set(Status::Error, os.str());   // adjust to Status' actual setter (see status.h)
        return false;
    }

    tab.defaultGround();

    // Toplevel: global params + top-level models/instances/conditionals + nested subckts.
    PTSubcircuitDefinition top;
    fillSubDef(top, nl, p);            // nl exposes params/models/instances/conditionals/subckts
    tab.setDefaultSubDef(std::move(top));

    for (const auto& g : nl.globals) tab.addGlobal(PTParsedIdentifier(sv(g).c_str()));

    // Analyses → control block.
    for (const auto& a : nl.analyses) {
        PTAnalysis desc(Id(sv(a.name).c_str()), Id(sv(a.analysis_type).c_str()));
        auto ps = paramString(a.params);
        if (!ps.empty()) desc.parameters().add(p.parseParameters(ps));
        tab.addCommand(std::move(desc));
    }
    return true;
}

}
```

> Note: `fillSubDef` is templated so it accepts both `netlist::Netlist` (top level) and `netlist::Subckt` — both expose `params/models/instances/conditionals/subckts` with identical field names. `netlist::Netlist` has no `ports`; only the `Subckt` branch reads `ports`.

- [ ] **Step 5: Build + run**

Run: `cmake --build ../build.VACASK/Release --target demo_netlistrs && (cd /tmp && "$OLDPWD/../build.VACASK/Release/demo/api/demo_netlistrs")`
Expected: PASS — prints a `ParserTables` dump showing subdef with params `c0`, models `res`/`cap`, instances `r1`/`c1` (with `c1`'s `c` as an *expression* `2*c0`), exit 0.

- [ ] **Step 6: Verify `Status::set` signature.** Read `../VACASK/include/status.h`; correct the `s.set(...)` call in Step 4 to the real API (e.g. `s = Status(...)` or `s.set(...)`). Rebuild. This is the one spot that may not compile verbatim.

- [ ] **Step 7: Commit** (in `../VACASK`)

```bash
cd ../VACASK && git add include/netlistrs.h lib/netlistrs.cpp lib/CMakeLists.txt demo/api/demo_netlistrs.cpp && git commit -m "feat: netlist::Netlist -> ParserTables adapter" && cd -
```

---

### Task 4: VACASK — end-to-end elaborate + tran (the vertical slice)

**Files:**
- Create: `../VACASK/demo/api/rc.scs`
- Modify: `../VACASK/demo/api/demo_netlistrs.cpp` (full driver)
- Modify: `../VACASK/demo/api/CMakeLists.txt` (add CTest)

**Interfaces:**
- Consumes: `buildParserTables` (Task 3); `OpenvafCompiler`, `Circuit`, `Analysis`, `PTLoad`, `PTSave` (VACASK, per `demo1.cpp`).

- [ ] **Step 1: Write the test netlist** `../VACASK/demo/api/rc.scs`:

```
simulator lang=spectre
parameters c0=1u v0=5
r1 (1 2) res r=1k
c1 (2 0) cap c=2*c0
v1 (1 0) vsrc type="pulse" val0=0 val1=v0 delay=1m rise=1u fall=1u width=4m
model res resistor
model cap capacitor
model vsrc vsource
```

- [ ] **Step 2: Write the full driver** — replace `demo/api/demo_netlistrs.cpp` (structure mirrors `demo/api/demo1.cpp`, but tables come from the adapter):

```cpp
#include "libplatform.h"
#include "simulator.h"
#include "parser.h"
#include "netlistrs.h"
#include "openvafcomp.h"
#include "circuit.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace sim;

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: demo_netlistrs <file.scs|.cir>\n"; return 2; }
    std::string path = argv[1];
    bool startSpice = !(path.size() >= 4 && path.substr(path.size() - 4) == ".scs");

    std::ifstream in(path);
    if (!in) { std::cerr << "cannot open " << path << "\n"; return 2; }
    std::stringstream ss; ss << in.rdbuf();
    std::string source = ss.str();

    Simulator::setup();
    Simulator::prependModulePath({"../../lib/vacask/mod"});

    ParserTables tab("rc from rust parser");
    Parser p(tab);
    Status s;

    // Loads: milestone-1 responsibility of the driver (see plan).
    tab.add(PTLoad("resistor.osdi"))
       .add(PTLoad("capacitor.osdi"))
       .add(PTLoad("vsource.osdi"));

    if (!buildParserTables(source, startSpice, tab, p, s)) {
        Simulator::err() << s.message() << "\n"; return 1;
    }
    if (!tab.verify(s)) { Simulator::err() << s.message() << "\n"; return 1; }
    tab.dump(0, Simulator::out());

    OpenvafCompiler comp;
    Circuit cir(tab, &comp, s);
    if (!cir.isValid()) { Simulator::err() << s.message() << "\n"; return 1; }

    if (!cir.elaborate({}, "__topdef__", "__topinst__", nullptr, s)) {
        Simulator::err() << "elaboration failed: " << s.message() << "\n"; return 1;
    }
    cir.dumpHierarchy(0, Simulator::out());

    auto tranDesc = PTAnalysis("tran1", "tran");
    tranDesc.add(PV{"step", 1e-5}).add(PV{"stop", 10e-3});
    auto* tran = Analysis::create(tranDesc, cir, s);
    if (!tran) { Simulator::err() << "analysis create failed: " << s.message() << "\n"; return 1; }
    tran->add(PTSave("default"));

    auto [ok, canResume] = tran->run(s);
    delete tran;
    if (!ok) { Simulator::err() << "analysis failed: " << s.message() << "\n"; return 1; }
    Simulator::out() << "Analysis OK.\n";
    return 0;
}
```

> If `PTAnalysis::add(PV&&)` is not available (Task 3 grep showed sweeps, not scalar `add`), build params instead with `tranDesc.parameters().add(p.parseParameters("step=1e-5 stop=10e-3"))`. Verify against `parseroutput.h` and use whichever compiles.

- [ ] **Step 3: Add the CTest** in `../VACASK/demo/api/CMakeLists.txt`:

```cmake
add_test(NAME demo_netlistrs
         COMMAND demo_netlistrs "${CMAKE_CURRENT_SOURCE_DIR}/rc.scs")
set_tests_properties(demo_netlistrs PROPERTIES WORKING_DIRECTORY "/tmp")
```

- [ ] **Step 4: Build + run the analysis**

Run:
```bash
cmake --build ../build.VACASK/Release --target demo_netlistrs
(cd /tmp && "$OLDPWD/../build.VACASK/Release/demo/api/demo_netlistrs" "$OLDPWD/../VACASK/demo/api/rc.scs")
```
Expected: dumps tables + hierarchy, prints `Analysis OK.`, exit 0.

- [ ] **Step 5: Verify against a reference oracle.** Parse the same RC circuit in Cadnip (`MNACircuit(spc"...")`, `tran!`) or ngspice; confirm node-2 rises with τ≈R·C=2ms toward v0=5. This is behavioral verification, not just "it ran" (see superpowers:verification-before-completion).

- [ ] **Step 6: Commit** (in `../VACASK`)

```bash
cd ../VACASK && git add demo/api/rc.scs demo/api/demo_netlistrs.cpp demo/api/CMakeLists.txt && git commit -m "test: end-to-end RC tran via Rust parser adapter" && cd -
```

---

### Task 5: C++ — file entry + include resolution + `.sim` dispatch

Include resolution lives in **C++** (clean separation: Rust parses one string; C++ owns the filesystem). The projected `netlist::Netlist` exposes top-level `includes` as `Include{path, section}`; the adapter reads each referenced file, re-parses it with `parse_netlist` (dialect by extension), and accumulates its members into the same toplevel `PTSubcircuitDefinition` before it is moved into `tab`. This avoids mutating any `rust::Vec`.

**Files:**
- Modify: `../VACASK/include/netlistrs.h` (add file entry + dispatch)
- Modify: `../VACASK/lib/netlistrs.cpp` (refactor `buildParserTables` to accumulate; add resolver + dispatch)
- Modify: `../VACASK/demo/api/demo_netlistrs.cpp` (call the file entry)
- Reference: `Cadnip.jl/src/spc/sema.jl` (`sema_include!`, most-recent-include-wins) for merge semantics.
- Test: fixtures `../VACASK/demo/api/models.scs` + `../VACASK/demo/api/rc_inc.scs`; a CTest.

**Interfaces:**
- Consumes: `netlist::parse_netlist` (Task 1); `buildParserTables` (Task 3).
- Produces (in `netlistrs.h`):
  ```cpp
  // Parse a netlist FILE, resolving includes relative to its directory.
  //  .sim  -> VACASK's native parser (Parser::parseNetlistFile); Rust path skipped.
  //  .scs  -> Rust parser starting in Spectre.
  //  else  -> Rust parser starting in SPICE.
  bool buildParserTablesFromFile(const std::string& path,
                                 ParserTables& tab, Parser& p, Status& s);
  ```

**Scope note (milestone 1):** plain `include "file"` at top level is resolved. Section-qualified includes (`include "lib.scs" section=tt` / SPICE `.lib file sect`) require the Rust projection to expose named library/section blocks (the flat `Netlist` does not carry them today) — flagged as a projection-extension follow-up, not done here. Includes *inside* a subckt body are likewise deferred (the `Subckt` projection drops `includes`).

- [ ] **Step 1: Write the failing test.** Create `../VACASK/demo/api/models.scs`:

```
simulator lang=spectre
model res resistor
model cap capacitor
```
Create `../VACASK/demo/api/rc_inc.scs` (same circuit as `rc.scs` but models come from the include):
```
simulator lang=spectre
include "models.scs"
parameters c0=1u v0=5
r1 (1 2) res r=1k
c1 (2 0) cap c=2*c0
v1 (1 0) vsrc type="pulse" val0=0 val1=v0 delay=1m rise=1u fall=1u width=4m
model vsrc vsource
```
Add a CTest in `../VACASK/demo/api/CMakeLists.txt`:
```cmake
add_test(NAME demo_netlistrs_include
         COMMAND demo_netlistrs "${CMAKE_CURRENT_SOURCE_DIR}/rc_inc.scs")
set_tests_properties(demo_netlistrs_include PROPERTIES WORKING_DIRECTORY "/tmp")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build ../build.VACASK/Release --target demo_netlistrs && (cd ../build.VACASK/Release && ctest -R demo_netlistrs_include --output-on-failure)`
Expected: FAIL — `res`/`cap` models unresolved (the `include` is currently ignored), so `verify`/`elaborate` errors on unknown master `res`.

- [ ] **Step 3: Refactor `buildParserTables` to accumulate into a caller-owned toplevel def.** In `lib/netlistrs.cpp`, split the toplevel fill out of `buildParserTables` so it can be called repeatedly. Change the core so that both the root and each included file merge into the *same* `PTSubcircuitDefinition top`:

```cpp
// (private) accumulate one parsed Netlist's toplevel members into `top`,
// its analyses/globals into `tab`, then recurse into its includes.
void mergeNetlist(const netlist::Netlist& nl, PTSubcircuitDefinition& top,
                  ParserTables& tab, Parser& p,
                  const std::filesystem::path& baseDir,
                  std::set<std::filesystem::path>& visited);
```
where `mergeNetlist`:
1. `fillSubDef(top, nl, p);` (existing helper — appends params/models/instances/conditionals/subckts).
2. globals → `tab.addGlobal(...)`; analyses → `tab.addCommand(...)` (moved out of the old `buildParserTables`).
3. for each `inc : nl.includes` **with empty `inc.section`** (section-qualified deferred — see scope note): resolve `path = baseDir / sv(inc.path)`; skip + continue if already in `visited`; insert into `visited`; read the file; `bool spice = !(ext == ".scs");`; `netlist::Netlist sub = netlist::parse_netlist(rust::Str(contents), spice);`; if `sub.errors` non-empty → set `s`, return; recurse `mergeNetlist(sub, top, tab, p, path.parent_path(), visited)`.

Rewrite `buildParserTables(source, startSpice, tab, p, s)` to: `tab.defaultGround(); PTSubcircuitDefinition top; std::set<std::filesystem::path> visited; netlist::Netlist nl = parse_netlist(rust::Str(source), startSpice); if(!nl.errors.empty()){...return false;} mergeNetlist(nl, top, tab, p, std::filesystem::current_path(), visited); tab.setDefaultSubDef(std::move(top)); return true;`

- [ ] **Step 4: Add `buildParserTablesFromFile` with `.sim` dispatch.** In `lib/netlistrs.cpp`:

```cpp
bool buildParserTablesFromFile(const std::string& path, ParserTables& tab,
                               Parser& p, Status& s) {
    namespace fs = std::filesystem;
    fs::path fp(path);
    std::string ext = fp.extension().string();

    if (ext == ".sim") {
        // VACASK's own native parser fills tab directly.
        auto idx = tab.fileStack().push(path, s);   // adjust to FileStack's real API
        return p.parseNetlistFile(idx, s);
    }

    std::ifstream in(path);
    if (!in) { s.set(Status::Error, "cannot open " + path); return false; }
    std::stringstream ss; ss << in.rdbuf();

    tab.defaultGround();
    PTSubcircuitDefinition top;
    std::set<fs::path> visited{ fs::absolute(fp) };
    bool spice = (ext != ".scs");
    netlist::Netlist nl = netlist::parse_netlist(rust::Str(ss.str()), spice);
    if (!nl.errors.empty()) { s.set(Status::Error, "netlist parse error"); return false; }
    mergeNetlist(nl, top, tab, p, fp.parent_path(), visited);
    tab.setDefaultSubDef(std::move(top));
    return true;
}
```
Declare it in `netlistrs.h`. Add includes: `<filesystem>`, `<set>`, `<fstream>`, `<sstream>`. Verify `Parser::parseNetlistFile`'s `FileStackFileIndex` acquisition against `include/filestack.h`/`parser.h` and correct the `tab.fileStack().push(...)` line to the real API.

- [ ] **Step 5: Switch the driver to the file entry.** In `demo/api/demo_netlistrs.cpp`, replace the manual `ifstream` + `buildParserTables(source, startSpice, ...)` with:

```cpp
    if (!buildParserTablesFromFile(path, tab, p, s)) {
        Simulator::err() << s.message() << "\n"; return 1;
    }
```
(Drop the now-unused `startSpice`/`source` locals. Keep the `PTLoad`s — the driver still adds `resistor/capacitor/vsource` osdi.)

- [ ] **Step 6: Build + run all three CTests**

Run: `cmake --build ../build.VACASK/Release --target demo_netlistrs && (cd ../build.VACASK/Release && ctest -R demo_netlistrs --output-on-failure)`
Expected: `demo_netlistrs`, `demo_netlistrs_include` (models resolved from the include) all PASS.

- [ ] **Step 7: Commit** (in `../VACASK`)

```bash
cd ../VACASK && git add include/netlistrs.h lib/netlistrs.cpp demo/api/demo_netlistrs.cpp demo/api/models.scs demo/api/rc_inc.scs demo/api/CMakeLists.txt && git commit -m "feat: C++ include resolution + file entry (.sim/.scs/spice dispatch)" && cd -
```

---

### Task 6: Hardening — nested subckt + parameter override end-to-end

**Files:**
- Create: `../VACASK/demo/api/rc_nested.scs`
- Modify: `../VACASK/demo/api/CMakeLists.txt` (second CTest)

**Interfaces:** Consumes everything above. This task proves VACASK's scope resolution works through our transcribed tables (the plan's key verification).

- [ ] **Step 1: Write a nested-subckt + override netlist** `../VACASK/demo/api/rc_nested.scs`:

```
simulator lang=spectre
parameters rext=2k
subckt rcblock (a b)
  parameters r=1k c=1u
  r1 (a b) res r=r
  c1 (b 0) cap c=c
ends rcblock
model res resistor
model cap capacitor
model vsrc vsource
x1 (1 0) rcblock r=rext c=3u
v1 (1 0) vsrc type="pulse" val0=0 val1=5 delay=1m rise=1u fall=1u width=4m
```

- [ ] **Step 2: Add a second CTest** in `demo/api/CMakeLists.txt`:

```cmake
add_test(NAME demo_netlistrs_nested
         COMMAND demo_netlistrs "${CMAKE_CURRENT_SOURCE_DIR}/rc_nested.scs")
set_tests_properties(demo_netlistrs_nested PROPERTIES WORKING_DIRECTORY "/tmp")
```

- [ ] **Step 3: Build + run both CTests.**

Run: `cmake --build ../build.VACASK/Release --target demo_netlistrs && (cd ../build.VACASK/Release && ctest -R demo_netlistrs --output-on-failure)`
Expected: both PASS.

- [ ] **Step 4: Verify the override took effect.** In the hierarchy dump, confirm `x1.r1` sees `r=rext=2k` (call-site override of the subckt default `r=1k`) and `c=3u`. Cross-check the waveform's time constant against Cadnip/ngspice on the same netlist. Evidence before the completion claim.

- [ ] **Step 5: Commit** (in `../VACASK`)

```bash
cd ../VACASK && git add demo/api/rc_nested.scs demo/api/CMakeLists.txt && git commit -m "test: nested subckt + param override via Rust parser" && cd -
```

---

### Task 7 (milestone 2): embed SPICE blocks — SpiceBlock projection + adapter path

**Not on the milestone-1 critical path** (the vertical slice is Spectre `rc.scs`). Scheduled after Task 6, when non-`.scs` SPICE netlists must work end-to-end. To be broken into TDD micro-steps when picked up; the design and acceptance criteria are fixed here.

**Design (mirrors Julia's `SP.*` vs `SC.*` split — see `src/spc/sema.jl:133` and `:362`):** a `simulator lang=spice` region parses into an embedded `SPICENetlistSource` subtree. Project it as a **distinct `SpiceBlock`**, never by casting SPICE devices into the Spectre `Instance` struct.

- **Rust (`crates/netlist-cxx/src/lib.rs`):** extend the bridge schema — add `struct SpiceBlock { instances, models, subckts, params }` using SPICE-native shapes, and give `Netlist` and `Subckt` a `spice_blocks: Vec<SpiceBlock>` field. In `collect_scope`, add a `SyntaxKind::SPICENetlistSource` arm that projects a `SpiceBlock` via the **SPICE typed AST** (`netlist_syntax::ast`: `Resistor`/`Capacitor`/`Inductor`/`Diode`/`MOSFET`/`BipolarTransistor`/`SubcktCall`/`Model`/`Subckt`/`MutualInductor`/controlled sources — full accessor coverage already exists). SPICE instances keep positional nodes + prefix-derived master (`R*`→resistor, `C*`→capacitor, …) and positional values as named params (`r`/`c`/`l`); no cast to Spectre `Instance`.
- **C++ (`../VACASK/lib/netlistrs.cpp`):** add a SPICE-block → `ParserTables` path (walk `spice_blocks`, emit `PTModel`/`PTInstance` with positional terminals + prefix master + positional-value param), distinct from the Spectre-instance path; both feed the neutral tables.

**Acceptance:** a `.cir`/SPICE netlist (`R`/`C`/`V` + `.model`) drives `buildParserTablesFromFile` → `verify` → `elaborate` → `tran`; and a Spectre file with an inline `simulator lang=spice` block elaborates with both dialects' devices present. Node voltages diffed against Cadnip/ngspice.

---

## Self-Review

**Spec coverage** (against milestone 1 of the scoping doc):
- "Expose typed AST + Spectre through the C++ boundary" → done pre-plan (netlist-cxx); Task 1 adds the SPICE start-language dispatch that milestone 1 also implies. ✅
- "C++ adapter typed-AST → ParserTables (no separate IR crate)" → Task 3. ✅
- "`.include`/`.lib` resolution" → Task 5, done in **C++** (revised from Rust for clean separation). Plain top-level includes covered; section-qualified + in-subckt includes flagged as follow-ups. ✅
- "Corrosion build" → Task 2. ✅
- "Parse a real netlist, build ParserTables, verify, elaborate, run one tran" → Task 4; nested + override verification → Task 6. ✅
- `.sim` files → VACASK's own native parser (Task 5 dispatch), leaving the existing flex/bison path intact.
- Out of scope (correctly deferred): Verilog-A parsing (OpenVAF owns `.va`→`.osdi`); PTLoad derivation from directives (milestone 2); circulax elaboration layer (milestone 3); section-qualified/in-subckt includes and mixed-dialect *projection* fidelity for SPICE-only forms the Spectre AST doesn't cover — milestone-2 breadth.

**Placeholder scan:** Concrete code throughout. Real risks flagged for header verification rather than hidden: `Status::set` (Task 3/5), `PTAnalysis` scalar `add` vs `parameters().add` (Task 4), and `Parser::parseNetlistFile`'s `FileStackFileIndex` acquisition (Task 5 Step 4). Each is a single line to reconcile against the named header.

**Type consistency:** `parse_netlist(src, start_spice)` / `parse_netlist_file(path, start_spice)` used consistently across Tasks 1/2/5; `buildParserTables(source, startSpice, tab, p, s)` consistent across Tasks 3/4; `netlist::Netlist` field names match the bridge schema pinned above; `fillSubDef` relies on `Netlist` and `Subckt` sharing field names (verified in the bridge struct defs).

## Verification (milestone-level)

Build VACASK with Corrosion; parse `rc.scs` → `buildParserTables` → `tab.verify` → `Circuit::elaborate` → `tran` → `Analysis OK`; compare the node-2 waveform / RC time constant against Cadnip or ngspice. Then `rc_nested.scs` proves VACASK's scope resolution + call-site override work end-to-end from our transcribed `ParserTables`.
