# SPICE Task 4 Report: Controlled Sources + Remaining Kinds + Includes

## Scratch Verification — AST shapes

Ran `dump_cst` on a test netlist covering all new kinds. Key findings:

### E/F/G/H ControlledSource — kind derivation

The parser emits a single `ControlledSource` SyntaxKind for all four letters.
The control child node distinguishes voltage- vs current-controlled:

| Source | Control SyntaxKind | Kind enum |
|--------|-------------------|-----------|
| E1 out 0 in 0 2.0 | VoltageControl | Vcvs |
| G1 o 0 a b 1e-3  | VoltageControl | Vccs |
| F1 o 0 vsense 10  | CurrentControl | Cccs |
| H1 o 0 vsense 5   | CurrentControl | Ccvs |

E vs G and F vs H cannot be distinguished by control kind alone — both E and G
use VoltageControl; both F and H use CurrentControl. Final disambiguation uses
`name.chars().next().to_ascii_uppercase()` (first letter of device name).

### ctrl_value — bare token issue

The gain/transconductance value (e.g. `2.0`, `1e-3`, `10`) is emitted by the
parser as a bare `NumberLiteral` TOKEN directly under VoltageControl/CurrentControl,
NOT wrapped in a `LiteralExpr` node. The typed-AST `value()` accessor uses
`expr_children()` which skips bare tokens, returning `None`.

Fix: `ctrl_value_text(node: &SyntaxNode)` helper in `lib.rs`:
1. Scans child NODES for any non-HierarchialNode (covers BinaryExpression etc.)
2. Falls back to scanning `children_with_tokens()` for bare `NumberLiteral`/`Literal` tokens.

### MutualInductor (K)

Coupling coefficient is also a bare `NumberLiteral` token — same issue and same fix.
`MutualInductor` has no `params()` method (only `name()`, `inductors()`, `coupling()`);
coupling stored in `value` field via `ctrl_value_text`.

### Switch (S)

`Switch::nodes()` yields all HierarchialNodes after name (nd1, nd2, cnd1, cnd2, model).
No `params()` method. All nodes projected flat; model is the last entry.

### Behavioral (B), JFET (J)

Standard `name()`, `pos()`/`neg()` or `drain()`/`gate()`/`source()`/`model()`, `params()`.

### OSDIDevice (N) — only N parses as OSDIDevice; Y parses as TransmissionLine

`nodes()` skips name and yields all remaining including model (last). Same
last-is-model split pattern as BJT/SubcktCall, with empty-guard.

### SPICE .include / .lib

- `.include "path"` → `SyntaxKind::IncludeStatement`, accessor `IncludeStatement::path()` → `StringLiteral`
- `.lib "path" section` → `SyntaxKind::LibInclude`, accessors `LibInclude::path()` + `LibInclude::section()`

Both are collected into `SpiceBlock.includes` (reusing the shared `Include` struct).
`SpiceSubckt` has no `includes` field, so subckt-body includes are not collected (none in schema).

## TDD

**RED:** Added `projects_controlled_sources_and_include` + `projects_full_breadth` tests.
Ran `cargo test -p netlist-cxx` → 2 failures:
- `projects_controlled_sources_and_include`: E1 not found (arm missing)
- `projects_full_breadth`: 9 devices instead of 17

**GREEN:** Implemented:
1. `ctrl_value_text()` helper
2. `ControlledSource` arm (E/F/G/H, POLY/TABLE deferred to ctrl_value text)
3. `MutualInductor` arm
4. `Switch` arm
5. `Behavioral` arm
6. `JFET` arm
7. `OSDIDevice` arm
8. `IncludeStatement` + `LibInclude` in `project_spice_block`

Ran `cargo test -p netlist-cxx` → **10 passed; 0 failed**.

## Files Changed

- `crates/netlist-cxx/src/lib.rs` — only file modified:
  - Added `ctrl_value_text()` helper (~20 lines)
  - Added 7 new arms to `project_spice_device` (~100 lines)
  - Added `IncludeStatement` + `LibInclude` arms in `project_spice_block` (~15 lines)
  - Added 2 new tests (~60 lines)
  - Total: ~200 lines added to a single file

## Self-Review

- Purely structural: verbatim text, positional nodes, no renaming, no Spectre casting.
- Malformed-input safe: `OSDIDevice` has `if all.is_empty()` guard (same pattern as BJT/SubcktCall).
  `ControlledSource` returns `None` for unknown name prefix — adapter warn-skips.
- All prior arms (R/C/L/V/I/D/M/Q/model/subckt + Spectre scope) are untouched.
- `MutualInductor` and `Switch` correctly use `vec![]` for params (no `params()` accessor on these types).

## Concerns / Deferrals

- **POLY/TABLE control**: `PolyControl` and `TableControl` are captured as the raw
  node text in `ctrl_value`. Full structured parsing (dimension, coefficients, table
  pairs) is deferred — noted in the code comment.
- **Y (TransmissionLine)**: The brief mentions OSDI as N/Y, but `Y` parses as
  `TransmissionLine` (verified via dump). The task brief's mention of Y under OSDI
  appears to be incorrect; `Y` should be covered by a future TransmissionLine arm.
- **SpiceSubckt includes**: `SpiceSubckt` has no `includes` field, so `.include`/`.lib`
  inside a `.subckt` body are silently ignored — this matches the schema as defined.
