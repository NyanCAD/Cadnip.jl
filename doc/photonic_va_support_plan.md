# Plan: Photonic Verilog-A Model Support

## Context

The [photonic VA model library](../../../Verilog-A-photonic-model-library/veriloga/) contains ~25 optical component models (waveguides, couplers, lasers, detectors, etc.) that can't run in Cadnip.jl due to missing features. These models use a custom `optical` discipline with `OptE()` access function, array ports (`[0:3]`), module instantiation for hierarchical composition, `$abstime`, `@(initial_step)`, `laplace_nd()`, and `$rdist_normal()`.

The optical discipline has **only a potential nature** (no flow), making `OptE()` semantically identical to `V()` in MNA terms. This key insight means we can treat custom potential access functions as voltage aliases rather than building a full multi-discipline framework.

## Model Tiers by Feature Dependency

| Tier | Models | Additional Features Needed |
|------|--------|---------------------------|
| 1 | Polar2Cartesian, CartesianMultiplier, CartAdd/Sub, PolToCart, Terminator | OptE access + array ports |
| 2 | Attenuator, Waveguide, DirectionalCoupler, Isolator, PhaseShifter | + module instantiation |
| 3 | CwLaser, PhaseModulator | + `$abstime`, `@(initial_step)` |
| 4 | PhotoDetector, TunableFilter | + `laplace_nd()` |
| 5 | NoisyEDFA | + `$rdist_normal()` |

## Phase 1: Array Ports + Custom Access Functions (Tier 1) ✅

**Status**: Implemented

### Changes Made

**Parser (VerilogAParser.jl):**
- `forms.jl`: Added `range` field to `NetDeclaration` struct
- `parse.jl`: Fixed `parse_range` to handle single-element subscripts `[0]` (was unconditionally parsing colon+max)
- `parse.jl`: Fixed `parse_lvalue` to call `parse_function_call` directly instead of `parse_analog_expression` (prevented `V(a[0]) <+ expr` from being parsed as `V(a[0]) < (+expr)`)
- `parse.jl`: Skip `fc_to_bpfc` for contributions with array-indexed args; keep as FunctionCall
- `parse.jl`: Pass range into `NetDeclaration` constructor
- `parse.jl`: Fixed all `parse_range(ps)` calls to use `parse_range(ps, false)`

**Code Generator (src/vasim.jl):**
- `pins()`: Expanded to return `(expanded_pins, array_nodes)` — array ports `[0:N]` expand to individual symbols
- `MNAScope`: Added `array_nodes` and `access_map` fields
- `resolve_array_ref()`: New helper to resolve `cart[0]` to `cart_0` at codegen time
- `build_access_map()`: Extracts access function roles from nature/discipline declarations, with fallback defaults for standard VAMS functions
- FunctionCall handler: Added custom potential access function support (OptE, Temp, etc.)
- ContributionStatement handlers: Both inline and collected paths now support FunctionCall lvalues and use `access_map` instead of hardcoded V/I

## Phase 2: Module Instantiation (Tier 2)

**Goal**: Simulate `Attenuator.va` which composes Polar2Cartesian + CartesianMultiplier.

### Parser: Module instantiation syntax
- Add `ModuleInstantiation` AST node to `forms.jl`
- In `parse_module_items`, detect `IDENTIFIER IDENTIFIER LPAREN` pattern → parse as instantiation

### Code generator: Compile-time flattening
- Collect all modules from parse tree into `Dict{Symbol, VANode{VerilogModule}}`
- `flatten_module_instance(child_ast, port_mapping, instance_prefix)`:
  - Map parent port connections to child ports (with array slice resolution)
  - Prefix child internal nodes with instance name
  - Inline child analog block with substituted node names
  - Recurse for nested instantiations

## Phase 3: `$abstime` + `@(initial_step)` (Tier 3)

### `$abstime`
- Add to SystemIdentifier handler: `$abstime → :(_mna_t_)` (already exists as stamp! parameter)

### `@(initial_step)` event control
- Change `initial_step`/`final_step` from RESERVED to proper token kinds
- Add `EventControlStatement` AST node
- Parse `@(initial_step|final_step) statement` in analog blocks
- Codegen: wrap body in `if _mna_spec_.mode == :dcop ... end`

## Phase 4: `laplace_nd()` Transfer Functions (Tier 4)

### Parser changes
- Move `laplace_nd/np/zd/zp` from RESERVED to filter_functions
- Parse array constants `{1, 1/bw}` → `ArrayLiteral` AST node

### Code generator: State-space implementation
- `laplace_nd(input, num, den)` → controllable canonical state-space form
- Allocate N internal state nodes (order of denominator - 1)
- Stamp state equations into G and C matrices

## Phase 5: `$rdist_normal()` (Tier 5)

- Map `$rdist_normal(seed, mean, std)` to Julia RNG
- DC: return mean; Transient: sample normally
