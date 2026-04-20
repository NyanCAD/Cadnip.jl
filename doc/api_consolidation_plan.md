# API Consolidation Plan

**Status:** Draft — iterating.

## Goal

One coherent, robust API for the full workflow:

1. Reference precompiled PDKs and VA device packages from within a netlist via `jlpkg://`.
2. Load a SPICE/Spectre netlist from a file (primary), string or macro (secondary).
3. For tests: take an already-compiled builder function directly.
4. Always initialize via Cedar's `dc_op` / `tran_op` (robust multi-Newton + homotopy).
5. Return a SciML-compatible solution that supports name-based state lookup.

Today, each of those steps has multiple partial paths that do not compose cleanly. The plan below collapses them.

## Architectural decision: two-tier model resolution

Three parallel mechanisms exist today for resolving a SPICE device name:

1. **`ModelRegistry.getmodel`** — global Julia dispatch. BSIM4.jl, VADistillerModels.jl, PSPModels.jl already register here.
2. **`imported_hdl_modules` kwarg** — a per-call list of modules, searched by name. Threaded through `make_mna_circuit`, `parse_spice_to_mna`, `parse_spice_file_to_mna`, `load_mna_pdk`, `load_mna_modules`. Cannot be passed through string macros.
3. **Netlist include graph** — `.hdl "foo.va"`, `.lib "foo.sp" section`, `.include "foo.sp"`, and `jlpkg://Package/path` forms. Sema already handles these; for `jlpkg://` it switches scope to the target package (`src/spc/sema.jl:456-482`).

The second mechanism and much of the third exist because standalone VA loading produced modules that weren't registered anywhere. Users had to route them back in via a kwarg. String macros couldn't. Corner selection on a PDK is fundamentally a namespacing problem that flat registration can't solve (same device name, three different parameterizations per section).

**Decision: two tiers with a clear boundary.**

- **Tier 1 — Builtins (registry).** Everything SPICE simulators are "supposed to know" out of the box: R, C, L, D, V, I, standard MOSFETs dispatched by level (1/2/3/6/9/14/17/...), standard BJTs, diode levels. Resolved via `ModelRegistry.getmodel`. Populated by method addition. Contributed by Cadnip (passives) and stdlib-style packages (VADistillerModels, BSIM4, PSPModels) that claim `(devkind, level, version)` triples via Julia dispatch. `AbstractSimulator` lives here for its original purpose — disambiguating e.g. ngspice's `level=14` from hspice's.
- **Tier 2 — Scope (netlist include graph).** Everything else: PDK-specific devices, custom VA, photonic models, anything proprietary. Brought into scope at sema time via `.hdl`, `.include`, `.lib`, or `jlpkg://Package/path`. Resolved by a scope walk over the ordered list of included modules; `GlobalRef` baked into the generated builder. No global claim, no cross-package collisions, corners handled naturally because `.lib ... tt` vs `.lib ... ff` push different modules onto the scope.

**`spice_select_device` resolution order** (`src/spc/sema.jl:295`): scope tier first (most-recent include wins); builtin tier as fallback. Users who don't want their custom `d` to shadow the builtin `d` just don't name it `d` — matches how every other language layers local over global.

**`imported_hdl_modules` removed from all public APIs.** The concept survives as internal sema state (the scope walk list), populated only by actual netlist directives. No user-facing kwarg.

## Why VADistillerModels / PSPModels / BSIM4 need no migration

They already register via `Cadnip.ModelRegistry.getmodel` method additions (e.g. `VADistillerModels.jl:109-134`, `PSPModels.jl:71-74`). Under the two-tier model, that IS the correct behavior for Tier-1 contributors. `.model foo d` dispatches through builtins → `sp_diode`. `.model foo nmos level=1` → `sp_mos1`. The `imported_hdl_modules=[sp_diode_module]` kwarg in `benchmarks/vacask/graetz/cedarsim/runme.jl:31` is unnecessary today and just evaporates when the kwarg is deleted. No cross-repo migration.

## Why jlpkg:// is the PDK precedent (and how it generalizes)

`.lib "jlpkg://Sky130PDK/sky130.lib.spice" tt` already causes sema to resolve `Sky130PDK` as a Julia package, switch `imps` and `parse_cache` to the package's, and pick up `Sky130PDK.var"#cedar_parse_cache#"` (the precompiled parse state the package ships). Scope resolution is already package-aware for this case. What's missing is the generalization: the same scope model should drive device-name resolution for all Tier-2 entries (including local `.hdl "foo.va"` files), and Tier-1 should fall out by `getmodel` fallback.

## Netlist directives and what they populate

- **`jlpkg://Package/path`** — resolves to a Julia package; sema switches scope to the package's compiled parse cache. Canonical form for precompiled PDKs and published VA device packages.
- **`.hdl "foo.va"`** — parses the VA file inline at sema time, creates device types in the netlist's gensym'd scope module. Non-precompiled; re-parsed every compile. For local/in-development VA.
- **`.include "foo.sp"` / `.lib "foo.sp" section`** — parses the SPICE file inline, brings its subckts/models into scope.

## PDK-authoring API (not for end users)

PDK packages need a Julia-level function to call at package build time: parse the SPICE source, compile device types, bake a parse cache that `jlpkg://` can use later. That function is `Cadnip.precompile_pdk` (renamed from `load_mna_pdk`). It's documented as the thing PDK authors call inside their package; end users never invoke it. Same for VA-only device packages: `Cadnip.precompile_va` (renamed from `load_mna_va_module`) for authors.

End-user code uses `using MyPDK` to bring the package into dependency graph, then references its content from within the netlist via `jlpkg://MyPDK/...`. No Julia-level loading calls in user code.

## Module naming: opaque containers

Generated modules (VA devices compiled inline via `.hdl`, PDK device bundles inside published packages) are gensym'd — never user-visible, never imported by name. Matches how `jlpkg://` already works: users don't reach into `Sky130PDK.var"#something#"`; they reference devices via netlist directives.

The `MODULES = [optical_source_module, ...]` pattern in `neurophos/cadnip/tw_mzm_simulation.jl:33-37` disappears. Under the new model that script's VA files get loaded via `.hdl "optical_source.va"` lines inside the generated netlist (paths resolved against `source_dir`), or by packaging the VA files as a local Julia package and using `jlpkg://`.

## Generated-code hygiene

The baremodule design is intentional and correct: SPICE and Verilog-A have their own `sin`, `log`, `exp`, `max`, etc., and the baremodule is the namespace where those language-level identifiers get the right bindings. Sometimes that binding is `using Base: sin` (Julia's `sin` has the right semantics); sometimes a custom function defined in the module (semantics differ — e.g. a VA operator threading through a `V()` / `I()` access, or a SPICE function with different rounding). Either way, the generated code writes `sin(x)` and the module provides the right meaning.

What's wrong is that the templates have accumulated two kinds of imports that don't belong in the language namespace:

**Type A — language-level (correct to keep).** `sin`, `cos`, `log`, `exp`, `max`, `min`, `abs`, `sqrt`, `pow`, ... Whether backed by `Base` or a custom definition, these map a source-language identifier to the language's semantics. Leave them in the baremodule.

**Type B — Julia-side plumbing (wrong to import; qualify instead).**

- `hasproperty`, `getproperty`, `getfield`, `error`, `!==`, `iszero` — Julia metaprogramming helpers used by the generated builder to unpack parameter NamedTuples, dispatch on types, and throw errors. No SPICE/VA meaning. A VA module with a parameter `error` collides silently with `using Base: error`.
- `Cadnip.MNA.stamp!`, `va_ddt`, `va_absdelay_V`, `stamp_current_contribution!`, `alloc_internal_node!`, `alloc_current!`, `ZERO_VECTOR`, `va_laplace_nd_dss`, `va_laplace_zp_dss`, `MNAContext`, `MNASpec` — runtime machinery invoked by the generated code. Not language-level.
- `ForwardDiff.Dual`, `value`, `partials` — autodiff implementation detail. Not language-level.
- `Cadnip.VerilogAEnvironment` bulk `using` — audit each export; Type-A stays, Type-B gets qualified.

**Deciding rule:** "is this a function with matching semantics in the target language?" Yes → import or define in the baremodule so generated `foo(x)` binds naturally. No → emit fully-qualified `Namespace.foo(x)` at the call site in codegen.

**Hygiene guarantee after the audit:** a Type-B collision can't happen. A VA module with a parameter `error` or a SPICE subckt named `getfield` works. An identifier in user source still binds to a Type-A language function, a Tier-1 builtin, or a Tier-2 scoped device — the intended shadow semantics are preserved.

**Audit findings (scoped):**

- **VA codegen is nearly clean.** `src/vasim.jl` codegen already emits most MNA runtime calls fully qualified (~60 sites already correct). Only **3 bare Type-B emissions** to fix: `va_ddt` at `src/vasim.jl:828`, `error` at `src/vasim.jl:1861`, plus a quick edge check around operators.
- **SPICE codegen has the bulk of the work.** `src/spc/codegen.jl` emits `stamp!(Device(...), ctx, ...)` as a bare symbol at **~43 sites** (lines 773, 802, 827, 850, 879, 910, 941, 980, 1019, 1239, 1270, 1459, ...). All device-stamp calls plus behavioral-source + VA-module stamping. Mechanical find-replace to `Cadnip.MNA.stamp!(...)`.
- **VA baremodule preamble** (`src/vasim.jl:3600`) — drop Type-B imports: `va_ddt`, `stamp_current_contribution!`, `MNAContext`, `MNASpec`, `alloc_internal_node!`, `alloc_current!`, `ZERO_VECTOR`, `Dual`, `value`, `partials`. Keep `va_laplace_nd_dss`, `va_laplace_zp_dss` (not shadowing any user identifier).
- **SPICE baremodule preamble** (`src/spc/codegen.jl:3293`) — drop Type-B imports: `MNAContext`, `MNASpec`, `get_node!`, `stamp!`, `alloc_internal_node!`, `alloc_current!`. Keep device-type imports (`Resistor`, `Capacitor`, `Inductor`, `VoltageSource`, `CurrentSource`) — no user names a parameter `Resistor`.
- **`src/va_env.jl` and `src/spectre_env.jl` are clean** — all exports are Type-A. No changes.

**Sizing:** 3–4 days of active work (43 `stamp!` rewrites + 3 VA rewrites + 2 preamble changes + a `Resistor`/`Capacitor` usage audit) plus 2–3 days of regression testing. ~1 week total.

Folded into Phase 2 as a prerequisite. Without it, `Base.include(mod, SpiceFile(path))` has to mutate `mod` with Cadnip `using` statements just so the generated code's Type-B references resolve — the exact failure mode this section fixes.

## Target API (file-first)

```julia
# --- Production: load PDK via using + netlist from file ---
using Sky130PDK                              # Tier-1 builtins stay unregistered here;
                                             # Tier-2 content is used via jlpkg:// below
using VADistillerModels                      # populates Tier-1 diode/MOSFET builtins
circuit = MNACircuit("amp.sp")               # netlist's .lib "jlpkg://Sky130PDK/..." does the work
sol = tran!(circuit, (0.0, 1e-3))

# --- Production, with named builder for reuse / perf ---
Base.include(@__MODULE__, SpiceFile("amp.sp"))   # defines `amp` at top level
c = MNACircuit(amp; params=(R1=1e3,))
sol = dc!(c)

# --- Tests, small inline samples ---
circuit = MNACircuit(sp"""
  V1 vcc 0 DC 5
  R1 vcc out 1k
  R2 out 0 1k
""")
circuit = MNACircuit(spc"""...""")           # Spectre, matching old macro name

# --- Programmatic (dynamically-built netlist string) ---
# Supported but secondary; the netlist itself usually wants to be parameterizable.
circuit = MNACircuit(build_netlist(); lang=:spectre, source_dir=@__DIR__)

# --- Already-compiled builder (tests, performance-sensitive code) ---
circuit = MNACircuit(my_builder_fn; params=(R=1e3,))

# --- Parameters, sweeps, analyses ---
circuit = alter(circuit; R1=150.0)
sol  = dc!(circuit)                          # CedarRobustNLSolve + homotopy
sol  = tran!(circuit, (0.0, 1e-3))           # CedarTranOp init
sol  = ac!(circuit, freqs)
res  = dc!(CircuitSweep(circuit, sweep))     # SweepResult; iterates (params, sol)

# --- Name-based state lookup, one interface for DC and transient ---
sol[:vout]           # scalar for DC, full trajectory for transient
sol[sys.x1.vout]     # hierarchical
sol(t)[:vout]        # transient at time
```

Priority: file-first. Macros and runtime strings exist for tests and inline samples. Dynamic-string netlists are supported but not first-class — they typically indicate the netlist should be a parameterized `.subckt` instead.

File loading mirrors the existing VA pattern (`Base.include(mod, VAFile(path))` at `src/vasim.jl:3431`). `SpiceFile` / `SpectreFile` types dispatch to SPICE-specific `Base.include` methods. `MNACircuit(path)` is a one-liner that internally `Base.include`s into an anonymous module.

## Resulting public surface (after refactor)

| Category | Functions |
|---|---|
| Netlist loading | `MNACircuit(path)`, `MNACircuit(code; lang)`, `MNACircuit(builder)`, `Base.include(mod, SpiceFile(path))`, `Base.include(mod, SpectreFile(path))` |
| String macros | `sp"..."` (SPICE), `spc"..."` (Spectre), `va"..."` (Verilog-A) |
| Parameter modification | `alter`, `CircuitSweep`, `ProductSweep` and friends |
| Analyses | `dc!`, `tran!`, `ac!` |
| Solution access | `sol[...]` via SII |
| MTK interop | `@declare_MSLConnector` |
| PDK-authoring (not for end users) | `Cadnip.precompile_pdk`, `Cadnip.precompile_va` |

Everything else in the current public surface is deleted.

**Already in the API today, untouched by this plan:** `ac!` (in `src/ac.jl`), `MNACircuit(builder::Function; kwargs...)` (`src/mna/solve.jl:1348`), `@declare_MSLConnector` (MTK extension).

## Resulting internal primitives

Each is shared across multiple entrypoints and cannot be inlined without re-introducing duplication.

- **`make_mna_circuit(ast; circuit_name) -> Expr`** — codegen from parsed AST to a builder `Expr`. Called by `sp"..."`, `spc"..."`, `Base.include(mod, SpiceFile)`, `MNACircuit(path)`, `MNACircuit(code; lang)`, `precompile_pdk`. Genuinely shared. Stays unexported; `imported_hdl_modules` kwarg removed.
- **One private "eval builder into a module, return the builder function" helper** — shared between `Base.include(mod, SpiceFile(path))` and `MNACircuit(code; lang)`. The four hand-written copies at `test/common.jl:47-76, 95-124, 225-253` and `src/spc/interface.jl:235-258` collapse into this. Returns the builder function **bare, not wrapped in an `invokelatest` closure** — see "Invokelatest policy".
- **`spice_select_device`** (`src/spc/sema.jl:295`) — post-refactor, walks the sema scope (Tier 2) then falls back to `ModelRegistry.getmodel` (Tier 1). Single dispatch point, handles `UnimplementedDevice` warning.

## Invokelatest policy

**No `invokelatest` on the per-builder-call critical path.** The ODE/DAE solver calls the builder at every Newton iteration at every timestep; an `invokelatest` wrapper is a permanent tax.

`MNACircuit{F,P,S}` already stores the builder as a type parameter (`src/mna/solve.jl:1303`), so `dc!`, `tran!`, `ac!` specialize on the concrete builder type F. When `MNACircuit` is constructed with a freshly-eval'd builder, the specializations of `dc!` / `tran!` / solver callbacks for that new F-type are generated at the call site in the **current** world — after the eval. Standard Julia function-barrier pattern.

Consequences:

- The consolidated private helper evals the builder into an anonymous or caller-supplied module and returns the bare function. No closure wrapping.
- Users can freely construct `MNACircuit(path)` or `MNACircuit(code; lang)` inside a function body; `dc!(circuit)` crosses the type-dispatch boundary and picks up the new world without `invokelatest`.
- Only world-age gotcha: directly calling `circuit.builder(...)` in the same function body where `MNACircuit(...)` was constructed bypasses the barrier. Not a normal use case; mention in docs.
- The four existing test-helper copies (`test/common.jl:249`, `src/spc/interface.jl:250`, etc.) wrap with `(args...; kwargs...) -> Base.invokelatest(circuit_fn, args...; kwargs...)`. This pays `invokelatest` per call. Collapsing into one helper and dropping the wrapper is strictly faster.

## Deleted

- `parse_spice_to_mna` — one-line composition of `make_mna_circuit` + `NyanSpectreNetlistParser.parse`. Users get `Base.include(mod, SpiceFile(path))` or `MNACircuit(code)` instead.
- `parse_spice_file_to_mna` — replaced by `Base.include(mod, SpiceFile(path))` / `MNACircuit(path)`.
- `solve_spice_mna` — replaced by `dc!(MNACircuit(code))`.
- `make_mna_spice_circuit` (test helper) — replaced by `MNACircuit(code; lang=:spice)`.
- `solve_mna_spice_code`, `solve_mna_spectre_code`, `solve_mna_circuit`, `tran_mna_circuit` (test helpers) — replaced by `dc!(MNACircuit(...))` / `tran!(MNACircuit(...), tspan)`.
- `load_mna_va_module`, `load_mna_va_modules`, `load_mna_modules`, `load_mna_pdk` as end-user APIs — renamed / merged into PDK-authoring helpers `precompile_va` and `precompile_pdk`. End users don't call these.
- `imported_hdl_modules` kwarg — removed end-to-end. Concept survives only as internal sema scope-list state, populated by netlist directives only.
- `MNASolutionAccessor` — subsumed by SII on the solution types. Hard break.
- `voltage(sol, :x)` / `current(sol, :x)` accessors — deleted. Users write `sol[:x]`.

## Concrete problems being fixed

Cited so future-us can verify the fixes land.

1. **`tran_mna_circuit` bypasses transient init.** `test/common.jl:172-185` builds `ODEProblem` directly via `make_ode_problem`, skipping `tran!` and therefore `CedarTranOp`. Test-side analogue of the robust-Newton DC bypass fixed in PR #169.
2. **Four copies of the parse → temp-module → eval → `invokelatest` dance** at `test/common.jl:47-76, 95-124, 225-253` and `src/spc/interface.jl:235-258`.
3. **No Spectre string macro.** `sp"..."` and `va"..."` exist; no `spc"..."`. Spectre netlists can only be consumed via the slow temp-module path today.
4. **DC and transient have different solution types with different name-lookup APIs.** DC: `voltage(sol, :x)`. Transient: `MNASolutionAccessor(sol, sys)[:x]`. `SymbolicIndexingInterface` is already a dep (`src/circsummary.jl:24`) but only used for `NetRef`.
5. **`(ctx, sol)` tuple returns.** Exist only because `sol` cannot answer name queries alone. Disappear once SII is on solutions.
6. **`imported_hdl_modules` leaks into user scripts.** `benchmarks/vacask/graetz/cedarsim/runme.jl:31` passes `[sp_diode_module]` unnecessarily. Under the two-tier model, `.model foo d` dispatches through Tier-1 builtins to `VADistillerModels.sp_diode`; the kwarg is gone and the script stops needing it without any change to VADistillerModels.
7. **Corner selection is broken by flat registration.** A PDK with `typical`/`fast`/`slow` sections has the same device names across corners. Flat `getmodel` can't represent that. Two-tier resolves via Tier 2: `.lib "jlpkg://Sky130PDK/..." tt` vs `... ff` push different modules onto sema's scope, `getmodel` never sees the conflict.
8. **Manual stamping in integration tests.** `test/mna/audio_integration.jl`, `photonic.jl`, `vadistiller_integration.jl` build circuits via `MNAContext` + `stamp!` rather than through SPICE/VA.
9. **Real users reach for `MNA.assemble!` as a pre-warm.** `benchmarks/vacask/graetz/cedarsim/runme.jl:43` calls it outside the timed region to separate setup from solve. Should be an explicit API.
10. **Sweep result iteration requires `zip(solutions, cs)`** (`neurophos/cadnip/tw_mzm_simulation.jl:145`). `dc!(cs)` returns `Vector{DCSolution}` and loses the sweep-point association.

## Phased work

Ordered so the correctness fix (Phase 2 closes the `tran_mna_circuit` bypass) lands early. Each phase is independently shippable.

### Phase 0 — audit

Grep every Newton / `ODEProblem` / `DAEProblem` / `NonlinearProblem` construction site in `src/` and `test/`. Confirm each goes through `dc!`/`tran!`/`ac!` or has a documented reason. Flag stragglers; no code changes.

### Phase 1 — formalize the two-tier boundary; kill `imported_hdl_modules`

Dramatically smaller than earlier drafts because VADistillerModels / PSPModels / BSIM4 are already doing the right thing (Tier-1 method additions).

- **Establish Tier 2 scope walk in sema.** `spice_select_device` (`src/spc/sema.jl:295`) currently does: registry lookup → HDL-module-list fallback. Reorder to: scope walk (populated from `.hdl` / `.include` / `.lib` / `jlpkg://` directives) → registry fallback. The scope list is sema-internal state, never exposed to users.
- **`.hdl "foo.va"` sema rewrite.** Parse VA file inline, create device types in the netlist's gensym'd module, push that module onto the sema scope. No global registration. Relative paths resolve against the containing SPICE file.
- **`jlpkg://` stays as-is** (`src/spc/sema.jl:456-482`). Already does the right scope switch. One cleanup: make sure `.hdl "jlpkg://..."` works symmetrically (today `.lib` and `.include` handle it; check `.hdl`).
- **Remove `imported_hdl_modules` kwarg** from `make_mna_circuit`, `load_mna_pdk`, and internal sema threading. `parse_spice_to_mna` / `parse_spice_file_to_mna` are deleted outright in Phase 2.
- **Rename `load_mna_pdk` → `Cadnip.precompile_pdk`** and `load_mna_va_module` → `Cadnip.precompile_va`. Documented as PDK-authoring APIs; package the existing `#cedar_parse_cache#` baking behavior for `jlpkg://` consumption.
- **Delete `load_mna_modules`, `load_mna_va_modules`.** Multi-module handling collapses into the rename above (auto-detect single vs multi inside `precompile_va`).
- **Test:** VADistillerModels-using netlists that previously relied on `imported_hdl_modules=[sp_diode_module]` now work with the kwarg removed. Graetz benchmark is the primary case.
- **Test:** `.hdl "local_dev.va"` followed by a device reference in the same netlist resolves correctly.
- **Test:** two sibling `.lib "jlpkg://Sky130PDK/..." tt` and `.lib "jlpkg://Sky130PDK/..." ff` compiles produce different builders with different device types, no collision.

### Phase 2 — file-first loading; delete the temp-module duplication

**Prerequisite — codegen hygiene audit.** Before any new entrypoint can cleanly eval into a caller-supplied module, codegen must emit fully-qualified references. Without this, every entrypoint has to mutate the target module with `using` statements, re-introducing the duplication. See "Generated-code hygiene". Start Phase 2 here.

**SpiceFile / SpectreFile types:**
```julia
struct SpiceFile
    path::String
    name::Symbol       # builder name; defaults to filename stem
end
SpiceFile(path; name=Symbol(first(splitext(basename(path))))) = SpiceFile(String(path), Symbol(name))
# SpectreFile: identical shape
```
Default `name` from the filename stem. Explicit `name=` for overrides.

**`Base.include` methods:** `Base.include(mod::Module, f::SpiceFile)` / `Base.include(mod::Module, f::SpectreFile)`, mirroring `Base.include(mod, VAFile)` at `src/vasim.jl:3431`. Each parses, codegens via `make_mna_circuit(ast; circuit_name=f.name)`, evals the builder `Expr` into `mod`. Returns `nothing`; builder accessible as `mod.$(f.name)`. Fully-qualified codegen means `mod` needs no prior `using` statements.

**`MNACircuit(path::AbstractString; lang=infer_from_ext(path), kwargs...)`** — extension rule: `.scs` → Spectre, else SPICE. `lang=` overrides. Creates an anonymous module, `Base.include`s into it, wraps the builder bare.

**`MNACircuit(code::String; lang=:spice, source_dir=nothing, kwargs...)`** — runtime string. Shares the single eval helper with `MNACircuit(path)`. `source_dir` resolves `.hdl` relative paths; error if `.hdl` hits an unresolvable relative path.

**`spc"..."` string macro** — reinstate the old name; symmetric with `sp"..."`. Resolves `.hdl` relative to `__source__.file`.

**Deletions (hard break):**
- `test/common.jl`: `solve_mna_spice_code`, `solve_mna_spectre_code`, `solve_mna_circuit`, `tran_mna_circuit`, `make_mna_spice_circuit`.
- `src/spc/interface.jl`: `solve_spice_mna`, `parse_spice_to_mna`, `parse_spice_file_to_mna`.
- Every call site migrates to `dc!(MNACircuit(...))` / `tran!(MNACircuit(...), tspan)`.

**Byproduct:** fixes the `tran_mna_circuit` transient-init bypass — every test routes through `tran!`, which uses `CedarTranOp` by default.

**Tests to add:**

- `MNACircuit(code; lang=:spice)` inside a function body followed by `dc!` / `tran!` works without `invokelatest` on the per-call path. Regression-proofs the function-barrier story.
- `Base.include(mod, SpiceFile("amp.sp"))` against a module with no Cadnip `using` succeeds (confirms hygiene prerequisite held).
- `CircuitSweep` audit: does it accept a bare builder (as `neurophos/cadnip/tw_mzm_simulation.jl:138` does today) or only `MNACircuit`? If both, narrow to `MNACircuit`-input only. Trivial.

### Phase 2b — explicit `prepare!` / eager setup

- Make `MNACircuit(builder)` eagerly run structure discovery (currently `build_with_detection`, `src/mna/solve.jl:1437-1449`). Or expose `prepare!(circuit)` and document the benchmark pattern.
- Goal: the graetz benchmark's `MNA.assemble!(circuit)` pre-warm becomes `prepare!(circuit)` (or disappears because construction is eager).

### Phase 3 — `SymbolicIndexingInterface` on solutions

Approach: **attach `MNASystem` to the `ODEFunction.sys` field** so SII flows through the native SciML solution (Path A). No wrapper type. If SciMLBase internals turn out to make assumptions `MNASystem` can't satisfy, deal with them when we hit them rather than pre-building a wrapper.

- Implement `SII.is_variable`, `variable_symbols`, `variable_index`, `state_values`, `observed` on `MNASystem`.
- Thread `sys=circuit.system` through the `ODEFunction` / `DAEFunction` / `DDEFunction` construction in `tran!`.
- Implement the same SII surface on `DCSolution` (`src/mna/solve.jl:138`) directly — DC has no native SciML analog, so SII on the struct itself.
- Delete `MNASolutionAccessor` (`src/mna/solve.jl:1127-1174`) once the native-solution path works.
- Delete `voltage(sol, :x)` / `current(sol, :x)` accessors. Convert call sites.

If Path A hits a wall during implementation, fall back to a `(sol, sys)` wrapper — cost is roughly the same work, just in a different location. Sizing stays at ~2-3 days with the understanding that it might expand if SciMLBase resists.

### Phase 3b — `SweepResult` ergonomics

- `dc!(cs::CircuitSweep)` returns a small `SweepResult{P,S}` that iterates `(params, sol)` pairs and exposes `.points` / `.solutions` aligned.
- **Not `EnsembleSolution`.** Investigated (SciMLBase): `EnsembleSolution` doesn't carry per-trajectory parameter metadata, its plot recipes assume time-series inners (breaks DC), and its indexing has known issues with threaded solves. Custom `SweepResult`.
- Apply the same type to `tran!(cs)` once that exists.

### Phase 4 — port integration tests to HDL inputs

- Convert `test/mna/audio_integration.jl`, `photonic.jl`, `vadistiller_integration.jl`, and any `*_integration.jl` file to use `MNACircuit(path)` / `sp"..."` / `MNACircuit(code)` rather than manual `MNAContext` + `stamp!`.
- Unit tests in `test/mna/core.jl`, `charge_formulation.jl`, etc. keep manual stamping — that's what they test.
- Rule: tests whose name ends in `_integration.jl` must drive the public API a user drives.
- `test/common.jl` retains only fixtures (`isapprox_deftol`, etc.), no solver wrappers.

### Phase 5 — documentation

- README table: rows = input forms (SPICE file, Spectre file, VA file, PDK via jlpkg, local .hdl, SPICE string, Spectre string, VA string, builder fn, MTK connector); cols = (loader, example).
- Document the two-tier model explicitly: Tier 1 for builtins via `using VADistillerModels` / `using BSIM4`, Tier 2 for PDKs via `jlpkg://`.
- Remove the "Top-Level Eval for SPICE Circuits" section from `CLAUDE.md` — canonical pattern is `MNACircuit(path)` or `Base.include(mod, SpiceFile(path))`.
- Document `sol[:name]` as the canonical name-lookup.

## Explicit non-goals

- Rewriting `DCSolution` and the transient solution into a single unified struct.
- Touching the core stamping / `MNAContext` API.
- Changing how VADistillerModels / PSPModels / BSIM4 register (they're already correct Tier-1 contributors).
- Converting unit tests that exercise stamping directly.
- Automating level-specific VA registration (magic comments, annotations). Defer.
- Fixing the `params.params.x` double-dotting in `ParamLens`. Orthogonal.

## Sizing

Rough, subject to revision once Phase 0 turns up surprises:

| Phase | Size |
|---|---|
| 0 — audit | ~0.5 day |
| 1 — two-tier boundary, sema scope walk, `.hdl` rewrite, kill `imported_hdl_modules`, rename to `precompile_*` | ~1 day (no cross-repo migration; just Cadnip) |
| 2 — codegen hygiene (3 VA sites + 43 SPICE `stamp!` sites + preamble trim + tests), `SpiceFile`/`SpectreFile` + `Base.include`, `MNACircuit(path)`, `spc"..."`, delete helpers | ~1 week |
| 2b — eager prepare | ~0.5 day |
| 3 — SII on solutions (spike risk) | ~2-3 days |
| 3b — SweepResult | ~0.5 day |
| 4 — port integration tests | ~1-2 days |
| 5 — docs | ~0.5 day |

Total: ~2 weeks of focused work. No downstream-package migration needed.

Landing order: 0 → 1 → 2 → 2b → 3 → 3b → 4 → 5. Phase 2 is the fastest route to closing the transient-init bypass; depends on Phase 1 (for sema scope walk and kwarg removal) and on its own codegen-hygiene prerequisite.

## Resolved

- **Model resolution architecture:** two tiers. Tier 1 is the existing `ModelRegistry.getmodel` narrowed to SPICE-standard builtins (R/C/L/D, level-dispatched standard MOSFETs/BJTs), contributed by Cadnip + stdlib packages (VADistillerModels, BSIM4, PSPModels) via Julia dispatch. Tier 2 is the sema scope walk populated by netlist include directives, for PDK-specific and custom devices.
- **Corner handling:** Tier 2. `.lib "jlpkg://Sky130PDK/..." tt` pushes the tt-section module onto scope; `ff` pushes a different one. Precompiled PDKs can ship all corners in one package; no registry collision.
- **Downstream model packages:** no migration. They already register correctly as Tier-1 contributors.
- **Deprecation policy:** hard break. One PR, no aliases.
- **Spectre macro name:** `spc"..."` — matches the historical name and the `SpectreFile` type prefix.
- **Extension inference:** `.scs` → Spectre, else SPICE. Override via `lang=`.
- **File loading style:** `Base.include(mod, SpiceFile(path))` / `Base.include(mod, SpectreFile(path))`, mirroring the existing `VAFile` pattern.
- **`voltage` / `current` aliases:** deleted.
- **`SweepResult` vs `EnsembleSolution`:** custom `SweepResult`.
- **`AbstractSimulator`:** stays in its original role (simulator-variant disambiguation in Tier 1). Not repurposed for corners.
- **`.hdl` in runtime strings:** error if a relative `.hdl` path is encountered without `source_dir=`; absolute paths and `jlpkg://` forms work unconditionally. Keeps the common inline case zero-ceremony and surfaces missing context clearly for dynamic-string callers.
- **`precompile_pdk` section handling:** one call per PDK file, not per section. The PDK package ships a single parse cache; section selection is a consumer-side decision via `.lib "jlpkg://..." section`. Matches how `jlpkg://` already works (`test/sky130/scale.spice:3`); no new concept.
- **`.hdl "jlpkg://..."` symmetry:** implementation verification, not a design question. Folded into Phase 1's task list: verify the existing `.hdl` sema path handles `jlpkg://` the way `.lib` and `.include` do (`src/spc/sema.jl:456-482`); add the handling if it's missing.
- **Phase 3 SII approach:** Path A — attach `MNASystem` to the `ODEFunction.sys` field so SII flows through the native SciML solution. No wrapper type. If SciMLBase internals resist, fall back to a `(sol, sys)` wrapper as an in-Phase-3 adjustment rather than a pre-emptive design. Sizing stays at ~2-3 days.

## Open questions

None currently open. Plan is ready to execute.
