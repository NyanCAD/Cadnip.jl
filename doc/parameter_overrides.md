# Parameter overrides: the lens, and what is still missing

How a netlist `.param`, a `.subckt` formal parameter, or a builder keyword
reaches the thing it names — what the design covers, the one piece of it that is
unimplemented, and three adjacent defects, sized against the code so the next
session doesn't re-derive them or walk into the traps.

Context: `src/spectre.jl` (the lens), `src/spc/codegen.jl` (parameter
resolution), `src/mna/solve.jl` (`alter`), `test/params.jl` (the contracts).

## The shape of an override tuple

A scope — the top level, or a subcircuit instance — has *parameters* of its own
and *children* it instantiates. Both are named, and the namespaces can collide
(`.param x1=2` next to an `X1` instance), so there are two shapes:

```julia
compact    (a, x1 = (b,))                            # what a user writes
canonical  (params = (a,), x1 = (params = (b,),))    # what the lens reads
```

**A leaf is a parameter of the scope, a group is a child.** `x1 = (rv = 2k,)`
always addresses instance `X1`; `x1 = 2.0` always addresses parameter `x1`. When
a name is both and you need the parameter, `params = (x1 = 2.0,)` names it
explicitly and outranks the flat spelling.

`canonicalize_params` maps compact → canonical and is idempotent, so the lens
accepts either; `compact_params` is the inverse and is what `ParamObserver`
reports, so an observed tree can be handed straight back as an override. Both
are `@generated`, so this folds away at compile time.

`ParamLens` canonicalizes on construction, so everything downstream reads one
unambiguous shape. `getproperty` does *child* lookup only — a scope's own
parameters are never descended into, which is what keeps the collision case
working in both directions.

### Precedence

Within a subcircuit, strongest first:

1. lens override — `alter(circuit; var"x1.r1val"=…)`, a sweep axis
2. instance-line value — `X1 a b divider r1val=2k`
3. the `.subckt` default

The lens has to outrank the instance line, or a parameter the netlist spells out
is unreachable from `alter` and from sweeps. Codegen gets this by feeding the
netlist value in as the lens *default*.

### One home per knob

Because an explicit `params=` outranks the flat spelling, a knob must not exist
in both places at once: `alter` updates a knob *where it already lives*, so
building with one spelling and re-binding with the other lands on the same
parameter. Without that, a circuit built `params=(vin=1.0,)` and altered
`vin=4.0` keeps the old value, and the sweep comes back a flat line with no
error.

## Status

The override path for **named** parameters is complete: `.param` (top level and
subcircuit-local), `.subckt` formal parameters including ones the instance line
sets, Spectre `parameters`, and parameter→parameter expression chains
(`.param rbot='rtop*3'` follows `rtop`, and is itself overridable) — at any
hierarchy depth, through `MNACircuit`, `alter` and sweeps. The PDK idiom, devices
wrapped in a subcircuit whose W/L are formal parameters as in
`test/testpdk/testpdk.spice`, sweeps correctly including overriding a value the
instance line spells out, two levels down.

An override that names nothing no longer passes silently (§2). One piece of the
design is still unimplemented — device instance parameters (§1). The two
sections after those are adjacent defects, not parts of this design: §3 and §4
are independent bugs that happened to surface nearby, both since fixed and kept
here for the reasoning.

## 1. Raw device instance parameters are not overridable (unfinished here)

`alter(c; var"r1.r"=2e3)`, `r1=(r=2e3,)` and `var"r1.params.r"` are all silently
ignored. The namespace rule above already accommodates them — a group under an
instance name is that instance's scope — but codegen never consults the lens at
device sites.

Narrower than it looks: the PDK wrapper-subcircuit path above already covers the
W/L case, so this only bites *raw* device lines in hand-written netlists
(`M1 d g 0 0 nch w=20u l=1u`).

Costlier than it looks: 17 `cg_mna_instance!` methods and ~57 sites where a value
or parameter expression is built, so it wants a shared helper rather than a hook
per site, and a per-device lens lookup inflates generated code for every device —
on c6288 (212k devices) compile time is already a known problem. Gate on the
zero-alloc transient tests *and* c6288 build time, not just the vacask
benchmarks.

Prior art: the commented-out CedarSim test in `test/basic.jl` ("device == param")
shows device parameters were in the lens tree by design (`i1=(dc=-1,)`,
`rload=(r=2000.0,)`), and the netlist-*text* `alter(io, ast; r1=(r=4.0,))` still
honours that spelling.

## 2. Unknown override names are diagnosed (done)

A name no scope declares used to be inert, which made a typo look like a
parameter with no effect. It now throws at construction.

The names are already discoverable, and it is the two-lens design that makes
them so. `ParamObserver` is an `AbstractParamLens` that *records* where
`ParamLens` overrides; building once with one in place of the other yields the
whole tree — this scope's parameters under `:params`, each instantiated
subcircuit under its instance name, recursively, with the effective defaults
attached. `src/param_overrides.jl` diffs the override tuple against that tree.

Dispatch is what keeps this off the hot path. The builder calls the lens
generically, so observation costs the transient path nothing: it still gets the
`@generated` `ParamLens`, which folds away. There was no need to emit a static
table from codegen, and no need to thread anything through the lens.

The one real cost is the builder pass, and it is paid once per builder: the
observation is memoized on the builder object, so `alter` — which reconstructs
an `MNACircuit` at every sweep point — hits the cache. It only fires at all when
`params` is non-empty, so a circuit driven with no overrides (the c6288
benchmark) pays nothing.

Knowing a name's *kind* affords a real message rather than just "unknown":
`x1 = 2.0` where `X1` is an instance says to write `x1 = (rv = …,)`, `vin =
(…,)` where `vin` is a parameter says the reverse, and an outright typo lists
what the scope does declare.

Two things follow from observing rather than tabulating. Device instances never
consult the lens, so they are absent from the tree and read as unknown names —
rejected, but without naming §1 as the reason. And a scope reached only through
a `.if` that the overrides themselves would select is not observed, so such a
name reports as unknown; that is the same blind spot `alter` has always had, and
it fails safe.

A builder that cannot be observed is not checked. A hand-written builder that
reads `params` as a NamedTuple, or wraps it in `ParamLens` unconditionally,
throws on the observer and is left alone — which is right, since only it knows
what its parameters mean. Generated builders are observable because they accept
whatever lens they are handed (`params isa AbstractParamLens ? params : …`).

The one case where observation succeeds but *lies* is a hand-written builder
that takes its lens as a parameter (`p = params.lens(; R=…)`, as in
`test/mna/core.jl`): reading `params.lens` off the observer mints a phantom
child scope called `lens`, and the matching `MNACircuit(b; lens=IdentityLens())`
would then be rejected as naming an instance. An override whose *value* is an
`AbstractParamLens` is therefore skipped — a lens addresses whatever the builder
does with it, which is not a name that could be validated in the first place.
This is the failure mode to keep in mind if the checker ever gains reach:
observing a builder that was not written to be observed can invent structure.

Two knock-on fixes, both cases of an override that reached nothing: the
`MNACircuit` keyword constructor now folds dotted selectors (`var"x1.r1val"=2e3`)
into the tree the way `alter` always has, and `CircuitSweep` seeds its base
circuit through `alter` instead of splicing the first sweep point in as flat
keywords.

## 3. Subcircuit builder names collide (fixed)

Nothing to do with overrides — it surfaced through mismatched parameter keywords,
which is how it was found. Builders are named after the `.subckt` (`divider` →
`divider_mna_builder`), and netlist codegen used to be eval'd straight into the
caller's module, so two decks that each define `.subckt divider` overwrote each
other. The positional signatures match, so nothing errors: whichever deck was
loaded second answers for both, and the first returns the second's answer.

Whether that bites is a question of evaluation order, since this is ordinary
Julia redefinition — load A, use A, load B, use B is fine; load A, load B, use A
is not. The second ordering is the one the docs recommend (load at module top
level, use inside functions later, to stay clear of world age), which is what
made this worth fixing rather than documenting.

### A deck is a namespace

The fix is not to decorate the generated names but to stop leaking them.
`SpiceFile`/`SpectreFile` load a *complete* deck — they parse with
`implicit_title=true`, so the first line is a title — and a `.subckt` name is
local to its deck in SPICE. So each deck now gets a module of its own
(`_eval_deck_into_module` in `src/spc/interface.jl`), and only the circuit
builder is bound in the caller's module. A collision *there* is a visible
redefinition of a name the caller chose, not of one codegen invented.

This is the isolation the rest of the pipeline already had — a Verilog-A file
gets a baremodule, a PDK gets a baremodule, and `MNACircuit(path)` already
eval'd into a fresh module. `Base.include(mod, SpiceFile(...))` was the one
loading path that did not, which is exactly where the bug was.

Two details worth keeping:

- The module and the alias are evaluated as one `Expr(:toplevel, ...)`, not a
  block. Reading the builder back out with `getfield` immediately after
  `Base.eval` trips Julia 1.12's "access to binding in a world prior to its
  definition world".
- The parse cache stays on the *caller's* module, so `.hdl` Verilog-A modules
  are still shared across decks loaded into it. Per-deck caches would mean
  recompiling a PSP/BSIM model once per netlist.

What this deliberately does not cover: two `sp"..."` decks in the *same local
scope* (one function body, one `@testset`) still collide, because a module
cannot be defined in expression position. That case is a redefinition inside a
scope the author wrote, where Julia's last-wins is the expected answer — and it
is loud whenever the two subcircuits take different parameters.

SPICE's own `.include`, which *does* splice a snippet into the surrounding deck,
is a netlist directive handled in sema and is unaffected: within one deck, a
duplicate `.subckt` is a redefinition and last-wins is correct.

Regression test: `test/mna/subckt_scoping.jl`, "each deck keeps its own
subcircuits", with fixtures in `test/mna/fixtures/subckt_collision/`.

## 4. `.model` cards reading a `.param` (fixed — independent scoping bug)

Unrelated to overrides: it used to fail with no override in play. `.param
vt0=0.7` + `.model nch nmos vto=vt0` failed at *load* with `UndefVarError: vt0`,
because `codegen_toplevel_models!` emits model cards as module-level `const`s
outside the builder, where the `.param` local does not exist.

What is there: `model_param_deps` asks which declared `.param`s a card actually
reads. A card that reads none is still a module-level `const`, unchanged. A card
that reads some is emitted **inline in every builder that binds it**, after the
parameter assignments — the top level from its locals, a subcircuit through
`parent_params` (the parameters are added to the subcircuit's
`exposed_parameters` when the card is propagated into it, so they arrive the same
way a `.subckt` default's parent references do).

That is the whole mechanism. It is also the *third* design this took, and the
two it replaced were both defences against costs that turned out not to exist.

### Four performance arguments that did not survive measurement

This section used to warn that a parameterized card must not be built in the
builder ("a PSP-sized struct literal there is the LLVM SROA blow-up
`doc/psp103_noinline_investigation.md` exists to avoid") and asked for cards to
be built *"once per parameter set — never per restamp"*. Successive write-ups of
the fix added two more: `@noinline` on a module-level factory as cheap
insurance, and the factory itself as avoiding k+1 copies of the card. All four
were wrong, and the whole edifice reduced to "just emit it inline". Recorded so
the next session doesn't rebuild it — and because three of the four were
inherited from this document rather than measured, which is exactly how they
survived.

**A `const` card never bought "once per parameter set" in the first place.** Any
device line carrying instance parameters lowers to a `setproperties` in the
builder body:

```julia
# M1 d g 0 0 nch W=10u L=1u  →  every builder pass:
let dev = spicecall(nch; W = 1.0e-5, L = 1.0e-6)     # fresh full-size struct
```

so the full-size struct is already reconstructed on every pass whether `nch` is
a module-level `const` or a factory call. The `const` only ever saved the
*card-level* construction, never the instance-level one. Hoisting the card
further — caching it per parameter set behind a builder-ABI change — would
therefore have solved half a problem and left the larger half untouched.

**And the card-level cost is not measurable anyway.** Same circuit, constant
card vs a card reading one `.param`:

| | time | memory | allocs |
|---|---|---|---|
| MOS1 (~30 fields), transient, const | 3.006 ms | 647.49 KiB | 15735 |
| MOS1, transient, parameterized | 2.999 ms | 647.49 KiB | 15735 |
| PSP103VA (782 fields), DC, const | 986.4 µs | 345.22 KiB | 7600 |
| PSP103VA, DC, parameterized | 928.9 µs | 344.98 KiB | 7600 |

Identical allocation counts at both sizes — the structs are immutable and never
reach the heap — and the timing difference is noise.

**And `@noinline` on the factory bought nothing, so it is not there.** It was
added defensively from the SROA warning above, never measured. Measured on
PSP103VA (782 fields), `@noinline` vs. letting the compiler decide:

| | cold compile | steady-state | allocs | native code |
|---|---|---|---|---|
| `@noinline` | ~462 s | 1.004 ms | 7600 | 87 lines |
| inlinable | 461.9 s | 952.7 µs | 7600 | 87 lines |

Identical on every axis, *including* with the card used from five scopes (four
subcircuits + top level), which was the one case the single-scope runs could not
see.

**And the factory itself bought nothing, so it is gone too.** Its last claim was
that inline emission would put k+1 copies of the card in the module. Measured,
PSP103VA card bound from the top level plus k subcircuits, factory vs inline:

| | expr nodes | source chars |
|---|---|---|
| k=1 | 1061 → 1074 | 12341 → 12679 |
| k=4 | 2198 → 2241 | 25838 → 26746 |

Inline costs ~13 expr nodes per scope — 2% at k=4 — and eval time is
indistinguishable (0.017 s vs 0.014 s; 0.030 s vs 0.031 s). The premise was
simply false: a card lowers to only the parameters it *spells out*, never the
model's field count, so the "782-field literal" being duplicated does not exist.

```julia
spicecall(ParsedModel, PSP103VA, (VFB = vfbn, TYPE = 1))   # 2 params, not 782
```

Generated-expression size therefore scales with **card size**, not model size.
The one case where the factory could still pay is a card spelling out ~200
parameters across many scopes — and those arrive through `make_mna_pdk_module`,
which does not use this path at all.

⚠️ **Measure cold-compile numbers in separate processes.** Timing both variants
in one process makes the second inherit the first's compiled PSP103 `stamp!`
path and reads as a 33× improvement for whichever ran second. That artifact is
convincing enough to build a mechanism around; it is not real.

The live question this leaves is *instance*-level construction per restamp, which
predates this change and is a much larger topic than model cards.

The two copies of the card-lowering logic — `codegen_toplevel_models!` and the
`codegen_mna!` body, previously verbatim duplicates — are now one
`cg_model_value!`, parameterized by how a value expression is lowered. That hook
is what lets a subcircuit-local card read the parent's `.param` too.

Contracts in `test/params.jl` (`".model cards read .param"`): the card reads the
netlist values, `MNACircuit(…; vt0=…)` / `alter` / a sweep axis all reach it, two
model parameters move independently, and the card resolves in all four
arrangements of card and parameter across the hierarchy (both at the top level,
both inside the subcircuit, and either one on its own).

Not covered: a `.model` inside a `.lib` section reading a `.param` from the
enclosing file, and the `make_mna_pdk_module` path, which parses model card
values as literals (`tryparse`) and skips anything it cannot, rather than going
through `cg_expr!` at all.
