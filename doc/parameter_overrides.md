# Parameter overrides: the lens, and what is still missing

How a netlist `.param`, a `.subckt` formal parameter, or a builder keyword
reaches the thing it names — what the design covers, the one piece of it that is
unimplemented, and the adjacent defects, sized against the code so the next
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

An override that names nothing no longer passes silently (§2), and device
instance parameters reach the lens (§1). The sections after those are adjacent
defects, not parts of this design: §3, §4 and §5 are independent bugs that
happened to surface nearby, all since fixed and kept here for the reasoning.

## 1. Raw device instance parameters (done)

`alter(c; var"r1.r"=2e3)` and `r1=(r=2e3,)` used to be silently ignored. The
namespace rule above already accommodated them — a group under an instance name
is that instance's scope — codegen just never consulted the lens at device
sites. It does now: a device line is a scope like any other, and the same
`getproperty`-then-call it uses for a `.subckt` instance is what a device gets.

### What a device declares

The knobs are what the card *resolves*, which is not the same as what the model
has fields for:

| device | declares |
|---|---|
| R / C / L (SPICE and Spectre) | `r` / `c` / `l`, `m` |
| V / I | `acmag`, `acphase`, `dc`, and a SIN/PULSE card's arguments under their SPICE names (`vo va freq td theta phase`, `v1 v2 td tr tf pw per`) |
| Spectre `vsource`/`isource` | `dc`, `mag`, `phase`, plus `vo va freq` at `type=sine` |
| E / G / H / F, Spectre `vcvs`/`vccs` | `gain` / `gm` / `rm` / `gain` |
| M / Q / D / N (model cards), and a VA module instantiated as `X1 … PSP103VA W=1u` | every instance parameter the line spells out, plus `m` |

A B (behavioral) source is the one device with no knobs: its value *is* an
expression, and the names in it are `.param`s, which the existing path already
reaches.

`acmag`/`acphase` are declared even by a card that carries no `AC` spec, at
zero — so an override can *introduce* an excitation where the netlist has none,
which is what an `.ac` sweep of a transient testbench wants. `dc` is declared
whenever the card spells it, and additionally when there is no transient
function to derive it from; on a SIN/PULSE/PWL card with no explicit `DC`, the
DC value follows the *resolved* offset (`vo`, `v1`, the first PWL point) rather
than shadowing it, so overriding the offset moves both.

The line that had to be drawn is the model-card one. A MOSFET has hundreds of
parameters and the card names four; the other 200 have no *default* codegen
could hand the lens, because they live in the model struct and are only known
once the card is resolved. Worse, `setproperties` silently ignores a kwarg that
is not a field, so routing an unspelled name through would have been a no-op
with no error — the exact failure this whole path exists to prevent. So a
parameter the card leaves at the model's default is not a knob, and spelling it
on the line (`M1 … nch w=1u`) is how you make it one. The PDK idiom — devices
wrapped in a subcircuit whose W/L are formal parameters — was never affected
either way.

Two knobs are deliberately *not* per-point: a SPICE `PWL(...)` list and a
Spectre `wave=[...]` are positional vectors, not named parameters, so only the
source's `dc` and AC phasor are reachable on a PWL card.

`m` is a knob on every device, and it is the one that has to be taken apart:
`_mfactor_expr!` emits the card's `m` *times* `_mna_m_`, the enclosing
subcircuit multiplicity, and the lens overrides only the first. Hence
`_mfactor_value!` — the card's own `m` — with the `* _mna_m_` moved to the use
site.

### The generated shape, and what it costs

One lens call per device, with the netlist's own values as the defaults, so the
override outranks the card exactly as it outranks a `.subckt` default:

```julia
let var"*dev#r1" = Base.getproperty(var"*lens#", :r1; type=:device)(; r = rtop, m = 1.0)
    stamp!(Resistor(getfield(var"*dev#r1", :r) / (getfield(var"*dev#r1", :m) * _mna_m_)), …)
end
```

A `let` rather than a statement, because an instance defined under an `.if` is
emitted as `cond && <expr>` and has to stay one expression.

The worry recorded here beforehand was cost: ~17 `cg_mna_instance!` methods and
a per-device lens lookup inflating generated code for every device, with c6288
(212k devices) already a known compile-time problem. Half of that was cheaper
than feared and half of it is real.

The methods were cheap: a shared helper (`cg_device_params!`) and a mechanical
edit each, not a hook per value site.

The generated code is not free, and the measurement is the useful part. Two
2000-element RC ladders, same 2002-variable system, one written flat and one as
2000 instances of a two-device `.subckt`:

| flat: 4002 raw device lines | before | after | |
|---|---|---|---|
| expr nodes | 424381 | 696565 | +64% |
| source chars | 5362633 | 6626049 | +24% |
| eval (s) | 9.8 | 15.6 | +59% |
| build + assemble (s) | 41.2 | 53.5 | +30% |
| first `dc!` (s) | 76.4 | 102.7 | +34% |

| hierarchical: 2 device lines, 2000 instances | before | after | |
|---|---|---|---|
| expr nodes | 260570 | 260800 | +0.09% |
| eval (s) | 3.29 | 3.36 | noise |
| build + assemble (s) | 22.2 | 22.9 | +3% |
| first `dc!` (s) | 48.967 | 48.965 | — |

So the cost is ~68 expression nodes per device *line in the source text*, and
nothing per *instance*: a `.subckt` body is emitted once and called 2000 times.
That is what settles the c6288 question the warning was really about — c6288 is
2419 gates built from ten `.subckt` definitions whose devices are all
subcircuit calls, so it has no raw device lines at all and pays nothing. A flat
netlist with thousands of device lines pays a third more compile time, which is
the honest price of the feature; it is a shape a generated deck can have, and
worth knowing before generating one.

Steady state is untouched: with a `ParamLens` carrying no override for `r1`,
`getproperty` returns an `IdentityLens` — `get(nt, :r1, (;))` on a `NamedTuple`
with a constant key constant-folds — and `IdentityLens(; kwargs...)` is
`values(kwargs)`, so the whole thing is the defaults tuple the card already
built. `test/mna/pcnr.jl`'s `@allocated rebuild() == 0` still holds, and the
hierarchical `dc!` above is identical to the digit.

The same call is what makes a device *observable*: with a `ParamObserver` in
place of the lens it records the device and its parameter names, which is what
lets §2's checker say `r1` declares `m, r` rather than reporting `r1` as an
unknown name. `type=:device` is passed so the diagnostic can say "a device
instance" rather than "a subcircuit instance".

One consequence worth naming: the top-level builder now binds its lens
unconditionally. `needs_lens` used to gate it on the deck declaring a `.param`
or instantiating a subcircuit; every device consults the lens now, so that gate
only decides whether the scope's *own* parameters go through it.

Contracts in `test/params.jl` (`"device instance parameter overrides"`): a
resistor's value in both spellings and against a `.param` the card reads, `m`,
a source's `dc` and `acmag`, a controlled source's gain, a device parameter as a
sweep axis, a device one level down inside a subcircuit
(`var"x1.r1.r"`), and an unspelled model parameter rejected.

Prior art, still honoured: the netlist-*text* `alter(io, ast; r1=(r=4.0,))`
always accepted this spelling, and the commented-out CedarSim test in
`test/basic.jl` ("device == param") shows device parameters were in the lens
tree by design (`i1=(dc=-1,)`, `rload=(r=2000.0,)`).

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

The line between the two is *whether the lens was consulted*, and that is why a
deck declaring nothing needed a nudge. A netlist with no `.param` and no
subcircuit instance has no reason to touch its lens, so it observed as an empty
tree — the same thing a hand-written builder that ignores `params` produces.
The two get opposite verdicts (check everything vs check nothing), the bare deck
took the wrong one, and every override on it passed silently; a swept axis on
such a deck came back a flat curve, which is the failure this whole path exists
to prevent. `codegen_mna!` now calls the lens with no parameters when the deck
declares none, so the observation reads "declares nothing" — checkable — instead
of "cannot be observed". The call is guarded by the same `params isa
AbstractParamLens` test the lens wrapping uses, which is decided at compile time
for the `NamedTuple`/`ParamLens` a solve actually passes, so nothing reaches the
hot path.

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
`cg_model_value!`. It took a hook for how a value expression is lowered, so that
a subcircuit-local card could reach the parent's `.param` through
`parent_params`; §5 binds those names as locals instead, and the hook is gone.

Contracts in `test/params.jl` (`".model cards read .param"`): the card reads the
netlist values, `MNACircuit(…; vt0=…)` / `alter` / a sweep axis all reach it, two
model parameters move independently, and the card resolves in all four
arrangements of card and parameter across the hierarchy (both at the top level,
both inside the subcircuit, and either one on its own).

Not covered: a `.model` inside a `.lib` section reading a `.param` from the
enclosing file, and the `make_mna_pdk_module` path, which parses model card
values as literals (`tryparse`) and skips anything it cannot, rather than going
through `cg_expr!` at all.

## 5. A subcircuit reading the parent's `.param` (fixed — same family as §4)

The general case of §4, and it failed the same way with no override in play.
`.param rtop=1k` at the top level, `R1 a out {rtop}` inside a `.subckt` that
declares no `rtop` of its own: `UndefVarError: rtop` on the first solve — while
the value sat right there in the call that raised it.

```
idivider_mna_builder(…, parent_params::@NamedTuple{rtop::Float64}, …)
```

Every call site had always built `parent_params` from the callee's exposed
names. The callee only ever *read* it in two places: a `.subckt` default
expression (`foo=foo+2000`) and, since §4, a `.model` card propagated into it.
Anything else in the body — a device value, a source value, an expression mixing
an inherited name with a formal — was emitted as a bare identifier that resolved
to nothing.

So the fix is not a third redirect but binding them once. The builder
destructures `parent_params` into locals at the top of its body, and every
expression in the scope spells the name bare, exactly as it would at the top
level. §4's `cg_val` hook and the `cg_scoped` branch that chose it are gone with
it; `cg_expr_with_parent_params!` survives for the one case where the local and
the parent's value genuinely differ — a self-referencing default, where `foo` on
the right of `foo=foo+2000` means the parent's.

`CodegenState` carries the bound names (`inherited_params`) so `_is_declared_param`
can count them, which settles precedence the way a reader expects: an inherited
`.param` outranks a `SpectreEnvironment` binding of the same name inside a
subcircuit, just as a declared one does at the top level.

### The set to bind is the transitive one, not the sema one

The obvious set is the wrong one, and picking it leaves the two-level case still
broken. A `.subckt` sema's `exposed_parameters` covers what its own body reads
plus what subcircuits defined **inside** it read: `resolve_scopes!` resolves a
callee through that scope's own `subckts` table, and a sibling defined at the top
level is not in it. The call sites have always used
`collect_exposed_parameters_*` instead — the transitive closure over callees —
so a middle subcircuit that instantiates a sibling and never mentions the name
itself *receives* it in `parent_params` and does not have it in
`exposed_parameters`.

Binding the sema set therefore fixed the one-level case and left the two-level
one failing with `UndefVarError: rtop` in the middle builder. `codegen_mna_subcircuit`
computes the same closure its callers do, which is by construction the field set
of the `parent_params` it will be handed.

That equality is also what let the three call sites collapse to one line. They
used to split on whether the caller declared the name (bare) or inherited it
(`parent_params.name`), with `isempty(subckt_semas)` standing in for "are we
inside a subcircuit" at the Spectre site. Both cases are locals of the calling
scope now, so all three just say `cg_expr!(state, name)`.

Contracts in `test/params.jl` (`"subcircuits inherit the parent's .param"`): a
device value one level down and two levels down through a subcircuit that never
mentions the name, a source value, an inherited name multiplied by a formal the
instance line sets, `MNACircuit`/`alter`/a sweep axis all reaching it at the
parent, and the instance rejecting it as a knob of its own — it is the parent's
parameter, so `x1=(rtop=…)` names nothing.
