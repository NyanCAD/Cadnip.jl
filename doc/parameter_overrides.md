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
are independent bugs that happen to surface nearby.

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

Two knock-on fixes, both cases of an override that reached nothing: the
`MNACircuit` keyword constructor now folds dotted selectors (`var"x1.r1val"=2e3`)
into the tree the way `alter` always has, and `CircuitSweep` seeds its base
circuit through `alter` instead of splicing the first sweep point in as flat
keywords.

## 3. Subcircuit builder names collide (independent codegen bug)

Nothing to do with overrides — it surfaces through mismatched parameter keywords,
which is how it was found. Builders are named after the `.subckt` (`divider` →
`divider_mna_builder`), so two netlists loaded into the same module that both
define `.subckt divider` silently overwrite each other — the second wins, and the
first netlist's instances then call it with the wrong keyword arguments. The name
needs the netlist's identity in it; the `sp"..."` gensym already has one.

## 4. `.model` cards cannot reference a `.param` (independent scoping bug)

Also unrelated to overrides: this fails with no override in play. `.param
vt0=0.7` + `.model nch nmos vto=vt0` fails at *load* with `UndefVarError: vt0`,
because `codegen_toplevel_models!` emits model cards as module-level `const`s
outside the builder, where the `.param` local does not exist.

Do **not** "just move them inside the builder". The hoisting is deliberate —
subcircuit builders and the main function share them — and with `is_large_model`
(200+ fields, `invoke` to stop LLVM SROA blow-up) it would mean constructing a
PSP-sized struct on every restamp, the exact cost
`doc/psp103_noinline_investigation.md` and `doc/sroa_exploration_results.md`
exist to avoid.

Demand is smaller than it looks: corners are normally `.LIB` sections with their
own model cards, which already work (`test/testpdk/testpdk.spice`). First move is
to turn the `UndefVarError` into a diagnostic naming the parameter and pointing
at `.lib` sections. If parameterized model cards are ever genuinely needed, build
*only* the cards that reference a parameter, once per parameter set — never per
restamp.
