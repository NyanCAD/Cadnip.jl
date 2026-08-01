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

One piece of the design is unimplemented — device instance parameters (§1). The
three sections after it are adjacent defects, not parts of this design: §2 is a
missing diagnostic, §3 an independent bug that happens to surface nearby, and §4
an independent bug that has since been fixed, kept here for the reasoning.

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

## 2. Unknown override names are silent (missing diagnostic, not a design gap)

A name no scope declares is inert. That is deliberate — it matches how Spectre
`alter` behaves — but nothing says so, and a typo therefore looks like a
parameter that has no effect.

Do **not** reach for `ParamObserver`: it needs a full builder pass, and
`MNACircuit` is a plain struct today (free) that `alter` reconstructs *per sweep
point*, so validating at construction would rebuild the whole circuit on every
point. Codegen already knows every declared name (`sema.params`, formal params,
the subckt semas) — emit a static table beside the builder and validate with a
set lookup. That also affords a real message: "`x1` is an instance, write
`x1=(rv=…)`".

## 3. Subcircuit builder names collide (independent codegen bug)

Nothing to do with overrides — it surfaces through mismatched parameter keywords,
which is how it was found. Builders are named after the `.subckt` (`divider` →
`divider_mna_builder`), so two netlists loaded into the same module that both
define `.subckt divider` silently overwrite each other — the second wins, and the
first netlist's instances then call it with the wrong keyword arguments. The name
needs the netlist's identity in it; the `sp"..."` gensym already has one.

## 4. `.model` cards reading a `.param` (fixed — independent scoping bug)

Unrelated to overrides: it used to fail with no override in play. `.param
vt0=0.7` + `.model nch nmos vto=vt0` failed at *load* with `UndefVarError: vt0`,
because `codegen_toplevel_models!` emits model cards as module-level `const`s
outside the builder, where the `.param` local does not exist.

What is there: `model_param_deps` asks which declared `.param`s a card actually
reads. A card that reads none is still a `const`, unchanged. A card that reads
some is hoisted as a plain *function* of exactly those parameters, and each
scope that uses the model binds one call to it — the top level with its locals,
a subcircuit through `parent_params` (the parameters are added to the
subcircuit's `exposed_parameters` when the card is propagated into it, so they
arrive the same way a `.subckt` default's parent references do). The card is
written once no matter how many subcircuits use the model.

### Three performance arguments that did not survive measurement

This section used to warn that a parameterized card must not be built in the
builder ("a PSP-sized struct literal there is the LLVM SROA blow-up
`doc/psp103_noinline_investigation.md` exists to avoid") and asked for cards to
be built *"once per parameter set — never per restamp"*. The first write-up of
the fix repeated both and added a third: that `@noinline` on the factory was
cheap insurance. All three were wrong. Recorded here so the next session doesn't
re-derive them — and because two of the three were inherited from this document
rather than measured, which is exactly how they survived.

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
see and the last argument for keeping it — the "one copy instead of k+1" story
is not real; the compiler emits the same code either way.

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
