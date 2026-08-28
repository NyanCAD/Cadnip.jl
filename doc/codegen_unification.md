# Codegen unification

Findings from a README review that measured what the generated code actually
needs. Three related pieces of work: one shared import list, one lowering
instead of two, and two runtime warnings. All line numbers are against
`src/spc/codegen.jl` and `src/spc/interface.jl` as of this writing.

§5 was added later, from the same measure-don't-read discipline: the *other*
codegen backend in the same file, the one §1–§4 kept carefully in sync with
nothing, turned out to be unreachable and was deleted.

**Status.** All five are done. §1 and §2: the PDK path calls
`codegen_toplevel_models!` / `_propagate_toplevel_models!` /
`_codegen_subckt_builders` instead of its own copies, and the import list is
`_baremodule_prelude` — now the `Base` essentials for the PDK `baremodule` and
nothing else, because §4 removed the last name a deck resolved through a `using`.
§3 and §4 have their own "how it was fixed" sections below. §4's measured table
is corrected twice over: `SpectreEnvironment` *was* load-bearing, and the fix
was not the one this document predicted.

## 1. Two hand-maintained import lists, drifted apart

Both SPICE codegen paths emit a preamble of `using` statements so that names
appearing *bare* in generated code resolve:

- circuit path — `_make_mna_circuit_with_sema`, codegen.jl:3334-3337
- PDK module path — `make_mna_pdk_module`, codegen.jl:3533-3535

They are written out by hand in two places and no longer agree:

| name | circuit path | PDK path |
| --- | --- | --- |
| `Resistor, Capacitor, Inductor, VoltageSource, CurrentSource` | yes | yes |
| `VCVS, VCCS, CCVS, CCCS` | yes | **no** |
| `ParamLens, IdentityLens` | yes | yes |
| `StaticArrays` | yes | no |
| `spicecall, ParsedModel` | no | yes |
| `SpectreEnvironment` | yes | yes |
| `Base` essentials | no (normal module) | yes (`baremodule`) |

### The missing controlled sources are a live bug

Codegen emits `VCVS`/`VCCS` bare into PDK subcircuit builders, and the PDK
module imports neither. A PDK subcircuit containing an E or G card therefore
raises `UndefVarError` the first time it is *called*.

It has gone unnoticed because nothing in the test suite calls one, and the
failure is invisible until then: a bare global inside a function body is
resolved at call time, not at definition time. So both of these report success
on a deck that cannot run:

```julia
mod_expr = Cadnip.make_mna_pdk_module(lib; name=:typical)   # OK
Core.eval(m, mod_expr)                                      # OK
```

The check that does show it is static — compare the names the module imports
against the device names the generated code references bare:

```
PDK module imports: …, Capacitor, CurrentSource, Inductor, Resistor, VoltageSource, …
device names used bare in PDK code: VCVS, VCCS
BARE BUT NOT IMPORTED: VCVS, VCCS
```

Deriving both preambles from one definition closes this and stops it recurring.
Note the `baremodule` prefix (`using Base: …` + `import Base`) is genuinely
PDK-only and should stay a parameter of the shared helper rather than be
dropped.

### How it was fixed

Both, in the end. The 32 emission sites now write `$(MNA).Resistor(…)` — the
form every other MNA reference in generated code already used — so no import
can be missing for them and there is nothing left to drift. What survives is
`_codegen_preamble(; bare)`, one definition shared by both entry points,
holding `using Cadnip.SpectreEnvironment` and, for the PDK `baremodule`, the
`Base` essentials.

The regression test has to *solve* something: a bare global in a function body
resolves at call time, so codegen and `Core.eval` both pass on a module that
cannot run. `test/testpdk/testpdk.spice` gained a `ctrl_amp` subcircuit built
from E and G cards, and `pdk_test.jl` calls its builder and solves.

## 2. Two implementations of one lowering

`make_mna_pdk_module` (3405-3542) restates `_make_mna_circuit_with_sema`
(3256-3382) four times over:

1. building the `subckt_semas` cross-reference dict (3261-3267 vs 3411-3417)
2. propagating parent-level `.model` definitions into subcircuit semas
   (3276-3294 vs 3491-3500)
3. looping subcircuits to emit builders (3313-3320 vs 3506-3516)
4. **the big one** — ~65 lines (3421-3486) that hand-inline `.model` card
   lowering already implemented by `codegen_toplevel_models!` /
   `cg_model_value!`: the case-insensitive parameter map, `level`/`version`/
   `type` meta-parameter handling, the `getparams` registry query, and the
   `spicecall(ParsedModel, …)` construction

### Where the copies have diverged

Item 2 is not a faithful copy. The circuit path also pushes
`model_param_deps` into the subcircuit's `exposed_parameters`:

```julia
for dep in model_param_deps(state, last(defs)[2].val[1])
    haskey(ss.params, dep) || push!(ss.exposed_parameters, dep)
end
```

That is what makes a `.model` card read a `.param` (`.model nch nmos vto=vt0`),
so a process corner can be an ordinary sweep axis. The PDK copy has no such
step, which makes that a circuit-path-only feature by accident rather than by
decision — precisely the kind of drift a single implementation prevents.

Two *entry points* are reasonable: one produces a builder plus subcircuit
builders for a deck, the other a `baremodule` of exported builders for a PDK.
Two *implementations* of the same lowering are not.

### How it was fixed

`_subckt_sema_index`, `_propagate_toplevel_models!` and
`_codegen_subckt_builders` cover items 1-3; item 4 is now a call to
`codegen_toplevel_models!`, and the ~65-line copy is gone.

The copy was not harmless, which is worth recording because the opposite looked
plausible right up to being measured. A builder emits its own inline binding
for every `.model` card in *its* sema, so the module-level bindings read like
dead weight. They are not: for the test PDK's `.model pdk_diode d`, only the
module-level const is emitted, and `diode_1v8_mna_builder` references it without
binding one of its own — because propagation is keyed on `exposed_models`, which
that card is not in. So the module-level binding is what a PDK subcircuit
actually calls, and the old copy built it by `tryparse`-ing each value and
silently `continue`-ing past anything that was not a literal number.

## 3. Runtime warnings

### World-age binding access

`MNACircuit(path_or_code)` installs a builder and then reads it straight back
with `getfield` (interface.jl:374 and :383):

```julia
Base.include(mod, SpiceFile(path; name=eff_name))
builder = getfield(mod, eff_name)
```

On Julia 1.12 this prints, on every load:

```
WARNING: Detected access to binding `##divider#328.divider` in a world prior to
its definition world.
  Julia 1.12 has introduced more strict world age semantics for global bindings.
  !!! This code will error in future versions of Julia.
Hint: Add an appropriate `invokelatest` around the access to this binding.
```

It still returns the right answer today, but 1.12 says it will not keep doing
so. The fix is already written down a few hundred lines up, in
`_eval_deck_into_module` (interface.jl:130-133), which hit exactly this and
solved it by letting the world update before the binding is read:

> `:toplevel`, not a block: the module has to be fully defined — and the world
> updated — before the alias that reads a binding out of it runs. Reading it
> with `getfield` right after `Base.eval` instead trips Julia 1.12's "access to
> binding in a world prior to its definition world".

Julia's own hint (`invokelatest` around the access) is the other option.

### SciMLBase import warning

```
WARNING: Imported binding SciMLBase.MatrixOperator was undeclared at import
time during import to MNA.
```

Fires on every load of Cadnip.

### How it was fixed

Two warnings were listed; the family turned out to be larger. There are two
shapes of fix, and which applies depends on whether the read can wait:

- **The world can be advanced first.** Put the read in a following `:toplevel`
  statement, so it runs in the world the definition created. This is what
  `_eval_deck_into_module` already did for its alias, and it is the better fix
  where it applies — no dynamic dispatch, and the value comes back typed.
- **It cannot** — codegen is mid-expression, with no later statement to move the
  read into. Then it is Julia's own hint: `latest_global(mod, name)` in
  `src/util.jl`, one `invokelatest`-wrapped `getglobal` shared by every such site.

Applied:

| site | shape |
| --- | --- |
| `_eval_deck_into_module` returning its builder | `:toplevel` |
| `MNACircuit(code; lang=)`'s inline branch | `:toplevel` |
| `codegen_hdl!` reading back a `.hdl` module | `:toplevel` |
| `parse_and_eval_vafile`, both branches (`vasim.jl`) | `:toplevel` |
| `precompile_pdk` reading back a PDK module | `:toplevel` |
| the five `getglobal`s on a model's Julia type (`codegen.jl`) | `latest_global` |
| `ensure_cache!` and the four `cache.jl` accessors | `latest_global` |
| `ModelLoader.load_VA_model`'s two hops | `latest_global` |

`MNACircuit(path)` and `Base.include(mod, SpiceFile(...))` therefore hand the
builder back rather than leaving callers to `getfield` it; that is also the more
conventional thing for an `include` method to do. The `test/common.jl` shims got
the same treatment.

**`SciMLBase.MatrixOperator`.** Nothing in the repository referenced it — the
import was the only mention. Deleted.

Also gone: `_eval_builder_into_module`, dead since the deck-module change, whose
body was the very `Base.eval`-then-`getfield` pattern above.

**What was left (now fixed).** One *shape* survived, in the `.hdl` Verilog-A
path: reading a device type out of the module that defines it. `test/basic.jl`
went from eighteen warnings across four distinct bindings to six across one
(`BasicVAResistor_module.BasicVAResistor`); the full suite printed eight, at
three VA modules — `BasicVAResistor`, `TM2D`, `TMRoundTrip` — which was that one
shape at three decks, not three separate problems.

It was not any of the reads listed above; those were wrapped, and the count
fell when they were. It did not yield to inspection either — localising it
needed a stack trace, which meant `--depwarn=error`, which used to die first on
an unrelated deprecation (`SemaResult(ast)` passing `Dict()` for `OrderedDict`
fields, since fixed).

With that blocker gone, `--depwarn=error` on `test/basic.jl` didn't print the
predicted warning-turned-error at all — it died somewhere else first, with a
plain `KeyError: key :basicvaresistor not found` out of `resolve_subckt`, no
world-age wording anywhere near it. `cg_mna_instance!` (and three siblings that
repeat the same lookup for VCVS/VCCS/subcircuit contexts) detect a `.hdl`
instance with `isdefined(hdl_mod, subckt_name)`, ahead of the `getfield` reads
this section already fixed. `isdefined` doesn't print the binding-partition
warning `getfield`/`getglobal` do — it just answers against the calling frame's
world, silently, and for a device module defined earlier in the very same
codegen call that answer can still be `false`. So the VA-module branch was
skipped, codegen fell through to "this must be a subcircuit", and *that*
failed loudly because no subcircuit by that name exists. Worse than the reads
this section already covered: those at least warn before they mislead.

Ten call sites had the pattern (`codegen.jl:814,1594,1601,1734,1742,2067,2074,
2454,2950,3456`), all `isdefined(hdl_mod, name)` immediately gating a `GlobalRef`
or `latest_global` read of that same name. A new `latest_isdefined(mod, name)`
in `src/util.jl` (`Base.invokelatest(isdefined, mod, name)`, the same shape as
`latest_global`) replaces all ten. `test/basic.jl`, `test/mna/table_model.jl`
(the `TM2D`/`TMRoundTrip`/`TM1D` fixtures) and `test/mna/vadistiller.jl` now run
clean under `--depwarn=error` — no warnings, no errors — and `test/mna/core.jl`
and `test/mna/va.jl` are unaffected under the normal flags.

## 4. Related: `sp"..."` / `spc"..."` inside a function body

The README claimed for a long time that the string macros "work transparently
in both top-level and function-body contexts". They do not — the macro splices
the generated block, `using` statements included, into the call site, and Julia
rejects `using` anywhere but top level:

```
ERROR: syntax: "using" expression not at top level
```

`va"..."` is unaffected. Making the old promise true needs the imports gone,
which is why this hangs off item 1.

### Which imports are actually load-bearing

Measured by stripping each `using` and eval'ing the result into a bare module
that provides nothing:

| `using` | verdict |
| --- | --- |
| `MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource` | **needed** |
| `MNA: VCVS, VCCS, CCVS, CCCS` | **needed**, E/G/F/H cards only |
| `Cadnip: ParamLens, IdentityLens, StaticArrays` | no deck needed it |
| `Cadnip.SpectreEnvironment` | ~~no deck needed it~~ — **wrong, see below** |

`Resistor` sits in a needed import but is itself never bare — an R card lowers
through `GlobalRef(SpectreEnvironment, :resistor)`, so only the other four
mattered. All nine are moot now: §1 rewrote the emission sites to
`$(MNA).Resistor(…)`.

An AST scan is not enough to establish any of this: it counts kwarg names as
bindings and misses names no test deck exercises (it reported `CCVS` unused
until an H card was added). Strip-and-run is what settles it.

#### Correction: `SpectreEnvironment` is load-bearing

The row above was measured against decks that never hit the case. Not every
SPICE name lowers to a `GlobalRef` — that is true of a function *call*
(codegen.jl:225, 298), but an *identifier* goes through
`cg_expr!(state, ::Symbol)`, whose fallback emits the bare symbol. So

```spice
.param foo = temper
```

emitted a bare `temper()`, and `test/basic.jl:583` has exercised exactly that
the whole time. `$time` (`test/basic.jl:221`, in a `bsource`) was the same.

Both are now `GlobalRef`s: they are special-cased before any parameter lookup,
so a `.param temper` cannot shadow them and naming the binding directly changes
nothing but where it resolves. What is left is the fallback — an identifier that
is neither a parameter nor a net, such as `M_1_PI` — which is why the preamble
still carries `using Cadnip.SpectreEnvironment`. Resolving that fallback against
`SpectreEnvironment` (without letting it outrank a `.param` of the same name) is
the remaining piece.

### What the fix buys, and the second blocker

Rewriting the bare device names to `GlobalRef`s and dropping all four
reproduces the baseline answer exactly on passives, E/G/F/H, `.param`, subckt,
PWL/SIN/PULSE, `temper`, `agauss`, `$scale`, and — with VADistillerModels
loaded — diode (0.669317 V), MOS1 (1.68 V) and BJT (4.9433 V).

Spliced into a function body, plain and subckt decks then work (2.5 V, 1.25 V).
A **second blocker** remains for `.model` decks: `model_defs` emits `const`
bindings (codegen.jl:3340) and `const` is illegal on a local, so those have to
become plain locals in the in-function case.

### How it was fixed — and why not this way

Both blockers were cleared, the whole thing was measured, and then it was thrown
away. The record is worth keeping, because the plan above looks right until you
benchmark it.

**The identifier fallback.** `cg_expr!(state, ::Symbol)` now resolves a name no
scope declares against `SpectreEnvironment` and emits a `GlobalRef`; a `.param`
of the same name is checked first and still wins. `_is_declared_param` is that
check, and it is *narrower* than it first looks: only `params` and
`formal_parameters` count, not `exposed_parameters`. A name inherited from an
enclosing scope is left free by codegen — `parent_params` is passed to a subckt
builder and never destructured — so counting it as a parameter turns a name the
environment could have resolved into an `UndefVarError`. (That free variable is
a real bug of its own: a `.subckt` body reading a parent `.param` it has no
local default for raises `UndefVarError` today, on `main` as much as here. It
is filed on the scratchpad, not fixed here.)

The lookup is case-insensitive, matching what function calls already did.
`M_1_PI` reaches codegen from SPICE lowercased to `m_1_pi`, so it had never
resolved from a SPICE deck — the `using` only ever served Spectre. It works from
both now.

With that, a *deck's* generated code needs no imports at all. A PDK
`baremodule` still needs a list, and the reason is narrower than it looks: not
the device names or the environment functions, which are all named outright now,
but the **operators**. An expression lowers to `Expr(:call, :-, lhs, rhs)` — a
bare `-`. A deck's builder module is an ordinary `module`, so that resolves
through `Base`; a `baremodule` has no `Base`.

Dropping `using Cadnip.SpectreEnvironment` from `_baremodule_prelude` therefore
broke VACASKModels at *precompile* time, with `UndefVarError: - not defined in
VACASKModels.vacask_models` — and the test PDK did not catch it, because its
arithmetic all lives in subcircuit bodies, which a `baremodule` only type-checks
when they are called. The card that catches it is one whose value is hoisted to
module scope, so the operator runs when the module is *defined*:

```spice
.MODEL pdk_diode_scaled d is='2e-14 - 1e-14' n='0.5 + 0.5'
```

`test/testpdk/testpdk.spice` carries that now, and `pdk_test.jl` solves with it.
Verified the way a regression test has to be: with the import removed the test
reproduces `UndefVarError: -`, with it restored it passes.

**The `const` blocker, and why the `let` was abandoned.** The obvious answer to
"`const` is illegal on a local" is to make the spliced block a `let`: plain
assignments in the header, `function` definitions in the body, the builder as
the block's value. That works — every case runs, two decks in one scope stop
colliding — and it costs too much. Subcircuit builders defined in a `let` body
are captured by the circuit builder through a `Core.Box`, every call through the
box infers to `Any`, and a `tran!` over a two-instance subckt deck went from
603,872 allocations to 811,168. Moving the model bindings into the `let` header
fixed *their* capture and not this one; typed captures for the subckt builders
need them bound in the header too, which means topologically sorting the subckt
call graph so no builder is named before it exists.

So the macros do what every other loader already did: `_eval_deck_into_module`,
at macro-expansion time, into a module of the caller's. The macro's *result* is
the builder object, which is a constant — no `using`, no `const`, no closure,
nothing for Julia to reject in expression position. Allocation parity is exact
(603,872 either way; 397,2xx on the MOSFET stage), a deck is a namespace on this
path too, and there is one loading path instead of two.

Three things had to be checked before believing it, and all three hold:

- **World age.** The methods are defined during expansion, which for a top-level
  statement is before that statement's thunk runs, and for a function body is
  when the enclosing `function` is defined. `dc!(MNACircuit(sp"..."))` as a
  single top-level statement works, and so does the same inside a function.
- **Precompilation.** A package with `const c = sp"..."` at module scope *and*
  an `sp"..."` inside a function body precompiles and runs from the cache. It
  matters that the deck module is a *child* of the caller's: the same macro
  written against a free-standing `Module(gensym(...))` fails with "Evaluation
  into the closed module `##deck#277` breaks incremental compilation". Which is
  why the macros reuse `_eval_deck_into_module` rather than
  `_fresh_netlist_module` — the latter is fine for `MNACircuit(path)`, which
  only ever runs at runtime.
- **Expansion-time side effects.** Already the norm here: `.hdl` Verilog-A
  modules are `Core.eval`'d into the caller during the same call.

## 5. The pre-MNA codegen path was dead, and is now gone

`src/spc/codegen.jl` carried two codegen backends against one `SemaResult`:
`codegen!`, which emitted `Named(spicecall(model; params...))(nets...)` device
objects for the DAECompiler era, and `codegen_mna!`, which emits `stamp!` calls.
Everything §1–§4 above unified was work on the second one. The first was
unreachable, and the trace is short enough to state in full.

### The trace

`codegen!` has one caller, and the chain above it terminates in a value nothing
constructs. Leaf first:

    codegen!(state)                       codegen.jl
      <- codegen(scope)                   codegen.jl
        <- generate_sp_code(...)          interface.jl
             the @generated body for
          <- (::SpCircuit)(nets...)       generated.jl

An `SpCircuit` was constructed in exactly two places, and both are inside the
legacy path itself:

* `cg_instance!(::SNode{SP.SubcktCall})`, reachable only from `codegen!` — the
  legacy path recursing into itself for a subcircuit;
* `sema_assign_ids`, whose only caller was its own recursion.

Nothing called `sema_assign_ids`, so no `CktID` was ever assigned,
`SemaResult.CktID` stayed `nothing` for every deck, no `SpCircuit` was ever
constructed, the generated function never fired, and `codegen`/`codegen!` never
ran.

Two undefined names corroborate it from the other side. `Cadnip.Named` and
`Cadnip.SimOptions`/`Cadnip.options` do not exist:

```julia
julia> [s => isdefined(Cadnip, s) for s in (:Named, :SimOptions, :options)]
:Named      => false
:SimOptions => false
:options    => false
```

Both were interpolated as *values* at codegen time — `cg_spice_instance!`
splices `Named` into every device it emits, and `codegen!`'s option block
splices `Cadnip.SimOptions`. So the first deck containing a resistor, or an
`.option temp/gmin/scale`, would have thrown `UndefVarError` while its builder
was being generated. The test suite compiles exactly such decks without error,
which is only possible because none of that code runs. (This is finding 2 in
`doc/FINDINGS.rst`; it is fixed by deletion rather than by defining the missing
names.)

### What went

| file | what |
|---|---|
| `src/spc/codegen.jl` | `cg_params!`, `cg_spice_instance!`, the five `cg_instance!` methods, `cg_model_def!`, `codegen!`, `codegen` |
| `src/spc/generated.jl` | the whole file — the generated function and its `reload()` |
| `src/spc/interface.jl` | `SpCircuit`, `getsema`, `generate_sp_code` (the deck-loading API in the rest of the file is untouched and live) |
| `src/spc/query.jl` | the whole file — `show`, `getproperty`, `SpRef`, `RefKind`, `MultipleKinds`, all dispatching on `SpCircuit` |
| `src/spc/sema.jl` | `assign_id!`, `sema_assign_ids`, the `SemaResult.CktID` field and its two readers |
| `src/spectre.jl` | `devtype_param`, superseded by `ModelRegistry.getparams` and with no callers left |
| `test/compiler_sanity.jl` | orphaned: not in `runtests.jl`, and `using DAECompiler` is not a dependency of the test project |

`query.jl`'s `getproperty` is worth a note, because it reads like a live user
API — `circuit.r1` returning a typed reference into the netlist. It dispatches on
`SpCircuit`, so no value of any type reachable today could have hit it, and its
`Param` branch returns a `SpRef` tagged `Parameter`, a name the `RefKind` enum
never defined. That branch would have thrown `UndefVarError` on its first
execution.

### The one thing that had to move

`is_ambiguous` lived in `query.jl` but is load-bearing for the *MNA* path:
`cg_net_name!` and `cg_model_name!` call it to rename a net or model whose SPICE
name lands in more than one namespace. It moved to the top of `codegen.jl`, next
to its two callers.

Two more names read as legacy but are live and were left alone: `spicecall` is
shared — `cg_mna_instance!` uses it for diodes, MOSFETs and BJTs — and
`UnimplementedDevice` is what `sema.jl` returns as a `GlobalRef` when no model
resolves.

### Not swept up

The rest of the Cedar-era surface is untraced and stays: `ParamSim` and the
netlist-text `alter(io, ast, ::ParamSim)` in `src/spectre.jl`, the
`CircuitElement`/`AbstractSim` abstract types, and `SimSpec`. `SimSpec` and the
`Cadnip.spec` ScopedValue in particular are *not* dead — `temper()` and
`var"$time"` in `src/spectre_env.jl` read them, and MNA-generated expressions can
reach those. Nothing writes them, which is finding 3 in `doc/FINDINGS.rst`.

### Worth porting, not reinventing

Four designs died with the code. None of them ever *ran* — that is the whole
point of this section — but each is a considered answer to a problem the MNA
path still has, and rediscovering them from scratch would be wasted work. The
deleted source is one command away: `git show 991c27a^:src/spc/codegen.jl`, and
likewise for `query.jl`, `generated.jl`, `interface.jl`, `sema.jl`.

Everything below was measured against the MNA path as of this writing, not read
off the source.

#### Model binning — the substantive one

**Done** — the codegen half is in, with `doc/model_binning.md` as its design and
`test/binning.jl` as its test. The port did not keep the runtime half as it
stood: the window bounds moved off the model card and into the bin, which is
what lets a device with no `LMIN` field be binned. The section below is the
survey that motivated it, left as written.

`cg_model_def!` + `codegen!` were the only producer of `BinnedModel` in the
repo. The **whole runtime half survives** in `src/spectre.jl` and now has nothing
calling it:

| what | where | note |
|---|---|---|
| `BinnedModel{B<:Tuple}` | `spectre.jl:377` | holds `scale` + the bin tuple |
| `find_bin(bm, l, w)` | `spectre.jl:444` | `@assume_effects` -tuned so it concrete-evals; `test/compiler_sanity.jl` existed to pin exactly that |
| `NoBinExpection` | `spectre.jl:437` | the diagnostic for an out-of-range geometry |
| `(bm::BinnedModel)(; l, w, …)` | `spectre.jl:456` | dispatch to the winning bin |
| `spicecall(bm::BinnedModel; l, w, …)` | `spectre.jl:509` | the `spicecall` entry |

Sema already does the front half too: `binning_rx` matches `nch.1`/`nch.2`, and
`provided_binned_models` (`sema.jl:786`) stops the base name `nch` being treated
as an unresolved exposed model. So the missing piece is exactly the codegen
step: aggregate the numbered cards under the base name and emit a `BinnedModel`.

What the MNA path does today with a binned deck — measured:

```
.model dm.1 d is=1e-14 lmin=0  lmax=1u  wmin=0 wmax=1u
.model dm.2 d is=2e-14 lmin=1u lmax=10u wmin=0 wmax=1u
D1 k 0 dm
```
```
MethodError: no method matching sp_diode(; is::Float64, lmin::Float64,
                                          lmax::Float64, wmin::Float64,
                                          wmax::Float64)
```

It passes the four binning parameters through to the device constructor as if
they were ordinary model parameters. The deleted code treated them as magic —
uppercased rather than case-matched against the model's fieldnames — precisely
because they are not the model's.

Two reasons this is worth real effort rather than a note: every foundry PDK bins
by L/W, so it blocks the sky130/gf180/ihp coverage the Production-readiness
pillar asks for; and there is **no binned `.model` card anywhere in the test
suite**, which is why nothing ever noticed. A port needs a test either way.

#### Netlist source positions in generated code

The legacy path pushed a `LineNumberNode(instance)` — built from the `SNode`, so
carrying the netlist's own file and line — ahead of each device it emitted.

The MNA path emits none. Measured on a three-device deck, the generated builder
contains 29 `LineNumberNode`s and every one of them has
`file = src/spc/codegen.jl`: they are artifacts of the `quote` blocks in the
codegen itself, not netlist positions. So a runtime error inside a stamp points
at the compiler, never at the line of SPICE that caused it.

Cheap to port and it improves every error a deck can raise.

#### A guard against two simultaneously active instances

For an instance name defined in more than one conditional branch, `codegen!`
emitted a counter, incremented it inside each branch, and raised
`"Multiple simultaneously active instances of $name"` if more than one fired.
The MNA path (`process_instance`) emits the branches independently with no
counter, so a deck whose `.if`/`.elseif` conditions both hold stamps the same
name twice, silently.

Port the *idea*, not the code: the legacy version read
`cond_syms[abs(instance.cond)]` unconditionally, which `BoundsError`s at
`abs(0)` on a set mixing conditional and unconditional definitions of one name —
a case the MNA path already handles correctly.

#### `.option gmin` / `.option scale`, and the `isdefault` precedence

`codegen!`'s option block was the only consumer of `sema.options[:gmin]` and
`[:scale]`. Measured: neither reaches the generated MNA builder — of the options
sema collects, the MNA path consumes `temp` alone. That is finding 2's practical
effect in `doc/FINDINGS.rst`.

The block is also the reference implementation of the precedence question §1 of
`doc/FINDINGS.rst` raises for `temp`: each option was emitted as
`isdefault(old_options.<opt>) ? <card value> : old_options.<opt>`, i.e. the card
fills in only what the caller left at its default. Whether that is the right
rule is a live argument (#275 decided against it for `temp`), but it is worth
knowing the shape existed before re-deriving it.

#### Lower priority: netlist introspection

`query.jl` gave `circuit.r1` → an `SpRef` tagged `Param`/`Model`/`Subckt`/
`Instance`/`SPNet`/`Ambiguous`, with a `show` that printed the defining netlist
line with its file, line number, and the nets coloured by degree. A genuinely
nice idea for exploring a deck, and nothing replaces it.

Temper the enthusiasm with what it actually was, though: it dispatched on
`SpCircuit`, so it never ran, and its parameter branch returned a `SpRef` tagged
`Parameter` — a name the `RefKind` enum never defined and that has no binding in
`Cadnip`, so that branch would have thrown `UndefVarError` the first time it
executed. It is a sketch to work from, not an implementation to restore.

`doc/FINDINGS.rst` separately asks for the more immediately useful cousin:
`node_names(sol)` / `branch_names(sol)`, classifying names in a *solution*
rather than in the netlist.
