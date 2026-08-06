# Codegen unification

Findings from a README review that measured what the generated code actually
needs. Three related pieces of work: one shared import list, one lowering
instead of two, and two runtime warnings. All line numbers are against
`src/spc/codegen.jl` and `src/spc/interface.jl` as of this writing.

**Status.** All four are done. §1 and §2: the PDK path calls
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

Three warnings, not two — a third of the same family turned up while measuring.

**The builder read-back.** `_eval_deck_into_module` now *returns* the builder:
its `:toplevel` sequence ends with the circuit name, so the read is a top-level
statement of its own, in the world the definition created. `MNACircuit(path)`
takes what `Base.include` hands back instead of reading the binding itself, and
the inline-string branch does the same with `Expr(:toplevel, builder_code,
eff_name)`. `Base.include(mod, SpiceFile(...))` returning the builder is also the
more conventional thing for an `include` method to do.

**`SciMLBase.MatrixOperator`.** Nothing in the repository referenced it — the
import was the only mention. Deleted.

**`getglobal` on a freshly-eval'd Verilog-A type.** Codegen reads a model's Julia
type to case-fold its parameter names (`T = getglobal(ref.mod, ref.name)`, five
sites). For a `.hdl` device that type was defined by a module eval'd during this
same codegen pass, so the read is from a world older than the binding:

```
WARNING: Detected access to binding `TM2D_module.TM2D` in a world prior to its
definition world.
```

Here the world genuinely cannot be advanced first — codegen is mid-expression —
so this is the case for Julia's own hint. `_globalref_value(ref)` wraps the read
in `invokelatest`, once, for all five.

Also gone: `_eval_builder_into_module`, dead since the deck-module change, whose
body was the very `Base.eval`-then-`getfield` pattern above.

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

With that, the deck path emits no imports at all and `_baremodule_prelude` is
just the PDK `baremodule`'s `Base` essentials.

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
