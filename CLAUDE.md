# Claude Development Notes

## Environment

**Use Julia 1.12.** Better compilation performance for long functions (like VA
models), and it builds a full c6288 (212k-variable, PSP103-heavy) circuit with
no trouble.

- On a local machine Julia is usually pre-installed via juliaup — just use `julia`.
- In a fresh cloud/remote session Julia is typically not pre-installed. Install it:
  ```bash
  curl -fsSL https://install.julialang.org | sh -s -- -y
  . ~/.bashrc
  ~/.juliaup/bin/juliaup add 1.12 && ~/.juliaup/bin/juliaup default 1.12
  ```
  then run Julia via `~/.juliaup/bin/julia` (full path).
- For a fresh local setup: `juliaup add 1.12 && juliaup default 1.12`.

**Heavy VA model precompilation** (PSP103VA with 200+ params, BSIM4) can be slow
or memory-hungry. To skip the compile workloads, create
`test/LocalPreferences.toml`:

```toml
[PSPModels]
precompile_workload = false

[VADistillerModels]
precompile_workload = false
```

This file is gitignored. The packages will still work but with slower first-call latency.
See [PrecompileTools docs](https://julialang.github.io/PrecompileTools.jl/stable/) for details.

### CI Environment

- CI uses Julia 1.11 (what Manifest.toml is locked to)
- Don't add compatibility hacks for older Julia versions

## Development Guidelines

### Code Modification Philosophy

- **ALWAYS update existing code** - refactor and modify in place
- **NEVER add compatibility layers** - no deprecation wrappers, no duplicate APIs
- **NEVER create parallel implementations** - one clean API, not old + new
- We are at early stage development where breaking changes are expected
- If you need to change behavior, change it directly - don't preserve the old way

### Measure before you claim

Run it before asserting it. Claims about how this codebase behaves — what a
lens returns, which branch is reachable, whether a builder can be observed —
are cheap to check in a REPL and expensive to get wrong, because a plausible
wrong one gets written into a design doc and believed later.

Two traps that have already produced wrong claims here:

- **A synthetic probe is not real usage.** Calling `getproperty(observer, :vin)`
  by hand "showed" a phantom-child bug in `ParamObserver`; no builder does that,
  and the real call path was correct. The fix derived from the probe broke the
  `.param x1` / `X1` collision case.
- **Reading the code is not running the code.** `ParamLens.getproperty` looks
  like it returns a `ValLens` for a leaf. It cannot — canonicalization moves
  leaves under `params` first — so that branch was dead for years.

When a measurement contradicts the reasoning, the measurement wins, and the
prose that ships should be written from the measurement.

### MNA Backend Migration

- **DO NOT maintain backward compatibility with DAECompiler**
- Update existing APIs to use the new MNA backend directly
- When modifying sweep/simulation code, replace DAECompiler patterns with MNA equivalents
- Do not create duplicate types (e.g., `MNACircuitSweep`) - modify existing types instead

### Key MNA Components

- `MNACircuit`: Parameterized circuit simulation wrapper
- `MNAContext`: Circuit builder context for stamping (structure discovery)
- `DirectStampContext`: Zero-allocation context for fast restamping during solve
- `alter()`: Create new simulation with modified parameters
- `dc!()` / `tran!()`: DC and transient analysis
- `CircuitSweep`: Parameter sweep over MNA circuits

See `doc/` for design documents. Check `git log --oneline -20 --name-only` for recently changed files relevant to current work.

## CI and Testing

### Workflow

1. **Sanity check** - run the specific test file for what you changed
2. **Commit and push** - CI runs `test-core` + `test-integration` in parallel
3. **Run full tests locally** - `all` tests + benchmarks while CI runs

### Commands

```bash
# Specific test file (sanity check)
~/.juliaup/bin/julia --project=test test/mna/core.jl

# All tests (core + integration)
~/.juliaup/bin/julia --project=test test/runtests.jl all

# Benchmarks
~/.juliaup/bin/julia --project=. benchmarks/vacask/run_benchmarks.jl

# Parser tests
~/.juliaup/bin/julia --project=NyanSpectreNetlistParser.jl -e 'using Pkg; Pkg.test()'
~/.juliaup/bin/julia --project=NyanVerilogAParser.jl -e 'using Pkg; Pkg.test()'
```

### Test Files

| File | What it tests |
|------|---------------|
| `test/mna/core.jl` | MNA stamping, matrix assembly, DC/AC |
| `test/mna/va.jl` | VA contribution stamping |
| `test/basic.jl` | SPICE codegen, simple circuits |
| `test/transients.jl` | PWL/SIN sources |
| `test/sweep.jl` | Parameter sweeps |
| `test/mna/vadistiller.jl` | VADistiller models |
| `test/mna/vadistiller_integration.jl` | Large VA models (BSIM4) |
| `test/mna/audio_integration.jl` | BJT circuits |

### Test Style: prefer netlists + the high-level API

**Default to SPICE/Spectre netlists driven through the high-level API for any
test that asserts on *circuit behavior* (a DC operating point, a transient
trajectory, convergence, model-card parameter handling, an AC response).**
Reserve hand-written `stamp!` / `MNAContext` / `get_node!` builders for unit
tests that specifically exercise *low-level stamping mechanics* — matrix
assembly, COO structure, positional-counter discipline, `stamp_G!`/`stamp_C!`,
the `alloc_*` primitives, `DirectStampContext` restamping. If a test is really
about "does this circuit solve to the right answer," it should be a netlist.

Preferred (declarative; exercises the real
parser → codegen → `ModelRegistry` → solve path that production uses):

```julia
const rectifier = sp"""
V1 vin 0 DC 5
R1 vin out 1k
D1 out 0 dmod
.model dmod d is=76.9p n=1.45
"""i

sol = dc!(MNACircuit(rectifier))
@test 0.6 < sol[:out] < 0.8        # name-based access, robust to system size
```

Avoid, for behavioral tests (hand-managed nodes, `_mna_x_` threading, and
fragile positional `sol.x[2]` indexing that breaks the moment the system gains
a variable — e.g. a `$limit` limiting row):

```julia
function rect(p, s, t=0.0; x=Float64[], ctx=MNAContext())
    reset_for_restamping!(ctx)
    vin = get_node!(ctx, :vin); out = get_node!(ctx, :out)
    stamp!(VoltageSource(5.0; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, out)
    stamp!(sp_diode(), ctx, out, 0; _mna_spec_=s, _mna_x_=x)
    return ctx
end
sol = solve_dc(rect, (;), MNASpec()); @test 0.6 < sol.x[2] < 0.8
```

Why: netlists cover the parser, codegen, and two-tier model resolution that
real users hit (a hand-stamped `sp_diode()` skips all of it); `sol[:name]`
survives added state variables where `sol.x[i]` silently shifts; and the
netlist form is a fraction of the boilerplate. Load netlists at module top
level (`const c = sp"""..."""i`, or `Base.include(@__MODULE__,
SpiceFile(path))`) and pass the builder to `MNACircuit` — see **File-First
Circuit Loading** below for the world-age rules.

## Releasing

**A release is a merged version bump. Nothing else is manual.**

Bump the `version` in the package's `Project.toml` as part of the PR that
motivates it, and merge to `main`. `.github/workflows/register.yml` notices the
changed version, comments at Registrator on the merge commit, and General's
AutoMerge merges the registry PR; `.github/workflows/tagbot.yml` then creates
the git tag and GitHub release. Subpackage tags are prefixed
(`NyanSpectreNetlistParser.jl-v0.7.0`); the root package tags as `v0.14.0`.

Four packages are registered in General — `Cadnip` and the three
`Nyan*` parsers. The `models/` packages and `SpiceArmyKnife.jl` are not
registered, so bumping their versions does nothing.

**Release early.** The one thing that makes this awkward is letting changes pile
up. When a parser feature and the Cadnip code consuming it land in the same
release, Cadnip's compat bound needs a parser version that isn't registered yet,
and AutoMerge rejects it. `register.yml` handles that case — it registers in
dependency order (NyanLexers → parsers → Cadnip) and waits for each tier to
land — but the case only exists because of the gap. Release each side as it
lands and the constraint never forms.

**The bump commit message is the changelog.** `register.yml` sends the body of
the commit that changed the `version` as the release notes, so they land on the
registry PR and in the GitHub release. AutoMerge *rejects* a breaking release
that has none, so a breaking bump with an empty commit body fails the workflow
before it registers anything. Write the body as user-facing notes: what broke
and what to do about it. Bump one package per commit when a release touches
several, or they all get the same notes.

**Version choice is the other human call.** `0.x.y` treats `x` as the breaking
component: adding a struct field or an enum member is breaking, because it
changes a positional constructor or shifts later enum values. Only widen a
compat bound to a range the code actually works across — `"0.6, 0.7"` is free
when the code spans both and a lie when it reads a field only `0.7` has.

Registration requires the `REGISTRATOR_TOKEN` secret: Registrator only honours
comments from repository collaborators or public org members, so the default
`GITHUB_TOKEN` (which comments as `github-actions[bot]`) is silently ignored.

## Gotchas and Patterns

### Builder Function Signature
MNA builder functions have signature:
```julia
function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
```
- `params`: NamedTuple of circuit parameters
- `spec`: MNASpec with temp and mode
- `t`: Current time for time-dependent sources
- `x`: Solution vector for nonlinear devices
- `ctx`: MNAContext or DirectStampContext (reused across rebuilds)

### File-First Circuit Loading (canonical)
Use `MNACircuit(path)` or `Base.include(@__MODULE__, SpiceFile(path))` for
production code. The latter defines a builder function at top level and avoids
all world-age and invokelatest overhead.

A file is a *deck*, and a deck is a namespace: its generated code lands in a
module of its own and only the builder is bound in yours, so two netlists that
each define `.subckt divider` do not overwrite each other. An `sp"..."` deck goes
down the same path at macro-expansion time, so it gets a module too — including
two in one local scope.

```julia
# File path — language inferred from extension (.scs → Spectre, else SPICE)
circuit = MNACircuit("amp.sp")
sol = dc!(circuit)

# For performance-sensitive code, define the builder at top level:
Base.include(@__MODULE__, SpiceFile("amp.sp"))   # defines `amp(params, spec, ...)`
c = MNACircuit(amp; R1=1e3)
sol = dc!(c)

# Inline SPICE (sp"...") and Spectre (spc"...") string macros work too:
circuit = MNACircuit(sp"""
* divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
""")
```

For runtime-parsed netlist strings, `MNACircuit(code; lang=:spice|:spectre,
source_dir=...)` works at the REPL or module top level. It eval's the builder
on the spot. **Not safe inside a function body** — Julia freezes the caller's
world age at entry and dispatch to the freshly-defined builder would error.
Inside a function body, load the circuit at top level first and pass the
builder fn:

```julia
Base.include(@__MODULE__, SpiceFile("amp.sp"))   # top level

function run_sim()
    c = MNACircuit(amp; R1=1e3)                  # no eval, no tax
    dc!(c)
end
```

The `sp"..."` / `spc"..."` / `va"..."` macros compile their deck when the macro
expands — which is before the enclosing code runs, and before an enclosing
function is even defined — and expand to the builder itself. No world-age tax,
and nothing at the call site Julia restricts to top level, so they work in both
contexts.

### Two-tier model resolution

Device names in SPICE netlists resolve via two tiers:

- **Tier 1 — builtins (registry).** R, C, L, D, and level-dispatched MOSFETs/BJTs
  are resolved via `Cadnip.ModelRegistry.getmodel`. Populated by Cadnip +
  stdlib packages (VADistillerModels, BSIM4, PSPModels). Just `using MyPkg`
  and your `.model foo d` / `.model foo nmos level=1` picks it up.
- **Tier 2 — scope (netlist directives).** PDK-specific and custom VA devices
  are resolved from the sema scope walk list, populated by `.hdl "foo.va"`,
  `.include "foo.sp"`, `.lib "foo.sp" section`, and `jlpkg://Package/path`
  forms in the netlist. Most-recent include wins.

PDK authors expose precompiled content via `Cadnip.precompile_pdk(@__MODULE__,
"pdk.spice")` and `Cadnip.precompile_va(@__MODULE__, "device.va")`. End users
reference the baked content via `.lib "jlpkg://MyPDK/..." typical` directives
in their netlist.

### Solution access: `sol[:name]`

DC and transient solutions support name-based access via SymbolicIndexingInterface.

```julia
sol = dc!(circuit)
sol[:vout]         # scalar voltage at node :vout
sol[:I_v1]         # current through V1
sol[:gnd]          # 0.0 (ground)

# Transient:
sol = tran!(circuit, (0.0, 1e-3))
sol[:vout]         # trajectory
sol(1e-4)          # state at t
```

### ParamLens Pattern

A scope (the top level, or a subcircuit instance) has *parameters* of its own
and *children* it instantiates, in two namespaces that can collide. One rule
tells them apart: **a leaf is a parameter of the scope, a group is a child.**

```julia
circuit = MNACircuit(my_circuit; R1=100.0, R2=200.0)   # parameters of the top scope
altered  = alter(circuit; R1=150.0)

circuit = MNACircuit(my_circuit; inner = (R1=100.0,))  # R1 of instance `inner`
altered  = alter(circuit; var"inner.R1"=200.0)
```

When a name is *both* a parameter and an instance (`.param x1` next to `X1`),
`x1=2.0` sets the parameter and `x1=(rv=…)` addresses the instance; write
`params=(x1=2.0,)` to name the parameter explicitly (it outranks the flat
spelling) when you need both at once.

`ParamLens` canonicalizes on construction — `canonicalize_params` maps the
compact form users write to the `(params=(…), child=(params=(…),))` form the
lens reads, and `compact_params` maps back (it is what `ParamObserver` reports,
so an observed tree can be handed straight back as an override). Both are
`@generated`, so the whole thing folds away at compile time.

An override outranks the netlist: a subcircuit parameter the instance line
spells out (`X1 a b divider r1val=2k`) is still reachable from `alter` and from
a sweep axis. Two things are *not* reachable — device instance parameters
(`r1=(r=2k,)`, `m1=(w=…)` — parameterize the netlist with a `.param` instead),
and a name no scope declares at all, i.e. a typo — and both now throw at
construction instead of running as a no-op. Codegen emits the declared names of
each scope beside the builder (`src/mna/param_scope.jl`); a hand-written builder
emits none, so its `params` are never checked.

### Parameter Sweeps (the sweep API)

Don't hand-roll a `for`-loop that rebuilds a circuit per value — use the sweep
API. Wrap the range(s) in a sweep, bind to a builder with `CircuitSweep`, and
run `dc!`/`tran!`, which return a `SweepResult` iterating `(params, sol)` pairs:

```julia
sweep = Sweep(vac = [0.5e-3, 1e-3, 2e-3])          # one axis
# ProductSweep(a=..., b=...)  → full grid;  TandemSweep(a=..., b=...) → zipped
cs = CircuitSweep(ce_amplifier, sweep; vac=1e-3, freq=1e3)   # base params as kwargs
for (params, sol) in tran!(cs, (0.0, 5e-3))
    @test sol[:coll] ...                            # params aligned to its sol
end
```

The `CircuitSweep` kwargs (`; vac=1e-3, freq=1e3` above) set the *base* value of
everything the sweep doesn't vary; a swept axis needs no seeding of its own. Any
netlist `.param` responds: DC source values, device values, and
runtime-evaluated SIN/PULSE amplitudes alike, at top level or inside a
subcircuit. A hand-coded builder has to read `params.<name>` itself
(`merge(defaults, params)` or `ParamLens`); a swept name a *netlist* scope does
not declare throws when the sweep is built, rather than quietly giving every
point the default. Netlist
examples: `test/params.jl` (DC values, device values, instance params),
`test/design_flow.jl` (a bias sweep as a DC transfer curve), and
`test/mna/audio_integration.jl` (`.param vac` / `.param freq` on a `SIN` source).

For hierarchical/subcircuit params use `var"a.b"` selectors with a matching base
(`inner=(R1=…,)`), same as `alter` (see ParamLens above).

### SPICE Name Collisions
PDK modules use `baremodule` so SPICE names like `inv`, `log`, `exp` don't
conflict with Julia builtins. Generated code uses explicit `Base.hasfield`,
`Base.getproperty` for any Base functions needed.

### Verilog-A Gotcha
Disciplines (electrical, V(), I()) are IMPLICIT in NyanVerilogAParser.
Do NOT use `include "disciplines.vams"` - causes parser bugs.

```julia
va"""
module VAResistor(p, n);
    parameter real R = 1000.0;
    inout p, n;
    electrical p, n;
    analog I(p,n) <+ V(p,n)/R;
endmodule
"""
```
