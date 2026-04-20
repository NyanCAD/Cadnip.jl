# Claude Development Notes

## Environment

### Local Development (Native)

When running on a local machine with full system access:

- **Use Julia 1.12** - better compilation performance for long functions (like VA models)
- Julia is typically pre-installed via juliaup, just use `julia` command
- Full system resources available - no memory limits
- For fresh setup: `juliaup add 1.12 && juliaup default 1.12`

### Web/Sandbox Environment (Claude Code Web)

When running in gVisor-sandboxed environment (check with `uname -r` showing `runsc` or kernel 4.4.0):

- **Julia is NOT pre-installed** - install juliaup first:
  - Run: `curl -fsSL https://install.julialang.org | sh -s -- -y`
  - Then source the profile: `. ~/.bashrc`
  - Add Julia 1.11: `~/.juliaup/bin/juliaup add 1.11`
  - Set as default: `~/.juliaup/bin/juliaup default 1.11`
  - Use `~/.juliaup/bin/julia` to run Julia (full path required)
- **Use Julia 1.11** - more stable in sandbox environment
  - Julia 1.12 has threading bugs that cause segfaults during artifact downloads in gVisor
- **Memory limited** - large VA model compilations (PSP103VA with 200+ params) may OOM
- **Precompilation issues** - may need to disable compile workloads

**Fix for precompilation segfaults:** Create `test/LocalPreferences.toml`:

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

The `sp"..."` / `spc"..."` / `va"..."` macros expand at the call site and
work in both contexts.

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
`ParamLens` navigates hierarchical params and merges overrides at `.params` fields:
```julia
# Flat circuit params
circuit = MNACircuit(my_circuit; params = (R1=100.0, R2=200.0))
altered = alter(circuit; var"params.R1"=150.0)

# Hierarchical subcircuit params
circuit = MNACircuit(my_circuit; inner = (params = (R1=100.0,),))
altered = alter(circuit; var"inner.params.R1"=200.0)
```
The `lens(; defaults...)` call merges with `lens.nt.params` if present.

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
