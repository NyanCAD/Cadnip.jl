# C6288 Benchmark Bottleneck — Investigation

**Status:** identified. The c6288 (16x16 multiplier, ~10k PSP103 MOSFETs)
benchmark blocks for ≥10 minutes inside Julia's first-call JIT and never
reaches the solver. The profile run was killed at 7+ minutes still inside
`build_with_detection`'s first `circuit.builder(...)` call — this is the
JIT bottleneck, not the simulator.

## What the user sees

```
~/.juliaup/bin/julia --project=. benchmarks/vacask/c6288/cedarsim/runme.jl
```

…hangs for 10+ minutes with no output, then either OOMs or is killed. Status
docs (`benchmarks/vacask/cedarsim/STATUS.md`, `doc/c6288_comparison.md`)
attribute it to "DC failure", "sparse Jacobian pattern mismatch", or similar.
**That diagnosis is stale.** Those issues were fixed (jac_prototype is now
unified G+C pattern; CedarUICOp is wired up). The current blocker is upstream
of any of that.

## The numbers

Profiled with `julia --project=benchmarks benchmarks/vacask/c6288/cedarsim/profile_c6288.jl`
(see `profile_c6288.jl` in this directory):

| phase                               | time          | notes |
|-------------------------------------|---------------|-------|
| `parsefile`                         | 0.35 s        | |
| `Cadnip.sema`                       | 1.09 s        | |
| `_make_mna_circuit_with_sema`       | 1.47 s        | builds the AST |
| `eval(quoted)`                      | 2.72 s        | adds bindings; JIT deferred |
| **first builder call** (inside `build_with_detection`) | **>7 min, killed** | LLVM register allocator |

`gdb -p <pid> -batch -ex 'thread apply all bt'` on the stuck process shows the
main thread is in:

```
llvm::RAGreedy::calcGapWeights(...)
llvm::RAGreedy::tryLocalSplit(...)
llvm::RAGreedy::selectOrSplitImpl(...)
llvm::RegAllocBase::allocatePhysRegs()
llvm::RAGreedy::runOnMachineFunction(...)
[ Julia ] _jl_compile_codeinst → jl_add_to_ee
[ Julia ] jl_compile_method_internal at gf.c:2538
```

i.e. **LLVM's greedy register allocator local-splitting on a single huge
function**. This is the Julia 1.11 code path for compiling a generated
function into the JIT execution engine. RAGreedy is super-linear in basic
block size; for a body that's a single straight-line block of thousands of
calls, this is where the time goes.

## Why this function is so big

`runme.sp` includes `multiplier.inc`, which defines a single SPICE subckt
`c6288` containing **2419 gate instances** (256 AND, 2128 NOR, 32 NOT). At
codegen time `_make_mna_circuit_with_sema` (`src/spc/codegen.jl:3094`) emits a
single `c6288_mna_builder(lens, spec, t, ctx, port_args..., parent_params, x,
_mna_prefix_; ...)` function whose body has **2419 SubcktCall expressions**.

Each one (per `cg_mna_instance!(state, ::SNode{SP.SubcktCall}, ...)` at
`src/spc/codegen.jl:1542-1546`) expands to:

```julia
let subckt_lens = Base.getproperty(var"*lens#", :xAND2_1)
    and_mna_builder(subckt_lens, spec, t, ctx,
                    port_a, port_b, port_c,        # port_exprs
                    (;),                            # parent_params NamedTuple
                    x,
                    _mna_prefix_ == Symbol("") ? :xAND2_1 :
                        Symbol(_mna_prefix_, "_", :xAND2_1);   # _scoped_sym_expr
                    _mna_h_=_mna_h_, _mna_h_p_=_mna_h_p_,
                    # explicit kwargs from the call (none here)
                    )
end
```

So per call the IR carries: 1 `getproperty`, 1 ternary, 1 `Symbol(...)` concat,
1 NamedTuple literal, 1 keyword-call lowering. **× 2419 = ~12k IR
operations in one basic block** before LLVM. `c6288_mna_builder` is the
fattest function in the program by an order of magnitude. Python-level
greedy register allocation on it doesn't terminate on a 16-core box in
reasonable time.

Two more multipliers stack on top:

1. **`build_with_detection` calls the builder 5 times** (`src/mna/solve.jl:
   1411-1440`). Even after JIT is paid once, passes 2-5 each go through
   `MNAContext` re-stamping which still hits push! into G_I/G_J/G_V vectors
   per stamp.

2. **`tran!` calls `build_with_detection` AGAIN** through
   `SciMLBase.ODEProblem(circuit, ...)` at `src/mna/solve.jl:1678`, then
   `compile_structure(...)` runs the builder a 6th time. Plus
   `setup_simulation()` in `runme.jl:50` calls `MNA.assemble!(circuit)` which
   itself calls `build_with_detection` (`src/mna/solve.jl:1392`). So the
   builder runs **12 times** between MNACircuit construction and the first
   timestep. The first call pays JIT; the rest pay re-stamp cost only.

## What's downstream (so the blocker is unambiguous)

These were *not* reached but matter once JIT is solved:

### KLU sparsity-mismatch claim is stale

`STATUS.md` says KLU rejects the Jacobian because G and C have different
sparsity patterns. That's wrong as of `compile_structure`
(`src/mna/precompile.jl:404-411`):

```julia
jac_pattern = sparse(vcat(G_I_resolved, C_I_resolved),
                     vcat(G_J_resolved, C_J_resolved), ...)
G = _pad_to_pattern(G_raw, jac_pattern)
C = _pad_to_pattern(C_raw, jac_pattern)
```

G and C are padded to a unified pattern. `fast_jacobian!` writes
`J_nz[i] = G_nz[i] + γ*C_nz[i]` over nz arrays of identical structure
(`src/mna/precompile.jl:557-574`). `jac_prototype = copy(cs.G)` is the same
pattern. KLU has nothing to complain about.

### `runme.jl` defaults to a solver that *will* OOM at 154k vars

```julia
solver_name = length(ARGS) >= 1 ? ARGS[1] : "Rodas5P"
solver = Rodas5P()    # no linsolve specified
```

`benchmarks/vacask/c6288/cedarsim/runme.jl:88-94`. Bare `Rodas5P()` uses
**dense LU**. Jacobian = 154k × 154k × 8 B = **189 GB** — instant OOM. The
top-level `run_benchmarks.jl` script gets this right
(`KLUFactorization()`); the standalone `runme.jl` does not. Even after JIT
finishes, running `runme.jl` with no args will OOM-kill.

### `CedarUICOp` warmup builds an ODE problem with no jac

`src/mna/dcop.jl:336-337`:

```julia
warmup_f = SciMLBase.ODEFunction(warmup_rhs!; mass_matrix=cs.C)   # no jac=, no jac_prototype=
warmup_prob = SciMLBase.ODEProblem(warmup_f, prob.u0, warmup_tspan, ws)
warmup_int = init(warmup_prob, ImplicitEuler(); ...)
```

`ImplicitEuler` with no `jac` defaults to `AutoForwardDiff()`. That AD pass
through 10k PSP103 stamps with Dual{Tag} vectors will retrigger the same
inference / specialization costs that `doc/psp103_noinline_investigation.md`
fought to eliminate (the c6288 case wasn't in scope for that fix). Even if
inference behaves, evaluating Jacobian columns over 154k vars without a
sparse coloring is impractical.

The main solve path passes `jac=jac!, jac_prototype=copy(cs.G)`
(`src/mna/solve.jl:1609-1615`); the warmup path does not. Pass the same
through to `warmup_f`.

### Internal-node bloat — the system is roughly 2× larger than it needs to be

PSP103's `PSP103_module.include` declares 17 internal nodes per device
(`NOI, GP, SI, DI, BP, BI, BS, BD, INT1..INT9`). **All 17 are allocated
unconditionally** by `generate_mna_stamp_method_nterm` (`src/vasim.jl:3086-
3117`). Each gets a `gmin` stamp on its diagonal (`src/vasim.jl:3120-3131`).
With ~10k MOSFETs that's ≈170k internal nodes ⇒ 154k system size. jax-spice
collapses them, lands at 86k.

`detect_short_circuits` (`src/vasim.jl:2535`) only catches the explicit
`if (...) V(int, ext) <+ 0` pattern. PSP103 has none. Many of the INT* nodes
*are* unused when `SWNQS=0` (default) — they exist for non-quasi-static
analysis only — but the codegen has no way to know. This is structural
overhead per device that no per-step fast-path can fix.

## What to do about it (in order)

1. **Split `c6288_mna_builder` into chunks.** In `_make_mna_circuit_with_sema`,
   when a subckt has more than ~100 instance calls, group the calls into
   helper functions (e.g. `c6288_mna_builder_chunk_1(...)`,
   `c6288_mna_builder_chunk_2(...)`). Each chunk gets its own LLVM module and
   register allocator pass, so RAGreedy stays linear-time. This is the only
   change that unblocks c6288 specifically; everything else after this is
   refinements that also help smaller circuits.

   Lower-effort variant: compile the c6288 case with `-O0`. Confirmed locally
   that with `-O0` Julia uses LLVM's *fast* register allocator instead of
   RAGreedy. We don't ship `-O0` to users, but it's a useful confirmation
   knob and a stopgap for benchmark CI.

2. **Shrink the per-instance IR template.** The `_scoped_sym_expr` ternary
   (`src/spc/codegen.jl:655-657`) inserts a runtime `_mna_prefix_ ==
   Symbol("")` check at every site. For a top-level non-subckt call,
   `_mna_prefix_` is statically `Symbol("")` (set on entry to the top-level
   builder, `src/spc/codegen.jl:3183`). Specialize the codegen to emit the
   bare `:xAND2_1` literal at the top level — saves 2419 ternaries plus
   2419 `Symbol(...)` calls in c6288_mna_builder.

3. **Wire `jac=jac!`, `jac_prototype` through `CedarUICOp`'s warmup.** Edit
   `SciMLBase.initialize_dae!(::IDAIntegrator, ::CedarUICOp)`
   (`src/mna/dcop.jl:314`) and the matching `ODEIntegrator` overload. Without
   this, even after JIT is solved, the warmup integrator will burn through
   AD compile and Jacobian-by-columns runtime.

4. **Fix `runme.jl` default solver** to `Rodas5P(linsolve=KLUFactorization())`
   or `IDA(linear_solver=:KLU)`. Add a bare-Rodas5P guard somewhere so users
   don't OOM their machines.

5. **Skip duplicate `build_with_detection` runs** in `tran!` when
   `assemble!(circuit)` was already called. Cache the detection ctx on the
   `MNACircuit` (currently each entry point rebuilds from scratch).

6. **PSP103 internal-node short-circuiting.** Add a manual annotation
   mechanism (or a flag-based pass) so that `INT1..INT9` are aliased to
   ground when `SWNQS == 0`. ~170k → ~80k variables, ~halves KLU
   factorization cost and quarters memory. Requires a Cadnip patch (the VA
   model has the structural information; codegen has to act on it). Not a
   blocker for c6288 to *run*, but a 2-4× wall-time speedup once it does.

## Why the prior "JIT compile time" story misses the actionable detail

`doc/ring_oscillator_investigation.md` reported the ring takes ~109s for the
PSP103 builder's first call. The natural read is "each PSP103 stamp costs
something, scale by N". But for c6288, the cost is **not in the per-PSP103
stamp**, which is precompiled in PSPModels and dispatched via `invoke`
(`doc/psp103_noinline_investigation.md`). The cost is in the **outer
generated function** that wires 2419 calls together — and that function
doesn't exist for the ring at this scale. The right mental model is:

- Per-device JIT cost: amortized, fixed per device *type*
- Outer-builder JIT cost: scales with **instance count in the largest
  enclosing subckt** because that's the function size LLVM has to compile

Ring oscillator: largest enclosing subckt has 9 stages × 2 transistors = 18
instances. C6288: largest enclosing subckt has 2419 instances. The growth
isn't 18 → 10000 (ratio 555×); it's 18 → 2419 (ratio 134×). And it hits a
non-linearity in LLVM's register allocator at that size.

## References

- `benchmarks/vacask/c6288/cedarsim/profile_c6288.jl` — phase-by-phase
  profiler that produced the table above
- gdb backtrace from `pid` of stuck `julia` process during build_with_detection
  — RAGreedy::calcGapWeights confirms the LLVM register-allocator hypothesis
