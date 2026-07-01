# C6288 Benchmark Bottleneck — Investigation

**Status:** RESOLVED. `benchmarks/vacask/c6288/cedarsim/runme.jl` now runs
end to end under `julia -O0` with `CedarDCOp` initialization: setup
completes in ~90s, DC solve finds an operating point via GMIN/source
stepping, and the transient reaches `retcode=Success` (17 accepted / 5
rejected steps over the 2ns window in one validation run). See "Resolution"
near the end of this document for the fixes and the remaining `-O0`
requirement. The section below is kept as the original investigation log.

---

The c6288 (16x16 multiplier, ~10k PSP103 MOSFETs)
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

Two profiles. Both used `julia --project=benchmarks benchmarks/vacask/c6288/cedarsim/profile_c6288.jl`.

### Default (`julia` ≡ `-O2`): hangs in LLVM RAGreedy

| phase                               | time          | notes |
|-------------------------------------|---------------|-------|
| `parsefile`                         | 0.35 s        | |
| `Cadnip.sema`                       | 1.09 s        | |
| `_make_mna_circuit_with_sema`       | 1.47 s        | builds the AST |
| `eval(quoted)`                      | 2.72 s        | adds bindings; JIT deferred |
| **first builder call** (inside `build_with_detection`) | **>7 min, killed** | LLVM register allocator |

`gdb -p <pid> -batch -ex 'thread apply all bt'` on the stuck process shows
the main thread is in:

```
llvm::RAGreedy::calcGapWeights(...)
llvm::RAGreedy::tryLocalSplit(...)
llvm::RAGreedy::selectOrSplitImpl(...)
llvm::RegAllocBase::allocatePhysRegs()
llvm::RAGreedy::runOnMachineFunction(...)
[ Julia ] _jl_compile_codeinst → jl_add_to_ee
[ Julia ] jl_compile_method_internal at gf.c:2538
```

LLVM's greedy register allocator local-splitting on a single huge function.
RAGreedy is super-linear in basic block size; for a body that's straight-line
with thousands of inlined calls, it doesn't terminate.

### `julia -O0`: completes — RAGreedy was the hangup

`-O0` switches LLVM to the *fast* register allocator. Same profile, same
PSPModels, same circuit:

```
parse:                    0.18 s
sema:                     0.80 s
codegen (build expr):     1.04 s
eval (compile):           3.36 s     # bindings only; deferred JIT
MNACircuit ctor:          0.00 s
build_with_detection:    58.31 s     # 5 passes; pass 1 includes JIT
  system size:          212228 vars
  n_nodes:               90850
  n_currents:            70818
  n_charges:             50560
  G_I COO entries:     4661828
  C_I COO entries:       819072
  internal nodes:        80896
compile_structure:        2.20 s
  G nnz:               2452148
  C nnz:               2452148
create_workspace:         0.03 s
fast_rebuild! (1st):     25.95 s     # DirectStampContext JIT, separate MI
fast_rebuild! (warm):    0.480 s
fast_residual! warm:     0.657 s
fast_jacobian! warm:     1.017 s
KLU full solve (1st):    2.95 s
KLU full solve warm:     2.72 s
setup total: 91.9 s
```

Total setup at -O0: **92 s**. So the -O2 hang is genuinely RAGreedy, not an
infinite loop in our code.

These numbers also overturn two assumptions in `STATUS.md` and
`c6288_comparison.md`:

- **System size is 212k vars, not 154k.** Both prior docs say "154k"; the
  actual MNA size after detection is 212,228 (90,850 nodes + 70,818 currents
  + 50,560 charges). Internal nodes alone = 80,896 — confirming the PSP103
  bloat thesis below.
- **There are *two* huge JIT compiles, not one.** `build_with_detection`
  triggers `c6288_circuit(MNAContext)` JIT (~50s of those 58s), and the
  *first* `fast_rebuild!` triggers `c6288_circuit(DirectStampContext)` JIT
  (25.95s). Julia specializes the builder per ctx type. Both functions have
  the 2419-call body and both hit RAGreedy at -O2.

### Per-Newton-step cost (after JIT, at -O0)

```
fast_rebuild!  (10k device stamps + nzval writes):  0.48 s
fast_residual! (= rebuild + 2× mul! sparse+vec):    0.66 s
fast_jacobian! (= rebuild + nzval combine):         1.02 s
KLU factorize + solve (n=212k, nnz=2.45M):          2.72 s
```

At -O2 (when JIT terminates), `fast_rebuild!` and the matvec ops should be
~10× faster (LLVM optimizing PSP103 stamps' inner loops). KLU factorize+solve
is C code in `Sundials`, so the 2.72s is essentially solver-floor regardless
of `-O0`/`-O2`. **One Newton iteration ≈ 3-4 seconds even after warmup**;
50 iterations through a few hundred timesteps puts c6288 in the 5-15 minute
*solve* range on top of the ~2-min JIT.

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

### Internal-node bloat — system is 212k vars (verified), should be ~half

PSP103's `PSP103_module.include` declares 17 internal nodes per device
(`NOI, GP, SI, DI, BP, BI, BS, BD, INT1..INT9`). **All 17 are allocated
unconditionally** by `generate_mna_stamp_method_nterm` (`src/vasim.jl:3086-
3117`). Each gets a `gmin` stamp on its diagonal (`src/vasim.jl:3120-3131`).
The profile reports **80,896 internal nodes** out of 90,850 total nodes —
so 89% of every voltage row is internal-to-PSP103 and barely participates
in the dynamics.

KLU pays for it: 2.45M nnz in G+C, 2.72s/factorize+solve. Halve the
internal-node count and KLU drops roughly proportionally.

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

## Update: with `-O0`, what's the *next* blocker?

Verified by actually running `julia -O0 --project=benchmarks
benchmarks/vacask/c6288/cedarsim/runme.jl Rodas5P` end to end (not just the
profiler). Two things:

**Fixed en route:** `runme.jl` itself was broken independent of `-O0` — it
referenced `sp_psp103va_module`, which doesn't exist in `PSPModels`
(`PSP103VA_module` is the exported name). This threw `UndefVarError` before
anything else ran. Fixed by dropping the stale alias.

**Real next blocker, confirmed live via gdb:** once the script actually gets
past setup (`build_with_detection`/`compile_structure`/first `fast_rebuild!`
all complete in the ~75-90s the profiler predicted — `-O0` genuinely clears
RAGreedy), the process hangs again, this time at runtime, not JIT. Repeated
`gdb -p <pid> -batch -ex 'thread apply all bt'` snapshots over 10+ minutes
all show the same stack:

```
_insert! () at .../SparseArrays/src/sparsematrix.jl:3218
julia__setindex_scalar!_... () at .../SparseArrays/src/sparsematrix.jl:3206-3207
insert! () at array.jl:1746
julia__growat!_... () at array.jl:1162
memmove () at cmem.jl:28
```

i.e. `SparseArrays.setindex!` falling into its scalar insertion path
(`_insert!`, which shifts `colptr`/`rowval`/`nzval` to make room for one new
stored entry at a time) over and over, for a 212,228×212,228 matrix with
~2.45M final nnz. Each such insertion is `O(current nnz)`, so building up
the structure entry-by-entry this way is `O(n²)`-ish — consistent with the
process still running (and still making progress, not deadlocked) after 10
minutes before it was killed.

This is exactly item 3 from the "what to do about it" list above:
`CedarUICOp`'s pseudo-transient warmup (`SciMLBase.initialize_dae!(...,
::CedarUICOp)` in `src/mna/dcop.jl:310-389`) runs `ImplicitEuler(autodiff=
ADTypes.AutoFiniteDiff())` for `warmup_steps` fixed timesteps. The main
`tran!` path does wire `jac_prototype` through for the outer solver
(`src/mna/solve.jl:1721-1725`, `1611-1615`), but `AutoFiniteDiff()` with no
declared sparsity/coloring makes OrdinaryDiffEq's Jacobian machinery
(`calc_J`/`calc_W` in `OrdinaryDiffEqDifferentiation`) build/refresh the
Jacobian without exploiting the shared sparsity pattern — degenerating into
scalar writes that repeatedly grow the sparse structure instead of reusing
existing storage.

**Net effect:** clearing the LLVM RAGreedy hang with `-O0` does not make
c6288 runnable end-to-end. It trades a ~10-minute-plus JIT hang for a
different runtime hang of the same order (or worse, given `O(n²)` scaling),
inside the `CedarUICOp` warmup's finite-difference Jacobian handling. Fixing
this requires giving the warmup's `ImplicitEuler` a proper sparse
`jac_prototype`/coloring (or an analytic `jac=`) instead of relying on
`AutoFiniteDiff()` to rediscover the 2.45M-entry sparsity pattern column by
column at runtime.

## Resolution

Four real bugs, found and fixed by actually running the benchmark end to
end (not just profiling setup) and reading gdb/Julia stack traces off the
stuck process, plus one initialization-algorithm swap:

1. **`runme.jl` itself was broken**, independent of everything else below:
   it referenced `sp_psp103va_module`, which doesn't exist in `PSPModels`
   (the export is `PSP103VA_module`). Fixed by dropping the stale alias.

2. **`CedarUICOp`'s DAE (IDA) warmup had no Jacobian at all**
   (`src/mna/dcop.jl`) — `ODEFunction(warmup_rhs!; mass_matrix=cs.C)` with no
   `jac`/`jac_prototype`, forcing `ImplicitEuler` to rediscover the
   Jacobian via autodiff/finite-diff. Fixed by adding an analytic
   `warmup_jac!` (`d(du)/du = -G`) with `jac_prototype=copy(cs.G)`.

3. **The main ODE `jac!` was `O(n²)` on sparse circuits, not zero-alloc**
   (`src/mna/solve.jl`) — it looped `for i in eachindex(J, G); J[i] = -G[i]; end`.
   For `SparseMatrixCSC`, `eachindex` over two sparse matrices is
   `IndexCartesian`, i.e. `CartesianIndices` over the *full* `n²` index
   space, not the ~nnz stored entries. At c6288's `n=212228` that's
   ~4.5e10 iterations, each a scalar sparse write that can insert a new
   structural entry (`O(current nnz)` per insert). This is what was
   actually hanging after fix (2) — confirmed by sending `SIGINT` to the
   stuck process and reading the resulting Julia stack trace, which pinned
   it to `jac!` calling into `SparseArrays.setindex!`'s scalar path.
   First fix attempt walked `nonzeros(J)`/`nonzeros(G)` directly (assuming
   matching patterns); a follow-up (`0540661`) switched to `copyto!`
   after that assumption broke under (4) below (see next point).

4. **`ShampineCollocationInit`'s default `nlsolve` OOMs on large sparse
   circuits** (`src/mna/dcop.jl`, all four call sites) — with no `nlsolve`
   specified, `OrdinaryDiffEqNonlinearSolve.default_nlsolve` picks
   `FastShortcutNonlinearPolyalg`, whose leading algorithms are dense
   QuasiNewton methods (Broyden/Klement) that allocate an `n×n` dense
   approximate-Jacobian regardless of the sparse `jac`/`jac_prototype`
   already threaded through. At c6288's scale that's a ~360GB allocation →
   instant `OutOfMemoryError`. Fixed by passing
   `nlsolve=NewtonRaphson(autodiff=nothing)` explicitly, so it uses the
   supplied sparse Jacobian directly. This changed `J`'s provenance (now
   built by `NonlinearSolveBase.JacobianCache` rather than a straight
   `deepcopy` of `jac_prototype`), which is what broke the `nonzeros`-based
   fix in (3) — `J` and `G` no longer reliably shared identical stored-entry
   counts (observed: 677700 vs. 2452148), hence the `copyto!`-based fix.

With all four in place, `runme.jl` reaches the solver: no more JIT hang, no
OOM, no dimension mismatch — but `retcode=InitialFailure` with only the
`t=0` point saved. `CedarUICOp`'s warmup (fixed `dt=1e-12`, `adaptive=false`,
10 steps by default) turned out to make **zero progress**: `|Δu|=0.0` and
the DC-style residual `‖G(u)u - b(u)‖` identical to six decimal places
across 300 steps in a targeted diagnostic — Newton fails identically on the
very first attempted step and never moves `u` away from zero. Because
`adaptive=false`, `force_dtmin=true` has nothing to act on: there is no
dt-shrinking retry loop the way ngspice's real `uic` transient stepper has
(ngspice's `.control ... tran 2p 2n uic` skips DC op entirely and leans on
its own per-step Newton-with-timestep-halving; SciML's initialization
framework instead requires a consistent `t=0` state up front). More warmup
steps at a fixed `dt` would not have helped — the fixed-dt, no-retry
approach itself is the wrong tool here, not an insufficient step count.

**Swapping `CedarUICOp` for `CedarDCOp`** (which already implements a
GMIN-stepping/source-stepping homotopy chain in
`_dc_solve_with_fallbacks` — the same class of technique vacask/ngspice use
to find an operating point for exactly this kind of circuit) resolves it:
a validation run reached `retcode=Success` with 17 accepted / 5 rejected
steps over the full 2ns window. `runme.jl` now uses `CedarDCOp()`.

**The `-O0` requirement remains and is not expected to go away** without
the deeper codegen work in "What to do about it" item 1/2 above (chunking
the generated builder so LLVM's register allocator stays linear-time). CI
therefore runs the Cadnip c6288 benchmark as its own `julia -O0` step,
separate from `run_benchmarks.jl` (which runs the other circuits at
default optimization) — see `.github/workflows/benchmark.yml`.

## References

- `benchmarks/vacask/c6288/cedarsim/profile_c6288.jl` — phase-by-phase
  profiler that produced the table above
- gdb backtrace from `pid` of stuck `julia` process during build_with_detection
  — RAGreedy::calcGapWeights confirms the LLVM register-allocator hypothesis
- gdb backtraces from `pid` of stuck `julia -O0` process during the
  `CedarUICOp` warmup — `SparseArrays._insert!`/`setindex_scalar!` confirms
  the scalar-sparse-insertion hypothesis above
- `SIGINT`-induced Julia stack trace pinning the post-fix hang to
  `jac!` at `src/mna/solve.jl` calling `SparseArrays.setindex!`'s scalar path
- Targeted diagnostic script replaying `CedarUICOp`'s warmup loop with a
  DC-residual printout per step, showing zero movement across 300 steps
- Diagnostic run of `tran!` with `CedarDCOp()` in place of `CedarUICOp()`,
  reaching `retcode=Success`
