# Ring Oscillator Benchmark Investigation

This document summarizes the investigation into why the ring oscillator benchmark "hangs and blows up."

## Summary of Findings

### 1. JIT Compilation Time (Root Cause of "Hang")

The PSP103 Verilog-A model generates a very large Julia `stamp!` method. First-call overhead:

| Phase | Time |
|-------|------|
| Model loading (parse + eval) | ~38s |
| JIT compilation of stamp! | ~150s |
| Actual stamp execution | ~0.02s |
| **Total first-call** | **~190s** |

**This is why the benchmark appears to "hang"** - it's waiting for JIT compilation, not stuck in an infinite loop.

### 2. System Properties

After assembly, the ring oscillator circuit has:
- **System size**: 371 unknowns
- **G matrix**: 2846 nonzeros, 137 zero diagonal entries
- **C matrix**: 1296 stored nonzeros, only 162 actual nonzeros (>1e-20)
- **C rank**: 72 / 371 (highly rank deficient)
- **DAE index**: The system is a DAE with 72 differential and 299 algebraic variables

### 3. Numerical Issues (Root Cause of "Blows Up")

The simulation fails due to:
1. **Singular Jacobian**: G has zero diagonal entries, making the Jacobian (G + C/dt) singular at small dt
2. **No stable DC equilibrium**: Ring oscillators don't have a stable DC operating point
3. **Very high condition number**: Even with GMIN regularization, cond(G) ≈ 6.57e18

### 4. Solver Behaviors

| Solver | Result |
|--------|--------|
| IDA | Segfault during transient |
| FBDF | "Newton steps could not converge" |
| ImplicitEuler | "Solver failed" (singular matrix) |

## Root Causes

1. **JIT compilation time**: The PSP103 model is ~3000 lines of Verilog-A that expands to a massive Julia function. Julia's JIT compiler takes ~150s for first compile.

2. **Singular Jacobian**: The MNA formulation creates a system where:
   - Many rows have zero diagonal (current variables, internal nodes)
   - The mass matrix C is rank-deficient (only capacitive nodes have dynamics)
   - Without regularization, the Jacobian is singular

3. **DAE structure**: Ring oscillators are true DAEs (not ODEs). They require:
   - Proper DAE solvers (IDA should work but has a bug)
   - Consistent initialization (no stable DC point to start from)
   - GMIN or source stepping to regularize during init

## GMIN Regularization Test

Adding GMIN = 1e-12 to all G diagonal entries:
- Removes all zero diagonal entries
- Enables linear solve to succeed
- Condition number still high but finite (6.57e18)
- V(vdd) correctly solved to 1.2V

## LevenbergMarquardt Success

Per `doc/circuit_initialization_ngspice_sciml.md`, LevenbergMarquardt acts as a GMIN-like
regularizer. Testing confirms **it works**:

```
Trying LevenbergMarquardt...
LM solve in 6.9s
Retcode: MaxIters
Max |u|: 1.1999809754198028
V(vdd): 1.1999802730503715
```

**Key finding**: `LevenbergMarquardt(damping_initial=1.0)` successfully finds a reasonable
pseudo-equilibrium for the ring oscillator in 6.9s (20 iterations).

The `MaxIters` retcode is expected - ring oscillators don't have a true DC equilibrium.

### Updated JIT Breakdown

Full JIT compilation with both code paths:

| Phase | Time |
|-------|------|
| Model loading | ~38s |
| MNAContext stamp! compile | ~145s |
| DirectStampContext stamp! compile | ~110s |
| Ring builder first call (PMOS variant) | ~109s |
| **Total first-call** | **~400s** |

After JIT, operations are fast:
- Residual evaluation: 0.1s
- Jacobian evaluation: 0.2s
- Linear solve: 2.4s
- LM solve (20 iters): 6.9s

## Recommendations

### Short-term Fixes

1. **Increase benchmark timeout**: Account for 400s+ JIT compilation (both code paths)
2. **Use LevenbergMarquardt**: Pass `nlsolve=LevenbergMarquardt(damping_initial=1.0)` to DC solve
3. **Precompile PSP103**: Add to precompile workload or use a precompiled device package
4. **Add GMIN option**: Implement GMIN stepping in CedarDCOp/CedarUICOp

### Medium-term Improvements

1. **Fix IDA integration**: The segfault during transient needs investigation
2. **Implement source stepping**: Homotopy method for difficult convergence
3. **Use BSIM4 instead**: The test BSIM4 model works and compiles in 76s (vs 150s for PSP103)

### Code Changes Needed

1. **dcop.jl**: Add GMIN option to CedarUICOp:
```julia
struct CedarUICOp <: DiffEqBase.DAEInitializationAlgorithm
    warmup_steps::Int
    dt::Float64
    gmin::Float64  # New: conductance to add to diagonal
end
CedarUICOp(; warmup_steps=10, dt=1e-12, gmin=0.0) = CedarUICOp(warmup_steps, dt, gmin)
```

2. **solve.jl**: Apply GMIN during assembly when spec.mode == :dcop

3. **Benchmark**: Use BSIM4 or add explicit precompile step before timing

## Test Results

The following test confirms GMIN regularization enables solving:
```julia
# After adding GMIN = 1e-12 to G diagonal
DC solve succeeded!
Max voltage: 16.18V  # Note: linear solve at u=0, not physically meaningful
V(vdd): 1.2V  # Correct supply voltage
```

## Successful Configuration (January 2026)

After extensive CI testing, a working configuration was found:

### Winning Configuration

```julia
tran!(circuit, (0.0, 1e-9);
    solver=FBDF(autodiff=false),
    dtmax=0.01e-9,  # 10ps max step - CRITICAL
    initializealg=CedarTranOp())
```

**Results:**
- Status: **Success** (100% completion)
- Timepoints: 180
- NR iterations: 531
- Iter/step: 3.0 (healthy)

### Key Findings

| Configuration | Status | Progress | Notes |
|---------------|--------|----------|-------|
| FBDF + dtmax=0.1ns | Unstable | 25% | Too large step |
| FBDF + dtmax=0.01ns | **Success** | **100%** | **Winner** |
| FBDF + loose tol | Unstable | 32% | Tolerances don't help |
| Rodas5P | Unstable | 31% | Wrong solver type |
| IDA | Segfault | - | Bug in DAE path |

### Why It Works

1. **CedarTranOp homotopy**: The initialization uses a fallback chain:
   - Regular DC solve → fails (expected, no stable DC point)
   - GMIN stepping → fails
   - Source stepping → **succeeds**

2. **FBDF solver**: BDF (Backward Differentiation Formula) methods are the standard
   for stiff circuit simulation. FBDF is SciML's modern BDF implementation.

3. **dtmax=0.01ns (10ps)**: This is the critical parameter. The ring oscillator
   switches at ~GHz frequencies, requiring small timesteps to capture the dynamics.
   With dtmax=0.1ns, the solver tries to take steps too large and goes unstable.

### Updated Recommendations

The ring oscillator benchmark (`benchmarks/vacask/ring/cedarsim/runme.jl`) has been
updated to use this winning configuration.

For ring oscillators and other fast-switching circuits:
- Use `FBDF(autodiff=false)` solver
- Use `CedarTranOp()` for initialization (enables homotopy)
- Set `dtmax` small enough to capture switching (e.g., 10ps for GHz circuits)

## Investigation Notes

Diagnostic scripts were used during investigation but have been cleaned up.
Key tests performed:
- Single PSP103 device stamping (identified JIT as bottleneck)
- Single BSIM4 device comparison (76s vs 150s JIT)
- GMIN regularization (enables linear solve)
- LevenbergMarquardt solver (successfully finds pseudo-equilibrium)
- CI testing of solver/tolerance/dtmax combinations (found winning config)

## Bottleneck Decomposition: RHS Evaluation vs. Linear Solve vs. Iteration Count (2026-07)

Follow-up question: is the PSP103 ring oscillator slow because each residual/
Jacobian *evaluation* is expensive (RHS eval), because the *linear solve* is
expensive, or because the solver needs far *more Newton iterations/timesteps*?
And would swapping in a simpler MOSFET model (mos6, BSIM3, BSIM4) actually
solve in a more reasonable time?

`benchmarks/vacask/ring/cedarsim/bottleneck_probe.jl` isolates these three
costs for a given model level (`mos1`, `mos6`, `bsim3`, `bsim4`, `psp103`):
per-call RHS/stamp evaluation, per-call analytic Jacobian fill, a from-scratch
sparse LU factorize+solve (an upper bound on linear-solve cost — real KLU
solves reuse the symbolic factorization across iterations, so this
over-estimates), and a real `tran!` run to get the actual per-iteration wall
time and NR-iteration count. `psp103` reuses the real `runme.sp` topology and
VACASKModels' precompiled builder; `mos1`/`mos6`/`bsim3`/`bsim4` build the
same 9-stage inverter chain via `VADistillerModels`' `ModelRegistry` with
small explicit 10fF load caps (these models lack PSP103's internal ~1fF
parasitics to self-oscillate against without them).

### Results

| Model  | Unknowns | Build+JIT | RHS eval | Jac fill | Naive LU (upper bound) | Actual cost/iter | NR iters | Sim span | Wall (steady-state) | ms / simulated ns |
|--------|---------:|----------:|---------:|---------:|------------------------:|------------------:|---------:|---------:|---------------------:|-------------------:|
| mos1   |       11 |     3.6 s |  36.5 us |  27.8 us |                  281 us |            49.5 us|      104 |   100 ns |               0.005 s|               0.05 |
| mos6   |       47 |     3.6 s |  38.5 us |  82.9 us |                  413 us |            54.8 us|      104 |   100 ns |               0.006 s|               0.06 |
| bsim4  |      191 |    18.1 s | 278.4 us |1338.8 us|                 2610 us|           793.3 us|      104 |   100 ns |               0.083 s|               0.83 |
| bsim3  |       83 |     5.0 s | 128.2 us | 315.8 us|                  695 us|         15296 us **|     107 |   100 ns |               1.637 s|               16.4 |
| psp103 |      371 |     8.4 s | 366.8 us|4098.5 us|                18579 us|          2073.6 us|     4510 |    20 ns |               9.35 s |              467.6 |

(** bsim3's per-iteration cost is a reproducible anomaly — see below.)

For reference, VACASK's own compiled-C/OSDI PSP103 (same no-cap 9-stage
topology, full 1us span) does 81875 NR iterations in 1.18s = **14.4 us/iter**.

### What actually dominates

1. **Per-call RHS-eval cost scales mildly with model complexity** (36.5 us →
   366.8 us, mos1 to psp103, ~10x) — real, but far too small on its own to
   explain the >100x blowup in total wall time.
2. **The naive from-scratch linear solve is *not* what's paid per iteration.**
   Actual per-iteration cost is always well below the naive-LU upper bound
   (e.g. bsim4: 793 us actual vs. 2610 us naive; psp103: 2074 us actual vs.
   18579 us naive), confirming KLU's incremental/symbolic-reuse solve is much
   cheaper than a cold factorization, and that linear algebra is not the
   primary bottleneck either.
3. **The dominant multiplier is the number of Newton iterations/timesteps
   needed per unit of simulated time.** mos1/mos6/bsim4 need ~1.0 iter/step
   and converge the 100ns span in 104-107 iterations. PSP103 needs 5.26
   iter/step *and* only covers 20ns in 4510 iterations — roughly **40-50x
   more Newton iterations per simulated ns** than the SPICE-level models.
   This reflects genuine physical stiffness: PSP103's femtofarad-scale
   internal parasitics and richer surface-potential formulation force much
   smaller stable steps, independent of how fast any single stamp call is.
4. Combining (1) and (3): Cadnip's PSP103 pays ~144x more wall-clock per NR
   iteration than VACASK's compiled-C/OSDI PSP103 (2074 us vs. 14.4 us) — so
   there is real headroom in per-call performance too — but the iteration-
   count blowup is the larger factor in the overall benchmark being slow.

### Would mos6/BSIM3/BSIM4 solve faster?

Yes, dramatically — **BSIM4 is ~560x faster wall-clock-per-simulated-ns than
PSP103** (0.83 vs 467.6 ms/ns) while still being an industry-realistic model
(unlike mos1, a toy level-1 model), and its build+JIT time (18s) is trivial
compared to what indirection-laden PSP103 has historically required. mos6 is
faster still but is a much more dated/simplistic model electrically.
**BSIM3 is not currently a safe substitute**: despite having a *smaller*
system (83 vs 191 unknowns) and *cheaper* isolated RHS/Jacobian/linsolve costs
than BSIM4, its real `tran!` run reproducibly costs ~15.3 ms/iteration —
roughly 20x worse than BSIM4's 793 us/iteration, with no rejected steps or
other stat anomaly visible in `sol.stats` to explain it. This looks like a
BSIM3-specific inefficiency in the DAE (`IDA`) residual/Jacobian path
specifically (as opposed to the ODE mass-matrix path, which was what was
isolated-benchmarked above) and needs its own follow-up investigation before
BSIM3 is used as a "cheap" reference model.

**Recommendation:** use BSIM4 (`level=14`) for a CI-friendly ring oscillator
benchmark/test alongside the existing PSP103 one. For PSP103 itself, the
highest-leverage optimization is reducing the required Newton-iteration count
(larger stable steps) rather than further micro-optimizing the stamp code.

### Correction: verify against the real full-length run, not a short slice

The PSP103 numbers above (`9.35 s`, `467.6 ms/ns`, and the "~144x slower than
VACASK" comparison) came from a `tspan` cut to 20ns instead of the real
benchmark's 1us, extrapolated for turnaround speed during the investigation —
and the VACASK comparison used the upstream README table (measured on a
Threadripper 7970), not this machine. Both of those are worth being honest
about, so here are the real, non-extrapolated, same-machine numbers:

```
# Real VACASK binary, same machine, official benchmark.py methodology
CASES="ring" RUNS=3 bash benchmarks/vacask/run_vacask.sh
→ 2.28 s, 26040 timepoints, 1 rejected, 81949 iterations   (full 1us span)

# Real, unmodified benchmarks/vacask/ring/cedarsim/runme.jl, same machine
→ 333.5 s median (327-337 s range), 43968 timepoints, 242293 NR iterations,
  5.51 iter/step, 63.50 GiB allocated, 21.47M allocations   (full 1us span)
```

Corrected ratios:

| | Cadnip | VACASK | Ratio |
|---|---:|---:|---:|
| Wall time | 333.5 s | 2.28 s | **146x** |
| NR iterations | 242,293 | 81,949 | **2.96x** |
| Iter/step | 5.51 | 3.15 | 1.75x |
| Wall time / iteration | 1376 us | 27.8 us | **49.5x** |

So the 146x total slowdown factors as **~3x more Newton iterations × ~50x
higher cost per iteration** — not one dominant cause. The iteration-count
multiplier (2.96x, full run) is consistent with what the 20ns slice suggested
(iter/step 5.51 vs 5.26), so the slice was a reasonable proxy for *local*
dynamics — just not for extrapolating total wall time, and definitely not a
substitute for measuring the real VACASK binary on the same hardware. The
507-refactorizations-≈-total-time hypothesis (linear solve not benefiting
from warm/cached KLU reuse) was derived from the 20ns slice's `sol.stats` and
has **not** been re-confirmed against the real full-length run — treat it as
a lead, not a conclusion.

The real run's own numbers point to an additional, concrete, unexplained
lead: **63.50 GiB allocated / 21.47M allocations for one run** (~88
allocations per Newton iteration) in what's supposed to be a zero-allocation
`DirectStampContext` hot path. GC time itself is only ~2.4% of wall time, so
this isn't directly GC-pause-bound, but it's a large, measurable, and much
more tractable target for a `Profile.Allocs` pass than anything above.

This also means the earlier BSIM4-vs-PSP103 "~560x faster" figure should be
revised down using the real PSP103 number: 333.5 ms/ns vs BSIM4's 0.83 ms/ns
(itself measured directly, not extrapolated) is **~400x**, not 560x.

## Future Directions: What Would Actually Help the Ring Benchmark (Speculation, 2026-07)

The measured facts so far: no explicit caps (dynamics are driven entirely by
PSP103's internal fF parasitics), `C` rank 72/371 so 299 of 371 variables are
algebraic, `cond(G) ≈ 6.6e18`, no stable DC operating point, sustained ~5.5
Newton iterations/step throughout the run, and the current workaround is
`force_dtmin=true, abstol=1e-4, reltol=1e-2`. None of this has been tested
yet — it's ranked speculation for a follow-up investigation, ordered by
expected payoff:

1. **Unit-aware error control (vector `abstol`).** The state vector mixes
   ~1V node voltages, µA-scale branch currents, and femto-scale charge
   states in one vector, and a single scalar `abstol` makes the
   tiniest-unit variables dominate the error test - chronic error-test
   failure looks exactly like "needs `force_dtmin` + relaxed tolerances."
   ngspice never uses one tolerance: it has `vntol` (1µV), `abstol` (1pA),
   `chgtol` (10fC) per variable class. SciML's solvers accept
   `abstol::Vector`, and `MNAContext` already knows each variable's class
   (node/current/charge), so this is cheap to build and is the top suspect
   for the actual binding constraint.
2. **Exclude algebraic variables from the error test.** With 299/371
   algebraic variables, LTE-testing them is a known cause of spurious step
   rejections in DAEs. IDA has `suppressalg=true` (blocked on the known IDA
   segfault on this circuit - see "Solver Behaviors" above); the FBDF-path
   equivalent is folding this into the weight vector from item 1 (an
   effectively-infinite `abstol` for algebraic entries excludes them from
   the error norm the same way).
3. **Always-on GMIN.** ngspice puts `GMIN=1e-12` across every junction
   permanently, not just during homotopy init - this codebase appears to
   only use a `gshunt`-style diagonal floor during DC/source stepping.
   `cond(G) ≈ 6.6e18` means the linear solves deliver almost no correct
   digits, which alone could explain sustained 5+ Newton iterations per
   step. A permanent diagonal floor is standard SPICE practice and should
   be a cheap, one-line experiment.
4. **Asymmetric initialization.** A ring's symmetric operating point is
   metastable with a near-singular Jacobian along the oscillation mode,
   and Newton can hover there. This doc's own "LevenbergMarquardt Success"
   section above already showed LM finds a pseudo-equilibrium; adding a
   deliberate nodeset-style kick (bias one stage high) on top of
   `CedarUICOp` would avoid fighting the singular direction and start the
   oscillation cleanly instead of relying on `force_dtmin` to survive a
   metastable start.
5. **`$limit` implementation.** Helps the VADistiller model family a lot -
   it's their designed-in convergence aid - but does nothing directly for
   PSP103, which has no `$limit` call sites.

For completeness, what's already been tried and empirically **doesn't**
help this circuit: VA voltage-dependent event detection (`va_events`, see
`doc/va_events_removal_findings.md` - increased Newton-iteration count and
degraded fidelity on every real model tested, and PSP103's internal
comparisons are numerical safety guards, not physical discontinuities),
`$bound_step` (`dtmax` is already pinned to 0.05ns, tighter than any
`$bound_step` value would realistically set), and `auto_tstops` (a
free-running ring oscillator has no source edges to derive breakpoints
from - it's autonomous). The ring's problem is conditioning and
error-weighting across mismatched variable units, not missed
discontinuities.

## Item 1 Tested and Refuted: Per-Class Vector `abstol` Makes Things Worse (2026-07)

The top-ranked lead above — unit-aware error control via a vector `abstol`
split by variable class (node voltage / branch current / charge) — was
implemented and empirically tested on this exact circuit, per an explicit
request to validate rather than just theorize. It was **not** a JUNCAP/PCNR
side-quest that motivated this: a competing hypothesis (that PSP103's
internal JUNCAP200 junction-diode submodel might implement its own
NR-limiting, motivating a general Predictor/Corrector Newton-Raphson
framework) was closed out by inspection first — JUNCAP does have a
`VMAX`-based diode-current linearization functionally similar to ngspice's
`pnjlim` (`models/PSPModels.jl/va/JUNCAP200_macrodefs.include:105-121,279-291`),
but `SWJUNCAP` defaults to `0` and is never set in this benchmark's model
card (`models/VACASKModels.jl/spice/models.inc`, loaded via `using
VACASKModels`), so none of that code is active in the circuit that's failing
to converge. PCNR itself remains pure proposal
(`doc/compass_artifact_wf-c99b8a86-d31f-4e59-9c4f-6f69e4edd847_text_markdown.md`)
with no code or precedent in the repo, and stays deferred.

### What was implemented

`MNA.state_abstol(sys::MNAData; vntol, iabstol, chgtol)` (`src/mna/build.jl`)
builds a per-class tolerance vector from the node/current/charge index ranges
`MNAData` already tracks. `tran!`'s `abstol` kwarg now accepts a
`(vntol=, iabstol=, chgtol=)` NamedTuple, expanded via `state_abstol` using
the `MNAData` the problem constructor attaches as `prob.f.sys`
(`src/sweeps.jl`, `_resolve_abstol`). This works correctly end-to-end — a new
`test/mna/oscillator_test.jl` testset confirms the small 3-stage inverter
ring still converges and oscillates with the vector form. **The plumbing is
not the problem; the numbers are.**

### What happened on the real PSP103 ring (20ns slice, same methodology as
### the Bottleneck Decomposition section above)

| Configuration | Result |
|---|---|
| Baseline: scalar `abstol=1e-4, reltol=1e-2, force_dtmin=true` | **Success**, 842 timepoints, 5.30 iter/step, 39.9s |
| Vector `(vntol=1e-6, iabstol=1e-12, chgtol=1e-14)`, `reltol=1e-2`, `force_dtmin=true` | **InitialFailure** — dies inside `CedarTranOp`'s consistency step (3 NR iters, 3.2s) |
| Vector `(vntol=1e-6, iabstol=1e-12, chgtol=1e-14)`, `reltol=1e-3`, `force_dtmin=false` | **InitialFailure**, same as above (2.9s) |
| Vector `(vntol=1e-4, iabstol=1e-9, chgtol=1e-12)` — vntol matched to baseline, currents/charges only "SPICE-realistic" | Init succeeds, but transient blows up: **MaxIters**, 66 timepoints (only reached 1.83ns of the 20ns span), **4,999,685 NR iterations = 74,622 iter/step** (14,000x worse than baseline's 5.3), 1628s wall |
| Vector `(vntol=1e-5, ...)` | Killed after 27min+ of the same MaxIters trajectory as the row above — trend was already unambiguous |

Two independent failure modes, both worse than the status quo:
1. **Tightening `vntol` alone** (1e-6 vs. the baseline's blanket 1e-4) breaks
   `CedarTranOp`'s DC/consistency initialization outright — the ring's
   near-singular Jacobian (`cond(G) ≈ 6.6e18`) apparently can't reach a
   node-voltage residual below 1e-6 during init, only below ~1e-4.
2. **Tightening only the current/charge tolerances** to textbook SPICE
   magnitudes (`1e-9` A / `1e-12` C) while leaving `vntol` at the baseline's
   1e-4 causes a catastrophic ~14,000x blowup in Newton iterations per step
   during the actual transient integration — the branch-current/charge
   *magnitudes* this circuit's PSP103 stamping actually produces are
   evidently far enough from `1e-9`/`1e-12` scale that demanding that level
   of absolute accuracy on them is effectively demanding many more correct
   digits than the ill-conditioned linear solves can deliver, so the
   step-size controller thrashes instead of converging.

### Conclusion

**The "unit-aware error control" hypothesis, as formulated with
SPICE-textbook-style default magnitudes, is refuted by direct measurement on
this circuit — not just untested-but-plausible anymore.** The scalar
`abstol=1e-4` "blunt workaround" isn't a symptom of a fixable unit mismatch;
empirically it sits in the one narrow tolerance regime that actually works
for this ill-conditioned system, and moving away from it in *either*
direction (tighter node voltages, or tighter currents/charges while leaving
node voltages alone) makes convergence dramatically worse, not better. This
is consistent with `cond(G) ≈ 6.6e18` being the real, underlying constraint:
no per-class reweighting fixes an 18-order-of-magnitude conditioning problem;
it just changes where the error test starts failing. The next candidates
from the ranked list above (item 3, always-on GMIN, or item 4, asymmetric
initialization) target the conditioning/singularity directly rather than
re-weighting the symptom, and are better-motivated next steps than any
further tolerance tuning.

**Disposition:** `MNA.state_abstol`/`tran!`'s NamedTuple `abstol` form is
kept as a working, tested, general capability (it may be exactly right for
circuits with real class-mismatched magnitudes that aren't this pathological)
but is **not** adopted as the ring oscillator benchmark's default — the
existing `force_dtmin=true, abstol=1e-4, reltol=1e-2` workaround remains the
documented, working configuration for `benchmarks/vacask/ring/cedarsim/runme.jl`.
