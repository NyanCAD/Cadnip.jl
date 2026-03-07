# Comparative Study: Cadnip.jl vs VAJAX vs Circulax

**Date:** 2026-03-07
**Purpose:** Evaluate three open-source analog circuit simulator projects to identify the best path forward for joining forces.

---

## 1. Honest Context

All three projects are single-developer efforts. Metrics like lines of code and commit counts are misleading:

- **Cadnip** (~18K src): Much of this is parser/codegen code inherited from CedarSim. The actual MNA simulator engine is ~7,600 LOC. CedarSim has a longer private history at JuliaComputing.
- **VAJAX** (~65K): Includes ~4,500 LOC of OpenVAF bindings, ~4,500 LOC of SPICE netlist converter, ~10K LOC of tests. Core solver engine is ~8-10K LOC. The MNA port to Cadnip and vajax as a whole relied more on vibe coding, which inflates commits/LOC.
- **Circulax** (~4.8K): Smallest and cleanest, but this is genuinely less code rather than hiding complexity — many features simply don't exist yet. Well-designed code by a developer who clearly cares about architecture.

What matters is: **what actually works, what's actually tested, and what would need to happen to make each one the unified core.**

---

## 2. What Each Simulator Actually Has (Feature Truth Table)

### 2.1 Solver & Analysis Capabilities

| Capability | Cadnip | VAJAX | Circulax |
|---|---|---|---|
| **DC Operating Point** | Newton-Raphson via NonlinearSolve.jl (RobustMultiNewton + LevenbergMarquardt + PseudoTransient) | Custom NR in `jax.lax.while_loop` | NR via Optimistix |
| **GMIN stepping** | Yes (exponential backoff, max 20 steps) | Yes (1e-3 → 1e-13, adaptive) | No |
| **Source stepping** | Yes (max 50 steps) | Yes (0→100%, with GMIN fallback) | No |
| **Homotopy chain** | Newton → GMIN → Source | gdev → gshunt → src (configurable) | None |
| **Transient solver** | Sundials IDA (DAE), Rosenbrock (ODE) via DifferentialEquations.jl | Custom BE/Trap/Gear2 (2nd order max) | Backward Euler only (despite Diffrax having more) |
| **Adaptive timestepping** | IDA's built-in (production-grade) | Custom LTE-based, polynomial predictor (2nd order) | PIDController via Diffrax |
| **Higher-order methods** | Rodas5P (5th order Rosenbrock), IDA (variable order BDF) | Gear2 max (2nd order) | BE only (1st order) |
| **AC small-signal** | Yes (descriptor state-space, frequency sweep) | Yes (linearization + sweep) | No |
| **Noise analysis** | No (stubs return 0.0) | Yes (thermal/shot/flicker, small-signal) | No |
| **Harmonic balance** | No | Yes (DDT-based) | Yes (FFT-based, but dense Jacobian limits scalability) |
| **Transfer function** | No | Yes (DC/AC) | No |
| **Corner/PVT sweeps** | Parameter sweeps (Product/Tandem/Serial) | Yes (dedicated corner analysis) | No (manual loop) |
| **Oscillator init** | CedarUICOp pseudo-transient relaxation | Not mentioned | No |

**Verdict on solvers:** Cadnip has the best solver *ecosystem* via DifferentialEquations.jl — IDA is a production-grade DAE solver used in industry, Rosenbrock methods are excellent for stiff circuits. VAJAX has the most analysis *types* but its custom transient solver is limited to 2nd order. Circulax underutilizes Diffrax (locked to BE despite having access to better methods).

### 2.2 Verilog-A / Device Model Support

| Aspect | Cadnip | VAJAX | Circulax |
|---|---|---|---|
| **VA compiler** | Custom Julia parser + codegen | OpenVAF (Rust, industry-standard) | None |
| **VA approach** | Parse VA → generate Julia stamp functions | OpenVAF MIR → JAX operations | N/A |
| **PSP103** | Compiles but generates 100K+ statement functions that blow up LLVM SROA | Works, validated against VACASK + ngspice | N/A |
| **BSIM4** | Via VADistillerModels (partial) | Via OpenVAF (validated) | N/A |
| **Model validation** | Basic tests, some regression | 3-way comparison (VACASK + ngspice) | N/A |
| **Production models tested** | PSP103 (with LLVM issues), BSIM4 | PSP103, BSIM4, EKV, HiSIM | None |
| **Static analysis** | Minimal (parser does less optimization) | OpenVAF does full static analysis, node collapse, parameter caching | N/A |

**Verdict on VA:** VAJAX wins decisively here. OpenVAF is an established, well-tested VA compiler that does proper static analysis, dead code elimination, and node collapsing. Cadnip's custom VA codegen is a heroic effort but fundamentally limited — it does less optimization, and complex models like PSP103 generate functions so large that LLVM chokes on them. This is a structural limitation of the "parse VA, generate native code" approach without a sophisticated intermediate optimization pass.

### 2.3 Circuit Scale & Validation

| Circuit | Cadnip | VAJAX | Circulax |
|---|---|---|---|
| Simple RC/RLC | Tested | Tested | Tested |
| Diode circuits | Tested | Graetz (99.5% — stiff transition issue) | Diode clipper tested |
| BJT amplifiers | Audio integration tests (CE, emitter degen) | Converter exists, not validated in solver | Ebers-Moll model, basic tests |
| MOSFET circuits | PSP103 (with LLVM caveats) | PSP103 ring osc, c6288 (86K MOSFETs), mul64 (266K MOSFETs) | Square-law only |
| Max scale tested | ~100s of nodes | ~133K nodes (mul64 on GPU) | ~10s of nodes |
| Reference comparison | Against expected values | VACASK (C++) + ngspice | Against expected values |

**Verdict on scale:** VAJAX is the only one that has been tested at serious circuit scale. The mul64 benchmark (266K MOSFETs, ~133K nodes) is genuinely impressive. Cadnip and Circulax are untested beyond small circuits.

### 2.4 Netlist & Input Format Support

| Aspect | Cadnip | VAJAX | Circulax |
|---|---|---|---|
| **SPICE netlist** | Full parser (SpectreNetlistParser.jl, ~10K LOC) | Converter from ngspice/HSPICE/LTSpice (~4.5K LOC) | None |
| **Spectre netlist** | Yes (native) | No | No |
| **Native format** | Julia code (compiled circuits) | VACASK .sim files | SAX Python dicts |
| **Subcircuits** | Yes (.subckt hierarchy) | Yes | No |

### 2.5 Platform & Ecosystem

| Aspect | Cadnip | VAJAX | Circulax |
|---|---|---|---|
| **Language** | Julia | Python/JAX | Python/JAX |
| **GPU** | Not currently, but Julia can compile to GPU (significant effort to eliminate allocations) | Yes, working (CUDA via JAX). 12.8x speedup on mul64 | Theoretically via JAX (untested, same JAX options as VAJAX) |
| **AD** | ForwardDiff (forward-mode, through entire solver) | JAX AD (forward + reverse). Less natural than Cadnip's for circuit-specific use | JAX AD (forward + reverse). Cleanest integration of the three |
| **Solver libraries** | DifferentialEquations.jl (IDA, Rosenbrock, many more) | Custom from scratch | Diffrax + Optimistix (underutilized) |
| **ML integration** | Julia ML ecosystem (Flux, Lux) | JAX/Flax/Optax native | JAX/Flax/Optax native |
| **Startup time** | Slow (Julia JIT compilation, minutes for first sim) | Fast | Fast |
| **PyPI/package** | No | Yes (`pip install vajax`) | Yes (pixi) |

### 2.6 Unique Capabilities

| Capability | Only in |
|---|---|
| Photonic circuit simulation (S-param → Y-matrix, ring resonators, MZI) | Circulax |
| SPICE/Spectre netlist parsing (native, not conversion) | Cadnip |
| GPU-accelerated large-circuit simulation (actually tested) | VAJAX |
| Production VA models via OpenVAF | VAJAX |
| IDA DAE solver (variable-order BDF, production-grade) | Cadnip |
| Noise analysis | VAJAX |
| Transfer function analysis | VAJAX |
| Mixed electronic-photonic simulation | Circulax |
| SAX photonic library integration | Circulax |
| `@component` decorator API for model definition | Circulax |
| Zero-allocation transient inner loop | Cadnip |
| Pseudo-transient oscillator initialization | Cadnip |

---

## 3. Three Paths: What Would Each Require?

### Path A: VAJAX as Core

**What you'd get for free:** OpenVAF VA models, GPU acceleration, noise/AC/HB/transfer function analyses, large-circuit scalability, SPICE netlist converter, PyPI distribution.

**What's missing and needs work:**

| Gap | Effort | Notes |
|---|---|---|
| Better transient solvers | **Large** | Current custom BE/Trap/Gear2 is limited to 2nd order. Options: (a) contribute higher-order methods to VAJAX, (b) integrate Diffrax as alternative backend, (c) port DifferentialEquations.jl solvers. Option (b) is most practical — Diffrax already has implicit RK methods. |
| Photonic simulation | **Medium** | Port Circulax's S-param→Y-matrix transform, complex-valued MNA assembly, and photonic component library. The three-layer architecture maps cleanly. |
| SPICE/Spectre parsing | **Medium** | Has ngspice converter already. Could improve it or integrate Cadnip's parser as a standalone frontend. |
| Controlled sources (VCVS, CCCS, etc.) | **Small-Medium** | Listed as TODO in VAJAX. Cadnip and Circulax both have these. |
| Transmission lines | **Medium** | Not in any of the three. |
| Convergence robustness | **Ongoing** | Graetz benchmark still has edge case. Homotopy chain exists but could be more robust. |
| Cleaner component API | **Small** | Circulax's `@component` pattern could be added for hand-written Python models alongside OpenVAF path. |

**Risk:** Custom solver quality is a long-term maintenance burden. Solver code is subtle — bugs manifest as wrong answers, not crashes. Relying on Diffrax/Optimistix (as Circulax does) or DifferentialEquations.jl (as Cadnip does) offloads this to dedicated library maintainers.

### Path B: Circulax as Core

**What you'd get for free:** Clean architecture, photonic simulation, Diffrax/Optimistix/Equinox ecosystem, elegant component API, HB solver, SAX integration.

**What's missing and needs work:**

| Gap | Effort | Notes |
|---|---|---|
| Verilog-A support | **Very Large** | The biggest gap. Options: (a) integrate OpenVAF (requires Python bindings for MIR→JAX, essentially rewriting openvaf_jax), (b) write a new VA→JAX compiler. Either way this is months of work and the hardest piece of the puzzle. |
| Convergence aids | **Medium** | No GMIN stepping, no source stepping, no homotopy. Must be added for any non-trivial circuit to converge reliably. |
| AC analysis | **Medium** | Standard linearization + frequency sweep. Well-understood algorithm. |
| Noise analysis | **Medium-Large** | Requires per-device noise models, correlation matrix, spectral density computation. |
| SPICE netlist parser | **Large** | Could use Cadnip's parser as standalone tool generating SAX-format output, or write a new one. |
| Unlock Diffrax solvers | **Small** | Currently locked to BE — should be straightforward to expose Diffrax's Kvaerno/RadauIIA methods. |
| Scale to large circuits | **Unknown** | The sparse infrastructure exists (KLU, BiCGStab) but has never been tested beyond ~10 nodes. May need significant work. |
| Subcircuit hierarchy | **Medium** | No .subckt support. |
| Parameter sweeps | **Small** | Just engineering, no algorithmic challenge. |
| Dense HB Jacobian | **Medium** | Current HB won't scale. Needs sparse HB formulation. |

**Risk:** The OpenVAF integration is a massive effort and is the critical path to production models. Without VA support, you can't run real PDK models, which limits the simulator to educational/research use.

### Path C: Cadnip as Core

**What you'd get for free:** DifferentialEquations.jl solver ecosystem (IDA, Rosenbrock — best transient solvers of the three), SPICE/Spectre parsing, ForwardDiff AD, convergence aids (GMIN + source stepping + pseudo-transient), subcircuit hierarchy, zero-allocation optimization.

**What's missing and needs work:**

| Gap | Effort | Notes |
|---|---|---|
| OpenVAF integration | **Large** | Replace custom VA codegen with OpenVAF. Need Julia bindings for OpenVAF (Rust). This solves the PSP103-blows-up-LLVM problem at the root — OpenVAF's static analysis produces much more compact output. |
| GPU acceleration | **Very Large** | Julia can theoretically compile to GPU (CUDA.jl, KernelAbstractions.jl) but eliminating allocations from the inner loop is significant work. The zero-allocation DirectStampContext is a good start but the full solver loop needs to be GPU-friendly. |
| Photonic simulation | **Medium** | Port Circulax's photonic components. Julia's type system could make this elegant (complex-valued MNA via parametric types). |
| Noise analysis | **Medium** | AC infrastructure exists. Need per-device noise sources. |
| Harmonic balance | **Medium-Large** | Not implemented. Could port Circulax's approach. |
| Transfer function | **Small** | Standard analysis, AC infrastructure exists. |
| Python ecosystem access | **Medium** | PythonCall.jl works but adds complexity. Julia's ML ecosystem is smaller. |
| Startup latency | **Ongoing** | Julia's JIT compilation means minutes to first simulation. PackageCompiler.jl can help but adds deployment complexity. |
| Controlled sources | **Small** | Basic structure exists, needs completion. |

**Risk:** Julia's smaller ecosystem is both a strength (better numerical libraries) and weakness (fewer users, harder to hire contributors). The GPU story is real but unproven in practice. The VA codegen limitation (PSP103 → 100K statements → LLVM explosion) is a fundamental issue that needs OpenVAF to solve properly.

---

## 4. Architectural Portability: VAJAX ↔ Circulax

A critical question: since both VAJAX and Circulax are Python/JAX MNA simulators, how much work is porting features between them — really?

### The MNA core is structurally identical

MNA is MNA. Both simulators do the same thing:
1. Assemble G, C, b in COO sparse format by iterating over devices
2. Each device stamps conductance/capacitance/current at (i,j) positions
3. Convert to CSR/CSC, solve `(G + sC)x = b`
4. Newton-Raphson: linearize nonlinear devices around current x, rebuild G/b, re-solve

The stamp patterns are textbook. A resistor stamps `1/R` at `(p,p)`, `(p,n)`, `(n,p)`, `(n,n)` in all three projects. A voltage source adds a current variable. The data structures (sparse COO → compressed sparse, RHS vector) are the same JAX arrays.

### What differs (and what doesn't)

| Aspect | VAJAX | Circulax | Portability |
|---|---|---|---|
| NR loop | `jax.lax.while_loop` (manual) | Optimistix (library) | Different API, same math |
| Transient | Custom BE/Trap/Gear2 | Diffrax BE | Different, but both are JAX |
| Device interface | Function: voltages → (I, Q) | `@component` decorator → same | Thin adapter |
| Sparse format | `jax.experimental.sparse` | `jax.experimental.sparse` | **Same** |
| AD | `jax.jacfwd`/`jax.jacrev` | `jax.jacfwd`/`jax.jacrev` | **Same** |
| JIT | `@jax.jit` | `@eqx.filter_jit` (Equinox) | Compatible |

### Porting effort (corrected estimates)

**OpenVAF from VAJAX → Circulax:**
`openvaf_jax` is a self-contained module (~4,500 LOC) that loads OpenVAF-compiled `.so` files and creates JAX-callable device functions: `(terminal_voltages, params) → (currents, charges)`. It doesn't depend on VAJAX's solver. To use it in Circulax, you need an adapter that maps OpenVAF output to Circulax's `@component` interface. Both sides are JAX pytrees.
**Effort: 1-3 weeks, not months.** The adapter is mechanical — match function signatures, wrap in Equinox module.

**Diffrax from Circulax → VAJAX:**
VAJAX's custom transient loop needs to be wrapped as a Diffrax `ODETerm(vector_field)` where `vector_field(t, y, args) → dy/dt`. This means expressing the MNA rebuild as a pure function (which it already is in JAX). The main subtlety is handling the mass matrix (C may be singular → DAE). Circulax already solved this pattern.
**Effort: 1-2 weeks.** Well-defined refactor with a working reference implementation.

**GMIN/source stepping:**
These are ~100-200 lines of algorithm code each. Independent of framework — just wrap the NR solve in a loop that adjusts gmin/source scaling. The algorithm is identical in Cadnip and VAJAX.
**Effort: days.** Literal copy-paste with minor API adaptation.

**Noise/AC/transfer function analysis:**
Standard textbook algorithms operating on the assembled G, C, b matrices. Once you have the matrices (which all three produce), the analysis code is framework-independent.
**Effort: 1-2 weeks each.** The hard part is per-device noise models, not the analysis framework.

### What's genuinely hard to port

| Task | Why it's actually hard |
|---|---|
| **VAJAX's GPU scaling** | Not just "run on GPU" — required solving JAX-specific sparse↔dense auto-switching, vmap over devices, memory layout optimization. This is engineering that took iteration and profiling. |
| **VAJAX's validation suite** | Years of comparison against VACASK + ngspice. Can't be ported — must be re-run against any new codebase. |
| **Cadnip's SPICE parser** | Written in Julia (~20K LOC). Would need rewriting in Python or calling via subprocess. Not portable. |
| **Circulax's photonic models** | Domain expertise in photonic circuit theory (S-param → Y-matrix → complex MNA). The code is small but the physics knowledge behind it isn't trivial. |

### Bottom line

VAJAX and Circulax are more similar than different. Most features can be ported in weeks, not months. The previous estimates in this study were inflated by treating each project as a monolith. In reality, both use the same JAX primitives, the same sparse matrix types, and the same MNA algorithm. The barriers to collaboration are developer preferences and project direction, not architecture.

---

## 5. Deeper Analysis: Solver Quality Matters

Your point about VAJAX using custom solvers deserves emphasis. Here's why this matters:

**DifferentialEquations.jl (Cadnip):**
- Sundials IDA: Variable-order BDF (up to 5th order), used in MATLAB/Simulink, battle-tested in industry
- Error control: Per-step and global, with automatic order selection
- Stiffness handling: Automatic detection and method switching
- Maintained by a large team (SciML), not one person

**VAJAX custom solver:**
- BE/Trap/Gear2 only (max 2nd order)
- LTE control with polynomial predictor (ok but basic)
- Force-accepts after 5 consecutive rejects (risky for accuracy)
- Known issues: Graetz benchmark stiff transitions
- One person maintaining solver code where bugs = wrong answers

**Circulax via Diffrax:**
- Currently only uses Backward Euler (1st order)
- But Diffrax has Kvaerno3/4/5, RadauIIA, Tsit5, Dopri5
- Unlocking these is straightforward engineering
- Maintained by Patrick Kidger (well-respected in scientific ML)

**The gap:** VAJAX's custom solver works for its demonstrated use cases but is fundamentally less mature than IDA or even what Diffrax offers. If you want the solver quality Cadnip has, contributing it to VAJAX means either:
1. Porting IDA-equivalent logic to JAX (very hard, JAX's functional constraints fight imperative solver algorithms)
2. Integrating Diffrax into VAJAX (medium — rewrite the transient loop to use Diffrax's ODE interface)
3. Contributing better methods to Diffrax itself (the Julia DifferentialEquations.jl team already has a pattern for this — Chris Rackauckas contributes to both ecosystems)

Option 2 seems most practical: use VAJAX's MNA builder + OpenVAF device evaluation + homotopy, but swap the time integration to Diffrax.

---

## 6. The OpenVAF Question

OpenVAF integration is the critical differentiator. Without it:
- You're limited to hand-written device models (Circulax's square-law MOSFET, Ebers-Moll BJT)
- No access to foundry PDK models
- Not usable for real IC design work

Cadnip's custom VA codegen is a heroic effort but hits structural limits:
- Less static analysis → larger generated code → LLVM problems on complex models
- Ongoing maintenance burden of a custom VA compiler
- Missing optimizations that OpenVAF has (node collapse, parameter caching, dead code elimination)

**Integrating OpenVAF into Cadnip** (replacing custom codegen) would solve the PSP103 problem and give access to all validated models. The cost is needing Julia↔Rust FFI (via CBinding.jl or similar). This is real work but bounded.

**Integrating OpenVAF into Circulax** is simpler than it first appears — `openvaf_jax` is a self-contained module that could be imported as a library. The work is writing an adapter between OpenVAF's device function signature and Circulax's `@component` interface (see Section 4).

---

## 7. The GPU Question (Honest Assessment)

Your pushback is fair. Let me be precise:

- **VAJAX**: GPU works and is tested at scale. 12.8x speedup on mul64 is real. But only pays off at 500+ nodes (small circuits are 5-10x slower than C++ due to JAX overhead).
- **Circulax**: Has the same JAX options as VAJAX in theory. In practice, untested. The sparse infrastructure (KLU) is CPU-only; the dense solver would work on GPU but won't scale. Would need significant work to match VAJAX's GPU story.
- **Cadnip**: Julia can compile to GPU via CUDA.jl/KernelAbstractions.jl. The zero-allocation DirectStampContext is a good foundation. But GPU-ifying the full solver loop (including IDA or Rosenbrock) is a substantial project — DifferentialEquations.jl's GPU support exists but is experimental for DAEs.

**Bottom line:** GPU is a genuine advantage of VAJAX today. It's theoretically possible for all three but only VAJAX has done the work.

---

## 8. Revised Recommendation

Rather than picking a winner, here's what each project should contribute to a unified effort, based on **actual demonstrated strengths** not potential:

### What Each Brings (Proven, Not Aspirational)

| Contributor | Proven Contribution | Why It's Hard to Replicate |
|---|---|---|
| **VAJAX (Rob)** | OpenVAF→JAX pipeline, GPU MNA solver, large-circuit validation, noise/AC/HB/xfer analyses | OpenVAF integration is ~4,500 LOC of careful MIR→JAX translation. GPU solver required solving JAX-specific challenges (lax.while_loop NR, vmap device eval, sparse auto-switching). Years of validation work against VACASK + ngspice. |
| **Cadnip (Pepijn)** | SPICE/Spectre parser (~20K LOC across two packages), DifferentialEquations.jl integration patterns, convergence aids, zero-allocation patterns | Parser covers multiple SPICE dialects + Spectre. The IDA/Rosenbrock integration with circuit-specific initialization (CedarDCOp, CedarTranOp, CedarUICOp) represents deep domain expertise. |
| **Circulax (Chris)** | Photonic simulation, clean component API, Diffrax/Optimistix integration, SAX ecosystem bridge | Photonic circuit simulation (S-param→Y-matrix→complex MNA) is genuinely unique. The `@component` decorator pattern is the best API design of the three. Building on Diffrax/Optimistix rather than custom solvers is a strategic advantage. |

### Three Viable Paths (All Require Real Work)

**Path A: VAJAX + Diffrax solvers + photonics + better parsing**
- Swap VAJAX's custom transient solver for Diffrax (1-2 weeks)
- Port Circulax's photonic support (2-4 weeks)
- Improve SPICE parsing (extend existing converter or integrate Cadnip's parser as frontend tool)
- Keep OpenVAF + GPU as-is
- **Estimated effort:** 1-2 months for Diffrax integration + photonics. Parsing is ongoing.

**Path B: Circulax + OpenVAF + convergence aids + scaling**
- Import openvaf_jax as library, write adapter for `@component` interface (1-3 weeks)
- Add GMIN/source stepping homotopy (days — copy algorithm from VAJAX)
- Add AC + noise analysis (2-4 weeks)
- Unlock Diffrax's higher-order solvers (days — currently stuck on BE)
- Validate at scale (ongoing — requires running benchmark suite)
- **Estimated effort:** 2-3 months. OpenVAF is weeks not months since openvaf_jax exists.

**Path C: Cadnip + OpenVAF + photonics**
- Replace custom VA codegen with OpenVAF (via Rust FFI) — genuinely harder here due to Julia↔Rust FFI
- Port photonic simulation from Circulax (needs Python→Julia rewrite)
- Add noise, HB, transfer function analyses
- GPU work is a separate, longer-term effort
- **Estimated effort:** 2-4 months for OpenVAF integration (harder than Python paths). Photonics + new analyses: 1-2 more months.

### What Actually Determines the Choice

The decision should come down to:

1. **Language preference:** If the team prefers Python/JAX (likely, larger ecosystem), it's Path A vs B. If Julia is important, Path C.

2. **GPU priority:** If GPU acceleration for large circuits is a near-term priority, VAJAX is ahead. If it's a long-term goal, it's less decisive.

3. **Solver quality priority:** If you want the best transient solvers *now*, Cadnip wins (IDA is hard to beat). If you're willing to invest in integrating Diffrax properly, Path A or B can get close.

4. **OpenVAF:** This is non-negotiable for production use. VAJAX has it. The others would need to build or port it.

5. **Photonics:** If mixed electronic-photonic simulation is important, Circulax's work is the starting point regardless of which path you choose.

6. **Developer preferences:** Each developer has made deep architectural choices. The most productive path is the one that lets each person contribute what they're best at without requiring them to abandon their approach.

---

## 9. A Pragmatic Middle Path

Rather than "pick one core and port everything," consider:

**Shared infrastructure, separate frontends:**

1. **OpenVAF bindings** as a shared library — VAJAX's openvaf_jax already works. Could be extracted and made reusable.
2. **SPICE parser** as a standalone tool — Cadnip's parser outputs an intermediate representation that any simulator can consume.
3. **Benchmark suite** — shared test circuits with reference results from VACASK/ngspice for cross-validation.
4. **Photonic components** — Circulax's component library, extracted as a reusable package.

Each simulator keeps its own solver/engine but shares the hard-to-build infrastructure pieces. This is less ambitious than a full merge but more realistic for three independent developers.

---

## 10. Summary

The first version of this study over-weighted VAJAX based on commit counts and LOC. Here's the corrected picture:

| What matters | Best | Why |
|---|---|---|
| Transient solver quality | **Cadnip** | IDA (production BDF) + Rosenbrock via DifferentialEquations.jl |
| Verilog-A / PDK models | **VAJAX** | OpenVAF with proper static analysis, validated at scale |
| GPU acceleration | **VAJAX** | Actually working, tested to 133K nodes |
| Analysis breadth | **VAJAX** | DC, AC, tran, noise, HB, xfer, corners |
| Architecture & code quality | **Circulax** | Cleanest design, best use of library ecosystem |
| Photonic simulation | **Circulax** | Unique capability, no equivalent in others |
| Netlist parsing | **Cadnip** | Full SPICE + Spectre parser |
| Convergence robustness | **Cadnip** ≈ VAJAX | Both have GMIN + source stepping; Cadnip also has pseudo-transient init |
| Scalability (tested) | **VAJAX** | Only one tested beyond ~100 nodes |
| Solver ecosystem leverage | **Cadnip** > Circulax > VAJAX | DiffEq.jl > Diffrax (underused) > custom from scratch |

There is no clear winner. The "right" choice depends on priorities, and the most productive collaboration might be sharing infrastructure rather than forcing a single core.
