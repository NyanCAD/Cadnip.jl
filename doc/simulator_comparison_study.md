# Comparative Study: Cadnip.jl vs VAJAX vs Circulax vs VACASK

**Date:** 2026-03-07 (updated 2026-03-12: VACASK added)
**Purpose:** Evaluate open-source analog circuit simulator projects to identify the best path forward for joining forces. VACASK is included as the reference C++ implementation that all others benchmark against.

---

## 1. Honest Context

All four projects are single-developer efforts. Metrics like lines of code and commit counts are misleading:

- **Cadnip** (~18K src): Much of this is parser/codegen code inherited from CedarSim. The actual MNA simulator engine is ~7,600 LOC. CedarSim has a longer private history at JuliaComputing.
- **VAJAX** (~65K): Includes ~4,500 LOC of OpenVAF bindings, ~4,500 LOC of SPICE netlist converter, ~10K LOC of tests. Core solver engine is ~8-10K LOC. The MNA port to Cadnip and vajax as a whole relied more on vibe coding, which inflates commits/LOC.
- **Circulax** (~4.8K): Smallest and cleanest, but this is genuinely less code rather than hiding complexity — many features simply don't exist yet. Well-designed code by a developer who clearly cares about architecture.
- **VACASK** (~45K C++): The most mature codebase. Written in C++20 by Árpád Bűrmen (University of Ljubljana). Includes full netlist parser, all standard SPICE analyses, harmonic balance, and a comprehensive benchmark suite. VACASK is already used as the reference simulator for validating VAJAX's results.

What matters is: **what actually works, what's actually tested, and what would need to happen to make each one the unified core.** VACASK occupies a different position than the other three — it's a traditional compiled-language simulator, not a Python/JAX or Julia project. It's included because: (1) it's the performance bar everyone else is trying to beat, (2) its architecture decisions are instructive, and (3) collaboration with VACASK (shared benchmarks, shared device models via OpenVAF/VADistiller) is already happening.

---

## 2. What Each Simulator Actually Has (Feature Truth Table)

### 2.1 Solver & Analysis Capabilities

| Capability | Cadnip | VAJAX | Circulax | VACASK |
|---|---|---|---|---|
| **DC Operating Point** | Newton-Raphson via NonlinearSolve.jl (RobustMultiNewton + LevenbergMarquardt + PseudoTransient) | Custom NR in `jax.lax.while_loop` | NR via Optimistix | Custom NR with residual convergence check |
| **GMIN stepping** | Yes (exponential backoff, max 20 steps) | Yes (1e-3 → 1e-13, adaptive) | No | Yes (SPICE3/adaptive gmin, gdev and gshunt modes) |
| **Source stepping** | Yes (max 50 steps) | Yes (0→100%, with GMIN fallback) | No | Yes (SPICE3-style source stepping) |
| **Homotopy chain** | Newton → GMIN → Source | gdev → gshunt → src (configurable) | None | Configurable chain with state storage between methods |
| **Transient solver** | Sundials IDA (DAE), Rosenbrock (ODE) via DifferentialEquations.jl | Custom BE/Trap/Gear2 (2nd order max) | Backward Euler only (Diffrax has more but lacks mass matrix support — see Section 6) | BE, Trapezoidal, Gear (BDF), Adams-Moulton. All with variable-step coefficients via general IntegratorCoeffs framework |
| **Adaptive timestepping** | IDA's built-in (production-grade) | Custom LTE-based, polynomial predictor (2nd order) | PIDController via Diffrax | Predictor-corrector LTE control with polynomial extrapolation predictor, breakpoint handling |
| **Higher-order methods** | Rodas5P (5th order Rosenbrock), IDA (variable order BDF) | Gear2 max (2nd order) | BE only (1st order) | General-order BDF/AM via IntegratorCoeffs (computes coefficients for arbitrary order from timestep history). Default Gear2/Trap |
| **AC small-signal** | Yes (descriptor state-space, frequency sweep) | Yes (linearization + sweep) | No | Yes (linearized, frequency sweep) |
| **Noise analysis** | No (stubs return 0.0) | Yes (thermal/shot/flicker, small-signal) | No | Yes (full noise analysis with per-device noise sources) |
| **Harmonic balance** | No | Yes (DDT-based) | Yes (FFT-based, but dense Jacobian limits scalability) | Yes (multitone HB with colocation method) |
| **Transfer function** | No | Yes (DC/AC) | No | Yes (DC and AC transfer function) |
| **Corner/PVT sweeps** | Parameter sweeps (Product/Tandem/Serial) | Yes (dedicated corner analysis) | No (manual loop) | Yes (arbitrary-depth nested parametric sweeps — anything sweepable can also be altered without re-elaboration) |
| **Oscillator init** | CedarUICOp pseudo-transient relaxation | Not mentioned | No | Nodesets + analysis results as nodesets (enables custom arc-length homotopy at netlist level) |

**Verdict on solvers:** VACASK has the most complete traditional solver implementation — it covers every standard SPICE analysis plus multitone HB, with a general-purpose integration coefficient framework that can compute BDF/AM/polynomial predictor coefficients at arbitrary order. Cadnip has the best solver *ecosystem* via DifferentialEquations.jl — IDA is a production-grade DAE solver used in industry, Rosenbrock methods are excellent for stiff circuits. VAJAX has the most analysis *types* among the Python projects but its custom transient solver is limited to 2nd order. Circulax is locked to BE — and contrary to first impressions, Diffrax's higher-order implicit solvers (Kvaerno3/4/5) are *not* usable for circuit simulation due to missing mass matrix / DAE support in the entire JAX ecosystem (see Section 6).

### 2.2 Verilog-A / Device Model Support

| Aspect | Cadnip | VAJAX | Circulax | VACASK |
|---|---|---|---|---|
| **VA compiler** | Custom Julia parser + codegen | OpenVAF (Rust, industry-standard) | None | OpenVAF-reloaded (OSDI 0.4, fork with extensions) |
| **VA approach** | Parse VA → generate Julia stamp functions | OpenVAF MIR → JAX operations | N/A | OpenVAF → .osdi shared library → runtime dlopen via OSDI API |
| **PSP103** | Compiles but generates 100K+ statement functions that blow up LLVM SROA | Works, validated against VACASK + ngspice | N/A | Reference implementation, validated with benchmarks |
| **BSIM4** | Via VADistillerModels (partial) | Via OpenVAF (validated) | N/A | Yes (native VA + VADistiller converted SPICE version) |
| **BSIM3** | No | Via OpenVAF | N/A | Yes (both native VA from Cogenda and VADistiller SPICE conversion) |
| **BSIMBULK** | No | No | N/A | Yes (106.2.0) |
| **VBIC** | No | No | N/A | Yes (1.3 — 3, 4, and 5 terminal variants) |
| **Legacy SPICE models** | No | No | N/A | Yes (~20 models via VADistiller: MOS1-3,6,9, VDMOS, JFET, MESFET, Gummel-Poon BJT, BSIM3/4) |
| **Model validation** | Basic tests, some regression | 3-way comparison (VACASK + ngspice) | N/A | Comprehensive benchmarks vs ngspice and Xyce on 5 circuits |
| **Production models tested** | PSP103 (with LLVM issues), BSIM4 | PSP103, BSIM4, EKV, HiSIM | None | PSP103, BSIM3, BSIM4, BSIMBULK, VBIC, plus all VADistiller legacy models |
| **Static analysis** | Minimal (parser does less optimization) | OpenVAF does full static analysis, node collapse, parameter caching | N/A | Full OpenVAF static analysis (same compiler) |
| **Model variants** | N/A | N/A | N/A | `sn` (simplified noise, no extra nodes — fastest), `default` (no extra output nodes, full noise), `full` (all output variables and noise) |

**Verdict on VA:** VACASK has the most comprehensive device model library — it's the only project with both native Verilog-A models AND converted legacy SPICE3 models via VADistiller, plus multiple model variants optimized for different use cases. VAJAX uses the same OpenVAF compiler (though the original, not the -reloaded fork) and has validated its models against VACASK. Cadnip's custom VA codegen is fundamentally limited — complex models like PSP103 generate functions so large that LLVM chokes on them. This is a structural limitation that needs OpenVAF integration to solve.

### 2.3 Circuit Scale & Validation

| Circuit | Cadnip | VAJAX | Circulax | VACASK |
|---|---|---|---|---|
| Simple RC/RLC | Tested | Tested | Tested | Tested (1M timesteps, 0.94s) |
| Diode circuits | Tested | Graetz (99.5% — stiff transition issue) | Diode clipper tested | Graetz (1M steps, 1.89s), voltage multiplier (500K steps, 0.97s) |
| BJT amplifiers | Audio integration tests (CE, emitter degen) | Converter exists, not validated in solver | Ebers-Moll model, basic tests | VBIC BJT ring oscillator (via VADistiller) |
| MOSFET circuits | PSP103 (with LLVM caveats) | PSP103 ring osc, c6288 (86K MOSFETs), mul64 (266K MOSFETs) | Square-law only | PSP103 ring osc (18 transistors, 1.18s), c6288 (10112 transistors, 57.98s) |
| Max scale tested | ~100s of nodes | ~133K nodes (mul64 on GPU) | ~10s of nodes | 25,380 nodes (c6288), 10,112 transistors |
| Memory usage | Not benchmarked | Not benchmarked | Not benchmarked | 138.5 MB for c6288 (vs Xyce 775.7 MB, ngspice+KLU 200.5 MB) |
| Reference comparison | Against expected values | VACASK (C++) + ngspice | Against expected values | Against ngspice + Xyce with matched timepoints/iterations |

**Verdict on scale:** VAJAX is the only Python/Julia project that has been tested at serious circuit scale. The mul64 benchmark (266K MOSFETs, ~133K nodes) is genuinely impressive. VACASK holds the performance crown on the c6288 benchmark — 57.98s vs ngspice 71.81s vs Xyce 151.57s — with the lowest memory footprint (138.5 MB). VACASK's comprehensive benchmark methodology (6 runs, cache warming, matched timepoints, std dev <2%) sets the bar for how performance comparisons should be done. Cadnip and Circulax are untested beyond small circuits.

### 2.4 Netlist & Input Format Support

| Aspect | Cadnip | VAJAX | Circulax | VACASK |
|---|---|---|---|---|
| **SPICE netlist** | Full parser (SpectreNetlistParser.jl, ~10K LOC) | Converter from ngspice/HSPICE/LTSpice (~4.5K LOC) | None | Ngspice converter (python/ng2vc.py, under development) |
| **Spectre netlist** | Yes (native) | No | No | Spectre-like native syntax (.sim files) — the original |
| **Native format** | Julia code (compiled circuits) | VACASK .sim files | SAX Python dicts | .sim netlist files + C++ API (ParserTables builder) |
| **Subcircuits** | Yes (.subckt hierarchy) | Yes | No | Yes (full parameterized hierarchy with conditional blocks) |
| **Conditional netlist** | No | No | No | Yes (@if/@elseif/@else/@end blocks — topology can change based on parameters) |
| **On-the-fly VA compilation** | N/A | N/A | N/A | Yes (detects .va files in `load` directive, invokes OpenVAF automatically) |
| **Xschem integration** | No | No | No | Yes (schematic entry, analysis setup, results display) |
| **PDK support** | No | No | No | Yes (IHP Open PDK SG13G2 demonstrated) |

### 2.5 Platform & Ecosystem

| Aspect | Cadnip | VAJAX | Circulax | VACASK |
|---|---|---|---|---|
| **Language** | Julia | Python/JAX | Python/JAX | C++20 |
| **GPU** | Two paths: (1) CuArray for large circuits (easy to get working, but CPU-orchestrated — won't match JAX's fused execution), (2) EnsembleGPUKernel for massively parallel sweeps (natural fit — see Section 9) | Yes, working (CUDA via JAX). 12.8x speedup on mul64 | Theoretically via JAX (untested, same JAX options as VAJAX) | No (planned: MPI, CUDA, OpenMP on roadmap) |
| **AD** | ForwardDiff (forward-mode, through entire solver) | JAX AD (forward + reverse). Less natural than Cadnip's for circuit-specific use | JAX AD (forward + reverse). Cleanest integration of the three | None (analytic Jacobians from OpenVAF OSDI) |
| **Solver libraries** | DifferentialEquations.jl (IDA, Rosenbrock, many more) | Custom from scratch | Diffrax + Optimistix (but Diffrax lacks mass matrix/DAE support — see Section 6) | All custom, KLU only external dependency |
| **ML integration** | Julia ML ecosystem (Flux, Lux) | JAX/Flax/Optax native | JAX/Flax/Optax native | None (not a goal — VACASK targets traditional circuit simulation) |
| **Startup time** | Slow (Julia JIT compilation, minutes for first sim) | Fast | Fast | Instant (compiled binary, including on-the-fly VA compilation) |
| **PyPI/package** | No | Yes (`pip install vajax`) | Yes (pixi) | Pre-built .deb/.tgz/.zip packages, nightly builds |
| **Embeddable API** | Via Julia `include` / package | Python import | Python import | C++ library API (ParserTables builder, Circuit, Analysis classes — see demo/api/) |
| **Dependencies** | Julia ecosystem (SciML, SparseArrays, etc.) | Python/JAX/NumPy/SciPy | Python/JAX/Equinox/Diffrax | Only KLU + Boost + toml++ (minimal by design) |
| **License** | MIT | MIT | Apache 2.0 | AGPL 3.0 |

### 2.6 Unique Capabilities

| Capability | Only in |
|---|---|
| Photonic circuit simulation (S-param → Y-matrix, ring resonators, MZI) | Circulax |
| SPICE/Spectre netlist parsing (native, not conversion) | Cadnip |
| GPU-accelerated large-circuit simulation (actually tested) | VAJAX |
| IDA DAE solver (variable-order BDF, production-grade) | Cadnip |
| Mixed electronic-photonic simulation | Circulax |
| SAX photonic library integration | Circulax |
| `@component` decorator API for model definition | Circulax |
| Zero-allocation transient inner loop | Cadnip |
| Pseudo-transient oscillator initialization | Cadnip |
| Conditional netlist blocks (@if/@else) with topology changes | VACASK |
| Multi-topology simulation without restart | VACASK |
| Automatic partial re-elaboration on parameter change | VACASK |
| VADistiller legacy SPICE3→VA model conversion | VACASK |
| Continuation bypass (skip first NR eval in continuation mode) | VACASK |
| Inactive element bypass with configurable tolerances | VACASK |
| Residual-based convergence test (stricter than SPICE) | VACASK |
| Multitone harmonic balance | VACASK |
| Xschem schematic integration | VACASK |
| IHP Open PDK support | VACASK |
| Analysis results as nodesets for subsequent analyses | VACASK |

---

## 2.7 VACASK Deep Dive: Architecture & Performance

VACASK deserves a dedicated section because it's the reference implementation that all other projects benchmark against, and its architectural decisions illuminate the design space for circuit simulators.

### Why VACASK Matters for This Comparison

VACASK is NOT a competitor to join forces with — it's a different kind of project:
- **Language:** C++20, not Python or Julia. No ML integration, no AD, no GPU.
- **Goal:** Be a better SPICE, not a differentiable simulator or ML-friendly tool.
- **License:** AGPL 3.0, which restricts proprietary use more than MIT/Apache.

But it matters because:
1. **Performance bar:** VACASK is 19-24% faster than ngspice and 2.6x faster than Xyce on the c6288 benchmark, with 5.6x less memory than Xyce. Any Julia/Python simulator needs to understand why.
2. **Shared infrastructure:** VAJAX already validates against VACASK. Cadnip uses VACASK's benchmarks. VADistiller models work in both VACASK and Cadnip.
3. **Design lessons:** VACASK's continuation bypass, residual convergence testing, and inactive element bypass are techniques that inform Cadnip's optimization path.
4. **OpenVAF ecosystem:** VACASK drives OpenVAF-reloaded development, which benefits everyone using OpenVAF.

### Architecture (~45K LOC C++20)

```
VACASK Architecture:

Netlist (.sim)  ──→  Parser  ──→  ParserTables  ──→  Circuit
                                                        │
                                  OpenVAF (.va)  ──→  OSDI (.osdi)  ──→  Device instances
                                                        │
                              ┌──────────────────────────┘
                              │
                    ┌─────────┼──────────────────────────────────────┐
                    │         │                                      │
                  Analysis  AnalysisCore  ──→  NRSolver              │
                    │         │                   │                   │
                    │    ┌────┴────┐          KluRealMatrix          │
                    │    │         │              │                   │
                    OP  Tran   AC/Noise/HB    SparsityMap            │
                    │    │         │              │                   │
                    │  IntegratorCoeffs       factor/solve           │
                    │    │                       │                   │
                    └────┼───────────────────────┘                   │
                         │                                           │
                    VectorRepository<double>  ── solution history    │
                         │                                           │
                    Homotopy chain (gdev → gshunt → src)            │
                         │                                           │
                    Sweep (arbitrary depth, auto re-elaboration)     │
                    └────────────────────────────────────────────────┘
```

**Key classes:**
- `Circuit`: Manages instances, models, subcircuits, topology. Supports conditional blocks and partial re-elaboration.
- `KluRealMatrix` / `KluComplexMatrix`: Sparse matrix wrappers around SuiteSparse KLU. Typed templates (`KluMatrixCore<IndexType, ValueType>`). SparsityMap tracks resistive vs reactive entries separately.
- `OpNRSolver`: Newton-Raphson with force slots for nodesets, ICs, and homotopy. Supports residual convergence check (not just delta convergence like SPICE).
- `TranNRSolver`: Extends OpNRSolver with integration coefficients.
- `IntegratorCoeffs`: General-purpose integration coefficient framework. Supports Adams-Moulton, BDF, Adams-Bashforth, and polynomial extrapolation at arbitrary order. Computes coefficients from variable timestep history by solving a linear system.
- `Homotopy`: Base class with gdev, gshunt, spice3Gmin, src, spice3Src derived classes. State storage between methods for continuation.
- `VectorRepository<T>`: Circular buffer of solution vectors for time history. Enables O(1) push/pop for BDF history management.

### What Makes VACASK Fast

VACASK's c6288 advantage over ngspice (57.98s vs 71.81s, 20.7ms/iter vs 15.6ms/iter with continuation bypass) comes from several techniques:

1. **Continuation bypass** (`nr_contbypass`, default enabled): In continuation mode (accepted transient step, sweep step, converged homotopy step), skip device evaluation in the first NR iteration — reuse the previous Jacobian and residual. Since continuation points usually converge in 2-3 iterations, this saves ~33-50% of device evaluations. This is the single biggest optimization, taking c6288 from 63.19s to 48.34s (23% improvement).

2. **Residual-based convergence test** (`nr_residualcheck`, default enabled): After delta convergence, also check that KCL residuals are within tolerance. This is *stricter* than SPICE (which only checks deltas) and actually *slows down* simulation slightly (3487 vs 3090 iterations on c6288) but produces more accurate results. Can be disabled for speed.

3. **Inactive element bypass** (`nr_bypass`, default disabled): Skip device evaluation for instances whose terminal voltages haven't changed significantly. Similar to SPICE's bypass mode but with configurable tolerances (`nr_convtol`, `nr_bypasstol`). Effective for large circuits where most elements are inactive most of the time (digital circuits).

4. **Minimal dependencies:** Only KLU + C++ stdlib. No framework overhead, no memory management layers, no abstraction taxes. The entire simulator is a single executable.

5. **Direct pointer binding:** Devices bind directly to matrix element pointers at elaboration time (`Device::bind()` → `MatrixAccess::valuePtr()`). During simulation, stamping is a direct pointer write — no hash lookups, no COO assembly, no sparse matrix rebuilds.

6. **General IntegratorCoeffs framework:** Instead of hardcoding BE/Trap/Gear2 coefficients, VACASK computes them from the timestep history by solving a linear system. This means arbitrary-order BDF/AM is supported by changing one parameter, without code changes. The coefficient computation is done once per timestep, not per NR iteration.

### Performance Data (from VACASK benchmark suite)

All benchmarks on AMD Threadripper 7970, single-threaded, KLU sparse solver, 5 timed runs after 1 warmup:

**RC circuit** (2 elements, ~1M timesteps, linear):

| Simulator | Time (s) | Timepoints | Iterations |
|---|---|---|---|
| Xyce | 9.39 | 1,011,527 | 2,029,571 |
| Xyce-fast | 4.12 | 1,011,527 | 2,029,571 |
| Gnucap | 8.54 | 1,006,982 | 2,018,061 |
| Ngspice | 1.31 | 1,006,013 | 2,012,031 |
| **VACASK** | **0.94** | 1,005,006 | 2,010,014 |

**Graetz rectifier** (4 diodes + RC, ~1M timesteps):

| Simulator | Time (s) | Timepoints | Rejected | Iterations |
|---|---|---|---|---|
| Gnucap | 15.16 | 1,000,026 | 12 | 3,459,503 |
| Xyce | 10.60 | 1,000,002 | 0 | 2,000,014 |
| Xyce-fast | 5.34 | 1,000,002 | 0 | 2,000,014 |
| Ngspice | 2.21 | 1,000,008 | 0 | 2,000,024 |
| **VACASK** | **1.89** | 1,000,003 | 0 | 2,000,277 |

**Voltage multiplier** (4 diodes + 4 caps, ~500K timesteps):

| Simulator | Time (s) | Timepoints | Rejected | Iterations |
|---|---|---|---|---|
| Gnucap | 9.94 | 520,797 | 739 | 2,300,992 |
| Xyce | 5.51 | 502,341 | 1,270 | 1,041,833 |
| Xyce-fast | 2.78 | 502,341 | 1,270 | 1,041,833 |
| Ngspice | 1.16 | 500,467 | 957 | 1,019,733 |
| **VACASK** | **0.97** | 500,056 | 3 | 1,001,233 |

**9-stage CMOS ring oscillator** (18 PSP103 transistors):

| Simulator | Time (s) | Timepoints | Rejected | Iterations |
|---|---|---|---|---|
| Xyce | 3.33 | 27,310 | 0 | 95,462 |
| Xyce-fast | 3.10 | 27,310 | 0 | 95,462 |
| Ngspice | 1.60 | 20,556 | 1,037 | 80,018 |
| **VACASK** | **1.18** | 26,066 | 0 | 81,875 |

**C6288 16×16 multiplier** (10,112 PSP103 transistors, 25,380 nodes):

| Simulator | Time (s) | Timepoints | Rejected | Iterations | Time/iter |
|---|---|---|---|---|---|
| Xyce | 151.57 | 1,013 | 37 | 3,559 | 42.6ms |
| Ngspice | 71.81 | 1,020 | 1 | 3,474 | 20.7ms |
| **VACASK** (default) | **57.98** | 1,021 | 7 | 3,487 | 16.6ms |
| **VACASK** (no rtc) | **48.34** | 1,024 | 8 | 3,090 | 15.6ms |

**Memory usage** (c6288):

| Simulator | Sparse solver | Memory (MB) |
|---|---|---|
| Xyce | KLU | 775.7 |
| Ngspice | KLU | 200.5 |
| Ngspice | SPARSE | 135.3 |
| **VACASK** | KLU | **138.5** |

### VACASK's Roadmap (from F-Si wiki)

Planned features: damped pseudo-transient analysis, S-parameter analysis, advanced harmonic balance, parallelization (MPI, CUDA, OpenMP), more benchmarks, Skywater PDK support, SPICE3 legacy model support improvements.

### What VACASK Lacks (And Why It Matters)

| Missing | Impact |
|---|---|
| **GPU acceleration** | Can't compete with VAJAX on large parallel workloads. Roadmap item but no code yet. |
| **Automatic differentiation** | No parameter sensitivity, no gradient-based optimization, no ML integration. Analytic Jacobians only (from OpenVAF). |
| **Higher-level solver library** | Custom solver means custom bugs. No access to method families beyond what's hand-coded. Compare Cadnip's one-line access to IDA, Rosenbrock, Radau, etc. |
| **Differentiable simulation** | The core value proposition of VAJAX and Circulax is missing entirely. No way to backprop through a simulation for ML training. |
| **Python/Julia ecosystem** | No Jupyter notebooks, no matplotlib integration beyond postprocessing scripts, no ML framework interop. |
| **AGPL license** | Restricts proprietary use. Companies may prefer MIT/Apache simulators for embedding. |

### Lessons for Cadnip

VACASK validates several optimization strategies that are directly applicable to Cadnip:

1. **Continuation bypass is the biggest win.** Cadnip's `DirectStampContext` (zero-allocation restamping) is the foundation — adding continuation bypass on top (skip builder call when continuing from a nearby solution) could yield 20-30% speedup on large transient simulations, matching VACASK's most impactful optimization.

2. **Residual convergence check is worth the cost.** VACASK enables it by default despite the speed penalty because it catches convergence failures that delta-only checking misses. Cadnip should consider adding this.

3. **Direct pointer binding matters.** VACASK's devices write directly to matrix element pointers. Cadnip's current COO → sparse assembly path adds overhead. The `DirectStampContext` already moves toward this, but full pointer binding (assign matrix pointers at elaboration, write through them during simulation) would close the gap.

4. **Minimal allocations are king.** VACASK's 138.5 MB footprint for 10K transistors comes from careful memory management — `VectorRepository` reuses buffers, `SparsityMap` is built once, matrices are zeroed not reallocated. Cadnip's GC-based approach (298 MiB + 36-62% GC overhead for a 2-element RC circuit) shows how much room there is to improve.

5. **General IntegratorCoeffs is elegant.** Instead of hard-coding integration methods, VACASK solves for coefficients from timestep history. Cadnip delegates this to IDA/Rosenbrock (which is fine), but the companion model projects (VAJAX, Circulax) could benefit from this approach when extending beyond Gear2.

---

## 3. Three Paths: What Would Each Require?

### Path A: VAJAX as Core

**What you'd get for free:** OpenVAF VA models, GPU acceleration, noise/AC/HB/transfer function analyses, large-circuit scalability, SPICE netlist converter, PyPI distribution.

**What's missing and needs work:**

| Gap | Effort | Notes |
|---|---|---|
| Better transient solvers | **Large** | Current custom BE/Trap/Gear2 is limited to 2nd order. Options: (a) extend BDF to order 3-5 in VAJAX's custom solver (reimplementing IDA), (b) integrate Diffrax — but Diffrax lacks mass matrix support (Section 6), so this only helps if mass matrix is contributed upstream first, (c) port DifferentialEquations.jl solvers to JAX. Option (a) is most practical but means maintaining a custom solver indefinitely. |
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
| Unlock Diffrax solvers | **Large — blocked upstream** | Currently locked to BE. Diffrax has NO native mass matrix / DAE support (see Section 6). The only workaround is a user-contributed ImplicitEuler-only hack ([diffrax#710](https://github.com/patrick-kidger/diffrax/issues/710)). Extending to Kvaerno/RadauIIA requires either contributing mass matrix support to Diffrax (significant upstream effort across all implicit solver types) or reimplementing each solver from scratch. This is NOT a small engineering task. |
| Scale to large circuits | **Unknown** | The sparse infrastructure exists (KLU, BiCGStab) but has never been tested beyond ~10 nodes. May need significant work. |
| Subcircuit hierarchy | **Medium** | No .subckt support. |
| Parameter sweeps | **Small** | Just engineering, no algorithmic challenge. |
| Dense HB Jacobian | **Medium** | Current HB won't scale. Needs sparse HB formulation. |

**Risk:** The OpenVAF integration is a massive effort and is the critical path to production models. Without VA support, you can't run real PDK models, which limits the simulator to educational/research use. The absence of convergence aids (GMIN stepping, source stepping, homotopy) is also telling — these aren't optional features for production circuits. Any circuit with more than a handful of diodes or transistors will fail to converge without them. Their absence suggests Circulax has not yet been tested on circuits where convergence is non-trivial, which raises questions about what other production-readiness gaps may surface at scale.

### Path C: Cadnip as Core

**What you'd get for free:** DifferentialEquations.jl solver ecosystem (IDA, Rosenbrock — best transient solvers of the three), SPICE/Spectre parsing, ForwardDiff AD, convergence aids (GMIN + source stepping + pseudo-transient), subcircuit hierarchy, zero-allocation optimization.

**What's missing and needs work:**

| Gap | Effort | Notes |
|---|---|---|
| OpenVAF integration | **Large** | Replace custom VA codegen with OpenVAF. Need Julia bindings for OpenVAF (Rust). This solves the PSP103-blows-up-LLVM problem at the root — OpenVAF's static analysis produces much more compact output. |
| GPU acceleration | **Medium (sweeps) / Medium-Large (single circuit)** | Two paths: EnsembleGPUKernel for parallel sweeps is tractable (parameter sweeps already work, just need GPU dispatch). CuArray for large single circuits is easier than it looks (no allocation-free requirement), but performance won't match JAX's fused execution for smaller circuits. See Section 9. |
| Photonic simulation | **Medium** | Port Circulax's photonic components. Julia's type system could make this elegant (complex-valued MNA via parametric types). |
| Noise analysis | **Medium** | AC infrastructure exists. Need per-device noise sources. |
| Harmonic balance | **Medium-Large** | Not implemented. Could port Circulax's approach. |
| Transfer function | **Small** | Standard analysis, AC infrastructure exists. |
| Python ecosystem access | **Medium** | PythonCall.jl works but adds complexity. Julia's ML ecosystem is smaller. |
| Startup latency | **Ongoing** | Julia's JIT compilation means minutes to first simulation. PackageCompiler.jl can help but adds deployment complexity. |
| Controlled sources | **Small** | Basic structure exists, needs completion. |

**Risk:** Julia's smaller ecosystem is both a strength (better numerical libraries) and weakness (fewer users, harder to hire contributors). The GPU story has two paths: EnsembleGPUKernel for parallel sweeps is tractable near-term; CuArray for large single circuits is easier than it looks but won't match JAX's fused execution performance (see Section 9). The VA codegen limitation (PSP103 → 100K statements → LLVM explosion) is a fundamental issue that needs OpenVAF to solve properly.

---

## 4. Architectural Portability: VAJAX ↔ Circulax (Code-Level Comparison)

A critical question: since both VAJAX and Circulax are Python/JAX MNA simulators, how much work is porting features between them — really? Having read the actual source code of both, here's what they look like inside.

### Device Interface: Different Abstractions, Same Shape

**Circulax** uses a `@component` decorator that compiles a physics function into an Equinox module:

```python
# circulax/components/base_component.py
@component(ports=("p1", "p2"))
def Resistor(signals: Signals, s: States, R: float = 1.0):
    i = (signals.p1 - signals.p2) / R
    return {"p1": i, "p2": -i}, {}  # f_dict, q_dict

# The decorator generates a solver-facing function:
#   _fast_physics(vars_vec, params, t) -> (f_vec, q_vec)
```

**VAJAX** uses OpenVAF-compiled functions via openvaf_jax:

```python
# vajax/devices/verilog_a.py
device = VerilogADevice.from_va_file("resistor.va")
# Produces: vmapped_split_eval(shared_params, device_params, shared_cache,
#           device_cache, simparams, limit_state)
#   -> (res_resist, res_react, jac_resist, jac_react, ...)
```

**Key difference:** Circulax devices return `(f_vec, q_vec)` — the Jacobian is computed externally via `jax.jvp`. VAJAX's OpenVAF devices return both residuals AND Jacobian entries directly from the compiled model (OpenVAF computes analytic derivatives internally). This is more efficient but means the device interface is fundamentally different — VAJAX devices produce 9-tuple outputs with separate resist/react residual AND Jacobian arrays.

### Assembly: Same Algorithm, Different Granularity

**Circulax** (`solvers/assembly.py`) — clean, compact:

```python
# For each component group, vmap the physics + jvp:
(f_l, q_l), (df_l, dq_l) = jax.vmap(
    partial(_primal_and_jac_real, physics_at_t1)
)(v_locs, group.params)
total_f = total_f.at[group.eq_indices].add(f_l)
total_q = total_q.at[group.eq_indices].add(q_l)
j_eff = df_l + (dq_l / dt)  # Effective Jacobian for transient
```

Components are grouped by type into `ComponentGroup` objects with batched `var_indices`, `eq_indices`, and `params` — all stacked JAX arrays for efficient vmap. The Jacobian is dense per-group (shape `(n_instances, n_vars, n_vars)`), extracted as non-zero values.

**VAJAX** (`analysis/mna_builder.py`) — more complex, handles two assembly modes:

```python
# COO mode: collect COO triples, assemble at end
(batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react,
 ...) = split_info["vmapped_split_eval"](
    shared_params, device_params_updated, shared_cache, cache, simparams, ...)

f_resist_parts.append(mask_coo_vector(res_idx, batch_res_resist.ravel()))
j_resist_parts.append(mask_coo_matrix(jac_row_idx, jac_col_idx, batch_jac_resist.ravel()))

# CSR direct mode: stamp directly into pre-allocated CSR array
csr_data = csr_data.at[model_positions].add(batch_jac_resist.ravel() + integ_c0 * ...)
```

VAJAX also handles full MNA augmentation (branch currents for voltage sources) inline during assembly, which adds complexity.

**Verdict:** The assembly is structurally similar — both vmap over batched component groups and scatter results into global arrays. But VAJAX carries significantly more complexity: separate resist/react channels, COO vs CSR mode switching, device limiting state, voltage source branch currents. This is not just style — it's needed for the features VAJAX supports (transient with reactive devices, large-circuit CSR optimization).

### Transient Solver: Fundamentally Different Integration

**Circulax** subclasses `diffrax.AbstractSolver`:

```python
# circulax/solvers/transient.py
class VectorizedTransientSolver(AbstractSolver):
    def step(self, terms, t0, t1, y0, args, solver_state, options):
        # Newton loop via Optimistix fixed_point:
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(newton_update_step, solver, y_pred, max_steps=20)
        return y_next, y_error, dense_info, new_state, result

# Used via Diffrax:
diffrax.diffeqsolve(terms=term, solver=tsolver, t0=t0, t1=t1, ...)
```

**VAJAX** uses `jax.lax.while_loop` with custom adaptive stepping:

```python
# vajax/analysis/transient/full_mna.py
# Custom NR with lax.while_loop, custom LTE-based adaptive timestep,
# integration methods (BE/Trap/Gear2), polynomial predictor
def _time_step_body(carry):
    # ... 200+ lines of NR solve + timestep control + checkpointing
    return new_carry

result = lax.while_loop(lambda c: c.step < max_steps, _time_step_body, init_carry)
```

**Verdict:** These are *not* plug-compatible. Circulax uses Diffrax's solver interface (inheriting `AbstractSolver`), which gives it access to Diffrax's step-size controllers and interpolation. VAJAX has a completely custom time integration loop. Swapping one for the other is NOT just "wrap MNA as ODETerm" — it requires rethinking how NR convergence interacts with time-step control.

### OpenVAF Integration: Deeper Coupling Than Expected

`openvaf_jax` is NOT just "load .so, call function." It's a MIR-to-JAX transpiler:

```python
# openvaf_jax/codegen/function_builder.py
class FunctionBuilder:
    def __init__(self, mir_func: MIRFunction):
        self.cfg = CFGAnalyzer(mir_func)      # Control flow graph analysis
        self.ssa = SSAAnalyzer(mir_func, self.cfg)  # SSA form analysis
```

It parses OpenVAF's MIR (mid-level IR), analyzes control flow, resolves SSA phi nodes, does constant propagation, and generates Python/JAX AST nodes. The output is a JIT-compilable JAX function. This is a proper compiler backend, not a thin wrapper.

The generated functions return a specific tuple structure that VAJAX's `mna_builder.py` expects:
`(res_resist, res_react, jac_resist, jac_react, lim_rhs_resist, lim_rhs_react, noise_resist, noise_react, limit_state_out)`

To use this in Circulax, you'd need an adapter that:
1. Takes openvaf_jax's 9-tuple output
2. Combines resist+react into `(f_vec, q_vec)` format
3. Either passes through the Jacobian (bypassing Circulax's `jax.jvp` path) or discards it and lets Circulax re-derive it via AD

Option 2 wastes the analytic Jacobian. Option 1 requires changes to Circulax's assembly loop to accept pre-computed Jacobians.

### What Differs and What Doesn't (Corrected)

| Aspect | VAJAX | Circulax | Portability |
|---|---|---|---|
| **Device output** | 9-tuple (resist/react residuals + Jacobians + noise + limiting) | 2-tuple `(f_vec, q_vec)`, Jacobian via AD | **Different** — adapter needed, design choice about Jacobian |
| **Assembly** | COO or CSR direct stamping, resist/react separate | Dense per-group Jacobian blocks, single effective J | **Similar pattern** but VAJAX is more granular |
| **NR loop** | `jax.lax.while_loop` (manual convergence check) | `optimistix.fixed_point` (library) | **Different API**, same math. Convergence criteria differ |
| **Transient** | Custom BE/Trap/Gear2 + adaptive LTE + predictor | Diffrax `AbstractSolver` subclass (BE only) | **Fundamentally different** integration approaches |
| **Component batching** | `jax.vmap` over device groups | `jax.vmap` over `ComponentGroup` | **Same pattern** |
| **Sparse format** | COO → dense or CSR (with Spineax/UMFPACK) | Dense per-group → BCOO sparse via klujax | **Different** sparse strategies |
| **Netlist** | Custom `.sim` parser + SPICE converter | SAX-compatible dicts + networkx connectivity | **Different** format, not interchangeable |
| **Complex support** | Not in core solver | Full unrolled complex assembly (real/imag block format) | **Circulax only** — needed for photonics |

### Porting Effort (Corrected with Code Evidence)

**OpenVAF from VAJAX → Circulax:**
`openvaf_jax` IS self-contained (separate package, own `__init__.py`). But the integration is not trivial:
- openvaf_jax generates functions returning 9-tuple with separate resist/react Jacobians
- Circulax expects `(f_vec, q_vec)` with Jacobian computed via `jax.jvp`
- Options: (a) adapter that discards openvaf Jacobians, re-derives via AD — wastes work but clean; (b) modify Circulax assembly to accept pre-computed Jacobians — more efficient but touches core loop
- Must also handle: device parameter preparation, voltage-to-branch mapping, init/eval split, simparams, device limiting state

**Effort: 2-4 weeks** for option (a), **4-6 weeks** for option (b). Not months, but not a weekend either.

**Diffrax into VAJAX (replacing custom solver):**
VAJAX's custom transient loop (`full_mna.py`) is deeply integrated:
- Custom adaptive timestep with LTE control
- Polynomial predictor (2nd order)
- Force-accept after 5 rejects
- Checkpoint intervals for GPU memory management
- Integration method enum (BE/Trap/Gear2) with coefficients baked into the build_system function

Replacing this with Diffrax means:
- Wrapping `build_system` as a Diffrax-compatible vector field (the resist/react split complicates this)
- Giving up VAJAX's checkpoint-based GPU memory management (Diffrax doesn't have this)
- Mapping Diffrax's step controller to VAJAX's LTE-based control
- The `integ_c0, integ_c1, integ_d1` coefficients that `build_system` takes as args are tied to the integration method — Diffrax handles this differently

**Effort: 4-8 weeks.** The build_system interface is entangled with the integration method. This is a significant refactor, not a wrapper.

**Important caveat (see Section 6):** Even after this effort, Diffrax's higher-order implicit solvers (Kvaerno3/4/5) would NOT be usable for circuit simulation because Diffrax lacks mass matrix / DAE support. The swap would give VAJAX Diffrax's adaptive stepping infrastructure but not better integration methods. The more pragmatic path is extending VAJAX's existing BDF solver to higher orders.

**GMIN/source stepping:**
VAJAX already has this (`analysis/homotopy.py`). Circulax doesn't.

Porting to Circulax: VAJAX's homotopy wraps the NR solve in a loop, adjusting `gmin`/`gshunt`/`srcFact` parameters. This IS mostly algorithm-level code.

**Effort: 1-2 weeks.** Need to add gmin/srcFact parameters to Circulax's component evaluation (currently not parameterized for this).

### What's genuinely hard to port

| Task | Why it's actually hard |
|---|---|
| **VAJAX's GPU scaling** | CSR direct stamping, Spineax/cuDSS GPU solver integration, checkpoint-based memory management, dense↔sparse auto-switching. This is ~2000 LOC of GPU-specific engineering. |
| **VAJAX's validation suite** | Comparison infrastructure against VACASK + ngspice. Not code to port — it's a methodology and reference data. |
| **Cadnip's SPICE parser** | Written in Julia (~20K LOC). Language barrier is real. |
| **Circulax's complex MNA** | Unrolled real/imag block format for frequency-domain analysis. VAJAX would need this for photonic support. |

### Bottom line (corrected)

VAJAX and Circulax are more similar than different at the *algorithm* level, but the *implementation* details matter more than I initially estimated. The key friction points are:

1. **Device interface mismatch** — openvaf's 9-tuple with pre-computed Jacobians vs Circulax's AD-derived Jacobians. This is a design choice, not just a format difference.
2. **Transient solver entanglement** — VAJAX's build_system takes integration coefficients as args, coupling the MNA assembly to the time integration method. Can't just swap the outer loop.
3. **Sparse strategy difference** — VAJAX's COO/CSR with UMFPACK/Spineax vs Circulax's BCOO/klujax. Different sparse ecosystem choices.

Most features CAN be ported in weeks (not months), but the estimates should be 2-6 weeks per feature, not days. The exception is GPU scaling work which is genuinely hard to replicate.

---

## 5. Companion Model vs Mass Matrix: Reactive Device Formulations

This is a fundamental design choice with real consequences for accuracy and stiffness.

### Background: How Simulators Handle ddt(Q(V))

When a device has a reactive contribution `I = ddt(Q(V))` — like a junction capacitor where `Q` is a nonlinear function of voltage — there are two fundamentally different approaches:

**Companion Model (ngspice, VAJAX, Circulax):**
Discretize `ddt(Q)` inside the solver's residual using the integration method. For backward Euler:
```
I_cap = (Q(V_n+1) - Q(V_n)) / dt
```
The device returns `f_resist` (DC currents) and `f_react` (charge contributions) separately. The solver combines them: `residual = f_resist + c0*Q_new + c1*Q_prev + ...`. The system is a nonlinear algebraic equation at each timestep — NOT a DAE. All states are node voltages. Charges are "hidden" inside the companion model. This applies to ALL capacitive devices — both constant and voltage-dependent.

This is what VAJAX does (`integration.py`):
```python
# dQ/dt = c0 * Q_new + c1 * Q_prev + d1 * dQdt_prev + c2 * Q_prev2
dQdt = coeffs.c0 * Q_new + coeffs.c1 * Q_prev
```

And Circulax (`transient.py`):
```python
residual = total_f + (total_q - q_prev) / dt  # BE discretization
j_eff = df_l + (dq_l / dt)  # Effective Jacobian
```

Both evaluate Q(V) at each Newton iteration, compute the companion current, and add it to the residual. The integration coefficients are baked into the system.

**Mass Matrix Formulation (Cadnip):**
Cadnip does NOT use companion models for capacitors at all — not even for basic constant capacitors. Instead, ALL reactive devices are handled through a mass matrix formulation:
```
C * dx/dt + G*x = b    (mass matrix ODE)
```
where `C` is the mass matrix containing capacitance entries and `x` includes both voltages and (for V-dependent caps) charge states.

For **constant** capacitors, the capacitance value is stamped directly into the `C` mass matrix:
```julia
# devices.jl: stamp_C!() — constant cap goes straight into mass matrix
ctx.C[i, j] += C_value
```
The ODE solver (IDA, Rosenbrock) handles the time integration internally. There is no companion model, no integration coefficient, no charge history — the solver sees `C*dV/dt = I` and integrates it with whatever method it chooses (variable-order BDF, Rosenbrock, etc.).

For **voltage-dependent** capacitors (junction caps, MOSFET gate charge), the mass matrix entry `C(V)` would be voltage-dependent, which breaks Rosenbrock methods that factor `(I - γ*dt*J)` assuming constant mass. Cadnip solves this by reformulating as charge state variables:

### Cadnip's Charge State Approach for V-Dependent Caps

Cadnip detects voltage-dependent capacitors via multi-pass evaluation:

```julia
# build_with_detection(): Run builder 5 times with random x
# Compare Q/V ratios - if they differ, capacitance is V-dependent
for pass in 1:N_DETECTION_PASSES
    x = random_operating_point()
    builder(params, spec, 0.0; x=x, ctx=ctx)
    # detect_or_cached!() compares Q/V ratios across passes
end
```

When a V-dependent cap is detected, it's reformulated as an explicit charge state variable:

```julia
# stamp_charge_state!(): Add q as explicit state with constant mass entry
# Instead of: C(V)*dV/dt (V-dependent mass matrix — breaks Rosenbrock)
# Reformulate as:
#   dq/dt = I          (constant mass entry = 1/CHARGE_SCALE)
#   q = Q(V)           (algebraic constraint, Newton-solved)
```

This adds an extra state variable per voltage-dependent capacitor, but the mass matrix `C` remains **constant** — critical for Rosenbrock methods which factor `(I - γ*dt*J)` once per step assuming constant mass.

The charge is scaled by `CHARGE_SCALE = 1e12` to improve Jacobian conditioning (charges are O(1e-12), voltages are O(1)).

**Note:** The term "Newton companion model" appears in Cadnip's `devices.jl` (around line 1035), but this refers to **Newton linearization of nonlinear I-V curves** (diode exponential → conductance + current source), not to capacitor time integration. This is standard Newton-Raphson linearization, unrelated to the "companion model" concept in the context of reactive device formulations.

### Summary: Who Does What

| | Constant Cap | V-Dependent Cap |
|---|---|---|
| **ngspice / VAJAX / Circulax** | Companion model: `I = (Q_new - Q_prev)/dt` | Companion model: same formula, Q(V) evaluated at each NR iteration |
| **Cadnip** | Direct C matrix stamp: solver integrates `C*dV/dt = I` | Charge state variable: `dq/dt = I`, `q = Q(V)` — keeps C constant |

### Trade-offs

| Aspect | Companion Model (VAJAX, Circulax) | Mass Matrix (Cadnip) |
|---|---|---|
| **System size** | N (node voltages only) | N + K (voltages + charge states for V-dep caps) |
| **DAE index** | Index-0 (algebraic at each step) | Index-1 DAE (or semi-explicit ODE with mass matrix) |
| **Error control on charges** | None — charges are internal, solver only controls voltage error | Full — charge states (and constant cap currents) are solver unknowns with their own error tolerances |
| **Higher-order methods** | Limited — must manually implement BE/Trap/Gear2 with Q history | Free — any ODE/DAE solver works (IDA, Rosenbrock, Radau, ...) |
| **Stiffness** | Charge dynamics don't see the stiff solver directly | Adding fast charge states can increase stiffness ratio |
| **V-dep cap accuracy** | Good with small enough dt; no error estimate on charge | Charge error controlled by solver tolerances |
| **Implementation complexity** | Simple — devices return (f, Q), solver combines | Complex — multi-pass detection, charge scaling, index analysis |
| **GPU compatibility** | Better — fixed system size, no dynamic state allocation | Harder — system size varies with V-dep cap count |

### Why This Matters for the Comparison

**VAJAX and Circulax** use the same approach as ngspice: companion model with the integration baked into the residual. This is simpler, GPU-friendly, and well-understood. But it means:
- Error control applies only to voltages, not charges
- You're locked to whatever integration methods you implement manually
- No free lunch from solver library advances

**Cadnip** takes a fundamentally different approach: ALL reactive devices go through the mass matrix, and the ODE/DAE solver handles time integration. Constant caps stamp directly into C; voltage-dependent caps become charge state variables to keep C constant. This gives Cadnip access to IDA's variable-order BDF and Rosenbrock methods. But:
- The V-dependent cap detection is heuristic (multi-pass Q/V ratio comparison)
- Extra charge states increase system size and potentially stiffness
- It's more complex to implement and debug

**For the unified project**, this is a real design decision, not just an implementation detail. The companion model is the pragmatic choice for GPU work and the JAX ecosystem, where no library-solver DAE support exists (see Section 6). The mass matrix formulation is the theoretically cleaner choice that enables better solver integration — but currently only the Julia/DifferentialEquations.jl ecosystem can leverage it. A merged project could support both: companion model for JAX/GPU path (extending BDF order manually), mass matrix for Julia/DifferentialEquations.jl path (accessing IDA, Rosenbrock, etc. for free).

---

## 6. The Diffrax Mass Matrix Barrier: Why JAX Simulators Are Locked to BDF Methods

A critical finding that affects both VAJAX and Circulax: **there is no DAE solver in the JAX ecosystem**. Not in Diffrax, not in JAX itself, not anywhere. This is not a missing feature in one library — it's a gap in the entire ecosystem, and it has direct consequences for any attempt to move beyond basic BDF methods.

### The Problem

Circuit simulation produces systems of the form `C * dx/dt = f(x, t)` where `C` is a (possibly singular) mass matrix. This is a differential-algebraic equation (DAE). To use a solver library's built-in implicit methods (Kvaerno, Radau, Rosenbrock, etc.), the library must understand mass matrices — it can't just solve `dx/dt = f(x, t)`.

Diffrax's implicit solvers (`ImplicitEuler`, `Kvaerno3`, `Kvaerno4`, `Kvaerno5`) solve standard ODEs: `dy/dt = f(t, y)`. They do not accept a mass matrix. You cannot simply invert `C` to get `dx/dt = C^{-1} * f(x, t)` because `C` is singular in MNA systems (nodes without reactive devices have zero rows in `C`).

### The Diffrax DAE Timeline: Four Years of "Not On My Roadmap"

This isn't a new request. The Diffrax issue tracker tells a clear story:

**[Issue #62](https://github.com/patrick-kidger/diffrax/issues/62) — February 2022:** First DAE feature request. Patrick Kidger responded with enthusiasm and sketched a `SemiExplicitConstrainedSolver` approach using existing `AbstractTerm`/`AbstractSolver` interfaces. He proposed a two-step algorithm: (1) advance the differential component with an existing solver, (2) project onto the algebraic constraint via nonlinear solve. But he explicitly caveatted: *"I've not looked too closely at the details of solving a DAE. I don't know how effective/stable/etc. the above approach is numerically."* The sketch set error estimation for the algebraic component to zero and used only linear interpolation — both serious limitations for production use.

**[Issue #261](https://github.com/patrick-kidger/diffrax/issues/261) — May 2023:** A user asked about DAE solving with adaptive stepping. Kidger said DAEs were *"on the roadmap for Diffrax"* with hopes for implementation *"in a few months."*

**[Issue #413](https://github.com/patrick-kidger/diffrax/issues/413) — May 2024:** Another user asked about DAE workarounds. No progress on native support.

**[Issue #457](https://github.com/patrick-kidger/diffrax/issues/457) — July 2024:** A user directly asked about DAE status, referencing Kidger's earlier promise. Kidger responded: *"I'm afraid this isn't on my roadmap at the moment. I'd love to have it in, but it's not something I have time for myself right now."* He added: *"I think what's needed is for someone to feel strongly enough about this to implement it :)"*

**[Issue #710](https://github.com/patrick-kidger/diffrax/issues/710) — December 2025:** A user contributed a workaround — a custom `ImplicitEulerMass` solver subclass that hacks mass matrix support into ImplicitEuler by modifying `_implicit_relation` and passing the mass transform through the `args` tuple. The author explicitly stated: *"I don't have the bandwidth/time/knowledge to make this work for general cases (e.g., DAEs, IRK/DIRK/SDIRK/ESDIRK solvers, etc)"*. Kidger's response: *"Very nice! Thankyou for writing this down – this will be an excellent example to share for solving with mass matrices."* — treating it as documentation, not a roadmap item.

**Summary: DAE support was requested in Feb 2022. It's now March 2026. The status is: one user-contributed ImplicitEuler-only workaround, no native support, not on the maintainer's roadmap, waiting for someone to "feel strongly enough" to implement it.**

### Why the #710 Workaround Doesn't Help

The `ImplicitEulerMass` hack from issue #710 has fundamental limitations:

1. **Only covers ImplicitEuler** — 1st order, the simplest possible implicit method. This is no better than what VAJAX and Circulax already have with companion model BE.

2. **Requires per-solver reimplementation.** The workaround modifies `_implicit_relation` inside a custom solver subclass. Each solver type (SDIRK, ESDIRK, fully implicit RK) has a different `_implicit_relation` structure. There is no generic hook to inject a mass matrix. To support Kvaerno3/4/5, you'd need to fork each solver class individually.

3. **No solver-level error control on algebraic states.** The mass transform is passed via the `args` tuple — a convention, not an API. Diffrax's adaptive stepping has no awareness of the mass matrix structure.

4. **No index reduction or consistent initialization.** Production DAE solvers (IDA, DASSL) handle index-1 DAEs with automatic consistent initialization. This workaround doesn't.

### The Complete JAX Ecosystem: Nothing Exists

This is not just a Diffrax limitation. We searched comprehensively:

| Library / Approach | Mass matrix / DAE support | Status |
|---|---|---|
| **Diffrax** | No. User workaround for ImplicitEuler only ([#710](https://github.com/patrick-kidger/diffrax/issues/710)). DAE requested since 2022 ([#62](https://github.com/patrick-kidger/diffrax/issues/62)), explicitly not on roadmap ([#457](https://github.com/patrick-kidger/diffrax/issues/457)). | No plans for native support |
| **jax.experimental.ode** | No. Dormand-Prince only (explicit, adaptive). No mass matrix, no implicit methods. | Experimental, not suitable for stiff circuits |
| **Sundials/IDA via JAX** | No JAX-compatible wrapper exists. Zero GitHub repos for "jax sundials" or "jax IDA solver". | Does not exist |
| **SciPy BDF via JAX** | SciPy has BDF(1-5) and Radau, but they use NumPy — not JAX-traceable, no `jit`/`vmap`/`grad`. | Incompatible with JAX transforms |
| **torchdiffeq** | PyTorch only. No DAE support even there. | Wrong ecosystem |
| **JAXopt** | Optimization library, not ODE/DAE. Now deprecated ("no longer maintained"). | Dead project |
| **diffeqpy** | Wraps Julia's DifferentialEquations.jl for Python. Has DAE support but via Julia runtime — not JAX-traceable. | No JAX integration |
| **Custom (VAJAX)** | Yes, via companion model (BDF1-2). | Limited to what you implement manually |

**There is literally no JAX-compatible DAE solver anywhere.** Not in any library, not in any GitHub repo, not even as an experimental prototype (beyond the ImplicitEuler hack in #710).

### What Would It Take to Get One?

There are four realistic paths, none easy:

**Option 1: Contribute mass matrix support to Diffrax upstream**

This means modifying `_implicit_relation` in every implicit solver (`ImplicitEuler`, `Kvaerno3`, `Kvaerno4`, `Kvaerno5`) to accept and apply a mass matrix. The changes touch the core solver loop: the implicit system `y1 = y0 + dt*f(t1, y1)` becomes `M*(y1 - y0) = dt*f(t1, y1)` where `M` can be singular.

- **Effort:** 2-4 months for someone who understands both Diffrax internals and DAE numerics.
- **Risk:** The maintainer has explicitly said this isn't his priority. A PR might languish. The semi-explicit approach he sketched in #62 (ODE step + algebraic projection) has unknown numerical stability — he said so himself.
- **Benefit:** If accepted, all of Diffrax's infrastructure (adaptive stepping, interpolation, adjoint methods) would work with mass matrices.
- **For circuit simulation specifically:** Still needs index analysis, consistent initialization, and singular mass matrix handling that Diffrax's architecture wasn't designed for.

**Option 2: Write a standalone JAX DAE solver from scratch**

Implement variable-order BDF (like IDA) or Radau IIA directly in JAX, with mass matrix support built in from the start.

- **Effort:** 4-8 months for a production-quality implementation. IDA's core algorithm (variable-order, variable-step BDF with Newton iteration, error estimation, order selection) is well-documented but subtle.
- **Risk:** Reimplementing 30+ years of Sundials engineering. Bugs in solver code produce wrong answers, not crashes. Would need extensive validation.
- **Benefit:** Full control over the solver, JAX-native (`jit`/`vmap`/`grad` compatible), no upstream dependency.
- **This is essentially what VAJAX is already doing** with its custom BDF1-2 solver — just extending it to higher orders. The question is whether to keep extending VAJAX's solver or start fresh with a cleaner design.

**Option 3: Wrap Sundials IDA for JAX via custom_vjp**

Call IDA (C library) for the forward pass, implement the adjoint sensitivity equations in JAX for the backward pass.

- **Effort:** 2-3 months. The forward pass is straightforward FFI. The adjoint is the hard part.
- **Risk:** Loses `jit` compilation for the forward solve (IDA runs in C, not on GPU). The `vmap` story is complicated — you'd need to batch IDA calls externally. Essentially gives up JAX's key advantages for the solver itself.
- **Benefit:** Production-quality solver immediately. Could use JAX for everything except the time integration.
- **For GPU:** Forward solve would be CPU-only. Could still use GPU for device evaluation if structured carefully, but the solver loop stays on CPU.

**Option 4: Stay on companion model, extend BDF order manually**

Keep the algebraic-at-each-timestep formulation, add BDF3-5 with variable step size. This is what traditional SPICE does.

- **Effort:** 1-3 months. Variable-step BDF coefficients are well-known (Nordsieck vector representation). Error estimation via LTE. Order selection via comparing LTE at adjacent orders.
- **Risk:** Still limited to the BDF family. No path to Rosenbrock, Radau, or other method families. But BDF5 with variable step and order is what IDA does, so the quality ceiling is the same — just the implementation maturity differs.
- **Benefit:** No upstream dependency. Works with companion model and GPU. Both VAJAX and Circulax could do this independently.
- **This is the most pragmatic path for the JAX ecosystem.** It accepts the companion model constraint and works within it.

### What This Means for Each JAX Simulator

**Circulax** already subclasses `diffrax.AbstractSolver` with a custom `VectorizedTransientSolver` that implements BE with companion model. It gains nothing from Diffrax's solver suite. To use Kvaerno3/4/5 from Diffrax for circuit simulation, you would need to contribute mass matrix support upstream (Option 1) — a large effort with uncertain acceptance.

**VAJAX** doesn't use Diffrax at all (custom `lax.while_loop` solver), and switching to Diffrax would NOT unlock higher-order methods — it would just replace one BE-equivalent with another unless Diffrax adds mass matrix support. VAJAX's best path is Option 4: extend its existing custom BDF solver to higher orders.

### Comparison with Julia Ecosystem

| | JAX (Diffrax + ecosystem) | Julia (DifferentialEquations.jl) |
|---|---|---|
| **Mass matrix** | Not supported anywhere | `ODEProblem(f, u0, tspan; mass_matrix=M)` — one keyword |
| **DAE solver** | Does not exist | IDA (variable-order BDF 1-5, production-grade) |
| **Solver count** | 4 implicit (IE, Kvaerno 3/4/5), none usable for circuits | 100+ solvers, many support mass matrix |
| **Adding a new solver** | Fork solver class, reimplement `_implicit_relation` with mass transform | Change one argument: `solve(prob, Rodas5P())` → `solve(prob, RadauIIA5())` |
| **Error control on charges** | Not possible (solver unaware of mass matrix) | Full — solver controls all state variables |
| **Time to production DAE** | 2-8 months depending on approach | Already done |

This is a **structural advantage** of the Julia/SciML ecosystem for circuit simulation, not a convenience difference. It doesn't mean JAX simulators can't work — VAJAX proves they can, and companion model BDF is how ngspice has operated for decades. But it means the JAX path to solver quality depends entirely on custom implementation effort, while the Julia path gets it from the ecosystem.

### Implications for Higher-Order Methods

The companion model approach (used by both JAX simulators) naturally supports **only multistep methods** (BDF family):

| Method family | Companion model feasibility | Mass matrix feasibility |
|---|---|---|
| **BDF 1-2** (BE, Trap) | Works — current state of VAJAX | Works — current state of Cadnip |
| **BDF 3-5** | Extend coefficients + history. Most pragmatic path for JAX. | Already available via IDA |
| **SDIRK/ESDIRK** (Kvaerno) | Each stage resembles BE with modified coefficients. Possible but complex | Works directly (if solver supports mass matrix) |
| **Fully implicit RK** (Radau) | All stages coupled — system size multiplied by stage count. Impractical with companion model | Works directly |
| **Rosenbrock** | Incompatible — needs explicit `M*y'` form | Works directly |
| **Variable-order adaptive** | Must implement Nordsieck vector or equivalent. Significant effort | Handled by solver library (IDA) |

The practical ceiling for companion model in JAX is **BDF5 with variable timestep** — equivalent to reimplementing IDA in JAX. This is doable (Option 4 above) but represents months of careful numerical work, and the result would be less tested than Sundials IDA (which has decades of production use).

For methods beyond the BDF family (Rosenbrock, Radau, SDIRK), the JAX ecosystem currently offers no path. Diffrax has the solver implementations but lacks the mass matrix support to use them for circuits. This is a real architectural barrier that has persisted for four years with no resolution in sight.

---

## 7. Deeper Analysis: Solver Quality Matters

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
- Diffrax has Kvaerno3/4/5, Tsit5, Dopri5 — but these solve `dy/dt = f(t,y)`, not `M*dy/dt = f(t,y)`
- Diffrax has no mass matrix or DAE support (see Section 6) — these solvers are unusable for circuit simulation
- Maintained by Patrick Kidger (well-respected in scientific ML), but DAE support has been requested since 2022 with no progress

**The gap:** VAJAX's custom solver works for its demonstrated use cases but is limited to 2nd order BDF. Diffrax's higher-order solvers exist but can't be used for circuits (Section 6). If you want the solver quality Cadnip has, the realistic options are:
1. Extend VAJAX's custom BDF solver to orders 3-5 (most pragmatic — months of work, stays on companion model)
2. Contribute mass matrix support to Diffrax upstream (2-4 months, uncertain acceptance — see Section 6)
3. Write a standalone JAX DAE solver from scratch (4-8 months — reimplementing Sundials)

Option 1 is the most pragmatic: keep VAJAX's companion model approach and extend BDF order, which is exactly what traditional SPICE simulators do.

---

## 8. The OpenVAF Question

OpenVAF integration is the critical differentiator. Without it:
- You're limited to hand-written device models (Circulax's square-law MOSFET, Ebers-Moll BJT)
- No access to foundry PDK models
- Not usable for real IC design work

Cadnip's custom VA codegen is a heroic effort but hits structural limits:
- Less static analysis → larger generated code → LLVM problems on complex models
- Ongoing maintenance burden of a custom VA compiler
- Missing optimizations that OpenVAF has (node collapse, parameter caching, dead code elimination)

**Integrating OpenVAF into Cadnip** (replacing custom codegen) would solve the PSP103 problem and give access to all validated models. The cost is needing Julia↔Rust FFI (via CBinding.jl or similar). This is real work but bounded.

**Integrating OpenVAF into Circulax** is simpler than it first appears — `openvaf_jax` is a self-contained module that could be imported as a library. The work is writing an adapter between OpenVAF's device function signature and Circulax's `@component` interface (see Section 4). The deeper question is whether Circulax should also adopt the companion model pattern to match OpenVAF's resist/react output format (see Section 5).

---

## 9. The GPU Question (Honest Assessment)

Your pushback is fair. Let me be precise:

- **VAJAX**: GPU works and is tested at scale. 12.8x speedup on mul64 is real. But only pays off at 500+ nodes (small circuits are 5-10x slower than C++ due to JAX overhead).
- **Circulax**: Has the same JAX options as VAJAX in theory. In practice, untested. The sparse infrastructure (KLU) is CPU-only; the dense solver would work on GPU but won't scale. Would need significant work to match VAJAX's GPU story.
- **Cadnip**: Julia's GPU story is more nuanced than "not currently." DifferentialEquations.jl supports two distinct GPU modes ([SciML GPU docs](https://docs.sciml.ai/Overview/stable/showcase/massively_parallel_gpu/)):
  - **CUDA arrays for large systems:** Replace `Array` with `CuArray` and the solver runs sparse linear algebra on GPU. This is the analog of what VAJAX does — accelerating a single large circuit. Unlike JAX, this does NOT require allocation-free CPU code — Julia handles GPU memory management transparently, so existing allocating code works with CuArray. The main requirements are GPU-compatible sparse solvers (CUSPARSE, CUSOLVER) and ensuring the linear algebra dispatch paths support CuArray types. This is actually more tractable than it first appears — the hard part is performance, not correctness. However, the performance profile differs from JAX: Julia still orchestrates from CPU with GPU kernel launches per operation, while JAX's XLA compiler fuses the entire computation graph into optimized GPU kernels. For large circuits where each sparse solve is expensive, the per-launch overhead is small. For smaller circuits, JAX's fused execution has a structural advantage.
  - **Ensemble problems for many small systems:** Run thousands of independent small simulations (e.g., parameter sweeps, Monte Carlo) in parallel on GPU via `EnsembleGPUArray` or `EnsembleGPUKernel`. This is fundamentally different from JAX's approach — JAX uses `vmap` over device instances *within* a single simulation, while Julia parallelizes *across* independent simulations. For parameter sweeps and corner analysis, this could be very effective without modifying the solver at all.

  The ensemble approach is the more tractable near-term path: parameter sweeps already work on CPU, and `EnsembleGPUKernel` can parallelize them with minimal code changes. The single-large-circuit GPU path requires more investment.

**Bottom line:** GPU is a genuine advantage of VAJAX today for large single-circuit simulation. But Julia has a distinct advantage for massively parallel sweeps via ensemble GPU support — a use case JAX handles less naturally (you'd need to restructure the entire simulation to vmap over parameter sets). The right framing is: VAJAX wins at GPU-accelerated large circuits, Julia/Cadnip has a natural path to GPU-accelerated parameter sweeps.

---

## 10. Revised Recommendation

Rather than picking a winner, here's what each project should contribute to a unified effort, based on **actual demonstrated strengths** not potential:

### What Each Brings (Proven, Not Aspirational)

| Contributor | Proven Contribution | Why It's Hard to Replicate |
|---|---|---|
| **VAJAX (Rob)** | OpenVAF→JAX pipeline, GPU MNA solver, large-circuit validation, noise/AC/HB/xfer analyses | OpenVAF integration is ~4,500 LOC of careful MIR→JAX translation. GPU solver required solving JAX-specific challenges (lax.while_loop NR, vmap device eval, sparse auto-switching). Years of validation work against VACASK + ngspice. |
| **Cadnip (Pepijn)** | SPICE/Spectre parser (~20K LOC across two packages), DifferentialEquations.jl integration patterns, convergence aids, zero-allocation patterns | Parser covers multiple SPICE dialects + Spectre. The IDA/Rosenbrock integration with circuit-specific initialization (CedarDCOp, CedarTranOp, CedarUICOp) represents deep domain expertise. |
| **Circulax (Chris)** | Photonic simulation, clean component API, Optimistix integration, SAX ecosystem bridge | Photonic circuit simulation (S-param→Y-matrix→complex MNA) is genuinely unique. The `@component` decorator pattern is the best API design of the three. Note: the Diffrax dependency provides less value than expected for circuit simulation due to missing mass matrix support (Section 6). |

### Three Viable Paths (All Require Real Work)

**Path A: VAJAX + higher-order BDF + photonics + better parsing**
- Extend VAJAX's custom BDF solver to orders 3-5 with variable step/order (1-3 months — Nordsieck vector, LTE-based order selection). Diffrax integration would NOT help here due to missing mass matrix support (Section 6).
- Port Circulax's photonic support including complex MNA (3-5 weeks)
- Improve SPICE parsing (extend existing converter or integrate Cadnip's parser as frontend tool)
- Keep OpenVAF + GPU as-is
- **Estimated effort:** 3-5 months for solver extension + photonics. Parsing is ongoing.

**Path B: Circulax + OpenVAF + convergence aids + scaling**
- Import openvaf_jax as library, write adapter for device interface (2-4 weeks — 9-tuple vs 2-tuple mismatch, Jacobian routing decision)
- Add GMIN/source stepping homotopy (1-2 weeks — algorithm is portable, but need to parameterize component eval for gmin/srcFact)
- Add AC + noise analysis (3-5 weeks)
- Higher-order transient: either extend companion model to BDF3-5 (1-3 months) or contribute mass matrix support to Diffrax upstream (2-4 months, uncertain acceptance — see Section 6). Diffrax's existing Kvaerno/Radau solvers are NOT usable for circuits without mass matrix support.
- Validate at scale (ongoing — requires running benchmark suite)
- **Estimated effort:** 4-6 months. openvaf_jax is self-contained but integration touches core assembly loop. Solver improvement is a separate, significant effort.

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

3. **Solver quality priority:** If you want the best transient solvers *now*, Cadnip wins (IDA is hard to beat). The JAX paths require building custom BDF3-5 solvers — Diffrax can't help here due to missing mass matrix support (Section 6). This is months of custom solver work for either JAX simulator.

4. **OpenVAF:** This is non-negotiable for production use. VAJAX has it. The others would need to build or port it.

5. **Photonics:** If mixed electronic-photonic simulation is important, Circulax's work is the starting point regardless of which path you choose.

6. **Developer preferences:** Each developer has made deep architectural choices. The most productive path is the one that lets each person contribute what they're best at without requiring them to abandon their approach.

---

## 11. A Pragmatic Middle Path

Rather than "pick one core and port everything," consider:

**Shared infrastructure, separate frontends:**

1. **OpenVAF bindings** as a shared library — VAJAX's openvaf_jax already works. Could be extracted and made reusable.
2. **SPICE parser** as a standalone tool — Cadnip's parser outputs an intermediate representation that any simulator can consume.
3. **Benchmark suite** — shared test circuits with reference results from VACASK/ngspice for cross-validation.
4. **Photonic components** — Circulax's component library, extracted as a reusable package.

Each simulator keeps its own solver/engine but shares the hard-to-build infrastructure pieces. This is less ambitious than a full merge but more realistic for three independent developers.

---

## 12. Summary

The first version of this study compared three projects. Adding VACASK as the C++ reference clarifies the picture — it shows where the performance ceiling is and what optimization techniques actually matter.

| What matters | Best | Why |
|---|---|---|
| Raw simulation performance | **VACASK** | Fastest on all benchmarks. 20% faster than ngspice, 2.6x faster than Xyce on c6288. Continuation bypass + direct pointer binding + minimal overhead. |
| Memory efficiency | **VACASK** | 138.5 MB for c6288 vs Xyce 775.7 MB. Minimal dependencies, careful buffer reuse. |
| Transient solver quality | **Cadnip** ≈ **VACASK** | Cadnip: IDA + Rosenbrock via DifferentialEquations.jl. VACASK: custom BE/Trap/Gear with general IntegratorCoeffs framework. VACASK has more optimizations (continuation bypass, residual check). Cadnip has more solver *methods* available. |
| Verilog-A / PDK models | **VACASK** | Most comprehensive library: native VA + VADistiller converted SPICE models + multiple variants. Drives OpenVAF-reloaded development. |
| GPU acceleration (large circuits) | **VAJAX** | Actually working, tested to 133K nodes. VACASK has no GPU. |
| GPU acceleration (parallel sweeps) | **Cadnip** (potential) | EnsembleGPUKernel parallelizes independent sims naturally; JAX requires restructuring |
| Analysis breadth | **VACASK** | DC, AC, tran, noise, HB (multitone), DC/AC xfer, arbitrary-depth parametric sweeps. Most complete of all four. |
| Differentiable simulation | **VAJAX** ≈ Circulax | The key capability VACASK completely lacks. JAX AD enables gradient-based optimization and ML integration. |
| Architecture & code quality | **Circulax** (Python), **VACASK** (C++) | Circulax: cleanest Python design. VACASK: 45K LOC of well-organized C++20 with clear separation of concerns. |
| Photonic simulation | **Circulax** | Unique capability, no equivalent in others |
| Netlist parsing & ecosystem | **VACASK** | Spectre-like native parser + conditional blocks + Xschem integration + IHP PDK support. Cadnip has good SPICE/Spectre parsing too. |
| Convergence robustness | **VACASK** ≈ Cadnip | VACASK: GMIN/source stepping + residual convergence check + continuation bypass. Cadnip: GMIN/source stepping + pseudo-transient init. |
| Scalability (tested) | **VAJAX** (GPU) / **VACASK** (CPU) | VAJAX: 133K nodes on GPU. VACASK: 25K nodes on CPU, fastest per-iteration. |
| Solver ecosystem leverage | **Cadnip** ≫ others | DiffEq.jl (native mass matrix, 100+ solvers) ≫ all custom solvers. One-line access to IDA, Rosenbrock, Radau, etc. |
| Benchmark methodology | **VACASK** | The gold standard: 6 runs, cache warming, matched timepoints, controlled for solver differences, std dev <2% |

### The Four-Way Picture

The four projects occupy distinct positions:

- **VACASK** is the **performance and correctness reference**. It shows what's achievable in single-threaded C++ with careful engineering. Any new simulator should benchmark against it and learn from its optimization techniques (continuation bypass, residual convergence, direct pointer binding). Its limitation is that it's a traditional simulator — no AD, no GPU, no ML integration.

- **VAJAX** is the **differentiable GPU simulator**. It's the only project with working GPU acceleration at scale AND differentiable simulation. Its limitation is solver maturity (2nd order BDF) and the JAX DAE gap (Section 6).

- **Cadnip** is the **solver ecosystem play**. It's the only project with native access to production DAE solvers (IDA, Rosenbrock) without reimplementing them. Its limitation is VA model support (needs OpenVAF) and GPU (needs work).

- **Circulax** is the **clean architecture + photonics** project. It's the only one with electronic-photonic simulation. Its limitation is that it's the least mature for traditional analog simulation (no convergence aids, untested at scale, BE only).

The most productive collaboration is shared infrastructure: OpenVAF/VADistiller device models, VACASK-calibrated benchmarks, and a clear division of effort where each project contributes its proven strengths rather than duplicating the others' work.
