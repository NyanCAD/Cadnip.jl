# Comparative Study: Cadnip.jl vs VAJAX vs Circulax

**Date:** 2026-03-07
**Purpose:** Evaluate three open-source one-person analog circuit simulator projects to identify the best path forward for joining forces.

---

## 1. Executive Summary

| Dimension | **Cadnip.jl** | **VAJAX** | **Circulax** |
|-----------|---------------|-----------|--------------|
| Language | Julia | Python/JAX | Python/JAX |
| LOC (core) | ~18K (src), ~52K total | ~65K | ~4.8K |
| Commits | 50 | 1,018 | 168 |
| Contributors | 1 (Pepijn de Vos) | 1 (Rob Taylor) | 1 (Chris Daunt) |
| Last active | Feb 2026 | Mar 2026 | Mar 2026 |
| License | MIT (CedarSim fork) | Apache-2.0 | Apache-2.0 |
| Maturity | Early-stage | Most mature | Early-stage |

All three projects are actively developed single-person efforts building open-source analog circuit simulators from scratch. They share the same MNA (Modified Nodal Analysis) core algorithm but differ significantly in language, architecture, device model strategy, and GPU acceleration approach.

---

## 2. Project-by-Project Analysis

### 2.1 Cadnip.jl (NyanCAD)

**What it is:** A Julia-based MNA circuit simulator, forked from JuliaComputing's CedarSim. Replaces the DAECompiler backend with a hand-written MNA engine. Targets simplicity and maintainability.

**Architecture:**
- **Simulation engine:** Custom MNA with G (conductance), C (capacitance), and b (source) matrix stamping. Newton-Raphson for DC, DifferentialEquations.jl ecosystem for transient (Sundials IDA, Rosenbrock methods).
- **Two-phase stamping:** `MNAContext` for structure discovery (sparse pattern), `DirectStampContext` for zero-allocation restamping during Newton iterations. This is a clever optimization - the circuit topology is discovered once, then values are updated in-place during solve.
- **Builder function pattern:** Circuits are compiled to Julia functions with signature `circuit(params, spec, t; x, ctx)`. This enables Julia's JIT compiler to optimize the inner loop.
- **Parser ecosystem:** Full SPICE multi-dialect parser (`SpectreNetlistParser.jl`) and Verilog-A parser (`VerilogAParser.jl`) - both written from scratch in Julia, totaling ~20K lines.

**Verilog-A approach:** Parses VA source code and generates Julia code that performs MNA stamping directly. The VA model becomes a native Julia function with `stamp_G!`, `stamp_C!` calls. This means VA models get full Julia JIT optimization, ForwardDiff AD, and type specialization. The tradeoff is that the parser must handle the full VA language (a significant effort).

**Analysis types:**
- DC operating point (Newton-Raphson with homotopy: GMIN stepping, source stepping)
- Transient (via DifferentialEquations.jl - many ODE/DAE solvers available)
- AC small-signal (linearized frequency sweep)
- Parameter sweeps (`CircuitSweep`)
- Full differentiability via ForwardDiff (sensitivities, optimization)

**Device models:**
- Basic passives (R, L, C) via direct MNA stamps
- Voltage/current sources (DC, PWL, SIN, PULSE)
- Verilog-A compiled models: PSP103 (MOSFET), BSIM4, BJT models
- PDK integration via `VADistillerModels` and `PSPModels` packages

**Key strengths:**
- Julia's type system enables zero-allocation inner loops after compilation
- Full ForwardDiff AD through the entire simulation (sensitivities for free)
- Rich ODE/DAE solver ecosystem via DifferentialEquations.jl (adaptive timestepping, stiff solvers, Rosenbrock methods)
- Native SPICE/Spectre netlist parsing
- Verilog-A models compile to native Julia (no FFI overhead)

**Key weaknesses:**
- Julia startup latency (time-to-first-simulation can be minutes due to compilation)
- Smaller ecosystem than Python for ML/optimization integration
- No GPU acceleration
- VA parser doesn't cover 100% of the Verilog-A spec

**Code quality:** Well-structured with clear module separation. Extensive documentation in `doc/` (40+ design documents). Good test coverage (38 test files). The codebase shows careful thought about performance (zero-allocation patterns, precompiled circuits).

---

### 2.2 VAJAX (ChipFlow)

**What it is:** A Python/JAX GPU-accelerated analog circuit simulator. The most mature of the three projects (1,018 commits). Focuses on running existing Verilog-A models on GPUs for large-circuit speedups.

**Architecture:**
- **Simulation engine:** Full MNA with JAX arrays. Newton-Raphson in `jax.lax.while_loop` for on-device iteration. Custom COO sparse matrix assembly.
- **MNA builder:** `mna_builder.py` constructs stamp index mappings. Device contributions are computed by OpenVAF and scattered into the MNA matrix.
- **Solver pipeline:** DC op (with homotopy/GMIN stepping) -> Transient (adaptive timestep with LTE control) or AC sweep.
- **GPU strategy:** Devices are evaluated via `jax.vmap` for GPU parallelism. Large circuits (>500 nodes) automatically use GPU. Small circuits stay on CPU to avoid kernel launch overhead.

**Verilog-A approach:** Uses [OpenVAF](https://openvaf.semimod.de/) to compile VA models to a MIR (Mid-level IR), then `openvaf_jax` translates MIR to JAX operations. This is a fundamentally different approach from Cadnip - instead of parsing VA and generating simulator-native code, VAJAX uses OpenVAF's established compiler and adds a JAX backend. This gives access to production VA models (PSP103, BSIM-CMG, etc.) with higher fidelity.

**Analysis types:**
- DC operating point (Newton-Raphson with homotopy chain)
- Transient (adaptive BDF2/Trap/BE with LTE control, predictor-corrector)
- AC small-signal frequency sweep
- Noise analysis
- Transfer function analysis (DC/AC)
- Corner sweeps
- Harmonic balance
- GPU-accelerated large-circuit simulation

**Device models:**
- All devices via Verilog-A/OpenVAF (resistor, capacitor, diode, PSP103, etc.)
- Voltage/current sources (DC, pulse, sine, PWL)
- Validated against VACASK (C++ reference) and ngspice

**Performance results (impressive):**
- C6288 16-bit multiplier (~5000 nodes): **2.9x faster than C++ (VACASK)** on GPU
- Mul64 64-bit multiplier (~133K nodes, ~266K MOSFETs): 12.8x GPU speedup over CPU; VACASK times out
- Small circuits: ~5-10x slower than C++ due to JAX overhead

**Key strengths:**
- GPU acceleration that actually works for large circuits
- OpenVAF integration gives access to production Verilog-A models
- Validated against two reference simulators (VACASK, ngspice)
- Comprehensive analysis suite (DC, AC, tran, noise, HB, transfer function, corners)
- CLI interface (`vajax circuit.sim`)
- SPICE netlist converter (ngspice -> VAJAX format)
- Published on PyPI (`pip install vajax`)
- Extensive benchmarking infrastructure
- Most features of any of the three simulators

**Key weaknesses:**
- JAX's functional paradigm adds per-step overhead for small circuits
- `jnp.where` evaluates both branches (no short-circuit), wasteful for conditionals
- COO matrix assembly adds indirection vs direct stamping
- Complex build system (OpenVAF Rust + Python bindings + JAX custom ops)
- Less natural AD integration compared to Cadnip's ForwardDiff

**Code quality:** Very well-organized with clear module separation. Comprehensive docstrings. Good test suite (38 test files). Detailed performance analysis documentation. The codebase is the largest and most feature-complete.

---

### 2.3 Circulax

**What it is:** A differentiable, functional circuit simulator built on JAX, Diffrax, and Optimistix. The smallest and youngest of the three. Unique in supporting both electronic and photonic circuit simulation.

**Architecture:**
- **Three-layer design:** Physics (components) / Topology (compiler) / Analysis (solvers) with strict separation.
- **Functional components:** Devices are plain Python functions decorated with `@component` or `@source`. They return `(f_dict, q_dict)` - flow equations and storage terms. This is the cleanest API of the three.
- **Compiler:** `compile_netlist()` takes a SAX-format netlist dict, groups components by type, creates `ComponentGroup` objects with batched parameters and pre-computed indices.
- **Solvers:** Strategy pattern for linear algebra (Dense/KLU/BiCGStab). Newton-Raphson via Optimistix. Transient via Diffrax. Jacobians computed automatically via `jax.jacfwd`.

**Verilog-A approach:** None. Circulax has no Verilog-A support. Device models are defined as Python functions. This is both its greatest strength (simplicity, full AD) and greatest weakness (no access to production PDK models).

**Analysis types:**
- DC operating point (Newton-Raphson via Optimistix)
- Transient (Diffrax implicit solvers, adaptive timestepping)
- Harmonic balance (FFT-based, Newton via Optimistix)
- Photonic simulation (S-parameter based, complex-valued nodal analysis)

**Device models:**
- All hand-written in Python: Resistor, Capacitor, Inductor
- Diode (Shockley), Zener diode
- NMOS/PMOS (square-law with channel-length modulation)
- NMOSDynamic (Meyer capacitance model)
- BJT_NPN, BJT_NPN_Dynamic (Ebers-Moll with junction charges)
- VCVS, VCCS, CCVS, CCCS (controlled sources)
- IdealOpAmp, VoltageControlledSwitch
- Photonic: OpticalWaveguide, DirectionalCoupler, MachZehnder, RingResonator, PhotoDetector, LaserSource

**Key strengths:**
- Cleanest, most elegant API of the three
- Photonic circuit support is unique
- Fully Jacobian-free: `jax.jacfwd` computes all derivatives automatically
- Built on well-maintained scientific Python libraries (Diffrax, Optimistix, Equinox)
- Mixed electronic-photonic simulation
- True differentiability through the solver (reverse-mode AD possible via Diffrax)
- SAX netlist format integrates with photonic design tools
- Harmonic balance implementation

**Key weaknesses:**
- No Verilog-A support (no path to production PDK models)
- Only basic device models (square-law MOSFET, Ebers-Moll BJT)
- Smallest codebase (~4.8K lines) - many features still missing
- No SPICE netlist parser
- Dense Jacobian computation via `jax.jacfwd` scales as O(n^2) - won't work for large circuits
- No GPU benchmarks
- No validation against reference simulators

**Code quality:** Excellent. Clean, well-documented code. Good use of JAX/Equinox patterns. The component decorator system is elegant. Type annotations throughout. Ruff linting. Small but well-structured test suite (8 test files).

---

## 3. Technical Comparison

### 3.1 Core Algorithms

| Algorithm | Cadnip | VAJAX | Circulax |
|-----------|--------|-------|----------|
| MNA formulation | G*x + C*dx/dt = b | J_res + J_react stamping | F(y) + dQ/dt = 0 |
| DC solver | Newton-Raphson + homotopy | Newton-Raphson + homotopy | Newton-Raphson (Optimistix) |
| Transient | DifferentialEquations.jl | Custom adaptive BDF2/Trap/BE | Diffrax implicit solvers |
| Jacobian | ForwardDiff AD + analytic stamps | OpenVAF analytic + JAX AD | jax.jacfwd (fully automatic) |
| Sparse solver | SparseArrays (Julia) | UMFPACK, dense JAX | KLU (klujax), dense, BiCGStab |
| AC analysis | Yes | Yes | No (but HB available) |
| Noise | No | Yes | No |
| HB | No | Yes | Yes |

### 3.2 Verilog-A Integration

| Aspect | Cadnip | VAJAX | Circulax |
|--------|--------|-------|----------|
| VA compiler | Custom Julia parser | OpenVAF (Rust) | None |
| Output | Julia stamp functions | JAX operations via MIR | N/A |
| PDK models | PSP103, BSIM4 (partial) | PSP103, full VACASK suite | None |
| Model validation | Basic tests | 3-way comparison (VACASK+ngspice) | N/A |

### 3.3 Performance & Scalability

| Metric | Cadnip | VAJAX | Circulax |
|--------|--------|-------|----------|
| GPU support | No | Yes (CUDA) | Theoretically (JAX) |
| Small circuits | Fast (native Julia) | ~5-10x slower than C++ | Unknown |
| Large circuits | Untested at scale | 2.9x faster than C++ on GPU | Won't scale (dense Jacobian) |
| Startup time | Slow (Julia compilation) | Fast (Python) | Fast (Python) |
| AD capability | ForwardDiff (forward-mode) | JAX (forward + reverse) | JAX (forward + reverse) |

### 3.4 Ecosystem & Usability

| Aspect | Cadnip | VAJAX | Circulax |
|--------|--------|-------|----------|
| Package manager | Julia Pkg | pip/uv | pip/pixi |
| CLI | No | Yes (`vajax circuit.sim`) | No |
| Netlist format | SPICE/Spectre | VACASK .sim files + ngspice converter | SAX dict (Python) |
| Documentation | Extensive internal docs | Growing docs site | MkDocs site |
| PyPI/registry | No | Yes | Yes |
| ML integration | Julia ML ecosystem | JAX/Flax/Optax native | JAX/Flax/Optax native |

---

## 4. Complementary Strengths

The three projects have remarkably complementary capabilities:

```
                  Cadnip          VAJAX           Circulax
                  ------          -----           --------
VA models:        Julia codegen   OpenVAF/MIR     (none)
GPU:              (none)          CUDA/JAX        (theoretical)
Photonics:        (none)          (none)          S-param/complex
Differentiability: ForwardDiff    JAX AD          JAX AD (best)
Netlist parsing:  SPICE/Spectre   VACASK+ngspice  SAX dict
Solver ecosystem: DiffEq.jl      Custom          Diffrax/Optimistix
Code elegance:    Good            Good            Best
Scale tested:     ~100 nodes      ~133K nodes     ~10 nodes
```

---

## 5. Path Forward: Recommendations

### Option A: Unite Around VAJAX (Recommended)

**Rationale:** VAJAX is the most mature, has the best performance story, validated results, and the critical OpenVAF integration that gives access to production models. The two key missing pieces (photonic simulation and cleaner component API) can be added.

**Concrete steps:**

1. **Circulax -> VAJAX:** Port the photonic component system and harmonic balance solver. The `@component` decorator pattern could inspire a cleaner Python-native device API in VAJAX alongside the OpenVAF path. Chris Daunt's S-parameter/Y-matrix transforms and complex-valued MNA assembly are directly reusable.

2. **Cadnip -> VAJAX:** Port the SPICE/Spectre netlist parser or create a converter. Cadnip's ngspice-style DC initialization heuristics (GMIN stepping, source stepping) are already in VAJAX. The ForwardDiff-through-the-solver story is less critical since JAX already provides AD.

3. **Keep Cadnip as Julia frontend:** For users who prefer the Julia ecosystem, Cadnip could become a Julia frontend that generates VAJAX-compatible netlists or calls VAJAX's Python solver via PythonCall.jl. The Julia VA parser could also serve as an alternative VA compilation path.

**Why this works:** VAJAX already has the hardest pieces done (GPU solver, OpenVAF integration, validation). Adding photonic support and a nicer component API is incremental work.

### Option B: Shared Core Library (More Ambitious)

Create a language-agnostic core simulation kernel (in Rust or C++) that all three projects use:

```
                    ┌─────────────────┐
                    │   Shared Core   │
                    │  (Rust/C++)     │
                    │  - MNA engine   │
                    │  - NR solver    │
                    │  - Sparse math  │
                    │  - OpenVAF      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         Julia FFI      Python FFI     Photonic ext
         (Cadnip)     (VAJAX/Circulax)  (Circulax)
```

**Pros:** Each project keeps its language/ecosystem. Shared effort on the hard parts.
**Cons:** FFI complexity. JAX's value proposition (GPU, AD) doesn't work through FFI. This is essentially rewriting VACASK.

### Option C: Convergence on a Common Specification

Rather than merging codebases, define a shared:
- **Netlist interchange format** (superset of SPICE/SAX/VACASK)
- **Device model interface** (input/output contract for VA models)
- **Result format** (for cross-validation)
- **Benchmark suite** (shared circuits for regression testing)

Each project implements the spec in its own language. This enables cross-validation and shared benchmarks without forcing a single implementation.

---

## 6. Specific Contributions Each Developer Could Make

### Pepijn (Cadnip)
- **SPICE/Spectre parser:** Already the most complete open-source parser. Could be extracted as a standalone tool that outputs a common netlist IR.
- **Julia VA codegen:** The approach of parsing VA and generating native simulator code is valuable for non-JAX backends.
- **DiffEq.jl integration patterns:** The ODE/DAE solver integration patterns are well-tested.

### Rob (VAJAX)
- **OpenVAF + JAX pipeline:** The critical path for production PDK model support on GPU.
- **Large-circuit performance:** The only project that has actually scaled to 100K+ nodes.
- **Validation infrastructure:** Three-way comparison tooling.

### Chris (Circulax)
- **Component API design:** The `@component` decorator pattern is the cleanest API. Could become the standard for hand-written models.
- **Photonic simulation:** Unique capability. S-parameter transforms, complex MNA, ring resonator models.
- **Diffrax/Optimistix integration:** Clean use of state-of-the-art numerical libraries.
- **Harmonic Balance:** Clean FFT-based implementation.

---

## 7. Conclusion

The three projects together cover nearly everything needed for a world-class open-source analog simulator:

| Need | Who has it |
|------|-----------|
| GPU-accelerated MNA solver | VAJAX |
| Production VA model support | VAJAX (OpenVAF) |
| Large-circuit validation | VAJAX |
| SPICE/Spectre netlist parsing | Cadnip |
| Julia ecosystem integration | Cadnip |
| Differentiable simulation | All three (Circulax best) |
| Photonic/mixed-domain | Circulax |
| Elegant component API | Circulax |
| Harmonic balance | VAJAX + Circulax |
| Noise analysis | VAJAX |

**The recommended path is Option A: unite around VAJAX as the primary engine**, with Circulax's photonic capabilities and component API patterns merged in, and Cadnip's parser potentially providing SPICE/Spectre frontend support. This avoids the language-barrier problem (VAJAX and Circulax are both Python/JAX) while leveraging the most mature and performant codebase.

The single biggest risk to any collaboration is that one-person projects are deeply personal - each developer has made fundamental architectural choices that reflect their priorities. The key to success is respecting these choices while finding concrete, bounded contributions each person can make to the shared effort.
