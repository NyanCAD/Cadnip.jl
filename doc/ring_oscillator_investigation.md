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
