# PSP103 Bridge Tests Documentation

## Overview

This document describes the PSP103 bridge test suite (`psp103_bridge.jl`) that incrementally bridges the gap between simple PSP103 integration tests and the full 9-stage ring oscillator benchmark.

## Test Progression

The bridge tests follow this progression:

### Level 1: DC Tests with Full Model Card (280+ parameters)

1. **Single NMOS (DC)**: Single transistor with complete model card
   - Tests basic model parsing and DC operating point
   - Verifies drain current is in expected range

2. **Single PMOS (DC)**: Single P-channel transistor
   - Complements NMOS test
   - Validates both transistor types work with full model

3. **Single Inverter (DC)**: CMOS inverter (NMOS + PMOS)
   - First multi-device circuit
   - Tests subcircuit hierarchy with full models
   - Verifies DC transfer characteristics

### Level 2: Ring Oscillator Transient Tests

4. **3-Stage Ring Oscillator**: Minimal odd-stage ring
   - Circuit: 3 inverters in feedback loop
   - Smallest oscillator configuration
   - ~125 unknowns (vs 371 for 9-stage)
   - Tests transient analysis with CedarTranOp initialization

5. **5-Stage Ring Oscillator**: Mid-size ring
   - Circuit: 5 inverters
   - Intermediate complexity
   - ~205 unknowns
   - Slower oscillation frequency than 3-stage

6. **7-Stage Ring Oscillator**: Near-full complexity
   - Circuit: 7 inverters
   - ~290 unknowns
   - Approaching full benchmark complexity

7. **9-Stage Ring Oscillator**: Matches full benchmark
   - Circuit: 9 inverters
   - 371 unknowns (matches benchmark)
   - Full benchmark configuration
   - Tests complete simulation capability

## Key Configuration

All ring oscillator tests use the **winning configuration** from benchmark development:

```julia
solver=FBDF(autodiff=false)
initializealg=CedarTranOp()  # Homotopy: GMIN → source stepping
dtmax=0.01e-9                # 10ps max timestep
maxiters=10_000_000
dense=false
```

## Model Card

All tests use the **full 280+ parameter model card** from the benchmark:

- Both NMOS (psp103n) and PMOS (psp103p) models
- Identical to `benchmarks/vacask/ring/cedarsim/models.inc`
- Includes all physical parameters: geometry, mobility, saturation, noise, leakage, junction models
- Realistic PDK-level model complexity

## Why This Progression?

### Skip Simple Transient Tests

We deliberately **skip** simple single-transistor transient tests because:
- Single transistors with PWL sources are numerically unstable
- They don't represent realistic circuit operation
- Ring oscillators provide much better coverage for transient behavior

### DC Before Transient

DC tests establish baseline functionality before attempting transient analysis:
- Faster to run (no dynamic simulation)
- Easier to debug
- Validates model stamping and Newton solver

### Progressive Ring Sizes

Ring oscillators scale naturally:
- Odd stages required for oscillation (3, 5, 7, 9)
- System size grows predictably
- Identifies scaling bottlenecks before full benchmark
- Each size validates previous complexity level

## Compilation Time

**Important:** First run has massive JIT compilation overhead:

| Component | Compilation Time |
|-----------|------------------|
| PSP103VA stamp! | ~150s |
| DirectStampContext variants | ~110s |
| Ring builder | ~109s |
| **Total** | **~400s** |

Subsequent runs are much faster (~10-20s solve time) once functions are compiled.

## Running the Tests

### Full Suite

```bash
julia --project=test test/mna/psp103_bridge.jl
```

### Individual Test Sets

Use Julia REPL to run specific tests:

```julia
using Test
includet("test/mna/psp103_bridge.jl")

# Run only DC tests (fast)
@testset "DC only" begin
    # Tests 1-3 complete in <1 minute
end

# Run only 3-stage ring (intermediate)
@testset "3-stage" begin
    # Test 4 takes ~1-2 minutes after compilation
end
```

## Expected Results

### DC Tests (Tests 1-3)
- All should pass quickly
- Currents in µA range
- Voltages match applied biases

### 3-Stage Ring (Test 4)
- ~60 timepoints over 1µs
- ~200 Newton iterations (≈3.3 per step)
- Success status

### 5-Stage Ring (Test 5)
- ~100 timepoints over 1µs
- More stable than 3-stage (longer propagation delay)
- Success status

### 7-Stage Ring (Test 6)
- ~140 timepoints over 1µs
- Approaching benchmark behavior
- Success status

### 9-Stage Ring (Test 7)
- ~180 timepoints over 1µs
- ~531 Newton iterations (≈3.0 per step)
- **Matches benchmark exactly**
- Success status

## Troubleshooting

### Segfaults During Precompilation (gVisor Sandbox)

Create `test/LocalPreferences.toml`:

```toml
[PSPModels]
precompile_workload = false

[VADistillerModels]
precompile_workload = false
```

### Out of Memory (Sandbox Environment)

gVisor sandbox has limited memory. Large VA model compilations may OOM:
- Use Julia 1.11 (not 1.12)
- Disable precompilation workloads
- Run tests individually instead of full suite

### Unstable Transient Solutions

If ring oscillators fail to converge:
- Check dtmax (should be 0.01ns or 0.1ns)
- Verify CedarTranOp is used
- Increase maxiters if needed
- Reduce dtmax for smaller rings (they oscillate faster)

## Integration with CI

These tests complement existing integration tests:

| Test File | Focus | Speed |
|-----------|-------|-------|
| `psp103_integration.jl` | Minimal model parsing | Fast |
| **`psp103_bridge.jl`** | **Full model complexity scaling** | **Slow (JIT)** |
| Benchmark (`runme.jl`) | Production performance | Very slow |

The bridge tests should run in CI to catch regressions in:
- Full model card support
- Transient initialization (CedarTranOp)
- Ring oscillator convergence
- Scaling behavior

## Related Files

- Benchmark: `benchmarks/vacask/ring/cedarsim/runme.jl`
- Model card: `benchmarks/vacask/ring/cedarsim/models.inc`
- Integration tests: `test/mna/psp103_integration.jl`
- VA model: `models/PSPModels.jl/va/psp103.va`

## Future Enhancements

Potential additions to bridge tests:

1. **AC Analysis**: Small-signal AC sweep of inverter stages
2. **Parameter Sweeps**: Vary W/L ratios across ring stages
3. **Temperature Sweep**: Test model at different temperatures
4. **Noise Analysis**: Verify noise parameters in full model
5. **Different Supply Voltages**: Test VDD = 0.8V, 1.0V, 1.5V

## References

- PSP103VA model documentation: [Compact Model Coalition](https://www.si2.org/standard-models/)
- Ring oscillator theory: Any VLSI textbook (Razavi, Baker, etc.)
- MNA backend design docs: `doc/` directory
