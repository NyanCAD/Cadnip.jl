# CI Setup for PSP103 Bridge Tests

This document describes the CI configuration for the PSP103 bridge tests.

## Overview

The bridge tests have been integrated into the CI pipeline as a separate job (`test-bridge`) that runs in parallel with core and integration tests.

## CI Jobs

The CI now runs three parallel jobs:

### 1. `test-core` (Fastest)
- **Duration**: ~5-10 minutes
- **Purpose**: Fast core tests without heavy VA models
- **Tests Run**:
  - SpectreNetlistParser tests
  - VerilogAParser tests
  - MNA core tests
  - Basic circuit tests
  - PDK precompilation tests

### 2. `test-integration` (Medium)
- **Duration**: ~10-15 minutes
- **Purpose**: Heavy VA model integration tests
- **Tests Run**:
  - VADistiller integration (BSIM4, large models)
  - Audio integration (BJT circuits)
  - PSP103 basic integration

### 3. `test-bridge` (New - Slowest)
- **Duration**: ~15-20 minutes
- **Purpose**: PSP103 progressive complexity scaling
- **Tests Run**:
  - Single NMOS/PMOS with full 280+ param model (DC)
  - Single inverter with full model (DC)
  - 3/5/7/9-stage ring oscillators (transient)

## Test Invocation

### Via Command Line

```bash
# Run only bridge tests
julia --project=test test/runtests.jl bridge

# Run only integration tests
julia --project=test test/runtests.jl integration

# Run only core tests (default)
julia --project=test test/runtests.jl

# Run all tests
julia --project=test test/runtests.jl all
```

### In runtests.jl

The test runner now supports four modes via `ARGS`:

```julia
const RUN_INTEGRATION = "integration" in ARGS || "all" in ARGS
const RUN_BRIDGE = "bridge" in ARGS || "all" in ARGS
const RUN_CORE = !("integration" in ARGS) && !("bridge" in ARGS) || "all" in ARGS
```

## CI Configuration Details

### test-bridge Job

```yaml
test-bridge:
  name: Bridge Tests (PSP103) - Julia ${{ matrix.version }}
  runs-on: ubuntu-latest
  strategy:
    fail-fast: false
    matrix:
      version:
        - '1.11'
        - '1.12'
```

### Key Features

1. **Precompile Workload Disabled**: Prevents segfaults during PSP103VA compilation
   ```bash
   cat > test/LocalPreferences.toml << 'EOF'
   [PSPModels]
   precompile_workload = false
   ...
   EOF
   ```

2. **20-Minute Timeout**: Allows for ~400s JIT compilation overhead
   ```yaml
   timeout-minutes: 20
   ```

3. **Julia Caching**: Reuses cached packages across runs
   ```yaml
   - uses: julia-actions/cache@v2
   ```

4. **Matrix Testing**: Tests on both Julia 1.11 and 1.12
   ```yaml
   matrix:
     version:
       - '1.11'
       - '1.12'
   ```

## Trigger Conditions

Bridge tests run on:
- ✅ All pull requests
- ✅ Pushes to `main` branch
- ✅ Pushes to `master` branch

## Expected CI Behavior

### First Run (Cold Cache)
- **Total Time**: ~18-20 minutes
  - Package installation: ~3 minutes
  - Precompilation: ~2 minutes
  - PSP103VA JIT compilation: ~6-7 minutes
  - Test execution: ~7-8 minutes

### Subsequent Runs (Warm Cache)
- **Total Time**: ~12-15 minutes
  - Package installation: <1 minute (cached)
  - Precompilation: ~1 minute (cached)
  - PSP103VA JIT compilation: ~6-7 minutes (required)
  - Test execution: ~5-6 minutes

### Failure Scenarios

The bridge tests will fail if:
1. PSP103VA model parsing breaks
2. Full model card support regresses
3. Ring oscillator convergence fails
4. Memory issues (OOM)
5. Numerical instability in transient solver

## Monitoring

Check CI status at:
- PR status checks
- GitHub Actions tab
- Commit status indicators

## Parallel Execution

All three jobs run in parallel:
```
test-core        [========] 8min
test-integration [===========] 13min
test-bridge      [===============] 18min
                 └─ Total wall time: 18min
```

Without parallelization, total time would be ~36 minutes.

## Future Improvements

Potential optimizations:
1. **Artifact caching**: Cache compiled PSP103VA functions
2. **Selective testing**: Run only affected tests on PRs
3. **Fast-fail mode**: Stop early if DC tests fail
4. **Benchmark comparison**: Compare ring oscillator performance across runs

## Debugging CI Failures

### Compilation Failures
Check for:
- PSP103VA syntax errors
- Model card parsing issues
- Missing dependencies

### Memory Issues
Check for:
- OOM errors in logs
- Swap usage
- System limits

### Timeout Issues
Check for:
- Infinite loops in solver
- Excessive Newton iterations
- Memory thrashing

### Numerical Issues
Check for:
- Solver convergence failures
- NaN/Inf values
- Unstable transient solutions

## Related Files

- CI workflow: `.github/workflows/ci.yml`
- Test runner: `test/runtests.jl`
- Bridge tests: `test/mna/psp103_bridge.jl`
- Documentation: `test/mna/PSP103_BRIDGE_README.md`
