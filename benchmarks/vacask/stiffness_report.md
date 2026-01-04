# VACASK Stiffness Analysis Report

Analyzed on Julia 1.11.8

## Condition Number Summary

The condition number κ(J) of the Jacobian J = G + γC measures system stiffness.
Higher values indicate more difficult numerical integration.

| Benchmark | Size | κ(dt=1ns) | κ(dt=1μs) | κ(dt=1ms) | Stiffness Ratio |
|-----------|------|-----------|-----------|-----------|-----------------|
| RC Circuit | 3 | 1.00e+03 | 1.00e+00 | 5.00e+02 | 1.00e+00 |
| Graetz Bridge | 9 | 8.00e+11 | 9.20e+08 | 2.86e+08 | 9.20e+08 |
| Voltage Multiplier | 10 | 1.12e+03 | 7.32e+03 | 7.31e+06 | 7.32e+03 |
| Ring Oscillator | 281 | 9.61e+06 | 5.52e+09 | 1.29e+10 | 5.52e+09 |
| C6288 Multiplier | - | Failed | Failed | Failed | - |

## Detailed Results

### RC Circuit

| Metric | Value |
|--------|-------|
| System Size | 3 |
| Voltage Nodes | 2 |
| Current Variables | 1 |
| Charge Variables | 0 |
| κ(J) at dt=1ns | 1.00e+03 |
| κ(J) at dt=1μs | 1.00e+00 |
| κ(J) at dt=1ms | 5.00e+02 |
| Max Eigenvalue | 1.00e+00 |
| Min Eigenvalue | 1.00e+00 |
| Stiffness Ratio | 1.00e+00 |
| Voltage Range | [0.00e+00, 0.00e+00] |
| Current Range | [0.00e+00, 0.00e+00] |
| Charge Range | - |

### Graetz Bridge

| Metric | Value |
|--------|-------|
| System Size | 9 |
| Voltage Nodes | 8 |
| Current Variables | 1 |
| Charge Variables | 0 |
| κ(J) at dt=1ns | 8.00e+11 |
| κ(J) at dt=1μs | 9.20e+08 |
| κ(J) at dt=1ms | 2.86e+08 |
| Max Eigenvalue | 2.30e+02 |
| Min Eigenvalue | 2.50e-07 |
| Stiffness Ratio | 9.20e+08 |
| Voltage Range | [0.00e+00, 0.00e+00] |
| Current Range | [0.00e+00, 0.00e+00] |
| Charge Range | - |

### Voltage Multiplier

| Metric | Value |
|--------|-------|
| System Size | 10 |
| Voltage Nodes | 9 |
| Current Variables | 1 |
| Charge Variables | 0 |
| κ(J) at dt=1ns | 1.12e+03 |
| κ(J) at dt=1μs | 7.32e+03 |
| κ(J) at dt=1ms | 7.31e+06 |
| Max Eigenvalue | 2.14e+02 |
| Min Eigenvalue | 2.93e-02 |
| Stiffness Ratio | 7.32e+03 |
| Voltage Range | [5.33e-12, 5.00e+01] |
| Current Range | [2.76e-10, 2.76e-10] |
| Charge Range | - |

### Ring Oscillator

| Metric | Value |
|--------|-------|
| System Size | 281 |
| Voltage Nodes | 154 |
| Current Variables | 127 |
| Charge Variables | 0 |
| κ(J) at dt=1ns | 9.61e+06 |
| κ(J) at dt=1μs | 5.52e+09 |
| κ(J) at dt=1ms | 1.29e+10 |
| Max Eigenvalue | 1.14e+02 |
| Min Eigenvalue | 2.06e-08 |
| Stiffness Ratio | 5.52e+09 |
| Voltage Range | [6.42e-01, 1.20e+00] |
| Current Range | [6.42e-13, 2.61e-03] |
| Charge Range | - |

### C6288 Multiplier

> ❌ Analysis failed: SystemError: opening file "multiplier.inc": No such file or directory

## Interpretation

- **Condition Number**: κ(J) > 10^6 indicates a stiff system
- **Stiffness Ratio**: λ_max/λ_min > 10^6 requires implicit solvers
- **State Magnitude Spread**: Large differences between voltage/current/charge
  magnitudes contribute to poor conditioning

### Recommendations

If the system is stiff due to charge state magnitude differences:
1. **Charge Scaling**: Scale charge variables to match voltage/current magnitudes
2. **State Normalization**: Use scaled state variables in the DAE formulation
3. **Preconditioner**: Apply diagonal scaling preconditioner

