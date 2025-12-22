# MNA Architecture Design Document

This document captures the design decisions for the MNA (Modified Nodal Analysis)
engine, based on analysis of CedarSim's existing patterns, OpenVAF/VACASK interfaces,
and GPU computing requirements.

## Design Principles

### 1. Out-of-Place Evaluation

Circuit evaluation functions return new matrices rather than mutating:

```julia
function eval_circuit(lens, spec, t, u)
    G = spzeros(n, n)
    C = spzeros(n, n)
    b = zeros(n)
    # ... stamp devices ...
    return (G, C, b)
end
```

**Rationale:**
- **GPU compatibility**: DiffEqGPU.jl requires out-of-place formulation (`ODEProblem{false}`)
- **Simplicity**: No need to track which entries to reset
- **Ensemble solving**: Parameter sweeps use `EnsembleGPUKernel` which needs out-of-place
- **JIT optimization**: Julia's compiler can still optimize constant assignments

### 2. Explicit Parameter Passing (No ScopedValue)

Parameters flow explicitly through function arguments, not via `ScopedValue`:

```julia
# NOT this (ScopedValue):
temper()  # reads from global spec[]

# THIS (explicit):
spec.temp  # passed as argument
```

**Rationale:**
- DAECompiler treats `ScopedValue` as constant via compiler magic - not available in plain Julia
- Explicit passing enables full JIT optimization
- Avoids Julia's closure boxing bug (captured variables become `Core.Box`)

### 3. Separation of Concerns

```julia
struct MNASpec
    temp::Float64      # Temperature (Celsius)
    mode::Symbol       # :dcop, :tran, :tranop, :ac
end

# lens: ParamLens for circuit parameters (from ParamSim)
# spec: Simulation specification
# t: Current time
# u: Current solution (for non-linear devices)
```

**Parameter access via ParamLens:**
```julia
function build_circuit(lens, spec, t, u)
    # lens.subcircuit returns scoped lens for subcircuit
    # lens(; R=1000.0) returns actual value with lens overrides
    p = lens(; R=1000.0, C=1e-6)

    stamp_resistor!(G, n1, n2, p.R)
end
```

### 4. JIT-Friendly Stamping

Since Julia JIT-compiles circuit functions, linear device stamps optimize to constant
assignments. No need to manually separate "setup" from "eval" phases:

```julia
function eval_circuit(lens, spec, t, u)
    p = lens(; R=1000.0)

    # Linear: compiler sees p.R as constant
    stamp_resistor!(G, n1, n2, p.R)  # → G[i,j] = const

    # Temperature-dependent: constant within simulation
    R_t = p.R * (1 + p.tc * (spec.temp - 27))
    stamp_resistor!(G, n3, n4, R_t)

    # Time-dependent: changes each timestep
    v = spec.mode == :dcop ? p.Vdc : pwl(p.times, p.vals, t)
    stamp_vsource!(G, b, n5, n6, v)

    # Non-linear: changes each Newton iteration
    V_d = u[n7] - u[n8]
    I_d, G_d = diode_model(V_d, p.Is, p.Vt)
    stamp_conductance!(G, n7, n8, G_d)
    stamp_current!(b, n7, n8, I_d - G_d * V_d)

    return (G, C, b)
end
```

## Device Categories

### Linear Devices (Constant Stamps)
- Resistor: stamps G matrix
- Capacitor: stamps C matrix
- Inductor: stamps G and C matrices (with current variable)
- Ideal voltage/current sources: stamps G and b

These have constant contributions that the JIT can optimize.

### Temperature-Dependent Devices
- Resistors with tempco: `R(T) = R0 * (1 + tc1*(T-Tnom) + tc2*(T-Tnom)²)`
- Semiconductor models with temperature scaling

Constant within a simulation run, recomputed for temperature sweeps.

### Time-Dependent Sources
- PWL (Piecewise Linear)
- PULSE, SIN, EXP waveforms
- Controlled sources with time-varying control

Evaluated at each timestep. In `:dcop` mode, return DC value.

### Non-Linear Devices
- Diodes: `I = Is*(exp(V/Vt) - 1)`
- MOSFETs, BJTs, etc.

Require Newton-Raphson iteration. Each iteration:
1. Evaluate device equations at current `u`
2. Compute companion model (linearized around operating point)
3. Stamp equivalent conductance and current source

## Integration with SPICE Codegen

SPICE codegen will emit circuit functions following this pattern:

```julia
# .TEMP 50 generates:
spec = @set spec.temp = 50.0

# .PARAM R1=2k generates:
params = @set params.R1 = 2000.0

# Device instantiation:
function circuit(lens, spec, t, u)
    p = lens(; R1=1000.0)  # default, lens overrides
    stamp_resistor!(G, n1, n2, p.R1)
    # ...
end
```

The `@set` macro (Accessors.jl) creates new bindings in SSA style,
enabling constant folding when values are known at compile time.

## GPU Support

### Ensemble GPU (Parameter Sweeps)
For many small circuits with different parameters:
- Use `ODEProblem{false}` (out-of-place)
- StaticArrays for state vectors
- `EnsembleGPUKernel(CUDA.CUDABackend())`

### Large Circuit GPU
For single large circuits:
- `CuSparseMatrixCSR` for sparse matrices
- Iterative solvers via Krylov.jl
- Direct solvers via CUDSS.jl

## Comparison with OpenVAF/OSDI

OpenVAF's OSDI interface separates:
- `setup_instance(temp, ...)` - one-time setup
- `eval(sim_info)` - per-iteration evaluation
- `load_jacobian_resist/react` - separate resistive/reactive
- `JACOBIAN_ENTRY_RESIST_CONST` flags for constant entries

This separation is for C ABI efficiency without JIT. In Julia, we can
let the JIT handle optimization and use a simpler unified `eval` function.

Key insight from OSDI:
- Temperature is passed explicitly to setup
- Time (`abstime`) flows through `sim_info` each iteration
- Analysis mode is a flag (ANALYSIS_DC, ANALYSIS_TRAN)

## References

- CedarSim ParamLens: `src/spectre.jl:140-180`
- CedarSim SimSpec: `src/simulate_ir.jl:20-32`
- OpenVAF OSDI: `refs/OpenVAF/melange/core/src/veriloga/osdi_0_4.rs`
- VACASK device eval: `refs/VACASK/lib/osdiinstance.cpp`
- DiffEqGPU: https://docs.sciml.ai/DiffEqGPU/stable/getting_started/
