# MNA Residual Function Type Stability Analysis

## Executive Summary

**The residual function allocates ~2KB per call**, which explains why the VACASK benchmarks show high allocation counts and are slower than ngspice. The root cause is that `fast_rebuild!` calls the circuit builder function on every Newton iteration, creating a **new MNAContext** each time.

For a 1-second simulation with ~2M iterations, this translates to **~4GB of allocations** that need to be garbage collected!

## Current Implementation Status

### Completed (Phase 3b) ✅

**Zero-allocation StaticCircuit for linear circuits:**

| Type | File | Status | Allocations |
|------|------|--------|-------------|
| `StaticCircuit{N,T,L}` | `src/mna/compiled.jl` | ✅ Complete | **0 bytes** |
| `CPUCircuit{T}` | `src/mna/compiled.jl` | ✅ Complete | Low (fixed matrices) |
| `NonlinearStaticCircuit` | `src/mna/nonlinear_compiled.jl` | 🔶 Prototype | 0 bytes (linear only) |

**Test coverage:** 28 tests passing in `test/mna/compiled.jl`

### Usage Example

```julia
using CedarSim.MNA
using StaticArrays

# Build a circuit
ctx = MNAContext()
vcc = get_node!(ctx, :vcc)
out = get_node!(ctx, :out)
stamp!(VoltageSource(5.0), ctx, vcc, 0)
stamp!(Resistor(1000.0), ctx, vcc, out)
stamp!(Resistor(1000.0), ctx, out, 0)

# Compile to StaticCircuit (zero allocation)
circuit = compile_static_circuit(ctx, Val(3))

# Zero-allocation residual evaluation
u = @SVector [5.0, 2.5, -0.0025]
du = @SVector zeros(3)
resid = static_residual(du, u, circuit, 0.0)  # Returns SVector{3,Float64}

# Verify zero allocation in loops
function benchmark(c::StaticCircuit{3,Float64,9}, n::Int)
    for _ in 1:n
        u = @SVector [5.0, 2.5, -0.0025]
        du = @SVector zeros(3)
        static_residual(du, u, c, 0.0)
    end
end
@allocated benchmark(circuit, 1000)  # Returns 0
```

## API Comparison

| API | Use Case | Allocations/iter | Nonlinear Support |
|-----|----------|------------------|-------------------|
| `PrecompiledCircuit` | Standard CPU transient | ~2KB | ✅ Yes |
| `MNACircuit` → `DAEProblem` | SciML integration | ~2KB (uses PrecompiledCircuit) | ✅ Yes |
| `CPUCircuit` (new) | Linear CPU circuits | ~0 (fixed G,C,b) | ❌ No |
| `StaticCircuit` (new) | GPU ensemble/Monte Carlo | **0 bytes** | ❌ No |
| `NonlinearStaticCircuit` (prototype) | Nonlinear + GPU | **0 bytes** | 🔶 Prototype |

### Key Difference

The **existing** `PrecompiledCircuit` rebuilds the circuit every Newton iteration:

```julia
# precompile.jl:440-482 - allocates ~2KB per call
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    ctx = pc.builder(pc.params, spec_t; x=u)  # <-- Creates new MNAContext!
    # ... copies values from ctx to pc
end
```

The **new** `StaticCircuit` stores pre-computed constant matrices:

```julia
# compiled.jl - zero allocation
@inline function static_residual(du::SVector{N,T}, u::SVector{N,T},
                                  circuit::StaticCircuit{N,T,L}, t) where {N,T,L}
    return circuit.C * du + circuit.G * u - circuit.b  # All stack-allocated
end
```

## Remaining Work

### Phase 3b Completion: Nonlinear StaticCircuit

The `NonlinearStaticCircuit` prototype in `src/mna/nonlinear_compiled.jl` shows the architecture:

```julia
# Store stamp operations instead of constant matrices
struct NonlinearStaticCircuit{N,T,K,Stamps<:Tuple}
    stamps::Stamps  # Tuple of StampOp (ConstantStamp, LinearStamp, FunctionStamp)
    node_names::NTuple{N,Symbol}
end

# @generated function unrolls evaluation at compile time
@generated function build_matrices(circuit::NonlinearStaticCircuit{N,T,K,Stamps}, u, t)
    # Generates inlined evaluation of all stamps
    # Returns fresh SMatrix/SVector (stack-allocated)
end
```

**Required work:**
1. Extract stamp evaluators from VA device models
2. Integrate with SPICE codegen to emit stamp operations
3. Handle time-dependent sources (PWL, SIN, etc.)

### Phase 3c: GPU Integration

Not started. Required work:
1. Add CUDA.jl as optional dependency
2. Implement `GPUCircuit{T}` with CuSparseMatrixCSC
3. Test with DiffEqGPU.jl ensemble problems

### Make Zero-Allocation Default

To automatically use zero-allocation for all simulations:

1. **Detection**: Check if circuit is linear (no x-dependent stamps)
2. **Auto-selection**:
   - Linear + small (N < 20) → `StaticCircuit`
   - Linear + large → `CPUCircuit` with pointer stamping
   - Nonlinear + small → `NonlinearStaticCircuit`
   - Nonlinear + large → `PrecompiledCircuit` (current behavior)
3. **Integration**: Modify `DAEProblem(circuit, tspan)` to auto-select

## Investigation Results

### Allocation Measurements

| Function | Allocations per Call |
|----------|---------------------|
| `fast_residual!` | 2160 bytes |
| `fast_rebuild!` | 2192 bytes |
| `build_rc (builder)` | 2224 bytes |
| `MNAContext()` | 640 bytes |
| `update_sparse_from_coo!` | 48 bytes |
| `MNASpec(...)` | 0 bytes |
| **`static_residual` (new)** | **0 bytes** |

### Root Cause

The problem is in `fast_rebuild!` (precompile.jl:440-482):

```julia
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    # Creates a new spec (0 bytes - ok)
    spec_t = MNASpec(temp=pc.spec.temp, mode=:tran, time=real_time(t))

    # PROBLEM: Creates a brand new MNAContext!
    # This allocates ~2KB per call:
    # - 640 bytes for MNAContext itself
    # - Additional bytes for push! into vectors during stamping
    ctx = pc.builder(pc.params, spec_t; x=u)  # <-- ALLOCATES!

    # The rest copies values from ctx to pc (OK, no allocations)
    ...
end
```

### Type Stability Analysis

The `@code_warntype` output shows that `fast_residual!` and `fast_rebuild!` are **type-stable** - all types are fully inferred. The problem is not type instability but rather the **allocating operations** inside the builder call.

## External Reference Implementations

### VACASK Architecture (C++)

VACASK separates circuit setup into distinct phases:

1. **Structure Discovery** (`populateStructures()`): Determines sparsity pattern
2. **Pointer Binding** (`bindCore()`): Gets raw pointers to sparse matrix entries
3. **Evaluation** (`eval()` + `load()`): Writes directly to pre-bound pointers - **zero allocation**

```cpp
// In bindCore() - called once during setup
jacResistArray[i] = matResist->valuePtr(MatrixEntryPosition(e, u), ...);

// In loadCore() - called every Newton iteration
// Writes directly through stored pointers - NO ALLOCATION
descriptor->load_jacobian_resist(instance, model);
```

### OpenVAF OSDI Interface

OpenVAF compiles Verilog-A to native code with the OSDI interface:

```c
typedef struct OsdiDescriptor {
    uint32_t jacobian_ptr_resist_offset;  // Where to store Jacobian pointers
    void (*load_jacobian_resist)(void *inst, void* model);  // Write through pointers
} OsdiDescriptor;
```

### SciML GPU Requirements

For DiffEqGPU.jl ensemble solving:

| Mode | Array Type | Formulation | Use Case |
|------|------------|-------------|----------|
| Within-method | `CuArray` | In-place `f!(du, u, p, t)` | Large single circuit |
| Ensemble | `SVector{N,T}` | Out-of-place `f(u, p, t) -> SVector` | Parameter sweeps, Monte Carlo |

## Architecture

### Implemented Types

```
src/mna/compiled.jl:
├── CPUCircuit{T}              # Sparse matrices, in-place
│   └── compile_cpu_circuit()
│   └── cpu_residual!()
│   └── solve_dc_cpu()
│
└── StaticCircuit{N,T,L}       # Dense SMatrix/SVector, out-of-place
    └── compile_static_circuit()
    └── static_residual()      # Zero allocation ✅
    └── static_residual!()
    └── solve_dc_static()
    └── to_static()            # Convert from CPUCircuit

src/mna/nonlinear_compiled.jl:
└── NonlinearStaticCircuit{N,T,K,Stamps}  # Prototype
    └── StampOp hierarchy (ConstantStamp, LinearStamp, etc.)
    └── @generated build_matrices()
    └── nonlinear_residual()
```

### Target Architecture (Full Implementation)

```
CompiledCircuit{VecType, MatType, T}
├── CPUCircuit{T}          # VecType=Vector{T}, MatType=SparseMatrixCSC{T,Int}
│   └── In-place: residual!(du, u, p, t)
│   └── Use: Standard CPU simulation
│
├── GPUCircuit{T}          # VecType=CuArray{T}, MatType=CuSparseMatrixCSC{T,Int}
│   └── In-place: residual!(du, u, p, t)
│   └── Use: Large single-circuit GPU simulation
│
└── StaticCircuit{N,T}     # VecType=SVector{N,T}, MatType=SMatrix{N,N,T}
    └── Out-of-place: residual(u, p, t) -> SVector{N,T}
    └── Use: Ensemble GPU for parameter sweeps, Monte Carlo
```

## Implementation Roadmap

### ✅ Phase 3b.1: Linear StaticCircuit (COMPLETE)

- [x] `StaticCircuit{N,T,L}` with SMatrix/SVector
- [x] `compile_static_circuit()` from MNAContext
- [x] `static_residual()` - zero allocation verified
- [x] Float32 support
- [x] 28 tests passing

### 🔶 Phase 3b.2: Nonlinear StaticCircuit (IN PROGRESS)

- [x] `NonlinearStaticCircuit` prototype
- [x] `StampOp` hierarchy (ConstantStamp, LinearStamp, etc.)
- [x] `@generated build_matrices()` for compile-time unrolling
- [ ] Extract stamp evaluators from VA models
- [ ] Integrate with SPICE codegen
- [ ] Handle time-dependent sources

### ⬜ Phase 3c: GPU Integration (NOT STARTED)

- [ ] Add CUDA.jl optional dependency
- [ ] Implement `GPUCircuit{T}` with CuSparseMatrixCSC
- [ ] Test with DiffEqGPU.jl EnsembleGPUArray
- [ ] Verify GPU kernel generation

### ⬜ Phase 4: Make Default (NOT STARTED)

- [ ] Add circuit linearity detection
- [ ] Auto-select optimal compiled representation
- [ ] Integrate with `DAEProblem(circuit, tspan)` API
- [ ] Benchmark against ngspice

## Files

| File | Description |
|------|-------------|
| `src/mna/compiled.jl` | StaticCircuit and CPUCircuit implementations |
| `src/mna/nonlinear_compiled.jl` | NonlinearStaticCircuit prototype |
| `src/mna/precompile.jl` | Existing PrecompiledCircuit (allocating) |
| `test/mna/compiled.jl` | Zero-allocation verification tests |
| `benchmarks/type_stability_analysis.jl` | Investigation script |
