# OpenVAF vs Cadnip.jl: Codegen Comparison for Large VA Models

## Overview

This document compares the code generation strategies of OpenVAF and Cadnip.jl for handling large Verilog-A models like PSP103VA (~200 parameters, ~96K IR statements).

## The Problem

Large VA models produce functions with so many statements that LLVM crashes during compilation. The crashes happen due to:
1. **SROA (Scalar Replacement of Aggregates)** - LLVM tries to decompose large structs (782 fields for PSP103VA) into individual scalar variables
2. **Inline specialization** - Julia tries to specialize and inline the `stamp!` method into the circuit function
3. **Memory exhaustion** - The generated IR is too large for LLVM's optimization passes

## OpenVAF Architecture

OpenVAF is a Rust-based compiler that compiles Verilog-A to native shared libraries (OSDI format).

### Key Components

```
Verilog-A Source
    ↓
Parser (lexer → syntax → hir)
    ↓
HIR Lowering (hir_lower)
    ↓
Custom MIR (mir)
    ↓
MIR Optimization (mir_opt)
    ↓
Automatic Differentiation (mir_autodiff)
    ↓
LLVM IR Generation (mir_llvm)
    ↓
Native Code (.so/.dll)
```

### Code Splitting Strategy

OpenVAF splits each VA module into **4 separate LLVM modules** that are compiled independently and linked later (`osdi/src/lib.rs:91-165`):

1. **access_{module}** - Accessor functions for parameters
2. **setup_model_{module}** - Model parameter initialization
3. **setup_instance_{module}** - Instance parameter initialization
4. **eval_{module}** - Main evaluation function (I/V relationships)

This is achieved through Rayon parallel compilation:
```rust
rayon_core::scope(|scope| {
    for (i, module) in modules.iter().enumerate() {
        scope.spawn(move |_| { /* compile access function */ });
        scope.spawn(move |_| { /* compile setup_model */ });
        scope.spawn(move |_| { /* compile setup_instance */ });
        scope.spawn(move |_| { /* compile eval */ });
    }
});
```

### Initialization/Eval Split

OpenVAF separates **operating point independent** calculations into an initialization function (`sim_back/src/init.rs`):

- **Initialization function**: Computes values that don't depend on voltages/currents (parameter processing, temperature-dependent calculations)
- **Eval function**: Only contains operating point dependent calculations (I/V relationships, derivatives)

This is done by analyzing "taint propagation" - identifying which values depend on operating point variables (`mir_opt/src/split_tainted.rs`).

### MIR-Level Optimization

Before LLVM sees the code, OpenVAF runs extensive optimization passes on its custom MIR (`sim_back/src/context.rs:68-99`):

1. **Dead Code Elimination** - Remove unused computations
2. **SCCP** - Sparse Conditional Constant Propagation
3. **Inst Combine** - Instruction combining/peephole optimization
4. **CFG Simplification** - Control flow graph simplification
5. **GVN** - Global Value Numbering (CSE on steroids)
6. **Aggressive DCE** - More aggressive dead code elimination

### Automatic Differentiation at MIR Level

Jacobian derivatives are computed at the MIR level (`mir_autodiff/`), not LLVM level. This allows:
- Symbolic derivative optimization before LLVM sees it
- Dead derivative elimination
- Chain rule simplification

### DAE Sparsification

The DAE system is sparsified before codegen (`sim_back/src/dae.rs:62-100`):
- Zero entries in Jacobian are removed
- Zero residuals are eliminated
- This reduces the generated code size significantly

## Cadnip.jl Architecture

Cadnip.jl generates Julia code that is JIT-compiled at runtime.

### Current Approach

```
Verilog-A Source
    ↓
VerilogAParser.jl
    ↓
Julia AST Generation (vasim.jl)
    ↓
Julia Compilation (JIT)
    ↓
Native Code
```

### Workarounds for LLVM Crashes

Cadnip.jl uses inference barriers to prevent LLVM from attempting problematic optimizations:

1. **`Base.invokelatest()`** - Forces runtime dispatch, prevents inlining (`src/spc/codegen.jl:1393-1400`)
2. **`Base.inferencebarrier()`** - Hides type information from compiler (`src/spectre.jl:468-475`)
3. **`@noinline`** - Prevents function inlining (`src/vasim.jl:1997`)
4. **PrecompileTools** - Pre-compiles `stamp!` methods during package load

These are workarounds that sacrifice performance for compilability.

## Key Differences

| Aspect | OpenVAF | Cadnip.jl |
|--------|---------|-----------|
| **Compilation** | Ahead-of-time | JIT at runtime |
| **IR** | Custom MIR → LLVM IR | Julia AST → Julia IR → LLVM IR |
| **Code splitting** | 4 separate LLVM modules | Single `stamp!` function |
| **Initialization** | Separate init function | Part of stamp! |
| **Optimization** | MIR-level before LLVM | Relies on Julia/LLVM |
| **Differentiation** | MIR-level symbolic | Julia-level AD (DiffRules) |
| **Large struct handling** | N/A (uses pointers) | SROA workarounds needed |

## Potential Improvements for Cadnip.jl

### 1. Code Splitting (High Impact)

Split the generated `stamp!` method into multiple smaller functions:
- **Static computations**: Temperature-dependent, parameter-only calculations
- **Dynamic computations**: Voltage/current-dependent calculations
- **Jacobian contributions**: Separate function for matrix stamping

This could be achieved by analyzing which computations depend on the solution vector `x`.

### 2. MIR-Level Optimization

Add optimization passes to the generated Julia AST before compilation:
- Constant folding for fixed parameters
- Dead code elimination for unused branches
- Common subexpression elimination

### 3. Caching Static Computations

Pre-compute and cache values that don't depend on the operating point:
```julia
struct PSP103VACache
    temp_coeffs::NTuple{100, Float64}
    # ... other cached values
end
```

### 4. Struct Flattening

Instead of a 782-field struct, use a flat vector with named indices:
```julia
const PSP103VA_PARAMS = (VTH0=1, K1=2, ...)  # NamedTuple of indices
params::Vector{Float64}
```

This avoids SROA issues entirely.

### 5. External Compilation Path

For very large models, consider an external compilation path similar to OpenVAF:
1. Generate C code instead of Julia code
2. Compile with clang/gcc
3. Load as shared library via `ccall`

## Conclusion

OpenVAF's approach of:
1. Custom MIR with optimization passes
2. Code splitting into multiple compilation units
3. Separating initialization from evaluation
4. Operating at a lower level than LLVM

...allows it to handle arbitrarily large VA models without LLVM crashes.

Cadnip.jl's JIT approach provides flexibility but hits LLVM limits with large models. The current workarounds (inference barriers, @noinline) sacrifice some performance.

The most promising improvements for Cadnip.jl would be:
1. **Code splitting** - Break stamp! into smaller functions
2. **Caching** - Pre-compute static values
3. **Struct flattening** - Avoid large struct SROA issues

These could be implemented incrementally without changing the fundamental architecture.
