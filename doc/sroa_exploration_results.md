# SROA Prevention Exploration Results

## Problem Statement

Large VA model structs (e.g., PSP103VA with 782 fields) cause LLVM compilation issues:
1. LLVM SROA (Scalar Replacement of Aggregates) tries to decompose structs into individual scalars
2. This creates massive IR (96K+ statements for PSP103VA stamp!)
3. LLVM can crash or take excessive time compiling such large functions

The current solution uses `invokelatest` to force runtime dispatch. This exploration investigates whether simpler annotations can achieve the same result with less overhead.

## Tested Approaches

| Approach | Description |
|----------|-------------|
| `direct` | No barriers - baseline for comparison |
| `@noinline` | Mark stamp function as non-inlineable |
| `invokelatest` | Force runtime dispatch (current approach) |
| `@nospecialize` | Prevent type specialization |
| `inferencebarrier` | Hide type from compiler |
| `Ref{Any}` | Force boxing via heap allocation |

## Results Summary

### Compile Time (600-field struct, circuit builder pattern)

| Approach | First Context | Second Context |
|----------|---------------|----------------|
| direct | 764 ms | - |
| @noinline | 90 ms | 74 ms |
| invokelatest | 104 ms | 109 ms |
| inferencebarrier | 91 ms | - |

**Key finding**: `@noinline` reduces compile time by ~8x compared to direct calls.

### Runtime Performance (600-field struct, 3 devices)

| Approach | Runtime (ns) | Overhead vs baseline |
|----------|-------------|---------------------|
| direct/inline | 1613 | 1.0x |
| @noinline | 1628 | 1.0x |
| invokelatest | 2658 | 1.6x |
| @nospecialize | 38915 | 24x |
| inferencebarrier | 1620 | 1.0x |

**Key finding**: `@noinline` has NO runtime overhead, while `invokelatest` has ~1.6x overhead.

### LLVM IR Analysis (parent circuit builder function)

| Approach | IR Lines | Float Loads | SROA Prevented? |
|----------|----------|-------------|-----------------|
| direct | 419 | 14 | No |
| @noinline | 320 | 0 | **Yes** |
| invokelatest | 348 | 0 | Yes |
| inferencebarrier | 419 | 14 | No |

**Key finding**: `@noinline` prevents SROA in the parent function (0 float loads = struct not decomposed).

## Conclusions

### 1. `@noinline` is the optimal solution

- **Prevents SROA**: The parent function's IR shows 0 float loads (struct passed as-is)
- **No runtime overhead**: Same performance as direct inlined calls
- **8x faster compile time**: Compile time doesn't grow with struct size
- **Simple**: Just one annotation, no runtime dispatch overhead

### 2. `invokelatest` should be removed

- Has 1.6x runtime overhead due to dynamic dispatch
- Not needed since `@noinline` already prevents SROA
- Was likely added as a workaround before `@noinline` was in place

### 3. `inferencebarrier` doesn't help

- Does NOT prevent SROA in the parent function (same IR as direct)
- Only hides type at the specific barrier point
- No benefit over `@noinline`

### 4. `@nospecialize` is harmful

- 24x runtime overhead due to runtime type dispatch on every field access
- Should never be used for hot paths

## Recommended Changes

1. **Keep** `@noinline` on `stamp!` methods (already present)
2. **Remove** `invokelatest` from codegen - replace with direct calls
3. **Remove** `inferencebarrier` from device construction - not needed

### Code Change Example

```julia
# Before (current)
Base.invokelatest($(MNA).stamp!, dev, ctx, $(port_exprs...); ...)

# After (recommended)
$(MNA).stamp!(dev, ctx, $(port_exprs...); ...)
```

The `@noinline` annotation already on stamp! methods ensures:
- stamp! is compiled separately, not inlined
- Circuit builder IR stays small
- No LLVM SROA explosion in parent function

## Test Files

The exploration was done in these files (can be deleted after review):
- `explore_sroa.jl` - Basic annotation testing
- `explore_sroa2.jl` - Circuit builder pattern
- `explore_sroa3.jl` - Understanding noinline advantage
- `explore_sroa4.jl` - Large struct with realistic stamp pattern
- `explore_sroa5.jl` - Keyword constructor pattern
- `explore_sroa6.jl` - MNAContext vs DirectStampContext paths
