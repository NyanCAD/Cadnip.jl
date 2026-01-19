# Phase 1: Two-Function Code Splitting Plan

## Goal

Split the monolithic `stamp!` function into two functions:
1. **`init_device!`** - Computes parameter/temperature-dependent values (called when params change)
2. **`stamp!`** - Computes voltage-dependent values and stamps matrices (called every Newton iteration)

This reduces the size of code that LLVM must compile in the hot path.

## Current Structure

```
stamp!(dev, ctx, nodes...; t, mode, x, spec, instance)
├── Parameter extraction (lines 2004)
├── Local variable initialization (lines 1400-1461)
├── Internal node allocation (lines 1741-1782)
├── GMIN stamping (lines 1783-1795)
├── Voltage extraction from x (lines 1797-1815)
├── Dual creation (lines 1826-1839)
├── Contribution evaluation (lines 2043)
├── Jacobian extraction (lines 1571-1605)
└── Matrix stamping (lines 1614-1736)
```

## Proposed Structure

```
init_device!(cache, dev, spec) -> cache
├── Parameter extraction
├── Temperature-dependent calculations
├── "Static" local variable computations
└── Store results in cache struct

stamp!(cache, ctx, nodes...; t, mode, x, instance)
├── Internal node allocation (must stay here - needs ctx)
├── GMIN stamping
├── Voltage extraction from x
├── Dual creation
├── "Dynamic" contribution evaluation (using cached values)
├── Jacobian extraction
└── Matrix stamping
```

## Implementation Plan

### Step 1: Add Simple Taint Tracking to Codegen

**File: `src/vasim.jl`**

Add a simple tracking mechanism during code generation:

```julia
mutable struct CodegenTaint
    # Variables that are "static" (don't depend on voltages)
    static_vars::Set{Symbol}
    # Variables that are "dynamic" (depend on voltages)
    dynamic_vars::Set{Symbol}
    # Statements that are static (can go in init)
    static_stmts::Vector{Expr}
    # Statements that are dynamic (must stay in stamp)
    dynamic_stmts::Vector{Expr}
end

function CodegenTaint()
    CodegenTaint(Set{Symbol}(), Set{Symbol}(), Expr[], Expr[])
end
```

**Taint Rules (simple, no AST analysis needed):**

```julia
function is_static(taint::CodegenTaint, expr)
    # Base cases
    if expr isa Number || expr isa String
        return true
    end
    if expr isa Symbol
        # Known dynamic symbols
        if expr in taint.dynamic_vars
            return false
        end
        # Known static symbols (params, temp)
        if expr in taint.static_vars
            return true
        end
        # Unknown - assume dynamic to be safe
        return false
    end
    if expr isa Expr
        if expr.head == :call
            fname = expr.args[1]
            # V() and I() are always dynamic
            if fname in (:V, :I)
                return false
            end
            # va_ddt is dynamic (involves time/state)
            if fname == :va_ddt
                return false
            end
            # Other calls: static if all args are static
            return all(is_static(taint, arg) for arg in expr.args[2:end])
        end
        # Other expressions: static if all subexpressions are static
        return all(is_static(taint, arg) for arg in expr.args)
    end
    return false
end
```

### Step 2: Create Cache Struct Generator

**File: `src/vasim.jl`**

Generate a cache struct for each VA device:

```julia
function generate_cache_struct(device_name::Symbol, static_vars::Dict{Symbol, Type})
    fields = [:($(name)::$(T)) for (name, T) in static_vars]

    quote
        mutable struct $(Symbol(device_name, "Cache"))
            $(fields...)
            _initialized::Bool

            function $(Symbol(device_name, "Cache"))()
                new($([:zero($T) for T in values(static_vars)]...), false)
            end
        end
    end
end
```

### Step 3: Generate init_device! Function

**Pseudo-code for init_device! generation:**

```julia
function generate_init_device(device_name, static_stmts, static_vars, param_extraction)
    cache_type = Symbol(device_name, "Cache")

    quote
        function init_device!(cache::$cache_type, dev::$device_name, spec::MNASpec)
            if cache._initialized
                return cache
            end

            # Extract parameters
            $param_extraction

            # Temperature
            temp = spec.temp + 273.15

            # Compute static values
            $(static_stmts...)

            # Store in cache
            $([:( cache.$var = $var ) for var in keys(static_vars)]...)

            cache._initialized = true
            return cache
        end
    end
end
```

### Step 4: Modify stamp! to Use Cache

**Changes to `generate_mna_stamp_method_nterm`:**

1. Add `cache` parameter
2. Remove static computations (they're in init_device!)
3. Load cached values at start
4. Keep all voltage-dependent code

```julia
function generate_mna_stamp_method_nterm(...; use_cache=true)
    if use_cache
        # Generate stamp! that uses cache
        quote
            Base.@noinline function CedarSim.MNA.stamp!(
                    cache::$(cache_type),
                    dev::$device_name,
                    ctx::CedarSim.MNA.AnyMNAContext,
                    $(_node_args...);
                    _mna_t_::Real=0.0,
                    # ... other kwargs
                )
                # Load cached values
                $([:( $var = cache.$var ) for var in cached_var_names]...)

                # Dynamic code only
                $dynamic_stmts

                return nothing
            end
        end
    else
        # Generate original monolithic stamp! (fallback)
        # ... existing code ...
    end
end
```

### Step 5: Update MNACircuit to Manage Caches

**File: `src/mna/circuit.jl`**

```julia
struct MNACircuit{T, CacheT}
    # ... existing fields ...
    device_caches::CacheT  # Tuple of device caches
end

function MNACircuit(builder, params)
    # ... existing code ...

    # Initialize caches for VA devices
    caches = map(devices) do dev
        if hasmethod(init_device!, (typeof(dev),))
            cache = make_cache(dev)
            init_device!(cache, dev, spec)
            cache
        else
            nothing
        end
    end

    MNACircuit(..., caches)
end
```

### Step 6: Identify Static vs Dynamic Code

**Modification to `mna_collect_contributions!`:**

During contribution collection, track which expressions are static:

```julia
function mna_collect_contributions!(to_julia::MNAScope, analog_stmts, taint::CodegenTaint)
    # Mark params as static
    for p in to_julia.param_names
        push!(taint.static_vars, p)
    end
    push!(taint.static_vars, :temp)
    push!(taint.static_vars, :_mna_spec_)

    for stmt in analog_stmts
        if stmt isa VerilogAST.Assignment
            lhs = Symbol(stmt.lhs.item)
            rhs_expr = translate_expr(to_julia, stmt.rhs)

            if is_static(taint, rhs_expr)
                push!(taint.static_vars, lhs)
                push!(taint.static_stmts, :($lhs = $rhs_expr))
            else
                push!(taint.dynamic_vars, lhs)
                push!(taint.dynamic_stmts, :($lhs = $rhs_expr))
            end
        elseif stmt isa VerilogAST.Contribution
            # Contributions are always dynamic
            push!(taint.dynamic_stmts, translate_contribution(to_julia, stmt))
        end
        # ... handle other statement types
    end
end
```

## Concrete Implementation Steps

### Phase 1a: Minimal Viable Split (1-2 days)

1. **Add `CodegenTaint` struct** to `vasim.jl`
2. **Implement `is_static()`** function with simple rules
3. **Modify `mna_collect_contributions!`** to populate taint
4. **Generate cache struct** for each device
5. **Generate `init_device!`** with static code
6. **Modify `stamp!`** to accept and use cache
7. **Test with simple VA model** (VAResistor)

### Phase 1b: Integration (1 day)

8. **Update `MNACircuit`** to manage caches
9. **Add `alter()` support** - reinitialize cache when params change
10. **Test with medium VA model** (diode, BJT)

### Phase 1c: Large Model Testing (1 day)

11. **Test with PSP103VA** - verify LLVM doesn't crash
12. **Benchmark** - compare compile time and runtime
13. **Profile** - identify any performance regressions

## Example: Before and After

### Before (monolithic):

```julia
function stamp!(dev::VAResistor, ctx, p, n; x, spec, ...)
    # Static
    R = undefault(dev.R)
    temp = spec.temp + 273.15
    Vt = temp * 8.617e-5
    R_temp = R * (1 + 0.001 * (temp - 300))

    # Dynamic
    V_p = x[p]
    V_n = x[n]
    V = Dual(V_p - V_n, 1.0, -1.0)
    I = V / R_temp
    stamp_G!(ctx, p, p, partials(I, 1))
    # ...
end
```

### After (split):

```julia
mutable struct VAResistorCache
    R::Float64
    R_temp::Float64
    _initialized::Bool
end

function init_device!(cache::VAResistorCache, dev::VAResistor, spec)
    R = undefault(dev.R)
    temp = spec.temp + 273.15
    Vt = temp * 8.617e-5
    cache.R = R
    cache.R_temp = R * (1 + 0.001 * (temp - 300))
    cache._initialized = true
    return cache
end

function stamp!(cache::VAResistorCache, dev::VAResistor, ctx, p, n; x, ...)
    R_temp = cache.R_temp  # Load from cache

    # Dynamic only
    V_p = x[p]
    V_n = x[n]
    V = Dual(V_p - V_n, 1.0, -1.0)
    I = V / R_temp
    stamp_G!(ctx, p, p, partials(I, 1))
    # ...
end
```

## Expected Benefits

| Metric | Before | After (estimated) |
|--------|--------|-------------------|
| stamp! LLVM IR size (PSP103) | ~96K statements | ~60K statements |
| stamp! compile time | Crashes or 30+ min | <5 min |
| Runtime per Newton iteration | Baseline | ~Same (maybe 5% faster) |
| init_device! compile time | N/A | <30s |
| init_device! runtime | N/A | <1ms (called once) |

## Risks and Mitigations

### Risk 1: Incorrect Static/Dynamic Classification

**Mitigation:** Conservative approach - if unsure, classify as dynamic. The worst case is slightly less optimization, not incorrect results.

### Risk 2: Cache Invalidation

**Mitigation:** Clear `_initialized` flag in `alter()`. User must call `init_device!` after parameter changes.

### Risk 3: Increased Memory Usage

**Mitigation:** Cache struct is small (one Float64 per static variable). For PSP103 with ~200 params, this is ~1.6KB per device instance.

### Risk 4: API Compatibility

**Mitigation:** Keep the original monolithic `stamp!` signature as a fallback. The cached version is opt-in.

## Files to Modify

| File | Changes |
|------|---------|
| `src/vasim.jl` | Add taint tracking, cache struct generation, split codegen |
| `src/mna/circuit.jl` | Add cache management to MNACircuit |
| `src/mna/contrib.jl` | Minor: ensure stamp functions work with cache |
| `test/mna/va.jl` | Add tests for split functions |
| `test/mna/vadistiller_integration.jl` | Test large models with split |

## Success Criteria

1. PSP103VA compiles without LLVM crash
2. PSP103VA compiles in <5 minutes (vs crash or 30+ min before)
3. All existing tests pass
4. No runtime performance regression >10%
5. Simulation results match within numerical tolerance
