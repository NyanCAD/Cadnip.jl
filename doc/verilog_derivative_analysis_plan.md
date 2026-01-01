# Verilog-A Derivative Analysis Plan

## Executive Summary

This document outlines a plan for implementing advanced Verilog-A derivative analysis in SpiceArmyKnife.jl. The goals are:

1. **Analytical derivatives**: Extract symbolic/analytical derivatives from VA expressions where possible
2. **Reactive vs resistive classification**: Statically classify contributions as resistive (f) or reactive (q)
3. **Dependent capacitor rewriting**: Detect and transform voltage-dependent charges to proper charge formulation

## Current State Analysis

### SpiceArmyKnife.jl Architecture

The current implementation uses an elegant **s-dual approach** (see `src/mna/contrib.jl`):

```julia
# s represents the Laplace variable (s = jω)
# ddt(x) = s * x in Laplace domain
va_ddt(x) = Dual{ContributionTag}(zero(x), x)
```

**Strengths:**
- Automatic resistive/reactive separation via ForwardDiff
- No AST walking required for basic cases
- Jacobians computed automatically
- Works well for simple VA models

**Limitations:**
- No static classification (can't determine at compile time if a contribution is purely resistive)
- idt() (integration) not supported
- ddx() implemented but limited
- No charge reformulation detection
- All devices go through AD even when analytical derivatives exist

### OpenVAF Architecture (Reference)

OpenVAF uses a sophisticated multi-pass approach:

1. **HIR (High-level IR)**: Preserves VA semantics, explicit contribution tracking
2. **MIR (Middle IR)**: SSA-based, enables dataflow analysis
3. **Linearization Pass**: Decides if ddt can be linearized vs needs implicit equation
4. **Automatic Differentiation**: Forward-mode AD on MIR for Jacobians
5. **Small-Signal Analysis**: Detects values that are always zero at DC

Key insights from OpenVAF:
- **Explicit classification**: `ContributeKind` enum marks `is_reactive: bool` at lowering time
- **Linearization decision**: `Evaluation::{Equation, Linear, Dead}` based on dependency analysis
- **Callback-based operators**: `ddt`, `idt`, `ddx` lowered as callbacks, replaced during AD

---

## Proposed Architecture

### Phase 1: Static Analysis Infrastructure

Create an AST analysis pass that runs before code generation to classify contributions.

#### 1.1 Contribution Classification

```julia
@enum ContributionKind begin
    RESISTIVE     # No ddt(), idt(), or time-dependent operators
    REACTIVE      # Contains ddt() but linear in ddt (can be linearized)
    DYNAMIC       # Contains ddt() used nonlinearly (needs implicit equation)
    INTEGRATING   # Contains idt() (needs state variable)
end

struct ContributionInfo
    kind::ContributionKind
    branch::Tuple{Symbol,Symbol}  # (p, n) or (internal, external)
    is_voltage_src::Bool
    depends_on::Set{Symbol}       # Parameters/nodes this depends on
    ddt_exprs::Vector{Any}        # Expressions wrapped by ddt()
    charge_expr::Union{Nothing,Any}  # q(V) if linearizable
end
```

#### 1.2 Analysis Pass

```julia
function analyze_contribution(contrib_stmt::ContributionStatement)
    # 1. Determine if current (I) or voltage (V) contribution
    # 2. Walk expression tree looking for ddt(), idt(), ddx()
    # 3. Classify based on how these operators are used:
    #    - ddt(linear_in_V) → REACTIVE (linearizable)
    #    - f(ddt(x)) where f is nonlinear → DYNAMIC
    #    - idt(...) → INTEGRATING
    # 4. Extract charge expression for reactive contributions
end
```

### Phase 2: Charge Formulation Detection

Detect when a current contribution is actually a capacitor that should use charge formulation.

#### 2.1 Pattern Recognition

The following patterns indicate a capacitor contribution that should be reformulated:

```verilog
// Pattern 1: Direct capacitive current
I(p,n) <+ C * ddt(V(p,n));
// → q = C * V, stamps into C matrix

// Pattern 2: Voltage-dependent capacitance
I(p,n) <+ C(V) * ddt(V(p,n));
// → MUST use charge formulation: q = ∫C(V)dV

// Pattern 3: Nonlinear charge
Q = f(V(p,n));  // e.g., junction charge
I(p,n) <+ ddt(Q);
// → Already in charge form, extract Q and use ∂Q/∂V

// Pattern 4: Multi-port charge (MOSFET)
Qg = f(Vgs, Vds, Vbs);
I(g,s) <+ ddt(Qg);
// → Multi-terminal charge stamping
```

#### 2.2 Charge Extraction Algorithm

```julia
function extract_charge_expression(contrib_expr)
    # Look for ddt() at the top level or as a linear factor
    if is_ddt_call(contrib_expr)
        # I <+ ddt(Q) → charge is Q
        return contrib_expr.arg, nothing
    elseif is_mul_with_ddt(contrib_expr)
        # I <+ factor * ddt(V) → charge is factor * V
        # BUT: if factor depends on V, need integral form
        factor, ddt_arg = extract_factor_and_ddt(contrib_expr)
        if depends_on_voltage(factor, ddt_arg)
            # C(V)*ddt(V) case - needs integral
            return :needs_integral, (factor, ddt_arg)
        else
            # constant * ddt(V) - simple case
            return :($(factor) * $(ddt_arg)), nothing
        end
    else
        # Mixed or complex - use runtime AD
        return nothing, contrib_expr
    end
end
```

### Phase 3: Analytical Derivative Database

Build a database of analytical derivatives for common functions, using ChainRulesCore-style rules.

#### 3.1 Derivative Rules

```julia
# Define rules similar to ChainRulesCore.frule
struct DerivativeRule{F}
    f::F
    df::Function  # Derivative function
end

const DERIVATIVE_RULES = Dict{Symbol, DerivativeRule}(
    :exp => DerivativeRule(exp, (x, dx) -> exp(x) * dx),
    :log => DerivativeRule(log, (x, dx) -> dx / x),
    :sqrt => DerivativeRule(sqrt, (x, dx) -> dx / (2 * sqrt(x))),
    :sin => DerivativeRule(sin, (x, dx) -> cos(x) * dx),
    :cos => DerivativeRule(cos, (x, dx) -> -sin(x) * dx),
    :tanh => DerivativeRule(tanh, (x, dx) -> (1 - tanh(x)^2) * dx),
    :sinh => DerivativeRule(sinh, (x, dx) -> cosh(x) * dx),
    :cosh => DerivativeRule(cosh, (x, dx) -> sinh(x) * dx),
    :atan => DerivativeRule(atan, (x, dx) -> dx / (1 + x^2)),
    # ... etc
)
```

#### 3.2 Expression Differentiation

```julia
function differentiate_expr(expr, wrt::Symbol)
    # Symbolic differentiation using rules
    if expr isa Symbol
        return expr == wrt ? 1.0 : 0.0
    elseif expr isa Number
        return 0.0
    elseif expr isa Expr
        if expr.head == :call
            f = expr.args[1]
            args = expr.args[2:end]
            if haskey(DERIVATIVE_RULES, f)
                # Apply chain rule
                rule = DERIVATIVE_RULES[f]
                return apply_chain_rule(rule, args, wrt)
            end
        elseif expr.head == :+ || expr.head == :call && expr.args[1] == :+
            # Sum rule
            return sum_differentiate(expr.args[2:end], wrt)
        elseif expr.head == :* || expr.head == :call && expr.args[1] == :*
            # Product rule
            return product_differentiate(expr.args[2:end], wrt)
        end
    end
    # Fallback: generate AD call
    return :(ForwardDiff.derivative($expr, $wrt))
end
```

### Phase 4: Integration with Symbolics.jl

For complex expressions, leverage Symbolics.jl for symbolic differentiation.

#### 4.1 Hybrid Approach

```julia
using Symbolics

function symbolic_jacobian(contrib_fn, nodes::Vector{Symbol})
    # Create symbolic variables for node voltages
    @variables V[1:length(nodes)]

    # Evaluate contribution symbolically
    I_symbolic = contrib_fn(V...)

    # Compute symbolic Jacobian
    J = Symbolics.jacobian([I_symbolic], V)

    # Attempt simplification
    J_simplified = Symbolics.simplify.(J)

    # Generate Julia code
    J_fn = Symbolics.build_function(J_simplified, V...)

    return J_fn
end
```

#### 4.2 When to Use Symbolic vs AD

| Case | Approach |
|------|----------|
| Simple polynomial | Symbolic (compile-time) |
| Transcendental (exp, log, etc.) | Symbolic or cached AD |
| Complex nested expressions | AD at runtime |
| Device model with many params | AD with sparsity detection |
| ddx() explicit derivative | Symbolic if possible |

### Phase 5: Reactive/Resistive Separation at Compile Time

#### 5.1 Static Classification

Instead of runtime s-dual, determine at compile time:

```julia
function classify_contribution_static(contrib_ast)
    has_ddt = contains_call(contrib_ast, :ddt)
    has_idt = contains_call(contrib_ast, :idt)
    has_time = contains_symbol(contrib_ast, :t)

    if !has_ddt && !has_idt && !has_time
        return :static_resistive
    elseif has_idt
        return :integrating  # Needs state variable
    elseif is_linear_in_ddt(contrib_ast)
        return :static_reactive
    else
        return :dynamic  # Runtime determination
    end
end
```

#### 5.2 Code Generation Strategy

```julia
function codegen_contribution(info::ContributionInfo)
    if info.kind == RESISTIVE
        # Pure resistive: only stamp G matrix
        return quote
            stamp_resistive!(ctx, $p, $n, $I_fn, x)
        end
    elseif info.kind == REACTIVE && info.charge_expr !== nothing
        # Known charge: stamp C matrix with analytical derivative
        q_fn = info.charge_expr
        C_fn = differentiate(q_fn, :V)
        return quote
            stamp_capacitive!(ctx, $p, $n, $C_fn, x)
        end
    else
        # Mixed or complex: use runtime AD
        return quote
            stamp_current_contribution!(ctx, $p, $n, $contrib_fn, x)
        end
    end
end
```

### Phase 6: Implicit Equation Generation

For `idt()` and nonlinear `ddt()` usage, generate implicit equations like OpenVAF.

#### 6.1 idt() Support

```julia
struct ImplicitEquation
    kind::Symbol  # :ddt, :idt, :ddt_idt
    equation_idx::Int
    associated_charge::Any
end

function lower_idt(expr, ic=nothing)
    # idt(f(t)) requires state variable
    # Let x be state, then dx/dt = f(t)
    eq_idx = alloc_implicit_equation!(ctx)
    return quote
        # x is implicit unknown
        x = ctx.implicit_unknowns[$eq_idx]
        # Residual: dx/dt - f(t) = 0
        ctx.residuals[$eq_idx] = ctx.d_unknowns[$eq_idx] - $(expr)
        x  # Return value
    end
end
```

#### 6.2 Linearization Decision (OpenVAF-inspired)

```julia
function can_linearize_ddt(ddt_expr, uses)
    # Check if ddt result is only used in linear operations
    for use in uses
        if use.op in (:+, :-, :*)
            # Multiplication is OK if other operand is OP-independent
            if use.op == :* && depends_on_operating_point(use.other_arg)
                return false
            end
        else
            # Nonlinear operation → can't linearize
            return false
        end
    end
    return true
end
```

---

## Implementation Phases

### Phase A: Analysis Infrastructure (2-3 weeks effort)

1. **Create `src/vasim/analysis.jl`**
   - AST walker for VA contribution analysis
   - ContributionInfo structure
   - ddt/idt/ddx detection

2. **Add tests in `test/mna/va_analysis.jl`**
   - Test classification of simple resistor, capacitor, diode
   - Test detection of voltage-dependent capacitance
   - Test multi-port charge detection

### Phase B: Charge Formulation (2-3 weeks effort)

1. **Extend `src/mna/contrib.jl`**
   - `stamp_charge_contribution!` already exists
   - Add charge extraction from ddt() expressions
   - Handle integral form for C(V)*ddt(V) patterns

2. **Add `src/vasim/charge_rewrite.jl`**
   - Transform `I <+ C(V)*ddt(V)` to charge form
   - Compute Q(V) = ∫C(V)dV when possible
   - Fall back to numerical integration if needed

### Phase C: Symbolic Differentiation (3-4 weeks effort)

1. **Create `src/vasim/symbolic.jl`**
   - Derivative rule database
   - Expression differentiation
   - Jacobian generation

2. **Integrate Symbolics.jl (optional)**
   - Use for complex expressions
   - Cache compiled derivative functions
   - Benchmark vs ForwardDiff

### Phase D: Static Classification (2-3 weeks effort)

1. **Modify `src/vasim.jl` codegen**
   - Classify contributions at codegen time
   - Generate specialized stamp! calls
   - Skip AD for static cases

2. **Update `src/spc/codegen.jl`**
   - Same classification for SPICE behavioral sources
   - Consistent handling across frontends

### Phase E: idt() Support (3-4 weeks effort)

1. **Create `src/mna/implicit.jl`**
   - ImplicitEquation structure
   - State variable allocation
   - Residual generation

2. **Extend DAE solver integration**
   - Handle implicit unknowns in IDA/DifferentialEquations.jl
   - Initial condition handling
   - Convergence for idt()

---

## Data Structures

### ContributionAST

```julia
struct ContributionAST
    lhs::Symbol          # :I or :V
    branch::BranchRef    # (node1, node2) or named branch
    rhs::Expr            # Expression tree
    location::SourceLoc  # For error messages
end
```

### AnalyzedContribution

```julia
struct AnalyzedContribution
    original::ContributionAST
    kind::ContributionKind

    # Resistive part (f in: I = f + dq/dt)
    resist_expr::Union{Expr, Nothing}
    resist_deps::Set{Symbol}  # Nodes it depends on

    # Reactive part (q in: I = f + dq/dt)
    charge_expr::Union{Expr, Nothing}
    charge_deps::Set{Symbol}
    needs_integral::Bool  # C(V)*ddt(V) pattern

    # Jacobian expressions (if analytically computable)
    jacobian_resist::Dict{Symbol, Expr}  # ∂f/∂Vi
    jacobian_charge::Dict{Symbol, Expr}  # ∂q/∂Vi = Cij

    # Implicit equations needed
    implicit_eqs::Vector{ImplicitEquation}
end
```

### DeviceAnalysis

```julia
struct DeviceAnalysis
    module_name::Symbol
    terminals::Vector{Symbol}
    internal_nodes::Vector{Symbol}
    parameters::Dict{Symbol, Any}

    contributions::Vector{AnalyzedContribution}

    # Summarized matrix structure
    G_sparsity::SparsityPattern
    C_sparsity::SparsityPattern
    has_nonlinear_G::Bool
    has_nonlinear_C::Bool
    has_implicit_eqs::Bool
end
```

---

## Integration with Julia AD Ecosystem

### ChainRulesCore Compatibility

Define rrules for VA operators to work with Zygote/ReverseDiff:

```julia
using ChainRulesCore

function ChainRulesCore.rrule(::typeof(va_ddt), x)
    y = va_ddt(x)
    function va_ddt_pullback(ȳ)
        # In Laplace domain: ddt(x) = s*x
        # Reverse mode: ∂L/∂x = s * ∂L/∂y (where s is imaginary)
        return NoTangent(), ȳ  # Simplified for transient
    end
    return y, va_ddt_pullback
end
```

### ForwardDiff Custom Tags

Already implemented with ContributionTag and JacobianTag. Can extend for:

```julia
struct SensitivityTag end  # For parameter sensitivity
struct HessianTag end      # For second derivatives

# Ordering: SensitivityTag ≺ JacobianTag ≺ ContributionTag
```

---

## Performance Considerations

### Compile-Time Optimization

1. **Constant folding**: Device parameters known at compile time should fold
2. **Sparsity exploitation**: Pre-compute Jacobian sparsity pattern
3. **Inlining**: Small contribution functions should inline

### Runtime Optimization

1. **Cached derivatives**: Reuse Jacobian evaluations across Newton iterations
2. **SIMD for sweeps**: Vectorize across parameter sweep points
3. **GPU batching**: For large sweeps, batch evaluations on GPU

### Benchmarks to Add

```julia
# test/benchmarks/va_derivative.jl

# 1. Simple resistor (should be ~free with constant folding)
# 2. Diode (exp derivative)
# 3. BSIM4 Cgg (complex charge)
# 4. PSP103 (many internal nodes)
```

---

## References

### OpenVAF Source Files

| File | Relevance |
|------|-----------|
| `openvaf/hir_lower/src/lib.rs` | HirInterner, ImplicitEquationKind |
| `openvaf/hir_lower/src/expr.rs` | ddt, idt, ddx lowering |
| `openvaf/mir_autodiff/src/` | Automatic differentiation on MIR |
| `openvaf/sim_back/src/topology/lineralize.rs` | Linearization decisions |

### Julia Packages

- [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) - Custom derivative rules
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) - Symbolic differentiation
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) - Current AD infrastructure

### Papers

- "Best Practices for Compact Modeling in Verilog-A" (MOS-AK)
- "Derivative Management in Verilog-A" (TU Dresden Workshop 2019)

---

## Success Criteria

1. **Correctness**: All existing VA tests pass
2. **Performance**: No regression for simple devices
3. **Coverage**:
   - ddt() fully supported with charge extraction
   - ddx() with analytical derivatives
   - idt() basic support with state variables
4. **Classification**: Static analysis correctly identifies 90%+ of contributions
5. **Charge formulation**: Automatic detection and rewriting of C(V)*ddt(V) patterns

---

## Appendix: OpenVAF Evaluation Types

From `openvaf/sim_back/src/topology/lineralize.rs`:

```rust
pub enum Evaluation {
    // Create implicit equation (needs state variable)
    Equation,

    // Direct contribution (can be linearized)
    Linear { contributes: Box<[(Value, Value)]> },

    // Dead code (replace with zero)
    Dead,
}
```

This informs our classification:
- `Linear` → REACTIVE (stamps directly)
- `Equation` → DYNAMIC (needs implicit variable)
- `Dead` → Optimize away

---

## Appendix: Charge Formulation Mathematics

For voltage-dependent capacitance C(V):

**Naive approach** (incorrect for simulation):
```
I = C(V) * dV/dt
```

**Charge formulation** (correct):
```
Q(V) = ∫C(V')dV'  (from 0 to V)
I = dQ/dt = (dQ/dV) * dV/dt = C_eff(V) * dV/dt
```

Where `C_eff = dQ/dV` is the tangent capacitance.

**Example: Junction capacitor**
```
C(V) = Cj0 / (1 - V/φ)^m

Q(V) = Cj0 * φ * (1 - (1 - V/φ)^(1-m)) / (1-m)   [for m ≠ 1]

C_eff = dQ/dV = Cj0 / (1 - V/φ)^m = C(V)  ✓
```

The charge formulation ensures:
1. Charge conservation
2. Consistent AC and transient results
3. No spurious dissipation
