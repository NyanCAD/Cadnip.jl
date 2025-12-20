# General VA Contribution Stamping with AD

## The Core Question

For a general Verilog-A contribution like:
```verilog
I(p,n) <+ complex_expression(V(a,b), V(c,d), params, ddt(...));
```

Can we stamp this into MNA matrices without AST analysis?

**Yes, using AD to automatically extract Jacobians.**

---

## The General Approach

### Step 1: Separate Resistive and Reactive Contributions

Every Verilog-A current contribution has the form:
```
I = f(V) + ddt(q(V))
```

Where:
- `f(V)` is the resistive (DC) part → stamps into G
- `q(V)` is the charge → stamps into C (via ∂q/∂V)

We need to separate these during evaluation.

### Step 2: Use Tagged Values for ddt

```julia
# A value that carries both resistive and reactive components
struct SplitValue{T}
    resist::T    # f(V) - resistive part
    react::T     # q(V) - reactive/charge part
end

# Arithmetic propagates both components
Base.:+(a::SplitValue, b::SplitValue) = SplitValue(a.resist + b.resist, a.react + b.react)
Base.:*(a::SplitValue, b::Number) = SplitValue(a.resist * b, a.react * b)
Base.:*(a::Number, b::SplitValue) = SplitValue(a * b.resist, a * b.react)
# ... etc for other operations

# ddt moves value from resist to react
function ddt(x::SplitValue)
    # ddt(f + ddt(q)) = ddt(f) + ddt(ddt(q))
    # But ddt(ddt(q)) is not supported, so:
    SplitValue(x.react, zero(x.react))  # react becomes new resist (it will be differentiated)
end

function ddt(x::Number)
    # Plain number becomes reactive contribution
    SplitValue(zero(x), x)
end
```

Wait, this isn't quite right. Let me think more carefully...

### Better Approach: Accumulate Separately

```julia
mutable struct ContributionAccumulator{T}
    resist::T    # Resistive contribution (no ddt)
    react::T     # Reactive contribution (inside ddt)
end

# When we see: I(p,n) <+ expr
# We accumulate into resist

# When we see: I(p,n) <+ ddt(q_expr)
# We accumulate q_expr into react

# The issue: how do we detect ddt in the middle of an expression?
```

The challenge is that `ddt` can appear anywhere in an expression:
```verilog
I(p,n) <+ V(p,n)/R + C1*ddt(V(p,n)) + ddt(C2*V(p,n)*V(p,n));
```

### Solution: ddt Returns a Tagged Type

```julia
struct ReactiveValue{T}
    value::T  # The charge q - what's inside ddt()
end

# ddt(x) marks x as reactive
ddt(x) = ReactiveValue(x)

# When adding ReactiveValue to regular value, track separately
struct MixedValue{T}
    resist::T
    react::T
end

Base.:+(a::Number, b::ReactiveValue) = MixedValue(a, b.value)
Base.:+(a::ReactiveValue, b::Number) = MixedValue(b, a.value)
Base.:+(a::MixedValue, b::ReactiveValue) = MixedValue(a.resist, a.react + b.value)
Base.:+(a::MixedValue, b::Number) = MixedValue(a.resist + b, a.react)
# ... etc
```

This way, when we evaluate a contribution expression, we get a `MixedValue` that tells us:
- `resist`: the f(V) part → stamp into G
- `react`: the q(V) part → stamp into C

---

## Complete Implementation Sketch

```julia
using ForwardDiff

# Tagged types for separating resistive/reactive
struct ReactiveValue{T}
    charge::T  # q(V) - the charge
end

struct MixedContribution{T}
    resist::T  # f(V) - stamps into G
    react::T   # q(V) - stamps into C
end

# Convert plain numbers
MixedContribution(x::Number) = MixedContribution(x, zero(x))
MixedContribution(x::ReactiveValue) = MixedContribution(zero(x.charge), x.charge)

# ddt creates reactive contribution
ddt(x::Number) = ReactiveValue(x)
ddt(x::ForwardDiff.Dual) = ReactiveValue(x)  # Works with AD too

# Arithmetic on MixedContribution
function Base.:+(a::MixedContribution, b::MixedContribution)
    MixedContribution(a.resist + b.resist, a.react + b.react)
end
function Base.:+(a::MixedContribution, b::Number)
    MixedContribution(a.resist + b, a.react)
end
function Base.:+(a::Number, b::MixedContribution)
    MixedContribution(a + b.resist, b.react)
end
function Base.:+(a::MixedContribution, b::ReactiveValue)
    MixedContribution(a.resist, a.react + b.charge)
end
function Base.:+(a::ReactiveValue, b::MixedContribution)
    MixedContribution(b.resist, a.charge + b.react)
end
function Base.:+(a::Number, b::ReactiveValue)
    MixedContribution(a, b.charge)
end
function Base.:+(a::ReactiveValue, b::Number)
    MixedContribution(b, a.charge)
end
function Base.:+(a::ReactiveValue, b::ReactiveValue)
    MixedContribution(zero(a.charge), a.charge + b.charge)
end

# Multiplication
function Base.:*(a::Number, b::MixedContribution)
    MixedContribution(a * b.resist, a * b.react)
end
function Base.:*(a::MixedContribution, b::Number)
    MixedContribution(a.resist * b, a.react * b)
end
function Base.:*(a::Number, b::ReactiveValue)
    ReactiveValue(a * b.charge)
end

# Division, etc... (similar patterns)

#==============================================================================#
# MNA Context and Stamping
#==============================================================================#

mutable struct MNAContext
    n_nodes::Int
    n_currents::Int

    # COO format for sparse matrices
    G_I::Vector{Int}
    G_J::Vector{Int}
    G_V::Vector{Float64}

    C_I::Vector{Int}
    C_J::Vector{Int}
    C_V::Vector{Float64}

    b::Vector{Float64}
end

function stamp_G!(ctx::MNAContext, i::Int, j::Int, val::Float64)
    (i == 0 || j == 0) && return  # Skip ground
    push!(ctx.G_I, i)
    push!(ctx.G_J, j)
    push!(ctx.G_V, val)
end

function stamp_C!(ctx::MNAContext, i::Int, j::Int, val::Float64)
    (i == 0 || j == 0) && return
    push!(ctx.C_I, i)
    push!(ctx.C_J, j)
    push!(ctx.C_V, val)
end

function stamp_b!(ctx::MNAContext, i::Int, val::Float64)
    i == 0 && return
    ctx.b[i] += val
end

#==============================================================================#
# General Current Contribution Stamping
#==============================================================================#

"""
    stamp_current_contribution!(ctx, p, n, contrib_func, node_voltages)

Stamp a general current contribution I(p,n) <+ expr into MNA matrices.

- `contrib_func(V...)` returns the contribution value (may be MixedContribution)
- `node_voltages` are the current voltage values for linearization
"""
function stamp_current_contribution!(
    ctx::MNAContext,
    p::Int,           # Positive node index (0 = ground)
    n::Int,           # Negative node index
    contrib_func,     # Function: (V1, V2, ...) -> contribution
    x::Vector{Float64}  # Current node voltages
)
    num_nodes = length(x)

    # Evaluate with ForwardDiff to get Jacobian
    # Create dual numbers for all node voltages
    function eval_with_dual(i)
        # Create vector of Duals where only variable i has partial = 1
        x_dual = [ForwardDiff.Dual(x[j], j == i ? 1.0 : 0.0) for j in 1:num_nodes]
        return contrib_func(x_dual...)
    end

    # Get value at operating point
    result = contrib_func(x...)

    # Handle different result types
    resist_val, react_val = if result isa MixedContribution
        (result.resist, result.react)
    elseif result isa ReactiveValue
        (0.0, result.charge)
    else
        (result, 0.0)
    end

    # Compute Jacobians via AD for non-zero components
    if resist_val != 0.0 || true  # Always compute for Newton
        # Resistive Jacobian: ∂f/∂V_i
        for i in 1:num_nodes
            x_dual = [ForwardDiff.Dual(x[j], j == i ? 1.0 : 0.0) for j in 1:num_nodes]
            result_dual = contrib_func(x_dual...)

            resist_dual = if result_dual isa MixedContribution
                result_dual.resist
            elseif result_dual isa ReactiveValue
                zero(result_dual.charge)
            else
                result_dual
            end

            # Extract partial derivative
            dfdVi = ForwardDiff.partials(resist_dual, 1)

            # Stamp into G: current into p, out of n
            stamp_G!(ctx, p, i,  dfdVi)
            stamp_G!(ctx, n, i, -dfdVi)
        end
    end

    if react_val != 0.0 || true  # Always compute for dynamics
        # Reactive Jacobian: ∂q/∂V_i → goes into C matrix
        for i in 1:num_nodes
            x_dual = [ForwardDiff.Dual(x[j], j == i ? 1.0 : 0.0) for j in 1:num_nodes]
            result_dual = contrib_func(x_dual...)

            react_dual = if result_dual isa MixedContribution
                result_dual.react
            elseif result_dual isa ReactiveValue
                result_dual.charge
            else
                zero(result_dual)
            end

            # Extract partial derivative
            dqdVi = ForwardDiff.partials(react_dual, 1)

            # Stamp into C
            stamp_C!(ctx, p, i,  dqdVi)
            stamp_C!(ctx, n, i, -dqdVi)
        end
    end

    # Stamp equivalent sources for Newton iteration
    # For resistive: I_eq = f(V) - J_f * V → stamp f(V) - sum(df/dVi * Vi)
    # This is handled implicitly by the Newton formulation

    # Actually for linearization: stamp the residual
    stamp_b!(ctx, p, -resist_val)
    stamp_b!(ctx, n,  resist_val)
end

#==============================================================================#
# Example: How Generated VA Code Would Look
#==============================================================================#

# For: I(p,n) <+ V(p,n)/R + C*ddt(V(p,n))
# Generated code:

struct RCDevice
    R::Float64
    C::Float64
end

function (dev::RCDevice)(p_net, n_net)
    # This returns a contribution function
    return (V...) -> begin
        Vp = V[p_net.idx]
        Vn = V[n_net.idx]
        Vpn = Vp - Vn

        # Resistive: V/R
        # Reactive: C*V (charge, inside ddt)
        Vpn / dev.R + ddt(dev.C * Vpn)
    end
end

# Usage:
function stamp_rc_device!(ctx, dev::RCDevice, p, n, x)
    contrib_func = (V...) -> begin
        Vpn = V[p] - V[n]
        Vpn / dev.R + ddt(dev.C * Vpn)
    end
    stamp_current_contribution!(ctx, p, n, contrib_func, x)
end
```

---

## Does This Handle All Cases?

### Case 1: Simple Resistor ✓
```verilog
I(p,n) <+ V(p,n)/R;
```
→ Returns `Float64`, stamps into G only

### Case 2: Simple Capacitor ✓
```verilog
I(p,n) <+ C * ddt(V(p,n));
```
→ Returns `ReactiveValue`, stamps into C only

### Case 3: RC Parallel ✓
```verilog
I(p,n) <+ V(p,n)/R + C*ddt(V(p,n));
```
→ Returns `MixedContribution`, stamps both G and C

### Case 4: Nonlinear Capacitor ✓
```verilog
q = C0 * V(p,n) * V(p,n);
I(p,n) <+ ddt(q);
```
→ `ddt(C0*V^2)` returns `ReactiveValue` with `charge = C0*V^2`
→ AD gives ∂q/∂V = 2*C0*V → stamps into C

### Case 5: Diode ✓
```verilog
I(p,n) <+ Is * (exp(V(p,n)/Vt) - 1);
```
→ Returns `Float64` (nonlinear)
→ AD gives ∂I/∂V = Is/Vt * exp(V/Vt) → stamps into G
→ Residual stamps into b

### Case 6: Voltage Source
```verilog
V(p,n) <+ Vdc;
```
→ **This is different** - need current variable
→ Handle separately (not a current contribution)

### Case 7: VCCS ✓
```verilog
I(out_p, out_n) <+ gm * V(in_p, in_n);
```
→ AD gives ∂I/∂V_inp = gm, ∂I/∂V_inn = -gm
→ Stamps off-diagonal G entries

---

## Key Insight

**We don't need AST analysis.** The combination of:
1. Tagged types for `ddt()` to separate resistive/reactive
2. ForwardDiff for automatic Jacobian computation

...gives us everything we need to stamp any current contribution correctly.

**Voltage contributions** still need special handling (allocate current variable), but those are rare and easy to detect at codegen time (LHS is V not I).
