# General VA Contribution Stamping with AD

## The Core Question

For a general Verilog-A contribution like:
```verilog
I(p,n) <+ complex_expression(V(a,b), V(c,d), params, ddt(...));
```

Can we stamp this into MNA matrices without AST analysis?

**Yes, using ForwardDiff to automatically extract both Jacobians and resistive/reactive separation.**

---

## The Key Insight: Use Laplace Variable s as a Dual

In the Laplace domain, `ddt(x) = s * x`. We can leverage this by representing `s`
as a ForwardDiff Dual number with `value=0` and `partial=1`:

```julia
using ForwardDiff

# The Laplace variable s as a Dual
# value = 0 (no DC contribution from s itself)
# partial = 1 (tracks reactive contributions)
const s = ForwardDiff.Dual(0.0, 1.0)

# ddt in Laplace domain is just multiplication by s
ddt(x) = s * x
```

Now when we evaluate a contribution:
```julia
result = V/R + C*ddt(V)
       = V/R + C*(s*V)
       = V/R + s*(C*V)
```

The dual arithmetic automatically separates:
- `ForwardDiff.value(result) = V/R` → **resistive part f(V)** → stamps into G
- `ForwardDiff.partials(result, 1) = C*V` → **charge q(V)** → stamps into C (via ∂q/∂V)

**No custom tagged types needed!** ForwardDiff does the bookkeeping for us.

---

## Complete Approach: Nested Duals

To get both the resistive/reactive separation AND the Jacobians, we use nested Duals:

1. **Inner Dual**: Partials for node voltages (∂/∂Vi) - gives us the Jacobian
2. **Outer Dual**: Partial for s - separates resistive from reactive

```julia
using ForwardDiff
using ForwardDiff: Dual, Tag, value, partials

# Define tags to avoid mixing up our duals
struct STag end    # Tag for the s (Laplace) dual
struct VTag end    # Tag for voltage partials

# Create the s variable with its own tag
const s = Dual{STag}(0.0, 1.0)

# ddt is multiplication by s
ddt(x) = s * x
ddt(x::Dual{STag}) = s * x  # Works with already-tagged values

#==============================================================================#
# Evaluating a Contribution
#==============================================================================#

"""
    evaluate_contribution(contrib_func, x::Vector{Float64})

Evaluate a VA contribution function and return:
- resist_val: f(V) value at operating point
- react_val: q(V) value at operating point
- ∂f/∂V: Jacobian of resistive part
- ∂q/∂V: Jacobian of reactive part (charge Jacobian)
"""
function evaluate_contribution(contrib_func, x::Vector{Float64})
    n = length(x)

    # Create voltage duals with VTag for Jacobian computation
    # Wrap each in an STag dual for resist/react separation
    function make_dual_voltage(i)
        # Inner: voltage partial (1.0 for variable i, 0.0 otherwise)
        v_partials = ntuple(j -> j == i ? 1.0 : 0.0, n)
        v_dual = Dual{VTag}(x[i], v_partials...)
        # Outer: wrap in STag (value only, s partial is 0 for voltages)
        return Dual{STag}(v_dual, zero(v_dual))
    end

    # Evaluate for each voltage partial
    results = [begin
        x_dual = [make_dual_voltage(j) for j in 1:n]
        contrib_func(x_dual...)
    end for i in 1:n]

    # Actually, simpler approach: evaluate once with all partials
    x_dual = [make_dual_voltage(i) for i in 1:n]
    result = contrib_func(x_dual...)

    # Extract components
    # result is Dual{STag}(resist_dual, react_dual)
    resist_dual = value(result)           # Dual{VTag} with f(V) and ∂f/∂V
    react_dual = partials(result, 1)      # Dual{VTag} with q(V) and ∂q/∂V

    resist_val = value(resist_dual)       # f(V₀)
    react_val = value(react_dual)         # q(V₀)

    df_dV = [partials(resist_dual, i) for i in 1:n]  # ∂f/∂Vᵢ
    dq_dV = [partials(react_dual, i) for i in 1:n]   # ∂q/∂Vᵢ

    return resist_val, react_val, df_dV, dq_dV
end
```

---

## Simplified Single-Dual Approach

For many cases, we can use a simpler approach with just the s-dual, computing
Jacobians via multiple evaluations:

```julia
using ForwardDiff: Dual, value, partials

# s as a simple Dual (no tag needed if we're careful)
const s = Dual(0.0, 1.0)
ddt(x) = s * x

"""
    stamp_current_contribution!(ctx, p, n, contrib_func, x)

Stamp a general current contribution I(p,n) <+ expr into MNA matrices.
Uses ForwardDiff with s-dual for resist/react separation.
"""
function stamp_current_contribution!(
    ctx::MNAContext,
    p::Int,           # Positive node (0 = ground)
    n::Int,           # Negative node
    contrib_func,     # (V₁, V₂, ...) -> contribution (may contain ddt)
    x::Vector{Float64}
)
    num_nodes = length(x)

    # Evaluate at operating point to get resist/react split
    result = contrib_func(x...)

    if result isa Dual
        resist_val = value(result)
        react_val = partials(result, 1)
    else
        resist_val = result
        react_val = 0.0
    end

    # Compute Jacobians via finite differencing or nested AD
    for i in 1:num_nodes
        # Create dual for voltage i
        x_dual = [j == i ? Dual(x[j], 1.0) : x[j] for j in 1:num_nodes]
        result_i = contrib_func(x_dual...)

        # result_i has structure: Dual(Dual(f, ∂f/∂Vi), Dual(q, ∂q/∂Vi))
        # or simpler if x[j] wasn't wrapped in s-dual

        if result_i isa Dual
            resist_i = value(result_i)
            react_i = partials(result_i, 1)
        else
            resist_i = result_i
            react_i = 0.0
        end

        # Extract ∂/∂Vi from resist and react
        df_dVi = resist_i isa Dual ? partials(resist_i, 1) : 0.0
        dq_dVi = react_i isa Dual ? partials(react_i, 1) : 0.0

        # Stamp into G (resistive Jacobian)
        stamp_G!(ctx, p, i,  df_dVi)
        stamp_G!(ctx, n, i, -df_dVi)

        # Stamp into C (reactive Jacobian = ∂q/∂V)
        stamp_C!(ctx, p, i,  dq_dVi)
        stamp_C!(ctx, n, i, -dq_dVi)
    end

    # Stamp residual into RHS
    stamp_b!(ctx, p, -resist_val)
    stamp_b!(ctx, n,  resist_val)
end
```

#==============================================================================#
# MNA Context and Stamping Primitives
#==============================================================================#

```julia
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
```

#==============================================================================#
# Example: How Generated VA Code Would Look
#==============================================================================#

```julia
using ForwardDiff: Dual, value, partials

# Global s variable for ddt
const s = Dual(0.0, 1.0)
ddt(x) = s * x

# For: I(p,n) <+ V(p,n)/R + C*ddt(V(p,n))
# Generated code:

struct RCDevice
    R::Float64
    C::Float64
end

# The contribution function - this is what gets generated from VA
function rc_contribution(dev::RCDevice, Vp, Vn)
    Vpn = Vp - Vn
    # V/R is resistive, C*ddt(V) becomes s*C*V
    Vpn / dev.R + dev.C * ddt(Vpn)
end

# Usage in stamping:
function stamp_rc_device!(ctx, dev::RCDevice, p, n, x)
    # Define contribution in terms of node voltages
    contrib_func = (V...) -> rc_contribution(dev, V[p], V[n])

    # Use the general stamping function
    stamp_current_contribution!(ctx, p, n, contrib_func, x)
end

# Example evaluation:
# If V[p]=1.0, V[n]=0.0, R=1000, C=1e-6:
#   result = 1.0/1000 + s * (1e-6 * 1.0)
#         = Dual(0.001, 1e-6)
#   value(result) = 0.001      → stamps into G as conductance
#   partials(result,1) = 1e-6  → stamps into C as capacitance
```

---

## Does This Handle All Cases?

### Case 1: Simple Resistor ✓
```verilog
I(p,n) <+ V(p,n)/R;
```
→ No `ddt`, so result is plain `Float64`
→ `value(result) = V/R`, `partials = 0`
→ Stamps into G only

### Case 2: Simple Capacitor ✓
```verilog
I(p,n) <+ C * ddt(V(p,n));
```
→ `ddt(V)` returns `Dual(0, V)`, then `C * Dual(0, V) = Dual(0, C*V)`
→ `value(result) = 0`, `partials = C*V`
→ Stamps into C only

### Case 3: RC Parallel ✓
```verilog
I(p,n) <+ V(p,n)/R + C*ddt(V(p,n));
```
→ Result is `Dual(V/R, C*V)`
→ `value = V/R` → stamps into G
→ `partials = C*V` → stamps into C (via ∂q/∂V = C)

### Case 4: Nonlinear Capacitor ✓
```verilog
q = C0 * V(p,n) * V(p,n);
I(p,n) <+ ddt(q);
```
→ `ddt(C0*V²)` returns `Dual(0, C0*V²)`
→ `partials = C0*V²` is the charge
→ Nested AD gives ∂q/∂V = 2*C0*V → stamps into C

### Case 5: Diode ✓
```verilog
I(p,n) <+ Is * (exp(V(p,n)/Vt) - 1);
```
→ No `ddt`, result is plain `Float64` (nonlinear in V)
→ Nested AD gives ∂I/∂V = Is/Vt * exp(V/Vt) → stamps into G
→ Residual stamps into b for Newton iteration

### Case 6: Voltage Source ⚠️
```verilog
V(p,n) <+ Vdc;
```
→ **Needs current variable** - voltage is constrained, current unknown
→ Detect at codegen time (LHS is V not I)
→ Use separate `stamp_voltage_contribution!` that allocates current var

### Case 7: VCCS ✓
```verilog
I(out_p, out_n) <+ gm * V(in_p, in_n);
```
→ When computing Jacobian w.r.t. V_inp: ∂I/∂V_inp = gm
→ When computing Jacobian w.r.t. V_inn: ∂I/∂V_inn = -gm
→ Stamps off-diagonal entries in G

### Case 8: Inductor ⚠️
```verilog
V(p,n) <+ L * ddt(I(p,n));
```
→ **Needs current variable** - current is a state variable
→ Current appears in ddt, making it part of the state vector
→ Handle specially at codegen time

---

## Key Insight

**We don't need AST analysis or custom tagged types.** ForwardDiff does everything:

1. **s-dual for `ddt()`**: `ddt(x) = s * x` where `s = Dual(0,1)`
   - `value(result)` = resistive part f(V) → stamps into G
   - `partials(result)` = charge q(V) → stamps into C (via ∂q/∂V)

2. **Nested duals for Jacobians**: Wrap voltages in inner Dual for ∂/∂Vi
   - Automatically gives us ∂f/∂V for G matrix
   - Automatically gives us ∂q/∂V for C matrix

3. **ForwardDiff tags**: Use `Dual{STag}` and `Dual{VTag}` to keep duals separated

**Only two special cases need current variables:**
- `V(p,n) <+ ...` - voltage contributions (detect LHS)
- `V(p,n) <+ L*ddt(I)` - inductors (I is state)
