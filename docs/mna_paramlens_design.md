# ParamLens Integration for MNA Backend

## Current Architecture

### DAECompiler Approach
```julia
# Circuit is a function taking a lens
function my_circuit(🔍::AbstractParamLens)
    (;R, L) = 🔍(R=1000.0, L=1e-6)  # Lens resolves params
    # Build circuit with R, L
end

# Usage with ParamSim
sim = ParamSim(my_circuit, R=500.0)  # R is overridable, L is constant
sys = CircuitIRODESystem(sim)        # Compiles with DAECompiler
sol = solve(sys, ...)
```

**Key insight**: `ParamLens{NT}` encodes which params are overridden in NT's type:
- `🔍(R=1000.0)` where R NOT in NT → returns constant `1000.0` → **compile-time folding**
- `🔍(R=1000.0)` where R IS in NT → returns `lens.nt.R` → **runtime value (sweepable)**

### Current MNA Approach
```julia
# Build circuit object
circuit = MNACircuit()
nmos = bsimcmg(circuit, :d, :g, :s, :b; L=1e-6, W=2e-6, ...)
stamp!(nmos, circuit)

# Solve with DiffEq
prob = transient_problem(circuit, tspan)
sol = solve(prob, Rodas4(); p=nothing)  # p unused for device params
```

**Problem**: All parameters baked into device struct at construction time.
No way to sweep parameters without rebuilding circuit.

## Proposed Design

### Goal
Enable parameter sweeps without recompilation while maintaining constant folding
for non-swept parameters. Use the same ParamLens/ParamSim infrastructure.

### Architecture

```julia
# 1. Circuit as function taking lens
function my_mna_circuit(🔍::AbstractParamLens)
    circuit = MNACircuit()

    # Device instantiation with lens
    nmos_lens = getproperty(🔍, :nmos1)

    # Params resolved: constants for non-overridden, lens refs for overridden
    nmos = mna_bsimcmg(circuit, :d, :g, :s, :b, nmos_lens;
        L=1e-6, W=2e-6, NFIN=10.0)  # Defaults
    stamp!(nmos, circuit)

    return circuit
end

# 2. ParamSim integration
sim = ParamSim(my_mna_circuit, var"nmos1.L"=2e-8)  # Sweep L
prob = mna_problem(sim, tspan)  # Creates ODEProblem with lens values in p
sol = solve(prob, ...)

# 3. Sweep without recompilation
for L_val in [1e-8, 2e-8, 3e-8]
    new_sim = ParamSim(sim; var"nmos1.L"=L_val)  # Same lens type, new value
    prob = remake(prob, p=extract_params(new_sim))
    sol = solve(prob, ...)
end
```

### Implementation Details

#### 1. Device Parameter Resolution

VA device needs to handle lens-based parameters:

```julia
# Generated VA device with lens support
function mna_bsimcmg(circuit, d, g, s, b, 🔍::AbstractParamLens; defaults...)
    # Resolve parameters via lens
    resolved = 🔍(; defaults...)

    # Split into: constant params (IdentityLens) vs sweepable (ValLens)
    # This is determined by the lens TYPE, known at compile time

    # Create device with resolved params
    params = bsimcmgParams(; resolved...)
    device = bsimcmg(circuit, d, g, s, b, params)

    return device
end
```

#### 2. Parameter Storage

Two options for sweepable parameters:

**Option A: Store in device, update via p**
```julia
struct bsimcmg <: MNADevice
    _params::bsimcmgParams  # Immutable, includes sweepable
    _p_indices::NamedTuple  # Maps param names to p vector indices
end

# Residual function extracts sweepable params from p
function _mna_residual(dev, x, p, circuit)
    L = haskey(dev._p_indices, :L) ? p[dev._p_indices.L] : dev._params.L
    # ...
end
```

**Option B: Closure captures lens** (simpler)
```julia
# Residual closure captures lens values
function stamp!(dev, circuit, 🔍::AbstractParamLens)
    # Closure captures lens - Julia specializes on lens type
    push!(circuit.nonlinear_elements, (x, p, circ) -> begin
        # For sweepable params, get from p
        # For constant params, use captured default
        _mna_residual(dev, x, extract_params(🔍, p), circ)
    end)
end
```

#### 3. p Vector Structure

The `p` parameter in DiffEq contains only the sweepable parameter values:

```julia
struct MNAParams{NT}
    lens::ParamLens{NT}  # Type encodes structure
    values::NT           # Current values
end

# Extract sweepable params to p vector for DiffEq
function to_p_vector(params::MNAParams)
    # Flatten nested named tuple to Vector{Float64}
    return collect(flatten_params(params.values))
end

# Reconstruct from p vector
function from_p_vector(::Type{MNAParams{NT}}, p::Vector{Float64}) where NT
    # Unflatten to named tuple
    values = unflatten_params(NT, p)
    return MNAParams(ParamLens(values), values)
end
```

### Key Benefits

1. **Same ParamSim/ParamLens infrastructure** as DAECompiler path
2. **Constant folding** for non-swept parameters (via Julia specialization on lens type)
3. **Fast parameter sweeps** - just change p vector values
4. **Compatible with DiffEq** - parameters in standard p argument

### Migration Path

1. Add `🔍::AbstractParamLens` argument to MNA circuit functions
2. Modify VA codegen to accept lens for parameter resolution
3. Update `transient_problem` to use MNAParams as p
4. Ensure residual functions access sweepable params from p

### Example: Complete Flow

```julia
# Define circuit with lens
function inverter_circuit(🔍::AbstractParamLens)
    circuit = MNACircuit()

    # NMOS with sweepable L, constant W
    nmos = mna_bsimcmg(circuit, :out, :in, :gnd, :gnd,
        getproperty(🔍, :nmos);
        L=1e-6, W=2e-6, DEVTYPE=1.0)
    stamp!(nmos, circuit, getproperty(🔍, :nmos))

    # VDD source
    stamp_voltage_source!(circuit, :vdd, :gnd, 1.0)

    return circuit
end

# Create simulation with L sweepable
sim = ParamSim(inverter_circuit; var"nmos.L"=2e-8)

# Solve
prob = mna_transient_problem(sim, (0.0, 1e-9))
sol = solve(prob, Rodas4())

# Sweep L without recompilation
for L in [1e-8, 2e-8, 3e-8]
    new_prob = remake(prob, p=[L])  # Just update p
    sol = solve(new_prob, Rodas4())
end
```
