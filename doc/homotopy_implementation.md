# Homotopy Methods Implementation Guide

## Problem

Ring oscillator DC solve fails because at x=0, MOSFETs are in cutoff and the Jacobian is nearly singular (κ = 3.6e11).

## Solution: GMIN Stepping

**Add to MNASpec:**
```julia
Base.@kwdef struct MNASpec{T<:Real}
    # ... existing fields ...
    gshunt::Float64 = 0.0      # Target diagonal shunt (usually 0)
    diagGmin::Float64 = 0.0    # Current homotopy value (0.01 → gshunt)
end
```

**Implement in dcop.jl:**
```julia
function dc_with_gmin_stepping(circuit; gmin_start=0.01, gmin_target=1e-12)
    gmin = gmin_start
    u = zeros(n)

    while gmin >= gmin_target
        # Add diagGmin to G matrix diagonal before solve
        G_reg = G + gmin * I
        u, converged = newton_solve(G_reg, b, u)

        if !converged
            return u, false
        end
        gmin /= 10  # Geometric reduction
    end
    return u, true
end
```

**Where to add diagGmin** (like ngspice's `LoadGmin`):
```julia
# In fast_rebuild! or before LU factorization
for i in 1:n
    G[i,i] += spec.diagGmin
end
```

## Fallback: Source Stepping

If GMIN stepping fails, scale voltage sources:
```julia
function dc_with_source_stepping(circuit; srcfact_steps=[0.0, 0.1, 0.25, 0.5, 1.0])
    for srcfact in srcfact_steps
        spec = MNASpec(mode=:dcop, srcfact=srcfact)
        # Rebuild circuit with scaled sources
        u, converged = solve_dc(circuit, spec)
        if !converged
            return u, false
        end
    end
    return u, true
end
```

## Runtime Limiting (for transient)

**Implement `limexp` in va_env.jl:**
```julia
limexp(x) = x < 80.0 ? exp(x) : exp(80.0) * (1.0 + x - 80.0)
```

**Implement `$limit` in vasim.jl** - the VA models already use it, just need to make it functional instead of pass-through.

## Priority Order

1. **GMIN stepping** - fixes the singular Jacobian problem directly
2. **Source stepping** - fallback for circuits where GMIN stepping fails
3. **limexp/pnjlim** - prevents overflow during transient, already in VA models

## Key Insight

The ring oscillator's condition number drops from 3.6e11 to ~100 when adding diagGmin=0.01. The problem isn't the source values—it's that cutoff MOSFETs provide no conductance.
