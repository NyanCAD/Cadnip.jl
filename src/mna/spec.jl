#==============================================================================#
# MNA: Simulation Specification
#
# `MNASpec` is the simulation-level state every stamp sees — temperature, the
# analysis mode, the current time, and the `$simparam` values a Verilog-A model
# can read. It lives this early in the include order because `MNAContext`
# records the spec a build resolved (`record_spec!`), so the struct has to exist
# before context.jl defines that field.
#==============================================================================#

"""
    MNASpec{T}

Simulation specification for MNA analysis.

Contains simulation-level parameters that are separate from circuit parameters.
Passed explicitly to circuit builders (not via ScopedValue) to enable JIT optimization.

# Fields
- `temp::Float64`: Temperature in Celsius (default: 27.0)
- `mode::Symbol`: Analysis mode - `:dcop`, `:tran`, `:tranop`, `:ac` (default: :tran)
- `time::T`: Current simulation time for transient sources (default: 0.0)

The time field is parameterized to support ForwardDiff automatic differentiation
during transient analysis with Rosenbrock ODE solvers.

# Example
```julia
spec = MNASpec(temp=50.0, mode=:dcop)
spec_at_1ms = MNASpec(temp=27.0, mode=:tran, time=1e-3)
```

# Design Rationale
Unlike Cadnip's ScopedValue-based SimSpec, MNASpec is passed explicitly.
This enables full JIT optimization since Julia's closure boxing issue
prevents optimization of captured ScopedValue accesses.
"""
Base.@kwdef struct MNASpec{T<:Real}
    temp::Float64 = 27.0
    mode::Symbol = :tran
    time::T = 0.0
    # Common simulator parameters (for $simparam access)
    gmin::Float64 = 1e-12      # Device-level minimum conductance (used in device models)
    gshunt::Float64 = 0.0      # Node-to-ground shunt conductance (for stepping/floating nodes)
    srcFact::Float64 = 1.0     # Source scaling factor (for source stepping, 0→1)
    tnom::Float64 = 27.0       # Nominal temperature (Celsius)
    abstol::Float64 = 1e-12    # Absolute tolerance
    reltol::Float64 = 1e-3     # Relative tolerance
    vntol::Float64 = 1e-6      # Voltage tolerance
    iabstol::Float64 = 1e-12   # Current absolute tolerance
end

export MNASpec

"""
    MNASpec(base::MNASpec; kwargs...) -> MNASpec

Copy `base`, overriding the named fields. Everything not named carries through —
including the `time` field's type, which a Rosenbrock transient fills with a
ForwardDiff dual.

This is the one place that enumerates the fields; the `with_*` helpers below and
the netlist `.option` rebind in `codegen_mna!` all go through it, so a new field
does not have to be threaded into a handful of near-identical constructors.

```julia
MNASpec(spec; temp=85.0)              # same as with_temp(spec, 85.0)
MNASpec(spec; gmin=1e-9, tnom=25.0)   # several at once
```
"""
MNASpec(base::MNASpec; temp=base.temp, mode=base.mode, time=base.time,
        gmin=base.gmin, gshunt=base.gshunt, srcFact=base.srcFact, tnom=base.tnom,
        abstol=base.abstol, reltol=base.reltol, vntol=base.vntol, iabstol=base.iabstol) =
    MNASpec(; temp, mode, time, gmin, gshunt, srcFact, tnom, abstol, reltol, vntol, iabstol)

"""
    with_temp(spec::MNASpec, temp::Real) -> MNASpec

Create new spec with different temperature. The rest of the spec carries through.
"""
with_temp(spec::MNASpec, temp::Real) = MNASpec(spec; temp=Float64(temp))

"""
    with_mode(spec::MNASpec, mode::Symbol) -> MNASpec

Create new spec with different mode. The rest of the spec carries through.
"""
with_mode(spec::MNASpec, mode::Symbol) = MNASpec(spec; mode)

"""
    with_time(spec::MNASpec, t::Real) -> MNASpec

Create new spec with different time.
Note: time type is preserved to support ForwardDiff Dual numbers.
"""
with_time(spec::MNASpec, t::Real) = MNASpec(spec; time=t)

"""
    with_gshunt(spec::MNASpec, gshunt::Real) -> MNASpec

Create new spec with different gshunt (node-to-ground shunt conductance).
Used for GMIN stepping and floating node stabilization.
"""
with_gshunt(spec::MNASpec, gshunt::Real) = MNASpec(spec; gshunt=Float64(gshunt))

"""
    with_srcfact(spec::MNASpec, srcFact::Real) -> MNASpec

Create new spec with different srcFact (source scaling factor).
Used for source stepping homotopy: scale all sources by srcFact (0→1).
When srcFact < 1.0, the b vector is scaled by srcFact after stamping.
"""
with_srcfact(spec::MNASpec, srcFact::Real) = MNASpec(spec; srcFact=Float64(srcFact))

export with_temp, with_mode, with_time, with_gshunt, with_srcfact
