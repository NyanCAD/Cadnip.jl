#==============================================================================#
# Model Registry: Extensible SPICE Device Type to VA Model Mapping
#
# This module provides a dispatch-based system for mapping SPICE device types
# (nmos, pmos, npn, etc.) with level/version parameters to actual model types.
#
# Usage:
#   model = getmodel(:nmos, 14)           # Returns bsim4 type or nothing
#   params = getparams(:nmos, 14)         # Returns (TYPE=1,) for NMOS
#   device = model(; params..., W=1e-6)   # Instantiate with merged params
#
# Registration (by VA model packages):
#   import CedarSim.ModelRegistry: getmodel, getparams
#   getmodel(::Val{:nmos}, ::Val{14}, ::Nothing, ::Type{<:AbstractSimulator}) = bsim4
#   getparams(::Val{:nmos}, ::Val{14}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=1,)
#
#==============================================================================#

module ModelRegistry

export getmodel, getparams, AbstractSimulator

"""
    AbstractSimulator

Base type for simulator-specific model selection.

Packages can define subtypes (e.g., `struct VACASK <: AbstractSimulator end`)
and register simulator-specific model overrides.
"""
abstract type AbstractSimulator end

#==============================================================================#
# Core Dispatch Functions
#==============================================================================#

"""
    getmodel(device_type::Val, level::Union{Val,Nothing}, version::Union{Val,Nothing}, sim::Type) -> Type or nothing

Get the model type for a SPICE device type with given level and version.

Returns the model type (constructor) or `nothing` if no model is registered.

# Dispatch hierarchy (most specific wins):
1. `getmodel(Val{:nmos}, Val{14}, Val{Symbol("4.8")}, Type{MySimulator})` - exact match
2. `getmodel(Val{:nmos}, Val{14}, Val{Symbol("4.8")}, Type{<:AbstractSimulator})` - any sim
3. `getmodel(Val{:nmos}, Val{14}, Nothing, Type{<:AbstractSimulator})` - any version
4. `getmodel(Val{:nmos}, Nothing, Nothing, Type{<:AbstractSimulator})` - default level
5. `getmodel(Val, Any, Any, Type)` - fallback (returns nothing)

# Examples
```julia
# Get BSIM4 model for NMOS level 14
model = getmodel(:nmos, 14)
model !== nothing && (dev = model(; TYPE=1, W=1e-6))

# With version constraint
model = getmodel(:nmos, 17, "107")  # BSIMCMG version 107
```
"""
getmodel(::Val, ::Any, ::Any, ::Type) = nothing

"""
    getparams(device_type::Val, level::Union{Val,Nothing}, version::Union{Val,Nothing}, sim::Type) -> NamedTuple

Get the default type parameters (e.g., polarity) for a device type mapping.

Returns a NamedTuple with parameters like `(TYPE=1,)` for NMOS or `(TYPE=-1,)` for PMOS.
These are typically merged with user parameters when instantiating the device.

# Examples
```julia
params = getparams(:nmos, 14)  # Returns (TYPE=1,)
params = getparams(:pmos, 14)  # Returns (TYPE=-1,)

# Usage pattern:
model = getmodel(:nmos, 14)
params = getparams(:nmos, 14)
device = model(; params..., W=1e-6, L=100e-9)
```
"""
getparams(::Val, ::Any, ::Any, ::Type) = (;)

#==============================================================================#
# Convenience API (Symbol/Int -> Val conversion)
#==============================================================================#

"""
    getmodel(type::Symbol, level=nothing, version=nothing, sim=AbstractSimulator) -> Type or nothing

Convenience method that converts arguments to `Val` types for dispatch.

# Arguments
- `type::Symbol`: Device type (`:nmos`, `:pmos`, `:npn`, `:r`, `:d`, etc.)
- `level::Union{Int,Nothing}`: SPICE model level (14 for BSIM4, 49 for BSIM3, etc.)
- `version::Union{String,Nothing}`: Model version string ("107", "4.8.2", etc.)
- `sim::Type{<:AbstractSimulator}`: Simulator type for simulator-specific overrides
"""
function getmodel(type::Symbol, level::Union{Int,Nothing}=nothing,
                  version::Union{String,Nothing}=nothing,
                  ::Type{S}=AbstractSimulator) where {S<:AbstractSimulator}
    getmodel(Val(type),
             level === nothing ? nothing : Val(level),
             version === nothing ? nothing : Val(Symbol(version)),
             S)
end

"""
    getparams(type::Symbol, level=nothing, version=nothing, sim=AbstractSimulator) -> NamedTuple

Convenience method that converts arguments to `Val` types for dispatch.
"""
function getparams(type::Symbol, level::Union{Int,Nothing}=nothing,
                   version::Union{String,Nothing}=nothing,
                   ::Type{S}=AbstractSimulator) where {S<:AbstractSimulator}
    getparams(Val(type),
              level === nothing ? nothing : Val(level),
              version === nothing ? nothing : Val(Symbol(version)),
              S)
end

end # module ModelRegistry
