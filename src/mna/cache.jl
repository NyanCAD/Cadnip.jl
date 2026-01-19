#==============================================================================#
# Device Cache Interface
#
# VA devices can use a cache to store static computations (parameter/temperature
# dependent values) that don't need to be recomputed every Newton iteration.
#
# - make_cache(DeviceType): Create a new cache instance
# - init_device!(cache, dev, spec): Initialize cache with static values
#
# This reduces stamp! function size for large VA models (e.g., PSP103VA).
#==============================================================================#

"""
    make_cache(::Type{T}) where T
    make_cache(dev::T) where T

Create a new cache instance for device type T.

Default implementation returns `nothing` (no cache).
VA devices override this to return their specific cache struct.
"""
function make_cache end
make_cache(::Type{T}) where T = nothing
make_cache(::T) where T = nothing

"""
    init_device!(cache, dev, spec::MNASpec)

Initialize device cache with static values.

Computes parameter/temperature dependent values that don't change during
Newton iterations. Called once when parameters change.

Default implementation returns the cache unchanged.
VA devices override this to populate their specific cache.
"""
function init_device! end
init_device!(cache, dev, spec::MNASpec) = cache
init_device!(::Nothing, dev, spec::MNASpec) = nothing

export make_cache, init_device!
