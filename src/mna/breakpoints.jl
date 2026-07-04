#==============================================================================#
# Breakpoint Protocol
#
# Time-dependent sources (PWL, PULSE, SIN) can expose the exact times at which
# their waveform has a kink, edge, or onset. These are collected during
# circuit construction (MNAContext) and later expanded into solver `tstops`
# so the adaptive integrator lands exactly on them instead of discovering
# them via rejected steps.
#==============================================================================#

"""
    Wave

Abstract supertype for callable waveform structs (`PWLWave`, `SinWave`,
`PulseWave`) usable as the `tran` field of `VoltageSource`/`CurrentSource`.
"""
abstract type Wave end

export Wave

"""
    BreakpointSpec(times, period, count)

A set of breakpoint times for a single source.

- `times`: absolute times of each breakpoint's first occurrence. For a
  periodic spec the whole set repeats at `times .+ k*period` for `k >= 0`.
- `period`: `0.0` for aperiodic (times occur exactly once); otherwise the
  source repeats every `period` seconds.
- `count`: only read when `period > 0`: number of repetitions, `-1` for
  unbounded (repeat until the end of the simulation tspan).
"""
struct BreakpointSpec
    times::Vector{Float64}
    period::Float64
    count::Int
end

BreakpointSpec(times::Vector{Float64}) = BreakpointSpec(times, 0.0, -1)

export BreakpointSpec

"""
    breakpoints(wave) -> Union{BreakpointSpec, Nothing}

Return the breakpoint spec for `wave`, or `nothing` if it has none. This is
the default for anything that isn't a recognized `Wave` — including
`nothing` and plain user-supplied closures — so registering breakpoints for
an arbitrary `tran` callable is always safe and inert.
"""
breakpoints(::Any) = nothing

export breakpoints

"""
    _collect_times(xs) -> Vector{Float64}

Collect `xs` into a plain `Vector{Float64}`. Deliberately avoids
`Float64[Float64(x) for x in xs]`: StaticArrays overrides comprehension
`collect` for its container types, so that idiom returns a `SizedVector`
(not a `Vector`) when `xs isa SVector` - `BreakpointSpec` requires a plain
`Vector{Float64}`.
"""
function _collect_times(xs)::Vector{Float64}
    out = Vector{Float64}(undef, length(xs))
    @inbounds for i in eachindex(xs)
        out[i] = Float64(xs[i])
    end
    return out
end

"""
    register_breakpoints!(ctx, wave)

Push `wave`'s breakpoint spec onto `ctx.breakpoints` if it has one.

No-op on `DirectStampContext` — breakpoints are collected once during
structure discovery on `MNAContext`; the zero-allocation restamping path
never needs to touch them.
"""
function register_breakpoints! end

export register_breakpoints!
