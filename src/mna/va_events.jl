#==============================================================================#
# VA Event Detection: Comparison Interception
#
# Verilog-A compact models switch branch equations based on voltage-dependent
# comparisons (region selection in BSIM/PSP-style models) - an "invisible"
# discontinuity the adaptive integrator can silently step over. vasim.jl
# intercepts every `>`/`<`/`>=`/`<=` comparison in analog-block scope and
# routes it through one of these va_cmp_* functions, which:
#
#   1. Always compute the signed distance `d = value(lhs) - value(rhs)` and
#      store it in a per-comparison condition slot (see breakpoints.jl,
#      context.jl, value_only.jl).
#   2. Mark the slot voltage-dependent (`_mark_condition_vdep!`) if either
#      operand is currently a `ForwardDiff.Dual` - node-derived quantities
#      are always Dual-wrapped by dual_creation, so this runtime dispatch
#      identifies genuine voltage-dependent comparisons exactly, distinct
#      from provably parameter-only ones (`if (L <= 0)`). `va_event_callback`
#      (solve.jl) only builds a VectorContinuousCallback root for the
#      voltage-dependent slots: a compact model packs in far more
#      comparisons than the physically meaningful region-selection ones
#      (parameter validation, junction-diode/capacitance-region checks), and
#      watching all of them - including ones that toggle every few
#      picoseconds - was found to disrupt the adaptive step schedule badly
#      enough to change simulated trajectories, not just slow them down.
#   3. Return the ordinary Bool result of the comparison.
#
# `ForwardDiff.value` already has fallback methods for plain reals
# (`value(x::Real) = x`) as well as `Dual`, so a single implementation
# handles both cases for the returned Bool - this makes computing `d` a pure
# side channel: since the *value* used for the Bool result and the branch
# taken are identical to what plain `>`/`<`/`>=`/`<=` would give, the
# analytic Jacobian (G/C stamps) is bit-identical whether or not va_events
# are enabled.
#
# `==`/`!=` are NOT intercepted - a "cross zero" comparator can't usefully
# rootfind exact equality (degenerate root).
#==============================================================================#

using ForwardDiff: value

"""
    _real_value(x) -> Float64

Fully unwrap `x` to a plain `Float64`, recursively stripping `ForwardDiff.Dual`
levels. `value(lhs) - value(rhs)` only strips one level, which is sufficient
for the returned Bool (comparison/subtraction on the remaining Dual level
still dispatches correctly), but condition_values storage needs a concrete
Float64 - and callers can be nested in an outer AD layer (e.g. NonlinearSolve's
own Jacobian wrapping our internal per-node JacobianTag Dual) that a single
`value()` strip doesn't remove.
"""
_real_value(x::Real) = Float64(x)
_real_value(x::ForwardDiff.Dual) = _real_value(value(x))

"""
    va_cmp_gt(ctx, base::Int, k::Int, lhs, rhs) -> Bool

Intercepted `lhs > rhs`. Stores `d = value(lhs) - value(rhs)` in condition
slot `base+k`, marks it voltage-dependent if either operand is a
`ForwardDiff.Dual`, and returns `d > 0`.
"""
@inline function va_cmp_gt(ctx, base::Int, k::Int, lhs, rhs)
    d = value(lhs) - value(rhs)
    idx = base + k
    _mark_condition_vdep!(ctx, idx, lhs, rhs)
    _store_condition!(ctx, idx, _real_value(d))
    return d > 0
end

"""
    va_cmp_lt(ctx, base::Int, k::Int, lhs, rhs) -> Bool

Intercepted `lhs < rhs`. Stores `d = value(lhs) - value(rhs)` in condition
slot `base+k`, marks it voltage-dependent if either operand is a
`ForwardDiff.Dual`, and returns `d < 0`.
"""
@inline function va_cmp_lt(ctx, base::Int, k::Int, lhs, rhs)
    d = value(lhs) - value(rhs)
    idx = base + k
    _mark_condition_vdep!(ctx, idx, lhs, rhs)
    _store_condition!(ctx, idx, _real_value(d))
    return d < 0
end

"""
    va_cmp_ge(ctx, base::Int, k::Int, lhs, rhs) -> Bool

Intercepted `lhs >= rhs`. Stores `d = value(lhs) - value(rhs)` in condition
slot `base+k`, marks it voltage-dependent if either operand is a
`ForwardDiff.Dual`, and returns `d >= 0`.
"""
@inline function va_cmp_ge(ctx, base::Int, k::Int, lhs, rhs)
    d = value(lhs) - value(rhs)
    idx = base + k
    _mark_condition_vdep!(ctx, idx, lhs, rhs)
    _store_condition!(ctx, idx, _real_value(d))
    return d >= 0
end

"""
    va_cmp_le(ctx, base::Int, k::Int, lhs, rhs) -> Bool

Intercepted `lhs <= rhs`. Stores `d = value(lhs) - value(rhs)` in condition
slot `base+k`, marks it voltage-dependent if either operand is a
`ForwardDiff.Dual`, and returns `d <= 0`.
"""
@inline function va_cmp_le(ctx, base::Int, k::Int, lhs, rhs)
    d = value(lhs) - value(rhs)
    idx = base + k
    _mark_condition_vdep!(ctx, idx, lhs, rhs)
    _store_condition!(ctx, idx, _real_value(d))
    return d <= 0
end

export va_cmp_gt, va_cmp_lt, va_cmp_ge, va_cmp_le
