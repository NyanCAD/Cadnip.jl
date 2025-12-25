# This gets imported by all generated VerilogA code. The function names
# exported here should correspond to what is made available by the VerilogA
# standard.
baremodule VerilogAEnvironment

import ..Base
import ..CedarSim
import ForwardDiff
import Compat
import NaNMath
import ChainRules

# Phase 0: Use stubs instead of DAECompiler
if CedarSim.USE_DAECOMPILER
    import DAECompiler
    import DAECompiler: ddt
else
    import ..CedarSim.DAECompilerStubs: ddt, observed!, epsilon
    const DAECompiler = CedarSim.DAECompilerStubs
end

import Base:
    +, *, -, ==, !=, >, <,  <=, >=,
    max, min, abs,
    exp,
    sinh, cosh, tanh,
    zero, atan, floor, %, NaN

export !, +, *, -, ==, !=, /, ^, var"**", >, <,  <=, >=,
    max, min, abs,
    exp, sqrt,
    sinh, cosh, tanh,
    sin, cos, tan, atan,
    floor, %, NaN

using Base: @inbounds, @inline, @noinline

# Safe division and other operations for ForwardDiff Duals to prevent NaN
# This is critical for compact models (BSIMCMG, etc.) where charge calculations
# can have dqi/idscv → 0/0 as Vds → 0. The physically correct limit is 0.
# Use 1e-20 as threshold - small enough to not affect normal calculations,
# but large enough to catch numerical precision issues
const SAFE_DIV_EPS = 1e-20
# Maximum value to use instead of Inf (prevents NaN from Inf*0 in subsequent calculations)
const SAFE_DIV_MAX = 1e20

# For regular numbers, use normal division
@inline /(a::Number, b::Number) = Base.:/(a, b)

# For Duals, use safe division that handles 0/0 → 0 instead of NaN
@inline function /(a::ForwardDiff.Dual{T}, b::ForwardDiff.Dual{T}) where T
    va = ForwardDiff.value(a)
    vb = ForwardDiff.value(b)
    # If denominator is near zero, return 0 to avoid NaN/Inf propagation
    # This is conservative but prevents numerical issues in circuit simulation
    if Base.abs(vb) < SAFE_DIV_EPS
        return zero(a)
    end
    return Base.:/(a, b)
end

# Mixed cases - one Dual, one regular number
@inline function /(a::ForwardDiff.Dual{T}, b::Number) where T
    va = ForwardDiff.value(a)
    # If both numerator and denominator are very small, return 0
    if Base.abs(va) < SAFE_DIV_EPS && Base.abs(b) < SAFE_DIV_EPS
        return zero(a)
    end
    # If denominator is near zero but numerator isn't, avoid NaN by returning
    # a clamped large value with zero partials (prevents Inf*0=NaN later)
    if Base.abs(b) < SAFE_DIV_EPS
        sign_val = Base.sign(va) * Base.sign(b)
        sign_val = sign_val == 0 ? 1.0 : sign_val
        return ForwardDiff.Dual{T}(sign_val * SAFE_DIV_MAX, zero(ForwardDiff.partials(a)))
    end
    return Base.:/(a, b)
end

@inline function /(a::Number, b::ForwardDiff.Dual{T}) where T
    vb = ForwardDiff.value(b)
    # If both numerator and denominator are very small, return 0
    if Base.abs(a) < SAFE_DIV_EPS && Base.abs(vb) < SAFE_DIV_EPS
        return zero(b)
    end
    # If denominator value is near zero but numerator isn't, avoid NaN by returning
    # a clamped large value with zero partials (prevents Inf*0=NaN later)
    if Base.abs(vb) < SAFE_DIV_EPS
        sign_val = Base.sign(a) * Base.sign(vb)
        sign_val = sign_val == 0 ? 1.0 : sign_val
        return ForwardDiff.Dual{T}(sign_val * SAFE_DIV_MAX, zero(ForwardDiff.partials(b)))
    end
    return Base.:/(a, b)
end

# Handle division between Duals with different tags (e.g., ContributionTag / Nothing)
# This is common in VA models where va_ddt creates ContributionTag duals
@inline function /(a::ForwardDiff.Dual{T1}, b::ForwardDiff.Dual{T2}) where {T1, T2}
    va = ForwardDiff.value(a)
    vb = ForwardDiff.value(b)
    # Extract the innermost float values for comparison
    va_float = va isa ForwardDiff.Dual ? ForwardDiff.value(va) : va
    vb_float = vb isa ForwardDiff.Dual ? ForwardDiff.value(vb) : vb
    # If denominator is near zero, return 0 to avoid NaN/Inf propagation
    # This is a conservative choice for circuit simulation edge cases
    if Base.abs(vb_float) < SAFE_DIV_EPS
        return zero(a)
    end
    return Base.:/(a, b)
end
using Base.Experimental: @overlay
export @inbounds, @inline, @overlay, var"$temperature"

export pow, ln, ddt, flicker_noise, white_noise, atan2, log

# Safe functions for circuit simulation - prevent NaN from problematic inputs
# For regular (non-Dual) numbers, use NaNMath
@noinline Base.@assume_effects :total pow(a::Real, b::Real) = NaNMath.pow(a, b)
@noinline Base.@assume_effects :total ln(x::Real) = NaNMath.log(x)
@noinline Base.@assume_effects :total sqrt(x::Real) = NaNMath.sqrt(x)

# Safe sqrt for Duals: if value < 0 (numerical noise), return zero
@inline function sqrt(x::ForwardDiff.Dual{T}) where T
    v = ForwardDiff.value(x)
    # Handle negative values (numerical noise) by returning zero
    if v < 0
        return zero(x)
    end
    # If value is very small, return zero to avoid Inf in derivative (1/(2*sqrt(x)))
    if v < SAFE_DIV_EPS
        return zero(x)
    end
    return NaNMath.sqrt(x)
end

# Safe pow for Duals
@inline function pow(x::ForwardDiff.Dual{T}, p::Real) where T
    v = ForwardDiff.value(x)
    # Handle negative base with fractional power by returning zero
    if v < 0 && !Base.isinteger(p)
        return zero(x)
    end
    # Handle zero base with non-positive power
    if Base.abs(v) < SAFE_DIV_EPS
        if p <= 0
            # 0^0 = 1, 0^negative = Inf, but we return 0 to avoid NaN in partials
            return zero(x)
        elseif p < 1
            # 0^fractional has Inf derivative, return 0
            return zero(x)
        end
    end
    return NaNMath.pow(x, p)
end

# Safe ln for Duals: handle x <= 0
@inline function ln(x::ForwardDiff.Dual{T}) where T
    v = ForwardDiff.value(x)
    # Handle non-positive values by returning large negative (like -1e30)
    if v <= SAFE_DIV_EPS
        return ForwardDiff.Dual{T}(-SAFE_DIV_MAX, zero(ForwardDiff.partials(x)))
    end
    return NaNMath.log(x)
end

log(x) = cedarerror("log not supported, use $log10 or $ln instead")
!(a) = Base.:!(a)
!(a::Int64) = a == zero(a)
atan2(x,y) = Base.atan(x,y)
var"**"(a, b) = pow(a, b)
^ = Base.:(⊻)
@noinline Base.@assume_effects :total sin(x) = NaNMath.sin(x)
@noinline Base.@assume_effects :total cos(x) = NaNMath.cos(x)
@noinline Base.@assume_effects :total tan(x) = Base.tan(x)

# Chain rules for differentiation - with safe handling
function ChainRules.frule((_, Δx), ::typeof(sqrt), x)
    v = x isa ForwardDiff.Dual ? ForwardDiff.value(x) : x
    if v < SAFE_DIV_EPS
        return (zero(x), zero(Δx))
    end
    Ω = sqrt(x)
    (Ω, Δx / (2Ω))
end

function ChainRules.frule((_, Δx), ::typeof(ln), x)
    v = x isa ForwardDiff.Dual ? ForwardDiff.value(x) : x
    if v <= SAFE_DIV_EPS
        return (ln(x), zero(Δx))
    end
    (ln(x), Δx / x)
end

function ChainRules.frule((_, Δx, Δp), ::typeof(pow), x::Number, p::Number)
    vx = x isa ForwardDiff.Dual ? ForwardDiff.value(x) : x
    # Safe handling for problematic cases
    if vx < 0 && !Base.isinteger(p)
        return (zero(x), zero(Δx))
    end
    if Base.abs(vx) < SAFE_DIV_EPS && p <= 1
        return (pow(x, p), zero(Δx))
    end
    y = pow(x, p)
    _dx = ChainRules._pow_grad_x(x, p, ChainRules.float(y))
    # Check for NaN in _dx and replace with 0
    if isnan(_dx)
        _dx = zero(_dx)
    end
    if ChainRules.iszero(Δp)
        # Treat this as a strong zero, to avoid NaN, and save the cost of log
        return y, _dx * Δx
    else
        # This may do real(log(complex(...))) which matches ProjectTo in rrule
        _dp = ChainRules._pow_grad_p(x, p, ChainRules.float(y))
        if isnan(_dp)
            _dp = zero(_dp)
        end
        return y, ChainRules.muladd(_dp, Δp, _dx * Δx)
    end
end

function ChainRules.frule((_, Δx), ::typeof(cos), x::Number)
    sinx, cosx = NaNMath.sincos(x)
    return (cosx, -sinx * Δx)
end

ChainRules.@scalar_rule tan(x) 1 + Ω * Ω

# Branch voltagesa
#function V(V₊, V₋)
#    return V₊.V - V₋.V
#end
#V(node) = node.V
function white_noise(dscope, pwr, name)
    DAECompiler.observed!(pwr, CedarSim.DScope(dscope, Symbol(name, :pwr)))
    DAECompiler.epsilon(CedarSim.DScope(dscope, Symbol(name)))
end
function flicker_noise(dscope, pwr, exp, name)
    DAECompiler.observed!(pwr, CedarSim.DScope(dscope, Symbol(name, :pwr)))
    DAECompiler.observed!(exp, CedarSim.DScope(dscope, Symbol(name, :exp)))
    DAECompiler.epsilon(CedarSim.DScope(dscope, Symbol(name)))
end

# MNA-compatible stubs for noise functions (no dscope parameter)
# These return 0 since MNA doesn't do noise analysis during DC/transient
white_noise(pwr, name) = 0.0
flicker_noise(pwr, exp, name) = 0.0

vaconvert(T::Type{<:Number}, x::CedarSim.Default) = CedarSim.Default(vaconvert(T, x.val))
vaconvert(T::Type{<:Number}, x::CedarSim.DefaultOr) = CedarSim.DefaultOr(vaconvert(T, x.val), x.is_default)
vaconvert(T::Type{<:Number}, x::Integer) = Base.convert(T, x)
vaconvert(::Type{<:Number}, x::Number) = x

"""
    vaconvert(::Type{Int}, x::Real)

Implements conversion of VA `real` to `integer` types.

VA-LRM 4.2.1.1 Real to integer conversion:

    If the fractional part of the real number is exactly 0.5, it shall be
    rounded away from zero.
"""
vaconvert(::Type{Int}, x::Real) = Base.round(Int, x, Base.RoundNearestTiesAway)
vaconvert(::Type{Int}, x::Integer) = x
vaconvert(T::Type{Int}, x::CedarSim.Default) = CedarSim.Default(vaconvert(T, x.val))
vaconvert(T::Type{Int}, x::CedarSim.DefaultOr) = CedarSim.DefaultOr(vaconvert(T, x.val), x.is_default)

export var"$simparam"

var"$simparam"(param) = CedarSim.undefault(Base.getproperty(CedarSim.spec[], Symbol(param)))
function var"$simparam"(param, default)
    if Base.hasproperty(CedarSim.spec[], Symbol(param))
        return CedarSim.undefault(Base.getproperty(CedarSim.spec[], Symbol(param)))
    else
        return default
    end
end

var"$temperature"() = CedarSim.undefault(CedarSim.spec[].temp)+273.15 # Kelvin

abstract type VAModel <: CedarSim.CircuitElement; end

end
