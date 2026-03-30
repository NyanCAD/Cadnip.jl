#==============================================================================#
# Laplace Transfer Function Support
#
# Runtime helpers for Verilog-A laplace_nd() and laplace_zp() operators.
# Converts transfer function specifications to descriptor state-space form
# (A, E, B, C, D) for stamping into MNA matrices.
#
# Uses DescriptorSystems.jl for numerically robust conversion and gprescale
# for balancing systems with extreme dynamic range (e.g., optical filters
# with coefficients spanning 70+ orders of magnitude).
#==============================================================================#

using DescriptorSystems: dss, gprescale, dssdata
using DescriptorSystems: RationalTransferFunction
using Polynomials: Polynomial

export va_laplace_nd_dss, va_laplace_zp_dss

"""
    va_laplace_nd_dss(num, den) -> (A, E, B, C, D)

Convert `laplace_nd` polynomial coefficients to a prescaled descriptor
state-space realization.

Coefficients are in ascending power order (Verilog-A convention):
  num = (n₀, n₁, ...) → N(s) = n₀ + n₁s + n₂s² + ...
  den = (d₀, d₁, ...) → D(s) = d₀ + d₁s + d₂s² + ...

`Polynomial()` takes ascending order, matching VA's `{d₀, d₁, ...}` directly.
"""
function va_laplace_nd_dss(num, den)
    n = Polynomial(float.(collect(num)))
    d = Polynomial(float.(collect(den)))
    sys = dss(RationalTransferFunction(n, d, 0))
    sysbal, _, _ = gprescale(sys)
    return dssdata(sysbal)
end

"""
    va_laplace_zp_dss(zeros, poles, gain) -> (A, E, B, C, D)

Convert `laplace_zp` zero/pole specification to a prescaled descriptor
state-space realization.

Zeros and poles are given as pairs `{magnitude, phase, ...}` in the
Verilog-A convention, where each complex value is `mag * exp(j*phase)`.
"""
function va_laplace_zp_dss(zeros, poles, gain)
    z = Complex{Float64}[zeros[i] * exp(im * zeros[i+1]) for i in 1:2:length(zeros)]
    p = Complex{Float64}[poles[i] * exp(im * poles[i+1]) for i in 1:2:length(poles)]
    # Build polynomial from roots: N(s) = gain * ∏(s - zᵢ), D(s) = ∏(s - pᵢ)
    n = isempty(z) ? Polynomial([Float64(gain)]) : Float64(gain) * fromroots(z)
    d = isempty(p) ? Polynomial([1.0]) : fromroots(p)
    sys = dss(RationalTransferFunction(n, d, 0))
    sysbal, _, _ = gprescale(sys)
    return dssdata(sysbal)
end
