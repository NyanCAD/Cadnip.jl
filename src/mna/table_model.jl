#==============================================================================#
# $table_model support (LRM 9.21)
#
# Thin wrapper over Interpolations.jl. Called once per unique (file, column,
# interp_modes, extrap_code) at VA baremodule eval time — the resulting
# interpolator is held as a module-level `const` and invoked by the stamp
# function via a plain call.
#
# Per-dim interp codes:
#   '1' -> Gridded(Linear())         (LRM linear)
#   'D' -> Gridded(Constant())       (LRM discrete / nearest)
#
# Whole-interpolator extrap codes:
#   'L' -> Line()   (linear extrapolation — LRM "L")
#   'C' -> Flat()   (constant hold      — LRM "C")
#   'E' -> Throw()  (error on OOR       — LRM "E")
#
# Higher-order interp ('2' quadratic, '3' cubic) and per-dim heterogeneous
# extrap codes are follow-ups.
#==============================================================================#

using Interpolations: interpolate, extrapolate, Gridded, Linear, Constant, Line, Flat, Throw

export va_table_model_build

function _tm_interp_mode(c::Char)
    c == '1' ? Gridded(Linear()) :
    c == 'D' ? Gridded(Constant()) :
    error("\$table_model: unsupported interp code '$c' (v1 supports '1' linear, 'D' discrete)")
end

function _tm_extrap_bc(c::Char)
    c == 'L' ? Line() :
    c == 'C' ? Flat() :
    c == 'E' ? Throw() :
    error("\$table_model: unsupported extrap code '$c' (v1 supports 'L' linear, 'C' constant, 'E' error)")
end

"""
    va_table_model_build(axes::NTuple{D,Vector{Float64}},
                         ys::Array{Float64,D},
                         interp_modes::NTuple{D,Char},
                         extrap_code::Char) -> callable

Construct the interpolator for a single dependent column of a `\$table_model`
table. Called at VA baremodule eval time; the returned object is stored as a
module-level `const` and invoked as `itp(x1, x2, …, xD)` from the stamp body.
"""
@inline function va_table_model_build(
    axes::NTuple{D,Vector{Float64}},
    ys::AbstractArray{Float64,D},
    interp_modes::NTuple{D,Char},
    extrap_code::Char,
) where {D}
    modes = ntuple(i -> _tm_interp_mode(interp_modes[i]), D)
    itp = interpolate(axes, ys, modes)
    extrapolate(itp, _tm_extrap_bc(extrap_code))
end
