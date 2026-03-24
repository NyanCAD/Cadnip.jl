#==============================================================================#
# DDE Delay Support for absdelay()
#
# Implements transport delay via the DDE history function h(p, t).
# h and h_p are threaded as kwargs through fast_rebuild! → builder → stamp!
# so they remain type-stable at every call site.
#==============================================================================#

export va_absdelay_V

"""
    va_absdelay_V(h, h_p, p_idx::Int, n_idx::Int, tdelay, current_val, t)

Look up delayed voltage V(p,n) at time `t - tdelay` using the DDE history function.

During DC analysis or detection (h === nothing), returns `current_val` (the current
voltage value, delay ignored per VAMS spec). During transient with DDEProblem,
uses `h(h_p, t-tdelay; idxs=...)` to interpolate past state.
"""
function va_absdelay_V(h, h_p, p_idx::Int, n_idx::Int, tdelay, current_val, t)
    if h === nothing
        return current_val
    end
    t_past = t - tdelay
    vp = p_idx > 0 ? h(h_p, t_past; idxs=p_idx) : 0.0
    vn = n_idx > 0 ? h(h_p, t_past; idxs=n_idx) : 0.0
    return vp - vn
end
