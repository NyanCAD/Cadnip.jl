# Interactive exploration of a transient simulation.
#
# `Cadnip.explore(circuit, tspan)` opens a Makie window with one log slider per
# scalar parameter; dragging a slider re-runs the transient and updates every
# node-voltage trace live. Loading GLMakie activates the CadnipMakieExt.
using GLMakie
using Cadnip

# Third-order Butterworth low-pass (ω_c = 1), driven by a sine source. The
# builder reads its element values from `params`, so every one of them becomes a
# slider in the explorer.
function butterworth(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    p = merge((L1=3/2, C2=4/3, L3=1/2, R4=1.0, freq=1.0, amp=1.0), params)

    ctx === nothing ? (ctx = Cadnip.MNA.MNAContext()) : Cadnip.MNA.reset_for_restamping!(ctx)
    get = n -> Cadnip.MNA.get_node!(ctx, n)
    vin, n1, vout = get(:vin), get(:n1), get(:vout)

    ω = 2π * p.freq
    Cadnip.MNA.stamp!(Cadnip.MNA.VoltageSource(p.amp * sin(ω * t); name=:V1), ctx, vin, 0)
    Cadnip.MNA.stamp!(Cadnip.MNA.Inductor(p.L1), ctx, vin, n1)
    Cadnip.MNA.stamp!(Cadnip.MNA.Capacitor(p.C2), ctx, n1, 0)
    Cadnip.MNA.stamp!(Cadnip.MNA.Inductor(p.L3), ctx, n1, vout)
    Cadnip.MNA.stamp!(Cadnip.MNA.Resistor(p.R4), ctx, vout, 0)
    return ctx
end

circuit = MNACircuit(butterworth; L1=3/2, C2=4/3, L3=1/2, R4=1.0, freq=1.0, amp=1.0)

fig = Cadnip.explore(circuit, (0.0, 100.0))
display(fig)
