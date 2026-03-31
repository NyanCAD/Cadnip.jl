using Cadnip
using Cadnip.MNA
using Cadnip.MNA: MNAContext, get_node!, stamp!, MNACircuit, voltage
using Cadnip.MNA: VoltageSource, Resistor
using PhotonicModels
using OrdinaryDiffEq
using CairoMakie

function make_optical_port!(ctx, name)
    nodes = [get_node!(ctx, Symbol(name, "_", i)) for i in 0:3]
    for n in nodes
        stamp!(Resistor(1.0), ctx, n, 0)
    end
    return nodes
end

function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    if ctx === nothing; ctx = MNAContext()
    else Cadnip.MNA.reset_for_restamping!(ctx) end
    inp = make_optical_port!(ctx, :inp)
    outp = make_optical_port!(ctx, :outp)
    stamp!(VoltageSource(1.0; name=:Vopt), ctx, inp[1], 0, t, spec.mode)
    stamp!(TunableFilter(), ctx, inp..., outp...;
           _mna_x_=x, _mna_t_=t, _mna_spec_=spec, _mna_instance_=Symbol(""))
    return ctx
end

println("Building and solving TunableFilter transient (wavelength=1551nm)...")
circ = MNACircuit(circuit)
sol = tran!(circ, (0.0, 50e-12))

node_names = sol.prob.p.structure.node_names
out0_idx = findfirst(==(:outp_0), node_names)
out1_idx = findfirst(==(:outp_1), node_names)

ts = range(0, 50e-12, length=1000)
out0 = [sol(t)[out0_idx] for t in ts]
out1 = [sol(t)[out1_idx] for t in ts]
mag = sqrt.(out0.^2 .+ out1.^2)

fig = Figure(size=(900, 700))

ax1 = Axis(fig[1, 1],
    title="TunableFilter Transient Output (λ=1551nm, Δf≈125GHz)",
    xlabel="Time [ps]",
    ylabel="OptE [V]")
lines!(ax1, ts .* 1e12, out0, label="Re(out[0])")
lines!(ax1, ts .* 1e12, out1, label="Im(out[1])")
axislegend(ax1)

ax2 = Axis(fig[2, 1],
    title="Output Magnitude",
    xlabel="Time [ps]",
    ylabel="|OptE| [V]")
lines!(ax2, ts .* 1e12, mag, color=:black)

save("tunablefilter_transient.png", fig)
println("Saved tunablefilter_transient.png")
