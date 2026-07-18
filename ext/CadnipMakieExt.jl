module CadnipMakieExt

using Makie
using Cadnip
using Cadnip: tran!
using Cadnip.MNA: MNACircuit, alter, assemble!
using StructArrays

# Collect the numeric leaf parameters of a (possibly nested) params NamedTuple as
# `dotted.path => value` pairs. The dotted paths are exactly the selectors that
# `alter(circuit; var"a.b.c" = ...)` understands, so a slider can drive any
# scalar parameter, however deeply nested.
function _numeric_params(nt::NamedTuple, prefix::String="")
    out = Pair{Symbol,Float64}[]
    for (k, v) in pairs(nt)
        path = isempty(prefix) ? string(k) : string(prefix, ".", k)
        if v isa NamedTuple
            append!(out, _numeric_params(v, path))
        elseif v isa Real
            push!(out, Symbol(path) => Float64(v))
        end
    end
    return out
end

"""
    explore(circuit::MNACircuit, tspan; solver=nothing, kwargs...) -> Figure

Interactively explore a transient simulation of `circuit`. Opens a Makie figure
with one logarithmic slider per scalar circuit parameter; moving a slider
re-runs `tran!(circuit, tspan)` with the altered parameter and updates every
node-voltage trace live.

Requires a Makie backend to be loaded (e.g. `using GLMakie` for an interactive
window, or `using CairoMakie` to render/save a static frame). `solver` and any
extra `kwargs` are forwarded to `tran!`.

```julia
using GLMakie
circuit = MNACircuit(build_rc; R=1e3, C=1e-6)
explore(circuit, (0.0, 5e-3))
```
"""
function Cadnip.explore(circuit::MNACircuit, tspan::Tuple{<:Real,<:Real};
                        solver=nothing, kwargs...)
    params = _numeric_params(circuit.params)
    isempty(params) &&
        error("explore: circuit has no scalar parameters to sweep")

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="t (s)", ylabel="V (V)")

    sg = SliderGrid(fig[1, 2],
        ((label = string(name),
          range = (10 .^ (-3:0.05:3)) .* value,
          format = "{:.2e}",
          startvalue = value) for (name, value) in params)...,
        width = 350,
        tellheight = false)

    # Circuit structure (and therefore the node set) is constant across parameter
    # changes, so the names only need to be resolved once.
    node_names = assemble!(circuit).node_names
    names = Tuple(first(p) for p in params)

    slider_values = [s.value for s in sg.sliders]
    solution = lift(slider_values...) do values...
        altered = alter(circuit; NamedTuple{names}(values)...)
        return tran!(altered, tspan; solver, kwargs...)
    end

    for name in node_names
        (name === :gnd || name === Symbol("0")) && continue
        trace = lift(solution) do sol
            StructArray{Point2f}((Float64.(sol.t), Float64.(sol[name])))
        end
        lines!(ax, trace; label = string(name))
    end

    axislegend(ax)
    return fig
end

end # module
