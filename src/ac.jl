using DescriptorSystems
using LinearAlgebra
using SparseArrays

export ac!, acdec, freqresp, noise!, IRODESystem

abstract type FreqSol end

"""
    ACSol

AC small-signal analysis result. Contains the descriptor state-space system
for computing frequency response at any output node.
"""
struct ACSol <: FreqSol
    dss::DescriptorStateSpace{Float64, Matrix{Float64}}
    x_dc::Vector{Float64}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
end

"""
    ACSystem

Accessor for AC analysis that provides node_xxx style access.
"""
struct ACSystem
    ac::ACSol
end

struct NodeRef
    name::Symbol
    ac::ACSol
end

function Base.getproperty(sys::ACSystem, name::Symbol)
    name === :ac && return getfield(sys, :ac)
    ac = getfield(sys, :ac)
    name_str = string(name)
    if startswith(name_str, "node_")
        return NodeRef(Symbol(name_str[6:end]), ac)
    end
    name in ac.node_names && return NodeRef(name, ac)
    error("Unknown property: $name")
end

# IRODESystem alias for compatibility
IRODESystem(ac::ACSol) = ACSystem(ac)

"""
    ac!(circuit::MNA.MNACircuit; abstol=1e-10, reltol=1e-6)

Construct an `ACSol` object for the given circuit, allowing solution of
AC analysis for observables within the circuit.
"""
function ac!(circuit::MNA.MNACircuit; abstol=1e-10, reltol=1e-6, kwargs...)
    # DC solve to get operating point
    dc_sol = MNA.solve_dc(circuit)
    x_dc = dc_sol.x
    n = length(x_dc)
    n == 0 && error("Circuit has no nodes")

    # Build at DC operating point with AC mode
    ac_spec = MNA.MNASpec(temp=circuit.spec.temp, mode=:ac, time=0.0)
    ctx = MNA.build_with_detection(circuit)
    MNA.reset_for_restamping!(ctx)
    circuit.builder(circuit.params, ac_spec, 0.0; x=x_dc, ctx=ctx)
    sys = MNA.assemble!(ctx)

    G = Matrix(sys.G)
    C = Matrix(sys.C)

    # Build AC excitation vector from tracked AC sources
    b_ac = zeros(n)
    for (idx, ac_val) in zip(ctx.ac_I, ctx.ac_V)
        resolved = MNA.resolve_index(ctx, idx)
        if 1 <= resolved <= n
            b_ac[resolved] += real(ac_val)  # Use real part for magnitude
        end
    end

    # DSS: E*dx/dt = A*x + B*u  =>  E=C, A=-G, B=b_ac
    C_out = Matrix{Float64}(I, n, n)
    dsys = dss(-G, C, reshape(b_ac, n, 1), C_out, zeros(n, 1))

    return ACSol(dsys, x_dc, sys.node_names, sys.current_names, sys.n_nodes)
end

"""
    freqresp(ac::ACSol, ref::NodeRef, ωs::Vector{Float64})

Calculate the frequency response of the given AC Solution for `ref` across the
frequencies listed in `ωs`. Returns a `Vector{ComplexF64}`.
"""
function DescriptorSystems.freqresp(ac::ACSol, ref::NodeRef, ωs::Vector{Float64})
    idx = findfirst(==(ref.name), ac.node_names)
    if idx === nothing
        (ref.name in (:gnd, Symbol("0"))) && return zeros(ComplexF64, length(ωs))
        error("Unknown node: $(ref.name)")
    end
    fr = DescriptorSystems.freqresp(ac.dss, ωs)
    return vec(fr[idx, 1, :])
end

"""
    getindex(ac::ACSol, ref::NodeRef)

Get descriptor state-space subsystem for a specific output node.
"""
function Base.getindex(ac::ACSol, ref::NodeRef)
    idx = findfirst(==(ref.name), ac.node_names)
    idx === nothing && error("Unknown node: $(ref.name)")
    A, E, B, C, D = dssdata(ac.dss)
    return dss(A, E, B, C[idx:idx, :], D[idx:idx, :])
end

Base.getindex(ac::ACSol, name::Symbol) = ac[NodeRef(name, ac)]

"""
    acdec(nd, fstart, fstop)

Generate a logarithmically spaced frequency vector from `fstart` to `fstop`
with `nd` points per decade. Equivalent to the SPICE command:

    .ac dec nd fstart fstop

Return value is a vector in hertz per second.
"""
function acdec(nd, fstart, fstop)
    fstart = log10(fstart)
    fstop = log10(fstop)
    points = Int(ceil((fstop-fstart)*nd))+1
    return exp10.(range(fstart, stop=fstop, length=points))
end

"""
    noise!(circuit; kwargs...)

Noise analysis - not yet implemented in MNA backend.
"""
function noise!(circuit; kwargs...)
    error("Noise analysis is not yet implemented in the MNA backend.")
end
