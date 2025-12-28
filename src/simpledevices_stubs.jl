# Stub types for DAECompiler-only devices (simpledevices.jl removed)
# These types are referenced by spectre_env.jl and netlist_utils.jl but
# only work with DAECompiler. They error if actually called.
# TODO: Port remaining functionality to MNA or remove these stubs entirely.

export SimpleResistor, SimpleCapacitor, SimpleInductor, VoltageSource, CurrentSource, Gnd
export @subckt, SubCircuit

abstract type CircuitElement end
abstract type SPICEDevice <: CircuitElement end

struct SubCircuit{T} <: CircuitElement
    ckt::T
end
(X::SubCircuit)(args...; kwargs...) = error("SubCircuit requires DAECompiler - use MNA circuit builders instead")

macro subckt(args...)
    error("@subckt macro requires DAECompiler - use MNA circuit builders instead")
end

struct SimpleResistor <: SPICEDevice
    r::DefaultOr{Float64}
    rsh::Float64
    w::Float64
    l::Float64
    narrow::Float64
    short::Float64
end
SimpleResistor(r) = SimpleResistor(;r)
SimpleResistor(;r=mkdefault(1), rsh=50, wdef=1e-6, w=wdef, l=wdef, narrow=0, short=0, kwargs...) =
    SimpleResistor(r, Float64(undefault(rsh)), Float64(undefault(w)), Float64(undefault(l)), Float64(undefault(narrow)), Float64(undefault(short)))
(R::SimpleResistor)(args...; kwargs...) = error("SimpleResistor requires DAECompiler - use MNA.Resistor with stamp! instead")

struct SimpleCapacitor{T} <: SPICEDevice
    capacitance::T
end
SimpleCapacitor(;c=1, kwargs...) = SimpleCapacitor(undefault(c))
(C::SimpleCapacitor)(args...; kwargs...) = error("SimpleCapacitor requires DAECompiler - use MNA.Capacitor with stamp! instead")

struct SimpleInductor{T} <: SPICEDevice
    inductance::T
end
SimpleInductor(;l, kwargs...) = SimpleInductor(undefault(l))
(L::SimpleInductor)(args...; kwargs...) = error("SimpleInductor requires DAECompiler - use MNA.Inductor with stamp! instead")

# VoltageSource for DAECompiler path (different from MNA.VoltageSource)
struct VoltageSource{T} <: SPICEDevice
    dc::T
    tran::T
    ac::Complex{T}
end
function VoltageSource(;type=nothing, dc=nothing, ac=0.0+0.0im, tran=nothing)
    dc = something(dc, tran, 0.0)
    tran = something(tran, dc)
    dc, tran, acre, acim = promote(dc, tran, reim(ac)...)
    VoltageSource(dc, tran, acre+im*acim)
end
(VS::VoltageSource)(args...; kwargs...) = error("VoltageSource (DAECompiler) requires DAECompiler - use MNA.VoltageSource with stamp! instead")

struct CurrentSource{T} <: SPICEDevice
    dc::T
    tran::T
    ac::Complex{T}
end
function CurrentSource(;type=nothing, dc=nothing, ac=0.0+0.0im, tran=nothing)
    dc = something(dc, Some(tran))
    tran = something(tran, dc)
    dc, tran, acre, acim = promote(dc, tran, reim(ac)...)
    CurrentSource(dc, tran, acre+im*acim)
end
(IS::CurrentSource)(args...; kwargs...) = error("CurrentSource (DAECompiler) requires DAECompiler - use MNA.CurrentSource with stamp! instead")

struct Gnd <: CircuitElement end
(::Gnd)(args...; kwargs...) = error("Gnd requires DAECompiler - use ground node (0) in MNA instead")

struct vcvs <: SPICEDevice
    gain::Float64
    voltage::Float64
end
vcvs(;gain=1.0, vol=nothing, value=nothing) = vcvs(undefault(gain), something(undefault(vol), undefault(value), 0.0))
(S::vcvs)(args...; kwargs...) = error("vcvs requires DAECompiler - use MNA.VCVS with stamp! instead")

struct vccs <: SPICEDevice
    gain::Float64
    current::Float64
end
vccs(;gain=1.0, cur=nothing, value=nothing) = vccs(undefault(gain), something(undefault(cur), undefault(value), 0.0))
(S::vccs)(args...; kwargs...) = error("vccs requires DAECompiler - use MNA.VCCS with stamp! instead")

struct UnimplementedDevice
    params
end
UnimplementedDevice(;kwargs...) = UnimplementedDevice(kwargs)
(::UnimplementedDevice)(args...; kwargs...) = error("Unimplemented device")

struct SimpleDiode <: SPICEDevice
    params
end
SimpleDiode(;kwargs...) = SimpleDiode(kwargs)
(::SimpleDiode)(args...; kwargs...) = error("SimpleDiode requires DAECompiler - use MNA.Diode with stamp! instead")

struct Switch <: SPICEDevice
    params
end
Switch(;kwargs...) = Switch(kwargs)
(::Switch)(args...; kwargs...) = error("Switch requires DAECompiler - not yet implemented in MNA")

# Stub for defaultscope - used by device constructors
defaultscope(dev::Symbol) = GenScope(debug_scope[], dev)
defaultscope(dev::Union{SPICEDevice, SubCircuit}) = defaultscope(Symbol("X"))
