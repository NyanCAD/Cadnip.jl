"""
    VADistillerModels

Pre-parsed and precompiled VADistiller device models for circuit simulation.

Provides basic analog device models (resistor, capacitor, inductor, diode,
BJT, JFETs, MOSFETs) as well as advanced models (BSIM3v3, BSIM4v8, VDMOS).

# Usage
```julia
using VADistillerModels
using Cadnip.MNA: MNAContext, stamp!, get_node!

ctx = MNAContext()
vcc = get_node!(ctx, :vcc)
stamp!(sp_resistor(resistance=1000.0), ctx, vcc, 0; _mna_spec_=spec, _mna_x_=Float64[])
```

# Model Registration
This package registers models with Cadnip.ModelRegistry for automatic
device type resolution. SPICE netlists can use standard model cards:

```spice
.model mymos nmos level=1 vto=0.7 kp=100e-6
M1 d g s b mymos w=10u l=1u
```

# Exported Models
- `sp_resistor`: Basic resistor
- `sp_capacitor`: Basic capacitor
- `sp_inductor`: Basic inductor
- `sp_diode`: PN junction diode
- `sp_bjt`: Bipolar junction transistor
- `sp_jfet1`, `sp_jfet2`: JFET models
- `sp_mes1`: MESFET model
- `sp_mos1` - `sp_mos9`: Level 1-9 MOSFET models
- `sp_vdmos`: Power MOSFET (5-terminal)
- `sp_bsim3v3`: Berkeley BSIM3v3 MOSFET
- `sp_bsim4v8`: Berkeley BSIM4v8 MOSFET
"""
module VADistillerModels

using Cadnip
using Cadnip: VAFile
using Cadnip.MNA: MNAContext, MNASpec, DirectStampContext, stamp!, get_node!,
                    compile_structure, create_workspace, fast_rebuild!, reset_direct_stamp!
using Cadnip.ModelRegistry: getmodel, getparams, AbstractSimulator
using NyanVerilogAParser
using PrecompileTools: @compile_workload
# Model directory
const VA_DIR = joinpath(@__DIR__, "..", "va")

# List of all model files (without .va extension)
const MODEL_NAMES = [
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "bjt",
    "jfet1",
    "jfet2",
    "mes1",
    "mos1",
    "mos2",
    "mos3",
    "mos6",
    "mos9",
    "vdmos",
    "bsim3v3",
    "bsim4v8",
]

# Export model file paths for external use
for name in MODEL_NAMES
    path_const = Symbol(name, "_va")
    @eval const $path_const = joinpath(VA_DIR, $name * ".va")
    @eval export $path_const
end

# Parse and evaluate all models at module load time
for name in MODEL_NAMES
    filepath = joinpath(VA_DIR, name * ".va")
    va = NyanVerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, Cadnip.make_mna_module(va))
end

# Export all sp_ types
export sp_resistor, sp_capacitor, sp_inductor, sp_diode, sp_bjt
export sp_jfet1, sp_jfet2, sp_mes1
export sp_mos1, sp_mos2, sp_mos3, sp_mos6, sp_mos9
export sp_vdmos, sp_bsim3v3, sp_bsim4v8

# Export module references for SPICE integration
export sp_resistor_module, sp_capacitor_module, sp_inductor_module
export sp_diode_module, sp_bjt_module
export sp_jfet1_module, sp_jfet2_module, sp_mes1_module
export sp_mos1_module, sp_mos2_module, sp_mos3_module, sp_mos6_module, sp_mos9_module
export sp_vdmos_module, sp_bsim3v3_module, sp_bsim4v8_module

#==============================================================================#
# Model Registry: Register SPICE device type mappings
#
# Maps SPICE device types (nmos, pmos, npn, pnp, etc.) with levels to the
# corresponding VADistiller model types. This enables automatic device
# resolution via .model statements in SPICE netlists.
#==============================================================================#

# MOSFET level 1 (Shichman-Hodges)
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos1
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos1
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)

# MOSFET level 2
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{2}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos2
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{2}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos2
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{2}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{2}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)

# MOSFET level 3
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{3}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos3
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{3}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos3
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{3}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{3}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)

# MOSFET level 6
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{6}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos6
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{6}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos6
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{6}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{6}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)

# MOSFET level 9
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{9}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos9
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{9}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_mos9
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{9}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{9}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)

# BSIM3v3 (levels 8, 49 in ngspice)
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{8}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bsim3v3
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{8}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bsim3v3
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{49}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bsim3v3
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{49}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bsim3v3
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{8}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{8}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=-1,)
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{49}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{49}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=-1,)

# BSIM4v8 (levels 14, 54 in ngspice)
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{14}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bsim4v8
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{14}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bsim4v8
Cadnip.ModelRegistry.getmodel(::Val{:nmos}, ::Val{54}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bsim4v8
Cadnip.ModelRegistry.getmodel(::Val{:pmos}, ::Val{54}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bsim4v8
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{14}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{14}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=-1,)
Cadnip.ModelRegistry.getparams(::Val{:nmos}, ::Val{54}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=1,)
Cadnip.ModelRegistry.getparams(::Val{:pmos}, ::Val{54}, ::Nothing, ::Type{<:AbstractSimulator}) = (TYPE=-1,)

# BJT (Gummel-Poon) - default level (1 or no level)
Cadnip.ModelRegistry.getmodel(::Val{:npn}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bjt
Cadnip.ModelRegistry.getmodel(::Val{:pnp}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bjt
Cadnip.ModelRegistry.getmodel(::Val{:npn}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bjt
Cadnip.ModelRegistry.getmodel(::Val{:pnp}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_bjt
Cadnip.ModelRegistry.getparams(::Val{:npn}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pnp}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)
Cadnip.ModelRegistry.getparams(::Val{:npn}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pnp}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)

# JFET (levels 1, 2)
Cadnip.ModelRegistry.getmodel(::Val{:njf}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = sp_jfet1
Cadnip.ModelRegistry.getmodel(::Val{:pjf}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = sp_jfet1
Cadnip.ModelRegistry.getmodel(::Val{:njf}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_jfet1
Cadnip.ModelRegistry.getmodel(::Val{:pjf}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_jfet1
Cadnip.ModelRegistry.getmodel(::Val{:njf}, ::Val{2}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_jfet2
Cadnip.ModelRegistry.getmodel(::Val{:pjf}, ::Val{2}, ::Nothing, ::Type{<:AbstractSimulator}) = sp_jfet2
Cadnip.ModelRegistry.getparams(::Val{:njf}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pjf}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)
Cadnip.ModelRegistry.getparams(::Val{:njf}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pjf}, ::Val{1}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)
Cadnip.ModelRegistry.getparams(::Val{:njf}, ::Val{2}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=1,)
Cadnip.ModelRegistry.getparams(::Val{:pjf}, ::Val{2}, ::Nothing, ::Type{<:AbstractSimulator}) = (type=-1,)

# Diode — `using VADistillerModels` overrides Cadnip's default `MNA.Diode`
# so `.model foo d <params>` in a SPICE netlist resolves to the full VA diode
# model. Tier-1 registry contribution per the API consolidation plan.
Cadnip.ModelRegistry.getmodel(::Val{:d}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = sp_diode
Cadnip.ModelRegistry.getparams(::Val{:d}, ::Nothing, ::Nothing, ::Type{<:AbstractSimulator}) = (;)

# Precompile stamp! for the exact kwarg signature Cadnip's SPICE codegen emits
# (_mna_t_, _mna_mode_, _mna_x_, _mna_spec_, _mna_instance_, _mna_h_, _mna_h_p_).
# A narrower workload doesn't match what codegen actually calls, so the
# precompiled MIs never get hit at runtime.
# Both ctx types are covered: MNAContext (structure discovery + initial Newton)
# and DirectStampContext (fast_rebuild! hot path).
@compile_workload begin
    using Cadnip.MNA: reset_for_restamping!, ZERO_VECTOR
    spec = MNASpec()

    # Exercises both ctx types × both x types (ZeroVector, Vector{Float64}).
    function precompile_device(builder, params)
        ctx = builder(params, spec, 0.0; x=ZERO_VECTOR)
        builder(params, spec, 0.0; x=Float64[], ctx=ctx)
        cs = compile_structure(builder, params, spec; ctx=ctx)
        ws = create_workspace(cs; ctx=ctx)
        reset_direct_stamp!(ws.dctx)
        fast_rebuild!(ws, zeros(cs.n), 0.0)
    end

    # stamp! call matching codegen's emitted signature (7 kwargs).
    function stamp_codegen!(dev, ctx, nodes...; t, spec, x, _mna_h_, _mna_h_p_)
        stamp!(dev, ctx, nodes...;
               _mna_t_=t, _mna_mode_=spec.mode, _mna_x_=x, _mna_spec_=spec,
               _mna_instance_=:d1, _mna_h_=_mna_h_, _mna_h_p_=_mna_h_p_)
    end

    # Generic builder factory: takes an N-tuple of node keys.
    function make_builder(device_fn, node_keys::Tuple)
        function builder(params, spec::MNASpec, t::Real=0.0;
                         x::AbstractVector=Float64[],
                         ctx::Union{MNAContext,DirectStampContext,Nothing}=nothing,
                         _mna_h_=nothing, _mna_h_p_=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            nodes = ntuple(i -> get_node!(ctx, node_keys[i]), length(node_keys))
            stamp_codegen!(device_fn(), ctx, nodes..., 0;
                           t=t, spec=spec, x=x, _mna_h_=_mna_h_, _mna_h_p_=_mna_h_p_)
            return ctx
        end
    end

    make_2term_builder(fn) = make_builder(fn, (:n1,))
    make_3term_builder(fn) = make_builder(fn, (:d, :g))
    make_4term_builder(fn) = make_builder(fn, (:d, :g, :s))
    function make_5term_builder(device_fn)
        function builder(params, spec::MNASpec, t::Real=0.0;
                         x::AbstractVector=Float64[],
                         ctx::Union{MNAContext,DirectStampContext,Nothing}=nothing,
                         _mna_h_=nothing, _mna_h_p_=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)
            s = get_node!(ctx, :s)
            tj = get_node!(ctx, :tj)
            stamp_codegen!(device_fn(), ctx, d, g, s, 0, tj;
                           t=t, spec=spec, x=x, _mna_h_=_mna_h_, _mna_h_p_=_mna_h_p_)
            return ctx
        end
    end

    # 2-terminal devices
    precompile_device(make_2term_builder(() -> sp_resistor_module.sp_resistor()), NamedTuple())
    precompile_device(make_2term_builder(() -> sp_capacitor_module.sp_capacitor()), NamedTuple())
    precompile_device(make_2term_builder(() -> sp_inductor_module.sp_inductor()), NamedTuple())
    precompile_device(make_2term_builder(() -> sp_diode_module.sp_diode()), NamedTuple())

    # 3-terminal devices (JFETs, MESFET)
    precompile_device(make_3term_builder(() -> sp_jfet1_module.sp_jfet1()), NamedTuple())
    precompile_device(make_3term_builder(() -> sp_jfet2_module.sp_jfet2()), NamedTuple())
    precompile_device(make_3term_builder(() -> sp_mes1_module.sp_mes1()), NamedTuple())

    # 4-terminal devices (BJT uses c,b,e,s but mapped to d,g,s,b positions)
    precompile_device(make_4term_builder(() -> sp_bjt_module.sp_bjt()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos1_module.sp_mos1()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos2_module.sp_mos2()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos3_module.sp_mos3()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos6_module.sp_mos6()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos9_module.sp_mos9()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_bsim3v3_module.sp_bsim3v3()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_bsim4v8_module.sp_bsim4v8()), NamedTuple())

    # 5-terminal devices (VDMOS with thermal node)
    precompile_device(make_5term_builder(() -> sp_vdmos_module.sp_vdmos()), NamedTuple())
end

end # module
