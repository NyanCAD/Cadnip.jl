"""
    PSPModels

Pre-parsed and precompiled PSP 103.4 MOSFET models for circuit simulation.

Provides the PSP (Penn State Philips) MOSFET model family from NXP Semiconductors,
including standard, self-heating, and non-quasi-static (NQS) variants.

# Usage
```julia
using PSPModels
using Cadnip.MNA: MNAContext, stamp!, get_node!

ctx = MNAContext()
d = get_node!(ctx, :d)
g = get_node!(ctx, :g)
stamp!(PSP103VA(), ctx, d, g, 0, 0; _mna_spec_=spec, _mna_x_=Float64[])
```

# Exported Models
- `JUNCAP200`: JUNCAP 200 junction diode model (2-terminal: A, K)
- `PSP103VA`: PSP 103 MOSFET (4-terminal: D, G, S, B)
- `PSP103TVA`: PSP 103 MOSFET with self-heating (5-terminal: D, G, S, B, DT)
- `PSPNQS103VA`: PSP 103 MOSFET non-quasi-static (4-terminal: D, G, S, B)
"""
module PSPModels

using Cadnip
using Cadnip: VAFile
using Cadnip.MNA: MNAContext, MNASpec, DirectStampContext, stamp!, get_node!,
                    compile_structure, create_workspace, fast_rebuild!, reset_direct_stamp!
using NyanVerilogAParser
using PrecompileTools: @compile_workload

# Model directory
const VA_DIR = joinpath(@__DIR__, "..", "va")

# List of model files (without .va extension)
# Note: Order matters - juncap200 should be loaded first as it's simpler
const MODEL_NAMES = [
    "juncap200",
    "psp103",
    "psp103t",
    "psp103_nqs",
]

# Export model file paths for external use
const juncap200_va = joinpath(VA_DIR, "juncap200.va")
const psp103_va = joinpath(VA_DIR, "psp103.va")
const psp103t_va = joinpath(VA_DIR, "psp103t.va")
const psp103_nqs_va = joinpath(VA_DIR, "psp103_nqs.va")

export juncap200_va, psp103_va, psp103t_va, psp103_nqs_va

# Parse and evaluate all models at module load time
for name in MODEL_NAMES
    filepath = joinpath(VA_DIR, name * ".va")
    va = NyanVerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, Cadnip.make_mna_module(va))
end

# Export device types (names match VA module declarations)
export JUNCAP200, PSP103VA, PSP103TVA, PSPNQS103VA

# Export module references for SPICE integration
export JUNCAP200_module, PSP103VA_module, PSP103TVA_module, PSPNQS103VA_module

# Register models with ModelRegistry for automatic resolution in SPICE
# This allows .model statements to use these types without imported_hdl_modules
import Cadnip.ModelRegistry: getmodel
getmodel(::Val{:juncap200}, ::Nothing, ::Nothing, ::Type{<:Cadnip.ModelRegistry.AbstractSimulator}) = JUNCAP200
getmodel(::Val{:psp103va}, ::Nothing, ::Nothing, ::Type{<:Cadnip.ModelRegistry.AbstractSimulator}) = PSP103VA
getmodel(::Val{:psp103tva}, ::Nothing, ::Nothing, ::Type{<:Cadnip.ModelRegistry.AbstractSimulator}) = PSP103TVA
getmodel(::Val{:pspnqs103va}, ::Nothing, ::Nothing, ::Type{<:Cadnip.ModelRegistry.AbstractSimulator}) = PSPNQS103VA

# Precompile stamp! methods for the exact kwarg signature Cadnip's SPICE codegen
# emits: _mna_t_, _mna_mode_, _mna_x_, _mna_spec_, _mna_instance_, _mna_h_, _mna_h_p_.
# A narrower workload (just _mna_spec_ + _mna_x_) doesn't match what the codegen
# actually calls, so the precompiled MIs never get hit. See
# doc/psp103_noinline_investigation.md.
# Both ctx types are covered: MNAContext (used during structure discovery and
# solve_dc's initial Newton) and DirectStampContext (fast_rebuild! hot path).
# Note: PSPNQS103VA skipped - requires idt() function not yet supported
# Note: PSP103TVA skipped - requires ln_1p_d() function not yet supported
@compile_workload begin
    using Cadnip.MNA: reset_for_restamping!, ZERO_VECTOR
    spec = MNASpec()

    # Exercises both ctx types (MNAContext, DirectStampContext) × both x types
    # (ZeroVector used by compile_structure, Vector{Float64} used by Newton/tran!)
    # matching the exact kwarg signature Cadnip's codegen emits.
    function precompile_device(builder, params)
        # MNAContext + x::ZeroVector — matches compile_structure's initial probe
        ctx = builder(params, spec, 0.0; x=ZERO_VECTOR)
        # MNAContext + x::Vector{Float64} — matches solve_dc's first Newton call
        builder(params, spec, 0.0; x=Float64[], ctx=ctx)

        # DirectStampContext + x::Vector{Float64} — fast_rebuild! hot path
        cs = compile_structure(builder, params, spec; ctx=ctx)
        ws = create_workspace(cs; ctx=ctx)
        reset_direct_stamp!(ws.dctx)
        fast_rebuild!(ws, zeros(cs.n), 0.0)
    end

    # JUNCAP200 (2-terminal diode)
    function juncap_builder(params, spec::MNASpec, t::Real=0.0;
                            x::AbstractVector=Float64[],
                            ctx::Union{MNAContext,DirectStampContext,Nothing}=nothing,
                            _mna_h_=nothing, _mna_h_p_=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        a = get_node!(ctx, :a)
        stamp!(JUNCAP200_module.JUNCAP200(), ctx, a, 0;
               _mna_t_=t, _mna_mode_=spec.mode, _mna_x_=x, _mna_spec_=spec,
               _mna_instance_=:d1, _mna_h_=_mna_h_, _mna_h_p_=_mna_h_p_)
        return ctx
    end
    precompile_device(juncap_builder, NamedTuple())

    # PSP103VA (4-terminal MOSFET)
    function psp_builder(params, spec::MNASpec, t::Real=0.0;
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
        stamp!(PSP103VA_module.PSP103VA(), ctx, d, g, s, 0;
               _mna_t_=t, _mna_mode_=spec.mode, _mna_x_=x, _mna_spec_=spec,
               _mna_instance_=:m1, _mna_h_=_mna_h_, _mna_h_p_=_mna_h_p_)
        return ctx
    end
    precompile_device(psp_builder, NamedTuple())
end

end # module
