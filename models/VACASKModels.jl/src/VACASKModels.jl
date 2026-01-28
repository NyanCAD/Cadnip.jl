"""
    VACASKModels

Pre-configured PSP103 MOSFET subcircuits for the VACASK benchmark.

Uses the proper SPICE codegen API to load model cards and generate
MNA builder functions. Precompiles stamp! methods for fast runtime.

# Exports
- `nmos_mna_builder`: Builder for nmos subcircuit (W/L-parameterized PSP103 NMOS)
- `pmos_mna_builder`: Builder for pmos subcircuit (W/L-parameterized PSP103 PMOS)
"""
module VACASKModels

using CedarSim
using CedarSim.SpectreNetlistParser
using CedarSim.MNA: MNAContext, MNASpec, stamp!, get_node!,
                    compile_structure, create_workspace, fast_rebuild!, reset_direct_stamp!
using PSPModels  # Registers PSP103VA etc with ModelRegistry
using PrecompileTools: @compile_workload

# Path to SPICE model cards
const MODELS_INC = joinpath(@__DIR__, "..", "spice", "models.inc")

# Parse and generate MNA module at package load time
# No imported_hdl_modules needed since PSPModels registers with ModelRegistry
const _ast = SpectreNetlistParser.parsefile(MODELS_INC; implicit_title=false)
const _mod_expr = CedarSim.make_mna_pdk_module(_ast; name=:vacask_models)
eval(_mod_expr)

# Re-export the builders from generated module
using .vacask_models: nmos_mna_builder, pmos_mna_builder
export nmos_mna_builder, pmos_mna_builder

# Precompile stamp! methods for VACASK-sized devices
@compile_workload begin
    using CedarSim: ParamLens
    using CedarSim.MNA: reset_for_restamping!, ZERO_VECTOR
    spec = MNASpec()

    function precompile_builder(builder_fn, sizing_params)
        # Build simple circuit using the builder
        function circuit_fn(params, spec, t=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            lens = ParamLens(params)
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)
            builder_fn(lens, spec, t, ctx, d, g, 0, 0, (;), x; sizing_params...)
            return ctx
        end

        # Phase 1: MNAContext discovery
        ctx = circuit_fn((;), spec)

        # Phase 2: Compile and create workspace
        cs = compile_structure(circuit_fn, (;), spec; ctx=ctx)
        ws = create_workspace(cs; ctx=ctx)

        # Phase 3: DirectStampContext fast rebuild
        reset_direct_stamp!(ws.dctx)
        fast_rebuild!(ws, zeros(cs.n), 0.0)
    end

    # Precompile with VACASK sizing
    precompile_builder(nmos_mna_builder, (w=10e-6, l=1e-6))
    precompile_builder(pmos_mna_builder, (w=20e-6, l=1e-6))
end

end # module
