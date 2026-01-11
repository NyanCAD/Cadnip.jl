#==============================================================================#
# Monostable Multivibrator Test with sp_bjt model
#
# Tests the VADistiller sp_bjt model with CedarDCOp using relaxed tolerance.
#
# Run with: julia --project=test test/mna/monostable_sp_bjt_test.jl
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNACircuit, MNASolutionAccessor
using CedarSim.MNA: voltage, assemble!, CedarDCOp
using VerilogAParser
using SciMLBase
using CedarSim: tran!, parse_spice_to_mna
using OrdinaryDiffEq: Rodas5P

#==============================================================================#
# Load sp_bjt model from VADistiller
#==============================================================================#

println("Loading sp_bjt model...")
const bjt_path = joinpath(@__DIR__, "..", "..", "models", "VADistillerModels.jl", "va", "bjt.va")
const bjt_va = VerilogAParser.parsefile(bjt_path)
if bjt_va.ps.errored
    error("Failed to parse bjt.va")
end
Core.eval(@__MODULE__, CedarSim.make_mna_module(bjt_va))

#==============================================================================#
# Load SPICE netlist
#==============================================================================#

const spice_path = joinpath(@__DIR__, "..", "vadistiller", "circuits", "monostable_multivibrator.spice")
const spice_code = read(spice_path, String)

const monostable_code = parse_spice_to_mna(spice_code;
    circuit_name=:monostable_multivibrator,
    imported_hdl_modules=[sp_bjt_module])
eval(monostable_code)

#==============================================================================#
# Test
#==============================================================================#

@testset "sp_bjt Monostable with CedarDCOp" begin
    circuit = MNACircuit(monostable_multivibrator)

    # Short simulation to test initialization
    tspan = (0.0, 1e-3)

    @info "Running sp_bjt monostable transient simulation" tspan

    # Use high maxiters (500) to allow convergence at tight tolerance (1e-6)
    # sp_bjt internal nodes (excess phase) may need more iterations
    sol = tran!(circuit, tspan;
                solver=Rodas5P(),
                initializealg=CedarDCOp(abstol=1e-6, maxiters=500),
                abstol=1e-6, reltol=1e-4,
                dtmax=1e-4)

    @info "Transient result" sol.retcode

    @test sol.retcode == SciMLBase.ReturnCode.Success

    if sol.retcode == SciMLBase.ReturnCode.Success
        sys = assemble!(circuit)
        acc = MNASolutionAccessor(sol, sys)

        v_q1_coll = voltage(acc, :q1_coll, 0.5e-3)
        v_q2_coll = voltage(acc, :q2_coll, 0.5e-3)

        @info "Node voltages at t=0.5ms" v_q1_coll v_q2_coll

        # Check we got valid values (not NaN)
        @test !isnan(v_q1_coll)
        @test !isnan(v_q2_coll)
    end
end
