using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, voltage
using CedarSim.MNA: VoltageSource
using VerilogAParser

const PHOTONIC_DIR = joinpath(@__DIR__, "photonic_models")

@testset "Photonic Models" begin

@testset "Parser: nature/discipline declarations" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    types = [CedarSim.formof(s) for s in va.stmts]
    @test VerilogAParser.VerilogACSTParser.VerilogModule in types
    @test VerilogAParser.VerilogACSTParser.NatureDeclaration in types
    @test VerilogAParser.VerilogACSTParser.DisciplineDeclaration in types
end

@testset "Access map from disciplines" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    access_map = CedarSim.build_access_map(va)

    # V and I should come from parsed electrical discipline, not hardcoded
    @test access_map[:V] == :potential
    @test access_map[:I] == :flow

    # OptE from optical discipline
    @test access_map[:OptE] == :potential

    # Other standard access functions from parsed disciplines.vams
    @test access_map[:Temp] == :potential  # thermal
    @test access_map[:Pwr] == :flow        # thermal
    @test access_map[:MMF] == :potential   # magnetic (MMF is potential, Phi is flow)
end

@testset "Array port expansion" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    vamod = va.stmts[end]
    (ps, array_nodes) = CedarSim.pins(vamod)

    @test length(ps) == 4
    @test :pol_0 in ps
    @test :pol_1 in ps
    @test :cart_0 in ps
    @test :cart_1 in ps
    @test array_nodes[:pol] == (0, [:pol_0, :pol_1])
    @test array_nodes[:cart] == (0, [:cart_0, :cart_1])
end

@testset "Simulate: optical constant source" begin
    # Test that OptE voltage contributions work with constant values.
    # NOTE: Nonlinear voltage contributions (depending on other node voltages)
    # need Jacobian enhancement — that's a separate issue tracked in the plan.
    va"""
    nature OpticalElectricField
        units = "V/m";
        access = OptE;
        abstol = 1e-12;
    endnature

    discipline optical
        potential OpticalElectricField;
    enddiscipline

    module OptSource(outp);
        output outp;
        optical outp;
        parameter real amplitude = 1.0;
        analog begin
            OptE(outp) <+ amplitude;
        end
    endmodule
    """

    # Circuit builder: optical source sets output to constant amplitude
    function opt_source_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            CedarSim.MNA.reset_for_restamping!(ctx)
        end
        outp = get_node!(ctx, :outp)
        stamp!(OptSource(amplitude=params.params.amplitude), ctx, outp)
        return ctx
    end

    # amplitude=1.5 → output=1.5
    circ = MNACircuit(opt_source_circuit; params=(amplitude=1.5,))
    sol = dc!(circ)
    @test voltage(sol, :outp) ≈ 1.5 atol=1e-6

    # amplitude=3.0 → output=3.0
    circ2 = MNACircuit(opt_source_circuit; params=(amplitude=3.0,))
    sol2 = dc!(circ2)
    @test voltage(sol2, :outp) ≈ 3.0 atol=1e-6
end

@testset "Code generation: Polar2Cartesian from file" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    access_map = CedarSim.build_access_map(va)
    vamod = va.stmts[end]
    expr = CedarSim.make_mna_device(vamod; access_map)
    @test expr isa Expr
end

end # Photonic Models
