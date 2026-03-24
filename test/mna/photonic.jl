using Test
using CedarSim
using CedarSim.MNA
using VerilogAParser

const PHOTONIC_DIR = joinpath(@__DIR__, "..", "..", "..", "Verilog-A-photonic-model-library", "veriloga")

@testset "Photonic Models" begin

@testset "Parser: disciplines.vams" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    # disciplines.vams has RESERVED words like `domain discrete` which cause parse errors
    # but the important structures (natures, disciplines, module) should still parse

    # Check that we got the expected structure types
    types = [CedarSim.formof(s) for s in va.stmts]
    @test VerilogAParser.VerilogACSTParser.VerilogModule in types
    @test VerilogAParser.VerilogACSTParser.NatureDeclaration in types
    @test VerilogAParser.VerilogACSTParser.DisciplineDeclaration in types
end

@testset "Access map from disciplines" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    access_map = CedarSim.build_access_map(va)
    @info "Access map" access_map

    # Standard access functions
    @test access_map[:V] == :potential
    @test access_map[:I] == :flow

    # Optical access function from disciplines.vams
    @test haskey(access_map, :OptE)
    if haskey(access_map, :OptE)
        @test access_map[:OptE] == :potential
    end
end

@testset "Array port expansion" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    vamod = va.stmts[end]
    (ps, array_nodes) = CedarSim.pins(vamod)

    # Polar2Cartesian has: input [0:1] pol; output [0:1] cart;
    # Should expand to: pol_0, pol_1, cart_0, cart_1
    @test length(ps) == 4
    @test :pol_0 in ps
    @test :pol_1 in ps
    @test :cart_0 in ps
    @test :cart_1 in ps

    # Array mapping should be present
    @test haskey(array_nodes, :pol)
    @test haskey(array_nodes, :cart)
    @test array_nodes[:pol] == (0, [:pol_0, :pol_1])
    @test array_nodes[:cart] == (0, [:cart_0, :cart_1])
end

@testset "Code generation: Polar2Cartesian" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    access_map = CedarSim.build_access_map(va)
    vamod = va.stmts[end]
    try
        expr = CedarSim.make_mna_device(vamod; access_map)
        @test expr isa Expr
    catch e
        @error "Code generation failed" exception=(e, catch_backtrace())
        @test false
    end
end

end # Photonic Models
