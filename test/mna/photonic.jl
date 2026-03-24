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

@testset "Simulate: Polar2Cartesian stamp verification" begin
    va"""
    nature OpticalElectricField
        units = "V/m";
        access = OptE;
        abstol = 1e-12;
    endnature

    discipline optical
        potential OpticalElectricField;
    enddiscipline

    module Polar2Cart(pol, cart);
        input [0:1] pol;
        output [0:1] cart;
        optical [0:1] pol, cart;
        analog begin
            OptE(cart[0]) <+ OptE(pol[0]) * cos(OptE(pol[1]));
            OptE(cart[1]) <+ OptE(pol[0]) * sin(OptE(pol[1]));
        end
    endmodule
    """

    # Manually invoke stamp! and verify the MNA system
    ctx = MNAContext()
    pol_0 = get_node!(ctx, :pol_0)
    pol_1 = get_node!(ctx, :pol_1)
    cart_0 = get_node!(ctx, :cart_0)
    cart_1 = get_node!(ctx, :cart_1)

    # Stamp at: amplitude=2.0, phase=π/4
    # Expected: cart_0 = 2*cos(π/4) ≈ √2, cart_1 = 2*sin(π/4) ≈ √2
    x = zeros(4)
    x[pol_0] = 2.0
    x[pol_1] = π/4
    stamp!(Polar2Cart(), ctx, pol_0, pol_1, cart_0, cart_1; _mna_x_=x)

    sys = assemble!(ctx)

    # The device creates two voltage contributions:
    #   OptE(cart[0]) <+ pol_0 * cos(pol_1)  → V(cart_0) = 2*cos(π/4)
    #   OptE(cart[1]) <+ pol_0 * sin(pol_1)  → V(cart_1) = 2*sin(π/4)
    #
    # Each voltage contribution stamps:
    #   G[cart, I_var] = 1,  G[I_var, cart] = 1,  b[I_var] = value
    # So the b vector contains the computed values at the branch current rows.

    # Find the branch current indices (they come after the 4 node voltages)
    I_cart0 = sys.n_nodes + 1  # first branch current
    I_cart1 = sys.n_nodes + 2  # second branch current

    # The b vector at the branch current rows should contain the contribution values
    @test sys.b[I_cart0] ≈ 2.0 * cos(π/4) atol=1e-10
    @test sys.b[I_cart1] ≈ 2.0 * sin(π/4) atol=1e-10

    # G matrix should enforce voltage constraints:
    # G[I_cart0, cart_0] = 1  (voltage constraint row)
    # G[cart_0, I_cart0] = 1  (KCL: current into cart_0)
    @test sys.G[I_cart0, cart_0] ≈ 1.0
    @test sys.G[cart_0, I_cart0] ≈ 1.0
    @test sys.G[I_cart1, cart_1] ≈ 1.0
    @test sys.G[cart_1, I_cart1] ≈ 1.0
end

@testset "Code generation: Polar2Cartesian from file" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    access_map = CedarSim.build_access_map(va)
    vamod = va.stmts[end]
    expr = CedarSim.make_mna_device(vamod; access_map)
    @test expr isa Expr
end

end # Photonic Models
