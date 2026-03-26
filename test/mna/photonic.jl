using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, voltage, current
using CedarSim.MNA: VoltageSource, Resistor, MNACircuit
using CedarSim: dc!
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

    # Find the branch current indices (they come after the 4 node voltages)
    I_cart0 = sys.n_nodes + 1
    I_cart1 = sys.n_nodes + 2

    # G matrix should enforce voltage constraints
    @test sys.G[I_cart0, cart_0] ≈ 1.0
    @test sys.G[cart_0, I_cart0] ≈ 1.0
    @test sys.G[I_cart1, cart_1] ≈ 1.0
    @test sys.G[cart_1, I_cart1] ≈ 1.0

    # Jacobian entries: ∂(pol_0*cos(pol_1))/∂pol_0 = cos(π/4), ∂/∂pol_1 = -pol_0*sin(π/4)
    @test sys.G[I_cart0, pol_0] ≈ -cos(π/4) atol=1e-10
    @test sys.G[I_cart0, pol_1] ≈ 2.0*sin(π/4) atol=1e-10

    # Verify the linearized constraint is self-consistent
    @test sys.G[I_cart0, cart_0] * 2*cos(π/4) + sys.G[I_cart0, pol_0] * 2.0 +
          sys.G[I_cart0, pol_1] * (π/4) ≈ sys.b[I_cart0] atol=1e-10
end

@testset "Code generation: Polar2Cartesian from file" begin
    va = VerilogAParser.parsefile(joinpath(PHOTONIC_DIR, "Polar2Cartesian.va"))
    access_map = CedarSim.build_access_map(va)
    vamod = va.stmts[end]
    expr = CedarSim.make_mna_device(vamod; access_map)
    @test expr isa Expr
end

end # Photonic Models

const deftol = 1e-6

@testset "Module Instantiation" begin

    @testset "Basic: two resistors in series" begin
        va"""
        module VARes(p, n);
            inout p, n;
            electrical p, n;
            parameter real R = 1000.0;
            analog I(p,n) <+ V(p,n)/R;
        endmodule

        module TwoRes(a, b);
            inout a, b;
            electrical a, b;
            electrical mid;
            VARes R1(a, mid);
            VARes R2(mid, b);
            analog begin
            end
        endmodule
        """

        function twores_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            stamp!(VoltageSource(1.0; name=:V1), ctx, vcc, 0)
            stamp!(TwoRes(), ctx, vcc, 0)
            return ctx
        end

        circuit = MNACircuit(twores_circuit)
        sol = dc!(circuit)

        # Two 1k resistors in series: R_total = 2000Ω, I = 0.5mA
        @test isapprox(voltage(sol, :vcc), 1.0; atol=deftol)
        @test isapprox(current(sol, :I_V1), -0.0005; atol=1e-5)
    end

    @testset "Array ports with slicing" begin
        va"""
        module ArrayCopy(inp, outp);
            inout [0:1] inp, outp;
            electrical [0:1] inp, outp;
            analog begin
                I(outp[0], inp[0]) <+ V(outp[0], inp[0]) * 1000;
                I(outp[1], inp[1]) <+ V(outp[1], inp[1]) * 1000;
            end
        endmodule

        module ArrayParent(a, b);
            inout [0:3] a;
            inout [0:3] b;
            electrical [0:3] a, b;
            ArrayCopy C1(a[0:1], b[0:1]);
            ArrayCopy C2(a[2:3], b[2:3]);
            analog begin
            end
        endmodule
        """

        function array_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            a0 = get_node!(ctx, :a0)
            a1 = get_node!(ctx, :a1)
            a2 = get_node!(ctx, :a2)
            a3 = get_node!(ctx, :a3)
            b0 = get_node!(ctx, :b0)
            b1 = get_node!(ctx, :b1)
            b2 = get_node!(ctx, :b2)
            b3 = get_node!(ctx, :b3)

            stamp!(VoltageSource(1.0; name=:Va0), ctx, a0, 0)
            stamp!(VoltageSource(2.0; name=:Va1), ctx, a1, 0)
            stamp!(VoltageSource(3.0; name=:Va2), ctx, a2, 0)
            stamp!(VoltageSource(4.0; name=:Va3), ctx, a3, 0)

            stamp!(Resistor(1.0), ctx, b0, 0)
            stamp!(Resistor(1.0), ctx, b1, 0)
            stamp!(Resistor(1.0), ctx, b2, 0)
            stamp!(Resistor(1.0), ctx, b3, 0)

            stamp!(ArrayParent(), ctx, a0, a1, a2, a3, b0, b1, b2, b3)

            return ctx
        end

        circuit = MNACircuit(array_circuit)
        sol = dc!(circuit)

        @test isapprox(voltage(sol, :a0), 1.0; atol=deftol)
        @test isapprox(voltage(sol, :a1), 2.0; atol=deftol)
        @test isapprox(voltage(sol, :a2), 3.0; atol=deftol)
        @test isapprox(voltage(sol, :a3), 4.0; atol=deftol)
        @test abs(voltage(sol, :b0)) > 0.1
        @test abs(voltage(sol, :b1)) > 0.1
        @test abs(voltage(sol, :b2)) > 0.1
        @test abs(voltage(sol, :b3)) > 0.1
    end

    @testset "Photonic-style: Polar2Cartesian + Attenuator" begin
        va"""
        module Polar2Cart(pol, cart);
            input [0:1] pol;
            output [0:1] cart;
            electrical [0:1] pol, cart;
            analog begin
                I(cart[0]) <+ V(cart[0]) - V(pol[0]) * cos(V(pol[1]));
                I(cart[1]) <+ V(cart[1]) - V(pol[0]) * sin(V(pol[1]));
            end
        endmodule

        module SimpleAtten(inp, outp);
            inout [0:3] inp, outp;
            electrical [0:3] inp, outp;
            electrical [0:1] transfer_pol, transfer;
            parameter real attenuation = 6.0;
            Polar2Cart P1(transfer_pol, transfer);
            analog begin
                I(transfer_pol[0]) <+ V(transfer_pol[0]) - pow(10, - attenuation / 20);
                I(transfer_pol[1]) <+ V(transfer_pol[1]) - 0;
            end
        endmodule
        """

        function atten_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            inp0 = get_node!(ctx, :inp0)
            inp1 = get_node!(ctx, :inp1)
            inp2 = get_node!(ctx, :inp2)
            inp3 = get_node!(ctx, :inp3)
            outp0 = get_node!(ctx, :outp0)
            outp1 = get_node!(ctx, :outp1)
            outp2 = get_node!(ctx, :outp2)
            outp3 = get_node!(ctx, :outp3)
            stamp!(Resistor(1.0), ctx, inp0, 0)
            stamp!(Resistor(1.0), ctx, inp1, 0)
            stamp!(Resistor(1.0), ctx, outp0, 0)
            stamp!(Resistor(1.0), ctx, outp1, 0)
            stamp!(SimpleAtten(attenuation=6.0), ctx, inp0, inp1, inp2, inp3, outp0, outp1, outp2, outp3)
            return ctx
        end

        circuit = MNACircuit(atten_circuit)
        sol = dc!(circuit)

        expected_gain = 10^(-6.0/20)

        transfer_0_idx = findfirst(==(:SimpleAtten_transfer_0), sol.node_names)
        transfer_1_idx = findfirst(==(:SimpleAtten_transfer_1), sol.node_names)
        transfer_pol_0_idx = findfirst(==(:SimpleAtten_transfer_pol_0), sol.node_names)

        @test transfer_0_idx !== nothing
        @test transfer_pol_0_idx !== nothing
        @test isapprox(sol.x[transfer_pol_0_idx], expected_gain; atol=1e-4)
        @test isapprox(sol.x[transfer_0_idx], expected_gain; atol=1e-4)
        @test isapprox(sol.x[transfer_1_idx], 0.0; atol=1e-4)
    end

end # Module Instantiation
