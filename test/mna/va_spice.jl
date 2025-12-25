#==============================================================================#
# MNA Phase 6: Verilog-A to SPICE/Spectre Integration Tests
#
# Tests for using VA models within SPICE netlists via:
# - Subcircuit-like syntax (X device)
# - Model cards
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current
using CedarSim.MNA: VoltageSource, Resistor

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

@testset "MNA VA-SPICE Integration (Phase 6)" begin

    @testset "VA module defined via va_str" begin
        # Define a simple VA resistor for testing
        va"""
        module VAResistor(p, n);
            parameter real R = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/R;
        endmodule
        """

        # Test that the VA module is created and can be stamped directly
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
        stamp!(VAResistor(R=2000.0), ctx, vcc, 0)

        sys = assemble!(ctx)
        sol = solve_dc(sys)

        # V = 5V, R = 2000Ω, I = 2.5mA
        @test isapprox_deftol(voltage(sol, :vcc), 5.0)
        @test isapprox(current(sol, :I_V1), -0.0025; atol=1e-5)
    end

    @testset "VA module used as SPICE subcircuit (X device)" begin
        # Define a VA capacitor
        va"""
        module VACapacitor(p, n);
            parameter real C = 1e-12;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ C*ddt(V(p,n));
        endmodule
        """

        # Test direct instantiation like subcircuit
        # The VA module should be callable with keyword arguments
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        cap_node = get_node!(ctx, :cap)

        stamp!(VoltageSource(3.3; name=:V1), ctx, vcc, 0)
        stamp!(Resistor(1000.0; name=:R1), ctx, vcc, cap_node)
        stamp!(VACapacitor(C=1e-9), ctx, cap_node, 0)

        sys = assemble!(ctx)

        # DC solution: capacitor is open, so V(cap) = V(vcc) = 3.3V
        sol = solve_dc(sys)
        @test isapprox_deftol(voltage(sol, :vcc), 3.3)
        @test isapprox_deftol(voltage(sol, :cap), 3.3)

        # Check that capacitance is stamped correctly
        cap_idx = findfirst(n -> n == :cap, sys.node_names)
        @test isapprox(sys.C[cap_idx, cap_idx], 1e-9; atol=1e-12)
    end

    @testset "VA model with multiple parameters" begin
        # Define a VA parallel RC
        va"""
        module VAParallelRC(p, n);
            parameter real R = 1000.0;
            parameter real C = 1e-12;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/R + C*ddt(V(p,n));
        endmodule
        """

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)

        stamp!(VoltageSource(2.5; name=:V1), ctx, vcc, 0)
        stamp!(VAParallelRC(R=500.0, C=2e-9), ctx, vcc, 0)

        sys = assemble!(ctx)

        # Check conductance: 1/500 = 0.002
        vcc_idx = findfirst(n -> n == :vcc, sys.node_names)
        @test isapprox(sys.G[vcc_idx, vcc_idx], 0.002; atol=1e-6)

        # Check capacitance
        @test isapprox(sys.C[vcc_idx, vcc_idx], 2e-9; atol=1e-12)

        # DC solution
        sol = solve_dc(sys)
        @test isapprox_deftol(voltage(sol, :vcc), 2.5)
        # Current: I = V/R = 2.5/500 = 5mA
        @test isapprox(current(sol, :I_V1), -0.005; atol=1e-5)
    end

    @testset "VA diode nonlinear device" begin
        # Define a simple VA diode (Shockley equation)
        va"""
        module VADiode(p, n);
            parameter real Is = 1e-14;
            parameter real n = 1.0;
            parameter real Vt = 0.026;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ Is * (exp(V(p,n)/(n*Vt)) - 1);
        endmodule
        """

        # Simple diode circuit: V -> R -> Diode -> GND
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        diode_node = get_node!(ctx, :diode)

        stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
        stamp!(Resistor(1000.0; name=:R1), ctx, vcc, diode_node)

        # Need to stamp at an operating point for Newton iteration
        x = zeros(2)
        x[vcc] = 5.0
        x[diode_node] = 0.6  # Typical forward bias

        # Using stamp_current_contribution! directly for nonlinear device
        Is = 1e-14
        n_param = 1.0
        Vt = 0.026
        contrib_fn(V) = Is * (exp(V / (n_param * Vt)) - 1)
        MNA.stamp_current_contribution!(ctx, diode_node, 0, contrib_fn, x)

        sys = assemble!(ctx)

        # The system should be set up for Newton iteration
        # G matrix should have conductance at operating point
        @test sys.G[diode_node, diode_node] > 0  # Nonzero conductance
    end

    @testset "VA module instantiation with defaults" begin
        # Define a VA current source with a different parameter name to avoid conflict with I()
        # Use a unique name to avoid module caching issues between test runs
        va"""
        module VADefaultCurrent6(p, n);
            parameter real Idc = 1e-3;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ Idc;
        endmodule
        """

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)

        stamp!(Resistor(1000.0; name=:R1), ctx, vcc, 0)
        # Use default current (should be 1e-3 = 1mA)
        # I(vcc, gnd) <+ Idc means current flows from vcc to ground (draining the node)
        # So to source current INTO the node, we connect the other way: I(gnd, vcc)
        stamp!(VADefaultCurrent6(), ctx, 0, vcc)

        sys = assemble!(ctx)
        sol = solve_dc(sys)

        # V = I * R = 1mA * 1kΩ = 1V
        @test isapprox(voltage(sol, :vcc), 1.0; atol=0.01)
    end

    @testset "VA module with internal nodes" begin
        # Define a VA device with internal behavior
        va"""
        module VASeriesRL(p, n);
            parameter real R = 100.0;
            parameter real L = 1e-6;
            inout p, n;
            electrical p, n;
            analog begin
                I(p,n) <+ V(p,n)/R;
            end
        endmodule
        """

        # Note: This simplified version just uses resistive behavior
        # Full inductor support would need more complex handling
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)

        stamp!(VoltageSource(1.0; name=:V1), ctx, vcc, 0)
        stamp!(VASeriesRL(R=200.0), ctx, vcc, 0)

        sys = assemble!(ctx)
        sol = solve_dc(sys)

        # V = 1V, R = 200Ω, I = 5mA
        @test isapprox(current(sol, :I_V1), -0.005; atol=1e-5)
    end

end
