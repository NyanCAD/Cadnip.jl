#==============================================================================#
# MNA Phase 6: Verilog-A to SPICE/Spectre Integration Tests
#
# Tests the integration of VA models with SPICE/Spectre netlists via
# the imported_hdl_modules mechanism in sema().
#
# This complements:
# - mna/core.jl: MNA matrix assembly and solve
# - mna/va.jl: VA→MNA stamping (va_str macro, stamp! methods)
# - mna/vadistiller.jl: Direct VA model tests (resistor, capacitor, MOSFET, etc.)
# - basic.jl: SPICE/Spectre parsing tests (solve_mna_spice_code, solve_mna_spectre_code)
#
# This file focuses specifically on:
# 1. Using VA modules from SPICE/Spectre netlists via X device syntax
# 2. The imported_hdl_modules mechanism for device resolution
# 3. Mixed circuits with VA and native SPICE/Spectre elements
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!
using CedarSim.MNA: voltage, current, MNACircuit
using CedarSim: dc!
using CedarSim.MNA: VoltageSource, Resistor

include(joinpath(@__DIR__, "..", "common.jl"))

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

#==============================================================================#
# Tests: VA Modules in SPICE Netlists
#==============================================================================#

@testset "VA-SPICE Integration" begin

    @testset "VA resistor in SPICE netlist" begin
        # Define VA resistor - lowercase for SPICE case-insensitivity
        va"""
        module varesistor(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        # SPICE netlist using VA module as X device
        spice = """
        * VA resistor in voltage divider
        V1 vcc 0 DC 10
        X1 vcc mid varesistor r=2k
        R1 mid 0 2k
        """

        ctx, sol = solve_mna_spice_code(spice; imported_hdl_modules=[varesistor_module])

        # Voltage divider: 10V * (2k / (2k + 2k)) = 5V
        @test isapprox_deftol(voltage(sol, :vcc), 10.0)
        @test isapprox_deftol(voltage(sol, :mid), 5.0)
    end

    @testset "VA capacitor in SPICE RC network" begin
        va"""
        module vacapacitor(p, n);
            parameter real c = 1e-12;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ c*ddt(V(p,n));
        endmodule
        """

        spice = """
        * VA capacitor in RC
        V1 vcc 0 DC 5
        R1 vcc cap 1k
        X1 cap 0 vacapacitor c=1n
        """

        ctx, sol = solve_mna_spice_code(spice; imported_hdl_modules=[vacapacitor_module])

        # DC: capacitor is open, V(cap) = V(vcc) = 5V
        @test isapprox_deftol(voltage(sol, :vcc), 5.0)
        @test isapprox_deftol(voltage(sol, :cap), 5.0)

        # Verify capacitance is stamped
        sys = assemble!(ctx)
        cap_idx = findfirst(n -> n == :cap, sys.node_names)
        @test isapprox(sys.C[cap_idx, cap_idx], 1e-9; atol=1e-12)
    end

    @testset "Multiple VA modules in SPICE circuit" begin
        va"""
        module vares(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        va"""
        module vacap(p, n);
            parameter real c = 1e-12;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ c*ddt(V(p,n));
        endmodule
        """

        spice = """
        * Multiple VA devices
        V1 vcc 0 DC 10
        X1 vcc n1 vares r=1k
        X2 n1 n2 vares r=2k
        X3 n2 0 vares r=1k
        X4 n1 0 vacap c=100p
        """

        ctx, sol = solve_mna_spice_code(spice; imported_hdl_modules=[vares_module, vacap_module])

        # Total R = 4k, I = 10/4k = 2.5mA
        # V(n1) = 10 - 2.5mA * 1k = 7.5V
        # V(n2) = 7.5 - 2.5mA * 2k = 2.5V
        @test isapprox(voltage(sol, :n1), 7.5; atol=0.01)
        @test isapprox(voltage(sol, :n2), 2.5; atol=0.01)
    end

end

#==============================================================================#
# Tests: VA Modules in Spectre Netlists
#==============================================================================#

@testset "VA-Spectre Integration" begin

    @testset "VA resistor in Spectre netlist" begin
        va"""
        module spectreres(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        spectre = """
        // VA resistor in Spectre
        v1 (vcc 0) vsource dc=10
        x1 (vcc mid) spectreres r=2k
        r1 (mid 0) resistor r=2k
        """

        ctx, sol = solve_mna_spectre_code(spectre; imported_hdl_modules=[spectreres_module])

        @test isapprox_deftol(voltage(sol, :vcc), 10.0)
        @test isapprox_deftol(voltage(sol, :mid), 5.0)
    end

    @testset "VA parallel RC in Spectre" begin
        va"""
        module spectrerc(p, n);
            parameter real r = 1000.0;
            parameter real c = 1e-12;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r + c*ddt(V(p,n));
        endmodule
        """

        spectre = """
        // VA parallel RC
        v1 (vcc 0) vsource dc=5
        x1 (vcc 0) spectrerc r=500 c=2n
        """

        ctx, sol = solve_mna_spectre_code(spectre; imported_hdl_modules=[spectrerc_module])

        @test isapprox_deftol(voltage(sol, :vcc), 5.0)
        # I = V/R = 5/500 = 10mA
        @test isapprox(current(sol, :I_v1), -0.01; atol=1e-5)
    end

    @testset "Mixed VA and native Spectre devices" begin
        va"""
        module spectreva(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        spectre = """
        // Mixed native and VA
        v1 (vcc 0) vsource dc=12
        r1 (vcc n1) resistor r=1k
        x1 (n1 n2) spectreva r=2k
        r2 (n2 0) resistor r=1k
        """

        ctx, sol = solve_mna_spectre_code(spectre; imported_hdl_modules=[spectreva_module])

        # Total R = 4k, I = 12/4k = 3mA
        # V(n1) = 12 - 3mA * 1k = 9V
        # V(n2) = 9 - 3mA * 2k = 3V
        @test isapprox(voltage(sol, :n1), 9.0; atol=0.01)
        @test isapprox(voltage(sol, :n2), 3.0; atol=0.01)
    end

end

#==============================================================================#
# Tests: VA Modules with Array Ports in SPICE/Spectre Netlists
#
# Photonic VA models use array ports like [0:1] in, out which expand to
# individual pins (in_0, in_1, out_0, out_1). These tests verify that
# the imported_hdl_modules mechanism handles expanded port names correctly.
#==============================================================================#

using PhotonicModels

@testset "VA Array Ports in Netlists" begin

    # Polar2Cartesian has 4 expanded ports: pol_0, pol_1, cart_0, cart_1
    # OptE(cart[0]) <+ OptE(pol[0]) * cos(OptE(pol[1]))
    # OptE(cart[1]) <+ OptE(pol[0]) * sin(OptE(pol[1]))
    # OptE is a potential (voltage-like), so these are voltage contributions.

    @testset "Array port VA model in Spectre netlist" begin
        spectre = """
        // Drive pol_0 = 1.0 (amplitude), pol_1 = 0.0 (phase)
        // Expected: cart_0 = cos(0)*1 = 1.0, cart_1 = sin(0)*1 = 0.0
        v_amp (pol_0 0) vsource dc=1.0
        v_phase (pol_1 0) vsource dc=0.0
        r_cart0 (cart_0 0) resistor r=1
        r_cart1 (cart_1 0) resistor r=1
        p2c (pol_0 pol_1 cart_0 cart_1) Polar2Cartesian
        """

        ctx, sol = solve_mna_spectre_code(spectre;
            imported_hdl_modules=[PhotonicModels.Polar2Cartesian_module])

        @test isapprox(voltage(sol, :pol_0), 1.0; atol=0.01)
        @test isapprox(voltage(sol, :pol_1), 0.0; atol=0.01)
        @test isapprox(voltage(sol, :cart_0), 1.0; atol=0.01)
        @test isapprox(voltage(sol, :cart_1), 0.0; atol=0.01)
    end

    @testset "Array port VA model in SPICE netlist" begin
        spice = """
        * Polar to Cartesian test in SPICE
        V_amp pol_0 0 DC 1.0
        V_phase pol_1 0 DC 0.0
        R_cart0 cart_0 0 1
        R_cart1 cart_1 0 1
        X1 pol_0 pol_1 cart_0 cart_1 Polar2Cartesian
        """

        ctx, sol = solve_mna_spice_code(spice;
            imported_hdl_modules=[PhotonicModels.Polar2Cartesian_module])

        @test isapprox(voltage(sol, :pol_0), 1.0; atol=0.01)
        @test isapprox(voltage(sol, :pol_1), 0.0; atol=0.01)
        @test isapprox(voltage(sol, :cart_0), 1.0; atol=0.01)
        @test isapprox(voltage(sol, :cart_1), 0.0; atol=0.01)
    end

    @testset "Array port VA model with nonzero phase" begin
        spectre = """
        // Drive amplitude=2.0, phase=pi/4 (45 degrees)
        // Expected: cart_0 = 2*cos(pi/4) ≈ 1.4142, cart_1 = 2*sin(pi/4) ≈ 1.4142
        v_amp (pol_0 0) vsource dc=2.0
        v_phase (pol_1 0) vsource dc=0.7853981633974483
        r_cart0 (cart_0 0) resistor r=1
        r_cart1 (cart_1 0) resistor r=1
        p2c (pol_0 pol_1 cart_0 cart_1) Polar2Cartesian
        """

        ctx, sol = solve_mna_spectre_code(spectre;
            imported_hdl_modules=[PhotonicModels.Polar2Cartesian_module])

        expected = 2.0 * cos(π/4)  # ≈ 1.4142
        @test isapprox(voltage(sol, :cart_0), expected; atol=0.01)
        @test isapprox(voltage(sol, :cart_1), expected; atol=0.01)
    end

    @testset "Mixed electrical and photonic VA devices" begin
        spectre = """
        // Electrical: voltage divider
        v1 (vcc 0) vsource dc=10
        r1 (vcc mid) resistor r=1k
        r2 (mid 0) resistor r=1k

        // Photonic: polar to cartesian with voltage-driven amplitude
        // mid ≈ 5V drives amplitude, phase = 0
        v_phase (phase 0) vsource dc=0.0
        r_cart0 (cart_0 0) resistor r=1
        r_cart1 (cart_1 0) resistor r=1
        p2c (mid phase cart_0 cart_1) Polar2Cartesian
        """

        ctx, sol = solve_mna_spectre_code(spectre;
            imported_hdl_modules=[PhotonicModels.Polar2Cartesian_module])

        # Electrical divider
        @test isapprox(voltage(sol, :mid), 5.0; atol=0.01)
        # Photonic output: amplitude=5V, phase=0 -> cart_0=5, cart_1=0
        @test isapprox(voltage(sol, :cart_0), 5.0; atol=0.01)
        @test isapprox(voltage(sol, :cart_1), 0.0; atol=0.01)
    end

end

#==============================================================================#
# Tests: Direct VA stamping (va_str without netlist parsing)
#
# These test the foundation that the netlist integration builds on.
# More comprehensive VA device tests are in mna/vadistiller.jl
#==============================================================================#

@testset "Direct VA Stamping" begin

    @testset "VA module with MNA primitives" begin
        va"""
        module directres(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        function directres_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            mid = get_node!(ctx, :mid)

            stamp!(VoltageSource(10.0; name=:V1), ctx, vcc, 0)
            stamp!(Resistor(1000.0; name=:R1), ctx, vcc, mid)
            stamp!(directres(r=1000.0), ctx, mid, 0)
            return ctx
        end

        circuit = MNACircuit(directres_circuit)
        sol = dc!(circuit)

        @test isapprox_deftol(voltage(sol, :vcc), 10.0)
        @test isapprox_deftol(voltage(sol, :mid), 5.0)
    end

    @testset "VA module chain" begin
        va"""
        module chainres(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        function chainres_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            n1 = get_node!(ctx, :n1)
            n2 = get_node!(ctx, :n2)

            stamp!(VoltageSource(10.0; name=:V1), ctx, vcc, 0)
            stamp!(chainres(r=1000.0), ctx, vcc, n1)
            stamp!(chainres(r=2000.0), ctx, n1, n2)
            stamp!(chainres(r=1000.0), ctx, n2, 0)
            return ctx
        end

        circuit = MNACircuit(chainres_circuit)
        sol = dc!(circuit)

        # Total = 4k, I = 10V/4k = 2.5mA
        @test isapprox(voltage(sol, :n1), 7.5; atol=0.01)
        @test isapprox(voltage(sol, :n2), 2.5; atol=0.01)
    end

end
