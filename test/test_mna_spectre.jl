# Tests for MNA-Spectre Integration

using Test
using CedarSim
using CedarSim: MNASpectreEnvironment
using ScopedValues

@testset "MNA-Spectre Integration" begin

    @testset "Basic MNA Circuit Building" begin
        # Build circuit using direct MNA API
        circuit = MNACircuit()
        vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
        resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
        resistor!(circuit, :out, :gnd, 1000.0; name=:R2)
        ground!(circuit, :gnd)

        # Solve DC
        result = dc!(circuit)

        # Check voltage divider
        vcc_idx = circuit.nets[:vcc].index
        out_idx = circuit.nets[:out].index

        @test isapprox(result.solution[vcc_idx], 5.0, atol=1e-6)
        @test isapprox(result.solution[out_idx], 2.5, atol=1e-6)
    end

    @testset "MNA Spectre-style Devices" begin
        # Test using MNASpectreEnvironment devices
        @test MNASpectreEnvironment.resistor === MNASimpleResistor
        @test MNASpectreEnvironment.capacitor === MNASimpleCapacitor
        @test MNASpectreEnvironment.vsource === SpcVoltageSource

        # Create devices
        r = MNASpectreEnvironment.resistor(r=1000.0)
        @test r isa MNASimpleResistor
        @test r.r == 1000.0

        c = MNASpectreEnvironment.capacitor(c=1e-6)
        @test c isa MNASimpleCapacitor
        @test c.capacitance == 1e-6

        v = MNASpectreEnvironment.vsource(dc=5.0)
        @test v isa SpcVoltageSource
        @test v.dc == 5.0
    end

    @testset "Circuit Context (mna_circuit[])" begin
        circuit = MNACircuit()

        # Build circuit using context
        @with mna_circuit => circuit begin
            # Create nets
            vcc = mna_net(:vcc)
            out = mna_net(:out)
            gnd = mna_ground()

            @test vcc isa MNANetRef
            @test out isa MNANetRef
            @test gnd isa MNANetRef
            @test gnd.net.index == 0  # Ground is index 0

            # Stamp devices
            v = SpcVoltageSource(dc=5.0)
            v(vcc, gnd)

            r1 = MNASimpleResistor(r=1000.0)
            r1(vcc, out)

            r2 = MNASimpleResistor(r=1000.0)
            r2(out, gnd)
        end

        # Solve
        result = dc!(circuit)

        # Check result
        out_idx = circuit.nets[:out].index
        @test isapprox(result.solution[out_idx], 2.5, atol=1e-6)
    end

    @testset "simulate_dc helper" begin
        function my_circuit()
            vcc = mna_net(:vcc)
            out = mna_net(:out)
            gnd = mna_ground()

            SpcVoltageSource(dc=3.3)(vcc, gnd)
            MNASimpleResistor(r=100.0)(vcc, out)
            MNASimpleResistor(r=200.0)(out, gnd)
        end

        solution, circuit = simulate_dc(my_circuit)

        out_idx = circuit.nets[:out].index
        expected_voltage = 3.3 * 200 / 300  # Voltage divider

        @test isapprox(solution[out_idx], expected_voltage, atol=1e-9)
    end

    @testset "Named wrapper" begin
        circuit = MNACircuit()

        @with mna_circuit => circuit begin
            vcc = mna_net(:vcc)
            gnd = mna_ground()

            # Use Named wrapper like spectre.jl codegen does
            dev = SpcVoltageSource(dc=1.5)
            named_dev = Named(dev, "V1")
            named_dev(vcc, gnd)
        end

        result = dc!(circuit)
        vcc_idx = circuit.nets[:vcc].index
        @test isapprox(result.solution[vcc_idx], 1.5, atol=1e-9)
    end

    @testset "ParallelInstances (m parameter)" begin
        circuit = MNACircuit()

        @with mna_circuit => circuit begin
            vcc = mna_net(:vcc)
            out = mna_net(:out)
            gnd = mna_ground()

            SpcVoltageSource(dc=10.0)(vcc, gnd)

            # Two parallel resistors using m=2
            # R_eff = 1000/2 = 500 ohms
            r_parallel = ParallelInstances(MNASimpleResistor(r=1000.0), 2.0)
            r_parallel(vcc, out)

            MNASimpleResistor(r=500.0)(out, gnd)
        end

        result = dc!(circuit)
        out_idx = circuit.nets[:out].index
        # Voltage divider: out = 10V * 500 / (500 + 500) = 5V
        @test isapprox(result.solution[out_idx], 5.0, atol=1e-6)
    end

    @testset "spicecall helper" begin
        circuit = MNACircuit()

        @with mna_circuit => circuit begin
            vcc = mna_net(:vcc)
            out = mna_net(:out)
            gnd = mna_ground()

            # Use spicecall like spectre.jl codegen does
            v_inst = spicecall(SpcVoltageSource; dc=2.0)
            v_inst(vcc, gnd)

            r_inst = spicecall(MNASimpleResistor; r=1000.0, m=1.0)
            r_inst(vcc, out)

            r2_inst = spicecall(MNASimpleResistor; r=1000.0, m=1.0)
            r2_inst(out, gnd)
        end

        result = dc!(circuit)
        out_idx = circuit.nets[:out].index
        @test isapprox(result.solution[out_idx], 1.0, atol=1e-9)
    end

    @testset "Current source" begin
        circuit = MNACircuit()

        @with mna_circuit => circuit begin
            out = mna_net(:out)
            gnd = mna_ground()

            # 1mA current source: current flows from gnd to out (SPICE convention: I from - to +)
            # This pushes current INTO out, creating +1V across 1k resistor
            SpcCurrentSource(dc=0.001)(gnd, out)  # Current from gnd INTO out
            MNASimpleResistor(r=1000.0)(out, gnd)
        end

        result = dc!(circuit)
        out_idx = circuit.nets[:out].index
        @test isapprox(result.solution[out_idx], 1.0, atol=1e-6)
    end

    @testset "Capacitor in DC" begin
        circuit = MNACircuit()

        @with mna_circuit => circuit begin
            vcc = mna_net(:vcc)
            out = mna_net(:out)
            gnd = mna_ground()

            SpcVoltageSource(dc=5.0)(vcc, gnd)
            MNASimpleResistor(r=1000.0)(vcc, out)
            MNASimpleCapacitor(c=1e-6)(out, gnd)
        end

        result = dc!(circuit)
        out_idx = circuit.nets[:out].index
        # In DC, capacitor is open circuit, so out = vcc (minus small gmin effect)
        @test isapprox(result.solution[out_idx], 5.0, atol=1e-6)
    end

    @testset "DefaultOr parameter handling" begin
        # Test that DefaultOr values work correctly
        r1 = MNASimpleResistor(r=mkdefault(1000.0))
        @test r1.r == 1000.0

        r2 = MNASimpleResistor(r=mknondefault(2000.0))
        @test r2.r == 2000.0
    end

end

println("All MNA-Spectre integration tests passed!")
