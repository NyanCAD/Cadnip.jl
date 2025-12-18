# Tests for clean API (without MNA prefixes)

using Test
using CedarSim
using ScopedValues

@testset "Clean API Type Aliases" begin
    # Test that aliases point to correct types
    @test Circuit === MNACircuit
    @test Net === MNANet
    @test NetRef === MNANetRef
    @test SimpleResistor === MNASimpleResistor
    @test SimpleCapacitor === MNASimpleCapacitor
    @test SimpleInductor === MNASimpleInductor
    @test SimpleDiode === MNASimpleDiode
    @test VoltageSource === SpcVoltageSource
    @test CurrentSource === SpcCurrentSource
    @test VCVS === SpcVCVS
    @test VCCS === SpcVCCS
end

@testset "Circuit Building with Clean API" begin
    # Test using Circuit instead of MNACircuit
    ckt = Circuit()
    vsource!(ckt, :vcc, :gnd; dc=5.0, name=:V1)
    resistor!(ckt, :vcc, :out, 1000.0; name=:R1)
    resistor!(ckt, :out, :gnd, 1000.0; name=:R2)
    ground!(ckt, :gnd)

    result = dc!(ckt)
    @test isapprox(node_voltage(CircuitSystem(ckt), result.solution, :vcc), 5.0, atol=1e-6)
    @test isapprox(node_voltage(CircuitSystem(ckt), result.solution, :out), 2.5, atol=1e-6)
end

@testset "Device Classes with Clean API" begin
    ckt = Circuit()

    @with circuit => ckt begin
        vcc = mna_net(:vcc)
        out = mna_net(:out)
        gnd = mna_ground()

        # Use clean device names
        VoltageSource(dc=3.3)(vcc, gnd)
        SimpleResistor(r=1000.0)(vcc, out)
        SimpleResistor(r=1000.0)(out, gnd)
    end

    result = dc!(ckt)
    sys = CircuitSystem(ckt)
    @test isapprox(node_voltage(sys, result.solution, :vcc), 3.3, atol=1e-6)
    @test isapprox(node_voltage(sys, result.solution, :out), 1.65, atol=1e-6)
end

@testset "Transient with Clean API" begin
    # RC circuit
    ckt = Circuit()
    vsource!(ckt, :vcc, :gnd; dc=5.0, name=:V1)
    resistor!(ckt, :vcc, :out, 1000.0; name=:R1)
    capacitor!(ckt, :out, :gnd, 1e-6; name=:C1)
    ground!(ckt, :gnd)

    sys = CircuitSystem(ckt)
    sol = solve_tran(sys, (0.0, 1e-3))

    # Named solution access should work
    @test length(sol[:vcc]) == length(sol.t)
    @test all(v -> isapprox(v, 5.0, atol=1e-6), sol[:vcc])
end

println("All clean API tests passed!")
