# Tests for CircuitSystem and SciML-compatible named solution access

using Test
using CedarSim
using ScopedValues
using SymbolicIndexingInterface
using SciMLBase

@testset "CircuitSystem Creation" begin
    # Build a simple circuit
    circuit = MNACircuit()
    vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
    resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
    resistor!(circuit, :out, :gnd, 1000.0; name=:R2)
    ground!(circuit, :gnd)

    # Create CircuitSystem
    sys = CircuitSystem(circuit)

    @test sys.circuit === circuit
    @test :vcc in sys.node_names
    @test :out in sys.node_names
    @test :V1_I in sys.branch_names
end

@testset "SymbolCache Integration" begin
    circuit = MNACircuit()
    vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
    resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
    ground!(circuit, :gnd)

    sys = CircuitSystem(circuit)
    sc = sys.symbol_cache

    # Test SymbolicIndexingInterface methods
    @test is_variable(sc, :vcc)
    @test is_variable(sc, :out)
    @test !is_variable(sc, :nonexistent)

    @test variable_index(sc, :vcc) isa Int
    @test variable_index(sc, :out) isa Int

    @test is_independent_variable(sc, :t)
    @test !is_independent_variable(sc, :vcc)
end

@testset "Voltage/Current Index Functions" begin
    circuit = MNACircuit()
    vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
    resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
    ground!(circuit, :gnd)

    sys = CircuitSystem(circuit)

    @test voltage(sys, :vcc) isa Int
    @test voltage(sys, :out) isa Int
    @test voltage(sys, :nonexistent) === nothing

    @test current(sys, :V1_I) isa Int
end

@testset "Named DC Solution Access" begin
    circuit = MNACircuit()
    vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
    resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
    resistor!(circuit, :out, :gnd, 1000.0; name=:R2)
    ground!(circuit, :gnd)

    result = dc!(circuit)
    sys = CircuitSystem(circuit)

    # Test node_voltage function
    @test isapprox(node_voltage(sys, result.solution, :vcc), 5.0, atol=1e-6)
    @test isapprox(node_voltage(sys, result.solution, :out), 2.5, atol=1e-6)
end

@testset "Transient with Named Access (sol[:sym])" begin
    # Build RC circuit
    circuit = MNACircuit()
    vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
    resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
    capacitor!(circuit, :out, :gnd, 1e-6; name=:C1)
    ground!(circuit, :gnd)

    sys = CircuitSystem(circuit)

    # Solve transient
    sol = solve_tran(sys, (0.0, 5e-3); reltol=1e-6, abstol=1e-8)

    # Test named solution access (sol[:vcc])
    @test length(sol[:vcc]) == length(sol.t)
    @test length(sol[:out]) == length(sol.t)

    # Voltage source should stay at 5V
    @test all(v -> isapprox(v, 5.0, atol=1e-6), sol[:vcc])

    # At DC equilibrium (capacitor acts as open circuit), out = vcc
    @test isapprox(sol[:out][end], 5.0, atol=0.1)
end

@testset "Create Problem with Custom Options" begin
    circuit = MNACircuit()
    vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
    resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
    ground!(circuit, :gnd)

    sys = CircuitSystem(circuit)

    # Create problem
    prob = create_problem(sys, (0.0, 1e-3))

    @test prob isa SciMLBase.AbstractODEProblem
    @test prob.tspan == (0.0, 1e-3)
end

println("All CircuitSystem tests passed!")
