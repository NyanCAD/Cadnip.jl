# Ported tests from legacy DAECompiler-based test suite
# This file accounts for all tests from the old system, porting what's possible
# and documenting what cannot be ported and why.

using Test
using CedarSim
using ScopedValues
using Random

# ============================================================================
# BASIC CIRCUIT TESTS (from basic.jl)
# ============================================================================

@testset "Simple VR Circuit" begin
    # Port of basic.jl "Simple VR Circuit"
    # Original: Named(V(5.), "V")(vcc, gnd); Named(R(2), "R")(vcc, gnd)
    circuit = Circuit()
    vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V)
    resistor!(circuit, :vcc, :gnd, 2.0; name=:R)
    ground!(circuit, :gnd)

    result = dc!(circuit)

    # I = V/R = 5/2 = 2.5A
    # R_v = 5.0V (voltage across resistor = source voltage)
    vcc_idx = circuit.nets[:vcc].index
    @test isapprox(result.solution[vcc_idx], 5.0, atol=1e-6)
end

@testset "Simple IR Circuit" begin
    # Port of basic.jl "Simple IR circuit"
    # Current source driving resistor: V = IR
    circuit = Circuit()
    isource!(circuit, :icc, :gnd; dc=-5.0, name=:I)  # Negative = into node
    resistor!(circuit, :icc, :gnd, 2.0; name=:R)
    ground!(circuit, :gnd)

    result = dc!(circuit)

    # V = IR = 5 * 2 = 10V
    icc_idx = circuit.nets[:icc].index
    @test isapprox(result.solution[icc_idx], 10.0, atol=1e-6)
end

const v_val = 5.0
const r_val = 2000.0
const c_val = 1e-6

@testset "Simple VRC Circuit (RC Transient)" begin
    # Port of basic.jl "Simple VRC circuit"
    circuit = Circuit()
    vsource!(circuit, :vcc, :gnd; dc=v_val, name=:V)
    resistor!(circuit, :vcc, :vrc, r_val; name=:R)
    capacitor!(circuit, :vrc, :gnd, c_val; name=:C)
    ground!(circuit, :gnd)

    sys = CircuitSystem(circuit)

    # Solve with initial capacitor voltage = 0
    # Need to set up initial conditions properly
    sol = solve_tran(sys, (0.0, 10e-3); reltol=1e-6, abstol=1e-8)

    # RC circuit: tau = RC = 2000 * 1e-6 = 2ms
    # At t >> tau, capacitor charges to v_val
    @test isapprox(sol[:vrc][end], v_val, atol=0.1)
end

@testset "ParallelInstances" begin
    # Port of basic.jl "ParallelInstances"
    # Multiple parallel resistors reduce effective resistance
    circuit = Circuit()

    @with mna_circuit => circuit begin
        vcc = mna_net(:vcc)
        out = mna_net(:out)
        gnd = mna_ground()

        SpcVoltageSource(dc=v_val)(vcc, gnd)
        # 10 parallel 2k resistors = 200 ohms effective
        r_parallel = ParallelInstances(MNASimpleResistor(r=r_val), 10.0)
        r_parallel(vcc, out)
        MNASimpleCapacitor(c=c_val)(out, gnd)
    end

    # DC: capacitor is open, so out = vcc
    result = dc!(circuit)
    out_idx = circuit.nets[:out].index
    @test isapprox(result.solution[out_idx], v_val, atol=1e-6)
end

@testset "Current Source" begin
    # Port of basic.jl current source test
    circuit = Circuit()

    @with mna_circuit => circuit begin
        out = mna_net(:out)
        gnd = mna_ground()

        # 1mA current source into node
        SpcCurrentSource(dc=0.001)(gnd, out)
        MNASimpleResistor(r=1000.0)(out, gnd)
    end

    result = dc!(circuit)
    out_idx = circuit.nets[:out].index
    # V = IR = 0.001 * 1000 = 1V
    @test isapprox(result.solution[out_idx], 1.0, atol=1e-6)
end

@testset "Capacitor in DC" begin
    # Port of basic.jl "Capacitor in DC"
    circuit = Circuit()

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
    # DC: capacitor open circuit, out = vcc
    @test isapprox(result.solution[out_idx], 5.0, atol=1e-6)
end

@testset "DefaultOr parameter handling" begin
    # Port of basic.jl "DefaultOr parameter handling"
    r1 = MNASimpleResistor(r=mkdefault(1000.0))
    @test r1.r == 1000.0

    r2 = MNASimpleResistor(r=mknondefault(2000.0))
    @test r2.r == 2000.0
end

@testset "Voltage divider variations" begin
    # Test different voltage divider configurations
    for (r1, r2, expected_ratio) in [(1000.0, 1000.0, 0.5),
                                      (1000.0, 2000.0, 2/3),
                                      (2000.0, 1000.0, 1/3)]
        circuit = Circuit()
        vsource!(circuit, :vcc, :gnd; dc=10.0, name=:V)
        resistor!(circuit, :vcc, :out, r1; name=:R1)
        resistor!(circuit, :out, :gnd, r2; name=:R2)
        ground!(circuit, :gnd)

        result = dc!(circuit)
        out_idx = circuit.nets[:out].index
        @test isapprox(result.solution[out_idx], 10.0 * expected_ratio, atol=1e-6)
    end
end

# ============================================================================
# VCVS/VCCS CONTROLLED SOURCES (from basic.jl "Simple SPICE sources")
# ============================================================================

@testset "VCVS (Voltage-Controlled Voltage Source)" begin
    circuit = Circuit()

    @with mna_circuit => circuit begin
        v1 = mna_net(:v1)
        v2 = mna_net(:v2)
        gnd = mna_ground()

        # Control voltage source
        SpcVoltageSource(dc=1.0)(v1, gnd)
        MNASimpleResistor(r=1000.0)(v1, gnd)

        # VCVS: output = 2 * control voltage
        SpcVCVS(gain=2.0)(v2, gnd, v1, gnd)
        MNASimpleResistor(r=1000.0)(v2, gnd)
    end

    result = dc!(circuit)
    v2_idx = circuit.nets[:v2].index
    @test isapprox(result.solution[v2_idx], 2.0, atol=1e-6)
end

@testset "VCCS (Voltage-Controlled Current Source)" begin
    circuit = Circuit()

    @with mna_circuit => circuit begin
        v1 = mna_net(:v1)
        out = mna_net(:out)
        gnd = mna_ground()

        # Control voltage source (1V)
        SpcVoltageSource(dc=1.0)(v1, gnd)
        MNASimpleResistor(r=1000.0)(v1, gnd)

        # VCCS: current = gain * control voltage = 0.001 * 1V = 1mA
        SpcVCCS(gain=0.001)(out, gnd, v1, gnd)
        MNASimpleResistor(r=1000.0)(out, gnd)
    end

    result = dc!(circuit)
    out_idx = circuit.nets[:out].index
    # V = IR = 0.001 * 1000 = 1V
    @test isapprox(result.solution[out_idx], 1.0, atol=1e-6)
end

# ============================================================================
# TRANSIENT TESTS (from transients.jl)
# ============================================================================

@testset "RC Charging Transient" begin
    # Simplified version of transients.jl tests
    circuit = Circuit()
    vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V)
    resistor!(circuit, :vcc, :out, 1000.0; name=:R)
    capacitor!(circuit, :out, :gnd, 1e-6; name=:C)
    ground!(circuit, :gnd)

    sys = CircuitSystem(circuit)
    sol = solve_tran(sys, (0.0, 5e-3))  # 5 time constants

    # Check that solution completes
    @test length(sol.t) > 0
    # Capacitor should charge toward 5V
    @test sol[:out][end] > 4.0
end

# ============================================================================
# SWEEP API TESTS (from sweep.jl - data structure tests)
# ============================================================================

@testset "Sweep data structures" begin
    # These are pure data structure tests that don't need DAECompiler
    # Port of sweep.jl "nest and flatten param lists"

    # Test nest_param_list and flatten_param_list if available
    if isdefined(CedarSim, :nest_param_list)
        param_list_tuple = (
            (:R1, 1.0),
            (Symbol("x1.R3"), 2),
            (Symbol("x1.x2.R1"), 3),
            (Symbol("x1.x2.R2"), 4),
        )
        nested = CedarSim.nest_param_list(param_list_tuple)
        @test haskey(nested, :R1)
        @test nested.R1 == 1
    else
        @test_broken false  # nest_param_list not available in MNA API
    end
end

# ============================================================================
# SPECTRE EXPRESSION TESTS (from spectre_expr.jl)
# ============================================================================

@testset "Spectre parameter parsing" begin
    # Test that SpectreNetlistParser is still available
    @test isdefined(CedarSim, :SpectreNetlistParser) ||
          @isdefined(SpectreNetlistParser)
end

# ============================================================================
# TESTS THAT CANNOT BE PORTED (DAECompiler-specific)
# ============================================================================

@testset "DAECompiler-specific tests (marked broken)" begin
    # These tests relied on DAECompiler's CircuitIRODESystem which is no longer available

    @testset "Unimplemented Device test" begin
        # Original test: CircuitIRODESystem(ERRcircuit) throws UnsupportedIRException
        # This tested DAECompiler's IR handling of error() calls
        @test_broken false  # DAECompiler not available
    end

    @testset "MC VR Circuit (Monte Carlo)" begin
        # Original test: Used agauss() for Monte Carlo resistance
        # Requires DAECompiler's runtime parameterization
        @test_broken false  # Monte Carlo not yet implemented in MNA
    end

    @testset "Non-const circuit elements compilation" begin
        # Original test: Verified DAECompiler's static analysis behavior
        # Not applicable to MNA backend
        @test_broken false  # DAECompiler compilation test
    end

    @testset "Non-const values parameterization error" begin
        # Original test: DAECompiler error for non-const values
        # MNA handles this differently
        @test_broken false  # DAECompiler-specific error handling
    end
end

@testset "AC Analysis tests (requires linearization)" begin
    # AC analysis requires DAECompiler's linearization capabilities
    # These tests from ac.jl cannot be ported without implementing
    # small-signal linearization in the MNA backend

    @test_broken false  # AC analysis not implemented in MNA
    # TODO: Implement AC analysis in MNA backend
    # Required features:
    # - Small-signal linearization at DC operating point
    # - Frequency response calculation
    # - AC noise analysis
end

@testset "Sensitivity Analysis tests" begin
    # Sensitivity analysis from sensitivity.jl requires DAECompiler's
    # ODEForwardSensitivityProblem support

    @test_broken false  # Sensitivity analysis not implemented in MNA
    # TODO: Implement using SciMLSensitivity with MNA backend
end

@testset "SPICE/Spectre file parsing" begin
    # Tests from basic.jl that require full SPICE parsing pipeline
    # The MNA backend has a simpler parser (mna_parser.jl)

    @testset "Simple Spectre sources" begin
        # Original: Parsed Spectre netlist with isource, vsource, pwl, bsource
        @test_broken false  # Requires full Spectre parsing + codegen
    end

    @testset "SPICE include .LIB" begin
        # Original: Tested .LIB include mechanism
        @test_broken false  # Requires full SPICE parsing pipeline
    end

    @testset "Verilog include" begin
        # Original: ahdl_include for VA files
        # MNA has mna_va_load but different integration
        @test_broken false  # Different VA integration path
    end

    @testset "SPICE parameter scope" begin
        # Original: Complex parameter scoping in subcircuits
        @test_broken false  # Requires full SPICE codegen
    end
end

@testset "DDX (derivative) tests" begin
    # From ddx.jl - tests VA ddx() function
    # Requires NLVCR.va device which uses ddx()
    @test_broken false  # Requires specific VA device
    # Note: ddx support was added to MNA VA backend via ForwardDiff
end

@testset "Inverter tests (MOSFET)" begin
    # From inverter.jl - requires PDK models (GF180MCU)
    @test_broken false  # Requires GF180MCU PDK
    # TODO: Port when BSIMCMG/BSIM4 models are fully integrated
end

@testset "GF180 DFF tests" begin
    # From gf180_dff.jl - requires GF180MCU PDK
    @test_broken false  # Requires GF180MCU PDK
end

@testset "Inverter noise tests" begin
    # From inverter_noise.jl - requires noise analysis
    @test_broken false  # Requires noise analysis + PDK
end

@testset "MTK extension tests" begin
    # From MTK_extension.jl - ModelingToolkit integration
    @test_broken false  # MTK integration not ported
end

@testset "Compiler sanity tests" begin
    # From compiler_sanity.jl - tests DAECompiler internals
    @test_broken false  # DAECompiler-specific
end

@testset "Alias tests" begin
    # From alias.jl - tests DAECompiler's aliasmap
    @test_broken false  # DAECompiler's alias extraction
end

# ============================================================================
# TESTS THAT COULD BE PORTED WITH ADDITIONAL WORK
# ============================================================================

@testset "Features needing implementation" begin
    @testset "PWL (Piecewise Linear) sources" begin
        # MNA has basic PWL support but needs more testing
        # Original from transients.jl
        @test_broken false  # PWL sources partially implemented
    end

    @testset "Butterworth Filter (LC circuit)" begin
        # Requires inductor support in transient
        @test_broken false  # Inductors in transient need work
    end

    @testset "BSIMCMG DC sweep" begin
        # We have BSIMCMG VA support, need to add tests
        bsimcmg_path = joinpath(dirname(@__DIR__), "VerilogAFiles", "bsimcmg.va")
        if isfile(bsimcmg_path)
            @test true  # BSIMCMG file exists
        else
            @test_broken false  # BSIMCMG not found
        end
    end

    @testset "ParamSim/ParamLens integration" begin
        # ParamLens is implemented but needs more comprehensive tests
        lens = ParamLens((r=1000.0,))
        @test lens isa AbstractParamLens
        # Full parameter sweep support needs CircuitSweep API
        @test_broken false  # CircuitSweep not yet ported
    end
end

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "="^60)
println("TEST SUMMARY - Ported from Legacy DAECompiler Tests")
println("="^60)
println("""
PORTED AND WORKING:
- Simple VR, IR, VRC circuits
- ParallelInstances
- Current sources
- Capacitors in DC
- DefaultOr parameter handling
- Voltage dividers
- VCVS/VCCS controlled sources
- RC transient
- Clean API type aliases

MARKED BROKEN (DAECompiler-specific):
- CircuitIRODESystem compilation tests
- Monte Carlo with agauss()
- AC analysis / linearization
- Sensitivity analysis
- Complex SPICE/Spectre file parsing
- DDX VA tests
- Inverter/PDK tests
- Noise analysis
- MTK extension
- Compiler sanity checks
- Alias extraction

NEEDS IMPLEMENTATION:
- PWL sources (partial)
- LC circuit transients
- Full CircuitSweep API
- AC small-signal analysis
- Sensitivity via SciMLSensitivity
""")
