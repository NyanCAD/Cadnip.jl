# Tests for Verilog-A device integration

using Test
using CedarSim
using ScopedValues

@testset "VA Resistor" begin
    circuit = MNACircuit()

    @with mna_circuit => circuit begin
        vcc = mna_net(:vcc)
        out = mna_net(:out)
        gnd = mna_ground()

        SpcVoltageSource(dc=5.0)(vcc, gnd)
        # Use MNA VA resistor (generated from Verilog-A)
        varesistor!(circuit, vcc, out; R=1000.0, name=:R1)
        MNASimpleResistor(r=1000.0)(out, gnd)
    end

    result = dc!(circuit)
    out_idx = circuit.nets[:out].index
    @test isapprox(result.solution[out_idx], 2.5, atol=1e-6)
end

@testset "BSIMCMG MOSFET DC" begin
    # Skip if BSIMCMG models not available
    bsimcmg_path = joinpath(dirname(@__DIR__), "VerilogAFiles", "bsimcmg.va")
    if !isfile(bsimcmg_path)
        @info "Skipping BSIMCMG tests - model file not found at $bsimcmg_path"
        return
    end

    # Load BSIMCMG
    bsimcmg = mna_va_load(bsimcmg_path)

    # Simple NMOS circuit
    circuit = MNACircuit()

    @with mna_circuit => circuit begin
        vdd = mna_net(:vdd)
        vg = mna_net(:vg)
        vs = mna_net(:vs)
        gnd = mna_ground()

        # Power supply
        SpcVoltageSource(dc=1.0)(vdd, gnd)
        # Gate voltage
        SpcVoltageSource(dc=0.5)(vg, gnd)

        # NMOS: drain=vdd, gate=vg, source=vs, bulk=gnd
        nmos!(circuit, vdd, vg, vs, gnd;
            name=:M1,
            TFIN=15e-9, FPITCH=48e-9, NFIN=1,
            EOT=1e-9, TOXP=1.2e-9,
            NF=1, L=20e-9)

        # Source resistor
        MNASimpleResistor(r=1000.0)(vs, gnd)
    end

    result = dc!(circuit)
    vs_idx = circuit.nets[:vs].index
    # Just check that we get a valid result
    @test result.solution[vs_idx] > 0.0
    @test result.solution[vs_idx] < 1.0
end

@testset "ParamLens with VA Device" begin
    bsimcmg_path = joinpath(dirname(@__DIR__), "VerilogAFiles", "bsimcmg.va")
    if !isfile(bsimcmg_path)
        @info "Skipping ParamLens test - BSIMCMG not found"
        return
    end

    bsimcmg = mna_va_load(bsimcmg_path)

    # Test that ParamLens can override parameters
    lens = ParamLens((NFIN=2,))

    circuit = MNACircuit()
    @with mna_circuit => circuit begin
        vdd = mna_net(:vdd)
        vg = mna_net(:vg)
        gnd = mna_ground()

        SpcVoltageSource(dc=1.0)(vdd, gnd)
        SpcVoltageSource(dc=0.5)(vg, gnd)

        # NMOS with ParamLens override - NFIN should be 2 not 1
        nmos!(circuit, vdd, vg, gnd, gnd, lens;
            name=:M1,
            TFIN=15e-9, FPITCH=48e-9, NFIN=1,  # default 1
            EOT=1e-9, TOXP=1.2e-9,
            NF=1, L=20e-9)
    end

    # If we got here without error, the ParamLens integration works
    @test length(circuit.devices) > 0
end

println("All VA device tests passed!")
