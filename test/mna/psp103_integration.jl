#!/usr/bin/env julia
#==============================================================================#
# PSP103VA Integration Tests
#
# Tests that PSP103VA model works correctly through the SPICE parsing and
# MNA codegen path. Key challenges addressed:
#
# 1. LLVM crashes on very large functions (96K+ IR statements) during compilation
#    - Fix: Use `invokelatest` for VA model stamp! calls in generated code
#    - This forces runtime dispatch to precompiled stamp! methods
#
# 2. Very large structs (782 fields) cause LLVM SROA to explode
#    - Fix: Use `inferencebarrier` to hide exact type from Julia compiler
#
# 3. Internal node naming collisions in subcircuits
#    - Fix: Use hierarchical instance names (prefix_localname) for internal nodes
#==============================================================================#

using CedarSim
using CedarSim.SpectreNetlistParser
using CedarSim.MNA: MNAContext, MNASpec, solve_dc, voltage, current, n_internal_nodes
using PSPModels
using Test

@testset "PSP103VA Integration" begin

    @testset "Basic DC with defaults" begin
        # Simple NMOS IV test with minimal parameters
        netlist = """
* PSP103VA NMOS IV test with defaults
.model nch psp103va type=1
M1 d g 0 0 nch W=10u L=1u
Vds d 0 DC 1.2
Vgs g 0 DC 0.6
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)
        @test circuit_fn !== nothing

        spec = MNASpec(temp=27.0, mode=:dcop)
        sol = solve_dc(circuit_fn, (;), spec)

        @test isapprox(voltage(sol, :d), 1.2, atol=1e-6)
        @test isapprox(voltage(sol, :g), 0.6, atol=1e-6)

        # Drain current should be reasonable (100s of µA for these bias conditions)
        Id = current(sol, :I_vds)
        @test abs(Id) > 100e-6 && abs(Id) < 1e-3
    end

    @testset "DC with full model parameters" begin
        # Test with many model parameters (subset of full VACASK model card)
        netlist = """
* PSP103VA with full model parameters
.model nch psp103va
+    type=1
+    tr=27.0
+    vfbo=-1.1
+    vfbl=0
+    vfbw=0
+    stvfbo=5.0e-4
+    toxo=1.5e-9
+    epsroxo=3.9
+    nsubo=3.0e+23
+    nsubw=0
+    wseg=1.5e-10
+    npck=1.0e+24
+    npckw=0
+    wsegp=0.9e-8
+    lpck=5.5e-8
+    lpckw=0
+    fol1=2.0e-2
+    fol2=5.0e-6
+    facneffaco=0.8
+    gfacnudo=0.1
+    npo=1.5e+26
+    npl=10.0e-18
+    cto=5.0e-15
+    ctl=4.0e-2
+    ctlexp=0.6
+    toxovo=1.5e-9
+    toxovdo=2.0e-9
+    lov=10.0e-9
+    lovd=0
+    wot=0
+    thesato=0.5
+    mueo=0.5
+    mue=0.5

M1 d g 0 0 nch W=10u L=1u
Vds d 0 DC 1.2
Vgs g 0 DC 0.6
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)
        @test circuit_fn !== nothing

        spec = MNASpec(temp=27.0, mode=:dcop)
        sol = solve_dc(circuit_fn, (;), spec)

        @test isapprox(voltage(sol, :d), 1.2, atol=1e-6)
        @test isapprox(voltage(sol, :g), 0.6, atol=1e-6)

        # Current should be reasonable
        Id = current(sol, :I_vds)
        @test abs(Id) > 10e-6 && abs(Id) < 10e-3
    end

    @testset "Internal nodes with subcircuits" begin
        # Test that internal nodes are correctly allocated with hierarchical naming
        # Each PSP103VA device has 8 internal nodes - with subcircuits, these must
        # have unique names (xu1_xmn_nm_PSP103VA_BD, etc.) not just (nm_PSP103VA_BD)
        netlist = """
* Two-transistor test through subcircuits
.model nch psp103va type=1
.model pch psp103va type=-1

.subckt nmos d g s b
  nm d g s b nch W=10u L=1u
.ends

.subckt pmos d g s b
  nm d g s b pch W=20u L=1u
.ends

.subckt inverter in out vdd vss
  xmp out in vdd vdd pmos
  xmn out in vss vss nmos
.ends

xu1 in1 out1 vdd 0 inverter
xu2 in2 out2 vdd 0 inverter

Vin1 in1 0 DC 0.6
Vin2 in2 0 DC 0.3
Vdd vdd 0 DC 1.2
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)

        # Build structure and check internal nodes
        spec = MNASpec(temp=27.0, mode=:dcop)
        ctx = circuit_fn((;), spec, 0.0)

        # 2 inverters × 2 MOSFETs × 8 internal nodes = 32 internal nodes
        n_internal = n_internal_nodes(ctx)
        @test n_internal == 32  # Each PSP103VA has 8 internal nodes

        # Check that internal node names are hierarchical (not colliding)
        internal_names = [name for (name, idx) in ctx.node_to_idx if ctx.internal_node_flags[idx]]

        # Should have unique prefixes for each instance path
        @test any(contains(String(n), "xu1_xmn_nm") for n in internal_names)
        @test any(contains(String(n), "xu1_xmp_nm") for n in internal_names)
        @test any(contains(String(n), "xu2_xmn_nm") for n in internal_names)
        @test any(contains(String(n), "xu2_xmp_nm") for n in internal_names)
    end

    # NOTE: DC solve with subcircuits requires DirectStampContext stamp! precompilation
    # in PSPModels. Currently triggers LLVM SROA crash during Newton iteration.
    # This is a known limitation - PSPModels needs PrecompileTools workload update.
end
