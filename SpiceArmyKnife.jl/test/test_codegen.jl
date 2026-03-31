# Test suite for SPICE/Spectre code generator

using Test
using SpiceArmyKnife
using NyanSpectreNetlistParser

"""
    check_roundtrip(code, simulator)

Parse code, generate output, parse again, generate again.
Check that the second generation matches the first (stability test).
"""
function check_roundtrip(code, simulator)
    # First round: parse original
    # Note: Use implicit_title=false since test cases don't have SPICE title lines
    lang = language(simulator)
    ast1 = if lang == :spice
        NyanSpectreNetlistParser.parse(IOBuffer(code); start_lang=:spice, implicit_title=false)
    else
        NyanSpectreNetlistParser.parse(IOBuffer(code); start_lang=:spectre)
    end
    gen1 = generate_code(ast1, simulator)

    # Second round: parse generated output
    ast2 = if lang == :spice
        NyanSpectreNetlistParser.parse(IOBuffer(gen1); start_lang=:spice, implicit_title=false)
    else
        NyanSpectreNetlistParser.parse(IOBuffer(gen1); start_lang=:spectre)
    end
    gen2 = generate_code(ast2, simulator)

    # Generated output should be stable (idempotent)
    @test gen1 == gen2

    return gen1 == gen2
end

@testset "Code Generator" begin

    @testset "Roundtrip Stability" begin
        # Note: These tests currently fail due to whitespace accumulation in terminals.
        # Each parse→generate cycle adds an extra space because terminals may contain
        # trivia (whitespace) from the source. This is a known issue but doesn't affect
        # semantic correctness - only formatting stability.
        # TODO: Strip all trivia from terminals or normalize whitespace in output

        @testset "SPICE roundtrip" begin
            spice = """
.subckt inv in out vdd gnd
M1 out in vdd vdd pmos w=2u l=100n
M2 out in gnd gnd nmos w=1u l=100n
.ends inv

.model nmos nmos level=14
R1 a b 1k
C1 c d 1p
X1 n1 n2 vdd gnd inv
"""
            @test check_roundtrip(spice, Ngspice())
        end

        @testset "Spectre roundtrip" begin
            spectre = """
subckt inv (in out vdd gnd)
M1 (out in vdd vdd) pmos w=2u l=100n
M2 (out in gnd gnd) nmos w=1u l=100n
ends inv

I1 (n1 n2 vdd gnd) inv
"""
            @test check_roundtrip(spectre, SpectreADE())
        end
    end

    @testset "SPICE Models" begin
        @testset "Simple model" begin
            spice = ".model nmos nmos level=14\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test output == ".model nmos nmos level=14\n"
        end

        @testset "Model with parameters" begin
            spice = ".model nmos nmos level=14 vto=0.7 kp=100u\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test output == ".model nmos nmos level=14 vto=0.7 kp=100u\n"
        end

        @testset "ngspice filters documentation parameters" begin
            # Model with documentation-only parameters (similar to Cordell models)
            # Changed values to avoid copyright - only parameter names matter
            spice = ".model testdiode D(Is=1n Rs=2.0 N=1.5 Cjo=3p M=.5 tt=10n Iave=100m Vpk=50 mfg=TEST001)\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())

            # Should contain standard parameters
            @test occursin("Is=1n", output)
            @test occursin("Rs=2.0", output)
            @test occursin("N=1.5", output)
            @test occursin("Cjo=3p", output)
            @test occursin("M=.5", output)
            @test occursin("tt=10n", output)

            # Should NOT contain documentation-only parameters
            @test !occursin("Iave", output)
            @test !occursin("Vpk", output)
            @test !occursin("mfg", output)
        end

        @testset "ngspice filters type and rating parameters" begin
            # BJT model with additional documentation parameters
            spice = ".model testbjt npn(Is=1e-15 BF=100 VAF=200 Vceo=100 Icrating=5 type=npn)\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())

            # Should contain standard parameters
            @test occursin("Is=1e-15", output)
            @test occursin("BF=100", output)
            @test occursin("VAF=200", output)

            # Should NOT contain documentation-only parameters
            @test !occursin("Vceo", output)
            @test !occursin("Icrating", output)
            @test !occursin("type=", output)
        end

        @testset "PSPICE temperature parameter conversion" begin
            @testset "ngspice converts PSPICE temperature parameters" begin
                # Model with PSPICE-specific temperature parameters (from MicroCap library)
                spice = ".model DBDMOD D IS=1E-015 N=0.5 T_MEASURED=25 T_ABS=25\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Ngspice())

                # Should contain standard parameters unchanged
                @test occursin("IS=1E-015", output)
                @test occursin("N=0.5", output)

                # PSPICE temperature parameters should be converted
                @test occursin("TNOM=25", output)    # T_MEASURED → TNOM
                @test occursin("TEMP=25", output)    # T_ABS → TEMP

                # Should NOT contain original PSPICE names
                @test !occursin("T_MEASURED", output)
                @test !occursin("T_ABS", output)
            end

            @testset "pspice preserves PSPICE temperature parameters" begin
                # Model with PSPICE-specific temperature parameters
                spice = ".model DBDMOD D IS=1E-015 N=0.5 T_MEASURED=25 T_ABS=25\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Pspice())

                # Should preserve PSPICE temperature parameter names
                @test occursin("T_MEASURED=25", output)
                @test occursin("T_ABS=25", output)

                # Should NOT have converted names
                @test !occursin("TNOM", output)
                @test !occursin("TEMP=", output)
            end

            @testset "T_REL_GLOBAL conversion" begin
                # Test T_REL_GLOBAL → dtemp conversion
                spice = ".model TESTMOD D IS=1E-12 T_REL_GLOBAL=5\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Ngspice())

                @test occursin("DTEMP=5", output)
                @test !occursin("T_REL_GLOBAL", output)
            end

            @testset "Mixed temperature and other parameters" begin
                # Real-world example from MicroCap vishaydiode.lib
                spice = ".MODEL RBVCMOD RES TC1=0.00107 T_MEASURED=25\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Ngspice())

                # Should preserve TC1 unchanged
                @test occursin("TC1=0.00107", output)
                # Should convert T_MEASURED
                @test occursin("TNOM=25", output)
                @test !occursin("T_MEASURED", output)
            end

            @testset "Case insensitive conversion" begin
                # Temperature parameter names should be case-insensitive
                spice = ".model TEST D t_abs=25 T_MEASURED=30 T_Rel_Global=5\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Ngspice())

                @test occursin("TEMP=25", output)
                @test occursin("TNOM=30", output)
                @test occursin("DTEMP=5", output)
            end
        end

        @testset "VACASK temperature parameter conversion" begin
            @testset "VACASK converts tref to tnom" begin
                # VACASK Verilog-A models use tnom as the primary parameter
                # ngspice models may use tref as a compatibility alias
                spice = ".model DMOD D IS=1E-12 TREF=27\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, VACASK())

                # Should convert TREF → tnom
                @test occursin("tnom=27", output)
                @test !occursin("tref", lowercase(output))
            end

            @testset "VACASK converts TREF (uppercase) to tnom" begin
                # Test case insensitive conversion
                spice = ".model DMOD D IS=1E-12 TREF=25\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, VACASK())

                @test occursin("tnom=25", output)
                @test !occursin("TREF", output)
            end

            @testset "VACASK preserves tnom unchanged" begin
                # If the model already uses tnom, keep it as-is
                spice = ".model DMOD D IS=1E-12 TNOM=30\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, VACASK())

                @test occursin("tnom=30", output)
            end

            @testset "VACASK converts tref in instances" begin
                # Test instance parameter conversion
                spice = "D1 1 0 DMOD TREF=27\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, VACASK())

                @test occursin("tnom=27", output)
                @test !occursin("tref", lowercase(output))
            end

            @testset "VACASK mixed parameters with tref" begin
                # Test that other parameters are preserved while tref is converted
                spice = ".model RMOD R RSH=10 TREF=25 TC1=0.001\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, VACASK())

                @test occursin("rsh=10", lowercase(output))
                @test occursin("tnom=25", output)
                @test occursin("tc1=0.001", lowercase(output))
                @test !occursin("tref", lowercase(output))
            end
        end

        @testset "Operator conversion" begin
            @testset "gnucap converts ** to pow()" begin
                # Gnucap does not support ** operator, requires pow() function
                spice = ".param x={2**3}\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Gnucap())

                # Should convert ** to pow()
                @test occursin("pow(2, 3)", output)
                @test !occursin("**", output)
            end

            @testset "ngspice preserves ** operator" begin
                # Ngspice supports ** operator natively
                spice = ".param x={2**3}\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Ngspice())

                # Should preserve ** operator
                @test occursin("**", output)
                @test !occursin("pow(", output)
            end

            @testset "gnucap converts ** in complex expressions" begin
                # Test ** conversion in nested expressions
                spice = ".param y={(a+b)**2}\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Gnucap())

                # Should convert ** to pow()
                # Note: Gnucap is Verilog-A based, so identifiers get backticks
                @test occursin("pow(", output)
                @test !occursin("**", output)
            end

            @testset "gnucap preserves other operators" begin
                # Test that other operators are not affected
                spice = ".param z={a+b*c/d}\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Gnucap())

                # Should preserve all operators
                @test occursin("+", output)
                @test occursin("*", output)
                @test occursin("/", output)
            end

            @testset "gnucap converts multiple ** operators" begin
                # Test expression with multiple power operations
                spice = ".param w={2**3 + 4**5}\n"
                ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
                output = generate_code(ast, Gnucap())

                # Both ** should be converted to pow()
                @test occursin("pow(2, 3)", output)
                @test occursin("pow(4, 5)", output)
                @test !occursin("**", output)
            end
        end
    end

    @testset "SPICE Subcircuits" begin
        @testset "Empty subcircuit" begin
            spice = ".subckt test a b\n.ends test\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            # Reconstructed code may have slightly different spacing
            @test occursin(".subckt test", output)
            @test occursin(".ends test", output)
        end

        @testset "Subcircuit with body" begin
            spice = ".subckt inv in out vdd gnd\nM1 out in vdd vdd pmos w=2u\n.ends inv\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            # Output will have reconstructed spacing
            @test occursin(".subckt inv", output)
            @test occursin("M1", output)
            @test occursin(".ends inv", output)
        end

        @testset "Subcircuit with parameters" begin
            spice = ".subckt res2 a b r=1k\nR1 a b r\n.ends\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test occursin("r=1k", output)
        end
    end

    @testset "SPICE Device Instances" begin
        @testset "Resistor" begin
            spice = "R1 in out 1k\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test output == "R1 in out 1k\n"
        end

        @testset "Capacitor" begin
            spice = "C1 n1 gnd 1p\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test occursin("C1", output)
            @test occursin("1p", output)
        end

        @testset "MOSFET" begin
            spice = "M1 d g s b nmos w=1u l=100n\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test occursin("M1", output)
            @test occursin("nmos", output)
            @test occursin("w=1u", output)
            @test occursin("l=100n", output)
        end

        @testset "Subcircuit call" begin
            spice = "X1 a b c d inverter\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test occursin("X1", output)
            @test occursin("inverter", output)
        end
    end

    @testset "Spectre Models" begin
        @testset "Simple model" begin
            spectre = "model nmos_mod bsim4 version=4.7\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spectre); start_lang=:spectre)
            output = generate_code(ast, SpectreADE())
            @test output == "model nmos_mod bsim4 version=4.7\n"
        end
    end

    @testset "Spectre Subcircuits" begin
        @testset "Subcircuit with nodes" begin
            spectre = "subckt inv (in out vdd gnd)\nends inv\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spectre); start_lang=:spectre)
            output = generate_code(ast, SpectreADE())
            @test occursin("subckt inv", output)
            # Spacing may vary in reconstruction
            @test occursin("(in", output) && occursin("gnd)", output)
            @test occursin("ends inv", output)
        end
    end

    @testset "Spectre Instances" begin
        @testset "Instance with parameters" begin
            spectre = "I1 (n1 n2 vdd gnd) inverter w=1u\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spectre); start_lang=:spectre)
            output = generate_code(ast, SpectreADE())
            @test occursin("I1", output)
            @test occursin("inverter", output)
            @test occursin("w=1u", output)
        end
    end

    @testset "Expressions" begin
        @testset "Binary expressions" begin
            spice = "R1 a b {2*rval}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test occursin("2 * rval", output) || occursin("2*rval", output)
        end

        @testset "Parameters with expressions" begin
            spice = "M1 d g s b nmos w={wval*2}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test occursin("wval", output)
        end
    end

    @testset "Comments and Formatting" begin
        @testset "Comments preserved" begin
            spice = "* This is a comment\nR1 a b 1k\n"
            # This test has a comment as first line, so it should use implicit_title=true
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=true)
            output = generate_code(ast, Ngspice())
            @test occursin("* This is a comment", output)
        end

        @testset "Blank lines handled" begin
            spice = "R1 a b 1k\n\nC1 c d 1p\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            @test occursin("R1", output)
            @test occursin("C1", output)
        end
    end

    @testset "Complex Circuits" begin
        @testset "Full SPICE circuit" begin
            spice = """
* RC Filter
.subckt rc_filter in out gnd r=1k c=1p
R1 in out r
C1 out gnd c
.ends rc_filter

.model nmos nmos level=14
M1 d g s b nmos w=1u
X1 a b gnd rc_filter
"""
            # This test has a title comment as first line, so use implicit_title=true
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=true)
            output = generate_code(ast, Ngspice())

            # Check all major components present
            @test occursin("RC Filter", output)
            @test occursin(".subckt rc_filter", output)
            @test occursin(".ends rc_filter", output)
            @test occursin(".model", output) && occursin("nmos", output)
            @test occursin("M1", output)
            @test occursin("X1", output)
        end
    end

    @testset "IOBuffer output" begin
        spice = "R1 a b 1k\n"
        ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)

        io = IOBuffer()
        generate_code(ast, io, Ngspice())
        output = String(take!(io))

        @test output == "R1 a b 1k\n"
    end

    @testset "Title and Brace handling" begin
        @testset "Title line preserved" begin
            spice = "* Test Circuit Title\nR1 a b 1k\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=true)
            output = generate_code(ast, Ngspice())
            @test occursin("Test Circuit Title", output)
        end

        @testset "Braced expressions" begin
            # SPICE allows braced expressions in parameter values
            spice = ".param test_val={2*3.14}\nR1 a b {test_val}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Ngspice())
            # Check for braced expression (may have spaces from expression handler)
            @test occursin(r"\{2\s*\*\s*3\.14\}", output)
            @test occursin("{test_val}", output)
        end
    end

    @testset "Verilog-A Code Generation" begin
        @testset "ngspice gauss 3-argument form" begin
            # Test gauss(nom, rvar, sigma) with division-by-zero protection
            spice = """
.subckt test_random in out
.param res={gauss(1000, 0.05, 3)}
R1 in out res
.ends test_random
"""
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Gnucap(), options=Dict{Symbol,Any}(:spice_dialect => :ngspice))

            # Should contain _rdist_seed parameter
            @test occursin("parameter integer _rdist_seed", output)

            # Should have ternary with division-by-zero protection
            @test occursin("((0.05 <= 0) || (3 <= 0) ? 1000 : \$rdist_normal", output)
            @test occursin("\$rdist_normal(_rdist_seed, 1000, (0.05 * 1000) / 3)", output)
        end

        @testset "ngspice agauss 3-argument form" begin
            # Test agauss(nom, avar, sigma) with division-by-zero protection
            spice = """
.subckt test_random2 in out
.param cap={agauss(1e-12, 1e-14, 3)}
C1 in out cap
.ends test_random2
"""
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Gnucap(), options=Dict{Symbol,Any}(:spice_dialect => :ngspice))

            # Should contain _rdist_seed parameter
            @test occursin("parameter integer _rdist_seed", output)

            # Should have ternary with division-by-zero protection
            @test occursin("((1e-14 <= 0) || (3 <= 0) ? 1e-12 : \$rdist_normal", output)
            @test occursin("\$rdist_normal(_rdist_seed, 1e-12, 1e-14 / 3)", output)
        end

        @testset "Xyce agauss 2-argument form" begin
            # Test Xyce AGAUSS(μ, α) with default n=1 (from IHP PDK)
            spice = """
.subckt test_xyce in out
.param rsh={agauss(100, 0.0833)}
R1 in out rsh
.ends test_xyce
"""
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Gnucap(), options=Dict{Symbol,Any}(:spice_dialect => :xyce))

            # Should use Xyce semantics with n default of 1
            @test occursin("((0.0833 <= 0) || (1 <= 0) ? 100 : \$rdist_normal", output)
            @test occursin("\$rdist_normal(_rdist_seed, 100, 0.0833 / 1)", output)
        end

        @testset "Xyce gauss 3-argument form" begin
            # Test Xyce GAUSS(μ, α, n) with explicit n
            spice = """
.subckt test_xyce_gauss in out
.param r={gauss(1000, 0.1, 3)}
R1 in out r
.ends test_xyce_gauss
"""
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Gnucap(), options=Dict{Symbol,Any}(:spice_dialect => :xyce))

            # Xyce semantics: stddev = (α * μ) / n
            @test occursin("((0.1 <= 0) || (3 <= 0) ? 1000 : \$rdist_normal", output)
            @test occursin("\$rdist_normal(_rdist_seed, 1000, (0.1 * 1000) / 3)", output)
        end

        @testset "division by zero protection with sigma=0" begin
            # Test the case from IHP PDK: gauss(1, 1, (mm_ok != 1 ? 0 : 1))
            spice = """
.subckt test_divzero in out
.param mm_ok=0
.param nsmm={gauss(1, 1, (mm_ok != 1 ? 0 : 1))}
R1 in out 1k
.ends test_divzero
"""
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Gnucap(), options=Dict{Symbol,Any}(:spice_dialect => :ngspice))

            # Should have protection: when sigma=0, returns nominal
            @test occursin("? 1 :", output)  # Ternary returns 1 when condition is true
        end

        @testset "_rdist_seed always added to modules" begin
            # Even modules without gauss/agauss should have _rdist_seed parameter
            spice = """
.subckt simple_res in out
R1 in out 1k
.ends simple_res
"""
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, Gnucap())

            # Should contain _rdist_seed parameter
            @test occursin("parameter integer _rdist_seed = 0", output)
        end
    end

    @testset "Device Type to Model Name Mapping" begin
        # Helper to create test scope
        function test_scope(sim=OpenVAF(); dialect=:ngspice)
            io = IOBuffer()
            options = Dict{Symbol, Any}(:spice_dialect => dialect)
            CodeGenScope{typeof(sim)}(io, 0, options)
        end

        @testset "BJT level mappings - OpenVAF (default)" begin
            @testset "Default Gummel-Poon (no level)" begin
                scope = test_scope()
                name, params = spice_device_type_to_model_name(scope, "NPN")
                @test name == "bjt"
                @test params[:type] == 1
            end

            @testset "NPN vs PNP polarity" begin
                scope = test_scope()
                name_npn, params_npn = spice_device_type_to_model_name(scope, "NPN")
                @test params_npn[:type] == 1

                name_pnp, params_pnp = spice_device_type_to_model_name(scope, "PNP")
                @test params_pnp[:type] == -1
            end

            @testset "Level 1 Gummel-Poon" begin
                scope = test_scope()
                name, params = spice_device_type_to_model_name(scope, "NPN", 1)
                @test name == "bjt"
                @test params[:type] == 1
            end

            @testset "ngspice VBIC levels" begin
                scope = test_scope(OpenVAF(); dialect=:ngspice)
                # Level 4
                name, params = spice_device_type_to_model_name(scope, "NPN", 4)
                @test name == "vbic_4T_et_cf"
                @test params[:type] == 1

                # Level 9
                name, params = spice_device_type_to_model_name(scope, "PNP", 9)
                @test name == "vbic_4T_et_cf"
                @test params[:type] == -1
            end

            @testset "Xyce VBIC levels" begin
                scope = test_scope(OpenVAF(); dialect=:xyce)
                # Level 11
                name, params = spice_device_type_to_model_name(scope, "NPN", 11)
                @test name == "vbic_4T_et_cf"
                @test params[:type] == 1

                # Level 12
                name, params = spice_device_type_to_model_name(scope, "PNP", 12)
                @test name == "vbic_4T_et_cf"
                @test params[:type] == -1
            end
        end

        @testset "BJT level mappings - Gnucap (sp_ prefix)" begin
            @testset "Default Gummel-Poon" begin
                scope = test_scope(Gnucap())
                name, params = spice_device_type_to_model_name(scope, "NPN")
                @test name == "sp_bjt"
                @test params[:type] == 1
            end

            @testset "Level 1" begin
                scope = test_scope(Gnucap())
                name, params = spice_device_type_to_model_name(scope, "PNP", 1)
                @test name == "sp_bjt"
                @test params[:type] == -1
            end
        end

        @testset "MOSFET level mappings" begin
            @testset "BSIM4 (level 14, 54)" begin
                scope = test_scope()
                name, params = spice_device_type_to_model_name(scope, "NMOS", 14)
                @test name == "bsim4"
                @test params[:TYPE] == 1

                name, params = spice_device_type_to_model_name(scope, "PMOS", 54)
                @test name == "bsim4"
                @test params[:TYPE] == -1
            end

            @testset "BSIM3 (level 8, 49)" begin
                scope = test_scope()
                name, params = spice_device_type_to_model_name(scope, "NMOS", 8)
                @test name == "bsim3"
                @test params[:TYPE] == 1

                name, params = spice_device_type_to_model_name(scope, "PMOS", 49)
                @test name == "bsim3"
                @test params[:TYPE] == -1
            end

            @testset "BSIMCMG (level 17, 72)" begin
                scope = test_scope()
                # Default version (107)
                name, params = spice_device_type_to_model_name(scope, "NMOS", 17)
                @test name == "bsimcmg107"
                @test params[:DEVTYPE] == 1

                name, params = spice_device_type_to_model_name(scope, "PMOS", 72, "107")
                @test name == "bsimcmg107"
                @test params[:DEVTYPE] == 0
            end

            @testset "MOSFET without level - fallback" begin
                scope = test_scope()
                # No error, just returns uppercase device type
                name, params = spice_device_type_to_model_name(scope, "NMOS")
                @test name == "NMOS"
                @test isempty(params)
            end

            @testset "Unsupported MOSFET level - fallback" begin
                scope = test_scope()
                # No error, just returns uppercase device type
                name, params = spice_device_type_to_model_name(scope, "NMOS", 99)
                @test name == "NMOS"
                @test isempty(params)
            end

            @testset "Unsupported BSIMCMG version - uses default mapping" begin
                scope = test_scope()
                # Version mismatch means it falls back to level-only match (without version constraint)
                name, params = spice_device_type_to_model_name(scope, "NMOS", 17, "200")
                @test name == "bsimcmg107"  # Still matches level 17 rule without version
                @test params[:DEVTYPE] == 1
            end
        end

        @testset "Passive devices - OpenVAF (no prefix)" begin
            scope = test_scope()

            @testset "Resistor" begin
                name, params = spice_device_type_to_model_name(scope, "R")
                @test name == "resistor"
                @test isempty(params)
            end

            @testset "Capacitor" begin
                name, params = spice_device_type_to_model_name(scope, "C")
                @test name == "capacitor"
                @test isempty(params)
            end

            @testset "Inductor" begin
                name, params = spice_device_type_to_model_name(scope, "L")
                @test name == "inductor"
                @test isempty(params)
            end

            @testset "Diode" begin
                name, params = spice_device_type_to_model_name(scope, "D")
                @test name == "diode"
                @test isempty(params)
            end
        end

        @testset "Passive devices - Gnucap (sp_ prefix)" begin
            scope = test_scope(Gnucap())

            @testset "Resistor" begin
                name, params = spice_device_type_to_model_name(scope, "R")
                @test name == "sp_resistor"
                @test isempty(params)
            end

            @testset "Diode" begin
                name, params = spice_device_type_to_model_name(scope, "D")
                @test name == "sp_diode"
                @test isempty(params)
            end
        end

        @testset "Case insensitivity" begin
            scope = test_scope(OpenVAF(); dialect=:ngspice)
            name_upper, params_upper = spice_device_type_to_model_name(scope, "NPN", 9)
            name_lower, params_lower = spice_device_type_to_model_name(scope, "npn", 9)
            @test name_upper == name_lower
            @test params_upper == params_lower
        end
    end

    @testset "VACASK Gaussian functions" begin
        @testset "gauss() maps to nominal value" begin
            # VACASK doesn't support gauss() - should return nominal value only
            spice = ".param r1={gauss(100, 5, 3)}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, VACASK())

            # Should have just the nominal value (100), not the full function
            @test occursin("100", output)
            @test !occursin("gauss", output)
        end

        @testset "agauss() maps to nominal value" begin
            # VACASK doesn't support agauss() - should return nominal value only
            spice = ".param cap={agauss(10p, 0.5p, 3)}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, VACASK())

            # Should have just the nominal value (10p), not the full function
            @test occursin("10p", output)  # VACASK is Spectre-style, preserves magnitude suffix
            @test !occursin("agauss", output)
        end

        @testset "aunif() maps to nominal value" begin
            # VACASK doesn't support aunif() - should return nominal value only
            spice = ".param tol={aunif(50, 5)}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, VACASK())

            # Should have just the nominal value (50), not the full function
            @test occursin("50", output)
            @test !occursin("aunif", output)
        end

        @testset "unif() maps to nominal value" begin
            # VACASK doesn't support unif() - should return nominal value only
            spice = ".param width={unif(2u, 0.1)}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, VACASK())

            # Should have just the nominal value (2u), not the full function
            @test occursin("2u", output)  # VACASK is Spectre-style, preserves magnitude suffix
            @test !occursin("unif", output)
        end

        @testset "limit() maps to nominal value" begin
            # VACASK doesn't support limit() - should return nominal value only
            spice = ".param vth={limit(0.7, 0.05)}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, VACASK())

            # Should have just the nominal value (0.7), not the full function
            @test occursin("0.7", output)
            @test !occursin("limit", output)
        end

        @testset "ngspice gauss with 1 arg defaults to 1.0" begin
            # Special case: ngspice gauss(sigma) defaults nominal=1.0
            spice = ".param factor={gauss(3)}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false, spice_dialect=:ngspice)
            output = generate_code(ast, VACASK(); options=Dict{Symbol, Any}(:spice_dialect => :ngspice))

            # Should use default nominal value 1.0
            @test occursin("1.0", output)
            @test !occursin("gauss", output)
        end

        @testset "Gaussian functions in model parameters" begin
            # Real-world example: model with statistical variation
            spice = ".model NMOS NMOS vto={gauss(0.7, 0.05, 3)} kp={agauss(100u, 10u, 3)}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, VACASK())

            # Should have nominal values only
            @test occursin("0.7", output)
            @test occursin("100u", output)  # VACASK is Spectre-style, preserves magnitude suffix
            @test !occursin("gauss", output)
            @test !occursin("agauss", output)
        end

        @testset "Other functions preserved" begin
            # Non-statistical functions should be preserved
            spice = ".param result={sqrt(100) + sin(3.14)}\n"
            ast = NyanSpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice, implicit_title=false)
            output = generate_code(ast, VACASK())

            # Should preserve regular function calls
            @test occursin("sqrt", output)
            @test occursin("sin", output)
        end
    end

end
