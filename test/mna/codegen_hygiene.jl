module codegen_hygiene_tests

include(joinpath(@__DIR__, "..", "common.jl"))

using Test
using Cadnip
using Cadnip.MNA: MNACircuit, MNASpec
using VADistillerModels   # `.model dmod d` → sp_diode, for the `.model` decks below

# Regression test: generated SPICE/VA code must not leak Cadnip-internal identifiers
# into the target module's namespace. A SPICE subckt parameter named `stamp` would
# previously shadow the bare `stamp!(...)` emission inside generated code.
#
# After the codegen hygiene pass, Type-B references are emitted fully-qualified
# (`Cadnip.MNA.stamp!(...)`, `Cadnip.MNA.get_node!(...)`, `Base.error(...)`, etc.),
# so user identifiers are free to use those names.

@testset "Codegen hygiene: structural — no bare Type-B emissions" begin

    @testset "SPICE generated code contains no bare stamp!" begin
        ast = Cadnip.NyanSpectreNetlistParser.parse(IOBuffer("""
        * test
        V1 vcc 0 DC 5
        R1 vcc 0 1k
        C1 vcc 0 1p
        """); start_lang=:spice, implicit_title=true)
        code = Cadnip.make_mna_circuit(ast; circuit_name=:hygiene_test)
        code_str = string(code)
        # After hygiene: no unqualified `stamp!(`, `get_node!(`, `pwl_at_time(`,
        # `get_current_idx(`. All must appear as `Cadnip.MNA.xxx` or via GlobalRef.
        @test !occursin(r"(?<![.A-Za-z])stamp!\(", code_str)
        @test !occursin(r"(?<![.A-Za-z])get_node!\(", code_str)
    end

    @testset "SPICE subckt with parameter named `stamp` resolves correctly" begin
        # Regression: if `stamp!` were emitted bare, this subckt would fail
        # because the parameter `stamp` would shadow the function at call time.
        circuit = MNACircuit(sp"""
        * param-name-collision regression
        .subckt myres a b stamp=2000.0
        R1 a b 'stamp'
        .ends
        V1 vcc 0 DC 1.0
        X1 vcc 0 myres stamp=1500.0
        """)
        sol = dc!(circuit)
        # I = V/R = 1V / 1500Ω
        @test isapprox(sol[:I_v1], -1.0/1500.0; atol=1e-8)
    end

    @testset "SPICE subckt with parameter named `get_node`" begin
        # Additional regression — `get_node!` emitted bare would collide here.
        circuit = MNACircuit(sp"""
        * another collision regression
        .subckt myres2 a b get_node=3000.0
        R1 a b 'get_node'
        .ends
        V1 vcc 0 DC 1.0
        X1 vcc 0 myres2 get_node=2500.0
        """)
        sol = dc!(circuit)
        @test isapprox(sol[:I_v1], -1.0/2500.0; atol=1e-8)
    end

end

# A netlist macro compiles its deck into a module at expansion time and expands
# to the builder object, so the call site holds a constant rather than the
# `using`/`const`/`function` forms Julia only accepts at top level. That is what
# makes `sp"..."` legal inside a function body.

@testset "Netlist macros expand inside a function body" begin

    @testset "structural — the deck's own code needs no import list" begin
        # A deck module used to open with `using Cadnip.SpectreEnvironment`.
        # Everything is named by `GlobalRef` or interpolated value now, and a
        # generated block with no imports is one nothing has to be in scope for.
        ast = Cadnip.NyanSpectreNetlistParser.parse(IOBuffer("""
        * a deck with a model card, a subckt, and an environment name
        .param k = 'M_1_PI'
        .subckt myres a b rval=1k
        R1 a b 'rval'
        .ends
        V1 vin 0 DC 'k'
        R1 vin out 1k
        D1 out 0 dmod
        X1 out 0 myres
        .model dmod d is=76.9p n=1.45
        """); start_lang=:spice, implicit_title=true)
        code = Cadnip.make_mna_circuit(ast; circuit_name=:no_imports_test)
        heads = Set{Symbol}()
        walk(x) = x isa Expr && (push!(heads, x.head); foreach(walk, x.args))
        walk(code)
        @test !(:using in heads)
        @test !(:import in heads)
    end

    @testset "the macro expands to a builder, not to a block of definitions" begin
        # `@macroexpand` on a netlist macro yields the builder object itself.
        expanded = @macroexpand Cadnip.@sp_str("V1 vcc 0 DC 5\nR1 vcc 0 1k\n", "i")
        @test !isa(expanded, Expr)
        @test isa(expanded, Function)
    end

    # Each of these builds *and* solves inside the one call. Before, the macro
    # expanded to a block carrying `using` statements and this was a syntax
    # error at parse time — the function could not even be defined.
    function _fn_spice()
        dc!(MNACircuit(sp"""
        V1 vcc 0 DC 5
        R1 vcc out 1k
        R2 out 0 1k
        """i))[:out]
    end

    function _fn_spectre()
        dc!(MNACircuit(spc"""
        v1 (vcc 0) vsource type=dc dc=5
        r1 (vcc out) resistor r=1k
        r2 (out 0) resistor r=1k
        """))[:out]
    end

    # `.model`, `.param` and `.subckt` are the three things codegen emits beside
    # the builder, and each was a separate reason the old expansion was
    # top-level-only.
    function _fn_model_param_subckt()
        c = MNACircuit(sp"""
        .param rload = 1k
        .subckt load a b r=1k
        Rl a b 'r'
        .ends
        V1 vin 0 DC 5
        R1 vin out 'rload'
        D1 out 0 dmod
        X1 out 0 load r=10k
        .model dmod d is=76.9p n=1.45
        """i)
        dc!(c)[:out]
    end

    # Two decks that each define `.subckt divider`, in one local scope. Each
    # expansion gets a deck module, so the second does not answer for the first.
    function _fn_two_decks()
        a = MNACircuit(sp"""
        .subckt divider p n
        R1 p n 1k
        .ends
        V1 vcc 0 DC 1
        X1 vcc 0 divider
        """i)
        b = MNACircuit(sp"""
        .subckt divider p n
        R1 p n 2k
        .ends
        V1 vcc 0 DC 1
        X1 vcc 0 divider
        """i)
        (dc!(a)[:I_v1], dc!(b)[:I_v1])
    end

    @testset "SPICE" begin
        @test isapprox(_fn_spice(), 2.5; atol=1e-9)
    end

    @testset "Spectre" begin
        @test isapprox(_fn_spectre(), 2.5; atol=1e-9)
    end

    @testset ".model / .param / .subckt" begin
        # Same deck at top level, for an answer to compare against.
        toplevel = MNACircuit(sp"""
        .param rload = 1k
        .subckt load a b r=1k
        Rl a b 'r'
        .ends
        V1 vin 0 DC 5
        R1 vin out 'rload'
        D1 out 0 dmod
        X1 out 0 load r=10k
        .model dmod d is=76.9p n=1.45
        """i)
        @test isapprox(_fn_model_param_subckt(), dc!(toplevel)[:out]; atol=1e-12)
    end

    @testset "two decks, one scope, same subckt name" begin
        @test all(isapprox.(_fn_two_decks(), (-1e-3, -5e-4); atol=1e-9))
    end

end

@testset "SpectreEnvironment names resolve without a using" begin

    @testset "an environment constant is reachable from SPICE" begin
        # SPICE lowercases identifiers, so this arrives at codegen as `m_1_pi`
        # while the binding is `M_1_PI`; the lookup is case-insensitive, the
        # same rule function calls already followed.
        c = MNACircuit(sp"""
        .param k = 'M_1_PI'
        V1 vin 0 DC 'k'
        R1 vin 0 1k
        """i)
        @test isapprox(dc!(c)[:vin], 1/pi; atol=1e-12)
    end

    @testset "a .param outranks the environment binding of the same name" begin
        # `exp` is a SpectreEnvironment name. Declared here, it is a local of
        # the builder and the environment must not get a look in.
        c = MNACircuit(sp"""
        .param exp = 3.0
        V1 vin 0 DC 'exp*2'
        R1 vin 0 1k
        """i)
        @test isapprox(dc!(c)[:vin], 6.0; atol=1e-12)
    end

end

end # module
