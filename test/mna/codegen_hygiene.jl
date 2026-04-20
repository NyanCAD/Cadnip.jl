module codegen_hygiene_tests

include(joinpath(@__DIR__, "..", "common.jl"))

using Test
using Cadnip
using Cadnip.MNA: MNACircuit, MNASpec

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

end # module
