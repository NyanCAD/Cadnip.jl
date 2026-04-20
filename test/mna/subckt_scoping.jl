module subckt_scoping_tests

include(joinpath(@__DIR__, "..", "common.jl"))

using Test
using Cadnip
using Cadnip.MNA: MNACircuit

# Regression test for subckt internal-node namespacing.
# Before the fix, two instances of the same subckt shared internal nodes in the
# flat MNAContext map — effectively shorting their internals together.
# After the fix, each instance's internal nodes are scoped by the instance name
# via the `_mna_prefix_` mechanism (e.g., :x1_mid, :x2_mid).

@testset "subckt scoping: flat, two instances" begin
    # Two 2+2=4Ω chains in series through :a → 8Ω total.
    # V_vcc=12 → I=1.5 A → V(a)=6, V(x1.mid)=9, V(x2.mid)=3.
    circuit = MNACircuit(spc"""
    subckt myres (p n)
        r1 (p mid) resistor r=2
        r2 (mid n) resistor r=2
    ends myres

    v1 (vcc 0) vsource dc=12
    x1 (vcc a) myres
    x2 (a 0) myres
    """)
    sol = dc!(circuit)

    @test Set(sol.node_names) == Set([:vcc, :a, :x1_mid, :x2_mid])
    @test isapprox(sol[:vcc], 12.0;   atol=1e-8)
    @test isapprox(sol[:a],    6.0;   atol=1e-8)
    @test isapprox(sol[:x1_mid], 9.0; atol=1e-8)
    @test isapprox(sol[:x2_mid], 3.0; atol=1e-8)
end

@testset "subckt scoping: nested, two outer × two inner" begin
    # outer contains xi1,xi2 → inner. Two outer instances xo1, xo2 share :a.
    # Each inner's :mid should be uniquely scoped down to e.g. :xo1_xi1_mid.
    # Resistor chain: 4 × 2Ω per outer = 8Ω, two outers in series = 16Ω.
    # V_vcc=16 → I=1 A → V(a)=8, each outer's inner-series midpoint…
    circuit = MNACircuit(spc"""
    subckt inner (p n)
        r1 (p mid) resistor r=1
        r2 (mid n) resistor r=1
    ends inner

    subckt outer (p n)
        xi1 (p mid) inner
        xi2 (mid n) inner
    ends outer

    v1 (vcc 0) vsource dc=16
    xo1 (vcc a) outer
    xo2 (a 0) outer
    """)
    sol = dc!(circuit)

    # Expected internal node names: top-level :vcc, :a, plus scoped nodes per (outer, inner).
    expected = Set([
        :vcc, :a,
        :xo1_mid, :xo1_xi1_mid, :xo1_xi2_mid,
        :xo2_mid, :xo2_xi1_mid, :xo2_xi2_mid,
    ])
    @test Set(sol.node_names) == expected

    # Voltage divider math:
    #   total R = 8 × 2Ω paths … wait, each inner is 1+1=2Ω, outer=2+2=4Ω, two outers in series → 8Ω.
    #   I = 16/8 = 2 A, V(a) = 16 - 4*2 = 8.
    #   xo1.mid is between xo1's two inner subckts → at V = 16 - 2*2 = 12.
    #   xo1.xi1.mid is the midpoint of the first inner in xo1 → (vcc + xo1_mid)/2 = 14.
    #   xo1.xi2.mid is midpoint of second inner in xo1 → (xo1_mid + a)/2 = 10.
    #   xo2.mid = (a + 0)/2 = 4.
    #   xo2.xi1.mid = (a + xo2_mid)/2 = 6.
    #   xo2.xi2.mid = (xo2_mid + 0)/2 = 2.
    @test isapprox(sol[:vcc], 16.0; atol=1e-8)
    @test isapprox(sol[:a],    8.0; atol=1e-8)
    @test isapprox(sol[:xo1_mid],     12.0; atol=1e-8)
    @test isapprox(sol[:xo1_xi1_mid], 14.0; atol=1e-8)
    @test isapprox(sol[:xo1_xi2_mid], 10.0; atol=1e-8)
    @test isapprox(sol[:xo2_mid],      4.0; atol=1e-8)
    @test isapprox(sol[:xo2_xi1_mid],  6.0; atol=1e-8)
    @test isapprox(sol[:xo2_xi2_mid],  2.0; atol=1e-8)
end

@testset "subckt scoping: device name= scoped too" begin
    # Two subckt instances, each with a named voltage source inside.
    # After scoping, the two current names should be :I_x1_vinner and :I_x2_vinner.
    circuit = MNACircuit(spc"""
    subckt arm (out)
        vinner (mid 0) vsource dc=3
        r (out mid) resistor r=1
    ends arm

    x1 (a) arm
    x2 (b) arm
    r_load1 (a 0) resistor r=1
    r_load2 (b 0) resistor r=1
    """)
    sol = dc!(circuit)

    @test :I_x1_vinner ∈ sol.current_names
    @test :I_x2_vinner ∈ sol.current_names
    # Each arm independently drives its load through 1Ω + 1Ω (series with internal v=3).
    # Steady state: v_internal holds mid=3 through 1Ω series + 1Ω load → V(a) = 1.5.
    @test isapprox(sol[:a], 1.5; atol=1e-8)
    @test isapprox(sol[:b], 1.5; atol=1e-8)
end

end # module
