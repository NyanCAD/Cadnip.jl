module introspect_tests

include("common.jl")

using Cadnip: netlist, InstanceRef, ParamRef, ModelRef, SubcktRef, NetRef, AmbiguousRef

using VADistillerModels   # so a `.model dmod d` card resolves to something

#==============================================================================#
# `netlist(circuit)` — what does this deck declare, and where is it written?
#
# The deck is loaded through the ordinary entry points (`sp"..."` and
# `MNACircuit(path)`), because the index has to survive the same codegen path a
# user's circuit takes, not a hand-assembled one.
#==============================================================================#

const divider = sp"""
* voltage divider
.param res=1k
V1 vcc 0 DC 5
R1 vcc out 'res'
R2 out 0 'res'
"""

@testset "the five kinds of name" begin
    nl = netlist(MNACircuit(divider))

    @testset "instance" begin
        r1 = nl.r1
        @test r1 isa InstanceRef
        @test r1.name === :r1
        @test r1.device === :Resistor
        @test r1.nets == [:vcc, :out]
        @test length(r1.cards) == 1
        @test occursin("R1 vcc out", r1.cards[1].text)
        # An `sp"..."` deck's lines are this file's own lines, so the deck's
        # order is what to assert on: R2 is written directly under R1.
        @test nl.r2.cards[1].line == r1.cards[1].line + 1
        @test String(r1.cards[1].file) == @__FILE__
    end

    @testset "parameter" begin
        res = nl.res
        @test res isa ParamRef
        @test res.name === :res
        @test occursin("1k", res.cards[1].text)
        # `.param res=1k` sits two lines above `V1`, one above `R1`'s source.
        @test res.cards[1].line == nl.v1.cards[1].line - 1
    end

    @testset "net" begin
        out = nl.out
        @test out isa NetRef
        # `out` is the tap: R1's far end and R2's near end.
        @test sort(out.instances) == [:r1, :r2]
        @test sort(nl.vcc.instances) == [:r1, :v1]
    end

    @testset "lookup is SPICE's, not Julia's" begin
        # SPICE names are case-insensitive, and the deck writes `R1`.
        @test nl.R1 === nl.r1 === nl[:R1]
        @test haskey(nl, :R1) && haskey(nl, :r1)
        @test !haskey(nl, :r3)
        @test nl.r1 in values(nl)
        @test :r1 in propertynames(nl)
        # The struct's own fields do not shadow the deck's names.
        @test_throws ArgumentError nl.entries
        @test_throws ArgumentError nl.nosuchname
        @test get(nl, :nosuchname, nothing) === nothing
    end
end

const with_model = sp"""
* diode with a model card
.model dmod d is=1e-14
V1 a 0 DC 0.7
D1 a 0 dmod
"""

@testset "model card" begin
    nl = netlist(MNACircuit(with_model))
    dmod = nl.dmod
    @test dmod isa ModelRef
    @test dmod.name === :dmod
    # Two-tier resolution picked a device for the card.
    @test dmod.device isa GlobalRef
    @test occursin("dmod", dmod.cards[1].text)
    @test nl.d1.device === :Diode
end

#==============================================================================#
# A subcircuit is a namespace of its own, and the index nests the same way.
#==============================================================================#

@testset "subcircuit, from a file" begin
    path = joinpath(@__DIR__, "mna", "fixtures", "subckt_collision", "tap_a.sp")
    nl = netlist(MNACircuit(path))

    div = nl.divider
    @test div isa SubcktRef
    @test div.ports == [:p, :out, :n]
    # A file-loaded deck points at the file, not at codegen.jl.
    @test endswith(String(div.cards[1].file), "tap_a.sp")
    @test div.cards[1].line == 2
    # The card is the line that names the subcircuit, not its whole body.
    @test div.cards[1].text == ".subckt divider p out n"

    # Inside it: the subcircuit's own r1, not the top level's.
    @test !haskey(nl, :r1)
    @test haskey(div, :r1)
    @test div[:r1].nets == [:p, :out]
    @test keys(div) === keys(div.index)

    # And the instance line that calls it.
    @test nl.x1 isa InstanceRef
    @test nl.x1.device === :SubcktCall
    @test nl.x1.nets == [:in, :vout, Symbol("0")]
end

#==============================================================================#
# One name, two kinds. This is the `.param x1` / `X1` collision the parameter
# lens has its own rule for — the index is how you see that it is there.
#==============================================================================#

const collision = sp"""
* a parameter and an instance sharing a name
.param x1=2.0
V1 vcc 0 DC 'x1'
R1 vcc 0 1k
X1 vcc 0 rdiv
.subckt rdiv a b
R3 a b 1k
.ends
"""

@testset "ambiguous name" begin
    nl = netlist(MNACircuit(collision))
    x1 = nl.x1
    @test x1 isa AmbiguousRef
    @test x1.name === :x1
    @test any(r -> r isa ParamRef, x1.refs)
    @test any(r -> r isa InstanceRef, x1.refs)
    # The subcircuit's own name is unambiguous, and the deck-level summary
    # still counts `x1` under both kinds.
    @test nl.rdiv isa SubcktRef
    summary = sprint(show, MIME"text/plain"(), nl)
    @test occursin("parameters", summary)
    @test occursin("subcircuits", summary)
end

#==============================================================================#
# Conditional definitions: one name, several cards. Nothing here decides which
# branch is live — the index reports what the deck wrote.
#==============================================================================#

const conditional = sp"""
* one name defined in two branches
.param switch=1
V1 vcc 0 DC 1
.if (switch == 1)
R1 vcc 0 1
.else
R1 vcc 0 2
.endif
"""

@testset "a name defined in two branches" begin
    nl = netlist(MNACircuit(conditional))
    @test length(nl.r1.cards) == 2
    @test occursin("definitions", sprint(show, MIME"text/plain"(), nl.r1))
    # The circuit still solves — introspection is a read, and changes nothing.
    @test isapprox(dc!(MNACircuit(conditional))[:I_v1], -1.0; atol=deftol*10)
end

#==============================================================================#
# Printing: the point of the whole thing is that it prints the card.
#==============================================================================#

@testset "show" begin
    nl = netlist(MNACircuit(divider))

    summary = sprint(show, MIME"text/plain"(), nl)
    @test occursin("SPICE netlist", summary)
    @test occursin("voltage divider", summary)      # the title line
    @test occursin("instances (3)", summary)
    @test occursin("parameters (1)", summary)

    card = sprint(show, MIME"text/plain"(), nl.r1)
    @test occursin("Instance", card)
    @test occursin("R1 vcc out", card)              # the netlist line, as written
    @test occursin(":$(nl.r1.cards[1].line)", card) # with file and line number
    @test occursin(basename(@__FILE__), card)
    @test occursin("vcc, out", card)                # and its nets

    @test occursin("Net out", sprint(show, MIME"text/plain"(), nl.out))
    @test sprint(show, nl.r1) == "Instance r1"
end

#==============================================================================#
# A builder that is not a netlist has no cards to point at, and says so.
#==============================================================================#

rc_builder(params, spec, t=0.0; x=Float64[], ctx=nothing) = MNAContext()

@testset "hand-written builder" begin
    @test_throws ArgumentError netlist(MNACircuit(rc_builder))
end

end # module
