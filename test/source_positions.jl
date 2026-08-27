#=
Generated code points at the netlist that produced it.

A builder is written by `src/spc/codegen.jl`, so every `LineNumberNode` in it
used to be a position *in that file* — a `quote` block's own. An error raised
while a deck builds therefore named the compiler, never the card at fault.
Codegen now relocates each generated statement onto the card it came from,
which is what these two tests measure: the positions in the expression, and the
stack frame a real failure produces.
=#

using Test
using Cadnip
using Cadnip.MNA: MNACircuit
using VADistillerModels                     # .model … d → VA diode model
using NyanSpectreNetlistParser

# Every LineNumberNode in a generated expression, in no particular order.
function line_nodes(x, out=LineNumberNode[])
    if x isa LineNumberNode
        push!(out, x)
    elseif x isa Expr
        for a in x.args
            line_nodes(a, out)
        end
    end
    return out
end

# A deck whose line numbers are the whole point: the title comment is line 1, so
# a card written on the nth line below is line n.
const POSITIONS_DECK = """
* a deck whose line numbers are the point
.param rval=1k
.model dmod d is=1e-14
V1 in 0 DC 1
R1 in mid 'rval'
D1 mid 0 dmod
.subckt div a b rr=2k
R3 a b 'rr'
.ends
X1 mid 0 div
"""

@testset "generated code carries netlist positions" begin
    path = joinpath(mktempdir(), "positions.sp")
    write(path, POSITIONS_DECK)

    ast = NyanSpectreNetlistParser.parsefile(path)
    code = Cadnip.make_mna_circuit(ast; circuit_name=:positions_circuit)
    lnns = line_nodes(code)

    @test !isempty(lnns)
    # Nothing in a deck's builder points at the compiler that wrote it.
    @test all(l -> l.file === Symbol(path), lnns)

    lines = Set(l.line for l in lnns)
    @test 2 in lines    # .param rval
    @test 3 in lines    # .model dmod
    @test 4 in lines    # V1
    @test 5 in lines    # R1
    @test 6 in lines    # D1
    @test 7 in lines    # .subckt div — the builder's own frame
    @test 8 in lines    # R3, inside the subcircuit
    @test 10 in lines   # X1
end

@testset "a failing card names its own line" begin
    path = joinpath(mktempdir(), "badmodel.sp")
    write(path, """
    * a model parameter the device has never heard of
    V1 in 0 DC 1
    R1 in out 1k
    D1 out 0 dmod
    .model dmod d is=1e-14 nosuchparam=3
    """)

    err = nothing
    bt = nothing
    try
        MNACircuit(path)
    catch e
        err = e
        bt = catch_backtrace()
    end

    # `nosuchparam` reaches the device constructor as a keyword it has no field
    # for. What matters here is not the MethodError but where it is reported.
    @test err isa MethodError
    frames = Base.stacktrace(bt)
    @test any(f -> String(f.file) == path && f.line == 5, frames)
    @test !any(f -> occursin(joinpath("spc", "codegen.jl"), String(f.file)), frames)
end
