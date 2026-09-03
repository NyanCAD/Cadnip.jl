# DC sensitivity analysis (`sens!`) — see doc/sensitivity_design.md.
#
# The reference every test here measures against is a brute-force re-solve: nudge
# the parameter, solve the whole operating point again, and difference the
# output. That is the definition of the quantity, computed the expensive way;
# `sens!` computes it from one adjoint solve, and the two have to agree. Where a
# circuit has a closed form (a resistive divider does), the analytic value is
# asserted too, so a bug that moved both methods the same way would still show.
module sensitivity

using Test
using Cadnip
using VADistillerModels                      # .model … d → the VA diode
using Cadnip.MNA: MNACircuit, MNAContext, MNASpec, alter, stamp!, get_node!,
                  reset_for_restamping!, Resistor, VoltageSource
using Cadnip: dc!, sens!, normalized_sensitivity

# The output's derivative taken the expensive way: two full DC solves per
# parameter. `sens!` never solves more than once, so this is an independent
# check, not a rearrangement of the same arithmetic.
function brute_force(circuit, output::Symbol, name::Symbol, value::Real; rel=1e-6)
    h = rel * abs(value)
    up = dc!(alter(circuit; (name => value + h,)...))[output]
    dn = dc!(alter(circuit; (name => value - h,)...))[output]
    return (up - dn) / 2h
end

const divider = sp"""
.title sens divider
.param rtop=1k
.param rbot=1k
V1 vcc 0 DC 5
R1 vcc out 'rtop'
R2 out 0 'rbot'
"""i

@testset "resistive divider: against the closed form" begin
    c = MNACircuit(divider)
    s = sens!(c, :out)

    # V(out) = 5·rbot/(rtop+rbot), so ∂/∂rtop = -5·rbot/(rtop+rbot)² and
    # ∂/∂rbot = +5·rtop/(rtop+rbot)². Both are 1.25 mV/Ω at 1k/1k.
    @test s[:rtop] ≈ -5 * 1000 / 2000^2 rtol=1e-6
    @test s[:rbot] ≈ +5 * 1000 / 2000^2 rtol=1e-6

    # ...and against the operating point re-solved from scratch.
    @test s[:rtop] ≈ brute_force(c, :out, :rtop, 1000.0) rtol=1e-5
    @test s[:rbot] ≈ brute_force(c, :out, :rbot, 1000.0) rtol=1e-5

    # The operating point it linearized at comes back with the result.
    @test s.op.converged
    @test s.op[:out] ≈ 2.5
    @test s.output === :out
end

@testset "the normalized column is per percent" begin
    c = MNACircuit(divider)
    s = sens!(c, :out)
    # A 1% rise in rtop (1000 → 1010) drops V(out) by ~12.5 mV.
    @test normalized_sensitivity(s, :rtop) ≈ s[:rtop] * 1000 / 100
    @test normalized_sensitivity(s, :rtop) ≈ -0.0125 rtol=1e-5

    exact = dc!(alter(c; rtop=1010.0))[:out] - 2.5
    @test normalized_sensitivity(s, :rtop) ≈ exact rtol=1e-2   # linear term of a 1% step

    @test Dict(normalized_sensitivity(s))[:rbot] ≈ normalized_sensitivity(s, :rbot)
end

@testset "an override is the value it differentiates at" begin
    # rtop is 3k here, not the netlist's 1k, so both sensitivities have to be
    # taken at the 3k/1k operating point.
    c = MNACircuit(divider; rtop=3e3)
    s = sens!(c, :out)
    @test s[:rtop] ≈ -5 * 1000 / 4000^2 rtol=1e-6
    @test s[:rbot] ≈ +5 * 3000 / 4000^2 rtol=1e-6
    # The reported value is the one in effect, so the normalized column scales
    # by 3k rather than by the netlist default.
    @test normalized_sensitivity(s, :rtop) ≈ s[:rtop] * 3000 / 100
end

@testset "a branch current is an output like any other" begin
    c = MNACircuit(divider)
    s = sens!(c, :I_v1)
    @test s[:rtop] ≈ brute_force(c, :I_v1, :rtop, 1000.0) rtol=1e-5
    @test s[:rbot] ≈ brute_force(c, :I_v1, :rbot, 1000.0) rtol=1e-5
    # I(V1) = -5/(rtop+rbot) in the MNA sign convention, so raising either
    # resistance moves the source current toward zero from below.
    @test s[:rtop] > 0
end

const rect = sp"""
.title sens rectifier
.param rs=1k
.param vin=5
.subckt divider a b rv=1k
R1 a mid 'rv'
R2 mid b 'rv'
.ends
V1 vcc 0 DC 'vin'
R1 vcc out 'rs'
D1 out 0 dmod
X1 vcc 0 divider rv=2k
.model dmod d is=76.9p n=1.45
"""i

@testset "nonlinear: the adjoint agrees with a re-solve" begin
    c = MNACircuit(rect)
    s = sens!(c, :out)

    # The diode makes V(out) a transcendental function of both; nothing here is
    # a rearrangement of a linear formula.
    @test s[:rs] ≈ brute_force(c, :out, :rs, 1000.0) rtol=1e-5
    @test s[:vin] ≈ brute_force(c, :out, :vin, 5.0) rtol=1e-5

    # A forward-biased diode holds the node: 5 V of supply moves it ~8.6 mV.
    @test 0 < s[:vin] < 0.05
    @test s[:rs] < 0
end

@testset "a subcircuit's parameters are reachable, dotted" begin
    c = MNACircuit(rect)
    s = sens!(c, :out)
    @test Symbol("x1.rv") in keys(s)

    # X1 hangs across the supply, so its divider ratio cannot move V(out)...
    @test isapprox(s[Symbol("x1.rv")], 0.0; atol=1e-12)
    # ...but it does move the current the supply delivers, and by the amount a
    # re-solve says.
    si = sens!(c, :I_v1)
    @test si[Symbol("x1.rv")] ≈
          brute_force(c, :I_v1, Symbol("x1.rv"), 2000.0) rtol=1e-5
    # Multiplicity is a parameter of the instance too, and doubling X1 doubles
    # the current it draws.
    @test si[Symbol("x1.m")] ≈ brute_force(c, :I_v1, Symbol("x1.m"), 1.0) rtol=1e-5
end

@testset "picking parameters, and naming one that does not exist" begin
    c = MNACircuit(rect)
    all_of_them = sens!(c, :out)
    @test Set(keys(all_of_them)) ==
          Set([:rs, :vin, Symbol("x1.rv"), Symbol("x1.m")])

    just_one = sens!(c, :out; params=[:rs])
    @test keys(just_one) == [:rs]
    @test just_one[:rs] == all_of_them[:rs]

    err = try; sens!(c, :out; params=[:vbais]); catch e; e; end
    @test err isa ArgumentError
    @test occursin("unknown parameter `vbais`", err.msg)
    @test occursin("rs", err.msg)          # names what it does have
end

@testset "outputs that are not outputs" begin
    c = MNACircuit(divider)
    @test_throws ErrorException sens!(c, :gnd)
    @test_throws ErrorException sens!(c, :nowhere)
end

@testset "a capacitor does not move the operating point" begin
    c = MNACircuit(sp"""
    .title sens rc
    .param rtop=1k
    .param cload=1u
    V1 vcc 0 DC 5
    R1 vcc out 'rtop'
    R2 out 0 1k
    C1 out 0 'cload'
    """i)
    s = sens!(c, :out)
    # DC sensitivity differentiates the operating point, where C carries no
    # current — so this is zero, not small.
    @test s[:cload] == 0.0
    @test s[:rtop] ≈ -5 * 1000 / 2000^2 rtol=1e-6
end

@testset "a floating island does not sink the adjoint" begin
    # R3 hangs between two nodes that reach nothing, so `G` is singular — the
    # operating point still solves (zeros satisfy it), and the gmin shunt is
    # what keeps the adjoint solvable, exactly as it does for `ac!`.
    c = MNACircuit(sp"""
    .title sens floating
    .param rtop=1k
    .param riso=1k
    V1 vcc 0 DC 5
    R1 vcc out 'rtop'
    R2 out 0 1k
    R3 b c 'riso'
    """i)
    @test dc!(c)[:b] == 0.0

    s = sens!(c, :out)
    @test s[:rtop] ≈ -5 * 1000 / 2000^2 rtol=1e-6
    @test isapprox(s[:riso], 0.0; atol=1e-12)

    # Turned off, there is no adjoint to be had, and the error says which knob.
    err = try; sens!(c, :out; gmin=0.0); catch e; e; end
    @test err isa ErrorException
    @test occursin("singular", err.msg)
end

@testset "a hand-written builder names its own parameters" begin
    # Nothing can enumerate what a plain-NamedTuple builder declares, so the
    # caller states the parameters and the values they sit at.
    function build_divider(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        p = merge((R1=1e3, R2=1e3), params)
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)
        stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
        stamp!(Resistor(p.R1), ctx, vcc, out)
        stamp!(Resistor(p.R2), ctx, out, 0)
        return ctx
    end

    c = MNACircuit(build_divider; R1=1e3, R2=2e3)
    @test dc!(c)[:out] ≈ 5 * 2 / 3

    s = sens!(c, :out; params=(R1=1e3, R2=2e3))
    @test s[:R1] ≈ -5 * 2000 / 3000^2 rtol=1e-6
    @test s[:R2] ≈ +5 * 1000 / 3000^2 rtol=1e-6

    # Pairs spell the same thing as a NamedTuple.
    @test sens!(c, :out; params=[:R1 => 1e3])[:R1] ≈ s[:R1]

    # Without the values there is nothing to enumerate, and the error says so.
    err = try; sens!(c, :out); catch e; e; end
    @test err isa ArgumentError
    @test occursin("hand-written builder", err.msg)
end

@testset "a parameter sitting at zero still has a derivative" begin
    # V(out) = (5 + voff)/2, so ∂/∂voff = 0.5 — and `voff` is exactly zero, where
    # a step relative to the value would be no step at all.
    c = MNACircuit(sp"""
    .title sens offset
    .param voff=0
    V1 vcc 0 DC 5
    V2 mid vcc DC 'voff'
    R1 mid out 1k
    R2 out 0 1k
    """i)
    @test dc!(c)[:out] ≈ 2.5
    @test sens!(c, :out)[:voff] ≈ 0.5 rtol=1e-6
end

@testset "the table prints the three columns" begin
    c = MNACircuit(divider)
    txt = sprint(show, MIME"text/plain"(), sens!(c, :out))
    @test occursin("∂out / ∂p", txt)
    @test occursin("per %", txt)
    @test occursin("rtop", txt) && occursin("rbot", txt)
end

end # module
