#==============================================================================#
# Named initial state: `u0` by name
#
# `dc!`, `tran!` and the problem constructors take their starting state either
# as the full solution vector or by the circuit's own names — what SPICE writes
# as `.nodeset` (a guess for the operating point) and `.ic` (the state the
# transient starts from). These tests drive netlists through the high-level API
# and assert on circuit behavior: which root of a two-solution circuit the
# operating point lands on, and where a transient starts from.
#==============================================================================#

module initial_state_tests

using Test
using Cadnip
using Cadnip.MNA: MNACircuit, CedarUICOp, named_u0, nameat, build_with_detection
using Cadnip: dc!, tran!, node_names, branch_names, state_index
using Cadnip: CircuitSweep, Sweep
using OrdinaryDiffEq: Rodas5P

# Two operating points, both real: (2 - V)/1 = V², i.e. V² + V - 2 = 0, so
# V(out) is either 1 V or -2 V. Newton from zeros finds 1; the -2 branch is
# reachable only by starting near it, which is exactly what `.nodeset` is for.
const bistable = sp"""
* two DC solutions
V1 vcc 0 DC 2
R1 vcc out 1
B1 out 0 i=V(out)**2
"""i

# An isolated capacitor: the operating point is 0 V everywhere, so anything the
# transient starts from other than 0 came from `u0`. τ = R·C = 1 ms.
const rc = sp"""
* rc decay
V1 vcc 0 DC 0
R1 vcc out 1k
C1 out 0 1u
"""i

@testset "state_index names the solution vector" begin
    sol = dc!(MNACircuit(bistable))

    # Every name the solution classifies indexes back to its own value.
    for name in vcat(node_names(sol), branch_names(sol))
        idx = state_index(sol, name)
        @test idx !== nothing
        @test sol.x[idx] == sol[name]
    end

    # Nodes come first, in order, then branch currents.
    @test state_index(sol, first(node_names(sol))) == 1
    @test state_index(sol, first(branch_names(sol))) == length(node_names(sol)) + 1

    # Ground has no row of its own, and neither has a name the circuit lacks.
    @test state_index(sol, :gnd) === nothing
    @test state_index(sol, Symbol("0")) === nothing
    @test state_index(sol, Symbol("gnd!")) === nothing
    @test state_index(sol, :nosuchnode) === nothing
end

@testset "named_u0 expands against the circuit's names" begin
    ctx = build_with_detection(MNACircuit(bistable))

    # `nothing` and a full vector pass through untouched.
    @test named_u0(ctx, nothing) === nothing
    full = collect(1.0:length(node_names(ctx)) + length(branch_names(ctx)))
    @test named_u0(ctx, full) === full

    # A named guess is partial: what it names takes its value, the rest is 0.
    u0 = named_u0(ctx, (out = -1.5,))
    @test u0[state_index(ctx, :out)] == -1.5
    @test count(!iszero, u0) == 1

    # The spellings agree.
    @test named_u0(ctx, Dict(:out => -1.5)) == u0
    @test named_u0(ctx, :out => -1.5) == u0
    @test named_u0(ctx, [:out => -1.5]) == u0
    @test named_u0(ctx, (:out => -1.5,)) == u0

    # Currents are states too, and so are named.
    @test named_u0(ctx, (vcc = 2.0, I_v1 = -1.0))[state_index(ctx, :I_v1)] == -1.0
end

@testset "an unreachable name throws, rather than doing nothing" begin
    circuit = MNACircuit(bistable)

    err = try dc!(circuit; u0=(nosuchnode = 1.0,)) catch e; e end
    @test err isa ArgumentError
    @test occursin("unknown initial-value name `nosuchnode`", err.msg)
    @test occursin("out", err.msg)          # it lists what the circuit does have

    # Ground is not a state: it is pinned at 0 V and has no row to seed.
    err = try dc!(circuit; u0=(gnd = 1.0,)) catch e; e end
    @test err isa ArgumentError
    @test occursin("ground is not a state variable", err.msg)

    # Same check on the transient path, which resolves at problem construction.
    @test_throws ArgumentError tran!(circuit, (0.0, 1e-6); u0=(nosuchnode = 1.0,))

    # And something that is neither a state vector nor a naming says so.
    @test_throws ArgumentError dc!(circuit; u0=5)
end

@testset "dc!: a nodeset picks the operating point" begin
    circuit = MNACircuit(bistable)

    # Cold, Newton finds the +1 V root.
    cold = dc!(circuit)
    @test cold.converged
    @test isapprox(cold[:out], 1.0; atol=1e-6)

    # Named near the other root, it finds that one instead. Only `out` is
    # named — `vcc` and `I_v1` start at zero, as a cold start leaves them.
    hinted = dc!(circuit; u0=(out = -1.5,))
    @test hinted.converged
    @test isapprox(hinted[:out], -2.0; atol=1e-6)
    @test isapprox(hinted[:vcc], 2.0; atol=1e-6)

    # A guess at the root it would have found anyway changes nothing.
    @test isapprox(dc!(circuit; u0=(out = 0.9,))[:out], 1.0; atol=1e-6)

    # A full vector still works: continuing from a solution reproduces it.
    @test isapprox(dc!(circuit; u0=hinted.x)[:out], -2.0; atol=1e-6)
end

@testset "a sweep follows the branch it was seeded on" begin
    # V² + V - vin = 0, so each point has two roots. Seeded onto the negative
    # one, continuation keeps the sweep there: -2 at vin=2, -3 at vin=6.
    swept = sp"""
    * two DC solutions, supply as a parameter
    .param vin=2
    V1 vcc 0 DC vin
    R1 vcc out 1
    B1 out 0 i=V(out)**2
    """i

    cs = CircuitSweep(swept, Sweep(vin = [2.0, 6.0]))

    cold = [sol[:out] for (_, sol) in dc!(cs)]
    @test isapprox(cold, [1.0, 2.0]; atol=1e-6)

    seeded = [sol[:out] for (_, sol) in dc!(cs; u0=(out = -1.5,))]
    @test isapprox(seeded, [-2.0, -3.0]; atol=1e-6)
end

@testset "tran!: the nodeset reaches the operating point the run starts at" begin
    circuit = MNACircuit(bistable)

    # The default initialization (CedarTranOp) solves the operating point, so
    # the guess it starts Newton from decides which branch the run sits on.
    cold = tran!(circuit, (0.0, 1e-6))
    @test isapprox(cold[:out][1], 1.0; atol=1e-6)

    hinted = tran!(circuit, (0.0, 1e-6); u0=(out = -1.5,))
    @test isapprox(hinted[:out][1], -2.0; atol=1e-6)

    # Same on the ODE path, which is a different initializer method.
    hinted_ode = tran!(circuit, (0.0, 1e-6); solver=Rodas5P(), u0=(out = -1.5,))
    @test isapprox(hinted_ode[:out][1], -2.0; atol=1e-6)
end

@testset "tran!: an initial condition is where the transient starts" begin
    circuit = MNACircuit(rc)

    # Nothing drives this circuit, so the operating point is 0 V: the default
    # run sits at zero for its whole span.
    quiet = tran!(circuit, (0.0, 3e-3))
    @test isapprox(quiet[:out][1], 0.0; atol=1e-9)

    # `.ic`: charge the capacitor to 1 V and let it decay. UIC relaxes the
    # algebraic constraints around the state given instead of solving for
    # equilibrium, which is what makes the initial condition survive.
    decay = tran!(circuit, (0.0, 3e-3); u0=(out = 1.0,), initializealg=CedarUICOp())
    @test isapprox(decay[:out][1], 1.0; atol=1e-6)
    # One time constant later, e⁻¹ of it is left.
    @test isapprox(nameat(decay, :out, 1e-3), exp(-1.0); rtol=1e-3)
    @test isapprox(nameat(decay, :out, 2e-3), exp(-2.0); rtol=1e-3)
end

@testset "a charged capacitor under a moving source" begin
    # UIC reads its derivative off a few fixed picosecond steps. When the deck's
    # sources are already moving at t=0 that estimate does not survive the DAE
    # consistency check that follows, and Shampine collocation — which refines
    # state and derivative together — is the documented fix.
    driven = MNACircuit(sp"""
    * driven rc, capacitor pre-charged
    V1 vcc 0 DC 0 SIN(0 1 100k)
    R1 vcc out 1k
    C1 out 0 1n
    """i)

    charged = tran!(driven, (0.0, 20e-6);
                    u0=(out = 0.5,), initializealg=CedarUICOp(use_shampine=true))
    @test isapprox(nameat(charged, :out, 0.0), 0.5; atol=1e-4)
end

end # module
