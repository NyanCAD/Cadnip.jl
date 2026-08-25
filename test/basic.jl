module basic_tests

include("common.jl")

using Cadnip.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using Cadnip.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using Cadnip.MNA: make_ode_problem
using Cadnip.MNA: terminal_currents, op_vars, nameat
using DiffEqBase: BrownFullBasicInit
using VADistillerModels                     # .model … d → VA diode model

@testset "Simple VR Circuit" begin
    # Simple V-R circuit using sp"..." macro
    circuit = MNACircuit(sp"""
    V1 vcc 0 DC 5
    R1 vcc 0 2
    """i)
    sol = dc!(circuit)

    # I = V/R = 5/2 = 2.5A
    R_v = sol[:vcc]
    R_i = -sol[:I_v1]  # Voltage source current is negative when sourcing (SPICE is case-insensitive)
    @test isapprox_deftol(R_v, 5.0)
    @test isapprox_deftol(R_i, 2.5)
end

@testset "Simple IR circuit" begin
    # Simple I-R circuit using sp"..." macro
    # Current source into resistor: V = IR = 5*2 = 10V
    circuit = MNACircuit(sp"""
    I1 0 icc DC 5
    R1 icc 0 2
    """i)
    sol = dc!(circuit)

    R_v = sol[:icc]
    @test isapprox_deftol(R_v, 10.0)
end

const v_val = 5.0
const r_val = 2000.0
const c_val = 1e-6
@testset "Simple VRC circuit" begin
    # Original tested RC transient with u0=[0.0]
    function VRCcircuit(params, spec)
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        vrc = get_node!(ctx, :vrc)
        stamp!(VoltageSource(v_val; name=:V), ctx, vcc, 0)
        stamp!(Resistor(r_val; name=:R), ctx, vcc, vrc)
        stamp!(Capacitor(c_val; name=:C), ctx, vrc, 0)
        return ctx
    end

    ctx = VRCcircuit((;), MNASpec(mode=:tran))
    sys = assemble!(ctx)

    # Simulate the RC circuit with capacitor starting at 0
    tau = r_val * c_val  # Time constant
    tspan = (0.0, 10 * tau)
    prob_data = make_ode_problem(sys, tspan)

    # Explicitly start with capacitor uncharged
    u0 = copy(prob_data.u0)
    vrc_idx = findfirst(n -> n == :vrc, sys.node_names)
    u0[vrc_idx] = 0.0

    f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                    jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
    prob = ODEProblem(f, u0, prob_data.tspan)
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(linsolve=KLUFactorization()); reltol=deftol, abstol=deftol,
                               initializealg=BrownFullBasicInit())

    # At t=0, capacitor voltage should be 0
    # At t=10τ, capacitor voltage approaches v_val (within ~0.005%)
    # Exact value: v_val * (1 - exp(-10)) ≈ 0.99995 * v_val
    c_v_start = sol.u[1][vrc_idx]
    c_v_end = sol.u[end][vrc_idx]
    @test isapprox_deftol(c_v_start, 0.0)
    @test isapprox(c_v_end, v_val; rtol=1e-3)  # Within 0.1% of final value

    # Current at start: I = V/R = v_val/r_val
    # Current at end: I ≈ 0 (capacitor nearly fully charged)
    # (Current through voltage source)
    I_V_idx = sys.n_nodes + findfirst(n -> n == :I_V, sys.current_names)
    c_i_start = -sol.u[1][I_V_idx]  # Negative because sourcing
    c_i_end = -sol.u[end][I_V_idx]
    @test isapprox_deftol(c_i_start, v_val/r_val)
    @test isapprox(c_i_end, 0.0; atol=1e-6)  # Nearly zero current
end

@testset "Simple Spectre sources" begin
    # Simple resistor divider in Spectre format
    spectre_code = """
    // Simple Spectre voltage divider
    v1 (vcc 0) vsource dc=5
    r1 (vcc out) resistor r=1k
    r2 (out 0) resistor r=1k
    """

    ctx, sol = solve_mna_spectre_code(spectre_code)
    # Voltage divider: 5V * (1k / (1k + 1k)) = 2.5V
    @test isapprox_deftol(sol[:out], 2.5)
    @test isapprox_deftol(sol[:vcc], 5.0)
end

@testset "Spectre current source" begin
    # Current source with resistor in Spectre format
    spectre_code = """
    // Spectre current source test
    i1 (vcc 0) isource dc=1m
    r1 (vcc 0) resistor r=1k
    """

    ctx, sol = solve_mna_spectre_code(spectre_code)
    # V = I * R = 1mA * 1kΩ = 1V
    @test isapprox_deftol(sol[:vcc], 1.0)
end

# Spectre `type=pwl wave=[...]`. Still disabled, but not for the reason the old
# comment gave ("requires transient simulation" — transient has worked for a
# while): sema cannot walk a `SpectreArray`, so the deck fails before codegen
# with `MethodError: no method matching sema_visit_ids!(…, ::SpectreArray, …)`.
# The `bsource v=$time*V(3)` half of this deck is independent of that.
#=
@testset "Full Spectre sources (transient)" begin
    # This is the comprehensive test from the old version
    # Tests PWL sources and B-source with time-varying expression
    mktempdir() do dir
        spectre_file = joinpath(dir, "sources.scs")
        open(spectre_file; write=true) do io
            write(io, \"\"\"
            I1 (0 1) isource dc=2.2u
            R1 (1 0) resistor r=1000

            I2 (0 2) isource type=pwl wave=[0 1m .5 2m 1 1.75m]
            R2 (2 0) resistor r=2k

            V3 (0 3) vsource dc=1.5
            R3 (3 0) resistor r=1k

            V4 (0 4) vsource type=pwl wave=[ 0 1 .5 2 \\
                    1 5]
            R4 (4 0) resistor r=4k

            B5 (0 5) bsource v=\$time*V(3)
            R5 (5 0) resistor r=1k
            \"\"\")
        end

        sys, sol = solve_spectre_file(spectre_file);
        @test all(isapprox.(sol[sys.node_1], 2.2e-3))
        @test all(isapprox.(sol[sys.R1.I], 2.2e-6))
        @test all(isapprox.(sol[sys.node_3], -1.5))
        @test all(isapprox.(sol[sys.R3.I], -1.5e-3))

        @test isapprox(sol[sys.node_2][end], 3.5)
        @test isapprox(sol[sys.R2.I][end], 1.75e-3)
        @test isapprox(sol[sys.node_4][end], -5.)
        @test isapprox(sol[sys.R4.I][end], -1.25e-3)
        @test isapprox(sol[sys.node_5][end], 1.5)
    end
end
=#

@testset "Spectre subcircuit" begin
    # Port of old "Simple Spectre subcircuit" test
    spectre_code = """
    subckt myres vcc gnd
        parameters r=1k
        r1 (vcc gnd) resistor r=r
    ends myres

    x1 (vcc 0) myres r=2k
    v1 (vcc 0) vsource dc=1
    """
    ctx, sol = solve_mna_spectre_code(spectre_code)
    # I = V/R = 1V/2kΩ = 0.5mA
    @test isapprox_deftol(sol[:vcc], 1.0)
    @test isapprox_deftol(sol[:I_v1], -0.5e-3)
end

@testset "Simple SPICE sources" begin
    # Original SPICE code (same as original test)
    spice_code = """
    * Simple SPICE sources
    V1 0 1 1
    R1 1 0 1k
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # Original: @test all(isapprox.(sol[sys.node_1], -1.0))
    # V1 has + at 0, - at 1, so node 1 = -1V
    # Note: SPICE numeric nodes become Symbol("1"), not :node_1
    @test isapprox_deftol(sol[Symbol("1")], -1.0)
end

@testset "Simple SPICE controlled sources" begin
    # Original SPICE code testing E (VCVS) and G (VCCS)
    spice_code = """
    * Simple SPICE sources with controlled sources
    V1 0 1 1
    R1 1 0 1k

    E6 0 6 0 1 2
    R6 6 0 r=1k

    G7 0 7 0 1 2
    R7 7 0 r=1k
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # V1 makes node 1 = -1V (+ at 0, - at 1)
    # E6: VCVS with gain=2, Vout = 2 * V(0,1) = 2 * 1 = 2V at node 6 relative to 0
    # Since E6 has + at 0 and - at 6, node 6 = -2V
    @test isapprox_deftol(sol[Symbol("1")], -1.0)
    @test isapprox(sol[Symbol("6")], -2.0; atol=deftol*10)
    # G7: VCCS with gm=2, I = 2 * V(0,1) = 2A into node 7
    # With R7=1k to ground: V = I*R = 2*1000 = 2000V (but sign depends on convention)
    # G7 outputs current from 0 to 7, so 2A flows into 7, V7 = -2000V
    @test isapprox(sol[Symbol("7")], -2000.0; atol=deftol*10)
end

@testset "SPICE B-source" begin
    # Test B-source with voltage expression referencing another node
    spice_code = """
    * B-source test
    V1 0 1 1
    R1 1 0 1k

    B5 0 5 v=V(1)*2
    R5 5 0 1k
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # V1 makes node 1 = -1V
    # B5: v = V(1)*2 = -1*2 = -2V, but B5 has + at 0, - at 5, so node 5 = 2V
    @test isapprox(sol[Symbol("5")], 2.0; atol=deftol*10)
end

@testset "SPICE B-source (nonlinear current)" begin
    # Test nonlinear B-source with i=V(1)**2 (current proportional to voltage squared)
    # Circuit: V1(2V) -> R1(1Ω) -> node 1 <- B1(i=V(1)^2) <- GND
    # At DC equilibrium: I_R1 = (V1 - V_node1) / R1 = I_B1 = V_node1^2
    # So: (2 - V) / 1 = V^2  =>  2 - V = V^2  =>  V^2 + V - 2 = 0
    # Solutions: V = (-1 ± 3) / 2 = 1 or -2
    # Physical solution (forward-biased): V = 1V
    spice_code = """
    * Nonlinear B-source test
    V1 vcc 0 DC 2
    R1 vcc 1 1
    B1 1 0 i=V(1)**2
    """
    # Use the builder-based solver for Newton iteration
    ast = Cadnip.NyanSpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = Cadnip.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using Cadnip.MNA))
    Base.eval(m, :(using Cadnip: ParamLens))
    Base.eval(m, :(using Cadnip.SpectreEnvironment))
    circuit_fn = Base.eval(m, code)

    spec = Cadnip.MNA.MNASpec(temp=27.0, mode=:dcop)
    sol = Cadnip.MNA.solve_dc(circuit_fn, (;), spec)

    # Node 1 should be 1V (the positive solution to V^2 + V - 2 = 0)
    @test isapprox(sol[Symbol("1")], 1.0; atol=1e-6)
    # V_vcc = 2V
    @test isapprox(sol[:vcc], 2.0; atol=1e-6)
end

# Alternate E/G forms with vol=/cur= syntax. Measured still failing, in the
# parser rather than sema: `MethodError: no method matching LString(::Nothing)`
# — the card has no positional value for `LString` to read.
#=
@testset "SPICE controlled sources (alternate syntax)" begin
    spice_code = """
    * Alternate E/G syntax
    V1 0 1 1
    R1 1 0 1k

    E8 0 8 vol=V(0, 1)*2
    R8 8 0 r=1k

    G9 0 9 cur=V(0, 1)*2
    R9 9 0 r=1k
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # E8: vol=V(0,1)*2 = 1*2 = 2V, since + at 0, - at 8, node 8 = -2V
    # G9: cur=V(0,1)*2 = 1*2 = 2A into node 9, V = 2*1000 = 2000V
    @test isapprox(sol[Symbol("8")], -2.0; atol=deftol*10)
    @test isapprox(sol[Symbol("9")], -2000.0; atol=deftol*10)
end
=#

@testset "Simple SPICE subcircuit" begin
    # Same SPICE code as original
    spice_code = """
    * Subcircuit test
    .subckt myres vcc gnd
    .param r=1k
    R1 vcc gnd 'r'
    .ends

    V1 vcc 0 DC 1
    X1 vcc 0 myres r=2k
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # Original: @test all(isapprox.(sol[sys.x1.r1.I], 0.5e-3))
    @test isapprox_deftol(sol[:vcc], 1.0)
    @test isapprox_deftol(sol[:I_v1], -0.5e-3)  # 1V / 2kΩ
end

@testset "SPICE include .LIB" begin
    # Test .LIB definition and include (self-referential)
    mktempdir() do dir
        spice_file = joinpath(dir, "selfinclude.cir")
        open(spice_file; write=true) do io
            write(io, """
            * .LIB definition and include test
            V1 vdd 0 1

            .LIB my_lib
            r1 vdd 0 1337
            .ENDL
            .LIB "selfinclude.cir" my_lib
            """)
        end

        # Parse and solve using MNA
        ast = NyanSpectreNetlistParser.parsefile(spice_file; start_lang=:spice)
        code = Cadnip.make_mna_circuit(ast)
        m = Module()
        Base.eval(m, :(using Cadnip.MNA))
        Base.eval(m, :(using Cadnip: ParamLens))
        Base.eval(m, :(using Cadnip.SpectreEnvironment))
        circuit_fn = Base.eval(m, code)

        # Wrap in MNACircuit for proper dc! solve
        spec = Cadnip.MNA.MNASpec(temp=27.0, mode=:dcop)
        wrapped = (args...; kwargs...) -> Base.invokelatest(circuit_fn, args...; kwargs...)
        circuit = Cadnip.MNA.MNACircuit(wrapped, (;), spec)
        sol = dc!(circuit)

        # I = V / R = 1V / 1337Ω
        @test isapprox_deftol(sol[:I_v1], -1/1337)
    end
end

# HDL include tests — `va"..."`-level fixture path is resolved at module
# top level so @testset closures (function bodies) can call MNACircuit/dc!
# without tripping world-age.
const _HDL_RESISTOR_VA = joinpath(dirname(pathof(Cadnip.NyanVerilogAParser)),
                                   "..", "test", "inputs", "resistor.va")
@assert isfile(_HDL_RESISTOR_VA) "fixture missing: $(_HDL_RESISTOR_VA)"

# File-path MNACircuit call requires the builder to be defined at top level.
# Write a persistent fixture netlist next to the .va in a subdir of @__DIR__.
const _HDL_FIXTURE_DIR = mktempdir(; cleanup=true)
cp(_HDL_RESISTOR_VA, joinpath(_HDL_FIXTURE_DIR, "resistor.va"))
const _HDL_NETLIST = joinpath(_HDL_FIXTURE_DIR, "hdl_test.sp")
open(_HDL_NETLIST; write=true) do io
    write(io, """
    * HDL via file path
    .hdl "resistor.va"
    X1 vcc 0 BasicVAResistor R=4000
    V1 vcc 0 DC 2
    """)
end
const _HDL_FILE_CIRCUIT = MNACircuit(_HDL_NETLIST)

const _HDL_SCS_NETLIST = joinpath(_HDL_FIXTURE_DIR, "hdl_test.scs")
open(_HDL_SCS_NETLIST; write=true) do io
    write(io, """
    ahdl_include "resistor.va"
    v1 (vcc 0) vsource type=dc dc=2
    x1 (vcc 0) BasicVAResistor R=4000
    """)
end
const _HDL_SCS_CIRCUIT = MNACircuit(_HDL_SCS_NETLIST)

# cwd-relative .hdl from an inline netlist (no source_dir, no srcfile).
# Tests the cache.jl fallback that handles `pathof(thismod) === nothing`.
const _HDL_CWD_DIR = mktempdir(; cleanup=true)
cp(_HDL_RESISTOR_VA, joinpath(_HDL_CWD_DIR, "resistor.va"))
const _HDL_CWD_CIRCUIT = cd(_HDL_CWD_DIR) do
    MNACircuit("""
    * relative .hdl, no source_dir — must resolve via cwd
    .hdl "resistor.va"
    V1 vcc 0 DC 1
    X1 vcc 0 BasicVAResistor R=1000
    """; lang=:spice)
end

@testset "SPICE .hdl include (on-the-fly VA codegen)" begin
    @testset "inline MNACircuit(code) with absolute path" begin
        code = """
        * HDL on-the-fly smoke test
        .hdl "$_HDL_RESISTOR_VA"
        X1 vcc 0 BasicVAResistor R=2000
        V1 vcc 0 DC 1
        """
        circuit = MNACircuit(code; lang=:spice)
        sol = dc!(circuit)
        @test isapprox_deftol(sol[:vcc], 1.0)
        @test isapprox_deftol(sol[:I_v1], -1/2000)
    end

    @testset "file-path MNACircuit with sibling .va" begin
        sol = dc!(_HDL_FILE_CIRCUIT)
        @test isapprox_deftol(sol[:vcc], 2.0)
        @test isapprox_deftol(sol[:I_v1], -2/4000)
    end

    @testset "double .hdl of same file is idempotent" begin
        # Two directives referencing the same file should not error; the second
        # finds the cached Pair and reuses the existing module.
        code = """
        * Double HDL include
        .hdl "$_HDL_RESISTOR_VA"
        .hdl "$_HDL_RESISTOR_VA"
        X1 vcc 0 BasicVAResistor R=1000
        V1 vcc 0 DC 1
        """
        circuit = MNACircuit(code; lang=:spice)
        sol = dc!(circuit)
        @test isapprox_deftol(sol[:I_v1], -1/1000)
    end

    @testset "Spectre ahdl_include (inline)" begin
        code = """
        ahdl_include "$_HDL_RESISTOR_VA"
        x1 (vcc 0) BasicVAResistor R=2000
        v1 (vcc 0) vsource type=dc dc=1
        """
        circuit = MNACircuit(code; lang=:spectre)
        sol = dc!(circuit)
        @test isapprox_deftol(sol[:vcc], 1.0)
    end

    @testset "Spectre .scs file with ahdl_include + sibling .va" begin
        sol = dc!(_HDL_SCS_CIRCUIT)
        @test isapprox_deftol(sol[:vcc], 2.0)
        @test isapprox_deftol(sol[:I_v1], -2/4000)
    end

    @testset "relative .hdl from MNACircuit(code; lang) resolves against cwd" begin
        sol = dc!(_HDL_CWD_CIRCUIT)
        @test isapprox_deftol(sol[:vcc], 1.0)
        @test isapprox_deftol(sol[:I_v1], -1/1000)
    end
end

@testset "SPICE parameter scope" begin
    # Same SPICE code as original
    # Sema provides topologically sorted parameter_order, so par_leff can reference l and par_l
    spice_code = """
    * Parameter scoping test

    .subckt subcircuit1 vss gnd l=11
    .param
    + par_l=1
    + par_leff='l-par_l'
    r1 vss gnd 'par_leff'
    .ends

    x1 vss 0 subcircuit1
    v1 vss 0 1
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # R = l - par_l = 11 - 1 = 10Ω
    # I = V/R = 1/10 = 0.1A
    @test isapprox_deftol(sol[:I_v1], -0.1)
end

@testset "SPICE parameter scope (nested subcircuits)" begin
    # A self-referencing `.subckt` default (`foo=foo+2000`) reads the *caller's*
    # `foo`, through a scope the name is only passed down through: `outer`
    # neither declares nor reads `foo`, it just carries it to `inner`.
    spice_code = """
    * Dynamic parameters
    .subckt inner a b foo=foo+2000
    R1 a b 'foo'
    .ends

    .subckt outer a b
    x1 a b inner
    .ends

    .param foo = 1
    i1 vcc 0 'foo'
    x1 vcc 0 outer
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # foo=1 at top level, inner sees foo+2000 = 2001
    # V = I * R = 1 * 2001 = 2001V
    @test isapprox_deftol(sol[:vcc], -2001.0)
end

# TODO: Test .option temp / .temp for temperature setting
@testset "SPICE parameter scope (.option temp)" begin
    # Test that temper in .param picks up .option temp
    spice_code = """
    * param temp
    .option temp=10
    .param foo = temper
    i1 vcc 0 'foo'
    r1 vcc 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # foo = temper = 10 (from .option temp)
    @test_broken isapprox_deftol(sol[:vcc], -10.0)
end

# `.temp <value>` is funneled through the same option channel as `.option
# temp=<value>` (see sema.jl), so it shares the same (currently broken)
# temper-in-.param propagation. What it no longer does is crash at the sema
# stage.
@testset "SPICE parameter scope (.temp)" begin
    # Test .temp directive
    spice_code = """
    * .temp
    .temp 10
    .param foo = temper
    i1 vcc 0 'foo'
    r1 vcc 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # foo = temper = 10 (from .temp)
    @test_broken isapprox_deftol(sol[:vcc], -10.0)
end

@testset "SPICE parameter scope (default temper)" begin
    # Test that the default temper is 27
    spice_code = """
    * temper
    .param foo = temper
    i1 vcc 0 'foo'
    r1 vcc 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # Default temper = 27
    @test isapprox_deftol(sol[:vcc], -27.0)
end

# `.temp`/`.option temp` is the deck's own analysis temperature — it applies
# unconditionally, the same way it always has (a caller who wants a different
# temperature should not be emitting the card in the first place; that's on
# whoever generates the deck, not a precedence Cadnip should adjudicate). What
# used to be broken: the rebind reset every other `MNASpec` field (`gmin`,
# `tnom`, `gshunt`, `srcFact`, the four tolerances, and the `time` field's
# ForwardDiff-dual-carrying type) to their defaults on the way past. It now
# goes through `with_temp`, which `test/mna/core.jl` already unit-tests for
# preserving the rest of the spec — this just confirms the card is still
# honored end to end through the netlist path. doc/FINDINGS.rst finding 1.
@testset "SPICE .temp card sets the deck's temperature" begin
    rect = sp"""
    V1 vin 0 DC 5
    R1 vin out 1k
    D1 out 0 dmod
    .model dmod d is=76.9p n=1.45
    """i
    carded = sp"""
    V1 vin 0 DC 5
    R1 vin out 1k
    D1 out 0 dmod
    .model dmod d is=76.9p n=1.45
    .temp 100
    """i

    v_100_via_card = dc!(MNACircuit(carded))[:out]
    v_100_via_spec = dc!(MNACircuit(rect; spec=MNASpec(temp=100.0)))[:out]
    v_27_via_spec  = dc!(MNACircuit(rect))[:out]

    # The card sets the operating point the same as an equivalent explicit
    # spec with no card — and differently from the 27 C default.
    @test isapprox_deftol(v_100_via_card, v_100_via_spec)
    @test !isapprox_deftol(v_100_via_card, v_27_via_spec)

    # The card wins even over a caller-supplied spec (unconditionally, by
    # design — see the comment above).
    v_via_carded_and_spec = dc!(MNACircuit(carded; spec=MNASpec(temp=50.0)))[:out]
    @test isapprox_deftol(v_via_carded_and_spec, v_100_via_card)
end

# Analysis (.ac/.dc/.tran), output (.print/.width), and initial-condition
# (.ic) control cards used to crash the semantic analysis stage with
# `@show stmt; error()`. They carry no information the MNA netlist builder
# consumes, so they must be ignored, not fatal.
@testset "Control/analysis dot-cards are ignored, not fatal" begin
    # A trivial resistive divider (out = 5 * 1k/(1k+1k) = 2.5V) sprinkled with
    # control cards the builder does not consume.
    base = """
    * control cards
    V1 vcc 0 DC 5
    R1 vcc out 1k
    R2 out 0 1k
    """
    for card in (
        ".ac dec 10 1 1e6",
        ".dc V1 0 5 0.1",
        # `.noise` used to fail one step earlier than the others — it had no
        # branch in the parser's dot dispatch, so the card came back an error
        # node and the deck would not load at all.
        ".noise v(out) V1 dec 10 1 1e6",
        ".noise v(out,vcc) V1 oct 3 10 1k 5",
        ".print dc v(out)",
        ".width out=80",
        ".ic v(out)=1",
    )
        spice_code = base * card * "\n"
        ctx, sol = solve_mna_spice_code(spice_code)
        @test isapprox_deftol(sol[:out], 2.5)
    end
end

# An instance parameter whose value reads another parameter of the same
# instance line (`x1 … subcircuit1 w=4 nrd='w/2'`). Measured still failing, at
# codegen rather than runtime: `UndefVarError: w not defined` — the kwarg
# expressions are built in the *caller's* scope, where `w` is the callee's
# parameter and so is not bound.
#=
@testset "SPICE parameter scope (instance params)" begin
    # Test that instance parameters can refer to other parameters
    spice_code = """
    * Parameter scoping test

    .subckt subcircuit1 vss gnd w=2 rsh=1 nrd=1
    r1 vss gnd 'rsh*nrd'
    .ends
    * this should not require a global w parameter
    x1 vss 0 subcircuit1 w=4 nrd='w/2'
    v1 vss 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # nrd='w/2' where w=4, so nrd=2, R = rsh*nrd = 1*2 = 2
    # I = V/R = 1/2 = 0.5A
    @test isapprox_deftol(sol[:I_v1], -0.5)
end
=#

# One source carrying all three specifications at once. Which one is read is the
# analysis's business, not the card's: `DC 5` is the operating point, `SIN(10 3
# 1k)` is the transient waveform, `AC 1` the small-signal phasor. The original
# form of this test asserted the same 10.0 across three DAECompiler
# initializers; what it was really pinning down is that the SIN offset — not the
# DC value — is where the transient starts.
@testset "multimode spice source" begin
    wrapped = _eval_spice_builder("""
    * multimode spice source
    v1 vcc 0 DC 5 AC 1 SIN(10 3 1k)
    r1 vcc 0 1k
    """, Module[])

    # DC operating point reads `DC 5`.
    @test isapprox_deftol(dc!(MNACircuit(wrapped, (;), MNASpec(temp=27.0, mode=:dcop)))[:vcc], 5.0)

    # Transient reads SIN(offset=10, amplitude=3, freq=1k): starts at the
    # offset and swings ±3 around it.
    tr = tran!(MNACircuit(wrapped, (;), MNASpec(temp=27.0, mode=:tran)), (0.0, 2e-3))
    @test isapprox_deftol(tr[:vcc][1], 10.0)
    @test isapprox(maximum(tr[:vcc]), 13.0; rtol=1e-2)
    @test isapprox(minimum(tr[:vcc]), 7.0; rtol=1e-2)
end

@testset "SPICE multiplicities" begin
    # Same SPICE code as original
    spice_code = """
    * multiplicities
    v1 vcc 0 DC 1

    r1a vcc 1 1 m=10
    r1b 1 0 1
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # With m=10, r1a is effectively 0.1Ω
    # Total R = 0.1 + 1 = 1.1Ω
    # V at node 1 = 1 * (1/1.1) = 0.909V (voltage divider)
    @test isapprox(sol[Symbol("1")], 10/11; atol=deftol*10)
end

# `m` on a `.subckt` is the instance's multiplicity, not a value the body reads
# by name: it scales every device inside, and composes with the multiplicity of
# every enclosing instance. An instance-line `m=` *replaces* the `.subckt`
# line's default rather than multiplying with it — the default is what the
# instance line would have said.
@testset "SPICE multiplicities (subcircuit m=)" begin
    spice_code = """
    * multiplicities with subcircuit
    v1 vcc 0 DC 1

    .subckt r10 a b m=10
    r2a a b 1
    .ends
    x2a vcc 2 r10
    r2b 2 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # Subcircuit with m=10 divides resistance by 10
    @test isapprox(sol[Symbol("2")], 10/11; atol=deftol*10)
end

@testset "SPICE multiplicities (nested subcircuits)" begin
    spice_code = """
    * multiplicities with nested subcircuits
    v1 vcc 0 DC 1

    .subckt r10 a b m=10
    r2a a b 1
    .ends

    .subckt r5t2 a b
    x5r1 a b r10 m=5
    x5r2 a b r10 m=5
    .ends
    x4a1 vcc 4 r5t2
    r4b 4 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # Each instance line's m=5 replaces r10's own default of 10, and the two
    # instances are in parallel: 5 + 5 = 10 copies of a 1Ω resistor.
    @test isapprox(sol[Symbol("4")], 10/11; atol=deftol*10)
end

@testset "SPICE multiplicities (nested m=)" begin
    spice_code = """
    * multiplicities with nested m= on subcircuit
    v1 vcc 0 DC 1

    .subckt r2 a b
    r2 a b 1 m=2
    .ends
    x5a vcc 5 r2 m=5
    r5b 5 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # r2 has m=2 internally, x5a has m=5, so effective m=10.
    # `r2` never declares `m`, so the instance line's m= is the multiplicity
    # outright — the builder accepts it either way.
    @test isapprox(sol[Symbol("5")], 10/11; atol=deftol*10)
end

@testset "SPICE multiplicities compose across nesting levels" begin
    # Nothing overrides the inner default, so the two multiply: an m=3 instance
    # of a `.subckt` whose body instantiates an m=4 one stamps 12 copies.
    ctx, sol = solve_mna_spice_code("""
    * multiplicity composes
    v1 vcc 0 DC 1

    .subckt inner a b m=4
    ri a b 1
    .ends

    .subckt outer a b
    xi a b inner
    .ends
    xo vcc 2 outer m=3
    rb 2 0 1
    """)
    @test isapprox(sol[Symbol("2")], 12/13; atol=deftol*10)
end

@testset "SPICE multiplicities reach reactive devices" begin
    # The resistor cases above divide by m; a capacitance multiplies and an
    # inductance divides, off the same enclosing multiplicity.
    for (card, field, expected) in (("c1 a b 1u", :C, 4e-6),
                                    ("l1 a b 1u", :C, 0.25e-6))
        ctx, _ = solve_mna_spice_code("""
        * reactive under m=
        v1 vcc 0 DC 0
        r1 vcc 0 1k
        .subckt dut a b
        $card
        .ends
        x1 vcc 0 dut m=4
        """)
        sys = Cadnip.MNA.assemble!(ctx)
        @test isapprox(maximum(abs, getfield(sys, field)), expected; rtol=1e-9)
    end
end

@testset "SPICE multiplicities (.model)" begin
    spice_code = """
    * multiplicities with .model
    v1 vcc 0 DC 1

    .model rm r R=1
    r6a vcc 6 rm m=10 l=1u
    r6b 6 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    @test isapprox(sol[Symbol("6")], 10/11; atol=deftol*10)
end

@testset ".model case sensitivity" begin
    spice_code = """
    * .model case sensitivity
    v1 vcc 0 DC 1
    .model rr r R=1
    r1 vcc 1 rr l=1u
    r2 1 0 rr R=2 l=1u
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # r1 uses model rr with R=1, r2 overrides R=2
    # Total resistance = 1 + 2 = 3, V at node 1 = 1 * 2/3
    @test isapprox(sol[Symbol("1")], 2/3; atol=deftol*10)
end

@testset "units and magnitudes" begin
    # Same SPICE code as original - tests mAmp (milli) and MegQux (mega) suffixes
    spice_code = """
    * units and magnitudes
    i1 vcc 0 DC -1mAmp
    r1 vcc 0 1MegQux
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # V = I*R = 1e-3 * 1e6 = 1000V
    @test isapprox(sol[:vcc], 1000.0; atol=deftol*10)

    spice_code2 = """
    * units and magnitudes 2
    i1 vcc 0 DC -1Amp
    r1 vcc 0 1Mil
    """

    ctx, sol = solve_mna_spice_code(spice_code2)
    # 1 mil = 25.4e-6 (25.4 micrometers)
    @test isapprox(sol[:vcc], 2.54e-5; atol=1e-8)
end

# TODO: Test that magnitudes don't introduce floating point errors
@testset "units and magnitudes (precision)" begin
    spice_code = """
    * units and magnitudes 3
    .param a=0.22u b=0.22e-6
    V1 vcc 0 'a'
    R1 vcc 0 1
    """
    ast = NyanSpectreNetlistParser.SPICENetlistParser.parse(spice_code)
    code = Cadnip.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using Cadnip.MNA))
    Base.eval(m, :(using Cadnip: ParamLens))
    Base.eval(m, :(using Cadnip.SpectreEnvironment))
    circuit_fn = Base.eval(m, code)

    # Use ParamObserver to check if a === b (exact equality, no floating point error)
    observer = Cadnip.ParamObserver(:top, nothing)
    spec = MNASpec(temp=27.0, mode=:dcop)
    Base.invokelatest(circuit_fn, observer, spec)
    p = getfield(observer, :params)[:params]
    # 0.22u should equal exactly 0.22e-6 (no floating point rounding from magnitude parsing)
    @test p[:a] === p[:b]
end

# `.option` cards that name an `MNASpec` field (`Cadnip.SPEC_OPTIONS`) rebind
# that field for the deck. Only `temp` used to, and the rest were parsed, stored
# and dropped (doc/FINDINGS.rst finding 2). `gmin` is the one with a device-side
# effect that does not need a temperature to observe: a Verilog-A model reads it
# back through `$simparam("gmin")`.
const _OPT_FIXTURE_DIR = mktempdir(; cleanup=true)
write(joinpath(_OPT_FIXTURE_DIR, "simparam_probe.va"), raw"""
module SimParamProbe(p, n);
    inout p, n;
    electrical p, n;
    analog I(p,n) <+ V(p,n) * $simparam("gmin", 1e-30);
endmodule
""")

# Same deck twice: once as written, once with a raised gmin. The probe conducts
# `gmin` siemens, so I(V1) reads the value the devices were stamped with.
const _OPT_GMIN_DECK = """
* gmin through .option
.hdl "simparam_probe.va"
V1 vcc 0 DC 1
X1 vcc 0 SimParamProbe
"""
const _OPT_DEFAULT_CIRCUIT, _OPT_GMIN_CIRCUIT = cd(_OPT_FIXTURE_DIR) do
    (MNACircuit(_OPT_GMIN_DECK; lang=:spice),
     MNACircuit(_OPT_GMIN_DECK * ".option gmin=1e-3\n"; lang=:spice))
end

# Every spec option at once, with a `.temp` card alongside the `.option` ones,
# against a caller spec whose other fields must survive. Top level, so the
# builder is callable directly below (`MNACircuit(code)` eval's it).
const _OPT_MULTI_CIRCUIT = MNACircuit("""
* every spec option at once
.option gmin=1e-9 tnom=25 abstol=1e-15 reltol=1e-5
.option vntol=1e-9 iabstol=1e-14
.temp 40
V1 vcc 0 DC 1
R1 vcc 0 1k
"""; lang=:spice, spec=MNASpec(gshunt=1e-11, srcFact=0.5))

@testset ".option" begin
    # Bare flags and options the backend does not know are ignored, not fatal.
    spice_ckt = """
    * .option
    .option temp=10 filemode=ascii noinit
    """
    ast = NyanSpectreNetlistParser.SPICENetlistParser.parse(spice_ckt)
    code = Cadnip.make_mna_circuit(ast)
    @test code !== nothing

    @testset "a spec option reaches the devices" begin
        # The probe conducts `gmin` siemens, so I(V1) is the gmin the devices
        # were stamped with — 1e-12 by default, the card's value with one.
        @test dc!(_OPT_DEFAULT_CIRCUIT)[:I_v1] ≈ -1e-12 rtol=1e-6
        @test dc!(_OPT_GMIN_CIRCUIT)[:I_v1] ≈ -1e-3 rtol=1e-6
    end

    @testset "the deck's resolved spec is recorded on the context" begin
        # What the builder resolved is readable back off the context it stamped
        # into — the caller's `circuit.spec` never sees a card.
        function stamped(circuit)
            ctx = MNAContext()
            Base.invokelatest(circuit.builder, circuit.params, circuit.spec, 0.0;
                              x=ZERO_VECTOR, ctx=ctx)
            Cadnip.MNA.stamped_spec(ctx)
        end
        @test stamped(_OPT_DEFAULT_CIRCUIT).gmin == 1e-12
        @test stamped(_OPT_GMIN_CIRCUIT).gmin == 1e-3
        @test _OPT_GMIN_CIRCUIT.spec.gmin == 1e-12   # the card is builder-local

        # Several at once, including the `$simparam` tolerances. Everything no
        # card names keeps the caller's value.
        s = stamped(_OPT_MULTI_CIRCUIT)
        @test (s.gmin, s.tnom, s.abstol, s.reltol, s.vntol, s.iabstol, s.temp) ==
              (1e-9, 25.0, 1e-15, 1e-5, 1e-9, 1e-14, 40.0)
        @test (s.gshunt, s.srcFact, s.mode) ==
              (1e-11, 0.5, _OPT_MULTI_CIRCUIT.spec.mode)
    end

    @testset "an unimplemented result-affecting option warns" begin
        # `.option scale=1` is the no-op every PDK that spells the default out
        # carries; anything else changes what the deck means and we drop it.
        @test_logs min_level=Base.CoreLogging.Warn MNACircuit("""
        * scale=1 is a no-op
        .option scale=1
        V1 vcc 0 DC 1
        R1 vcc 0 1k
        """; lang=:spice)
        @test_logs (:warn, r"`.option scale` is parsed but not implemented") min_level=Base.CoreLogging.Warn MNACircuit("""
        * scale=2 is not
        .option scale=2
        V1 vcc 0 DC 1
        R1 vcc 0 1k
        """; lang=:spice)
    end
end

@testset "functions" begin
    # Same SPICE code as original - test parameter functions
    # These work because SpectreEnvironment exports int, nint, floor, ceil, pow, ln
    spice_ckt = """
    * functions
    .param
    + intp=int(1.5)
    + intn=int(-1.5)
    + nintp = nint(1.6)
    + nintn = nint(-1.6)
    + floorp=floor(1.5)
    + floorn=floor(-1.5)
    + ceilp=ceil(1.5)
    + ceiln=ceil(-1.5)
    + powp=pow(2.0, 3)
    + pown=pow(2.0, -3)
    + lnp=ln(2.0)
    V1 vcc 0 'intp + intn + floorp'
    R1 vcc 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_ckt)
    # intp=1, intn=-1, floorp=1 -> V = 1 + (-1) + 1 = 1V
    @test isapprox(sol[:vcc], 1.0; atol=deftol)
end

# TODO: Extended functions test - verify all function results via ParamObserver
@testset "functions (full verification)" begin
    spice_ckt = """
    * functions full
    .param
    + intp=int(1.5)
    + intn=int(-1.5)
    + nintp = nint(1.6)
    + nintn = nint(-1.6)
    + floorp=floor(1.5)
    + floorn=floor(-1.5)
    + ceilp=ceil(1.5)
    + ceiln=ceil(-1.5)
    + powp=pow(2.0, 3)
    + pown=pow(2.0, -3)
    + lnp=ln(2.0)
    V1 vcc 0 1
    R1 vcc 0 1
    """
    ast = NyanSpectreNetlistParser.SPICENetlistParser.parse(spice_ckt)
    code = Cadnip.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using Cadnip.MNA))
    Base.eval(m, :(using Cadnip: ParamLens))
    Base.eval(m, :(using Cadnip.SpectreEnvironment))
    circuit_fn = Base.eval(m, code)

    observer = Cadnip.ParamObserver(:top, nothing)
    spec = MNASpec(temp=27.0, mode=:dcop)
    Base.invokelatest(circuit_fn, observer, spec)
    p = getfield(observer, :params)[:params]

    @test p[:intp] == 1
    @test p[:intn] == -1
    @test p[:nintp] == 2
    @test p[:nintn] == -2
    @test p[:floorp] == 1
    @test p[:floorn] == -2
    @test p[:ceilp] == 2
    @test p[:ceiln] == -1
    @test p[:powp] == 8
    @test p[:pown] == 0.125
    @test p[:lnp] == log(2.0)
end

@testset "device == param (ParamObserver)" begin
    # Test ParamObserver integration with MNA codegen
    # ParamObserver records which parameters are used and their values
    using Cadnip: ParamObserver, @param

    # Use explicit parameter passing (factor is a formal parameter of subcircuit)
    spice_code = """
    * device == param
    .subckt myres p n factor=1
        .param rload=1k
        r1 p n 'rload*factor'
    .ends
    i1 vcc 0 DC -1
    x1 vcc 0 myres factor=2
    """

    # Parse and generate MNA circuit
    ast = NyanSpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = Cadnip.make_mna_circuit(ast)

    # Evaluate in temp module
    m = Module()
    Base.eval(m, :(using Cadnip.MNA))
    Base.eval(m, :(using Cadnip: ParamLens, AbstractParamLens))
    Base.eval(m, :(using Cadnip.SpectreEnvironment))
    circuit_fn = Base.eval(m, code)

    # Use ParamObserver to record parameters
    observer = ParamObserver(:top, nothing)
    spec = MNASpec(temp=27.0, mode=:dcop)
    ctx = Base.invokelatest(circuit_fn, observer, spec)

    # Test that ParamObserver recorded the parameter hierarchy
    # The subcircuit x1 should have rload and factor parameters
    @test haskey(getfield(observer, :params), :x1)
    x1_obs = getfield(observer, :params)[:x1]
    @test x1_obs isa ParamObserver
    @test haskey(getfield(x1_obs, :params), :params)
    x1_params = getfield(x1_obs, :params)[:params]
    @test haskey(x1_params, :rload)
    @test x1_params[:rload] == 1000.0  # default value recorded

    # Test @param macro works
    @test @param(observer.x1.rload) == 1000.0

    # Test that the parameter was applied by solving
    # Wrap in MNACircuit for proper dc! solve
    wrapped = (args...; kwargs...) -> Base.invokelatest(circuit_fn, args...; kwargs...)
    circuit = MNACircuit(wrapped, (;), spec)
    sol = dc!(circuit)

    # With factor=2, R = rload * factor = 1000 * 2 = 2000Ω
    # I = 1A (from current source), V = I*R
    # Current source I1: -1A means extracting 1A from vcc, injecting into 0
    # V_vcc = I * R = 1 * 2000 = 2000V
    @test isapprox(sol[:vcc], 2000.0; rtol=1e-6)
end

# TODO: Extended device == param tests
@testset "device == param (canonicalize_params)" begin
    # Test canonicalize_params function
    @test Cadnip.canonicalize_params((; params=(;boo=4), foo=2, bar=(; baz=3))) == (params = (boo = 4, foo = 2), bar = (params = (baz = 3,),))
end

# Semiconductor resistor: `.model … r` + `R1 … themodel w= l=`.
# The resistance each case is expected to reach is read back through the 1V
# source: `-I_v1` is `1/R`.
@testset "semiconductor resistor" begin
    conductance(code; kwargs...) = -solve_mna_spice_code(code; kwargs...)[2][:I_v1]

    @testset "sheet resistance, geometry from the instance line" begin
        # R1 = rsh * l / w = 500 * 2m / 1m = 1000Ω; R2 = res = 1000Ω.
        # In parallel: 500Ω, so I = 2mA. `res` is a `.param`, not a card —
        # the value position tells the two apart by whether a card owns the name.
        g = conductance("""
        * semiconductor resistor
        .model myres r rsh=500
        .param res=1k
        v1 vcc 0 1
        R1 vcc 0 myres w=1m l=2m
        R2 vcc 0 res
        """)
        @test isapprox(g, 2e-3; atol=deftol*10)
    end

    @testset "geometry defaulted by the card" begin
        # Neither `l` nor `w` on the instance line: `l` from the card, `w` from
        # its `defw`. R = 100 * 2u / 1u = 200Ω.
        g = conductance("""
        * card geometry
        .model rm r rsh=100 l=2u defw=1u
        v1 vcc 0 1
        R1 vcc 0 rm
        """)
        @test isapprox(g, 1/200; atol=deftol*10)
    end

    @testset "side etching" begin
        # R = rsh * (l - short) / (w - narrow) = 100 * 2u / 1u = 200Ω
        g = conductance("""
        * etching
        .model rn r rsh=100 narrow=0.1u short=0.2u
        v1 vcc 0 1
        R1 vcc 0 rn l=2.2u w=1.1u
        """)
        @test isapprox(g, 1/200; atol=deftol*10)
    end

    @testset "instance r= outranks the card's geometry" begin
        g = conductance("""
        * instance override
        .model rm r rsh=100 l=2u defw=1u
        v1 vcc 0 1
        R1 vcc 0 rm r=400
        """)
        @test isapprox(g, 1/400; atol=deftol*10)
    end

    @testset "a card that is only a resistance" begin
        g = conductance("""
        * plain card
        .model rr r r=2k
        v1 vcc 0 1
        R1 vcc 0 rr
        """)
        @test isapprox(g, 1/2000; atol=deftol*10)
    end

    @testset "temperature coefficients" begin
        # 100°C above the card's own tnom, tc1=1e-3: R = 1000 * 1.1 = 1100Ω
        card = """
        * tc on the card
        .model rt r rsh=1k l=1u defw=1u tc1=1e-3 tnom=27
        v1 vcc 0 1
        R1 vcc 0 rt
        """
        @test isapprox(conductance(card; temp=27.0), 1/1000; atol=deftol*10)
        @test isapprox(conductance(card; temp=127.0), 1/1100; atol=deftol*10)

        # tc2 too: R = 1000 * (1 + 1e-3*100 + 1e-5*100^2) = 1200Ω
        @test isapprox(conductance("""
        * tc2
        .model rt2 r rsh=1k l=1u defw=1u tc1=1e-3 tc2=1e-5
        v1 vcc 0 1
        R1 vcc 0 rt2
        """; temp=127.0), 1/1200; atol=deftol*10)

        # The instance line's own coefficients, with no card behind them
        @test isapprox(conductance("""
        * instance tc
        v1 vcc 0 1
        R1 vcc 0 1k tc1=1e-3
        """; temp=127.0), 1/1100; atol=deftol*10)

        # and they outrank the card's
        @test isapprox(conductance("""
        * instance tc wins
        .model rt3 r rsh=1k l=1u defw=1u tc1=1e-3
        v1 vcc 0 1
        R1 vcc 0 rt3 tc1=2e-3
        """; temp=127.0), 1/1200; atol=deftol*10)
    end

    @testset "sheet resistance with no card at all" begin
        # `rsh` spelled on the instance line: R = 500 * 2m / 1m = 1000Ω
        g = conductance("""
        * instance rsh
        v1 vcc 0 1
        R1 vcc 0 rsh=500 l=2m w=1m
        """)
        @test isapprox(g, 1/1000; atol=deftol*10)
    end

    @testset "a card with no resistance in it is an error" begin
        @test_throws Exception conductance("""
        * nothing to resolve
        .model rbad r narrow=1u
        v1 vcc 0 1
        R1 vcc 0 rbad
        """)
    end
end

@testset "ifelse" begin
    # Same SPICE code as original
    spice_code = """
    * ifelse resistor
    .param switch=1
    v1 vcc 0 1
    .if (switch == 1)
    R1 vcc 0 1
    .else
    R1 vcc 0 2
    .endif
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # With switch=1, R1=1Ω, I = V/R = 1A
    @test isapprox(sol[:I_v1], -1.0; atol=deftol*10)
end

@testset "SPICE CCVS (H element)" begin
    # Current-controlled voltage source
    # Uses zero-volt source for sensing (standard SPICE approach)
    spice_code = """
    * CCVS test with zero-volt sense source
    Vin vcc 0 DC 5
    R1 vcc sense 1k
    Vsense sense 0 DC 0
    H1 out 0 Vsense 200
    Rload out 0 1Meg
    """
    ctx, sol = solve_mna_spice_code(spice_code)

    # Current through Vsense = 5V/1kΩ = 5mA
    # Vout = rm * I = 200 * 5mA = 1V
    @test isapprox(sol[:vcc], 5.0; atol=deftol)
    @test isapprox(sol[:sense], 0.0; atol=deftol)
    @test isapprox(sol[:out], 1.0; atol=deftol)
end

@testset "SPICE CCCS (F element)" begin
    # Current-controlled current source
    # Uses zero-volt source for sensing (standard SPICE approach)
    spice_code = """
    * CCCS test with zero-volt sense source
    Vin vcc 0 DC 5
    R1 vcc sense 1k
    Vsense sense 0 DC 0
    F1 out 0 Vsense 2
    Rload out 0 100
    """
    ctx, sol = solve_mna_spice_code(spice_code)

    # Current through Vsense = 5V/1kΩ = 5mA
    # I_out = gain * I = 2 * 5mA = 10mA
    # V_out = I_out * R = 10mA * 100Ω = 1V
    @test isapprox(sol[:vcc], 5.0; atol=deftol)
    @test isapprox(sol[:sense], 0.0; atol=deftol)
    @test isapprox(sol[:out], 1.0; atol=deftol)
end

@testset "SPICE CCVS/CCCS sense a source in their own .subckt" begin
    # A sense source inside a subckt allocates its current under the instance
    # prefix (`I_x1_vsense`), so the F/H card must resolve the local name it
    # spells against that prefix — and two instances of the same subckt must
    # each sense their own copy.
    circuit = MNACircuit(sp"""
    .subckt tia inp outp
    R1 inp sense 1k
    Vsense sense 0 DC 0
    H1 outp 0 Vsense 200
    .ends
    .subckt mirror inp outp
    R1 inp sense 1k
    Vsense sense 0 DC 0
    F1 outp 0 Vsense 2
    .ends
    V1 vcc 0 DC 5
    V2 vcc2 0 DC 2
    X1 vcc outh tia
    X2 vcc2 outh2 tia
    X3 vcc outf mirror
    Rl1 outh 0 1Meg
    Rl2 outh2 0 1Meg
    Rl3 outf 0 100
    """i)
    sol = dc!(circuit)

    # Each instance senses its own current: 5V/1kΩ = 5mA and 2V/1kΩ = 2mA.
    @test isapprox(sol[:I_x1_vsense], 5e-3; atol=deftol)
    @test isapprox(sol[:I_x2_vsense], 2e-3; atol=deftol)

    # CCVS: rm · I = 200 · 5mA = 1V, and 200 · 2mA = 0.4V.
    @test isapprox(sol[:outh], 1.0; atol=deftol)
    @test isapprox(sol[:outh2], 0.4; atol=deftol)

    # CCCS: gain · I · Rload = 2 · 5mA · 100Ω = 1V.
    @test isapprox(sol[:outf], 1.0; atol=deftol)
end

@testset "SPICE CCVS senses through a nested .subckt" begin
    # The prefix chains, so the sense current is `I_xw_xin_vsense`.
    circuit = MNACircuit(sp"""
    .subckt tia inp outp
    R1 inp sense 1k
    Vsense sense 0 DC 0
    H1 outp 0 Vsense 200
    .ends
    .subckt wrapper a b
    Xin a b tia
    .ends
    V1 vcc 0 DC 5
    Xw vcc outh wrapper
    Rl outh 0 1Meg
    """i)
    sol = dc!(circuit)

    @test isapprox(sol[:I_xw_xin_vsense], 5e-3; atol=deftol)
    @test isapprox(sol[:outh], 1.0; atol=deftol)
end

@testset "DCSolution operating-point introspection" begin
    # Resistive divider: out = 6·2k/(1k+2k) = 4 V, in = 6 V.
    circuit = MNACircuit(sp"""
    * resistive divider
    V1 in 0 DC 6
    R1 in out 1k
    R2 out 0 2k
    """i)
    sol = dc!(circuit)

    cur = only(sol.current_names)        # single V source ⇒ one branch current

    # keys enumerate node voltages, branch currents, device terminal currents,
    # then device operating-point variables (what `show` prints); internal
    # charge/limit state variables are excluded from the enumeration.
    ks = keys(sol)
    @test :in in ks && :out in ks
    @test cur in ks
    @test :i_r1_p in ks                  # device terminal currents (see test/opinfo.jl)
    @test length(ks) == length(sol.node_names) + length(sol.current_names) +
                        length(sol.terminal_currents) + length(sol.op_vars)

    # values align positionally with keys
    vs = values(sol)
    @test length(vs) == length(ks)
    kv = Dict(zip(ks, vs))
    @test kv[:in]  ≈ 6.0
    @test kv[:out] ≈ 4.0
    @test kv[cur]  ≈ sol[cur]

    # pairs / Dict round-trip, and every pair agrees with name-based indexing
    @test Dict(pairs(sol)) == kv
    for (k, v) in pairs(sol)
        @test sol[k] == v
    end

    # haskey / get: safe, non-throwing lookups (ground is readable as 0.0)
    @test haskey(sol, :out)
    @test haskey(sol, :gnd)
    @test !haskey(sol, :does_not_exist)
    @test get(sol, :out, NaN) ≈ 4.0
    @test isnan(get(sol, :does_not_exist, NaN))
    @test_throws Exception sol[:does_not_exist]   # sol[name] still throws

    # string names work across getindex / haskey / get
    @test sol["out"] ≈ 4.0
    @test haskey(sol, "out")
    @test get(sol, "does_not_exist", -1.0) == -1.0
end

@testset "node_names / branch_names classify a solution's names" begin
    # One RC low-pass, driven so the same circuit answers DC, AC and transient:
    # two nodes (`in`, `out`), one branch current (V1's), and the device
    # terminal-current channel on top.
    circuit = MNACircuit(sp"""
    * classified readout
    V1 in 0 DC 1 AC 1
    R1 in out 1k
    R2 out 0 3k
    C1 out 0 1n
    """i)

    op = dc!(circuit)

    @test node_names(op) == [:in, :out]
    @test op[:out] ≈ 0.75                        # 1 V · 3k/(1k+3k)
    cur = only(branch_names(op))                 # one voltage source ⇒ one branch
    @test op[cur] ≈ -1 / 4000                    # 1 V across 4 kΩ, out of V1

    # Ground is not an unknown, so it names no column — even though `op[:gnd]`
    # reads as 0.0.
    @test :gnd ∉ node_names(op) && Symbol("0") ∉ node_names(op)
    @test isempty(intersect(node_names(op), branch_names(op)))

    # The classification partitions the enumeration: node voltages, then branch
    # currents, then the two device channels, in `keys` order. This is what
    # `keys(sol)` alone cannot tell you — the `I_` spelling of a source current
    # is a convention, not an interface.
    @test keys(op) == vcat(node_names(op), branch_names(op),
                           [p.first for p in terminal_currents(op)],
                           [p.first for p in op_vars(op)])
    @test all(n -> haskey(op, n), node_names(op))
    @test all(n -> haskey(op, n), branch_names(op))

    # The lists are copies: mutating one cannot corrupt the solution.
    push!(node_names(op), :bogus)
    @test :bogus ∉ node_names(op)

    # A transient solution reads the same, through the MNA system its problem
    # carries — the object `nameat` looks names up in.
    tsol = tran!(circuit, (0.0, 20e-6))
    @test node_names(tsol) == node_names(op)
    @test branch_names(tsol) == branch_names(op)
    @test nameat(tsol, first(node_names(tsol)), 0.0) ≈ 1.0

    # So does an AC solution.
    acsol = ac!(circuit, acdec(4, 1e3, 1e6))
    @test node_names(acsol) == node_names(op)
    @test branch_names(acsol) == branch_names(op)
    @test length(acsol[first(node_names(acsol))]) == length(acsol.freqs)

    # And so do the system and the context it was assembled from, which is where
    # the names are actually born.
    ctx = MNAContext()
    circuit.builder(circuit.params, circuit.spec, 0.0; x=ZERO_VECTOR, ctx=ctx)
    sys = assemble!(ctx)
    @test node_names(ctx) == node_names(op)
    @test branch_names(ctx) == branch_names(op)
    @test node_names(sys) == node_names(op)
    @test branch_names(sys) == branch_names(op)
end

end # basic_tests
