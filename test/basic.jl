module basic_tests

include("common.jl")

using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using CedarSim.MNA: voltage, current, make_ode_problem

@testset "Simple VR Circuit" begin
    function VRcircuit(params, spec)
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        stamp!(VoltageSource(5.0; name=:V), ctx, vcc, 0)
        stamp!(Resistor(2.0; name=:R), ctx, vcc, 0)
        return ctx
    end

    ctx = VRcircuit((;), MNASpec())
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    # I = V/R = 5/2 = 2.5A
    @test isapprox_deftol(voltage(sol, :vcc), 5.0)
    # Current through voltage source (negative = sourcing current)
    @test isapprox_deftol(current(sol, :I_V), -2.5)
end

@testset "Simple IR circuit" begin
    function IRcircuit(params, spec)
        ctx = MNAContext()
        icc = get_node!(ctx, :icc)
        # Note: we follow the SPICE convention here and use negative current
        # to denote current flowing from the negative to positive terminals
        # of the current source.
        stamp!(CurrentSource(-5.0; name=:I), ctx, icc, 0)
        stamp!(Resistor(2.0; name=:R), ctx, icc, 0)
        return ctx
    end

    ctx = IRcircuit((;), MNASpec())
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    # V = IR = 5*2 = 10V
    @test isapprox_deftol(voltage(sol, :icc), 10.0)
end

const v_val = 5.0
const r_val = 2000.0
const c_val = 1e-6
@testset "Simple VRC circuit" begin
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

    # Simulate the RC circuit
    tau = r_val * c_val  # Time constant
    tspan = (0.0, 10 * tau)
    prob_data = make_ode_problem(sys, tspan)

    f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                    jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
    prob = ODEProblem(f, prob_data.u0, prob_data.tspan)
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=deftol, abstol=deftol)

    # Get the index of vrc node
    vrc_idx = findfirst(n -> n == :vrc, sys.node_names)

    # At t=0, capacitor starts at 0 (from DC solution)
    # At t=∞, capacitor voltage should approach v_val
    @test isapprox_deftol(sol.u[end][vrc_idx], v_val)

    # Current at start: I = V/R
    # Current at end: I ≈ 0 (capacitor fully charged)
end

@testset "Simple SPICE sources" begin
    spice_code = """
    * Simple SPICE sources
    V1 0 1 1
    R1 1 0 1k
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    @test isapprox_deftol(voltage(sol, :node_1), -1.0)
end

@testset "Simple SPICE subcircuit" begin
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
    @test isapprox_deftol(voltage(sol, :vcc), 1.0)
    @test isapprox_deftol(current(sol, :I_v1), -0.5e-3)  # 1V / 2kΩ
end

@testset "SPICE parameter scope" begin
    # Ensure we can use parameters scoped to sub-circuits
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
    @test isapprox_deftol(current(sol, :I_v1), -0.1)
end

@testset "SPICE multiplicities" begin
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
    @test isapprox(voltage(sol, :node_1), 10/11; atol=deftol*10)
end

@testset "units and magnitudes" begin
    spice_code = """
    * units and magnitudes
    i1 vcc 0 DC -1mAmp
    r1 vcc 0 1MegQux
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # V = I*R = 1e-3 * 1e6 = 1000V
    @test isapprox(voltage(sol, :vcc), 1000.0; atol=deftol*10)

    spice_code2 = """
    * units and magnitudes 2
    i1 vcc 0 DC -1Amp
    r1 vcc 0 1Mil
    """

    ctx, sol = solve_mna_spice_code(spice_code2)
    # 1 mil = 25.4e-6 (25.4 micrometers)
    @test isapprox(voltage(sol, :vcc), 2.54e-5; atol=1e-8)
end

@testset "ifelse" begin
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
    @test isapprox(current(sol, :I_v1), -1.0; atol=deftol*10)
end

@testset "Voltage Divider" begin
    function divider(params, spec)
        ctx = MNAContext()
        vin = get_node!(ctx, :vin)
        vout = get_node!(ctx, :vout)
        stamp!(VoltageSource(10.0; name=:V), ctx, vin, 0)
        stamp!(Resistor(1000.0; name=:R1), ctx, vin, vout)
        stamp!(Resistor(2000.0; name=:R2), ctx, vout, 0)
        return ctx
    end

    ctx = divider((;), MNASpec())
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    # Vout = Vin * R2/(R1+R2) = 10 * 2k/(1k+2k) = 6.67V
    @test isapprox_deftol(voltage(sol, :vin), 10.0)
    @test isapprox_deftol(voltage(sol, :vout), 10.0 * 2000 / 3000)
end

@testset "Two Voltage Sources" begin
    function two_sources(params, spec)
        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)
        stamp!(VoltageSource(5.0; name=:V1), ctx, n1, 0)
        stamp!(VoltageSource(3.0; name=:V2), ctx, n2, n1)
        stamp!(Resistor(1000.0; name=:R), ctx, n2, 0)
        return ctx
    end

    ctx = two_sources((;), MNASpec())
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    @test isapprox_deftol(voltage(sol, :n1), 5.0)
    @test isapprox_deftol(voltage(sol, :n2), 8.0)  # 5 + 3
end

@testset "RL Transient" begin
    const v_rl = 10.0
    const r_rl = 100.0
    const l_rl = 1e-3
    const tau_rl = l_rl / r_rl  # Time constant = 10μs

    function VRLcircuit(params, spec)
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        vrl = get_node!(ctx, :vrl)
        stamp!(VoltageSource(v_rl; name=:V), ctx, vcc, 0)
        stamp!(Resistor(r_rl; name=:R), ctx, vcc, vrl)
        stamp!(Inductor(l_rl; name=:L), ctx, vrl, 0)
        return ctx
    end

    ctx = VRLcircuit((;), MNASpec(mode=:tran))
    sys = assemble!(ctx)

    tspan = (0.0, 10 * tau_rl)
    prob_data = make_ode_problem(sys, tspan)

    f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                    jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
    prob = ODEProblem(f, prob_data.u0, prob_data.tspan)
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-8)

    # At steady state, inductor is a short circuit
    # All voltage drops across resistor, I = V/R = 0.1A
    I_ss = v_rl / r_rl
    I_L_idx = findfirst(n -> n == :I_L, sys.current_names)

    @test isapprox(sol.u[end][sys.n_nodes + I_L_idx], I_ss; atol=0.001)
end

@testset "SPICE CCVS (H element)" begin
    # CCVS: Transresistance amplifier using zero-volt source for sensing
    # Vin provides 5V, R1 sets current = 5V/1kΩ = 5mA through Vsense
    # Vsense is a zero-volt source that senses this current
    # H1 outputs voltage = rm * I = 200 * 5mA = 1V
    spice_code = """
    * CCVS test with zero-volt sense source
    Vin vcc 0 DC 5
    R1 vcc sense 1k
    Vsense sense 0 DC 0
    H1 out 0 Vsense 200
    Rload out 0 1Meg
    """
    ctx, sol = solve_mna_spice_code(spice_code)

    # Current through Vsense = 5V/1kΩ = 5mA (positive, flowing from sense to ground)
    # Vout = rm * I = 200 * 5mA = 1V
    @test isapprox(voltage(sol, :vcc), 5.0; atol=deftol)
    @test isapprox(voltage(sol, :sense), 0.0; atol=deftol)
    @test isapprox(voltage(sol, :out), 1.0; atol=deftol)
end

@testset "SPICE CCCS (F element)" begin
    # CCCS: Current mirror using zero-volt source for sensing
    # Vin provides 5V, R1 sets current = 5V/1kΩ = 5mA through Vsense
    # Vsense is a zero-volt source that senses this current
    # F1 outputs current = gain * I = 2 * 5mA = 10mA
    # V_out = I_out * R = 10mA * 100Ω = 1V
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
    @test isapprox(voltage(sol, :vcc), 5.0; atol=deftol)
    @test isapprox(voltage(sol, :sense), 0.0; atol=deftol)
    @test isapprox(voltage(sol, :out), 1.0; atol=deftol)
end

end # basic_tests
