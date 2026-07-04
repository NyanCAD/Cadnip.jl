module transient_tests

include("common.jl")

using Cadnip.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc, reset_for_restamping!
using Cadnip.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using Cadnip.MNA: pwl_at_time
using Cadnip.MNA: MNACircuit
using SciMLBase: ODEProblem

# We'll create a piecewise linear current source that goes through a resistor
#
# The circuit diagram is:
#
#  ┌──┬── +
#  I  R
#  └──┴── -

const i_max = 2
const r_val_pwl = 2
@testset "PWL" begin
    # Helper function that creates the piecewise linear ramp
    # from 0 -> 1 over the course of 1ms -> 9ms
    function pwl_val(t)
        if t < 1e-3
            0.0
        elseif t > 9e-3
            1.0
        else
            (t-1e-3)/8e-3
        end
    end

    # The analytic solution of this circuit is easily calculated in terms of `pwl_val(t)`
    vout_analytic_sol(t) = pwl_val(t) * i_max * r_val_pwl

    # Test using SPICE PWL source
    # SPICE convention: I n+ n- injects current into n-, extracts from n+
    # So i1 0 vout injects current into vout (n-)
    spice_code = """
    * PWL test
    i1 0 vout PWL(1m 0 9m $(i_max))
    R1 vout 0 r=$(r_val_pwl)
    """

    # Parse and generate MNA builder
    ast = NyanSpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = Cadnip.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using Cadnip.MNA))
    Base.eval(m, :(using Cadnip: ParamLens))
    Base.eval(m, :(using Cadnip.SpectreEnvironment))
    builder = Base.eval(m, code)

    # Solve for 10ms using MNACircuit API
    tspan = (0.0, 10e-3)
    spec = MNASpec(temp=27.0, mode=:tran, time=0.0)
    circuit = Base.invokelatest(MNACircuit, builder, (;), spec)
    prob = Base.invokelatest(ODEProblem, circuit, tspan)
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(linsolve=KLUFactorization()); reltol=1e-6, abstol=1e-6)

    # Get node index for vout
    dc_spec = MNASpec(temp=27.0, mode=:dcop, time=0.0)
    ctx = Base.invokelatest(builder, (;), dc_spec)
    sys = Cadnip.MNA.assemble!(ctx)
    vout_idx = findfirst(n -> n == :vout, sys.node_names)

    # Check that solution matches analytic
    for (i, t) in enumerate(sol.t)
        expected = vout_analytic_sol(t)
        actual = sol.u[i][vout_idx]
        @test isapprox(actual, expected; atol=0.1)
    end

    # Also test using direct MNA API with unified CurrentSource
    function PWLIRcircuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        vout = get_node!(ctx, :vout)
        # PWL: 0->0 at 1ms, 0->i_max at 9ms
        times = [1e-3, 9e-3]
        values = [0.0, Float64(i_max)]
        # Use unified CurrentSource with transient function
        stamp!(CurrentSource(values[1]; tran=_t -> pwl_at_time(times, values, _t), name=:I),
               ctx, vout, 0, t, spec.mode)
        stamp!(Resistor(Float64(r_val_pwl); name=:R), ctx, vout, 0)
        return ctx
    end

    circuit2 = MNACircuit(PWLIRcircuit, (;), MNASpec(temp=27.0))
    prob2 = ODEProblem(circuit2, tspan)
    sol2 = OrdinaryDiffEq.solve(prob2, Rodas5P(linsolve=KLUFactorization()); reltol=1e-6, abstol=1e-6, maxiters=100000)

    # Check direct API matches
    for (i, t) in enumerate(sol2.t)
        expected = vout_analytic_sol(t)
        actual = sol2.u[i][1]  # vout is node 1
        @test isapprox(actual, expected; atol=0.1)
    end
end

#=
@testset "PWL derivative" begin
    # This test requires Diffractor which may not be available
    # Skip for now - the PWL interpolation is tested above
end
=#

@testset "PULSE repeats" begin
    # A PULSE source must be periodic. Regression test: previously the codegen
    # built PWL points for only a single period, so after the first cycle the
    # output held v1 forever instead of repeating.
    using Cadnip.MNA: pulse_at_time

    v1, v2 = 0.0, 1.0
    td, tr, tf, pw, per = 1e-3, 1e-6, 1e-6, 2e-3, 5e-3

    # Direct helper checks across several periods. Sample mid-plateau (phase
    # tr + pw/2) so the value is unambiguously v2 regardless of float rounding.
    high_phase = tr + pw / 2
    low_phase = tr + tf + pw + (per - (tr + tf + pw)) / 2  # solidly in the flat-bottom region
    @test pulse_at_time(v1, v2, td, tr, tf, pw, per, 0.0) == v1                       # before delay
    @test pulse_at_time(v1, v2, td, tr, tf, pw, per, td + high_phase) == v2           # 1st pulse
    # The high portion must recur in later periods, not stay flat at v1.
    @test pulse_at_time(v1, v2, td, tr, tf, pw, per, td + per + high_phase) == v2     # 2nd pulse
    @test pulse_at_time(v1, v2, td, tr, tf, pw, per, td + 3*per + high_phase) == v2   # 4th pulse
    # And the low portion must recur too.
    @test pulse_at_time(v1, v2, td, tr, tf, pw, per, td + 2*per + low_phase) == v1

    # Now drive a node through a SPICE PULSE source and confirm the simulated
    # waveform pulses more than once. Feed a PULSE voltage into an RC low-pass
    # filter so the output tracks (smoothed) the periodic input.
    R_val, C_val = 1e3, 1e-9  # τ = 1µs ≪ pw, so vout settles near v2/v1 each cycle
    spice_code = """
    * PULSE repeat test
    V1 vin 0 PULSE($(v1) $(v2) $(td) $(tr) $(tf) $(pw) $(per))
    R1 vin vout $(R_val)
    C1 vout 0 $(C_val)
    """

    ast = NyanSpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = Cadnip.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using Cadnip.MNA))
    Base.eval(m, :(using Cadnip: ParamLens))
    Base.eval(m, :(using Cadnip.SpectreEnvironment))
    builder = Base.eval(m, code)

    tspan = (0.0, td + 4*per)  # span several periods (room to sample the 4th pulse plateau)
    circuit = Base.invokelatest(MNACircuit, builder, (;), MNASpec(temp=27.0))
    n = Cadnip.MNA.system_size(circuit)
    prob = Base.invokelatest(ODEProblem, circuit, tspan; u0=zeros(n))
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(linsolve=KLUFactorization()); reltol=1e-6, abstol=1e-6, maxiters=1_000_000)

    dc_spec = MNASpec(temp=27.0, mode=:dcop, time=0.0)
    ctx = Base.invokelatest(builder, (;), dc_spec)
    sys = Cadnip.MNA.assemble!(ctx)
    vout_idx = findfirst(nm -> nm == :vout, sys.node_names)

    # Sample the output near the middle of the high portion of each pulse.
    # With τ ≪ pw the filtered output reaches close to v2 on every cycle.
    for k in 0:3
        t_high = td + k*per + tr + pw/2
        @test isapprox(sol(t_high)[vout_idx], v2; atol=0.1)
    end

    # And near the end of each low portion it should be back near v1.
    for k in 0:2
        t_low = td + k*per + per - 1e-6
        @test isapprox(sol(t_low)[vout_idx], v1; atol=0.1)
    end
end

# Create a third-order Butterworth filter, according to https://en.wikipedia.org/wiki/Butterworth_filter#Example
# The circuit diagram is:
#
#  ┌─L1─┬─L3─┬── +
#  V    C2   R4
#  └────┴────┴── -
#
# We take the simple example, with values:
#  L1 = 3/2 H
#  C2 = 4/3 F
#  L3 = 1/2 H
#  R4 = 1 Ω
#
# This yields a transfer function of:
#   H(s) = 1/(1 + 2s + 2s^2 + s^3)
# The magnitude of the steady-state response is:
#   G(ω) = 1/sqrt(1 + ω^6)
# so at ω=1 we should get 1/2 gain (note, ω is supplied in radians, so the actual value
# will be divided by 2π!)
#
# If we drive this system with a sinusoidal input with frequency 1, we get the following transfer function:
#   H(s) = 1/(s^2 + 1) * 1/(1 + 2s + 2s^2 + s^3)
# This corresponds to a time-domain solution via the inverse laplace transform of:
#   vout(t) = (e^(-t) - sin(t) - cos(t))/2 + (2 * sin((sqrt(3) * t)/2))/(sqrt(3) * sqrt(e^t))
const L1_val = 3/2
const C2_val = 4/3
const L3_val = 1/2
const R4_val = 1
const ω_val = 1

@testset "Butterworth Filter" begin
    # Helper function to calculate RMS of a signal
    rms(sig) = sqrt(sum(sig.^2)/length(sig))

    vout_analytic_sol(t) = (exp(-t) - sin(t) - cos(t))/2 + (2 * sin((sqrt(3) * t)/2))/(sqrt(3) * sqrt(exp(t)))

    # Test using SPICE SIN source
    spice_code = """
    *Third order low pass filter, butterworth, with ω_c = 1

    V1 vin 0 SIN(0, 1, $(ω_val/2π))
    L1 vin n1 $(L1_val)
    C2 n1 0 $(C2_val)
    L3 n1 vout $(L3_val)
    R4 vout 0 $(R4_val)
    """

    # Parse and generate MNA builder
    ast = NyanSpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = Cadnip.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using Cadnip.MNA))
    Base.eval(m, :(using Cadnip: ParamLens))
    Base.eval(m, :(using Cadnip.SpectreEnvironment))
    builder = Base.eval(m, code)

    # This is a very low-frequency circuit; simulate for a long enough time
    # that we can get a nice steady-state response in the end
    tspan = (0.0, 100.0)

    # Get initial conditions (all zeros for inductors/capacitors)
    spec = Cadnip.MNA.MNASpec(temp=27.0, mode=:dcop, time=0.0)
    ctx = Base.invokelatest(builder, (;), spec)
    sys = Cadnip.MNA.assemble!(ctx)

    # Use MNACircuit API with zero initial conditions (capacitor/inductor start uncharged)
    circuit = Base.invokelatest(MNACircuit, builder, (;), MNASpec(temp=27.0))
    n = Cadnip.MNA.system_size(circuit)
    u0 = zeros(n)
    prob = Base.invokelatest(ODEProblem, circuit, tspan; u0=u0)
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(linsolve=KLUFactorization()); reltol=1e-6, abstol=1e-6, maxiters=100000)

    # Get node index for vout
    vout_idx = findfirst(n -> n == :vout, sys.node_names)

    # Check that solution matches analytic at sample points
    @test isapprox(sol.u[1][vout_idx], vout_analytic_sol(sol.t[1]); atol=0.1)
    @test isapprox(sol.u[end][vout_idx], vout_analytic_sol(sol.t[end]); atol=0.1)

    # Also assert that the RMS of the steady-state portion is approximately correct
    # At ω=1, gain should be ~0.5
    steady_state_vout = [sol.u[i][vout_idx] for i in (length(sol.u)÷2):length(sol.u)]
    @test isapprox(rms(steady_state_vout), 0.5; atol=0.15, rtol=0.15)

    # Test using direct MNA API with unified VoltageSource
    function butterworth_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)
        n1 = get_node!(ctx, :n1)
        vout = get_node!(ctx, :vout)

        # SIN source: V(t) = sin(ω*t) using unified VoltageSource with transient function
        freq = ω_val / 2π
        stamp!(VoltageSource(0.0; tran=_t -> sin(2π * freq * _t), name=:V), ctx, vin, 0, t, spec.mode)
        stamp!(Inductor(L1_val; name=:L1), ctx, vin, n1)
        stamp!(Capacitor(C2_val; name=:C2), ctx, n1, 0)
        stamp!(Inductor(L3_val; name=:L3), ctx, n1, vout)
        stamp!(Resistor(R4_val; name=:R4), ctx, vout, 0)
        return ctx
    end

    circuit2 = MNACircuit(butterworth_circuit, (;), MNASpec(temp=27.0))
    n2 = Cadnip.MNA.system_size(circuit2)
    u0_2 = zeros(n2)
    prob2 = ODEProblem(circuit2, tspan; u0=u0_2)
    sol2 = OrdinaryDiffEq.solve(prob2, Rodas5P(linsolve=KLUFactorization()); reltol=1e-6, abstol=1e-6, maxiters=100000)

    # Get vout index from direct API circuit
    ctx2 = butterworth_circuit((;), Cadnip.MNA.MNASpec(temp=27.0, mode=:dcop, time=0.0), 0.0)
    sys2 = Cadnip.MNA.assemble!(ctx2)
    vout_idx2 = findfirst(n -> n == :vout, sys2.node_names)

    # Check direct API also matches
    @test isapprox(sol2.u[1][vout_idx2], vout_analytic_sol(sol2.t[1]); atol=0.1)
    @test isapprox(sol2.u[end][vout_idx2], vout_analytic_sol(sol2.t[end]); atol=0.1)

    steady_state_vout2 = [sol2.u[i][vout_idx2] for i in (length(sol2.u)÷2):length(sol2.u)]
    @test isapprox(rms(steady_state_vout2), 0.5; atol=0.15, rtol=0.15)
end

@testset "Auto tstops (breakpoint detection)" begin
    using Cadnip.MNA: PulseWave, PWLWave, expand_breakpoints, breakpoints

    # PULSE edges: td, td+tr, td+tr+pw, td+tr+pw+tf, repeating every `per`.
    function build_pulse_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        defaults = (td=1e-6, per=10e-6)
        p = merge(defaults, params)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)
        vout = get_node!(ctx, :vout)
        wave = PulseWave(0.0, 1.0, p.td, 1e-6, 1e-6, 3e-6, p.per)
        stamp!(VoltageSource(0.0; tran=wave, name=:V1), ctx, vin, 0, t, spec.mode)
        stamp!(Resistor(1e3; name=:R), ctx, vin, vout)
        stamp!(Capacitor(1e-9; name=:C), ctx, vout, 0)
        return ctx
    end

    # Expected edges derive from breakpoints() on the same wave the circuit
    # stamps, so this testset exercises the real edge math rather than a
    # hand-copied reimplementation of it.
    pulse_edges(td, per, tspan) = expand_breakpoints(
        [breakpoints(PulseWave(0.0, 1.0, td, 1e-6, 1e-6, 3e-6, per))], tspan)

    tspan = (0.0, 30e-6)
    circuit = MNACircuit(build_pulse_rc; td=1e-6, per=10e-6)
    edges = pulse_edges(1e-6, 10e-6, tspan)
    @test length(edges) == 12  # 4 edges/period × 3 periods within tspan

    hits(sol, es) = all(e -> any(t -> abs(t - e) < 1e-15, sol.t), es)

    # Default solver (IDA, DAE path): tstops only (no d_discontinuities - Sundials rejects it)
    sol_ida = tran!(circuit, tspan)
    @test hits(sol_ida, edges)

    # ODE path (Rodas5P): tstops + d_discontinuities
    sol_ode = tran!(circuit, tspan; solver=Rodas5P())
    @test hits(sol_ode, edges)

    prob = ODEProblem(circuit, tspan)
    @test haskey(prob.kwargs, :tstops) && !isempty(prob.kwargs[:tstops])
    @test haskey(prob.kwargs, :d_discontinuities) && !isempty(prob.kwargs[:d_discontinuities])

    prob_dae = Base.invokelatest(SciMLBase.DAEProblem, circuit, tspan; explicit_jacobian=true)
    @test haskey(prob_dae.kwargs, :tstops) && !isempty(prob_dae.kwargs[:tstops])
    @test !haskey(prob_dae.kwargs, :d_discontinuities)  # Sundials/IDA rejects this kwarg

    # auto_tstops=false: no breakpoints injected at all
    prob_off = ODEProblem(circuit, tspan; auto_tstops=false)
    @test !haskey(prob_off.kwargs, :tstops)
    @test !haskey(prob_off.kwargs, :d_discontinuities)

    # User-supplied tstops merge with (don't clobber) the auto-computed ones.
    # tran! forwards tstops to the problem constructor, which owns the merge:
    user_t = 7.5e-6
    prob_user = ODEProblem(circuit, tspan; tstops=[user_t])
    @test user_t in prob_user.kwargs[:tstops]
    @test all(e -> e in prob_user.kwargs[:tstops], edges)

    sol_merged = tran!(circuit, tspan; tstops=[user_t])
    @test any(t -> abs(t - user_t) < 1e-15, sol_merged.t)
    @test hits(sol_merged, edges)

    sol_merged_ode = tran!(circuit, tspan; solver=Rodas5P(), tstops=[user_t])
    @test any(t -> abs(t - user_t) < 1e-15, sol_merged_ode.t)
    @test hits(sol_merged_ode, edges)

    # alter() recomputes edges from the new `td` parameter
    circuit2 = Cadnip.MNA.alter(circuit; td=3e-6)
    edges2 = pulse_edges(3e-6, 10e-6, tspan)
    prob2 = ODEProblem(circuit2, tspan)
    @test all(e -> any(t2 -> abs(t2 - e) < 1e-9, prob2.kwargs[:tstops]), edges2)
    @test !any(t2 -> abs(t2 - 1e-6) < 1e-9, prob2.kwargs[:tstops])  # old td's edge is gone

    # PWL vertex times land exactly in sol.t too (SPICE-parsed netlist path)
    pwl_code = """
    * PWL tstop test
    i1 0 vout PWL(1e-6 0 5e-6 1 9e-6 0.5)
    R1 vout 0 1e3
    """
    ast_pwl = NyanSpectreNetlistParser.parse(IOBuffer(pwl_code); start_lang=:spice, implicit_title=true)
    code_pwl = Cadnip.make_mna_circuit(ast_pwl)
    m_pwl = Module()
    Base.eval(m_pwl, :(using Cadnip.MNA))
    Base.eval(m_pwl, :(using Cadnip: ParamLens))
    Base.eval(m_pwl, :(using Cadnip.SpectreEnvironment))
    builder_pwl = Base.eval(m_pwl, code_pwl)
    circuit_pwl = Base.invokelatest(MNACircuit, builder_pwl, (;), MNASpec(temp=27.0))
    sol_pwl = tran!(circuit_pwl, (0.0, 10e-6))
    @test hits(sol_pwl, [1e-6, 5e-6, 9e-6])
end

end # module transient_tests
