using Cadnip
using Cadnip.MNA
using Test

@testset "Laplace and IDT operators" begin

@testset "idt() codegen" begin
    va"""
    module IDTTest(p, n);
        inout p, n;
        electrical p, n;
        analog begin
            V(p, n) <+ idt(1.0, 0.0);
        end
    endmodule
    """
    ctx = MNAContext()
    MNA.stamp!(IDTTest(), ctx, 1, 0; _mna_x_=zeros(10), _mna_instance_=Symbol(""))
    # Should have at least 1 state node for idt
    @test ctx.n_nodes >= 1
end

@testset "laplace_nd() first-order low-pass codegen" begin
    va"""
    module LaplaceLP(inp, out);
        inout inp, out;
        electrical inp, out;
        parameter real RC = 1e-3;
        analog begin
            V(out) <+ laplace_nd(V(inp), {1.0}, {1.0, RC});
        end
    endmodule
    """
    ctx = MNAContext()
    # Pre-register inp and out nodes
    MNA.get_node!(ctx, :inp)
    MNA.get_node!(ctx, :out)
    MNA.stamp!(LaplaceLP(), ctx, 1, 2; _mna_x_=zeros(10), _mna_instance_=Symbol(""))
    # Should have inp, out + 1 state node = 3
    @test ctx.n_nodes == 3
end

@testset "laplace_nd() 6th-order with extreme coefficients" begin
    va"""
    module LaplaceHigh(inp, out);
        inout inp, out;
        electrical inp, out;
        analog begin
            V(out) <+ laplace_nd(V(inp),
                {2.347141585877207e71, 0, 0, 0, 0, 0, 0},
                {2.347141585877208e71, 1.154657487839621e60, 2.840122475453116e48, 4.428868818445329e36, 4.604233134433859e24, 3.034545479782387e12, 1});
        end
    endmodule
    """
    ctx = MNAContext()
    MNA.get_node!(ctx, :inp)
    MNA.get_node!(ctx, :out)
    MNA.stamp!(LaplaceHigh(), ctx, 1, 2; _mna_x_=zeros(20), _mna_instance_=Symbol(""))
    # Should have inp, out + 6 state nodes = 8
    @test ctx.n_nodes == 8
end

@testset "laplace_nd() in full circuit - DC unity gain" begin
    # A low-pass filter H(s) = 1/(1+sRC) should pass DC with unity gain
    # Circuit: V1(1V) -> LPFilter -> 1Ω load to ground
    va"""
    module LPFilter2(inp, out);
        inout inp, out;
        electrical inp, out;
        analog begin
            I(out) <+ laplace_nd(V(inp), {1.0}, {1.0, 1e-3});
        end
    endmodule
    """

    function lp_builder(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        inp = MNA.get_node!(ctx, :inp)
        out = MNA.get_node!(ctx, :out)
        MNA.stamp!(MNA.VoltageSource(1.0; name=:V1), ctx, inp, 0)
        MNA.stamp!(LPFilter2(), ctx, inp, out; _mna_x_=x, _mna_t_=t, _mna_spec_=spec, _mna_instance_=Symbol(""))
        MNA.stamp!(MNA.Resistor(1.0), ctx, out, 0)
        return ctx
    end

    circuit = MNACircuit(lp_builder)
    sol = dc!(circuit)
    # At DC, H(0) = 1, so filter current = V(inp) * 1 = 1A
    # Through 1Ω: V(out) = 1V
    # I(out) <+ 1.0 means 1A flowing out of 'out' node
    # With 1Ω to ground: V(out) = -1.0 (current leaves node)
    @test abs(voltage(sol, :out)) ≈ 1.0 atol=0.01
end

@testset "idt() DC with non-zero initial condition" begin
    # idt(0.0, 5.0) should return 5.0 at DC (integrand is 0, IC is 5)
    va"""
    module IDTic(inp, out);
        inout inp, out;
        electrical inp, out;
        analog begin
            V(out) <+ idt(0.0, 5.0);
        end
    endmodule
    """

    function idt_ic_builder(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        inp = MNA.get_node!(ctx, :inp)
        out = MNA.get_node!(ctx, :out)
        MNA.stamp!(MNA.VoltageSource(1.0; name=:V1), ctx, inp, 0)
        MNA.stamp!(IDTic(), ctx, inp, out; _mna_x_=x, _mna_t_=t, _mna_spec_=spec, _mna_instance_=Symbol(""))
        return ctx
    end

    circuit = MNACircuit(idt_ic_builder)
    sol = dc!(circuit)
    # At DC, idt(0.0, 5.0) should return ic = 5.0
    # V(out) <+ 5.0 means V(out) - V(gnd) is constrained to 5.0
    @test voltage(sol, :out) ≈ 5.0 atol=0.01
end

@testset "laplace_nd() AC frequency response" begin
    va"""
    module LPAC(inp, out);
        inout inp, out;
        electrical inp, out;
        analog begin
            I(out) <+ laplace_nd(V(inp), {1.0}, {1.0, 1e-3});
        end
    endmodule
    """

    function lpac_builder(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        inp = MNA.get_node!(ctx, :inp)
        out = MNA.get_node!(ctx, :out)
        MNA.stamp!(MNA.VoltageSource(0.0; name=:V1, ac=1.0), ctx, inp, 0, t, spec.mode)
        MNA.stamp!(LPAC(), ctx, inp, out; _mna_x_=x, _mna_t_=t, _mna_spec_=spec, _mna_instance_=Symbol(""))
        MNA.stamp!(MNA.Resistor(1.0), ctx, out, 0)
        return ctx
    end

    circuit = MNACircuit(lpac_builder)
    ac_sol = ac!(circuit)

    # Check DC gain (should be ~1.0 = 0 dB)
    ωs_low = [2π * 1.0]  # 1 Hz
    resp_low = freqresp(ac_sol, :out, ωs_low)
    @test abs(resp_low[1]) ≈ 1.0 atol=0.1

    # Check at -3dB frequency: |H| ≈ 1/√2
    f_3dB = 1.0 / (2π * 1e-3)  # ≈ 159.15 Hz
    ωs_3dB = [2π * f_3dB]
    resp_3dB = freqresp(ac_sol, :out, ωs_3dB)
    @test abs(resp_3dB[1]) ≈ 1/√2 atol=0.1
end

@testset "laplace_nd() runtime conversion correctness" begin
    # Verify va_laplace_nd_dss produces correct state-space for simple filter
    # H(s) = 1/(1+s*RC), RC = 1e-3 → pole at s = -1000
    A, E, B, C, D = MNA.va_laplace_nd_dss((1.0,), (1.0, 1e-3))

    # Should be a 1st-order system (1 state)
    @test size(A, 1) == 1

    # DC gain: H(0) = -C*inv(A)*B + D should be 1.0
    dc_gain = -C * (A \ B) + D
    @test dc_gain[1,1] ≈ 1.0 atol=1e-10
end

@testset "laplace_nd() transient step response" begin
    # First-order LP: H(s) = 1/(1+s*RC), RC=1e-6
    # PWL step at t=1e-6: DC=0V, stays 0 until t=1µs, then jumps to 1V
    # After the step, output follows y(t) = 1 - exp(-(t-t_step)/RC)
    using OrdinaryDiffEq

    va"""
    module LPTran(inp, out);
        inout inp, out;
        electrical inp, out;
        analog begin
            I(out) <+ laplace_nd(V(inp), {1.0}, {1.0, 1e-6});
        end
    endmodule
    """

    RC = 1e-6
    t_step = 1e-6  # step occurs at t=1µs

    function lptran_builder(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        inp = MNA.get_node!(ctx, :inp)
        out = MNA.get_node!(ctx, :out)
        # PWL step: 0V until t_step, then 1V
        MNA.stamp!(MNA.VoltageSource(0.0; name=:V1, tran=(_t -> _t < t_step ? 0.0 : 1.0)), ctx, inp, 0, t, spec.mode)
        MNA.stamp!(LPTran(), ctx, inp, out; _mna_x_=x, _mna_t_=t, _mna_spec_=spec, _mna_instance_=Symbol(""))
        MNA.stamp!(MNA.Resistor(1.0), ctx, out, 0)
        return ctx
    end

    circuit = MNACircuit(lptran_builder)
    tspan = (0.0, t_step + 5*RC)
    sol = tran!(circuit, tspan; tstops=[t_step])

    out_idx = findfirst(==(:out), sol.prob.p.structure.node_names)

    # Before step: output should be ~0
    v_before = sol(t_step * 0.5)[out_idx]
    @test abs(v_before) < 0.01

    # At t_step + RC: output ≈ 1 - 1/e ≈ 0.632
    v_at_RC = sol(t_step + RC)[out_idx]
    @test isapprox(abs(v_at_RC), 1 - exp(-1); rtol=0.15)

    # At t_step + 5RC: near steady state ≈ 1.0
    v_at_5RC = sol(t_step + 5*RC)[out_idx]
    @test isapprox(abs(v_at_5RC), 1.0; rtol=0.05)
end

end # testset
