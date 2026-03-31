#==============================================================================#
# absdelay / DDE Tests
#
# Tests for transport delay via DDEProblem + MethodOfSteps.
#==============================================================================#

using Test
using Cadnip
using Cadnip.MNA
using Cadnip.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!
using Cadnip.MNA: MNACircuit, reset_for_restamping!
using Cadnip.MNA: VoltageSource, Resistor
using Cadnip: tran!
using DelayDiffEq: MethodOfSteps
using OrdinaryDiffEq: Rodas5P

# VA delay element: output tracks delayed input voltage
# I(out) <+ V(out) - absdelay(V(inp), tau)
# This creates: at steady state V(out) ≈ V(inp, t-tau) (current enforces equality)
va"""
module VADelay(inp, out);
    parameter real tau = 1e-9;
    inout inp, out;
    electrical inp, out;
    analog begin
        I(out) <+ V(out) - absdelay(V(inp), tau);
    end
endmodule
"""

@testset "absdelay basic" begin
    # Circuit: V_source(inp) -> VADelay -> out with load resistor
    function build_delay(params, spec, t::Real=0.0; x=Float64[], ctx=nothing, _mna_h_=nothing, _mna_h_p_=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        inp = get_node!(ctx, :inp)
        out = get_node!(ctx, :out)

        # Step voltage source: 0V at t<=0, 1V for t>0
        stamp!(VoltageSource(0.0, 0.0im, t -> t > 0 ? 1.0 : 0.0, :V1), ctx, inp, 0, t, spec.mode)

        # VA delay device with tau parameter
        stamp!(VADelay(; tau=params.tau), ctx, inp, out;
               _mna_t_=t, _mna_mode_=spec.mode, _mna_x_=x, _mna_spec_=spec,
               _mna_h_=_mna_h_, _mna_h_p_=_mna_h_p_)

        # Load resistor on output
        stamp!(Resistor(1000.0; name=:Rload), ctx, out, 0)

        return ctx
    end

    tau = 1e-9
    circuit = MNACircuit(build_delay; tau=tau)

    # Solve with DDE solver
    tspan = (0.0, 5e-9)
    sol = tran!(circuit, tspan;
                solver=MethodOfSteps(Rodas5P()),
                constant_lags=[tau])

    # Find the output node index
    # The workspace is in sol.prob.p, node names in the compiled structure
    ws = sol.prob.p
    out_idx = ws.dctx.node_to_idx[:out]

    # Before the delay time, output should be near 0V (initial DC with delay)
    t_early = tau / 2
    v_early = sol(t_early)[out_idx]
    @test abs(v_early) < 0.2

    # After delay has passed, output should approach 1V
    t_late = 4 * tau
    v_late = sol(t_late)[out_idx]
    @test abs(v_late - 1.0) < 0.2
end
