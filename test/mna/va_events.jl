#==============================================================================#
# VA Voltage-Dependent Event Detection Tests
#
# Tests for the Part B pipeline: comparison interception (vasim.jl),
# condition slots (context.jl/value_only.jl/precompile.jl), va_cmp_*
# runtime (va_events.jl), and va_event_callback/tran! wiring (solve.jl,
# sweeps.jl).
#==============================================================================#

using Test
using Cadnip
using Cadnip.MNA
using Cadnip.MNA: MNAContext, MNASpec, get_node!, stamp!, reset_for_restamping!
using Cadnip.MNA: VoltageSource, Resistor, Capacitor, MNACircuit
using Cadnip.MNA: build_with_detection, create_direct_stamp_context, reset_direct_stamp!
using SciMLBase
using OrdinaryDiffEq: Rodas5P
using LinearSolve: KLUFactorization

@testset "VA Event Detection" begin

    #==========================================================================#
    # Comparator model driven by a slow SIN - crossing time should land in
    # sol.t within tight tolerance when va_events=true.
    #==========================================================================#

    va"""
    module va_ev_comparator(p, n, out);
        inout p, n, out;
        electrical p, n, out;
        parameter real vth = 0.5;
        real vin;
        analog begin
            vin = V(p, n);
            if (vin > vth) begin
                V(out) <+ 1.0;
            end else begin
                V(out) <+ 0.0;
            end
        end
    endmodule
    """

    function comparator_sin_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        vsrc = get_node!(ctx, :vsrc)
        vp = get_node!(ctx, :vp)
        vout = get_node!(ctx, :vout)
        # Slow half-sine ramp 0->1 over the sim window, threshold crossed once.
        # Filtered through a fast RC (tau=1us, tspan=1s) before reaching the
        # comparator so `vin` is a genuine differential state - a purely
        # algebraic node (an ideal voltage source read straight into the
        # comparator) makes the ODE mass matrix fully singular, which starves
        # Rosenbrock's dense output of the accuracy needed for precise event
        # localization (confirmed: same source driven directly gives ~1e-2
        # crossing error vs. ~1e-9 through this filter). Real VA models always
        # sit inside a circuit with real parasitic capacitance, so this
        # matches the realistic case Part B targets.
        stamp!(VoltageSource(0.0; tran=MNA.SinWave(0.5, 0.5, 0.5, 0.0, 0.0, -90.0), name=:V1),
               ctx, vsrc, 0, t, spec.mode)
        stamp!(Resistor(1.0), ctx, vsrc, vp)
        stamp!(Capacitor(1e-4), ctx, vp, 0)
        stamp!(va_ev_comparator(), ctx, vp, 0, vout; _mna_x_=x)
        stamp!(Resistor(1e3), ctx, vout, 0)
        return ctx
    end

    @testset "crossing time lands in sol.t" begin
        circuit = MNACircuit(comparator_sin_circuit)
        tspan = (0.0, 1.0)

        ctx = build_with_detection(circuit)
        @test ctx.n_conditions == 1
        @test ctx.condition_is_vdep == [true]  # vin is V(p,n)-derived, genuinely voltage-dependent

        # SinWave(vo=0.5,va=0.5,freq=0.5,phase=-90): vsrc(t) = 0.5 - 0.5*cos(pi*t),
        # crosses vth=0.5 at t=0.5 (quarter period); vp tracks vsrc within the
        # RC=100us lag (negligible against the 1s timescale).
        sol = tran!(circuit, tspan; solver=Rodas5P(linsolve=KLUFactorization()), va_events=true,
                    abstol=1e-9, reltol=1e-7, maxiters=1_000_000)
        @test any(t -> abs(t - 0.5) < 1e-3, sol.t)
    end

    @testset "n_conditions matches lexical comparison count" begin
        ctx = build_with_detection(MNACircuit(comparator_sin_circuit))
        @test ctx.n_conditions == 1
    end

    #==========================================================================#
    # Two instances -> 2x slots
    #==========================================================================#

    @testset "two instances allocate 2x slots" begin
        function two_instance(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vp1 = get_node!(ctx, :vp1)
            vp2 = get_node!(ctx, :vp2)
            vout1 = get_node!(ctx, :vout1)
            vout2 = get_node!(ctx, :vout2)
            stamp!(VoltageSource(1.0; name=:V1), ctx, vp1, 0)
            stamp!(VoltageSource(0.2; name=:V2), ctx, vp2, 0)
            stamp!(va_ev_comparator(), ctx, vp1, 0, vout1; _mna_x_=x)
            stamp!(va_ev_comparator(), ctx, vp2, 0, vout2; _mna_x_=x)
            stamp!(Resistor(1e3), ctx, vout1, 0)
            stamp!(Resistor(1e3), ctx, vout2, 0)
            return ctx
        end
        ctx = build_with_detection(MNACircuit(two_instance))
        @test ctx.n_conditions == 2
        @test ctx.condition_is_vdep == [true, true]
    end

    #==========================================================================#
    # Nested-if model: stable slot count over discovery + many restamps,
    # no overflow warning.
    #==========================================================================#

    va"""
    module va_ev_nested(p, n, out);
        inout p, n, out;
        electrical p, n, out;
        parameter real vth1 = 0.5;
        parameter real vth2 = 1.5;
        real vin;
        analog begin
            vin = V(p, n);
            if (vin > vth1) begin
                if (vin > vth2) begin
                    V(out) <+ 2.0;
                end else begin
                    V(out) <+ 1.0;
                end
            end else begin
                V(out) <+ 0.0;
            end
        end
    endmodule
    """

    function nested_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        vp = get_node!(ctx, :vp)
        vout = get_node!(ctx, :vout)
        stamp!(VoltageSource(get(params, :vin, 1.0); name=:V1), ctx, vp, 0)
        stamp!(va_ev_nested(), ctx, vp, 0, vout; _mna_x_=x)
        stamp!(Resistor(1e3), ctx, vout, 0)
        return ctx
    end

    @testset "nested-if: correct count, stable over many restamps" begin
        circuit = MNACircuit(nested_circuit; vin=2.0)
        ctx = build_with_detection(circuit)
        @test ctx.n_conditions == 2
        @test ctx.condition_is_vdep == [true, true]

        # Region selection sanity check (not a precision test - see the
        # inline devices.jl comment on n-terminal branch stamping for why
        # this isn't exactly 2.0/1.0/0.0).
        for (vin, expected) in [(2.0, 2.0), (1.0, 1.0), (0.1, 0.0)]
            sol = dc!(MNACircuit(nested_circuit; vin=vin))
            @test isapprox(sol[:vout], expected; atol=0.01)
        end

        # Stability over many restamps: no overflow warning, slot count fixed.
        cs = MNA.compile_structure(circuit.builder, circuit.params, circuit.spec; ctx=ctx)
        ws = MNA.create_workspace(cs; ctx=ctx)
        @test ws.structure.n_conditions == 2
        @test_logs begin
            for _ in 1:50
                MNA.fast_rebuild!(ws, zeros(MNA.system_size(ws)), 0.0)
            end
        end
    end

    #==========================================================================#
    # Parameter-only-conditional model: slots exist, zero events, identical
    # results to a plain (non-branching) resistor divider.
    #==========================================================================#

    va"""
    module va_ev_param_only(p, n, out);
        inout p, n, out;
        electrical p, n, out;
        parameter real mode_sel = 1.0;
        analog begin
            if (mode_sel > 0.5) begin
                V(out, n) <+ 0.5 * V(p, n);
            end else begin
                V(out, n) <+ 0.25 * V(p, n);
            end
        end
    endmodule
    """

    @testset "parameter-only conditional: slots exist, zero runtime events" begin
        function param_only_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vp = get_node!(ctx, :vp)
            vout = get_node!(ctx, :vout)
            stamp!(VoltageSource(2.0; name=:V1), ctx, vp, 0)
            stamp!(va_ev_param_only(), ctx, vp, 0, vout; _mna_x_=x)
            return ctx
        end

        ctx = build_with_detection(MNACircuit(param_only_circuit))
        @test ctx.n_conditions == 1  # slot exists even though it never toggles at runtime
        @test ctx.condition_is_vdep == [false]  # mode_sel is a parameter, never a Dual

        sol = dc!(MNACircuit(param_only_circuit))
        @test isapprox(sol[:vout], 1.0; atol=1e-6)  # 0.5 * 2.0

        # va_event_callback excludes the non-voltage-dependent slot entirely
        cs = MNA.compile_structure(MNACircuit(param_only_circuit).builder,
                                    MNACircuit(param_only_circuit).params,
                                    MNACircuit(param_only_circuit).spec; ctx=ctx)
        ws = MNA.create_workspace(cs; ctx=ctx)
        @test MNA.va_event_callback(ws) === nothing
    end

end
