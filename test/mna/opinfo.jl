#==============================================================================#
# Operating-point info channel — device terminal currents
# (doc/operating_point_info.md)
#
# These are low-level stamping-mechanics tests: that devices register the
# current into each of their terminals on the deferred MNAContext channel as
# they stamp, that the readout evaluates and folds those entries against a
# solution vector, and — crucially — that none of it is visible to the DC or
# transient value path (the hot-path DirectStampContext carries no op channel at
# all). Circuit-behavior assertions live in `test/opinfo.jl`, on netlists.
#==============================================================================#

using Test
using Cadnip
using Cadnip.MNA
using Cadnip.MNA: MNAContext, get_node!, stamp!, Resistor, Capacitor, Diode, DiodeWithCap
using Cadnip.MNA: SimpleMOSFET, VoltageSource, CurrentSource, resolve_index
using Cadnip.MNA: reset_for_restamping!, op_enabled, terminal_current_name
using Cadnip.MNA: register_terminal_current!, register_ohmic_terminal_current!
using Cadnip.MNA: num_terminal_currents, terminal_currents
using Cadnip.MNA: solve_dc, MNASpec, MNACircuit
using Cadnip.SpectreEnvironment

# A Verilog-A resistor with a series "contact" branch through an internal node:
# the terminal current has to come off the branch that reaches `p`, not off the
# branch that carries the resistance.
va"""
module VASeriesRes(p, n);
    parameter real r = 900.0;
    parameter real rc = 100.0;
    inout p, n;
    electrical p, n, mid;
    analog begin
        I(p,mid) <+ V(p,mid)/rc;
        I(mid,n) <+ V(mid,n)/r;
    end
endmodule
"""

@testset "operating-point info channel" begin

    @testset "terminal_current_name composes instance and terminal" begin
        @test terminal_current_name(:m1, :d) === :i_m1_d
        @test terminal_current_name(:r1, Symbol("")) === :i_r1
        @test terminal_current_name(Symbol(""), :d) === :i_d
    end

    @testset "resistor registers both ends, evaluated against the solution" begin
        # The resistor never sees `x`, so it registers a conductance and its
        # branch; the readout turns that into G·(V_p − V_n).
        ctx = MNAContext()
        a = get_node!(ctx, :a)      # index 1
        b = get_node!(ctx, :b)      # index 2
        stamp!(Resistor(1000.0; name=:r1), ctx, a, b)

        @test num_terminal_currents(ctx) == 2
        tc = Dict(terminal_currents(ctx, [3.0, 1.0]))
        @test tc[:i_r1_p] ≈ 2e-3    # (3 − 1)/1k, into the device at p
        @test tc[:i_r1_n] ≈ -2e-3   # and out the other end
    end

    @testset "diode registers its junction current at the bias" begin
        Is, Vt = 1e-14, 0.026
        Vbias = 0.6
        I0 = Is * (exp(Vbias / Vt) - 1.0)

        ctx = MNAContext()
        a = get_node!(ctx, :a)
        stamp!(Diode(Is=Is, Vt=Vt, limit=false, name=:d1), ctx, a, 0; x=[Vbias])

        tc = Dict(terminal_currents(ctx, [Vbias]))
        @test tc[:i_d1_p] ≈ I0
        @test tc[:i_d1_n] ≈ -I0

        # Same for the junction-capacitance variant (the charge carries no DC).
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        stamp!(DiodeWithCap(Is=Is, Vt=Vt, name=:d2), ctx, a, 0; x=[Vbias])
        @test Dict(terminal_currents(ctx, [Vbias]))[:i_d2_p] ≈ I0
    end

    @testset "MOSFET reports drain, gate and source" begin
        Vth, K = 0.5, 1e-3
        Vg, Vd = 1.5, 2.0                     # saturation: Vds > Vgs − Vth
        Ids = K / 2 * (Vg - Vth)^2

        ctx = MNAContext()
        d = get_node!(ctx, :d)                # index 1
        g = get_node!(ctx, :g)                # index 2
        stamp!(SimpleMOSFET(Vth=Vth, K=K, lambda=0.0, name=:m1), ctx, d, g, 0; x=[Vd, Vg])

        tc = Dict(terminal_currents(ctx, [Vd, Vg]))
        @test tc[:i_m1_d] ≈ Ids
        @test tc[:i_m1_s] ≈ -Ids
        @test tc[:i_m1_g] == 0.0              # capacitive gate: no DC current
        @test tc[:i_m1_d] + tc[:i_m1_g] + tc[:i_m1_s] ≈ 0.0 atol=1e-15
    end

    @testset "current source reports current into the device" begin
        # The source drives 1 mA into node p, so 1 mA flows out of the device
        # there — the reported sign is the current *into* the terminal.
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        stamp!(CurrentSource(1e-3; name=:i1), ctx, p, 0)
        tc = Dict(terminal_currents(ctx, [0.0]))
        @test tc[:i_i1_p] ≈ -1e-3
        @test tc[:i_i1_n] ≈ 1e-3
    end

    @testset "repeated names sum; a rebuild does not duplicate" begin
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        register_terminal_current!(ctx, :x1, :a, 1e-3)
        register_terminal_current!(ctx, :x1, :a, 2e-3)
        register_ohmic_terminal_current!(ctx, :x1, :a, p, 0, 1e-3)   # +1 mA at V=1
        @test num_terminal_currents(ctx) == 3
        tc = terminal_currents(ctx, [1.0])
        @test length(tc) == 1                 # folded into one entry
        @test tc[1] == (:i_x1_a => 4e-3)

        # Restamping starts the channel over rather than appending to it.
        reset_for_restamping!(ctx)
        @test num_terminal_currents(ctx) == 0
        get_node!(ctx, :a); get_node!(ctx, :b)
        stamp!(Resistor(1000.0; name=:r1), ctx, 1, 2)
        @test num_terminal_currents(ctx) == 2
    end

    @testset "ohmic entries read ground and short vectors as zero" begin
        # Structure discovery stamps against ZERO_VECTOR / a stale-length x; the
        # readout must not throw there, it just reports nothing.
        ctx = MNAContext()
        a = get_node!(ctx, :a)
        b = get_node!(ctx, :b)
        stamp!(Resistor(1000.0; name=:r1), ctx, a, b)
        @test Dict(terminal_currents(ctx, Float64[]))[:i_r1_p] == 0.0
        @test Dict(terminal_currents(ctx, [5.0]))[:i_r1_p] ≈ 5e-3   # V_b unknown ⇒ 0
    end

    @testset "Verilog-A terminals sum over the branches that reach them" begin
        # The current into `p` arrives through the contact branch I(p,mid); the
        # internal node's own branches stay inside the device.
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)
        stamp!(VASeriesRes(), ctx, p, n; _mna_instance_=:x1, _mna_x_=Float64[])
        names = [q.first for q in terminal_currents(ctx, Float64[])]
        @test :i_x1_p in names
        @test :i_x1_n in names
        @test all(!occursin("mid", String(nm)) for nm in names)   # internal node stays inside
    end

    @testset "hot path carries no op channel" begin
        # `op_enabled` is the compile-time gate the Verilog-A accumulation sits
        # behind; on the restamping context it is a constant false, which is what
        # lets the whole accumulation fold away.
        @test op_enabled(MNAContext()) === true

        # An RC low-pass through the high-level API drives DirectStampContext
        # restamping, where the resistor stamp calls register_ohmic_terminal_current!
        # — a no-op there. If the no-op were missing this would error.
        circuit = MNACircuit(sp"""
        V1 in 0 DC 1
        R1 in out 1k
        C1 out 0 1u
        """i)
        sol = tran!(circuit, (0.0, 20e-3))
        @test sol[:out][end] ≈ 1.0 atol=1e-2
    end

    @testset "numerics unchanged: divider still solves exactly" begin
        circuit = MNACircuit(sp"""
        V1 in 0 DC 6
        R1 in out 1k
        R2 out 0 2k
        """i)
        sol = dc!(circuit)
        @test sol[:out] ≈ 4.0
        @test sol[:in] ≈ 6.0
    end
end
