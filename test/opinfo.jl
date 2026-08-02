#==============================================================================#
# Device terminal currents and operating-point variables in the operating point
# (doc/operating_point_info.md)
#
# The behavioral half of the op-info channel: netlists driven through the
# high-level API, asserting that `dc!` reports each device's terminal currents
# and small-signal variables, and that they agree with what the circuit says
# they must be (KCL at a node, the branch current of the source feeding them,
# the hand-derived bias of a stage, a numerical derivative of the reported
# current). The stamping mechanics live in `test/mna/opinfo.jl`.
#==============================================================================#

module opinfo_tests

using Test
using Cadnip
using VADistillerModels                     # .model … d / nmos level=1 → VA models
using Cadnip.MNA: MNACircuit, terminal_currents, op_vars
using Cadnip: dc!

const divider = sp"""
* resistive divider
V1 in 0 DC 6
R1 in out 1k
R2 out 0 2k
"""i

const rectifier = sp"""
V1 in 0 DC 5
R1 in out 1k
D1 out 0 dmod
.model dmod d is=76.9p n=1.45
"""i

const cs_stage = sp"""
.model nch nmos level=1 vto=0.7 kp=100u lambda=0.01
.param vbias=1.1472
Vdd vdd 0 DC 5
Vin gate 0 DC vbias
M1 drain gate 0 0 nch w=20u l=1u
Rd vdd drain 10k
"""i

const with_subckt = sp"""
.subckt divider a b
R1 a mid 1k
R2 mid b 1k
.ends
V1 in 0 DC 4
X1 in 0 divider
"""i

@testset "device terminal currents" begin

    @testset "resistors: KCL at a node, from the report alone" begin
        op = dc!(MNACircuit(divider))
        I = 6.0 / 3000.0                    # loop current

        @test op[:i_r1_p] ≈ I
        @test op[:i_r1_n] ≈ -I
        @test op[:i_r2_p] ≈ I
        @test op[:i_r2_n] ≈ -I

        # The two devices that meet at :out balance — this is the thing the
        # solution vector cannot tell you, because KCL already summed them.
        @test op[:i_r1_n] + op[:i_r2_p] ≈ 0.0 atol=1e-15

        # And they agree with the one branch current the system does carry.
        @test abs(op[:I_v1]) ≈ I
    end

    @testset "diode: the junction current is the loop current" begin
        op = dc!(MNACircuit(rectifier))
        @test 0.6 < op[:out] < 0.8

        # The VA diode's terminals are (a, c). Current in at the anode, out at
        # the cathode, and equal to what the series resistor delivers.
        ir = (op[:in] - op[:out]) / 1e3
        @test op[:i_d1_a] ≈ ir rtol=1e-6
        @test op[:i_d1_c] ≈ -ir rtol=1e-6
        @test op[:i_r1_n] ≈ -ir rtol=1e-9
    end

    @testset "MOSFET: the drain current without inferring it" begin
        op = dc!(MNACircuit(cs_stage))

        # Square law at the design bias: ID = ½·kp·(W/L)·(VGS − VTO)² ≈ 200 µA
        # (λ = 0.01 V⁻¹ pushes it a few percent up).
        @test isapprox(op[:i_m1_d], 200e-6; rtol=0.05)

        # It matches the supply branch, which is how this had to be read before.
        @test isapprox(op[:i_m1_d], -op[:I_vdd]; rtol=1e-6)
        # …and the load resistor carrying the same current.
        @test isapprox(op[:i_m1_d], op[:i_rd_p]; rtol=1e-6)

        # A DC gate draws nothing, and the device conserves charge across its
        # four terminals.
        @test op[:i_m1_g] ≈ 0.0 atol=1e-9
        @test op[:i_m1_d] + op[:i_m1_g] + op[:i_m1_s] + op[:i_m1_b] ≈ 0.0 atol=1e-9
    end

    @testset "subcircuit instances report under their hierarchical name" begin
        op = dc!(MNACircuit(with_subckt))
        names = [p.first for p in terminal_currents(op)]
        @test :i_x1_r1_p in names
        @test :i_x1_r2_p in names
        I = 4.0 / 2000.0
        @test op[:i_x1_r1_p] ≈ I
        @test op[:i_x1_r1_n] + op[:i_x1_r2_p] ≈ 0.0 atol=1e-15
    end

    @testset "the operating point enumerates them" begin
        op = dc!(MNACircuit(divider))

        ks = keys(op)
        vs = values(op)
        @test length(ks) == length(vs)
        @test :i_r1_p in ks
        @test length(ks) == length(op.node_names) + length(op.current_names) +
                            length(terminal_currents(op)) + length(op_vars(op))

        kv = Dict(pairs(op))
        @test kv[:i_r1_p] ≈ op[:i_r1_p]
        for (k, v) in pairs(op)
            @test op[k] == v
        end

        @test haskey(op, :i_r1_p)
        @test !haskey(op, :i_r9_p)
        @test get(op, :i_r1_p, NaN) ≈ op[:i_r1_p]
        @test isnan(get(op, :i_r9_p, NaN))

        # `show` reports them alongside the voltages and branch currents.
        out = sprint(show, MIME"text/plain"(), op)
        @test occursin("Device Terminal Currents:", out)
        @test occursin("i_r1_p", out)
    end

    @testset "a re-solve reports the moved operating point" begin
        # `alter` re-solves; the terminal currents follow the new bias rather
        # than carrying over from the first solve's channel.
        c = MNACircuit(cs_stage)
        cold = dc!(c)
        hot = dc!(Cadnip.MNA.alter(c; vbias=1.20))
        @test hot[:i_m1_d] > cold[:i_m1_d] * 1.1
    end
end

#==============================================================================#
# Device operating-point variables — the model's own small-signal numbers.
#
# These come from the Verilog-A `(* desc = … *) real gm;` declarations the
# VADistiller models already carry, so `.model nch nmos level=1` reports MOS1's
# whole `.op` output with nothing to configure. The hand derivation below is the
# same square law `cs_stage` was sized against.
#==============================================================================#

# Hand derivation at the design bias (Shichman-Hodges, level 1):
#   K = kp·W/L = 100 µA/V² · 20 = 2 mA/V²,  VOV = 1.1472 − 0.7 = 447.2 mV
#   ID = ½·K·VOV² = 200 µA,  gm = K·VOV = 894.4 µS
const VOV = 1.1472 - 0.7
const GM  = 100e-6 * 20.0 * VOV

# The same stage wrapped in a subcircuit, to check the hierarchical instance
# name. Netlists with a `.subckt` have to be loaded at top level — codegen emits
# the subcircuit builder as a `const`. The model card is named `nchw`, not
# `nch`: two netlists in one module that name the same `.model` collide, and the
# later card silently wins (see doc/scratchpad.md, the codegen name-collision
# item) — a second `nch` here would strip `lambda` off `cs_stage` above.
const wrapped_stage = sp"""
.model nchw nmos level=1 vto=0.7 kp=100u lambda=0.01
.subckt stage d g
M1 d g 0 0 nchw w=20u l=1u
.ends
Vdd vdd 0 DC 5
Vin gate 0 DC 1.1472
X1 drain gate stage
Rd vdd drain 10k
"""i

@testset "device operating-point variables" begin

    @testset "MOS1 reports gm, gds and the saturation voltage" begin
        op = dc!(MNACircuit(cs_stage))

        # The transconductance, straight from the model rather than differenced
        # out of a transfer curve. λ = 0.01 V⁻¹ lifts ID (and with it gm) a few
        # percent above the λ = 0 hand number.
        @test isapprox(op[:m1_gm], GM; rtol=0.05)

        # gm is a derivative, so check it *is* one: bump the gate and difference
        # the reported drain current across the bias.
        δ = 1e-3
        up = dc!(Cadnip.MNA.alter(MNACircuit(cs_stage); vbias=1.1472 + δ))
        dn = dc!(Cadnip.MNA.alter(MNACircuit(cs_stage); vbias=1.1472 - δ))
        @test isapprox(op[:m1_gm], (up[:i_m1_d] - dn[:i_m1_d]) / 2δ; rtol=0.02)

        # Output conductance: level 1 puts it at λ·ID.
        @test isapprox(op[:m1_gds], 0.01 * op[:i_m1_d]; rtol=0.10)

        # Region, in the model's own numbers rather than a hand-derived
        # overdrive: the device is saturated when VDS clears VDSAT.
        @test isapprox(op[:m1_vdsat], VOV; rtol=0.05)
        @test op[:m1_vds] > op[:m1_vdsat]
        @test isapprox(op[:m1_vgs], 1.1472; rtol=1e-6)
        @test isapprox(op[:m1_vds], op[:drain]; rtol=1e-6)
    end

    @testset "the diode reports its junction conductance" begin
        op = dc!(MNACircuit(rectifier))
        # gd = dI/dV at the junction — the small-signal resistance the AC
        # analysis linearizes around, which is I/(n·Vt) for an ideal junction.
        Vt = 1.380649e-23 * 300.15 / 1.602176634e-19
        @test isapprox(op[:d1_gd], op[:i_d1_a] / (1.45 * Vt); rtol=0.05)
        @test isapprox(op[:d1_vd], op[:out]; rtol=1e-6)
    end

    @testset "they enumerate, print, and follow a re-solve" begin
        op = dc!(MNACircuit(cs_stage))

        names = [p.first for p in op_vars(op)]
        @test :m1_gm in names
        @test :m1_gds in names
        @test all(startswith(String(n), "m1_") for n in names)   # only the one device

        @test :m1_gm in keys(op)
        @test Dict(pairs(op))[:m1_gm] ≈ op[:m1_gm]
        @test haskey(op, :m1_gm)
        @test !haskey(op, :m1_nosuchvar)
        @test isnan(get(op, :m1_nosuchvar, NaN))

        out = sprint(show, MIME"text/plain"(), op)
        @test occursin("Device Operating-Point Variables:", out)
        @test occursin("m1_gm", out)

        # More gate drive, more gm — the channel is refilled at each solve.
        hot = dc!(Cadnip.MNA.alter(MNACircuit(cs_stage); vbias=1.30))
        @test hot[:m1_gm] > op[:m1_gm] * 1.1
    end

    @testset "hierarchical instances, and devices that declare none" begin
        # A builtin resistor registers terminal currents but no variables, so a
        # netlist of nothing but resistors reports an empty channel.
        @test isempty(op_vars(dc!(MNACircuit(divider))))
        @test isempty(op_vars(dc!(MNACircuit(with_subckt))))

        # Inside a subcircuit the instance name is the hierarchical one, same as
        # the terminal currents.
        op = dc!(MNACircuit(wrapped_stage))
        @test isapprox(op[:x1_m1_gm], GM; rtol=0.05)
    end
end

end # module opinfo_tests
