module ddx_tests

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current
using CedarSim.MNA: VoltageSource
using Test

# Load the NLVCR model from .va file
# The model is:
#   cdrain = R*V(g,s)**2
#   I(d,s) <+ V(d,s)*ddx(cdrain, V(g,s))
#
# Where ddx(cdrain, V(g,s)) = 2*R*V(g,s)
# So I(d,s) = V(d,s) * 2*R*V(g,s) = 2*R*V(d,s)*V(g,s)
const NLVCR = CedarSim.load_VA_model(joinpath(@__DIR__, "NLVCR.va"))

@testset "ddx() 3-terminal VA device" begin
    # Circuit:
    # V1(5V) between vcc and gnd
    # V2(3V) between vg and gnd
    # NLVCR(R=2) with d=vcc, g=vg, s=gnd
    #
    # Expected current:
    # I(d,s) = 2*R*V(d,s)*V(g,s) = 2*2*5*3 = 60A
    # V1 sources this current, so I_V1 = -60A

    function VRcircuit(params, spec)
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        vg = get_node!(ctx, :vg)

        stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
        stamp!(VoltageSource(3.0; name=:V2), ctx, vg, 0)
        stamp!(NLVCR(R=2.0), ctx, vcc, vg, 0)

        return ctx
    end

    ctx = VRcircuit((;), MNASpec())
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    # Verify voltages
    @test isapprox(voltage(sol, :vcc), 5.0; atol=1e-10)
    @test isapprox(voltage(sol, :vg), 3.0; atol=1e-10)

    # Verify current: V1 sources 60A, so I_V1 = -60A
    # I(d,s) = 2*R*V(d,s)*V(g,s) = 2*2*5*3 = 60A
    expected_I = -5.0 * 2.0 * 2.0 * 3.0
    @test isapprox(current(sol, :I_V1), expected_I; atol=1e-6)
end

end # module ddx_tests
