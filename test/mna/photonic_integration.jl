using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, get_node!, stamp!, MNACircuit, voltage, assemble!
using CedarSim.MNA: VoltageSource, Resistor
using CedarSim: dc!
using PhotonicModels

# Helper: create 4 optical port nodes and ground them with 1Ω resistors
function make_optical_port!(ctx, name)
    nodes = [get_node!(ctx, Symbol(name, "_", i)) for i in 0:3]
    for n in nodes
        stamp!(Resistor(1.0), ctx, n, 0)
    end
    return nodes
end

@testset "Photonic Integration" begin

    @testset "Tier 1: Leaf model compilation" begin
        @test isdefined(PhotonicModels, :Polar2Cartesian)
        @test isdefined(PhotonicModels, :Cartesian2Polar)
        @test isdefined(PhotonicModels, :PolToCart)
        @test isdefined(PhotonicModels, :CartToPol)
        @test isdefined(PhotonicModels, :CartesianMultiplier)
        @test isdefined(PhotonicModels, :CartesianAdder)
        @test isdefined(PhotonicModels, :CartAdd)
        @test isdefined(PhotonicModels, :CartMul)
        @test isdefined(PhotonicModels, :CartSub)
        @test isdefined(PhotonicModels, :Terminator)
        @test isdefined(PhotonicModels, :ReflectionInterface)
        @test isdefined(PhotonicModels, :DirectionalCoupler)
        @test isdefined(PhotonicModels, :OneTwoCoupler)
        @test isdefined(PhotonicModels, :OneTwoLoopback)
        @test isdefined(PhotonicModels, :OneTwoSplitter)
        @test isdefined(PhotonicModels, :TwoOneCombiner)
        @test isdefined(PhotonicModels, :NonlinearCapacitor)
    end

    @testset "Tier 2: Composite model compilation" begin
        @test isdefined(PhotonicModels, :Attenuator)
        @test isdefined(PhotonicModels, :Isolator)
        @test isdefined(PhotonicModels, :PhaseShifter)
    end

    @testset "Attenuator: 6dB transfer function" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            inp = make_optical_port!(ctx, :inp)
            outp = make_optical_port!(ctx, :outp)
            stamp!(Attenuator(attenuation=6.0), ctx, inp..., outp...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))
        gain = 10^(-6.0/20)  # ≈ 0.5012

        tp0 = findfirst(==(:Attenuator_transfer_pol_0), sol.node_names)
        t0 = findfirst(==(:Attenuator_transfer_0), sol.node_names)
        t1 = findfirst(==(:Attenuator_transfer_1), sol.node_names)

        @test tp0 !== nothing
        @test isapprox(sol.x[tp0], gain; atol=1e-4)
        @test isapprox(sol.x[t0], gain; atol=1e-4)
        @test isapprox(sol.x[t1], 0.0; atol=1e-4)
    end

    @testset "Isolator: forward pass-through + backward attenuation" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            inp = make_optical_port!(ctx, :inp)
            outp = make_optical_port!(ctx, :outp)
            stamp!(Isolator(isolation=20.0), ctx, inp..., outp...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))
        gain = 10^(-20.0/20)  # = 0.1

        tp0 = findfirst(==(:Isolator_transfer_pol_0), sol.node_names)
        t0 = findfirst(==(:Isolator_transfer_0), sol.node_names)

        @test tp0 !== nothing
        @test isapprox(sol.x[tp0], gain; atol=1e-4)
        @test isapprox(sol.x[t0], gain; atol=1e-4)
    end

    @testset "PhaseShifter: 90deg transfer" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            inp = make_optical_port!(ctx, :inp)
            outp = make_optical_port!(ctx, :outp)
            stamp!(PhaseShifter(phase=90.0), ctx, inp..., outp...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))

        tp0 = findfirst(==(:PhaseShifter_transfer_pol_0), sol.node_names)
        tp1 = findfirst(==(:PhaseShifter_transfer_pol_1), sol.node_names)
        t0 = findfirst(==(:PhaseShifter_transfer_0), sol.node_names)
        t1 = findfirst(==(:PhaseShifter_transfer_1), sol.node_names)

        @test tp0 !== nothing
        @test isapprox(sol.x[tp0], 1.0; atol=1e-4)
        expected_angle = (-90.0 / 180 * π) % (2π)
        @test isapprox(sol.x[tp1], expected_angle; atol=1e-4)
        @test t0 !== nothing
        @test t1 !== nothing
    end

    @testset "DirectionalCoupler: kappa=0.5" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            in1 = make_optical_port!(ctx, :in1)
            in2 = make_optical_port!(ctx, :in2)
            out1 = make_optical_port!(ctx, :out1)
            out2 = make_optical_port!(ctx, :out2)
            stamp!(DirectionalCoupler(kappa=0.5), ctx, in1..., in2..., out1..., out2...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))
        @test length(sol.x) > 0
    end

    @testset "Terminator: stamp and solve" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            inp = make_optical_port!(ctx, :inp)
            stamp!(Terminator(), ctx, inp...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))
        @test length(sol.x) >= 4
    end

    @testset "Attenuator: matrix structure" begin
        ctx = MNAContext()
        nodes = [get_node!(ctx, Symbol("n$i")) for i in 1:8]
        stamp!(Attenuator(attenuation=3.0), ctx, nodes...)
        sys = assemble!(ctx)
        # 8 port nodes + 4 internal (transfer_pol[0:1], transfer[0:1]) + child nodes
        @test sys.n_nodes >= 12
    end

end # Photonic Integration
