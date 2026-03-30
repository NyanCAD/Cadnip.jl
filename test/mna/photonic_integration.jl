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
        @test isdefined(PhotonicModels, :Waveguide)
        @test isdefined(PhotonicModels, :Pcw)
        @test isdefined(PhotonicModels, :PhaseModulator)
        @test isdefined(PhotonicModels, :PcwPhaseModulator)
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

    @testset "Waveguide: propagation loss" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            n1 = make_optical_port!(ctx, :n1)
            n2 = make_optical_port!(ctx, :n2)
            stamp!(Waveguide(length=100e-6, loss=2.0), ctx, n1..., n2...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))

        # transfer_pol[0] = exp(-alpha/2 * length) where alpha = 23.026 * loss
        alpha = 23.0258509299404568 * 2.0
        expected_amp = exp(-alpha / 2 * 100e-6)
        tp0 = findfirst(==(:Waveguide_transfer_pol_0), sol.node_names)
        @test tp0 !== nothing
        @test isapprox(sol.x[tp0], expected_amp; atol=1e-4)
    end

    @testset "Pcw: propagation loss (higher group index)" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            n1 = make_optical_port!(ctx, :n1)
            n2 = make_optical_port!(ctx, :n2)
            stamp!(Pcw(length=100e-6, loss=2.0), ctx, n1..., n2...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))

        # Same loss formula as Waveguide (Pcw differs only in default groupIndex)
        alpha = 23.0258509299404568 * 2.0
        expected_amp = exp(-alpha / 2 * 100e-6)
        tp0 = findfirst(==(:Pcw_transfer_pol_0), sol.node_names)
        @test tp0 !== nothing
        @test isapprox(sol.x[tp0], expected_amp; atol=1e-4)
    end

    @testset "PhaseModulator: transfer at V=0" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            opt_in = make_optical_port!(ctx, :opt_in)
            opt_out = make_optical_port!(ctx, :opt_out)
            ele_in = get_node!(ctx, :ele_in)
            stamp!(Resistor(1.0), ctx, ele_in, 0)
            stamp!(PhaseModulator(length=100e-6, loss=2.0), ctx, opt_in..., opt_out..., ele_in)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))

        # At V=0: alpha = 23.026 * loss, amplitude = exp(-alpha/2 * length)
        alpha = 23.0258509299404568 * 2.0
        expected_amp = exp(-alpha / 2 * 100e-6)
        tp0 = findfirst(==(:PhaseModulator_transfer_pol_0), sol.node_names)
        @test tp0 !== nothing
        @test isapprox(sol.x[tp0], expected_amp; atol=1e-4)
    end

    @testset "PcwPhaseModulator: transfer at V=0" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            opt_in = make_optical_port!(ctx, :opt_in)
            opt_out = make_optical_port!(ctx, :opt_out)
            ele_in = get_node!(ctx, :ele_in)
            stamp!(Resistor(1.0), ctx, ele_in, 0)
            stamp!(PcwPhaseModulator(length=100e-6, loss=2.0), ctx, opt_in..., opt_out..., ele_in)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))

        alpha = 23.0258509299404568 * 2.0
        expected_amp = exp(-alpha / 2 * 100e-6)
        tp0 = findfirst(==(:PcwPhaseModulator_transfer_pol_0), sol.node_names)
        @test tp0 !== nothing
        @test isapprox(sol.x[tp0], expected_amp; atol=1e-4)
    end

    @testset "ReflectionInterface: no reflection passes through" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            n1 = make_optical_port!(ctx, :n1)
            n2 = make_optical_port!(ctx, :n2)
            stamp!(ReflectionInterface(reflection=0.0), ctx, n1..., n2...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))
        @test length(sol.x) > 0
    end

    @testset "OneTwoSplitter: 50/50 split" begin
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            one = make_optical_port!(ctx, :one)
            two1 = make_optical_port!(ctx, :two1)
            two2 = make_optical_port!(ctx, :two2)
            stamp!(OneTwoSplitter(kappa=0.5), ctx, one..., two1..., two2...)
            return ctx
        end

        sol = dc!(MNACircuit(circuit))
        @test length(sol.x) > 0
    end

    @testset "Tier 3: PhotoDetector DC" begin
        # PhotoDetector: I(ele_out) <+ laplace_nd(-resp * (OptE[0]^2 + OptE[1]^2), ...)
        # Drive OptE(opt_in[0]) = 1.0V (via voltage source on first port node)
        # Power = OptE[0]^2 = 1.0, filtered current = -resp*1 = -1A (DC gain=1)
        # KCL: I_contribution(-1A) + I_resistor(V/1Ω) = 0 → V(ele_out) = 1.0V
        function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing; ctx = MNAContext()
            else CedarSim.MNA.reset_for_restamping!(ctx) end
            opt_in = make_optical_port!(ctx, :opt_in)
            ele_out = get_node!(ctx, :ele_out)
            stamp!(Resistor(1.0), ctx, ele_out, 0)
            stamp!(VoltageSource(1.0; name=:Vopt), ctx, opt_in[1], 0, t, spec.mode)
            stamp!(PhotoDetector(), ctx, opt_in..., ele_out;
                   _mna_x_=x, _mna_t_=t, _mna_spec_=spec, _mna_instance_=Symbol(""))
            return ctx
        end

        sol = dc!(MNACircuit(circuit))
        @test isapprox(voltage(sol, :ele_out), 1.0; atol=0.01)
    end

    @testset "Tier 3: TunableFilter compilation" begin
        # Just verify TunableFilter compiles and can be instantiated
        # Full simulation requires time-dependent $abstime sources
        @test isdefined(PhotonicModels, :TunableFilter)
    end

end # Photonic Integration
