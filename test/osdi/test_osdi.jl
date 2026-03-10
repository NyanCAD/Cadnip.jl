using CedarSim
using CedarSim.MNA
using CedarSim.OsdiLoader
using Test

const OSDI_DIR = @__DIR__

# Compiled from models/VADistillerModels.jl/va/resistor.va (openvaf)
const RESISTOR_OSDI = joinpath(OSDI_DIR, "resistor.osdi")

@testset "OSDI Loader" begin

@testset "Load .osdi file" begin
    @test isfile(RESISTOR_OSDI)
    f = osdi_load(RESISTOR_OSDI)
    @test length(f.devices) == 1

    dev = f.devices[1]
    @test dev.name == "sp_resistor"
    @test dev.num_terminals == 2
    @test dev.instance_size > 0
    @test dev.model_size > 0

    # Check nodes
    @test dev.nodes[1].name == "pos"
    @test dev.nodes[2].name == "neg"

    # Check parameters
    @test haskey(dev.param_by_name, "resistance")
    R_param = dev.param_by_name["resistance"]
    @test R_param.type == Float64
    @test R_param.kind == :instance

    println("  Device: $(dev.name)")
    println("  Nodes: $(dev.num_nodes) ($(dev.num_terminals) terminals)")
    println("  Params: $(length(dev.params))")
    println("  Jacobian entries: $(length(dev.jacobian_entries))")
    println("  Resistive entries: $(dev.num_resistive_entries)")
    println("  Reactive entries: $(dev.num_reactive_entries)")
    println("  Instance size: $(dev.instance_size)")
    println("  Model size: $(dev.model_size)")
    println("  States: $(dev.num_states)")
end

@testset "Model and Instance setup" begin
    f = osdi_load(RESISTOR_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    setup_model!(model)
    @test model.initialized

    inst = OsdiInstance(model)
    set_param!(inst, "resistance", 1000.0)
    setup_instance!(inst)
    @test inst.initialized
end

@testset "Parameter setting" begin
    f = osdi_load(RESISTOR_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    setup_model!(model)

    inst = OsdiInstance(model)
    set_param!(inst, "resistance", 2000.0)
    setup_instance!(inst)
end

@testset "Discovery stamp (MNAContext)" begin
    f = osdi_load(RESISTOR_OSDI)
    dev = f.devices[1]

    # Create OSDI resistor with resistance=1000
    model = OsdiModel(dev)
    setup_model!(model)
    inst = OsdiInstance(model)
    set_param!(inst, "resistance", 1000.0)
    setup_instance!(inst)

    # Stamp into MNAContext
    ctx = MNAContext()
    n1 = get_node!(ctx, :n1)
    n2 = get_node!(ctx, :n2)
    stamp!(inst, ctx, n1, n2)

    # Build matrices
    sys = assemble!(ctx)

    # For a 1kΩ resistor: G = 1/1000 = 0.001
    # G matrix should be:
    #   [ +G  -G ]
    #   [ -G  +G ]
    G_dense = Matrix(sys.G)
    expected_G = 1.0 / 1000.0
    @test G_dense[n1, n1] ≈ expected_G atol=1e-10
    @test G_dense[n1, n2] ≈ -expected_G atol=1e-10
    @test G_dense[n2, n1] ≈ -expected_G atol=1e-10
    @test G_dense[n2, n2] ≈ expected_G atol=1e-10

    println("  G matrix (OSDI resistor):")
    display(G_dense)
    println()
end

@testset "Compare OSDI vs native Resistor" begin
    f = osdi_load(RESISTOR_OSDI)
    dev = f.devices[1]

    # OSDI path
    model = OsdiModel(dev)
    setup_model!(model)
    inst = OsdiInstance(model)
    set_param!(inst, "resistance", 1000.0)
    setup_instance!(inst)

    ctx_osdi = MNAContext()
    n1 = get_node!(ctx_osdi, :n1)
    n2 = get_node!(ctx_osdi, :n2)
    stamp!(inst, ctx_osdi, n1, n2)
    sys_osdi = assemble!(ctx_osdi)

    # Native path
    ctx_native = MNAContext()
    n1n = get_node!(ctx_native, :n1)
    n2n = get_node!(ctx_native, :n2)
    stamp!(Resistor(1000.0), ctx_native, n1n, n2n)
    sys_native = assemble!(ctx_native)

    # Compare G matrices
    G_osdi = Matrix(sys_osdi.G)
    G_native = Matrix(sys_native.G)
    @test G_osdi ≈ G_native atol=1e-10

    println("  OSDI G matches native Resistor G: ✓")
end

@testset "DC solve with OSDI resistor" begin
    f = osdi_load(RESISTOR_OSDI)
    dev = f.devices[1]

    # Create model and instance ONCE outside the builder
    model_r1 = OsdiModel(dev)
    setup_model!(model_r1)
    inst_r1 = OsdiInstance(model_r1)
    set_param!(inst_r1, "resistance", 1000.0)
    setup_instance!(inst_r1)

    # Build a voltage divider: Vsrc -> R1(OSDI) -> R2(native) -> GND
    function osdi_divider(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        vcc = get_node!(ctx, :vcc)
        mid = get_node!(ctx, :mid)

        # Voltage source
        stamp!(VoltageSource(5.0), ctx, vcc, 0)

        # OSDI resistor (R=1000)
        stamp!(inst_r1, ctx, vcc, mid; _mna_x_=x, _mna_spec_=spec, t=t)

        # Native resistor (R=1000)
        stamp!(Resistor(1000.0), ctx, mid, 0)

        return ctx
    end

    circuit = MNACircuit(osdi_divider)
    sol = dc!(circuit)

    # Voltage divider: Vmid = 5.0 * R2/(R1+R2) = 5.0 * 1000/2000 = 2.5V
    @test voltage(sol, :vcc) ≈ 5.0 atol=1e-6
    @test voltage(sol, :mid) ≈ 2.5 atol=1e-6

    println("  Vcc = $(voltage(sol, :vcc))")
    println("  Vmid = $(voltage(sol, :mid)) (expected 2.5)")
end

end # testset "OSDI Loader"

#==============================================================================#
# Diode Tests (nonlinear, 2-terminal with internal nodes, reactive entries)
#==============================================================================#

const DIODE_OSDI = joinpath(OSDI_DIR, "diode.osdi")

@testset "OSDI Diode" begin

@testset "Load diode" begin
    @test isfile(DIODE_OSDI)
    f = osdi_load(DIODE_OSDI)
    dev = f.devices[1]
    @test dev.name == "sp_diode"
    @test dev.num_terminals == 2
    @test dev.num_nodes == 4  # a, c, a_int, implicit_equation_0
    @test dev.num_states == 1
    @test dev.num_resistive_entries > 0
    println("  $(dev.name): $(dev.num_nodes) nodes, $(dev.num_resistive_entries) resist, $(dev.num_reactive_entries) react")
end

@testset "Diode forward bias DC" begin
    f = osdi_load(DIODE_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    setup_model!(model)
    inst = OsdiInstance(model)
    setup_instance!(inst)

    # Forward-biased diode: Vsrc=0.6V -> Diode -> GND
    function diode_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        anode = get_node!(ctx, :anode)

        stamp!(VoltageSource(0.6; name=:V1), ctx, anode, 0)
        stamp!(inst, ctx, anode, 0; _mna_x_=x, _mna_spec_=spec, t=t,
               instance_name=:D1)

        return ctx
    end

    circuit = MNACircuit(diode_circuit)
    sol = dc!(circuit)

    @test voltage(sol, :anode) ≈ 0.6 atol=1e-6

    # Diode current should be positive and in reasonable range for Is=1e-14, N=1
    I_diode = -current(sol, :I_V1)
    println("  V_anode = $(voltage(sol, :anode)), I_diode = $I_diode")
    @test I_diode > 0  # forward bias → positive current
    @test I_diode > 1e-6  # should be significant at 0.6V
end

@testset "Diode with series resistor" begin
    f = osdi_load(DIODE_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    setup_model!(model)
    inst = OsdiInstance(model)
    setup_instance!(inst)

    # Vsrc=5V -> R(1k) -> Diode -> GND
    # Diode should drop ~0.6-0.7V, resistor drops the rest
    function diode_r_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        vcc = get_node!(ctx, :vcc)
        anode = get_node!(ctx, :anode)

        stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
        stamp!(Resistor(1000.0), ctx, vcc, anode)
        stamp!(inst, ctx, anode, 0; _mna_x_=x, _mna_spec_=spec, t=t,
               instance_name=:D1)

        return ctx
    end

    circuit = MNACircuit(diode_r_circuit)
    sol = dc!(circuit)

    V_anode = voltage(sol, :anode)
    println("  V_anode = $V_anode (expect ~0.6-0.7V)")
    @test voltage(sol, :vcc) ≈ 5.0 atol=1e-6
    @test V_anode > 0.4 && V_anode < 0.9  # typical diode drop
end

end # testset "OSDI Diode"

#==============================================================================#
# MOSFET Level 1 Tests (4-terminal, internal nodes, reactive)
#==============================================================================#

const MOS1_OSDI = joinpath(OSDI_DIR, "mos1.osdi")

@testset "OSDI MOS1" begin

@testset "Load mos1" begin
    @test isfile(MOS1_OSDI)
    f = osdi_load(MOS1_OSDI)
    dev = f.devices[1]
    @test dev.name == "sp_mos1"
    @test dev.num_terminals == 4
    @test dev.num_nodes == 11  # d,g,s,b + 7 internal
    println("  $(dev.name): $(dev.num_nodes) nodes, $(dev.num_resistive_entries) resist, $(dev.num_reactive_entries) react, $(dev.num_states) states")
end

@testset "NMOS DC IV" begin
    f = osdi_load(MOS1_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    set_param!(model, "type", 1)      # NMOS
    set_param!(model, "vto", 0.7)
    set_param!(model, "kp", 100e-6)
    setup_model!(model)

    inst = OsdiInstance(model)
    set_param!(inst, "w", 10e-6)
    set_param!(inst, "l", 1e-6)
    setup_instance!(inst)

    # NMOS: Vds=1.2, Vgs=1.0, Vs=Vb=0
    function nmos_iv(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)

        stamp!(VoltageSource(1.2; name=:Vds), ctx, d, 0)
        stamp!(VoltageSource(1.0; name=:Vgs), ctx, g, 0)
        # d=1, g=2, s=0(gnd), b=0(gnd)
        stamp!(inst, ctx, d, g, 0, 0; _mna_x_=x, _mna_spec_=spec, t=t,
               instance_name=:M1)

        return ctx
    end

    circuit = MNACircuit(nmos_iv)
    sol = dc!(circuit)

    @test voltage(sol, :d) ≈ 1.2 atol=1e-6
    @test voltage(sol, :g) ≈ 1.0 atol=1e-6

    # Drain current: saturation region (Vgs-Vth=0.3, Vds=1.2 > Vgs-Vth)
    # Id = 0.5 * kp * W/L * (Vgs-Vth)^2 = 0.5 * 100e-6 * 10 * 0.09 = 45µA
    Id = -current(sol, :I_Vds)
    println("  Vds=$(voltage(sol, :d)), Vgs=$(voltage(sol, :g)), Id=$Id")
    @test Id > 10e-6 && Id < 200e-6  # reasonable range
end

end # testset "OSDI MOS1"

#==============================================================================#
# PSP103 Tests (complex 4-terminal MOSFET with 9 internal nodes)
#==============================================================================#

const PSP103_OSDI = joinpath(OSDI_DIR, "psp103.osdi")

@testset "OSDI PSP103" begin

@testset "Load psp103" begin
    @test isfile(PSP103_OSDI)
    f = osdi_load(PSP103_OSDI)
    dev = f.devices[1]
    @test dev.name == "PSP103VA"
    @test dev.num_terminals == 4
    @test dev.num_nodes == 13
    println("  $(dev.name): $(dev.num_nodes) nodes ($(dev.num_terminals) terminals + $(dev.num_nodes - dev.num_terminals) internal)")
    println("  Jacobian: $(dev.num_resistive_entries) resist, $(dev.num_reactive_entries) react")
    println("  Model params: $(length([p for p in dev.params if p.kind == :model]))")
    println("  Instance params: $(length([p for p in dev.params if p.kind == :instance]))")
end

@testset "PSP103 DC with defaults" begin
    f = osdi_load(PSP103_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    set_param!(model, "TYPE", 1)  # NMOS
    setup_model!(model)

    inst = OsdiInstance(model)
    set_param!(inst, "L", 1e-6)
    set_param!(inst, "W", 10e-6)
    setup_instance!(inst)

    # NMOS: Vds=1.2, Vgs=0.6
    function psp103_iv(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)

        stamp!(VoltageSource(1.2; name=:Vds), ctx, d, 0)
        stamp!(VoltageSource(0.6; name=:Vgs), ctx, g, 0)
        stamp!(inst, ctx, d, g, 0, 0; _mna_x_=x, _mna_spec_=spec, t=t,
               instance_name=:M1)

        return ctx
    end

    circuit = MNACircuit(psp103_iv)
    sol = dc!(circuit)

    @test voltage(sol, :d) ≈ 1.2 atol=1e-6
    @test voltage(sol, :g) ≈ 0.6 atol=1e-6

    Id = -current(sol, :I_Vds)
    println("  PSP103 DC: Vds=1.2, Vgs=0.6, Id=$Id")
    # PSP103 with default model parameters (no Vth etc.) may produce zero current.
    # Just verify convergence and that drain current is non-negative for NMOS.
    @test Id >= 0.0
end

@testset "PSP103 DC with model parameters" begin
    f = osdi_load(PSP103_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    set_param!(model, "TYPE", 1)
    set_param!(model, "TR", 27.0)
    set_param!(model, "TOXO", 1.5e-9)
    set_param!(model, "NSUBO", 3.0e23)
    set_param!(model, "THESATO", 0.5)
    set_param!(model, "MUEO", 0.5)
    setup_model!(model)

    inst = OsdiInstance(model)
    set_param!(inst, "L", 1e-6)
    set_param!(inst, "W", 10e-6)
    setup_instance!(inst)

    # Test at multiple bias points
    for (vds, vgs) in [(0.5, 0.4), (1.0, 0.6), (1.2, 1.0)]
        function psp_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            end
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)

            stamp!(VoltageSource(vds; name=:Vds), ctx, d, 0)
            stamp!(VoltageSource(vgs; name=:Vgs), ctx, g, 0)
            stamp!(inst, ctx, d, g, 0, 0; _mna_x_=x, _mna_spec_=spec, t=t,
                   instance_name=:M1)

            return ctx
        end

        circuit = MNACircuit(psp_circuit)
        sol = dc!(circuit)

        @test voltage(sol, :d) ≈ vds atol=1e-6
        @test voltage(sol, :g) ≈ vgs atol=1e-6

        Id = -current(sol, :I_Vds)
        println("  PSP103: Vds=$vds, Vgs=$vgs → Id=$Id")
        @test isfinite(Id)
        @test abs(Id) < 0.1  # sanity: less than 100mA
    end
end

end # testset "OSDI PSP103"
