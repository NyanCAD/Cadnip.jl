using CedarSim
using CedarSim.MNA
using CedarSim.OsdiLoader
using Test

const OSDI_DIR = @__DIR__

# Compiled from models/VADistillerModels.jl/va/resistor.va (openvaf-r)
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

    @test dev.nodes[1].name == "pos"
    @test dev.nodes[2].name == "neg"

    @test haskey(dev.param_by_name, "resistance")
    R_param = dev.param_by_name["resistance"]
    @test R_param.type == Float64

    println("  Device: $(dev.name)")
    println("  Nodes: $(dev.num_nodes) ($(dev.num_terminals) terminals)")
    println("  Params: $(length(dev.params))")
    println("  Jacobian entries: $(length(dev.jacobian_entries))")
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

    model = OsdiModel(dev)
    setup_model!(model)
    inst = OsdiInstance(model)
    set_param!(inst, "resistance", 1000.0)
    setup_instance!(inst)

    ctx = MNAContext()
    n1 = get_node!(ctx, :n1)
    n2 = get_node!(ctx, :n2)
    stamp!(inst, ctx, n1, n2)

    sys = assemble!(ctx)
    G_dense = Matrix(sys.G)
    expected_G = 1.0 / 1000.0
    @test G_dense[n1, n1] ≈ expected_G atol=1e-10
    @test G_dense[n1, n2] ≈ -expected_G atol=1e-10
    @test G_dense[n2, n1] ≈ -expected_G atol=1e-10
    @test G_dense[n2, n2] ≈ expected_G atol=1e-10
end

@testset "DC solve with OSDI resistor" begin
    f = osdi_load(RESISTOR_OSDI)
    dev = f.devices[1]

    model_r1 = OsdiModel(dev)
    setup_model!(model_r1)
    inst_r1 = OsdiInstance(model_r1)
    set_param!(inst_r1, "resistance", 1000.0)
    setup_instance!(inst_r1)

    function osdi_divider(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        vcc = get_node!(ctx, :vcc)
        mid = get_node!(ctx, :mid)

        stamp!(VoltageSource(5.0), ctx, vcc, 0)
        stamp!(inst_r1, ctx, vcc, mid; _mna_x_=x, _mna_spec_=spec, t=t)
        stamp!(Resistor(1000.0), ctx, mid, 0)

        return ctx
    end

    circuit = MNACircuit(osdi_divider)
    sol = dc!(circuit)

    @test voltage(sol, :vcc) ≈ 5.0 atol=1e-6
    @test voltage(sol, :mid) ≈ 2.5 atol=1e-6
end

end # testset "OSDI Loader"

#==============================================================================#
# Diode Tests (nonlinear, 2-terminal with internal nodes, reactive entries)
#==============================================================================#

const DIODE_OSDI = joinpath(OSDI_DIR, "diode.osdi")

@testset "OSDI Diode" begin

@testset "Diode with series resistor" begin
    f = osdi_load(DIODE_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    setup_model!(model)
    inst = OsdiInstance(model)
    setup_instance!(inst)

    function diode_r_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        vcc = get_node!(ctx, :vcc)
        anode = get_node!(ctx, :anode)

        stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
        stamp!(Resistor(1000.0), ctx, vcc, anode)
        stamp!(inst, ctx, anode, 0; _mna_x_=x, _mna_spec_=spec, t=t,
               _mna_instance_=:D1)

        return ctx
    end

    circuit = MNACircuit(diode_r_circuit)
    sol = dc!(circuit)

    V_anode = voltage(sol, :anode)
    println("  V_anode = $V_anode (expect ~0.6-0.7V)")
    @test voltage(sol, :vcc) ≈ 5.0 atol=1e-6
    @test V_anode > 0.4 && V_anode < 0.9
end

end # testset "OSDI Diode"

#==============================================================================#
# MOSFET Level 1 Tests (4-terminal, internal nodes, reactive)
#==============================================================================#

const MOS1_OSDI = joinpath(OSDI_DIR, "mos1.osdi")

@testset "OSDI MOS1" begin

@testset "NMOS DC IV" begin
    f = osdi_load(MOS1_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    set_param!(model, "type", 1)
    set_param!(model, "vto", 0.7)
    set_param!(model, "kp", 100e-6)
    setup_model!(model)

    inst = OsdiInstance(model)
    set_param!(inst, "w", 10e-6)
    set_param!(inst, "l", 1e-6)
    setup_instance!(inst)

    function nmos_iv(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)

        stamp!(VoltageSource(1.2; name=:Vds), ctx, d, 0)
        stamp!(VoltageSource(1.0; name=:Vgs), ctx, g, 0)
        stamp!(inst, ctx, d, g, 0, 0; _mna_x_=x, _mna_spec_=spec, t=t,
               _mna_instance_=:M1)

        return ctx
    end

    circuit = MNACircuit(nmos_iv)
    sol = dc!(circuit)

    @test voltage(sol, :d) ≈ 1.2 atol=1e-6
    @test voltage(sol, :g) ≈ 1.0 atol=1e-6

    Id = -current(sol, :I_Vds)
    println("  Vds=$(voltage(sol, :d)), Vgs=$(voltage(sol, :g)), Id=$Id")
    @test Id > 10e-6 && Id < 200e-6
end

end # testset "OSDI MOS1"

#==============================================================================#
# PSP103 Tests (complex 4-terminal MOSFET with 9 internal nodes)
#==============================================================================#

const PSP103_OSDI = joinpath(OSDI_DIR, "psp103.osdi")

@testset "OSDI PSP103" begin

@testset "PSP103 DC with defaults" begin
    f = osdi_load(PSP103_OSDI)
    dev = f.devices[1]

    model = OsdiModel(dev)
    set_param!(model, "TYPE", 1)
    setup_model!(model)

    inst = OsdiInstance(model)
    set_param!(inst, "L", 1e-6)
    set_param!(inst, "W", 10e-6)
    setup_instance!(inst)

    function psp103_iv(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        end
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)

        stamp!(VoltageSource(1.2; name=:Vds), ctx, d, 0)
        stamp!(VoltageSource(0.6; name=:Vgs), ctx, g, 0)
        stamp!(inst, ctx, d, g, 0, 0; _mna_x_=x, _mna_spec_=spec, t=t,
               _mna_instance_=:M1)

        return ctx
    end

    circuit = MNACircuit(psp103_iv)
    sol = dc!(circuit)

    @test voltage(sol, :d) ≈ 1.2 atol=1e-6
    @test voltage(sol, :g) ≈ 0.6 atol=1e-6

    Id = -current(sol, :I_Vds)
    println("  PSP103 DC: Vds=1.2, Vgs=0.6, Id=$Id")
    @test Id >= 0.0
end

end # testset "OSDI PSP103"

#==============================================================================#
# SPICE Netlist Integration Tests
#==============================================================================#

include(joinpath(@__DIR__, "..", "common.jl"))

@testset "OSDI via SPICE" begin

@testset "Resistor divider" begin
    spice = """* OSDI resistor SPICE test
    .model myres sp_resistor resistance=1000
    V1 vcc 0 DC 10
    N1 vcc mid myres
    R1 mid 0 1k
    """
    ctx, sol = solve_mna_spice_code(spice; osdi_files=[RESISTOR_OSDI])
    @test voltage(sol, :mid) ≈ 5.0 atol=1e-4
    println("  SPICE Resistor: Vmid = ", voltage(sol, :mid))
end

@testset "Diode with series resistor" begin
    spice = """* OSDI diode SPICE test
    .model d1 sp_diode
    V1 a 0 DC 5
    R1 a anode 1k
    N1 anode 0 d1
    """
    ctx, sol = solve_mna_spice_code(spice; osdi_files=[DIODE_OSDI])
    V_anode = voltage(sol, :anode)
    println("  SPICE Diode: V_anode = ", V_anode)
    @test V_anode > 0.4 && V_anode < 0.9
end

@testset "MOS1 NMOS DC" begin
    spice = """* OSDI mos1 SPICE test
    .model nmos1 sp_mos1 type=1 vto=0.7 kp=100e-6
    Vds d 0 DC 1.2
    Vgs g 0 DC 1.0
    N1 d g 0 0 nmos1 w=10e-6 l=1e-6
    """
    ctx, sol = solve_mna_spice_code(spice; osdi_files=[MOS1_OSDI])
    @test voltage(sol, :d) ≈ 1.2 atol=1e-6
    @test voltage(sol, :g) ≈ 1.0 atol=1e-6
    Id = -current(sol, :I_vds)
    println("  SPICE MOS1: Id = ", Id)
    @test Id > 10e-6 && Id < 200e-6
end

@testset "Resistor in subcircuit" begin
    spice = """* OSDI resistor subcircuit test
    .model myres sp_resistor resistance=1000
    .subckt my_resistor p n
    N1 p n myres
    .ends
    V1 vcc 0 DC 10
    X1 vcc mid my_resistor
    R1 mid 0 1k
    """
    ctx, sol = solve_mna_spice_code(spice; osdi_files=[RESISTOR_OSDI])
    @test voltage(sol, :mid) ≈ 5.0 atol=1e-4
    println("  SPICE Subcircuit Resistor: Vmid = ", voltage(sol, :mid))
end

@testset "MOS1 in subcircuit (IHP-style)" begin
    spice = """* OSDI mos1 subcircuit test
    .model nmos1 sp_mos1 type=1 vto=0.7 kp=100e-6
    .subckt my_nmos d g s b
    + w=10e-6 l=1e-6
    N1 d g s b nmos1 w=w l=l
    .ends
    Vds d 0 DC 1.2
    Vgs g 0 DC 1.0
    X1 d g 0 0 my_nmos w=10e-6 l=1e-6
    """
    ctx, sol = solve_mna_spice_code(spice; osdi_files=[MOS1_OSDI])
    @test voltage(sol, :d) ≈ 1.2 atol=1e-6
    @test voltage(sol, :g) ≈ 1.0 atol=1e-6
    Id = -current(sol, :I_vds)
    println("  SPICE Subcircuit MOS1: Id = ", Id)
    @test Id > 10e-6 && Id < 200e-6
end

end # testset "OSDI via SPICE"
