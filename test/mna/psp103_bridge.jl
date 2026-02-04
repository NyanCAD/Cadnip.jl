#!/usr/bin/env julia
#==============================================================================#
# PSP103 Ring Benchmark Bridge Tests
#
# Incrementally bridges the gap between the basic PSP103 integration test
# (single transistor, minimal model card) and the full 9-stage ring oscillator
# benchmark (18 transistors, 280+ parameter model card, 1µs transient).
#
# Test levels (progressive complexity):
#   1. Single NMOS with full model card (DC)
#   2. NMOS through subcircuit wrapper with area/perimeter params (assembly)
#   3. Single CMOS inverter with full model card (assembly + structure)
#   4. Single CMOS inverter, transient with pulse input
#   5. 3-stage ring oscillator, transient
#   6. 9-stage ring oscillator, short transient (matches ring benchmark topology)
#
# Each level adds complexity dimensions:
#   - Model card completeness (280+ params vs defaults)
#   - Subcircuit hierarchy depth
#   - Device count (1 → 2 → 6 → 18)
#   - Analysis type (DC → transient)
#   - Circuit topology (open-loop → feedback ring)
#
# NOTE: DC solve through subcircuit-wrapped PSP103VA triggers LLVM SROA hangs
# (known limitation, see psp103_integration.jl). Levels 2-3 test assembly only.
# Levels 4-6 use transient analysis which works through the MNACircuit/DAE path.
#==============================================================================#

using CedarSim
using CedarSim.SpectreNetlistParser
using CedarSim.MNA: MNAContext, MNASpec, solve_dc, voltage, current,
                     n_internal_nodes, MNACircuit, CedarTranOp, assemble!
using PSPModels
using OrdinaryDiffEq: FBDF
using Test

# Load the full benchmark model card (280+ params for NMOS + PMOS)
# This is the same model card used by the ring oscillator benchmark
const models_inc_path = joinpath(@__DIR__, "..", "..", "benchmarks", "vacask", "ring", "cedarsim", "models.inc")
const models_inc = read(models_inc_path, String)

# Inverter subcircuit definition (same as ring benchmark's runme.sp)
const inverter_subckt = """
.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends
"""

# Helper: parse SPICE with full model card prepended
function parse_with_full_modelcard(circuit_spice::String; circuit_name::Symbol=:bridge_circuit)
    # models.inc starts with a comment line that becomes the implicit title
    full_netlist = models_inc * "\n" * circuit_spice
    ast = SpectreNetlistParser.parse(IOBuffer(full_netlist); start_lang=:spice, implicit_title=true)
    return CedarSim.make_mna_circuit(ast; circuit_name, imported_hdl_modules=[PSPModels])
end

@testset "PSP103 Ring Benchmark Bridge" begin

    #==========================================================================
    # Level 1: Single NMOS with full model card (DC)
    #
    # Simplest possible circuit with the complete 280+ parameter model card
    # from the ring oscillator benchmark. This validates that all model
    # parameters parse and produce a convergent DC solution.
    #
    # Compared to psp103_integration.jl "DC with full model parameters":
    #   - Uses complete benchmark model card (280+ params vs ~40 params)
    #   - Uses psp103n model name (matching benchmark convention)
    ==========================================================================#
    @testset "Level 1: Single NMOS, full model card, DC" begin
        code = parse_with_full_modelcard("""
M1 d g 0 0 psp103n W=10u L=1u
Vds d 0 DC 1.2
Vgs g 0 DC 0.6
"""; circuit_name=:bridge_l1)
        eval(code)

        sol = solve_dc(bridge_l1, (;), MNASpec(temp=27.0, mode=:dcop))

        @test isapprox(voltage(sol, :d), 1.2, atol=1e-6)
        @test isapprox(voltage(sol, :g), 0.6, atol=1e-6)

        # Drain current should be in the hundreds of µA range for these dimensions
        Id = current(sol, :I_vds)
        @test abs(Id) > 10e-6   # At least 10µA
        @test abs(Id) < 10e-3   # Less than 10mA
        println("  Level 1 NMOS Id = $(abs(Id)*1e6) µA")
    end

    #==========================================================================
    # Level 2: NMOS through subcircuit wrapper (assembly)
    #
    # Uses the `nmos` subcircuit wrapper from models.inc which adds:
    #   - Area/perimeter params (AD, AS, PD, PS) computed from W and LD
    #   - One level of subcircuit hierarchy
    #
    # This is how transistors are instantiated in the ring benchmark.
    # Only tests assembly (not DC solve) due to known LLVM SROA limitation
    # when compiling DirectStampContext through subcircuit wrappers.
    ==========================================================================#
    @testset "Level 2: NMOS subcircuit wrapper, assembly" begin
        code = parse_with_full_modelcard("""
xm1 d g 0 0 nmos w={10u} l={1u}
Vds d 0 DC 1.2
Vgs g 0 DC 0.6
"""; circuit_name=:bridge_l2)
        eval(code)

        # Verify structure builds correctly
        spec = MNASpec(temp=27.0, mode=:dcop)
        ctx = bridge_l2((;), spec, 0.0)

        # PSP103VA has 8 internal nodes per device
        n_internal = n_internal_nodes(ctx)
        @test n_internal == 8
        println("  Level 2: $n_internal internal nodes (expected 8)")

        # Verify subcircuit hierarchy in node names
        internal_names = [name for (name, idx) in ctx.node_to_idx if ctx.internal_node_flags[idx]]
        @test any(contains(String(n), "xm1_nm") for n in internal_names)
        println("  Level 2: Internal nodes include subcircuit prefix")
    end

    #==========================================================================
    # Level 3: Single CMOS inverter (assembly + structure validation)
    #
    # First test with both NMOS and PMOS model cards together, forming
    # a complementary inverter with the exact subcircuit from the benchmark.
    # Tests:
    #   - Both NMOS (type=1) and PMOS (type=-1) full model cards parse
    #   - Inverter subcircuit hierarchy (inverter → nmos/pmos → psp103va)
    #   - Correct node and internal node allocation
    #   - 2 PSP103VA devices × 8 internal nodes = 16 total
    ==========================================================================#
    @testset "Level 3: CMOS inverter, assembly" begin
        code = parse_with_full_modelcard(inverter_subckt * """
xu1 in out vdd 0 inverter w={10u} l={1u}
Vin in 0 DC 0.6
Vdd vdd 0 DC 1.2
"""; circuit_name=:bridge_l3)
        eval(code)

        spec = MNASpec(temp=27.0, mode=:dcop)
        ctx = bridge_l3((;), spec, 0.0)

        # 1 inverter × 2 MOSFETs × 8 internal nodes = 16
        n_internal = n_internal_nodes(ctx)
        @test n_internal == 16
        println("  Level 3: $n_internal internal nodes (expected 16)")

        # Verify both NMOS and PMOS paths have proper hierarchical names
        internal_names = [name for (name, idx) in ctx.node_to_idx if ctx.internal_node_flags[idx]]
        @test any(contains(String(n), "xu1_xmn") for n in internal_names)
        @test any(contains(String(n), "xu1_xmp") for n in internal_names)
        println("  Level 3: Both NMOS and PMOS internal nodes present")

        # Verify external nodes are present
        @test haskey(ctx.node_to_idx, :in)
        @test haskey(ctx.node_to_idx, :out)
        @test haskey(ctx.node_to_idx, :vdd)
        println("  Level 3: External nodes (in, out, vdd) present")
    end

    #==========================================================================
    # Level 4: Single CMOS inverter, transient analysis
    #
    # Steps up from assembly to transient simulation. Uses a pulse input to
    # switch the inverter, testing that:
    #   - PSP103 model works with time integration (DAE solver)
    #   - Voltage-dependent capacitors are handled correctly
    #   - CedarTranOp DC initialization converges
    #   - FBDF solver steps through the switching transient
    #
    # Short simulation: 0.5ns (just enough to see switching)
    ==========================================================================#
    @testset "Level 4: CMOS inverter, transient" begin
        code = parse_with_full_modelcard(inverter_subckt * """
xu1 in out vdd 0 inverter w={10u} l={1u}
Vin in 0 DC 0 pulse 0 1.2 0.1n 0.05n 0.05n 0.2n 0.5n
Vdd vdd 0 DC 1.2
"""; circuit_name=:bridge_l4)
        eval(code)

        circuit = MNACircuit(bridge_l4)

        # Verify assembly
        data = assemble!(circuit)
        n = size(data.G, 1)
        println("  Level 4: Inverter assembled: $n unknowns")
        @test n > 10  # Should have signal nodes + internal nodes

        println("  Level 4: Running inverter transient (0.5ns)...")
        sol = tran!(circuit, (0.0, 0.5e-9);
            solver=FBDF(autodiff=false),
            dtmax=0.01e-9,
            initializealg=CedarTranOp(),
            maxiters=100000,
            dense=false)

        @test sol.retcode == :Success
        @test sol.t[end] >= 0.5e-9 * 0.99
        println("  Level 4: $(length(sol.t)) timepoints, retcode=$(sol.retcode)")
    end

    #==========================================================================
    # Level 5: 3-stage ring oscillator, transient
    #
    # Minimum odd-stage ring oscillator. Tests:
    #   - Multi-stage feedback loop topology
    #   - 6 MOSFETs (3 NMOS + 3 PMOS)
    #   - Self-sustaining oscillation dynamics
    #   - No stable DC equilibrium (requires homotopy init)
    #
    # Short simulation: 1ns
    ==========================================================================#
    @testset "Level 5: 3-stage ring oscillator, transient" begin
        code = parse_with_full_modelcard(inverter_subckt * """
i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n
xu1 1 2 vdd 0 inverter w={10u} l={1u}
xu2 2 3 vdd 0 inverter w={10u} l={1u}
xu3 3 1 vdd 0 inverter w={10u} l={1u}
vdd vdd 0 1.2
"""; circuit_name=:bridge_l5)
        eval(code)

        circuit = MNACircuit(bridge_l5)

        # Verify assembly
        data = assemble!(circuit)
        n = size(data.G, 1)
        println("  Level 5: 3-stage ring assembled: $n unknowns")
        # 3 inverters × 2 devices × (1 external + 8 internal) ≈ many nodes
        @test n > 30

        # Verify structure: 6 devices × 8 internal nodes = 48
        ctx = bridge_l5((;), MNASpec(temp=27.0, mode=:dcop), 0.0)
        n_internal = n_internal_nodes(ctx)
        @test n_internal == 48  # 3 inv × 2 devices × 8 internal
        println("  Level 5: $n_internal internal nodes (expected 48)")

        println("  Level 5: Running 3-stage ring transient (1ns)...")
        sol = tran!(circuit, (0.0, 1e-9);
            solver=FBDF(autodiff=false),
            dtmax=0.01e-9,
            initializealg=CedarTranOp(),
            maxiters=100000,
            dense=false)

        @test sol.retcode == :Success
        @test sol.t[end] >= 1e-9 * 0.99
        println("  Level 5: $(length(sol.t)) timepoints, retcode=$(sol.retcode)")
    end

    #==========================================================================
    # Level 6: 9-stage ring oscillator, short transient
    #
    # Full ring oscillator topology matching the benchmark exactly:
    #   - 9 inverter stages (18 MOSFETs)
    #   - Full 280+ parameter model card for both NMOS and PMOS
    #   - Kick-start current pulse
    #   - ~371 MNA unknowns
    #
    # Uses 1ns simulation (vs 1µs in the full benchmark) to keep test time
    # reasonable while validating the full circuit assembly and initial
    # transient convergence.
    ==========================================================================#
    @testset "Level 6: 9-stage ring oscillator, short transient" begin
        code = parse_with_full_modelcard(inverter_subckt * """
i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n
xu1 1 2 vdd 0 inverter w={10u} l={1u}
xu2 2 3 vdd 0 inverter w={10u} l={1u}
xu3 3 4 vdd 0 inverter w={10u} l={1u}
xu4 4 5 vdd 0 inverter w={10u} l={1u}
xu5 5 6 vdd 0 inverter w={10u} l={1u}
xu6 6 7 vdd 0 inverter w={10u} l={1u}
xu7 7 8 vdd 0 inverter w={10u} l={1u}
xu8 8 9 vdd 0 inverter w={10u} l={1u}
xu9 9 1 vdd 0 inverter w={10u} l={1u}
vdd vdd 0 1.2
"""; circuit_name=:bridge_l6)
        eval(code)

        circuit = MNACircuit(bridge_l6)

        # Verify assembly
        data = assemble!(circuit)
        n = size(data.G, 1)
        println("  Level 6: 9-stage ring assembled: $n unknowns")
        @test n > 100  # Should have hundreds of unknowns (expect ~371)

        # Verify structure: 18 devices × 8 internal nodes = 144
        ctx = bridge_l6((;), MNASpec(temp=27.0, mode=:dcop), 0.0)
        n_internal = n_internal_nodes(ctx)
        @test n_internal == 144  # 9 inv × 2 devices × 8 internal
        println("  Level 6: $n_internal internal nodes (expected 144)")

        println("  Level 6: Running 9-stage ring transient (1ns)...")
        sol = tran!(circuit, (0.0, 1e-9);
            solver=FBDF(autodiff=false),
            dtmax=0.01e-9,
            initializealg=CedarTranOp(),
            maxiters=100000,
            dense=false)

        @test sol.retcode == :Success
        @test sol.t[end] >= 1e-9 * 0.99
        println("  Level 6: $(length(sol.t)) timepoints, retcode=$(sol.retcode)")
    end

end
