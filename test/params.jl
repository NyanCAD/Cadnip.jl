module params_tests

include("common.jl")

# MNA imports for parameter tests
using Cadnip.MNA: MNAContext, MNACircuit, MNASpec, get_node!, stamp!, assemble!, solve_dc
using Cadnip.MNA: Resistor, Capacitor, VoltageSource

using Cadnip.MNA: alter, reset_for_restamping!  # MNA-specific alter for MNACircuit
using Cadnip: ParamLens, IdentityLens
using Cadnip: dc!, CircuitSweep, Sweep

# Loading a Makie backend activates CadnipMakieExt, which defines Cadnip.explore.
import CairoMakie

#==============================================================================#
# Test 1: Simple parameterized circuit (replaces ParCir struct)
#
# Original: struct ParCir with R and V fields, callable to build circuit
# New: MNA builder function with params NamedTuple
#==============================================================================#

# MNA builder function equivalent to ParCir struct
# Default values: R=2.0, V=5.0
function build_par_cir(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    # Merge with defaults (like @kwdef did for the struct)
    defaults = (R=2.0, V=5.0)
    p = merge(defaults, params)

    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end
    vcc = get_node!(ctx, :vcc)
    gnd = get_node!(ctx, :gnd)  # Use explicit gnd node for clarity

    stamp!(VoltageSource(p.V; name=:V), ctx, vcc, 0)
    stamp!(Resistor(p.R), ctx, vcc, 0)

    return ctx
end

# Test with R=1.0 (equivalent to ParamSim(ParCir, R=1.0, temp=340.0))
circuit = MNACircuit(build_par_cir; spec=MNASpec(temp=340.0), R=1.0)
sol = dc!(circuit)
# Current through voltage source: I = -V/R = -5.0/1.0 = -5.0
@test sol[:I_V] == -5.0

#==============================================================================#
# Test 2: Nested subcircuit with ParamLens (replaces NestedParCir)
#
# Original: NestedParCir with child::ParCir, using SubCircuit
# New: MNA builder using ParamLens for hierarchical parameter access
#==============================================================================#

# MNA builder using ParamLens for hierarchical access
# Structure: (child=(params=(R=..., V=...),),)
function build_nested_par_cir(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    lens = ParamLens(params)
    # lens.child(; defaults...) merges defaults with overrides
    p = lens.child(; R=2.0, V=5.0)

    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end
    vcc = get_node!(ctx, :vcc)

    stamp!(VoltageSource(p.V; name=:V), ctx, vcc, 0)
    stamp!(Resistor(p.R), ctx, vcc, 0)

    return ctx
end

# Test with var"child.R"=1.0 (equivalent to ParamSim(NestedParCir, var"child.R"=1.0))
# ParamLens expects (child=(params=(R=...,),)) structure
circuit = MNACircuit(build_nested_par_cir;
             spec=MNASpec(temp=340.0),
             child=(params=(R=1.0,),))
sol = dc!(circuit)
# Current through voltage source: I = -V/R = -5.0/1.0 = -5.0
@test sol[:I_V] == -5.0

#==============================================================================#
# Test 3: Function-based circuit with lens (replaces FuncCir)
#
# Original: function FuncCir(lens) using lens(V=...).V pattern
# New: MNA builder using ParamLens with same pattern
#==============================================================================#

# MNA builder using ParamLens for parameter defaults with overrides
function build_func_cir(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    lens = ParamLens(params)
    # Call lens with defaults - returns merged params
    p = lens(; V=5.0, R=2.0)

    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end
    vcc = get_node!(ctx, :vcc)

    stamp!(VoltageSource(p.V::Float64; name=:V), ctx, vcc, 0)
    stamp!(Resistor(p.R::Float64), ctx, vcc, 0)

    return ctx
end

# Test with R=1.0 (equivalent to ParamSim(FuncCir, var"R"=1.0))
# ParamLens expects (params=(R=...,),) for top-level lens() calls
circuit = MNACircuit(build_func_cir;
             spec=MNASpec(temp=340.0),
             params=(R=1.0,))
sol = dc!(circuit)
# Current through voltage source: I = -V/R = -5.0/1.0 = -5.0
@test sol[:I_V] == -5.0

#==============================================================================#
# Test 4: Makie exploration (CadnipMakieExt)
#
# `Cadnip.explore(circuit, tspan)` builds an interactive figure with one slider
# per scalar parameter and a live node-voltage trace per node. Loading a Makie
# backend (CairoMakie here) activates the extension. We render to a PNG to
# force the observable/solve/plot pipeline to run end-to-end headlessly.
#==============================================================================#

# RC low-pass with sweepable Vcc/R/C, driven to a step response over the tspan.
function build_rc_explore(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    p = merge((Vcc=5.0, R=1e3, C=1e-6), params)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)
    stamp!(VoltageSource(p.Vcc; name=:V), ctx, vcc, 0)
    stamp!(Resistor(p.R), ctx, vcc, out)
    stamp!(Capacitor(p.C), ctx, out, 0)
    return ctx
end

@testset "CadnipMakieExt explore()" begin
    circuit = MNACircuit(build_rc_explore; Vcc=5.0, R=1e3, C=1e-6)
    fig = Cadnip.explore(circuit, (0.0, 5e-3))
    @test fig isa CairoMakie.Makie.Figure

    # Rendering exercises the full lift → alter → tran! → plot pipeline.
    plot_path = joinpath(mktempdir(), "explore.png")
    CairoMakie.save(plot_path, fig)
    @test isfile(plot_path) && filesize(plot_path) > 0

    # No numeric parameters → nothing to sweep → informative error.
    empty_circuit = MNACircuit(build_rc_explore)
    @test_throws ErrorException Cadnip.explore(empty_circuit, (0.0, 5e-3))
end

#==============================================================================#
# Test 5: SPICE netlist with parameters - using MNA codegen
#
# Original: make_spectre_circuit, ParamObserver, alter()
# New: make_mna_circuit with nested subcircuit parameter passing
#==============================================================================#

spice_ckt = """
* Subcircuit parameters
.subckt inner a b foo=foo+2000
R1 a b r= 'foo'
.ends

.subckt outer a b
x1 a b inner
.ends

.param inner  =1
.param foo =  1
i1 vcc 0 'foo'
l1 vcc out 1m
x1 out 0 outer
"""

ast = NyanSpectreNetlistParser.SPICENetlistParser.parse(spice_ckt)

# Test MNA circuit building with parameters
# Generate MNA builder from SPICE
mna_code = Cadnip.make_mna_circuit(ast)
m = Module()
Base.eval(m, :(using Cadnip.MNA))
Base.eval(m, :(using Cadnip: ParamLens, AbstractParamLens, ParamObserver, @param))
Base.eval(m, :(using Cadnip.SpectreEnvironment))
mna_builder = Base.eval(m, mna_code)

# Test ParamObserver with MNA circuits
# ParamObserver collects parameter hierarchy when circuit is called
observer = Cadnip.ParamObserver(foo=200)
spec = MNASpec(temp=27.0, mode=:dcop)
ctx = Base.invokelatest(mna_builder, observer, spec)

# Convert observer to NamedTuple and verify parameter values
observed = convert(NamedTuple, observer)

# Helper function for subset comparison
⊑(x::Number, y::Number) = x == y
⊑(x::NamedTuple, y::NamedTuple) = all(haskey(x, k) && (x[k] ⊑ y[k]) for k in keys(y))

# With foo=200, i1's DC value should be 200
# inner subcircuit has foo=foo+2000, so at x1.x1 level, foo should be 2200
# R1 uses 'foo' which is 2200 at that scope
# Note: ParamObserver puts parameters at top level, not under 'params'
expected = (
    foo = 200,
    x1 = (; x1 = (; foo = 2200.0,))
)
@test observed ⊑ expected

# Test that we can pass parameters to subcircuits via MNACircuit
# ParamLens expects params wrapped in (params=(...),) for merging to work
# Test with x1.x1.params.foo=2.0 - should set inner resistor R1 = foo = 2.0
circuit = MNACircuit(mna_builder; x1=(x1=(params=(foo=2.0,),),))
sol = dc!(circuit)
# With foo=2.0 override at x1.x1 level, R1 = foo = 2.0 Ohms
# Current source i1 uses top-level foo=1 by default
# V = I * R = 1A * 2Ω = 2V across R1
@test isapprox_deftol(sol[:out], -2)

# Test with top-level foo=2.0 - should affect both i1 and (via expression) R1
# params=(foo=2.0,) at top level merges with defaults
circuit = MNACircuit(mna_builder; params=(foo=2.0,))
sol = dc!(circuit)
# foo=2.0 at top level: i1 DC = 2A, R1 = foo+2000 = 2002Ω at inner level
# With default foo expression in subcircuit: foo = parent_foo + 2000 = 2002
# V = I * R = 2A * 2002Ω = 4004V
@test isapprox_deftol(sol[:out], -4004.0)

# Note: Direct component-level parameter overrides (like r1=(params=(r=100.0,),))
# are not yet supported by MNA codegen - the resistor stamp uses the foo parameter
# directly without checking lens for component-level overrides.
# For now, test that we can override foo at the subcircuit level with explicit value
circuit = MNACircuit(mna_builder; x1=(x1=(params=(foo=100.0,),),))
sol = dc!(circuit)
# Override foo=100.0 at x1.x1 level, R1 = foo = 100Ω, i1 uses foo=1, so I = 1A
# V = I * R = 1A * 100Ω = 100V
@test isapprox_deftol(sol[:out], -100)

# Test that our 'default parameterization' helper sees `foo` and `inner`
default_params = Cadnip.get_default_parameterization(ast)
@test (:inner => 1.0) ∈ default_params
@test (:foo => 1.0) ∈ default_params

#==============================================================================#
# Test 6: alter() on SPICE AST (still uses old API for AST manipulation)
#
# These tests modify the SPICE AST text directly, not the circuit.
# The alter() function for ASTs is separate from MNA parameter handling.
#==============================================================================#

io = IOBuffer()
Cadnip.alter(io, ast, foo=2.0, inner=(foo=3.0, r1=(r=4.0,)))
modified = String(take!(io))
replaced = replace(spice_ckt,
    "foo =  1" => "foo =  2.0",
    "foo=foo+2000" => "foo=3.0",
    "r= 'foo'" => "r= 4.0")
@test modified == replaced
new_ast = NyanSpectreNetlistParser.SPICENetlistParser.parse(modified)
default_params = Cadnip.get_default_parameterization(new_ast)
@test (:foo => 2) ∈ default_params

# Test that AST alter works with ParamLens
Cadnip.alter(io, ast, Cadnip.ParamLens((foo=2.0, inner=(foo=3.0, r1=(r=4.0,)))))
modified = String(take!(io))
@test modified == replaced

#==============================================================================#
# Test 7: ParamLens hierarchical access patterns
#
# Additional tests for ParamLens behavior with MNA circuits
#==============================================================================#

@testset "ParamLens with MNA circuits" begin
    # Test IdentityLens returns defaults unchanged
    ident = IdentityLens()
    defaults = ident(; R=1000.0, V=5.0)
    @test defaults.R == 1000.0
    @test defaults.V == 5.0

    # Test ParamLens with partial overrides
    partial_lens = ParamLens((params=(R=2000.0,),))
    merged = partial_lens(; R=1000.0, V=5.0)
    @test merged.R == 2000.0  # Overridden
    @test merged.V == 5.0     # Uses default (unmodified)

    # Test hierarchical lens traversal
    hier_lens = ParamLens((child=(params=(R=500.0,),),))
    child_lens = getproperty(hier_lens, :child)
    child_params = child_lens(; R=1000.0, V=5.0)
    @test child_params.R == 500.0  # Override from child
    @test child_params.V == 5.0    # Default (unmodified)

    # Accessing undefined subcircuit returns IdentityLens
    other_lens = getproperty(hier_lens, :other)
    @test other_lens isa IdentityLens
    other_params = other_lens(; R=1000.0)
    @test other_params.R == 1000.0  # All defaults
end

@testset "MNACircuit alter() for parameter sweeps" begin
    # Test alter() on MNACircuit objects
    circuit = MNACircuit(build_par_cir; R=1000.0, V=5.0)
    sol = dc!(circuit)
    @test sol[:vcc] ≈ 5.0

    # Alter R parameter
    circuit2 = alter(circuit; R=500.0)
    @test circuit2.params.R == 500.0
    @test circuit2.params.V == 5.0  # Unchanged

    # Both should give correct DC solution
    sol2 = dc!(circuit2)
    @test sol2[:vcc] ≈ 5.0

    # Current should reflect new R: I = -V/R = -5/500 = -0.01
    @test sol2[:I_V] ≈ -0.01
end

#==============================================================================#
# Test 8: Netlist `.param` overrides through the high-level API
#
# A `.param` is the designer's knob: bias point, device size, source amplitude.
# Overriding one has to work in the spelling a user reaches for first —
# `MNACircuit(ckt; vin=4.0)` and `Sweep(vin=...)` — not only in the qualified
# `params=(vin=4.0,)` form. Getting this wrong is silent: every sweep point
# returns the netlist default and the sweep looks like a perfectly flat curve.
#==============================================================================#

# Netlists live at module top level (world age; see CLAUDE.md "File-First Loading").
const divider_ckt = sp"""
.param vin=1.0
.param rtop=1k
V1 in 0 DC vin
R1 in out {rtop}
R2 out 0 1k
"""i

const hier_ckt = sp"""
.subckt divider a out b r1val=1k r2val=1k
R1 a out {r1val}
R2 out b {r2val}
.ends
.param vin=3.0
V1 in 0 DC vin
X1 in vout 0 divider r1val=2k r2val=1k
"""i

# `.param x1` next to an `X1` instance: the one case where a name means two
# different things in the same scope. (Subcircuit builders are named after the
# `.subckt`, so this one cannot reuse `divider` above — two netlists in one
# module would generate the same builder name.)
const collide_ckt = sp"""
.subckt pair a out b rv=1k
R1 a out {rv}
R2 out b 1k
.ends
.param x1=2000
V1 in 0 DC 4
Rs in mid {x1}
X1 mid vout 0 pair rv=1k
"""i

@testset "netlist .param overrides" begin
    # Baseline: the netlist's own values.
    @test dc!(MNACircuit(divider_ckt))[:out] ≈ 0.5

    # Flat spelling — the one the README and CLAUDE.md document.
    @test dc!(MNACircuit(divider_ckt; vin=4.0))[:out] ≈ 2.0
    # Qualified spelling — the ParamLens `params=` form.
    @test dc!(MNACircuit(divider_ckt; params=(vin=4.0,)))[:out] ≈ 2.0
    # Both at once: the explicit `params=` wins.
    @test dc!(MNACircuit(divider_ckt; vin=6.0, params=(vin=4.0,)))[:out] ≈ 2.0

    # A `.param` feeding a device value, not a source value.
    @test dc!(MNACircuit(divider_ckt; rtop=3e3))[:out] ≈ 0.25

    # Overriding both knobs at once.
    @test dc!(MNACircuit(divider_ckt; vin=4.0, rtop=3e3))[:out] ≈ 1.0

    # alter() re-parameterizes an existing circuit.
    @test dc!(alter(MNACircuit(divider_ckt; vin=4.0); vin=2.0))[:out] ≈ 1.0

    # A name no scope declares is inert (it is not a parameter of this circuit).
    @test dc!(MNACircuit(divider_ckt; nosuch=1.0))[:out] ≈ 0.5
end

@testset "netlist .param sweeps" begin
    # The sweep axis must actually move the operating point — a swept `.param`
    # that silently resolves to its default gives a flat curve, and every
    # "gain is constant across the sweep" assertion passes for the wrong reason.
    cs = CircuitSweep(divider_ckt, Sweep(vin = [1.0, 2.0, 4.0]); vin=1.0)
    @test [sol[:out] for (_, sol) in dc!(cs)] ≈ [0.5, 1.0, 2.0]

    # The qualified selector addresses the same parameter.
    cs = CircuitSweep(divider_ckt, Sweep(var"params.vin" = [1.0, 2.0]); params=(vin=1.0,))
    @test [sol[:out] for (_, sol) in dc!(cs)] ≈ [0.5, 1.0]

    # A swept axis needs no base value of its own — `alter` introduces the knob.
    cs = CircuitSweep(divider_ckt, Sweep(vin = [1.0, 4.0]))
    @test [sol[:out] for (_, sol) in dc!(cs)] ≈ [0.5, 2.0]

    # Points and solutions stay aligned.
    cs = CircuitSweep(divider_ckt, Sweep(rtop = [1e3, 3e3]); rtop=1e3)
    for (p, sol) in dc!(cs)
        @test sol[:out] ≈ 1.0 * 1e3 / (p.rtop + 1e3)
    end
end

@testset "subcircuit instance parameter overrides" begin
    # X1 spells out r1val=2k: 3V across 2k+1k → 1V at the tap.
    @test dc!(MNACircuit(hier_ckt))[:vout] ≈ 1.0

    # An override has to outrank the instance line — otherwise a parameter the
    # netlist happens to set is unreachable from alter() and from sweeps.
    @test dc!(MNACircuit(hier_ckt; x1=(r1val=1e3,)))[:vout] ≈ 1.5
    @test dc!(MNACircuit(hier_ckt; x1=(params=(r1val=1e3,),)))[:vout] ≈ 1.5

    # Parent-scope and instance-scope parameters override independently.
    @test dc!(MNACircuit(hier_ckt; vin=6.0, x1=(r1val=1e3,)))[:vout] ≈ 3.0

    # ... and a subcircuit instance parameter is a usable sweep axis.
    cs = CircuitSweep(hier_ckt, Sweep(var"x1.r1val" = [1e3, 2e3]); x1=(r1val=1e3,))
    @test [sol[:vout] for (_, sol) in dc!(cs)] ≈ [1.5, 1.0]

    # A `.subckt` default the instance line does not set stays overridable.
    @test dc!(MNACircuit(hier_ckt; x1=(r2val=3e3,)))[:vout] ≈ 1.8
end

@testset "parameter/instance name collision" begin
    # `.param x1=2000` (the series resistor Rs) and instance `X1` (a 1k/1k
    # divider) share the name. The *shape* of the override decides which one it
    # addresses: a leaf is the parameter, a group is the instance.
    tap(rs, rv) = 4 * 1e3 / (rs + rv + 1e3)

    @test dc!(MNACircuit(collide_ckt))[:vout] ≈ tap(2e3, 1e3)

    # Leaf → the parameter: Rs goes 2k → 6k.
    param_hit = dc!(MNACircuit(collide_ckt; x1=6000.0))[:vout]
    @test param_hit ≈ tap(6e3, 1e3)

    # Group → the instance: its rv goes 1k → 3k, Rs stays 2k.
    @test dc!(MNACircuit(collide_ckt; x1=(rv=3e3,)))[:vout] ≈ tap(2e3, 3e3)

    # `params=` names the parameter explicitly — same as the leaf spelling, and
    # the form to reach for when both meanings are needed at once.
    @test dc!(MNACircuit(collide_ckt; params=(x1=6000.0,)))[:vout] ≈ param_hit
    @test dc!(MNACircuit(collide_ckt; params=(x1=6000.0,), x1=(rv=3e3,)))[:vout] ≈
          tap(6e3, 3e3)
end

@testset "canonical / compact parameter trees" begin
    canon = Cadnip.canonicalize_params
    compact = Cadnip.compact_params

    # Compact (what a user writes) → canonical (what the lens reads).
    @test canon((; params=(;boo=4), foo=2, bar=(; baz=3))) ==
          (params = (boo = 4, foo = 2), bar = (params = (baz = 3,),))

    # Canonicalization is idempotent, so the lens accepts either shape.
    c = canon((; params=(;boo=4), foo=2, bar=(; baz=3)))
    @test canon(c) == c

    # An explicit `params=` outranks the flat spelling, written either order.
    @test canon((; vin=1.0, params=(;vin=2.0))).params.vin == 2.0
    @test canon((; params=(;vin=2.0), vin=1.0)).params.vin == 2.0

    # A leaf that isn't a Number is still a parameter, not something to drop.
    @test canon((; mode=:fast)).params.mode == :fast

    # `compact_params` is the inverse — this is the shape `ParamObserver`
    # reports, so an observed tree can be handed straight back as an override.
    @test compact(canon((; vin=1.0, x1=(rv=2.0,)))) == (vin=1.0, x1=(rv=2.0,))
    # ...and it keeps the qualified form exactly where it is needed.
    @test compact(canon((; params=(x1=2.0,), x1=(rv=3.0,)))) ==
          (params=(x1=2.0,), x1=(rv=3.0,))

    # Lens-level: leaf is a parameter, group is a child, and a leaf is never
    # descended into.
    @test ParamLens((vin=2.0,))(; vin=1.0) == (vin=2.0,)
    @test ParamLens((params=(vin=2.0,),))(; vin=1.0) == (vin=2.0,)
    @test ParamLens((x1=(rv=2.0,),))(; vin=1.0) == (vin=1.0,)
    @test getproperty(ParamLens((x1=(rv=2.0,),)), :x1)(; rv=1.0) == (rv=2.0,)
    @test getproperty(ParamLens((vin=2.0,)), :x1) isa IdentityLens
end

end # module params_tests
