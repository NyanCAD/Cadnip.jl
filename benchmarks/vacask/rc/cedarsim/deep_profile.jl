#!/usr/bin/env julia
#==============================================================================#
# Deep Memory Allocation Profiling for VACASK RC Circuit
#
# This script profiles individual operations in the fast_rebuild! path
# to identify the exact source of remaining allocations.
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!
using CedarSim.MNA: reset_value_only!, ValueOnlyContext, get_node!, stamp!
using CedarSim.MNA: PWLVoltageSource, Resistor, Capacitor, stamp_G!, stamp_b!
using StaticArrays
using BenchmarkTools

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

# Create circuit and compile structure
spec = MNASpec()
cs = compile_structure(rc_circuit, NamedTuple(), spec)
ws = create_workspace(cs)
u = zeros(Float64, cs.n)
t = 0.5

println("=" ^ 80)
println("Deep Allocation Profiling")
println("=" ^ 80)

# Warmup
for _ in 1:100
    fast_rebuild!(ws, u, t)
end

# Overall fast_rebuild!
alloc_total = @allocated fast_rebuild!(ws, u, t)
println("\n1. Total fast_rebuild!: $alloc_total bytes")

# Test individual components of fast_rebuild!
vctx = ws.vctx

# reset_value_only!
for _ in 1:10
    reset_value_only!(vctx)
end
alloc_reset = @allocated reset_value_only!(vctx)
println("2. reset_value_only!: $alloc_reset bytes")

# Builder call with vctx
reset_value_only!(vctx)
for _ in 1:10
    reset_value_only!(vctx)
    cs.builder(cs.params, cs.spec, t; x=u, ctx=vctx)
end
reset_value_only!(vctx)
alloc_builder = @allocated cs.builder(cs.params, cs.spec, t; x=u, ctx=vctx)
println("3. Builder call: $alloc_builder bytes")

# Test individual device stamps
println("\nDevice-level breakdown:")

# get_node! calls
p1 = get_node!(vctx, Symbol("1"))
n1 = get_node!(vctx, Symbol("0"))
for _ in 1:10
    get_node!(vctx, Symbol("1"))
end
alloc_get_node = @allocated get_node!(vctx, Symbol("1"))
println("   get_node!: $alloc_get_node bytes")

# PWLVoltageSource with SVector
times_sv = SVector{6,Float64}(0.0, 1.0e-6, 2.0e-6, 0.001002, 0.001003, 0.002)
values_sv = SVector{6,Float64}(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
pwl = PWLVoltageSource(times_sv, values_sv; name=:vs)
for _ in 1:10
    reset_value_only!(vctx)
    stamp!(pwl, vctx, 1, 0; t=t, _sim_mode_=:tran)
end
reset_value_only!(vctx)
alloc_pwl = @allocated stamp!(pwl, vctx, 1, 0; t=t, _sim_mode_=:tran)
println("   stamp!(PWLVoltageSource): $alloc_pwl bytes")

# Resistor
res = Resistor(1000.0; name=:r1)
for _ in 1:10
    stamp!(res, vctx, 1, 2)
end
alloc_res = @allocated stamp!(res, vctx, 1, 2)
println("   stamp!(Resistor): $alloc_res bytes")

# Capacitor
cap = Capacitor(1.0e-6; name=:c1)
for _ in 1:10
    stamp!(cap, vctx, 2, 0)
end
alloc_cap = @allocated stamp!(cap, vctx, 2, 0)
println("   stamp!(Capacitor): $alloc_cap bytes")

# Test with Float64 time (no Dual type)
println("\nTime type tests:")
t_f64 = Float64(0.5)
for _ in 1:10
    reset_value_only!(vctx)
    stamp!(pwl, vctx, 1, 0; t=t_f64, _sim_mode_=:tran)
end
reset_value_only!(vctx)
alloc_pwl_f64 = @allocated stamp!(pwl, vctx, 1, 0; t=t_f64, _sim_mode_=:tran)
println("   stamp!(PWL, t::Float64): $alloc_pwl_f64 bytes")

# Test PWL lookup directly
using CedarSim.MNA: get_source_value
for _ in 1:10
    get_source_value(pwl, t_f64, :tran)
end
alloc_pwl_lookup = @allocated get_source_value(pwl, t_f64, :tran)
println("   get_source_value(PWL): $alloc_pwl_lookup bytes")

println("\n" * "=" ^ 80)
println("BenchmarkTools measurement for fast_rebuild!:")
println("=" ^ 80)
display(@benchmark fast_rebuild!($ws, $u, $t))
println()
