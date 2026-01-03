#!/usr/bin/env julia
#==============================================================================#
# Type Stability Analysis for PWL stamp!
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace
using CedarSim.MNA: reset_value_only!, ValueOnlyContext, stamp!
using CedarSim.MNA: PWLVoltageSource, get_source_value
using StaticArrays
using InteractiveUtils

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
vctx = ws.vctx

# Create PWL device
times_sv = SVector{6,Float64}(0.0, 1.0e-6, 2.0e-6, 0.001002, 0.001003, 0.002)
values_sv = SVector{6,Float64}(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
pwl = PWLVoltageSource(times_sv, values_sv; name=:vs)

t = Float64(0.5)

println("=" ^ 80)
println("Type Analysis")
println("=" ^ 80)

println("\n1. PWL source type:")
println("   $(typeof(pwl))")

println("\n2. ValueOnlyContext type:")
println("   $(typeof(vctx))")

println("\n3. get_source_value @code_warntype:")
println("-" ^ 40)
@code_warntype get_source_value(pwl, t, :tran)
println("-" ^ 40)

println("\n4. stamp!(PWL, ValueOnlyContext) @code_warntype:")
reset_value_only!(vctx)
println("-" ^ 40)
@code_warntype stamp!(pwl, vctx, 1, 0; t=t, _sim_mode_=:tran)
println("-" ^ 40)

println("\n5. Effects analysis with @code_typed:")
println("-" ^ 40)
@code_typed stamp!(pwl, vctx, 1, 0; t=t, _sim_mode_=:tran)
println("-" ^ 40)

# Check the builder call
println("\n6. Builder call @code_warntype:")
u = zeros(Float64, cs.n)
reset_value_only!(vctx)
println("-" ^ 40)
@code_warntype cs.builder(cs.params, cs.spec, t; x=u, ctx=vctx)
println("-" ^ 40)
