#!/usr/bin/env julia
#==============================================================================#
# Test if keyword arguments cause the 24 bytes allocation
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!, reset_value_only!
using CedarSim.MNA: stamp!, PWLVoltageSource, get_node!
using StaticArrays

const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

spec = MNASpec()
cs = compile_structure(rc_circuit, NamedTuple(), spec)
ws = create_workspace(cs)
vctx = ws.vctx
u = zeros(Float64, cs.n)
t = Float64(0.5)

# Create PWL device
pwl = PWLVoltageSource(
    SVector{6,Float64}(0.0, 1.0e-6, 2.0e-6, 0.001002, 0.001003, 0.002),
    SVector{6,Float64}(0.0, 0.0, 1.0, 1.0, 0.0, 0.0);
    name=:vs
)

p_idx = vctx.node_to_idx[Symbol("1")]
n_idx = 0  # Ground node is always 0

println("PWL device: ", typeof(pwl))
println("p_idx: $p_idx, n_idx: $n_idx")
println()

# Warmup
for _ in 1:5000
    reset_value_only!(vctx)
    stamp!(pwl, vctx, p_idx, n_idx; t=t, _sim_mode_=:tran)
end

GC.gc(true)

println("stamp!(pwl, ctx, p, n; t=t, _sim_mode_=:tran) with keyword args:")
for i in 1:5
    reset_value_only!(vctx)
    a = @allocated stamp!(pwl, vctx, p_idx, n_idx; t=t, _sim_mode_=:tran)
    println("  call $i: $a bytes")
end

# Now let's test what the allocation looks like if we bypass kwargs
println("\nDirect call with forced positional (if available):")

# Check if there's a positional stamp! method we can call
methods_stamp = methods(stamp!, (typeof(pwl), typeof(vctx), Int, Int))
println("Available stamp! methods for this signature:")
for m in methods_stamp
    println("  ", m)
end

println("\n" * "=" ^ 60)
println("Compare to full fast_rebuild!")
println("=" ^ 60)

for _ in 1:5000
    fast_rebuild!(ws, u, t)
end
GC.gc(true)

println("\nfast_rebuild!:")
for i in 1:5
    a = @allocated fast_rebuild!(ws, u, t)
    println("  call $i: $a bytes")
end
