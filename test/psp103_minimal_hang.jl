#!/usr/bin/env julia
#==============================================================================#
# Minimal Example: PSP103VA Compiler Hang
#
# This script demonstrates the LLVM compiler hang that occurs when:
# 1. Removing `Base.invokelatest` from stamp! calls (forces inline compilation)
# 2. Removing `Base.inferencebarrier` from device construction (exposes 782-field struct)
#
# The PSP103VA model generates a stamp! function with ~96,000 IR statements.
# Without the barriers, Julia/LLVM attempts to:
# - Specialize on the exact 782-field struct type (SROA explosion)
# - Inline the massive stamp! function into the circuit builder
# Both cause the compiler to hang or crash.
#
# Usage:
#   julia --project=test test/psp103_minimal_hang.jl
#
# Expected behavior:
# - With barriers (USE_BARRIERS=true): Compiles and runs in seconds
# - Without barriers (USE_BARRIERS=false): Compiler hangs indefinitely
#==============================================================================#

using CedarSim
using CedarSim.MNA: MNAContext, MNASpec, stamp!, get_node!, reset_for_restamping!, solve_dc
using PSPModels

# Toggle to demonstrate the hang
const USE_BARRIERS = parse(Bool, get(ENV, "USE_BARRIERS", "true"))

println("PSP103VA Minimal Hang Example")
println("="^60)
println("USE_BARRIERS = $USE_BARRIERS")
println()

if USE_BARRIERS
    println("Using invokelatest + inferencebarrier (this should work)")
else
    println("WARNING: Removing barriers - compiler will likely hang!")
    println("Press Ctrl+C to abort if it hangs for more than 60 seconds")
end
println()

# Show PSP103VA struct size
let T = typeof(PSP103VA_module.PSP103VA())
    n_fields = length(fieldnames(T))
    println("PSP103VA struct has $n_fields fields")
end
println()

#==============================================================================#
# Circuit Builder Functions
#
# Version 1: With barriers (works)
# Version 2: Without barriers (hangs compiler)
#==============================================================================#

"""
Circuit builder WITH invokelatest and inferencebarrier.
This is how production code handles large VA models.
"""
function circuit_with_barriers(params, spec::MNASpec, t::Real=0.0;
                               x::AbstractVector{Float64}=Float64[],
                               ctx::Union{MNAContext,Nothing}=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end

    # Get node indices
    d = get_node!(ctx, :d)
    g = get_node!(ctx, :g)

    # DC voltage sources (simple stamps)
    Vds_idx = get_node!(ctx, :I_vds)
    Vgs_idx = get_node!(ctx, :I_vgs)
    CedarSim.MNA.stamp_voltage_source!(ctx, d, 0, Vds_idx, 1.2)
    CedarSim.MNA.stamp_voltage_source!(ctx, g, 0, Vgs_idx, 0.6)

    # PSP103VA device with BOTH barriers:
    # 1. inferencebarrier: Hides 782-field struct type from Julia compiler
    # 2. invokelatest: Forces runtime dispatch, prevents inlining 96K IR statements
    let dev = Base.inferencebarrier(PSP103VA_module.PSP103VA(; TYPE=CedarSim.DefaultOr{Int}(1)))
        Base.invokelatest(stamp!, dev, ctx, d, g, 0, 0;
            _mna_t_=t, _mna_mode_=spec.mode, _mna_x_=x, _mna_spec_=spec,
            _mna_instance_=:m1)
    end

    return ctx
end

"""
Circuit builder WITHOUT barriers.
This triggers the compiler hang on large VA models.
"""
function circuit_without_barriers(params, spec::MNASpec, t::Real=0.0;
                                  x::AbstractVector{Float64}=Float64[],
                                  ctx::Union{MNAContext,Nothing}=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end

    # Get node indices
    d = get_node!(ctx, :d)
    g = get_node!(ctx, :g)

    # DC voltage sources
    Vds_idx = get_node!(ctx, :I_vds)
    Vgs_idx = get_node!(ctx, :I_vgs)
    CedarSim.MNA.stamp_voltage_source!(ctx, d, 0, Vds_idx, 1.2)
    CedarSim.MNA.stamp_voltage_source!(ctx, g, 0, Vgs_idx, 0.6)

    # PSP103VA device WITHOUT barriers - direct call
    # This exposes the full type and allows inlining, causing:
    # - LLVM SROA to attempt decomposing 782-field struct
    # - Julia to try specializing on full type signature
    # - Potential inlining of 96K IR statement stamp! function
    dev = PSP103VA_module.PSP103VA(; TYPE=CedarSim.DefaultOr{Int}(1))
    stamp!(dev, ctx, d, g, 0, 0;
        _mna_t_=t, _mna_mode_=spec.mode, _mna_x_=x, _mna_spec_=spec,
        _mna_instance_=:m1)

    return ctx
end

#==============================================================================#
# Test Execution
#==============================================================================#

println("Defining circuit function...")
circuit_fn = USE_BARRIERS ? circuit_with_barriers : circuit_without_barriers

println("Creating MNASpec...")
spec = MNASpec(temp=27.0, mode=:dcop)

println("Building circuit (this is where compilation happens)...")
println("  Start time: $(Dates.now())")

# This call triggers compilation - with barriers it's fast, without it hangs
using Dates
t_start = time()
ctx = circuit_fn((;), spec, 0.0)
t_elapsed = time() - t_start

println("  End time: $(Dates.now())")
println("  Elapsed: $(round(t_elapsed, digits=2)) seconds")
println()

println("Circuit built successfully!")
println("  Number of nodes: $(length(ctx.node_to_idx))")
println("  Matrix size: $(size(ctx.G))")

# Try solving DC
println()
println("Solving DC operating point...")
sol = solve_dc(circuit_fn, (;), spec)
println("  Vd = $(CedarSim.MNA.voltage(sol, :d)) V")
println("  Vg = $(CedarSim.MNA.voltage(sol, :g)) V")
println("  Id = $(CedarSim.MNA.current(sol, :I_vds)) A")
