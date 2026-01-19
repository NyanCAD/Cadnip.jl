#!/usr/bin/env julia
#==============================================================================#
# PSP103VA Minimal Example - Large Function Compilation Issue
#
# This demonstrates the compiler scalability issue with the PSP103VA model.
#
# Key facts:
# - PSP103VA struct has 782 Float64 parameter fields (12KB)
# - The stamp! method has 96,561 lowered IR statements
# - First compilation takes ~140 seconds
#
# The problem:
# Without barriers, when stamp! is embedded in a circuit builder function,
# Julia may try to inline/specialize aggressively, causing:
# - Very long compile times (minutes)
# - Memory pressure during LLVM compilation
# - In some cases, LLVM crashes (SROA pass exploding on large structs)
#
# The workarounds:
# 1. Base.invokelatest(stamp!, ...) - prevents inlining, forces runtime dispatch
# 2. Base.inferencebarrier(dev) - hides struct type from compiler
#
# Usage:
#   julia --project=test test/psp103_minimal_example.jl
#
#==============================================================================#

using CedarSim
using CedarSim.MNA: MNAContext, MNASpec, stamp!, get_node!
using PSPModels

println("PSP103VA Compilation Scalability Example")
println("="^60)
println()

# Show model size
dev = PSP103VA_module.PSP103VA()
println("PSP103VA model statistics:")
println("  Struct fields: $(length(fieldnames(typeof(dev))))")
println("  Struct size: $(sizeof(dev)) bytes")
println()

# Get lowered code size
inner_fn = PSP103VA_module.var"#stamp!#2"
args = (Real, Symbol, AbstractVector, MNASpec, Symbol, typeof(stamp!), typeof(dev),
        Union{CedarSim.MNA.AnyMNAContext, CedarSim.MNA.AnyStampContext}, Int64, Int64, Int64, Int64)
lowered = code_lowered(inner_fn, args)
if !isempty(lowered)
    println("  stamp! lowered IR statements: $(length(lowered[1].code))")
end
println()

#==============================================================================#
# Test 1: Direct stamp! call (first compilation)
#==============================================================================#

println("Test 1: First stamp! call (triggers 96K-statement compilation)")
println("-"^60)

ctx = MNAContext()
d = get_node!(ctx, :d)
g = get_node!(ctx, :g)
spec = MNASpec()

println("Calling stamp! directly (no barriers)...")
t1 = time()
stamp!(dev, ctx, d, g, 0, 0;
    _mna_t_=0.0, _mna_mode_=spec.mode, _mna_x_=Float64[], _mna_spec_=spec,
    _mna_instance_=:m1)
elapsed1 = time() - t1
println("First call: $(round(elapsed1, digits=2)) seconds")
println()

#==============================================================================#
# Test 2: Second call (already compiled)
#==============================================================================#

println("Test 2: Second stamp! call (already compiled)")
println("-"^60)

t2 = time()
stamp!(dev, ctx, d, g, 0, 0;
    _mna_t_=0.0, _mna_mode_=spec.mode, _mna_x_=Float64[], _mna_spec_=spec,
    _mna_instance_=:m1)
elapsed2 = time() - t2
println("Second call: $(round(elapsed2 * 1000, digits=2)) milliseconds")
println()

#==============================================================================#
# Test 3: Circuit builder with barriers (production pattern)
#==============================================================================#

println("Test 3: Circuit builder WITH barriers")
println("-"^60)

function circuit_with_barriers(spec::MNASpec)
    ctx = MNAContext()
    d = get_node!(ctx, :d)
    g = get_node!(ctx, :g)

    # Barriers: hide type + prevent inlining
    dev = Base.inferencebarrier(PSP103VA_module.PSP103VA())
    Base.invokelatest(stamp!, dev, ctx, d, g, 0, 0;
        _mna_t_=0.0, _mna_mode_=spec.mode, _mna_x_=Float64[], _mna_spec_=spec,
        _mna_instance_=:m1)

    return ctx
end

t3 = time()
ctx3 = circuit_with_barriers(MNASpec())
elapsed3 = time() - t3
println("With barriers: $(round(elapsed3 * 1000, digits=2)) milliseconds")
println()

#==============================================================================#
# Test 4: Circuit builder without barriers
#==============================================================================#

println("Test 4: Circuit builder WITHOUT barriers")
println("-"^60)

function circuit_no_barriers(spec::MNASpec)
    ctx = MNAContext()
    d = get_node!(ctx, :d)
    g = get_node!(ctx, :g)

    # No barriers - relies on stamp! already being compiled
    dev = PSP103VA_module.PSP103VA()
    stamp!(dev, ctx, d, g, 0, 0;
        _mna_t_=0.0, _mna_mode_=spec.mode, _mna_x_=Float64[], _mna_spec_=spec,
        _mna_instance_=:m1)

    return ctx
end

t4 = time()
ctx4 = circuit_no_barriers(MNASpec())
elapsed4 = time() - t4
println("No barriers: $(round(elapsed4 * 1000, digits=2)) milliseconds")
println()

#==============================================================================#
# Summary
#==============================================================================#

println("="^60)
println("Summary")
println("="^60)
println()
println("First stamp! compilation: $(round(elapsed1, digits=1)) seconds (96K IR statements)")
println("Subsequent calls: <1 millisecond")
println()
println("The barriers (invokelatest + inferencebarrier) ensure that:")
println("1. stamp! is compiled separately (not inlined into circuit builder)")
println("2. The 782-field struct type is hidden from aggressive optimization")
println()
println("Without barriers, if stamp! weren't already compiled, Julia would try")
println("to compile the entire circuit builder + stamp! together, potentially")
println("causing very long compile times or LLVM crashes on some systems.")
