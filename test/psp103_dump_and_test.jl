#!/usr/bin/env julia
#==============================================================================#
# PSP103VA Code Dump and Minimal Reproducer
#
# This script:
# 1. Loads PSPModels to get the real PSP103VA device
# 2. Dumps the struct definition and stamp! method code
# 3. Creates a minimal standalone file that reproduces the compiler hang
#
# Usage:
#   julia --project=test test/psp103_dump_and_test.jl
#==============================================================================#

using InteractiveUtils
using CedarSim
using CedarSim.MNA: MNAContext, MNASpec, stamp!, get_node!
using PSPModels

println("="^70)
println("PSP103VA Code Dump and Analysis")
println("="^70)
println()

#==============================================================================#
# Analyze PSP103VA struct
#==============================================================================#

const PSP103VA_Type = typeof(PSP103VA_module.PSP103VA())

println("PSP103VA Struct Analysis")
println("-"^40)
println("Type: $PSP103VA_Type")
println("Number of fields: $(length(fieldnames(PSP103VA_Type)))")
println("Struct size: $(sizeof(PSP103VA_module.PSP103VA())) bytes")
println()

# List all field names and types
println("Field names (first 20):")
for (i, name) in enumerate(fieldnames(PSP103VA_Type)[1:min(20, end)])
    T = fieldtype(PSP103VA_Type, i)
    println("  $name::$T")
end
if length(fieldnames(PSP103VA_Type)) > 20
    println("  ... and $(length(fieldnames(PSP103VA_Type)) - 20) more fields")
end
println()

#==============================================================================#
# Analyze stamp! method
#==============================================================================#

println("stamp! Method Analysis")
println("-"^40)

# Find the stamp! method for PSP103VA
dev = PSP103VA_module.PSP103VA()
ctx = MNAContext()
d = get_node!(ctx, :d)
g = get_node!(ctx, :g)
spec = MNASpec()

# Get the method that would be called
method = @which stamp!(dev, ctx, d, g, 0, 0; _mna_spec_=spec, _mna_x_=Float64[], _mna_instance_=:m1)
println("Method: $method")
println()

# Get lowered code size
println("Getting lowered code statistics...")
try
    # This gives us the lowered IR
    lowered = code_lowered(stamp!, (typeof(dev), typeof(ctx), Int, Int, Int, Int))
    if !isempty(lowered)
        n_stmts = length(lowered[1].code)
        println("Lowered IR statements: $n_stmts")
    end
catch e
    println("Could not get lowered code: $e")
end
println()

#==============================================================================#
# Write minimal reproducer file
#==============================================================================#

println("Generating minimal reproducer...")
println()

# Create a simplified struct with same number of fields
field_names = collect(fieldnames(PSP103VA_Type))
n_fields = length(field_names)

# Generate the minimal reproducer code
reproducer_code = """
#!/usr/bin/env julia
#==============================================================================#
# Minimal PSP103VA Compiler Hang Reproducer
#
# Auto-generated from psp103_dump_and_test.jl
#
# This file demonstrates the compiler hang with:
# - A struct with $n_fields fields (like PSP103VA)
# - A large computation function
#
# The issue: Julia/LLVM hangs when trying to compile functions that:
# 1. Take large structs as arguments (SROA tries to decompose all fields)
# 2. Generate very large IR (>90K statements in real PSP103VA)
#
# Workarounds:
# - Base.invokelatest() - forces runtime dispatch, prevents inlining
# - Base.inferencebarrier() - hides type from compiler
#==============================================================================#

println("Building struct with $n_fields fields...")

# Generate struct with same field count as PSP103VA
struct PSP103Stub
$(join(["    $(name)::Float64" for name in field_names], "\n"))
end

# Default constructor
function PSP103Stub()
    PSP103Stub($(join(["0.0" for _ in 1:n_fields], ", ")))
end

println("Struct created: \$(sizeof(PSP103Stub())) bytes")

# Generate a computation function that accesses many fields
# This simulates the stamp! function's field access pattern
function stamp_stub!(dev::PSP103Stub, result::Vector{Float64})
    # Access all fields (simulating parameter extraction)
$(join(["    p$i = dev.$(name)" for (i, name) in enumerate(field_names[1:min(100, end)])], "\n"))

    # Computations (simplified from real stamp!)
    acc = 0.0
$(join(["    acc += p$(mod1(i, 100)) * p$(mod1(i+1, 100))" for i in 1:200], "\n"))

    result[1] = acc
    return nothing
end

println("Function defined")
println()

#==============================================================================#
# Test Cases
#==============================================================================#

println("="^60)
println("Test: Direct call (this may hang)")
println("="^60)

dev = PSP103Stub()
result = [0.0]

println("Calling stamp_stub! directly...")
t_start = time()
stamp_stub!(dev, result)
t_elapsed = time() - t_start
println("Completed in \$(round(t_elapsed, digits=2))s, result = \$(result[1])")
println()

println("="^60)
println("Test: With invokelatest (workaround)")
println("="^60)

println("Calling via invokelatest...")
t_start = time()
Base.invokelatest(stamp_stub!, dev, result)
t_elapsed = time() - t_start
println("Completed in \$(round(t_elapsed, digits=2))s, result = \$(result[1])")
"""

# Write the reproducer file
reproducer_path = joinpath(@__DIR__, "psp103_reproducer.jl")
open(reproducer_path, "w") do io
    write(io, reproducer_code)
end

println("✓ Wrote minimal reproducer to: $reproducer_path")
println()

#==============================================================================#
# Test the actual PSP103VA without barriers
#==============================================================================#

println("="^70)
println("Testing Real PSP103VA")
println("="^70)
println()

println("Test 1: With invokelatest + inferencebarrier (production code path)")
println("-"^40)

function test_with_barriers()
    ctx = MNAContext()
    d = get_node!(ctx, :d)
    g = get_node!(ctx, :g)
    spec = MNASpec()

    dev = Base.inferencebarrier(PSP103VA_module.PSP103VA(; TYPE=CedarSim.DefaultOr{Int}(1)))
    Base.invokelatest(stamp!, dev, ctx, d, g, 0, 0;
        _mna_spec_=spec, _mna_x_=Float64[], _mna_instance_=:m1)

    return ctx
end

println("Calling with barriers...")
t_start = time()
ctx1 = test_with_barriers()
t_elapsed = time() - t_start
println("✓ Completed in $(round(t_elapsed, digits=2)) seconds")
println("  Nodes: $(length(ctx1.node_to_idx))")
println()

println("Test 2: Direct call WITHOUT barriers (may hang or be slow)")
println("-"^40)

# WARNING: This test may hang the Julia compiler!
# Comment out if you want to skip

function test_without_barriers()
    ctx = MNAContext()
    d = get_node!(ctx, :d)
    g = get_node!(ctx, :g)
    spec = MNASpec()

    # Direct call - no barriers
    dev = PSP103VA_module.PSP103VA(; TYPE=CedarSim.DefaultOr{Int}(1))
    stamp!(dev, ctx, d, g, 0, 0;
        _mna_spec_=spec, _mna_x_=Float64[], _mna_instance_=:m1)

    return ctx
end

println("Calling WITHOUT barriers (this triggers full compilation)...")
println("If this takes more than 60 seconds, the compiler is hanging.")
println()

t_start = time()
ctx2 = test_without_barriers()
t_elapsed = time() - t_start

println("✓ Completed in $(round(t_elapsed, digits=2)) seconds")
println("  Nodes: $(length(ctx2.node_to_idx))")
println()

println("="^70)
println("Done!")
println("="^70)
