#!/usr/bin/env julia
#==============================================================================#
# Julia/LLVM Compiler Hang Reproducer
#
# Demonstrates compiler hang with large structs + large functions.
#
# Real-world context: PSP103VA transistor model from circuit simulation
# - 782 Float64 parameter fields in struct
# - stamp! function generates ~96,000 LLVM IR statements
# - Used in circuit simulation (CedarSim/Cadnip.jl)
#
# The problem:
# - LLVM's SROA (Scalar Replacement of Aggregates) pass explodes on large structs
# - Julia tries to specialize on the full struct type
# - Large functions get inlined, compounding the problem
#
# Workarounds in production:
# - Base.invokelatest(f, args...) - forces runtime dispatch
# - Base.inferencebarrier(x) - hides type from inference
#
# This reproducer is self-contained (no external dependencies).
#
# Usage:
#   julia psp103_compiler_bug_reproducer.jl [n_fields] [n_ops]
#
# Example (should complete):
#   julia psp103_compiler_bug_reproducer.jl 100 100
#
# Example (may hang with high values):
#   julia psp103_compiler_bug_reproducer.jl 500 1000
#
# Real PSP103VA scale (definitely hangs without workarounds):
#   julia psp103_compiler_bug_reproducer.jl 782 2000
#==============================================================================#

# Parse command line args
const N_FIELDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
const N_OPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500

println("Julia Compiler Hang Reproducer")
println("="^60)
println("Julia version: $(VERSION)")
println("N_FIELDS: $N_FIELDS (PSP103VA has 782)")
println("N_OPS: $N_OPS (PSP103VA stamp! has ~2000+ operations)")
println()

#==============================================================================#
# Step 1: Generate large struct (like PSP103VA's 782 parameters)
#==============================================================================#

println("Generating struct with $N_FIELDS fields...")

# Build struct expression
field_exprs = [:($(Symbol("p$i"))::Float64 = $(0.001 * i)) for i in 1:N_FIELDS]
struct_name = :LargeVADevice

# Use @kwdef-style struct with defaults
struct_code = quote
    Base.@kwdef struct $struct_name
        $(field_exprs...)
    end
end

eval(struct_code)
println("✓ Struct created: $(sizeof(eval(struct_name)())) bytes")

#==============================================================================#
# Step 2: Generate large function (like PSP103VA's stamp! method)
#
# The real stamp! function pattern:
# 1. Extract all parameters from device struct
# 2. Perform hundreds of arithmetic operations
# 3. Branch based on operating region (cutoff/linear/saturation)
# 4. Compute currents, charges, and their derivatives
# 5. Stamp results into sparse matrix
#==============================================================================#

println("Generating function with $N_OPS operations...")

# Build the function body
body_exprs = Any[]

# Parameter extraction (like real stamp! does)
for i in 1:N_FIELDS
    push!(body_exprs, :($(Symbol("v$i")) = dev.$(Symbol("p$i"))))
end

# Intermediate variables
push!(body_exprs, :(acc = 0.0))
push!(body_exprs, :(Ids = 0.0))
push!(body_exprs, :(Qgs = 0.0))

# Generate many operations (simulating real VA model equations)
for i in 1:N_OPS
    # Cycle through variables
    v1 = Symbol("v$(mod1(i, N_FIELDS))")
    v2 = Symbol("v$(mod1(i + 3, N_FIELDS))")
    v3 = Symbol("v$(mod1(i + 7, N_FIELDS))")

    if i % 5 == 0
        # Exponential (common in MOSFET models)
        push!(body_exprs, :(acc += exp(-abs($v1)) * $v2))
    elseif i % 5 == 1
        # Square root (threshold calculations)
        push!(body_exprs, :(acc += sqrt(abs($v1) + 1e-12) * $v2))
    elseif i % 5 == 2
        # Polynomial (IV curves)
        push!(body_exprs, :(acc += $v1 * $v1 * $v2 + $v3))
    elseif i % 5 == 3
        # Logarithm (subthreshold)
        push!(body_exprs, :(acc += log1p(abs($v1)) * $v2 / (1 + abs($v3))))
    else
        # Branching (region-dependent)
        push!(body_exprs, quote
            if $v1 > 0.5
                Ids += $v2 * $v3 * $v3
            else
                Ids += $v2 * (2 * $v3 - $v1)
            end
        end)
    end
end

# Final accumulation
push!(body_exprs, :(Ids += acc * 1e-6))
push!(body_exprs, :(result[1] = Ids))
push!(body_exprs, :(return nothing))

# Build the function
fn_body = Expr(:block, body_exprs...)
fn_expr = :(function stamp_large!(dev::$struct_name, result::Vector{Float64})
    $fn_body
end)

eval(fn_expr)
println("✓ Function created")
println()

#==============================================================================#
# Step 3: Show code statistics
#==============================================================================#

println("Code Statistics")
println("-"^40)

# Get lowered code
lowered = code_lowered(stamp_large!, (eval(struct_name), Vector{Float64}))
n_stmts = isempty(lowered) ? 0 : length(lowered[1].code)
println("Lowered IR statements: $n_stmts")
println()

#==============================================================================#
# Step 4: Test compilation
#==============================================================================#

println("="^60)
println("TEST 1: Direct call (may trigger compiler hang)")
println("="^60)
println()

# Create instance
DevType = eval(struct_name)
dev = DevType()
result = [0.0]

println("Calling stamp_large! directly...")
println("This triggers full type-specialized compilation.")
println("With high N_FIELDS/N_OPS, this may hang for minutes or crash.")
println()

t1 = time()
stamp_large!(dev, result)
elapsed1 = time() - t1

println("✓ Direct call completed in $(round(elapsed1, digits=2))s")
println("  Result: $(result[1])")
println()

#==============================================================================#
# Step 5: Test with workarounds
#==============================================================================#

println("="^60)
println("TEST 2: With invokelatest (workaround)")
println("="^60)
println()

result[1] = 0.0
t2 = time()
Base.invokelatest(stamp_large!, dev, result)
elapsed2 = time() - t2

println("✓ invokelatest call completed in $(round(elapsed2, digits=2))s")
println("  Result: $(result[1])")
println()

println("="^60)
println("TEST 3: With inferencebarrier (workaround)")
println("="^60)
println()

# Function that uses inferencebarrier
function stamp_with_barrier!(dev, result)
    hidden = Base.inferencebarrier(dev)
    stamp_large!(hidden, result)
end

result[1] = 0.0
t3 = time()
stamp_with_barrier!(dev, result)
elapsed3 = time() - t3

println("✓ inferencebarrier call completed in $(round(elapsed3, digits=2))s")
println("  Result: $(result[1])")
println()

#==============================================================================#
# Summary
#==============================================================================#

println("="^60)
println("SUMMARY")
println("="^60)
println()
println("Direct call:       $(round(elapsed1, digits=2))s")
println("invokelatest:      $(round(elapsed2, digits=2))s")
println("inferencebarrier:  $(round(elapsed3, digits=2))s")
println()
println("If direct call was significantly slower or hung, it demonstrates")
println("the compiler overhead from specializing on large structs.")
println()
println("To reproduce real PSP103VA behavior, try:")
println("  julia $PROGRAM_FILE 782 2000")
println()
println("The workarounds (invokelatest, inferencebarrier) prevent the")
println("compiler from attempting full specialization/inlining.")
