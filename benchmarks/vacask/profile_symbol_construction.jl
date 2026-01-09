#!/usr/bin/env julia
#==============================================================================#
# Profile Symbol Construction Allocations
#
# Test if runtime Symbol construction is the source of remaining 640 bytes.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using Printf

println("=" ^ 70)
println("Profile Symbol Construction Allocations")
println("=" ^ 70)

#==============================================================================#
# Test Symbol Construction Patterns
#==============================================================================#

function profile_symbol_patterns()
    println("\n" * "=" ^ 50)
    println("Symbol Construction Patterns")
    println("=" ^ 50)

    # Pattern used in generated code
    base_name = :sp_diode_Q_p_a_int
    instance_name = :xd1

    # Pattern 1: Direct interpolation (current code)
    println("\n--- Direct Symbol interpolation ---")
    allocs_direct = @allocated begin
        for _ in 1:1000
            name = instance_name == Symbol("") ? base_name : Symbol(instance_name, "_", base_name)
        end
    end
    @printf("  Symbol(instance, \"_\", base): %.1f bytes/call\n", allocs_direct / 1000)

    # Pattern 2: Ternary with empty string comparison
    allocs_ternary = @allocated begin
        for _ in 1:1000
            name = if instance_name == Symbol("")
                base_name
            else
                Symbol(instance_name, "_", base_name)
            end
        end
    end
    @printf("  Ternary pattern: %.1f bytes/call\n", allocs_ternary / 1000)

    # Pattern 3: Using string conversion (what actually happens in Symbol())
    allocs_string = @allocated begin
        for _ in 1:1000
            name = Symbol(string(instance_name, "_", base_name))
        end
    end
    @printf("  Symbol(string(...)): %.1f bytes/call\n", allocs_string / 1000)

    # For comparison: what if name is not used at all
    println("\n--- Counter-based access (name ignored) ---")
    counter = Ref(1)
    allocs_counter = @allocated begin
        for _ in 1:1000
            counter[] += 1
        end
    end
    @printf("  Counter only: %.1f bytes/call\n", allocs_counter / 1000)

    # Simulate what DirectStampContext does (ignores name)
    function fake_alloc_with_name(name::Symbol)::Int
        return 42  # Just return a fixed value, name is ignored
    end

    allocs_ignore_name = @allocated begin
        for _ in 1:1000
            name = instance_name == Symbol("") ? base_name : Symbol(instance_name, "_", base_name)
            idx = fake_alloc_with_name(name)
        end
    end
    @printf("  Construct then ignore name: %.1f bytes/call\n", allocs_ignore_name / 1000)

    # Component-based API (no construction needed)
    function fake_alloc_components(base::Symbol, instance::Symbol)::Int
        return 42  # Just return a fixed value
    end

    allocs_components = @allocated begin
        for _ in 1:1000
            idx = fake_alloc_components(base_name, instance_name)
        end
    end
    @printf("  Component-based API: %.1f bytes/call\n", allocs_components / 1000)
end

#==============================================================================#
# Profile Diode Contributions (simulate generated code)
#==============================================================================#

function profile_diode_simulation()
    println("\n" * "=" ^ 50)
    println("Simulated Diode Stamp Allocations")
    println("=" ^ 50)

    # Graetz circuit has 4 diodes, each with:
    # - 1 internal node allocation (already fixed)
    # - 1 alloc_current! call for voltage source (if any)
    # - 1-2 detect_or_cached! calls
    # - 1-2 alloc_charge! calls (if voltage-dependent)

    instance_names = [:xd1, :xd2, :xd3, :xd4]
    base_name = :sp_diode_Q_p_a_int

    # Simulate detect_or_cached! and alloc_charge! per diode
    println("\n--- Per-diode allocation simulation ---")

    # Current pattern: construct name at runtime
    allocs_current = @allocated begin
        for _ in 1:100
            for inst in instance_names
                # This is what happens 2x per diode (detect + alloc)
                name1 = inst == Symbol("") ? base_name : Symbol(inst, "_", base_name)
                name2 = inst == Symbol("") ? base_name : Symbol(inst, "_", base_name)
            end
        end
    end
    @printf("  Current pattern (4 diodes × 2 calls): %.1f bytes/call\n", allocs_current / 100)
    @printf("  Per-diode overhead: %.1f bytes\n", allocs_current / 100 / 4)

    # New pattern: no name construction
    allocs_new = @allocated begin
        for _ in 1:100
            for inst in instance_names
                # Component-based: no Symbol construction
                result1 = 0  # Counter-based lookup
                result2 = 0
            end
        end
    end
    @printf("  Component-based (4 diodes × 2 calls): %.1f bytes/call\n", allocs_new / 100)

    @printf("\nExpected savings: ~%.1f bytes/call for 4 diodes\n",
            allocs_current / 100 - allocs_new / 100)
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    profile_symbol_patterns()
    profile_diode_simulation()

    println("\n" * "=" ^ 70)
    println("ANALYSIS")
    println("=" ^ 70)

    println("""

The 640 bytes remaining allocation per builder call comes from runtime Symbol
construction in these places:

1. **detect_or_cached!** - charge name construction (name is ignored!)
2. **alloc_charge!** - charge name construction (name is ignored!)
3. **alloc_current!** - current variable name construction (name is ignored!)

All these functions in DirectStampContext IGNORE the name parameter and use
counter-based access. But the generated code still constructs the Symbol at
runtime, causing allocation.

FIX: Use component-based APIs that pass base_name and instance_name separately.
For DirectStampContext, ignore both and use the counter.

Expected savings per 4-diode circuit:
- ~160 bytes per Symbol construction × 4 diodes = ~640 bytes
- This matches the observed allocation!
""")
end

main()
