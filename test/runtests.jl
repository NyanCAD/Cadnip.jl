using Test
using Random

# avoid random CI failures due to randomness
# fixed seed chosen by fair dice roll
Random.seed!(10)

# Core MNA tests
@testset "MNA Integration" begin
    include("test_mna_spectre.jl")
end

# CircuitSystem tests (named solution access)
@testset "CircuitSystem" begin
    include("test_circuit_system.jl")
end

# VA Device tests
@testset "VA Devices" begin
    include("test_va_devices.jl")
end
