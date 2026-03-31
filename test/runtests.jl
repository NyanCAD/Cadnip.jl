using Test
using Random

# avoid random CI failures due to randomness
# fixed seed chosen by fair dice roll
Random.seed!(10)

# Test group filtering via ARGS (from Pkg.test(test_args=[...]))
# Supported groups:
#   - "integration": Run only vadistiller integration tests (large VA models)
#   - "core": Run core tests only (excludes integration)
#   - "all": Run all tests including integration
#   - (default): Same as "core" - integration tests are opt-in
const RUN_INTEGRATION = "integration" in ARGS || "all" in ARGS
const RUN_CORE = !("integration" in ARGS) || "all" in ARGS

if RUN_INTEGRATION && !RUN_CORE
    # Integration-only mode
    @info "Running integration tests only (large VA models)"
    using Cadnip
    @testset "VADistiller Integration" begin
        @testset "mna/vadistiller_integration.jl" include("mna/vadistiller_integration.jl")
    end
    @testset "Audio Integration" begin
        @testset "mna/audio_integration.jl" include("mna/audio_integration.jl")
    end
    @testset "PSP103VA Integration" begin
        @testset "mna/psp103_integration.jl" include("mna/psp103_integration.jl")
    end
    @testset "Photonic Integration" begin
        @testset "mna/photonic_integration.jl" include("mna/photonic_integration.jl")
    end
elseif RUN_CORE
    @info "Running core tests"

    using Cadnip

    # Tests that work with parsing only (no simulation required)
    @testset "Phase 0: Parsing Tests" begin
        @testset "spectre_expr.jl" include("spectre_expr.jl")
        @testset "sweep.jl" include("sweep.jl")
    end

    # MNA core tests
    @testset "Phase 1: MNA Core" begin
        @testset "mna/core.jl" include("mna/core.jl")
        @testset "mna/precompile.jl" include("mna/precompile.jl")
    end

    # VA integration tests (s-dual contribution stamping)
    @testset "Phase 5: MNA VA Integration" begin
        @testset "mna/va.jl" include("mna/va.jl")
        @testset "ddx.jl" include("ddx.jl")
        @testset "varegress.jl" include("varegress.jl")
    end

    # Multi-terminal VA devices and MOSFET tests
    @testset "Phase 6: MNA VA Multi-Terminal" begin
        @testset "mna/va_mosfet.jl" include("mna/va_mosfet.jl")
        @testset "mna/vadistiller.jl" include("mna/vadistiller.jl")
    end

    # Basic tests using MNA backend
    @testset "Phase 4: MNA Basic Tests" begin
        @testset "basic.jl" include("basic.jl")
        @testset "transients.jl" include("transients.jl")
        @testset "params.jl" include("params.jl")
    end

    # Photonic unit tests (array ports, custom access functions, module instantiation)
    @testset "Photonic" begin
        @testset "mna/photonic.jl" include("mna/photonic.jl")
    end

    # Laplace/idt operator tests
    @testset "Laplace/IDT" begin
        @testset "mna/laplace.jl" include("mna/laplace.jl")
    end

    # AC small-signal analysis
    @testset "AC Analysis" begin
        @testset "ac.jl" include("ac.jl")
    end

    # PDK Precompilation tests
    @testset "PDK Precompilation" begin
        @testset "testpdk/pdk_test.jl" include("testpdk/pdk_test.jl")
    end

    # Integration tests (only if explicitly requested with "all")
    if RUN_INTEGRATION
        GC.gc()  # Clean up before heavy tests
        @testset "VADistiller Integration" begin
            @testset "mna/vadistiller_integration.jl" include("mna/vadistiller_integration.jl")
        end
        @testset "Audio Integration" begin
            @testset "mna/audio_integration.jl" include("mna/audio_integration.jl")
        end
        @testset "PSP103VA Integration" begin
            @testset "mna/psp103_integration.jl" include("mna/psp103_integration.jl")
        end
        @testset "Photonic Integration" begin
            @testset "mna/photonic_integration.jl" include("mna/photonic_integration.jl")
        end
    end
end
