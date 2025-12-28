using Test
using Random

# avoid random CI failures due to randomness
# fixed seed chosen by fair dice roll
Random.seed!(10)

# DAECompiler has been removed - MNA is the only backend
# Tests are organized by feature, not by phase

using CedarSim

@info "Running MNA test suite"

# Parsing and codegen tests
@testset "Parsing Tests" begin
    @testset "spectre_expr.jl" include("spectre_expr.jl")
    @testset "sweep.jl" include("sweep.jl")
end

# MNA core tests
@testset "MNA Core" begin
    @testset "mna/core.jl" include("mna/core.jl")
    @testset "mna/precompile.jl" include("mna/precompile.jl")
end

# VA integration tests (s-dual contribution stamping)
@testset "MNA VA Integration" begin
    @testset "mna/va.jl" include("mna/va.jl")
    @testset "ddx.jl" include("ddx.jl")
    @testset "varegress.jl" include("varegress.jl")
end

# Multi-terminal VA devices and MOSFET tests
@testset "MNA VA Multi-Terminal" begin
    @testset "mna/va_mosfet.jl" include("mna/va_mosfet.jl")
    @testset "mna/vadistiller.jl" include("mna/vadistiller.jl")
end

# Basic circuit tests using MNA backend
@testset "MNA Basic Tests" begin
    @testset "basic.jl" include("basic.jl")
    @testset "transients.jl" include("transients.jl")
    @testset "params.jl" include("params.jl")
end

# NOTE: The following tests require DAECompiler and are kept as reference for future porting:
# - compilation.jl, compiler_sanity.jl (DAECompiler compilation tests)
# - ac.jl, sensitivity.jl, inverter_noise.jl (AC/noise analysis - needs MNA porting)
# - alias.jl, inverter.jl, gf180_dff.jl (full simulation tests)
# - bsimcmg/*.jl, sky130/*.jl, binning/*.jl (PDK-specific tests)
# - MTK_extension.jl (ModelingToolkit extension)
