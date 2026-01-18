#==============================================================================#
# Adaptive Timestep Allocation Test
#
# This test verifies which ODE solvers can achieve zero-allocation adaptive
# (variable) timestep stepping. Key findings:
#
# ZERO ALLOCATION with adaptive=true:
# - QNDF: Variable-order quasi-constant BDF (recommended for stiff circuits)
# - Rodas5P: 5th order Rosenbrock (recommended for moderate stiffness)
# - Rosenbrock23: 2-3 order Rosenbrock (robust fallback)
# - ImplicitEuler: 1st order (simple, stable)
#
# ALLOCATES with adaptive=true:
# - FBDF: ~56 bytes/step due to Lagrange interpolation in order control
#   (calc_Lagrange_interp! at bdf_utils.jl:157 and choose_order! at controllers.jl:178)
#
# REQUIREMENTS for zero-allocation adaptive stepping:
# 1. Dense matrices (dense=true) - Sparse UMFPACK lu! allocates ~1696 bytes/call
# 2. blind_step!() wrapper - step!() returns ReturnCode causing 16 bytes boxing
# 3. autodiff=false - Use explicit Jacobian from MNA
# 4. Compatible solver (see list above)
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNACircuit, voltage, blind_step!
using OrdinaryDiffEq
using OrdinaryDiffEq: FBDF, QNDF, Rodas5P, Rosenbrock23, ImplicitEuler
using SciMLBase

include(joinpath(@__DIR__, "..", "common.jl"))

#==============================================================================#
# BJT Device Model
#==============================================================================#

va"""
module npnbjt_adapt(b, e, c);
    inout b, e, c;
    electrical b, e, c;
    analog I(b,e) <+ 1.0e-14*(exp(V(b,e)/25.0e-3) - 1) - 1.0/(1 + 100.0)*1.0e-14*(exp(V(b,c)/25.0e-3) - 1);
    analog I(c,b) <+ 100.0/(1 + 100.0)*1.0e-14*(exp(V(b,e)/25.0e-3) - 1) - 1.0e-14*(exp(V(b,c)/25.0e-3) - 1);
endmodule
"""

const bjt_circuit_code = CedarSim.parse_spice_to_mna("""
* BJT with collector resistor
Vcc vcc 0 DC 12.0
Vb base 0 DC 0.65
Rc vcc coll 4.7k
X1 base 0 coll npnbjt_adapt
"""; circuit_name=:bjt_adaptive_test, imported_hdl_modules=[npnbjt_adapt_module])
eval(bjt_circuit_code)

#==============================================================================#
# Allocation measurement helper
#==============================================================================#

function measure_allocations(f::Function; warmup=1000, iters=10000)
    for _ in 1:warmup
        f()
    end
    GC.gc()

    GC.enable(false)
    try
        allocs = @allocated begin
            for _ in 1:iters
                f()
            end
        end
        return allocs ÷ iters
    finally
        GC.enable(true)
    end
end

#==============================================================================#
# Tests
#==============================================================================#

@testset "Adaptive Timestep Allocation" begin

    circuit = MNACircuit(bjt_adaptive_test)
    prob = ODEProblem(circuit, (0.0, 1e-6); dense=true)

    @testset "Fixed timestep baseline (QNDF)" begin
        # Fixed timestep should always be zero-allocation
        integrator = init(prob, QNDF(autodiff=false);
            adaptive=false,
            dt=1e-9,
            save_on=false,
            dense=false,
            maxiters=10_000_000,
            initializealg=MNA.CedarTranOp())

        # Warmup
        for _ in 1:1000
            blind_step!(integrator)
        end

        allocs = measure_allocations(; warmup=100, iters=10000) do
            blind_step!(integrator)
        end
        @test allocs == 0
        @info "QNDF fixed timestep: $allocs bytes/step"
    end

    @testset "QNDF adaptive (zero allocation)" begin
        integrator = init(prob, QNDF(autodiff=false);
            adaptive=true,
            dt=1e-9,
            save_on=false,
            dense=false,
            maxiters=10_000_000,
            initializealg=MNA.CedarTranOp())

        for _ in 1:1000
            blind_step!(integrator)
        end

        reinit!(integrator, prob.u0)
        allocs = measure_allocations(; warmup=100, iters=10000) do
            blind_step!(integrator)
        end
        @test allocs == 0
        @info "QNDF adaptive: $allocs bytes/step"
    end

    @testset "Rodas5P adaptive (zero allocation)" begin
        integrator = init(prob, Rodas5P(autodiff=false);
            adaptive=true,
            dt=1e-9,
            save_on=false,
            dense=false,
            maxiters=10_000_000,
            initializealg=MNA.CedarTranOp())

        for _ in 1:1000
            blind_step!(integrator)
        end

        reinit!(integrator, prob.u0)
        allocs = measure_allocations(; warmup=100, iters=10000) do
            blind_step!(integrator)
        end
        @test allocs == 0
        @info "Rodas5P adaptive: $allocs bytes/step"
    end

    @testset "Rosenbrock23 adaptive (zero allocation)" begin
        integrator = init(prob, Rosenbrock23(autodiff=false);
            adaptive=true,
            dt=1e-9,
            save_on=false,
            dense=false,
            maxiters=10_000_000,
            initializealg=MNA.CedarTranOp())

        for _ in 1:1000
            blind_step!(integrator)
        end

        reinit!(integrator, prob.u0)
        allocs = measure_allocations(; warmup=100, iters=10000) do
            blind_step!(integrator)
        end
        @test allocs == 0
        @info "Rosenbrock23 adaptive: $allocs bytes/step"
    end

    @testset "ImplicitEuler adaptive (zero allocation)" begin
        integrator = init(prob, ImplicitEuler(autodiff=false);
            adaptive=true,
            dt=1e-9,
            save_on=false,
            dense=false,
            maxiters=10_000_000,
            initializealg=MNA.CedarTranOp())

        for _ in 1:1000
            blind_step!(integrator)
        end

        reinit!(integrator, prob.u0)
        allocs = measure_allocations(; warmup=100, iters=10000) do
            blind_step!(integrator)
        end
        @test allocs == 0
        @info "ImplicitEuler adaptive: $allocs bytes/step"
    end

    @testset "FBDF adaptive (allocates - documented behavior)" begin
        # FBDF allocates due to Lagrange interpolation in variable order control
        # This is expected behavior - use QNDF instead for zero-allocation BDF
        integrator = init(prob, FBDF(autodiff=false);
            adaptive=true,
            dt=1e-9,
            save_on=false,
            dense=false,
            maxiters=10_000_000,
            initializealg=MNA.CedarTranOp())

        for _ in 1:1000
            blind_step!(integrator)
        end

        reinit!(integrator, prob.u0)
        allocs = measure_allocations(; warmup=100, iters=10000) do
            blind_step!(integrator)
        end
        # FBDF allocates ~56 bytes/step in choose_order! and calc_Lagrange_interp!
        @test allocs > 0  # Expected to allocate
        @info "FBDF adaptive: $allocs bytes/step (expected to allocate)"
    end

end  # testset
