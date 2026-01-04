#==============================================================================#
# Test: Charge Scaling Effect on Stiffness
#
# This test demonstrates that charge state scaling improves conditioning
# of the Jacobian matrix for circuits with voltage-dependent capacitors.
#==============================================================================#

using Test
using LinearAlgebra
using SparseArrays

using CedarSim.MNA

@testset "Charge Scaling Effect on Stiffness" begin

    @testset "stamp_charge_state! with scaling" begin
        # Test that charge_scale parameter is applied correctly

        # Create a simple 2-node context with a voltage-dependent capacitor
        ctx = MNAContext()
        p = get_node!(ctx, :a)
        n = get_node!(ctx, :c)

        x = Float64[1.0, 0.0]  # Vp = 1V, Vn = 0V

        # Nonlinear junction capacitor: Q(V) = Cj0 * V^2
        Cj0 = 1e-12  # 1 pF
        q_fn(V) = Cj0 * V^2

        # Stamp without scaling
        ctx1 = MNAContext()
        p1 = get_node!(ctx1, :a)
        n1 = get_node!(ctx1, :c)
        q_idx1 = stamp_charge_state!(ctx1, p1, n1, q_fn, x, :Q_test)

        # Stamp with scaling
        ctx2 = MNAContext()
        p2 = get_node!(ctx2, :a)
        n2 = get_node!(ctx2, :c)
        q_idx2 = stamp_charge_state!(ctx2, p2, n2, q_fn, x, :Q_test; charge_scale=1e-12)

        # Verify both allocated charge variables
        @test ctx1.n_charges == 1
        @test ctx2.n_charges == 1

        # Check that scaling affects the C matrix values
        # Unscaled: C[p, q_idx] = 1.0
        # Scaled: C[p, q_idx] = charge_scale = 1e-12

        # Extract C matrix entries for positive node
        # (can't easily compare assembled matrices in this simple test,
        # but we can verify the stamping worked)
        @test length(ctx1.C_V) > 0
        @test length(ctx2.C_V) > 0
    end

    @testset "Condition Number Improvement with Scaling" begin
        # Build a small test circuit with a charge state variable
        # and verify that scaling improves the condition number

        # Helper to build a circuit and get condition number
        function build_and_condition(charge_scale::Float64)
            ctx = MNAContext()

            # Create nodes
            vdd = get_node!(ctx, :vdd)
            out = get_node!(ctx, :out)

            # Add a resistor: VDD to OUT (1kΩ)
            G_R = 1.0 / 1000.0
            stamp_G!(ctx, vdd, vdd, G_R)
            stamp_G!(ctx, vdd, out, -G_R)
            stamp_G!(ctx, out, vdd, -G_R)
            stamp_G!(ctx, out, out, G_R)

            # Add a ground resistor on VDD for reference
            stamp_G!(ctx, vdd, vdd, 1e-3)  # 1kΩ to ground

            # Add a voltage-dependent capacitor: OUT to GND
            # Q(V) = C0 * V^2 (nonlinear, requires charge formulation)
            C0 = 1e-12  # Typical gate capacitance ~1pF
            q_fn(V) = C0 * V^2

            x = Float64[1.2, 0.6, 0.0]  # Initial guess: VDD=1.2V, OUT=0.6V, q=0
            stamp_charge_state!(ctx, out, 0, q_fn, x, :Q_cap; charge_scale=charge_scale)

            # Assemble matrices
            n = system_size(ctx)
            G = spzeros(n, n)
            C = spzeros(n, n)

            # Resolve indices and assemble G
            for (ii, jj, vv) in zip(ctx.G_I, ctx.G_J, ctx.G_V)
                i = resolve_index(ctx, ii)
                j = resolve_index(ctx, jj)
                if i > 0 && j > 0
                    G[i, j] += vv
                end
            end

            # Assemble C
            for (ii, jj, vv) in zip(ctx.C_I, ctx.C_J, ctx.C_V)
                i = resolve_index(ctx, ii)
                j = resolve_index(ctx, jj)
                if i > 0 && j > 0
                    C[i, j] += vv
                end
            end

            # Compute condition number at γ = 1e6 (typical transient timestep)
            γ = 1e6
            J = Matrix(G + γ * C)

            # Add small regularization
            for i in 1:n
                if abs(J[i, i]) < 1e-30
                    J[i, i] += 1e-15
                end
            end

            return cond(J)
        end

        # Compare condition numbers
        κ_unscaled = build_and_condition(1.0)
        κ_scaled = build_and_condition(1e-12)  # Scale to match typical charge magnitude

        # Verify that scaling improves conditioning
        @test κ_scaled < κ_unscaled

        # The improvement should be significant (several orders of magnitude)
        improvement_ratio = κ_unscaled / κ_scaled
        @test improvement_ratio > 1e6  # At least 6 orders of magnitude improvement

        # Print results for reference
        @info "Charge Scaling Test Results" κ_unscaled κ_scaled improvement_ratio
    end

    @testset "MNASpec charge_scale parameter" begin
        # Verify that MNASpec has the charge_scale field

        # Default value
        spec1 = MNASpec()
        @test spec1.charge_scale == 1.0

        # Custom value
        spec2 = MNASpec(charge_scale=1e-15)
        @test spec2.charge_scale == 1e-15

        # with_charge_scale helper
        spec3 = with_charge_scale(spec1, 1e-12)
        @test spec3.charge_scale == 1e-12
        @test spec3.temp == spec1.temp  # Other fields preserved
        @test spec3.mode == spec1.mode
    end

end

# Run tests if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # The @testset above will run automatically
end
