using Cadnip
using Cadnip.MNA
using Test
using StaticArrays: SVector

# Import codegen-time helpers directly for unit tests.
const _tm_parse_file = Cadnip._tm_parse_file
const _tm_parse_control = Cadnip._tm_parse_control

# Set up fixture files at toplevel — `va"..."` must run at toplevel to avoid
# world-age errors when the defined type is used in the same scope.
const _TM_TESTDIR = mktempdir(; cleanup=true)
write(joinpath(_TM_TESTDIR, "tm_rt.tbl"), """
    # column 1: wl, column 2: g
    1.54  0.01
    1.55  0.02
    1.56  0.03
    """)

# cd so the relative "tm_rt.tbl" in the VA module resolves against _TM_TESTDIR
# when the va macro invokes codegen.
const _TM_OLD_CWD = pwd()
cd(_TM_TESTDIR)

va"""
module TMRoundTrip(p, n);
    inout p, n;
    electrical p, n;
    parameter real wl = 1.55;
    analog begin
        I(p, n) <+ $table_model(wl, "tm_rt.tbl", "1L;1");
    end
endmodule
"""

cd(_TM_OLD_CWD)

function _tm_build_at(wl_val)
    function tm_builder(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        p = MNA.get_node!(ctx, :p)
        MNA.stamp!(MNA.Resistor(10.0; name=:R1), ctx, p, 0)
        MNA.stamp!(TMRoundTrip(wl=wl_val), ctx, p, 0;
                   _mna_x_=x, _mna_t_=t, _mna_spec_=spec,
                   _mna_instance_=Symbol(""))
        return ctx
    end
    MNACircuit(tm_builder)
end

@testset "\$table_model support (LRM 9.21)" begin

@testset "pwl_at_time extrap parameter" begin
    ts = SVector{3,Float64}(1.0, 2.0, 3.0)
    ys = SVector{3,Float64}(10.0, 20.0, 30.0)

    # Interior: same result regardless of extrap.
    @test MNA.pwl_at_time(ts, ys, 1.5) ≈ 15.0
    @test MNA.pwl_at_time(ts, ys, 1.5, 0.0) ≈ 15.0
    @test MNA.pwl_at_time(ts, ys, 1.5, 1.0) ≈ 15.0
    @test MNA.pwl_at_time(ts, ys, 2.5, 1.0) ≈ 25.0

    # Sample points match exactly.
    @test MNA.pwl_at_time(ts, ys, 1.0) ≈ 10.0
    @test MNA.pwl_at_time(ts, ys, 3.0) ≈ 30.0

    # Off-the-ends with extrap=0.0 (default): hold — existing behavior.
    @test MNA.pwl_at_time(ts, ys, 0.5) ≈ 10.0
    @test MNA.pwl_at_time(ts, ys, 3.5) ≈ 30.0
    @test MNA.pwl_at_time(ts, ys, -100.0) ≈ 10.0
    @test MNA.pwl_at_time(ts, ys, 100.0) ≈ 30.0

    # Off-the-ends with extrap=1.0: linear extrapolation (slope 10).
    @test MNA.pwl_at_time(ts, ys, 0.5, 1.0) ≈ 5.0
    @test MNA.pwl_at_time(ts, ys, 3.5, 1.0) ≈ 35.0
    @test MNA.pwl_at_time(ts, ys, 0.0, 1.0) ≈ 0.0
    @test MNA.pwl_at_time(ts, ys, 4.0, 1.0) ≈ 40.0
end

@testset "_tm_parse_file: basic parsing" begin
    mktempdir() do dir
        path = joinpath(dir, "basic.tbl")
        write(path, """
            # Comment line
            1.0 10.0 100.0

            2.0 20.0 200.0
            3.0 30.0 300.0  # trailing comment
            """)
        tbl = _tm_parse_file(path)
        @test tbl.input == [1.0, 2.0, 3.0]
        @test tbl.outputs == [10.0 100.0; 20.0 200.0; 30.0 300.0]
        @test tbl.source_path == path
    end
end

@testset "_tm_parse_file: unsorted rows are sorted" begin
    mktempdir() do dir
        path = joinpath(dir, "unsorted.tbl")
        write(path, """
            3.0 30.0
            1.0 10.0
            2.0 20.0
            """)
        tbl = _tm_parse_file(path)
        @test tbl.input == [1.0, 2.0, 3.0]
        @test tbl.outputs == reshape([10.0, 20.0, 30.0], 3, 1)
    end
end

@testset "_tm_parse_file: error on inconsistent columns" begin
    mktempdir() do dir
        path = joinpath(dir, "ragged.tbl")
        write(path, "1.0 10.0\n2.0 20.0 extra\n")
        @test_throws ErrorException _tm_parse_file(path)
    end
end

@testset "_tm_parse_file: error on missing file" begin
    @test_throws ErrorException _tm_parse_file("/nonexistent/path/to/table.tbl")
end

@testset "_tm_parse_control" begin
    @test _tm_parse_control("1L;1") == 1
    @test _tm_parse_control("1L;42") == 42
    @test_throws ErrorException _tm_parse_control("1L")              # no semicolon
    @test_throws ErrorException _tm_parse_control("1L;1;extra")      # too many
    @test_throws ErrorException _tm_parse_control("2L;1")            # unsupported interp
    @test_throws ErrorException _tm_parse_control("1C;1")            # unsupported extrap
end

@testset "codegen round-trip: inline VA module" begin
    # Sample point: g(1.55) = 0.02 ⇒ |V(p)| = 0.2V through 10Ω load.
    sol = dc!(_tm_build_at(1.55))
    @test isapprox(abs(sol[:p]), 0.2; atol=1e-8)

    # Midpoint: g(1.545) = 0.015 (linear between 0.01 and 0.02).
    sol_mid = dc!(_tm_build_at(1.545))
    @test isapprox(abs(sol_mid[:p]), 0.15; atol=1e-8)

    # Below range with extrap=1.0: g(1.53) = 0.01 + slope*(-0.01) = 0.0
    sol_lo = dc!(_tm_build_at(1.53))
    @test isapprox(abs(sol_lo[:p]), 0.0; atol=1e-8)

    # Above range with extrap=1.0: g(1.57) = 0.03 + slope*(0.01) = 0.04
    sol_hi = dc!(_tm_build_at(1.57))
    @test isapprox(abs(sol_hi[:p]), 0.4; atol=1e-8)
end

@testset "codegen: missing file errors cleanly" begin
    # `va"..."` codegen calls `_tm_parse_file(abspath(filename))`. Since the
    # macro runs at toplevel of this file, the error propagates through the
    # macro and lands here. The `include("table_model.jl")` is itself inside
    # a `@testset ... include(...)` call — so we wrap in a try/catch and use
    # `@eval Main` to run the va macro at an outer toplevel scope where
    # errors propagate as expected.
    err = try
        Base.include_string(Main, """
            using Cadnip
            va\"\"\"
            module TMMissing(p, n);
                inout p, n;
                electrical p, n;
                parameter real wl = 1.55;
                analog begin
                    V(p, n) <+ \$table_model(wl, "/nonexistent_dir_xyz/does_not_exist.tbl", "1L;1");
                end
            endmodule
            \"\"\"
            """)
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test occursin("does_not_exist.tbl", msg)
end

@testset "codegen: bad control string errors cleanly" begin
    # Write a valid .tbl so only the control string is bad.
    mktempdir() do dir
        path = joinpath(dir, "tm_bad.tbl")
        write(path, "1.0 10.0\n2.0 20.0\n")
        err = try
            Base.include_string(Main, """
                using Cadnip
                va\"\"\"
                module TMBadCtrl(p, n);
                    inout p, n;
                    electrical p, n;
                    parameter real wl = 1.5;
                    analog begin
                        V(p, n) <+ \$table_model(wl, "$path", "2L;1");
                    end
                endmodule
                \"\"\"
                """)
            nothing
        catch e
            e
        end
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("1L", msg)
    end
end

end  # outer testset
