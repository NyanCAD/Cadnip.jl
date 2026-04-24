using Cadnip
using Cadnip.MNA
using Test
using ForwardDiff

# Import codegen-time helpers directly for unit tests.
const _tm_parse_file = Cadnip._tm_parse_file
const _tm_parse_control = Cadnip._tm_parse_control

# Set up 1-D fixture at toplevel — `va"..."` must run at toplevel to avoid
# world-age errors when the defined type is used in the same scope.
const _TM_TESTDIR = mktempdir(; cleanup=true)
write(joinpath(_TM_TESTDIR, "tm_rt.tbl"), """
    # column 1: wl, column 2: g
    1.54  0.01
    1.55  0.02
    1.56  0.03
    """)

# 2-D fixture: y = 2*wl + 3*T + 5 on a 3x3 grid.
# Rows ordered as wl outer, T inner.
const _TM_2D_PATH = joinpath(_TM_TESTDIR, "tm_2d.tbl")
open(_TM_2D_PATH; write=true) do io
    for wl in (1.54, 1.55, 1.56), T in (20.0, 25.0, 30.0)
        println(io, "$wl  $T  $(2wl + 3T + 5)")
    end
end

# 3-D fixture: y = 10*a + 100*b + 1000*c on a 3x3x3 grid.
const _TM_3D_PATH = joinpath(_TM_TESTDIR, "tm_3d.tbl")
open(_TM_3D_PATH; write=true) do io
    for a in (1.0, 2.0, 3.0), b in (10.0, 20.0, 30.0), c in (100.0, 200.0, 300.0)
        println(io, "$a  $b  $c  $(10a + 100b + 1000c)")
    end
end

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

va"""
module TM2D(p, n);
    inout p, n;
    electrical p, n;
    parameter real wl = 1.55;
    parameter real T = 25.0;
    analog begin
        I(p, n) <+ $table_model(wl, T, "tm_2d.tbl", "1L,1L;1");
    end
endmodule
"""

# Two $table_model calls against the same (file, col) — tests hoisting dedup.
va"""
module TMDedup(p, n);
    inout p, n;
    electrical p, n;
    parameter real wl = 1.55;
    real g1;
    real g2;
    analog begin
        g1 = $table_model(wl, "tm_rt.tbl", "1L;1");
        g2 = $table_model(wl, "tm_rt.tbl", "1L;1");
        I(p, n) <+ (g1 + g2) * V(p, n);
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

function _tm_build_2d(wl_val, T_val)
    function tm_builder(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        p = MNA.get_node!(ctx, :p)
        MNA.stamp!(MNA.Resistor(10.0; name=:R1), ctx, p, 0)
        MNA.stamp!(TM2D(wl=wl_val, T=T_val), ctx, p, 0;
                   _mna_x_=x, _mna_t_=t, _mna_spec_=spec,
                   _mna_instance_=Symbol(""))
        return ctx
    end
    MNACircuit(tm_builder)
end

@testset "\$table_model support (LRM 9.21)" begin

@testset "_tm_parse_file: 1-D basic parsing" begin
    mktempdir() do dir
        path = joinpath(dir, "basic.tbl")
        write(path, """
            # Comment line
            1.0 10.0 100.0

            2.0 20.0 200.0
            3.0 30.0 300.0  # trailing comment
            """)
        tbl = _tm_parse_file(path; n_inputs=1)
        @test tbl.axes == ([1.0, 2.0, 3.0],)
        # outputs is (3, 2) — axis_1 × n_deps
        @test size(tbl.outputs) == (3, 2)
        @test tbl.outputs[:, 1] == [10.0, 20.0, 30.0]
        @test tbl.outputs[:, 2] == [100.0, 200.0, 300.0]
        @test tbl.source_path == path
    end
end

@testset "_tm_parse_file: 1-D unsorted rows are sorted" begin
    mktempdir() do dir
        path = joinpath(dir, "unsorted.tbl")
        write(path, """
            3.0 30.0
            1.0 10.0
            2.0 20.0
            """)
        tbl = _tm_parse_file(path; n_inputs=1)
        @test tbl.axes == ([1.0, 2.0, 3.0],)
        @test tbl.outputs[:, 1] == [10.0, 20.0, 30.0]
    end
end

@testset "_tm_parse_file: error on inconsistent columns" begin
    mktempdir() do dir
        path = joinpath(dir, "ragged.tbl")
        write(path, "1.0 10.0\n2.0 20.0 extra\n")
        @test_throws ErrorException _tm_parse_file(path; n_inputs=1)
    end
end

@testset "_tm_parse_file: error on missing file" begin
    @test_throws ErrorException _tm_parse_file("/nonexistent/path/to/table.tbl"; n_inputs=1)
end

@testset "_tm_parse_file: 2-D regular grid" begin
    tbl = _tm_parse_file(_TM_2D_PATH; n_inputs=2)
    @test tbl.axes[1] == [1.54, 1.55, 1.56]
    @test tbl.axes[2] == [20.0, 25.0, 30.0]
    @test size(tbl.outputs) == (3, 3, 1)
    @test tbl.outputs[2, 2, 1] ≈ 2*1.55 + 3*25.0 + 5
end

@testset "_tm_parse_file: 2-D ragged grid errors" begin
    mktempdir() do dir
        path = joinpath(dir, "ragged.tbl")
        # (1.0, 10.0) missing — not a product grid
        write(path, """
            1.0 20.0 1.0
            2.0 10.0 2.0
            2.0 20.0 3.0
            """)
        @test_throws ErrorException _tm_parse_file(path; n_inputs=2)
    end
end

@testset "_tm_parse_control: 1-D" begin
    # Returns (modes, extrap, col).
    @test _tm_parse_control("1L;1", 1) == (('1',), 'L', 1)
    @test _tm_parse_control("1L;42", 1) == (('1',), 'L', 42)
    @test _tm_parse_control("1C;1", 1) == (('1',), 'C', 1)   # constant extrap
    @test _tm_parse_control("1E;1", 1) == (('1',), 'E', 1)   # error extrap
    @test _tm_parse_control("1;1", 1) == (('1',), 'L', 1)    # default extrap = L
    @test _tm_parse_control("D;1", 1) == (('D',), 'L', 1)    # discrete interp
    @test_throws ErrorException _tm_parse_control("1L", 1)              # no semicolon
    @test_throws ErrorException _tm_parse_control("1L;1;extra", 1)      # too many
    @test_throws ErrorException _tm_parse_control("2L;1", 1)            # unsupported interp
    @test_throws ErrorException _tm_parse_control("1X;1", 1)            # unsupported extrap
    @test_throws ErrorException _tm_parse_control("1L,1L;1", 1)         # dim mismatch
end

@testset "_tm_parse_control: N-D" begin
    @test _tm_parse_control("1L,1L;1", 2) == (('1', '1'), 'L', 1)
    @test _tm_parse_control("1,D;2", 2) == (('1', 'D'), 'L', 2)         # mixed interp
    @test _tm_parse_control("1L,1L,1L;3", 3) == (('1', '1', '1'), 'L', 3)
    @test_throws ErrorException _tm_parse_control("1L;1", 2)            # dim mismatch
    @test_throws ErrorException _tm_parse_control("1L,1C;1", 2)         # non-uniform extrap
end

@testset "codegen round-trip: 1-D VA module" begin
    # Sample point: g(1.55) = 0.02 ⇒ |V(p)| = 0.2V through 10Ω load.
    sol = dc!(_tm_build_at(1.55))
    @test isapprox(abs(sol[:p]), 0.2; atol=1e-8)

    # Midpoint: g(1.545) = 0.015 (linear between 0.01 and 0.02).
    sol_mid = dc!(_tm_build_at(1.545))
    @test isapprox(abs(sol_mid[:p]), 0.15; atol=1e-8)

    # Below range with "L" extrap: g(1.53) = 0.01 + slope*(-0.01) = 0.0
    sol_lo = dc!(_tm_build_at(1.53))
    @test isapprox(abs(sol_lo[:p]), 0.0; atol=1e-8)

    # Above range with "L" extrap: g(1.57) = 0.03 + slope*(0.01) = 0.04
    sol_hi = dc!(_tm_build_at(1.57))
    @test isapprox(abs(sol_hi[:p]), 0.4; atol=1e-8)
end

@testset "codegen round-trip: 2-D VA module" begin
    # y = 2*wl + 3*T + 5; interp is exact on linear data.
    # At (wl=1.55, T=25.0): g = 2*1.55 + 3*25 + 5 = 83.1 → |V(p)| = 83.1 * 10
    sol = dc!(_tm_build_2d(1.55, 25.0))
    expected = 2*1.55 + 3*25.0 + 5
    @test isapprox(abs(sol[:p]), expected * 10; atol=1e-6)

    # Interior midpoint (wl=1.545, T=22.5):
    sol_mid = dc!(_tm_build_2d(1.545, 22.5))
    expected_mid = 2*1.545 + 3*22.5 + 5
    @test isapprox(abs(sol_mid[:p]), expected_mid * 10; atol=1e-6)

    # Linear extrap: below both axes
    sol_lo = dc!(_tm_build_2d(1.53, 15.0))
    expected_lo = 2*1.53 + 3*15.0 + 5
    @test isapprox(abs(sol_lo[:p]), expected_lo * 10; atol=1e-6)

    # Linear extrap: above both axes
    sol_hi = dc!(_tm_build_2d(1.57, 35.0))
    expected_hi = 2*1.57 + 3*35.0 + 5
    @test isapprox(abs(sol_hi[:p]), expected_hi * 10; atol=1e-6)
end

@testset "codegen: missing file errors cleanly" begin
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
        @test occursin("interp", msg) || occursin("2L", msg) || occursin("'2'", msg)
    end
end

@testset "ForwardDiff sensitivity through 1-D interpolator" begin
    # Compile a small 1-D table and build the interpolator directly (no codegen).
    tbl = _tm_parse_file(_TM_TESTDIR * "/tm_rt.tbl"; n_inputs=1)
    itp = MNA.va_table_model_build(
        tbl.axes,
        collect(selectdim(tbl.outputs, ndims(tbl.outputs), 1)),
        ('1',),
        'L',
    )
    # Analytic slope over [1.54, 1.55] is (0.02 - 0.01) / 0.01 = 1.0
    @test ForwardDiff.derivative(itp, 1.545) ≈ 1.0 atol=1e-10
end

@testset "hoisting dedup: two calls to same (file, col) → one const" begin
    # TMDedup module was defined at toplevel with two identical
    # $table_model calls. Both should hoist to the same const — count
    # _tm_itp_* names in the generated baremodule.
    itp_names = filter(
        n -> occursin("_tm_itp_", string(n)),
        names(Main.TMDedup_module; all=true),
    )
    @test length(itp_names) == 1
end

end  # outer testset
