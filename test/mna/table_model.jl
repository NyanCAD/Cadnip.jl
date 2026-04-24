using Cadnip
using Cadnip.MNA
using Cadnip: SpectreFile
using Test
using ForwardDiff

# Codegen-time helpers exercised directly in a few unit tests.
const _tm_parse_file = Cadnip._tm_parse_file
const _tm_parse_control = Cadnip._tm_parse_control

const FIXTURE_DIR = joinpath(@__DIR__, "fixtures", "table_model")

# Load the test circuits via the file-first API. The `.scs` rigs each
# `ahdl_include` their sibling `.va`, which in turn references the sibling
# `.tbl`. `$table_model`'s file paths resolve against cwd at VA codegen time,
# so cd to the fixture dir for the duration of the load.
const _OLDCWD = pwd()
cd(FIXTURE_DIR)
Base.include(@__MODULE__, SpectreFile(joinpath(FIXTURE_DIR, "tm_1d.scs"); name=:tm_1d_circuit))
Base.include(@__MODULE__, SpectreFile(joinpath(FIXTURE_DIR, "tm_2d.scs"); name=:tm_2d_circuit))
cd(_OLDCWD)

# Inline VA module for the hoisting-dedup structural test, which needs to
# inspect the generated baremodule's symbol table.
cd(FIXTURE_DIR)
va"""
module TMDedup(p, n);
    inout p, n;
    electrical p, n;
    parameter real wl = 1.55;
    real g1, g2;
    analog begin
        g1 = $table_model(wl, "tm_1d.tbl", "1L;1");
        g2 = $table_model(wl, "tm_1d.tbl", "1L;1");
        I(p, n) <+ (g1 + g2) * V(p, n);
    end
endmodule
"""
cd(_OLDCWD)

@testset "\$table_model support (LRM 9.21)" begin

# ── Direct interpolator numerics: no solver, no stamping ────────────────────

@testset "1-D interpolator numerics" begin
    tbl = _tm_parse_file(joinpath(FIXTURE_DIR, "tm_1d.tbl"); n_inputs=1)
    itp = MNA.va_table_model_build(
        tbl.axes,
        collect(selectdim(tbl.outputs, ndims(tbl.outputs), 1)),
        ('1',), 'L',
    )
    @test itp(1.55) ≈ 0.02                            # sample
    @test itp(1.545) ≈ 0.015                          # interior
    @test isapprox(itp(1.53), 0.0; atol=1e-12)        # below-range extrap
    @test itp(1.57) ≈ 0.04                            # above-range extrap
    @test ForwardDiff.derivative(itp, 1.545) ≈ 1.0 atol=1e-10
end

@testset "2-D interpolator numerics" begin
    tbl = _tm_parse_file(joinpath(FIXTURE_DIR, "tm_2d.tbl"); n_inputs=2)
    itp = MNA.va_table_model_build(
        tbl.axes,
        collect(selectdim(tbl.outputs, ndims(tbl.outputs), 1)),
        ('1', '1'), 'L',
    )
    @test itp(1.55, 25.0) ≈ 2*1.55 + 3*25 + 5         # sample
    @test itp(1.545, 22.5) ≈ 2*1.545 + 3*22.5 + 5     # interior
    @test itp(1.53, 15.0) ≈ 2*1.53 + 3*15 + 5         # below-range extrap
    @test itp(1.57, 35.0) ≈ 2*1.57 + 3*35 + 5         # above-range extrap
end

# ── Codegen round-trip: file-first API, no manual stamping ──────────────────

@testset "codegen round-trip (1-D)" begin
    # V1 forces V(p) = 1; X1 contributes g amps into p via its I <+ stamp;
    # V1 sinks that current, so sol[:I_V1] = -g.
    c = MNACircuit(tm_1d_circuit; params=(wl_sweep=1.55,))
    @test isapprox(dc!(c)[:I_V1], -0.02; atol=1e-10)

    c2 = MNA.alter(c; var"params.wl_sweep"=1.545)
    @test isapprox(dc!(c2)[:I_V1], -0.015; atol=1e-10)
end

@testset "codegen round-trip (2-D)" begin
    c = MNACircuit(tm_2d_circuit; params=(wl_sweep=1.55, T_sweep=25.0))
    g = 2*1.55 + 3*25 + 5
    @test isapprox(dc!(c)[:I_V1], -g; atol=1e-8)
end

# ── Parser + control-string unit tests ──────────────────────────────────────

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
        @test size(tbl.outputs) == (3, 2)
        @test tbl.outputs[:, 1] == [10.0, 20.0, 30.0]
        @test tbl.outputs[:, 2] == [100.0, 200.0, 300.0]
        @test tbl.source_path == path
    end
end

@testset "_tm_parse_file: 1-D unsorted rows are sorted" begin
    mktempdir() do dir
        path = joinpath(dir, "unsorted.tbl")
        write(path, "3.0 30.0\n1.0 10.0\n2.0 20.0\n")
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
    tbl = _tm_parse_file(joinpath(FIXTURE_DIR, "tm_2d.tbl"); n_inputs=2)
    @test tbl.axes[1] == [1.54, 1.55, 1.56]
    @test tbl.axes[2] == [20.0, 25.0, 30.0]
    @test size(tbl.outputs) == (3, 3, 1)
    @test tbl.outputs[2, 2, 1] ≈ 2*1.55 + 3*25.0 + 5
end

@testset "_tm_parse_file: 2-D ragged grid errors" begin
    mktempdir() do dir
        path = joinpath(dir, "ragged.tbl")
        # (1.0, 10.0) missing — not a product grid
        write(path, "1.0 20.0 1.0\n2.0 10.0 2.0\n2.0 20.0 3.0\n")
        @test_throws ErrorException _tm_parse_file(path; n_inputs=2)
    end
end

@testset "_tm_parse_control: 1-D" begin
    @test _tm_parse_control("1L;1", 1) == (('1',), 'L', 1)
    @test _tm_parse_control("1L;42", 1) == (('1',), 'L', 42)
    @test _tm_parse_control("1C;1", 1) == (('1',), 'C', 1)
    @test _tm_parse_control("1E;1", 1) == (('1',), 'E', 1)
    @test _tm_parse_control("1;1", 1) == (('1',), 'L', 1)
    @test _tm_parse_control("D;1", 1) == (('D',), 'L', 1)
    @test_throws ErrorException _tm_parse_control("1L", 1)
    @test_throws ErrorException _tm_parse_control("1L;1;extra", 1)
    @test_throws ErrorException _tm_parse_control("2L;1", 1)
    @test_throws ErrorException _tm_parse_control("1X;1", 1)
    @test_throws ErrorException _tm_parse_control("1L,1L;1", 1)
end

@testset "_tm_parse_control: N-D" begin
    @test _tm_parse_control("1L,1L;1", 2) == (('1', '1'), 'L', 1)
    @test _tm_parse_control("1,D;2", 2) == (('1', 'D'), 'L', 2)
    @test _tm_parse_control("1L,1L,1L;3", 3) == (('1', '1', '1'), 'L', 3)
    @test_throws ErrorException _tm_parse_control("1L;1", 2)
    @test_throws ErrorException _tm_parse_control("1L,1C;1", 2)
end

# ── Error-path codegen tests ────────────────────────────────────────────────

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
    @test occursin("does_not_exist.tbl", sprint(showerror, err))
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

# ── Hoisting dedup structural test ──────────────────────────────────────────

@testset "hoisting dedup: two calls to same (file, col) → one const" begin
    # TMDedup defined at module top level (above) with two identical
    # $table_model calls. Both should hoist to the same const — count
    # _tm_itp_* names in the generated baremodule.
    itp_names = filter(
        n -> occursin("_tm_itp_", string(n)),
        names(Main.TMDedup_module; all=true),
    )
    @test length(itp_names) == 1
end

end  # outer testset
