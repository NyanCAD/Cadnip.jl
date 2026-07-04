#==============================================================================#
# Breakpoint Protocol Tests
#
# Tests for src/mna/breakpoints.jl: the Wave callable structs' breakpoints()
# edge lists, expand_breakpoints() periodic expansion/clipping/dedupe, and
# the _merge_tstops precedence helper used by tran!.
#==============================================================================#

using Test
using StaticArrays

using Cadnip
using Cadnip.MNA: BreakpointSpec, breakpoints, expand_breakpoints
using Cadnip.MNA: PWLWave, SinWave, PulseWave

@testset "Breakpoint Protocol" begin

    #==========================================================================#
    # breakpoints() per Wave type
    #==========================================================================#

    @testset "breakpoints(::PWLWave)" begin
        w = PWLWave(SVector(0.0, 1e-3, 2e-3), SVector(0.0, 1.0, 0.0))
        bp = breakpoints(w)
        @test bp.times == [0.0, 1e-3, 2e-3]
        @test bp.period == 0.0  # aperiodic

        # Dynamic-array fallback path
        wd = PWLWave([0.0, 1e-3, 2e-3], [0.0, 1.0, 0.0])
        bpd = breakpoints(wd)
        @test bpd.times == [0.0, 1e-3, 2e-3]
        @test bpd.period == 0.0
    end

    @testset "breakpoints(::SinWave)" begin
        # td == 0: no onset breakpoint (source is already smooth from t=0)
        w0 = SinWave(0.0, 1.0, 1e3)
        @test breakpoints(w0) === nothing

        # td > 0: onset is a slope discontinuity
        wtd = SinWave(0.0, 1.0, 1e3, 1e-3)
        bp = breakpoints(wtd)
        @test bp.times == [1e-3]
        @test bp.period == 0.0
    end

    @testset "breakpoints(::PulseWave) edge cases" begin
        # Normal case: 4 distinct edges (td, td+tr, td+tr+pw, td+tr+pw+tf), periodic
        w = PulseWave(0.0, 1.0, 1e-6, 1e-6, 1e-6, 1e-3, 2e-3)
        bp = breakpoints(w)
        @test bp.times == [1e-6, 2e-6, 1.002e-3, 1.003e-3]
        @test bp.period == 2e-3
        @test bp.count == -1  # unbounded repetition

        # tr=0/tf=0: instantaneous edges collapse to coincident times (still
        # emitted here; expand_breakpoints is what dedupes coincident points)
        wz = PulseWave(0.0, 1.0, 1e-6, 0.0, 0.0, 1e-3, 2e-3)
        bpz = breakpoints(wz)
        @test bpz.times == [1e-6, 1e-6, 1.001e-3, 1.001e-3]

        # per <= 0: single non-repeating pulse (aperiodic)
        wnp = PulseWave(0.0, 1.0, 1e-6, 1e-6, 1e-6, 1e-3, 0.0)
        bpnp = breakpoints(wnp)
        @test bpnp.period == 0.0
        @test bpnp.times == [1e-6, 2e-6, 1.002e-3, 1.003e-3]
    end

    @testset "breakpoints(::Any) default" begin
        @test breakpoints(nothing) === nothing
        @test breakpoints(t -> 5.0 * t) === nothing  # arbitrary user closure
        @test breakpoints(42) === nothing
    end

    #==========================================================================#
    # expand_breakpoints
    #==========================================================================#

    @testset "expand_breakpoints: aperiodic" begin
        specs = [BreakpointSpec([1e-6, 5e-6, 1e-5])]
        out = expand_breakpoints(specs, (0.0, 1e-5))
        @test out == [1e-6, 5e-6]  # endpoints of tspan excluded (open interval)
    end

    @testset "expand_breakpoints: periodic expansion" begin
        # PulseWave-style spec: edges at [1,2] us within each 10us period, count=-1
        specs = [BreakpointSpec([1e-6, 2e-6], 10e-6, -1)]
        out = expand_breakpoints(specs, (0.0, 35e-6))
        @test out ≈ [1e-6, 2e-6, 11e-6, 12e-6, 21e-6, 22e-6, 31e-6, 32e-6]
    end

    @testset "expand_breakpoints: tspan clipping" begin
        specs = [BreakpointSpec([1e-6, 2e-6], 10e-6, -1)]
        # Window starts mid-way through the periodic sequence
        out = expand_breakpoints(specs, (15e-6, 25e-6))
        @test out ≈ [21e-6, 22e-6]
    end

    @testset "expand_breakpoints: count-limited" begin
        specs = [BreakpointSpec([1e-6, 2e-6], 10e-6, 2)]  # only 2 repetitions (k=0,1)
        out = expand_breakpoints(specs, (0.0, 100e-6))
        @test out ≈ [1e-6, 2e-6, 11e-6, 12e-6]
    end

    @testset "expand_breakpoints: coincident-edge dedupe" begin
        # tr=0/tf=0 PulseWave produces duplicate times; also merge across two specs.
        # Tolerance is scaled to the ULP of the compared values (~1e-6 here), not
        # to the tspan width - a few ULP (~1e-22) is within tolerance (dedupes)
        # while 1e-15 (many orders of magnitude larger than eps(1e-6)) is not
        # (stays distinct). This is deliberately independent of tspan width: the
        # same offsets behave the same regardless of how long/short (0.0, 1e-5) is.
        specs = [
            BreakpointSpec([1e-6, 1e-6, 5e-6, 5e-6]),
            BreakpointSpec([1e-6 + 1e-22, 3e-6]),  # within a few ULP of 1e-6
        ]
        out = expand_breakpoints(specs, (0.0, 1e-5))
        @test out == [1e-6, 3e-6, 5e-6]

        # Outside tolerance: stays as two distinct points
        specs2 = [BreakpointSpec([1e-6, 1e-6 + 1e-15])]
        out2 = expand_breakpoints(specs2, (0.0, 1e-5))
        @test length(out2) == 2
    end

    @testset "expand_breakpoints: empty specs / no breakpoints" begin
        @test expand_breakpoints(BreakpointSpec[], (0.0, 1.0)) == Float64[]
        # spec entirely outside tspan
        specs = [BreakpointSpec([100.0])]
        @test expand_breakpoints(specs, (0.0, 1.0)) == Float64[]
    end

    @testset "expand_breakpoints: max_points warning (tiny period, long tspan)" begin
        # 1ns period over a 1ms window would be ~1e6 points; cap tightly and expect a warning
        specs = [BreakpointSpec([0.0], 1e-9, -1)]
        local out
        @test_logs (:warn, r"max_points") begin
            out = expand_breakpoints(specs, (0.0, 1e-3); max_points=100)
        end
        @test length(out) <= 100
    end

    @testset "expand_breakpoints: astronomically tiny period doesn't overflow k-range" begin
        # (t1-tmin)/period can vastly exceed typemax(Int) for a pathological
        # period; the k-range must be clamped in Float64 before converting to
        # Int rather than throwing InexactError.
        specs = [BreakpointSpec([0.0], 1e-300, -1)]
        local out
        @test_logs (:warn, r"max_points") begin
            out = expand_breakpoints(specs, (0.0, 1.0); max_points=100)
        end
        @test length(out) <= 100
    end

    @testset "expand_breakpoints: sort-before-truncate keeps globally earliest points" begin
        # specA has many points clustered LATE in the tspan; specB (processed
        # second) has a single, chronologically much earlier point. Truncating
        # mid-accumulation (in spec order) would drop specB's point entirely
        # since specA alone already exceeds max_points; truncating only after
        # a global sort keeps the earliest points regardless of spec order.
        specA = BreakpointSpec(collect(5.0:0.001:5.2))  # 201 points, late in tspan
        specB = BreakpointSpec([0.0001])                 # one point, very early
        local out
        @test_logs (:warn, r"max_points") begin
            out = expand_breakpoints([specA, specB], (0.0, 10.0); max_points=100)
        end
        @test 0.0001 in out
        @test issorted(out)
        @test length(out) == 100
    end

    #==========================================================================#
    # _merge_tstops precedence (internal helper used by tran!)
    #==========================================================================#

    @testset "_merge_tstops" begin
        merge_tstops = Cadnip.MNA._merge_tstops
        @test merge_tstops(nothing, nothing) === nothing
        @test merge_tstops([1.0, 2.0], nothing) == [1.0, 2.0]
        @test merge_tstops(nothing, [1.0, 2.0]) == [1.0, 2.0]
        @test merge_tstops([1.0, 3.0], [2.0, 3.0]) == [1.0, 2.0, 3.0]  # sorted, deduped
    end
end
