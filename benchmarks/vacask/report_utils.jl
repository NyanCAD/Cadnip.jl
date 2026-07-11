#==============================================================================#
# Shared benchmark-result reporting: BenchmarkResult struct, VACASK reference
# TSV loader, and markdown table generation. Included by run_benchmarks.jl and
# by any standalone benchmark script (e.g. c6288/cedarsim/runme.jl) that needs
# to emit a comparison table in the same format.
#==============================================================================#

using Printf

# Results storage
struct BenchmarkResult
    name::String
    solver::String
    status::Symbol  # :success, :failed, :skipped
    median_time::Float64  # seconds
    min_time::Float64
    max_time::Float64
    memory::Float64  # MB
    allocs::Int
    timepoints::Int
    rejected::Int
    nr_iters::Int   # nonlinear (Newton) iterations; 0 = not reported
    error_msg::String
end

BenchmarkResult(name, solver, status, error_msg="") = BenchmarkResult(name, solver, status, NaN, NaN, NaN, NaN, 0, 0, 0, 0, error_msg)

# Label used for rows sourced from the real VACASK simulator (run_vacask.sh).
const VACASK_REF_SOLVER = "VACASK"

#==============================================================================#
# VACASK reference numbers
#
# run_vacask.sh runs the real VACASK binary via the upstream benchmark.py and
# writes a TSV (benchmark, time_s, timepoints, rejected, iterations). If the
# path is provided via $VACASK_REFERENCE_TSV we fold those rows into the same
# comparison tables for an apples-to-apples, same-machine comparison.
#==============================================================================#
function load_vacask_reference()
    refs = BenchmarkResult[]
    path = get(ENV, "VACASK_REFERENCE_TSV", "")
    (isempty(path) || !isfile(path)) && return refs
    for line in eachline(path)
        isempty(strip(line)) && continue
        cols = split(line, '\t')
        length(cols) < 5 && continue
        name = String(cols[1])
        t = tryparse(Float64, cols[2])
        t === nothing && continue
        tp = something(tryparse(Int, cols[3]), 0)
        rej = something(tryparse(Int, cols[4]), 0)
        iters = something(tryparse(Int, cols[5]), 0)
        push!(refs, BenchmarkResult(name, VACASK_REF_SOLVER, :success,
                                    t, t, t, NaN, 0, tp, rej, iters, ""))
    end
    return refs
end

function format_time(seconds::Float64)
    if isnan(seconds)
        return "-"
    elseif seconds < 1e-3
        return @sprintf("%.2f µs", seconds * 1e6)
    elseif seconds < 1
        return @sprintf("%.2f ms", seconds * 1e3)
    else
        return @sprintf("%.2f s", seconds)
    end
end

function format_memory(mb::Float64)
    if isnan(mb)
        return "-"
    elseif mb < 1
        return @sprintf("%.1f KB", mb * 1024)
    elseif mb < 1024
        return @sprintf("%.1f MB", mb)
    else
        return @sprintf("%.2f GB", mb / 1024)
    end
end

#==============================================================================#
# Generate Markdown Report
#==============================================================================#
function generate_markdown(results::Vector{BenchmarkResult}; title::String="VACASK Benchmark Results")
    io = IOBuffer()

    println(io, "# $title")
    println(io)
    println(io, "Benchmarks run on Julia $(VERSION)")
    println(io)
    if any(r -> r.solver == VACASK_REF_SOLVER, results)
        println(io, "Rows labelled **$(VACASK_REF_SOLVER)** are the real VACASK simulator measured on the same machine (see `run_vacask.sh`), for an apples-to-apples comparison.")
        println(io)
    end

    # Summary table with all solvers
    println(io, "## Summary")
    println(io)
    println(io, "| Benchmark | Solver | Status | Median Time | Timepoints | Rejected | NR iters | Iters/step | Memory |")
    println(io, "|-----------|--------|--------|-------------|------------|----------|----------|------------|--------|")

    for r in results
        status_emoji = r.status == :success ? "✅" : r.status == :skipped ? "⏭️" : "❌"
        rejected_str = r.rejected >= 0 ? string(r.rejected) : "-"
        iters_str = r.nr_iters > 0 ? string(r.nr_iters) : "-"
        iters_per_step = (r.nr_iters > 0 && r.timepoints > 0) ?
            @sprintf("%.2f", r.nr_iters / r.timepoints) : "-"
        println(io, "| $(r.name) | $(r.solver) | $(status_emoji) | $(format_time(r.median_time)) | $(r.timepoints > 0 ? r.timepoints : "-") | $(rejected_str) | $(iters_str) | $(iters_per_step) | $(format_memory(r.memory)) |")
    end
    println(io)

    # Detailed results grouped by benchmark
    println(io, "## Detailed Results")
    println(io)

    # Group results by benchmark name
    benchmarks = unique(r.name for r in results)
    for bench_name in benchmarks
        println(io, "### $(bench_name)")
        println(io)

        bench_results = filter(r -> r.name == bench_name, results)
        successful = filter(r -> r.status == :success, bench_results)

        if !isempty(successful)
            println(io, "| Solver | Median | Min | Max | Rejected | NR iters | Memory |")
            println(io, "|--------|--------|-----|-----|----------|----------|--------|")
            for r in successful
                iters_str = r.nr_iters > 0 ? string(r.nr_iters) : "-"
                println(io, "| $(r.solver) | $(format_time(r.median_time)) | $(format_time(r.min_time)) | $(format_time(r.max_time)) | $(r.rejected) | $(iters_str) | $(format_memory(r.memory)) |")
            end
            println(io)

            # Show fastest Cadnip solver (excluding the VACASK reference row)
            cadnip_successful = filter(r -> r.solver != VACASK_REF_SOLVER, successful)
            if !isempty(cadnip_successful)
                fastest = argmin(r -> r.median_time, cadnip_successful)
                note = ""
                ref = filter(r -> r.solver == VACASK_REF_SOLVER, successful)
                if !isempty(ref)
                    ratio = ref[1].median_time / fastest.median_time
                    note = ratio >= 1 ? " — $(round(ratio, digits=2))× faster than VACASK" :
                                        " — $(round(1/ratio, digits=2))× slower than VACASK"
                end
                println(io, "> 🏆 Fastest Cadnip solver: **$(fastest.solver)** ($(format_time(fastest.median_time)))$(note)")
                println(io)
            end
        end

        # Show failures
        failed = filter(r -> r.status == :failed, bench_results)
        for r in failed
            println(io, "> ❌ $(r.solver) failed: $(r.error_msg)")
        end
        println(io)
    end

    return String(take!(io))
end
