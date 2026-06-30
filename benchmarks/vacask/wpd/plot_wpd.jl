#!/usr/bin/env julia
#==============================================================================#
# Work-Precision Diagram benchmark - analysis + plotting (pure Julia)
#
# Reads the Cadnip and VACASK waveforms/summaries produced by cadnip_wpd.jl and
# vacask_wpd.jl, computes the relative-L2 error of every run against the golden
# reference, and writes one log-log work-precision diagram per circuit plus a
# markdown summary.
#
# Golden reference per case (in order of preference):
#   out/ref_<case>.csv       VACASK at tight tolerance (the mature simulator)
#   out/analytic_<case>.csv  closed-form solution (linear cases)
#   tightest successful Cadnip run (last resort)
#
# Usage:
#   julia --project=benchmarks benchmarks/vacask/wpd/plot_wpd.jl [case ...]
#==============================================================================#

ENV["GKSwstype"] = "100"  # headless GR: render straight to files

using Pkg
try
    @eval import JSON
catch
    Pkg.add("JSON"); @eval import JSON
end
try
    @eval using Plots
catch
    Pkg.add("Plots"); @eval using Plots
end
using Printf
using Statistics

const HERE = @__DIR__
const OUT = joinpath(HERE, "out")
const PLOTS = joinpath(HERE, "plots")
mkpath(PLOTS)
const CFG = JSON.parsefile(joinpath(HERE, "config.json"))

#------------------------------------------------------------------------------#
# IO + math helpers
#------------------------------------------------------------------------------#
"Read a t,v waveform CSV; returns (t::Vector, v::Vector)."
function read_wave(path)
    t = Float64[]; v = Float64[]
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue               # header t,v
        isempty(strip(line)) && continue
        a, b = split(line, ',')
        push!(t, parse(Float64, a)); push!(v, parse(Float64, b))
    end
    return t, v
end

function interp_to(xs, ys, q)
    out = Vector{Float64}(undef, length(q)); n = length(xs)
    for (k, x) in enumerate(q)
        if x <= xs[1]
            out[k] = ys[1]
        elseif x >= xs[n]
            out[k] = ys[n]
        else
            j = searchsortedlast(xs, x)
            x1, x2 = xs[j], xs[j+1]; y1, y2 = ys[j], ys[j+1]
            out[k] = x2 == x1 ? y1 : y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        end
    end
    return out
end

rel_l2(v, ref) = sqrt(mean((v .- ref) .^ 2)) / sqrt(mean(ref .^ 2))
reltol_tag(r) = @sprintf("%.0e", r)

"Read a header-and-rows CSV into a Vector of NamedTuple-ish Dicts (String values)."
function read_table(path)
    rows = Dict{String,String}[]
    lines = collect(eachline(path))
    isempty(lines) && return rows, String[]
    header = String.(split(lines[1], ','))
    for line in lines[2:end]
        isempty(strip(line)) && continue
        cells = String.(split(line, ','))
        length(cells) < length(header) && continue
        push!(rows, Dict(header[i] => cells[i] for i in eachindex(header)))
    end
    return rows, header
end

#------------------------------------------------------------------------------#
# Per-case processing
#------------------------------------------------------------------------------#
# Returns the DENSE reference as raw (t, v) arrays plus a label. The reference is
# kept dense (analytic: 200k pts; VACASK-tight: ~100k pts) so it can be
# interpolated accurately onto any solver's *own* timepoints - that way a
# high-order solver taking large steps is not penalised by interpolation of its
# own (sparse) output.
function load_golden(case, spec, reltols)
    anap = joinpath(OUT, "analytic_$(case).csv")
    if get(spec, "analytic", false) && isfile(anap)
        return read_wave(anap)..., "analytic (exact)"
    end
    refp = joinpath(OUT, "ref_$(case).csv")
    if isfile(refp)
        return read_wave(refp)..., "VACASK (tight)"
    end
    if isfile(anap)
        return read_wave(anap)..., "analytic"
    end
    for r in sort(reltols), s in ("Rodas5P", "FBDF", "IDA")
        p = joinpath(OUT, "cadnip_$(case)_$(s)_$(reltol_tag(r)).csv")
        isfile(p) && return read_wave(p)..., "Cadnip $s @ $(reltol_tag(r))"
    end
    return nothing, nothing, ""
end

"Relative-L2 error of run waveform (rt,rv) vs the dense reference (gt,gv), evaluated at the run's OWN points."
function run_error(rt, rv, gt, gv)
    ref = interp_to(gt, gv, rt)
    return sqrt(mean((rv .- ref) .^ 2)) / sqrt(mean(ref .^ 2))
end

function process_case(case)
    spec = CFG["cases"][case]
    reltols = Float64.(CFG["reltols"])

    gt, gv, gsrc = load_golden(case, spec, reltols)
    gt === nothing && (@warn "no golden for $case; skipping"; return nothing)
    println("$case: golden = $gsrc")

    # Cross-check the analytic solution against the tight VACASK run when both
    # exist, as an independent confirmation that the two references agree.
    anap = joinpath(OUT, "analytic_$(case).csv")
    refp = joinpath(OUT, "ref_$(case).csv")
    if isfile(anap) && isfile(refp)
        ta, va = read_wave(anap); tr, vr = read_wave(refp)
        @printf("  analytic vs VACASK-tight cross-check: rel-L2 = %.2e\n",
                run_error(tr, vr, ta, va))
    end

    curves = Dict{String,Vector{Tuple{Float64,Float64}}}()  # label -> (err,time)
    table = Tuple{String,Float64,Float64,Float64}[]          # (sim,reltol,err,time)

    # Each run's error is measured at its OWN timepoints vs the dense reference.
    cpath = joinpath(OUT, "cadnip_$(case).csv")
    if isfile(cpath)
        rows, _ = read_table(cpath)
        for row in rows
            solver = row["solver"]; r = parse(Float64, row["reltol"])
            t = tryparse(Float64, get(row, "median_time_s", "NaN"))
            wp = joinpath(OUT, "cadnip_$(case)_$(solver)_$(reltol_tag(r)).csv")
            isfile(wp) || continue
            tw, vw = read_wave(wp)
            err = run_error(tw, vw, gt, gv)
            label = "Cadnip $solver"
            if isfinite(err) && t !== nothing && isfinite(t)
                push!(get!(curves, label, Tuple{Float64,Float64}[]), (err, t))
            end
            push!(table, (label, r, err, something(t, NaN)))
        end
    end

    vpath = joinpath(OUT, "vacask_$(case).csv")
    if isfile(vpath)
        rows, _ = read_table(vpath)
        for row in rows
            r = parse(Float64, row["reltol"]); t = tryparse(Float64, get(row, "time_s", "NaN"))
            wp = joinpath(OUT, "vacask_$(case)_$(reltol_tag(r)).csv")
            isfile(wp) || continue
            tw, vw = read_wave(wp)
            err = run_error(tw, vw, gt, gv)
            if isfinite(err) && t !== nothing && isfinite(t)
                push!(get!(curves, "VACASK", Tuple{Float64,Float64}[]), (err, t))
            end
            push!(table, ("VACASK", r, err, something(t, NaN)))
        end
    end

    plot_case(case, spec["title"], curves)
    return case, gsrc, table
end

function plot_case(case, title, curves)
    plt = plot(; xscale=:log10, yscale=:log10, xlabel="relative L2 error vs golden",
               ylabel="runtime [s]", title="Work-precision: $title", legend=:topright,
               minorgrid=true)
    for label in sort(collect(keys(curves)))
        pts = sort(curves[label])             # by error
        isempty(pts) && continue
        errs = first.(pts); times = last.(pts)
        if label == "VACASK"
            plot!(plt, errs, times; label=label, marker=:square, linestyle=:dash, color=:black, lw=2)
        else
            plot!(plt, errs, times; label=label, marker=:circle, lw=1.8)
        end
    end
    for ext in ("png", "svg")
        savefig(plt, joinpath(PLOTS, "$(case).$(ext)"))
    end
    println("  wrote $(joinpath(PLOTS, case)).png / .svg")
end

function write_markdown(results)
    io = IOBuffer()
    println(io, "# Work-Precision Diagram Results\n")
    println(io, "Error = relative L2 of the output node vs the golden reference, sampled on a")
    println(io, "common time grid. Each simulator runs *adaptively* across a tolerance sweep")
    println(io, "(no forced timestep), so the diagram shows accuracy-per-unit-runtime.\n")
    for (case, gsrc, table) in results
        spec = CFG["cases"][case]
        println(io, "## $(spec["title"])\n")
        println(io, "Golden reference: **$gsrc**. Plot: `plots/$(case).png`.\n")
        println(io, "| Simulator | reltol | rel-L2 error | runtime (s) |")
        println(io, "|-----------|--------|--------------|-------------|")
        for (sim, r, err, t) in table
            es = isfinite(err) ? @sprintf("%.2e", err) : "-"
            ts = isfinite(t) ? @sprintf("%.3f", t) : "-"
            println(io, "| $sim | $(@sprintf("%.0e", r)) | $es | $ts |")
        end
        println(io)
    end
    path = joinpath(OUT, "wpd_results.md")
    write(path, String(take!(io)))
    println("\nMarkdown summary: $path")
end

function main()
    cases = isempty(ARGS) ? collect(keys(CFG["cases"])) : ARGS
    results = []
    for case in cases
        haskey(CFG["cases"], case) || (@warn "unknown case $case"; continue)
        res = process_case(String(case))
        res !== nothing && push!(results, res)
    end
    isempty(results) || write_markdown(results)
end

main()
