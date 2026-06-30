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
function load_golden(case, spec, grid, reltols)
    anap = joinpath(OUT, "analytic_$(case).csv")
    # Prefer the exact analytic solution when the case has one (no reference
    # floor); otherwise use the tight VACASK run (the mature simulator).
    if get(spec, "analytic", false) && isfile(anap)
        t, v = read_wave(anap)
        return interp_to(t, v, grid), "analytic (exact)"
    end
    refp = joinpath(OUT, "ref_$(case).csv")
    if isfile(refp)
        t, v = read_wave(refp)
        return interp_to(t, v, grid), "VACASK (tight)"
    end
    if isfile(anap)
        t, v = read_wave(anap)
        return interp_to(t, v, grid), "analytic"
    end
    for r in sort(reltols), s in ("Rodas5P", "FBDF", "IDA")
        p = joinpath(OUT, "cadnip_$(case)_$(s)_$(reltol_tag(r)).csv")
        if isfile(p)
            t, v = read_wave(p)
            return interp_to(t, v, grid), "Cadnip $s @ $(reltol_tag(r))"
        end
    end
    return nothing, ""
end

function process_case(case)
    spec = CFG["cases"][case]
    tspan = (Float64(spec["tspan"][1]), Float64(spec["tspan"][2]))
    n_grid = Int(CFG["n_grid"])
    grid = collect(range(tspan[1], tspan[2]; length=n_grid))
    reltols = Float64.(CFG["reltols"])

    golden, gsrc = load_golden(case, spec, grid, reltols)
    golden === nothing && (@warn "no golden for $case; skipping"; return nothing)
    println("$case: golden = $gsrc")

    # Cross-check the analytic solution against the tight VACASK run when both
    # exist, as an independent confirmation that the two references agree.
    anap = joinpath(OUT, "analytic_$(case).csv")
    refp = joinpath(OUT, "ref_$(case).csv")
    if isfile(anap) && isfile(refp)
        ta, va = read_wave(anap); tr, vr = read_wave(refp)
        @printf("  analytic vs VACASK-tight cross-check: rel-L2 = %.2e\n",
                rel_l2(interp_to(tr, vr, grid), interp_to(ta, va, grid)))
    end

    curves = Dict{String,Vector{Tuple{Float64,Float64}}}()  # label -> (err,time)
    table = Tuple{String,Float64,Float64,Float64}[]          # (sim,reltol,err,time)

    # Cadnip
    cpath = joinpath(OUT, "cadnip_$(case).csv")
    if isfile(cpath)
        rows, _ = read_table(cpath)
        for row in rows
            solver = row["solver"]; r = parse(Float64, row["reltol"])
            t = tryparse(Float64, get(row, "median_time_s", "NaN"))
            wp = joinpath(OUT, "cadnip_$(case)_$(solver)_$(reltol_tag(r)).csv")
            isfile(wp) || continue
            tw, vw = read_wave(wp)
            err = rel_l2(interp_to(tw, vw, grid), golden)
            label = "Cadnip $solver"
            if isfinite(err) && t !== nothing && isfinite(t)
                push!(get!(curves, label, Tuple{Float64,Float64}[]), (err, t))
            end
            push!(table, (label, r, err, something(t, NaN)))
        end
    end

    # VACASK
    vpath = joinpath(OUT, "vacask_$(case).csv")
    if isfile(vpath)
        rows, _ = read_table(vpath)
        for row in rows
            r = parse(Float64, row["reltol"]); t = tryparse(Float64, get(row, "time_s", "NaN"))
            wp = joinpath(OUT, "vacask_$(case)_$(reltol_tag(r)).csv")
            isfile(wp) || continue
            tw, vw = read_wave(wp)
            err = rel_l2(interp_to(tw, vw, grid), golden)
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
