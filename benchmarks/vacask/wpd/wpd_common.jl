# Shared helpers for the work-precision-diagram benchmark (run_wpd.jl).
#
# Pure, lightweight utilities: config loading, CSV/raw IO, interpolation, the
# error metric, and VACASK binary discovery. No heavy deps (Cadnip / UnicodePlots
# are loaded by run_wpd.jl itself).

using Printf

# JSON is declared in benchmarks/Project.toml; add on first use as a fallback.
try
    @eval import JSON
catch
    import Pkg; Pkg.add("JSON"); @eval import JSON
end

const HERE = @__DIR__
const VACASK_DIR = normpath(joinpath(HERE, ".."))   # benchmarks/vacask
const OUT = joinpath(HERE, "out")

const CFG = JSON.parsefile(joinpath(HERE, "config.json"))

# Number of VACASK timed repetitions; the minimum runtime is reported.
const VACASK_REPEATS = parse(Int, get(ENV, "WPD_VACASK_REPEATS", "3"))

#------------------------------------------------------------------------------#
# Small numeric / IO helpers
#------------------------------------------------------------------------------#
reltol_tag(r) = @sprintf("%.0e", r)   # 1.0e-6 -> "1e-06"

"Linear interpolation of (xs,ys) onto query points q (xs ascending)."
function interp_to(xs::AbstractVector, ys::AbstractVector, q::AbstractVector)
    out = Vector{Float64}(undef, length(q))
    n = length(xs)
    for (k, x) in enumerate(q)
        if x <= xs[1]
            out[k] = ys[1]
        elseif x >= xs[n]
            out[k] = ys[n]
        else
            j = searchsortedlast(xs, x)
            x1, x2 = xs[j], xs[j+1]
            y1, y2 = ys[j], ys[j+1]
            out[k] = x2 == x1 ? y1 : y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        end
    end
    return out
end

"Relative-L2 error of run waveform (rt,rv) vs the dense reference (gt,gv), at the run's own points."
run_error(rt, rv, gt, gv) = (ref = interp_to(gt, gv, rt);
                             sqrt(sum(abs2, rv .- ref) / length(ref)) / sqrt(sum(abs2, ref) / length(ref)))

function write_wave(path, t, v)
    open(path, "w") do io
        println(io, "t,v")
        @inbounds for i in eachindex(t)
            @printf(io, "%.17g,%.17g\n", t[i], v[i])
        end
    end
end

"Read a t,v waveform CSV; returns (t::Vector, v::Vector)."
function read_wave(path)
    t = Float64[]; v = Float64[]
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue                # header t,v
        isempty(strip(line)) && continue
        a, b = split(line, ',')
        push!(t, parse(Float64, a)); push!(v, parse(Float64, b))
    end
    return t, v
end

"Read a header+rows CSV into (rows::Vector{Dict{String,String}}, header)."
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
# SPICE3 binary raw reader (the format VACASK writes; see lib/vacask/python/rawfile.py)
#------------------------------------------------------------------------------#
"Return (names::Vector{String}, M) where M[var, point] is the data matrix."
function read_rawfile(path::AbstractString)
    io = open(path)
    nvars = 0; npoints = 0; iscomplex = false
    names = String[]
    while true
        eof(io) && error("unexpected EOF reading raw header of $path")
        line = readline(io)
        if startswith(line, "Variables:")
            for _ in 1:nvars
                push!(names, String(split(strip(readline(io)))[2]))
            end
        elseif startswith(line, "Binary:")
            break
        else
            kv = split(line, ":"; limit=2)
            if length(kv) == 2
                key = lowercase(strip(kv[1])); val = strip(kv[2])
                key == "no. variables" && (nvars = parse(Int, val))
                key == "no. points" && (npoints = parse(Int, val))
                key == "flags" && (iscomplex = occursin("complex", lowercase(val)))
            end
        end
    end
    iscomplex && error("complex raw files not supported (transient should be real)")
    raw = Vector{Float64}(undef, npoints * nvars)
    read!(io, raw)
    close(io)
    return names, reshape(raw, nvars, npoints)   # column-major: M[var, point]
end

#------------------------------------------------------------------------------#
# VACASK binary discovery (mirrors fetch_vacask.sh / run_vacask.sh cache layout)
#------------------------------------------------------------------------------#
"Return (binary_path, lib_dir) or (nothing, nothing) if VACASK is not available."
function locate_vacask()
    cmd = get(ENV, "VACASK_COMMAND", "")
    if !isempty(cmd) && isfile(cmd)
        return cmd, joinpath(dirname(cmd), "lib")
    end
    cache = get(ENV, "CACHE_DIR", joinpath(homedir(), ".cache", "cadnip-vacask"))
    cand = joinpath(cache, "vacask", "simulator", "vacask")
    isfile(cand) && return cand, joinpath(dirname(cand), "lib")
    return nothing, nothing
end
