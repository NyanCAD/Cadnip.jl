#!/usr/bin/env julia
#==============================================================================#
# Work-Precision Diagram benchmark - VACASK side (pure Julia)
#
# Shells out to the real VACASK binary across the same tolerance sweep used by
# cadnip_wpd.jl, reads VACASK's SPICE3 binary raw output, and writes the output
# waveform per (case, reltol) plus a tight-tolerance golden reference. No Python
# / InSpice dependency: VACASK is just a subprocess and its raw file is parsed
# directly.
#
# Outputs (under wpd/out/):
#   vacask_<case>_<reltol>.csv   columns: t,v   (the waveform, VACASK's own grid)
#   vacask_<case>.csv            summary per run (reltol, time, timepoints)
#   ref_<case>.csv               t,v on the common grid - the golden reference
#
# The VACASK binary is found via $VACASK_COMMAND or the cache populated by
# ../run_vacask.sh (~/.cache/cadnip-vacask/...). Run that script once first.
#
# Usage:
#   julia benchmarks/vacask/wpd/vacask_wpd.jl [case ...]
#==============================================================================#

using Printf

const HERE = @__DIR__
const VACASK_DIR = normpath(joinpath(HERE, ".."))
const OUT = joinpath(HERE, "out")
mkpath(OUT)

# Minimal JSON-free config read: reuse Julia's JSON if present, else a tiny parse.
function load_config()
    path = joinpath(HERE, "config.json")
    try
        @eval import JSON
        return Base.invokelatest(JSON.parsefile, path)
    catch
        error("JSON package required; run cadnip_wpd.jl first (it adds JSON), or `]add JSON`.")
    end
end
const CFG = load_config()

const VACASK_REPEATS = parse(Int, get(ENV, "WPD_VACASK_REPEATS", "3"))

# VACASK defaults to trapezoidal (2nd order); use its variable-order Gear/BDF so
# it is benchmarked at its best rather than limited to 2nd order.
const VTRAN_METHOD = String(get(CFG, "vacask_tran_method", "gear"))
const VTRAN_MAXORD = Int(get(CFG, "vacask_tran_maxord", 5))

#------------------------------------------------------------------------------#
# Locate the VACASK binary (mirrors run_vacask.sh's cache layout)
#------------------------------------------------------------------------------#
function locate_vacask()
    cmd = get(ENV, "VACASK_COMMAND", "")
    if !isempty(cmd) && isfile(cmd)
        libdir = joinpath(dirname(cmd), "lib")
        return cmd, libdir
    end
    cache = get(ENV, "CACHE_DIR", joinpath(homedir(), ".cache", "cadnip-vacask"))
    cand = joinpath(cache, "vacask", "simulator", "vacask")
    if isfile(cand)
        return cand, joinpath(dirname(cand), "lib")
    end
    return nothing, nothing
end
const VACASK_BIN, VACASK_LIB = locate_vacask()

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
                toks = split(strip(readline(io)))
                push!(names, String(toks[2]))
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
    # File order is point-major (all vars of point 0, then point 1, ...), so a
    # column-major reshape with nvars first gives M[var, point].
    M = reshape(raw, nvars, npoints)
    return names, M
end

"Linear interpolation of (xs,ys) onto sorted query points q (both ascending)."
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

#------------------------------------------------------------------------------#
# VACASK netlists (match the cedarsim/runme.sp circuits Cadnip runs)
#------------------------------------------------------------------------------#
function sim_body(case::String)
    if case == "graetz"
        return """
        Graetz bridge full-wave rectifier
        load "spice/resistor.osdi"
        load "spice/capacitor.osdi"
        load "spice/sn/diode.osdi"
        model r sp_resistor
        model c sp_capacitor
        model vsource vsource
        model dmod sp_diode ( is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45 )
        vs (inp inn) vsource dc=0 type="sine" sinedc=0.0 ampl=20 freq=50.0
        d1 (inp outp) dmod
        d2 (outn inp) dmod
        d3 (inn outp) dmod
        d4 (outn inn) dmod
        cl (outp outn) c c=100u
        rl (outp outn) r r=1k
        rgnd1 (inn 0) r r=1meg
        rgnd2 (outn 0) r r=1meg
        """
    elseif case == "rc"
        return """
        RC circuit excited by a pulse train
        load "spice/resistor.osdi"
        load "spice/capacitor.osdi"
        model r sp_resistor
        model c sp_capacitor
        model vsource vsource
        vs (1 0) vsource dc=0 type="pulse" val0=0 val1=1 delay=1u rise=1u fall=1u width=1m period=2m
        r1 (1 2) r r=1k
        c1 (2 0) c c=1u
        """
    elseif case == "filter"
        return """
        Butterworth LC filter
        load "spice/resistor.osdi"
        load "spice/capacitor.osdi"
        load "spice/inductor.osdi"
        model r sp_resistor
        model c sp_capacitor
        model l sp_inductor
        model vsource vsource
        v1 (vin 0) vsource dc=0 type="sine" sinedc=0 ampl=1 freq=0.15915494309189535
        l1 (vin n1) l l=1.5
        c2 (n1 0) c c=1.3333333333333333
        l3 (n1 vout) l l=0.5
        r4 (vout 0) r r=1
        """
    else
        error("unknown case $case")
    end
end

#------------------------------------------------------------------------------#
# Run VACASK once, return (time_vec, signal_vec, timepoints, runtime_s)
#------------------------------------------------------------------------------#
function run_vacask_once(case, reltol, vntol, tspan, n_grid, out_nodes; maxstep=tspan[2])
    step = (tspan[2] - tspan[1]) / n_grid
    sim = sim_body(case) * """
    control
      options reltol=$(reltol) vntol=$(vntol) tran_method="$(VTRAN_METHOD)" tran_maxord=$(VTRAN_MAXORD)
      analysis tran1 tran step=$(step) stop=$(tspan[2]) maxstep=$(maxstep)
      print stats
    endc
    """
    workdir = mktempdir()
    write(joinpath(workdir, "runme.sim"), sim)
    env = copy(ENV)
    env["LD_LIBRARY_PATH"] = string(VACASK_LIB, get(ENV, "LD_LIBRARY_PATH", "") == "" ? "" : ":" * ENV["LD_LIBRARY_PATH"])
    cmd = Cmd(`$VACASK_BIN --skip-embed --skip-postprocess runme.sim`; dir=workdir, env=env)

    best = Inf
    local names, M
    for _ in 1:VACASK_REPEATS
        t0 = time()
        run(pipeline(cmd; stdout=devnull, stderr=devnull))
        best = min(best, time() - t0)
        names, M = read_rawfile(joinpath(workdir, "tran1.raw"))
    end
    rm(workdir; recursive=true, force=true)

    vi = findfirst(==("time"), names)
    vi === nothing && error("no 'time' column in raw output")
    ti = M[vi, :]
    idx(n) = (j = findfirst(==(n), names); j === nothing ? error("node '$n' not in $(names)") : j)
    sig = length(out_nodes) == 1 ? M[idx(out_nodes[1]), :] :
          M[idx(out_nodes[1]), :] .- M[idx(out_nodes[2]), :]
    return ti, sig, length(ti), best
end

function write_wave(path, t, v)
    open(path, "w") do io
        println(io, "t,v")
        @inbounds for i in eachindex(t)
            @printf(io, "%.17g,%.17g\n", t[i], v[i])
        end
    end
end

#------------------------------------------------------------------------------#
# Per-case sweep
#------------------------------------------------------------------------------#
function run_case(case::String)
    spec = CFG["cases"][case]
    tspan = (Float64(spec["tspan"][1]), Float64(spec["tspan"][2]))
    n_grid = Int(CFG["n_grid"])
    grid = collect(range(tspan[1], tspan[2]; length=n_grid))
    out_nodes = String.(spec["output"])
    reltols = Float64.(CFG["reltols"])
    ref_reltol = Float64(CFG["ref_reltol"])
    ref_abstol = Float64(CFG["ref_abstol"])
    ref_factor = Float64(get(CFG, "ref_maxstep_factor", 20))
    ref_maxstep = (tspan[2] - tspan[1]) / (n_grid * ref_factor)

    println("\n", "="^70, "\n", spec["title"], "  ($case)\n", "="^70)

    # Golden reference: moderately tight reltol with a fine maxstep so the
    # trajectory is accurate without tripping VACASK's min-timestep abort.
    @printf("  golden: reltol=%.0e vntol=%.0e maxstep=%.1e ... ", ref_reltol, ref_abstol, ref_maxstep)
    try
        ti, sig, tp, rt = run_vacask_once(case, ref_reltol, ref_abstol, tspan, n_grid, out_nodes; maxstep=ref_maxstep)
        write_wave(joinpath(OUT, "ref_$(case).csv"), grid, interp_to(ti, sig, grid))
        @printf("%.3fs  %d points\n", rt, tp)
    catch e
        println("FAILED (no golden, skipping VACASK for this case): ",
                first(sprint(showerror, e), 160))
        return
    end

    summary = open(joinpath(OUT, "vacask_$(case).csv"), "w")
    println(summary, "reltol,time_s,timepoints")
    for r in reltols
        @printf("  reltol=%.0e ... ", r)
        try
            ti, sig, tp, rt = run_vacask_once(case, r, r, tspan, n_grid, out_nodes)
            write_wave(joinpath(OUT, "vacask_$(case)_$(@sprintf("%.0e", r)).csv"), ti, sig)
            println(summary, "$r,$rt,$tp")
            @printf("%.3fs  %d points\n", rt, tp)
        catch e
            println(summary, "$r,NaN,0")
            println("FAILED: ", first(sprint(showerror, e), 160))
        end
        flush(summary)
    end
    close(summary)
end

function main()
    if VACASK_BIN === nothing
        error("VACASK binary not found. Set \$VACASK_COMMAND or run ../run_vacask.sh once.")
    end
    println("VACASK: ", VACASK_BIN)
    cases = isempty(ARGS) ? collect(keys(CFG["cases"])) : ARGS
    for case in cases
        haskey(CFG["cases"], case) || (@warn "unknown case $case"; continue)
        run_case(String(case))
    end
    println("\nVACASK waveforms + golden written to $OUT")
end

main()
