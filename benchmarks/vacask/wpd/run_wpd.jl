#!/usr/bin/env julia
#==============================================================================#
# Work-Precision Diagram benchmark - single entry point.
#
# Unlike run_benchmarks.jl (which forces a tiny fixed dtmax and measures raw
# throughput), this runs each solver *adaptively* across a range of tolerances
# and reports accuracy-per-runtime. For every case it:
#   1. sweeps Cadnip's solver families over `reltols`,
#   2. builds the pinned golden reference (analytic | vacask | cadnip),
#   3. sweeps the real VACASK (high-order Gear/BDF),
#   4. computes each run's relative-L2 error at its OWN timepoints vs the dense
#      golden, and writes out/wpd_results.md with per-case tables and inline
#      ASCII (UnicodePlots) work-precision diagrams.
#
# Usage:
#   julia --project=benchmarks benchmarks/vacask/wpd/run_wpd.jl [case ...]
# With no args, runs every case in config.json.
#==============================================================================#

using Pkg
Pkg.instantiate()

include(joinpath(@__DIR__, "wpd_common.jl"))

using Cadnip
using Cadnip.MNA
using Sundials: IDA
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqSDIRK: ImplicitEuler
using ADTypes: AutoFiniteDiff
using LinearSolve: KLUFactorization
using BenchmarkTools
using Statistics
using SciMLBase: ReturnCode
using VADistillerModels     # registers the SPICE diode model
using UnicodePlots

mkpath(OUT)

#------------------------------------------------------------------------------#
# Circuit builders (top level so dispatch to the freshly-defined builders is OK)
#------------------------------------------------------------------------------#
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "graetz", "cedarsim", "runme.sp"); name=:graetz_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "mul", "cedarsim", "runme.sp"); name=:mul_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "rc", "cedarsim", "runme.sp"); name=:rc_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(HERE, "filter.sp"); name=:filter_circuit))

const BUILDERS = Dict(
    "graetz" => graetz_circuit, "mul" => mul_circuit,
    "rc" => rc_circuit, "filter" => filter_circuit,
)
const NONLINEAR = ("graetz", "mul")

#------------------------------------------------------------------------------#
# Analytic references (exact closed forms)
#------------------------------------------------------------------------------#
# Butterworth 3rd-order LC filter driven by sin(t) (see test/transients.jl).
filter_analytic(t) = (exp(-t) - sin(t) - cos(t)) / 2 +
                     (2 * sin((sqrt(3) * t) / 2)) / (sqrt(3) * sqrt(exp(t)))

# RC (R=1k, C=1u, τ=1ms) driven by PULSE(0 1 td=1u tr=1u tf=1u pw=1m per=2m).
# Exact response to the piecewise-linear source, extended over all periods.
function rc_analytic(t)
    τ = 1e-3
    td, tr, pw, tf, per = 1e-6, 1e-6, 1e-3, 1e-6, 2e-3
    v0 = 0.0; k = 0
    while true
        base = k * per
        base > t && return v0
        segs = ((base,             base + td,               0.0, 0.0),
                (base + td,         base + td + tr,          0.0, 1.0 / tr),
                (base + td + tr,    base + td + tr + pw,     1.0, 0.0),
                (base + td+tr+pw,   base + td + tr + pw+tf,  1.0, -1.0 / tf),
                (base + td+tr+pw+tf, base + per,             0.0, 0.0))
        for (s0, s1, a, b) in segs
            if t <= s1
                dt = t - s0
                return (v0 - a + b * τ) * exp(-dt / τ) + a + b * dt - b * τ
            else
                dt = s1 - s0
                v0 = (v0 - a + b * τ) * exp(-dt / τ) + a + b * dt - b * τ
            end
        end
        k += 1
    end
end

const ANALYTIC = Dict("filter" => filter_analytic, "rc" => rc_analytic)

#------------------------------------------------------------------------------#
# Solver families
#------------------------------------------------------------------------------#
mk_ida()     = IDA(linear_solver=:KLU, max_error_test_failures=20, max_nonlinear_iters=100)
mk_fbdf()    = FBDF(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
mk_rodas5p() = Rodas5P(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
mk_ie()      = ImplicitEuler(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())

# (name, constructor, min_reltol). ImplicitEuler is order 1: capped to loose
# tolerances (runaway step counts otherwise) and to visibly lag on the WPD.
# Rodas5P degenerates on the diode DAEs, so it is used only on the linear cases.
solvers_for(case) = case in NONLINEAR ?
    [("IDA", mk_ida, 0.0), ("FBDF", mk_fbdf, 0.0), ("ImplicitEuler", mk_ie, 1e-6)] :
    [("IDA", mk_ida, 0.0), ("FBDF", mk_fbdf, 0.0), ("Rodas5P", mk_rodas5p, 0.0), ("ImplicitEuler", mk_ie, 1e-6)]

#------------------------------------------------------------------------------#
# Helpers
#------------------------------------------------------------------------------#
setup(builder) = (c = MNACircuit(builder); MNA.assemble!(c); c)

function output_signal(sol, out_nodes)
    length(out_nodes) == 1 && return Vector{Float64}(sol[Symbol(out_nodes[1])])
    return Vector{Float64}(sol[Symbol(out_nodes[1])]) .- Vector{Float64}(sol[Symbol(out_nodes[2])])
end

"Pulse-edge breakpoints so adaptive solvers don't step over the sharp source edges."
function case_tstops(case, tspan)
    case == "rc" || return Float64[]
    td, tr, pw, tf, per = 1e-6, 1e-6, 1e-3, 1e-6, 2e-3
    bps = Float64[]; k = 0
    while true
        base = td + k * per
        base > tspan[2] && break
        for off in (0.0, tr, tr + pw, tr + pw + tf)
            e = base + off
            tspan[1] <= e <= tspan[2] && push!(bps, e)
        end
        k += 1
    end
    return bps
end

#------------------------------------------------------------------------------#
# VACASK netlists (match the cedarsim/runme.sp circuits)
#------------------------------------------------------------------------------#
const VDIODE = """
load "spice/resistor.osdi"
load "spice/capacitor.osdi"
load "spice/sn/diode.osdi"
model r sp_resistor
model c sp_capacitor
model vsource vsource
model dmod sp_diode ( is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45 )"""

function sim_body(case::String)
    if case == "graetz"
        return "Graetz bridge\n$VDIODE\n" * """
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
    elseif case == "mul"
        return "Diode voltage multiplier\n$VDIODE\n" * """
        vs (a 0) vsource dc=0 type="sine" sinedc=0 ampl=50 freq=100k tdphase=90
        r1 (a 1) r r=0.01
        c1 (1 2) c c=100n
        d1 (0 1) dmod
        c2 (0 10) c c=100n
        d2 (1 10) dmod
        c3 (1 2) c c=100n
        d3 (10 2) dmod
        c4 (10 20) c c=100n
        d4 (2 20) dmod
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

const VACASK_BIN, VACASK_LIB = locate_vacask()

"Run VACASK once; return (t, signal, timepoints, runtime_s) or throw on failure."
function run_vacask_once(case, reltol, vntol, tspan, out_nodes; maxstep=tspan[2])
    step = (tspan[2] - tspan[1]) / Int(CFG["n_grid"])
    method = String(get(CFG, "vacask_tran_method", "gear"))
    maxord = Int(get(CFG, "vacask_tran_maxord", 5))
    sim = sim_body(case) * """
    control
      options reltol=$(reltol) vntol=$(vntol) tran_method="$(method)" tran_maxord=$(maxord)
      analysis tran1 tran step=$(step) stop=$(tspan[2]) maxstep=$(maxstep)
    endc
    """
    workdir = mktempdir()
    write(joinpath(workdir, "runme.sim"), sim)
    env = copy(ENV)
    prev = get(ENV, "LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = isempty(prev) ? VACASK_LIB : "$VACASK_LIB:$prev"
    cmd = Cmd(`$VACASK_BIN --skip-embed --skip-postprocess runme.sim`; dir=workdir, env=env)
    best = Inf; local names, M
    for _ in 1:VACASK_REPEATS
        t0 = time()
        run(pipeline(cmd; stdout=devnull, stderr=devnull))
        best = min(best, time() - t0)
        names, M = read_rawfile(joinpath(workdir, "tran1.raw"))
    end
    rm(workdir; recursive=true, force=true)
    ti = M[findfirst(==("time"), names), :]
    idx(n) = (j = findfirst(==(n), names); j === nothing ? error("node '$n' not in $names") : j)
    sig = length(out_nodes) == 1 ? M[idx(out_nodes[1]), :] :
          M[idx(out_nodes[1]), :] .- M[idx(out_nodes[2]), :]
    return ti, sig, length(ti), best
end

#------------------------------------------------------------------------------#
# Per-case: Cadnip sweep, goldens, VACASK sweep
#------------------------------------------------------------------------------#
function run_cadnip_sweep(case, spec)
    builder = BUILDERS[case]
    t0, t1 = Float64(spec["tspan"][1]), Float64(spec["tspan"][2])
    out_nodes = String.(spec["output"])
    reltols = Float64.(CFG["reltols"])
    ascale = Float64(CFG["abstol_scale"])
    tstops = case_tstops(case, (t0, t1))

    summary = open(joinpath(OUT, "cadnip_$(case).csv"), "w")
    println(summary, "solver,reltol,median_time_s,steps,rejects,nniter,retcode")
    for (sname, sfn, min_rel) in solvers_for(case), r in reltols
        r < min_rel && continue
        a = r * ascale
        @printf("  cadnip %-14s reltol=%.0e ... ", sname, r); flush(stdout)
        try
            c = setup(builder)
            sol = tran!(c, (t0, t1); abstol=a, reltol=r, solver=sfn(),
                        tstops=tstops, dense=false, maxiters=50_000_000)
            steps = hasproperty(sol.stats, :naccept) && sol.stats.naccept > 0 ? sol.stats.naccept : length(sol.t)
            rejects = hasproperty(sol.stats, :nreject) ? sol.stats.nreject : 0
            nniter = hasproperty(sol.stats, :nnonliniter) ? sol.stats.nnonliniter : 0
            write_wave(joinpath(OUT, "cadnip_$(case)_$(sname)_$(reltol_tag(r)).csv"),
                       sol.t, output_signal(sol, out_nodes))
            ct = setup(builder)
            bench = @benchmark tran!($ct, ($t0, $t1); abstol=$a, reltol=$r, solver=$(sfn()),
                                     tstops=$tstops, dense=false, maxiters=50_000_000) samples=3 evals=1 seconds=120
            tmed = median(bench.times) / 1e9
            println(summary, "$sname,$r,$tmed,$steps,$rejects,$nniter,$(sol.retcode)")
            @printf("%.3fs steps=%d\n", tmed, steps)
        catch e
            println(summary, "$sname,$r,NaN,0,0,0,failed")
            println("FAILED: ", first(sprint(showerror, e), 120))
        end
        flush(summary); flush(stdout)
    end
    close(summary)
end

"Dense Cadnip-tight golden (robust IDA) for cases VACASK can't reference."
function cadnip_golden(case, spec)
    builder = BUILDERS[case]
    t0, t1 = Float64(spec["tspan"][1]), Float64(spec["tspan"][2])
    out_nodes = String.(spec["output"])
    grid = collect(range(t0, t1; length=Int(CFG["n_grid"]) * 50))
    tstops = case_tstops(case, (t0, t1))
    c = setup(builder)
    sol = tran!(c, (t0, t1); abstol=1e-12, reltol=1e-11, solver=mk_ida(),
                saveat=grid, tstops=tstops, maxiters=100_000_000)
    write_wave(joinpath(OUT, "cadnip_ref_$(case).csv"), sol.t, output_signal(sol, out_nodes))
end

function run_vacask_sweep(case, spec, want_golden)
    t0, t1 = Float64(spec["tspan"][1]), Float64(spec["tspan"][2])
    out_nodes = String.(spec["output"])
    reltols = Float64.(CFG["reltols"])
    ascale = Float64(CFG["abstol_scale"])
    ref_reltol = Float64(CFG["ref_reltol"]); ref_abstol = Float64(CFG["ref_abstol"])
    ref_factor = Float64(get(CFG, "ref_maxstep_factor", 50))

    if want_golden
        ms = (t1 - t0) / (Int(CFG["n_grid"]) * ref_factor)
        @printf("  vacask golden reltol=%.0e maxstep=%.1e ... ", ref_reltol, ms); flush(stdout)
        ti, sig, tp, rt = run_vacask_once(case, ref_reltol, ref_abstol, (t0, t1), out_nodes; maxstep=ms)
        write_wave(joinpath(OUT, "ref_$(case).csv"), ti, sig)
        @printf("%.3fs %d pts\n", rt, tp); flush(stdout)
    end

    summary = open(joinpath(OUT, "vacask_$(case).csv"), "w")
    println(summary, "reltol,time_s,timepoints")
    for r in reltols
        @printf("  vacask reltol=%.0e ... ", r); flush(stdout)
        try
            ti, sig, tp, rt = run_vacask_once(case, r, r * ascale, (t0, t1), out_nodes)
            write_wave(joinpath(OUT, "vacask_$(case)_$(reltol_tag(r)).csv"), ti, sig)
            println(summary, "$r,$rt,$tp")
            @printf("%.3fs %d pts\n", rt, tp)
        catch e
            println(summary, "$r,NaN,0")
            println("ABORT/skip")
        end
        flush(summary); flush(stdout)
    end
    close(summary)
end

#------------------------------------------------------------------------------#
# Golden selection + analysis + report
#------------------------------------------------------------------------------#
function load_golden(case, spec)
    g = String(spec["golden"])
    if g == "analytic"
        return read_wave(joinpath(OUT, "analytic_$(case).csv"))..., "analytic (exact)"
    elseif g == "vacask"
        return read_wave(joinpath(OUT, "ref_$(case).csv"))..., "VACASK (tight)"
    elseif g == "cadnip"
        return read_wave(joinpath(OUT, "cadnip_ref_$(case).csv"))..., "Cadnip IDA (tight)"
    else
        error("case $case: unknown golden '$g' (use analytic|vacask|cadnip)")
    end
end

function analyze(case, spec)
    gt, gv, gsrc = load_golden(case, spec)
    println("$case: golden = $gsrc")
    curves = Dict{String,Vector{Tuple{Float64,Float64}}}()
    table = Tuple{String,Float64,Float64,Float64}[]

    for row in first(read_table(joinpath(OUT, "cadnip_$(case).csv")))
        solver = row["solver"]; r = parse(Float64, row["reltol"])
        t = tryparse(Float64, get(row, "median_time_s", "NaN"))
        wp = joinpath(OUT, "cadnip_$(case)_$(solver)_$(reltol_tag(r)).csv")
        isfile(wp) || continue
        tw, vw = read_wave(wp)
        err = run_error(tw, vw, gt, gv)
        isfinite(err) && t !== nothing && isfinite(t) &&
            push!(get!(curves, "Cadnip $solver", Tuple{Float64,Float64}[]), (err, t))
        push!(table, ("Cadnip $solver", r, err, something(t, NaN)))
    end

    vpath = joinpath(OUT, "vacask_$(case).csv")
    if isfile(vpath)
        for row in first(read_table(vpath))
            r = parse(Float64, row["reltol"]); t = tryparse(Float64, get(row, "time_s", "NaN"))
            wp = joinpath(OUT, "vacask_$(case)_$(reltol_tag(r)).csv")
            isfile(wp) || continue
            tw, vw = read_wave(wp)
            err = run_error(tw, vw, gt, gv)
            isfinite(err) && t !== nothing && isfinite(t) &&
                push!(get!(curves, "VACASK", Tuple{Float64,Float64}[]), (err, t))
            push!(table, ("VACASK", r, err, something(t, NaN)))
        end
    end

    # cross-check: if both analytic and a VACASK ref exist, report their agreement
    ap = joinpath(OUT, "analytic_$(case).csv"); rp = joinpath(OUT, "ref_$(case).csv")
    xcheck = ""
    if isfile(ap) && isfile(rp)
        ta, va = read_wave(ap); tr, vr = read_wave(rp)
        xcheck = @sprintf("analytic vs VACASK-tight cross-check: rel-L2 = %.2e", run_error(tr, vr, ta, va))
        println("  ", xcheck)
    end
    return gsrc, curves, table, xcheck
end

function ascii_plot(title, curves)
    plt = nothing
    for label in sort(collect(keys(curves)))
        pts = sort(curves[label]); isempty(pts) && continue
        x = Float64[p[1] for p in pts]; y = Float64[p[2] for p in pts]
        if plt === nothing
            plt = lineplot(x, y; name=label, xlabel="rel-L2 error", ylabel="runtime s",
                           title=title, xscale=:log10, yscale=:log10,
                           width=64, height=16, canvas=DotCanvas)
        else
            lineplot!(plt, x, y; name=label)
        end
    end
    plt === nothing && return "(no data)"
    io = IOBuffer()
    show(IOContext(io, :color => false), plt)
    return String(take!(io))
end

function report(results)
    io = IOBuffer()
    println(io, "# Work-Precision Diagram Results\n")
    println(io, "Each solver runs *adaptively* across a tolerance sweep (no forced timestep).")
    println(io, "Error = relative L2 of the output node at each run's own timepoints vs the")
    println(io, "golden reference. VACASK uses high-order Gear/BDF (`tran_maxord=5`).\n")
    for (case, gsrc, curves, table, xcheck) in results
        spec = CFG["cases"][case]
        println(io, "## $(spec["title"])\n")
        println(io, "Golden reference: **$gsrc**.", isempty(xcheck) ? "" : " ($xcheck)", "\n")
        if !isempty(curves)
            println(io, "```")
            println(io, ascii_plot(String(case), curves))
            println(io, "```\n")
        end
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
    println("\nReport: $path")
end

#------------------------------------------------------------------------------#
# Main
#------------------------------------------------------------------------------#
function main()
    cases = isempty(ARGS) ? collect(keys(CFG["cases"])) : ARGS
    results = []
    for case in cases
        haskey(CFG["cases"], case) || (@warn "unknown case $case"; continue)
        spec = CFG["cases"][case]
        golden = String(spec["golden"])
        println("\n", "="^70, "\n", spec["title"], "  ($case)\n", "="^70)

        run_cadnip_sweep(case, spec)

        if golden == "analytic"
            t0, t1 = Float64(spec["tspan"][1]), Float64(spec["tspan"][2])
            fine = collect(range(t0, t1; length=200_000))
            write_wave(joinpath(OUT, "analytic_$(case).csv"), fine, ANALYTIC[case].(fine))
        elseif golden == "cadnip"
            cadnip_golden(case, spec)
        end

        # VACASK sweep always runs where it can; also produce the VACASK golden
        # only when this case is pinned to it.
        if VACASK_BIN !== nothing
            run_vacask_sweep(case, spec, golden == "vacask")
        elseif golden == "vacask"
            error("case $case pinned to VACASK golden but VACASK binary not found")
        end

        push!(results, (String(case), analyze(case, spec)...))
    end
    report(results)
end

main()
