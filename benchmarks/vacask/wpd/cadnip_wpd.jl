#!/usr/bin/env julia
#==============================================================================#
# Work-Precision Diagram benchmark - Cadnip side
#
# Unlike run_benchmarks.jl (which forces a tiny fixed dtmax and measures raw
# throughput), this script runs each solver *adaptively* across a range of
# tolerances and dumps the output-node waveform per (solver, tolerance) so the
# Python/InSpice harness (run_wpd.py) can compute error-vs-runtime against the
# VACASK golden reference and plot a work-precision diagram.
#
# Outputs (under wpd/out/):
#   cadnip_<case>_<solver>_<reltol>.csv   columns: t,v        (the waveform)
#   cadnip_<case>.csv                     summary per run     (timing, steps, ...)
#   analytic_<case>.csv                   t,v   (linear cases with closed form)
#
# Usage:
#   julia --project=benchmarks benchmarks/vacask/wpd/cadnip_wpd.jl [case ...]
# With no args, runs every case in config.json.
#==============================================================================#

using Pkg
Pkg.instantiate()

# config.json is the single source of truth shared with run_wpd.py. JSON is not
# a hard dependency of the benchmarks project, so add it on first use.
try
    @eval using JSON
catch
    Pkg.add("JSON")
    @eval using JSON
end

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
using Printf
using SciMLBase: ReturnCode

# Registers the SPICE diode model (sp_diode) used by graetz/mul via ModelRegistry.
using VADistillerModels

const HERE = @__DIR__
const VACASK_DIR = normpath(joinpath(HERE, ".."))   # benchmarks/vacask
const OUT = joinpath(HERE, "out")
mkpath(OUT)

const CFG = JSON.parsefile(joinpath(HERE, "config.json"))

#------------------------------------------------------------------------------#
# Circuit builders (top-level so dispatch to the freshly-defined builders is OK)
#------------------------------------------------------------------------------#
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "graetz", "cedarsim", "runme.sp"); name=:graetz_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "rc", "cedarsim", "runme.sp"); name=:rc_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(HERE, "filter.sp"); name=:filter_circuit))

const BUILDERS = Dict(
    "graetz" => graetz_circuit,
    "rc"     => rc_circuit,
    "filter" => filter_circuit,
)

# Closed-form output for the linear cases (independent cross-check of the golden).
# Butterworth 3rd-order LC filter driven by sin(t) (see test/transients.jl).
filter_analytic(t) = (exp(-t) - sin(t) - cos(t)) / 2 +
                     (2 * sin((sqrt(3) * t) / 2)) / (sqrt(3) * sqrt(exp(t)))
const ANALYTIC = Dict("filter" => filter_analytic)

#------------------------------------------------------------------------------#
# Solver families. Higher-order BDF / Rosenbrock should pull ahead at tight
# tolerance - the whole point the throughput benchmark cannot show.
#------------------------------------------------------------------------------#
solver_ida()           = IDA(linear_solver=:KLU, max_error_test_failures=20, max_nonlinear_iters=100)
solver_fbdf()          = FBDF(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
solver_rodas5p()       = Rodas5P(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
solver_implicit_euler()= ImplicitEuler(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())

# (name, constructor, min_reltol) - ImplicitEuler is order 1, so it is capped to
# loose tolerances to avoid runaway step counts (and to visibly lag on the WPD).
const SOLVERS_NONLINEAR = [
    ("IDA",     solver_ida,     0.0),
    ("FBDF",    solver_fbdf,    0.0),
    ("Rodas5P", solver_rodas5p, 0.0),
]
const SOLVERS_LINEAR = [
    ("IDA",           solver_ida,            0.0),
    ("FBDF",          solver_fbdf,           0.0),
    ("Rodas5P",       solver_rodas5p,        0.0),
    ("ImplicitEuler", solver_implicit_euler, 1e-6),
]

#------------------------------------------------------------------------------#
# Helpers
#------------------------------------------------------------------------------#
setup(builder) = (c = MNACircuit(builder); MNA.assemble!(c); c)

"Index of node `name` in the assembled state vector (matches test/transients.jl)."
function node_index(builder, name::AbstractString)
    spec = MNA.MNASpec(temp=27.0, mode=:dcop, time=0.0)
    ctx = Base.invokelatest(builder, (;), spec)
    sys = MNA.assemble!(ctx)
    idx = findfirst(==(Symbol(name)), sys.node_names)
    idx === nothing && error("node '$name' not found; available: $(sys.node_names)")
    return idx
end

"Output signal over sol.t: single node-to-ground, or differential of two nodes."
function output_signal(sol, idxs::Vector{Int})
    if length(idxs) == 1
        return [sol.u[i][idxs[1]] for i in eachindex(sol.t)]
    else
        return [sol.u[i][idxs[1]] - sol.u[i][idxs[2]] for i in eachindex(sol.t)]
    end
end

function write_wave(path, t, v)
    open(path, "w") do io
        println(io, "t,v")
        @inbounds for i in eachindex(t)
            @printf(io, "%.17g,%.17g\n", t[i], v[i])
        end
    end
end

reltol_tag(r) = @sprintf("%.0e", r)   # 1.0e-6 -> "1e-06"

"""
Source breakpoints for a case. Adaptive solvers will otherwise step over the
sharp edges of a pulse source (FBDF takes ~3 steps and misses the whole train);
SPICE engines like VACASK break at source breakpoints internally, so passing
these as `tstops` makes the comparison fair and correct. Empty for smooth drives.
"""
function case_tstops(case, tspan)
    case == "rc" || return Float64[]
    td, tr, tf, pw, per = 1e-6, 1e-6, 1e-6, 1e-3, 2e-3  # PULSE(0 1 1u 1u 1u 1m 2m)
    bps = Float64[]
    k = 0
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
# Run one case
#------------------------------------------------------------------------------#
function run_case(case::String)
    spec = CFG["cases"][case]
    builder = BUILDERS[case]
    t0, t1 = Float64(spec["tspan"][1]), Float64(spec["tspan"][2])
    n_grid = Int(CFG["n_grid"])
    grid = collect(range(t0, t1; length=n_grid))
    out_nodes = String.(spec["output"])
    reltols = Float64.(CFG["reltols"])
    ascale = Float64(CFG["abstol_scale"])
    solvers = spec["linear"] ? SOLVERS_LINEAR : SOLVERS_NONLINEAR

    println("\n", "="^70, "\n", spec["title"], "  ($case, tspan=($t0,$t1))\n", "="^70)

    idxs = [node_index(builder, n) for n in out_nodes]
    tstops = case_tstops(case, (t0, t1))

    # Analytic reference, written DENSELY (not on the 2000-pt grid) so the plot
    # stage can interpolate it accurately onto any solver's native timepoints.
    if get(spec, "analytic", false) && haskey(ANALYTIC, case)
        f = ANALYTIC[case]
        fine = collect(range(t0, t1; length=200_000))
        write_wave(joinpath(OUT, "analytic_$(case).csv"), fine, f.(fine))
    end

    summary = open(joinpath(OUT, "cadnip_$(case).csv"), "w")
    println(summary, "solver,reltol,median_time_s,steps,rejects,nniter,retcode")

    for (sname, sfn, min_rel) in solvers, r in reltols
        r < min_rel && continue
        a = r * ascale
        @printf("  %-14s reltol=%.0e abstol=%.0e ... ", sname, r, a)
        try
            # One untimed solve for the waveform + stats.
            c = setup(builder)
            sol = tran!(c, (t0, t1); abstol=a, reltol=r, solver=sfn(),
                        saveat=grid, tstops=tstops, maxiters=50_000_000)
            # saveat fixes length(sol.t) to the grid size, so report the actual
            # number of accepted internal steps (the true work) when available.
            steps = hasproperty(sol.stats, :naccept) && sol.stats.naccept > 0 ?
                    sol.stats.naccept : length(sol.t)
            rejects = hasproperty(sol.stats, :nreject) ? sol.stats.nreject : 0
            nniter = hasproperty(sol.stats, :nnonliniter) ? sol.stats.nnonliniter : 0
            ok = sol.retcode == ReturnCode.Success && isapprox(sol.t[end], t1; rtol=1e-6)

            v = output_signal(sol, idxs)
            write_wave(joinpath(OUT, "cadnip_$(case)_$(sname)_$(reltol_tag(r)).csv"), sol.t, v)

            # Timed runs (reuse a prepared circuit; tran! does not mutate structure).
            ct = setup(builder)
            bench = @benchmark tran!($ct, ($t0, $t1); abstol=$a, reltol=$r, solver=$(sfn()),
                                     saveat=$grid, tstops=$tstops, maxiters=50_000_000) samples=3 evals=1 seconds=120
            tmed = median(bench.times) / 1e9

            println(summary, "$sname,$r,$tmed,$steps,$rejects,$nniter,$(sol.retcode)")
            @printf("%.3fs  steps=%d rej=%d  %s\n", tmed, steps, rejects, ok ? "ok" : string(sol.retcode))
        catch e
            msg = sprint(showerror, e)
            println(summary, "$sname,$r,NaN,0,0,0,failed")
            println("FAILED: ", first(msg, 120))
        end
        flush(summary)
    end
    close(summary)
end

function main()
    cases = isempty(ARGS) ? collect(keys(CFG["cases"])) : ARGS
    for case in cases
        haskey(BUILDERS, case) || (@warn "unknown case $case"; continue)
        run_case(String(case))
    end
    println("\nCadnip waveforms + summaries written to $OUT")
    println("Next: vacask_wpd.jl (VACASK sweep), then plot_wpd.jl (diagrams).")
end

main()
