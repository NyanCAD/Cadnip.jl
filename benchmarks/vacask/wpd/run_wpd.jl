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
#      ASCII (UnicodePlots) work-precision diagrams, plus higher-quality
#      PNG/SVG plots (Plots.jl/GR) under out/plots/ for the CI artifact - the
#      job summary only renders the ASCII version.
#
# Usage:
#   julia --project=benchmarks benchmarks/vacask/wpd/run_wpd.jl [case ...]
# With no args, runs every case in config.json.
#==============================================================================#

using Pkg
Pkg.instantiate()

# Must be set before Plots/GR initializes: renders to file with no display
# server (CI has none). File output only - no markdown embedding, so none of
# the earlier headless-rendering complications apply here.
ENV["GKSwstype"] = "100"

include(joinpath(@__DIR__, "wpd_common.jl"))

using Cadnip
using Cadnip.MNA
using Sundials: IDA
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqRosenbrock: Rodas5P, Rodas6P
using OrdinaryDiffEqSDIRK: Kvaerno3, Kvaerno5, KenCarp4
using OrdinaryDiffEqFIRK: RadauIIA5
using ADTypes: AutoFiniteDiff
using LinearSolve: KLUFactorization
using BenchmarkTools
using Statistics
using SciMLBase: ReturnCode
using VADistillerModels     # registers the SPICE diode model
using UnicodePlots
using Plots
gr()

mkpath(OUT)
mkpath(PLOTS_DIR)

#------------------------------------------------------------------------------#
# Circuit builders (top level so dispatch to the freshly-defined builders is OK)
#------------------------------------------------------------------------------#
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "graetz", "cedarsim", "runme.sp"); name=:graetz_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "mul", "cedarsim", "runme.sp"); name=:mul_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "rc", "cedarsim", "runme.sp"); name=:rc_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(HERE, "filter.sp"); name=:filter_circuit))
Base.include(@__MODULE__, SpiceFile(joinpath(VACASK_DIR, "darlington", "cedarsim", "runme.sp"); name=:darlington_circuit))

const BUILDERS = Dict(
    "graetz" => graetz_circuit, "mul" => mul_circuit,
    "rc" => rc_circuit, "filter" => filter_circuit,
    "darlington" => darlington_circuit,
)

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
mk_ida()      = IDA(linear_solver=:KLU, max_error_test_failures=20, max_nonlinear_iters=100)
mk_fbdf()     = FBDF(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
mk_rodas5p()  = Rodas5P(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
mk_rodas6p()  = Rodas6P(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
mk_kvaerno5() = Kvaerno5(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
mk_radau()    = RadauIIA5(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
mk_kencarp4() = KenCarp4(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff())
# `smooth_est=false` variants - see the SOLVERS comment below for why these
# exist as separate constructors rather than changing the defaults above:
# filter/rc never needed this (Kvaerno5 already works fine there), so the
# plain constructors stay untouched for them.
mk_kvaerno3_rawest() = Kvaerno3(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff(), smooth_est=false)
mk_kvaerno5_rawest() = Kvaerno5(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff(), smooth_est=false)
mk_kencarp4_rawest() = KenCarp4(linsolve=KLUFactorization(), autodiff=AutoFiniteDiff(), smooth_est=false)

# (name, constructor, min_reltol) per case. Not a blanket linear/nonlinear split -
# solver viability varies by circuit, confirmed empirically (see
# ../wpd/README.md "Solver survey" for the full candidate sweep this was drawn
# from, across filter/rc/graetz/mul plus a later darlington-only follow-up):
#   - **`smooth_est=false` unlocks the whole Kvaerno/KenCarp (ESDIRK) family
#     on the nonlinear cases - this was the actual bug, not the diode/BJT
#     turn-on itself.** Root-caused by reading `OrdinaryDiffEqSDIRK`'s shared
#     ESDIRK step implementation (`generic_imex_perform_step.jl`): the local
#     error estimate branches on `isnewton(nlsolver) && alg.smooth_est`
#     (`smooth_est=true` is every one of these algorithms' default) - when
#     true, the raw embedded-difference vector is run through an *extra
#     linear solve* against the cached iteration matrix
#     `W = -(G + C/(γdt))` ("smoothing", a real Hairer/Wanner technique for
#     damping noisy error estimates on stiff problems). Confirmed directly
#     (`darlington`, native `NLNewton`, nothing else changed): the `dt`
#     sequence at the failure point shrinks by exactly 1/5 every rejected
#     retry, forever, down past `1e-23` before the integrator gives up -
#     while a corrector-activation trace showed Newton itself converges
#     cleanly (`θ≈0.3`, nowhere near the `θ>2` divergence threshold) on
#     *every single one* of those rejected attempts. So the failure was
#     never in the nonlinear solve - it's the smoothed estimate itself that
#     never shrinks properly as `dt→0` for this circuit's structure (MNA's
#     mass matrix `C` has zero rows for non-capacitive/algebraic nodes -
#     `W`'s scaling is genuinely inhomogeneous across rows as `1/(γdt)→∞`,
#     exactly the kind of thing an extra ill-conditioned solve can turn
#     into a spurious non-vanishing LTE). `Kvaerno5(smooth_est=false)` on
#     the *native* path (no bridge, no PCNR, nothing else touched) alone
#     reproduces success - conclusively isolating `smooth_est` as the cause,
#     not Jacobian source (native already uses an explicit analytic
#     `jac!=-G`, not autodiff - confirmed by reading `solve.jl`'s
#     `ODEFunction` construction - so the earlier `AutoFiniteDiff` vs
#     `AutoForwardDiff` test never actually varied the Jacobian at all) and
#     not the W-construction formula (`jacobian2W!`'s generic combination
#     `-(mass_matrix/(γdt)) + J` is algebraically identical to
#     `CedarPCNRStageJac`'s hand-rolled `-(G + c·C)` - same inputs, same
#     formula, verified by reading both). The `_rawest` constructors above
#     (`mk_kvaerno3_rawest`, `mk_kvaerno5_rawest`, `mk_kencarp4_rawest`) are
#     the fix - see below for what it unlocks per case.
#   - Kvaerno3/Kvaerno5 (SDIRK) with the *default* `smooth_est=true` get
#     stuck in the stiff turn-on/turn-off transient on all three nonlinear
#     cases - `graetz`/`mul` (diodes) AND `darlington` (BJTs: `:Unstable` at
#     reltol 1e-3/1e-5/1e-7, burning up to 34118 steps without making it
#     past t~2e-6 of the 2e-5 tspan at the worst point). With
#     `smooth_est=false`: full `:Success` on `graetz` at every reltol
#     checked (1e-3 to 1e-5, a bounded exploration - see `min_reltol=1e-5`
#     on both in `SOLVERS`, not verified tighter); on `darlington`, full
#     `:Success` from 1e-3 through 1e-8 (`min_reltol=1e-8` excludes just the
#     one failure at the tightest 1e-9). `mul` stays the exception: Kvaerno5
#     only succeeds at the single loosest reltol=1e-3 before hitting the
#     `maxiters=1e6` bound without finishing (not a clean failure like
#     before, just genuinely slow - not worth the one marginal point, so
#     not added), and Kvaerno3 hits the same `maxiters` bound at *every*
#     tested reltol including 1e-3, reaching only ~10% of the tspan - `mul`
#     really is the stiffest case in this suite, `smooth_est` or not.
#     Kvaerno3 is left out of `darlington`'s `SOLVERS` (redundant with the
#     cheaper Kvaerno5 there, same reasoning as the TRBDF2/QNDF calls
#     below).
#   - Rodas5P works on `graetz` (correct rectified output at reltol 1e-3/1e-6,
#     :Unstable only at the tightest 1e-9, already excluded by the retcode filter)
#     but hangs on `mul` (its faster 100kHz cascaded-diode switching is far
#     stiffer) even at the loosest reltol=1e-3 - excluded there. On `darlington`
#     it's fully robust (`:Success` at every reltol 1e-3 through 1e-9) but
#     Rodas6P dominates it there (see below), so it's left out in favor of
#     Rodas6P alone, same call as `filter`.
#   - RadauIIA5 (5th-order FIRK) matches or beats Rodas5P's accuracy-per-step on
#     the linear cases and on `graetz` at loose/medium reltol, going :Unstable
#     only past reltol=1e-5 there - added to filter/rc/graetz. Like Kvaerno5 it
#     is not viable on `mul` (:Unstable at every reltol) - excluded there. On
#     `darlington` it's the cheapest solver at tight tolerance by a wide margin
#     (5821 steps at reltol=1e-9 vs IDA's 30355 and Rodas6P's 547386) but has a
#     two-tolerance outlier: `:Unstable` specifically at reltol=1e-3 and 1e-5,
#     `:Success` at every other point tested (1e-4, 1e-6, 1e-7, 1e-8, 1e-9) -
#     not investigated further (non-monotonic in reltol, unlike `rc`'s single-
#     tolerance outlier below); added anyway since the pipeline already drops
#     failed points from the plotted curve rather than erroring on them.
#   - KenCarp4 (ESDIRK) with the *default* `smooth_est=true` fails
#     everywhere on `graetz`, gets only partial coverage on `mul` (reltol
#     1e-3/1e-5 before hitting the 100kHz switching wall), and is fully
#     `:Unstable` on `darlington`. With `smooth_est=false` it's the biggest
#     winner of the fix: fully `:Success` and cheap on `graetz` (707-1410
#     steps at reltol 1e-3 to 1e-5, the bounded range checked), fully
#     `:Success` and cheap on `mul` too (4864-12857 steps, same range -
#     `mk_kencarp4_rawest` replaces the old unconditional `mk_kencarp4` in
#     `mul`'s `SOLVERS` entry), and the *only* nonlinear-case solver that's
#     fully `:Success` across `darlington`'s *entire* reltol range, 1e-3
#     through 1e-9 - no `min_reltol` needed there at all.
#   - Rodas5P vs Rodas6P (6th-order, newest of the Rodas family) is
#     genuinely case-dependent, not a strict order: Rodas5P is more
#     accurate at every reltol on `rc` (though Rodas6P takes fewer steps
#     there); on `filter` Rodas6P strictly dominates (lower error *and*
#     fewer steps at every reltol) - used there instead of Rodas5P; on
#     `graetz` they cross over (Rodas5P wins loose reltol, Rodas6P wins
#     medium, tied at 1e-7) - Rodas5P kept since the crossover favors it at
#     the more commonly-used loose end; on `mul` Rodas6P is more accurate
#     and reaches one more tolerance point - used there instead of Rodas5P;
#     on `darlington` Rodas6P strictly dominates too (fewer steps than
#     Rodas5P at all 7 reltols, e.g. 547386 vs 758645 at reltol=1e-9) -
#     same call as `filter`.
#     Net: each case gets whichever one wins there, never both (one
#     Rosenbrock representative per case).
#   - True (implicit-first-stage) SDIRK - SDIRK2, Cash4, Hairer4, Hairer42 -
#     were tried on `graetz`/`mul` on the theory that PLECS (which defaults
#     to (E)SDIRK for MNA circuits and notes "SDIRK is typically more
#     stable" than ESDIRK) might handle the diode turn-on better than the
#     ESDIRK family (Kvaerno5/KenCarp4/TRBDF2). They don't: all four go
#     `:Unstable` at nearly every tolerance on both diode circuits, doing
#     *worse* than KenCarp4's partial success on `mul` - not added anywhere.
#   - IDA is robust on every case; FBDF is robust on every case except
#     `darlington`, where it converges at the loosest reltol=1e-3 (9381
#     steps) but goes `:Unstable` at every tighter tolerance (`dt` forced
#     below floating-point epsilon during the sharp BJT turn-on/turn-off
#     edges) - confirmed via a local Cadnip-only sweep (no min-reltol cutoff
#     needed: the failures are cheap/clean, not slow/hanging like Rodas6P on
#     `mul` below, so there's no runtime reason to bound it, just an
#     accuracy ceiling). Left in `SOLVERS` unconditionally since the
#     `run_cadnip_sweep`/`analyze` pipeline already excludes non-`Success`
#     runs from the plotted curve rather than erroring - FBDF just shows a
#     single point on `darlington`'s work-precision diagram (KenCarp4 with
#     `smooth_est=false`, above, now covers the full range instead).
#   - Rodas6P on `mul` degrades catastrophically (not just fails) below
#     reltol=1e-5: it took 451s/6.4M steps to *finish* reltol=1e-6 (vs
#     KenCarp4's 2.2s/43775 steps at the same tolerance) and then hung
#     entirely at reltol=1e-7 until CI's 60-minute job timeout killed it -
#     it doesn't cleanly error like the exploration sweep (bounded by a
#     much lower maxiters) suggested. min_reltol=1e-5 keeps it to the
#     tolerances it's actually fast at; it's still redundant with KenCarp4
#     there, but the two together are more informative than either alone.
const SOLVERS = Dict(
    "filter" => [("IDA", mk_ida, 0.0), ("FBDF", mk_fbdf, 0.0), ("Rodas6P", mk_rodas6p, 0.0), ("Kvaerno5", mk_kvaerno5, 0.0), ("RadauIIA5", mk_radau, 0.0)],
    "rc"     => [("IDA", mk_ida, 0.0), ("FBDF", mk_fbdf, 0.0), ("Rodas5P", mk_rodas5p, 0.0), ("Kvaerno5", mk_kvaerno5, 0.0), ("RadauIIA5", mk_radau, 0.0)],
    "graetz" => [("IDA", mk_ida, 0.0), ("FBDF", mk_fbdf, 0.0), ("Rodas5P", mk_rodas5p, 0.0), ("RadauIIA5", mk_radau, 0.0),
                 ("Kvaerno3", mk_kvaerno3_rawest, 1e-5), ("Kvaerno5", mk_kvaerno5_rawest, 1e-5), ("KenCarp4", mk_kencarp4_rawest, 1e-5)],
    "mul"    => [("IDA", mk_ida, 0.0), ("FBDF", mk_fbdf, 0.0), ("KenCarp4", mk_kencarp4_rawest, 0.0), ("Rodas6P", mk_rodas6p, 1e-5)],
    "darlington" => [("IDA", mk_ida, 0.0), ("FBDF", mk_fbdf, 0.0), ("Rodas6P", mk_rodas6p, 0.0), ("RadauIIA5", mk_radau, 0.0),
                      ("Kvaerno5", mk_kvaerno5_rawest, 1e-8), ("KenCarp4", mk_kencarp4_rawest, 0.0)],
)
solvers_for(case) = SOLVERS[case]

#------------------------------------------------------------------------------#
# Helpers
#------------------------------------------------------------------------------#
# Step cap for the reltol sweep. Sized to sit well above every solver's largest
# *successful* run in this suite (the current worst is graetz Kvaerno3 at
# reltol=1e-5, ~760k accepted steps) while still killing a stuck solve quickly.
# The old 50_000_000 let a single pathological point churn for ~19 min before
# bailing out (darlington FBDF at reltol=1e-6 walked to the 50M ceiling having
# covered 0.2% of the tspan), which alone pushed the CI job past its 60-minute
# timeout. A run that needs >2M steps here is diverging, not converging - cap it
# and record the MaxIters skip instead of waiting it out. Does not change any
# accepted point (all finish far below this); only bounds the runaways.
const SWEEP_MAXITERS = 2_000_000

setup(builder) = (c = MNACircuit(builder); MNA.assemble!(c); c)

function output_signal(sol, out_nodes)
    length(out_nodes) == 1 && return Vector{Float64}(sol[Symbol(out_nodes[1])])
    return Vector{Float64}(sol[Symbol(out_nodes[1])]) .- Vector{Float64}(sol[Symbol(out_nodes[2])])
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
    elseif case == "darlington"
        return "Darlington pair switch\n" * """
        load "spice/resistor.osdi"
        load "spice/capacitor.osdi"
        load "spice/sn/bjt.osdi"
        model r sp_resistor
        model c sp_capacitor
        model vsource vsource
        model qmod sp_bjt ( bf=100 is=1e-15 cje=10p cjc=5p tf=0.3n )
        vcc (vcc 0) vsource dc=5
        vs (vin 0) vsource dc=0 type="pulse" val0=0 val1=3 rise=10n fall=10n width=0.99u period=2u
        rb (vin b1) r r=10k
        q1 (coll b1 b2 0) qmod
        q2 (coll b2 0 0) qmod
        rbleed (b2 0) r r=10k
        rl (vcc coll) r r=1k
        cl (coll 0) c c=100p
        """
    else
        error("unknown case $case")
    end
end

const VACASK_BIN, VACASK_LIB = locate_vacask()

"""
Parse VACASK's self-reported analysis time from stdout: it prints
`  Elapsed time: <seconds>` right after `Running analysis 'tran1'.`,
unconditionally (no `print stats` needed). This is the transient analysis's own
wall time - it excludes process spawn, netlist parsing, and OSDI model loading,
which is what an external wall-clock around the whole subprocess would include.
Cadnip's timing (via @benchmark on a pre-built circuit) is the equivalent
solve-only figure, so this is the fair, apples-to-apples number to compare.
"""
function parse_vacask_elapsed(stdout_text::AbstractString)
    m = match(r"Elapsed time:\s*([0-9.eE+-]+)", stdout_text)
    m === nothing && error("could not find 'Elapsed time:' in VACASK output")
    return parse(Float64, m.captures[1])
end

"Run VACASK once; return (t, signal, timepoints, runtime_s) or throw on failure."
function run_vacask_once(case, reltol, vntol, tspan, out_nodes; maxstep=tspan[2],
                          method=nothing, maxord=nothing, extra_opts="")
    # `analysis tran ... step=` sets VACASK's INITIAL timestep, not just an
    # output stride - confirmed empirically (see README "Findings about
    # VACASK"): using tspan/n_grid here (previously) silently cost `filter`
    # ~50x accuracy at tight reltol, because a too-coarse first step's error
    # never damps out on that lightly-damped LC circuit, while `rc` was
    # completely unaffected either way (its floor has a different cause).
    # A tiny, tspan-independent initial step costs at most one or two
    # negligibly cheap extra steps before the adaptive controller takes
    # over - confirmed safe and fast on all 4 cases - so there's no
    # accuracy/cost tradeoff to make: always start as fine as possible.
    step = 1e-12
    m = method === nothing ? String(get(CFG, "vacask_tran_method", "gear")) : method
    mo = maxord === nothing ? Int(get(CFG, "vacask_tran_maxord", 5)) : maxord
    sim = sim_body(case) * """
    control
      options reltol=$(reltol) vntol=$(vntol) tran_method="$(m)" tran_maxord=$(mo) $(extra_opts)
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
        out = IOBuffer()
        run(pipeline(cmd; stdout=out, stderr=devnull))
        best = min(best, parse_vacask_elapsed(String(take!(out))))
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

    summary = open(joinpath(OUT, "cadnip_$(case).csv"), "w")
    println(summary, "solver,reltol,median_time_s,steps,rejects,nniter,retcode")
    for (sname, sfn, min_rel) in solvers_for(case), r in reltols
        r < min_rel && continue
        a = r * ascale
        @printf("  cadnip %-14s reltol=%.0e ... ", sname, r); flush(stdout)
        try
            c = setup(builder)
            # auto_tstops=true (tran!'s default) derives PULSE/PWL/SIN source
            # breakpoints automatically - no more hand-derived case_tstops.
            sol = tran!(c, (t0, t1); abstol=a, reltol=r, solver=sfn(),
                        dense=false, maxiters=SWEEP_MAXITERS)
            # A solver can bail out early (e.g. retcode :Unstable) without
            # throwing. Only accept runs that actually reached t1 - otherwise
            # the truncated waveform is not comparable to the others and
            # corrupts the error/work-precision curve.
            reached = sol.retcode == ReturnCode.Success && isapprox(sol.t[end], t1; rtol=1e-6)
            if !reached
                println(summary, "$sname,$r,NaN,0,0,0,$(sol.retcode)")
                @printf("SKIP (%s, reached t=%.4g of %.4g)\n", sol.retcode, sol.t[end], t1)
                flush(summary); flush(stdout)
                continue
            end
            steps = hasproperty(sol.stats, :naccept) && sol.stats.naccept > 0 ? sol.stats.naccept : length(sol.t)
            rejects = hasproperty(sol.stats, :nreject) ? sol.stats.nreject : 0
            nniter = hasproperty(sol.stats, :nnonliniter) ? sol.stats.nnonliniter : 0
            write_wave(joinpath(OUT, "cadnip_$(case)_$(sname)_$(reltol_tag(r)).csv"),
                       sol.t, output_signal(sol, out_nodes))
            ct = setup(builder)
            bench = @benchmark tran!($ct, ($t0, $t1); abstol=$a, reltol=$r, solver=$(sfn()),
                                     dense=false, maxiters=$SWEEP_MAXITERS) samples=3 evals=1 seconds=120
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

"""
Dense Cadnip-tight golden (robust IDA) for cases VACASK can't reference.

Tries progressively looser tolerances starting from `1e-11` - a stiff circuit
(e.g. the diode multiplier) can fail to converge at the tightest tolerance even
with IDA, so back off until one actually solves to `t1`. Errors if even the
loosest of these fails, since the case's pinned golden must exist.
"""
function cadnip_golden(case, spec)
    builder = BUILDERS[case]
    t0, t1 = Float64(spec["tspan"][1]), Float64(spec["tspan"][2])
    out_nodes = String.(spec["output"])
    grid = collect(range(t0, t1; length=Int(CFG["n_grid"]) * 50))
    for reltol in (1e-11, 1e-10, 1e-9, 1e-8, 1e-7)
        @printf("  cadnip golden reltol=%.0e ... ", reltol); flush(stdout)
        try
            c = setup(builder)
            sol = tran!(c, (t0, t1); abstol=reltol, reltol=reltol, solver=mk_ida(),
                        saveat=grid, maxiters=100_000_000)
            if sol.retcode == ReturnCode.Success && isapprox(sol.t[end], t1; rtol=1e-6)
                write_wave(joinpath(OUT, "cadnip_ref_$(case).csv"), sol.t, output_signal(sol, out_nodes))
                println("ok"); flush(stdout)
                return
            end
            println("failed ($(sol.retcode)), backing off"); flush(stdout)
        catch e
            println("failed (", first(sprint(showerror, e), 80), "), backing off"); flush(stdout)
        end
    end
    error("case $case: Cadnip golden failed to converge at every tolerance tried")
end

"""
`tag`/`maxord_force` add a second, independent VACASK series without
touching the primary one: pass `tag="_gear2", maxord_force=2` to sweep
gear2 (2nd-order Gear/BDF, A-stable) instead of whatever order the case's
`vacask_override`/`vacask_tran_maxord` picks. Output files are namespaced
by `tag` so the two series never collide. `want_golden` is only ever true
for the primary (tag="") call - the golden is independent of which
comparison curves get plotted against it.
"""
function run_vacask_sweep(case, spec, want_golden; tag::String="", maxord_force=nothing)
    t0, t1 = Float64(spec["tspan"][1]), Float64(spec["tspan"][2])
    out_nodes = String.(spec["output"])
    reltols = Float64.(CFG["reltols"])
    ascale = Float64(CFG["abstol_scale"])
    ref_reltol = Float64(CFG["ref_reltol"]); ref_abstol = Float64(CFG["ref_abstol"])
    ref_factor = Float64(get(CFG, "ref_maxstep_factor", 50))

    # A case can override VACASK's default (unbounded maxstep, benchmark's
    # standard tran_method=gear/tran_maxord=5, no extra options) via
    # config.json's `vacask_override` - see per-case `_vacask_override_comment`
    # for why. This is the ONE VACASK run plotted for the case, not an extra
    # series alongside the raw default - a case either needs a fair override
    # or it doesn't, there's no value in also showing the version we already
    # know is unrepresentative. The golden (tight-maxstep) run below applies
    # the same method/maxord/extra_opts (but keeps its own fine maxstep, not
    # `oms`) - a circuit that needs e.g. nr_residualcheck=0 to avoid aborting
    # in the reltol sweep needs it just as much for its own golden (mul's
    # golden aborted before this was applied here). `maxord_force` (the
    # gear2 comparison series) overrides the order only - any maxstep/
    # extra_opts a case needs to avoid aborting still apply regardless of
    # order (confirmed on `mul`: the residualcheck abort happens at every
    # order 1-5 alike).
    ov = get(spec, "vacask_override", Dict{String,Any}())
    oms = haskey(ov, "maxstep") ? Float64(ov["maxstep"]) : t1
    omethod = haskey(ov, "method") ? String(ov["method"]) : nothing
    omaxord = maxord_force !== nothing ? maxord_force : (haskey(ov, "maxord") ? Int(ov["maxord"]) : nothing)
    oextra = String(get(ov, "extra_opts", ""))

    if want_golden
        ms = (t1 - t0) / (Int(CFG["n_grid"]) * ref_factor)
        @printf("  vacask golden reltol=%.0e maxstep=%.1e ... ", ref_reltol, ms); flush(stdout)
        ti, sig, tp, rt = run_vacask_once(case, ref_reltol, ref_abstol, (t0, t1), out_nodes;
                                           maxstep=ms, method=omethod, maxord=omaxord, extra_opts=oextra)
        write_wave(joinpath(OUT, "ref_$(case).csv"), ti, sig)
        @printf("%.3fs %d pts\n", rt, tp); flush(stdout)
    end

    summary = open(joinpath(OUT, "vacask$(tag)_$(case).csv"), "w")
    println(summary, "reltol,time_s,timepoints")
    for r in reltols
        @printf("  vacask%s reltol=%.0e ... ", tag, r); flush(stdout)
        try
            ti, sig, tp, rt = run_vacask_once(case, r, r * ascale, (t0, t1), out_nodes;
                                               maxstep=oms, method=omethod, maxord=omaxord, extra_opts=oextra)
            write_wave(joinpath(OUT, "vacask$(tag)_$(case)_$(reltol_tag(r)).csv"), ti, sig)
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
"""
`golden="self"` means each simulator is scored against its OWN tight
reference rather than one shared cross-simulator golden - see README
"Findings about VACASK" on `mul`: a Cadnip-vs-VACASK gap that looks like a
tolerance-tuning floor turned out to be VACASK converging cleanly to an
answer that just differs from Cadnip's by a small, fixed amount (most likely
the two simulators' separately-compiled diode models, not a solver-accuracy
problem on either side). Scoring each simulator against a foreign golden
would silently fold that gap into "error", overstating whichever simulator
doesn't own the shared golden. `sim` picks which tight reference applies to
`self`; it's ignored for the other golden kinds since they're already
simulator-agnostic (analytic) or already only exist for one simulator.
"""
function load_golden(case, spec, sim::Symbol)
    g = String(spec["golden"])
    eff = g == "self" ? (sim == :cadnip ? "cadnip" : "vacask") : g
    if eff == "analytic"
        return read_wave(joinpath(OUT, "analytic_$(case).csv"))..., "analytic (exact)"
    elseif eff == "vacask"
        return read_wave(joinpath(OUT, "ref_$(case).csv"))..., "VACASK (tight)"
    elseif eff == "cadnip"
        return read_wave(joinpath(OUT, "cadnip_ref_$(case).csv"))..., "Cadnip IDA (tight)"
    else
        error("case $case: unknown golden '$g' (use analytic|vacask|cadnip|self)")
    end
end

function analyze(case, spec)
    cgt, cgv, cgsrc = load_golden(case, spec, :cadnip)
    vgt, vgv, vgsrc = load_golden(case, spec, :vacask)
    gsrc = cgsrc == vgsrc ? cgsrc : "$cgsrc (Cadnip) / $vgsrc (VACASK) - each scored against its own"
    println("$case: golden = $gsrc")
    curves = Dict{String,Vector{Tuple{Float64,Float64}}}()
    table = Tuple{String,Float64,Float64,Float64}[]

    for row in first(read_table(joinpath(OUT, "cadnip_$(case).csv")))
        solver = row["solver"]; r = parse(Float64, row["reltol"])
        t = tryparse(Float64, get(row, "median_time_s", "NaN"))
        wp = joinpath(OUT, "cadnip_$(case)_$(solver)_$(reltol_tag(r)).csv")
        isfile(wp) || continue
        tw, vw = read_wave(wp)
        err = run_error(tw, vw, cgt, cgv)
        isfinite(err) && t !== nothing && isfinite(t) &&
            push!(get!(curves, "Cadnip $solver", Tuple{Float64,Float64}[]), (err, t))
        push!(table, ("Cadnip $solver", r, err, something(t, NaN)))
    end

    # Two VACASK series: the primary (case's own picked order/overrides) and
    # a fixed gear2 comparison (2nd-order Gear/BDF, A-stable - the order the
    # maintainer says circuit simulators historically stick to). Both are
    # scored against the SAME golden (vgt/vgv) - the question is how an
    # A-stable low-order method scales against whatever order this case
    # actually uses, not against a different truth.
    for (label, tag) in (("VACASK", ""), ("VACASK gear2", "_gear2"))
        vpath = joinpath(OUT, "vacask$(tag)_$(case).csv")
        isfile(vpath) || continue
        for row in first(read_table(vpath))
            r = parse(Float64, row["reltol"]); t = tryparse(Float64, get(row, "time_s", "NaN"))
            wp = joinpath(OUT, "vacask$(tag)_$(case)_$(reltol_tag(r)).csv")
            isfile(wp) || continue
            tw, vw = read_wave(wp)
            err = run_error(tw, vw, vgt, vgv)
            isfinite(err) && t !== nothing && isfinite(t) &&
                push!(get!(curves, label, Tuple{Float64,Float64}[]), (err, t))
            push!(table, (label, r, err, something(t, NaN)))
        end
    end

    # cross-check: whenever two independent tight references exist for this
    # case (analytic+VACASK, or - under golden="self" - Cadnip-tight+VACASK-
    # tight), report their mutual agreement. This is the number that matters
    # for `self` cases: it's the open, unexplained gap between the two
    # simulators' converged answers, kept separate from either curve's error.
    ap = joinpath(OUT, "analytic_$(case).csv"); rp = joinpath(OUT, "ref_$(case).csv")
    cp = joinpath(OUT, "cadnip_ref_$(case).csv")
    xcheck = ""
    if isfile(ap) && isfile(rp)
        ta, va = read_wave(ap); tr, vr = read_wave(rp)
        xcheck = @sprintf("analytic vs VACASK-tight cross-check: rel-L2 = %.2e", run_error(tr, vr, ta, va))
        println("  ", xcheck)
    elseif isfile(cp) && isfile(rp)
        tc, vc = read_wave(cp); tr, vr = read_wave(rp)
        xcheck = @sprintf("Cadnip-tight vs VACASK-tight cross-check: rel-L2 = %.2e (open item, not a tolerance-tuning gap - see README)", run_error(tr, vr, tc, vc))
        println("  ", xcheck)
    end
    return gsrc, curves, table, xcheck
end

"""
Fixed label -> style-index mapping so the same solver (or VACASK) gets the
same marker/color in every case's plot, not just a consistent position
within whichever subset of solvers happens to run on that particular case
(e.g. RadauIIA5 would otherwise shift slots between filter/rc, which also
run Kvaerno5, and graetz, which doesn't). Covers every label `solvers_for`
or the VACASK sweep can produce; unrecognized labels fall back to the next
free slot rather than erroring, so a new solver added to `SOLVERS` degrades
gracefully instead of blowing up plotting.
"""
const SERIES_ORDER = ["Cadnip IDA", "Cadnip FBDF", "Cadnip Rodas5P", "Cadnip Rodas6P",
                       "Cadnip Kvaerno5", "Cadnip RadauIIA5", "Cadnip KenCarp4", "VACASK",
                       "VACASK gear2"]
series_style_index(label) = something(findfirst(==(label), SERIES_ORDER), length(SERIES_ORDER) + 1)

"""
Distinct pure-ASCII markers, one per series, so curves stay distinguishable even
without color (color support in GITHUB_STEP_SUMMARY is unconfirmed, so it's kept
off - see README). `canvas=AsciiCanvas`/`border=:ascii` are used for the same
reason: named UnicodePlots markers/canvases (Braille dots, box-drawing borders)
have known cross-font/cross-renderer width bugs that misalign the plot; plain
ASCII (0-127) is guaranteed single-column in any monospace font.
"""
const ASCII_MARKERS = ["o", "x", "+", "*", "#", "@", "%", "&", "="]

function ascii_plot(title, curves)
    labels = sort(collect(keys(curves)))
    filter!(l -> !isempty(curves[l]), labels)
    isempty(labels) && return "(no data)"

    # scatterplot!'s axis range is fixed by the FIRST series plotted (it does not
    # auto-rescale for later series), so points from other series outside that
    # range are silently dropped. Compute the range across ALL series up front,
    # with margin - points exactly at the raw extrema are also clipped - and pass
    # it explicitly to the first scatterplot() call.
    allx = Float64[]; ally = Float64[]
    for l in labels, (e, t) in curves[l]
        push!(allx, e); push!(ally, t)
    end
    xlim = (minimum(allx) / 2, maximum(allx) * 2)
    ylim = (minimum(ally) / 2, maximum(ally) * 2)

    plt = nothing
    for label in labels
        pts = sort(curves[label])
        x = Float64[p[1] for p in pts]; y = Float64[p[2] for p in pts]
        marker = ASCII_MARKERS[mod1(series_style_index(label), length(ASCII_MARKERS))]
        # Embed the marker in the legend text itself ("[o] Cadnip IDA") - the
        # legend otherwise lists only names, with no way to tell which marker
        # glyph belongs to which series once color is off.
        legend_name = "[$marker] $label"
        if plt === nothing
            plt = scatterplot(x, y; name=legend_name, xlabel="rel-L2 error", ylabel="runtime s",
                              title=title, xscale=:log10, yscale=:log10,
                              width=64, height=16, canvas=AsciiCanvas, border=:ascii,
                              marker=marker, xlim=xlim, ylim=ylim, unicode_exponent=false)
        else
            scatterplot!(plt, x, y; name=legend_name, marker=marker)
        end
    end
    io = IOBuffer()
    show(IOContext(io, :color => false), plt)
    return String(take!(io))
end

"""
High-quality PNG/SVG work-precision plot (real color, proper fonts, vector
curves) for the downloaded artifact - complements the plain-ASCII plot
embedded in the job summary (`ascii_plot`), which is legible without
downloading anything but is inherently coarser. Written to
`PLOTS_DIR/<case>.{png,svg}`; same underlying (error, runtime) data as the
ASCII plot and the per-case table.
"""
function save_plot(case, title, curves)
    labels = sort(collect(keys(curves)))
    filter!(l -> !isempty(curves[l]), labels)
    isempty(labels) && return

    plt = Plots.plot(; xscale=:log10, yscale=:log10, xlabel="relative L2 error",
                      ylabel="runtime (s)", title="Work-precision: $title",
                      legend=:outertopright, size=(900, 600), dpi=150)
    markers = (:circle, :xcross, :rect, :diamond, :utriangle, :star5, :pentagon, :hexagon, :cross)
    colors = (:dodgerblue, :orangered, :seagreen, :purple, :goldenrod, :teal, :magenta, :black, :brown)
    for label in labels
        pts = sort(curves[label])
        x = Float64[p[1] for p in pts]; y = Float64[p[2] for p in pts]
        k = mod1(series_style_index(label), length(markers))
        Plots.plot!(plt, x, y; label=label, marker=markers[k], color=colors[k],
                    markersize=6, linewidth=2)
    end
    for ext in ("png", "svg")
        Plots.savefig(plt, joinpath(PLOTS_DIR, "$(case).$(ext)"))
    end
end

function report(results)
    io = IOBuffer()
    println(io, "# Work-Precision Diagram Results\n")
    println(io, "Each solver runs *adaptively* across a tolerance sweep (no forced timestep).")
    println(io, "Error = relative L2 of the output node at each run's own timepoints vs the")
    println(io, "golden reference. VACASK uses Gear/BDF, order picked per case (see config.json)")
    println(io, "since a fixed order isn't uniformly best across circuits (issue #83).\n")
    for (case, gsrc, curves, table, xcheck) in results
        spec = CFG["cases"][case]
        println(io, "## $(spec["title"])\n")
        println(io, "Golden reference: **$gsrc**.", isempty(xcheck) ? "" : " ($xcheck)", "\n")
        if String(spec["golden"]) == "self"
            println(io, "*Each simulator is scored against its own tight reference, not a shared*")
            println(io, "*cross-simulator golden - the cross-check figure above is the open gap*")
            println(io, "*between the two, not error attributable to either curve below.*\n")
        end
        if !isempty(curves)
            # GitHub renders ANSI SGR color codes in ```ansi fences for
            # issues/PRs/READMEs, but confirmed NOT in GITHUB_STEP_SUMMARY (shows
            # raw escape-code garbage there instead) - use a plain fence and rely
            # on the distinct ASCII markers (o/x/+/*) to keep series legible.
            println(io, "```")
            println(io, ascii_plot(String(case), curves))
            println(io, "```\n")
            save_plot(String(case), spec["title"], curves)
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
        elseif golden == "cadnip" || golden == "self"
            # "self" needs its own Cadnip-tight golden too - Cadnip's curves
            # must never be scored against VACASK's answer or vice versa.
            cadnip_golden(case, spec)
        end

        # VACASK sweep always runs where it can; also produce the VACASK golden
        # when this case is pinned to it OR uses "self" (which needs both
        # goldens, not just one). The gear2 comparison series never builds a
        # golden of its own - it's scored against the same one as the primary.
        if VACASK_BIN !== nothing
            run_vacask_sweep(case, spec, golden == "vacask" || golden == "self")
            run_vacask_sweep(case, spec, false; tag="_gear2", maxord_force=2)
        elseif golden == "vacask"
            error("case $case pinned to VACASK golden but VACASK binary not found")
        end

        push!(results, (String(case), analyze(case, spec)...))
    end
    report(results)
end

main()
