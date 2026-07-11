#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: C6288 16x16 Multiplier
#
# A 16x16 bit multiplier circuit using PSP103 MOSFETs (212228 variables).
#
# STATUS: Uses CedarDCOp initialization (DC solve with GMIN/source stepping)
#
# Notes:
# ngspice's 'uic' skips DC operating point analysis and starts transient
# integration directly from u=0, relying on its own per-step Newton retry
# with timestep halving to work through the harsh startup (vdd hard-on at
# t=0). SciML's DAE initialization framework requires a consistent t=0
# state up front rather than tolerating repeated per-step failures, so the
# naive translation of 'uic' -- CedarUICOp's fixed-dt pseudo-transient
# relaxation -- cannot take even a single non-degenerate step here (Newton
# fails identically at every dt, including with unlimited shrinking room).
# CedarDCOp's existing GMIN-stepping/source-stepping homotopy chain (the
# same machinery vacask/ngspice use to find a DC operating point for
# awkward circuits) does find a usable t=0 state, and the transient
# proceeds normally from there. See doc/c6288_bottleneck_findings.md.
#
# FBDF is deliberately NOT offered here: bare `FBDF()` throws OutOfMemoryError
# inside `jacobian2W!` at this scale -- its own default `autodiff` resolves to
# `AutoSparse{AutoForwardDiff, KnownJacobianSparsityDetector,
# GreedyColoringAlgorithm}`, which ignores the analytic `jac=` we supply and
# rebuilds/colors its own Jacobian instead; combining that with the mass
# matrix inside `jacobian2W!`'s generic sparse broadcast blows up memory at
# n=212228. `autodiff=AutoFiniteDiff()` does not fix it either -- it only
# changes which backend gets wrapped in the same `AutoSparse` + coloring
# reconstruction. See doc/c6288_bottleneck_findings.md ("IDA and FBDF needed
# their own solver-option fixes").
#
# Usage: julia -O0 runme.jl [output_file]
#   Runs every solver in SOLVERS_C6288 (IDA, Rodas5P) and writes a markdown
#   comparison table (same format as run_benchmarks.jl) to output_file, or
#   stdout if omitted. Requires `-O0`: the generated builder (2419 inlined
#   subckt calls) hits LLVM RAGreedy's super-linear blowup at default -O2.
#   Set $VACASK_REFERENCE_TSV to fold in the real VACASK "C6288 Multiplier"
#   reference row (see run_vacask.sh).
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Cadnip.MNA: CedarDCOp
using Sundials: IDA
using OrdinaryDiffEqRosenbrock: Rodas5P
using BenchmarkTools
using SciMLBase: ReturnCode
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "report_utils.jl"))

const BENCHMARK_NAME = "C6288 Multiplier"

# linear_solver=:KLU is required for IDA at this scale: Sundials' own default
# dense solver allocates an n x n matrix (n=212228 -> ~360GB), which doesn't
# raise a catchable OutOfMemoryError -- it segfaults inside SUNMatZero_Dense.
# Rodas5P needs no such override: its Rosenbrock `calc_J`/`calc_J!` path calls
# `f.jac` directly when `has_jac(f)` is true, so it uses the sparse analytic
# Jacobian `tran!` already threads through rather than building its own dense
# default. It does NOT OOM, but it is far from free: measured at ~30-50GB
# (vs IDA's ~11GB) at reltol=1e-5, and ~4-5x slower wall-clock than IDA. A
# standard CI runner may not have that much memory headroom -- see the
# incremental-write comment below for how a hard OOM-kill on this solver is
# handled without losing the IDA row already collected.
const SOLVERS_C6288 = [
    ("IDA", () -> IDA(linear_solver=:KLU, max_nonlinear_iters=100, max_error_test_failures=20)),
    ("Rodas5P", () -> Rodas5P()),
]

# Import pre-parsed PSP103 model from PSPModels package
using PSPModels
println("PSP103VA loaded from PSPModels package")

# Parse SPICE file, inject PSP103VA module as Tier-2 scope so `.model` cards
# referring to PSP103VA resolve. Codegen runs at top level (no world-age tax).
const spice_file = joinpath(@__DIR__, "runme.sp")
let ast = Cadnip.NyanSpectreNetlistParser.parsefile(spice_file; start_lang=:spice, implicit_title=true),
    sema_result = Cadnip.sema(ast; imported_hdl_modules=[PSP103VA_module])
    eval(Cadnip._make_mna_circuit_with_sema(sema_result; circuit_name=:c6288_circuit))
end

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
"""
function setup_simulation()
    circuit = MNACircuit(c6288_circuit)
    MNA.assemble!(circuit)
    return circuit
end

# reltol=1e-5 (tightened from an earlier 1e-3 default): at 1e-3 IDA/Rodas5P
# accept only ~18-56 steps over the 2ns window versus VACASK/ngspice/xyce's
# ~1000-1024 (see benchmarks/vacask/README.md's c6288 table) -- three orders
# of magnitude fewer, raising the question of whether Cadnip is resolving the
# multiplier's switching activity or adaptively stepping past it. Verified by
# sweeping reltol from 1e-3 to 1e-6 (56/70/118/139 accepted steps for IDA):
# the *converged* output at t=2ns is bit-for-bit identical (to mV precision)
# at every reltol, and Rodas5P at reltol=1e-5 converges to the same values
# via an independent solver family/step-size-control algorithm. So the coarse
# reltol=1e-3 trajectory was NOT skipping switching transitions -- the answer
# is reltol-independent. The residual mismatch against the fully-settled
# 0xFFFF*0xFFFF=0xFFFE0001 product (p0-p16 read high instead of the expected
# pattern) reflects that 2ns is short relative to this multiplier's full
# carry-chain propagation depth (2419 gates), not a solver artifact -- the
# real reference simulators only run this same 2ns window too.
#
# reltol=1e-5 is chosen as the point of diminishing returns: three more
# orders of tightening (1e-3 -> 1e-6) only bought ~2.5x more accepted steps
# (56 -> 139) for IDA, so chasing exact 1000-step parity with VACASK via
# reltol alone is not practical at this scale/cost. 1e-5 roughly doubles
# resolution over the old default while keeping wall-clock reasonable for CI.
function run_benchmark(solver; reltol=1e-5, abstol=1e-6, maxiters=10_000_000)
    tspan = (0.0, 2e-9)  # 2ns simulation (same window VACASK/ngspice/xyce use)
    solver_name = nameof(typeof(solver))

    # Setup the simulation outside the timed region
    circuit = setup_simulation()
    n = MNA.system_size(circuit)
    println("Circuit size: $n variables")

    # Use CedarDCOp for initialization: the GMIN/source-stepping fallback
    # chain in _dc_solve_with_fallbacks finds a usable operating point for
    # this circuit where CedarUICOp's pseudo-transient warmup cannot take a
    # single step. See doc/c6288_bottleneck_findings.md.
    init = CedarDCOp()

    # abstol=1e-6 (looser than tran!'s default of 1e-10, which is unreachable
    # for a 212k-variable circuit and was driving dt down toward the
    # femtosecond scale) directly explained IDA's original hmin
    # corrector-convergence failure.
    #
    # Deliberately NOT using force_dtmin/unstable_check here, unlike the ring
    # oscillator benchmark. Ring uses CedarTranOp (a homotopy warmup with no
    # real DC solve), so it genuinely needs to push through switching
    # transitions with no valid intermediate state. c6288 uses CedarDCOp --
    # a proper GMIN/source-stepping DC operating point solve -- so forcing
    # through steps that don't converge isn't crossing a
    # known-hard-but-tractable region the way it does for ring; empirically
    # it just burns CPU time for hours without reaching t=2ns. See
    # doc/c6288_bottleneck_findings.md.
    println("\nBenchmarking transient analysis with $solver_name (reltol=$reltol, abstol=$abstol)...")
    bench = @benchmark tran!($circuit, $tspan; solver=$solver, reltol=$reltol, abstol=$abstol,
                              maxiters=$maxiters, initializealg=$init, dense=false) samples=3 evals=1 seconds=300

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; solver=solver, reltol=reltol, abstol=abstol, maxiters=maxiters,
                initializealg=init, dense=false)

    println("\n=== Results ($solver_name) ===")
    println("Status:     $(sol.retcode)")
    @printf("Timepoints: %d\n", length(sol.t))
    @printf("Final time: %.3e s (target %.3e s)\n", sol.t[end], tspan[2])
    @printf("NR iters:   %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

function make_solver(solver_name)
    for (name, solver_fn) in SOLVERS_C6288
        name == solver_name && return solver_fn()
    end
    error("Unknown solver: $solver_name. Use " * join(first.(SOLVERS_C6288), ", "))
end

function result_for_solver(solver_name)
    println("Running $BENCHMARK_NAME with $solver_name...")
    try
        bench, sol = run_benchmark(make_solver(solver_name))
        if bench === nothing || sol === nothing
            return BenchmarkResult(BENCHMARK_NAME, solver_name, :failed, "Benchmark returned nothing")
        end

        tspan_end = sol.prob.tspan[2]
        reached_end = isapprox(sol.t[end], tspan_end; rtol=1e-6)

        warning = ""
        if sol.retcode != ReturnCode.Success
            warning = " ($(sol.retcode))"
            @warn "$BENCHMARK_NAME with $solver_name: $(sol.retcode)"
        end

        if !reached_end
            return BenchmarkResult(BENCHMARK_NAME, solver_name, :failed,
                "Stopped at t=$(sol.t[end]), expected $(tspan_end)")
        end

        rejected = hasproperty(sol.stats, :nreject) ? sol.stats.nreject : 0
        nr_iters = hasproperty(sol.stats, :nnonliniter) ? sol.stats.nnonliniter : 0

        return BenchmarkResult(
            BENCHMARK_NAME, solver_name, :success,
            median(bench.times) / 1e9, minimum(bench.times) / 1e9, maximum(bench.times) / 1e9,
            bench.memory / 1e6, bench.allocs, length(sol.t), rejected, nr_iters, warning
        )
    catch e
        return BenchmarkResult(BENCHMARK_NAME, solver_name, :failed, sprint(showerror, e))
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    known_solvers = first.(SOLVERS_C6288)
    if length(ARGS) >= 1 && ARGS[1] in known_solvers
        # Single-solver debug mode: verbose run_benchmark() output only, no
        # markdown table.
        run_benchmark(make_solver(ARGS[1]))
    else
        # Full report mode (default): run every solver, emit the same
        # markdown table format as run_benchmarks.jl, folding in the VACASK
        # reference row for BENCHMARK_NAME if $VACASK_REFERENCE_TSV is set.
        #
        # Rodas5P's memory estimate at this scale runs ~30GB (vs IDA's
        # ~11GB) -- comfortably beyond what a standard CI runner may have
        # free. If the OS OOM-kills the process outright (not a catchable
        # Julia OutOfMemoryError), nothing after that point runs -- so the
        # report is (re)written after every solver, not just at the end,
        # meaning a hard crash on a later solver still leaves prior results
        # on disk instead of losing the whole report.
        results = BenchmarkResult[]
        function write_report(output_arg)
            markdown = generate_markdown(results; title="C6288 Multiplier Benchmark Results")
            if output_arg !== nothing
                open(output_arg, "w") do f
                    write(f, markdown)
                end
            end
            return markdown
        end

        output_arg = length(ARGS) >= 1 ? ARGS[1] : nothing
        for name in known_solvers
            push!(results, result_for_solver(name))
            write_report(output_arg)
        end
        for ref in load_vacask_reference()
            ref.name == BENCHMARK_NAME && push!(results, ref)
        end

        markdown = write_report(output_arg)
        if output_arg !== nothing
            println("Report written to: $(ARGS[1])")
        else
            println()
            println(markdown)
        end
    end
end
