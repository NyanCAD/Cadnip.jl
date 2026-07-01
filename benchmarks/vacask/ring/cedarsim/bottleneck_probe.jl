#!/usr/bin/env julia
#==============================================================================#
# Ring oscillator bottleneck probe: RHS (stamp) evaluation vs linear solve
# vs Newton-iteration count.
#
# The PSP103 ring oscillator (runme.jl) needs indirection to even compile
# (see doc/ring_oscillator_investigation.md), and historically appeared to
# "hang and blow up". This script isolates, for a given MOSFET compact model,
# three independent costs so we can tell which one actually drives wall time:
#
#   1. Per-call cost of the RHS/stamp evaluation (fast_rebuild! + matvec)
#   2. Per-call cost of a from-scratch sparse LU factorize+solve on the
#      resulting Jacobian (an upper-bound stand-in for the KLU factorize+solve
#      IDA/FBDF do per Newton iteration -- real solves reuse the symbolic
#      factorization, so this over-estimates the true linear-algebra cost)
#   3. The actual number of Newton iterations/timesteps needed to cover a
#      given span of simulated time (reflects how "stiff"/well-conditioned
#      the model's dynamics are, independent of per-call cost)
#
# level=psp103 reuses VACASKModels' precompiled PSP103 builder and the real
# benchmark topology (runme.sp: no load caps, current-pulse kickstart, FBDF).
# level=mos1/mos6/bsim3/bsim4 build the same 9-stage inverter-chain topology
# via VADistillerModels' ModelRegistry (.model level=N), with small explicit
# 10fF load caps (these models don't have PSP103's ~1fF internal parasitics
# to self-oscillate against with the same kickstart, so caps put them on
# equal footing) and use tran!'s default IDA+CedarTranOp path.
#
# Usage: julia --project=../../../../benchmarks bottleneck_probe.jl <level>
#   level in: mos1 mos6 bsim3 bsim4 psp103
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Cadnip.MNA: CedarTranOp
using OrdinaryDiffEqBDF: FBDF
using ADTypes: AutoFiniteDiff
using SciMLBase
using LinearAlgebra
using Printf

const LEVEL = length(ARGS) >= 1 ? ARGS[1] : "mos1"

println("="^70)
println("Ring oscillator bottleneck probe -- level=$LEVEL")
println("="^70)

#==============================================================================#
# Circuit construction (level-specific)
#==============================================================================#

if LEVEL == "psp103"
    using VACASKModels
    const spice_file = joinpath(@__DIR__, "runme.sp")

    t_build = @elapsed let ast = Cadnip.NyanSpectreNetlistParser.parsefile(spice_file; start_lang=:spice, implicit_title=true),
            sema_result = Cadnip.sema(ast; imported_hdl_modules=[VACASKModels])
        eval(Cadnip._make_mna_circuit_with_sema(sema_result; circuit_name=:ring_circuit))
    end
    make_circuit() = MNACircuit(ring_circuit)
    tspan = (0.0, 20e-9)          # PSP103 needs many more NR iters/ns -- keep short
    dtmax = 0.05e-9
    solver = FBDF(autodiff=AutoFiniteDiff())
    tran_kwargs = (; dtmax, solver, initializealg=CedarTranOp(), dense=false,
                     force_dtmin=true, abstol=1e-4, reltol=1e-2,
                     unstable_check=(dt,u,p,t)->false)
else
    using VADistillerModels

    const MODEL_CARD = Dict(
        "mos1"  => """
        .model nmosX nmos level=1 vto=0.7 kp=100e-6
        .model pmosX pmos level=1 vto=-0.7 kp=50e-6
        """,
        "mos6"  => """
        .model nmosX nmos level=6 vto=0.7 u0=600.0 tox=10e-9
        .model pmosX pmos level=6 vto=-0.7 u0=250.0 tox=10e-9
        """,
        "bsim3" => """
        .model nmosX nmos level=8
        .model pmosX pmos level=8
        """,
        "bsim4" => """
        .model nmosX nmos level=14
        .model pmosX pmos level=14
        """,
    )
    haskey(MODEL_CARD, LEVEL) || error("unknown level $LEVEL, must be one of psp103 $(keys(MODEL_CARD))")

    netlist = """
    * 9-stage ring oscillator ($LEVEL)
    $(MODEL_CARD[LEVEL])
    Vdd vdd 0 DC 3.3
    $(join(("""
    MP$i $(i%9+1) $i vdd vdd pmosX w=2u l=1u
    MN$i $(i%9+1) $i 0   0   nmosX w=1u l=1u
    C$i $(i%9+1) 0 10f
    """ for i in 1:9), ""))
    .end
    """

    t_build = @elapsed circuit = Cadnip.MNACircuit(netlist; lang=:spice, source_dir=@__DIR__)
    make_circuit() = Cadnip.MNACircuit(netlist; lang=:spice, source_dir=@__DIR__)
    tspan = (0.0, 100e-9)
    dtmax = 1e-9
    tran_kwargs = (; dtmax, initializealg=CedarTranOp(use_shampine=true))
end

@printf("Circuit construction (parse+sema+eval, first call):  %.2f s\n", t_build)
circuit = @isdefined(circuit) ? circuit : make_circuit()

t_probfirst = @elapsed prob = SciMLBase.ODEProblem(circuit, tspan)
@printf("ODEProblem construction (structure/JIT, first call): %.2f s\n", t_probfirst)

n = length(prob.u0)
println("System size (unknowns): $n")

#==============================================================================#
# 1+2. Isolated RHS-eval / Jacobian-fill / linear-solve costs
#==============================================================================#

rhs! = prob.f.f
jac! = prob.f.jac
ws = prob.p

du = similar(prob.u0)
u0 = fill(0.6, n)  # away from the exact symmetric fixed point

rhs!(du, u0, ws, 0.0)  # warm up / trigger JIT
J = similar(prob.f.jac_prototype)
jac!(J, u0, ws, 0.0)

n_rhs = 2000
t0 = time()
for _ in 1:n_rhs
    rhs!(du, u0, ws, 0.0)
end
t_rhs = (time() - t0) / n_rhs
@printf("Per-call RHS/stamp evaluation:      %8.3f us\n", t_rhs * 1e6)

t0 = time()
for _ in 1:n_rhs
    jac!(J, u0, ws, 0.0)
end
t_jac = (time() - t0) / n_rhs
@printf("Per-call analytic Jacobian fill:    %8.3f us\n", t_jac * 1e6)

# Naive from-scratch sparse LU: over-estimates real per-iteration linear
# solve cost (KLU reuses the symbolic factorization across iterations/steps).
Jd = -J
b = ones(n)
n_lin = 200
t0 = time()
for _ in 1:n_lin
    F = lu(Jd)
    ldiv!(similar(b), F, b)
end
t_lin = (time() - t0) / n_lin
@printf("Per-call sparse LU factorize+solve: %8.3f us  (upper bound, see note above)\n", t_lin * 1e6)

#==============================================================================#
# 3. Real transient run -- first call pays for JIT of the DAE/ODE solver's
#    own residual!/jacobian! (a *different* compiled path from the rhs!/jac!
#    above), second call on the same builder is steady state.
#==============================================================================#

t_tran = NaN
niter = 0
try
    println("\nRunning transient, call #1 (JIT + solve)...")
    t_tran1 = @elapsed sol1 = tran!(circuit, tspan; tran_kwargs...)
    @printf("Call #1 wall time: %.3f s  (status=%s, timepoints=%d)\n",
            t_tran1, sol1.retcode, length(sol1.t))

    println("Running transient, call #2 (steady-state, JIT amortized)...")
    # Reuse the SAME circuit/builder (not make_circuit()): a fresh MNACircuit()
    # call generates a gensym'd builder of a new type, forcing a full recompile
    # of compile_structure/create_workspace/fast_rebuild! all over again.
    global t_tran = @elapsed sol = tran!(circuit, tspan; tran_kwargs...)

    @printf("Transient wall time:  %.3f s\n", t_tran)
    println("Status:     $(sol.retcode)")
    println("Full stats: ", sol.stats)
    @printf("Timepoints: %d\n", length(sol.t))
    if sol.stats !== nothing && hasproperty(sol.stats, :nnonliniter)
        global niter = sol.stats.nnonliniter
        @printf("NR iters:   %d  (%.2f iter/step)\n", niter, niter / length(sol.t))
        if niter > 0
            @printf("Wall time per NR iter (actual, incl. all overhead): %.3f us\n", t_tran / niter * 1e6)
            @printf("  isolated RHS-eval cost:  %8.3f us (%5.1f%% of actual)\n", t_rhs*1e6, 100*t_rhs*niter/t_tran)
            @printf("  isolated linsolve cost:  %8.3f us (%5.1f%% of actual)\n", t_lin*1e6, 100*t_lin*niter/t_tran)
        end
    end
catch e
    println("\nTransient run FAILED: ", sprint(showerror, e)[1:min(end, 500)])
end

println("\n=== SUMMARY ($LEVEL, n=$n) ===")
@printf("build+jit=%.2fs  rhs=%.3fus  jac=%.3fus  linsolve=%.3fus  tran(%gns)=%.3fs  niter=%s  ms/ns=%.4g\n",
        t_build + t_probfirst, t_rhs*1e6, t_jac*1e6, t_lin*1e6, tspan[2]*1e9, t_tran, niter,
        1e3 * t_tran / (tspan[2]*1e9))
