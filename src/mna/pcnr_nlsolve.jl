#==============================================================================#
# PCNR as a NonlinearSolve.jl algorithm: full-Newton FBDF stage solve with a
# dormant in-step limiting corrector
#
# Packages the PCNR predict/correct loop (see doc/pcnr_plan.md and
# _dc_pcnr_newton in solve.jl) as an AbstractNonlinearSolveAlgorithm so that
# implicit OrdinaryDiffEq steppers can run it as their per-stage nonlinear
# solve:
#
#     FBDF(nlsolve = NonlinearSolveAlg(CedarPCNR()))
#
# with zero OrdinaryDiffEq changes. Sundials IDA can never participate (its
# Newton loop is C); use `pcnr_fbdf()` from `tran!` instead.
#
# What this actually buys over plain FBDF (measured): a *full* Newton stage
# solve — restamp + refactor J every iteration — versus FBDF's modified
# Newton (frozen-W reuse). That is a ~2x warm speedup on diode-switching
# circuits. The SPICE junction-limiting corrector is carried along but is
# *inert during normal stepping*: the integrator's predictor warm-starts each
# step within 2*vt, so pnjlim never fires (adopt count measured at 0 across
# rectifier/graetz/BJT cases). Limiting is a cold-start mechanism, handled by
# _dc_pcnr_newton at DC/init; here the corrector is a dormant safety net that
# engages only if a junction swings hard in a single step. See doc/pcnr_plan.md
# "Finding — the in-step corrector is measured to be inert".
#
# The file is split in two clearly-fenced sections:
#
#   Section A: the generic algorithm (PCNRSolver). Written against the
#     SciMLBase + NonlinearSolveBase + LinearSolve public API only — no
#     Cadnip types — so it can be upstreamed as-is. The single PCNR-specific
#     API is the post-update hook `correct!(u_proposed, u_prev, p)` run after
#     every Newton update, exactly the hook shape proposed for the
#     NonlinearSolveBase RFC (doc/pcnr_plan.md, "Deferred upstreaming plan").
#
#   Section B: Cadnip glue. Decodes OrdinaryDiffEqNonlinearSolve's stage
#     params tuple to reach the EvalWorkspace (and with it the recorded
#     limit_w buffer), and supplies the stage Jacobian from the just-stamped
#     G/C matrices. This is the only place in the codebase that knows the
#     bridge's params-tuple layout.
#
# DC/in-step asymmetry (deliberate): _dc_pcnr_newton's initjct arming,
# vcrit seeding, and convergence-settle are *cold-start* mechanisms — they
# encode "the solver has no iterate history yet". In-step solves are warm
# starts: the integrator's predictor extrapolates x_lim along with the rest
# of the state, so vold ≈ vnew and pnjlim is a natural no-op away from
# switching instants. Nothing here touches ws.dctx.initjct.
#==============================================================================#

using SciMLBase: ReturnCode, NLStats
using NonlinearSolveBase: NonlinearSolveBase, InternalAPI
using OrdinaryDiffEq.OrdinaryDiffEqCore: MethodType, COEFFICIENT_MULTISTEP
import OrdinaryDiffEqNonlinearSolve
using OrdinaryDiffEqNonlinearSolve: NonlinearSolveAlg
using OrdinaryDiffEqBDF: FBDF

export PCNRSolver, CedarPCNR, pcnr_fbdf

#==============================================================================#
# Section A: generic PCNR algorithm (upstreamable; no Cadnip types)
#==============================================================================#

"""
    PCNRSolver(correct!; jac!, jac_prototype, name=:PCNRSolver)

Newton's method with a post-update correction hook — the Predictor/Corrector
Newton-Raphson scheme of Aadithya, Keiter & Mei (SAND2018-5689C,
doi:10.1007/978-3-030-44101-2_19), where the "refine" computation lives in
the correction hook.

Each iteration:

 1. evaluate the residual `prob.f(fu, u, p)` and the Jacobian
    `jac!(J, u, p)` at the current iterate,
 2. take a plain Newton step `u ← u − J \\ fu` (**predict**), then
 3. call `correct!(u_proposed, u_prev, p)`, which may overwrite components
    of the proposed iterate (**correct**) — e.g. replace junction-limiting
    variables with the limited voltages the model evaluation recorded.

The correction runs *between* iterations, not inside the linearized solve:
a limiting equation solved simultaneously with the circuit equations tracks
the new iterate to first order and never fires (see doc/pcnr_plan.md,
"v1 design"). With an identity `correct!` this is plain Newton.

# Arguments
- `correct!`: `(u_proposed, u_prev, p) -> nothing`, mutates `u_proposed`.
- `jac!`: `(J, u, p) -> nothing`, the Jacobian of `prob.f` at `u`. Required —
  this algorithm never uses AD or `prob.f.jac` (SPICE-style simulators
  re-linearize on every iteration as a side effect of residual evaluation).
- `jac_prototype`: a matrix with the Jacobian's sparsity pattern, or a
  callable `(u0, p) -> matrix`.

Designed to be driven either standalone (`init`/`step!`) or as the inner
algorithm of `OrdinaryDiffEqNonlinearSolve.NonlinearSolveAlg`, which gives
implicit ODE steppers a per-stage predict/correct loop.
"""
struct PCNRSolver{C, J, P} <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
    name::Symbol
    correct!::C
    jac!::J
    jac_prototype::P
end

function PCNRSolver(correct!; jac!, jac_prototype, name::Symbol=:PCNRSolver)
    return PCNRSolver(name, correct!, jac!, jac_prototype)
end

_materialize_prototype(proto::AbstractMatrix, u0, p) = copy(proto)
_materialize_prototype(proto, u0, p) = proto(u0, p)

mutable struct PCNRSolverCache{PR, A, T, M, L, P} <:
               NonlinearSolveBase.AbstractNonlinearSolveCache
    prob::PR
    alg::A
    u::Vector{T}
    fu::Vector{T}
    uprev::Vector{T}
    J::M
    lincache::L
    p::P
    stats::NLStats
    nsteps::Int
    maxiters::Int
    retcode::ReturnCode.T
    force_stop::Bool
    timer::Any
    trace::Any
    abstol::Any
    reltol::Any
end

NonlinearSolveBase.get_abstol(cache::PCNRSolverCache) = cache.abstol
NonlinearSolveBase.get_reltol(cache::PCNRSolverCache) = cache.reltol

function SciMLBase.__init(prob::SciMLBase.NonlinearProblem, alg::PCNRSolver, args...;
                          maxiters::Int=1000, abstol=nothing, reltol=nothing,
                          kwargs...)
    u = collect(Float64, prob.u0)
    fu = similar(u)
    uprev = copy(u)
    J = _materialize_prototype(alg.jac_prototype, u, prob.p)
    linalg = J isa SparseMatrixCSC ? KLUFactorization() : nothing
    lincache = init(LinearProblem(J, fu), linalg)
    return PCNRSolverCache(prob, alg, u, fu, uprev, J, lincache, prob.p,
                           NLStats(0, 0, 0, 0, 0), 0, maxiters,
                           ReturnCode.Default, false,
                           NonlinearSolveBase.get_timer_output(), nothing,
                           abstol, reltol)
end

function InternalAPI.step!(cache::PCNRSolverCache; recompute_jacobian=nothing,
                           kwargs...)
    # `recompute_jacobian` is deliberately ignored: the residual evaluation
    # re-stamps the system anyway, so a fresh Jacobian is free — and PCNR's
    # convergence argument assumes evaluation-point linearization.
    prob = cache.prob
    alg = cache.alg
    u = cache.u
    copyto!(cache.uprev, u)

    prob.f(cache.fu, u, cache.p)
    cache.stats.nf += 1
    alg.jac!(cache.J, u, cache.p)
    cache.stats.njacs += 1

    ok = all(isfinite, cache.fu)
    if ok
        # PREDICT: cached factorization, numeric-only refactor (pattern fixed).
        cache.lincache.A = cache.J
        cache.lincache.b = cache.fu
        δ = try
            solve!(cache.lincache).u
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
            nothing
        end
        cache.stats.nsolve += 1
        ok = δ !== nothing && all(isfinite, δ)
        if ok
            @inbounds for i in eachindex(u, δ)
                u[i] -= δ[i]
            end
        end
    end

    if !ok
        # Poison the iterate so callers that measure progress by increment
        # (OrdinaryDiffEqNonlinearSolve's outer loop) see a non-finite norm
        # and declare divergence — an unchanged `u` would read as a zero
        # increment, i.e. spurious convergence.
        fill!(u, NaN)
        cache.retcode = ReturnCode.Failure
        cache.force_stop = true
        return nothing
    end

    # CORRECT: the hook may overwrite components of the proposed iterate.
    alg.correct!(u, cache.uprev, cache.p)
    return nothing
end

function InternalAPI.reinit!(cache::PCNRSolverCache; u0=nothing, p=nothing,
                             kwargs...)
    u0 !== nothing && copyto!(cache.u, u0)
    p !== nothing && (cache.p = p)
    cache.nsteps = 0
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    InternalAPI.reinit!(cache.stats)
    return cache
end

#==============================================================================#
# Section B: Cadnip glue for OrdinaryDiffEqNonlinearSolve's stage solves
#
# The bridge builds one inner NonlinearProblem whose params are refreshed per
# stage (OrdinaryDiffEqNonlinearSolve/src/newton.jl, initialize!):
#
#   nlp_params = (tmp, ustep, γ, α, tstep, k, invγdt, method, p, dt, f)
#
# where p (index 9) is the ODEProblem's parameter object — for Cadnip's
# ODEProblem(circuit, ...), the EvalWorkspace itself. The stage residual is
#
#   COEFFICIENT_MULTISTEP (FBDF):  R(z) = tmp + f(z) − (α/γdt)·C·z,  u ≡ z
#   DIRK:                          R(z) = (dt·f(u) − C·z)/γdt,  u = tmp + γ·z
#
# with f(u) = b − G·u, so the stage Jacobian is −(G + c·C) with
# c = α·invγdt (FBDF) or invγdt (DIRK). The DAE layout (DAEFunction) is a
# different tuple shape and is not supported here — use IDA or the DC path.
#==============================================================================#

# Accessors for the bridge's ODE params tuple. The typed getindex on the
# EvalWorkspace slot doubles as a version canary: if a future
# OrdinaryDiffEqNonlinearSolve reshapes the tuple, this throws loudly instead
# of silently mis-stepping.
_stage_tmp(p::Tuple)    = p[1]
_stage_γ(p::Tuple)      = p[3]
_stage_α(p::Tuple)      = p[4]
_stage_invγdt(p::Tuple) = p[7]
_stage_method(p::Tuple) = p[8]::MethodType
_stage_ws(p::Tuple)     = p[9]::EvalWorkspace

"""
    CedarPCNRCorrect()

PCNR correction hook for MNA stage solves: for each limiting variable whose
limiter was **active** this iteration (`limit_active[k]`, i.e. the device
evaluated at a compressed `w ≠ V_branch`), overwrite that slot of the
proposed stage iterate with `limit_w[k]` — the voltage the device actually
evaluated at (recorded by `limit!` / `record_limit_w!`) — so it becomes
`vold` for the next iteration.

Inert branches are deliberately left as the Newton step produced them: the
linear `g_lim` row is then solved exactly (`x_lim = V_branch`), and adopting
the lagged `limit_w` (recorded at the previous iterate's probe) would inject
a one-iteration lag into the *accepted* stage state — which the BDF error
estimator reads as spurious displacement and can drive to `dt` underflow.
The DC loop tolerates that lag because it re-verifies with a settle step at
convergence (`_dc_pcnr_newton`); the in-step solve has no settle hook, so it
avoids the lag at the source. At a true fixed point every limiter is inert
(`pnjlim(V, V) = V`), so acceptance always lands on a consistent state.

For FBDF (`COEFFICIENT_MULTISTEP`) the stage variable is `u(t+dt)` itself;
for DIRK methods the stage variable `z` maps through `u = tmp + γ·z`, so the
overwrite is applied in `z`-space.
"""
struct CedarPCNRCorrect end

function (::CedarPCNRCorrect)(u, uprev, p)
    ws = _stage_ws(p)
    L = ws.structure.n_limits
    L == 0 && return nothing
    limit_w = ws.dctx.limit_w
    limit_active = ws.dctx.limit_active
    lim0 = length(u) - L
    if _stage_method(p) === COEFFICIENT_MULTISTEP
        # FBDF: stage variable is u(t+dt) itself, so the map is the identity —
        # the same active-aware adopt the DC loop uses.
        _pcnr_adopt_limits!(u, limit_w, limit_active, lim0, L)
    else
        tmp = _stage_tmp(p)
        γ = _stage_γ(p)
        @inbounds for k in 1:L
            limit_active[k] && (u[lim0 + k] = (limit_w[k] - tmp[lim0 + k]) / γ)
        end
    end
    return nothing
end

"""
    CedarPCNRStageJac()

Stage Jacobian for MNA stage solves: `J = −(G + c·C)` assembled from the
workspace's G/C nzval (already stamped at the current iterate by the
residual evaluation that precedes it — see `InternalAPI.step!`). G and C are
compiled onto a shared sparsity pattern (`compile_structure`), so a single
fused nzval loop suffices, as in `fast_jacobian!`.
"""
struct CedarPCNRStageJac end

function (::CedarPCNRStageJac)(J::SparseMatrixCSC, u, p)
    ws = _stage_ws(p)
    cs = ws.structure
    c = _stage_method(p) === COEFFICIENT_MULTISTEP ?
        _stage_α(p) * _stage_invγdt(p) : _stage_invγdt(p)
    J_nz = nonzeros(J)
    G_nz = nonzeros(cs.G)
    C_nz = nonzeros(cs.C)
    @inbounds for i in eachindex(J_nz, G_nz, C_nz)
        J_nz[i] = -(G_nz[i] + c * C_nz[i])
    end
    return nothing
end

_cedar_pcnr_prototype(u0, p) = copy(_stage_ws(p).structure.G)

"""
    CedarPCNR()

The Cadnip instantiation of [`PCNRSolver`](@ref) for in-step transient
limiting: recorded-`w` corrector + `−(G + c·C)` stage Jacobian, both reading
the `EvalWorkspace` that Cadnip's `ODEProblem` carries as `prob.p`.
Stateless — safe to construct anywhere, reusable across circuits.
"""
CedarPCNR() = PCNRSolver(CedarPCNRCorrect();
                         jac! = CedarPCNRStageJac(),
                         jac_prototype = _cedar_pcnr_prototype,
                         name = :CedarPCNR)

"""
    pcnr_fbdf(; nlsolve_kwargs=(;), kwargs...)

`FBDF` configured for SPICE-style in-step junction limiting:
`FBDF(nlsolve = NonlinearSolveAlg(CedarPCNR()); kwargs...)`.

Use as `tran!(circuit, tspan; solver=pcnr_fbdf())`. On circuits with no
limiting variables the corrector is inert and this behaves as plain FBDF
with a restamp-every-iteration Newton. `nlsolve_kwargs` are forwarded to
`NonlinearSolveAlg` (`κ`, `max_iter`, `fast_convergence_cutoff`, ...).
"""
function pcnr_fbdf(; nlsolve_kwargs=(;), kwargs...)
    return FBDF(; autodiff=ADTypes.AutoFiniteDiff(),
                nlsolve=NonlinearSolveAlg(CedarPCNR(); nlsolve_kwargs...),
                kwargs...)
end
