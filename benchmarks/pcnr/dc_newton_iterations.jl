#==============================================================================#
# PCNR vs. NonlinearSolve.jl DC Newton convergence benchmark
#
# Compares DC Newton iteration counts across all the nonlinear methods Cadnip
# uses (see doc/pcnr_plan.md and src/mna/solve.jl) on a handful of diode
# rectifier / cascade topologies. Every circuit is a hand-written native
# `Diode` builder (not VA), each available in a `limit=true` and `limit=false`
# variant so PCNR (which needs the limiting-augmented system) and the plain
# NonlinearSolve algorithms (which run on the natural, unaugmented system) can
# both be exercised fairly.
#
# Run with:
#   ~/.juliaup/bin/julia --project=test benchmarks/pcnr/dc_newton_iterations.jl
#==============================================================================#

using Cadnip
using Cadnip.MNA
import Cadnip.MNA as MNA
using LinearAlgebra
using Printf

# NonlinearSolve/SciMLBase are not direct dependencies of test/Project.toml,
# but MNA does `using NonlinearSolve` / `using SciMLBase` internally, which
# makes their exported names resolvable via qualification on the MNA module.
const NewtonRaphson = MNA.NewtonRaphson
const TrustRegion = MNA.TrustRegion
const RobustMultiNewton = MNA.RobustMultiNewton
const LevenbergMarquardt = MNA.LevenbergMarquardt
const PseudoTransient = MNA.PseudoTransient
const NonlinearProblem = MNA.NonlinearProblem
const NonlinearFunction = MNA.NonlinearFunction
const nlsolve = MNA.solve
const ReturnCode = MNA.SciMLBase.ReturnCode
const CedarRobustNLSolve = MNA.CedarRobustNLSolve

#==============================================================================#
# Circuit builders (native Diode, mirrors test/mna/pcnr.jl style)
#
# All nodes are allocated via get_node! before any device is stamped -- this
# ordering is mandatory so DirectStampContext's positional counters stay in
# sync between the structure-detection pass and later fast restamps.
#==============================================================================#

# d1n4007-like diode parameters (rs/cjo ignored -- this is a DC benchmark)
const DIODE_IS = 76.9e-12
const DIODE_N = 1.45
const DIODE_VT = 0.026

make_diode(name::Symbol, limit::Bool) =
    Diode(Is=DIODE_IS, n=DIODE_N, Vt=DIODE_VT, limit=limit, name=name)

# 1. Half-wave rectifier: Vsrc -[1k]- out -D1- gnd
function rectifier(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        MNA.reset_for_restamping!(ctx)
    end
    vin = get_node!(ctx, :vin)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(params.Vsrc; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, out)
    stamp!(make_diode(:D1, params.limit), ctx, out, 0; x=x)
    return ctx
end

# 2. Series chain: Vsrc -[1k]- n1 -D1- n2 -D2- n3 -D3- gnd
function chain3(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        MNA.reset_for_restamping!(ctx)
    end
    vin = get_node!(ctx, :vin)
    n1 = get_node!(ctx, :n1)
    n2 = get_node!(ctx, :n2)
    n3 = get_node!(ctx, :n3)

    stamp!(VoltageSource(params.Vsrc; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, n1)
    stamp!(make_diode(:D1, params.limit), ctx, n1, n2; x=x)
    stamp!(make_diode(:D2, params.limit), ctx, n2, n3; x=x)
    stamp!(make_diode(:D3, params.limit), ctx, n3, 0; x=x)
    return ctx
end

# 3. Full-wave (Graetz) bridge, DC only (100uF smoothing cap is open at DC,
# so it's omitted -- see benchmarks/vacask/graetz/cedarsim/runme.sp).
# vs inp inn 0 sin ... -> DC value Vsrc, stamped node-to-node (two-node
# VoltageSource is supported directly, no need to route through ground).
function graetz(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        MNA.reset_for_restamping!(ctx)
    end
    inp = get_node!(ctx, :inp)
    inn = get_node!(ctx, :inn)
    outp = get_node!(ctx, :outp)
    outn = get_node!(ctx, :outn)

    stamp!(VoltageSource(params.Vsrc; name=:VS), ctx, inp, inn)
    stamp!(make_diode(:D1, params.limit), ctx, inp, outp; x=x)
    stamp!(make_diode(:D2, params.limit), ctx, outn, inp; x=x)
    stamp!(make_diode(:D3, params.limit), ctx, inn, outp; x=x)
    stamp!(make_diode(:D4, params.limit), ctx, outn, inn; x=x)
    stamp!(Resistor(1e3), ctx, outp, outn)     # rl
    stamp!(Resistor(1e6), ctx, inn, 0)          # rgnd1
    stamp!(Resistor(1e6), ctx, outn, 0)         # rgnd2
    return ctx
end

# 4. Diode multiplier (mul4), DC only -- caps replaced by 100k resistors
# (open at DC; the resistors keep the DC problem meaningful, preserving the
# cascade stiffness without leaving nodes floating through reverse-diode
# leakage alone). See benchmarks/vacask/mul/cedarsim/runme.sp.
function mul4(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        MNA.reset_for_restamping!(ctx)
    end
    a = get_node!(ctx, :a)
    n1 = get_node!(ctx, :n1)
    n2 = get_node!(ctx, :n2)
    n10 = get_node!(ctx, :n10)
    n20 = get_node!(ctx, :n20)

    stamp!(VoltageSource(params.Vsrc; name=:VS), ctx, a, 0)
    stamp!(Resistor(0.01), ctx, a, n1)          # r1
    stamp!(make_diode(:D1, params.limit), ctx, 0, n1; x=x)
    stamp!(make_diode(:D2, params.limit), ctx, n1, n10; x=x)
    stamp!(make_diode(:D3, params.limit), ctx, n10, n2; x=x)
    stamp!(make_diode(:D4, params.limit), ctx, n2, n20; x=x)
    stamp!(Resistor(100e3), ctx, n1, n2)        # c1 || c3 (dedup'd)
    stamp!(Resistor(100e3), ctx, 0, n10)        # c2
    stamp!(Resistor(100e3), ctx, n10, n20)      # c4
    return ctx
end

#==============================================================================#
# Circuit list: (name, builder, [Vsrc...])
#==============================================================================#

const CIRCUITS = [
    ("rectifier", rectifier, [5.0, 50.0]),
    ("chain3", chain3, [50.0]),
    ("graetz", graetz, [20.0, 325.0]),
    ("mul4", mul4, [50.0]),
]

#==============================================================================#
# Sanity check: DC solution via the default (fully robust) solve path
#==============================================================================#

function sanity_check(name, builder, Vsrc)
    params = (Vsrc=Vsrc, limit=true)
    spec = MNASpec(mode=:dcop)
    sol = MNA.solve_dc(builder, params, spec)
    println("== $name (Vsrc=$Vsrc) DC sanity check ==")
    for nm in sol.node_names
        @printf("  %-8s % .6g\n", nm, sol[nm])
    end
    println()
    return sol
end

#==============================================================================#
# Plain (uncorrected) Newton loop -- same predictor as _dc_pcnr_newton but
# without the refine/correct step, run on the SAME limit-augmented system.
# This shows what raw Newton does on the augmented problem when the corrector
# is skipped (reference point only, not a solver Cadnip actually offers).
#==============================================================================#

function plain_newton_loop(cs, ws, u0::AbstractVector; abstol::Real=1e-10, maxiters::Int=200)
    n = length(u0)
    u = copy(u0)
    F = zeros(n)
    for iter in 1:maxiters
        MNA.fast_rebuild!(ws, cs, u, 0.0)
        mul!(F, cs.G, u)
        F .-= ws.dctx.b

        if norm(F) < abstol
            return u, true, iter - 1, norm(F)
        end

        δ = try
            cs.G \ F
        catch err
            err isa LinearAlgebra.SingularException && return u, false, iter - 1, norm(F)
            rethrow()
        end
        all(isfinite, δ) || return u, false, iter - 1, norm(F)
        u .-= δ
    end
    MNA.fast_rebuild!(ws, cs, u, 0.0)
    mul!(F, cs.G, u)
    F .-= ws.dctx.b
    return u, false, maxiters, norm(F)
end

#==============================================================================#
# NonlinearSolve.jl algorithms under test (mirrors _dc_newton_compiled)
#==============================================================================#

const NLS_METHODS = [
    ("NewtonRaphson", () -> NewtonRaphson(autodiff=nothing)),
    ("TrustRegion", () -> TrustRegion(autodiff=nothing)),
    ("RobustMultiNewton", () -> RobustMultiNewton(autodiff=nothing)),
    ("LevenbergMarquardt", () -> LevenbergMarquardt(autodiff=nothing)),
    ("PseudoTransient", () -> PseudoTransient(autodiff=nothing)),
    ("CedarRobustNLSolve", () -> CedarRobustNLSolve()),
]

function run_nls_method(cs, ws, alg_fn, u0; abstol=1e-10, maxiters=200)
    function residual!(F, u, p)
        MNA.fast_rebuild!(ws, cs, u, 0.0)
        mul!(F, cs.G, u)
        F .-= ws.dctx.b
        return nothing
    end
    function jacobian!(J, u, p)
        MNA.fast_rebuild!(ws, cs, u, 0.0)
        copyto!(J, cs.G)
        return nothing
    end

    nlfunc = NonlinearFunction(residual!; jac=jacobian!, jac_prototype=cs.G)
    nlprob = NonlinearProblem(nlfunc, copy(u0))

    try
        sol = nlsolve(nlprob, alg_fn(); abstol=abstol, maxiters=maxiters)
        Ffinal = zeros(length(u0))
        residual!(Ffinal, sol.u, nothing)
        return (retcode=string(sol.retcode), nsteps=sol.stats.nsteps, nf=sol.stats.nf,
                resid=norm(Ffinal), converged=(sol.retcode == ReturnCode.Success))
    catch err
        return (retcode="threw: $(nameof(typeof(err)))", nsteps=-1, nf=-1,
                resid=NaN, converged=false)
    end
end

#==============================================================================#
# Benchmark driver
#==============================================================================#

struct Row
    circuit::String
    Vsrc::Float64
    method::String
    converged::Bool
    iters::Int
    nf::Int
    resid::Float64
    retcode::String
end

function build_problem(builder, params, spec)
    ctx = MNA._detect_structure(builder, params, spec)
    cs = MNA.compile_structure(builder, params, spec; ctx=ctx)
    ws = MNA.create_workspace(cs; ctx=ctx)
    n = MNA.system_size(ctx)
    return cs, ws, n
end

function run_circuit(name, builder, Vsrc)
    spec = MNASpec(mode=:dcop)
    rows = Row[]

    # --- limit=true variant: PCNR + reference plain-Newton-on-augmented ---
    params_lim = (Vsrc=Vsrc, limit=true)
    cs_lim, ws_lim, n_lim = build_problem(builder, params_lim, spec)

    u_pcnr, ok_pcnr, iters_pcnr = MNA._dc_pcnr_newton(cs_lim, ws_lim, zeros(n_lim);
                                                       abstol=1e-10, maxiters=200)
    Fpcnr = zeros(n_lim)
    MNA.fast_rebuild!(ws_lim, cs_lim, u_pcnr, 0.0)
    mul!(Fpcnr, cs_lim.G, u_pcnr)
    Fpcnr .-= ws_lim.dctx.b
    push!(rows, Row(name, Vsrc, "PCNR", ok_pcnr, iters_pcnr, iters_pcnr, norm(Fpcnr),
                     ok_pcnr ? "Success" : "Failure"))

    u_plain, ok_plain, iters_plain, resid_plain = plain_newton_loop(cs_lim, ws_lim, zeros(n_lim);
                                                                     abstol=1e-10, maxiters=200)
    push!(rows, Row(name, Vsrc, "PlainNewton(augmented,no-correct)", ok_plain, iters_plain,
                     iters_plain, resid_plain, ok_plain ? "Success" : "Failure"))

    # --- limit=false variant: NonlinearSolve.jl algorithms on the natural system ---
    params_nolim = (Vsrc=Vsrc, limit=false)
    cs_nolim, ws_nolim, n_nolim = build_problem(builder, params_nolim, spec)
    u0 = zeros(n_nolim)

    for (mname, alg_fn) in NLS_METHODS
        r = run_nls_method(cs_nolim, ws_nolim, alg_fn, u0; abstol=1e-10, maxiters=200)
        push!(rows, Row(name, Vsrc, mname, r.converged, r.nsteps, r.nf, r.resid, r.retcode))
    end

    return rows
end

#==============================================================================#
# Output
#==============================================================================#

function print_table(all_rows::Vector{Row})
    @printf("%-11s %8s %-34s %-9s %8s %8s %12s  %s\n",
            "circuit", "Vsrc", "method", "converged", "iters", "nf", "||F||", "retcode")
    println("-"^120)
    for r in all_rows
        @printf("%-11s %8.4g %-34s %-9s %8d %8d %12.3e  %s\n",
                r.circuit, r.Vsrc, r.method, r.converged ? "yes" : "no",
                r.iters, r.nf, r.resid, r.retcode)
    end
end

function print_summary(all_rows::Vector{Row})
    println()
    println("Summary (fewest iterations among converged methods, per circuit/Vsrc):")
    seen = Tuple{String,Float64}[]
    for r in all_rows
        key = (r.circuit, r.Vsrc)
        key in seen && continue
        push!(seen, key)
    end
    for (circuit, Vsrc) in seen
        candidates = filter(r -> r.circuit == circuit && r.Vsrc == Vsrc && r.converged, all_rows)
        if isempty(candidates)
            println("  $circuit (Vsrc=$Vsrc): no method converged")
        else
            best = candidates[argmin([r.iters for r in candidates])]
            @printf("  %-10s (Vsrc=%-8.4g): %s wins with %d iterations\n",
                    circuit, Vsrc, best.method, best.iters)
        end
    end
end

function main()
    println("PCNR vs. NonlinearSolve.jl DC Newton convergence benchmark")
    println("="^100)
    println()

    for (name, builder, Vsrcs) in CIRCUITS
        for Vsrc in Vsrcs
            sanity_check(name, builder, Vsrc)
        end
    end

    all_rows = Row[]
    for (name, builder, Vsrcs) in CIRCUITS
        for Vsrc in Vsrcs
            append!(all_rows, run_circuit(name, builder, Vsrc))
        end
    end

    println()
    print_table(all_rows)
    print_summary(all_rows)
end

main()
