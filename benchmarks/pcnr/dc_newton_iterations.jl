#==============================================================================#
# PCNR vs. NonlinearSolve.jl DC Newton convergence benchmark
#
# Compares DC Newton iteration counts across all the nonlinear methods Cadnip
# uses (see doc/pcnr_plan.md and src/mna/solve.jl) on a handful of real
# VADistiller `sp_diode` rectifier / cascade topologies -- the end goal of the
# `$limit` work: the distilled ngspice diode with SPICE limiting active.
#
# sp_diode always carries a PCNR limiting variable (via the Verilog-A `$limit`
# codegen), so there is no unlimited twin: PCNR, a reference plain-Newton loop
# (corrector disabled), and every NonlinearSolve algorithm all run on the one
# limit-augmented system. The correctorless methods see an inert limiter (the
# linear g_lim rows pin x_lim = V), i.e. the natural problem -- so this is
# exactly the comparison of interest: limiting-aware PCNR vs. methods that
# don't support limiting, on the same circuit.
#
# Run with:
#   ~/.juliaup/bin/julia --project=test benchmarks/pcnr/dc_newton_iterations.jl [output_file]
#
# If output_file is provided, a markdown report is written there (same
# convention as benchmarks/vacask/run_benchmarks.jl). Otherwise, the original
# fixed-width text table is printed to stdout.
#==============================================================================#

using Cadnip
using Cadnip.MNA
import Cadnip.MNA as MNA
using LinearAlgebra
using Printf
# Real distilled ngspice devices (register sp_diode / sp_bjt; each carries PCNR
# limiting variables via the Verilog-A `$limit` codegen). See doc/pcnr_plan.md.
using VADistillerModels: sp_diode, sp_bjt

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

# Real distilled ngspice diode (d1n4007-like is/n; rs/cjo irrelevant at DC).
# sp_diode ships its own SPICE limiting logic in Verilog-A and, via the
# `$limit` codegen, always carries a PCNR limiting variable. There is no
# "unlimited" twin, so every method (PCNR, plain-Newton, and the NonlinearSolve
# algorithms) runs on the same limit-augmented system: under a correctorless
# solver the limiter is provably inert (the linear g_lim rows keep x_lim = V),
# so those methods see the natural problem; only PCNR's corrector activates it.
# That is exactly the comparison of interest -- limiting-aware PCNR vs. methods
# that don't support limiting -- so a native/unlimited reference is unnecessary.
const DIODE_IS = 76.9e-12
const DIODE_N = 1.45
d(name::Symbol) = sp_diode(; is=DIODE_IS, n=DIODE_N)

# 1. Half-wave rectifier: Vsrc -[1k]- out -D1- gnd
function rectifier(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    ctx = ctx === nothing ? MNAContext() : (MNA.reset_for_restamping!(ctx); ctx)
    vin = get_node!(ctx, :vin)
    out = get_node!(ctx, :out)
    stamp!(VoltageSource(params.Vsrc; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, out)
    stamp!(d(:D1), ctx, out, 0; _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D1)
    return ctx
end

# 2. Series chain: Vsrc -[1k]- n1 -D1- n2 -D2- n3 -D3- gnd
function chain3(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    ctx = ctx === nothing ? MNAContext() : (MNA.reset_for_restamping!(ctx); ctx)
    vin = get_node!(ctx, :vin); n1 = get_node!(ctx, :n1)
    n2 = get_node!(ctx, :n2); n3 = get_node!(ctx, :n3)
    stamp!(VoltageSource(params.Vsrc; name=:V1), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, n1)
    stamp!(d(:D1), ctx, n1, n2; _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D1)
    stamp!(d(:D2), ctx, n2, n3; _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D2)
    stamp!(d(:D3), ctx, n3, 0;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D3)
    return ctx
end

# 3. Full-wave (Graetz) bridge, DC only (100uF smoothing cap is open at DC,
# so it's omitted -- see benchmarks/vacask/graetz/cedarsim/runme.sp).
function graetz(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    ctx = ctx === nothing ? MNAContext() : (MNA.reset_for_restamping!(ctx); ctx)
    inp = get_node!(ctx, :inp); inn = get_node!(ctx, :inn)
    outp = get_node!(ctx, :outp); outn = get_node!(ctx, :outn)
    stamp!(VoltageSource(params.Vsrc; name=:VS), ctx, inp, inn)
    stamp!(d(:D1), ctx, inp, outp;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D1)
    stamp!(d(:D2), ctx, outn, inp;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D2)
    stamp!(d(:D3), ctx, inn, outp;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D3)
    stamp!(d(:D4), ctx, outn, inn;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D4)
    stamp!(Resistor(1e3), ctx, outp, outn)     # rl
    stamp!(Resistor(1e6), ctx, inn, 0)          # rgnd1
    stamp!(Resistor(1e6), ctx, outn, 0)         # rgnd2
    return ctx
end

# 4. Diode multiplier (mul4), DC only -- caps replaced by 100k resistors
# (open at DC; the resistors keep the DC problem meaningful). See
# benchmarks/vacask/mul/cedarsim/runme.sp.
function mul4(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    ctx = ctx === nothing ? MNAContext() : (MNA.reset_for_restamping!(ctx); ctx)
    a = get_node!(ctx, :a); n1 = get_node!(ctx, :n1); n2 = get_node!(ctx, :n2)
    n10 = get_node!(ctx, :n10); n20 = get_node!(ctx, :n20)
    stamp!(VoltageSource(params.Vsrc; name=:VS), ctx, a, 0)
    stamp!(Resistor(0.01), ctx, a, n1)          # r1
    stamp!(d(:D1), ctx, 0, n1;    _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D1)
    stamp!(d(:D2), ctx, n1, n10;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D2)
    stamp!(d(:D3), ctx, n10, n2;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D3)
    stamp!(d(:D4), ctx, n2, n20;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:D4)
    stamp!(Resistor(100e3), ctx, n1, n2)        # c1 || c3 (dedup'd)
    stamp!(Resistor(100e3), ctx, 0, n10)        # c2
    stamp!(Resistor(100e3), ctx, n10, n20)      # c4
    return ctx
end

# 5. Darlington pair (NPN): Q1's emitter drives Q2's base, so the two Vbe drops
# stack and the current gains multiply — the BJT analog of the diode chain, and
# the multi-branch stress case (each sp_bjt has 3 limited junctions: vbe, vbc,
# vsub). Vsrc drives the input base through Rb; the output emitter sits at
# ~Vsrc − 2·Vbe. sp_bjt terminals: collector, base, emitter, substrate.
q(name::Symbol) = sp_bjt(; bf=100.0, is=1e-15)
function darlington(params, spec, t::Real=0.0; x=MNA.ZERO_VECTOR, ctx=nothing)
    ctx = ctx === nothing ? MNAContext() : (MNA.reset_for_restamping!(ctx); ctx)
    vcc = get_node!(ctx, :vcc); vin = get_node!(ctx, :vin); base = get_node!(ctx, :base)
    mid = get_node!(ctx, :mid); out = get_node!(ctx, :out)
    stamp!(VoltageSource(5.0; name=:Vcc), ctx, vcc, 0)
    stamp!(VoltageSource(params.Vsrc; name=:Vin), ctx, vin, 0)
    stamp!(Resistor(10e3), ctx, vin, base)                 # rb
    stamp!(q(:Q1), ctx, vcc, base, mid, 0; _mna_spec_=spec, _mna_x_=x, _mna_instance_=:Q1)
    stamp!(q(:Q2), ctx, vcc, mid, out, 0;  _mna_spec_=spec, _mna_x_=x, _mna_instance_=:Q2)
    stamp!(Resistor(1e3), ctx, out, 0)                     # rl
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
    ("darlington", darlington, [3.0, 5.0]),
]

#==============================================================================#
# Sanity check: DC solution via the default (fully robust) solve path
#==============================================================================#

function sanity_check(name, builder, Vsrc)
    params = (Vsrc=Vsrc,)
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

# sp_diode has no `limit=false` twin (the `$limit` codegen always allocates a
# limiting variable), so PCNR, plain-Newton, and every NonlinearSolve method run
# on the *same* limit-augmented system. The correctorless methods see an inert
# limiter (linear g_lim rows pin x_lim = V), i.e. the natural problem; only
# PCNR's corrector activates it.
function run_circuit(name, builder, Vsrc)
    spec = MNASpec(mode=:dcop)
    rows = Row[]
    params = (Vsrc=Vsrc,)
    cs, ws, n = build_problem(builder, params, spec)

    u_pcnr, ok_pcnr, iters_pcnr = MNA._dc_pcnr_newton(cs, ws, zeros(n);
                                                       abstol=1e-10, maxiters=200)
    Fpcnr = zeros(n)
    MNA.fast_rebuild!(ws, cs, u_pcnr, 0.0)
    mul!(Fpcnr, cs.G, u_pcnr)
    Fpcnr .-= ws.dctx.b
    push!(rows, Row(name, Vsrc, "PCNR", ok_pcnr, iters_pcnr, iters_pcnr, norm(Fpcnr),
                     ok_pcnr ? "Success" : "Failure"))

    u_plain, ok_plain, iters_plain, resid_plain = plain_newton_loop(cs, ws, zeros(n);
                                                                     abstol=1e-10, maxiters=200)
    push!(rows, Row(name, Vsrc, "PlainNewton(augmented,no-correct)", ok_plain, iters_plain,
                     iters_plain, resid_plain, ok_plain ? "Success" : "Failure"))

    u0 = zeros(n)
    for (mname, alg_fn) in NLS_METHODS
        r = run_nls_method(cs, ws, alg_fn, u0; abstol=1e-10, maxiters=200)
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

#==============================================================================#
# Markdown report (for CI job summaries -- see benchmarks/vacask/run_benchmarks.jl
# for the same output_file convention)
#==============================================================================#

function generate_markdown(all_rows::Vector{Row})
    io = IOBuffer()

    println(io, "## DC Newton method comparison (PCNR)")
    println(io)
    println(io, "Compares DC Newton *iteration counts* (not wall-clock) on the real ",
                 "distilled `sp_diode` across every nonlinear method Cadnip uses -- the ",
                 "PCNR limiting-augmented loop, a reference plain-Newton loop with the ",
                 "corrector disabled, and each `NonlinearSolve.jl` algorithm -- all on ",
                 "the one limit-augmented system (the correctorless methods see an inert ",
                 "limiter, i.e. the natural problem).")
    println(io)
    println(io, "Benchmarks run on Julia $(VERSION)")
    println(io)

    seen = Tuple{String,Float64}[]
    for r in all_rows
        key = (r.circuit, r.Vsrc)
        key in seen && continue
        push!(seen, key)
    end

    for (circuit, Vsrc) in seen
        println(io, "### $circuit (Vsrc=$Vsrc)")
        println(io)
        println(io, "| Method | Converged | Iters | nf | \\|\\|F\\|\\| | Retcode |")
        println(io, "|--------|-----------|------:|---:|--------|---------|")
        for r in filter(r -> r.circuit == circuit && r.Vsrc == Vsrc, all_rows)
            @printf(io, "| %s | %s | %d | %d | %.3e | %s |\n",
                    r.method, r.converged ? "yes" : "no", r.iters, r.nf, r.resid, r.retcode)
        end
        println(io)
    end

    println(io, "### Summary")
    println(io)
    println(io, "Fewest iterations among converged methods, per circuit/Vsrc:")
    println(io)
    for (circuit, Vsrc) in seen
        candidates = filter(r -> r.circuit == circuit && r.Vsrc == Vsrc && r.converged, all_rows)
        if isempty(candidates)
            println(io, "- **$circuit** (Vsrc=$Vsrc): no method converged")
        else
            best = candidates[argmin([r.iters for r in candidates])]
            println(io, "- **$circuit** (Vsrc=$Vsrc): `$(best.method)` wins with $(best.iters) iterations")
        end
    end
    println(io)

    return String(take!(io))
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

    if length(ARGS) >= 1
        output_file = ARGS[1]
        markdown = generate_markdown(all_rows)
        open(output_file, "w") do f
            write(f, markdown)
        end
        println()
        println("Report written to: $output_file")
    end
end

# Only run the full benchmark when invoked as a script; `include`-ing this file
# (e.g. from a baseline-capture harness) reuses the circuit builders without
# triggering the run.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
