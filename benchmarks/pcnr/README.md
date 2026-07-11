# PCNR DC Newton iteration benchmark

> **In-step transient limiting** (PCNR inside every timestep, via
> `pcnr_fbdf()` = `FBDF(nlsolve=NonlinearSolveAlg(CedarPCNR()))`) is benchmarked
> separately, as a transient wall-clock + NR-iteration comparison in the VACASK
> suite (`benchmarks/vacask/run_benchmarks.jl`): the `FBDF+PCNR` solver row on
> the **Graetz Bridge**, **Voltage Multiplier**, and **Darlington Pair** cases,
> plus a `QBDF+PCNR` probe on Graetz. This file stays DC-only.

`dc_newton_iterations.jl` compares DC Newton *iteration counts* (not
wall-clock) across every nonlinear method Cadnip uses -- the PCNR
limiting-augmented loop (`MNA._dc_pcnr_newton`), a hand-rolled plain-Newton
loop on the same augmented system with the corrector disabled (reference
only), and each `NonlinearSolve.jl` algorithm in `_dc_newton_compiled`'s
fallback chain (`NewtonRaphson`, `TrustRegion`, `RobustMultiNewton`,
`LevenbergMarquardt`, `PseudoTransient`, `CedarRobustNLSolve`) -- on four
hand-written native-`Diode` circuits (half-wave rectifier, 3-diode series
chain, full-wave Graetz bridge, 4-stage diode multiplier) at DC, each stamped
with the d1n4007 diode parameters.

Run with:

```
~/.juliaup/bin/julia --project=test benchmarks/pcnr/dc_newton_iterations.jl
```

`--project=benchmarks` works too (that's what CI uses).

Takes a couple of minutes (package precompilation dominates; the solves
themselves are fast). No plotting or CSV output -- just a fixed-width table
plus a per-circuit summary line naming the fewest-iteration converged method.

An optional output-file argument writes the same results as a markdown
report instead (same convention as `benchmarks/vacask/run_benchmarks.jl`):

```
julia --project=benchmarks benchmarks/pcnr/dc_newton_iterations.jl pcnr_results.md
```

CI (`.github/workflows/benchmark.yml`) runs this and publishes
`pcnr_results.md` in the job summary alongside the VACASK benchmark tables.

**Interpretation:** PCNR converges on every circuit here and wins iteration
count on all of them: 6-7 iterations on the rectifier/chain cases and 14 on
graetz/mul4 (with `initjct` seeding and evaluation-anchored `lim_rhs`
companions; see the "Measured" notes in `doc/pcnr_plan.md`). Against the
true classic diode model (the `limit=false` baseline is the unclamped
exponential, identical to pre-PCNR semantics), plain `NewtonRaphson` goes
Unstable or hits MaxIters on most stiff cases — exp overflow at the first
overshoot — and the best trust-region methods take 15-30+ where they
survive. The graetz rows are the robustness headline: `RobustMultiNewton`
and `CedarRobustNLSolve` throw there, TrustRegion hits MaxIters, LM stalls.
(Earlier revisions of this branch accidentally clamped the baseline model,
which made plain Newton look like it converged in 65-104 iterations; that
was an artifact.)
`RobustMultiNewton`/`CedarRobustNLSolve` throw `MethodError` on the Graetz
bridge -- a known issue (see `CedarShampineNLSolve`'s docstring in
`src/mna/solve.jl`): its `TrustRegion(Bastin)` member needs a
Jacobian-vector product that ignores `autodiff=nothing` and falls back to
`AutoForwardDiff()`, which chokes on our residual closure.
