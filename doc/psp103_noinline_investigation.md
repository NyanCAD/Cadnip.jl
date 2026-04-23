# PSP103 Compile-Time Blowup — Investigation & Fix

**Status: fixed.** Two orthogonal compile-time blowups, two targeted fixes.
Both shipped. No more `invokelatest`. No belt-and-suspenders.

## What ships

| File | Change |
|---|---|
| `src/mna/solve.jl` | `RobustMultiNewton(autodiff=nothing)` (same for LM, PseudoTransient). Tells NonlinearSolvePolyAlgorithm not to construct its own ForwardDiff.Tag machinery, since we already provide an explicit Jacobian. |
| `src/spc/codegen.jl` | At each of the six VA-model codegen sites, replace `Base.invokelatest(stamp!, …)` with `invoke(stamp!, Tuple{DeviceType, AnyMNAContext, Int…}, …)`. New `va_device_type(state, model_sym)` helper returns the concrete Type. `is_large_va_model` refactored to share it. |
| `test/mna/psp103_integration.jl` | Comment rewrite explaining the two fixes. |

Verified (Julia 1.12.5):
- `test/mna/psp103_integration.jl`: 13/13 pass, **18.6s total** (down from 30+ min without autodiff=nothing).
- Ring oscillator (`benchmarks/vacask/ring`, 18 PSP103 MOSFETs): **55.6s mean** compile+solve for 200ns, clean oscillation.

## The two problems, dissected

### 1. Caller-side inference walk (→ `invoke`)

Even with `@noinline` on stamp!, Julia's inference runs *transitively* through
called methods to determine return type and effects. For a direct
`stamp!(…)` call from a generated circuit builder, inference walks the 782-
field body. This burned ~180s and ~6 GiB of transient inference/SROA allocs
for output that's byte-identical to the precompiled stamp!.

`invokelatest` dodges this because it's defined as:

```julia
invokelatest(@nospecialize(f), @nospecialize args...; kwargs...) =
    Core._call_latest(f, args..., kwargs...)
```

`Core._call_latest` is an intrinsic. Julia's inference treats it opaquely —
returns `Any`, can't look inside. That cuts the inference chain.

**The world-age semantics are incidental.** The thing we needed was the
inference-opaque boundary.

`invoke(f, Tuple{T…}, args…; kw…)` gives us the same inference-cutting
behavior via static method dispatch. Julia dispatches to the specific
MethodInstance matching the type tuple. The MI's return type is cached —
inference reads the cache instead of walking the body. No runtime dispatch;
no world-age lookup.

Per-call measurement on PSP103 `stamp!` after JIT warmup:

| call form | alloc / 1000 calls | time / 1000 calls |
|---|---:|---:|
| direct `stamp!(…)` | 0 bytes | 6.55 ms |
| `invoke(stamp!, Tuple{…}, …)` | **0 bytes** | 6.59 ms |
| `Base.invokelatest(stamp!, …)` | 12.8 MB | 7.26 ms |

`invoke` is strictly better than `invokelatest` per-call.

The `Tuple{DeviceType, AnyMNAContext, Int…}` includes a 2-way Union in the
ctx slot (`MNAContext` or `DirectStampContext`). Measured vs hand-splitting
the call on `ctx isa MNAContext`: identical runtime and allocation. The
Union is fine.

### 2. Nested-AD specialization (→ `autodiff=nothing`)

Two layers of ForwardDiff:

- **Inner AD** — `Dual{Cadnip.MNA.JacobianTag, Float64, N}` (N up to 30)
  inside `stamp!` for device Jacobian entries. Stable Tag, shared across
  circuits.
- **Outer AD** — `ForwardDiff.Tag{NonlinearFunction{…, typeof(circuit), …},
  Float64}` constructed by `NonlinearSolvePolyAlgorithm` for safety checks /
  JVP fallback *even though we provide an explicit `jacobian!`*. Embeds
  `typeof(circuit)` — unique per circuit.

When the outer Tag's Duals reached `stamp!`'s `_mna_x_`, Julia tried to
specialize the 782-field body for a `Vector{Dual{Tag{NLF{…}}, Float64, 1}}`
input. Inside, inner AD produced nested Duals. Transitive inference over 782
fields with nested Duals was the 30+ min compile.

`RobustMultiNewton(autodiff=nothing)` tells NonlinearSolve "don't build AD
machinery, we have our Jacobian." Outer Tag never constructed. `_mna_x_`
stays `Vector{Float64}`. Precompiled `stamp!` MI hit directly.

## Why I ended up *not* needing `@noinline` or `@nospecialize`

Tried both. Removed both. `invoke` alone is sufficient because it cuts the
inference walk at the call site — no walk means no caller-side specialization
attempt, which means the inliner never sees the 782-field body in scope.

Explicitly verified: with codegen using `invoke` but `make_mna_device` NOT
auto-applying `@noinline`, `test/mna/psp103_integration.jl` runs in 18.6s.
Same as with `@noinline`. Belt-and-suspenders removed.

## Paths not taken

- **`@nospecialize _mna_x_`** on the generated `stamp!`. Prevents caller
  specialization per element type of x, but inference still walks the body
  for return type. Doesn't replace `invoke`.
- **`@noinline` on the generated `stamp!`**. Blocks inlining, not inference.
  Doesn't replace `invoke`.
- **Widening PSPModels `@compile_workload`** to cover all kwarg shapes. Moved
  cost into precompile time (134s → 2813s). Unacceptable.

## References

- `doc/sroa_exploration_results.md` — earlier 600-field-era exploration of
  `@noinline` vs `inferencebarrier` vs `invokelatest`.
- `~/.claude/projects/.../memory/project_psp103_invokelatest.md` — memory
  pointer for future sessions.
