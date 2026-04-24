# N-Dimensional `$table_model` — Design Note

## Context

Cadnip currently supports the 1-D form of LRM 9.21 `$table_model`:

```verilog
y = $table_model(wl, "data.tbl", "1L;<col>");
```

Implementation lives at:

- Codegen dispatch: `src/vasim.jl:962-992`
- Compile-time helpers: `_tm_parse_control` (~L732), `_tm_parse_file`
  (~L697) — both hard-code D=1.
- Runtime: `pwl_at_time` in `src/mna/devices.jl:47-73`, called with
  `extrap=1.0` to match LRM "1L" extrapolation semantics.

**What fires this design:** the companion SAX-to-Verilog emitter at
`/Users/pepijndevos/code/sax-in-the-sheets-verilog-in-the-streets/sax_to_verilog.py`
now emits N-D tables. For a 2-D wl × loss sweep, it writes lines like:

```verilog
a_0_0 = $table_model(wl, loss_dB, "mmi2x2_2d.tbl", "1L,1L;1");
```

HSpice handles this per LRM. Cadnip currently doesn't — variadic
`$table_model` calls fail the `length(stmt.args) == 3` assertion at
codegen time. Closing this gap lets the round-trip tests in
`/Users/pepijndevos/code/sax-in-the-sheets-verilog-in-the-streets/test_lynxnip.jl`
extend to multi-parameter sweeps and makes Cadnip a viable verifier
for foundry-style photonic compact models that depend on wl + (T | V
| geometry | …).

## Scope for v1

- **Dimensionality**: arbitrary D ≥ 1. The 1-D code path stays
  identical in behavior and test coverage; N-D is additive.
- **Interpolation**: linear only (`"1"`). Per-dim heterogeneous control
  strings (e.g. `"1L,DC;1"`) should parse but need not implement
  every interp method yet — `"1"` with any extrap is required; `"D"`
  / `"2"` / `"3"` can error cleanly with "not implemented" for a
  follow-up.
- **Extrapolation**: linear (`L`). Constant (`C`) and error (`E`) are
  easy follow-ups once the N-D recursion lands; not required for the
  emitter's output today since it always emits `"1L"` per dim.
- **Input type**: each dim input is a scalar expression, typically a
  parameter via `.param`. No Jacobian contributions — input is
  assumed constant w.r.t. solver node voltages. Matches the 1-D case.
- **Data source**: file-based (isoline format), same as 1-D. Inline
  array form (LRM `table_model_array`) stays out of scope.
- **Cache**: lookup by absolute file path, same as 1-D — at most one
  parsed copy per `.tbl` across the simulation.

## Isoline file format recap (LRM 9.21.1)

For a D-dim input with M dependent columns, each row is
`x₁ x₂ … x_D  y₁ y₂ … y_M`. Rows are grouped into **isolines** where
the outer D−1 input columns are held constant and the innermost column
varies. For a D>1 table:

- Exact-match boundaries on the outer D−1 columns define isoline
  transitions.
- LRM allows (but doesn't require) rows in any order — implementations
  must sort per dim.
- A well-formed file must produce at least 2 distinct values per
  dim. Ragged grids (unequal inner-sample counts per outer ordinate)
  are legal per LRM; every isoline is interpolated/extrapolated on
  its own sample set, and cross-isoline interpolation happens on the
  bracketing isoline's result.

The emitter we need to support produces **regular grids** — every
isoline has the same inner sample set — so v1 can optimise for that
and bail to a clearer error on ragged data. Cache the flat sorted axis
vectors per dim plus a dense N-D output tensor, computed at parse
time. Ragged-grid support is a follow-up.

## Changes

### 1. `_tm_parse_control` — `src/vasim.jl` ~L732-739

Parse a possibly-multi-dim control string. Splits on `;` into
`(interp_parts, col)`. Splits `interp_parts` on `,` into per-dim
substrings. Returns something like:

```julia
struct TMControl
    dims::Vector{Tuple{Char,Char,Char}}  # (interp, extrap_lo, extrap_hi) per dim
    col::Int
end
```

v1 accepts per-dim `"1L"` (and `"1"` meaning default linear-extrap);
any other per-dim code errors with a clear message. The parser should
validate that the number of per-dim entries matches the number of
input expressions at the call site (caller passes `D` to validate).

Estimated delta: +15 lines replacing the current 1-D check.

### 2. `_tm_parse_file` — `src/vasim.jl` ~L697-730

Generalise on `n_inputs::Int`. Expected flow:

1. Parse rows (skipping `#` comments) into a `Vector{Vector{Float64}}`,
   validating consistent column count.
2. Assert `ncols == n_inputs + n_deps` and `n_deps ≥ 1`.
3. For each dim `k` in `0..n_inputs-1`, compute the sorted unique
   ordinates of column `k` — that's the dim's axis vector. Sizes
   `(n_1, n_2, …, n_D)`.
4. Validate `nrows == ∏ n_k`. Bail on ragged grids in v1.
5. Assemble a dense `Array{Float64, D+1}` of shape
   `(n_1, …, n_D, n_deps)` by indexing each row into the axis-product
   grid.

Store the axis vectors + tensor in a new struct:

```julia
struct TableData{D}
    axes::NTuple{D,Vector{Float64}}
    outputs::Array{Float64}   # size (axes..., n_deps)
    source_path::String
end
```

Specialise `D=1` at the storage level so the existing
`pwl_at_time(tbl.input, tbl.outputs[:, col], …)` call site still
works — or migrate the 1-D call site to the new helper in a single
pass.

Estimated delta: +80 lines of new parser + ~20 deleted from the 1-D
path.

### 3. Runtime N-D interp — `src/mna/table_model.jl` (new) or extend `devices.jl`

```julia
"""
    va_table_model_interp_nd(tbl::TableData{D}, inputs::NTuple{D,<:Real},
                             col::Int, extrap_codes::NTuple{D,Char})

Recursive linear interp + extrapolation matching LRM 9.21.4.
"""
function va_table_model_interp_nd(tbl::TableData{D}, inputs::NTuple{D},
                                  col::Int, extrap::NTuple{D,Char}) where D
    _interp_rec(tbl.axes, tbl.outputs, inputs, col, extrap, 1)
end

function _interp_rec(axes, tensor, inputs, col, extrap, dim_idx)
    # Base case: reached the dependent columns → return output[col]
    if dim_idx > length(axes)
        return tensor[col]
    end
    ax = axes[dim_idx]
    x = inputs[dim_idx]
    n = length(ax)
    if x <= ax[1]
        # Extrapolate off low end
        lo = _slice(tensor, dim_idx, 1)
        hi = _slice(tensor, dim_idx, 2)
        y0 = _interp_rec(axes, lo, inputs, col, extrap, dim_idx + 1)
        y1 = _interp_rec(axes, hi, inputs, col, extrap, dim_idx + 1)
        slope = (y1 - y0) / (ax[2] - ax[1])
        return y0 + slope * (x - ax[1])
    elseif x >= ax[end]
        # Extrapolate off high end
        lo = _slice(tensor, dim_idx, n - 1)
        hi = _slice(tensor, dim_idx, n)
        y0 = _interp_rec(axes, lo, inputs, col, extrap, dim_idx + 1)
        y1 = _interp_rec(axes, hi, inputs, col, extrap, dim_idx + 1)
        slope = (y1 - y0) / (ax[n] - ax[n - 1])
        return y1 + slope * (x - ax[n])
    else
        i = searchsortedfirst(ax, x)
        lo = _slice(tensor, dim_idx, i - 1)
        hi = _slice(tensor, dim_idx, i)
        y0 = _interp_rec(axes, lo, inputs, col, extrap, dim_idx + 1)
        y1 = _interp_rec(axes, hi, inputs, col, extrap, dim_idx + 1)
        t = (x - ax[i - 1]) / (ax[i] - ax[i - 1])
        return y0 + t * (y1 - y0)
    end
end
```

Base case (D=1) should reduce to a call equivalent to the existing
`pwl_at_time` so we keep identical numerics on the regression tests.

`_slice(tensor, dim_idx, i)` is `selectdim(tensor, dim_idx, i)` —
`selectdim` is allocation-free in Julia for a single index.

Needs careful attention to **type stability** for `ForwardDiff.Dual`
values — the same pattern as `pwl_at_time` at `devices.jl:47-73`
(explicit `type_stable_time = 0.0 * t`). Recursive helpers propagate
the dual type through the `t * (y1 - y0)` term automatically; spot-check
with a sensitivity test.

Estimated delta: +150 lines in a new `src/mna/table_model.jl`
including the struct, parser, and interp helpers (moved out of
`devices.jl` / `vasim.jl` for cleanliness).

### 4. Codegen dispatch — `src/vasim.jl` ~L962-992

Change:

- Accept `length(stmt.args) ≥ 3` instead of `== 3`; derive `D = len -
  2`.
- Parse each of the first `D` args as input expressions via
  `to_julia`.
- Extract filename + control strings from the trailing two args as
  today.
- Call `_tm_parse_control(ctrl, D)` — pass D for validation.
- Call the (new) `_tm_parse_file(abspath(filename); n_inputs=D)`.
- Emit the interp call. Keep the existing 1-D path bit-identical by
  specialising on D:

  ```julia
  if D == 1
      # Current bakery: SVector of xs + SVector of ys + pwl_at_time
      # unchanged.
  else
      # Bake the axis tuple + flat output tensor into the generated
      # function via nested SVector or a const `TableData{D}` literal,
      # then call va_table_model_interp_nd.
  end
  ```

SVector nesting past D=2 starts to get unwieldy — probably emit the
axes as individual SVectors and the output tensor as a plain
`Array{Float64, D+1}` literal inside a `let` block. World-age is safe
because the array literal is built inside the generated function, not
through an `eval`.

Estimated delta: +40 lines in the codegen branch.

### 5. Tests — `test/mna/table_model.jl`

Extend existing test file. New fixtures:

- 2-D fixture: wl × T isoline file, 5 × 3 grid with a known linear
  function so interp is exact. Test interior + 8 corners + extrap.
- 3-D fixture: 3 × 3 × 3 on a cubic polynomial. Verify linear interp
  has the expected second-order error.
- Parse-error tests: ragged grid (unequal inner sizes per outer),
  mismatched columns, control-string dim count mismatch.
- Sensitivity: pass a `ForwardDiff.Dual` input, verify non-zero
  partial where appropriate, zero where the input is off the
  dimension's axis.
- Regression: re-run the existing 1-D tests unchanged — they must
  pass bit-identically on the new code path.

Estimated delta: +200 lines.

## Scope, sequence, and non-goals

### Phased implementation

1. Split the file parser + interp out of `vasim.jl` into
   `src/mna/table_model.jl`. Keep 1-D-only behavior identical.
2. Generalise the parser + codegen to `D ≥ 1`, specialise storage on
   `D=1` so no runtime cost for the common case.
3. Recursive N-D interp helper, wired into codegen for `D ≥ 2`.
4. Test coverage (2-D, 3-D, sensitivity).

### Non-goals

- Per-dim heterogeneous interpolation (e.g. cubic in wl, discrete in
  mode index). LRM allows it; can add later.
- Ragged grids. Bail with a clear error in v1.
- Inline-array data source (`table_model_array`).
- `ahdl_include` fix for Spectre — tracked separately
  (`src/spc/cache.jl:51`).
- Differentiable table inputs (input depending on solver node
  voltages). Our use case is always `.param`-driven.

## Line count estimate

- parser/control: +15
- file parser with axis detection: +80
- struct + cache: +20
- codegen N-D branch: +40
- runtime recursive interp: +150
- tests: +200

Total: ~500 lines added, ~20 removed. ~1 focused week of work.

## Follow-ups (once v1 lands)

- `"D"` nearest-neighbour, `"2"`/`"3"` splines per LRM 9.21.4. Spline
  support tends to be the biggest ask because ring-resonator S-parameters
  are smoother than a linear fit can reasonably approximate at coarse
  grids.
- `"C"`/`"E"` extrapolation.
- Ragged grids (each isoline has its own inner sample set).
- Investigate whether exposing the `TableData{D}` cache keyed by path
  across modules helps large PDK flows where many devices share one
  characterization file.
