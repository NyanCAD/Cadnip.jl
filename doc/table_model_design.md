# `$table_model` Support in Cadnip — Design

## Goal

Add enough `$table_model` (LRM 9.21) support to Cadnip for **running
SAX-derived photonic compact models emitted by sax_to_verilog.py**
end-to-end, as a round-trip check against HSpice (the actual target
simulator for these models). The emitter lives at
`/Users/pepijndevos/code/sax-in-the-sheets-verilog-in-the-streets/sax_to_verilog.py`
and produces, per SAX model:

- a `.va` module with scalar `parameter real wl`, array ports
  `inout [0:1] oN; electrical [0:1] oN;`, and one
  `$table_model(wl, table_file, "1L;k")` call per Re/Im matrix
  element, feeding a standard KCL `I(o_i[k]) <+ …` stamping of Y·V;
- a companion `.tbl` text data file in LRM 9.21.1 column format
  (col 1 = wl, subsequent columns = Re/Im of each Y_ij).

Python round-trip (`verify_table.py` in that repo) confirms the
emission matches SAX to ~1e-6 with 201 samples across a 20 nm band
for a 10 µm straight waveguide. HSpice accepts the pair natively.
Cadnip currently does not — this doc scopes the minimal work to
close that gap.

## Non-goals / future scope

Explicitly **out of v1**:

- Multi-dimensional tables (2D+ sweeps). v1 is strictly 1-D input.
- Interpolation schemes other than linear (`1`). Quadratic (`2`),
  cubic splines (`3`), and discrete (`D`) can be added incrementally.
- Extrapolation methods other than linear (`L`). Constant (`C`) and
  error (`E`) are easy follow-ups; skip for now.
- Column-ignore (`I`) control char. Our emitter doesn't use it.
- Inline array-literal data source (LRM's
  `table_model_array` variant). File-based only.
- `$table_model` taking a node voltage as input (would require a
  Jacobian slope contribution on the bracketing interval). Our
  input is always a `parameter real`, so ∂/∂V = 0 everywhere.
- Supporting `parameter string` generally across the parser if it's
  missing — only need enough for `$table_model`'s filename arg,
  which can also accept a string literal directly.

Coverage target: exactly what sax_to_verilog.py emits today. If we
want 2-D (e.g., wl × temperature) later, we extend the emitter and
the Cadnip runtime together.

## LRM syntax summary

What we must parse and interpret:

```
$table_model(input_expr, filename_or_string_param, control_string)
```

- `input_expr`: any real expression. In our case, the module
  parameter `wl`. Always a compile-time-resolvable scalar at stamp
  call time (it's a sweep parameter, not a node voltage).
- `filename_or_string_param`: either a string literal `"foo.tbl"`
  or an identifier bound to a string parameter. Our emitter uses
  a `parameter string table_file = "foo.tbl"` with a literal default.
- `control_string`: string literal. Our emitter always produces
  `"1L;k"` where `k` is a positive integer (1-based dependent
  column index per LRM 9.21.2). Parse the `k` at compile time —
  it's not a runtime expression.

## Files & existing patterns to follow

The closest analog in the codebase is `src/mna/laplace.jl` +
`src/vasim.jl:874-889` (codegen for `laplace_nd`). Pattern:

1. Codegen recognises the system function name, extracts the
   arguments, and emits a call to a runtime helper in
   `Cadnip.MNA`.
2. Runtime helper does the heavy lifting (polynomial → DSS for
   laplace_nd; file parse + interpolation for `$table_model`).
3. Per-instance state (if any) is allocated via the stamp machinery.

For `$table_model`, the state is a *global* cache keyed by resolved
filename — LRM 9.21.1 says "state is captured on the first call
… any change after this point is ignored". One parse per file per
simulation run, shared across every call and every instance.

## Changes to make

### 1. Parser — `NyanVerilogAParser.jl`

Verified state:

- `$table_model(...)` call syntax: should already parse. System
  identifiers route through `SYSTEM_IDENTIFIER` → `FunctionCall`,
  and variadic mixed-type args (real expr, string literal, string
  literal) are all accepted by the existing call-expression grammar.
  Worth a smoke test; no changes expected.
- String literals as function args: already supported (used for
  `$strobe`, `$warning`).
- `parameter string foo = "..."`: **already supported**. Verified
  at `NyanVerilogAParser.jl/src/tokenize/token_kinds.jl:305` —
  `is_parameter_type(k) = k in (INTEGER, REAL, REALTIME, TIME,
  STRING)`. Parse path at `parse.jl:501-548`. No parser work needed.

One remaining caveat at `parse.jl:528`: the parameter parser still
has `@assert prange === nothing # TODO` blocking ARRAY-typed
parameters (e.g., `parameter real tbl[0:N]`). That's for the
inline-array data-source variant of `$table_model` which we're
*not* implementing — leave the assertion alone.

### 2. Runtime helper — new `src/mna/table_model.jl`

Mirror the `src/mna/laplace.jl` layout.

```julia
# src/mna/table_model.jl
using DelimitedFiles  # or a bespoke parser — LRM allows '#' comments

struct TableData
    # Column 1 = input; columns 2..K+1 = dependent variables.
    input::Vector{Float64}   # sorted ascending
    outputs::Matrix{Float64} # size (n_samples, n_deps)
end

const _table_cache = Dict{String, TableData}()   # keyed by absolute path

function _load_table(path::AbstractString)::TableData
    abs = abspath(path)
    get!(_table_cache, abs) do
        _parse_table_file(abs)
    end
end

function _parse_table_file(abs::AbstractString)::TableData
    # Skip '#'-prefixed comment lines and blanks; split rest on
    # whitespace. All rows must have the same column count. Column
    # 1 is input, rest are dependents. LRM allows unsorted input,
    # with auto-sort — implement it.
    ...
end

"""
    va_table_model_interp(filename, col_index, input)

Evaluate a LRM 9.21 `$table_model` call with control string "1L" on
dependent column `col_index` (1-based). Linear interpolation in the
bracketing interval; linear extrapolation off either end.
"""
@inline function va_table_model_interp(
    filename::AbstractString, col_index::Int, input::Real,
)::Float64
    tbl = _load_table(filename)
    _linear_interp_extrap(tbl.input, view(tbl.outputs, :, col_index), input)
end

@inline function _linear_interp_extrap(xs, ys, x)
    # searchsortedfirst gives the index of the first xs[i] ≥ x.
    n = length(xs)
    i = searchsortedfirst(xs, x)
    if i == 1
        # Extrapolate off the low end using slope of segment 1.
        slope = (ys[2] - ys[1]) / (xs[2] - xs[1])
        return ys[1] + slope * (x - xs[1])
    elseif i > n
        # Extrapolate off the high end.
        slope = (ys[n] - ys[n-1]) / (xs[n] - xs[n-1])
        return ys[n] + slope * (x - xs[n])
    else
        # Interpolate.
        x0, x1 = xs[i-1], xs[i]
        y0, y1 = ys[i-1], ys[i]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    end
end

export va_table_model_interp
```

Cache thread-safety: for v1, a plain `Dict` under the assumption
DC/AC sweeps are single-threaded. If sweeps become parallel,
replace with `ReentrantLock` + read/check/insert.

File path resolution: start with `abspath` relative to the current
working directory. The `.tbl` lives next to the `.va`; users drop
both into the simulation dir. Document this. A future extension
could thread the .va file's directory as a search path but that's
plumbing we don't need yet.

### 3. Codegen hookup — `src/vasim.jl`

Verified location: the system-function dispatch starts at
`src/vasim.jl:940` (`$temperature` branch), with `$vt` at 942,
`$param_given` at 944, `$simparam` at 949. A second cluster in the
statement-level handler around line 1863 covers `$warning`,
`$strobe`, `$error`, `$discontinuity`. The closest full-featured
analog is the `laplace_nd` branch at line 882 — shows how to pull
typed arguments out of the AST and emit a call to a runtime helper.

Add a branch:

```julia
elseif fname == :table_model
    # $table_model(input, filename, control_string)
    @assert length(stmt.args) == 3 "\$table_model requires 3 arguments"
    input_expr = to_julia(stmt.args[1])
    filename_expr = _tm_extract_filename(stmt.args[2])   # string literal or param ref
    ctrl = _tm_extract_ctrl_string(stmt.args[3])         # parse "<interp>;<k>"
    col = _tm_parse_col(ctrl)                             # compile-time Int
    # Only "1L" interp supported in v1; error out on others.
    _tm_assert_linear(ctrl)
    return :(Cadnip.MNA.va_table_model_interp($filename_expr, $col, $input_expr))
end
```

Helpers parse compile-time args:

- `_tm_extract_filename` pulls the filename either from a
  `StringLiteral` AST node (common path) or resolves a
  `parameter string` reference through the normal param-lookup
  mechanism.
- `_tm_extract_ctrl_string` requires a `StringLiteral` — control
  strings are always compile-time constants in our emission.
- `_tm_parse_col` splits on `;`, takes the integer after it.
  v1: raise on missing (default is column N+1, but we don't need
  that path).
- `_tm_assert_linear` checks the first substring is `"1L"`; raise
  a nice error on anything else.

No changes to the stamp-hoisting logic since `$table_model` is a
pure function with no state nodes — it's as simple as `$temperature`
from the hoister's point of view. Verify by looking at how the
hoister treats `$temperature` calls.

### 4. Tests — new `test/mna/table_model.jl`

Three layers of coverage:

1. **Runtime helper unit test.** Feed a small in-memory table via
   the parser, call `va_table_model_interp` at sample points (should
   return exact sample values) and mid-sample points (should linearly
   interpolate) and off-the-end points (should linearly extrapolate).
   No Cadnip codegen involved.

2. **Codegen round-trip on a toy module.** Hand-write a tiny
   `test_tm.va`::

   ```verilog
   `include "disciplines.vams"
   module test_tm(p, n);
       inout p, n;
       electrical p, n;
       parameter real wl = 1.55;
       real g;
       analog begin
           g = $table_model(wl, "test_tm.tbl", "1L;1");
           I(p, n) <+ g * V(p, n);
       end
   endmodule
   ```

   and a `test_tm.tbl`::

   ```
   1.54  0.1
   1.55  0.2
   1.56  0.3
   ```

   Sweep `wl` across the band, confirm node voltages match the
   expected KCL solution for the interpolated `g`.

3. **End-to-end with the sibling repo's emitted pair.** Copy
   `mmi2x2.va` + `mmi2x2.tbl` (and `disciplines.vams`) into a test
   directory, sweep `wl`, assert each port voltage matches the
   `verify_table.py` reference output to 1e-8.

## Open questions to resolve during implementation

1. **String parameter plumbing to the stamp call.** The emitted
   code wants an expression like
   `Cadnip.MNA.va_table_model_interp(some_instance_param, col, input)`.
   `some_instance_param` should be the string value of
   `parameter string table_file`. Walk through the `laplace_nd`
   codegen path with a string parameter and confirm it arrives as
   a plain Julia `String` — the param-plumbing is set up for
   `parameter real`, so some adjustment may be needed for the
   string case. If adjustment is heavy, fall back to requiring the
   filename as a compile-time string literal inside each
   `$table_model` call (the emitter already produces a literal
   default; the instance-override capability is the only thing
   lost).
2. **Thread safety of `_table_cache`.** Start single-threaded;
   revisit if a sweep becomes parallel.
3. **File path in error messages.** When parsing fails or the
   file is missing, report the path that was attempted so the user
   can fix it without trial and error.
4. **Resolving relative paths.** cwd is the v1 answer. HSpice
   resolves relative paths against the netlist include path;
   follow that convention once we're willing to thread the
   netlist directory into the runtime helper.

## Effort estimate

- Parser spike + any `parameter string` patch: <1 day. Maybe zero
  work if it already accepts strings.
- Runtime helper (file parser + interpolator): half a day. Cached
  dictionary + linear interp is trivial; the tricky part is being
  lenient about comment lines, blank lines, column-count checks,
  and error messages.
- Codegen hookup: half a day. One new branch in the system-function
  dispatch, three small helpers to parse compile-time args.
- Tests: half a day across the three layers.
- Running `mmi2x2.va` + `mmi2x2.tbl` end-to-end and fixing whatever
  breaks: open-ended but usually a few hours.

Realistic total: 2–3 focused days.

## Out-of-scope complement: SaxInTheSheets.jl (native device)

A separate track — not implemented here, but referenced because it's
the other obvious direction. If in the future we want live
parameter-sweep integration (where Cadnip calls Python SAX code at
each sweep point instead of reading a precomputed table), that's the
`SaxInTheSheets.jl` subpackage concept. It uses PythonCall and a
native Julia Cadnip device, bypassing Verilog-A entirely. It's
faster for design-space exploration (no re-emission per design
point) but requires a Python runtime and doesn't produce shareable
`.va` artifacts. The `$table_model` path above is the right choice
for the HSpice-compatibility use case; `SaxInTheSheets.jl` is the
right choice for tight SAX-Cadnip iteration. They don't conflict.
