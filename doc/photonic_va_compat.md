# Cadnip Simulation Limitations for Photonic VA Models

Status as of 2026-03-31. This documents limitations encountered when porting `src/tests/test_verilog_a.py` (ngspice/OSDI) to `test_cadnip.jl` (Cadnip MNA backend).

## Setup

- Verilog-A models live in `compact_models/verilog_a/`
- Standard `disciplines.vams` and `constants.vams` were copied from `Cadnip.jl/models/PhotonicModels.jl/va/` into that directory so the parser can resolve `include` directives
- OSDI files for macOS were compiled with `/Users/pepijndevos/Downloads/vacask_0.3.2-dev2_darwin-arm64/simulator/openvaf-r` (the originals were Linux x86-64 ELF binaries)
- The Linux OSDI originals are backed up in `compact_models/verilog_a/osdi_linux_backup/`

## Bug fixes applied to Cadnip

### 1. Preprocessor VirtPos underflow (`VerilogAParser.jl/src/parse/preproc.jl`)

**Problem**: When a VA file uses `` `include "disciplines.vams" ``, the preprocessor creates chunk tree nodes for the included content. Between two adjacent macro expansions, the text chunk can be empty (`fileend < filestart`). The `virtrange()` function at line 198 computed `virtstart + (fileend - filestart)` where the subtraction produced a negative value, causing `UInt32` underflow (`InexactError: trunc(UInt32, -1)`).

**Fix**: Added an early return for empty text chunks:
```julia
fileend < filestart && return virtstart:(virtstart - 1)
```

**Location**: `VerilogAParser.jl/src/parse/preproc.jl:198`

### 2. Stamp hoisting scope bug (`src/vasim.jl`)

**Problem**: When a VA model has an `if/else` block with voltage contributions in both branches (e.g. `optical_phase_shifter_simp`'s `include_time_delay` conditional), the hoisting optimization collects all `let I_var = alloc_current!(...)` bindings into a flat `let_var_map`. Since the let variable is always named `I_var`, later bindings overwrite earlier ones. All stamp `get_G_idx!` calls end up using the last allocation's hoisted symbol, wiring both `V(optReal_out)` and `V(optImag_out)` to the same branch current variable. This produces incorrect results (both outputs get the same value).

**Fix**: Made `collect_stamp_calls!` scope-aware. It now accepts the `allocs` vector and tracks which `AllocInfo` each let binding corresponds to via identity comparison (`alloc.original_expr === expr`). Each `StampInfo` carries a `scope::Dict{Symbol, Symbol}` snapshot of the let_var-to-hoisted_sym mapping at its location in the tree. The stamp resolution step uses `stamp.scope` instead of the global `let_var_map`.

**Location**: `src/vasim.jl` — `StampInfo` struct, `collect_stamp_calls!`, and the stamp resolution loop in `hoist_conditional_stamps`.

## Parser limitations

### `@(initial_step)` not supported

The VerilogAParser does not handle `@(initial_step)` event control statements inside `analog begin ... end` blocks. It hits a `TODO` error when encountering the `AT_SIGN` token.

**Affected models** (6 of 22):

| Model | File | What `@(initial_step)` does |
|---|---|---|
| `coupler_2x2` | coupler_2x2.va | Computes `loss`, `alpha`, `t_bar`, `t_cross`, `mach_prec` |
| `optical_combiner_2x1` | optical_combiner_2x1.va | Computes `loss`, `alpha`, `t_top`, `t_below` |
| `optical_loss_phase_generic` | optical_loss_phase_generic.va | Computes `pi`, `mach_prec`, `transmission`, `phase` |
| `triggered_optical_source` | triggered_optical_source.va | Initializes `mach_prec` |
| `photodiode` | photodiode.va | Temperature-dependent initialization |
| `diode_basic` | diode_basic.va | Lauritzen-Ma model initialization |

**Workaround**: In all these models, the `@(initial_step)` block only computes values from parameters (no state dependency). Moving those assignments out of the `@(initial_step)` block and into the main `analog begin` body would make them parse. This changes semantics slightly (recomputed every iteration instead of once) but is functionally identical for DC analysis since the values are parameter-only.

**Example transformation**:
```verilog
// Before (fails to parse):
analog begin
  @(initial_step) begin
    mach_prec = 1e-15;
    loss = pow(10.0, -0.1 * excess_loss_dB);
  end
  // ... rest of model
end

// After (parses fine):
analog begin
  mach_prec = 1e-15;
  loss = pow(10.0, -0.1 * excess_loss_dB);
  // ... rest of model
end
```

**Proper fix**: Add `@(initial_step)` parsing support to `VerilogAParser.jl/src/parse/parse.jl` around line 377 (`parse_primary`), where the `AT_SIGN` token triggers the TODO error. The PhotonicModels library in Cadnip already supports `@(initial_step)` semantics in codegen (see `src/vasim.jl` photonic integration tests), so only the parser needs updating.

### `diode_basic.va` parse error

This model reports `PARSE_ERR` (not a crash) even though it does use `@(initial_step)`. The error may be related to additional Verilog-A constructs used in the Lauritzen-Ma diode model. Not investigated further.

## MNA backend limitations

### `I(a,b)` probe in contributions

The MNA backend does not support reading branch current `I(a,b)` as a probe inside contributions. It throws: `"I(a,b) probe not supported in MNA contribution"`.

**Affected models** (2 of 22):

| Model | File | Usage |
|---|---|---|
| `SkinEffectResistor` | rse.va | `V(A, B) <+ I(A, B) * (R_dc + R_se)` |
| `CapacitorWithDf` | cap_with_df.va | `V(mid, n) <+ I(mid, n) * esr` |

Both models use `I(a,b)` to implement resistive elements (V = I*R). The SkinEffectResistor is a simple DC resistance. The CapacitorWithDf uses it for ESR modeling.

**Workaround**: Replace with equivalent current contributions:
```verilog
// Before (not supported):
V(A, B) <+ I(A, B) * R;

// After (equivalent):
I(A, B) <+ V(A, B) / R;
```

Or use Cadnip's native `Resistor` for the resistive parts.

**Proper fix**: Implement `I(a,b)` probe support in the MNA stamp codegen. This requires allocating a branch current variable for the probed branch and making it available as a readable value in the contribution expressions.

### Nonlinear DC convergence with multi-device optical chains

The MNA nonlinear DC solver fails to converge for circuits with multiple interconnected optical VA devices, specifically the MZM chain topology (source -> splitter -> 2x phase_shifter -> combiner -> photodetector).

**Symptom**: `Warning: Nonlinear DC solve did not converge` — the solution vector contains near-zero values instead of the expected optical field amplitudes.

**Root cause**: The optical models use `mach_prec = 1e-15` threshold logic:
```verilog
rcomp = mach_prec;
if (abs(V(optReal_in)) > mach_prec) begin
    rcomp = V(optReal_in);
end
```
This creates discontinuities in the Jacobian that the Newton-Raphson solver struggles with when multiple devices are chained. Each device's output depends on its input, creating a cascade of discontinuous functions.

**Working**: Per-voltage-point simulations (the voltage sweep test) work because each creates a fresh circuit. Individual devices and simple 2-device circuits also converge fine.

**Failing**: The 6-device MZM chain with shared nonlinear state. The `test_cadnip.jl` marks 3 tests as `@test_broken` for this.

**Potential fixes**:
1. **Smoothing**: Replace the hard `if/else` threshold with a smooth approximation (e.g., `rcomp = mach_prec + V * tanh(V / mach_prec)`)
2. **Source stepping**: Start with small source amplitudes and ramp up
3. **Better initial guess**: Use a linear pre-solve to get initial node voltages closer to the operating point
4. **GMIN stepping**: Temporarily add larger minimum conductances and reduce

## Model-by-model status

| Model | Parse | Simulate | Notes |
|---|---|---|---|
| optical_source | OK | OK | 0/10 dBm tests pass |
| mmi_splitter_1x2 | OK | OK | 50/50 power split verified |
| mmi_2x2_combiner | OK | OK | Single-input 50/50 split verified |
| optical_phase_shifter_ideal | OK | OK | Zero bias and pi-shift verified |
| optical_phase_shifter_simp | OK | OK | Power conservation verified (requires hoisting fix) |
| optical_waveguide | OK | OK | Zero-loss and with-loss verified |
| optical_pd | OK | OK | Field magnitude output verified |
| photodiode_ideal | OK | OK | Photocurrent and dark current verified |
| optical_termination | OK | OK | Smoke test passes |
| mzm_circuit (flattened) | OK | OK | Nonzero output and symmetric bias verified |
| diode_va (diode_model) | OK | OK | Forward and reverse bias verified |
| rc_network | OK | OK | DC open-circuit verified |
| coupler_2x2 | FAIL | - | `@(initial_step)` not parsed |
| optical_combiner_2x1 | FAIL | - | `@(initial_step)` not parsed |
| optical_loss_phase_generic | FAIL | - | `@(initial_step)` not parsed |
| triggered_optical_source | FAIL | - | `@(initial_step)` not parsed |
| photodiode | FAIL | - | `@(initial_step)` not parsed |
| diode_basic | FAIL | - | Parse error (not investigated) |
| SkinEffectResistor (rse) | OK | FAIL | `I(a,b)` probe not supported |
| CapacitorWithDf (cap_with_df) | OK | FAIL | `I(a,b)` probe not supported |
| resistor_va | OK | not tested | Trivial model |
| MZM chain (multi-device) | OK | PARTIAL | Convergence issues with chained devices |

## Files

- `test_cadnip.jl` — Julia test file (equivalent to `src/tests/test_verilog_a.py`)
- `compact_models/verilog_a/disciplines.vams` — Standard Verilog-A discipline definitions (copied from Cadnip PhotonicModels)
- `compact_models/verilog_a/constants.vams` — Standard Verilog-A constants (copied from Cadnip PhotonicModels)
- `compact_models/verilog_a/osdi_linux_backup/` — Original Linux x86-64 OSDI files
