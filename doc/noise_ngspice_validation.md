# Validating `noise!` against ngspice

What `test/noise_ngspice.jl` pins down, how to regenerate its reference, and the
two defects it found — one fixed here, one still open.

Until now `noise!` was checked only against closed-form results (`test/noise.jl`:
4kT·R shaped by an RC pole, and its band integral kT/C). That validates the
analysis — the adjoint solve, the incoherent sum, the input referral — but not
the *device* noise models, which is the half a designer actually depends on: the
shot/flicker split of a junction, the channel thermal noise of a MOSFET, the five
sources SPICE3 gives a BJT. Those are worth checking against a simulator that has
had them for thirty years.

## One deck, two simulators

The decks live in `test/ngspice/noise/` and are read by *both* simulators. Each
carries its own `.noise` and `.print noise` cards, so

```
ngspice -b test/ngspice/noise/diode.cir
```

prints the reference table, and Cadnip loads the same file — sema already accepts
and ignores analysis and output cards, so nothing had to change for this. There
is no second copy of the netlist to drift out of step with the reference, which
is the failure mode of the inline-netlist style used in `test/ac.jl`.

ngspice reports `onoise_spectrum`/`inoise_spectrum` as spectral densities in
V/√Hz; `noise!` reports a PSD in V²/Hz, so the test compares against
`sqrt.(ns[:onoise])`.

| deck | what it exercises |
|------|-------------------|
| `rc_lowpass.cir` | resistor thermal noise, RC shaping, input referral |
| `diode.cir` | junction shot noise + flicker (`kf=1e-14 af=1`), corner ≈ 30 kHz |
| `mos1.cir` | level-1 MOSFET channel thermal noise |
| `mos1_flicker.cir` | the same stage with `kf=1e-30 af=1` — **broken**, see below |
| `bjt.cir` | the whole SPICE3 BJT model: rb/re/rc thermal, base and collector shot, base flicker |

Agreement is to ~1e-7 relative on every passing deck. The test asserts 1e-4,
whose floor is not either simulator: ngspice carries the pre-2019 CODATA
Boltzmann constant, worth 2e-5 in a thermal PSD.

## The bug this found: a limit variable read out from under itself

`mos1.cir` came out flat at the wrong level and its `inoise` was off by exactly
2×. The operating point said why:

```
ngspice   @m1[id] = 1.029412e-03   @m1[gm] = 2.058824e-03   v(d) = 2.941176
Cadnip    m1_i_d  = 2.561029e-04   m1_gm  = 1.024412e-03    v(d) = 2.941176(5)
```

The node voltage agreed to seven digits while the device's own reported bias did
not: `m1_vgs` read 1.5 V where the gate sits at 2.0 V, and everything downstream
— `gm`, `id`, the AC gain, the channel noise PSD — followed from that wrong bias.
Bisecting the model card put it on `tox`, which is what gives a level-1 MOSFET a
gate charge.

A charge is a state variable, and a state variable moves the ones after it.
`resolve_index` lays the system out as `[nodes | currents | charges | limits]`
and says so in its own docstring: it is correct **at assembly**, when the counts
are final. `limit!` was calling it mid-pass, to read the voltage the device
evaluated at last iteration out of `x`:

```julia
lidx = alloc_limit!(ctx, base_name, instance_name, p, n; init)
li = resolve_index(ctx, lidx)          # n_charges is still climbing here
vold = x[li]
```

`sp_mos1` allocates its four limit variables before its gate charge, so at that
moment `n_charges == 0` and the read landed one slot early — on the charge
itself. `fetlim` then limited from a "previous voltage" of 0.046 (a scaled
charge) and returned 1.5 V.

Why the DC solution survived it: Newton runs on a `DirectStampContext`, whose
structure is precompiled and whose counts are therefore final before any stamping
happens. Only rebuilds on an `MNAContext` were affected — which is to say the
reported operating point, `ac!` and `noise!`, and nothing that would show up as a
failed solve. A silent 2× error in every small-signal quantity of a MOSFET with a
gate charge.

The fix is `limit_state_index` (`src/mna/context.jl`), used by `limit!` and by the
`$limit` preamble the Verilog-A lowering emits. It resolves against
`prev_limit_base` — where the limit block sat in the pass that laid out `x` —
rather than where it will sit in the pass being built. `reset_for_restamping!`
snapshots that offset before clearing, guarded on "a pass actually ran since the
last reset": resets come in pairs (a caller resets, then the builder resets again
on the way in), and an unguarded snapshot would record the zero the first reset
just wrote.

On a `DirectStampContext` the two indices agree by construction, so the hot path
is untouched — `limit_state_index` is `resolve_index` there.

Two smaller things fell out of the same investigation:

- `ac!` and `noise!` did their own single cold structure pass (`x = ZERO_VECTOR`)
  instead of the multi-pass `build_with_detection` every other path uses. On the
  `tox` deck that sized their context at 9 states against a 10-long `dc_sol.x`.
  Both now call `build_with_detection`.
- `build_with_detection` and `_detect_structure` were the same five-pass loop
  written twice; the former now delegates to the latter.

## Still open: the level-1 MOSFET has no flicker noise

`mos1_flicker.cir` is the same stage with `kf=1e-30 af=1`. ngspice's curve is 1/f
across the whole band (5.76e-6 V/√Hz at 1 Hz, down to 1.24e-8 at 1 MHz); Cadnip's
is the white floor of the `kf`-less deck. `sp_mos1` *registers* a flicker source —
it appears in `ns.contributions` as `:m1_flicker` — and its PSD is identically
zero.

Measured: the PSD does not respond to `kf` (1e-30 and 1e-20, ten orders apart,
give the same curve) and does not respond to `nlev`. In
`models/VADistillerModels.jl/va/mos1.va` the flicker power is the one quantity in
the noise block assigned inside a `case (nlev)`:

```verilog
case (nlev)
  0: begin flicker_psd = kf*exp(af*ln(...))/((MOS1l-2*ld)*(MOS1l-2*ld)*sqrt(coxSquared)); ... end
  ...
endcase
I(d_int, s_int) <+ flicker_noise(flicker_psd, flicker_exp, "flicker");
```

`flicker_psd` is a local initialised to 0. Everything else in that block — the
`rd`/`rs` thermal sources, the `id` channel source, whose `Sid` comes out of an
ordinary `if`/`else` — is right against ngspice, and every other device's flicker
term is right (the diode and BJT decks pass, flicker included). That is what
points at the lowering of `case` rather than at the model or at the noise
channel. Not yet confirmed: nobody has read the generated code for this module.

`test/noise_ngspice.jl` carries the deck and a `@test_broken`, so it will
announce itself when the underlying gap closes.

## Adding a deck

1. Write the netlist in `test/ngspice/noise/`, ending with its own `.noise` and
   `.print noise onoise_spectrum inoise_spectrum` cards, and give the input
   source an `AC 1` — ngspice refuses to run `.noise` without one.
2. `ngspice -b test/ngspice/noise/<deck>.cir` and paste the two printed columns
   into the test as a reference table.
3. `Base.include(@__MODULE__, SpiceFile(...))` at the top of
   `test/noise_ngspice.jl` — the builder takes the filename's stem — and add a
   `@testset` comparing against it.

Keep the grid `dec 2 1 1e6`; the reference tables are row-aligned to
`NGSPICE_FREQS` positionally.
