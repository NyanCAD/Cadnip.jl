# Model binning

A foundry PDK does not give one `.model nch nmos …` card. It gives a family —
`nch.1`, `nch.2`, … — each card fitted over a window of channel geometry, and
every instance line names the bare `nch`:

```spice
.model nch.1 nmos level=54 lmin=0    lmax=0.2u wmin=0   wmax=1u  …
.model nch.2 nmos level=54 lmin=0.2u lmax=1u   wmin=0   wmax=1u  …
M1 d g s b nch w=0.5u l=0.1u          $ picks nch.1
M2 d g s b nch w=0.5u l=0.5u          $ picks nch.2
```

Every open-source PDK worth testing against (sky130, gf180, ihp) bins this way,
so nothing that reads one works without it.

## What was there before

The runtime half of binning survived the deletion of the pre-MNA codegen path
(`doc/codegen_unification.md` §5): `BinnedModel`, `find_bin`, the `spicecall`
entry. Sema's front half survived too — `binning_rx` recognises `nch.1`, and
`provided_binned_models` keeps the base name `nch` from being reported as an
unresolved model. The producer in the middle — the codegen step that aggregates
the numbered cards into a `BinnedModel` — went with `codegen!`, and had never
run even before that.

What the MNA path did with a binned deck, measured before this change:

```
MethodError: no method matching sp_diode(; is::Float64, lmin::Float64,
                                          lmax::Float64, wmin::Float64,
                                          wmax::Float64)
```

Each numbered card was lowered as an ordinary model card, so the four window
bounds were handed to the device constructor as if they were device parameters.
They are not, and the bare `nch` the instances named was bound to nothing at all.

## Where the window lives

The deleted design read the window back off the *model card*: `find_bin`
destructured `(; LMIN, LMAX, WMIN, WMAX) = bin.model`. That only works for a
device model that happens to declare those four fields — BSIM4 does, a level-1
MOSFET and a diode do not — and it means the bounds have to be passed through the
constructor that has no field for them, which is exactly the `MethodError` above.

So the window moved out of the model and into the bin:

```julia
struct ModelBin{M}
    lmin::Float64; lmax::Float64
    wmin::Float64; wmax::Float64
    model::M
end

struct BinnedModel{B<:Tuple}
    name::Symbol
    bins::B
end
```

`lmin`/`lmax`/`wmin`/`wmax` are stripped from a bin card's parameters before it
is lowered (`cg_model_value!(…; skip_params=BINNING_PARAMS)`) and become the
`ModelBin` around it. A model that *does* declare `LMIN` keeps its own default;
nothing about which bin was picked is a device parameter.

An unspecified bound is unbounded on that side (`0.0` below, `Inf` above), so a
family that bins on length alone need not write `wmin=0 wmax=1` on every card.

`find_bin` keeps the half-open HSPICE rule on both axes — `lmin <= l < lmax`,
`wmin <= w < wmax` — so bins that share an edge do not overlap. It keeps its
`@assume_effects :consistent :effect_free :terminates_globally @noinline`
annotations. Measured on a two-bin family:

```julia
julia> Base.infer_effects(find_bin, (typeof(bm), Float64, Float64))
(+c,+e,!n,+t,+s,+m,+u,+o,+r)

julia> Base.return_types(find_bin, (typeof(bm), Float64, Float64))
1-element Vector{Any}: …          # the bins' shared model type, concrete
```

The concrete return type is the one that matters downstream: every bin of a
family is the same device type, so the `spicecall` the instance site emits
infers to a concrete device and the stamp does not heap-box it — the same
property the module-level `const` on a plain model card exists to protect.

The `scale` field is gone. It existed to apply `.option scale` to the geometry
before comparing it against the window — but `.option scale` has to scale every
device's geometric parameters, at the instance site, and once it does the value
reaching `find_bin` is already scaled. A `scale` here would double-apply it. See
`doc/FINDINGS.rst` 2 for the state of `.option scale`.

## Codegen

`binned_model_families(state)` groups a scope's `.model` cards by base name,
ordered by the numeric suffix. Both places that lower model cards — the
module-level `const` hoist in `codegen_toplevel_models!` and the in-builder
emission in `codegen_mna!` — then emit the family binding after the bins it
names. For `.model nch.1 nmos level=1 lmin=0 lmax=1u vto=0.4 kp=100u` and its
`nch.2` sibling, `make_mna_circuit` emits (lightly wrapped):

```julia
const var"nch.1" = (Cadnip.spicecall)(Cadnip.ParsedModel,
    VADistillerModels.sp_mos1_module.sp_mos1, (vto = 0.4, kp = 1.0e-4, type = 1))
const var"nch.2" = (Cadnip.spicecall)(Cadnip.ParsedModel,
    VADistillerModels.sp_mos1_module.sp_mos1, (vto = 1.0, kp = 1.0e-4, type = 1))
const nch = (Cadnip.BinnedModel)(:nch,
    ((Cadnip.ModelBin)(0.0,    1.0e-6, 0.0, Inf, var"nch.1"),
     (Cadnip.ModelBin)(1.0e-6, 1.0,    0.0, Inf, var"nch.2")))
```

and each instance line becomes `spicecall(nch; w = …, l = …)`.

The instance site needs no new code: `spicecall(bm::BinnedModel; l, w, …)`
already dispatches through `find_bin`. What it did need is for the codegen-time
questions it asks *about* a model — what device type does this stamp (so a
PSP/BSIM-sized card takes the `invoke` path), what are its field names (for the
case-insensitive instance-parameter lookup) — to be answerable for a name that
is a family rather than a card. `model_card_def(state, sym)` answers both,
falling back to the family's first bin; every bin of a family is the same device
type, which is the whole reason they can share an instance line.

Three consequences worth naming:

- **A bin that reads a `.param` defers its whole family.** A card reading a
  `.param` cannot be a module-level `const` (the parameter is a local of the
  builder), and a `BinnedModel` has to be built in the same scope as the bins it
  names — so one deferred bin defers the family.
- **A family referenced from inside a `.subckt` is propagated into it**, bins and
  all, by `_propagate_toplevel_models!`, which regroups them in that scope. Same
  reason the non-binned case is propagated: `is_large_va_model` and `.param`
  resolution both need the card visible where the instance is.
- **`binning_rx` is now anchored** (`^(.*)\.([0-9]+)$`). Unanchored, it matched
  `nch.1x` as bin 1 of `nch`.

## Diagnostics

Two errors, both `CedarException`s so the stack trace stays out of the way:

- `NoBinError` — no card covers this geometry. Prints the geometry and every
  bin's window, which is the question you actually have when a PDK rejects a
  size.
- `NoGeometryError` — the instance line gives no `l`/`w`, so there is nothing to
  bin on. SPICE would fall back to `.option defl`/`defw`, which are not
  implemented, and guessing a bin means guessing a device.

`.model nch` written out *next to* `.model nch.1` is a deck contradicting itself.
The direct card wins — it is what the deck says `nch` is — and a warning names
the numbered cards that were left unbinned.

## Not covered

- **Binning an `r` card.** A `.model rsh.1 r …` family lowers to a NamedTuple,
  not a device model, and the resistor instance path does not go through
  `spicecall`. Codegen raises a clear error rather than emitting something that
  silently ignores the bins.
- **`.option scale`**, as above.
- **`.option defl` / `defw`**, the SPICE default geometry — see `NoGeometryError`.
- **Binning on anything but L/W.** Some PDKs also bin on `nf` or temperature;
  nothing here generalises to that yet.
