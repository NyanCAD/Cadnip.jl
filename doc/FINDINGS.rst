==========================================================
 Cadnip.jl findings from building the InSpice backend
==========================================================

Notes taken while writing :file:`InSpice/Spice/Cadnip/`, kept because most of
them are about Cadnip rather than about InSpice and are worth filing upstream:
behaviour that surprised the backend, one unreachable code path, and the feature
gaps the backend has to raise :exc:`NotImplementedError` for.

Every claim here was measured, not read off the source or the docs — twice the
documentation and once my own reading turned out to disagree with what the code
does.  Each entry carries the reproduction that produced it.

Versions
========

============================ =================================================
Cadnip                       0.14.0 (General registry) and ``main`` @ 739fdac,
                             16 commits past the 0.14.0 bump (81bec51)
Julia                        1.11.7 and 1.12.1
juliacall / PythonCall       0.9.35
============================ =================================================

The probes below were run on Julia 1.11.7; finding 4 was repeated on ``main``
and behaves identically there.  The backend's own test suite passes identically
on 1.11.7 and 1.12.1 and against both Cadnip versions, which is the basis for
saying the two versions agree.

What works
==========

Verified end to end through the backend, against analytical values: ``dc!``
(node voltages, branch currents, device terminal currents, device
operating-point variables), ``tran!`` including ``CedarUICOp`` for SPICE `uic`,
``ac!`` over an ``acdec`` grid, ``noise!`` with per-source contributions and
``total_noise``, ``with_temp`` for a temperature sweep with Newton warm starts,
and subcircuit instance-parameter overrides.

Defects
=======

1. A temperature card overrides the caller's ``MNASpec``
-------------------------------------------------------

``.options temp=<T>`` and ``.temp <T>`` compile, on the MNA path, to a rebinding
at the top of the generated builder::

    julia> sa = NyanSpectreNetlistParser.parse(IOBuffer(deck); start_lang=:spice,
                                               implicit_title=true);
    julia> string(Cadnip.make_mna_circuit(sa))
    # ".options temp=100" or ".temp 100" in the deck yields, in the body:
    spec = (Cadnip.MNA.MNASpec)(temp = 100.0, mode = spec.mode)

Three consequences, in decreasing order of severity:

**The card outranks the caller.** ``codegen_mna!`` (``src/spc/codegen.jl``,
~line 2939) has no ``isdefault`` guard, unlike the Spectre-side ``codegen!``
(~line 623) which honours an explicitly-set value over the netlist's.  So
``dc!(with_temp(circuit, T))`` is a no-op for the devices of any deck that
carries a temperature card — a temperature *sweep* silently returns the same
point over and over.  This is why the InSpice backend emits no temperature card
and warns if one reaches the deck through ``raw_spice`` or an `.include`:
InSpice writes ``.options TEMP = 27`` into every Ngspice deck, so the natural
translation would pin every simulation at 27 °C.

**The rebinding drops the rest of the spec.**  It carries ``temp`` and ``mode``
only, so ``tnom``, ``gmin``, ``gshunt``, ``srcFact`` and the four tolerances
revert to their defaults.

**Noise disagrees with the devices.**  ``noise!`` reads ``circuit.spec.temp``
(``src/noise.jl``), which the card never touches, so with a card the devices are
at the card's temperature and the noise at the spec's.  Measured on a 1 kΩ +
1 µF circuit, output PSD at 1 Hz::

    no card, default spec (27 C)     1.6575415088490336e-17
    .options temp=100 in the deck    1.6575415088490336e-17   <- unchanged
    .temp 100 in the deck            1.6575415088490336e-17   <- unchanged
    MNASpec(temp=100.0)              2.0606750425687716e-17
    with_temp(circuit, 100)          2.0606750425687716e-17

    ratio 2.0607e-17 / 1.6575e-17 = 1.2432 = 373.15 / 300.15  (exact)

*Suggested fix*: guard the MNA rebinding with ``isdefault`` as the Spectre-side
one does; carry the incoming spec's other fields through the rebinding; and give
``noise!`` the same temperature the builder ended up using rather than
``circuit.spec``.

*Status*: the second consequence is fixed; the first is now treated as
intended behavior, not a defect. ``.temp``/``.option temp`` is the netlist's
own control card — SPICE has no notion of a caller-language override
outranking it, and manufacturing that precedence inside Cadnip would mean
guessing "did the caller mean this" from whether ``spec.temp`` happens to
differ from ``MNASpec``'s own default, which is exactly the kind of ambiguous,
unmeasurable claim this document exists to avoid. (The InSpice backend writing
``.options TEMP=27`` into every deck is a translation choice on that side, not
a precedence bug here.) So the card wins unconditionally, same as before — what
changed is that the rebind now goes through ``with_temp`` instead of
reconstructing ``MNASpec`` with only ``temp``/``mode`` set, so
``tnom``/``gmin``/``gshunt``/``srcFact``/tolerances (and the ``time`` field's
type, needed for ForwardDiff duals) survive it instead of resetting to
defaults. The third consequence is unchanged and, with the card winning
unconditionally, is now simply "whenever a `.temp` card is present at all":
``circuit.spec.temp`` never reflects it, so ``noise!`` disagrees with the
devices any time a deck carries the card. Fixing that needs ``circuit.spec``
itself to carry the resolved temperature, which the codegen has no channel
back to today.

A cleaner long-term shape, not attempted here: route ``.temp`` through
``ParamLens`` the way a ``.subckt`` default or a ``.param`` already work — the
card supplies the *default*, and an explicit override (``alter(circuit;
temp=...)``, a sweep axis) is unambiguous because it is either present in the
override tree or it isn't, with no magic-default guessing. That would also
give ``circuit.spec``/``noise!`` a real value to read, closing the third
consequence for free. Two things stand in the way: ``temp`` would need a name
that cannot collide with a user's own ``.param temp=...`` in the same scope
(unlike the ``x1``/``X1`` case, both would be leaves in the same namespace),
and ``MNASpec`` is deliberately kept outside the ``params``/lens tree today —
"passed explicitly... to enable full JIT optimization" per its own docstring —
so folding temperature into it crosses a boundary the design has held to so
far.

2. ``Cadnip.SimOptions`` and ``Cadnip.options`` do not exist
-----------------------------------------------------------

::

    julia> using Cadnip
    julia> isdefined(Cadnip, :SimOptions), isdefined(Cadnip, :options)
    (false, false)

Both are referenced by ``src/spc/codegen.jl`` (~line 627), inside the option
block of ``codegen!``.  That function is reached only from ``codegen(scope)``
(~line 725), which the MNA pipeline never calls — ``make_mna_circuit`` goes
through ``codegen_mna!`` — so the branch is unreachable dead code that would
throw ``UndefVarError`` at codegen time if it ever were reached.

Practical effect: of the options its sema collects, the MNA path consumes only
``temp``.  ``.option gmin=`` and ``.option scale=`` are parsed, stored, and
dropped.

3. ``temper()`` in a ``.param`` sees no temperature at all
---------------------------------------------------------

Already known upstream — ``test/basic.jl`` marks it ``@test_broken`` — but worth
recording alongside the two above, because together they make temperature a
three-way split (builder-local spec / ``circuit.spec`` / the ``Cadnip.spec``
ScopedValue that ``temper()`` reads).  With ``.param tt='temper()*10'`` and
``R1`` = ``tt`` in a divider, V(out) stays at its 27 °C value under every
route::

    no card, default spec (27 C)     V(out) = 3.937
    .options temp=100 in the deck    V(out) = 3.937
    .temp 100 in the deck            V(out) = 3.937
    MNASpec(temp=100.0)              V(out) = 3.937
    with_temp(circuit, 100)          V(out) = 3.937
                                     (100 C would be 2.5)

4. An unreachable override fails three different ways
-----------------------------------------------------

Device instance parameters are the documented gap (``doc/parameter_overrides.md``
§1), and ``test/params.jl`` asserts they throw.  What is inconsistent is *how*,
measured against a ``V1``/``R1``/``R2`` divider whose V(out) is 2.5 V::

    # deck with no .param at all
    MNACircuit(deck; r1=(r=2e3,))      MethodError: no method matching
                                       var"##circuit#284"(::@NamedTuple{...})
    alter(circuit; r1=(r=2e3,))        silently ignored, V(out) = 2.5
    alter(circuit; var"r1.r"=2e3)      silently ignored, V(out) = 2.5
    alter(circuit; v1=(dc=2.0,))       silently ignored, V(out) = 2.5

    # same deck plus any .param
    alter(circuit; r1=(r=2e3,))        ArgumentError: unknown parameter
                                       override `r1` — the top level declares
                                       no parameter `r1`. It declares: rtop.

So §2's "an override that names nothing no longer passes silently" holds only
when the deck declares at least one ``.param``; with none, the observed tree is
empty and the override is either ignored (``alter``) or reported as a raw
``MethodError`` from the generated builder (the constructor).

*Suggested fix*: check overrides against the empty tree too, and let the
constructor path raise the same ``ArgumentError`` the ``alter`` path does.

*Status*: fixed.  A generated builder now registers its scope with the lens even
when the deck declares nothing (``codegen_mna!``), so a bare deck observes as
``(params = (),)`` — "declares nothing", which is checkable — rather than as an
empty tree indistinguishable from a hand-written builder that ignores its
``params``.  Both spellings now raise the same ``ArgumentError``, and so does a
sweep axis, which previously came back as a flat curve.

One correction to the measurements above, made while reproducing them on
``main`` @ fbce56d.  The constructor's ``MethodError`` was not a second failure
mode of the override path: it is the world-age error of finding 5, and it
appeared here only because the probe built the circuit inside a closure.  Built
at top level, the ``MNACircuit(deck; r1=(r=2e3,))`` of the table simply returned
a circuit and solved to 2.5 V, exactly like the ``alter`` rows.  What the two
spellings had in common was the silence, not the inconsistency.  The remaining
rows reproduce as written::

    # deck with no .param at all, all four measured at top level
    MNACircuit(deck; r1=(r=2e3,))      silently ignored, V(out) = 2.5
    alter(circuit; r1=(r=2e3,))        silently ignored, V(out) = 2.5
    alter(circuit; var"r1.r"=2e3)      silently ignored, V(out) = 2.5
    alter(circuit; v1=(dc=2.0,))       silently ignored, V(out) = 2.5
    CircuitSweep(deck, Sweep(vbais=…)) silently ignored, [2.5, 2.5]

    # after the fix, every one of the five
    ArgumentError: unknown parameter override `…` — the top level declares no
    parameter `…`. It declares no parameters at all.

A deck with a subcircuit instance but no top-level ``.param`` was already
checked before the fix — instances register on the observer through
``getproperty`` — which is why the gap only ever showed up on the barest decks.

5. The README's world-age example is stricter than reality
----------------------------------------------------------

The README says ``dc!(MNACircuit("amp.sp"))`` in a single statement fails with
*"method too new"*.  Measured on 0.14.0, a same-statement build and solve works
at top level; what fails is doing both inside a *function* body, where the world
is frozen at entry::

    same-statement build+solve (top level)   OK, V(out) = 2.5
    inside a function body                   MethodError
    inside a function + Base.invokelatest    OK, V(out) = 2.5

Worth correcting, because the stricter claim leads callers to reach for
``invokelatest`` where they do not need it.  A driver that builds in one call
and solves in the next — which is what a foreign-language binding does — never
needs it at all.

Dead code
=========

6. The pre-MNA codegen path is unreachable
------------------------------------------

Traced on ``main`` @ 739fdac while working out where `.options temp` goes.  The
chain, leaf first::

    codegen!(state)                       src/spc/codegen.jl:607-723
      <- codegen(scope)                   src/spc/codegen.jl:725-727
        <- generate_sp_code(...)          src/spc/interface.jl:9-33
             the @generated body for
          <- (::SpCircuit)(nets...)       src/spc/generated.jl:1-7

and an ``SpCircuit`` value is constructed in exactly two places:

* ``codegen.jl:455``, inside ``cg_instance!(::SNode{SP.SubcktCall})`` — which is
  itself reachable only from ``codegen!``, so that is the legacy path recursing
  into itself for a subcircuit;
* ``sema.jl:823``, inside ``sema_assign_ids``, whose only caller is its own
  recursion at line 815.

Nothing else calls ``sema_assign_ids``, so no ``CktID`` is ever assigned,
``SemaResult.CktID`` stays ``nothing``, no ``SpCircuit`` is ever constructed, the
generated function never fires, and ``codegen`` / ``codegen!`` never run.  Every
live entry — ``MNACircuit(path)``, ``MNACircuit(code; lang)``,
``Base.include(mod, SpiceFile(...))``, ``sp"..."``, ``spc"..."`` — goes through
``_eval_deck_into_module`` → ``make_mna_circuit`` → ``codegen_mna!``, which emits
``stamp!`` calls instead.

Two undefined names corroborate it from the other side.  Measured::

    julia> using Cadnip
    julia> [s => isdefined(Cadnip, s) for s in
            (:Named, :SimOptions, :options, :spicecall, :BinnedModel)]
    :Named       => false
    :SimOptions  => false
    :options     => false
    :spicecall   => true
    :BinnedModel => true

``cg_spice_instance!`` (``codegen.jl:426-431``) interpolates ``Named`` into every
device it emits, and ``codegen!``'s option block interpolates
``Cadnip.SimOptions`` / ``Cadnip.options``.  Both are interpolations of a *value*
at codegen time, so the first deck with a resistor — or with
`.option temp/gmin/scale` — would throw ``UndefVarError`` while its builder was
being generated.  Cadnip's own tests compile exactly such decks without error,
which is only possible because they never reach that code.

What that makes unreachable, with line numbers on ``main`` @ 739fdac:

====================== ========== ===============================================
file                   lines      what
====================== ========== ===============================================
src/spc/codegen.jl     411-431    ``cg_params!``, ``cg_spice_instance!``
src/spc/codegen.jl     433-523    the five ``cg_instance!`` methods
src/spc/codegen.jl     527-605    ``cg_model_def!`` (only caller is ``codegen!``
                                  at line 692)
src/spc/codegen.jl     607-727    ``codegen!``, ``codegen``
src/spc/generated.jl   1-7        the generated function and its ``reload()``
src/spc/interface.jl   1-33       ``SpCircuit``, ``getsema``,
                                  ``generate_sp_code``
src/spc/sema.jl        802-826    ``assign_id!``, ``sema_assign_ids``, and with
                                  them the ``CktID`` field and its readers
                                  (lines 685, 810)
src/spc/query.jl       most of it everything dispatching on ``SpCircuit``:
                                  ``show``, ``getproperty``, ``SpRef``,
                                  ``MultipleKinds``
====================== ========== ===============================================

That is roughly 410 of ``codegen.jl``'s 3751 lines, plus two small files.

Three things not to sweep up with it
------------------------------------

**``is_ambiguous``** lives in :file:`src/spc/query.jl` (line 109) but is
load-bearing for the *MNA* path: ``cg_net_name!`` and ``cg_model_name!`` call it
at ``codegen.jl:27`` and ``:37``.  It has to move before the file goes.

**``spicecall``** is shared — ``cg_mna_instance!`` uses it for diodes, MOSFETs
and BJTs.  Only ``Named`` is legacy-only, and its remaining uses are inside
``#= =#`` blocks in :file:`test/basic.jl`.

**``UnimplementedDevice``** reads as Cedar-era but is live: ``sema.jl:332``
returns a ``GlobalRef`` to it as the fallback when no model resolves.

Not traced, so not claimed either way: the rest of the Cedar-era surface —
``ParamSim`` and the netlist-text ``alter(io, ast, ::ParamSim)`` in
:file:`src/spectre.jl`, the ``CircuitElement`` / ``AbstractSim`` abstract types,
and ``SimSpec``.  ``SimSpec`` and the ``Cadnip.spec`` ScopedValue in particular
are *not* dead: ``temper()`` and ``var"$time"`` in :file:`src/spectre_env.jl`
read them, and MNA-generated expressions can reach those.  Nothing writes them,
which is finding 3.

Verifying
---------

The whole trace is four greps::

    grep -rn '\bcodegen!\?(' --include=*.jl .      # 2 hits: 725, interface.jl:33
    grep -rn 'sema_assign_ids' --include=*.jl .    # 2 hits: its definition and
                                                   #         its own recursion
    grep -rn 'SpCircuit{' --include=*.jl src/      # constructed at 455 and 823
    grep -rn 'is_ambiguous' --include=*.jl src/    # query.jl:109, codegen.jl:27,37

Gaps
====

Not defects, just unimplemented; the backend raises :exc:`NotImplementedError`
for each, and :file:`README.rst` records what Cadnip would need.

============================== ================================================
SPICE feature                  what is missing
============================== ================================================
`.dc` of a source or resistor  raw-device instance overrides (see 4 above)
`.sens`                        a sensitivity analysis over the ForwardDiff
                               derivatives Cadnip already computes
`.pz`                          a pole-zero entry point over the descriptor
                               system ``subsystem(ac, :name)`` returns
`.tf`                          DC gain with input/output resistance
`.disto`                       distortion analysis, ``distof1``/``distof2``
`.meas`                        any measurement API
`.ic` values                   a named-node initial state on ``tran!``
                               (``CedarUICOp`` covers `uic` itself)
`.nodeset`                     a named partial guess for ``dc!``; only a full
                               ``u0`` vector is accepted
`.ac oct` / `.ac lin`          octave and linear counterparts to ``acdec``
`.tran` output interval        no output-interval option; SciML's ``saveat``
                               reaches it, Cadnip does not name it
`.options` tolerances          spec tolerances are ``$simparam`` values, not
                               solver controls
SAVECURRENTS in transient      terminal currents exist only on the DC rebuild
AC device op variables         ``ac!`` exposes solution variables only
`.noise v(a,b)`                node-pair output; only a single name is indexed
`.noise` with a current input  input referral accepts voltage sources only
============================== ================================================

Two more ergonomic ones, which cost this backend real code:

**No accessor classifies a solution name.**  ``keys(sol)`` returns node
voltages, branch currents, terminal currents and device variables mixed
together, and only the ``I_`` naming convention separates the first two.  The
bridge reads the ``node_names`` / ``current_names`` fields of ``DCSolution``,
``ACSol`` and ``MNAData`` instead.  ``node_names(sol)`` / ``branch_names(sol)``
would make that a supported operation.

**Hierarchical nodes flatten with an underscore.**  ``x1_out`` where SPICE
writes ``x1.out``.  Reversing it is unsafe — an underscore is legal in a net
name — so a portable script cannot name a subcircuit node the same way across
simulators.

Reproducing
===========

The probes are small enough to inline.  With a Julia environment holding Cadnip:

.. code-block:: julia

    using Cadnip
    using Cadnip.MNA: MNACircuit, MNASpec, alter, with_temp
    using Cadnip.NyanSpectreNetlistParser

    divider = """
    .title divider
    V1 vcc 0 DC 5
    R1 vcc out 1k
    R2 out 0 1k
    """

    c = MNACircuit(divider; lang=:spice)
    dc!(c)[:out]                                  # 2.5 — the baseline
    dc!(alter(c; r1=(r=2e3,)))[:out]              # still 2.5 — finding 4

    # finding 1: the card rebinds the spec in the generated builder
    carded = replace(divider, ".title divider" => ".title divider\n.temp 100")
    sa = NyanSpectreNetlistParser.parse(IOBuffer(carded); start_lang=:spice,
                                        implicit_title=true)
    occursin("MNASpec)(temp = 100.0", string(Cadnip.make_mna_circuit(sa)))   # true

    # finding 1: noise follows circuit.spec, which the card never touches
    nd = """
    .title noise
    V1 vcc 0 DC 0
    R1 vcc out 1k
    C1 out 0 1u
    """
    noise!(MNACircuit(nd; lang=:spice), :out; freqs=[1.0])[:onoise][1]
    noise!(MNACircuit(replace(nd, ".title noise" => ".title noise\n.temp 100");
                      lang=:spice), :out; freqs=[1.0])[:onoise][1]   # same
    noise!(MNACircuit(nd; lang=:spice, spec=MNASpec(temp=100.0)),
           :out; freqs=[1.0])[:onoise][1]                            # x 1.2432

    # finding 2
    isdefined(Cadnip, :SimOptions), isdefined(Cadnip, :options)      # (false, false)
