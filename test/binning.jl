#==============================================================================#
# Model binning: `.model nch.1` / `.model nch.2` … referenced as `nch`
#
# A foundry PDK does not ship one model card per device flavour; it ships a
# family of cards, each fitted over a window of channel geometry, and every
# instance line names the bare family. `lmin`/`lmax`/`wmin`/`wmax` on a card say
# which window it covers — they are not device parameters, and most device
# models have no field for them.
#
# The assertions here are mostly of one shape: a binned deck must solve to
# exactly what the deck that spells out the winning card solves to. That pins
# bin selection and the parameter handoff at once, without hard-coding numbers
# the level-1 model is free to change.
#
# See `doc/model_binning.md`.
#==============================================================================#

module binning_tests

using Test
using Cadnip
using VADistillerModels                   # .model … nmos level=1 → MOS1
using Cadnip: dc!
using Cadnip.MNA: MNACircuit

# Two length bins of one family. Only `vto` differs, so which bin was picked is
# visible in the drain node of a resistively loaded stage.
const length_binned = sp"""
* length-binned nmos family
.model nch.1 nmos level=1 lmin=0   lmax=1u vto=0.4 kp=100u
.model nch.2 nmos level=1 lmin=1u  lmax=1  vto=1.0 kp=100u
Vdd vdd 0 DC 2
Vg  g   0 DC 0.8
Rshort vdd short 10k
Mshort short g 0 0 nch w=10u l=0.5u
Rlong  vdd long  10k
Mlong  long  g 0 0 nch w=10u l=2u
"""i

# The same stage with the winning card written out by hand, once per bin.
const unbinned_lo = sp"""
.model nch nmos level=1 vto=0.4 kp=100u
Vdd vdd 0 DC 2
Vg  g   0 DC 0.8
Rshort vdd short 10k
Mshort short g 0 0 nch w=10u l=0.5u
"""i

const unbinned_hi = sp"""
.model nch nmos level=1 vto=1.0 kp=100u
Vdd vdd 0 DC 2
Vg  g   0 DC 0.8
Rlong vdd long 10k
Mlong long  g 0 0 nch w=10u l=2u
"""i

@testset "length binning picks the card the geometry falls in" begin
    sol = dc!(MNACircuit(length_binned))
    lo  = dc!(MNACircuit(unbinned_lo))
    hi  = dc!(MNACircuit(unbinned_hi))

    @test sol[:short] ≈ lo[:short] rtol=1e-9
    @test sol[:long]  ≈ hi[:long]  rtol=1e-9

    # And the two bins really are different devices at this bias: vto=0.4 with
    # vgs=0.8 conducts, vto=1.0 does not.
    @test sol[:short] < 1.0
    @test sol[:long]  > 1.99
end

# Binning on width, with the length axis left unbounded — a family need not
# spell out every bound.
const width_binned = sp"""
* width-binned nmos family
.model nch.1 nmos level=1 wmin=0  wmax=5u vto=0.4 kp=100u
.model nch.2 nmos level=1 wmin=5u wmax=1  vto=1.0 kp=100u
Vdd vdd 0 DC 2
Vg  g   0 DC 0.8
Rnarrow vdd narrow 10k
Mnarrow narrow g 0 0 nch w=1u l=1u
Rwide   vdd wide   10k
Mwide   wide   g 0 0 nch w=10u l=1u
"""i

@testset "width binning, unbounded length axis" begin
    sol = dc!(MNACircuit(width_binned))
    @test sol[:narrow] < 1.99      # vto = 0.4 → conducting
    @test sol[:wide]   > 1.99      # vto = 1.0 → off
end

# A window bound is not a device parameter. MOS1 has no LMIN/LMAX/WMIN/WMAX
# field at all, so a family whose cards carry all four only builds if they are
# stripped before the model card is constructed.
@testset "window bounds do not reach the device model" begin
    deck = sp"""
    .model nch.1 nmos level=1 lmin=0 lmax=1 wmin=0 wmax=1 vto=0.4 kp=100u
    Vdd vdd 0 DC 2
    Vg  g   0 DC 0.8
    R1 vdd d 10k
    M1 d g 0 0 nch w=10u l=1u
    """i
    sol = dc!(MNACircuit(deck))
    @test sol[:d] < 1.99
end

# A geometry no card covers is an error, not a silent fallback to bin 1.
@testset "geometry outside every bin" begin
    deck = sp"""
    .model nch.1 nmos level=1 lmin=0  lmax=1u vto=0.4 kp=100u
    .model nch.2 nmos level=1 lmin=1u lmax=2u vto=1.0 kp=100u
    Vdd vdd 0 DC 2
    Vg  g   0 DC 0.8
    R1 vdd d 10k
    M1 d g 0 0 nch w=10u l=5u
    """i
    err = try
        dc!(MNACircuit(deck))
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test occursin("NoBinError", msg)
    @test occursin("nch", msg)
end

# Nothing to bin on: SPICE has no default geometry here (`.option defl`/`defw`
# are not implemented), so guessing a bin would be guessing a device.
@testset "instance with no geometry" begin
    deck = sp"""
    .model nch.1 nmos level=1 lmin=0 lmax=1 vto=0.4 kp=100u
    Vdd vdd 0 DC 2
    Vg  g   0 DC 0.8
    R1 vdd d 10k
    M1 d g 0 0 nch
    """i
    err = try
        dc!(MNACircuit(deck))
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("NoGeometryError", sprint(showerror, err))
end

# The family is defined at the top level and referenced from inside a subcircuit,
# which is where a PDK's cards and a design's devices actually sit.
const binned_in_subckt = sp"""
* binned family used from a subcircuit
.model nch.1 nmos level=1 lmin=0  lmax=1u vto=0.4 kp=100u
.model nch.2 nmos level=1 lmin=1u lmax=1  vto=1.0 kp=100u
.subckt stage d g len=1u
M1 d g 0 0 nch w=10u l=len
.ends
Vdd vdd 0 DC 2
Vg  g   0 DC 0.8
Rshort vdd short 10k
Xshort short g stage len=0.5u
Rlong  vdd long  10k
Xlong  long  g stage len=2u
"""i

@testset "binned family referenced from a subcircuit" begin
    sol = dc!(MNACircuit(binned_in_subckt))
    @test sol[:short] < 1.0
    @test sol[:long]  > 1.99
end

# A bin card is an ordinary card otherwise: it can read a `.param`, which means
# the family cannot be a module-level constant and is built in the builder body.
const binned_param = sp"""
.param vt_lo=0.4
.model nch.1 nmos level=1 lmin=0  lmax=1u vto=vt_lo kp=100u
.model nch.2 nmos level=1 lmin=1u lmax=1  vto=1.0   kp=100u
Vdd vdd 0 DC 2
Vg  g   0 DC 0.8
R1 vdd d 10k
M1 d g 0 0 nch w=10u l=0.5u
"""i

@testset "a bin card reading a .param, overridable" begin
    conducting = dc!(MNACircuit(binned_param))
    @test conducting[:d] < 1.0

    # Raise the threshold of the bin this geometry lands in and it turns off.
    off = dc!(MNACircuit(binned_param; vt_lo = 1.0))
    @test off[:d] > 1.99
end

# The `.model nch` a deck writes out is what `nch` means, bins or no bins. This
# one is asserted on the generated code rather than on a solution: `sp"..."`
# compiles its deck when the macro expands, which is before any `@test_logs`
# here could be watching.
@testset "a direct card beats a numbered one of the same base" begin
    code = """
    * a deck contradicting itself
    .model nch   nmos level=1 vto=1.0 kp=100u
    .model nch.1 nmos level=1 lmin=0 lmax=1 vto=0.4 kp=100u
    Vdd vdd 0 DC 2
    Vg  g   0 DC 0.8
    R1 vdd d 10k
    M1 d g 0 0 nch w=10u l=0.5u
    """
    ast = Cadnip.NyanSpectreNetlistParser.parse(IOBuffer(code); start_lang=:spice, implicit_title=true)
    generated = @test_logs (:warn, r"defined directly as well as in bins") match_mode=:any Cadnip.make_mna_circuit(ast)
    @test !occursin("BinnedModel", string(generated))
end

end # module
