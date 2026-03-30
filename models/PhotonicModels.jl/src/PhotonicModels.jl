"""
    PhotonicModels

Pre-parsed photonic Verilog-A device models for optical circuit simulation.

Models use the custom `optical` discipline with `OptE()` access function.

# Tier 1 (leaf modules)
Coordinate converters, Cartesian arithmetic, passive optical components, couplers/splitters.

# Tier 2 (composite modules - use module instantiation)
Attenuator, Isolator, PhaseShifter, Waveguide, Pcw, PhaseModulator, PcwPhaseModulator.

# Tier 3 (not yet loadable)
CwLaser, NoisyEDFA (need @(initial_step)), PhotoDetector, TunableFilter (need laplace_nd).
"""
module PhotonicModels

using CedarSim
using CedarSim: VAFile
using CedarSim.MNA: MNAContext, MNASpec, stamp!, get_node!
using VerilogAParser

const VA_DIR = joinpath(@__DIR__, "..", "va")

# Tier 1: Leaf modules (no module instantiation)
# parsefile resolves `include directives via the va/ search path.
const TIER1_MODELS = [
    # Coordinate converters
    "Polar2Cartesian",
    "Cartesian2Polar",
    "PolToCart",
    "CartToPol",
    # Cartesian arithmetic
    "CartesianMultiplier",
    "CartesianAdder",
    "CartAdd",
    "CartMul",
    "CartSub",
    # Passive optical
    "Terminator",
    "ReflectionInterface",
    # Couplers and splitters
    "DirectionalCoupler",
    "OneTwoCoupler",
    "OneTwoLoopback",
    "OneTwoSplitter",
    "TwoOneCombiner",
    # Electrical
    "NonlinearCapacitor",
]

for name in TIER1_MODELS
    filepath = joinpath(VA_DIR, name * ".va")
    va = VerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
end

# Tier 2: Composite modules (use module instantiation)
# These reference child modules in separate files. The `deps` kwarg generates
# `using` statements to import child types from their already-loaded baremodules.
# Load order matters: Tier 1 must be loaded first.
const TIER2_MODELS = [
    ("Attenuator",        [:Polar2Cartesian, :CartesianMultiplier]),
    ("Isolator",          [:Polar2Cartesian, :CartesianMultiplier]),
    ("PhaseShifter",      [:Polar2Cartesian, :CartesianMultiplier]),
    ("Waveguide",         [:Polar2Cartesian, :CartesianMultiplier]),
    ("Pcw",               [:Polar2Cartesian, :CartesianMultiplier]),
    ("PhaseModulator",    [:Polar2Cartesian, :CartesianMultiplier]),
    ("PcwPhaseModulator", [:Polar2Cartesian, :CartesianMultiplier]),
]

for (name, deps) in TIER2_MODELS
    filepath = joinpath(VA_DIR, name * ".va")
    va = VerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, CedarSim.make_mna_module(va; deps))
end

# Tier 3: Models requiring laplace_nd (now supported)
const TIER3_LEAF_MODELS = ["PhotoDetector"]
for name in TIER3_LEAF_MODELS
    filepath = joinpath(VA_DIR, name * ".va")
    va = VerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
end

const TIER3_COMPOSITE_MODELS = [
    ("TunableFilter", [:CartesianMultiplier]),
]
for (name, deps) in TIER3_COMPOSITE_MODELS
    filepath = joinpath(VA_DIR, name * ".va")
    va = VerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, CedarSim.make_mna_module(va; deps))
end

# Not yet supported:
# CwLaser:   @(initial_step) event control
# NoisyEDFA: @(initial_step) + $rdist_normal()

# Tier 1 exports
export Polar2Cartesian, Cartesian2Polar, PolToCart, CartToPol
export CartesianMultiplier, CartesianAdder, CartAdd, CartMul, CartSub
export Terminator, ReflectionInterface
export DirectionalCoupler, OneTwoCoupler, OneTwoLoopback, OneTwoSplitter, TwoOneCombiner
export NonlinearCapacitor
# Tier 2 exports
export Attenuator, Isolator, PhaseShifter
export Waveguide, Pcw, PhaseModulator, PcwPhaseModulator
# Tier 3 exports
export PhotoDetector, TunableFilter

end # module
