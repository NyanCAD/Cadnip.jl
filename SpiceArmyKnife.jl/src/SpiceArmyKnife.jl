module SpiceArmyKnife

using UUIDs
using Downloads
using p7zip_jll
using NyanSpectreNetlistParser
using NyanSpectreNetlistParser: SpectreNetlistCSTParser, SPICENetlistParser
using .SPICENetlistParser: SPICENetlistCSTParser
using .SpectreNetlistCSTParser: SpectreNetlistSource
using .SPICENetlistCSTParser: SPICENetlistSource
using NyanSpectreNetlistParser.RedTree: fullcontents
using NyanVerilogAParser
using StringEncodings

const SNode = SpectreNetlistCSTParser.Node
const SC = SpectreNetlistCSTParser
const SP = SPICENetlistCSTParser

LSymbol(s) = Symbol(lowercase(String(s)))

# Include Verilog-A model extraction
include("va_models.jl")

# Include code generation
include("codegen.jl")

# Include parsing code
include("parse.jl")

# Code generation exports
export CodeGenScope, generate_code

# Verilog-A model extraction exports
export ModelParameter, ModelDefinition, ModelDatabase
export extract_model_definitions, merge_model_databases
export get_model

# Include app submodules
include("Generate.jl")
include("Convert.jl")

end # module SpiceArmyKnife
