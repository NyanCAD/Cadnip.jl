#==============================================================================#
# OsdiLoader: Load OpenVAF-compiled .osdi shared libraries into Cadnip MNA
#
# Usage:
#   using CedarSim
#   f = OsdiLoader.osdi_load("path/to/device.osdi")
#   dev = f.devices[1]
#   model = OsdiModel(dev)
#   setup_model!(model)
#   inst = OsdiInstance(model)
#   setup_instance!(inst)
#   # Then use inst in a circuit builder with stamp!(inst, ctx, p, n)
#==============================================================================#

module OsdiLoader

using ..MNA

include("types.jl")
include("loader.jl")
include("model.jl")
include("stamp.jl")

export osdi_load, OsdiFile, OsdiDeviceType, OsdiParamInfo, OsdiNodeInfo
export OsdiModel, OsdiInstance, OsdiLimitState
export setup_model!, setup_instance!, set_param!
export bind_jacobian_pointers!, apply_node_collapse!
export reset_from_converged!, promote_converged!

end # module OsdiLoader
