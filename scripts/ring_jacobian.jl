#!/usr/bin/env julia
#==============================================================================#
# Ring Oscillator Jacobian Analysis
#
# Uses sp_mos1 model loaded directly via make_mna_module
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNACircuit, MNASpec, build_with_detection, assemble!, system_size
using CedarSim.MNA: compile_structure, create_workspace, fast_rebuild!
using CedarSim.MNA: MNAContext, get_node!, stamp!, ZERO_VECTOR, reset_for_restamping!
using VerilogAParser
using LinearAlgebra
using SparseArrays
using Printf

println("="^70)
println("Ring Oscillator Jacobian Analysis")
println("="^70)
println()

# Load sp_mos1 model from VA file
println("Loading sp_mos1 VA model...")
const mos1_va_file = joinpath(@__DIR__, "..", "models", "VADistillerModels.jl", "va", "mos1.va")
va = VerilogAParser.parsefile(mos1_va_file)
# Create and eval the module - this defines sp_mos1_module in Main
eval(CedarSim.make_mna_module(va))
println("sp_mos1_module created: $(sp_mos1_module)")

# Now parse the ring oscillator circuit
println("\nParsing ring oscillator SPICE netlist...")
const ring_code = parse_spice_to_mna("""
* 3-stage CMOS Ring Oscillator
Vdd vdd 0 DC 3.3
XMP1 out1 in1 vdd vdd sp_mos1 type=-1 vto=-0.7 kp=50e-6 w=2e-6 l=1e-6
XMN1 out1 in1 0 0 sp_mos1 type=1 vto=0.7 kp=100e-6 w=1e-6 l=1e-6
XMP2 out2 out1 vdd vdd sp_mos1 type=-1 vto=-0.7 kp=50e-6 w=2e-6 l=1e-6
XMN2 out2 out1 0 0 sp_mos1 type=1 vto=0.7 kp=100e-6 w=1e-6 l=1e-6
XMP3 in1 out2 vdd vdd sp_mos1 type=-1 vto=-0.7 kp=50e-6 w=2e-6 l=1e-6
XMN3 in1 out2 0 0 sp_mos1 type=1 vto=0.7 kp=100e-6 w=1e-6 l=1e-6
C1 out1 0 10f
C2 out2 0 10f
C3 in1 0 10f
.END
"""; circuit_name=:ring_oscillator, imported_hdl_modules=[sp_mos1_module])
eval(ring_code)

# Create circuit
spec = MNASpec(mode=:dcop)
circuit = MNACircuit(ring_oscillator, (;), spec)

println("Building circuit...")
ctx = build_with_detection(circuit)
sys = assemble!(ctx)

n = system_size(sys)
println("\n=== System Size ===")
println("  Total unknowns: $n")
println("  Node voltages:  $(sys.n_nodes)")
println("  Branch currents: $(length(sys.current_names))")
println("  Node names: $(sys.node_names)")

# Get matrices
G = sys.G
C = sys.C
b = sys.b

println("\n=== Matrix Statistics ===")
println("  G matrix: $(nnz(G)) nonzeros")
println("  C matrix: $(nnz(C)) nonzeros")

# Compute condition number
println("\n=== Condition Number Analysis ===")
G_dense = Matrix(G)
C_dense = Matrix(C)

try
    cond_G = cond(G_dense)
    @printf("  Condition number of G: %.2e\n", cond_G)
catch e
    println("  G is singular: $e")
end

# Row norms
println("\n=== Row Scaling Analysis ===")
row_norms_G = [norm(G_dense[i, :]) for i in 1:n]
row_norms_C = [norm(C_dense[i, :]) for i in 1:n]

println("  Row norms (G | C | G/C ratio):")
for i in 1:n
    name = i <= sys.n_nodes ? string(sys.node_names[i]) : sys.current_names[i - sys.n_nodes]
    g_norm = row_norms_G[i]
    c_norm = row_norms_C[i]
    ratio = c_norm > 1e-30 ? g_norm / c_norm : Inf
    @printf("    Row %3d (%-20s): G=%.2e, C=%.2e, ratio=%.2e\n", i, name, g_norm, c_norm, ratio)
end

# SVD analysis
println("\n=== SVD Analysis ===")
try
    svd_G = svd(G_dense)
    @printf("  Max singular value: %.2e\n", svd_G.S[1])
    @printf("  Min singular value: %.2e\n", svd_G.S[end])
    @printf("  Condition number:   %.2e\n", svd_G.S[1] / max(svd_G.S[end], 1e-30))
catch e
    println("  SVD failed: $e")
end

println("\n" * "="^70)
println("Analysis complete")
println("="^70)
