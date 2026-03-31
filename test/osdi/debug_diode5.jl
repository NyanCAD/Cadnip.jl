#!/usr/bin/env julia
#==============================================================================#
# Debug: Compare OSDI vs VA transient waveforms on short sim
# Then verify residual consistency at a point where OSDI struggles
#==============================================================================#

using Cadnip
using Cadnip.MNA
using Cadnip.MNA: compile_structure, create_workspace, build_with_detection, EvalWorkspace, fast_rebuild!
using Cadnip.OsdiLoader
using VADistillerModels
using LinearAlgebra, SparseArrays
using OrdinaryDiffEq: Rodas5P
using LinearSolve: KLUFactorization

const DIODE_OSDI = joinpath(@__DIR__, "diode.osdi")

# Simple single-diode test
const spice_va = """* VA diode
vs a 0 sin 0.0 5 1000
R1 a anode 1k
xd1 anode 0 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
.end
"""
const spice_osdi = """* OSDI diode
.model d1 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
vs a 0 sin 0.0 5 1000
R1 a anode 1k
N1 anode 0 d1
.end
"""

println("=== Building circuits ===")
va_code = Cadnip.parse_spice_to_mna(spice_va; circuit_name=:va_ckt,
                                       imported_hdl_modules=[sp_diode_module])
eval(va_code)
osdi_code = Cadnip.parse_spice_to_mna(spice_osdi; circuit_name=:osdi_ckt,
                                          osdi_files=[DIODE_OSDI])
eval(osdi_code)

solver = Rodas5P(linsolve=KLUFactorization())

# Short transient
tspan = (0.0, 0.001)

println("\n=== VA transient ===")
va_circuit = MNACircuit(va_ckt)
MNA.assemble!(va_circuit)
va_sol = tran!(va_circuit, tspan; dtmax=1e-5, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
println("Steps: $(length(va_sol.t)), retcode=$(va_sol.retcode)")

println("\n=== OSDI transient ===")
osdi_circuit = MNACircuitFromSetup((p) -> Base.invokelatest(osdi_ckt, p), (;), MNASpec())
MNA.assemble!(osdi_circuit)
osdi_sol = tran!(osdi_circuit, tspan; dtmax=1e-5, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=100000, dense=false)
println("Steps: $(length(osdi_sol.t)), retcode=$(osdi_sol.retcode)")

# Compare at specific timepoints
println("\n=== Waveform comparison ===")
# Find VA node indices
va_anode_idx = findfirst(==(Symbol("anode")), va_sol.prob.p.structure.node_names)
va_a_idx = findfirst(==(Symbol("a")), va_sol.prob.p.structure.node_names)
# Find OSDI node indices
osdi_anode_idx = findfirst(==(Symbol("anode")), osdi_sol.prob.p.structure.node_names)
osdi_a_idx = findfirst(==(Symbol("a")), osdi_sol.prob.p.structure.node_names)

for t_check in [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]
    # Find closest VA timepoint
    va_idx = argmin(abs.(va_sol.t .- t_check))
    osdi_idx = argmin(abs.(osdi_sol.t .- t_check))

    va_anode = va_sol.u[va_idx][va_anode_idx]
    osdi_anode = osdi_sol.u[osdi_idx][osdi_anode_idx]
    va_a = va_sol.u[va_idx][va_a_idx]
    osdi_a = osdi_sol.u[osdi_idx][osdi_a_idx]

    println("t=$t_check:")
    println("  V(a):     VA=$(round(va_a, digits=6)), OSDI=$(round(osdi_a, digits=6))")
    println("  V(anode): VA=$(round(va_anode, digits=6)), OSDI=$(round(osdi_anode, digits=6))")
    println("  diff(a):  $(abs(va_a - osdi_a))")
    println("  diff(anode): $(abs(va_anode - osdi_anode))")
end

# Now test with dtmax=1e-6 (the failing case) but for short time
println("\n=== Short sim with dtmax=1e-6 ===")
osdi_circuit2 = MNACircuitFromSetup((p) -> Base.invokelatest(osdi_ckt, p), (;), MNASpec())
MNA.assemble!(osdi_circuit2)
osdi_sol2 = tran!(osdi_circuit2, (0.0, 0.001); dtmax=1e-6, solver=solver, abstol=1e-3, reltol=1e-3, maxiters=1000000, dense=false)
println("Steps: $(length(osdi_sol2.t)), retcode=$(osdi_sol2.retcode), t_end=$(osdi_sol2.t[end])")

# Check step sizes around the failure point
if length(osdi_sol2.t) > 10
    dts = diff(osdi_sol2.t)
    println("First 10 dt: ", [round(dt, sigdigits=3) for dt in dts[1:min(10,end)]])
    min_dt_idx = argmin(dts)
    println("Smallest dt: $(dts[min_dt_idx]) at t=$(osdi_sol2.t[min_dt_idx])")
    println("Last 10 dt: ", [round(dt, sigdigits=3) for dt in dts[max(1,end-9):end]])
end
