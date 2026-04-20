using Cadnip
using Cadnip.SpectreEnvironment
using NyanSpectreNetlistParser
using Test
using Random

# Simulation packages
using OrdinaryDiffEq
using SciMLBase
using Sundials
using LinearSolve: KLUFactorization

using Cadnip.MNA: MNAContext, MNASpec, assemble!, solve_dc, solve_ac
using Cadnip.MNA: MNACircuit
using Cadnip.MNA: get_node!, stamp!
using Cadnip.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using Cadnip.MNA: make_ode_problem, ZERO_VECTOR

const deftol = 1e-8

# Our default tolerances are one order of magnitude above our default solve tolerances
isapprox_deftol(x, y) = isapprox(x, y; atol=deftol*10, rtol=deftol*10)
isapprox_deftol(x) = y->isapprox(x, y; atol=deftol*10, rtol=deftol*10)

allapprox_deftol(itr) = isempty(itr) ? true : all(isapprox_deftol(first(itr)), itr)

#==============================================================================#
# Compat shims for test files — thin wrappers around the new MNACircuit API.
#
# Tests that still return `(ctx, sol)` tuples are supported here; the idiomatic
# call is `dc!(MNACircuit(code; lang=...))`, which returns just the solution.
# These shims make old test code compile while Phase 4 migration is in progress.
#==============================================================================#

# These shims eval the builder inside a function body (we're called from an
# `@testset`). `MNACircuit(code; lang=...)` from production would error with
# "method too new" here (by design), so we wrap the builder in an explicit
# invokelatest closure ourselves — same pattern tests have always used for
# runtime-parsed circuits.
function _eval_spice_builder(spice_code, imported_hdl_modules)
    code = parse_spice_to_mna(spice_code; circuit_name=:circuit, imported_hdl_modules)
    m = Module()
    Base.eval(m, code)
    builder = getfield(m, :circuit)
    return (args...; kwargs...) -> Base.invokelatest(builder, args...; kwargs...)
end

function _eval_spectre_builder(spectre_code, imported_hdl_modules)
    ast = Cadnip.NyanSpectreNetlistParser.parse(IOBuffer(spectre_code); start_lang=:spectre)
    sema_result = Cadnip.sema(ast; imported_hdl_modules)
    code = Cadnip._make_mna_circuit_with_sema(sema_result; circuit_name=:circuit)
    m = Module()
    Base.eval(m, code)
    builder = getfield(m, :circuit)
    return (args...; kwargs...) -> Base.invokelatest(builder, args...; kwargs...)
end

function solve_mna_spice_code(spice_code::AbstractString; temp::Real=27.0, maxiters::Int=100,
                              imported_hdl_modules::Vector{Module}=Module[])
    wrapped = _eval_spice_builder(spice_code, imported_hdl_modules)
    circuit = MNACircuit(wrapped, (;), MNASpec(temp=Float64(temp), mode=:dcop))
    sol = solve_dc(circuit)
    ctx = circuit.builder(circuit.params, circuit.spec, 0.0; x=sol.x)
    return ctx, sol
end

function solve_mna_spectre_code(spectre_code::AbstractString; temp::Real=27.0, maxiters::Int=100,
                                imported_hdl_modules::Vector{Module}=Module[])
    wrapped = _eval_spectre_builder(spectre_code, imported_hdl_modules)
    circuit = MNACircuit(wrapped, (;), MNASpec(temp=Float64(temp), mode=:dcop))
    sol = solve_dc(circuit)
    ctx = circuit.builder(circuit.params, circuit.spec, 0.0; x=sol.x)
    return ctx, sol
end

# `solve_mna_circuit`, `tran_mna_circuit`, and `make_mna_spice_circuit` were
# compat shims for old test code; no current test calls them. Deleted.

#==============================================================================#
# Test-only shim: `parse_spice_to_mna` returns a builder Expr and accepts an
# internal `imported_hdl_modules` list for test code that defines VA devices via
# the `va"""..."""` macro. Not a public API.
#==============================================================================#
function parse_spice_to_mna(spice_code::AbstractString;
                            circuit_name::Symbol=:circuit,
                            imported_hdl_modules::Vector{Module}=Module[])
    ast = Cadnip.NyanSpectreNetlistParser.parse(IOBuffer(spice_code);
        start_lang=:spice, implicit_title=true)
    sema_result = Cadnip.sema(ast; imported_hdl_modules)
    return _make_mna_circuit_from_sema(sema_result, circuit_name)
end

function parse_spice_file_to_mna(filepath::AbstractString;
                                  circuit_name::Symbol=:circuit,
                                  imported_hdl_modules::Vector{Module}=Module[])
    ast = Cadnip.NyanSpectreNetlistParser.parsefile(filepath;
        start_lang=:spice, implicit_title=true)
    sema_result = Cadnip.sema(ast; imported_hdl_modules)
    return _make_mna_circuit_from_sema(sema_result, circuit_name)
end

# Reuse codegen with a pre-built sema result. Duplicates a bit of codegen.jl
# logic but avoids surfacing imported_hdl_modules on a public API.
function _make_mna_circuit_from_sema(sema_result, circuit_name::Symbol)
    # Reach into Cadnip's codegen — this is an internal/test-only bridge.
    # We assemble the same expression make_mna_circuit would, but with a sema
    # that already has imported_hdl_modules populated.
    Cadnip._make_mna_circuit_with_sema(sema_result; circuit_name)
end
