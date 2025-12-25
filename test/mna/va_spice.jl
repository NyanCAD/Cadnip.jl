#==============================================================================#
# MNA Phase 6: Verilog-A to SPICE/Spectre Integration Tests
#
# Tests the integration of VA models with SPICE/Spectre netlists via
# the imported_hdl_modules mechanism in sema().
#
# This complements:
# - mna/core.jl: MNA matrix assembly and solve
# - mna/va.jl: VA→MNA stamping (va_str macro, stamp! methods)
# - mna/vadistiller.jl: Direct VA model tests (resistor, capacitor, MOSFET, etc.)
# - basic.jl: SPICE/Spectre parsing tests (solve_mna_spice_code, solve_mna_spectre_code)
#
# This file focuses specifically on:
# 1. Using VA modules from SPICE/Spectre netlists via X device syntax
# 2. The imported_hdl_modules mechanism for device resolution
# 3. Mixed circuits with VA and native SPICE/Spectre elements
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current
using CedarSim.MNA: VoltageSource, Resistor
using SpectreNetlistParser

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

#==============================================================================#
# Helper: Build MNA circuit from SPICE/Spectre with VA modules
#
# This mirrors how spice_select_device works for built-in devices (BSIM4, etc.)
# but uses imported_hdl_modules for user-provided VA modules.
#==============================================================================#

"""
    solve_with_va_modules(code, hdl_modules; lang=:spice, temp=27.0)

Parse SPICE or Spectre code with imported VA modules and solve DC.

This is the integration point that connects VA models to SPICE/Spectre netlists.
VA modules are passed via `imported_hdl_modules` to `sema()`, which then
recognizes them when device types are referenced in the netlist.

# Arguments
- `code`: SPICE or Spectre netlist code
- `hdl_modules`: Vector of baremodules containing VA device types
- `lang`: `:spice` or `:spectre`
- `temp`: Temperature for simulation
"""
function solve_with_va_modules(code::String, hdl_modules::Vector{Module};
                                lang::Symbol=:spice, temp::Real=27.0)
    # Parse the netlist
    if lang == :spice
        ast = SpectreNetlistParser.parse(IOBuffer(code); start_lang=:spice, implicit_title=true)
    else
        ast = SpectreNetlistParser.parse(IOBuffer(code); start_lang=:spectre)
    end

    # Run sema with imported_hdl_modules - this is the key integration point
    sema_result = CedarSim.sema(ast; imported_hdl_modules=hdl_modules)

    # Generate MNA code
    state = CedarSim.CodegenState(sema_result)

    # Generate subcircuit builders
    subckt_defs = Expr[]
    for (name, subckt_list) in sema_result.subckts
        if !isempty(subckt_list)
            _, subckt_sema = first(subckt_list)
            push!(subckt_defs, CedarSim.codegen_mna_subcircuit(subckt_sema.val, name))
        end
    end

    body = CedarSim.codegen_mna!(state)

    code_expr = quote
        $(subckt_defs...)
        function circuit(params, spec::$(MNASpec)=$(MNASpec)(); x::AbstractVector=Float64[])
            ctx = $(MNAContext)()
            $body
            return ctx
        end
    end

    # Evaluate in a module with VA types imported
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    for hdl_mod in hdl_modules
        for name in names(hdl_mod; all=true, imported=false)
            if !startswith(String(name), "#") && isdefined(hdl_mod, name)
                val = getfield(hdl_mod, name)
                isa(val, Type) && Base.eval(m, :(const $name = $val))
            end
        end
    end
    circuit_fn = Base.eval(m, code_expr)

    spec = MNASpec(temp=Float64(temp), mode=:dcop)
    ctx = Base.invokelatest(circuit_fn, (;), spec)
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    return sys, sol
end

#==============================================================================#
# Tests: VA Modules in SPICE Netlists
#==============================================================================#

@testset "VA-SPICE Integration" begin

    @testset "VA resistor in SPICE netlist" begin
        # Define VA resistor - lowercase for SPICE case-insensitivity
        va"""
        module varesistor(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        # SPICE netlist using VA module as X device
        spice = """
        * VA resistor in voltage divider
        V1 vcc 0 DC 10
        X1 vcc mid varesistor r=2k
        R1 mid 0 2k
        """

        sys, sol = solve_with_va_modules(spice, [varesistor_module]; lang=:spice)

        # Voltage divider: 10V * (2k / (2k + 2k)) = 5V
        @test isapprox_deftol(voltage(sol, :vcc), 10.0)
        @test isapprox_deftol(voltage(sol, :mid), 5.0)
    end

    @testset "VA capacitor in SPICE RC network" begin
        va"""
        module vacapacitor(p, n);
            parameter real c = 1e-12;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ c*ddt(V(p,n));
        endmodule
        """

        spice = """
        * VA capacitor in RC
        V1 vcc 0 DC 5
        R1 vcc cap 1k
        X1 cap 0 vacapacitor c=1n
        """

        sys, sol = solve_with_va_modules(spice, [vacapacitor_module]; lang=:spice)

        # DC: capacitor is open, V(cap) = V(vcc) = 5V
        @test isapprox_deftol(voltage(sol, :vcc), 5.0)
        @test isapprox_deftol(voltage(sol, :cap), 5.0)

        # Verify capacitance is stamped
        cap_idx = findfirst(n -> n == :cap, sys.node_names)
        @test isapprox(sys.C[cap_idx, cap_idx], 1e-9; atol=1e-12)
    end

    @testset "Multiple VA modules in SPICE circuit" begin
        va"""
        module vares(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        va"""
        module vacap(p, n);
            parameter real c = 1e-12;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ c*ddt(V(p,n));
        endmodule
        """

        spice = """
        * Multiple VA devices
        V1 vcc 0 DC 10
        X1 vcc n1 vares r=1k
        X2 n1 n2 vares r=2k
        X3 n2 0 vares r=1k
        X4 n1 0 vacap c=100p
        """

        sys, sol = solve_with_va_modules(spice, [vares_module, vacap_module]; lang=:spice)

        # Total R = 4k, I = 10/4k = 2.5mA
        # V(n1) = 10 - 2.5mA * 1k = 7.5V
        # V(n2) = 7.5 - 2.5mA * 2k = 2.5V
        @test isapprox(voltage(sol, :n1), 7.5; atol=0.01)
        @test isapprox(voltage(sol, :n2), 2.5; atol=0.01)
    end

end

#==============================================================================#
# Tests: VA Modules in Spectre Netlists
#==============================================================================#

@testset "VA-Spectre Integration" begin

    @testset "VA resistor in Spectre netlist" begin
        va"""
        module spectreres(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        spectre = """
        // VA resistor in Spectre
        v1 (vcc 0) vsource dc=10
        x1 (vcc mid) spectreres r=2k
        r1 (mid 0) resistor r=2k
        """

        sys, sol = solve_with_va_modules(spectre, [spectreres_module]; lang=:spectre)

        @test isapprox_deftol(voltage(sol, :vcc), 10.0)
        @test isapprox_deftol(voltage(sol, :mid), 5.0)
    end

    @testset "VA parallel RC in Spectre" begin
        va"""
        module spectrerc(p, n);
            parameter real r = 1000.0;
            parameter real c = 1e-12;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r + c*ddt(V(p,n));
        endmodule
        """

        spectre = """
        // VA parallel RC
        v1 (vcc 0) vsource dc=5
        x1 (vcc 0) spectrerc r=500 c=2n
        """

        sys, sol = solve_with_va_modules(spectre, [spectrerc_module]; lang=:spectre)

        @test isapprox_deftol(voltage(sol, :vcc), 5.0)
        # I = V/R = 5/500 = 10mA
        @test isapprox(current(sol, :I_v1), -0.01; atol=1e-5)
    end

    @testset "Mixed VA and native Spectre devices" begin
        va"""
        module spectreva(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        spectre = """
        // Mixed native and VA
        v1 (vcc 0) vsource dc=12
        r1 (vcc n1) resistor r=1k
        x1 (n1 n2) spectreva r=2k
        r2 (n2 0) resistor r=1k
        """

        sys, sol = solve_with_va_modules(spectre, [spectreva_module]; lang=:spectre)

        # Total R = 4k, I = 12/4k = 3mA
        # V(n1) = 12 - 3mA * 1k = 9V
        # V(n2) = 9 - 3mA * 2k = 3V
        @test isapprox(voltage(sol, :n1), 9.0; atol=0.01)
        @test isapprox(voltage(sol, :n2), 3.0; atol=0.01)
    end

end

#==============================================================================#
# Tests: Direct VA stamping (va_str without netlist parsing)
#
# These test the foundation that the netlist integration builds on.
# More comprehensive VA device tests are in mna/vadistiller.jl
#==============================================================================#

@testset "Direct VA Stamping" begin

    @testset "VA module with MNA primitives" begin
        va"""
        module directres(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        mid = get_node!(ctx, :mid)

        stamp!(VoltageSource(10.0; name=:V1), ctx, vcc, 0)
        stamp!(Resistor(1000.0; name=:R1), ctx, vcc, mid)
        stamp!(directres(r=1000.0), ctx, mid, 0)

        sys = assemble!(ctx)
        sol = solve_dc(sys)

        @test isapprox_deftol(voltage(sol, :vcc), 10.0)
        @test isapprox_deftol(voltage(sol, :mid), 5.0)
    end

    @testset "VA module chain" begin
        va"""
        module chainres(p, n);
            parameter real r = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/r;
        endmodule
        """

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)

        stamp!(VoltageSource(10.0; name=:V1), ctx, vcc, 0)
        stamp!(chainres(r=1000.0), ctx, vcc, n1)
        stamp!(chainres(r=2000.0), ctx, n1, n2)
        stamp!(chainres(r=1000.0), ctx, n2, 0)

        sys = assemble!(ctx)
        sol = solve_dc(sys)

        # Total = 4k, I = 10V/4k = 2.5mA
        @test isapprox(voltage(sol, :n1), 7.5; atol=0.01)
        @test isapprox(voltage(sol, :n2), 2.5; atol=0.01)
    end

end
