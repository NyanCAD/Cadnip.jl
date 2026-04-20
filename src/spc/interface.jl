struct SpCircuit{CktID, Subckts}
    params::NamedTuple
    models::NamedTuple
end

function getsema end
getsema(ckt::SpCircuit{CktID}) where {CktID} = getsema(CktID)

function generate_sp_code(world::UInt64, source::LineNumberNode, ::Type{SpCircuit{CktId, Subckts}}, args...) where {CktId, Subckts}
    sig = Tuple{typeof(getsema), Type{CktId}}
    mthds = Base._methods_by_ftype(sig, -1, world)
    gen = Core.GeneratedFunctionStub(identity, Core.svec(:var"self", :args), Core.svec())
    if mthds === nothing || length(mthds) != 1
        return gen(world, source, :(getsema($CktID); error("Cedar Internal ERROR: Could not find spice method")))
    end
    match = only(mthds)

    mi = Core.Compiler.specialize_method(match)
    ci = Core.Compiler.retrieve_code_info(mi, world)
    if ci === nothing
        return gen(world, source, :(getsema($CktID); error("Cedar Internal ERROR: Could not find spice source")))
    end

    sema = ci.code[end].val
    if isa(sema, Core.SSAValue)
        sema = ci.code[sema.id]
    end
    if isa(sema, QuoteNode)
        sema = sema.value
    end
    @assert isa(sema, SemaResult)

    return gen(world, source, codegen(sema))
end

function analyze_mosfet_import(dialect, level)
    if dialect == :ngspice
        if level == 5
            #error("bsim2 not supported")
            #return :bsim2
        elseif level == 8 || level == 49
            #error("bsim3 not supported")
            #return :bsim3
        elseif level == 14 || level == 54
            return :BSIM4
        end
    end
    return nothing
end

function analyze_imports!(n::SNode, parse_cache::Union{CedarParseCache, Nothing}, traverse_imports::Bool=false;
        imports=Set{Symbol}(),
        hdl_imports=Set{String}(),
        includes::Set{String}=Set{String}(),
        pkg_hdl_imports=Set{String}(),
        pkg_spc_import=Set{String}(),
        thispath::Union{String, Nothing}=nothing)
    for stmt in n.stmts
        if isa(stmt, SNode{SP.IncludeStatement}) || isa(stmt, SNode{SP.LibInclude}) || isa(stmt, SNode{SP.HDLStatement})
            str = strip(unescape_string(String(stmt.path)), ['"', '\'']) # verify??
            if startswith(str, JLPATH_PREFIX)
                path = str[sizeof(JLPATH_PREFIX)+1:end]
                components = splitpath(path)
                push!(imports, Symbol(components[1]))
                if isa(stmt, SNode{SP.HDLStatement})
                    push!(pkg_hdl_imports, str)
                else
                    push!(pkg_spc_import, str)
                end
            else
                if thispath !== nothing
                    str = joinpath(dirname(thispath), str)
                end
                if parse_cache !== nothing
                    if isa(stmt, SNode{SP.HDLStatement})
                        parse_and_cache_va!(parse_cache, str)
                    else
                        str in includes && continue
                        push!(includes, str)
                        analyze_imports!(parse_and_cache_spc!(parse_cache, str), parse_cache; imports, hdl_imports, includes, thispath=str)
                    end
                else
                    if isa(stmt, SNode{SP.HDLStatement})
                        push!(hdl_imports, str)
                    else
                        push!(includes, str)
                    end
                end
            end
        elseif isa(stmt, SNode{SP.Model})
            typ = LSymbol(stmt.typ)
            mosfet_type = typ in (:nmos, :pmos) ? typ : nothing
            local level = 1
            for p in stmt.parameters
                name = LSymbol(p.name)
                if name == :level
                    # TODO
                    level = parse(Float64, String(p.val))
                    continue
                end
            end
            mosfet_type === nothing && continue
            imp = analyze_mosfet_import(:ngspice, level)
            imp !== nothing && push!(imports, imp)
        elseif isa(stmt, Union{SNode{SPICENetlistSource}, SNode{SP.Subckt}, SNode{SP.LibStatement}})
            analyze_imports!(stmt, parse_cache; imports, hdl_imports, includes, pkg_hdl_imports, pkg_spc_import, thispath)
        end
    end
    return imports, hdl_imports, includes, pkg_hdl_imports, pkg_spc_import
end

function ensure_cache!(mod::Module)
    if isdefined(mod, :var"#cedar_parse_cache#")
        return mod.var"#cedar_parse_cache#"
    end
    cache = CedarParseCache(mod)
    Core.eval(mod, :(const var"#cedar_parse_cache#" = $cache))
    return cache
end

function codegen_missing_imports!(thismod::Module, imps::Union{Dict{Symbol, Module}, NamedTuple}, pkg_hdl_imports::Set{String}, pkg_spc_import::Set{String})
    if isa(imps, NamedTuple)
        imps = Dict{Symbol, Module}(pairs(imps)...)
    end
    for imp in pkg_spc_import
        @assert startswith(imp, JLPATH_PREFIX)
        path = imp[sizeof(JLPATH_PREFIX)+1:end]
        components = splitpath(path)
        mod = imps[Symbol(components[1])]
        localpath = joinpath(components[2:end])
        cache = ensure_cache!(mod)
        if haskey(cache.spc_cache, localpath)
            continue
        end
        imports, _, _, sub_pkg_hdl_imports, sub_pkg_spc_import = analyze_imports!(parse_and_cache_spc!(cache, localpath), cache, thispath=localpath)
        sub_imps = Dict{Symbol, Module}(Symbol(pkg) => Base.require(mod, Symbol(pkg)) for pkg in imports)
        codegen_missing_imports!(mod, sub_imps, sub_pkg_hdl_imports, sub_pkg_spc_import)
    end
    for imp in pkg_hdl_imports
        @assert startswith(imp, JLPATH_PREFIX)
        path = imp[sizeof(JLPATH_PREFIX)+1:end]
        components = splitpath(path)
        mod = imps[Symbol(components[1])]
        localpath = joinpath(components[2:end])
        cache = ensure_cache!(mod)
        codegen_hdl_import!(mod, cache, localpath)
    end
end

function codegen_hdl_import!(mod::Module, cache::CedarParseCache, imp::String)
    va = get(cache.va_cache, imp, nothing)
    va isa Pair && return
    if va === nothing
        va = parse_and_cache_va!(cache, imp)
    end

    vamod = va.stmts[end]
    s = gensym(String(vamod.id))
    sm = Core.eval(mod, :(baremodule $s
        const VerilogAEnvironment = $(Cadnip.VerilogAEnvironment)
        using .VerilogAEnvironment
        $(Cadnip.make_spice_device(vamod))
        const $(Symbol(lowercase(String(vamod.id)))) = $(Symbol(vamod.id))
    end))

    recache_va!(mod, imp, Pair{VANode, Module}(va, sm))
end

function codegen_hdl_imports!(mod::Module, hdl_imports)
    cache = mod.var"#cedar_parse_cache#"
    for imp in hdl_imports
        codegen_hdl_import!(mod, cache, imp)
    end
end

#==============================================================================#
# Internal helper: eval a builder expression into a module
#
# Shared by Base.include(mod, SpiceFile/SpectreFile), MNACircuit(path), and
# MNACircuit(code; lang). Returns the builder function bare (no invokelatest
# wrapper) — MNACircuit{F,...} specializes on F, and dc!/tran!/ac! cross the
# function-barrier boundary in the current world. See docs on Invokelatest policy.
#==============================================================================#
function _eval_builder_into_module(mod::Module, code::Expr, circuit_name::Symbol)
    # After codegen hygiene pass, `mod` does not need any Cadnip `using` statements —
    # generated code emits fully-qualified references.
    Base.eval(mod, code)
    return getfield(mod, circuit_name)
end

"""
    sp"..."

Parse SPICE code and generate an MNA builder function.

The result is a callable that takes (params, spec) and returns an MNAContext.

# Flags
- No flag (default): `implicit_title=true` - first line is treated as title
- `i` flag: `implicit_title=false` - inline mode, no title expected
- `e` flag: enable Julia escape sequences in string

# Example
```julia
# Default mode requires a title line (first line is treated as comment)
circuit = sp\"\"\"
* Voltage divider
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
\"\"\"
ctx = circuit((;), MNASpec())
sol = solve_dc(ctx)
sol[:out]  # Returns 2.5

# Inline mode (i flag) - no title line needed
circuit2 = sp\"\"\"
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
\"\"\"i
```
"""
macro sp_str(str, flag="")
    enable_julia_escape = 'e' in flag
    inline = 'i' in flag
    sa = NyanSpectreNetlistParser.parse(IOBuffer(str); start_lang=:spice, enable_julia_escape,
        implicit_title = !inline, fname=String(__source__.file), srcline=__source__.line)

    # Generate MNA builder function
    circuit_name = gensym(:circuit)
    code = make_mna_circuit(sa; circuit_name)

    # Return the builder function
    return esc(quote
        $code
        $circuit_name
    end)
end

"""
    spc"..."

Parse Spectre code and generate an MNA builder function. Symmetric with `sp"..."`.

# Example
```julia
circuit = spc\"\"\"
v1 (vcc 0) vsource type=dc dc=5
r1 (vcc out) resistor r=1k
r2 (out 0) resistor r=1k
\"\"\"
sol = dc!(MNACircuit(circuit))
```
"""
macro spc_str(str, flag="")
    enable_julia_escape = 'e' in flag
    sa = NyanSpectreNetlistParser.parse(IOBuffer(str); start_lang=:spectre, enable_julia_escape,
        fname=String(__source__.file), srcline=__source__.line)

    circuit_name = gensym(:circuit)
    code = make_mna_circuit(sa; circuit_name)

    return esc(quote
        $code
        $circuit_name
    end)
end

#==============================================================================#
# SpiceFile / SpectreFile + Base.include — file-first loading API
#
# Mirrors the existing VAFile pattern (src/vasim.jl:3431). Preserves file path
# for relative .hdl / .include resolution.
#==============================================================================#

"""
    SpiceFile(path; name=<stem>)

File-loading wrapper for SPICE netlists. Used with `Base.include(mod, SpiceFile(path))`
to define a builder function named `name` in `mod`.

Default `name` is the filename stem (e.g. `SpiceFile("amp.sp").name === :amp`).

See also: `SpectreFile`, `MNACircuit(path)`.
"""
struct SpiceFile
    path::String
    name::Symbol
end
SpiceFile(path::AbstractString; name=Symbol(first(splitext(basename(String(path)))))) =
    SpiceFile(String(path), Symbol(name))

"""
    SpectreFile(path; name=<stem>)

File-loading wrapper for Spectre netlists. Symmetric with `SpiceFile`.
"""
struct SpectreFile
    path::String
    name::Symbol
end
SpectreFile(path::AbstractString; name=Symbol(first(splitext(basename(String(path)))))) =
    SpectreFile(String(path), Symbol(name))

export SpiceFile, SpectreFile

function _parse_netlist_file(path::String, lang::Symbol)
    if lang === :spice
        sa = NyanSpectreNetlistParser.parsefile(path; start_lang=:spice, implicit_title=true)
    elseif lang === :spectre
        sa = NyanSpectreNetlistParser.parsefile(path; start_lang=:spectre)
    else
        error("Unknown netlist language: $lang (expected :spice or :spectre)")
    end
    if sa.ps.errored
        throw(LoadError(path, 0, SpectreParseError(sa)))
    end
    return sa
end

function _parse_netlist_string(code::AbstractString, lang::Symbol; source_dir=nothing)
    fname = source_dir === nothing ? "<string>" : joinpath(source_dir, "<string>")
    if lang === :spice
        sa = NyanSpectreNetlistParser.parse(IOBuffer(code); start_lang=:spice,
            implicit_title=true, fname=String(fname))
    elseif lang === :spectre
        sa = NyanSpectreNetlistParser.parse(IOBuffer(code); start_lang=:spectre,
            fname=String(fname))
    else
        error("Unknown netlist language: $lang (expected :spice or :spectre)")
    end
    if sa.ps.errored
        throw(LoadError(fname, 0, SpectreParseError(sa)))
    end
    return sa
end

function Base.include(mod::Module, f::SpiceFile)
    sa = _parse_netlist_file(f.path, :spice)
    code = make_mna_circuit(sa; circuit_name=f.name)
    Base.eval(mod, code)
    return nothing
end

function Base.include(mod::Module, f::SpectreFile)
    sa = _parse_netlist_file(f.path, :spectre)
    code = make_mna_circuit(sa; circuit_name=f.name)
    Base.eval(mod, code)
    return nothing
end

"""
    infer_lang_from_ext(path) -> Symbol

Infer netlist language from file extension. `.scs` → `:spectre`, else `:spice`.
"""
function infer_lang_from_ext(path::AbstractString)
    _, ext = splitext(String(path))
    return lowercase(ext) == ".scs" ? :spectre : :spice
end

#==============================================================================#
# MNACircuit(path) and MNACircuit(code; lang) — file-first entry points
#
# One AbstractString method: if the argument names an existing file, treat it as
# a path (extension-inferred language); otherwise treat it as inline netlist code
# (language defaults to :spice, override with lang=).
#==============================================================================#

"""
    MNACircuit(path_or_code::AbstractString; lang=nothing, source_dir=nothing,
               name=<stem>, kwargs...) -> MNACircuit

If `path_or_code` names an existing file, loads the netlist from disk and infers
language from the extension (`.scs` → Spectre, else SPICE). Otherwise treats the
string as an inline netlist (default `:spice`, override with `lang=`).

`source_dir` is used to resolve relative `.hdl` / `.include` paths for inline
netlists; absent, relative paths fail.

!!! note "Top-level use only"
    This constructor calls `Base.eval` internally to install the generated
    builder. Julia captures the caller's world age at function entry, so when
    called *inside a function body* the subsequent `dc!`/`tran!`/`ac!` will
    error with _"method too new"_. Use this form at the REPL or module top
    level.

    For loads inside a function body, bring the circuit into scope at top
    level first:
    ```julia
    Base.include(@__MODULE__, SpiceFile("amp.sp"))  # top level — defines `amp`

    function run_sim()
        c = MNACircuit(amp; R1=1e3)                 # no eval, no world-age
        dc!(c)
    end
    ```

```julia
# Top level:
circuit = MNACircuit("amp.sp")              # file
circuit = MNACircuit("V1 vcc 0 5\\nR1 vcc 0 1k"; lang=:spice)  # inline
sol = dc!(circuit)
```
"""
function MNA.MNACircuit(path_or_code::AbstractString;
                        lang::Union{Symbol,Nothing}=nothing,
                        source_dir=nothing,
                        name::Union{Symbol,Nothing}=nothing,
                        spec::MNA.MNASpec=MNA.MNASpec(), kwargs...)
    is_file = lang === nothing && source_dir === nothing && isfile(path_or_code)
    if is_file
        path = String(path_or_code)
        eff_lang = infer_lang_from_ext(path)
        eff_name = name === nothing ?
            Symbol(first(splitext(basename(path)))) : name
        mod = Module(gensym(eff_name))
        if eff_lang === :spice
            Base.include(mod, SpiceFile(path; name=eff_name))
        else
            Base.include(mod, SpectreFile(path; name=eff_name))
        end
        builder = getfield(mod, eff_name)
    else
        eff_lang = something(lang, :spice)
        sa = _parse_netlist_string(path_or_code, eff_lang; source_dir=source_dir)
        eff_name = name === nothing ? gensym(:circuit) : name
        builder_code = make_mna_circuit(sa; circuit_name=eff_name)
        mod = Module(gensym(:netlist))
        Base.eval(mod, builder_code)
        builder = getfield(mod, eff_name)
    end
    # Return the bare builder. This constructor is documented as top-level-only;
    # the call to `Base.eval` has advanced the world so a following top-level
    # `dc!(circuit)` sees the fresh methods. Inside a function body it errors,
    # which is a clear signal to move the load to top level.
    params = NamedTuple(kwargs)
    return MNA.MNACircuit(builder, params, spec)
end
