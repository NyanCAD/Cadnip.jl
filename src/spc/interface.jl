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

# Create a fresh top-level-ish module for holding a generated SPICE builder
# and any on-the-fly VA baremodules. Imports Cadnip so the generated
# `import ..Cadnip` inside VA baremodules resolves.
function _fresh_netlist_module(name::Symbol)
    mod = Module(gensym(name))
    Core.eval(mod, :(import Cadnip))
    return mod
end

function ensure_cache!(mod::Module)
    if isdefined(mod, :var"#cedar_parse_cache#")
        return mod.var"#cedar_parse_cache#"
    end
    cache = CedarParseCache(mod)
    Core.eval(mod, :(const var"#cedar_parse_cache#" = $cache))
    return cache
end

"""
    codegen_hdl!(cache::CedarParseCache, path::String) -> Module

Parse and codegen a Verilog-A file referenced by a SPICE `.hdl` directive,
returning the generated baremodule. Idempotent: if the cache already holds
a `(VANode, Module)` pair (e.g. from PDK precompilation), the cached module
is returned unchanged.

Called from the sema walk in `src/spc/sema.jl` when a `.hdl` directive is
encountered. Evals into `cache.thismod`, which is the same module where the
SPICE builder function will be eval'd — so the generated device types are
visible at stamp time via ordinary module-level resolution.
"""
codegen_hdl!(cache::CedarParseCache, path::AbstractString) = codegen_hdl!(cache, String(path))

function codegen_hdl!(cache::CedarParseCache, path::String)
    entry = get(cache.va_cache, path, nothing)
    if entry isa Pair{VANode, Module}
        return entry.second
    end
    va = entry === nothing ? parse_and_cache_va!(cache, path) : entry
    @assert va isa VANode
    expr = Cadnip.make_mna_module(va)
    @assert expr.head === :toplevel
    module_def = expr.args[1]
    @assert module_def.head === :module
    mod_name = module_def.args[2]::Symbol
    Core.eval(cache.thismod, expr)
    sm = getfield(cache.thismod, mod_name)
    recache_va!(cache.thismod, path, Pair{VANode, Module}(va, sm))
    return sm
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

    # Generate MNA builder function. Pass the caller's cache so `.hdl` VA
    # includes codegen into the caller's module alongside the SPICE builder.
    parse_cache = ensure_cache!(__module__)
    circuit_name = gensym(:circuit)
    code = make_mna_circuit(sa; circuit_name, parse_cache)

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

    parse_cache = ensure_cache!(__module__)
    circuit_name = gensym(:circuit)
    code = make_mna_circuit(sa; circuit_name, parse_cache)

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
    parse_cache = ensure_cache!(mod)
    code = make_mna_circuit(sa; circuit_name=f.name, parse_cache)
    Base.eval(mod, code)
    return nothing
end

function Base.include(mod::Module, f::SpectreFile)
    sa = _parse_netlist_file(f.path, :spectre)
    parse_cache = ensure_cache!(mod)
    code = make_mna_circuit(sa; circuit_name=f.name, parse_cache)
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
        mod = _fresh_netlist_module(eff_name)
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
        mod = _fresh_netlist_module(:netlist)
        parse_cache = ensure_cache!(mod)
        builder_code = make_mna_circuit(sa; circuit_name=eff_name, parse_cache)
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
