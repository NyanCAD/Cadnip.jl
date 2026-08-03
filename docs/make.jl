using Cadnip
using VADistillerModels     # model cards used by the examples in devices.md
using Documenter

makedocs(
    sitename = "Cadnip",
    # Pin the source links instead of deriving them from `origin`, so a
    # checkout with a mirror/proxy remote builds the same site as CI does.
    repo = Documenter.Remotes.GitHub("NyanCAD", "Cadnip.jl"),
    format = Documenter.HTML(),
    modules = [Cadnip],
    pages = Any[
        "Home" => "index.md",
        "netlists.md",
        "parameters.md",
        "analyses.md",
        "devices.md",
    ],
    warnonly = true,
)


# Normalize the documenter key if it's not already base64-encoded
using Base64
documenter_key = get(ENV, "DOCUMENTER_KEY", "")
try
    base64decode(documenter_key)
catch e
    if isa(e, ArgumentError)
        if !endswith(documenter_key, "\n")
            global documenter_key = string(documenter_key, "\n")
        end
        ENV["DOCUMENTER_KEY"] = base64encode(documenter_key)
    else
        rethrow(e)
    end
end

deploydocs(
    repo = "github.com/NyanCAD/Cadnip.jl.git",
    branch = "docs",
)
