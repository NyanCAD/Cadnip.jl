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
    # GitHub Pages serves this repository from `gh-pages` — that is the branch
    # https://nyancad.github.io/Cadnip.jl/ is built from. The old `branch =
    # "docs"` named a branch that does not exist, so a deploy would have landed
    # somewhere nothing serves.
    branch = "gh-pages",
    # Explicit rather than auto-detected: Documenter's default runs
    # `git remote show origin` and falls back to "master" if that call fails,
    # which would silently skip the deploy on a push to main.
    devbranch = "main",
)
