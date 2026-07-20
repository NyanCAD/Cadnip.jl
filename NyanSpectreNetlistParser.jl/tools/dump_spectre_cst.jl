import NyanSpectreNetlistParser
import AbstractTrees

# Canonical Spectre CST dumper — the ground-truth generator for the Rust
# differential test. Mirrors tools/dump_cst.jl (the SPICE dumper) but drives the
# Spectre entry point. Preorder DFS, one line per node,
# "<indent><FormStructName> <start>-<stop>" with a half-open, 0-based *content*
# byte span (leading/trailing trivia excluded); zero-width nodes are skipped.

function dump_node(node, depth::Int)
    node === nothing && return
    # A NodeList (a synthetic wrapper around an EXPRList field, e.g.
    # SubcktNodes.nodes::Maybe{EXPRList{SNode}}) is not a form node — its `.expr`
    # is an EXPRList with no width/off/form. Recurse into its elements at the
    # SAME depth so they appear as direct children of the enclosing form, exactly
    # as they do in the rowan CST (which has no intermediate list node).
    if !hasfield(typeof(node.expr), :width)
        for child in AbstractTrees.children(node)
            dump_node(child, depth)
        end
        return
    end
    w = node.expr.width
    w == 0 && return
    start = Int(node.startof) + Int(node.expr.off)
    stop  = start + Int(w)
    kind  = string(nameof(typeof(node.expr.form)))
    indent = "  " ^ depth
    println("$(indent)$(kind) $(start)-$(stop)")
    for child in AbstractTrees.children(node)
        dump_node(child, depth + 1)
    end
end

function main()
    length(ARGS) < 1 && error("Usage: dump_spectre_cst.jl <file.scs> [start_lang]")
    path = ARGS[1]
    start_lang = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :spectre
    src  = read(path, String)
    root = NyanSpectreNetlistParser.SpectreNetlistCSTParser.parse(src; start_lang, implicit_title=true)
    dump_node(root, 0)
end

main()
