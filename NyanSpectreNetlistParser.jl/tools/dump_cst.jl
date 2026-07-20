import NyanSpectreNetlistParser
import AbstractTrees

const SPICENetlistParser = NyanSpectreNetlistParser.SPICENetlistParser

function dump_node(node, depth::Int)
    node === nothing && return
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
    length(ARGS) < 1 && error("Usage: dump_cst.jl <file.sp> [dialect]")
    path = ARGS[1]
    dialect = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :ngspice
    src  = read(path, String)
    root = SPICENetlistParser.parse(src; spice_dialect=dialect, implicit_title=true)
    dump_node(root, 0)
end

main()
