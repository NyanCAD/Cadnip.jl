using AbstractTrees

const SC = SpectreNetlistCSTParser
const SP = SPICENetlistParser.SPICENetlistCSTParser

function Base.LineNumberNode(n::SC.Node)
    sf = n.ps.srcfile
    lsf = sf.lineinfo
    lno_first = LineNumbers.compute_line(lsf, n.startof+n.expr.off)
    # line 1 is at offset 0
    offset = n.ps.srcline - 1
    LineNumberNode(lno_first+offset, Symbol(sf.path))
end

function visit_errors(sa; io=stdout, verbose=false)
    print("\n")
    for node in PreOrderDFS(sa)
        if node isa SC.Node{<:Union{SC.Error, SP.Error}}
            start = node.expr.off
            len = max(node.expr.width, 1)
            context = node
            while context.parent isa SC.Node{<:Union{SC.Incomplete, SP.Incomplete}}
                context = context.parent
            end
            start += context.expr.fullwidth - node.expr.fullwidth
            if node.expr.kind in (SC.UnexpectedToken, SP.UnexpectedToken)
                printstyled(io, "ERROR: ", bold=true, color=:red)
                print(io, "unexpected token")
                if node.parent isa SC.Node{<:Union{SC.Incomplete, SP.Incomplete}}
                    print(io, " while parsing a ", nameof(typeof(context.expr.form).parameters[1]))
                end
                println(io, ":")
            elseif  node.expr.kind == SC.UnknownStatement
                printstyled(io, "ERROR: ", bold=true, color=:red)
                println(io, "unknown named statement. If you intended to create an instance, add parentheses around the net names")
            end
            lnn = LineNumberNode(context)
            println(io, lnn.file, ":", lnn.line, ":")
            line = fullcontents(context)
            pointer = " "^start * "^"^len
            println(rstrip(line))
            printstyled(io, pointer, "\n"; color=:light_green)
            println(io, "got: $(node.expr.got)")
            if node.expr.expected !== nothing
                println(io, "expected: $(node.expr.expected)")
            end
        end
    end
end