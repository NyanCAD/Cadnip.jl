abstract type CircRef; end

# Various references to SPICE-namespaced things
struct InstanceRef
    ref::SNode
end
struct ParamRef
    ref::SNode
end
struct ModelRef
    ref::SNode
end
struct NetRef
    refs::Vector{SNode}
end

struct AmbiguousRef
    instance::Union{InstanceRef, Nothing}
    param::Union{ParamRef, Nothing}
    model::Union{ModelRef, Nothing}
    net::Union{NetRef, Nothing}
end

# Mapping of SPICE-namespace to DAECompiler namespace
# Phase 0: Guard DAECompiler method extensions
@static if CedarSim.USE_DAECOMPILER
    function ScopeRef(sys::DAECompiler.IRODESystem, ref::NetRef)
        getproperty(sys, Symbol(string("node_", lowercase(String(ref.refs[1].name)))))
    end
end


using SymbolicIndexingInterface
SymbolicIndexingInterface.symbolic_type(::NetRef) = ScalarSymbolic()
SymbolicIndexingInterface.symbolic_type(::Type{<:NetRef}) = ScalarSymbolic()

# Phase 0: Guard DAECompiler method extensions
@static if CedarSim.USE_DAECOMPILER
    SymbolicIndexingInterface.is_independent_variable(sys::DAECompiler.TransformedIRODESystem,
        sym::NetRef) = false
    function SymbolicIndexingInterface.is_variable(sys::DAECompiler.TransformedIRODESystem,
        sym::NetRef)
        SymbolicIndexingInterface.is_variable(sys, ScopeRef(DAECompiler.get_sys(sys), sym))
    end
    function SymbolicIndexingInterface.is_observed(sys::DAECompiler.TransformedIRODESystem,
        sym::NetRef)
        SymbolicIndexingInterface.is_observed(sys, ScopeRef(DAECompiler.get_sys(sys), sym))
    end
    SymbolicIndexingInterface.variable_index(sys::DAECompiler.TransformedIRODESystem,
        sym::NetRef) = SymbolicIndexingInterface.variable_index(sys, ScopeRef(DAECompiler.get_sys(sys), sym))
    SymbolicIndexingInterface.is_parameter(sys::DAECompiler.TransformedIRODESystem, sym::NetRef) = false
    function (this::DAECompiler.DAEReconstructedObserved)(sym::NetRef, args...)
        this(ScopeRef(this.sys, sym), args...)
    end
end
