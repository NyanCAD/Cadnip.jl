# TODO:
# - Move endl inline
# - Nodes can be integers

import ...@case, ....@trynext, ....@trysetup, ....@donext

function parse_spice_toplevel(ps)::EXPR
    @case kind(nt(ps)) begin
        SIMULATOR  => return parse_simulator(ps)
        DOT        => return parse_dot(ps)
        TITLE_LINE => return parse_title(ps)
        NEWLINE    => return error!(ps, UnexpectedToken, "SPICE statement")
        _          => nothing # fall-through
    end
    if !is_ident(kind(nt(ps)))
        # TODO: This should be an "unrecognized SPICE directive/instance" error
        return error!(ps, UnexpectedToken)
    end
    return parse_instance(ps)
end


function parse_dot(ps)
    dot = take(ps, DOT)
    parse_dot(ps, dot)
end

function parse_dot(ps, dot)
    @trysetup AbstractASTNode
    @trynext dot
    @case kind(nt(ps)) begin
        DC => parse_dc(ps, dot)
        AC => parse_ac(ps, dot)
        LIB => parse_lib(ps, dot)
        END => parse_end(ps, dot)
        ENDL => parse_endl(ps, dot) # Only when parsing a library?
        INCLUDE => parse_include(ps, dot)
        PARAMETERS => parse_param(ps, dot)
        CSPARAM => parse_csparam(ps, dot)
        SUBCKT => parse_subckt(ps, dot)
        MODEL => parse_model(ps, dot)
        OPTIONS => parse_options(ps, dot)
        TITLE => parse_title(ps, dot)
        GLOBAL => parse_global(ps, dot)
        DATA => parse_data(ps, dot)
        IC   => parse_ic(ps, dot)
        MEASURE => parse_measure(ps, dot)
        TRAN => parse_tran(ps, dot)
        PRINT => parse_print(ps, dot)
        TEMP => parse_temp(ps, dot)
        WIDTH => parse_width(ps, dot)
        HDL => parse_hdl(ps, dot)
        IF => parse_if(ps, dot)
        # TODO: This should return an "unexpected DOT command" error
        _ => @trynext error!(ps, UnexpectedToken)
    end
end

function parse_title(ps, dot)
    @trysetup Title dot
    @trynext kw = take_kw(ps, TITLE)
    line = nothing
    if kind(nt(ps)) == TITLE_LINE
        @trynext line = take(ps, TITLE_LINE)
    end
    @trynext nl = accept_newline(ps)
    return EXPR(Title(dot, kw, line, nl))
end

function parse_title(ps)
    @trysetup Title
    @trynext line = take(ps, TITLE_LINE)
    @trynext nl = accept_newline(ps)
    return EXPR(Title(nothing, nothing, line, nl))
end

function parse_tran(ps, dot)
    @trysetup Tran dot
    @trynext kw = take_kw(ps, TRAN)
    @trynext tstep_or_stop = take_literal(ps)
    tstop = if is_literal(kind(nt(ps)))
        tstep = tstep_or_stop
        @trynext take_literal(ps)
    else
        tstep = nothing
        tstep_or_stop
    end
    tstart = if is_literal(kind(nt(ps)))
        @trynext take_literal(ps)
    end
    tmax = if is_literal(kind(nt(ps)))
        @trynext take_literal(ps)
    end
    uic = if is_ident(kind(nt(ps)))
        @trynext take_identifier(ps)
    end
    @trynext nl = accept_newline(ps)
    if tstep === nothing && ps.lexer.strict && (ps.lexer.spice_dialect in (:ngspice, :hspice))
        return @trynext error!(ps, StrictModeViolation)
    end
    return EXPR(Tran(dot, kw, tstep, tstop, tstart, tmax, uic, nl))
end

function parse_measure(ps, dot)
    @trysetup MeasurePointStatement dot
    @trynext kw = take_kw(ps, MEASURE)
    @trynext type = take_kw(ps, (AC, DC, OP, TRAN, TF, NOISE))
    @trynext name = if kind(nt(ps)) == STRING
        take_string(ps)
    else
        take_identifier(ps)
    end

    if kind(nt(ps)) in (FIND, DERIV, PARAMETERS, WHEN, AT)
        return parse_measure_point(ps, dot, kw, type, name)
    else
        return parse_measure_range(ps, dot, kw, type, name)
    end
end

function parse_trig_targ(ps)
    @trysetup TrigTarg
    @trynext kw = take_kw(ps, (TRIG, TARG))
    @trynext lhs = parse_expression(ps)

    val = nothing
    if kind(nt(ps)) == VAL
        @trynext val_kw = take_kw(ps)
        @trynext eq = take(ps, EQ)
        @trynext rhs = parse_expression(ps)
        val = EXPR(Val_(val_kw, eq, rhs))
    end

    td = nothing
    if kind(nt(ps)) == TD
        @trynext td = parse_td(ps)
    end

    rfc = nothing
    if kind(nt(ps)) in (RISE, FALL, CROSS)
        @trynext rfc = parse_risefallcross(ps)
    end
    return EXPR(TrigTarg(kw, lhs, val, td, rfc))
end

function parse_measure_range(ps, dot, kw, type, name)
    @trysetup MeasureRangeStatement dot kw type name
    avgmaxminpprmsinteg = nothing

    if kind(nt(ps)) in (AVG, MAX, MIN, PP, RMS, INTEG)
        @trynext kw_op = take_kw(ps)
        @trynext expr = parse_expression(ps)
        avgmaxminpprmsinteg = EXPR(AvgMaxMinPPRmsInteg(kw_op, expr))
    end

    trig = nothing
    targ = nothing
    if kind(nt(ps)) == TRIG
        @trynext trig = parse_trig_targ(ps)
    end
    if kind(nt(ps)) == TARG
        @trynext targ = parse_trig_targ(ps)
    end

    @trynext nl = accept_newline(ps)
    return EXPR(MeasureRangeStatement(dot, kw, type, name, avgmaxminpprmsinteg, trig, targ, nl))
end

function parse_risefallcross(ps)
    @trysetup RiseFallCross
    @trynext rfc_kw = take_kw(ps, (RISE, FALL, CROSS))
    @trynext eq_rfc = take(ps, EQ)
    val = if kind(nt(ps)) == LAST
        @trynext take_kw(ps)
    else
        @trynext take_literal(ps)
    end
    return EXPR(RiseFallCross(rfc_kw, eq_rfc, val))
end

function parse_td(ps)
    @trysetup TD_
    @trynext td_kw = take_kw(ps, TD)
    @trynext eq_td = take(ps, EQ)
    @trynext expr = parse_expression(ps)
    return EXPR(TD_(td_kw, eq_td, expr))
end

function parse_measure_point(ps, dot, kw, type, name)
    @trysetup MeasurePointStatement
    @trynext dot
    @trynext kw
    @trynext type
    @trynext name
    fdp = nothing
    whenat = nothing
    td = nothing
    rfc = nothing
    if kind(nt(ps)) in (FIND, DERIV, PARAMETERS)
        @trynext fdp_kw = take_kw(ps, (FIND, DERIV, PARAMETERS))
        eq = if kind(nt(ps)) == EQ
            @trynext take(ps, EQ)
        else
            nothing
        end
        @trynext expr = parse_expression(ps)
        fdp = EXPR(FindDerivParam(fdp_kw, eq, expr))
    end
    if kind(nt(ps)) in (WHEN, AT)
        if kind(nt(ps)) == AT
            @trynext at_kw = take_kw(ps)
            @trynext eq = take(ps, EQ)
            @trynext at_expr = parse_expression(ps)
            whenat = EXPR(At(at_kw, eq, at_expr))
        else
            @trynext when_kw = take_kw(ps)
            @trynext when_expr = parse_expression(ps)
            whenat = EXPR(When(when_kw, when_expr))
        end
    end
    if kind(nt(ps)) in (RISE, FALL, CROSS)
        @trynext rfc = parse_risefallcross(ps)
    end
    if kind(nt(ps)) == TD
        @trynext td = parse_td(ps)
    end
    @trynext nl = accept_newline(ps)
    return EXPR(MeasurePointStatement(dot, kw, type, name, fdp, whenat, rfc, td, nl))
end


function parse_ic_statement(ps)
    if kind(nt(ps)) == STAR
        return EXPR(WildCard(nothing, take(ps, STAR)))
    elseif kind(nt(ps)) == NUMBER
        int = take_literal(ps)
        if kind(nt(ps)) == STAR
            return EXPR(WildCard(int, take(ps, STAR)))
        end
        return int
    elseif is_ident(kind(nt(ps)))
        l = take_identifier(ps)
        if kind(nt(ps)) == COLON
            colon = take(ps, COLON)
            r = take_identifier(ps)
            return EXPR(Coloned(l, colon, r))
        end
        return l
    else
        error("unhandled ic statement")
    end
end

function parse_ic(ps, dot)
    @trysetup ICStatement
    @trynext kw = take_kw(ps, IC)
    entries = EXPRList{ICEntry}()
    while !eol(ps)
        @trynext name = take_identifier(ps)
        @trynext lparen = take(ps, LPAREN)
        @trynext arg = parse_ic_statement(ps)

        @trynext rparen = take(ps, RPAREN)
        @trynext eq = take(ps, EQ)
        @trynext val = parse_expression(ps)
        entry = EXPR(ICEntry(name, lparen, arg, rparen, eq, val))
        push!(entries, entry)
    end
    @trynext nl = accept_newline(ps)
    return EXPR(ICStatement(dot, kw, entries, nl))
end

function parse_print(ps, dot)
    @trysetup PrintStatement dot
    @trynext kw = take_kw(ps, PRINT)
    entries = EXPRList{Any}()
    while !eol(ps)
        push!(entries, @trynext parse_expression(ps))
    end
    @trynext nl = accept_newline(ps)
    return EXPR(PrintStatement(dot, kw, entries, nl))
end

function parse_data(ps, dot)
    @trysetup DataStatement dot
    @trynext kw = take_kw(ps, DATA)
    @trynext blockname = take_identifier(ps)

    n_rows = 0
    row_names = EXPRList{Identifier}()
    while is_ident(kind(nt(ps)))
        @trynext id = take_identifier(ps)
        push!(row_names, id)
        n_rows += 1
    end

    values = EXPRList{NumberLiteral}()
    while !eol(ps)
        for i in 1:n_rows
            push!(values, @trynext take_literal(ps))  # NumberLiteral directly
        end
    end

    @trynext nl = accept_newline(ps)
    @trynext dot2 = take(ps, DOT)
    @trynext endkw = take_kw(ps, ENDDATA)
    @trynext nl2 = accept_newline(ps)
    return EXPR(DataStatement(dot, kw, blockname, row_names, values, nl, dot2, endkw, nl2))
end

function parse_options(ps, dot)
    @trysetup OptionStatement dot
    @trynext kw = take_kw(ps, OPTIONS)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(OptionStatement(dot, kw, params, nl))
end

function parse_width(ps, dot)
    @trysetup WidthStatement dot
    @trynext kw = take_kw(ps, WIDTH)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(WidthStatement(dot, kw, params, nl))
end

function parse_node(ps)
    @trysetup NodeName
    if kind(nt(ps)) == NUMBER
        return EXPR(NodeName(@trynext take_literal(ps)))
    elseif is_ident(kind(nt(ps)))
        return EXPR(NodeName(@trynext take_identifier(ps)))
    else
        return error!(ps, UnexpectedToken, "circuit node")
    end
end

function parse_node_list(ps)
    nodes = EXPRList{NodeName}()
    return parse_node_list!(nodes, ps)
end

function parse_node_list!(nodes, ps)
    @trysetup NodeName
    while !eol(ps)
        @trynext p = parse_node(ps)
        push!(nodes, p)
    end
    return nodes
end


function parse_hierarchial_node(ps, name=nothing)
    @trysetup HierarchialNode
    if name === nothing
        @trynext name = parse_node(ps)
    end
    subnodes = EXPRList{SubNode}()
    while kind(nt(ps)) == DOT
        @trynext dot = take(ps, DOT)
        @trynext subnode = parse_node(ps)
        push!(subnodes, EXPR(SubNode(dot, subnode)))
    end
    return EXPR(HierarchialNode(name, subnodes))
end

function parse_hierarchial_node_list(ps)
    nodes = EXPRList{HierarchialNode}()
    return parse_hierarchial_node_list!(nodes, ps)
end

function parse_hierarchial_node_list!(nodes, ps)
    @trysetup HierarchialNode
    while !eol(ps)
        @trynext p = parse_hierarchial_node(ps)
        push!(nodes, p)
    end
    return nodes
end

function parse_model(ps, dot)
    @trysetup Model dot
    @trynext kw = take_kw(ps, MODEL)
    @trynext name = parse_hierarchial_node(ps)
    @trynext typ = take_identifier_or_number(ps)  # Model types can start with digits
    @trynext parameters = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(Model(dot, kw, name, typ, parameters, nl))
end

function parse_subckt(ps, dot)
    @trysetup Subckt dot
    @trynext kw = take_kw(ps, SUBCKT)
    @trynext name = take_identifier_or_number(ps)
    nodes = EXPRList{HierarchialNode}()
    parameters = EXPRList{Parameter}()
    while !eol(ps)
        if kind(nnt(ps)) == EQ
            @trynext parameters = parse_parameter_list(ps)
        else
            push!(nodes, @trynext parse_hierarchial_node(ps))
        end
    end
    @trynext nl = accept_newline(ps)
    exprs = EXPRList{Any}()
    while kind(nt(ps)) != ENDMARKER
        if kind(nt(ps)) == DOT
            dot2 = take(ps, DOT)  # Don't capture yet - will be captured in appropriate branch
            if kind(nt(ps)) == ENDS
                @trynext dot2     # Capture it now since we're using it in return
                @trynext ends = take_kw(ps, ENDS)
                name_end = eol(ps) ? nothing : @trynext take_identifier_or_number(ps)
                @trynext nl2 = accept_newline(ps)
                return EXPR(Subckt(dot, kw, name, nodes, parameters, nl, exprs, dot2, ends, name_end, nl2))
            else
                @donext expr = parse_dot(ps, dot2)  # parse_dot will capture dot2
            end
        else
            @donext expr = parse_spice_toplevel(ps)
        end
        push!(exprs, expr)
    end
    @trynext error!(ps, UnexpectedToken, ENDS)
end

function parse_dc(ps, dot)
    @trysetup DCStatement dot
    @trynext kw = take_kw(ps, DC)
    commands = EXPRList{DCCommand}()
    while !eol(ps)
        @trynext command = parse_dc_command(ps)
        push!(commands, command)
    end
    @trynext nl = accept_newline(ps)
    return EXPR(DCStatement(dot, kw, commands, nl))
end

function parse_dc_command(ps)
    @trysetup DCCommand
    @trynext name = take_identifier(ps)
    @trynext start = parse_expression(ps)
    @trynext stop = parse_expression(ps)
    @trynext step = parse_expression(ps)
    return EXPR(DCCommand(name, start, stop, step))
end

function parse_ac(ps, dot)
    @trysetup ACStatement dot
    @trynext kw = take_kw(ps, AC)
    @trynext command = parse_ac_command(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(ACStatement(dot, kw, command, nl))
end

function parse_ac_command(ps)
    @trysetup ACCommand
    @trynext name = take_identifier(ps) # Should be keyword (lin/dec/oct)?
    @trynext n = parse_expression(ps)
    @trynext fstart = parse_expression(ps)
    @trynext fstop = parse_expression(ps)
    return EXPR(ACCommand(name, n, fstart, fstop))
end

function parse_endl(ps, dot)
    @trysetup EndlStatement dot
    @trynext endd = take_kw(ps, ENDL)
    endl_id = nothing
    if is_ident(kind(nt(ps)))
        @trynext endl_id = take_identifier(ps)
    end
    @trynext nl = accept_newline(ps)
    return EXPR(EndlStatement(dot, endd, endl_id, nl))
end

function parse_lib(ps, dot)
    @trysetup Any
    @trynext kw = take_kw(ps, LIB)
    # library section is of the form .lib identifier
    # library include is of the form .lib [string|identifier] identifier
    if !is_ident(kind(nnt(ps)))
        @trynext name = take_identifier(ps)
        @trynext nl = accept_newline(ps)
        stmt = nothing
        exprs = EXPRList{Any}()
        while true
            # TODO: What can be inside a lib statement?
            stmt = parse_spice_toplevel(ps)
            # Failed to parse toplevel, should have an endl here
            if stmt isa EXPR{EndlStatement}
                break
            end
            push!(exprs, stmt)
        end
        return EXPR(LibStatement(dot, kw, name, nl, exprs, stmt))
    else
        @trynext path = take_path(ps)
        @trynext name = take_identifier(ps)
        @trynext nl = accept_newline(ps)
        return EXPR(LibInclude(dot, kw, path, name, nl))
    end
end

function parse_condition(ps)
    @trysetup Condition
    @trynext lparen = accept(ps, LPAREN)
    @trynext expr = parse_expression(ps)
    @trynext rparen = accept(ps, RPAREN)
    return EXPR(Condition(lparen, expr, rparen))
end

function parse_ifelse_block(ps, dot, kws=(IF, ELSE, ELSEIF))
    # real capture of dot and kw happens here
    @trysetup IfElseCase dot
    @trynext kw = take_kw(ps, kws)
    condition = nothing
    if kw.form.kw in (IF, ELSEIF)
        @trynext condition = parse_condition(ps)
    end
    @trynext nl = accept_newline(ps)
    stmts = EXPRList{Any}()
    while true
        if kind(nt(ps)) == DOT && kind(nnt(ps)) in (ELSE, ELSEIF, ENDIF)
            break
        end
        if kind(nt(ps)) == ENDMARKER
            break
        end
        stmt = parse_spice_toplevel(ps)
        push!(stmts, stmt)
    end
    return EXPR(IfElseCase(dot, kw, condition, nl, stmts))
end

function parse_if(ps, dot)
    @trysetup IfBlock # dot is captured in parse_ifelse_block
    cases = EXPRList{IfElseCase}()
    push!(cases, @trynext parse_ifelse_block(ps, dot, (IF,)))
    kw = nothing
    while true
        dot = take(ps, DOT)
        tok = kind(nt(ps))
        if tok === ENDIF
            @trynext dot
            @trynext kw = take_kw(ps, ENDIF)
            break
        end
        push!(cases, @trynext parse_ifelse_block(ps, dot))
    end
    @trynext nl = accept_newline(ps)
    return EXPR(IfBlock(cases, dot, kw, nl))
end

function parse_primary_or_unary(ps)
    if is_unary_operator(kind(nt(ps)))
        parse_unary_op(ps)
    else
        parse_primary(ps)
    end
end


function parse_unary_op(ps)
    @trysetup UnaryOp
    @trynext op = take_operator(ps)
    @trynext primary = parse_primary(ps)
    return EXPR(UnaryOp(op, primary))
end

function parse_primary(ps)
    if is_number(kind(nt(ps)))
        lit = take_literal(ps)
        # TODO: Units?
        return lit
    elseif is_literal(kind(nt(ps)))
        return take_literal(ps)
    elseif is_ident(kind(nt(ps)))
        id = take_identifier(ps)
        if kind(nt(ps)) == LPAREN
            return parse_function_call(ps, id)
        elseif kind(nt(ps)) == DOT
            # Local trysetup for compound hierarchical node - id captured here since
            # parse_hierarchial_node doesn't capture pre-parsed arguments
            @trysetup HierarchialNode id
            return @trynext parse_hierarchial_node(ps, EXPR(NodeName(id)))
        else
            return id
        end
    elseif kind(nt(ps)) == STRING
        return take_string(ps)
    elseif kind(nt(ps)) == LSQUARE
        return parse_array(ps)
    elseif is_kw(kind(nt(ps)))
        return take_kw(ps)
    elseif kind(nt(ps)) == LBRACE
        # Local trysetup for compound Brace expression
        @trysetup Brace
        @trynext lparen = take(ps, LBRACE)
        @trynext e = parse_expression(ps)
        @trynext rparen = accept(ps, RBRACE)
        return EXPR(Brace(lparen, e, rparen))
    elseif kind(nt(ps)) == LPAREN
        # Local trysetup for compound Parens expression
        @trysetup Parens
        @trynext lparen = take(ps, LPAREN)
        @trynext e = parse_expression(ps)
        @trynext rparen = accept(ps, RPAREN)
        return EXPR(Parens(lparen, e, rparen))
    elseif kind(nt(ps)) == PRIME
        # Local trysetup for compound Prime expression
        @trysetup Prime
        @trynext lparen = take(ps, PRIME)
        @trynext e = parse_expression(ps)
        @trynext rparen = accept(ps, PRIME)
        return EXPR(Prime(lparen, e, rparen))
    elseif kind(nt(ps)) == JULIA_ESCAPE_BEGIN
        # Switch to julia parser
        contents = ps.srcfile.contents
        # TODO: Could we have a base interface that doesn't require a copy?
        thispos = ps.nnt.startbyte+1
        (je, newpos) = Base.Meta.parse(String(isa(contents, IOBuffer) ? copy(contents.data) : contents), thispos-1; raise=false, greedy=false)
        Base.seek(ps.lexer, newpos-2)
        ps.nnt = Token(JULIA_ESCAPE, Int64(ps.nnt.startbyte), newpos-3)
        ps.nnpos += newpos - thispos - 1
        ps.tok_storage = next_token(ps.lexer)
        open = take(ps, JULIA_ESCAPE_BEGIN)
        body = take_julia_escape_body(ps)
        close = accept(ps, RPAREN)
        return EXPR(JuliaEscape(open, body, close))
    end
    return error!(ps, UnexpectedToken, "expression")
end

function parse_comma_list(T, parse_item, ps)
    ElementType = T.parameters[1]
    @trysetup ElementType
    list = T()
    comma = nothing
    while true
        @trynext ref = parse_item(ps)
        push!(list, EXPR(ElementType(comma, ref)))
        kind(nt(ps)) == COMMA || break
        @trynext comma = take(ps, COMMA)
    end
    return list
end

function parse_function_call(ps, name)
    @trysetup FunctionCall name
    @trynext lparen = accept(ps, LPAREN)
    args = EXPRList{FunctionArgs{EXPR}}()
    if kind(nt(ps)) != RPAREN
        @trynext args = parse_comma_list(typeof(args), parse_expression, ps)
    end
    return EXPR(FunctionCall(name, lparen, args, @trynext accept(ps, RPAREN)))
end

function parse_array(ps)
    @trysetup Square
    @trynext lsquare = accept(ps, LSQUARE)
    args = EXPRList{Any}()
    while kind(nt(ps)) != RSQUARE
        push!(args, @trynext parse_expression(ps))
    end
    @trynext rsquare = accept(ps, RSQUARE)
    return EXPR(Square(lsquare, args, rsquare))
end

function parse_expression(ps)
    if kind(nt(ps)) == PRIME
        # Local trysetup for compound Prime expression
        @trysetup Prime
        @trynext lprime = take(ps, PRIME)
        @trynext expr = parse_expression(ps)
        @trynext rprime = take(ps, PRIME)
        return EXPR(Prime(lprime, expr, rprime))
    end
    ex = parse_primary_or_unary(ps)
    if is_operator(kind(nt(ps)))
        # caller captures
        @trysetup BinaryExpression ex
        @trynext op = take_operator(ps)
        @trynext ex = parse_binop(ps, ex, op)
    end
    if kind(nt(ps)) == CONDITIONAL
        # Local trysetup for compound ternary expression
        @trysetup TernaryExpr ex
        @trynext not = take(ps, CONDITIONAL)
        @trynext ifcase = parse_expression(ps)
        @trynext colon = accept(ps, COLON)
        @trynext elsecase = parse_expression(ps)
        ex = EXPR(TernaryExpr(ex, not, ifcase, colon, elsecase))
    end
    return ex
end

# this is a tricky one. Caller captures!!
function parse_binop(ps, ex, op, opterm = nothing)
    @trysetup BinaryExpression
    local rhs
    while true
        @trynext rhs = parse_primary_or_unary(ps)
        is_operator(kind(nt(ps))) || break
        ntprec = prec(kind(nt(ps)))
        if prec(op) >= ntprec
            # caller captured ex and op, we can captured rhs
            ex = EXPR(BinaryExpression(ex, op, rhs))
            (opterm !== nothing && prec(opterm) >= ntprec) && return ex
            @trynext op = take_operator(ps)
            continue
        else
            # we're the caller now, so we can capture ex, op and opterm
            @trynext rhs = parse_binop(ps, rhs, @trynext(take_operator(ps)), op)
            # ex is made up of captured elements
            # op is captured
            # rhs is captured, but if Error, does not contain ex and op
            ex = EXPR(BinaryExpression(ex, op, rhs))
            is_operator(kind(nt(ps))) || return ex
            @trynext op = take_operator(ps)
            continue
        end
    end
    ret = EXPR(BinaryExpression(ex, op, rhs))
    return ret
end

function parse_simulator(ps)
    @trysetup Simulator
    @trynext kw = take_kw(ps, SIMULATOR)
    @trynext langkw = take_kw(ps, LANG)
    @trynext eq = take(ps, EQ)
    if kind(nt(ps)) == SPECTRE
        ps.lang_swapped=true
    end
    @trynext lang = take_kw(ps, (SPECTRE, SPICE))
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(Simulator(kw, langkw, eq, lang, params, nl))
end


function parse_parameter_list(ps)
    parameters = EXPRList{Parameter}()
    return parse_parameter_list!(parameters, ps)
end

function parse_parameter_list!(parameters, ps)
    @trysetup Parameter
    while !eol(ps) && kind(nt(ps)) != RPAREN
        name = take_identifier(ps)
        @trynext p = parse_parameter(ps, name)
        push!(parameters, p)
    end
    return parameters
end


function parse_parameter_mod(ps)
    @trysetup DevMod
    # TODO: Lot
    @trynext kw = take_kw(ps, DEV)
    slash, distr = nothing, nothing
    if kind(nt(ps)) == SLASH
        @trynext slash = take(ps, SLASH)
        @trynext distr = take_identifier(ps)
    end
    @trynext eq = take(ps, EQ)
    @trynext val = parse_expression(ps)
    return EXPR(DevMod(kw, slash, distr, eq, val))
end

function parse_parameter(ps, name)
    @trysetup Parameter name
    eq = nothing
    val = nothing
    mod = nothing
    if kind(nt(ps)) == EQ # flags like savecurrents
        @trynext eq = accept(ps, EQ)
        @trynext val = parse_expression(ps)
    end
    if kind(nt(ps)) == DEV
        @trynext mod = parse_parameter_mod(ps)
    end
    return EXPR(Parameter(name, eq, val, mod))
end


function parse_param(ps, dot)
    @trysetup ParamStatement dot
    @trynext kw = take_kw(ps)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(ParamStatement(dot, kw, params, nl))
end

function parse_temp(ps, dot)
    @trysetup TempStatement dot
    @trynext kw = take_kw(ps)
    @trynext val = parse_expression(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(TempStatement(dot, kw, val, nl))
end

function parse_global(ps, dot)
    @trysetup GlobalStatement dot
    @trynext kw = take_kw(ps, GLOBAL)
    @trynext nodes = parse_hierarchial_node_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(GlobalStatement(dot, kw, nodes, nl))
end

function parse_csparam(ps, dot)
    @trysetup ParamStatement dot
    @trynext kw = take_kw(ps)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(ParamStatement(dot, kw, params, nl))
end


function parse_include(ps, dot)
    @trysetup IncludeStatement dot
    @trynext kw = take_kw(ps, INCLUDE)
    @trynext path = take_path(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(IncludeStatement(dot, kw, path, nl))
end

function parse_hdl(ps, dot)
    @trysetup HDLStatement dot
    @trynext kw = take_kw(ps, HDL)
    @trynext path = take_path(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(HDLStatement(dot, kw, path, nl))
end


function parse_end(ps, dot)
    @trysetup EndStatement dot
    @trynext endd = take_kw(ps, END)
    @trynext nl = accept_newline(ps)
    return EXPR(EndStatement(dot, endd, nl))
end

function unimplemented_instance_error(ps)
    # TODO: Emit more specific "unimplemented instance" error
    return error!(ps, UnexpectedToken)
end

# Parse an "instance" line
#
# The current next token must be an IDENTIFIER.
function parse_instance(ps)
    @case kind(nt(ps)) begin
        IDENTIFIER_BEHAVIORAL                 => parse_behavioral(ps)
        IDENTIFIER_CAPACITOR                  => parse_capacitor(ps)
        IDENTIFIER_DIODE                      => parse_diode(ps)
        IDENTIFIER_VOLTAGE_CONTROLLED_CURRENT => parse_controlled(ControlledSource{:V, :C}, ps)
        IDENTIFIER_VOLTAGE_CONTROLLED_VOLTAGE => parse_controlled(ControlledSource{:V, :V}, ps)
        IDENTIFIER_CURRENT_CONTROLLED_CURRENT => parse_controlled(ControlledSource{:C, :C}, ps)
        IDENTIFIER_CURRENT_CONTROLLED_VOLTAGE => parse_controlled(ControlledSource{:C, :V}, ps)
        IDENTIFIER_CURRENT                    => parse_current(ps)
        IDENTIFIER_JFET                       => unimplemented_instance_error(ps) # TODO
        IDENTIFIER_HFET_MESA                  => unimplemented_instance_error(ps) # TODO
        IDENTIFIER_LINEAR_MUTUAL_INDUCTOR     => unimplemented_instance_error(ps) # TODO
        IDENTIFIER_LINEAR_INDUCTOR            => parse_inductor(ps)
        IDENTIFIER_MOSFET                     => parse_mosfet(ps)
        IDENTIFIER_OSDI                       => parse_subckt_call(ps, OSDIDevice) # kinda a subckt call
        IDENTIFIER_PORT                       => unimplemented_instance_error(ps) # TODO
        IDENTIFIER_BIPOLAR_TRANSISTOR         => parse_bipolar_transistor(ps)
        IDENTIFIER_RESISTOR                   => parse_resistor(ps)
        IDENTIFIER_S_PARAMETER_ELEMENT        => parse_s_parameter_element(ps)
        IDENTIFIER_SWITCH                     => parse_switch(ps)
        IDENTIFIER_VOLTAGE                    => parse_voltage(ps)
        IDENTIFIER_TRANSMISSION_LINE          => unimplemented_instance_error(ps) # TODO
        IDENTIFIER_SUBCIRCUIT_CALL            => parse_subckt_call(ps)
        # TODO: This should be an "unexpected instance type" or "unexpected instance prefix" error
        IDENTIFIER_UNKNOWN_INSTANCE           => unimplemented_instance_error(ps)
        # TODO: .lib path/section, escaping backslash, unknown (non-keyword) identifier
        IDENTIFIER                            => unimplemented_instance_error(ps)
        _ => error!(ps, UnexpectedToken, "instance identifier")
    end
end

function parse_julia_device(ps, name, nodes...)
    @trysetup JuliaDevice
    ns = EXPRList{HierarchialNode}()
    for n in nodes
        push!(ns, n)
    end
    @trynext dev = parse_primary(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(JuliaDevice(name, ns, dev, nl))
end

function parse_inductor(ps)
    @trysetup Inductor
    @trynext name = parse_hierarchial_node(ps)
    @trynext pos = parse_hierarchial_node(ps)
    @trynext neg = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, pos, neg)
    val = kind(nnt(ps)) == EQ ? nothing : @trynext parse_expression(ps)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(Inductor(name, pos, neg, val, params, nl))
end

function parse_controlled(cs::Type{ControlledSource{in, out}}, ps) where {in, out}
    @trysetup cs
    @trynext name = parse_hierarchial_node(ps)
    @trynext pos = parse_hierarchial_node(ps)
    @trynext neg = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, pos, neg)

    # Check if this is a POLY expression
    if kind(nt(ps)) == POLY
        # Parse POLY expression: POLY(N) control_nodes... coefficients...
        # Note: parentheses are treated as token separators and not emitted as tokens
        @trynext poly_token = take_kw(ps, POLY)
        @trynext n_dims_token = take_literal(ps)  # Should be a number literal

        # Parse all remaining arguments into one big list
        # This includes: control_nodes + coefficients
        args = EXPRList{Any}()
        while !eol(ps)
            push!(args, @trynext parse_expression(ps))
        end

        # Create a special POLY expression structure
        poly_expr = EXPR(PolyControl(poly_token, n_dims_token, args))

        @trynext nl = accept_newline(ps)
        return EXPR(cs(name, pos, neg, poly_expr, nl))
    elseif kind(nt(ps)) == TABLE
        # Table controlled source parsing
        @trynext table_token = take_kw(ps, TABLE)
        @trynext expr = parse_expression(ps)
        eq = if kind(nt(ps)) == EQ
                 accept(ps, EQ)
        end
        # Parse all remaining number pairs into one big list
        args = EXPRList{Any}()
        while !eol(ps)
            push!(args, @trynext parse_expression(ps))
        end
        table_expr = EXPR(TableControl(table_token, expr, eq, args))
        @trynext nl = accept_newline(ps)
        return EXPR(cs(name, pos, neg, table_expr, nl))
    elseif in == :V
        # Standard voltage controlled source parsing
        cpos = kind(nnt(ps)) == EQ ? nothing : @trynext parse_hierarchial_node(ps)
        cneg = kind(nnt(ps)) == EQ ? nothing : @trynext parse_hierarchial_node(ps)
        val = kind(nnt(ps)) == EQ ? nothing : @trynext parse_expression(ps)
        @trynext params = parse_parameter_list(ps)
        expr = EXPR(VoltageControl(cpos, cneg, val, params))
        @trynext nl = accept_newline(ps)
        return EXPR(cs(name, pos, neg, expr, nl))
    elseif in == :C
        # Standard current controlled source parsing
        vnam = kind(nnt(ps)) == EQ ? nothing : @trynext parse_hierarchial_node(ps)
        val = kind(nnt(ps)) == EQ ? nothing : @trynext parse_expression(ps)
        @trynext params = parse_parameter_list(ps)
        expr = EXPR(CurrentControl(vnam, val, params))
        @trynext nl = accept_newline(ps)
        return EXPR(cs(name, pos, neg, expr, nl))
    end
end

function parse_behavioral(ps)
    @trysetup Behavioral
    @trynext name = parse_hierarchial_node(ps)
    @trynext pos = parse_hierarchial_node(ps)
    @trynext neg = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, pos, neg)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(Behavioral(name, pos, neg, params, nl))
end

parse_voltage(ps) = parse_voltage_or_current(ps, true)
parse_current(ps) = parse_voltage_or_current(ps, false)

function parse_tran_fn(ps)
    @trysetup TranSource
    @trynext kw = take_kw(ps)
    # TODO: Check divisible by two?
    vals = EXPRList{Any}()
    while !eol(ps) && !is_kw(kind(nt(ps)))
        @trynext ref = parse_expression(ps)
        push!(vals, ref)
    end
    return EXPR(TranSource(kw, vals))
end

function parse_voltage_or_current(ps, isvoltage)
    T = isvoltage ? Voltage : Current
    @trysetup T
    @trynext name = parse_hierarchial_node(ps)
    @trynext pos = parse_hierarchial_node(ps)
    @trynext neg = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, pos, neg)
    vals = EXPRList{Union{ACSource, DCSource, TranSource}}()
    while !eol(ps)
        if kind(nt(ps)) == DC
            @trynext dc = take_kw(ps, DC)
            eq = if kind(nt(ps)) == EQ
                @trynext take(ps, EQ)
            end
            @trynext expr = parse_expression(ps)
            push!(vals, EXPR(DCSource(dc, eq, expr)))
        elseif kind(nt(ps)) == AC
            @trynext ac = take_kw(ps, AC)
            eq = if kind(nt(ps)) == EQ
                @trynext take(ps, EQ)
            end
            @trynext expr = parse_expression(ps)
            # TODO Phase
            push!(vals, EXPR(ACSource(ac, eq, expr)))
        elseif is_source_type(kind(nt(ps)))
            @trynext fn = parse_tran_fn(ps)
            push!(vals, fn)
        else
            @trynext expr = parse_expression(ps)
            push!(vals, EXPR(DCSource(nothing, nothing, expr)))
        end
    end
    @trynext nl = accept_newline(ps)
    return EXPR(T(name, pos, neg, vals, nl))
end


function parse_bipolar_transistor(ps)
    @trysetup BipolarTransistor
    @trynext name = parse_hierarchial_node(ps)
    @trynext c = parse_hierarchial_node(ps)
    @trynext b = parse_hierarchial_node(ps)
    @trynext e = parse_hierarchial_node(ps)
    optional = EXPR{HierarchialNode}[]
    # [s] [t] model
    for _ in 1:3
        @trynext o = parse_hierarchial_node(ps)
        push!(optional, o)
        kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, c, b, e, optional...)
        is_ident(kind(nt(ps))) || break
        kind(nnt(ps)) == EQ && break
    end
    model = pop!(optional)
    s = get(optional, 1, nothing)
    t = get(optional, 2, nothing)
    if is_ident(kind(nt(ps))) && kind(nnt(ps)) == EQ
        @trynext params = parse_parameter_list(ps)
    else
        params = EXPRList{Parameter}()
    end
    @trynext nl = accept_newline(ps)
    return EXPR(BipolarTransistor(name, c, b, e, s, t, model, params, nl))
end

function parse_capacitor(ps)
    @trysetup Capacitor
    @trynext name = parse_hierarchial_node(ps)
    @trynext pos = parse_hierarchial_node(ps)
    @trynext neg = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, pos, neg)
    val = kind(nnt(ps)) == EQ ? nothing : @trynext parse_expression(ps)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(Capacitor(name, pos, neg, val, params, nl))
end


function parse_diode(ps)
    @trysetup Diode
    @trynext name = parse_hierarchial_node(ps)
    @trynext pos = parse_hierarchial_node(ps)
    @trynext neg = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, pos, neg)

    @trynext model = parse_hierarchial_node(ps)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(Diode(name, pos, neg, model, params, nl))
end

function parse_resistor(ps)
    @trysetup Resistor
    @trynext name = parse_hierarchial_node(ps)
    @trynext pos = parse_hierarchial_node(ps)
    @trynext neg = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, pos, neg)
    val = kind(nnt(ps)) == EQ ? nothing : @trynext parse_expression(ps)
    @trynext params = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(Resistor(name, pos, neg, val, params, nl))
end


function parse_mosfet(ps)
    @trysetup MOSFET
    @trynext name = parse_hierarchial_node(ps)
    @trynext d = parse_hierarchial_node(ps)
    @trynext g = parse_hierarchial_node(ps)
    @trynext s = parse_hierarchial_node(ps)
    @trynext b = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, d, g, s, b)
    @trynext model = parse_hierarchial_node(ps)
    @trynext parameters = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(MOSFET(name, d, g, s, b, model, parameters, nl))
end

function parse_subckt_call(ps, type=SubcktCall)
    @trysetup type
    @trynext name = parse_hierarchial_node(ps)
    nodes = EXPRList{HierarchialNode}()
    while kind(nnt(ps)) !== EQ && kind(nt(ps)) !== NEWLINE && kind(nt(ps)) !== JULIA_ESCAPE_BEGIN
        push!(nodes, @trynext parse_hierarchial_node(ps))
    end
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, nodes...)
    if isempty(nodes)
        return @trynext error!(ps, UnexpectedToken, HierarchialNode)
    end

    @trynext parameters = parse_parameter_list(ps)

    # Check if last parameter is actually the model name (no '=' sign)
    model = nothing
    model_after = nothing
    if !isempty(parameters)
        last_param = parameters[end]
        if last_param.eq === nothing && last_param.val === nothing
            # Last "parameter" is actually the model name appearing after parameters
            param_node = pop!(parameters)
            model_after = param_node.name  # Extract just the identifier
        else
            # Model appears before parameters (standard position)
            model = pop!(nodes)
        end
    else
        # No parameters, model must be in nodes
        model = pop!(nodes)
    end

    @trynext nl = accept_newline(ps)
    return EXPR(type(name, nodes, model, parameters, model_after, nl))
end

function parse_s_parameter_element(ps)
    @trysetup SParameterElement
    @trynext name = parse_hierarchial_node(ps)
    @trynext nd1 = parse_hierarchial_node(ps)
    @trynext nd2 = parse_hierarchial_node(ps)
    @trynext model = parse_hierarchial_node(ps)
    @trynext parameters = parse_parameter_list(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(SParameterElement(name, nd1, nd2, model, parameters, nl))
end

function parse_switch(ps)
    @trysetup Switch
    @trynext name = parse_hierarchial_node(ps)
    @trynext nd1 = parse_hierarchial_node(ps)
    @trynext nd2 = parse_hierarchial_node(ps)
    @trynext cnd1 = parse_hierarchial_node(ps)
    @trynext cnd2 = parse_hierarchial_node(ps)
    kind(nt(ps)) == JULIA_ESCAPE_BEGIN && return parse_julia_device(ps, name, nd1, nd2, cnd1, cnd2)
    @trynext model = parse_hierarchial_node(ps)
    @trynext onoff = take_kw(ps)
    @trynext nl = accept_newline(ps)
    return EXPR(Switch(name, nd1, nd2, cnd1, cnd2, model, onoff, nl))
end

eol(ps) = (t = nt(ps); kind(t) == NEWLINE || kind(t) == ENDMARKER)


function take_kw(ps)
    kwkind = kind(nt(ps))
    is_kw(kwkind) || return error!(ps, UnexpectedToken, "keyword")
    EXPR!(Keyword(kwkind), ps)
end

function take_kw(ps, tkind)
    !isa(tkind, Tuple) && (tkind = (tkind,))
    kwkind = kind(nt(ps))
    if !is_kw(kwkind) || !(kwkind in tkind)
        return error!(ps, UnexpectedToken, tkind)
    end
    EXPR!(Keyword(kwkind), ps)
end

function take_identifier(ps)
    if !(is_ident(kind(nt(ps))))
        return error!(ps, UnexpectedToken)
    end
    return EXPR!(Identifier(), ps)
end

function take_identifier_or_number(ps)
    # Accept either an identifier or a number token (for model names like "1N3064")
    if is_ident(kind(nt(ps)))
        return EXPR!(Identifier(), ps)
    elseif kind(nt(ps)) == NUMBER
        return EXPR!(NumberLiteral(), ps)
    else
        return error!(ps, UnexpectedToken)
    end
end

function take_literal(ps)
    ntkind = kind(nt(ps))
    if !is_literal(ntkind)
        return error!(ps, UnexpectedToken)
    end
    EXPR!(ntkind == NUMBER ? NumberLiteral() :
          Literal(), ps)
end

function maybe_take_unit(ps)
    # This function is now a no-op
end

function accept_identifier(ps)
    if kind(nt(ps)) == IDENTIFIER
        return take_identifier(ps)
    else
        return error!(ps, UnexpectedToken)
    end
end

function accept(ps, tkind)
    !isa(tkind, Tuple) && (tkind = (tkind,))
    if kind(nt(ps)) in tkind
        return EXPR!(Notation(), ps)
    else
        return error!(ps, UnexpectedToken, tkind)
    end
end

function accept_newline(ps)
    if kind(nt(ps)) in (NEWLINE, ENDMARKER)
        return EXPR!(Notation(), ps)
    else
        return error!(ps, UnexpectedToken, "newline")
    end
end

function accept_kw(ps, tkind)
    !isa(tkind, Tuple) && (tkind = (tkind,))
    if !all(is_kw, tkind)
        return error!(ps, UnexpectedToken)
    end
    kwkind = kind(nt(ps))
    if kwkind in tkind
        return EXPR!(Keyword(kwkind), ps)
    else
        return error!(ps, UnexpectedToken)
    end
end

function accept_kw(ps)
    kwkind = kind(nt(ps))
    if is_kw(kwkind)
        return EXPR!(Keyword(kwkind), ps)
    else
        return error!(ps, UnexpectedToken)
    end
end

function take_string(ps)
    if kind(nt(ps)) != STRING
        return error!(ps, UnexpectedToken, "STRING")
    end
    return EXPR!(StringLiteral(), ps)
end

function take_path(ps)
    # an unquoted path might be lexed as an identifier
    if !is_ident(kind(nt(ps))) && kind(nt(ps)) != STRING
        return error!(ps, UnexpectedToken, "STRING or IDENTIFIER")
    end
    return EXPR!(StringLiteral(), ps)
end

function take(ps, tkind)
    !isa(tkind, Tuple) && (tkind = (tkind,))
    if !(kind(nt(ps)) in tkind)
        return error!(ps, UnexpectedToken)
    end
    return EXPR!(Notation(), ps)
end

function take_operator(ps)
    ntkind = kind(nt(ps))
    if !is_operator(ntkind)
        return error!(ps, UnexpectedToken)
    end
    return EXPR!(Operator(ntkind), ps)
end

function take_builtin_func(ps)
    if !is_builtin_func(kind(nt(ps)))
        return error!(ps, UnexpectedToken)
    end
    return EXPR!(BuiltinFunc(), ps)
end

function take_builtin_const(ps)
    if !is_builtin_const(kind(nt(ps)))
        return error!(ps, UnexpectedToken)
    end
    return EXPR!(BuiltinConst(), ps)
end

function take_julia_escape_body(ps)
    if kind(nt(ps)) != JULIA_ESCAPE
        return error!(ps, UnexpectedToken)
    end
    return EXPR!(JuliaEscapeBody(), ps)
end

function error!(ps, errkind, expected=nothing, expand=true)
    ps.errored = true
    if !eol(ps) && expand
        expr = EXPR!(Error(errkind, expected, kind(nt(ps))), ps)
        return extend_to_line_end(expr, ps)
    else
        return EXPR!(Error(errkind, expected, kind(nt(ps))), ps)
    end
end

struct SPICEParserError <: Exception
    ps
    kind
    expected
end

function Base.showerror(io::IO, s::SPICEParserError)
    ps = s.ps
    nt = ps.nt
    lb = LineBreaking(UInt64(0), ps.srcfile.lineinfo, nothing)
    line, col1 = LineNumbers.indtransform(lb, nt.startbyte)
    _, col2 = LineNumbers.indtransform(lb, nt.endbyte)
    line_str = String(ps.srcfile.lineinfo[line])
    print(io, "SPICEParser error at line $(line):")
    if s.expected !== nothing
        print(io, "expected $(s.expected)")
    end
    println(io)
    println(io, "  ", chomp(line_str))
    print(io, "  ")
    for i in 1:length(line_str)
        if i >= col1 && i <= col2
            printstyled(io, "^"; color=:light_green)
        else
            print(io, " ")
        end
    end
end

@enum(PrecedenceLevels,
    PREC_LOGICAL,
    PREC_AND_AND,
    PREC_OR,
    PREC_XOR,
    PREC_AND,
    PREC_EQ,
    PREC_LT,
    PREC_SHIFT,
    PREC_PLUS,
    PREC_MUL,
    PREC_STAR_STAR)

function prec(opkind::Kind)
    if opkind in (LAZY_OR, EVENT_OR)
        return PREC_LOGICAL
    elseif opkind in (LAZY_AND,)
        return PREC_AND_AND
    elseif opkind in (OR,)
        return PREC_OR
    elseif opkind in (XOR, XOR_TILDE, TILDE_XOR)
        return PREC_XOR
    elseif opkind in (AND,)
        return PREC_AND
    elseif opkind in (EQ, EQEQ, NOT_EQ, EQEQEQ, NOT_EQEQ)
        return PREC_EQ
    elseif opkind in (LESS, GREATER, LESS_EQ, GREATER_EQ)
        return PREC_LT
    elseif opkind in (LBITSHIFT, RBITSHIFT, LBITSHIFT_A, RBITSHIFT_A)
        return PREC_SHIFT
    elseif opkind in (PLUS, MINUS)
        return PREC_PLUS
    elseif opkind in (STAR, SLASH, PERCENT)
        return PREC_MUL
    elseif opkind in (STAR_STAR,)
        return PREC_STAR_STAR
    else
        error("Unknown operator")
    end
end
prec(op::Operator) = prec(op.op)
prec(ex::EXPR{Operator}) = prec(ex.form)
