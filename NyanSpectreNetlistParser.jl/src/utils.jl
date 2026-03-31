using Base.Meta

const exprlistname = gensym("exprlist")
const exprtypename = gensym("exprtype")

"""
    @trysetup(type, init...)

Initializes error handling for a parser function. This macro must be called at the beginning
of any function that uses `@trynext` to ensure proper token capturing and error recovery.

# Token Capturing Rules

The fundamental principle is **each token must be captured exactly once** in the error recovery
mechanism. This prevents both token loss and double-counting
that would result in unrecoverable misalignment in the red-green tree.

## Arguments

- `type`: The AST node type being parsed (e.g., `Title`, `MeasurePointStatement`)
- `init...`: Optional pre-parsed tokens/values to capture immediately

## How Token Capturing Works

1. **EXPRList Creation**: Creates a list to accumulate all tokens/values for error recovery
2. **Type Storage**: Stores the target AST node type for error reporting
3. **Initial Capture**: Any `init` arguments are immediately captured via `@trynext`

## Key Patterns and Examples

### Pattern 1: Simple Function with Pre-parsed Arguments

```julia
function parse_dot(ps, dot)
    @trysetup AbstractASTNode dot # Capture the pre-parsed dot token
    # ... rest of parsing
end
```

The `dot` parameter was parsed by the caller and passed in. We must capture it here
because if parsing fails later, we need it for error recovery.

### Pattern 2: Deferred Capturing for Control Flow

```julia
function parse_if(ps, dot)
    @trysetup IfBlock # dot is captured in parse_ifelse_block
    cases = EXPRList{IfElseCase}()
    push!(cases, @trynext parse_ifelse_block(ps, dot, (IF,)))
    # ...
    while true
        dot = take(ps, DOT)  # New dot token
        tok = kind(nt(ps))
        if tok === ENDIF
            @trynext dot      # Capture the ENDIF dot
            @trynext kw = take_kw(ps, ENDIF)
            break
        end
        push!(cases, @trynext parse_ifelse_block(ps, dot))
    end
end
```

Here, the initial `dot` is NOT captured in `@trysetup` because it will be captured
inside `parse_ifelse_block`. This prevents double-counting while ensuring the token
is still captured exactly once.

### Pattern 3: Capturing vs Non-Capturing in Called Functions

```julia
function parse_parameter_list!(parameters, ps)
    @trysetup Parameter
    while !eol(ps) && kind(nt(ps)) != RPAREN
        name = take_identifier(ps)        # DON'T capture here
        @trynext p = parse_parameter(ps, name)  # parse_parameter captures it
        push!(parameters, p)
    end
    return parameters
end

function parse_parameter(ps, name)
    @trysetup Parameter name  # Capture the name passed from caller
    # ... continue parsing
end
```

The caller (`parse_parameter_list!`) doesn't capture `name` because the callee
(`parse_parameter`) will capture it. This ensures single capture.

### Pattern 4: Tail-Call Pattern with Shared Pre-parsed Arguments

```julia
function parse_measure(ps, dot)
    @trysetup MeasurePointStatement dot  # Capture in case of early failure
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

function parse_measure_range(ps, dot, kw, type, name)
    @trysetup MeasureRangeStatement dot kw type name  # Re-capture all pre-parsed
    # ... continue parsing
end
```

**Key Insight**: The pre-parsed arguments (`dot`, `kw`, `type`, `name`) are captured
TWICE - once in `parse_measure` and again in `parse_measure_range`. This isn't
double-counting because:

1. **Early Failure Protection**: If parsing fails in `parse_measure` (e.g., during
   `name` parsing), we need those tokens in `parse_measure`'s accumulator
2. **Tail Call**: `parse_measure_range` is called in tail position and its result
   is returned directly - no `@trynext` wrapper
3. **Accumulator Abandonment**: On success, `parse_measure`'s accumulator is
   abandoned and `parse_measure_range`'s result (complete or incomplete) is returned
4. **Error Preservation**: If `parse_measure_range` fails, its `Incomplete` node
   contains all the pre-parsed tokens needed for error recovery

This pattern ensures that no matter where parsing fails, all successfully parsed
tokens are preserved exactly once in the final error information.

## Critical Guidelines

### DO capture when:
- You have pre-parsed arguments that won't be captured elsewhere
- You parse a token and use it directly in the current function
- You need to capture intermediate results that might error

### DON'T capture when:
- A token will be captured by a called function
- You're passing a token to another parser function that will handle it
- The token is captured in a different control flow branch

### Error Recovery Behavior

When any `@trynext` operation fails (returns an Error or Incomplete EXPR):
1. All previously captured tokens/values are collected from `exprlistname`
2. An `Incomplete{type}` node is created containing:
   - The accumulated tokens (for partial parsing information)
   - The error that caused the failure
3. This allows error recovery to preserve all successfully parsed content

## Common Mistakes

1. **Double Capture**: Capturing a token both in caller and callee
   ```julia
   # WRONG
   @trynext name = take_identifier(ps)
   @trynext p = parse_parameter(ps, name)  # parse_parameter also captures name
   ```

2. **Missing Capture**: Not capturing tokens that aren't captured elsewhere
   ```julia
   # WRONG - if parse_expression fails, kw is lost
   kw = take_kw(ps, MEASURE)
   @trynext expr = parse_expression(ps)
   ```

3. **Wrong Setup Order**: Not calling @trysetup first
   ```julia
   # WRONG - @trynext used before @trysetup
   @trynext name = take_identifier(ps)
   @trysetup SomeType
   ```

The key insight is that token capturing is about error recovery: if parsing fails,
we want to preserve all the tokens we successfully parsed up to that point, but
we want each token to appear exactly once in the error information.
"""
macro trysetup(type, init...)
    esc(quote
        $exprlistname = EXPRList{Any}()
        $exprtypename = $type
        for __i in [$(init...)]
            @trynext __i
        end
    end)
end

"""
    @trynext assignment

Wraps a parsing operation for error handling and token capture. If the operation
succeeds, the result is captured in the error recovery list. If it fails (returns
Error or Incomplete), immediately returns an Incomplete node with all captured tokens.

# Usage Patterns

## Standard Assignment
```julia
@trynext kw = take_kw(ps, MEASURE)
@trynext name = parse_expression(ps)
```

## Direct Expression (creates anonymous variable)
```julia
push!(entries, @trynext parse_expression(ps))
return EXPR(SomeType(@trynext take_literal(ps)))
```

## Conditional Capture
```julia
val = kind(nnt(ps)) == EQ ? nothing : @trynext parse_expression(ps)
```

# When NOT to Use @trynext

Don't use `@trynext` for:
- Control flow tokens that will be captured elsewhere
- Tokens passed to functions that will capture them
- Simple assignments that don't involve parsing operations

# Error Recovery Mechanism

When a `@trynext` operation encounters an Error or Incomplete EXPR:
1. Stops execution immediately
2. Creates `Incomplete{type}(captured_tokens, error)`
3. Returns this to the caller for error propagation

This ensures that partial parsing information is preserved while maintaining
the single-capture invariant for all tokens.
"""
macro trynext(assignment)
    if !(assignment isa Expr) || assignment.head !== :(=)
        assignment = Expr(:(=), gensym(), assignment)
    end
    lhs = assignment.args[1]
    esc(quote
        $assignment
        if $lhs isa EXPR && ($lhs.form isa Error || $lhs.form isa Incomplete)
            return EXPR(Incomplete{$exprtypename}($exprlistname, $lhs))
        end
        push!($exprlistname, $lhs)
        $lhs
    end)
end

macro donext(assignment)
    if !(assignment isa Expr) || assignment.head !== :(=)
        assignment = Expr(:(=), gensym(), assignment)
    end
    lhs = assignment.args[1]
    esc(quote
        $assignment
        push!($exprlistname, $lhs)
        $lhs
    end)
end

function mkcond(cond)
    if isexpr(cond, :call) && cond.args[1] == :(|)
        return Expr(:(||),
            mkcond(cond.args[2]),
            mkcond(cond.args[3]))
    else
        return :(cond == $(esc(cond)))
    end
end

macro case(cond, cases)
    block = ret = Expr(:block, Expr(:(=), :cond, esc(cond)))
    done = false
    first = true
    for case in cases.args
        isa(case, LineNumberNode) && continue
        isa(case, Expr) || error("Expected Expr, got $(typeof(case))a")
        done && error("Extra statements after catchall")
        if isexpr(case, :call) && case.args[1] == :(=>)
            casecond = case.args[2]
            casestmt = case.args[3]
            if casecond == :(_)
                push!(block.args, esc(casestmt))
                done = true
                continue
            end
            stmt = Expr(first ? :if : :elseif,
                mkcond(casecond),
                esc(casestmt))
            push!(block.args, stmt)
            block = stmt
            first = false
        else
            error("Unknown stmt kind $(case)")
        end
    end
    ret
end
