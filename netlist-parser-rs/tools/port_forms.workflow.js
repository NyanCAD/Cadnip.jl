export const meta = {
  name: 'port-spice-forms',
  description: 'Port remaining SPICE parser forms to Rust, each self-verified against the Julia parser in an isolated worktree',
  phases: [{ title: 'Port', detail: 'one worktree-isolated agent per form group' }],
}

// Each agent works in its OWN git worktree (isolation), ports one group of
// productions, verifies byte-exact against the Julia parser via the differential
// test, iterating to green, then returns the verified code pieces. The main
// agent re-assembles the returned pieces into the tree.

const JULIAENV =
  '/tmp/claude-1000/-home-pepijn-code-nyanodide-Cadnip-jl/905b4c9c-ed15-489c-b71b-e77609ca97c0/scratchpad/juliaenv'

const PORT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    status: { type: 'string', enum: ['verified', 'failed'] },
    functions: {
      type: 'string',
      description: 'Exact Rust source of the parse method(s) to append to the `impl Parser` productions block.',
    },
    instance_dispatch: {
      type: 'string',
      description: 'Match arm(s) to add to parse_instance (empty string if none).',
    },
    dot_dispatch: {
      type: 'string',
      description: 'Match arm(s) to add to parse_dot_body (empty string if none).',
    },
    corpus: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          name: { type: 'string', description: 'file name only, e.g. wf_switch_1.sp' },
          content: { type: 'string' },
        },
        required: ['name', 'content'],
      },
    },
    differential_entries: {
      type: 'array',
      items: { type: 'string', description: 'e.g. `switch_basic => "wf_switch_1",`' },
    },
    syntax_kinds_added: {
      type: 'string',
      description: 'Any SyntaxKind enum + dump_label additions you had to make (empty if none).',
    },
    notes: { type: 'string' },
  },
  required: ['status', 'functions', 'corpus', 'differential_entries', 'notes'],
}

const PREAMBLE = `You are porting ONE group of SPICE parser productions from Julia to Rust, in an ISOLATED git worktree of the Cadnip.jl repo (your cwd is the worktree root). This is a faithful CST port — the Rust tree must match the Julia parser's tree byte-for-byte.

FIRST, read these to learn conventions (paths relative to your cwd):
- netlist-parser-rs/crates/netlist-syntax/src/parser.rs — the parser. Study existing productions (parse_rcl, parse_diode, parse_v_or_i, parse_subckt, parse_model, parse_dc, parse_parameter_list, the expression parser). Note the helpers: wrapped(cp, SyntaxKind::X, |p|{...}) closes [cp..] as X on success / Incomplete on error (use ? for child errors); wrapped_on_err(cp, |p|{...}) wraps as Incomplete only on error (for inlined lists); take_kw(&[K]), take_kw_any(), take_identifier(), take_identifier_or_number(), take_literal(), take_string(), take_path(), take(&[K]) and accept(&[K]) (both emit Notation), accept_newline(), take_operator(), parse_expression(), parse_hierarchial_node(), parse_node(), parse_parameter_list(), self.error() (emits Error, returns Err), self.checkpoint(), self.nt.kind, self.nnt.kind, self.eol(). TokenKind::* is already imported (DC, AC, IC, LPAREN, RPAREN, EQ, COLON, STAR, NUMBER, ENDS, ENDDATA, ELSE, ELSEIF, ENDIF, ON, OFF, ...).
- netlist-parser-rs/crates/netlist-syntax/src/syntax_kind.rs — confirm the SyntaxKind variant names for your forms (most already exist). Only add a variant (to the enum AND dump_label) if genuinely missing; report it in syntax_kinds_added.
- The Julia source you are porting: NyanSpectreNetlistParser.jl/src/SPICE/parse/parse.jl (the parse_* functions) and .../forms.jl (the node STRUCTS — the CST child order MUST equal struct field order, with EXPRList fields inlined as direct children and Nothing/omitted fields producing no child).

CST RULES (critical): a production maps to wrapped(cp, SyntaxKind::<FormName>, ...) where <FormName> is the Julia struct name. Terminals: keyword -> Keyword (take_kw), punctuation/newline -> Notation (take/accept/accept_newline), identifier -> Identifier (take_identifier), number -> NumberLiteral (take_literal on NUMBER), string -> StringLiteral. Lists (EXPRList fields) are inlined (no wrapper node) — loop and emit elements directly; on error they get a list-level Incomplete via wrapped_on_err (see parse_parameter_list). Reproduce Julia's structure EXACTLY, including which sub-nodes wrap which tokens.

DISPATCH: instance devices are added as arms in fn parse_instance (match self.nt.kind on IDENTIFIER_* -> self.parse_xxx()). Dot-commands are arms in fn parse_dot_body (match self.nt.kind AFTER the dot was consumed; each production takes cp: Checkpoint and is called as self.parse_xxx(cp)).

VERIFY (do this, iterate until green):
1. Add your parse fn(s) to the productions impl in parser.rs and your dispatch arm(s).
2. Create 1-4 test netlists at netlist-parser-rs/tests/corpus/<name>.sp. EVERY netlist's FIRST LINE is the implicit title — start each with a comment/title line like "* t". Exercise the interesting shapes of your form.
3. Generate Julia ground truth for each:
   ~/.juliaup/bin/julia --project=${JULIAENV} NyanSpectreNetlistParser.jl/tools/dump_cst.jl netlist-parser-rs/tests/corpus/<name>.sp > netlist-parser-rs/tests/expected/<name>.txt
4. Add matching entries to netlist-parser-rs/crates/netlist-syntax/tests/differential.rs (inside the diff_tests! macro): \`<ident> => "<name-without-.sp>",\`
5. cd netlist-parser-rs && cargo test  — fix until YOUR new tests pass with NO regressions. The differential test compares your Rust dump to the Julia dump byte-for-byte; diffs pinpoint divergences.
   You can also eyeball: target/debug/dump_cst <file> vs the Julia dumper output.

If after real effort a specific shape cannot match Julia (rare; note it precisely), still return your best verified subset with status accordingly.

RETURN the pieces so the main agent can re-apply them to the main tree (your worktree is discarded): the exact Rust source of the fn(s) you added (functions), the dispatch arm text (instance_dispatch / dot_dispatch, empty if n/a), the corpus files you created (name + full content), the differential.rs entry lines (differential_entries), any SyntaxKind additions (syntax_kinds_added), and notes. Report status "verified" only if cargo test is green.
`

const GROUPS = [
  {
    label: 'controlled-sources',
    spec: `GROUP: controlled sources E/F/G/H. Julia: parse_controlled(cs::Type{ControlledSource{in,out}}, ps) and its sub-forms VoltageControl, CurrentControl, PolyControl, TableControl (all in parse.jl/forms.jl). Node kind for the device is ControlledSource (SyntaxKind::ControlledSource) for all of E/F/G/H.
The Julia type param {in,out}: in = control type. E=ControlledSource{:V,:V}, G=ControlledSource{:V,:C} -> in=:V (voltage-controlled); F=ControlledSource{:C,:C}, H=ControlledSource{:C,:V} -> in=:C (current-controlled). Since Rust has no type param, add a bool param (e.g. in_is_voltage) to your parse fn: E,G -> true; F,H -> false. It selects the VoltageControl vs CurrentControl branch (see parse_controlled). Also handle the POLY(...) and TABLE branches (PolyControl / TableControl). Skip the JULIA_ESCAPE_BEGIN branch (that token never occurs — julia escape is disabled).
Dispatch (instance) — Julia parse_instance mapping:
  IDENTIFIER_VOLTAGE_CONTROLLED_VOLTAGE => self.parse_controlled(true),   // E
  IDENTIFIER_VOLTAGE_CONTROLLED_CURRENT => self.parse_controlled(true),   // G
  IDENTIFIER_CURRENT_CONTROLLED_CURRENT => self.parse_controlled(false),  // F
  IDENTIFIER_CURRENT_CONTROLLED_VOLTAGE => self.parse_controlled(false),  // H
Test shapes: linear VCVS "E1 out 0 in 0 2", CCCS "F1 out 0 V1 3", a POLY form, and a value= param form.`,
  },
  {
    label: 'switch',
    spec: `GROUP: switch (S and W in ngspice). Julia: parse_switch (parse.jl), form Switch (forms.jl): name nd1 nd2 cnd1 cnd2 model onoff(keyword ON/OFF) nl. Dispatch (instance): IDENTIFIER_SWITCH => self.parse_switch(). Both S* and W* lex to IDENTIFIER_SWITCH in ngspice. Test shapes: "S1 out 0 ctl 0 swmod ON" and "W1 a b c d wmod OFF".`,
  },
  {
    label: 'ic',
    spec: `GROUP: .ic. Julia: parse_ic + parse_ic_statement (parse.jl); forms ICStatement, ICEntry, WildCard, Coloned. Dispatch (dot): IC => self.parse_ic(cp). Note the lexer already treats .ic as an implicit-expression dot-command. parse_ic loops ICEntry = name '(' arg ')' '=' expr where arg is parse_ic_statement (a NUMBER, an identifier, an identifier ':' identifier -> Coloned, a '*' -> WildCard, or NUMBER '*' -> WildCard). Test shapes: ".ic v(out)=1.5", ".ic v(a:b)=0", and a bare number/wildcard arg if valid.`,
  },
  {
    label: 'lib-endl',
    spec: `GROUP: .lib and .endl. Julia: parse_lib (LibStatement block form + LibInclude form) and parse_endl (EndlStatement). Dispatch (dot): LIB => self.parse_lib(cp), ENDL => self.parse_endl(cp). parse_lib distinguishes the two forms by whether the token after the name is an identifier (see the Julia code): section/block form ".lib name ... .endl" (LibStatement, which parses toplevel statements until an EndlStatement) vs include form ".lib path section" (LibInclude). The block form calls parse_spice_toplevel in a loop until it gets an EndlStatement — in Rust, loop calling self.parse_spice_toplevel() / detecting the .endl. Study how parse_subckt handles its body loop and .ends for the pattern. Test shapes: a ".lib mylib\\n R1 a b 1k\\n .endl\\n" block and a ".lib 'file.lib' section" include.`,
  },
  {
    label: 'data',
    spec: `GROUP: .data. Julia: parse_data (parse.jl), form DataStatement (forms.jl): dot kw blockname row_names(idents) values(numbers) nl dot2 ENDDATA nl2. It reads N column-name identifiers then rows of N numbers until end, then ".enddata". Dispatch (dot): DATA => self.parse_data(cp). Test shape: ".data tbl\\n a b\\n 1 2\\n 3 4\\n .enddata\\n".`,
  },
  {
    label: 'ifelse',
    spec: `GROUP: .if/.else/.elseif/.endif. Julia: parse_if, parse_ifelse_block, parse_condition (parse.jl); forms IfBlock, IfElseCase, Condition. Dispatch (dot): IF => self.parse_if(cp). parse_if builds an IfBlock of IfElseCase cases (each: dot kw condition? nl stmts) terminated by ".endif". Conditions are (expr). The cases contain nested toplevel statements (call self.parse_spice_toplevel() in the body loop, breaking on .else/.elseif/.endif — study parse_subckt's body loop). NOTE: the initial dot for the first case is already consumed (cp marks it); subsequent .else/.elseif/.endif dots are consumed inside the loop. Test shape: ".if (a>1)\\n R1 a b 1k\\n .else\\n R1 a b 2k\\n .endif\\n".`,
  },
  {
    label: 'measure',
    spec: `GROUP: .measure (.meas). Julia: parse_measure and its helpers parse_measure_point, parse_measure_range, parse_trig_targ, parse_risefallcross, parse_td, parse_measure_point (parse.jl); forms MeasurePointStatement, MeasureRangeStatement, When, At, RiseFallCross, TD_, FindDerivParam, AvgMaxMinPPRmsInteg, Val_, TrigTarg. Dispatch (dot): MEASURE => self.parse_measure(cp). This is the largest group — port the point form (FIND/DERIV/PARAM + WHEN/AT + RISE/FALL/CROSS + TD) and the range form (AVG/MAX/MIN/PP/RMS/INTEG + TRIG + TARG). SyntaxKind variants TD_ and Val_ have trailing underscores (they match the Julia struct names). Test shapes: ".meas tran res1 FIND v(out) AT=5m" and ".meas tran m2 TRIG v(a) VAL=1 TD=1n RISE=1 TARG v(b) VAL=2".`,
  },
]

phase('Port')

const results = await parallel(
  GROUPS.map((g) => async () => {
    const prompt = `${PREAMBLE}\n\n${g.spec}`
    let r = await agent(prompt, {
      label: `port:${g.label}`,
      phase: 'Port',
      isolation: 'worktree',
      model: 'claude-sonnet-5',
      schema: PORT_SCHEMA,
    })
    // Escalate a failure to a stronger model, once.
    if (!r || r.status !== 'verified') {
      log(`${g.label}: first pass ${r ? r.status : 'null'} — escalating to opus`)
      const r2 = await agent(
        `${prompt}\n\nA previous attempt did not fully verify. Notes from it: ${r ? r.notes : 'none'}. Get it to a fully verified (cargo test green) state.`,
        {
          label: `port:${g.label}:opus`,
          phase: 'Port',
          isolation: 'worktree',
          model: 'claude-opus-4-8',
          effort: 'high',
          schema: PORT_SCHEMA,
        },
      )
      if (r2) r = r2
    }
    return { group: g.label, result: r }
  }),
)

return results.filter((x) => x && x.result)
