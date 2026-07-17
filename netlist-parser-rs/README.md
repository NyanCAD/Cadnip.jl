# netlist-parser-rs

A standalone Rust port of the SPICE/Spectre netlist parser that lives inside
Cadnip.jl (`NyanLexers.jl` + `NyanSpectreNetlistParser.jl`), so C++/Python/other
consumers can parse netlists without a Julia runtime.

**Status: spike.** One dialect (SPICE), CST-only (no semantic layer). The spike
validates that a `rowan`-based lossless-CST design reproduces the Julia parser's
tree *exactly*, and that the layered bindings are ergonomic — before investing in
the full grammar or a resolved-netlist semantic layer.

## Layout

```
crates/
  netlist-syntax/   core: SyntaxKind, lexer, keyword trie, parser, dump (rowan)
  netlist-cabi/     C ABI (cbindgen -> include/netlist_parser.h), cdylib+staticlib
  netlist-py/       PyO3 Python extension (maturin)
tests/
  corpus/           SPICE netlists within the spike grammar subset
  expected/         canonical CST dumps captured from the Julia parser (ground truth)
  regen_expected.sh regenerate expected/ from the Julia parser
```

## Design (Julia → Rust)

A textbook red-green lossless CST + hand-written recursive descent +
precedence-climbing expression parser — the model `rowan` implements.

| Julia | Rust |
|---|---|
| `EXPR{F}` green / `Node{T}` red tree (`EXPRS.jl`, `RedTree.jl`) | `rowan` green/red tree — not ported by hand |
| `Tokens.Kind` `@enum` (token_kinds.jl) | `TokenKind` (syntax_kind.rs), same order → `begin_*`/`end_*` range checks port as ordinal comparisons |
| ~148 `forms.jl` structs | one `SyntaxKind` enum; the dump label of each node/token = its Julia form struct name |
| `Tries.jl` keyword trie + prefix completion | `KeywordTrie` (keywords.rs), same unique-prefix completion (`.param`→`PARAMETERS`) |
| `NyanLexers` buffered IO + 3-char lookahead, `lexing_expression_stack` etc. | `Lexer` over a `Vec<char>` cursor (lexer.rs), context-sensitive state preserved verbatim |
| `ParseState` + `get_next_action_token` | parser.rs token layer: tokenize up front, classify significant vs. trivia, fold `+` continuations |
| `@trynext` / `Incomplete{T}` recovery | `wrapped()` closes each production as its form kind (ok) or `Incomplete` (err); `error()` = `error!` + `extend_to_line_end` |
| `prec(::Kind)` + `parse_binop` | `prec()` + `parse_binop` (precedence climbing), same associativity |

Trivia (whitespace, comments, non-significant newlines, `+` continuations) are
emitted as `rowan` tokens so the tree tiles the source, but carry trivia
`SyntaxKind`s and are omitted from the canonical dump — matching Julia, where
trivia is absorbed into node `off`/`fullwidth` and never a node.

## Spike grammar subset

Title (implicit + `.title`), `.end`, `.model`, `.param`/`.csparam`,
`.subckt`/`.ends`; R/C/L, V/I (with DC/AC/tran-function sources), X subckt calls
(incl. `model_after`); full expressions (precedence climbing, ternary, unary,
function calls, `{...}`/`'...'`/`(...)`); context-sensitive lexing; and error
recovery (malformed lines → `Incomplete`/`Error` nodes, rest of tree intact).

Out-of-subset dot-commands/devices intentionally produce error nodes; they are
mechanical breadth work (see below).

## Build & test

```bash
cargo test                              # unit + differential (vs captured Julia dumps)
cargo run -p netlist-syntax --bin dump_cst <file.sp>   # print the canonical CST dump
./crates/netlist-cabi/tests/run_c_smoke.sh             # C ABI smoke test

# Python (PyO3, in a venv with maturin):
cd crates/netlist-py && VIRTUAL_ENV=<venv> maturin develop && python -m pytest tests/
```

### Test layers & coverage

Tests exercise every stage of the pipeline:

- **Lexer** — the Julia `tokenize.jl` token table ported as a Rust unit test,
  plus operator/dialect/edge tests (all operators, dialect device letters,
  base specifiers, continuations, unterminated strings, julia-escape).
- **Parser/CST** — 57 differential tests byte-exact vs the Julia parser,
  including the parser package's own SPICE example netlists.
- **Reconstruct-source** — round-trip losslessness (`tree.text() == source`)
  over the whole corpus.
- **Error recovery** — a robustness suite of malformed inputs that must parse
  without panicking and stay lossless.
- **Real SPICE** — coverage netlists (`cov_*.sp`) are validated as runnable
  ngspice via `tools/validate_ngspice.sh`.
- **Xyce dialect** — `.step`/`.func`/`.global_param`/`.nodeset` and `Y`-devices
  (`Dialect::Xyce`) are supported beyond the Julia parser and validated against
  the Xyce simulator (`tools/validate_xyce.sh`, `-syntax`); `tests/xyce.rs`
  asserts no-error + lossless parse of `tests/xyce/full.cir`.

Measure coverage with [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov):

```bash
cargo llvm-cov --no-cfg-coverage -p netlist-syntax --ignore-filename-regex 'bin/' --summary-only
```

Current: ~90% regions / ~94% lines of `netlist-syntax`. The remainder is code
that is **not reachable from valid ngspice** and is exercised only at the lexer
level or is intentionally dead: Verilog-A/Spectre operators (`<<`/`>>`/`<+`/
`~&`/…) that the parser's `prec()` doesn't accept, non-ngspice dialect paths
(hspice/pspice/xyce), the disabled julia-escape feature, `dump_label` arms for
forms the SPICE parser never produces, and the `A`/`O`/`Z` device-letter arms
that are dead because `is_instance_first_char` omits them (faithful to Julia).
These close out as Spectre/Verilog-A and the other dialects are ported.

### Differential testing

The validation gate: the Rust CST dump must byte-match the Julia parser's dump.
Both sides emit a canonical preorder form — `<kind> <start>-<end>` per node,
content spans (trivia excluded), 0-based half-open byte offsets. `tests/expected/`
holds dumps captured from the Julia parser; `cargo test` checks the Rust output
against them (no Julia needed at test time). To refresh ground truth after a
Julia-side change, set up a Julia env and run `tests/regen_expected.sh` (see the
script header).

## Decision gate (spike findings)

- **Fidelity:** byte-exact across the corpus covering every hard mechanism
  (context-sensitive expression lexing, precedence climbing, subckt-call
  `model_after`, line continuations, `;`/`$` comments, brace/prime expressions,
  and deep error recovery — `Incomplete` nesting for errors inside expressions,
  parameter lists, binops and ternaries) plus a real production netlist
  (`RLC_test.cir`). An adversarial review ran ~60 further inputs through both
  parsers: the success path and all tested error cases are byte-identical.
- **One known divergence (malformed input only):** an unterminated *function-call
  argument list* (e.g. `f(a,` with nothing after the comma). Julia's
  `parse_comma_list` discards the per-arg `FunctionArgs` wrapping and emits one
  flat `Incomplete`; a forward-only rowan builder can't retro-unwrap finished
  nodes, so the Rust CST keeps the `FunctionArgs` wrapper and wraps the tail as
  `Incomplete`. Spans and leaf tokens are identical — only the wrapper shape
  differs, and only on this malformed case. Well-formed calls are byte-exact.
- **Effort per production:** low and formulaic. The load-bearing work was the
  lexer state machine, the token layer, and the error-recovery infra; each
  additional grammar production is a short `wrapped()` closure mirroring its
  Julia `parse_*` function 1:1.
- **Bindings:** both build and pass smoke tests. The C ABI (opaque handle + node
  walk + error API) builds as a `.so`/`.a` with a cbindgen-generated header and
  passes a C smoke test. The PyO3 extension (`netlist_parser`, abi3) builds with
  `maturin develop` and passes a pytest smoke test on Python 3.14. Effort per
  adapter was small over the frozen core — they compose without touching it.

### Next (post-spike, opt-in)

- **Breadth:** the remaining ~140 `forms.jl` device/dot-command node types
  (embarrassingly parallel, each gated by the differential test), then Spectre +
  the `simulator lang=spice` mid-file switch.
- **Depth:** a semantic layer (number/unit eval, name/scope resolution,
  `.include`, subckt/model/param binding) producing a *resolved* netlist — what
  most C++/Python consumers actually want. CST-only serves editors/linters/
  formatters.
