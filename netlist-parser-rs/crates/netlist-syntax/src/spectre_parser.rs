//! Spectre parser — a faithful port of
//! `NyanSpectreNetlistParser.jl/src/parse/parse.jl` plus the
//! `get_next_action_token` token layer from `parserstate.jl`.
//!
//! Structurally this mirrors the SPICE `parser.rs` (rowan `GreenNodeBuilder`,
//! `bump`/`flush_trivia`, `wrapped()`/Incomplete, `error()`/`extend_to_line_end`)
//! over the SEPARATE Spectre token set (`spectre_syntax_kind`,
//! `spectre_lexer`). The rowan CST kind space (`syntax_kind::SyntaxKind`) is
//! SHARED, so node/terminal forms reuse the same variants (their `dump_label`
//! equals the Julia form struct name).
//!
//! Key Spectre-specific behaviours (differences from SPICE, discovered by
//! validating against the real Julia parser):
//! - Expression primaries are emitted as BARE terminals (no `LiteralExpr` /
//!   `NameRef` wrapper): a number → `NumberLiteral`, a string/other literal →
//!   `Literal`, a bare identifier → `Identifier`.
//! - `take_identifier` is LENIENT (parse.jl lines 638-643 have no `return`
//!   before `error!`): on a non-identifier it performs error recovery
//!   (`error!`, discarded) and then STILL consumes the *next* token as an
//!   `Identifier`. It therefore never fails. The consumed offending token(s)
//!   become `Skipped` trivia (the Julia `Error` EXPR is discarded, so its bytes
//!   vanish from the tree). This is what makes `model RESMOD` (no master) parse
//!   as a `Model` with just kw+name rather than an `Incomplete`.
//! - `accept_identifier` (used for the subckt name) is STRICT: it requires a
//!   plain `IDENTIFIER` and reports an error otherwise.
//! - No `+` line continuation (the token layer never folds `+`); continuation is
//!   a trailing backslash → `ESCD_NEWLINE`, skipped as trivia.
//! - `prec()` (from `spectre_syntax_kind`) panics on `TILDE_AND`/`TILD_OR` —
//!   only ever called when a following operator exists, matching Julia.

use crate::lexer::Dialect;
use crate::spectre_lexer::{Lexer, RawTok};
use crate::spectre_syntax_kind::TokenKind::*;
use crate::spectre_syntax_kind::{prec, TokenKind};
use crate::syntax_kind::{NetlistLang, SyntaxKind, SyntaxNode};
use rowan::{Checkpoint, GreenNodeBuilder, Language};

/// Which dialect the top-level source opens in. `.scs` files start in Spectre;
/// `.cir` files start in SPICE. Either may switch via `simulator lang=`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StartLang {
    Spectre,
    Spice,
}

/// `Ok(())` = success; `Err(())` = this subtree became an Error/Incomplete and
/// the failure should propagate (the caller wraps as `Incomplete`).
type PResult = Result<(), ()>;

#[derive(Clone, Copy)]
struct Sig {
    idx: usize,
    kind: TokenKind,
}

pub struct Parser<'a> {
    src: &'a str,
    raw: Vec<RawTok>,

    // get_next_action state (parserstate.jl)
    p: usize,
    started: bool,

    // significant-token lookahead
    nt: Sig,
    nnt: Sig,

    /// Index of the next raw token to emit into the builder.
    emit_idx: usize,

    builder: GreenNodeBuilder<'static>,
    errored: bool,

    /// Set by `parse_simulator` when a `simulator lang=spice` switches the
    /// active dialect to SPICE. The driver (`parse_toplevel`) then hands off to
    /// the SPICE parser. Mirrors `ParseState.lang_swapped`; like Julia it is
    /// never cleared (sticky) — after the final switch every remaining Spectre
    /// statement re-triggers a (possibly empty, and thus invisible) handoff.
    lang_swapped: bool,

    /// When set, all builder writes are suppressed (the cursor still advances).
    /// Used by `instance_tail_ok` to decide, without emitting, whether an
    /// instance's `master`/params/newline tail parses cleanly — which in Julia
    /// governs whether the `SNodeList` (a plain assignment, not a captured
    /// `@trynext` value) appears at all. See `parse_instance`.
    dry: bool,
}

fn to_raw(kind: SyntaxKind) -> rowan::SyntaxKind {
    NetlistLang::kind_to_raw(kind)
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        let raw = Lexer::tokenize(src, ERROR);
        let mut p = Parser {
            src,
            raw,
            p: 0,
            started: false,
            nt: Sig { idx: 0, kind: ERROR },
            nnt: Sig { idx: 0, kind: ERROR },
            emit_idx: 0,
            builder: GreenNodeBuilder::new(),
            errored: false,
            lang_swapped: false,
            dry: false,
        };
        p.nt = p.next_sig();
        p.nnt = p.next_sig();
        p
    }

    /// The next byte to be emitted into the builder (= end of the last emitted
    /// token). Hands a contiguous boundary to the SPICE parser at a switch.
    fn next_emit_byte(&self) -> u32 {
        if self.emit_idx < self.raw.len() {
            self.raw[self.emit_idx].start
        } else {
            self.src.len() as u32
        }
    }

    /// Resume the Spectre token cursor at byte `byte` after a SPICE region has
    /// emitted up to it. Mirrors `transition_from_spice!` + `reinit_at_pos!`:
    /// re-derive the lookahead from the raw token starting at `byte`, and reset
    /// `emit_idx` there so subsequent trivia flush contiguously.
    fn resync_at(&mut self, byte: u32) {
        let idx = self
            .raw
            .iter()
            .position(|t| t.start >= byte)
            .unwrap_or(self.raw.len() - 1);
        self.p = idx;
        self.emit_idx = idx;
        self.started = false;
        self.nt = self.next_sig();
        self.nnt = self.next_sig();
    }

    /// Hand off to the SPICE parser for a `simulator lang=spice` region: move
    /// the shared builder across, parse a `SPICENetlistSource` from `start_byte`
    /// until the dialect switches back, then resync the Spectre cursor.
    fn handoff_to_spice(&mut self, start_byte: u32) {
        let builder = std::mem::replace(&mut self.builder, GreenNodeBuilder::new());
        let (builder, stop, errored) =
            crate::parser::parse_spice_region(self.src, Dialect::Ngspice, builder, start_byte, true);
        self.builder = builder;
        self.errored |= errored;
        self.resync_at(stop);
    }

    // --- token layer (parserstate.jl) ---

    /// `get_next_token`: return the current raw index, advancing past it (but
    /// `ENDMARKER` stays put, matching Julia's guard).
    fn get_next_raw(&mut self) -> usize {
        let k = self.p;
        if self.raw[k].kind != ENDMARKER {
            self.p += 1;
        }
        k
    }

    fn storage_kind(&self) -> TokenKind {
        self.raw[self.p].kind
    }

    /// `get_next_action_token`: skip trivia, fold `+` line continuations, and
    /// decide which newlines are significant. Mirrors `parserstate.jl`
    /// `get_next_action_token` (lines 83-86): after a NEWLINE, a leading `+` on
    /// the next line continues the statement, so that newline is NOT significant.
    fn get_next_action(&mut self) -> usize {
        let mut idx = self.get_next_raw();
        while matches!(self.raw[idx].kind, WHITESPACE | COMMENT | ESCD_NEWLINE)
            || (self.raw[idx].kind == NEWLINE && !self.started)
        {
            idx = self.get_next_raw();
        }
        self.started = true;

        if self.raw[idx].kind == NEWLINE {
            while self.storage_kind() == WHITESPACE {
                self.get_next_raw();
            }
            if self.storage_kind() == PLUS {
                self.get_next_raw(); // eat the continuation `+`
                return self.get_next_action();
            }
            loop {
                let knt = self.storage_kind();
                if knt == WHITESPACE {
                    self.get_next_raw();
                } else if knt == NEWLINE || knt == COMMENT {
                    return self.get_next_action();
                } else {
                    return idx; // significant newline
                }
            }
        }
        idx
    }

    fn next_sig(&mut self) -> Sig {
        let idx = self.get_next_action();
        Sig { idx, kind: self.raw[idx].kind }
    }

    fn advance(&mut self) {
        self.nt = self.nnt;
        self.nnt = self.next_sig();
    }

    fn eol(&self) -> bool {
        self.nt.kind == NEWLINE || self.nt.kind == ENDMARKER
    }

    // --- builder emission ---

    fn trivia_kind(k: TokenKind) -> SyntaxKind {
        match k {
            WHITESPACE => SyntaxKind::Whitespace,
            COMMENT => SyntaxKind::Comment,
            NEWLINE => SyntaxKind::Newline,
            ESCD_NEWLINE => SyntaxKind::EscdNewline,
            PLUS => SyntaxKind::Continuation,
            _ => SyntaxKind::Skipped,
        }
    }

    fn flush_trivia(&mut self, up_to: usize) {
        while self.emit_idx < up_to {
            let t = self.raw[self.emit_idx];
            if t.end > t.start && !self.dry {
                let sk = Self::trivia_kind(t.kind);
                self.builder
                    .token(to_raw(sk), &self.src[t.start as usize..t.end as usize]);
            }
            self.emit_idx += 1;
        }
    }

    /// Consume `nt` as a leaf of the given form kind.
    fn bump(&mut self, kind: SyntaxKind) {
        self.flush_trivia(self.nt.idx);
        let t = self.raw[self.nt.idx];
        if t.end > t.start && !self.dry {
            self.builder
                .token(to_raw(kind), &self.src[t.start as usize..t.end as usize]);
        }
        self.emit_idx = self.nt.idx + 1;
        self.advance();
    }

    /// Consume `nt` as `Skipped` trivia (error recovery's line-eater, and the
    /// discarded offending tokens of the lenient `take_identifier`).
    fn bump_skipped(&mut self) {
        self.flush_trivia(self.nt.idx);
        let t = self.raw[self.nt.idx];
        if t.end > t.start && !self.dry {
            self.builder.token(
                to_raw(SyntaxKind::Skipped),
                &self.src[t.start as usize..t.end as usize],
            );
        }
        self.emit_idx = self.nt.idx + 1;
        self.advance();
    }

    fn checkpoint(&mut self) -> Checkpoint {
        // Flush pending leading trivia before marking, so a node's span starts
        // at its first real token (matches SPICE `parser.rs`).
        self.flush_trivia(self.nt.idx);
        self.builder.checkpoint()
    }

    fn wrap_at(&mut self, cp: Checkpoint, kind: SyntaxKind) {
        if self.dry {
            return;
        }
        self.builder.start_node_at(cp, to_raw(kind));
        self.builder.finish_node();
    }

    /// Close `[cp..]` as `ok_kind` on success or `Incomplete` on error.
    fn wrapped<F: FnOnce(&mut Self) -> PResult>(
        &mut self,
        cp: Checkpoint,
        ok_kind: SyntaxKind,
        body: F,
    ) -> PResult {
        let r = body(self);
        if !self.dry {
            let kind = if r.is_ok() { ok_kind } else { SyntaxKind::Incomplete };
            self.builder.start_node_at(cp, to_raw(kind));
            self.builder.finish_node();
        }
        r
    }

    /// Wrap `[cp..]` as `Incomplete` *only* on error (no node on success).
    /// Mirrors the Julia list helpers (`parse_parameter_list`, `parse_nodes`):
    /// a clean list is inlined, but a failure re-wraps the accumulated elements
    /// in an `Incomplete` at the list level.
    fn wrapped_on_err<F: FnOnce(&mut Self) -> PResult>(
        &mut self,
        cp: Checkpoint,
        body: F,
    ) -> PResult {
        let r = body(self);
        if r.is_err() && !self.dry {
            self.builder.start_node_at(cp, to_raw(SyntaxKind::Incomplete));
            self.builder.finish_node();
        }
        r
    }

    // --- error recovery (error! + extend_to_line_end) ---

    fn error(&mut self) -> PResult {
        self.errored = true;
        if !self.eol() {
            self.bump(SyntaxKind::Error); // offending token = Error content
            while !self.eol() {
                self.bump_skipped();
            }
            if self.nt.kind == NEWLINE {
                self.bump_skipped();
            } else {
                self.flush_trivia(self.nt.idx); // ENDMARKER: flush trailing trivia
            }
        } else if self.nt.kind == NEWLINE {
            self.bump(SyntaxKind::Error); // newline as (width-1) Error content
        } else {
            self.flush_trivia(self.nt.idx); // zero-width error at EOF: emit nothing
        }
        Err(())
    }

    /// The lenient-`take_identifier` error path: `error!` (discarded) followed
    /// by `EXPR!(Identifier)`. The `error!` tokens become `Skipped` trivia; then
    /// the next token is consumed as an `Identifier` (possibly `ENDMARKER`,
    /// zero-width → emits nothing). Never fails.
    fn recover_then_identifier(&mut self) {
        self.errored = true;
        if !self.eol() {
            self.bump_skipped(); // offending token (discarded Error content)
            while !self.eol() {
                self.bump_skipped();
            }
            if self.nt.kind == NEWLINE {
                self.bump_skipped();
            }
        } else if self.nt.kind == NEWLINE {
            self.bump_skipped(); // consume the newline (discarded Error)
        }
        // EXPR!(Identifier) consumes whatever is now current.
        self.bump(SyntaxKind::Identifier);
    }

    // --- take_* helpers (parse.jl) ---

    fn take_kw_any(&mut self) -> PResult {
        if self.nt.kind.is_kw() {
            self.bump(SyntaxKind::Keyword);
            Ok(())
        } else {
            self.error()
        }
    }

    fn take_kw(&mut self, kinds: &[TokenKind]) -> PResult {
        if self.nt.kind.is_kw() && kinds.contains(&self.nt.kind) {
            self.bump(SyntaxKind::Keyword);
            Ok(())
        } else {
            self.error()
        }
    }

    /// LENIENT identifier (parse.jl `take_identifier`): never fails.
    fn take_identifier(&mut self) -> PResult {
        if self.nt.kind.is_ident() {
            self.bump(SyntaxKind::Identifier);
        } else {
            self.recover_then_identifier();
        }
        Ok(())
    }

    /// STRICT identifier (parse.jl `accept_identifier`): requires a plain
    /// `IDENTIFIER` token; errors otherwise.
    fn accept_identifier(&mut self) -> PResult {
        if self.nt.kind == IDENTIFIER {
            self.take_identifier()
        } else {
            self.error()
        }
    }

    /// `take_node` (parse.jl): a `NUMBER` literal or an identifier.
    fn take_node(&mut self) -> PResult {
        if self.nt.kind == NUMBER {
            self.take_literal()
        } else if self.nt.kind.is_ident() {
            self.bump(SyntaxKind::Identifier);
            Ok(())
        } else {
            self.error()
        }
    }

    fn take_literal(&mut self) -> PResult {
        if !self.nt.kind.is_literal() {
            return self.error();
        }
        let sk = if self.nt.kind == NUMBER {
            SyntaxKind::NumberLiteral
        } else {
            SyntaxKind::Literal
        };
        self.bump(sk);
        Ok(())
    }

    fn take_string(&mut self) -> PResult {
        if self.nt.kind == STRING {
            self.bump(SyntaxKind::StringLiteral);
            Ok(())
        } else {
            self.error()
        }
    }

    /// `take(ps, tkind)` / `accept(ps, tkind)` — a `Notation` leaf.
    fn take(&mut self, kinds: &[TokenKind]) -> PResult {
        if kinds.contains(&self.nt.kind) {
            self.bump(SyntaxKind::Notation);
            Ok(())
        } else {
            self.error()
        }
    }

    fn accept(&mut self, kinds: &[TokenKind]) -> PResult {
        self.take(kinds)
    }

    fn accept_newline(&mut self) -> PResult {
        if self.nt.kind == NEWLINE {
            self.bump(SyntaxKind::Notation);
            Ok(())
        } else if self.nt.kind == ENDMARKER {
            self.flush_trivia(self.nt.idx); // zero-width nl at EOF
            Ok(())
        } else {
            self.error()
        }
    }

    fn take_operator(&mut self) -> PResult {
        if self.nt.kind.is_operator() {
            self.bump(SyntaxKind::Operator);
            Ok(())
        } else {
            self.error()
        }
    }

    fn take_builtin_const(&mut self) -> PResult {
        if self.nt.kind.is_builtin_const() {
            self.bump(SyntaxKind::BuiltinConst);
            Ok(())
        } else {
            self.error()
        }
    }
}

// --- grammar productions ---

impl<'a> Parser<'a> {
    fn parse_toplevel(&mut self, start_lang: StartLang) {
        self.builder
            .start_node(to_raw(SyntaxKind::SpectreNetlistSource));
        // A SPICE-start file (`.cir`) opens with a leading SPICE region parsed
        // from byte 0 (mirrors `parse(...; start_lang=:spice)`); control then
        // returns to the Spectre driver. This handoff does NOT set the Spectre
        // `lang_swapped` flag (the SPICE parser's own flag drove the return).
        if start_lang == StartLang::Spice {
            self.handoff_to_spice(0);
        }
        while self.nt.kind != ENDMARKER {
            let _ = self.parse_source();
            if self.lang_swapped {
                let byte = self.next_emit_byte();
                self.handoff_to_spice(byte);
            }
        }
        self.flush_trivia(self.raw.len()); // trailing trivia (ENDMARKER is zero-width)
        self.builder.finish_node();
    }

    /// `parse_spectrenetlist_source`.
    fn parse_source(&mut self) -> PResult {
        match self.nt.kind {
            SIMULATOR => self.parse_simulator(),
            MODEL => self.parse_model(),
            INCLUDE => self.parse_include(),
            AHDL_INCLUDE => self.parse_ahdl_include(),
            GLOBAL => self.parse_global(),
            PARAMETERS => self.parse_parameters(),
            INLINE => {
                let cp = self.checkpoint();
                self.parse_subckt(cp, true)
            }
            SUBCKT => {
                let cp = self.checkpoint();
                self.parse_subckt(cp, false)
            }
            SAVE => self.parse_save(),
            IC => self.parse_ic(),
            NODESET => self.parse_nodeset(),
            REAL => self.parse_function_decl(),
            IF => self.parse_conditional_block(),
            k if k.is_ident() => {
                let cp = self.checkpoint();
                self.bump(SyntaxKind::Identifier); // name
                self.parse_other(cp)
            }
            _ => self.error(),
        }
    }

    // --- simulator / language ---

    fn parse_simulator(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Simulator, |p| {
            p.take_kw(&[SIMULATOR])?;
            p.take_kw(&[LANG])?;
            p.take(&[EQ])?;
            // A switch to SPICE flags the driver to hand off after this
            // statement (mirrors `parse_simulator` setting `ps.lang_swapped`).
            if p.nt.kind == SPICE {
                p.lang_swapped = true;
            }
            p.take_kw(&[SPECTRE, SPICE])?;
            p.parse_parameter_list()?;
            p.accept_newline()
        })
    }

    // --- model ---

    fn parse_model(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Model, |p| {
            p.take_kw(&[MODEL])?;
            p.take_identifier()?; // name (lenient)
            p.take_identifier()?; // master (lenient)
            p.parse_parameter_list()?;
            p.accept_newline()
        })
    }

    // --- parameters + parameter list ---

    fn parse_parameters(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Parameters, |p| {
            p.take_kw(&[PARAMETERS])?;
            p.parse_parameter_list()?;
            p.accept_newline()
        })
    }

    fn parse_parameter_list(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped_on_err(cp, |p| {
            while !p.eol() {
                p.parse_parameter()?;
            }
            Ok(())
        })
    }

    fn parse_parameter(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Parameter, |p| {
            p.take_identifier()?; // name (lenient)
            p.accept(&[EQ])?;
            p.parse_expression()
        })
    }

    // --- subckt ---

    fn parse_subckt(&mut self, cp: Checkpoint, has_inline: bool) -> PResult {
        self.wrapped(cp, SyntaxKind::Subckt, |p| {
            if has_inline {
                p.take_kw(&[INLINE])?;
            }
            p.take_kw(&[SUBCKT])?;
            p.accept_identifier()?; // name (strict)
            if p.nt.kind == LPAREN {
                let snc = p.checkpoint();
                p.accept(&[LPAREN])?;
                p.parse_nodes()?;
                p.accept(&[RPAREN])?;
                p.wrap_at(snc, SyntaxKind::SubcktNodes);
            } else if !p.eol() {
                let snc = p.checkpoint();
                p.parse_nodes()?;
                p.wrap_at(snc, SyntaxKind::SubcktNodes);
            }
            p.accept_newline()?; // nl1
            while p.nt.kind != ENDS {
                p.parse_source()?; // body statement (@trynext: propagates)
            }
            p.take_kw(&[ENDS])?;
            if p.nt.kind == IDENTIFIER {
                p.take_identifier()?; // end_name (lenient)
            }
            p.accept_newline() // nl2
        })
    }

    // --- nodes (SNode / SubcktNode) ---

    fn parse_nodes(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped_on_err(cp, |p| {
            while p.nt.kind != RPAREN && !p.eol() {
                p.parse_node()?;
            }
            Ok(())
        })
    }

    fn parse_node(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::SNode, |p| {
            let mut idcp = p.checkpoint();
            p.take_node()?; // first id
            while p.nt.kind == DOT {
                p.take(&[DOT])?;
                p.wrap_at(idcp, SyntaxKind::SubcktNode); // [id, dot]
                idcp = p.checkpoint();
                p.take_node()?; // next id (final one stays bare)
            }
            Ok(())
        })
    }

    // --- instance / analysis ---

    /// `parse_instance(name)` — the name has already been bumped at `cp`. The
    /// `(nodes)` head is `Instance`-`@trysetup`; on the analysis branch the
    /// whole thing becomes an unconditional `Analysis` instead.
    fn parse_instance(&mut self, cp: Checkpoint) -> PResult {
        let snlcp = self.checkpoint();
        let head = (|p: &mut Self| -> PResult {
            p.accept(&[LPAREN])?;
            p.parse_nodes()?;
            p.accept(&[RPAREN])
        })(self);
        if head.is_err() {
            // `nodelist` is never built (rparen not reached) — flat nodes.
            self.wrap_at(cp, SyntaxKind::Incomplete);
            return Err(());
        }
        if self.nt.kind.is_analysis() {
            // `parse_analysis(name, nodelist)` always succeeds → SNodeList present.
            self.wrap_at(snlcp, SyntaxKind::SNodeList);
            return self.parse_analysis_tail(cp);
        }
        // In Julia the `SNodeList` is a plain assignment, not a captured
        // `@trynext` value, so it only appears in the *success* `Instance` node;
        // on a tail failure the Incomplete{Instance} keeps flat nodes. Decide
        // via a dry run before committing the SNodeList wrapper.
        if self.instance_tail_ok() {
            self.wrap_at(snlcp, SyntaxKind::SNodeList);
            let _ = self.take_identifier(); // master (lenient)
            let _ = self.parse_parameter_list();
            let _ = self.accept_newline();
            self.wrap_at(cp, SyntaxKind::Instance);
            Ok(())
        } else {
            // Flat nodes (no SNodeList); the tail re-parses to the same error.
            let _ = self.take_identifier(); // master (lenient)
            let _ = self.parse_parameter_list();
            let _ = self.accept_newline();
            self.wrap_at(cp, SyntaxKind::Incomplete);
            Err(())
        }
    }

    /// Dry-run the instance tail (`master` params newline) to learn whether it
    /// parses cleanly, without emitting anything. Restores all cursor/emit
    /// state afterwards. `master` (lenient `take_identifier`) never fails, so
    /// this reflects whether the parameter list is well-formed.
    fn instance_tail_ok(&mut self) -> bool {
        let saved = self.save_state();
        self.dry = true;
        let _ = self.take_identifier();
        let r = self.parse_parameter_list().and_then(|_| self.accept_newline());
        self.restore_state(saved);
        r.is_ok()
    }

    fn save_state(&self) -> (usize, Sig, Sig, bool, usize, bool, bool) {
        (
            self.p,
            self.nt,
            self.nnt,
            self.started,
            self.emit_idx,
            self.errored,
            self.dry,
        )
    }

    fn restore_state(&mut self, s: (usize, Sig, Sig, bool, usize, bool, bool)) {
        self.p = s.0;
        self.nt = s.1;
        self.nnt = s.2;
        self.started = s.3;
        self.emit_idx = s.4;
        self.errored = s.5;
        self.dry = s.6;
    }

    /// Emit `nt`'s text as a stand-alone `Identifier` token WITHOUT advancing
    /// the cursor. Reproduces the Julia double-capture (`@trynext name` in
    /// `parse_if`/`parse_elseif`/`parse_else` plus `@trynext name` inside
    /// `parse_instance`): on the failure path the same name token is rendered
    /// twice, inflating the tree past the source by the name's width.
    fn emit_phantom_identifier(&mut self) {
        self.flush_trivia(self.nt.idx); // leading trivia before the phantom
        let t = self.raw[self.nt.idx];
        if t.end > t.start && !self.dry {
            self.builder
                .token(to_raw(SyntaxKind::Identifier), &self.src[t.start as usize..t.end as usize]);
        }
        // Deliberately do NOT advance the cursor or `emit_idx`: the *real*
        // `take_identifier` re-emits the same token next.
    }

    /// `parse_analysis` — `@trysetup Analysis`: a params/newline error propagates,
    /// closing the whole node as `Incomplete`. `cp` precedes the (already-emitted)
    /// name + optional nodelist.
    fn parse_analysis_tail(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::Analysis, |p| {
            p.take_kw_any()?;
            p.parse_parameter_list()?;
            p.accept_newline()
        })
    }

    // --- parse_other (named statements dispatched on the trailing keyword) ---

    fn parse_other(&mut self, cp: Checkpoint) -> PResult {
        if self.nt.kind.is_analysis() {
            return self.parse_analysis_tail(cp);
        }
        match self.nt.kind {
            LPAREN => self.parse_instance(cp),
            ALTERGROUP => self.parse_altergroup(cp),
            ALTER => self.parse_named_control(cp, ALTER, SyntaxKind::Alter),
            CHECK => self.parse_named_control(cp, CHECK, SyntaxKind::Check),
            CHECKLIMIT => self.parse_named_control(cp, CHECKLIMIT, SyntaxKind::CheckLimit),
            INFO => self.parse_named_control(cp, INFO, SyntaxKind::Info),
            OPTIONS => self.parse_named_control(cp, OPTIONS, SyntaxKind::Options),
            SET => self.parse_named_control(cp, SET, SyntaxKind::Set),
            SHELL => self.parse_named_control(cp, SHELL, SyntaxKind::Shell),
            PARAMTEST => self.parse_named_control(cp, PARAMTEST, SyntaxKind::ParamTest),
            _ => {
                let _ = self.error(); // error!(UnknownStatement)
                self.wrap_at(cp, SyntaxKind::Incomplete);
                Err(())
            }
        }
    }

    /// The `name kw param=val ...` control statements (`alter`, `check`,
    /// `checklimit`, `info`, `options`, `set`, `shell`, `paramtest`). No
    /// `@trysetup`: the form node is always produced.
    fn parse_named_control(&mut self, cp: Checkpoint, kw: TokenKind, form: SyntaxKind) -> PResult {
        self.wrapped(cp, form, |p| {
            p.take_kw(&[kw])?;
            p.parse_parameter_list()?;
            p.accept_newline()
        })
    }

    fn parse_altergroup(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::AlterGroup, |p| {
            p.take_kw(&[ALTERGROUP])?;
            p.accept(&[LBRACE])?;
            p.accept_newline()?;
            while p.nt.kind != RBRACE && p.nt.kind != ENDMARKER {
                p.parse_source()?;
            }
            p.accept(&[RBRACE])?;
            p.accept_newline()
        })
    }

    // --- control: save / ic / nodeset / global ---

    fn parse_save(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Save, |p| {
            p.take_kw_any()?;
            p.parse_save_list()?;
            p.accept_newline()
        })
    }

    /// `parse_save_list` — a list layer that re-wraps as `Incomplete` on a failing
    /// signal (mirrors `parse_parameter_list` / Julia's `@trysetup`).
    fn parse_save_list(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped_on_err(cp, |p| {
            while !p.eol() {
                p.parse_save_signal()?;
            }
            Ok(())
        })
    }

    fn parse_save_signal(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::SaveSignal, |p| {
            if p.nt.kind != COLON {
                p.parse_node()?; // signalname (SNode)
            }
            if p.nt.kind == COLON {
                let mcp = p.checkpoint();
                p.take(&[COLON])?;
                if p.nt.kind.is_save_kw() {
                    p.take_kw_any()?;
                } else if p.nt.kind.is_number() {
                    p.take_literal()?;
                } else if p.nt.kind.is_ident() {
                    p.take_identifier()?;
                } else {
                    // No valid modifier: the error propagates and the whole
                    // SaveSignal closes as Incomplete, with no SaveSignalModifier
                    // node (matching Julia's `@trynext mod = error!(...)`).
                    p.error()?;
                }
                p.wrap_at(mcp, SyntaxKind::SaveSignalModifier);
            }
            Ok(())
        })
    }

    fn parse_ic(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Ic, |p| {
            p.take_kw(&[IC])?;
            p.parse_ic_parameter_list()?;
            p.take(&[NEWLINE])
        })
    }

    fn parse_ic_parameter_list(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped_on_err(cp, |p| {
            while !p.eol() {
                p.parse_ic_parameter()?;
            }
            Ok(())
        })
    }

    fn parse_ic_parameter(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::ICParameter, |p| {
            p.parse_node()?;
            p.accept(&[EQ])?;
            p.parse_expression()
        })
    }

    fn parse_nodeset(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::NodeSet, |p| {
            p.take_kw(&[NODESET])?;
            p.parse_parameter_list()?;
            p.take(&[NEWLINE])
        })
    }

    fn parse_global(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Global, |p| {
            p.take_kw(&[GLOBAL])?;
            while !p.eol() {
                p.parse_node()?;
            }
            p.accept_newline()
        })
    }

    // --- include / ahdl_include ---

    fn parse_include(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Include, |p| {
            p.take_kw(&[INCLUDE])?;
            p.take_string()?;
            if p.nt.kind == SECTION {
                p.parse_include_section()?;
            }
            p.accept_newline()
        })
    }

    fn parse_include_section(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::IncludeSection, |p| {
            p.take_kw(&[SECTION])?;
            p.take(&[EQ])?;
            p.take_identifier() // lenient
        })
    }

    fn parse_ahdl_include(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::AHDLInclude, |p| {
            p.take_kw(&[AHDL_INCLUDE])?;
            p.take_string()?;
            p.accept_newline()
        })
    }

    // --- function declaration ---

    fn parse_function_decl(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::FunctionDecl, |p| {
            p.take_kw(&[REAL])?; // rtype
            p.take_identifier()?; // id (lenient)
            p.take(&[LPAREN])?;
            if p.nt.kind != RPAREN {
                p.parse_comma_list(Self::parse_function_decl_arg)?;
            }
            p.take(&[RPAREN])?;
            p.take(&[LBRACE])?;
            p.take(&[NEWLINE])?; // nl1
            p.take_kw(&[RETURN])?;
            p.parse_expression()?; // body
            p.take(&[SEMICOLON])?;
            p.take(&[NEWLINE])?; // nl2
            p.take(&[RBRACE])?;
            p.take(&[NEWLINE]) // nl3
        })
    }

    fn parse_function_decl_arg(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::FunctionDeclArg, |p| {
            p.take_kw(&[REAL])?; // typ
            p.take_identifier() // id (lenient)
        })
    }

    // --- conditional block (if / else if / else) ---

    fn parse_conditional_block(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::ConditionalBlock, |p| {
            p.parse_if()?;
            while p.nt.kind == ELSE {
                let ecp = p.checkpoint();
                p.bump(SyntaxKind::Keyword); // take_kw(ELSE) (not @trynext)
                if p.nt.kind == IF {
                    p.parse_elseif(ecp)?;
                } else {
                    p.parse_else(ecp)?;
                    break;
                }
            }
            p.accept_newline()
        })
    }

    fn parse_if(&mut self) -> PResult {
        let cp = self.checkpoint();
        let head = (|p: &mut Self| -> PResult {
            p.take_kw(&[IF])?;
            p.accept(&[LPAREN])?;
            p.parse_expression()?;
            p.accept(&[RPAREN])?;
            p.accept(&[LBRACE])?;
            p.accept_newline() // nl1
        })(self);
        self.finish_conditional(cp, head, SyntaxKind::If)
    }

    /// `else if (...) {...}` — the leading `else` keyword has already been
    /// bumped at `cp`.
    fn parse_elseif(&mut self, cp: Checkpoint) -> PResult {
        let head = (|p: &mut Self| -> PResult {
            p.take_kw(&[IF])?; // kw2
            p.accept(&[LPAREN])?;
            p.parse_expression()?;
            p.accept(&[RPAREN])?;
            p.accept(&[LBRACE])?;
            p.accept_newline()
        })(self);
        self.finish_conditional(cp, head, SyntaxKind::ElseIf)
    }

    /// `else {...}` — the leading `else` keyword has already been bumped at `cp`.
    fn parse_else(&mut self, cp: Checkpoint) -> PResult {
        let head = (|p: &mut Self| -> PResult {
            p.accept(&[LBRACE])?;
            p.accept_newline()
        })(self);
        self.finish_conditional(cp, head, SyntaxKind::Else)
    }

    /// Shared close for `if`/`else if`/`else`: if the head already failed, wrap
    /// `Incomplete`; otherwise parse the `name instance rbrace` body (with the
    /// Julia double-capture quirk on failure) and wrap the form kind.
    fn finish_conditional(&mut self, cp: Checkpoint, head: PResult, form: SyntaxKind) -> PResult {
        if head.is_err() {
            self.wrap_at(cp, SyntaxKind::Incomplete);
            return Err(());
        }
        let tail = self.parse_conditional_body();
        let kind = if tail.is_ok() { form } else { SyntaxKind::Incomplete };
        self.wrap_at(cp, kind);
        tail
    }

    /// The body of a conditional clause: `name = take_identifier;
    /// parse_instance(name); rbrace`. On the failure path Julia's double-capture
    /// duplicates the name (see `emit_phantom_identifier`), so we decide via a
    /// dry run and prepend the phantom name only when the body fails.
    fn parse_conditional_body(&mut self) -> PResult {
        if self.conditional_body_ok() {
            let icp = self.checkpoint();
            let _ = self.take_identifier();
            let _ = self.parse_instance(icp);
            self.accept(&[RBRACE])
        } else {
            self.emit_phantom_identifier(); // captured `name` (double capture)
            let icp = self.checkpoint();
            let _ = self.take_identifier();
            let _ = self.parse_instance(icp);
            let _ = self.accept(&[RBRACE]);
            Err(())
        }
    }

    /// Dry-run `name instance rbrace` to learn whether the clause body parses
    /// cleanly (governs the double-capture rendering).
    fn conditional_body_ok(&mut self) -> bool {
        let saved = self.save_state();
        self.dry = true;
        let _ = self.take_identifier();
        let icp = self.checkpoint();
        let r = self.parse_instance(icp).and_then(|_| self.accept(&[RBRACE]));
        self.restore_state(saved);
        r.is_ok()
    }

    // --- expressions (precedence climbing) ---

    fn parse_expression(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.parse_primary_or_unary()?;
        if self.nt.kind.is_operator() {
            let op = self.nt.kind;
            self.bump(SyntaxKind::Operator);
            if self.parse_binop(cp, op, None).is_err() {
                self.wrap_at(cp, SyntaxKind::Incomplete);
                return Err(());
            }
        } else if self.nt.kind == CONDITIONAL {
            return self.wrapped(cp, SyntaxKind::TernaryExpr, |p| {
                p.take(&[CONDITIONAL])?;
                p.parse_expression()?; // ifcase
                p.accept(&[COLON])?;
                p.parse_expression() // elsecase
            });
        }
        Ok(())
    }

    fn parse_primary_or_unary(&mut self) -> PResult {
        if self.nt.kind.is_unary_operator() {
            self.parse_unary_op()
        } else {
            self.parse_primary()
        }
    }

    fn parse_unary_op(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::UnaryOp, |p| {
            p.take_operator()?;
            p.parse_primary()
        })
    }

    /// Left operand at `[cp_ex..]`; `op` (already emitted) is the pending
    /// operator; `opterm` is the caller's operator for the precedence guard.
    /// Direct port of `parse_binop` (parse.jl); identical to SPICE.
    fn parse_binop(
        &mut self,
        cp_ex: Checkpoint,
        mut op: TokenKind,
        opterm: Option<TokenKind>,
    ) -> PResult {
        loop {
            let cp_rhs = self.checkpoint();
            if self.parse_primary_or_unary().is_err() {
                self.wrap_at(cp_rhs, SyntaxKind::Incomplete);
                return Err(());
            }
            if !self.nt.kind.is_operator() {
                self.wrap_at(cp_ex, SyntaxKind::BinaryExpression);
                return Ok(());
            }
            let ntprec = prec(self.nt.kind);
            if prec(op) >= ntprec {
                self.wrap_at(cp_ex, SyntaxKind::BinaryExpression);
                if let Some(t) = opterm {
                    if prec(t) >= ntprec {
                        return Ok(());
                    }
                }
                op = self.nt.kind;
                self.bump(SyntaxKind::Operator);
            } else {
                let next_op = self.nt.kind;
                self.bump(SyntaxKind::Operator);
                if self.parse_binop(cp_rhs, next_op, Some(op)).is_err() {
                    self.wrap_at(cp_rhs, SyntaxKind::Incomplete);
                    return Err(());
                }
                self.wrap_at(cp_ex, SyntaxKind::BinaryExpression);
                if !self.nt.kind.is_operator() {
                    return Ok(());
                }
                op = self.nt.kind;
                self.bump(SyntaxKind::Operator);
            }
        }
    }

    fn parse_primary(&mut self) -> PResult {
        let k = self.nt.kind;
        // number / literal (string counts as `Literal`): bare terminal.
        if k.is_number() || k.is_literal() {
            return self.take_literal();
        }
        if k.is_builtin_const() {
            return self.take_builtin_const();
        }
        if k.is_ident() || k.is_builtin_func() {
            let cp = self.checkpoint();
            if k.is_builtin_func() {
                self.bump(SyntaxKind::BuiltinFunc);
            } else {
                self.bump(SyntaxKind::Identifier);
            }
            if self.nt.kind == LPAREN {
                return self.parse_function_call(cp);
            }
            return Ok(()); // bare id / builtin func (no wrapper)
        }
        if k == STRING {
            return self.take_string(); // dead (is_literal catches STRING first)
        }
        if k == LSQUARE {
            return self.parse_array();
        }
        if k == LPAREN {
            return self.parse_paren();
        }
        self.error()
    }

    fn parse_function_call(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::FunctionCall, |p| {
            p.accept(&[LPAREN])?;
            if p.nt.kind != RPAREN {
                p.parse_comma_list(Self::parse_expression)?;
            }
            p.accept(&[RPAREN])
        })
    }

    /// `parse_comma_list`: a run of `FunctionArgs` nodes. The first wraps just
    /// its item; each subsequent one wraps the preceding comma plus its item.
    fn parse_comma_list<F: Fn(&mut Self) -> PResult>(&mut self, item: F) -> PResult {
        let a0 = self.checkpoint();
        self.wrapped(a0, SyntaxKind::FunctionArgs, |p| item(p))?;
        while self.nt.kind == COMMA {
            let a = self.checkpoint();
            self.wrapped(a, SyntaxKind::FunctionArgs, |p| {
                p.take(&[COMMA])?;
                item(p)
            })?;
        }
        Ok(())
    }

    fn parse_paren(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Parens, |p| {
            p.take(&[LPAREN])?;
            p.parse_expression()?;
            p.accept(&[RPAREN])
        })
    }

    fn parse_array(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::SpectreArray, |p| {
            p.take(&[LSQUARE])?;
            while p.nt.kind != RSQUARE {
                if p.nt.kind == LPAREN {
                    p.parse_paren()?;
                } else if p.nt.kind.is_unary_operator() {
                    p.parse_unary_op()?;
                } else if p.nt.kind.is_ident() {
                    p.take_identifier()?; // lenient
                } else if p.nt.kind.is_literal() {
                    p.take_literal()?;
                } else {
                    return p.error();
                }
            }
            p.take(&[RSQUARE])
        })
    }
}

/// Parse Spectre source into a lossless rowan CST rooted at
/// `SpectreNetlistSource`.
pub fn parse(src: &str) -> SyntaxNode {
    parse_with(src, StartLang::Spectre)
}

/// Parse a netlist that may switch dialects via `simulator lang=`, starting in
/// `start_lang`. The root is always `SpectreNetlistSource`; SPICE regions nest
/// as `SPICENetlistSource` subtrees. Mirrors `SpectreNetlistCSTParser.parse`.
pub fn parse_with(src: &str, start_lang: StartLang) -> SyntaxNode {
    let mut p = Parser::new(src);
    p.parse_toplevel(start_lang);
    SyntaxNode::new_root(p.builder.finish())
}
