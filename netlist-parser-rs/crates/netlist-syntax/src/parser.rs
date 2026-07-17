//! Recursive-descent + precedence-climbing parser that emits a lossless rowan
//! CST. A faithful port of `SPICE/parse/{parse.jl,parserstate.jl}` for the spike
//! grammar subset (title/`.end`/`.model`/`.param`/`.subckt`, R/C/L, V/I, X, and
//! full expressions).
//!
//! Token layer: the SPICE lexer is standalone, so we tokenize the whole buffer
//! up front, then a `get_next_action`-style classifier (ported from
//! `parserstate.jl`) walks it, skipping trivia and folding `+` line
//! continuations. Significant tokens drive the recursive descent; trivia are
//! flushed into the builder around them so the tree tiles the source.
//!
//! Error recovery: `wrapped()` closes each production as its form kind on
//! success or `Incomplete` on the first child error (mirrors `@trynext` /
//! `Incomplete{T}`). `error()` emits an `Error` leaf spanning the offending
//! token and consumes the rest of the line as `Skipped` trivia
//! (`error!` + `extend_to_line_end`).

use crate::lexer::{Dialect, Lexer, RawTok};
use crate::syntax_kind::TokenKind::*;
use crate::syntax_kind::{prec, NetlistLang, SyntaxKind, SyntaxNode, TokenKind};
use rowan::{Checkpoint, GreenNodeBuilder, Language};

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

    // get_next_action state (parserstate.jl SigStream)
    p: usize,
    started: bool,

    // significant-token lookahead
    nt: Sig,
    nnt: Sig,

    /// Index of the next raw token to emit into the builder.
    emit_idx: usize,

    builder: GreenNodeBuilder<'static>,
    errored: bool,
}

fn to_raw(kind: SyntaxKind) -> rowan::SyntaxKind {
    NetlistLang::kind_to_raw(kind)
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, dialect: Dialect) -> Self {
        let raw = Lexer::tokenize(src, dialect, false, false);
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
        };
        p.nt = p.next_sig();
        p.nnt = p.next_sig();
        p
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

    /// `get_next_action_token`: skip trivia, fold `+` continuations, and decide
    /// which newlines are significant.
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
            if t.end > t.start {
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
        if t.end > t.start {
            self.builder
                .token(to_raw(kind), &self.src[t.start as usize..t.end as usize]);
        }
        self.emit_idx = self.nt.idx + 1;
        self.advance();
    }

    /// Consume `nt` as `Skipped` trivia (used by error recovery's line-eater).
    fn bump_skipped(&mut self) {
        self.flush_trivia(self.nt.idx);
        let t = self.raw[self.nt.idx];
        if t.end > t.start {
            self.builder
                .token(to_raw(SyntaxKind::Skipped), &self.src[t.start as usize..t.end as usize]);
        }
        self.emit_idx = self.nt.idx + 1;
        self.advance();
    }

    fn checkpoint(&self) -> Checkpoint {
        self.builder.checkpoint()
    }

    fn wrap_at(&mut self, cp: Checkpoint, kind: SyntaxKind) {
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
        let kind = if r.is_ok() { ok_kind } else { SyntaxKind::Incomplete };
        self.builder.start_node_at(cp, to_raw(kind));
        self.builder.finish_node();
        r
    }

    /// Wrap `[cp..]` as `Incomplete` *only* on error (no node on success). This
    /// mirrors the Julia list helpers (`parse_parameter_list!` etc.): a list is
    /// inlined when it parses cleanly, but a failure re-wraps the accumulated
    /// elements in an `Incomplete` at the list level — one extra layer on top of
    /// the failing element's own `Incomplete`.
    fn wrapped_on_err<F: FnOnce(&mut Self) -> PResult>(&mut self, cp: Checkpoint, body: F) -> PResult {
        let r = body(self);
        if r.is_err() {
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

    fn take_identifier(&mut self) -> PResult {
        if self.nt.kind.is_ident() {
            self.bump(SyntaxKind::Identifier);
            Ok(())
        } else {
            self.error()
        }
    }

    fn take_identifier_or_number(&mut self) -> PResult {
        if self.nt.kind.is_ident() {
            self.bump(SyntaxKind::Identifier);
            Ok(())
        } else if self.nt.kind == NUMBER {
            self.bump(SyntaxKind::NumberLiteral);
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

    /// For `.include`/`.lib` paths — outside the spike subset, kept for breadth.
    #[allow(dead_code)]
    fn take_path(&mut self) -> PResult {
        if self.nt.kind.is_ident() || self.nt.kind == STRING {
            self.bump(SyntaxKind::StringLiteral);
            Ok(())
        } else {
            self.error()
        }
    }

    /// `take(ps, tkind)` — a `Notation` leaf.
    fn take(&mut self, kinds: &[TokenKind]) -> PResult {
        if kinds.contains(&self.nt.kind) {
            self.bump(SyntaxKind::Notation);
            Ok(())
        } else {
            self.error()
        }
    }

    /// `accept(ps, tkind)` — like `take` but also `Notation`.
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
}

// --- grammar productions ---

impl<'a> Parser<'a> {
    fn parse_toplevel(&mut self) {
        self.builder.start_node(to_raw(SyntaxKind::SPICENetlistSource));
        while self.nt.kind != ENDMARKER {
            let _ = self.parse_spice_toplevel();
        }
        self.flush_trivia(self.raw.len()); // trailing trivia (ENDMARKER is zero-width)
        self.builder.finish_node();
    }

    fn parse_spice_toplevel(&mut self) -> PResult {
        match self.nt.kind {
            DOT => self.parse_dot(),
            TITLE_LINE => self.parse_title_implicit(),
            NEWLINE => self.error(),
            k if k.is_ident() => self.parse_instance(),
            _ => self.error(),
        }
    }

    fn parse_title_implicit(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Title, |p| {
            p.take(&[TITLE_LINE])?;
            p.accept_newline()
        })
    }

    fn parse_dot(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.bump(SyntaxKind::Notation); // dot
        self.parse_dot_body(cp)
    }

    /// Dispatch a dot-command whose `.` has already been consumed at `cp`.
    fn parse_dot_body(&mut self, cp: Checkpoint) -> PResult {
        match self.nt.kind {
            MODEL => self.parse_model(cp),
            PARAMETERS | CSPARAM => self.parse_param(cp),
            SUBCKT => self.parse_subckt(cp),
            END => self.parse_end(cp),
            TITLE => self.parse_title_dot(cp),
            _ => self.error(), // dot commands outside the spike subset
        }
    }

    fn parse_title_dot(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::Title, |p| {
            p.take_kw(&[TITLE])?;
            if p.nt.kind == TITLE_LINE {
                p.take(&[TITLE_LINE])?;
            }
            p.accept_newline()
        })
    }

    fn parse_end(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::EndStatement, |p| {
            p.take_kw(&[END])?;
            p.accept_newline()
        })
    }

    fn parse_param(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::ParamStatement, |p| {
            p.take_kw_any()?;
            p.parse_parameter_list()?;
            p.accept_newline()
        })
    }

    fn parse_model(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::Model, |p| {
            p.take_kw(&[MODEL])?;
            p.parse_hierarchial_node()?;
            p.take_identifier_or_number()?; // typ (may start with a digit)
            p.parse_parameter_list()?;
            p.accept_newline()
        })
    }

    fn parse_subckt(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::Subckt, |p| {
            p.take_kw(&[SUBCKT])?;
            p.take_identifier_or_number()?; // name
            while !p.eol() {
                if p.nnt.kind == EQ {
                    p.parse_parameter_list()?;
                } else {
                    p.parse_hierarchial_node()?;
                }
            }
            p.accept_newline()?; // nl1
            loop {
                if p.nt.kind == ENDMARKER {
                    return p.error(); // reached EOF before `.ends`
                }
                if p.nt.kind == DOT {
                    let cp2 = p.checkpoint();
                    p.bump(SyntaxKind::Notation); // dot2
                    if p.nt.kind == ENDS {
                        p.take_kw(&[ENDS])?;
                        if !p.eol() {
                            p.take_identifier_or_number()?; // name_end
                        }
                        return p.accept_newline(); // nl2
                    } else {
                        let _ = p.parse_dot_body(cp2); // @donext: body errors are contained
                    }
                } else {
                    let _ = p.parse_spice_toplevel();
                }
            }
        })
    }

    fn parse_parameter_list(&mut self) -> PResult {
        let cp_list = self.checkpoint();
        self.wrapped_on_err(cp_list, |p| {
            while !p.eol() && p.nt.kind != RPAREN {
                let cp = p.checkpoint();
                p.wrapped(cp, SyntaxKind::Parameter, |p| {
                    p.take_identifier()?; // name
                    if p.nt.kind == EQ {
                        p.accept(&[EQ])?;
                        p.parse_expression()?;
                    }
                    if p.nt.kind == DEV {
                        p.parse_parameter_mod()?;
                    }
                    Ok(())
                })?;
            }
            Ok(())
        })
    }

    fn parse_parameter_mod(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::DevMod, |p| {
            p.take_kw(&[DEV])?;
            if p.nt.kind == SLASH {
                p.take(&[SLASH])?;
                p.take_identifier()?;
            }
            p.take(&[EQ])?;
            p.parse_expression()
        })
    }

    fn parse_node(&mut self) -> PResult {
        if self.nt.kind == NUMBER || self.nt.kind.is_ident() {
            let cp = self.checkpoint();
            self.wrapped(cp, SyntaxKind::NodeName, |p| {
                if p.nt.kind == NUMBER {
                    p.take_literal()
                } else {
                    p.take_identifier()
                }
            })
        } else {
            self.error()
        }
    }

    fn parse_hierarchial_node(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::HierarchialNode, |p| {
            p.parse_node()?; // base
            p.parse_subnodes()
        })
    }

    fn parse_subnodes(&mut self) -> PResult {
        while self.nt.kind == DOT {
            let cp = self.checkpoint();
            self.wrapped(cp, SyntaxKind::SubNode, |p| {
                p.take(&[DOT])?;
                p.parse_node()
            })?;
        }
        Ok(())
    }

    fn parse_instance(&mut self) -> PResult {
        match self.nt.kind {
            IDENTIFIER_RESISTOR => self.parse_rcl(SyntaxKind::Resistor),
            IDENTIFIER_CAPACITOR => self.parse_rcl(SyntaxKind::Capacitor),
            IDENTIFIER_LINEAR_INDUCTOR => self.parse_rcl(SyntaxKind::Inductor),
            IDENTIFIER_VOLTAGE => self.parse_v_or_i(SyntaxKind::Voltage),
            IDENTIFIER_CURRENT => self.parse_v_or_i(SyntaxKind::Current),
            IDENTIFIER_SUBCIRCUIT_CALL => self.parse_subckt_call(),
            _ => self.error(), // other instance types outside the spike subset
        }
    }

    /// R / C / L: `name pos neg [value] params nl`.
    fn parse_rcl(&mut self, kind: SyntaxKind) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, kind, |p| {
            p.parse_hierarchial_node()?; // name
            p.parse_hierarchial_node()?; // pos
            p.parse_hierarchial_node()?; // neg
            if p.nnt.kind != EQ {
                p.parse_expression()?; // value (unless the next token is a param)
            }
            p.parse_parameter_list()?;
            p.accept_newline()
        })
    }

    /// V / I: `name pos neg (DC|AC|<tran fn>|<expr>)* nl`.
    fn parse_v_or_i(&mut self, kind: SyntaxKind) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, kind, |p| {
            p.parse_hierarchial_node()?; // name
            p.parse_hierarchial_node()?; // pos
            p.parse_hierarchial_node()?; // neg
            while !p.eol() {
                if p.nt.kind == DC {
                    let cpd = p.checkpoint();
                    p.wrapped(cpd, SyntaxKind::DCSource, |p| {
                        p.take_kw(&[DC])?;
                        if p.nt.kind == EQ {
                            p.take(&[EQ])?;
                        }
                        p.parse_expression()
                    })?;
                } else if p.nt.kind == AC {
                    let cpa = p.checkpoint();
                    p.wrapped(cpa, SyntaxKind::ACSource, |p| {
                        p.take_kw(&[AC])?;
                        if p.nt.kind == EQ {
                            p.take(&[EQ])?;
                        }
                        p.parse_expression()?; // acmag
                        if !p.eol() && !p.nt.kind.is_kw() {
                            p.parse_expression()?; // acphase
                        }
                        Ok(())
                    })?;
                } else if p.nt.kind.is_source_type() {
                    let cpt = p.checkpoint();
                    p.wrapped(cpt, SyntaxKind::TranSource, |p| {
                        p.take_kw_any()?;
                        while !p.eol() && !p.nt.kind.is_kw() {
                            p.parse_expression()?;
                        }
                        Ok(())
                    })?;
                } else {
                    // bare DCSource: `DCSource(nothing, nothing, expr)`
                    let cpe = p.checkpoint();
                    p.wrapped(cpe, SyntaxKind::DCSource, |p| p.parse_expression())?;
                }
            }
            p.accept_newline()
        })
    }

    /// X subckt call. The trailing bare word (a model name appearing after
    /// parameters) is emitted as a bare `Identifier`, matching Julia's
    /// `model_after` extraction; the "model before params" case needs no special
    /// handling since the popped node keeps its `HierarchialNode` kind/position.
    fn parse_subckt_call(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::SubcktCall, |p| {
            p.parse_hierarchial_node()?; // name
            let mut node_count = 0;
            while p.nnt.kind != EQ
                && p.nt.kind != NEWLINE
                && p.nt.kind != ENDMARKER
                && p.nt.kind != JULIA_ESCAPE_BEGIN
            {
                p.parse_hierarchial_node()?;
                node_count += 1;
            }
            if node_count == 0 {
                return p.error();
            }
            loop {
                if p.eol() || p.nt.kind == RPAREN {
                    break;
                }
                // A bare identifier immediately before end-of-line is the
                // model-after-params name → emit as a bare Identifier.
                if p.nt.kind.is_ident()
                    && (p.nnt.kind == NEWLINE || p.nnt.kind == ENDMARKER)
                {
                    p.bump(SyntaxKind::Identifier);
                    break;
                }
                let cpp = p.checkpoint();
                p.wrapped(cpp, SyntaxKind::Parameter, |p| {
                    p.take_identifier()?;
                    if p.nt.kind == EQ {
                        p.accept(&[EQ])?;
                        p.parse_expression()?;
                    }
                    if p.nt.kind == DEV {
                        p.parse_parameter_mod()?;
                    }
                    Ok(())
                })?;
            }
            p.accept_newline()
        })
    }

    // --- expressions (precedence climbing) ---

    fn parse_expression(&mut self) -> PResult {
        let cp = self.checkpoint();
        if self.nt.kind == PRIME {
            return self.wrapped(cp, SyntaxKind::Prime, |p| {
                p.take(&[PRIME])?;
                p.parse_expression()?;
                p.take(&[PRIME])
            });
        }
        self.parse_primary_or_unary()?;
        if self.nt.kind.is_operator() {
            let op = self.nt.kind;
            self.bump(SyntaxKind::Operator);
            if self.parse_binop(cp, op, None).is_err() {
                self.wrap_at(cp, SyntaxKind::Incomplete);
                return Err(());
            }
        }
        if self.nt.kind == CONDITIONAL {
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
            let cp = self.checkpoint();
            self.wrapped(cp, SyntaxKind::UnaryOp, |p| {
                p.take_operator()?;
                p.parse_primary()
            })
        } else {
            self.parse_primary()
        }
    }

    /// Left operand is at `[cp_ex..]`; `op` (already emitted) is the pending
    /// operator; `opterm` is the caller's operator for the precedence guard.
    /// A direct port of `parse_binop` (parse.jl).
    fn parse_binop(
        &mut self,
        cp_ex: Checkpoint,
        mut op: TokenKind,
        opterm: Option<TokenKind>,
    ) -> PResult {
        loop {
            let cp_rhs = self.checkpoint();
            // Julia's `parse_binop` `@trysetup BinaryExpression` (no captured
            // `ex`/`op`) wraps a failed rhs in its own `Incomplete` spanning just
            // the rhs region.
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
        if self.nt.kind.is_number() || self.nt.kind.is_literal() {
            return self.take_literal();
        }
        if self.nt.kind.is_ident() {
            let cp = self.checkpoint();
            self.bump(SyntaxKind::Identifier);
            if self.nt.kind == LPAREN {
                return self.parse_function_call(cp);
            } else if self.nt.kind == DOT {
                // `parse_hierarchial_node(ps, NodeName(id))`
                self.wrap_at(cp, SyntaxKind::NodeName);
                return self.wrapped(cp, SyntaxKind::HierarchialNode, |p| p.parse_subnodes());
            }
            return Ok(());
        }
        if self.nt.kind == STRING {
            return self.take_string();
        }
        if self.nt.kind == LSQUARE {
            return self.parse_array();
        }
        let cp = self.checkpoint();
        match self.nt.kind {
            LBRACE => self.wrapped(cp, SyntaxKind::Brace, |p| {
                p.take(&[LBRACE])?;
                p.parse_expression()?;
                p.accept(&[RBRACE])
            }),
            LPAREN => self.wrapped(cp, SyntaxKind::Parens, |p| {
                p.take(&[LPAREN])?;
                p.parse_expression()?;
                p.accept(&[RPAREN])
            }),
            PRIME => self.wrapped(cp, SyntaxKind::Prime, |p| {
                p.take(&[PRIME])?;
                p.parse_expression()?;
                p.accept(&[PRIME])
            }),
            _ => self.error(),
        }
    }

    // KNOWN divergence (malformed input only): Julia's `parse_comma_list`
    // accumulates *raw* args/commas and, on an arg error, discards the per-arg
    // `FunctionArgs` wrapping to emit one flat `Incomplete{FunctionArgs}`. A
    // forward-only rowan builder cannot retro-unwrap already-finished
    // `FunctionArgs` nodes, so on an unterminated arg list we retain them and
    // wrap the tail as `Incomplete`. Spans and leaves match; only the wrapper
    // shape differs. Well-formed calls are byte-identical.
    fn parse_function_call(&mut self, cp: Checkpoint) -> PResult {
        self.wrapped(cp, SyntaxKind::FunctionCall, |p| {
            p.accept(&[LPAREN])?;
            if p.nt.kind != RPAREN {
                let cpa = p.checkpoint();
                p.wrapped(cpa, SyntaxKind::FunctionArgs, |p| p.parse_expression())?;
                while p.nt.kind == COMMA {
                    let cpa = p.checkpoint();
                    p.wrapped(cpa, SyntaxKind::FunctionArgs, |p| {
                        p.take(&[COMMA])?;
                        p.parse_expression()
                    })?;
                }
            }
            p.accept(&[RPAREN])
        })
    }

    fn parse_array(&mut self) -> PResult {
        let cp = self.checkpoint();
        self.wrapped(cp, SyntaxKind::Square, |p| {
            p.accept(&[LSQUARE])?;
            while p.nt.kind != RSQUARE {
                if p.eol() {
                    return p.error();
                }
                p.parse_expression()?;
            }
            p.accept(&[RSQUARE])
        })
    }
}

/// Parse SPICE source into a lossless rowan CST rooted at `SPICENetlistSource`.
pub fn parse(src: &str, dialect: Dialect) -> SyntaxNode {
    let mut p = Parser::new(src, dialect);
    p.parse_toplevel();
    SyntaxNode::new_root(p.builder.finish())
}
