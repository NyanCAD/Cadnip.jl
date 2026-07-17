//! SPICE lexer — a faithful port of `NyanLexers.jl/src/lexer.jl` +
//! `SPICE/tokenize/lexer.jl`.
//!
//! The Julia lexer runs over an `IO` with a 3-char lookahead; here we run over a
//! pre-collected `Vec<char>` with a cursor, which is equivalent (see the
//! position bookkeeping notes below) and simpler. Because the SPICE lexer takes
//! no feedback from the parser (Julia-escape and the Spectre `simulator lang=`
//! switch are the only exceptions, both out of spike scope), we can tokenize the
//! whole buffer up front.
//!
//! Byte-span model: `tok_start` is the token's first byte; `position()` is the
//! byte offset one past the last consumed char. `emit` produces `[tok_start,
//! position())` (half-open) and then re-anchors `tok_start = position()`. This
//! matches Julia's `Token(kind, token_startpos, position-1)` with an exclusive
//! end.

use crate::keywords::KeywordTrie;
use crate::syntax_kind::TokenKind;
use crate::syntax_kind::TokenKind::*;

/// Sentinel for "past end of input". `char::MAX` matches Julia's
/// `EOF_CHAR = typemax(Char)`: every character-class predicate returns false.
const EOF_CHAR: char = '\u{10FFFF}';

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    Ngspice,
    Hspice,
    Pspice,
    Xyce,
}

/// A raw lexer token: kind + half-open byte span `[start, end)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RawTok {
    pub kind: TokenKind,
    pub start: u32,
    pub end: u32,
}

// --- character classifiers (tokenize/lexer.jl) ---

const INSTANCE_SPECIAL_START: &[char] =
    &['~', '!', '@', '#', '%', '^', '&', '_', '<', '>', '?', '/', '|'];
const INSTANCE_SPECIAL: &[char] =
    &['$', '*', '-', '+', '{', '}', '[', ']', '\\', ';', ':'];

fn is_instance_start_char(c: char) -> bool {
    c.is_alphabetic() || c.is_ascii_digit() || INSTANCE_SPECIAL_START.contains(&c)
}

fn is_instance_char(c: char) -> bool {
    c.is_alphabetic()
        || c.is_ascii_digit()
        || INSTANCE_SPECIAL_START.contains(&c)
        || INSTANCE_SPECIAL.contains(&c)
}

fn is_identifier_char(c: char) -> bool {
    c.is_alphabetic() || c.is_ascii_digit() || c == '_' || c == '$' || c == '#'
}

fn is_identifier_start_char(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_instance_first_char(c: char) -> bool {
    matches!(
        c,
        'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L' | 'M' | 'N' | 'P' | 'Q'
            | 'R' | 'S' | 'T' | 'U' | 'V' | 'W' | 'X' | 'Y'
    )
}

fn ishex(c: char) -> bool {
    c.is_ascii_digit() || ('a'..='f').contains(&c) || ('A'..='F').contains(&c)
}

fn is_whitespace(c: char) -> bool {
    c.is_whitespace() && c != '\n'
}

fn is_token_seperator(c: char) -> bool {
    (c.is_whitespace() || c == ',' || c == '(' || c == ')') && c != '\n'
}

fn is_base_char(c: char) -> bool {
    matches!(c, 'd' | 'D' | 'h' | 'H' | 'o' | 'O' | 'b' | 'B')
}

fn is_logic_char(c: char) -> bool {
    matches!(c, '0' | '1' | 'x' | 'X' | 'z' | 'Z')
}

pub struct Lexer {
    chars: Vec<char>,
    /// `offs[i]` = byte offset of `chars[i]`; `offs[len]` = total byte length.
    offs: Vec<usize>,
    /// Cursor: index of the next char to read (what `peekchar` returns).
    i: usize,
    /// Byte offset of the current token's start.
    tok_start: usize,
    /// Char index of the current token's start (for the `simulator` check).
    tok_start_ci: usize,

    spice_dialect: Dialect,
    /// Carried for the breadth phase (`.tran` strict-mode checks); unused in the
    /// spike subset.
    #[allow(dead_code)]
    strict: bool,
    enable_julia_escape: bool,

    last_token: TokenKind,
    last_nontriv_token: TokenKind,
    lexed_nontriv_token_line: bool,
    lexing_expression_stack: Vec<TokenKind>,

    kw: KeywordTrie,
}

impl Lexer {
    /// `case_sensitive=false` (as the SPICE `ParseState` sets): ASCII input is
    /// upper-cased for classification/keyword matching while byte offsets stay
    /// keyed to the original source. `implicit_title` seeds `last_nontriv_token`
    /// with `TITLE` so the first line lexes as a `TITLE_LINE` (what the parser
    /// wants); the raw lexer (e.g. token-stream tests) passes `false`.
    pub fn new(
        src: &str,
        spice_dialect: Dialect,
        strict: bool,
        enable_julia_escape: bool,
        implicit_title: bool,
    ) -> Self {
        let mut chars = Vec::new();
        let mut offs = Vec::new();
        for (b, c) in src.char_indices() {
            offs.push(b);
            chars.push(c.to_ascii_uppercase());
        }
        offs.push(src.len());
        Lexer {
            chars,
            offs,
            i: 0,
            tok_start: 0,
            tok_start_ci: 0,
            spice_dialect,
            strict,
            enable_julia_escape,
            last_token: ERROR,
            // SPICE files open with an implicit `.TITLE` (parserstate.jl).
            last_nontriv_token: if implicit_title { TITLE } else { ERROR },
            lexed_nontriv_token_line: false,
            lexing_expression_stack: Vec::new(),
            kw: KeywordTrie::new(),
        }
    }

    /// Tokenize the whole buffer, including the final `ENDMARKER`.
    pub fn tokenize(
        src: &str,
        dialect: Dialect,
        strict: bool,
        enable_julia_escape: bool,
        implicit_title: bool,
    ) -> Vec<RawTok> {
        let mut lx = Lexer::new(src, dialect, strict, enable_julia_escape, implicit_title);
        let mut out = Vec::new();
        loop {
            let t = lx.next_token();
            let is_end = t.kind == ENDMARKER;
            out.push(t);
            if is_end {
                break;
            }
        }
        out
    }

    // --- cursor primitives (NyanLexers.jl) ---

    fn peekchar(&self) -> char {
        self.chars.get(self.i).copied().unwrap_or(EOF_CHAR)
    }

    fn dpeekchar(&self) -> (char, char) {
        (
            self.chars.get(self.i).copied().unwrap_or(EOF_CHAR),
            self.chars.get(self.i + 1).copied().unwrap_or(EOF_CHAR),
        )
    }

    fn position(&self) -> usize {
        self.offs[self.i.min(self.chars.len())]
    }

    fn readchar(&mut self) -> char {
        if self.i >= self.chars.len() {
            return EOF_CHAR;
        }
        let c = self.chars[self.i];
        self.i += 1;
        c
    }

    fn accept_ch(&mut self, want: char) -> bool {
        let c = self.peekchar();
        if c == EOF_CHAR {
            return false;
        }
        if c == want {
            self.readchar();
            true
        } else {
            false
        }
    }

    fn accept_f<F: Fn(char) -> bool>(&mut self, f: F) -> bool {
        let c = self.peekchar();
        if c == EOF_CHAR {
            return false;
        }
        if f(c) {
            self.readchar();
            true
        } else {
            false
        }
    }

    fn accept_batch<F: Fn(char) -> bool>(&mut self, f: F) {
        while self.accept_f(&f) {}
    }

    fn start_token(&mut self) {
        self.tok_start = self.position();
        self.tok_start_ci = self.i;
    }

    fn emit(&mut self, kind: TokenKind) -> RawTok {
        let tok = RawTok { kind, start: self.tok_start as u32, end: self.position() as u32 };
        self.last_token = kind;
        if !kind.is_triv() {
            self.lexed_nontriv_token_line = true;
            self.last_nontriv_token = kind;
        }
        if kind.is_newline() {
            self.lexed_nontriv_token_line = false;
        }
        self.start_token();
        tok
    }

    fn last_stack_is(&self, k: TokenKind) -> bool {
        self.lexing_expression_stack.last() == Some(&k)
    }

    // --- the state machine (tokenize/lexer.jl `next_token`) ---

    fn next_token(&mut self) -> RawTok {
        if self.last_nontriv_token == TITLE {
            self.accept_batch(|c| c != '\n' && c != '\r');
            return self.emit(TITLE_LINE);
        }

        let c = self.readchar();

        // If the first non-trivial token of a line is not a `+` continuation,
        // clear the expression-delimiter stack.
        if !self.lexed_nontriv_token_line && !is_whitespace(c) && c != '+' {
            self.lexing_expression_stack.clear();
        }

        if c == EOF_CHAR {
            self.emit(ENDMARKER)
        } else if c == '\n' || c == '\r' {
            self.lex_newline(c)
        } else if (!self.lexing_expression_stack.is_empty() && is_whitespace(c))
            || (self.lexing_expression_stack.is_empty() && is_token_seperator(c))
        {
            self.lex_whitespace()
        } else if matches!(self.last_nontriv_token, INCLUDE | LIB | HDL)
            && self.lexing_expression_stack.is_empty()
        {
            self.lex_path(c)
        } else if !self.lexed_nontriv_token_line && is_instance_first_char(c) {
            self.lex_instance(c)
        } else if c == '/' {
            self.emit(SLASH)
        } else if c == '\\' {
            self.lex_backslash()
        } else if c == '$' {
            self.lex_dollar()
        } else if c == '`' {
            self.emit(BACKTICK)
        } else if c == '*' {
            self.lex_star()
        } else if c == '[' {
            self.lexing_expression_stack.push(LSQUARE);
            self.emit(LSQUARE)
        } else if c == ']' {
            self.lex_rsquare()
        } else if c == '{' {
            self.lexing_expression_stack.push(LBRACE);
            self.emit(LBRACE)
        } else if c == ';' {
            self.lex_semicolon()
        } else if c == '}' {
            self.lex_rbrace()
        } else if c == '(' {
            self.lexing_expression_stack.push(LPAREN);
            self.emit(LPAREN)
        } else if c == ')' {
            self.lex_rparen()
        } else if c == ',' {
            self.lex_comma()
        } else if c == '^' {
            self.lex_circumflex()
        } else if c == '~' {
            self.lex_tilde()
        } else if c == '@' {
            self.emit(AT_SIGN)
        } else if c == '?' {
            self.emit(CONDITIONAL)
        } else if c == '=' {
            self.lex_equal()
        } else if c == '!' {
            self.lex_exclaim()
        } else if c == '>' {
            self.lex_greater()
        } else if c == '<' {
            self.lex_less()
        } else if c == ':' {
            self.emit(COLON)
        } else if c == '&' {
            self.lex_amper()
        } else if c == '|' {
            self.lex_bar()
        } else if c == '\'' && self.spice_dialect == Dialect::Pspice {
            self.lex_number()
        } else if c == '\'' {
            self.lex_prime()
        } else if c == '"' {
            self.lex_prime()
        } else if c == '%' {
            self.emit(PERCENT)
        } else if c == '.' {
            self.lex_dot()
        } else if c == '+' {
            self.emit(PLUS)
        } else if c == '-' {
            self.emit(MINUS)
        } else if is_identifier_start_char(c) {
            self.lex_identifier()
        } else if c.is_ascii_digit() {
            self.lex_number()
        } else {
            self.emit(UNKNOWN)
        }
    }

    fn lex_rbrace(&mut self) -> RawTok {
        // Julia asserts the stack top is LBRACE; on malformed input we just skip
        // the pop rather than panic.
        if self.last_stack_is(LBRACE) {
            self.lexing_expression_stack.pop();
        }
        if self.last_stack_is(EQ) {
            self.lexing_expression_stack.pop();
        }
        self.emit(RBRACE)
    }

    fn lex_rsquare(&mut self) -> RawTok {
        if self.last_stack_is(LSQUARE) {
            self.lexing_expression_stack.pop();
        }
        self.emit(RSQUARE)
    }

    fn lex_comma(&mut self) -> RawTok {
        if self.last_stack_is(EQ) {
            self.lexing_expression_stack.pop();
            self.lex_whitespace()
        } else {
            self.emit(COMMA)
        }
    }

    fn lex_path(&mut self, c: char) -> RawTok {
        if c == '"' || c == '\'' {
            self.lex_quote()
        } else {
            self.accept_batch(|c| !c.is_whitespace());
            self.emit(IDENTIFIER)
        }
    }

    fn lex_rparen(&mut self) -> RawTok {
        if self.last_stack_is(LPAREN) || self.last_stack_is(JULIA_ESCAPE_BEGIN) {
            self.lexing_expression_stack.pop();
            self.emit(RPAREN)
        } else {
            self.lex_whitespace()
        }
    }

    fn lex_whitespace(&mut self) -> RawTok {
        if !self.lexing_expression_stack.is_empty() {
            self.accept_batch(is_whitespace);
            if self.last_stack_is(EQ) && self.last_nontriv_token != EQ {
                self.lexing_expression_stack.pop();
            }
        } else {
            self.accept_batch(is_token_seperator);
        }
        self.emit(WHITESPACE)
    }

    fn lex_escaped_identifier(&mut self) -> RawTok {
        if !self.lexing_expression_stack.is_empty() {
            self.accept_batch(|c| !is_whitespace(c));
        } else {
            self.accept_batch(|c| !is_token_seperator(c));
        }
        self.emit(IDENTIFIER)
    }

    fn lex_newline(&mut self, c: char) -> RawTok {
        if self.last_stack_is(EQ) {
            self.lexing_expression_stack.pop();
        }
        if c == '\n' {
            self.emit(NEWLINE)
        } else {
            // c == '\r'
            if self.accept_ch('\n') {
                self.emit(NEWLINE)
            } else {
                self.emit(WHITESPACE)
            }
        }
    }

    fn lex_backslash(&mut self) -> RawTok {
        if self.accept_ch('\n') {
            return self.emit(ESCD_NEWLINE);
        }
        // Julia: `accept('\r') && accept('\n')`. Rust `&&` short-circuits the
        // same way, so a lone `\r` after `\\` does not consume.
        if self.accept_ch('\r') && self.accept_ch('\n') {
            return self.emit(ESCD_NEWLINE);
        }
        self.lex_escaped_identifier()
    }

    fn lex_greater(&mut self) -> RawTok {
        if self.accept_ch('>') {
            if self.accept_ch('>') {
                self.emit(RRBITSHIFT_A)
            } else {
                self.emit(RBITSHIFT)
            }
        } else if self.accept_ch('=') {
            self.emit(GREATER_EQ)
        } else {
            self.emit(GREATER)
        }
    }

    fn lex_less(&mut self) -> RawTok {
        if self.accept_ch('<') {
            if self.accept_ch('<') {
                self.emit(LBITSHIFT_A)
            } else {
                self.emit(LBITSHIFT)
            }
        } else if self.accept_ch('=') {
            self.emit(LESS_EQ)
        } else if self.accept_ch('+') {
            self.emit(CASSIGN)
        } else {
            self.emit(LESS)
        }
    }

    fn lex_dot(&mut self) -> RawTok {
        if self.last_token.is_ident() {
            return self.emit(DOT);
        }
        if self.peekchar().is_ascii_digit() {
            self.lex_number()
        } else {
            self.emit(DOT)
        }
    }

    fn lex_equal(&mut self) -> RawTok {
        if self.accept_ch('=') {
            if self.accept_ch('=') {
                self.emit(EQEQEQ)
            } else {
                self.emit(EQEQ)
            }
        } else {
            if self.lexing_expression_stack.is_empty() {
                self.lexing_expression_stack.push(EQ);
            }
            self.emit(EQ)
        }
    }

    fn lex_exclaim(&mut self) -> RawTok {
        if self.accept_ch('=') {
            if self.accept_ch('=') {
                self.emit(NOT_IS)
            } else {
                self.emit(NOT_EQ)
            }
        } else {
            self.emit(NOT)
        }
    }

    fn lex_dollar(&mut self) -> RawTok {
        if self.enable_julia_escape && self.accept_ch('(') {
            self.lexing_expression_stack.push(JULIA_ESCAPE_BEGIN);
            return self.emit(JULIA_ESCAPE_BEGIN);
        }
        self.accept_batch(|c| c != '\n');
        self.emit(COMMENT)
    }

    fn lex_semicolon(&mut self) -> RawTok {
        self.accept_batch(|c| c != '\n');
        self.emit(COMMENT)
    }

    fn lex_star(&mut self) -> RawTok {
        if !self.lexed_nontriv_token_line {
            self.accept_batch(|c| c != '\n');
            return self.emit(COMMENT);
        }
        if self.accept_ch('*') {
            return self.emit(STAR_STAR);
        }
        self.emit(STAR)
    }

    fn lex_circumflex(&mut self) -> RawTok {
        if self.accept_ch('~') {
            self.emit(XOR_TILDE)
        } else {
            self.emit(XOR)
        }
    }

    fn lex_tilde(&mut self) -> RawTok {
        if self.accept_ch('^') {
            self.emit(TILDE_XOR)
        } else if self.accept_ch('&') {
            self.emit(TILDE_AND)
        } else if self.accept_ch('|') {
            self.emit(TILD_OR)
        } else {
            self.emit(TILDE)
        }
    }

    fn lex_number(&mut self) -> RawTok {
        while self.peekchar().is_ascii_digit() {
            self.readchar();
        }
        if self.peekchar() == '.' {
            self.readchar();
            while self.peekchar().is_ascii_digit() {
                self.readchar();
            }
        }
        let (pc, ppc) = self.dpeekchar();
        if pc == '\'' && is_base_char(ppc) {
            self.readchar();
            self.readchar();
            while ishex(self.peekchar()) {
                self.readchar();
            }
        } else if pc == '\'' && is_logic_char(ppc) {
            self.readchar();
            self.readchar();
        } else if pc == 'e' || pc == 'E' {
            self.readchar();
            let pc = self.peekchar();
            if pc == '+' || pc == '-' {
                self.readchar();
            }
            while self.peekchar().is_ascii_digit() {
                self.readchar();
            }
        }
        // Scale factors / units — context-sensitive like `lex_identifier`.
        loop {
            let pc = self.peekchar();
            if (is_identifier_char(pc) && !self.lexing_expression_stack.is_empty())
                || (is_instance_char(pc) && self.lexing_expression_stack.is_empty())
            {
                self.readchar();
            } else {
                break;
            }
        }
        self.emit(NUMBER)
    }

    fn lex_prime(&mut self) -> RawTok {
        if matches!(self.last_nontriv_token, LIB | INCLUDE) {
            return self.lex_quote();
        }
        if self.last_stack_is(PRIME) {
            self.lexing_expression_stack.pop();
            if self.last_stack_is(EQ) {
                self.lexing_expression_stack.pop();
            }
        } else {
            self.lexing_expression_stack.push(PRIME);
        }
        self.emit(PRIME)
    }

    fn lex_amper(&mut self) -> RawTok {
        if self.accept_ch('&') {
            self.emit(LAZY_AND)
        } else {
            self.emit(AND)
        }
    }

    fn lex_bar(&mut self) -> RawTok {
        if self.accept_ch('|') {
            self.emit(LAZY_OR)
        } else {
            self.emit(OR)
        }
    }

    fn lex_quote(&mut self) -> RawTok {
        if self.accept_ch('"') || self.accept_ch('\'') {
            self.emit(STRING)
        } else if self.read_string() {
            self.emit(STRING)
        } else {
            self.emit(EOF_STRING)
        }
    }

    /// We just consumed the opening `"` or `'`. Julia's `read_string` tracks the
    /// terminator by inspecting `l.chars[1]` (the just-read char) after each
    /// `readchar`; a `\` escapes the next char.
    fn read_string(&mut self) -> bool {
        loop {
            let c = self.readchar();
            if c == '\\' {
                if self.readchar() == EOF_CHAR {
                    return false;
                }
                continue;
            }
            if c == '"' || c == '\'' {
                return true;
            } else if c == EOF_CHAR {
                return false;
            }
        }
    }

    fn lex_identifier(&mut self) -> RawTok {
        loop {
            let pc = self.peekchar();
            if (is_identifier_char(pc) && !self.lexing_expression_stack.is_empty())
                || (is_instance_char(pc) && self.lexing_expression_stack.is_empty())
            {
                self.readchar();
            } else {
                break;
            }
        }
        let text: String = self.chars[self.tok_start_ci..self.i].iter().collect();
        if let Some(kind) = self.kw.lookup(&text) {
            // Some dot-commands begin an implicit expression.
            if self.last_nontriv_token == DOT
                && matches!(kind, PARAMETERS | IC | MEASURE | PRINT | IF | ELSEIF)
            {
                self.lexing_expression_stack.push(kind);
            }
            return self.emit(kind);
        }
        self.emit(IDENTIFIER)
    }

    fn lex_instance(&mut self, c: char) -> RawTok {
        use Dialect::*;
        let typ = match c {
            'B' => IDENTIFIER_BEHAVIORAL,
            'C' => IDENTIFIER_CAPACITOR,
            'D' => IDENTIFIER_DIODE,
            'E' => IDENTIFIER_VOLTAGE_CONTROLLED_VOLTAGE,
            'F' => IDENTIFIER_CURRENT_CONTROLLED_CURRENT,
            'G' => IDENTIFIER_VOLTAGE_CONTROLLED_CURRENT,
            'H' => IDENTIFIER_CURRENT_CONTROLLED_VOLTAGE,
            'I' => IDENTIFIER_CURRENT,
            'J' => IDENTIFIER_JFET,
            'K' => IDENTIFIER_LINEAR_MUTUAL_INDUCTOR,
            'L' => IDENTIFIER_LINEAR_INDUCTOR,
            'M' => IDENTIFIER_MOSFET,
            'N' if self.spice_dialect == Ngspice => IDENTIFIER_OSDI,
            'O' => IDENTIFIER_TRANSMISSION_LINE,
            'P' if self.spice_dialect == Hspice => IDENTIFIER_PORT,
            'P' if self.spice_dialect == Ngspice => IDENTIFIER_TRANSMISSION_LINE,
            'Q' => IDENTIFIER_BIPOLAR_TRANSISTOR,
            'R' => IDENTIFIER_RESISTOR,
            'S' if self.spice_dialect == Hspice => IDENTIFIER_S_PARAMETER_ELEMENT,
            'S' if self.spice_dialect == Ngspice => IDENTIFIER_SWITCH,
            'V' => IDENTIFIER_VOLTAGE,
            'T' => IDENTIFIER_TRANSMISSION_LINE,
            'U' => IDENTIFIER_TRANSMISSION_LINE,
            'W' if self.spice_dialect == Hspice => IDENTIFIER_TRANSMISSION_LINE,
            'W' if self.spice_dialect == Ngspice => IDENTIFIER_SWITCH,
            'X' => IDENTIFIER_SUBCIRCUIT_CALL,
            'Y' if self.spice_dialect == Ngspice => IDENTIFIER_TRANSMISSION_LINE,
            'Y' if self.spice_dialect == Xyce => IDENTIFIER_OSDI,
            'Z' if self.spice_dialect == Ngspice => IDENTIFIER_HFET_MESA,
            _ => IDENTIFIER_UNKNOWN_INSTANCE,
        };

        if self.accept_f(is_instance_start_char) {
            self.accept_batch(is_instance_char);
            if matches!(typ, IDENTIFIER_S_PARAMETER_ELEMENT | IDENTIFIER_SWITCH) {
                let text: String = self.chars[self.tok_start_ci..self.i]
                    .iter()
                    .collect::<String>()
                    .to_lowercase();
                if text == "simulator" {
                    return self.emit(SIMULATOR);
                }
            }
            return self.emit(typ);
        }
        self.emit(ERROR)
    }
}

#[cfg(test)]
mod token_tests {
    //! Port of the Julia lexer test table (`test/SPICE/tokenize.jl`): each SPICE
    //! snippet must tokenize to the same non-trivia `TokenKind` sequence. Uses
    //! the raw lexer (no implicit title), matching the Julia test's
    //! `tokenize(str, ERROR, ...)`.
    use crate::lexer::{Dialect, Lexer};
    use crate::syntax_kind::TokenKind::{self, *};

    fn kinds(src: &str) -> Vec<TokenKind> {
        Lexer::tokenize(src, Dialect::Ngspice, false, false, false)
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| !k.is_triv() && *k != ENDMARKER)
            .collect()
    }

    #[test]
    fn julia_tokenize_table() {
        let cases: &[(&str, &[TokenKind])] = &[
            (".TITLE THIS IS A TILE WITH \u{a4}%&/)/& stuff", &[DOT, TITLE, TITLE_LINE]),
            ("Vv-_-{} A B 0", &[IDENTIFIER_VOLTAGE, IDENTIFIER, IDENTIFIER, NUMBER]),
            ("* MOSFET", &[]),
            (".GLOBAL VDD NET1", &[DOT, GLOBAL, IDENTIFIER, IDENTIFIER]),
            (".MODEL BJT_modName NPN (BF=val)", &[DOT, MODEL, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, VAL]),
            ("Rname N1 N2 0.1 $ comment", &[IDENTIFIER_RESISTOR, IDENTIFIER, IDENTIFIER, NUMBER]),
            ("sky130.0", &[IDENTIFIER_SWITCH, DOT, NUMBER]),
            (".param freq = 1Meg", &[DOT, PARAMETERS, IDENTIFIER, EQ, NUMBER]),
            (".parameter freq = 1Meg", &[DOT, PARAMETERS, IDENTIFIER, EQ, NUMBER]),
            ("*comment", &[]),
            ("    *comment", &[]),
            (";comment", &[]),
            ("    ;comment", &[]),
            ("1    ;comment", &[NUMBER]),
            ("{1*1}", &[LBRACE, NUMBER, STAR, NUMBER, RBRACE]),
            (".lib 'with spaces/sm141064.ngspice' nmos_6p0_t", &[DOT, LIB, STRING, IDENTIFIER]),
            (".lib sm141064.ngspice nmos_6p0_t", &[DOT, LIB, IDENTIFIER, IDENTIFIER]),
            (".include ./foo", &[DOT, INCLUDE, IDENTIFIER]),
            (".include './foo.bar'", &[DOT, INCLUDE, STRING]),
            (
                ".param r_l='s*(r_length-2*r_dl)'",
                &[DOT, PARAMETERS, IDENTIFIER, EQ, PRIME, IDENTIFIER, STAR, LPAREN, IDENTIFIER, MINUS, NUMBER, STAR, IDENTIFIER, RPAREN, PRIME],
            ),
            ("Q1 Net-_Q1-C_ Net-_Q1-B_ 0 BC546B", &[IDENTIFIER_BIPOLAR_TRANSISTOR, IDENTIFIER, IDENTIFIER, NUMBER, IDENTIFIER]),
            (
                "X1 v+ v- r={a-b} f-o-o=1",
                &[IDENTIFIER_SUBCIRCUIT_CALL, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, LBRACE, IDENTIFIER, MINUS, IDENTIFIER, RBRACE, IDENTIFIER, EQ, NUMBER],
            ),
            (
                ".ic v( m_tn4:d )=  1.225e-08",
                &[DOT, IC, VAL, LPAREN, IDENTIFIER, COLON, IDENTIFIER, RPAREN, EQ, NUMBER],
            ),
            ("V1 vin 0 SIN (0, 1, 1k)", &[IDENTIFIER_VOLTAGE, IDENTIFIER, NUMBER, SIN, NUMBER, NUMBER, NUMBER]),
            (
                "R1 (a b) r=\"foo+bar\"",
                &[IDENTIFIER_RESISTOR, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, PRIME, IDENTIFIER, PLUS, IDENTIFIER, PRIME],
            ),
            (
                ".MEAS TRAN res1 FIND V(out) AT=5m",
                &[DOT, MEASURE, TRAN, IDENTIFIER, FIND, VAL, LPAREN, IDENTIFIER, RPAREN, AT, EQ, NUMBER],
            ),
            (".tran 1ns 60ns", &[DOT, TRAN, NUMBER, NUMBER]),
            (".model 1N3064 D", &[DOT, MODEL, NUMBER, IDENTIFIER]),
            ("2N2222", &[NUMBER]),
            ("R1 n1 n2 r={1.5e-3}", &[IDENTIFIER_RESISTOR, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, LBRACE, NUMBER, RBRACE]),
            (
                "R2 n1 n2 r={1foe-bar}",
                &[IDENTIFIER_RESISTOR, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, LBRACE, NUMBER, MINUS, IDENTIFIER, RBRACE],
            ),
            ("C1 n1 n2 1.5e-12F", &[IDENTIFIER_CAPACITOR, IDENTIFIER, IDENTIFIER, NUMBER]),
            (".OPTIONS montequantiles=[0.134 99.865]", &[DOT, OPTIONS, IDENTIFIER, EQ, LSQUARE, NUMBER, NUMBER, RSQUARE]),
            (".param vals=[1 2 3]", &[DOT, PARAMETERS, IDENTIFIER, EQ, LSQUARE, NUMBER, NUMBER, NUMBER, RSQUARE]),
            (".OPTIONS someopt=[1,2]", &[DOT, OPTIONS, IDENTIFIER, EQ, LSQUARE, NUMBER, COMMA, NUMBER, RSQUARE]),
            (".param x=[1.5e-3]", &[DOT, PARAMETERS, IDENTIFIER, EQ, LSQUARE, NUMBER, RSQUARE]),
            ("123'hAB", &[NUMBER]),
            ("8'hFF", &[NUMBER]),
            ("4'b1010", &[NUMBER]),
            ("8'o377", &[NUMBER]),
            ("16'h1234", &[NUMBER]),
            (".param myval=123'hAB", &[DOT, PARAMETERS, IDENTIFIER, EQ, NUMBER]),
            (
                "EOS 7 1 POLY(1) 16 49 2E-3 1",
                &[IDENTIFIER_VOLTAGE_CONTROLLED_VOLTAGE, NUMBER, NUMBER, POLY, NUMBER, NUMBER, NUMBER, NUMBER, NUMBER],
            ),
            (
                "GD16 16 1 TABLE {V(16,1)} ((-100,-1p)(0,0)(1m,1u)(2m,1m))",
                &[IDENTIFIER_VOLTAGE_CONTROLLED_CURRENT, NUMBER, NUMBER, TABLE, LBRACE, VAL, LPAREN, NUMBER, COMMA, NUMBER, RPAREN, RBRACE, MINUS, NUMBER, MINUS, NUMBER, NUMBER, NUMBER, NUMBER, NUMBER, NUMBER, NUMBER],
            ),
            ("", &[]),
        ];
        for (src, expected) in cases {
            assert_eq!(&kinds(src), expected, "tokenizing {src:?}");
        }
    }
}
