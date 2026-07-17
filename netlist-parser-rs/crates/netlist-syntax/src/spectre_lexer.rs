//! Spectre lexer — a faithful port of `NyanSpectreNetlistParser.jl/src/tokenize/lexer.jl`
//! (built on `NyanLexers.jl/src/lexer.jl`).
//!
//! Structurally this mirrors the SPICE `lexer.rs` (a pre-collected `Vec<char>`
//! cursor emitting half-open `[start, end)` byte spans), but the Spectre lexer is
//! considerably SIMPLER:
//! - No `lexing_expression_stack` (there is no `+` continuation, no instance-name
//!   char class, no delimiter-driven mode switching).
//! - No implicit `TITLE` first line, no SPICE dialects, no `.include` path lexing.
//! - CASE SENSITIVE: source is kept verbatim (SPICE upper-cases everything). The
//!   keyword trie stores lower-cased spellings for most reserved words and
//!   upper-cased spellings for builtin constants (`M_PI`, `P_Q`), so `tan` is the
//!   `TAN` keyword while `TAN`/`MHz` are plain identifiers.
//!
//! Spectre-specific lexer gotchas (all handled below):
//! - `is_identifier_char` includes `'!'` and `'$'`; `is_identifier_start_char`
//!   is `isletter | '_'`. So `vdd!` is one identifier, and `a!=b` lexes as
//!   `IDENTIFIER("a!") EQ IDENTIFIER("b")`.
//! - Numbers greedily absorb a trailing scale-factor + unit + identifier chars
//!   (`23pf`, `0.3MHz`, `6.3ns`, `6_Ohms` are each a single NUMBER); `.3`,
//!   `1e-3`, `2E+5`, `011`, `1.0` are also NUMBER. (Julia defines
//!   `accept_float`/`is_scale_factor`/`maybe_lex_unit` but `lex_number` never
//!   calls them — the trailing identifier-char loop subsumes them. We port
//!   `lex_number` faithfully and omit the dead helpers.)
//! - Comments: `//` (`lex_slash`) and a leading `*` at line start (`lex_star`
//!   when `last_token == NEWLINE`). `;` is `SEMICOLON`, not a comment.
//! - Continuation: a trailing backslash lexes to `ESCD_NEWLINE`. A leading `+`
//!   on a following line is also folded as a continuation, but that is handled
//!   in the parser's token layer (`get_next_action`), not here.
//! - `~|` lexes to `TILDE_OR` (`lex_tilde`), matching the Julia lexer.

use crate::spectre_keywords::KeywordTrie;
use crate::spectre_syntax_kind::TokenKind;
use crate::spectre_syntax_kind::TokenKind::*;

/// Sentinel for "past end of input". `char::MAX` matches Julia's
/// `EOF_CHAR = typemax(Char)`: every character-class predicate returns false.
const EOF_CHAR: char = '\u{10FFFF}';

/// A raw lexer token: kind + half-open byte span `[start, end)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RawTok {
    pub kind: TokenKind,
    pub start: u32,
    pub end: u32,
}

// --- character classifiers (tokenize/lexer.jl) ---

fn is_identifier_char(c: char) -> bool {
    c.is_alphabetic() || c.is_ascii_digit() || c == '_' || c == '$' || c == '!'
}

fn is_identifier_start_char(c: char) -> bool {
    // CN: allows unicode letters, though the standard does not specify whether
    // this is permitted.
    c.is_alphabetic() || c == '_'
}

fn ishex(c: char) -> bool {
    c.is_ascii_digit() || ('a'..='f').contains(&c) || ('A'..='F').contains(&c)
}

fn is_whitespace(c: char) -> bool {
    c.is_whitespace() && c != '\n'
}

fn is_base_char(c: char) -> bool {
    matches!(c, 'd' | 'D' | 'h' | 'H' | 'o' | 'O' | 'b' | 'B')
}

pub struct Lexer {
    chars: Vec<char>,
    /// `offs[i]` = byte offset of `chars[i]`; `offs[len]` = total byte length.
    offs: Vec<usize>,
    /// Cursor: index of the next char to read (what `peekchar` returns).
    i: usize,
    /// Byte offset of the current token's start.
    tok_start: usize,
    /// Char index of the current token's start (for keyword-text slicing).
    tok_start_ci: usize,

    last_token: TokenKind,

    kw: KeywordTrie,
}

impl Lexer {
    /// `last_token` seeds the "previous token" state (the Julia `tokenize` entry
    /// point takes it as an argument; the token-table test passes `ERROR`). It
    /// only affects `lex_star`'s leading-`*`-comment decision.
    pub fn new(src: &str, last_token: TokenKind) -> Self {
        let mut chars = Vec::new();
        let mut offs = Vec::new();
        for (b, c) in src.char_indices() {
            offs.push(b);
            chars.push(c);
        }
        offs.push(src.len());
        Lexer { chars, offs, i: 0, tok_start: 0, tok_start_ci: 0, last_token, kw: KeywordTrie::new() }
    }

    /// Tokenize the whole buffer, including the final `ENDMARKER`.
    pub fn tokenize(src: &str, last_token: TokenKind) -> Vec<RawTok> {
        let mut lx = Lexer::new(src, last_token);
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
        self.start_token();
        tok
    }

    // --- the state machine (tokenize/lexer.jl `next_token`) ---

    fn next_token(&mut self) -> RawTok {
        let c = self.readchar();

        if c == EOF_CHAR {
            self.emit(ENDMARKER)
        } else if c == '\n' || c == '\r' {
            self.lex_newline(c)
        } else if is_whitespace(c) {
            self.lex_whitespace()
        } else if c == '"' {
            self.lex_quote()
        } else if c == '/' {
            self.lex_slash()
        } else if c == '\\' {
            self.lex_backslash()
        } else if c == '$' {
            self.lex_identifier()
        } else if c == '`' {
            self.emit(BACKTICK)
        } else if c == '(' {
            self.emit(LPAREN)
        } else if c == '*' {
            self.lex_star()
        } else if c == '[' {
            self.emit(LSQUARE)
        } else if c == ']' {
            self.emit(RSQUARE)
        } else if c == '{' {
            self.emit(LBRACE)
        } else if c == ';' {
            self.emit(SEMICOLON)
        } else if c == '}' {
            self.emit(RBRACE)
        } else if c == ')' {
            self.emit(RPAREN)
        } else if c == ',' {
            self.emit(COMMA)
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
        } else if c == '\'' {
            self.emit(PRIME)
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

    fn lex_whitespace(&mut self) -> RawTok {
        self.accept_batch(is_whitespace);
        self.emit(WHITESPACE)
    }

    // Lex identifier after an escaping backslash.
    fn lex_escaped_identifier(&mut self) -> RawTok {
        self.accept_batch(|c| !is_whitespace(c));
        self.emit(IDENTIFIER)
    }

    fn lex_slash(&mut self) -> RawTok {
        if self.peekchar() == '/' {
            // Line comment.
            self.accept_batch(|c| c != '\n');
            self.emit(COMMENT)
        } else {
            self.emit(SLASH)
        }
    }

    fn lex_newline(&mut self, c: char) -> RawTok {
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

    fn lex_star(&mut self) -> RawTok {
        if self.last_token == NEWLINE {
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
            self.emit(TILDE_OR)
        } else {
            self.emit(TILDE)
        }
    }

    // A digit (or a `.` immediately before a digit) has been consumed.
    fn lex_number(&mut self) -> RawTok {
        // Accept digits.
        while self.peekchar().is_ascii_digit() {
            self.readchar();
        }
        // Accept decimal point and more digits.
        if self.peekchar() == '.' {
            self.readchar();
            while self.peekchar().is_ascii_digit() {
                self.readchar();
            }
        }
        // Accept base specifiers (e.g. `123'hAB`) or scientific notation.
        let (pc, ppc) = self.dpeekchar();
        if pc == '\'' && is_base_char(ppc) {
            self.readchar(); // consume '\''
            self.readchar(); // consume base char
            while ishex(self.peekchar()) {
                self.readchar();
            }
        } else if pc == 'e' || pc == 'E' {
            self.readchar(); // consume 'e' or 'E'
            let pc = self.peekchar();
            if pc == '+' || pc == '-' {
                self.readchar();
            }
            while self.peekchar().is_ascii_digit() {
                self.readchar();
            }
        }
        // Accept scale factors, units, and identifier chars (e.g. `1N3064`).
        loop {
            let pc = self.peekchar();
            if is_identifier_char(pc) {
                self.readchar();
            } else {
                break;
            }
        }
        self.emit(NUMBER)
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

    // Parse a token starting with a quote. A `"` has been consumed.
    fn lex_quote(&mut self) -> RawTok {
        if self.accept_ch('"') {
            // Empty string `""`.
            self.emit(STRING)
        } else if self.read_string() {
            self.emit(STRING)
        } else {
            self.emit(EOF_STRING)
        }
    }

    /// We just consumed the opening `"`. Julia's `read_string` inspects the
    /// just-read char (`l.chars[1]` after each `readchar`) and terminates on a
    /// `"`; a `\` escapes the next char. Spectre strings are only `"`-delimited.
    fn read_string(&mut self) -> bool {
        loop {
            let c = self.readchar();
            if c == '\\' {
                if self.readchar() == EOF_CHAR {
                    return false;
                }
                continue;
            }
            if c == '"' {
                return true;
            } else if c == EOF_CHAR {
                return false;
            }
        }
    }

    fn lex_identifier(&mut self) -> RawTok {
        loop {
            let pc = self.peekchar();
            if is_identifier_char(pc) {
                self.readchar();
            } else {
                break;
            }
        }
        // Exact-match the full identifier text against the keyword trie (no
        // unique-prefix completion, unlike SPICE).
        let text: String = self.chars[self.tok_start_ci..self.i].iter().collect();
        if let Some(kind) = self.kw.lookup(&text) {
            return self.emit(kind);
        }
        self.emit(IDENTIFIER)
    }
}

#[cfg(test)]
mod token_tests {
    //! Verbatim port of the Julia lexer test table
    //! (`NyanSpectreNetlistParser.jl/test/tokenize.jl`): the snippets are joined
    //! by newlines, tokenized with an initial `last_token` of `ERROR`, trivia is
    //! filtered out (but `ENDMARKER` is kept, matching the Julia `!is_triv`
    //! filter), and the flattened non-`nothing` expectations must match exactly.
    use crate::spectre_lexer::Lexer;
    use crate::spectre_syntax_kind::TokenKind::{self, *};

    #[test]
    fn julia_tokenize_table() {
        // (source, Some(expected kinds) | None for a comment-only line)
        let cases: &[(&str, Option<&[TokenKind]>)] = &[
            ("// foo", None),
            ("011", Some(&[NUMBER])),
            ("1.0", Some(&[NUMBER])),
            ("vdd!", Some(&[IDENTIFIER])),
            ("tan", Some(&[TAN])),
            ("tanh", Some(&[TANH])),
            ("march", Some(&[MARCH])),
            ("int", Some(&[INT])),
            ("2pf", Some(&[NUMBER])),
            ("6.3ns", Some(&[NUMBER])),
            ("6_Ohms", Some(&[NUMBER])),
            ("0.3MHz", Some(&[NUMBER])),
            ("MHz", Some(&[IDENTIFIER])),
            ("a = 1 \\\nb=2", Some(&[IDENTIFIER, EQ, NUMBER, ESCD_NEWLINE, IDENTIFIER, EQ, NUMBER])),
            ("name info info=foo", Some(&[IDENTIFIER, INFO, INFO, EQ, IDENTIFIER])),
            ("tran tran tran=tran", Some(&[TRAN, TRAN, TRAN, EQ, TRAN])),
            ("save save=foo", Some(&[SAVE, SAVE, EQ, IDENTIFIER])),
            ("* comment", None),
            ("", Some(&[ENDMARKER])),
        ];

        let src = cases.iter().map(|(s, _)| *s).collect::<Vec<_>>().join("\n");
        let got: Vec<TokenKind> = Lexer::tokenize(&src, ERROR)
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| !k.is_triv())
            .collect();
        let expected: Vec<TokenKind> =
            cases.iter().filter_map(|(_, e)| *e).flatten().copied().collect();
        assert_eq!(got, expected);
    }
}
