//! Spectre token kinds + classifier predicates + operator precedence.
//!
//! A faithful transcription of the Julia `Tokens.Kind` `@enum` in
//! `NyanSpectreNetlistParser.jl/src/tokenize/token_kinds.jl`, in the same order,
//! including the `begin_*`/`end_*` marker values so the range-based classifier
//! predicates (`is_kw`, `is_operator`, ...) port over as plain ordinal
//! comparisons.
//!
//! The Spectre token set DIFFERS from the SPICE one (`syntax_kind.rs`): a
//! different keyword set (control/analysis/save/builtin groups), an `EVENT_OR`
//! operator, `INCLUDE_FNAME`, and distinct operator ordering. It therefore lives
//! in its own module. The rowan CST kind space (`SyntaxKind`) is SHARED and
//! lives in `syntax_kind.rs`; only this token layer is Spectre-specific.
//!
//! Lexer-vs-enum note (see `lexer.jl`): `lex_greater` can emit `RRBITSHIFT_A`
//! (`>>>`), `lex_less` can emit `CASSIGN` (`<+`), `lex_exclaim` can emit `NOT_IS`
//! (`!==`). None of `CASSIGN`, `RRBITSHIFT_A`, or `NOT_IS` exists in the Julia
//! `Tokens.Kind` enum (referencing them there would `UndefVarError` / throw at lex
//! time). We give them real variants inside the ops range so the lexer is total;
//! `prec()` panics on them (matching Julia, which has no arm for them). `~|`
//! lexes to the real `TILDE_OR` variant (Julia's `lex_tilde` historically had a
//! typo here that threw; that has been fixed in the Julia source too).

#![allow(non_camel_case_types)]

/// Lexer token kind. Mirrors Spectre `Tokens.Kind` in Julia, in the same order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u16)]
pub enum TokenKind {
    ENDMARKER, // EOF
    COMMENT,   // //
    WHITESPACE,
    NEWLINE,
    ESCD_NEWLINE, // \ \n
    IDENTIFIER,
    BASE_SPEC,     // 'b (but not hex)
    HEX_BASE_SPEC, // 'h or 'H
    BACKTICK,      // `
    COMMA,
    COLON,
    SEMICOLON,
    EQ, // =
    DOT,
    AT_SIGN,     // @
    HASH,        // #
    DOLLAR,      // $
    CONDITIONAL, // ?
    PRIME,       // '

    INCLUDE_FNAME,
    JULIA_ESCAPE, // $( if enabled

    begin_errors,
    ERROR,
    EOF_STRING,
    UNKNOWN,
    end_errors,

    begin_literal,
    LITERAL,
    begin_number,
    NUMBER,
    end_number,
    STRING,
    end_literal,

    begin_delimiters,
    LSQUARE,
    RSQUARE,
    LBRACE,
    RBRACE,
    LPAREN,
    RPAREN,
    LATTR, // (*
    RATTR, // *)
    end_delimiters,

    begin_keywords,
    CORRELATE,
    ELSE,
    END,
    ENDS,
    EXPORT,
    FOR,
    FUNCTION,
    GLOBAL,
    IF,
    INLINE,
    LIBRARY,
    LOCAL,
    MARCH,
    MODEL,
    PARAMETERS,
    PARAMTEST,
    PLOT,
    PRINT,
    REAL,
    RETURN,

    SUBCKT,

    TO,
    VARY,

    SIMULATOR,
    LANG,
    SPECTRE,
    SPICE,
    INCLUDE,
    AHDL_INCLUDE,

    begin_control,
    begin_second_control,
    ALTER,
    ALTERGROUP,
    ASSERT,
    CHECK,
    CHECKLIMIT,
    INFO,
    OPTIONS,
    PARAMSET,
    SET,
    SHELL,
    end_second_control,
    begin_first_control,
    IC,
    NODESET,
    SAVE,
    end_first_control,
    STATISTICS,
    end_control,

    begin_analyses,
    DC,
    AC,
    NOISE,
    XF,
    SP,
    TRAN,
    TDR,
    PZ,

    ENVLP,
    PAC,
    PDISTO,
    PNOISE,
    PSS,
    PXF,

    SENS,
    FOURIER,
    DCMATCH,
    STB,
    SWEEP,
    MONTECARLO,
    end_analyses,

    SECTION,

    begin_builtin_constants,
    M_1_PI,
    M_2_PI,
    M_2_SQRTP,
    M_DEGPERRAD,
    M_E,
    M_LN10,
    M_LN2,
    M_LOG10E,
    M_LOG2E,
    M_PI,
    M_PI_2,
    M_PI_4,
    M_SQRT1_2,
    M_SQRT2,
    M_TWO_PI,
    P_C,
    P_CELSIUS0,
    P_EPS0,
    P_H,
    P_K,
    P_Q,
    P_U0,
    end_builtin_constants,

    begin_builtin_functions,
    ABS,
    ACOS,
    ACOSH,
    ASIN,
    ASINH,
    ATAN,
    ATAN2,
    ATANH,
    CEIL,
    COS,
    COSH,
    EXP,
    FLOOR,
    FMOD,
    HYPOT,
    INT,
    LOG,
    LOG10,
    NINT,
    MAX,
    MIN,
    POW,
    SIN,
    SINH,
    SQRT,
    TAN,
    TANH,
    TRUNCATE,
    end_builtin_functions,

    begin_save_keywords,
    CURRENTS,
    STATIC,
    DISPLACEMENT,
    DYNAMIC,
    OPPOINT,
    PROBE,
    PWR,
    ALL,
    end_save_keywords,
    end_keywords,

    begin_ops,
    OP, // general

    // Arithmetic
    PLUS,      // +
    MINUS,     // -
    STAR,      // *
    SLASH,     // /
    STAR_STAR, // **

    // Modulus
    PERCENT, // %

    // Relational
    GREATER,    // >
    LESS,       // <
    GREATER_EQ, // >=
    LESS_EQ,    // <=

    // Logical equality
    EQEQ,   // ==
    NOT_EQ, // !=

    // Case equality
    EQEQEQ,  // ===
    NOT_EQEQ, // !==

    // Logical negation
    NOT, // !

    // Logical and
    LAZY_AND, // &&

    // Logical or
    LAZY_OR, // ||

    // Bitwise
    TILDE, // ~
    AND,   // &
    OR,    // |
    XOR,   // ^

    // Reduction
    XOR_TILDE, // ^~
    TILDE_XOR, // ~^
    TILDE_AND, // ~&
    TILDE_OR,  // ~|

    // Bitshifts
    LBITSHIFT,   // <<
    RBITSHIFT,   // >>
    LBITSHIFT_A, // <<<
    RBITSHIFT_A, // >>>

    // Literal "or"
    EVENT_OR, // or

    // Emitted by the lexer but absent from the Julia enum (see module docs).
    CASSIGN,      // <+
    RRBITSHIFT_A, // >>>
    NOT_IS,       // !==
    end_ops,
}

use TokenKind::*;

impl TokenKind {
    #[inline]
    fn ord(self) -> u16 {
        self as u16
    }

    /// `is_kw`: strictly between the keyword markers.
    pub fn is_kw(self) -> bool {
        begin_keywords.ord() < self.ord() && self.ord() < end_keywords.ord()
    }

    /// `is_ident`: a plain identifier token *or* a keyword (Spectre resolves the
    /// keyword/identifier distinction in the parser, not the lexer).
    pub fn is_ident(self) -> bool {
        self == IDENTIFIER || self.is_kw()
    }

    pub fn is_operator(self) -> bool {
        begin_ops.ord() < self.ord() && self.ord() < end_ops.ord()
    }

    pub fn is_unary_operator(self) -> bool {
        self == PLUS || self == MINUS
    }

    pub fn is_literal(self) -> bool {
        begin_literal.ord() < self.ord() && self.ord() < end_literal.ord()
    }

    pub fn is_number(self) -> bool {
        begin_number.ord() < self.ord() && self.ord() < end_number.ord()
    }

    pub fn is_builtin_func(self) -> bool {
        begin_builtin_functions.ord() < self.ord() && self.ord() < end_builtin_functions.ord()
    }

    pub fn is_builtin_const(self) -> bool {
        begin_builtin_constants.ord() < self.ord() && self.ord() < end_builtin_constants.ord()
    }

    pub fn is_analysis(self) -> bool {
        begin_analyses.ord() < self.ord() && self.ord() < end_analyses.ord()
    }

    pub fn is_save_kw(self) -> bool {
        begin_save_keywords.ord() < self.ord() && self.ord() < end_save_keywords.ord()
    }

    /// `NyanLexers.is_triv`
    pub fn is_triv(self) -> bool {
        self == COMMENT || self == WHITESPACE || self == NEWLINE
    }

    /// `NyanLexers.is_newline`
    pub fn is_newline(self) -> bool {
        self == NEWLINE
    }
}

/// Precedence levels, mirroring the Julia `PrecedenceLevels` `@enum` in
/// `parse.jl`. Higher binds tighter.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Prec {
    Logical,
    AndAnd,
    Or,
    Xor,
    And,
    Eq,
    Lt,
    Shift,
    Plus,
    Mul,
    StarStar,
}

/// `prec(opkind)` from parse.jl. Panics on any kind with no arm — matching
/// Julia's `error("Unknown operator")`. `TILDE_AND` (`~&`) groups with `AND` and
/// `TILDE_OR` (`~|`) groups with `OR`, matching the Julia `prec()` arms; the
/// lexer-only operators (`CASSIGN`, `RRBITSHIFT_A`, `NOT_IS`) have no arm and
/// panic, as in Julia.
pub fn prec(op: TokenKind) -> Prec {
    match op {
        LAZY_OR | EVENT_OR => Prec::Logical,
        LAZY_AND => Prec::AndAnd,
        OR | TILDE_OR => Prec::Or,
        XOR | XOR_TILDE | TILDE_XOR => Prec::Xor,
        AND | TILDE_AND => Prec::And,
        EQEQ | NOT_EQ | EQEQEQ | NOT_EQEQ => Prec::Eq,
        LESS | GREATER | LESS_EQ | GREATER_EQ => Prec::Lt,
        LBITSHIFT | RBITSHIFT | LBITSHIFT_A | RBITSHIFT_A => Prec::Shift,
        PLUS | MINUS => Prec::Plus,
        STAR | SLASH | PERCENT => Prec::Mul,
        STAR_STAR => Prec::StarStar,
        _ => panic!("Unknown operator: {op:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_ranges() {
        assert!(MODEL.is_kw());
        assert!(SUBCKT.is_kw());
        assert!(DC.is_kw() && DC.is_analysis());
        assert!(SAVE.is_kw() && !SAVE.is_save_kw()); // control kw, not a save signal kw
        assert!(CURRENTS.is_kw() && CURRENTS.is_save_kw());
        assert!(M_PI.is_kw() && M_PI.is_builtin_const());
        assert!(SQRT.is_kw() && SQRT.is_builtin_func());
        assert!(!PLUS.is_kw());
    }

    #[test]
    fn ident_predicate() {
        assert!(IDENTIFIER.is_ident());
        assert!(MODEL.is_ident()); // keywords double as identifiers
        assert!(!PLUS.is_ident());
        assert!(!NUMBER.is_ident());
    }

    #[test]
    fn operator_ranges() {
        assert!(PLUS.is_operator() && PLUS.is_unary_operator());
        assert!(MINUS.is_unary_operator());
        assert!(EVENT_OR.is_operator());
        assert!(!STAR.is_unary_operator());
        assert!(!IDENTIFIER.is_operator());
    }

    #[test]
    fn precedence() {
        assert!(prec(STAR_STAR) > prec(STAR));
        assert!(prec(STAR) > prec(PLUS));
        assert_eq!(prec(EVENT_OR), Prec::Logical);
        assert_eq!(prec(LAZY_OR), Prec::Logical);
    }

    #[test]
    fn prec_reduction_ops() {
        // ~& groups with &, ~| with | (matching the fixed Julia prec() arms).
        assert_eq!(prec(TILDE_AND), Prec::And);
        assert_eq!(prec(TILDE_OR), Prec::Or);
    }

    #[test]
    #[should_panic]
    fn prec_panics_on_lexer_only_op() {
        let _ = prec(NOT_IS);
    }
}
