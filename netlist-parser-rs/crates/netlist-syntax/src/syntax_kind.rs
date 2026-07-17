//! Token kinds and syntax-tree kinds.
//!
//! `TokenKind` is a faithful transcription of the Julia `Tokens.Kind` `@enum`
//! (`NyanSpectreNetlistParser.jl/src/SPICE/tokenize/token_kinds.jl`), including
//! the `begin_*`/`end_*` marker values so the range-based classifier predicates
//! (`is_kw`, `is_operator`, ...) port over as plain ordinal comparisons.
//!
//! A handful of operator kinds are emitted by the Julia lexer but were missing
//! from its enum (`CASSIGN`, `RRBITSHIFT_A`, `NOT_IS`, `TILD_OR`) — referencing
//! them in Julia would `UndefVarError` at runtime, but they only occur for
//! exotic operators that never appear in real SPICE. We give them real variants
//! (inside the ops range) so the lexer is total; `prec()` still panics on them,
//! matching Julia's `error("Unknown operator")`.
//!
//! `SyntaxKind` is the kind space of the rowan CST: trivia, terminal *forms*
//! (the `Terminal` structs in `forms.jl` — `Notation`, `Keyword`, ...), and
//! node forms (the `AbstractASTNode` struct names). The dump label of a node is
//! exactly the Julia form struct name (see `dump_label`).

#![allow(non_camel_case_types)]

/// Lexer token kind. Mirrors `Tokens.Kind` in Julia, in the same order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u16)]
pub enum TokenKind {
    ENDMARKER,
    COMMENT,
    WHITESPACE,
    NEWLINE,
    ESCD_NEWLINE,
    TITLE_LINE,
    JULIA_ESCAPE_BEGIN,
    JULIA_ESCAPE,

    begin_identifiers,
    IDENTIFIER,
    IDENTIFIER_BEHAVIORAL,
    IDENTIFIER_CAPACITOR,
    IDENTIFIER_DIODE,
    IDENTIFIER_VOLTAGE_CONTROLLED_VOLTAGE,
    IDENTIFIER_VOLTAGE_CONTROLLED_CURRENT,
    IDENTIFIER_CURRENT_CONTROLLED_VOLTAGE,
    IDENTIFIER_CURRENT_CONTROLLED_CURRENT,
    IDENTIFIER_CURRENT,
    IDENTIFIER_JFET,
    IDENTIFIER_HFET_MESA,
    IDENTIFIER_LINEAR_MUTUAL_INDUCTOR,
    IDENTIFIER_LINEAR_INDUCTOR,
    IDENTIFIER_MOSFET,
    IDENTIFIER_OSDI,
    IDENTIFIER_PORT,
    IDENTIFIER_BIPOLAR_TRANSISTOR,
    IDENTIFIER_RESISTOR,
    IDENTIFIER_S_PARAMETER_ELEMENT,
    IDENTIFIER_SWITCH,
    IDENTIFIER_VOLTAGE,
    IDENTIFIER_TRANSMISSION_LINE,
    IDENTIFIER_SUBCIRCUIT_CALL,
    IDENTIFIER_UNKNOWN_INSTANCE,
    IDENTIFIER_XSPICE,
    end_identifiers,

    BASE_SPEC,
    HEX_BASE_SPEC,
    BACKTICK,
    COMMA,
    COLON,
    SEMICOLON,
    DOT,
    AT_SIGN,
    HASH,
    DOLLAR,
    CONDITIONAL,
    PRIME,

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
    LATTR,
    RATTR,
    end_delimiters,

    begin_keywords,
    begin_analyses,
    DC,
    AC,
    TRAN,
    IC,
    TF,
    NOISE,
    end_analyses,
    begin_output,
    PRINT,
    PLOT,
    MEASURE,
    end_output,
    IF,
    ELSE,
    ELSEIF,
    ENDIF,
    MODEL,
    LIB,
    INCLUDE,
    SUBCKT,
    END,
    ENDL,
    ENDS,
    PARAMETERS,
    CSPARAM,
    OPTIONS,
    TEMP,
    WIDTH,
    GLOBAL,
    DEV,
    LOT,
    SIMULATOR,
    LANG,
    SPECTRE,
    SPICE,
    TITLE,
    DATA,
    ENDDATA,
    ON,
    OFF,
    HDL,
    FIND,
    DERIV,
    WHEN,
    AT,
    TD,
    RISE,
    FALL,
    CROSS,
    LAST,
    AVG,
    MAX,
    MIN,
    PP,
    RMS,
    INTEG,
    TRIG,
    VAL,
    TARG,
    begin_sources,
    PULSE,
    SIN,
    EXP,
    PWL,
    SFFM,
    AM,
    TRNOISE,
    TRRANDOM,
    POLY,
    TABLE,
    end_sources,
    end_keywords,

    begin_ops,
    OP,
    PLUS,
    MINUS,
    STAR,
    SLASH,
    STAR_STAR,
    PERCENT,
    GREATER,
    LESS,
    GREATER_EQ,
    LESS_EQ,
    EQ,
    EQEQ,
    NOT_EQ,
    EQEQEQ,
    NOT_EQEQ,
    NOT,
    LAZY_AND,
    LAZY_OR,
    TILDE,
    AND,
    OR,
    XOR,
    XOR_TILDE,
    TILDE_XOR,
    TILDE_AND,
    TILDE_OR,
    LBITSHIFT,
    RBITSHIFT,
    LBITSHIFT_A,
    RBITSHIFT_A,
    EVENT_OR,
    // Emitted by the lexer but absent from the Julia enum (see module docs).
    CASSIGN,
    RRBITSHIFT_A,
    NOT_IS,
    TILD_OR,
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

    /// `is_ident`: an identifier token *or* a keyword (SPICE lets keywords act
    /// as identifiers, e.g. node names).
    pub fn is_ident(self) -> bool {
        (begin_identifiers.ord() < self.ord() && self.ord() < end_identifiers.ord())
            || self.is_kw()
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

    pub fn is_source_type(self) -> bool {
        begin_sources.ord() < self.ord() && self.ord() < end_sources.ord()
    }

    pub fn is_analysis(self) -> bool {
        begin_analyses.ord() < self.ord() && self.ord() < end_analyses.ord()
    }

    /// `NyanLexers.is_triv`
    pub fn is_triv(self) -> bool {
        self == COMMENT || self == WHITESPACE || self == NEWLINE
    }

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

/// `prec(opkind)` from parse.jl. Panics on a non-operator (matching Julia's
/// `error("Unknown operator")`).
pub fn prec(op: TokenKind) -> Prec {
    match op {
        LAZY_OR | EVENT_OR => Prec::Logical,
        LAZY_AND => Prec::AndAnd,
        OR => Prec::Or,
        XOR | XOR_TILDE | TILDE_XOR => Prec::Xor,
        AND => Prec::And,
        EQ | EQEQ | NOT_EQ | EQEQEQ | NOT_EQEQ => Prec::Eq,
        LESS | GREATER | LESS_EQ | GREATER_EQ => Prec::Lt,
        LBITSHIFT | RBITSHIFT | LBITSHIFT_A | RBITSHIFT_A => Prec::Shift,
        PLUS | MINUS => Prec::Plus,
        STAR | SLASH | PERCENT => Prec::Mul,
        STAR_STAR => Prec::StarStar,
        _ => panic!("Unknown operator: {op:?}"),
    }
}

/// Syntax-tree kind: the rowan node/token kind space.
///
/// Trivia + terminal forms + node forms. The `dump_label` of each equals the
/// corresponding Julia form struct name (for terminals and nodes); trivia are
/// never emitted in dumps.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u16)]
pub enum SyntaxKind {
    // --- trivia (never appear in a dump) ---
    Whitespace,
    Comment,
    Newline,
    EscdNewline,
    /// `+` line-continuation, consumed as trivia by the parser token layer.
    Continuation,
    /// Tokens swallowed by error recovery (`extend_to_line_end`); never dumped.
    Skipped,

    // --- terminal forms (forms.jl `Terminal` structs) ---
    Keyword,
    Operator,
    Identifier,
    SystemIdentifier,
    Notation,
    BuiltinFunc,
    BuiltinConst,
    Literal,
    StringLiteral,
    NumberLiteral,
    JuliaEscapeBody,
    Error,

    // --- node forms (forms.jl `AbstractASTNode` structs) ---
    SPICENetlistSource,
    Incomplete,
    Title,
    EndStatement,
    EndlStatement,
    ParamStatement,
    TempStatement,
    GlobalStatement,
    Model,
    Subckt,
    Parameter,
    DevMod,
    NodeName,
    SubNode,
    HierarchialNode,
    Resistor,
    Capacitor,
    Inductor,
    Voltage,
    Current,
    Behavioral,
    Diode,
    MOSFET,
    BipolarTransistor,
    SubcktCall,
    OSDIDevice,
    ControlledSource,
    VoltageControl,
    CurrentControl,
    PolyControl,
    TableControl,
    Switch,
    SParameterElement,
    JuliaDevice,
    JuliaEscape,
    DCSource,
    ACSource,
    TranSource,
    DCStatement,
    DCCommand,
    ACStatement,
    ACCommand,
    IncludeStatement,
    HDLStatement,
    LibStatement,
    LibInclude,
    OptionStatement,
    WidthStatement,
    UnaryOp,
    BinaryExpression,
    TernaryExpr,
    Brace,
    Parens,
    Prime,
    Square,
    FunctionCall,
    FunctionArgs,
    /// Wrapper node around a literal token used as an expression (rust-analyzer
    /// `Literal` pattern). Dumped transparently as its inner token so the
    /// differential stays byte-exact against Julia's bare terminal.
    LiteralExpr,
    /// Wrapper node around a bare identifier used as an expression (rust-analyzer
    /// `PathExpr`/`NameRef` pattern). Dumped transparently as its inner token.
    NameRef,

    // Analysis / dot-command + measure forms (breadth phase).
    Tran,
    PrintStatement,
    ICStatement,
    ICEntry,
    WildCard,
    Coloned,
    DataStatement,
    IfBlock,
    IfElseCase,
    Condition,
    When,
    At,
    RiseFallCross,
    TD_,
    FindDerivParam,
    MeasurePointStatement,
    MeasureRangeStatement,
    AvgMaxMinPPRmsInteg,
    Val_,
    TrigTarg,

    // Xyce-dialect dot-commands (not present in the Julia parser; validated
    // against the Xyce simulator instead of the differential harness).
    StepStatement,
    FuncStatement,
    GlobalParamStatement,
    NodeSetStatement,
    NodeSetEntry,

    // Device types unimplemented in the Julia parser but accepted by
    // ngspice/Xyce (validated against those simulators).
    MutualInductor,
    JFET,
    TransmissionLine,
    Mesfet,
    XspiceDevice,

    #[doc(hidden)]
    __Last,
}

impl SyntaxKind {
    /// True for trivia kinds, which are omitted from the canonical dump.
    pub fn is_trivia(self) -> bool {
        matches!(
            self,
            SyntaxKind::Whitespace
                | SyntaxKind::Comment
                | SyntaxKind::Newline
                | SyntaxKind::EscdNewline
                | SyntaxKind::Continuation
                | SyntaxKind::Skipped
        )
    }

    /// The canonical dump label: the Julia form struct name. For all node and
    /// terminal forms this is just the variant name; only trivia (never dumped)
    /// have no Julia analogue.
    pub fn dump_label(self) -> &'static str {
        use SyntaxKind::*;
        match self {
            Whitespace => "Whitespace",
            Comment => "Comment",
            Newline => "Newline",
            EscdNewline => "EscdNewline",
            Continuation => "Continuation",
            Skipped => "Skipped",
            Keyword => "Keyword",
            Operator => "Operator",
            Identifier => "Identifier",
            SystemIdentifier => "SystemIdentifier",
            Notation => "Notation",
            BuiltinFunc => "BuiltinFunc",
            BuiltinConst => "BuiltinConst",
            Literal => "Literal",
            StringLiteral => "StringLiteral",
            NumberLiteral => "NumberLiteral",
            JuliaEscapeBody => "JuliaEscapeBody",
            Error => "Error",
            SPICENetlistSource => "SPICENetlistSource",
            Incomplete => "Incomplete",
            Title => "Title",
            EndStatement => "EndStatement",
            EndlStatement => "EndlStatement",
            ParamStatement => "ParamStatement",
            TempStatement => "TempStatement",
            GlobalStatement => "GlobalStatement",
            Model => "Model",
            Subckt => "Subckt",
            Parameter => "Parameter",
            DevMod => "DevMod",
            NodeName => "NodeName",
            SubNode => "SubNode",
            HierarchialNode => "HierarchialNode",
            Resistor => "Resistor",
            Capacitor => "Capacitor",
            Inductor => "Inductor",
            Voltage => "Voltage",
            Current => "Current",
            Behavioral => "Behavioral",
            Diode => "Diode",
            MOSFET => "MOSFET",
            BipolarTransistor => "BipolarTransistor",
            SubcktCall => "SubcktCall",
            OSDIDevice => "OSDIDevice",
            ControlledSource => "ControlledSource",
            VoltageControl => "VoltageControl",
            CurrentControl => "CurrentControl",
            PolyControl => "PolyControl",
            TableControl => "TableControl",
            Switch => "Switch",
            SParameterElement => "SParameterElement",
            JuliaDevice => "JuliaDevice",
            JuliaEscape => "JuliaEscape",
            DCSource => "DCSource",
            ACSource => "ACSource",
            TranSource => "TranSource",
            DCStatement => "DCStatement",
            DCCommand => "DCCommand",
            ACStatement => "ACStatement",
            ACCommand => "ACCommand",
            IncludeStatement => "IncludeStatement",
            HDLStatement => "HDLStatement",
            LibStatement => "LibStatement",
            LibInclude => "LibInclude",
            OptionStatement => "OptionStatement",
            WidthStatement => "WidthStatement",
            UnaryOp => "UnaryOp",
            BinaryExpression => "BinaryExpression",
            TernaryExpr => "TernaryExpr",
            Brace => "Brace",
            Parens => "Parens",
            Prime => "Prime",
            Square => "Square",
            FunctionCall => "FunctionCall",
            FunctionArgs => "FunctionArgs",
            LiteralExpr => "LiteralExpr",
            NameRef => "NameRef",
            Tran => "Tran",
            PrintStatement => "PrintStatement",
            ICStatement => "ICStatement",
            ICEntry => "ICEntry",
            WildCard => "WildCard",
            Coloned => "Coloned",
            DataStatement => "DataStatement",
            IfBlock => "IfBlock",
            IfElseCase => "IfElseCase",
            Condition => "Condition",
            When => "When",
            At => "At",
            RiseFallCross => "RiseFallCross",
            TD_ => "TD_",
            FindDerivParam => "FindDerivParam",
            MeasurePointStatement => "MeasurePointStatement",
            MeasureRangeStatement => "MeasureRangeStatement",
            AvgMaxMinPPRmsInteg => "AvgMaxMinPPRmsInteg",
            Val_ => "Val_",
            TrigTarg => "TrigTarg",
            StepStatement => "StepStatement",
            FuncStatement => "FuncStatement",
            GlobalParamStatement => "GlobalParamStatement",
            NodeSetStatement => "NodeSetStatement",
            NodeSetEntry => "NodeSetEntry",
            MutualInductor => "MutualInductor",
            JFET => "JFET",
            TransmissionLine => "TransmissionLine",
            Mesfet => "Mesfet",
            XspiceDevice => "XspiceDevice",
            __Last => "__Last",
        }
    }
}

impl From<SyntaxKind> for rowan::SyntaxKind {
    fn from(k: SyntaxKind) -> Self {
        rowan::SyntaxKind(k as u16)
    }
}

/// The rowan `Language` for netlist CSTs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NetlistLang {}

impl rowan::Language for NetlistLang {
    type Kind = SyntaxKind;

    fn kind_from_raw(raw: rowan::SyntaxKind) -> SyntaxKind {
        assert!(raw.0 < SyntaxKind::__Last as u16, "raw kind out of range");
        // Safe: `SyntaxKind` is `repr(u16)` with contiguous discriminants
        // `0..=__Last`, and we just bounds-checked.
        unsafe { std::mem::transmute::<u16, SyntaxKind>(raw.0) }
    }

    fn kind_to_raw(kind: SyntaxKind) -> rowan::SyntaxKind {
        kind.into()
    }
}

pub type SyntaxNode = rowan::SyntaxNode<NetlistLang>;
pub type SyntaxToken = rowan::SyntaxToken<NetlistLang>;
pub type SyntaxElement = rowan::SyntaxElement<NetlistLang>;
