//! A standalone SPICE/Spectre netlist parser producing a lossless rowan CST.
//!
//! Spike scope: the SPICE dialect only, CST (no semantic layer). The public
//! entry point is [`parse_spice`]; [`dump::dump`] renders the canonical tree
//! form used for differential testing against the Julia parser.

pub mod ast;
pub mod dump;
pub mod keywords;
pub mod lexer;
pub mod parser;
pub mod syntax_kind;

// Spectre dialect: separate token layer (token kinds, keyword trie, lexer,
// parser); the rowan CST kind space in `syntax_kind` is shared.
pub mod spectre_keywords;
pub mod spectre_lexer;
pub mod spectre_parser;
pub mod spectre_syntax_kind;

pub use lexer::Dialect;
pub use spectre_parser::StartLang;
pub use syntax_kind::{NetlistLang, SyntaxKind, SyntaxNode, SyntaxToken};

/// Parse SPICE source into a lossless rowan CST (ngspice dialect).
pub fn parse_spice(src: &str) -> SyntaxNode {
    parser::parse(src, Dialect::Ngspice)
}

/// Parse SPICE source under a specific dialect.
pub fn parse_spice_dialect(src: &str, dialect: Dialect) -> SyntaxNode {
    parser::parse(src, dialect)
}

/// Parse Spectre source into a lossless rowan CST (rooted at
/// `SpectreNetlistSource`).
pub fn parse_spectre(src: &str) -> SyntaxNode {
    spectre_parser::parse(src)
}

/// Parse a netlist that may switch dialects via `simulator lang=`, starting in
/// `start_lang` (`.scs` → Spectre, `.cir` → SPICE). The root is always
/// `SpectreNetlistSource`.
pub fn parse_spectre_with(src: &str, start_lang: StartLang) -> SyntaxNode {
    spectre_parser::parse_with(src, start_lang)
}
