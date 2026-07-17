//! A standalone SPICE/Spectre netlist parser producing a lossless rowan CST.
//!
//! Spike scope: the SPICE dialect only, CST (no semantic layer). The public
//! entry point is [`parse_spice`]; [`dump::dump`] renders the canonical tree
//! form used for differential testing against the Julia parser.

pub mod dump;
pub mod keywords;
pub mod lexer;
pub mod parser;
pub mod syntax_kind;

pub use lexer::Dialect;
pub use syntax_kind::{NetlistLang, SyntaxKind, SyntaxNode, SyntaxToken};

/// Parse SPICE source into a lossless rowan CST (ngspice dialect).
pub fn parse_spice(src: &str) -> SyntaxNode {
    parser::parse(src, Dialect::Ngspice)
}

/// Parse SPICE source under a specific dialect.
pub fn parse_spice_dialect(src: &str, dialect: Dialect) -> SyntaxNode {
    parser::parse(src, dialect)
}
