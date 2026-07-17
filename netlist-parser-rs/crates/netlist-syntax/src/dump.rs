//! Canonical CST dumper — the Rust side of the differential test.
//!
//! Matches `NyanSpectreNetlistParser.jl/tools/dump_cst.jl`: preorder DFS, one
//! line per node, `<indent><KindLabel> <start>-<end>` where the span is the
//! node's *content* span (leading/trailing trivia excluded), half-open, 0-based
//! byte offsets. Trivia tokens and zero-content nodes are skipped; indentation
//! counts only emitted ancestors.

use crate::syntax_kind::{SyntaxKind, SyntaxNode};
use rowan::{NodeOrToken, TextSize};

/// Content span `[start, end)` of an element: the byte range from its first to
/// its last non-trivia descendant token. `None` if it contains no such token.
fn content_span(elem: &NodeOrToken<SyntaxNode, crate::syntax_kind::SyntaxToken>) -> Option<(u32, u32)> {
    match elem {
        NodeOrToken::Token(t) => {
            let k = t.kind();
            if k.is_trivia() {
                None
            } else {
                let r = t.text_range();
                let (s, e) = (u32::from(r.start()), u32::from(r.end()));
                if e > s {
                    Some((s, e))
                } else {
                    None
                }
            }
        }
        NodeOrToken::Node(n) => {
            let mut lo: Option<TextSize> = None;
            let mut hi: Option<TextSize> = None;
            for t in n.descendants_with_tokens() {
                if let NodeOrToken::Token(t) = t {
                    let k = t.kind();
                    if k.is_trivia() {
                        continue;
                    }
                    let r = t.text_range();
                    if r.end() == r.start() {
                        continue;
                    }
                    lo = Some(lo.map_or(r.start(), |x| x.min(r.start())));
                    hi = Some(hi.map_or(r.end(), |x| x.max(r.end())));
                }
            }
            match (lo, hi) {
                (Some(a), Some(b)) if b > a => Some((u32::from(a), u32::from(b))),
                _ => None,
            }
        }
    }
}

fn kind_of(elem: &NodeOrToken<SyntaxNode, crate::syntax_kind::SyntaxToken>) -> SyntaxKind {
    match elem {
        NodeOrToken::Node(n) => n.kind(),
        NodeOrToken::Token(t) => t.kind(),
    }
}

fn dump_rec(
    elem: NodeOrToken<SyntaxNode, crate::syntax_kind::SyntaxToken>,
    depth: usize,
    out: &mut String,
) {
    // `LiteralExpr`/`NameRef` are transparent expression wrappers: dump their
    // inner token in place, so the output matches Julia's bare terminal (keeps
    // the differential parity while the Rust AST stays uniformly node-based).
    if let NodeOrToken::Node(n) = &elem {
        if matches!(n.kind(), SyntaxKind::LiteralExpr | SyntaxKind::NameRef) {
            for child in n.children_with_tokens() {
                dump_rec(child, depth, out);
            }
            return;
        }
    }
    let span = match content_span(&elem) {
        Some(s) => s,
        None => return, // trivia / zero-content: skip (and, for nodes, its subtree has no content)
    };
    let kind = kind_of(&elem);
    for _ in 0..depth {
        out.push_str("  ");
    }
    out.push_str(kind.dump_label());
    out.push(' ');
    out.push_str(&span.0.to_string());
    out.push('-');
    out.push_str(&span.1.to_string());
    out.push('\n');

    if let NodeOrToken::Node(n) = elem {
        for child in n.children_with_tokens() {
            dump_rec(child, depth + 1, out);
        }
    }
}

/// Produce the canonical dump of a parsed tree.
pub fn dump(root: &SyntaxNode) -> String {
    let mut out = String::new();
    dump_rec(NodeOrToken::Node(root.clone()), 0, &mut out);
    out
}
