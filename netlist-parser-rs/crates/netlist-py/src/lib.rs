//! PyO3 Python bindings for the `netlist-syntax` SPICE parser.
//!
//! Exposes a single `#[pyclass] Node` that wraps any rowan element (node or
//! token), and two module-level functions:
//!   - `parse_spice(src: str) -> Node`  — parse and return the CST root.
//!   - `errors(root: Node) -> list[tuple[int,int]]` — collect Error-token spans.

use netlist_syntax::syntax_kind::{SyntaxElement, SyntaxKind};
use pyo3::prelude::*;
use rowan::NodeOrToken;

/// A single CST element — either an inner node or a leaf token.
/// Both are represented uniformly; tokens have empty `.children`.
///
/// `unsendable` because rowan's `SyntaxNode`/`SyntaxToken` use `Rc`-backed
/// internals that are not `Send`/`Sync`.
#[pyclass(unsendable)]
pub struct Node {
    inner: SyntaxElement,
}

#[pymethods]
impl Node {
    /// Syntax kind label (e.g. `"SPICENetlistSource"`, `"Resistor"`, `"Identifier"`).
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            NodeOrToken::Node(n) => n.kind().dump_label(),
            NodeOrToken::Token(t) => t.kind().dump_label(),
        }
    }

    /// Full source text covered by this element (including descendant trivia).
    #[getter]
    fn text(&self) -> String {
        match &self.inner {
            NodeOrToken::Node(n) => n.text().to_string(),
            NodeOrToken::Token(t) => t.text().to_string(),
        }
    }

    /// Byte span `(start, end)` — half-open, matching `text_range()`.
    #[getter]
    fn span(&self) -> (u32, u32) {
        let r = match &self.inner {
            NodeOrToken::Node(n) => n.text_range(),
            NodeOrToken::Token(t) => t.text_range(),
        };
        (u32::from(r.start()), u32::from(r.end()))
    }

    /// Direct children (nodes and tokens). Tokens return an empty list.
    #[getter]
    fn children(&self) -> Vec<Node> {
        match &self.inner {
            NodeOrToken::Node(n) => n
                .children_with_tokens()
                .map(|el| Node { inner: el })
                .collect(),
            NodeOrToken::Token(_) => vec![],
        }
    }

    /// True when this element is trivia (whitespace, comment, newline, …).
    #[getter]
    fn is_trivia(&self) -> bool {
        match &self.inner {
            NodeOrToken::Node(n) => n.kind().is_trivia(),
            NodeOrToken::Token(t) => t.kind().is_trivia(),
        }
    }

    /// True when this element has kind `Error`.
    #[getter]
    fn is_error(&self) -> bool {
        let kind = match &self.inner {
            NodeOrToken::Node(n) => n.kind(),
            NodeOrToken::Token(t) => t.kind(),
        };
        kind == SyntaxKind::Error
    }

    fn __repr__(&self) -> String {
        let (start, end) = self.span();
        format!("Node({}, {}..{})", self.kind(), start, end)
    }
}

/// Recursively collect `Error`-token spans from a CST element.
fn collect_errors(el: &SyntaxElement, out: &mut Vec<(u32, u32)>) {
    match el {
        NodeOrToken::Token(t) => {
            if t.kind() == SyntaxKind::Error {
                let r = t.text_range();
                out.push((u32::from(r.start()), u32::from(r.end())));
            }
        }
        NodeOrToken::Node(n) => {
            for child in n.children_with_tokens() {
                collect_errors(&child, out);
            }
        }
    }
}

/// Parse a SPICE netlist string and return the CST root node.
#[pyfunction]
#[pyo3(name = "parse_spice")]
fn parse_spice_py(src: &str) -> Node {
    let root = netlist_syntax::parse_spice(src);
    Node {
        inner: NodeOrToken::Node(root),
    }
}

/// Walk a CST root and return all `Error`-token spans as `(start, end)` tuples.
#[pyfunction]
fn errors(root: &Node) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    collect_errors(&root.inner, &mut result);
    result
}

#[pymodule]
fn netlist_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_spice_py, m)?)?;
    m.add_function(wrap_pyfunction!(errors, m)?)?;
    m.add_class::<Node>()?;
    Ok(())
}
