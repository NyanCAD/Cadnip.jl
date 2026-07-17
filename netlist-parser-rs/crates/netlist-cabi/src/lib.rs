//! C ABI over `netlist-syntax` — the lowest-common-denominator, ABI-stable
//! `.so` any language can `dlopen` (C, Julia, Go, cffi, ...).
//!
//! An `NlTree` owns the source buffer + green tree; `NlNode` handles are opaque,
//! independently-owned boxes over a rowan element (node or token) — each must be
//! released with `nl_node_free`. Byte spans are `[start, end)` full spans
//! (including trivia); `nl_node_kind` returns the `SyntaxKind` discriminant.

use netlist_syntax::syntax_kind::{SyntaxElement, SyntaxNode};
use netlist_syntax::Dialect;
use rowan::NodeOrToken;
use std::os::raw::c_char;

/// Owns the source and parsed tree, plus precomputed error spans.
pub struct NlTree {
    root: SyntaxNode,
    errors: Vec<[u32; 2]>,
    // Keep the source alive for the tree's lifetime (rowan tokens copy their
    // text into the green tree, but we keep it for parity with the plan's API).
    _src: String,
}

/// Opaque handle to a tree element (node or token).
pub struct NlNode(SyntaxElement);

fn dialect_from(d: u32) -> Dialect {
    match d {
        1 => Dialect::Hspice,
        2 => Dialect::Pspice,
        3 => Dialect::Xyce,
        _ => Dialect::Ngspice,
    }
}

/// Parse SPICE source of `len` bytes. `dialect`: 0=ngspice, 1=hspice, 2=pspice,
/// 3=xyce. Returns null on invalid UTF-8 or a null pointer.
///
/// # Safety
/// `src` must point to at least `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn nl_parse_spice(src: *const c_char, len: usize, dialect: u32) -> *mut NlTree {
    if src.is_null() {
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(src as *const u8, len);
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => return std::ptr::null_mut(),
    };
    let root = netlist_syntax::parser::parse(&s, dialect_from(dialect));
    let mut errors = Vec::new();
    for el in root.descendants_with_tokens() {
        if let NodeOrToken::Token(t) = el {
            if t.kind() == netlist_syntax::SyntaxKind::Error {
                let r = t.text_range();
                errors.push([u32::from(r.start()), u32::from(r.end())]);
            }
        }
    }
    Box::into_raw(Box::new(NlTree { root, errors, _src: s }))
}

/// Free a tree. Safe to call with null.
///
/// # Safety
/// `tree` must be a pointer returned by `nl_parse_spice`, freed at most once.
#[no_mangle]
pub unsafe extern "C" fn nl_tree_free(tree: *mut NlTree) {
    if !tree.is_null() {
        drop(Box::from_raw(tree));
    }
}

/// Get the root node handle. Caller must `nl_node_free` it.
///
/// # Safety
/// `tree` must be a valid `NlTree` pointer.
#[no_mangle]
pub unsafe extern "C" fn nl_tree_root(tree: *const NlTree) -> *mut NlNode {
    let tree = &*tree;
    Box::into_raw(Box::new(NlNode(NodeOrToken::Node(tree.root.clone()))))
}

/// Release a node handle. Safe to call with null.
///
/// # Safety
/// `node` must be a pointer returned by `nl_tree_root`/`nl_node_child`, freed
/// at most once.
#[no_mangle]
pub unsafe extern "C" fn nl_node_free(node: *mut NlNode) {
    if !node.is_null() {
        drop(Box::from_raw(node));
    }
}

/// The element's `SyntaxKind` discriminant.
///
/// # Safety
/// `node` must be a valid `NlNode` pointer.
#[no_mangle]
pub unsafe extern "C" fn nl_node_kind(node: *const NlNode) -> u16 {
    let k = match &(*node).0 {
        NodeOrToken::Node(n) => n.kind(),
        NodeOrToken::Token(t) => t.kind(),
    };
    k as u16
}

/// Number of direct children (nodes and tokens). Tokens have 0.
///
/// # Safety
/// `node` must be a valid `NlNode` pointer.
#[no_mangle]
pub unsafe extern "C" fn nl_node_child_count(node: *const NlNode) -> usize {
    match &(*node).0 {
        NodeOrToken::Node(n) => n.children_with_tokens().count(),
        NodeOrToken::Token(_) => 0,
    }
}

/// The `i`th child handle, or null if out of range. Caller must `nl_node_free`.
///
/// # Safety
/// `node` must be a valid `NlNode` pointer.
#[no_mangle]
pub unsafe extern "C" fn nl_node_child(node: *const NlNode, i: usize) -> *mut NlNode {
    match &(*node).0 {
        NodeOrToken::Node(n) => match n.children_with_tokens().nth(i) {
            Some(child) => Box::into_raw(Box::new(NlNode(child))),
            None => std::ptr::null_mut(),
        },
        NodeOrToken::Token(_) => std::ptr::null_mut(),
    }
}

/// Write the element's full byte span `[start, end)`.
///
/// # Safety
/// `node`, `start`, `end` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn nl_node_span(node: *const NlNode, start: *mut u32, end: *mut u32) {
    let r = match &(*node).0 {
        NodeOrToken::Node(n) => n.text_range(),
        NodeOrToken::Token(t) => t.text_range(),
    };
    *start = u32::from(r.start());
    *end = u32::from(r.end());
}

/// Copy the element's source text into `buf` (up to `cap` bytes, no NUL).
/// Returns the full text length; if it exceeds `cap`, the text was truncated.
///
/// # Safety
/// `node` must be valid; `buf` must point to at least `cap` writable bytes
/// (or be null when `cap == 0`).
#[no_mangle]
pub unsafe extern "C" fn nl_node_text(node: *const NlNode, buf: *mut c_char, cap: usize) -> usize {
    let text = match &(*node).0 {
        NodeOrToken::Node(n) => n.text().to_string(),
        NodeOrToken::Token(t) => t.text().to_string(),
    };
    let bytes = text.as_bytes();
    let n = bytes.len().min(cap);
    if n > 0 && !buf.is_null() {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
    }
    bytes.len()
}

/// Number of `Error` leaves in the tree.
///
/// # Safety
/// `tree` must be a valid `NlTree` pointer.
#[no_mangle]
pub unsafe extern "C" fn nl_tree_error_count(tree: *const NlTree) -> usize {
    (*tree).errors.len()
}

/// Write the `i`th error's span `[start, end)`. Returns false if out of range.
///
/// # Safety
/// `tree`, `start`, `end` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn nl_tree_error(
    tree: *const NlTree,
    i: usize,
    start: *mut u32,
    end: *mut u32,
) -> bool {
    match (&(*tree).errors).get(i) {
        Some(&[s, e]) => {
            *start = s;
            *end = e;
            true
        }
        None => false,
    }
}
