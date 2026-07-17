//! Simulator-validated extensions beyond the Julia parser: devices and
//! operators the Julia parser rejects but ngspice/Xyce accept. Validated by:
//! parses with NO Error/Incomplete nodes + lossless CST. The corresponding
//! netlists are checked against the real simulators via
//! tools/validate_ngspice.sh (tests/sim/*.cir) and tools/validate_xyce.sh.

use netlist_syntax::{parse_spice_dialect, Dialect, SyntaxKind};
use rowan::NodeOrToken;

fn clean(src: &str) {
    let tree = parse_spice_dialect(src, Dialect::Ngspice);
    assert_eq!(tree.text().to_string(), src, "not lossless: {src:?}");
    // Also exercise the dumper (covers dump_label for the new node kinds).
    assert!(!netlist_syntax::dump::dump(&tree).is_empty());
    for el in tree.descendants_with_tokens() {
        let kind = match &el {
            NodeOrToken::Node(n) => n.kind(),
            NodeOrToken::Token(t) => t.kind(),
        };
        assert!(
            kind != SyntaxKind::Error && kind != SyntaxKind::Incomplete,
            "unexpected {kind:?} at {:?} in {src:?}",
            el.text_range()
        );
    }
}

/// Assert a specific node kind appears (the device was recognized, not just
/// parsed as something generic).
fn has_kind(src: &str, want: SyntaxKind) {
    let tree = parse_spice_dialect(src, Dialect::Ngspice);
    assert!(
        tree.descendants().any(|n| n.kind() == want),
        "expected a {want:?} node in {src:?}"
    );
}

#[test]
fn extended_devices() {
    // K = mutual inductor, J = JFET, O = LTRA, Z = MESFET, A = XSPICE.
    clean("* t\nL1 1 3 1u\nL2 2 4 1u\nK1 L1 L2 0.9\n");
    has_kind("* t\nK1 L1 L2 0.9\n", SyntaxKind::MutualInductor);
    clean("* t\nJ1 d g s jmod\n");
    has_kind("* t\nJ1 d g s jmod\n", SyntaxKind::JFET);
    clean("* t\nO1 1 0 2 0 omod\n");
    has_kind("* t\nO1 1 0 2 0 omod\n", SyntaxKind::TransmissionLine);
    has_kind("* t\nT1 1 0 2 0 Z0=50 TD=1n\n", SyntaxKind::TransmissionLine);
    clean("* t\nZ1 d g s zmod\n");
    has_kind("* t\nZ1 d g s zmod\n", SyntaxKind::Mesfet);
    clean("* t\nA1 in out gain_block\n");
    has_kind("* t\nA1 in out gain_block\n", SyntaxKind::XspiceDevice);
}

#[test]
fn extended_operators() {
    // Unary ~ / ! and bitwise & | ^ and shifts — accepted by ngspice.
    clean("* t\n.param p = {~1}\n");
    clean("* t\n.param p = {!0}\n");
    clean("* t\n.param p = {2 & 3}\n");
    clean("* t\n.param p = {4 | 1}\n");
    clean("* t\n.param p = {5 ^ 1}\n");
    clean("* t\n.param p = {1 << 2}\n");
    clean("* t\n.param p = {8 >> 1}\n");
    clean("* t\n.param p = {~1 + !0 + (2&3) + (4|1) + (5^1)}\n");
    has_kind("* t\n.param p = {~1}\n", SyntaxKind::UnaryOp);
}
