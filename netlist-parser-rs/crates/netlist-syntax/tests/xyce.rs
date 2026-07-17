//! Xyce-dialect support. These netlists are valid Xyce (verified with the Xyce
//! simulator's `-syntax` check via tools/validate_xyce.sh) but use constructs
//! the Julia parser rejects (.step/.func/.global_param/.nodeset, Y=OSDI), so
//! they're validated here by: parses with NO Error/Incomplete nodes, and the
//! CST is lossless (`tree.text() == source`).

use netlist_syntax::{parse_spice_dialect, Dialect, SyntaxKind};
use rowan::NodeOrToken;

/// Assert the tree has no error/incomplete nodes and reconstructs its source.
fn clean_xyce(src: &str) {
    let tree = parse_spice_dialect(src, Dialect::Xyce);
    assert_eq!(tree.text().to_string(), src, "not lossless: {src:?}");
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

#[test]
fn xyce_dot_commands() {
    // .step (param sweep), .func, .global_param, .nodeset — all Xyce-valid,
    // none accepted by the Julia parser.
    clean_xyce("* t\n.global_param tc=27\n");
    clean_xyce("* t\n.step rval 1k 5k 1k\n");
    clean_xyce("* t\n.step LIN rval 0 5 10\n");
    clean_xyce("* t\n.step rval LIST 1k 2k 5k\n");
    clean_xyce("* t\n.func half(x) {x/2}\n");
    clean_xyce("* t\n.func f(a,b) {a+b}\n");
    clean_xyce("* t\n.nodeset V(out)=0\n");
}

#[test]
fn xyce_full_netlist() {
    // A representative Xyce netlist exercising the dialect end-to-end.
    let src = "Xyce feature sweep\n\
        .param rval=1k\n\
        .global_param tc=27\n\
        .func half(x) {x/2}\n\
        V1 in 0 DC 1 AC 1\n\
        R1 in out {rval}\n\
        C1 out 0 1u\n\
        M1 d g s b nmos w=1u l=0.18u\n\
        .model nmos nmos level=1 vto=0.7\n\
        .subckt buf a y\n\
        Rb a y 1k\n\
        .ends\n\
        Xb in mid buf\n\
        .step rval 1k 5k 1k\n\
        .tran 1u 1m\n\
        .op\n\
        .print tran format=csv V(out) I(V1)\n\
        .measure tran vmax MAX V(out)\n\
        .options timeint reltol=1e-4\n\
        .nodeset V(out)=0\n\
        .ic V(out)=0\n\
        .end\n";
    clean_xyce(src);
}

#[test]
fn xyce_y_device_is_osdi() {
    // In the Xyce dialect a `Y…` device lexes as OSDI and parses like a subckt
    // call (name, nodes, model) rather than erroring.
    let tree = parse_spice_dialect("* t\nYGENEXT g1 a b osdimod\n", Dialect::Xyce);
    assert!(
        tree.descendants().any(|n| n.kind() == SyntaxKind::OSDIDevice),
        "expected an OSDIDevice node for the Y device"
    );
}
