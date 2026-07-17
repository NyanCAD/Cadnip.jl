//! Parser robustness / error-path coverage. Every input — malformed or edge —
//! must parse without panicking and reconstruct its source exactly (lossless
//! error recovery). These aren't checked against Julia (deep error-recovery
//! `Incomplete` nesting can differ, as documented), only for no-panic + the
//! lossless-tree guarantee, which is what error recovery must uphold.

fn ok(src: &str) {
    let tree = netlist_syntax::parse_spice(src);
    assert_eq!(
        tree.text().to_string(),
        src,
        "lossless round-trip failed for {src:?}"
    );
}

#[test]
fn malformed_and_edge_inputs_are_lossless() {
    let inputs = &[
        // --- incomplete instances (hit each device's Incomplete path) ---
        "* t\nR1 a\n",
        "* t\nR1\n",
        "* t\nC1 a\n",
        "* t\nL1 a b\n",
        "* t\nV1 a\n",
        "* t\nI1 a b\n",
        "* t\nD1 a b\n",
        "* t\nM1 a b c\n",
        "* t\nQ1 a b\n",
        "* t\nB1 a\n",
        "* t\nE1 a b\n",
        "* t\nF1 a b\n",
        "* t\nS1 a b c\n",
        "* t\nX1\n",
        // errors at EOF (no trailing newline → error() ENDMARKER/eol branches)
        "* t\nR1",
        "* t\nR1 a b",
        "* t\n.model",
        // --- incomplete / malformed dot-commands ---
        "* t\n.model foo\n",
        "* t\n.subckt\n",
        "* t\n.subckt s a b\n",             // no .ends before EOF
        "* t\n.param x =\n",
        "* t\n.param\n",
        "* t\n.dc\n",
        "* t\n.dc V1 0 5\n",
        "* t\n.ac\n",
        "* t\n.tran\n",
        "* t\n.ic\n",
        "* t\n.ic v(\n",
        "* t\n.global +\n",
        "* t\n.temp\n",
        "* t\n.width\n",
        "* t\n.options\n",
        "* t\n.include\n",
        "* t\n.hdl\n",
        "* t\n.print\n",
        "* t\n.data\n",
        "* t\n.data blk a\n",
        "* t\n.if\n",
        "* t\n.if (1)\nR1 a b 1k\n",       // no .endif
        "* t\n.lib\n",
        "* t\n.lib mylib\nR1 a b 1k\n",     // no .endl before EOF
        "* t\n.measure\n",
        "* t\n.measure tran\n",
        "* t\n.measure tran nm find\n",
        "* t\n.measure tran nm avg\n",
        "* t\n.FOO\n",
        "* t\n.INVALID_COMMAND arg\n",
        // --- unknown / unsupported instance prefixes ---
        "* t\nz1 a b model\n",
        // --- expression errors ---
        "* t\n.param x = a+\n",
        "* t\n.param x = (a+b\n",
        "* t\n.param x = {a*b\n",
        "* t\n.param x = 'a+b\n",
        "* t\n.param x = f(a,\n",
        "* t\n.param x = a ? b\n",
        "* t\n.param x = [1 2\n",
        // --- valid edge cases (success sub-branches) ---
        "* t\n.param x = [1 2 3]\n",         // array literal
        "* t\n.param x = a ? b : c\n",       // full ternary
        "* t\nV1 a 0 DC=5 AC=1 90\n",        // DC=/AC= with eq + acphase
        "* t\nV1 a 0 PULSE(0 5 1n 1n 1n 5n 10n)\n", // tran source
        "* t\n.dc V1 0 5 1 V2 0 3 1\n",      // two dc commands
        "* t\n.meas tran m WHEN v(a)=1\n",   // measure WHEN
        "* t\n.meas tran m FIND v(a) AT=1n RISE=2\n", // find + at + rise
        "* t\n.meas tran m TRIG v(a) VAL=1 TD=1n TARG v(b) VAL=2\n",
        "* t\nE1 out 0 POLY(2) a 0 b 0 1 2 3\n",
        "* t\nG1 out 0 TABLE {v(a)} (0,0) (1,1)\n",
        "* t\nX1 a b sub p1=1 p2=2\n",
        "* t\nX1 a b p1=1 subname\n",        // model_after
        // --- token-layer + trivia edge cases ---
        "* t\nR1 a b 1k\n\n* comment\nR2 a b 2k\n", // blank + comment between stmts
        "* t\nR1 a b\\\n1k\n",                 // backslash line continuation
        // --- sub-branches that need specific shapes ---
        ".subckt s a b p1=1\nR1 a b 1k\n.ends\n", // subckt with parameters
        "* t\n.model rm r r=1k dev/gauss=0.1\n",   // DevMod with slash/distr
        "* t\n.model rm r r=1k dev=0.1\n",         // DevMod without slash
        "* t\n.param y = f(a, b, c)\n",            // multi-arg function call
        "* t\nV1 vcc 0 DC +",                      // error extending to EOF
        "* t\nG1 out 0 TABLE {v(a)}=(0,0)(1,1)\n", // TABLE with '='
        "* t\nQ1 c b e s m\n",                     // BJT with substrate node
        "* t\nQ1 c b e s t m\n",                   // BJT with substrate + thermal
    ];
    for src in inputs {
        ok(src);
    }
}

/// Expression-operator coverage for the Pratt parser and `prec()`. These use
/// operators that aren't valid ngspice but DO parse (prec is defined), so they
/// exercise `parse_binop`/`prec` arms; checked for no-panic + losslessness only.
#[test]
fn expression_operators_parse_losslessly() {
    let exprs = &[
        "a+b", "a-b", "a*b", "a/b", "a%b", "a**b",
        "a<b", "a>b", "a<=b", "a>=b", "a==b", "a!=b", "a===b",
        "a<<b", "a>>b", "a<<<b",
        "a&b", "a|b", "a^b", "a^~b", "a~^b",
        "a&&b", "a||b",
        // mixed precedence → recursion + opterm branches
        "a+b*c", "a*b+c", "a+b+c", "a-b-c", "a*b/c", "a**b**c",
        "a+b*c-d/e", "a<b+c", "a&&b||c", "a==b&&c",
        "-a+b", "+a-b", "a+-b",
    ];
    for e in exprs {
        ok(&format!("* t\n.param x = {{{e}}}\n"));
    }
}

/// Spectre inputs that made the *Julia* reference parser throw before it was
/// fixed (a lexer `~|` typo, a `mod` undefined-variable, `parse_primary`
/// `unreachable`, and non-`@trynext` control statements hitting `convert`
/// errors). Julia now recovers into `Error`/`Incomplete` nodes on all of these,
/// so they live in the differential corpus proper; this test additionally pins
/// that the Rust port never panics and stays lossless on them.
#[test]
fn spectre_formerly_crashing_inputs_are_lossless() {
    let cases = [
        "ic node value\n",              // ic parameter missing '='
        "nodeset x=\n",                 // parameter value missing
        "save foo:\n",                  // dangling save-signal ':'
        "ahdl_include device.va\n",     // ahdl_include filename not a string
        "include\n",                    // include with no filename
        "myalt alter param=\n",         // control statement, missing value
        "tr1 tran stop=\n",             // analysis, missing value
        "parameters n=a~|b~|c\n",       // '~|' reduction-or, now a real operator
    ];
    for src in cases {
        let tree = netlist_syntax::parse_spectre(src);
        assert_eq!(
            tree.text().to_string(),
            src,
            "\nlossless round-trip failed for {src:?}"
        );
    }
}
