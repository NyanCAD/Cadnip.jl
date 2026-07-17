//! Typed AST accessor layer for the **Spectre** dialect — the rust-analyzer
//! `ast` pattern, mirroring [`crate::ast`] for SPICE.
//!
//! Why a separate module? The rowan CST kind space is *shared* between SPICE
//! and Spectre (see `syntax_kind.rs`), so several kinds (`Model`, `Parameter`,
//! `Subckt`, …) are reused by both dialects — but their **child layout differs**
//! (Spectre models carry bare `Identifier` name/type tokens; SPICE models carry
//! a `HierarchialNode`). The SPICE accessors in [`crate::ast`] therefore return
//! the wrong thing on a Spectre tree. This module provides Spectre-shaped
//! newtypes for those kinds, kept distinct by living in their own module.
//!
//! **Expressions are captured as verbatim source text.** Spectre emits scalar
//! expression leaves as *bare tokens* (a number is a bare `NumberLiteral`, an
//! identifier a bare `Identifier`), unlike SPICE which wraps them in
//! `LiteralExpr`/`NameRef` nodes. Rather than build a second typed `Expr` enum
//! for the bare-token grammar, parameter/condition/value accessors return the
//! RHS source text (`String`). This matches how downstream consumers (VACASK,
//! circulax) treat parameters — symbolic, re-parsed by their own expression
//! engine — and keeps this layer thin.

use crate::ast::{ast_node, support, AstNode};
use crate::syntax_kind::{SyntaxKind, SyntaxNode, SyntaxToken};
use rowan::NodeOrToken;

// --- shared helpers (Spectre-specific, bare-token aware) ---

/// The `n`th direct child token of a given kind.
fn nth_token(parent: &SyntaxNode, kind: SyntaxKind, n: usize) -> Option<SyntaxToken> {
    parent
        .children_with_tokens()
        .filter_map(|e| e.into_token())
        .filter(|t| t.kind() == kind)
        .nth(n)
}

/// Source text of the first non-trivia element following the `=` (`Notation`)
/// token — the RHS of a `name = value` construct. Handles Spectre's bare-token
/// expression leaves as well as wrapper expression nodes.
fn value_text_after_eq(parent: &SyntaxNode) -> Option<String> {
    element_text_after(parent, |el| {
        matches!(el, NodeOrToken::Token(t) if t.kind() == SyntaxKind::Notation && t.text() == "=")
    })
}

/// Source text of the first non-trivia element appearing after the first
/// element matching `is_sep`. `None` if no separator or nothing follows.
fn element_text_after(
    parent: &SyntaxNode,
    is_sep: impl Fn(&NodeOrToken<SyntaxNode, SyntaxToken>) -> bool,
) -> Option<String> {
    let mut seen = false;
    for el in parent.children_with_tokens() {
        if !seen {
            if is_sep(&el) {
                seen = true;
            }
            continue;
        }
        if el.kind().is_trivia() {
            continue;
        }
        return Some(element_text(&el));
    }
    None
}

fn element_text(el: &NodeOrToken<SyntaxNode, SyntaxToken>) -> String {
    match el {
        NodeOrToken::Node(n) => n.text().to_string(),
        NodeOrToken::Token(t) => t.text().to_string(),
    }
}

// --- root ---

ast_node!(SpectreNetlistSource);
impl SpectreNetlistSource {
    /// All top-level statement nodes (trivia excluded).
    pub fn statements(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.0.children()
    }
}

// --- node names ---

ast_node!(SNode);
impl SNode {
    /// The dotted node path as source text (e.g. `X3.na`).
    pub fn path(&self) -> String {
        self.0.text().to_string().trim().to_string()
    }
}

ast_node!(SubcktNode);
ast_node!(SubcktNodes);
impl SubcktNodes {
    pub fn nodes(&self) -> impl Iterator<Item = SNode> + '_ {
        support::all(&self.0)
    }
}

ast_node!(SNodeList);
impl SNodeList {
    pub fn nodes(&self) -> impl Iterator<Item = SNode> + '_ {
        support::all(&self.0)
    }
}

// --- parameters ---

ast_node!(Parameter);
impl Parameter {
    /// Parameter name (`Identifier` token).
    pub fn name(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    /// Parameter value as verbatim source text (present when `name=value`).
    pub fn value_text(&self) -> Option<String> {
        value_text_after_eq(&self.0)
    }
}

ast_node!(Parameters);
impl Parameters {
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

// --- simulator ---

ast_node!(Simulator);
impl Simulator {
    /// The selected language keyword text (`spectre` or `spice`), i.e. the RHS
    /// of `lang = …`.
    pub fn lang(&self) -> Option<String> {
        value_text_after_eq(&self.0)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

// --- model ---

ast_node!(Model);
impl Model {
    /// Model name (first bare `Identifier` token).
    pub fn name(&self) -> Option<SyntaxToken> {
        nth_token(&self.0, SyntaxKind::Identifier, 0)
    }
    /// Master/base device name (second bare `Identifier` token).
    pub fn master(&self) -> Option<SyntaxToken> {
        nth_token(&self.0, SyntaxKind::Identifier, 1)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

// --- subckt ---

ast_node!(Subckt);
impl Subckt {
    /// True for `inline subckt` definitions.
    pub fn is_inline(&self) -> bool {
        self.0
            .children_with_tokens()
            .filter_map(|e| e.into_token())
            .any(|t| t.kind() == SyntaxKind::Keyword && t.text() == "inline")
    }
    /// Subcircuit name (first `Identifier` token — precedes the `ends` name).
    pub fn name(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    /// Port node names, from the `SubcktNodes` child (bare or parenthesised).
    pub fn ports(&self) -> Vec<SNode> {
        support::child::<SubcktNodes>(&self.0)
            .map(|n| n.nodes().collect())
            .unwrap_or_default()
    }
    /// Body statements (everything except the port list).
    pub fn body(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.0
            .children()
            .filter(|n| n.kind() != SyntaxKind::SubcktNodes)
    }
}

// --- instance ---

ast_node!(Instance);
impl Instance {
    /// Instance name (first bare `Identifier` token).
    pub fn name(&self) -> Option<SyntaxToken> {
        nth_token(&self.0, SyntaxKind::Identifier, 0)
    }
    /// Master (model/subckt) name — the second bare `Identifier` token, after
    /// the connection-node list.
    pub fn master(&self) -> Option<SyntaxToken> {
        nth_token(&self.0, SyntaxKind::Identifier, 1)
    }
    /// Connection nodes. Handles both the success path (wrapped in `SNodeList`)
    /// and the error path (flat `SNode` children under the instance).
    pub fn nodes(&self) -> Vec<SNode> {
        match support::child::<SNodeList>(&self.0) {
            Some(list) => list.nodes().collect(),
            None => support::all::<SNode>(&self.0).collect(),
        }
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

// --- analysis ---

ast_node!(Analysis);
impl Analysis {
    /// Analysis name (first bare `Identifier` token).
    pub fn name(&self) -> Option<SyntaxToken> {
        nth_token(&self.0, SyntaxKind::Identifier, 0)
    }
    /// Analysis type keyword (e.g. `tran`, `dc`, `ac`).
    pub fn analysis_type(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Keyword)
    }
    /// Optional connection-node list.
    pub fn nodes(&self) -> Vec<SNode> {
        support::child::<SNodeList>(&self.0)
            .map(|l| l.nodes().collect())
            .unwrap_or_default()
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

// --- save ---

ast_node!(Save);
impl Save {
    pub fn signals(&self) -> impl Iterator<Item = SaveSignal> + '_ {
        support::all(&self.0)
    }
}

ast_node!(SaveSignal);
impl SaveSignal {
    /// The saved signal's node path, if any.
    pub fn node(&self) -> Option<SNode> {
        support::child(&self.0)
    }
    pub fn modifier(&self) -> Option<SaveSignalModifier> {
        support::child(&self.0)
    }
}

ast_node!(SaveSignalModifier);
impl SaveSignalModifier {
    /// The modifier token after the `:` (a save keyword, number, or identifier).
    pub fn value(&self) -> Option<String> {
        element_text_after(&self.0, |el| {
            matches!(el, NodeOrToken::Token(t) if t.kind() == SyntaxKind::Notation && t.text() == ":")
        })
    }
}

// --- ic / nodeset ---

ast_node!(Ic);
impl Ic {
    pub fn params(&self) -> impl Iterator<Item = ICParameter> + '_ {
        support::all(&self.0)
    }
}

ast_node!(ICParameter);
impl ICParameter {
    pub fn node(&self) -> Option<SNode> {
        support::child(&self.0)
    }
    pub fn value_text(&self) -> Option<String> {
        value_text_after_eq(&self.0)
    }
}

ast_node!(NodeSet);
impl NodeSet {
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

// --- global ---

ast_node!(Global);
impl Global {
    pub fn nodes(&self) -> impl Iterator<Item = SNode> + '_ {
        support::all(&self.0)
    }
}

// --- include / ahdl_include ---

ast_node!(Include);
impl Include {
    /// The included file path (`StringLiteral` token, quotes included).
    pub fn path(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::StringLiteral)
    }
    pub fn section(&self) -> Option<IncludeSection> {
        support::child(&self.0)
    }
}

ast_node!(IncludeSection);
impl IncludeSection {
    /// The section identifier.
    pub fn id(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
}

ast_node!(AHDLInclude);
impl AHDLInclude {
    pub fn path(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::StringLiteral)
    }
}

// --- function declaration ---

ast_node!(FunctionDecl);
impl FunctionDecl {
    /// Function name (`Identifier` token).
    pub fn name(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    /// Declared arguments (each nested inside a `FunctionArgs` wrapper node).
    pub fn args(&self) -> Vec<FunctionDeclArg> {
        self.0
            .descendants()
            .filter_map(FunctionDeclArg::cast)
            .collect()
    }
    /// The returned expression as source text.
    pub fn return_expr(&self) -> Option<String> {
        element_text_after(&self.0, |el| {
            matches!(el, NodeOrToken::Token(t) if t.kind() == SyntaxKind::Keyword && t.text() == "return")
        })
    }
}

ast_node!(FunctionDeclArg);
impl FunctionDeclArg {
    pub fn name(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
}

// --- conditional block ---

ast_node!(ConditionalBlock);
impl ConditionalBlock {
    pub fn if_clause(&self) -> Option<If> {
        support::child(&self.0)
    }
    pub fn else_ifs(&self) -> impl Iterator<Item = ElseIf> + '_ {
        support::all(&self.0)
    }
    pub fn else_clause(&self) -> Option<Else> {
        support::child(&self.0)
    }
}

/// The condition expression text between `(` and `)`, plus the instantiated
/// body instance, shared by `if`/`else if` clauses.
fn condition_text(parent: &SyntaxNode) -> Option<String> {
    element_text_after(parent, |el| {
        matches!(el, NodeOrToken::Token(t) if t.kind() == SyntaxKind::Notation && t.text() == "(")
    })
}

ast_node!(If);
impl If {
    pub fn condition(&self) -> Option<String> {
        condition_text(&self.0)
    }
    pub fn body_instance(&self) -> Option<Instance> {
        support::child(&self.0)
    }
}

ast_node!(ElseIf);
impl ElseIf {
    pub fn condition(&self) -> Option<String> {
        condition_text(&self.0)
    }
    pub fn body_instance(&self) -> Option<Instance> {
        support::child(&self.0)
    }
}

ast_node!(Else);
impl Else {
    pub fn body_instance(&self) -> Option<Instance> {
        support::child(&self.0)
    }
}

// --- named control statements ---

/// Generates a Spectre named-control wrapper: `<name> <keyword> params…`.
macro_rules! named_control {
    ($name:ident) => {
        ast_node!($name);
        impl $name {
            /// The statement name (leading `Identifier` token).
            pub fn name(&self) -> Option<SyntaxToken> {
                nth_token(&self.0, SyntaxKind::Identifier, 0)
            }
            /// The control keyword (e.g. `alter`, `options`).
            pub fn control_keyword(&self) -> Option<SyntaxToken> {
                support::token(&self.0, SyntaxKind::Keyword)
            }
            pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
                support::all(&self.0)
            }
        }
    };
}

named_control!(Alter);
named_control!(Check);
named_control!(CheckLimit);
named_control!(Info);
named_control!(Options);
named_control!(Set);
named_control!(Shell);
named_control!(ParamTest);

ast_node!(AlterGroup);
impl AlterGroup {
    pub fn name(&self) -> Option<SyntaxToken> {
        nth_token(&self.0, SyntaxKind::Identifier, 0)
    }
    /// Statements inside the `{ … }` block.
    pub fn body(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.0.children()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(src: &str) -> SpectreNetlistSource {
        SpectreNetlistSource::cast(crate::parse_spectre(src)).unwrap()
    }

    /// First top-level statement castable to `N`.
    fn first<N: AstNode>(src: &str) -> N {
        root(src)
            .statements()
            .find_map(N::cast)
            .expect("no matching statement")
    }

    fn tok(t: Option<SyntaxToken>) -> String {
        t.map(|t| t.text().to_string()).unwrap_or_default()
    }

    #[test]
    fn simulator_lang() {
        let sim: Simulator = first("simulator lang=spectre\n");
        assert_eq!(sim.lang().as_deref(), Some("spectre"));
    }

    #[test]
    fn model_name_master_params() {
        let m: Model = first("model nch bsim4 version=4.5 l=1u\n");
        assert_eq!(tok(m.name()), "nch");
        assert_eq!(tok(m.master()), "bsim4");
        let params: Vec<_> = m.params().collect();
        assert_eq!(params.len(), 2);
        assert_eq!(tok(params[0].name()), "version");
        assert_eq!(params[0].value_text().as_deref(), Some("4.5"));
        assert_eq!(tok(params[1].name()), "l");
        assert_eq!(params[1].value_text().as_deref(), Some("1u"));
    }

    #[test]
    fn parameters_statement() {
        let p: Parameters = first("parameters vdd=1.8 temp=27\n");
        let params: Vec<_> = p.params().collect();
        assert_eq!(params.len(), 2);
        assert_eq!(tok(params[0].name()), "vdd");
        assert_eq!(params[0].value_text().as_deref(), Some("1.8"));
    }

    #[test]
    fn param_expression_value_text() {
        // Bare-token grammar + a compound expression RHS.
        let p: Parameters = first("parameters a=1 b=a*2+1\n");
        let params: Vec<_> = p.params().collect();
        assert_eq!(params[1].value_text().as_deref(), Some("a*2+1"));
    }

    #[test]
    fn subckt_ports_and_body() {
        let s: Subckt = first(
            "subckt inv (in out vdd vss)\n  m1 (out in vss vss) nch\nends inv\n",
        );
        assert_eq!(tok(s.name()), "inv");
        assert!(!s.is_inline());
        let ports: Vec<String> = s.ports().iter().map(|n| n.path()).collect();
        assert_eq!(ports, vec!["in", "out", "vdd", "vss"]);
        let insts: Vec<Instance> = s.body().filter_map(Instance::cast).collect();
        assert_eq!(insts.len(), 1);
        assert_eq!(tok(insts[0].name()), "m1");
    }

    #[test]
    fn instance_nodes_master_params() {
        let i: Instance = first("r1 (a b) resistor r=1k\n");
        assert_eq!(tok(i.name()), "r1");
        assert_eq!(tok(i.master()), "resistor");
        let nodes: Vec<String> = i.nodes().iter().map(|n| n.path()).collect();
        assert_eq!(nodes, vec!["a", "b"]);
        let params: Vec<_> = i.params().collect();
        assert_eq!(tok(params[0].name()), "r");
        assert_eq!(params[0].value_text().as_deref(), Some("1k"));
    }

    #[test]
    fn analysis_type_and_params() {
        let a: Analysis = first("stepResponse tran stop=100ns\n");
        assert_eq!(tok(a.name()), "stepResponse");
        assert_eq!(tok(a.analysis_type()), "tran");
        let params: Vec<_> = a.params().collect();
        assert_eq!(tok(params[0].name()), "stop");
        assert_eq!(params[0].value_text().as_deref(), Some("100ns"));
    }

    #[test]
    fn global_nodes() {
        let g: Global = first("global 0 gnd vdd\n");
        let nodes: Vec<String> = g.nodes().map(|n| n.path()).collect();
        assert_eq!(nodes, vec!["0", "gnd", "vdd"]);
    }

    #[test]
    fn ic_params() {
        let ic: Ic = first("ic vout=2.5 vin=0\n");
        let params: Vec<_> = ic.params().collect();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].node().map(|n| n.path()).as_deref(), Some("vout"));
        assert_eq!(params[0].value_text().as_deref(), Some("2.5"));
    }

    #[test]
    fn include_with_section() {
        let inc: Include = first("include \"models.scs\" section=tt\n");
        assert_eq!(tok(inc.path()), "\"models.scs\"");
        assert_eq!(
            inc.section().and_then(|s| s.id()).map(|t| t.text().to_string()).as_deref(),
            Some("tt")
        );
    }

    #[test]
    fn conditional_block_clauses() {
        let src = "simulator lang=spectre\n\
                   subckt s1 (d g s b)\n\
                   parameters l=1u\n\
                   if ( l < 0.5u ) {\n\
                   m1 (d g s b) shortmod l=l\n\
                   } else {\n\
                   m2 (d g s b) longmod l=l\n\
                   }\n\
                   ends s1\n";
        let s: Subckt = first(src);
        let cond: ConditionalBlock = s
            .body()
            .find_map(ConditionalBlock::cast)
            .expect("conditional block");
        let iff = cond.if_clause().expect("if clause");
        assert_eq!(iff.condition().as_deref(), Some("l < 0.5u"));
        assert_eq!(
            iff.body_instance().and_then(|i| i.name()).map(|t| t.text().to_string()).as_deref(),
            Some("m1")
        );
        let els = cond.else_clause().expect("else clause");
        assert_eq!(
            els.body_instance().and_then(|i| i.master()).map(|t| t.text().to_string()).as_deref(),
            Some("longmod")
        );
    }
}
