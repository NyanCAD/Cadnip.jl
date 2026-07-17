//! Typed AST accessor layer over the rowan CST — the rust-analyzer `ast`
//! pattern. Each grammar node gets a thin typed wrapper with named field
//! accessors, giving Rust the same ergonomics as the Julia parser's `forms.jl`
//! structs + `RedTree` named-field access.
//!
//! Note on literals: in this CST, terminals (`NumberLiteral`, `Identifier`,
//! `Notation`, ...) are leaf *tokens*, not nodes. So an "expression" position
//! can be a node (`BinaryExpression`, ...) or a token (`NumberLiteral`); the
//! [`Expr`] enum unifies both. Accessors for same-typed positional fields
//! (e.g. a device's `name`/`pos`/`neg`, all `HierarchialNode`) select by child
//! order, matching the Julia field layout.

use crate::syntax_kind::{SyntaxKind, SyntaxNode, SyntaxToken};
use rowan::NodeOrToken;

/// A typed handle onto a CST node.
pub trait AstNode: Sized {
    fn can_cast(kind: SyntaxKind) -> bool;
    fn cast(node: SyntaxNode) -> Option<Self>;
    fn syntax(&self) -> &SyntaxNode;
    /// The node's source text (including trivia within its span).
    fn text(&self) -> String {
        self.syntax().text().to_string()
    }
}

mod support {
    use super::*;

    /// First child node castable to `N`.
    pub(super) fn child<N: AstNode>(parent: &SyntaxNode) -> Option<N> {
        parent.children().find_map(N::cast)
    }
    /// `n`th child node castable to `N`.
    pub(super) fn nth<N: AstNode>(parent: &SyntaxNode, n: usize) -> Option<N> {
        parent.children().filter_map(N::cast).nth(n)
    }
    /// All child nodes castable to `N`.
    pub(super) fn all<'a, N: AstNode + 'a>(parent: &'a SyntaxNode) -> impl Iterator<Item = N> + 'a {
        parent.children().filter_map(N::cast)
    }
    /// First direct child token of the given kind.
    pub(super) fn token(parent: &SyntaxNode, kind: SyntaxKind) -> Option<SyntaxToken> {
        parent
            .children_with_tokens()
            .filter_map(|e| e.into_token())
            .find(|t| t.kind() == kind)
    }
}

macro_rules! ast_node {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(SyntaxNode);
        impl AstNode for $name {
            fn can_cast(kind: SyntaxKind) -> bool {
                kind == SyntaxKind::$name
            }
            fn cast(node: SyntaxNode) -> Option<Self> {
                if Self::can_cast(node.kind()) {
                    Some($name(node))
                } else {
                    None
                }
            }
            fn syntax(&self) -> &SyntaxNode {
                &self.0
            }
        }
    };
}

// --- Expr: a node-or-token expression ---

/// An expression: an operator/call/grouping node, a `HierarchialNode`, or a
/// literal/identifier leaf token.
#[derive(Debug, Clone)]
pub enum Expr {
    Binary(BinaryExpression),
    Unary(UnaryOp),
    Ternary(TernaryExpr),
    Call(FunctionCall),
    Parens(Parens),
    Brace(Brace),
    Prime(Prime),
    Square(Square),
    Hier(HierarchialNode),
    /// A literal (`NumberLiteral`/`Literal`/`StringLiteral`) wrapped in a
    /// `LiteralExpr` node.
    Literal(LiteralExpr),
    /// A bare identifier used as an expression, wrapped in a `NameRef` node.
    Name(NameRef),
}

impl Expr {
    fn cast_element(el: rowan::SyntaxElement<crate::syntax_kind::NetlistLang>) -> Option<Expr> {
        match el {
            NodeOrToken::Node(n) => match n.kind() {
                SyntaxKind::BinaryExpression => Some(Expr::Binary(BinaryExpression(n))),
                SyntaxKind::UnaryOp => Some(Expr::Unary(UnaryOp(n))),
                SyntaxKind::TernaryExpr => Some(Expr::Ternary(TernaryExpr(n))),
                SyntaxKind::FunctionCall => Some(Expr::Call(FunctionCall(n))),
                SyntaxKind::Parens => Some(Expr::Parens(Parens(n))),
                SyntaxKind::Brace => Some(Expr::Brace(Brace(n))),
                SyntaxKind::Prime => Some(Expr::Prime(Prime(n))),
                SyntaxKind::Square => Some(Expr::Square(Square(n))),
                SyntaxKind::HierarchialNode => Some(Expr::Hier(HierarchialNode(n))),
                SyntaxKind::LiteralExpr => Some(Expr::Literal(LiteralExpr(n))),
                SyntaxKind::NameRef => Some(Expr::Name(NameRef(n))),
                _ => None,
            },
            // Expression leaves are always wrapped in nodes (LiteralExpr /
            // NameRef / HierarchialNode), so there are no bare-token exprs.
            NodeOrToken::Token(_) => None,
        }
    }

    /// Source text of the expression.
    pub fn text(&self) -> String {
        match self {
            Expr::Binary(n) => n.text(),
            Expr::Unary(n) => n.text(),
            Expr::Ternary(n) => n.text(),
            Expr::Call(n) => n.text(),
            Expr::Parens(n) => n.text(),
            Expr::Brace(n) => n.text(),
            Expr::Prime(n) => n.text(),
            Expr::Square(n) => n.text(),
            Expr::Hier(n) => n.text(),
            Expr::Literal(n) => n.text(),
            Expr::Name(n) => n.text(),
        }
    }
}

ast_node!(LiteralExpr);
impl LiteralExpr {
    /// The inner literal token (`NumberLiteral`, `Literal`, or `StringLiteral`).
    pub fn token(&self) -> Option<SyntaxToken> {
        self.0
            .children_with_tokens()
            .filter_map(|e| e.into_token())
            .find(|t| {
                matches!(
                    t.kind(),
                    SyntaxKind::NumberLiteral | SyntaxKind::Literal | SyntaxKind::StringLiteral
                )
            })
    }
}

ast_node!(NameRef);
impl NameRef {
    /// The inner identifier token.
    pub fn token(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
}

/// Expression children of `parent`, in order (nodes and literal tokens).
fn expr_children(parent: &SyntaxNode) -> impl Iterator<Item = Expr> + '_ {
    parent.children_with_tokens().filter_map(Expr::cast_element)
}

/// The expression appearing after an `=` (`Notation`) token — a value in a
/// `name = value` construct (an `Identifier` name is itself an `Expr::Token`, so
/// `expr_children().next()` would wrongly return the name).
fn value_after_eq(parent: &SyntaxNode) -> Option<Expr> {
    let mut seen_eq = false;
    for el in parent.children_with_tokens() {
        if let NodeOrToken::Token(t) = &el {
            if t.kind() == SyntaxKind::Notation && t.text() == "=" {
                seen_eq = true;
                continue;
            }
        }
        if seen_eq {
            if let Some(e) = Expr::cast_element(el) {
                return Some(e);
            }
        }
    }
    None
}

/// The value expression of a two-terminal-ish device: the first expression
/// after the leading `skip_hier` `HierarchialNode` terminals (which are the
/// node connections). Matches the Julia `val` field position.
fn value_after_hier(parent: &SyntaxNode, skip_hier: usize) -> Option<Expr> {
    let mut seen_hier = 0;
    for el in parent.children_with_tokens() {
        if let NodeOrToken::Node(n) = &el {
            if n.kind() == SyntaxKind::HierarchialNode && seen_hier < skip_hier {
                seen_hier += 1;
                continue;
            }
        }
        if seen_hier >= skip_hier {
            if let Some(e) = Expr::cast_element(el) {
                return Some(e);
            }
        }
    }
    None
}

// --- root & structural ---

ast_node!(SPICENetlistSource);
impl SPICENetlistSource {
    /// All top-level statement nodes.
    pub fn statements(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.0.children()
    }
}

ast_node!(NodeName);
impl NodeName {
    /// The underlying `Identifier` or `NumberLiteral` token.
    pub fn token(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
            .or_else(|| support::token(&self.0, SyntaxKind::NumberLiteral))
    }
}

ast_node!(SubNode);
impl SubNode {
    pub fn node(&self) -> Option<NodeName> {
        support::child(&self.0)
    }
}

ast_node!(HierarchialNode);
impl HierarchialNode {
    pub fn base(&self) -> Option<NodeName> {
        support::child(&self.0)
    }
    pub fn subnodes(&self) -> impl Iterator<Item = SubNode> + '_ {
        support::all(&self.0)
    }
}

ast_node!(Parameter);
impl Parameter {
    /// Parameter name (`Identifier` token).
    pub fn name(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    /// Parameter value expression (present when `name=value`).
    pub fn value(&self) -> Option<Expr> {
        value_after_eq(&self.0)
    }
}

// --- passive / source instances ---

macro_rules! two_terminal_value {
    ($name:ident) => {
        ast_node!($name);
        impl $name {
            pub fn name(&self) -> Option<HierarchialNode> {
                support::nth(&self.0, 0)
            }
            pub fn pos(&self) -> Option<HierarchialNode> {
                support::nth(&self.0, 1)
            }
            pub fn neg(&self) -> Option<HierarchialNode> {
                support::nth(&self.0, 2)
            }
            /// The optional value expression (unless the value is given as a
            /// parameter).
            pub fn value(&self) -> Option<Expr> {
                value_after_hier(&self.0, 3)
            }
            pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
                support::all(&self.0)
            }
        }
    };
}

two_terminal_value!(Resistor);
two_terminal_value!(Capacitor);
two_terminal_value!(Inductor);

macro_rules! source_instance {
    ($name:ident) => {
        ast_node!($name);
        impl $name {
            pub fn name(&self) -> Option<HierarchialNode> {
                support::nth(&self.0, 0)
            }
            pub fn pos(&self) -> Option<HierarchialNode> {
                support::nth(&self.0, 1)
            }
            pub fn neg(&self) -> Option<HierarchialNode> {
                support::nth(&self.0, 2)
            }
            /// The DC/AC/transient source specs.
            pub fn sources(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
                self.0.children().filter(|n| {
                    matches!(
                        n.kind(),
                        SyntaxKind::DCSource | SyntaxKind::ACSource | SyntaxKind::TranSource
                    )
                })
            }
        }
    };
}

source_instance!(Voltage);
source_instance!(Current);

ast_node!(Diode);
impl Diode {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    pub fn pos(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 1)
    }
    pub fn neg(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 2)
    }
    pub fn model(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 3)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

ast_node!(MOSFET);
impl MOSFET {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    pub fn drain(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 1)
    }
    pub fn gate(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 2)
    }
    pub fn source(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 3)
    }
    pub fn bulk(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 4)
    }
    pub fn model(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 5)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

ast_node!(BipolarTransistor);
impl BipolarTransistor {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    /// Collector, base, emitter, and any optional substrate/thermal/model nodes,
    /// in order (all `HierarchialNode`s).
    pub fn nodes(&self) -> impl Iterator<Item = HierarchialNode> + '_ {
        support::all(&self.0)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

ast_node!(SubcktCall);
impl SubcktCall {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    /// Connection nodes and (positional) model name — all `HierarchialNode`s.
    pub fn nodes(&self) -> impl Iterator<Item = HierarchialNode> + '_ {
        support::all(&self.0)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

// --- statements ---

ast_node!(Title);

ast_node!(Model);
impl Model {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::child(&self.0)
    }
    /// Model type — an `Identifier` or `NumberLiteral` token after the name.
    pub fn model_type(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
            .or_else(|| support::token(&self.0, SyntaxKind::NumberLiteral))
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

ast_node!(Subckt);
impl Subckt {
    /// Subckt name (`Identifier` or `NumberLiteral` token).
    pub fn name(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
            .or_else(|| support::token(&self.0, SyntaxKind::NumberLiteral))
    }
    pub fn ports(&self) -> impl Iterator<Item = HierarchialNode> + '_ {
        support::all(&self.0)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
    /// Body statements between the header and `.ends`.
    pub fn body(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.0.children().filter(|n| {
            !matches!(n.kind(), SyntaxKind::HierarchialNode | SyntaxKind::Parameter)
        })
    }
}

macro_rules! param_statement {
    ($name:ident) => {
        ast_node!($name);
        impl $name {
            pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
                support::all(&self.0)
            }
        }
    };
}
param_statement!(ParamStatement);
param_statement!(OptionStatement);
param_statement!(WidthStatement);
param_statement!(GlobalParamStatement);

ast_node!(PrintStatement);
impl PrintStatement {
    pub fn args(&self) -> impl Iterator<Item = Expr> + '_ {
        expr_children(&self.0)
    }
}

ast_node!(NodeSetStatement);
impl NodeSetStatement {
    pub fn entries(&self) -> impl Iterator<Item = NodeSetEntry> + '_ {
        support::all(&self.0)
    }
}

ast_node!(NodeSetEntry);
impl NodeSetEntry {
    /// The accessor function token (`V`/`I`).
    pub fn func(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    pub fn node(&self) -> Option<NodeName> {
        support::child(&self.0)
    }
    pub fn value(&self) -> Option<Expr> {
        value_after_eq(&self.0)
    }
}

// --- generic extension devices (name, nodes, params) ---

macro_rules! generic_device {
    ($name:ident) => {
        ast_node!($name);
        impl $name {
            pub fn name(&self) -> Option<HierarchialNode> {
                support::nth(&self.0, 0)
            }
            /// Trailing node/value terminals (after the name).
            pub fn nodes(&self) -> impl Iterator<Item = HierarchialNode> + '_ {
                support::all::<HierarchialNode>(&self.0).skip(1)
            }
            pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
                support::all(&self.0)
            }
        }
    };
}
generic_device!(MutualInductor);
generic_device!(JFET);
generic_device!(TransmissionLine);
generic_device!(Mesfet);
generic_device!(XspiceDevice);

// --- expressions ---

ast_node!(BinaryExpression);
impl BinaryExpression {
    pub fn lhs(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
    pub fn op(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Operator)
    }
    pub fn rhs(&self) -> Option<Expr> {
        expr_children(&self.0).nth(1)
    }
}

ast_node!(UnaryOp);
impl UnaryOp {
    pub fn op(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Operator)
    }
    pub fn operand(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

ast_node!(TernaryExpr);
impl TernaryExpr {
    pub fn condition(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
    pub fn if_case(&self) -> Option<Expr> {
        expr_children(&self.0).nth(1)
    }
    pub fn else_case(&self) -> Option<Expr> {
        expr_children(&self.0).nth(2)
    }
}

ast_node!(FunctionCall);
impl FunctionCall {
    /// Function name (`Identifier` token).
    pub fn callee(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    pub fn args(&self) -> impl Iterator<Item = FunctionArgs> + '_ {
        support::all(&self.0)
    }
}

ast_node!(FunctionArgs);
impl FunctionArgs {
    pub fn expr(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

macro_rules! grouping {
    ($name:ident) => {
        ast_node!($name);
        impl $name {
            /// The inner expression.
            pub fn inner(&self) -> Option<Expr> {
                expr_children(&self.0).next()
            }
        }
    };
}
grouping!(Parens);
grouping!(Brace);
grouping!(Prime);

ast_node!(Square);
impl Square {
    pub fn elements(&self) -> impl Iterator<Item = Expr> + '_ {
        expr_children(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(src: &str) -> SPICENetlistSource {
        SPICENetlistSource::cast(crate::parse_spice(src)).unwrap()
    }

    #[test]
    fn resistor_fields() {
        let r = root("* t\nR1 in out 1k\n");
        let res = r
            .statements()
            .find_map(Resistor::cast)
            .expect("a Resistor");
        assert_eq!(res.name().unwrap().text(), "R1");
        assert_eq!(res.pos().unwrap().text(), "in");
        assert_eq!(res.neg().unwrap().text(), "out");
        assert_eq!(res.value().unwrap().text(), "1k");
    }

    #[test]
    fn resistor_with_param_value() {
        let r = root("* t\nR1 a b r=1k\n");
        let res = r.statements().find_map(Resistor::cast).unwrap();
        assert!(res.value().is_none(), "value is a parameter, not positional");
        let p = res.params().next().unwrap();
        assert_eq!(p.name().unwrap().text(), "r");
        assert_eq!(p.value().unwrap().text(), "1k");
    }

    #[test]
    fn mosfet_and_model() {
        let r = root("* t\nM1 d g s b nmos w=1u l=0.18u\n.model nmos nmos level=1\n");
        let m = r.statements().find_map(MOSFET::cast).unwrap();
        assert_eq!(m.drain().unwrap().text(), "d");
        assert_eq!(m.bulk().unwrap().text(), "b");
        assert_eq!(m.model().unwrap().text(), "nmos");
        assert_eq!(m.params().count(), 2);
        let model = r.statements().find_map(Model::cast).unwrap();
        assert_eq!(model.model_type().unwrap().text(), "nmos");
    }

    #[test]
    fn binary_expression_tree() {
        let r = root("* t\n.param x = 1k + 2*3\n");
        let ps = r.statements().find_map(ParamStatement::cast).unwrap();
        let val = ps.params().next().unwrap().value().unwrap();
        let top = match val {
            Expr::Binary(b) => b,
            other => panic!("expected binary, got {other:?}"),
        };
        assert_eq!(top.op().unwrap().text(), "+");
        assert_eq!(top.lhs().unwrap().text(), "1k");
        // rhs is the tighter-binding 2*3
        match top.rhs().unwrap() {
            Expr::Binary(inner) => assert_eq!(inner.op().unwrap().text(), "*"),
            other => panic!("expected nested binary, got {other:?}"),
        }
    }

    #[test]
    fn expr_is_uniformly_node_based() {
        // A literal, a bare identifier, and a dotted name are all node-based Expr
        // variants (no bare-token expressions).
        let r = root("* t\n.param a = 1k\n.param b = foo\n.param c = x.y\n");
        let vals: Vec<_> = r
            .statements()
            .filter_map(ParamStatement::cast)
            .filter_map(|p| p.params().next()?.value())
            .collect();
        assert!(matches!(vals[0], Expr::Literal(_)), "1k is a Literal");
        match &vals[1] {
            Expr::Name(n) => assert_eq!(n.token().unwrap().text(), "foo"),
            other => panic!("expected NameRef, got {other:?}"),
        }
        assert!(matches!(vals[2], Expr::Hier(_)), "x.y is a HierarchialNode");
    }

    #[test]
    fn nodeset_entries() {
        let r = root("* t\n.nodeset V(1)=0.5 V(out)=0.3\n");
        let ns = r.statements().find_map(NodeSetStatement::cast).unwrap();
        let entries: Vec<_> = ns.entries().collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].node().unwrap().text(), "1");
        assert_eq!(entries[0].value().unwrap().text(), "0.5");
        assert_eq!(entries[1].node().unwrap().text(), "out");
    }
}
