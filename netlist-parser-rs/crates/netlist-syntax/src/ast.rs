//! Typed AST accessor layer over the rowan CST — the rust-analyzer `ast`
//! pattern. Each grammar node gets a thin typed wrapper with named field
//! accessors, giving Rust the same ergonomics as the Julia parser's `forms.jl`
//! structs + `RedTree` named-field access.
//!
//! Expressions are uniformly node-based: literal and bare-identifier leaves are
//! wrapped in `LiteralExpr`/`NameRef` nodes (the rust-analyzer `Literal`/
//! `NameRef` pattern) so the [`Expr`] enum has no bare-token variant. Those
//! wrappers are rendered transparently by the dumper, so the differential stays
//! byte-exact against Julia while the AST stays idiomatic. Accessors for
//! same-typed positional fields (e.g. a device's `name`/`pos`/`neg`, all
//! `HierarchialNode`) select by child order — the rowan convention for a fixed
//! number of repeated same-kind children — matching the Julia field layout.

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

/// Direct child tokens of a given kind (for fields the parser emits as bare
/// tokens rather than wrapped nodes — e.g. `.tran`/`.data` numeric values).
fn tokens(parent: &SyntaxNode, kind: SyntaxKind) -> impl Iterator<Item = SyntaxToken> + '_ {
    parent
        .children_with_tokens()
        .filter_map(|e| e.into_token())
        .filter(move |t| t.kind() == kind)
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

// --- remaining instances / source specs ---

ast_node!(Behavioral);
impl Behavioral {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    pub fn pos(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 1)
    }
    pub fn neg(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 2)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

ast_node!(OSDIDevice);
impl OSDIDevice {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    /// Connection nodes and (positional) model name.
    pub fn nodes(&self) -> impl Iterator<Item = HierarchialNode> + '_ {
        support::all::<HierarchialNode>(&self.0).skip(1)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
}

ast_node!(Switch);
impl Switch {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    /// nd1, nd2, cnd1, cnd2, model — all `HierarchialNode`s in order.
    pub fn nodes(&self) -> impl Iterator<Item = HierarchialNode> + '_ {
        support::all::<HierarchialNode>(&self.0).skip(1)
    }
    /// The `ON`/`OFF` keyword.
    pub fn state(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Keyword)
    }
}

ast_node!(ControlledSource);
impl ControlledSource {
    pub fn name(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    pub fn pos(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 1)
    }
    pub fn neg(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 2)
    }
    /// The control spec: a `VoltageControl`/`CurrentControl`/`PolyControl`/
    /// `TableControl` node.
    pub fn control(&self) -> Option<SyntaxNode> {
        self.0.children().find(|n| {
            matches!(
                n.kind(),
                SyntaxKind::VoltageControl
                    | SyntaxKind::CurrentControl
                    | SyntaxKind::PolyControl
                    | SyntaxKind::TableControl
            )
        })
    }
}

ast_node!(VoltageControl);
impl VoltageControl {
    /// Control-node connections (cpos, cneg).
    pub fn control_nodes(&self) -> impl Iterator<Item = HierarchialNode> + '_ {
        support::all(&self.0)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
    /// Nonlinear value expression, if given as a non-parameter.
    pub fn value(&self) -> Option<Expr> {
        expr_children(&self.0).find(|e| !matches!(e, Expr::Hier(_)))
    }
}

ast_node!(CurrentControl);
impl CurrentControl {
    /// Controlling source name.
    pub fn vnam(&self) -> Option<HierarchialNode> {
        support::nth(&self.0, 0)
    }
    pub fn params(&self) -> impl Iterator<Item = Parameter> + '_ {
        support::all(&self.0)
    }
    pub fn value(&self) -> Option<Expr> {
        expr_children(&self.0).find(|e| !matches!(e, Expr::Hier(_)))
    }
}

ast_node!(PolyControl);
impl PolyControl {
    /// The `N` dimension of `POLY(N)`.
    pub fn dimensions(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::NumberLiteral)
    }
    /// Control nodes and coefficients.
    pub fn args(&self) -> impl Iterator<Item = Expr> + '_ {
        expr_children(&self.0)
    }
}

ast_node!(TableControl);
impl TableControl {
    /// The controlling expression.
    pub fn expr(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
    /// The (x, y) table points and any following expressions.
    pub fn args(&self) -> impl Iterator<Item = Expr> + '_ {
        expr_children(&self.0).skip(1)
    }
}

ast_node!(DCSource);
impl DCSource {
    pub fn value(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

ast_node!(ACSource);
impl ACSource {
    pub fn magnitude(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
    pub fn phase(&self) -> Option<Expr> {
        expr_children(&self.0).nth(1)
    }
}

ast_node!(TranSource);
impl TranSource {
    /// The source-function keyword (`PULSE`/`SIN`/`PWL`/...).
    pub fn function(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Keyword)
    }
    pub fn values(&self) -> impl Iterator<Item = Expr> + '_ {
        expr_children(&self.0)
    }
}

// --- analysis + simple dot-commands ---

ast_node!(DCStatement);
impl DCStatement {
    pub fn commands(&self) -> impl Iterator<Item = DCCommand> + '_ {
        support::all(&self.0)
    }
}

ast_node!(DCCommand);
impl DCCommand {
    pub fn source(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    pub fn start(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
    pub fn stop(&self) -> Option<Expr> {
        expr_children(&self.0).nth(1)
    }
    pub fn step(&self) -> Option<Expr> {
        expr_children(&self.0).nth(2)
    }
}

ast_node!(ACStatement);
impl ACStatement {
    pub fn command(&self) -> Option<ACCommand> {
        support::child(&self.0)
    }
}

ast_node!(ACCommand);
impl ACCommand {
    pub fn variation(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    pub fn points(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
    pub fn fstart(&self) -> Option<Expr> {
        expr_children(&self.0).nth(1)
    }
    pub fn fstop(&self) -> Option<Expr> {
        expr_children(&self.0).nth(2)
    }
}

ast_node!(Tran);
impl Tran {
    /// The numeric time arguments (tstep/tstop/tstart/tmax), in order.
    pub fn values(&self) -> impl Iterator<Item = SyntaxToken> + '_ {
        tokens(&self.0, SyntaxKind::NumberLiteral)
    }
    /// The optional `uic` flag.
    pub fn uic(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
}

ast_node!(TempStatement);
impl TempStatement {
    pub fn value(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

ast_node!(GlobalStatement);
impl GlobalStatement {
    pub fn nodes(&self) -> impl Iterator<Item = HierarchialNode> + '_ {
        support::all(&self.0)
    }
}

ast_node!(IncludeStatement);
impl IncludeStatement {
    pub fn path(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::StringLiteral)
    }
}

ast_node!(HDLStatement);
impl HDLStatement {
    pub fn path(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::StringLiteral)
    }
}

ast_node!(EndStatement);
ast_node!(EndlStatement);
impl EndlStatement {
    /// The optional library-section identifier.
    pub fn id(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
}

ast_node!(LibInclude);
impl LibInclude {
    pub fn path(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::StringLiteral)
    }
    pub fn section(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
}

ast_node!(LibStatement);
impl LibStatement {
    pub fn name(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    /// Statements inside the `.lib`/`.endl` block.
    pub fn statements(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.0.children().filter(|n| n.kind() != SyntaxKind::EndlStatement)
    }
    pub fn end(&self) -> Option<EndlStatement> {
        support::child(&self.0)
    }
}

ast_node!(DevMod);
impl DevMod {
    /// The distribution identifier (after `dev/`).
    pub fn distribution(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    pub fn value(&self) -> Option<Expr> {
        value_after_eq(&self.0)
    }
}

// --- .ic / .nodeset / .data ---

ast_node!(ICStatement);
impl ICStatement {
    pub fn entries(&self) -> impl Iterator<Item = ICEntry> + '_ {
        support::all(&self.0)
    }
}

ast_node!(ICEntry);
impl ICEntry {
    pub fn name(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Identifier)
    }
    /// The node argument inside the parentheses.
    pub fn arg(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
    pub fn value(&self) -> Option<Expr> {
        value_after_eq(&self.0)
    }
}

ast_node!(WildCard);
impl WildCard {
    /// The optional value before the `*`.
    pub fn value(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

ast_node!(Coloned);
impl Coloned {
    pub fn left(&self) -> Option<SyntaxToken> {
        tokens(&self.0, SyntaxKind::Identifier).next()
    }
    pub fn right(&self) -> Option<SyntaxToken> {
        tokens(&self.0, SyntaxKind::Identifier).nth(1)
    }
}

ast_node!(DataStatement);
impl DataStatement {
    /// The data block name.
    pub fn name(&self) -> Option<SyntaxToken> {
        tokens(&self.0, SyntaxKind::Identifier).next()
    }
    /// Column names.
    pub fn columns(&self) -> impl Iterator<Item = SyntaxToken> + '_ {
        tokens(&self.0, SyntaxKind::Identifier).skip(1)
    }
    /// Row values (flattened).
    pub fn values(&self) -> impl Iterator<Item = SyntaxToken> + '_ {
        tokens(&self.0, SyntaxKind::NumberLiteral)
    }
}

// --- .if / .else / .endif ---

ast_node!(IfBlock);
impl IfBlock {
    pub fn cases(&self) -> impl Iterator<Item = IfElseCase> + '_ {
        support::all(&self.0)
    }
}

ast_node!(IfElseCase);
impl IfElseCase {
    /// The `IF`/`ELSE`/`ELSEIF` keyword.
    pub fn keyword(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Keyword)
    }
    pub fn condition(&self) -> Option<Condition> {
        support::child(&self.0)
    }
    /// Body statements in this branch.
    pub fn body(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.0.children().filter(|n| n.kind() != SyntaxKind::Condition)
    }
}

ast_node!(Condition);
impl Condition {
    pub fn expr(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

// --- Xyce .step / .func ---

macro_rules! named_expr_list {
    ($name:ident) => {
        ast_node!($name);
        impl $name {
            /// The command word (e.g. `step`, `func`).
            pub fn command(&self) -> Option<SyntaxToken> {
                support::token(&self.0, SyntaxKind::Identifier)
            }
            pub fn args(&self) -> impl Iterator<Item = Expr> + '_ {
                expr_children(&self.0)
            }
        }
    };
}
named_expr_list!(StepStatement);
named_expr_list!(FuncStatement);

// --- .measure family ---

ast_node!(MeasurePointStatement);
impl MeasurePointStatement {
    /// The analysis-type keyword (`TRAN`/`AC`/...) - the 2nd keyword after `.meas`.
    pub fn analysis(&self) -> Option<SyntaxToken> {
        tokens(&self.0, SyntaxKind::Keyword).nth(1)
    }
    pub fn find_deriv_param(&self) -> Option<FindDerivParam> {
        support::child(&self.0)
    }
    pub fn td(&self) -> Option<TD_> {
        support::child(&self.0)
    }
    pub fn rise_fall_cross(&self) -> Option<RiseFallCross> {
        support::child(&self.0)
    }
}

ast_node!(MeasureRangeStatement);
impl MeasureRangeStatement {
    pub fn analysis(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::Keyword)
    }
    pub fn agg(&self) -> Option<AvgMaxMinPPRmsInteg> {
        support::child(&self.0)
    }
    pub fn trig(&self) -> Option<TrigTarg> {
        support::all::<TrigTarg>(&self.0).next()
    }
    pub fn targ(&self) -> Option<TrigTarg> {
        support::all::<TrigTarg>(&self.0).nth(1)
    }
}

ast_node!(FindDerivParam);
impl FindDerivParam {
    pub fn body(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

ast_node!(When);
impl When {
    pub fn body(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

ast_node!(At);
impl At {
    pub fn body(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

ast_node!(RiseFallCross);
impl RiseFallCross {
    /// The count value or `LAST` keyword.
    pub fn value(&self) -> Option<SyntaxToken> {
        support::token(&self.0, SyntaxKind::NumberLiteral)
            .or_else(|| support::token(&self.0, SyntaxKind::Keyword))
    }
}

ast_node!(TD_);
impl TD_ {
    pub fn value(&self) -> Option<Expr> {
        value_after_eq(&self.0)
    }
}

ast_node!(AvgMaxMinPPRmsInteg);
impl AvgMaxMinPPRmsInteg {
    pub fn body(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
}

ast_node!(Val_);
impl Val_ {
    pub fn value(&self) -> Option<Expr> {
        value_after_eq(&self.0)
    }
}

ast_node!(TrigTarg);
impl TrigTarg {
    pub fn lhs(&self) -> Option<Expr> {
        expr_children(&self.0).next()
    }
    pub fn val(&self) -> Option<Val_> {
        support::child(&self.0)
    }
    pub fn td(&self) -> Option<TD_> {
        support::child(&self.0)
    }
    pub fn rise_fall_cross(&self) -> Option<RiseFallCross> {
        support::child(&self.0)
    }
}

// --- out-of-spike forms: castable, no field accessors ---
// (present as SyntaxKinds but not produced by the current parser, or error
// nodes; provided for completeness so every node kind can be `cast`.)
ast_node!(Incomplete);
ast_node!(SParameterElement);
ast_node!(JuliaDevice);
ast_node!(JuliaEscape);

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
    fn dc_command_fields() {
        let r = root("* t\n.dc V1 0 5 0.5\n");
        let dc = r.statements().find_map(DCStatement::cast).unwrap();
        let cmd = dc.commands().next().unwrap();
        assert_eq!(cmd.source().unwrap().text(), "V1");
        assert_eq!(cmd.start().unwrap().text(), "0");
        assert_eq!(cmd.stop().unwrap().text(), "5");
        assert_eq!(cmd.step().unwrap().text(), "0.5");
    }

    #[test]
    fn tran_and_sources() {
        let r = root("* t\nV1 in 0 DC 5 SIN(0 1 1k)\n.tran 1n 100n 0 uic\n");
        let v = r.statements().find_map(Voltage::cast).unwrap();
        let srcs: Vec<_> = v.sources().collect();
        assert!(srcs.iter().any(|n| n.kind() == SyntaxKind::DCSource));
        let tran = TranSource::cast(
            srcs.iter().find(|n| n.kind() == SyntaxKind::TranSource).unwrap().clone(),
        )
        .unwrap();
        assert_eq!(tran.function().unwrap().text(), "SIN");
        assert_eq!(tran.values().count(), 3);
        let t = r.statements().find_map(Tran::cast).unwrap();
        assert_eq!(t.values().count(), 3);
        assert_eq!(t.uic().unwrap().text(), "uic");
    }

    #[test]
    fn if_block_and_measure() {
        let r = root("* t\n.if (a>1)\nR1 a b 1k\n.endif\n.meas tran m FIND v(a) AT=5m\n");
        let ib = r.statements().find_map(IfBlock::cast).unwrap();
        let case = ib.cases().next().unwrap();
        assert_eq!(case.keyword().unwrap().text(), "if");
        assert!(case.condition().is_some());
        assert!(case.body().any(|n| n.kind() == SyntaxKind::Resistor));
        let m = r.statements().find_map(MeasurePointStatement::cast).unwrap();
        assert_eq!(m.analysis().unwrap().text(), "tran");
        assert!(m.find_deriv_param().is_some());
    }

    #[test]
    fn controlled_source_fields() {
        let r = root("* t\nE1 out 0 in 0 2\n");
        let e = r.statements().find_map(ControlledSource::cast).unwrap();
        assert_eq!(e.name().unwrap().text(), "E1");
        let ctrl = e.control().unwrap();
        assert_eq!(ctrl.kind(), SyntaxKind::VoltageControl);
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
