//! Clean C++ binding for the Spectre netlist parser via [`cxx`].
//!
//! This crate exposes an **eager, owned projection** of the Spectre typed AST
//! ([`netlist_syntax::spectre_ast`]) as plain-old-data structs shared with C++.
//! Rather than hand callers opaque node handles to traverse across the FFI
//! boundary (one heap allocation + one boundary crossing per node touched), we
//! walk the tree once in Rust and return fully-materialised value structs
//! (`rust::String`, `rust::Vec`) in a single transfer. For a consumer that
//! reads most of the netlist — building a simulator's parser tables, lowering
//! to a flat device list — that is both nicer to use and cheaper.
//!
//! Parameter/expression values are carried as **verbatim source text**: the
//! consumer re-parses them with its own expression engine (this mirrors how
//! VACASK round-trips expressions through `parseExpression`, and keeps this
//! layer free of a bespoke Spectre expression evaluator).

use netlist_syntax::ast::AstNode;
use netlist_syntax::spectre_ast as sast;
use netlist_syntax::{parse_spectre, SyntaxKind, SyntaxNode, SyntaxToken};

#[cxx::bridge(namespace = "netlist")]
mod ffi {
    /// A `name = value` parameter; `value` is verbatim expression source text.
    struct Param {
        name: String,
        value: String,
    }

    /// A device/subckt instance line.
    struct Instance {
        name: String,
        master: String,
        nodes: Vec<String>,
        params: Vec<Param>,
    }

    /// A `model` card.
    struct Model {
        name: String,
        master: String,
        params: Vec<Param>,
    }

    /// An analysis statement (`<name> <type> params…`).
    struct Analysis {
        name: String,
        analysis_type: String,
        nodes: Vec<String>,
        params: Vec<Param>,
    }

    /// One saved signal (`node` and optional `:modifier`).
    struct SaveItem {
        signal: String,
        modifier: String,
    }

    /// One initial-condition assignment (`node = value`).
    struct IcItem {
        node: String,
        value: String,
    }

    /// An `include`/`ahdl_include`; `section` is empty when absent.
    struct Include {
        path: String,
        section: String,
    }

    /// One clause of a conditional instantiation. `condition` is empty for the
    /// trailing `else`.
    struct CondClause {
        condition: String,
        instance: Instance,
    }

    /// A conditional-instantiation block (`if/else if/else` over instances).
    struct Conditional {
        clauses: Vec<CondClause>,
    }

    /// A `subckt`/`inline subckt` definition. Nested definitions are kept nested
    /// under `subckts`.
    struct Subckt {
        name: String,
        is_inline: bool,
        ports: Vec<String>,
        params: Vec<Param>,
        models: Vec<Model>,
        instances: Vec<Instance>,
        subckts: Vec<Subckt>,
        conditionals: Vec<Conditional>,
    }

    /// The byte span `[start, end)` of an error node in the source.
    struct ParseError {
        start: u32,
        end: u32,
    }

    /// The whole netlist, projected into grouped value structs.
    struct Netlist {
        params: Vec<Param>,
        models: Vec<Model>,
        subckts: Vec<Subckt>,
        instances: Vec<Instance>,
        analyses: Vec<Analysis>,
        saves: Vec<SaveItem>,
        ics: Vec<IcItem>,
        globals: Vec<String>,
        includes: Vec<Include>,
        ahdl_includes: Vec<String>,
        errors: Vec<ParseError>,
    }

    extern "Rust" {
        /// Parse Spectre source and project it into a flat, owned [`Netlist`].
        fn parse_spectre_netlist(src: &str) -> Netlist;
    }
}

// --- small text helpers ---

fn tok_text(t: Option<SyntaxToken>) -> String {
    t.map(|t| t.text().to_string()).unwrap_or_default()
}

/// Strip one layer of matching surrounding quotes from a string literal.
fn unquote(s: &str) -> String {
    let s = s.trim();
    let b = s.as_bytes();
    if b.len() >= 2 && (b[0] == b'"' || b[0] == b'\'') && b[b.len() - 1] == b[0] {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

// --- element projections ---

fn project_param(p: &sast::Parameter) -> ffi::Param {
    ffi::Param {
        name: tok_text(p.name()),
        value: p.value_text().unwrap_or_default(),
    }
}

fn project_instance(i: &sast::Instance) -> ffi::Instance {
    ffi::Instance {
        name: tok_text(i.name()),
        master: tok_text(i.master()),
        nodes: i.nodes().iter().map(sast::SNode::path).collect(),
        params: i.params().map(|p| project_param(&p)).collect(),
    }
}

fn project_model(m: &sast::Model) -> ffi::Model {
    ffi::Model {
        name: tok_text(m.name()),
        master: tok_text(m.master()),
        params: m.params().map(|p| project_param(&p)).collect(),
    }
}

fn project_analysis(a: &sast::Analysis) -> ffi::Analysis {
    ffi::Analysis {
        name: tok_text(a.name()),
        analysis_type: tok_text(a.analysis_type()),
        nodes: a.nodes().iter().map(sast::SNode::path).collect(),
        params: a.params().map(|p| project_param(&p)).collect(),
    }
}

fn project_conditional(c: &sast::ConditionalBlock) -> ffi::Conditional {
    let mut clauses = Vec::new();
    let mut push = |condition: String, inst: Option<sast::Instance>| {
        if let Some(inst) = inst {
            clauses.push(ffi::CondClause {
                condition,
                instance: project_instance(&inst),
            });
        }
    };
    if let Some(iff) = c.if_clause() {
        push(iff.condition().unwrap_or_default(), iff.body_instance());
    }
    for ei in c.else_ifs() {
        push(ei.condition().unwrap_or_default(), ei.body_instance());
    }
    if let Some(els) = c.else_clause() {
        push(String::new(), els.body_instance());
    }
    ffi::Conditional { clauses }
}

fn project_subckt(s: &sast::Subckt) -> ffi::Subckt {
    let scope = collect_scope(s.body());
    ffi::Subckt {
        name: tok_text(s.name()),
        is_inline: s.is_inline(),
        ports: s.ports().iter().map(sast::SNode::path).collect(),
        params: scope.params,
        models: scope.models,
        instances: scope.instances,
        subckts: scope.subckts,
        conditionals: scope.conditionals,
    }
}

/// The definition-level content shared by a netlist and a subckt body.
#[derive(Default)]
struct Scope {
    params: Vec<ffi::Param>,
    models: Vec<ffi::Model>,
    instances: Vec<ffi::Instance>,
    subckts: Vec<ffi::Subckt>,
    conditionals: Vec<ffi::Conditional>,
    analyses: Vec<ffi::Analysis>,
    saves: Vec<ffi::SaveItem>,
    ics: Vec<ffi::IcItem>,
    globals: Vec<String>,
    includes: Vec<ffi::Include>,
    ahdl_includes: Vec<String>,
}

fn collect_scope(stmts: impl Iterator<Item = SyntaxNode>) -> Scope {
    let mut scope = Scope::default();
    for stmt in stmts {
        match stmt.kind() {
            SyntaxKind::Model => {
                if let Some(m) = sast::Model::cast(stmt) {
                    scope.models.push(project_model(&m));
                }
            }
            SyntaxKind::Instance => {
                if let Some(i) = sast::Instance::cast(stmt) {
                    scope.instances.push(project_instance(&i));
                }
            }
            SyntaxKind::Subckt => {
                if let Some(s) = sast::Subckt::cast(stmt) {
                    scope.subckts.push(project_subckt(&s));
                }
            }
            SyntaxKind::Parameters => {
                if let Some(p) = sast::Parameters::cast(stmt) {
                    scope.params.extend(p.params().map(|p| project_param(&p)));
                }
            }
            SyntaxKind::ConditionalBlock => {
                if let Some(c) = sast::ConditionalBlock::cast(stmt) {
                    scope.conditionals.push(project_conditional(&c));
                }
            }
            SyntaxKind::Analysis => {
                if let Some(a) = sast::Analysis::cast(stmt) {
                    scope.analyses.push(project_analysis(&a));
                }
            }
            SyntaxKind::Save => {
                if let Some(s) = sast::Save::cast(stmt) {
                    scope.saves.extend(s.signals().map(|sig| ffi::SaveItem {
                        signal: sig.node().map(|n| n.path()).unwrap_or_default(),
                        modifier: sig.modifier().and_then(|m| m.value()).unwrap_or_default(),
                    }));
                }
            }
            SyntaxKind::Ic => {
                if let Some(ic) = sast::Ic::cast(stmt) {
                    scope.ics.extend(ic.params().map(|p| ffi::IcItem {
                        node: p.node().map(|n| n.path()).unwrap_or_default(),
                        value: p.value_text().unwrap_or_default(),
                    }));
                }
            }
            SyntaxKind::Global => {
                if let Some(g) = sast::Global::cast(stmt) {
                    scope.globals.extend(g.nodes().map(|n| n.path()));
                }
            }
            SyntaxKind::Include => {
                if let Some(inc) = sast::Include::cast(stmt) {
                    scope.includes.push(ffi::Include {
                        path: unquote(&tok_text(inc.path())),
                        section: inc
                            .section()
                            .and_then(|s| s.id())
                            .map(|t| t.text().to_string())
                            .unwrap_or_default(),
                    });
                }
            }
            SyntaxKind::AHDLInclude => {
                if let Some(a) = sast::AHDLInclude::cast(stmt) {
                    scope.ahdl_includes.push(unquote(&tok_text(a.path())));
                }
            }
            _ => {}
        }
    }
    scope
}

fn collect_errors(root: &SyntaxNode) -> Vec<ffi::ParseError> {
    root.descendants_with_tokens()
        .filter(|e| e.kind() == SyntaxKind::Error)
        .map(|e| {
            let r = e.text_range();
            ffi::ParseError {
                start: r.start().into(),
                end: r.end().into(),
            }
        })
        .collect()
}

pub fn parse_spectre_netlist(src: &str) -> ffi::Netlist {
    let root = parse_spectre(src);
    let errors = collect_errors(&root);
    let source = sast::SpectreNetlistSource::cast(root).expect("root is SpectreNetlistSource");
    let scope = collect_scope(source.statements());
    ffi::Netlist {
        params: scope.params,
        models: scope.models,
        subckts: scope.subckts,
        instances: scope.instances,
        analyses: scope.analyses,
        saves: scope.saves,
        ics: scope.ics,
        globals: scope.globals,
        includes: scope.includes,
        ahdl_includes: scope.ahdl_includes,
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_spectre_netlist;

    #[test]
    fn projects_top_level() {
        let nl = parse_spectre_netlist(
            "simulator lang=spectre\n\
             parameters vdd=1.8\n\
             r1 (a b) resistor r=1k\n\
             model nch bsim4 l=1u\n\
             stepResponse tran stop=100ns\n",
        );
        assert_eq!(nl.params.len(), 1);
        assert_eq!(nl.params[0].name, "vdd");
        assert_eq!(nl.params[0].value, "1.8");
        assert_eq!(nl.instances.len(), 1);
        assert_eq!(nl.instances[0].name, "r1");
        assert_eq!(nl.instances[0].master, "resistor");
        assert_eq!(nl.instances[0].nodes, vec!["a", "b"]);
        assert_eq!(nl.models.len(), 1);
        assert_eq!(nl.models[0].master, "bsim4");
        assert_eq!(nl.analyses.len(), 1);
        assert_eq!(nl.analyses[0].analysis_type, "tran");
    }

    #[test]
    fn projects_nested_subckt_and_conditional() {
        let nl = parse_spectre_netlist(
            "subckt s1 (d g s b)\n\
             parameters l=1u\n\
             if ( l < 0.5u ) {\n\
             m1 (d g s b) shortmod l=l\n\
             } else {\n\
             m2 (d g s b) longmod l=l\n\
             }\n\
             ends s1\n",
        );
        assert_eq!(nl.subckts.len(), 1);
        let s = &nl.subckts[0];
        assert_eq!(s.name, "s1");
        assert_eq!(s.ports, vec!["d", "g", "s", "b"]);
        assert_eq!(s.params.len(), 1);
        assert_eq!(s.conditionals.len(), 1);
        let cl = &s.conditionals[0].clauses;
        assert_eq!(cl.len(), 2);
        assert_eq!(cl[0].condition, "l < 0.5u");
        assert_eq!(cl[0].instance.name, "m1");
        assert_eq!(cl[1].condition, "");
        assert_eq!(cl[1].instance.master, "longmod");
    }

    #[test]
    fn collects_include_and_errors() {
        let nl = parse_spectre_netlist("include \"models.scs\" section=tt\n");
        assert_eq!(nl.includes.len(), 1);
        assert_eq!(nl.includes[0].path, "models.scs");
        assert_eq!(nl.includes[0].section, "tt");
    }
}
