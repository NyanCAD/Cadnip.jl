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

use netlist_syntax::ast::{self, AstNode};
use netlist_syntax::spectre_ast as sast;
use netlist_syntax::{parse_spectre_with, StartLang, SyntaxKind, SyntaxNode, SyntaxToken};

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
        spice_blocks: Vec<SpiceBlock>,
    }

    /// The byte span `[start, end)` of an error node in the source.
    struct ParseError {
        start: u32,
        end: u32,
    }

    // --- SPICE block schema (full enum + structs; later tasks populate more
    // device kinds — defined in full now so the C++ side compiles against the
    // complete schema) ---

    #[derive(Debug, Clone, Copy)]
    enum SpiceDeviceKind {
        Resistor,
        Capacitor,
        Inductor,
        VSource,
        ISource,
        Diode,
        Mosfet,
        Bjt,
        Jfet,
        SubcktCall,
        Vcvs,
        Vccs,
        Ccvs,
        Cccs,
        MutualInductor,
        Behavioral,
        Switch,
        Osdi,
    }

    /// Voltage/current source waveform data (DC, AC, and transient).
    struct SpiceSource {
        dc: String,
        ac_mag: String,
        ac_phase: String,
        tran_kind: String,
        tran_args: Vec<String>,
    }

    /// A single SPICE device instance (R, C, L, V, I, D, M, …).
    struct SpiceDevice {
        kind: SpiceDeviceKind,
        name: String,
        nodes: Vec<String>,
        value: String,
        model: String,
        params: Vec<Param>,
        ctrl_nodes: Vec<String>,
        ctrl_value: String,
        source: SpiceSource,
    }

    /// A SPICE `.model` card.
    struct SpiceModel {
        name: String,
        model_type: String,
        level: String,
        params: Vec<Param>,
    }

    /// A SPICE `.subckt` definition (nested definitions kept nested).
    struct SpiceSubckt {
        name: String,
        ports: Vec<String>,
        params: Vec<Param>,
        devices: Vec<SpiceDevice>,
        models: Vec<SpiceModel>,
        subckts: Vec<SpiceSubckt>,
    }

    /// One contiguous SPICE block (everything between two `simulator lang=`
    /// switches, or the whole file when starting in SPICE).
    struct SpiceBlock {
        params: Vec<Param>,
        models: Vec<SpiceModel>,
        subckts: Vec<SpiceSubckt>,
        devices: Vec<SpiceDevice>,
        includes: Vec<Include>,
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
        spice_blocks: Vec<SpiceBlock>,
    }

    extern "Rust" {
        /// Parse netlist source and project it into a flat, owned `Netlist`.
        /// `start_spice` selects the starting dialect (`.cir` → true / SPICE,
        /// `.scs` → false / Spectre); a `simulator lang=` line may still switch
        /// mid-file.
        fn parse_netlist(src: &str, start_spice: bool) -> Netlist;
        /// Back-compat: parse starting in the Spectre dialect.
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
        spice_blocks: scope.spice_blocks,
    }
}

// --- SPICE projection helpers ---

/// Source text of an optional `HierarchialNode`.
fn hier_text(h: Option<ast::HierarchialNode>) -> String {
    h.map(|n| n.text()).unwrap_or_default()
}

/// Project a SPICE `Parameter` (name-only or `name=value`).
fn project_spice_param(p: &ast::Parameter) -> ffi::Param {
    ffi::Param {
        name: tok_text(p.name()),
        value: p.value().map(|e| e.text()).unwrap_or_default(),
    }
}

fn empty_source() -> ffi::SpiceSource {
    ffi::SpiceSource {
        dc: String::new(),
        ac_mag: String::new(),
        ac_phase: String::new(),
        tran_kind: String::new(),
        tran_args: vec![],
    }
}

/// Build a `SpiceSource` by walking the `DCSource`/`ACSource`/`TranSource`
/// child nodes yielded by `Voltage::sources()` or `Current::sources()`.
fn project_spice_source(sources: impl Iterator<Item = SyntaxNode>) -> ffi::SpiceSource {
    let mut out = empty_source();
    for src in sources {
        match src.kind() {
            SyntaxKind::DCSource => {
                if let Some(dc) = ast::DCSource::cast(src) {
                    out.dc = dc.value().map(|e| e.text()).unwrap_or_default();
                }
            }
            SyntaxKind::ACSource => {
                if let Some(ac) = ast::ACSource::cast(src) {
                    out.ac_mag = ac.magnitude().map(|e| e.text()).unwrap_or_default();
                    out.ac_phase = ac.phase().map(|e| e.text()).unwrap_or_default();
                }
            }
            SyntaxKind::TranSource => {
                if let Some(tran) = ast::TranSource::cast(src) {
                    out.tran_kind = tok_text(tran.function());
                    out.tran_args = tran.values().map(|e| e.text()).collect();
                }
            }
            _ => {}
        }
    }
    out
}

/// Project a `SPICENetlistSource` node into a `SpiceBlock`.
/// R/C/L are projected now; other device kinds fall through (`_ => {}`).
fn project_spice_block(node: SyntaxNode) -> ffi::SpiceBlock {
    let mut block = ffi::SpiceBlock {
        params: vec![],
        models: vec![],
        subckts: vec![],
        devices: vec![],
        includes: vec![],
    };
    for child in node.children() {
        match child.kind() {
            SyntaxKind::Resistor => {
                if let Some(r) = ast::Resistor::cast(child) {
                    block.devices.push(ffi::SpiceDevice {
                        kind: ffi::SpiceDeviceKind::Resistor,
                        name: hier_text(r.name()),
                        nodes: vec![hier_text(r.pos()), hier_text(r.neg())],
                        value: r.value().map(|e| e.text()).unwrap_or_default(),
                        model: String::new(),
                        params: r.params().map(|p| project_spice_param(&p)).collect(),
                        ctrl_nodes: vec![],
                        ctrl_value: String::new(),
                        source: empty_source(),
                    });
                }
            }
            SyntaxKind::Capacitor => {
                if let Some(c) = ast::Capacitor::cast(child) {
                    block.devices.push(ffi::SpiceDevice {
                        kind: ffi::SpiceDeviceKind::Capacitor,
                        name: hier_text(c.name()),
                        nodes: vec![hier_text(c.pos()), hier_text(c.neg())],
                        value: c.value().map(|e| e.text()).unwrap_or_default(),
                        model: String::new(),
                        params: c.params().map(|p| project_spice_param(&p)).collect(),
                        ctrl_nodes: vec![],
                        ctrl_value: String::new(),
                        source: empty_source(),
                    });
                }
            }
            SyntaxKind::Inductor => {
                if let Some(l) = ast::Inductor::cast(child) {
                    block.devices.push(ffi::SpiceDevice {
                        kind: ffi::SpiceDeviceKind::Inductor,
                        name: hier_text(l.name()),
                        nodes: vec![hier_text(l.pos()), hier_text(l.neg())],
                        value: l.value().map(|e| e.text()).unwrap_or_default(),
                        model: String::new(),
                        params: l.params().map(|p| project_spice_param(&p)).collect(),
                        ctrl_nodes: vec![],
                        ctrl_value: String::new(),
                        source: empty_source(),
                    });
                }
            }
            SyntaxKind::Voltage => {
                if let Some(v) = ast::Voltage::cast(child) {
                    block.devices.push(ffi::SpiceDevice {
                        kind: ffi::SpiceDeviceKind::VSource,
                        name: hier_text(v.name()),
                        nodes: vec![hier_text(v.pos()), hier_text(v.neg())],
                        value: String::new(),
                        model: String::new(),
                        params: vec![],
                        ctrl_nodes: vec![],
                        ctrl_value: String::new(),
                        source: project_spice_source(v.sources()),
                    });
                }
            }
            SyntaxKind::Current => {
                if let Some(i) = ast::Current::cast(child) {
                    block.devices.push(ffi::SpiceDevice {
                        kind: ffi::SpiceDeviceKind::ISource,
                        name: hier_text(i.name()),
                        nodes: vec![hier_text(i.pos()), hier_text(i.neg())],
                        value: String::new(),
                        model: String::new(),
                        params: vec![],
                        ctrl_nodes: vec![],
                        ctrl_value: String::new(),
                        source: project_spice_source(i.sources()),
                    });
                }
            }
            _ => {}
        }
    }
    block
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
    spice_blocks: Vec<ffi::SpiceBlock>,
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
            SyntaxKind::SPICENetlistSource => {
                scope.spice_blocks.push(project_spice_block(stmt));
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

pub fn parse_netlist(src: &str, start_spice: bool) -> ffi::Netlist {
    let lang = if start_spice { StartLang::Spice } else { StartLang::Spectre };
    let root = parse_spectre_with(src, lang);
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
        spice_blocks: scope.spice_blocks,
    }
}

/// Back-compat wrapper: parse starting in Spectre.
pub fn parse_spectre_netlist(src: &str) -> ffi::Netlist {
    parse_netlist(src, false)
}

#[cfg(test)]
mod tests {
    use super::{parse_netlist, parse_spectre_netlist};

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

    #[test]
    fn projects_spice_block_passives() {
        let nl = super::parse_netlist("* t\nR1 a b 1k\nC1 b 0 1u\nL1 a b 2n\n", true);
        assert!(nl.errors.is_empty());
        assert_eq!(nl.spice_blocks.len(), 1);
        let d = &nl.spice_blocks[0].devices;
        assert_eq!(d.len(), 3);
        assert_eq!(d[0].kind, super::ffi::SpiceDeviceKind::Resistor);
        assert_eq!(d[0].name, "R1");
        assert_eq!(d[0].nodes, vec!["a", "b"]);
        assert_eq!(d[0].value, "1k");
        assert_eq!(d[1].kind, super::ffi::SpiceDeviceKind::Capacitor);
        assert_eq!(d[1].name, "C1");
        assert_eq!(d[1].nodes, vec!["b", "0"]);
        assert_eq!(d[1].value, "1u");
        assert_eq!(d[2].kind, super::ffi::SpiceDeviceKind::Inductor);
        assert_eq!(d[2].name, "L1");
        assert_eq!(d[2].nodes, vec!["a", "b"]);
        assert_eq!(d[2].value, "2n");
    }

    #[test]
    fn projects_spice_sources() {
        let nl = super::parse_netlist(
            "* t\nV1 1 0 DC 5 AC 1 PULSE(0 5 1m 1u 1u 4m 10m)\nI1 0 2 DC 1m\n",
            true,
        );
        assert!(nl.errors.is_empty(), "unexpected parse errors");
        assert_eq!(nl.spice_blocks.len(), 1);
        let d = &nl.spice_blocks[0].devices;
        // V1
        let v = d.iter().find(|x| x.name == "V1").unwrap();
        assert_eq!(v.kind, super::ffi::SpiceDeviceKind::VSource);
        assert_eq!(v.nodes, vec!["1", "0"]);
        assert_eq!(v.source.dc, "5");
        assert_eq!(v.source.ac_mag, "1");
        assert_eq!(v.source.ac_phase, "");
        assert_eq!(v.source.tran_kind.to_lowercase(), "pulse");
        assert_eq!(v.source.tran_args, vec!["0", "5", "1m", "1u", "1u", "4m", "10m"]);
        // I1
        let cur = d.iter().find(|x| x.name == "I1").unwrap();
        assert_eq!(cur.kind, super::ffi::SpiceDeviceKind::ISource);
        assert_eq!(cur.nodes, vec!["0", "2"]);
        assert_eq!(cur.source.dc, "1m");
        assert_eq!(cur.source.tran_kind, "");
        assert!(cur.source.tran_args.is_empty());
    }

    #[test]
    fn start_language_dispatch() {
        // SPICE block, then switch to Spectre, then a Spectre instance.
        let src = "* title\nR1 a b 1k\nsimulator lang=spectre\nr2 (a b) resistor r=2k\n";

        // Start in SPICE: the leading SPICE block parses cleanly; after the
        // `simulator lang=spectre` switch, control returns to the Spectre driver
        // and the trailing Spectre instance r2 projects. (SPICE-device projection
        // of the leading block is deferred to Task 7 — not asserted here.)
        let spice = parse_netlist(src, /*start_spice=*/ true);
        assert!(spice.errors.is_empty(), "spice-start should parse cleanly");
        assert_eq!(spice.instances.len(), 1);
        assert_eq!(spice.instances[0].name, "r2");

        // Start in Spectre: the same leading SPICE line `R1 a b 1k` is invalid
        // Spectre and produces error node(s).
        let spectre = parse_netlist(src, /*start_spice=*/ false);
        assert!(!spectre.errors.is_empty(), "spectre-start should error on the SPICE line");
    }
}
