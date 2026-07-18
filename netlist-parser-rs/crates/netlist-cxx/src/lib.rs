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

/// Extract the value text from a `VoltageControl` or `CurrentControl` syntax node.
///
/// Controlled-source gain/transconductance values are emitted by the parser as
/// bare `NumberLiteral` tokens rather than wrapped `LiteralExpr` nodes, so the
/// typed-AST `value()` accessor (which uses `expr_children`) misses them.  This
/// helper tries expression-node children first (covers `BinaryExpression`, etc.)
/// then falls back to scanning direct token children for `NumberLiteral`/`Literal`.
fn ctrl_value_text(node: &SyntaxNode) -> String {
    // Expression-node children first (wraps complex expressions).
    for child in node.children() {
        if !matches!(child.kind(), SyntaxKind::HierarchialNode) {
            return child.text().to_string();
        }
    }
    // Fall back: bare NumberLiteral or Literal token child.
    for el in node.children_with_tokens() {
        if let Some(tok) = el.into_token() {
            if matches!(tok.kind(), SyntaxKind::NumberLiteral | SyntaxKind::Literal) {
                return tok.text().to_string();
            }
        }
    }
    String::new()
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

/// Project a single SPICE device `SyntaxNode` into a `SpiceDevice`, or return
/// `None` for non-device nodes (statements, `.model`, `.subckt`, …).
///
/// This helper is shared by `project_spice_block` and `project_spice_subckt`
/// so that subckt bodies receive the same device coverage as the top-level block.
fn project_spice_device(child: SyntaxNode) -> Option<ffi::SpiceDevice> {
    match child.kind() {
        SyntaxKind::Resistor => ast::Resistor::cast(child).map(|r| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::Resistor,
            name: hier_text(r.name()),
            nodes: vec![hier_text(r.pos()), hier_text(r.neg())],
            value: r.value().map(|e| e.text()).unwrap_or_default(),
            model: String::new(),
            params: r.params().map(|p| project_spice_param(&p)).collect(),
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: empty_source(),
        }),
        SyntaxKind::Capacitor => ast::Capacitor::cast(child).map(|c| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::Capacitor,
            name: hier_text(c.name()),
            nodes: vec![hier_text(c.pos()), hier_text(c.neg())],
            value: c.value().map(|e| e.text()).unwrap_or_default(),
            model: String::new(),
            params: c.params().map(|p| project_spice_param(&p)).collect(),
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: empty_source(),
        }),
        SyntaxKind::Inductor => ast::Inductor::cast(child).map(|l| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::Inductor,
            name: hier_text(l.name()),
            nodes: vec![hier_text(l.pos()), hier_text(l.neg())],
            value: l.value().map(|e| e.text()).unwrap_or_default(),
            model: String::new(),
            params: l.params().map(|p| project_spice_param(&p)).collect(),
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: empty_source(),
        }),
        SyntaxKind::Voltage => ast::Voltage::cast(child).map(|v| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::VSource,
            name: hier_text(v.name()),
            nodes: vec![hier_text(v.pos()), hier_text(v.neg())],
            value: String::new(),
            model: String::new(),
            params: vec![],
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: project_spice_source(v.sources()),
        }),
        SyntaxKind::Current => ast::Current::cast(child).map(|i| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::ISource,
            name: hier_text(i.name()),
            nodes: vec![hier_text(i.pos()), hier_text(i.neg())],
            value: String::new(),
            model: String::new(),
            params: vec![],
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: project_spice_source(i.sources()),
        }),
        SyntaxKind::Diode => ast::Diode::cast(child).map(|d| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::Diode,
            name: hier_text(d.name()),
            nodes: vec![hier_text(d.pos()), hier_text(d.neg())],
            value: String::new(),
            model: hier_text(d.model()),
            params: d.params().map(|p| project_spice_param(&p)).collect(),
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: empty_source(),
        }),
        SyntaxKind::MOSFET => ast::MOSFET::cast(child).map(|m| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::Mosfet,
            name: hier_text(m.name()),
            nodes: vec![
                hier_text(m.drain()),
                hier_text(m.gate()),
                hier_text(m.source()),
                hier_text(m.bulk()),
            ],
            value: String::new(),
            model: hier_text(m.model()),
            params: m.params().map(|p| project_spice_param(&p)).collect(),
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: empty_source(),
        }),
        SyntaxKind::BipolarTransistor => ast::BipolarTransistor::cast(child).map(|q| {
            // `nodes()` yields ALL HierarchialNode children in source order:
            // [instance-name, c, b, e, [s,] model] — mirroring ngspice syntax
            // `Q <c> <b> <e> [<s>] <model>`. Skip the first entry (instance name),
            // treat the last entry as the model reference, and the remainder as
            // terminal connections. A follow-up can refine substrate (4th terminal)
            // vs model disambiguation if needed.
            let all: Vec<String> = q.nodes().map(|n| n.text()).collect();
            // Guard: a malformed bare `Q1` has len<=1 → slice [1..0] would panic.
            let (terminals, model) = if all.len() < 2 {
                (Vec::new(), String::new())
            } else {
                (all[1..all.len() - 1].to_vec(), all[all.len() - 1].clone())
            };
            ffi::SpiceDevice {
                kind: ffi::SpiceDeviceKind::Bjt,
                name: hier_text(q.name()),
                nodes: terminals,
                value: String::new(),
                model,
                params: q.params().map(|p| project_spice_param(&p)).collect(),
                ctrl_nodes: vec![],
                ctrl_value: String::new(),
                source: empty_source(),
            }
        }),
        SyntaxKind::SubcktCall => ast::SubcktCall::cast(child).map(|x| {
            // `nodes()` yields ALL HierarchialNode children in source order:
            // [instance-name, conn1, conn2, ..., master-subckt]. Skip the first
            // entry (instance name), treat the last as the master reference, and
            // the remainder as the connection nodes.
            let all: Vec<String> = x.nodes().map(|n| n.text()).collect();
            // Guard: a malformed bare `X1` has len<=1 → slice [1..0] would panic.
            let (conn_nodes, model) = if all.len() < 2 {
                (Vec::new(), String::new())
            } else {
                (all[1..all.len() - 1].to_vec(), all[all.len() - 1].clone())
            };
            ffi::SpiceDevice {
                kind: ffi::SpiceDeviceKind::SubcktCall,
                name: hier_text(x.name()),
                nodes: conn_nodes,
                value: String::new(),
                model,
                params: x.params().map(|p| project_spice_param(&p)).collect(),
                ctrl_nodes: vec![],
                ctrl_value: String::new(),
                source: empty_source(),
            }
        }),
        // --- Controlled sources (E/F/G/H) ---
        //
        // The parser emits a single `ControlledSource` node for all four letters.
        // The control kind distinguishes voltage- vs current-controlled:
        //   VoltageControl → E (Vcvs) or G (Vccs)
        //   CurrentControl → F (Cccs) or H (Ccvs)
        // The final E-vs-G / F-vs-H disambiguation uses the device-name prefix.
        // ctrl_nodes: VoltageControl → control_nodes(); CurrentControl → [vnam()]
        // ctrl_value: gain/transconductance (bare NumberLiteral token in the CST)
        // POLY/TABLE: captured best-effort as ctrl_value text; full parsing deferred.
        SyntaxKind::ControlledSource => ast::ControlledSource::cast(child).and_then(|cs| {
            let name = hier_text(cs.name());
            let kind = match name.chars().next().map(|c| c.to_ascii_uppercase()) {
                Some('E') => ffi::SpiceDeviceKind::Vcvs,
                Some('F') => ffi::SpiceDeviceKind::Cccs,
                Some('G') => ffi::SpiceDeviceKind::Vccs,
                Some('H') => ffi::SpiceDeviceKind::Ccvs,
                _ => return None,
            };
            let nodes = vec![hier_text(cs.pos()), hier_text(cs.neg())];
            let (ctrl_nodes, ctrl_value, params) = match cs.control() {
                Some(ctrl) => match ctrl.kind() {
                    SyntaxKind::VoltageControl => {
                        if let Some(vc) = ast::VoltageControl::cast(ctrl) {
                            let cnodes: Vec<String> =
                                vc.control_nodes().map(|n| n.text()).collect();
                            let cval = ctrl_value_text(vc.syntax());
                            let ps: Vec<ffi::Param> =
                                vc.params().map(|p| project_spice_param(&p)).collect();
                            (cnodes, cval, ps)
                        } else {
                            (vec![], String::new(), vec![])
                        }
                    }
                    SyntaxKind::CurrentControl => {
                        if let Some(cc) = ast::CurrentControl::cast(ctrl) {
                            let vnam = hier_text(cc.vnam());
                            let cval = ctrl_value_text(cc.syntax());
                            let ps: Vec<ffi::Param> =
                                cc.params().map(|p| project_spice_param(&p)).collect();
                            (vec![vnam], cval, ps)
                        } else {
                            (vec![], String::new(), vec![])
                        }
                    }
                    // POLY/TABLE: capture full node text as ctrl_value (deferred).
                    _ => (vec![], ctrl.text().to_string(), vec![]),
                },
                None => (vec![], String::new(), vec![]),
            };
            Some(ffi::SpiceDevice {
                kind,
                name,
                nodes,
                value: String::new(),
                model: String::new(),
                params,
                ctrl_nodes,
                ctrl_value,
                source: empty_source(),
            })
        }),

        // --- Mutual inductor (K) ---
        // nodes = [L1, L2, ...]; coupling value stored in `value`.
        SyntaxKind::MutualInductor => ast::MutualInductor::cast(child).map(|k| {
            let nodes: Vec<String> = k.inductors().map(|n| n.text()).collect();
            let coupling = ctrl_value_text(k.syntax());
            ffi::SpiceDevice {
                kind: ffi::SpiceDeviceKind::MutualInductor,
                name: hier_text(k.name()),
                nodes,
                value: coupling,
                model: String::new(),
                params: vec![],
                ctrl_nodes: vec![],
                ctrl_value: String::new(),
                source: empty_source(),
            }
        }),

        // --- Voltage-controlled / current-controlled switch (S/W) ---
        // nodes() yields nd1, nd2, cnd1, cnd2, model in order (after name).
        SyntaxKind::Switch => ast::Switch::cast(child).map(|sw| {
            let nodes: Vec<String> = sw.nodes().map(|n| n.text()).collect();
            ffi::SpiceDevice {
                kind: ffi::SpiceDeviceKind::Switch,
                name: hier_text(sw.name()),
                nodes,
                value: String::new(),
                model: String::new(),
                params: vec![],
                ctrl_nodes: vec![],
                ctrl_value: String::new(),
                source: empty_source(),
            }
        }),

        // --- Behavioral source (B) ---
        SyntaxKind::Behavioral => ast::Behavioral::cast(child).map(|b| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::Behavioral,
            name: hier_text(b.name()),
            nodes: vec![hier_text(b.pos()), hier_text(b.neg())],
            value: String::new(),
            model: String::new(),
            params: b.params().map(|p| project_spice_param(&p)).collect(),
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: empty_source(),
        }),

        // --- JFET (J) ---
        // Accessor layout mirrors MOSFET (drain/gate/source/model via fet_device!).
        SyntaxKind::JFET => ast::JFET::cast(child).map(|j| ffi::SpiceDevice {
            kind: ffi::SpiceDeviceKind::Jfet,
            name: hier_text(j.name()),
            nodes: vec![hier_text(j.drain()), hier_text(j.gate()), hier_text(j.source())],
            value: j.area().map(|e| e.text()).unwrap_or_default(),
            model: hier_text(j.model()),
            params: j.params().map(|p| project_spice_param(&p)).collect(),
            ctrl_nodes: vec![],
            ctrl_value: String::new(),
            source: empty_source(),
        }),

        // --- OSDI device (N) ---
        // nodes() = [conn..., model] (all after name); last is the model reference.
        // Guard: empty list → no panic.
        SyntaxKind::OSDIDevice => ast::OSDIDevice::cast(child).map(|n| {
            let all: Vec<String> = n.nodes().map(|h| h.text()).collect();
            let (terminals, model) = if all.is_empty() {
                (Vec::new(), String::new())
            } else {
                (all[..all.len() - 1].to_vec(), all[all.len() - 1].clone())
            };
            ffi::SpiceDevice {
                kind: ffi::SpiceDeviceKind::Osdi,
                name: hier_text(n.name()),
                nodes: terminals,
                value: String::new(),
                model,
                params: n.params().map(|p| project_spice_param(&p)).collect(),
                ctrl_nodes: vec![],
                ctrl_value: String::new(),
                source: empty_source(),
            }
        }),

        _ => None,
    }
}

/// Project a SPICE `.model` card into a `SpiceModel`.
/// Extracts `level` from the params list if present (case-insensitive match).
fn project_spice_model(m: &ast::Model) -> ffi::SpiceModel {
    let params: Vec<ffi::Param> = m.params().map(|p| project_spice_param(&p)).collect();
    let level = params
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case("level"))
        .map(|p| p.value.clone())
        .unwrap_or_default();
    ffi::SpiceModel {
        name: hier_text(m.name()),
        model_type: tok_text(m.model_type()),
        level,
        params,
    }
}

/// Project a SPICE `.subckt` definition into a `SpiceSubckt`, recursively
/// walking the body for nested devices, `.model` cards, and `.subckt` definitions.
fn project_spice_subckt(s: &ast::Subckt) -> ffi::SpiceSubckt {
    let mut devices = vec![];
    let mut models = vec![];
    let mut subckts = vec![];
    for child in s.body() {
        match child.kind() {
            SyntaxKind::Model => {
                if let Some(m) = ast::Model::cast(child) {
                    models.push(project_spice_model(&m));
                }
            }
            SyntaxKind::Subckt => {
                if let Some(sub) = ast::Subckt::cast(child) {
                    subckts.push(project_spice_subckt(&sub));
                }
            }
            _ => {
                if let Some(dev) = project_spice_device(child) {
                    devices.push(dev);
                }
            }
        }
    }
    ffi::SpiceSubckt {
        name: tok_text(s.name()),
        ports: s.ports().map(|p| p.text()).collect(),
        params: s.params().map(|p| project_spice_param(&p)).collect(),
        devices,
        models,
        subckts,
    }
}

/// Project a `SPICENetlistSource` node into a `SpiceBlock`.
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
            SyntaxKind::Model => {
                if let Some(m) = ast::Model::cast(child) {
                    block.models.push(project_spice_model(&m));
                }
            }
            SyntaxKind::Subckt => {
                if let Some(s) = ast::Subckt::cast(child) {
                    block.subckts.push(project_spice_subckt(&s));
                }
            }
            // .include "path" → Include { path, section: "" }
            SyntaxKind::IncludeStatement => {
                if let Some(inc) = ast::IncludeStatement::cast(child) {
                    block.includes.push(ffi::Include {
                        path: unquote(&tok_text(inc.path())),
                        section: String::new(),
                    });
                }
            }
            // .lib "path" section  → Include { path, section }
            SyntaxKind::LibInclude => {
                if let Some(lib) = ast::LibInclude::cast(child) {
                    block.includes.push(ffi::Include {
                        path: unquote(&tok_text(lib.path())),
                        section: tok_text(lib.section()),
                    });
                }
            }
            _ => {
                if let Some(dev) = project_spice_device(child) {
                    block.devices.push(dev);
                }
            }
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
    fn projects_spice_semiconductors_and_subckt() {
        let src = "* t\n\
            D1 a k dmod\n\
            M1 d g s b nch w=1u l=0.1u\n\
            Q1 c b e qmod\n\
            X1 in out amp gain=2\n\
            .model dmod d is=1e-14\n\
            .subckt amp in out\n R1 in out 1k\n .ends\n";
        let nl = super::parse_netlist(src, true);
        let b = &nl.spice_blocks[0];
        // MOSFET
        let m = b.devices.iter().find(|x| x.name == "M1").unwrap();
        assert_eq!(m.kind, super::ffi::SpiceDeviceKind::Mosfet);
        assert_eq!(m.nodes, vec!["d", "g", "s", "b"]);
        assert_eq!(m.model, "nch");
        // Diode
        let d = b.devices.iter().find(|x| x.name == "D1").unwrap();
        assert_eq!(d.kind, super::ffi::SpiceDeviceKind::Diode);
        assert_eq!(d.nodes, vec!["a", "k"]);
        assert_eq!(d.model, "dmod");
        // BJT: nodes() = [Q1, c, b, e, qmod]; skip name, last is model
        let q = b.devices.iter().find(|x| x.name == "Q1").unwrap();
        assert_eq!(q.kind, super::ffi::SpiceDeviceKind::Bjt);
        assert_eq!(q.nodes, vec!["c", "b", "e"]);
        assert_eq!(q.model, "qmod");
        // SubcktCall: nodes() = [X1, in, out, amp]; skip name, last is master
        let x = b.devices.iter().find(|x| x.name == "X1").unwrap();
        assert_eq!(x.kind, super::ffi::SpiceDeviceKind::SubcktCall);
        assert_eq!(x.nodes, vec!["in", "out"]);
        assert_eq!(x.model, "amp");
        // .model card
        assert_eq!(b.models.len(), 1);
        assert_eq!(b.models[0].name, "dmod");
        assert_eq!(b.models[0].model_type, "d");
        // .subckt definition
        assert_eq!(b.subckts.len(), 1);
        assert_eq!(b.subckts[0].name, "amp");
        assert_eq!(b.subckts[0].ports, vec!["in", "out"]);
        assert_eq!(b.subckts[0].devices.len(), 1);
        assert_eq!(b.subckts[0].devices[0].name, "R1");
    }

    /// Regression test: malformed BJT/SubcktCall with ≤1 node children must not
    /// panic. A bare `Q1` and `X1` (no terminals, no model) hit the len<2 guard
    /// and project with empty nodes and model rather than slicing out-of-bounds.
    #[test]
    fn projects_malformed_bjt_subckt_no_panic() {
        // Bare device names only — parser produces minimal/error AST nodes with
        // very few HierarchialNode children, exercising the len<2 guard.
        let nl = super::parse_netlist("* t\nQ1\nX1\n", true);
        // Must not panic; any devices that project should have empty nodes/model.
        for dev in &nl.spice_blocks[0].devices {
            assert!(dev.nodes.is_empty(), "malformed device should have no nodes");
            assert!(dev.model.is_empty(), "malformed device should have no model");
        }
    }

    /// E/F/G/H controlled sources + .include collected into SpiceBlock.includes.
    #[test]
    fn projects_controlled_sources_and_include() {
        let src = "* t\n\
            E1 out 0 in 0 2.0\n\
            G1 o 0 a b 1e-3\n\
            F1 o 0 vsense 10\n\
            .include \"models.spice\"\n";
        let nl = super::parse_netlist(src, true);
        let b = &nl.spice_blocks[0];
        // E1 — voltage-controlled voltage source
        let e = b.devices.iter().find(|x| x.name == "E1").unwrap();
        assert_eq!(e.kind, super::ffi::SpiceDeviceKind::Vcvs);
        assert_eq!(e.nodes, vec!["out", "0"]);
        assert_eq!(e.ctrl_nodes, vec!["in", "0"]);
        assert_eq!(e.ctrl_value, "2.0");
        // G1 — voltage-controlled current source
        let g = b.devices.iter().find(|x| x.name == "G1").unwrap();
        assert_eq!(g.kind, super::ffi::SpiceDeviceKind::Vccs);
        assert_eq!(g.ctrl_nodes, vec!["a", "b"]);
        assert_eq!(g.ctrl_value, "1e-3");
        // F1 — current-controlled current source
        let f = b.devices.iter().find(|x| x.name == "F1").unwrap();
        assert_eq!(f.kind, super::ffi::SpiceDeviceKind::Cccs);
        assert_eq!(f.ctrl_nodes, vec!["vsense"]);
        assert_eq!(f.ctrl_value, "10");
        // include
        assert_eq!(b.includes.len(), 1);
        assert_eq!(b.includes[0].path, "models.spice");
    }

    /// Smoke test: one netlist covering every device kind; checks count + no errors.
    #[test]
    fn projects_full_breadth() {
        let src = "* full breadth\n\
            R1 a b 1k\n\
            C1 b 0 1u\n\
            L1 a b 2n\n\
            V1 vin 0 DC 5\n\
            I1 0 vin DC 1m\n\
            D1 a k dmod\n\
            M1 d g s b nch w=1u\n\
            Q1 c b e qmod\n\
            X1 in out amp\n\
            E1 out 0 in 0 2.0\n\
            G1 o 0 a b 1e-3\n\
            F1 o 0 vsense 10\n\
            H1 o 0 vsense 5\n\
            K1 L1 L2 0.9\n\
            S1 nd1 nd2 cnd1 cnd2 smod ON\n\
            B1 out 0 V=1+2\n\
            J1 d g s jmod\n\
            .model dmod d\n\
            .subckt amp in out\n R1 in out 1k\n .ends\n\
            .include \"models.spice\"\n\
            .lib \"mylib.sp\" tt\n";
        let nl = super::parse_netlist(src, true);
        assert!(nl.errors.is_empty(), "unexpected parse errors: {} error(s)", nl.errors.len());
        let b = &nl.spice_blocks[0];
        // 17 device lines (R C L V I D M Q X E G F H K S B J)
        assert_eq!(b.devices.len(), 17, "expected 17 devices, got {}: {:?}",
            b.devices.len(),
            b.devices.iter().map(|d| &d.name).collect::<Vec<_>>());
        // .include + .lib → 2 includes
        assert_eq!(b.includes.len(), 2);
        assert_eq!(b.includes[0].path, "models.spice");
        assert_eq!(b.includes[1].path, "mylib.sp");
        assert_eq!(b.includes[1].section, "tt");
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
