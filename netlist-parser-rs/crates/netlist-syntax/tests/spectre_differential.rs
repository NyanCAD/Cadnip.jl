//! Spectre differential test: the Rust CST dump must byte-match the captured
//! Julia parser dump for every corpus file under `tests/corpus/spectre/**`.
//!
//! Mirrors `differential.rs` (the SPICE harness). The expected `.txt` files are
//! canonical dumps produced by the Julia parser
//! (`NyanSpectreNetlistParser.jl/tools/dump_spectre_cst.jl`).
//!
//! We exclude only the `quarantine/` directory (known-divergent / untestable
//! inputs). Language-switch files (`*.cir`, and `*.scs` containing `lang=spice`)
//! are now covered: `.cir` starts in SPICE, `.scs` starts in Spectre, and the
//! parser hands off across `simulator lang=` boundaries.
//! Every remaining corpus file is discovered by recursively walking the tree,
//! so new files are picked up automatically.

use netlist_syntax::StartLang;
use std::fs;
use std::path::{Path, PathBuf};

fn tests_dir() -> PathBuf {
    // crates/netlist-syntax/tests -> ../../tests
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests")
}

/// Recursively collect the spectre-only corpus files under `dir`.
fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().and_then(|s| s.to_str()) == Some("quarantine") {
                continue; // excluded this phase
            }
            collect(&path, out);
            continue;
        }
        let ext = path.extension().and_then(|s| s.to_str());
        if ext != Some("scs") && ext != Some("cir") {
            continue; // netlist sources only
        }
        out.push(path);
    }
}

#[test]
fn spectre_corpus_differential() {
    let base = tests_dir();
    let corpus = base.join("corpus/spectre");
    let expected_root = base.join("expected/spectre");

    let mut files = Vec::new();
    collect(&corpus, &mut files);
    files.sort();
    assert!(!files.is_empty(), "no spectre corpus files found");

    let mut failures = Vec::new();
    let mut count = 0usize;
    for path in &files {
        let rel = path.strip_prefix(&corpus).unwrap();
        let expected_path = expected_root.join(rel).with_extension("txt");
        let src = fs::read_to_string(path).unwrap();
        let expected = match fs::read_to_string(&expected_path) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{}: missing expected ({e})", rel.display()));
                continue;
            }
        };
        // `.cir` opens in SPICE, `.scs` in Spectre; either may switch dialects.
        let start_lang = match path.extension().and_then(|s| s.to_str()) {
            Some("cir") => StartLang::Spice,
            _ => StartLang::Spectre,
        };
        let got = netlist_syntax::dump::dump(&netlist_syntax::parse_spectre_with(&src, start_lang));
        count += 1;
        if got.trim_end() != expected.trim_end() {
            failures.push(format!(
                "=== {} ===\n--- rust ---\n{}\n--- julia ---\n{}",
                rel.display(),
                got.trim_end(),
                expected.trim_end()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} of {} spectre corpus files differ:\n\n{}",
            failures.len(),
            count,
            failures.join("\n\n")
        );
    }
}
