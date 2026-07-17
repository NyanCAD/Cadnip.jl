//! Losslessness: the CST must tile the source exactly, so `tree.text() == src`
//! for every corpus netlist. This is the Rust analogue of the Julia parser's
//! `check_roundtrip` (parse → print → identical source), covering the
//! reconstruct-source stage of the pipeline over the whole corpus at once.

use netlist_syntax::StartLang;
use std::fs;
use std::path::{Path, PathBuf};

fn corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/corpus")
}

/// Recursively collect Spectre corpus files (`.scs`/`.cir`) under `dir`,
/// including `quarantine/` — losslessness must hold even on the intentionally
/// malformed inputs the differential suite skips.
fn collect_spectre(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_spectre(&path, out);
            continue;
        }
        match path.extension().and_then(|e| e.to_str()) {
            Some("scs") | Some("cir") => out.push(path),
            _ => {}
        }
    }
}

#[test]
fn roundtrip_all_corpus() {
    let dir = corpus();
    let mut checked = 0;
    for entry in fs::read_dir(&dir).expect("read corpus dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("sp") {
            continue;
        }
        let src = fs::read_to_string(&path).unwrap();
        let tree = netlist_syntax::parse_spice(&src);
        let reprinted = tree.text().to_string();
        assert_eq!(
            reprinted,
            src,
            "\nlossless round-trip failed for {}: tree.text() != source",
            path.display()
        );
        checked += 1;
    }
    assert!(checked > 0, "no corpus files found in {}", dir.display());
    eprintln!("roundtrip: {checked} corpus files reconstruct their source exactly");
}

#[test]
fn roundtrip_all_spectre_corpus() {
    let dir = corpus().join("spectre");
    let mut files = Vec::new();
    collect_spectre(&dir, &mut files);
    files.sort();
    assert!(!files.is_empty(), "no spectre corpus files found in {}", dir.display());
    for path in &files {
        // The one documented exception to text==source: on an if/elseif/else
        // *body* failure the Julia parser double-captures the block name and
        // emits it twice (`parse_if` @trynext AND `parse_instance` @trynext).
        // The Rust port faithfully reproduces this (see `emit_phantom_identifier`
        // in spectre_parser.rs) so its *dump* byte-matches Julia — verified by
        // the differential suite. The cost is that this single error-recovery
        // input is intentionally non-lossless (and non-idempotent, since each
        // re-parse re-doubles), so it cannot satisfy text==source. Skip it here;
        // the differential test is its ground truth.
        if path.file_name().and_then(|s| s.to_str()) == Some("err_missing_rbrace.scs") {
            continue;
        }
        let src = fs::read_to_string(path).unwrap();
        // `.cir` opens in SPICE, `.scs` in Spectre; either may switch dialects.
        let start_lang = match path.extension().and_then(|e| e.to_str()) {
            Some("cir") => StartLang::Spice,
            _ => StartLang::Spectre,
        };
        let tree = netlist_syntax::parse_spectre_with(&src, start_lang);
        assert_eq!(
            tree.text().to_string(),
            src,
            "\nlossless round-trip failed for {}: tree.text() != source",
            path.display()
        );
    }
    eprintln!(
        "roundtrip: {} spectre corpus files reconstruct their source exactly",
        files.len()
    );
}
