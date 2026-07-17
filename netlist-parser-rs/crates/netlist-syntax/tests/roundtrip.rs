//! Losslessness: the CST must tile the source exactly, so `tree.text() == src`
//! for every corpus netlist. This is the Rust analogue of the Julia parser's
//! `check_roundtrip` (parse → print → identical source), covering the
//! reconstruct-source stage of the pipeline over the whole corpus at once.

use std::fs;
use std::path::PathBuf;

fn corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/corpus")
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
