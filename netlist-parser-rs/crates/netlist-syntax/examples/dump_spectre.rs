//! Throwaway differential helper: dump the Spectre CST for a file.
//! usage: dump_spectre <file.scs> [spice|spectre]
use netlist_syntax::StartLang;
use std::process::ExitCode;

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: dump_spectre <file> [spice|spectre]");
            return ExitCode::FAILURE;
        }
    };
    let start = match std::env::args().nth(2).as_deref() {
        Some("spice") => StartLang::Spice,
        _ => StartLang::Spectre,
    };
    let src = std::fs::read_to_string(&path).unwrap();
    let tree = netlist_syntax::parse_spectre_with(&src, start);
    print!("{}", netlist_syntax::dump::dump(&tree));
    ExitCode::SUCCESS
}
