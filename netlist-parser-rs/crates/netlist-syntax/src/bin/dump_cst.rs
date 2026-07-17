//! Rust side of the differential test: parse a SPICE file and print the
//! canonical CST dump (see `netlist_syntax::dump`).

use std::process::ExitCode;

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: dump_cst <file.sp>");
            return ExitCode::FAILURE;
        }
    };
    let dialect = match std::env::args().nth(2).as_deref() {
        Some("hspice") => netlist_syntax::Dialect::Hspice,
        Some("pspice") => netlist_syntax::Dialect::Pspice,
        Some("xyce") => netlist_syntax::Dialect::Xyce,
        _ => netlist_syntax::Dialect::Ngspice,
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tree = netlist_syntax::parse_spice_dialect(&src, dialect);
    print!("{}", netlist_syntax::dump::dump(&tree));
    ExitCode::SUCCESS
}
