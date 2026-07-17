//! Differential test: the Rust CST dump must byte-match the captured Julia
//! parser dump for every corpus file in the spike grammar subset.
//!
//! The expected `.txt` files are canonical dumps produced by the Julia parser
//! (`NyanSpectreNetlistParser.jl/tools/dump_cst.jl`); regenerate them with
//! `netlist-parser-rs/tests/regen_expected.sh` when the Julia parser changes.

use std::fs;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    // crates/netlist-syntax/tests -> ../../tests/{corpus,expected}
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests")
}

fn check(name: &str) {
    let base = corpus_dir();
    let src = fs::read_to_string(base.join("corpus").join(format!("{name}.sp")))
        .unwrap_or_else(|e| panic!("read corpus/{name}.sp: {e}"));
    let expected = fs::read_to_string(base.join("expected").join(format!("{name}.txt")))
        .unwrap_or_else(|e| panic!("read expected/{name}.txt: {e}"));
    let got = netlist_syntax::dump::dump(&netlist_syntax::parse_spice(&src));
    assert_eq!(
        got.trim_end(),
        expected.trim_end(),
        "\n=== {name}.sp: Rust dump differs from Julia ground truth ===\n\
         --- rust ---\n{got}\n--- julia ---\n{expected}"
    );
}

macro_rules! diff_tests {
    ($($name:ident => $file:literal),* $(,)?) => {
        $(#[test] fn $name() { check($file); })*
    };
}

diff_tests! {
    basic_resistor      => "a",
    param_expression    => "b",
    subckt              => "c",
    error_recovery      => "d",
    voltage_current_src => "e",
    ternary_funcall     => "f",
    subckt_call         => "g",
    continuation_comment => "h",
    brace_prime         => "i",
    nested_subckt       => "j",
    real_rlc_filter     => "rlc",
    // Deep (mid-expression / mid-param-list) error recovery: exact Incomplete
    // nesting vs Julia.
    err_param_list      => "err_paramlist",
    err_binop_rhs       => "err_binop",
    err_missing_rparen  => "err_parens",
    err_ternary_colon   => "err_ternary",
    // Breadth batch 1: instance devices + analysis/dot-commands.
    b1_devices          => "b1_dev",
    b1_dot_commands     => "b1_dots",
    b1_includes         => "b1_inc",
    b1_osdi_device      => "b1_osdi",
    controlled_linear   => "controlled_linear",
    controlled_poly     => "controlled_poly",
    controlled_value    => "controlled_value",
    controlled_table    => "controlled_table",
    data_basic          => "data_basic",
    data_single_col     => "data_single_col",
    ic_basic            => "wf_ic_1",
    ic_coloned          => "wf_ic_2",
    ic_wildcard_multi   => "wf_ic_3",
    if_basic            => "if_basic",
    if_elseif           => "if_elseif",
    if_noelse           => "if_noelse",
    if_nested           => "if_nested",
    lib_block           => "lib_block",
    lib_block_named     => "lib_block_named",
    lib_include         => "lib_include",
    measure_point_find  => "wf_meas1",
    measure_point_when  => "wf_meas2",
    measure_range_trig  => "wf_meas3",
    measure_range_avg   => "wf_meas4",
    switch_basic        => "wf_switch_1",
    switch_wmod         => "wf_switch_2",
    // Real Julia SPICE example netlists (ported parse-test inputs).
    real_1n3064 => "ex_1n3064",
    real_comprt => "ex_comprt",
    real_data => "ex_data",
    real_global => "ex_global",
    real_hspice_isms => "ex_hspice_isms",
    real_ic0 => "ex_ic0",
    real_measure => "ex_measure",
    real_microcap => "ex_microcap",
    real_names => "ex_names",
    real_options => "ex_options",
    real_print => "ex_print",
    real_rc_ladder => "ex_rc_ladder",
    real_sources => "ex_sources",
    real_tran => "ex_tran",
    real_voltage_divider => "ex_voltage_divider",
}
