#!/usr/bin/env bash
# Build the cdylib, compile the C smoke test against the generated header, run it.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> netlist-parser-rs
cargo build -p netlist-cabi
CRATE=crates/netlist-cabi
CC="${CC:-cc}"
"$CC" -std=c11 -Wall -Wextra \
    -I "$CRATE/include" \
    "$CRATE/tests/c_smoke.c" \
    -L target/debug -lnetlist_cabi \
    -o target/debug/c_smoke
LD_LIBRARY_PATH="target/debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" target/debug/c_smoke
