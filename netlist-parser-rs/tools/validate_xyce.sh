#!/usr/bin/env bash
# Validate that netlists are accepted by the real Xyce simulator (parse-only,
# `-syntax`). Used to ground the Xyce-dialect support: the Rust parser must
# accept (no Error nodes) exactly the netlists Xyce accepts.
#
#   XYCE=/path/to/Xyce.AppImage ./tools/validate_xyce.sh file1.cir [file2.cir ...]
set -uo pipefail
XYCE="${XYCE:-/home/pepijn/code/nyanodide/Xyce-build/Xyce-1.0.0-x86_64.AppImage}"
if [ ! -x "$XYCE" ]; then echo "Xyce not found at $XYCE (set XYCE)"; exit 2; fi
if [ $# -eq 0 ]; then
  cd "$(dirname "$0")/.." && set -- tests/xyce/*.cir
fi

fail=0
for f in "$@"; do
  out="$("$XYCE" -syntax "$f" 2>&1)"
  if echo "$out" | grep -q "Netlist syntax OK"; then
    echo "OK       $f"
  else
    echo "INVALID  $f"
    echo "$out" | grep -iE "error" | head -5 | sed 's/^/    /'
    fail=1
  fi
done
exit $fail
