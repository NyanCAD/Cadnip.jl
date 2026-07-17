#!/usr/bin/env bash
# Validate that corpus netlists are real, valid ngspice (they parse without a
# syntax/fatal error under ngspice -b). Intended for the `cov_*` coverage
# netlists, which are authored to be runnable ngspice; edge-case/malformed
# corpus files (err_*, some ex_* fragments) are NOT expected to pass and are
# skipped unless named explicitly.
#
#   ./tools/validate_ngspice.sh                 # validates tests/corpus/cov_*.sp
#   ./tools/validate_ngspice.sh path/to/file.sp # validates one file
set -uo pipefail
cd "$(dirname "$0")/.."
NGSPICE="${NGSPICE:-ngspice}"
if ! command -v "$NGSPICE" >/dev/null; then
  echo "ngspice not found (set NGSPICE)"; exit 2
fi

files=("$@")
if [ ${#files[@]} -eq 0 ]; then
  files=(tests/corpus/cov_*.sp)
fi

fail=0
for f in "${files[@]}"; do
  out="$("$NGSPICE" -b -o /dev/null "$f" 2>&1)"
  # ngspice prints "Error on line ..." / "fatal" / "can't find" for bad input.
  if echo "$out" | grep -qiE '^\s*(error|fatal)\b|syntax error|unrecognized'; then
    echo "INVALID  $f"
    echo "$out" | grep -iE 'error|fatal|syntax|unrecognized' | head -5 | sed 's/^/    /'
    fail=1
  else
    echo "OK       $f"
  fi
done
exit $fail
