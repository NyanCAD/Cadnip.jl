#!/usr/bin/env bash
# Regenerate the expected CST dumps from the Julia parser (ground truth for the
# differential test). Requires a Julia environment with the local
# NyanSpectreNetlistParser dev-added. Set JULIA_ENV to that project dir.
#
#   JULIA_ENV=/path/to/env ./tests/regen_expected.sh
#
# To create such an env:
#   julia --project=$JULIA_ENV -e 'using Pkg;
#     Pkg.develop(path="../../NyanSpectreNetlistParser.jl");
#     Pkg.add("AbstractTrees"); Pkg.instantiate()'
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd ../.. && pwd)"   # netlist-parser-rs lives at <repo>/netlist-parser-rs
DUMP="$REPO_ROOT/NyanSpectreNetlistParser.jl/tools/dump_cst.jl"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
: "${JULIA_ENV:?set JULIA_ENV to a project dir with NyanSpectreNetlistParser dev-added}"

for f in tests/corpus/*.sp; do
  name="$(basename "$f" .sp)"
  "$JULIA" --project="$JULIA_ENV" "$DUMP" "$f" > "tests/expected/$name.txt"
  echo "regenerated tests/expected/$name.txt"
done
