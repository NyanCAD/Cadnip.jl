#!/usr/bin/env bash
#==============================================================================#
# VACASK reference benchmark runner
#
# Runs the *real* VACASK simulator on every benchmark case using the official
# upstream methodology (benchmark.py: 1 warmup run + N timed runs, averaged).
#
# This gives apples-to-apples reference numbers on the same machine that runs
# the Cadnip benchmarks (benchmarks/vacask/run_benchmarks.jl), instead of the
# upstream-published numbers in README.md which were measured on a different
# machine (AMD Threadripper 7970).
#
# Usage:
#   ./run_vacask.sh [output.md]
#
# When an output file is given, a machine-readable <output>.tsv is written
# alongside it (columns: benchmark, time_s, timepoints, rejected, iterations).
# run_benchmarks.jl reads this TSV (via $VACASK_REFERENCE_TSV) to fold the
# reference numbers into its own comparison tables.
#
# The VACASK binary is located in this order:
#   1. $VACASK_DIR              -- a pre-extracted release dir (must contain
#                                  simulator/vacask and simulator/lib)
#   2. a previously cached download under $CACHE_DIR
#   3. downloaded from $VACASK_URL (pinned release below)
#
# Environment overrides:
#   VACASK_DIR   path to an extracted release (skips download)
#   VACASK_URL   release tarball URL
#   CACHE_DIR    where to cache the download (default: ~/.cache/cadnip-vacask)
#   RUNS         number of timed runs (default 5)
#   CASES        space-separated case list (default: rc graetz mul ring c6288)
#==============================================================================#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VACASK_URL="${VACASK_URL:-https://github.com/pepijndevos/VACASK/releases/download/_0.3.3-dev/vacask_0.3.3-dev_linux-x86_64.tar.gz}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/cadnip-vacask}"
RUNS="${RUNS:-5}"
CASES="${CASES:-rc graetz mul ring c6288}"
OUT="${1:-}"

#------------------------------------------------------------------------------#
# Locate (or fetch) the VACASK release
#------------------------------------------------------------------------------#
locate_vacask() {
    if [[ -n "${VACASK_DIR:-}" && -x "$VACASK_DIR/simulator/vacask" ]]; then
        echo "$VACASK_DIR"
        return
    fi
    mkdir -p "$CACHE_DIR"
    local extracted="$CACHE_DIR/vacask"
    if [[ ! -x "$extracted/simulator/vacask" ]]; then
        local tarball="$CACHE_DIR/vacask.tar.gz"
        if [[ ! -f "$tarball" ]]; then
            echo "Downloading VACASK release..." >&2
            curl -fsSL -o "$tarball" "$VACASK_URL"
        fi
        echo "Extracting VACASK release..." >&2
        rm -rf "$extracted"
        mkdir -p "$extracted"
        tar xzf "$tarball" -C "$extracted" --strip-components=1
    fi
    chmod +x "$extracted/simulator/vacask" "$extracted/simulator/openvaf-r" 2>/dev/null || true
    echo "$extracted"
}

VC="$(locate_vacask)"
BIN="$VC/simulator/vacask"
export LD_LIBRARY_PATH="$VC/simulator/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VERSION="$("$BIN" --help 2>&1 | head -1 | sed 's/This is //; s/\.$//')"
CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"

echo "Simulator: $VERSION" >&2
echo "Binary:    $BIN" >&2
echo "CPU:       $CPU" >&2
echo "Runs:      $RUNS (+1 warmup) per case" >&2
echo >&2

#------------------------------------------------------------------------------#
# Run each case through the official benchmark.py and collect results
#------------------------------------------------------------------------------#
emit() { if [[ -n "$OUT" ]]; then echo "$1" >>"$OUT"; else echo "$1"; fi; }

TSV=""
if [[ -n "$OUT" ]]; then
    : >"$OUT"
    TSV="${OUT%.md}.tsv"
    : >"$TSV"
fi
tsv() { [[ -n "$TSV" ]] && printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >>"$TSV"; }

emit "# VACASK reference benchmark results"
emit ""
emit "Measured with the upstream methodology (\`benchmark.py\`: 1 warmup + $RUNS timed runs, averaged)."
emit ""
emit "- Simulator: \`$VERSION\`"
emit "- CPU: $CPU"
emit ""
emit "| Benchmark | Time (s) | Rel. std | Timepoints | Rejected | Iterations |"
emit "|-----------|----------|----------|------------|----------|------------|"

# Names must match the benchmark names in run_benchmarks.jl so the TSV can be
# joined into its comparison tables.
declare -A NAMES=(
    [rc]="RC Circuit"
    [graetz]="Graetz Bridge"
    [mul]="Voltage Multiplier"
    [ring]="Ring Oscillator (PSP103)"
    [c6288]="C6288 Multiplier"
)

for c in $CASES; do
    log="$(mktemp)"
    echo "Running $c ..." >&2
    python3 "$HERE/benchmark.py" -n "$RUNS" -nd 1 "$HERE/$c/vacask" "$BIN" >"$log" 2>&1 || {
        echo "  FAILED (see below)" >&2; tail -20 "$log" >&2; emit "| ${NAMES[$c]:-$c} | FAILED | - | - | - | - |"; rm -f "$log"; continue;
    }
    avg="$(grep -E 'Average runtime|Runtime:' "$log" | tail -1 | grep -oE '[0-9.]+')"
    rel="$(grep -E 'relative:' "$log" | tail -1 | grep -oE '[0-9.]+' || echo '')"
    tp="$(grep -E 'Accepted timepoints:' "$log" | tail -1 | grep -oE '[0-9]+')"
    rej="$(grep -E 'Rejected timepoints:' "$log" | tail -1 | grep -oE '[0-9]+')"
    it="$(grep -E 'NR solver iterations:' "$log" | tail -1 | grep -oE '[0-9]+')"
    relpct=""
    [[ -n "$rel" ]] && relpct="$(awk -v r="$rel" 'BEGIN{printf "%.1f%%", r*100}')"
    avgfmt="$(awk -v a="$avg" 'BEGIN{printf "%.2f", a}')"
    emit "| ${NAMES[$c]:-$c} | $avgfmt | ${relpct:--} | ${tp:--} | ${rej:--} | ${it:--} |"
    tsv "${NAMES[$c]:-$c}" "$avg" "${tp:-0}" "${rej:-0}" "${it:-0}"
    echo "  done: ${avgfmt}s, ${tp} timepoints, ${rej} rejected, ${it} iterations" >&2
    rm -f "$log"
done

[[ -n "$OUT" ]] && echo "Report written to: $OUT" >&2
echo "Done." >&2
