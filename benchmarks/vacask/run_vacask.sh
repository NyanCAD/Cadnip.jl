#!/usr/bin/env bash
#==============================================================================#
# VACASK reference benchmark runner
#
# Runs the *real* VACASK simulator on every benchmark case using the official
# upstream run-count methodology (benchmark.py: 1 warmup run + N timed runs).
# The reported time is VACASK's own solve-only "Elapsed time" (parsed from its
# stdout), not benchmark.py's wall-clock around the whole subprocess - the
# latter also includes process spawn, netlist parse, and OSDI model load,
# which would unfairly inflate VACASK's numbers next to Cadnip's solve-only
# @benchmark timing in run_benchmarks.jl.
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
#   CASES        space-separated case list (default: rc graetz mul darlington ring c6288)
#==============================================================================#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS="${RUNS:-5}"
CASES="${CASES:-rc graetz mul darlington ring c6288}"
OUT="${1:-}"

#------------------------------------------------------------------------------#
# Locate (or fetch) the VACASK release (shared with the work-precision CI job)
#------------------------------------------------------------------------------#
VC="$(bash "$HERE/fetch_vacask.sh")"
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
emit "Measured with the upstream run-count methodology (\`benchmark.py\`: 1 warmup + $RUNS timed runs). Time is VACASK's own solve-only elapsed time (not process/parse overhead), averaged over the $RUNS timed runs."
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
    [darlington]="Darlington Pair"
    [ring]="Ring Oscillator (PSP103)"
    [c6288]="C6288 Multiplier"
)

for c in $CASES; do
    log="$(mktemp)"
    echo "Running $c ..." >&2
    python3 "$HERE/benchmark.py" -n "$RUNS" -nd 1 "$HERE/$c/vacask" "$BIN" >"$log" 2>&1 || {
        echo "  FAILED (see below)" >&2; tail -20 "$log" >&2; emit "| ${NAMES[$c]:-$c} | FAILED | - | - | - | - |"; rm -f "$log"; continue;
    }
    # VACASK prints its own solve-only "Elapsed time: X" after each tran
    # analysis, unconditionally - independent of benchmark.py's own wall-clock
    # timer (grep target above, now unused), which also includes process
    # spawn, netlist parse, and OSDI model load. This is the same
    # solve-vs-binary-runtime distinction fixed in the work-precision
    # benchmark (benchmarks/vacask/wpd/), applied here too so Cadnip-vs-VACASK
    # timing is apples-to-apples. benchmark.py runs 1 warmup ("-nd 1") + $RUNS
    # timed invocations in order (each case has exactly one `analysis` in its
    # .sim, so exactly one Elapsed-time line per invocation): skip the first
    # (warmup) reading and average the remaining $RUNS.
    read -r avg rel <<<"$(grep -oE 'Elapsed time:[[:space:]]*[0-9.eE+-]+' "$log" \
        | grep -oE '[0-9.eE+-]+$' | tail -n +2 \
        | awk '{a[NR]=$1; s+=$1} END{
            if (NR==0) { print "nan nan"; exit }
            m=s/NR
            if (NR>1) { for(i=1;i<=NR;i++) ss+=(a[i]-m)^2; r=sqrt(ss/(NR-1))/m } else r=""
            printf "%.6f %s\n", m, r
        }')"
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
