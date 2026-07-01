#!/usr/bin/env bash
#==============================================================================#
# Locate (or download + extract) a prebuilt VACASK release, and print the path
# to the extracted release directory on stdout (diagnostics go to stderr).
#
# Used by run_vacask.sh (throughput benchmarks) and the work-precision CI job.
# The WPD Julia harness (benchmarks/vacask/wpd/wpd_common.jl `locate_vacask`)
# finds the same cached binary at $CACHE_DIR/vacask/simulator/vacask.
#
# Usage:
#   VC="$(benchmarks/vacask/fetch_vacask.sh)"        # download if needed, get dir
#   "$VC/simulator/vacask" ...                       # run it
#
# Environment overrides:
#   VACASK_DIR   path to an already-extracted release (skips download)
#   VACASK_URL   release tarball URL
#   CACHE_DIR    where to cache the download (default: ~/.cache/cadnip-vacask)
#==============================================================================#
set -euo pipefail

VACASK_URL="${VACASK_URL:-https://github.com/pepijndevos/VACASK/releases/download/_0.3.3-dev/vacask_0.3.3-dev_linux-x86_64.tar.gz}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/cadnip-vacask}"

if [[ -n "${VACASK_DIR:-}" && -x "$VACASK_DIR/simulator/vacask" ]]; then
    echo "$VACASK_DIR"
    exit 0
fi

extracted="$CACHE_DIR/vacask"
if [[ ! -x "$extracted/simulator/vacask" ]]; then
    mkdir -p "$CACHE_DIR"
    tarball="$CACHE_DIR/vacask.tar.gz"
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
