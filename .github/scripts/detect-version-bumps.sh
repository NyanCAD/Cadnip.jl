#!/usr/bin/env bash
# Print the packages whose Project.toml `version` changed between two commits.
#
#   usage: detect-version-bumps.sh <before-sha> <after-sha>
#
# Output is one colon-separated record per changed package:
#
#   <name>:<subdir>:<version>:<tier>
#
# The separator is deliberately not a tab: `read` collapses runs of IFS
# *whitespace*, so the root package's empty subdir field would silently vanish
# and shift every later field left. Colons do not collapse.
#
# `subdir` is empty for the root package. `tier` orders registration so a
# package is never registered before an in-repo dependency it may have just
# bumped a compat bound for: NyanLexers (0) → the two parsers (1) → Cadnip (2).
# General's AutoMerge rejects a version whose compat bound nothing registered
# satisfies, so the order matters on the rare release that bumps both sides of
# an internal edge in one commit.
set -euo pipefail

before="${1:?usage: detect-version-bumps.sh <before-sha> <after-sha>}"
after="${2:?usage: detect-version-bumps.sh <before-sha> <after-sha>}"

# name : subdir : tier — keep in dependency order.
packages=(
  "NyanLexers:NyanLexers.jl:0"
  "NyanSpectreNetlistParser:NyanSpectreNetlistParser.jl:1"
  "NyanVerilogAParser:NyanVerilogAParser.jl:1"
  "Cadnip::2"
)

# Both commits must be present locally. A shallow checkout (actions/checkout
# defaults to fetch-depth: 1) silently lacks the pushed-from commit, which would
# otherwise read as "no package changed" and skip the whole release.
for sha in "${before}" "${after}"; do
  if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
    echo "detect-version-bumps: commit ${sha} is not present locally." >&2
    echo "  (a shallow checkout will do this — use actions/checkout with fetch-depth: 0)" >&2
    exit 1
  fi
done

# Empty when the manifest does not exist at that commit, which is the normal
# reading for a package added in this push.
version_at() {
  local sha="$1" path="$2"
  git cat-file -e "${sha}:${path}" 2>/dev/null || return 0
  git show "${sha}:${path}" |
    sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' |
    head -1
}

for entry in "${packages[@]}"; do
  IFS=: read -r name subdir tier <<<"${entry}"
  manifest="Project.toml"
  [ -n "${subdir}" ] && manifest="${subdir}/Project.toml"

  new_version="$(version_at "${after}" "${manifest}")"
  # No manifest at HEAD (deleted, or a path that moved) — nothing to register.
  [ -n "${new_version}" ] || continue

  # An empty old version means the package is new in this push; that is still a
  # bump worth registering.
  old_version="$(version_at "${before}" "${manifest}")"
  [ "${old_version}" = "${new_version}" ] && continue

  printf '%s:%s:%s:%s\n' "${name}" "${subdir}" "${new_version}" "${tier}"
done
