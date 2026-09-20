#!/usr/bin/env bash
#
# Run the cross-platform probe battery inside Linux containers, one per distro.
#
# Why containers for Linux at all, when CI already has ubuntu-latest: CI has
# exactly ONE Linux (ubuntu-latest, which is Ubuntu LTS). A user on Linux Mint,
# Debian, Fedora or a different LTS point release is running a different
# userland, and the mechanisms lop depends on -- a user systemd instance,
# `notify-send`, `xdg-open`, the terminal it is launched from -- are exactly
# the things a distro changes. This script is how those differences get a
# reading instead of an assumption.
#
#   scripts/xplat_linux_matrix.sh                    # ubuntu + mint
#   scripts/xplat_linux_matrix.sh ubuntu:24.04       # one distro
#   OUT=~/workspace/xplat-audit/matrix scripts/xplat_linux_matrix.sh
#   PROBE_ARGS="--only tui cli" scripts/xplat_linux_matrix.sh
#
# Output: one JSON per image under $OUT, a combined table on stdout, and a
# non-zero exit if any probe FAILed on any image.
#
# On Windows: this script is Linux-only BY CONSTRUCTION, and that is not a gap
# that can be closed here. Docker on a macOS or Linux host runs a Linux kernel,
# and a Windows container image needs a Windows kernel underneath it. Windows
# validation therefore happens on a real Windows runner -- see the
# `xplat-probe-windows` job in `.github/workflows/ci.yml` -- rather than in a
# Wine or emulation container that would not be evidence of anything.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$HOME/workspace/xplat-audit/matrix}"
PROBE_ARGS="${PROBE_ARGS:-}"
IMAGES=("$@")
if [ ${#IMAGES[@]} -eq 0 ]; then
  IMAGES=("ubuntu:24.04" "linuxmintd/mint22-amd64")
fi

mkdir -p "$OUT"
overall=0

for image in "${IMAGES[@]}"; do
  slug="$(printf '%s' "$image" | tr '/:' '__')"
  tag="lop-xplat:$slug"
  echo "=== $image -> $tag ==="
  if ! docker build -f "$REPO/scripts/xplat/Dockerfile.probe" \
      --build-arg "BASE=$image" -t "$tag" "$REPO"; then
    echo "BUILD FAILED for $image" >&2
    overall=1
    continue
  fi
  # `--init` so the container reaps children: the probe starts a server and a
  # daemon, and a container with no init leaves their orphans parented to
  # nothing.
  # shellcheck disable=SC2086 -- PROBE_ARGS is a deliberate word-split.
  docker run --rm --init -v "$OUT:/out" "$tag" \
    /venv/bin/python scripts/xplat_probe.py --json "/out/$slug.json" $PROBE_ARGS
  rc=$?
  echo "  -> rc=$rc"
  [ "$rc" -eq 0 ] || overall=1
  # A FAIL is a finding, not a reason to abort: the NEXT distro's result is the
  # comparison that makes the finding legible.
done

echo
echo "=== combined ==="
python3 "$REPO/scripts/xplat_report.py" "$OUT"
report_rc=$?
[ "$report_rc" -eq 0 ] || overall=1

echo "JSON written to $OUT"
exit "$overall"
