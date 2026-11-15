#!/bin/bash
# Assemble and sign "Local Operator.lopruntime" — the launcher every LaunchAgent names.
#
# WHY A SEPARATE SCRIPT FROM assemble_keyagent_bundle.sh. The key agent needs a
# provisioning profile and entitlements, and most of that script exists to keep that
# pair consistent. The launcher has neither: it is a plain Developer-ID-signed Mach-O
# whose only job is to give Background Task Management a signed identity to name
# (see lop-launcher/launcher.c). Sharing one script would make every launcher build
# carry the profile machinery and every key agent change risk the launcher.
#
# What it shares with the key agent, deliberately: universal2 (the macOS wheel is
# tagged macosx_11_0_universal2), hardened runtime, and a SECURE TIMESTAMP that is
# read back rather than trusted — a Developer ID signature without one stops
# validating when the certificate expires, for every wheel already installed.
#
# usage:
#   assemble_launcher_bundle.sh --out BUNDLE --identity ID [--timestamp secure|none]
#
#   --identity -   ad-hoc signature, for LOCAL tests only: an ad-hoc launcher is never
#                  selected by update.daemon_program() (it requires team SHA2U6KT7V and
#                  a Radient Inc. leaf), so it cannot reach a user by accident.
#   --timestamp none  refuses unless LOP_LAUNCHER_ALLOW_UNTIMESTAMPED_BUILD=1; implied
#                  for an ad-hoc signature, which has no timestamp to read back.
#
# NEVER run this with an individual Developer ID (e.g. the login keychain's personal
# identity): the release job asserts the Radient Inc. authority after signing, and
# update.daemon_program() refuses any other leaf on the user's machine.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=""
IDENTITY=""
TIMESTAMP_MODE="secure"
SOURCE="$HERE/lop-launcher/launcher.c"

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --identity) IDENTITY="$2"; shift 2 ;;
    --timestamp) TIMESTAMP_MODE="$2"; shift 2 ;;
    *) echo "assemble_launcher_bundle.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "$OUT" ] || [ -z "$IDENTITY" ]; then
  echo "assemble_launcher_bundle.sh: --out and --identity are required" >&2
  exit 2
fi
if [ "$IDENTITY" = "-" ]; then
  TIMESTAMP_MODE="none"
elif [ "$TIMESTAMP_MODE" = "none" ] && [ "${LOP_LAUNCHER_ALLOW_UNTIMESTAMPED_BUILD:-}" != "1" ]; then
  echo "assemble_launcher_bundle.sh: refusing --timestamp=none for a Developer ID build;" >&2
  echo "set LOP_LAUNCHER_ALLOW_UNTIMESTAMPED_BUILD=1 for a local build. Never in a release job." >&2
  exit 2
fi
case "$TIMESTAMP_MODE" in
  secure) TS_KW=(--timestamp) ;;
  none) TS_KW=(--timestamp=none) ;;
  *) echo "assemble_launcher_bundle.sh: --timestamp takes 'secure' or 'none'" >&2; exit 2 ;;
esac

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/lop-launcher-stage.$$.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

clang -O2 -Wall -Wextra -Werror -arch arm64 -arch x86_64 -mmacosx-version-min=11 \
  -o "$STAGE/launcher" "$SOURCE"
ARCHS="$(lipo -archs "$STAGE/launcher")"
for arch in x86_64 arm64; do
  case " $ARCHS " in
    *" $arch "*) ;;
    *) echo "the launcher is not universal2: lipo -archs reports '$ARCHS'" >&2; exit 2 ;;
  esac
done
echo "  launcher arches: $ARCHS"

rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS"
cp "$HERE/lop-launcher/Info.plist" "$OUT/Contents/Info.plist"
cp "$STAGE/launcher" "$OUT/Contents/MacOS/Local Operator"
chmod 0755 "$OUT/Contents/MacOS/Local Operator"

# --identifier is pinned rather than left to Info.plist so a plist edit cannot change
# the identity update.daemon_program() verifies without this line changing too.
codesign --force --options runtime "${TS_KW[@]}" \
  --identifier com.local-operator.launcher --sign "$IDENTITY" "$OUT"
codesign --verify --strict --verbose=2 "$OUT"

if [ "$TIMESTAMP_MODE" = "secure" ]; then
  if ! codesign -dv --verbose=4 "$OUT" 2>&1 | grep -q '^Timestamp='; then
    echo "the launcher signature carries no secure timestamp; refusing to ship it" >&2
    exit 1
  fi
fi
echo "assembled $OUT"
