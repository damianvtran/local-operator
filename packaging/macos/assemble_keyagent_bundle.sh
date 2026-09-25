#!/bin/bash
# Assemble and sign lop-keyagent.app — the one place the bundle's shape is built.
#
# WHY A SCRIPT AND NOT FIVE LINES IN THE WORKFLOW. Two of the steps below are
# failure modes that are either silent or unreportable, so they belong in one
# reviewed place that CI and a developer run identically:
#
#   * a bundle identifier that does not match the profile's application
#     identifier is SILENTLY BUILT and then SIGKILLed by the kernel on first use
#     (measured shape Q, exit 137) — so it is checked here, at build time;
#   * a bundle with no embedded profile is likewise a kernel kill, so the profile
#     is required, not optional;
#   * a signature that does not verify is a plain runtime failure, so the whole
#     thing is verified before it can ship.
#
# usage:
#   assemble_keyagent_bundle.sh --out APP --identity ID --profile FILE [options]
#
# options:
#   --binary FILE      prebuilt universal binary (default: compile from source)
#   --source FILE      the C source (default: lop-keyagent/se-keyagent.c)
#   --keychain FILE    a throwaway keychain holding the identity (CI)
#   --asan             build a sanitizer variant: -fsanitize=address, -O1 -g, and
#                      NO hardened runtime. Hardened runtime blocks the sanitizer
#                      runtime's library load, and the sanitizer variant is a QA
#                      artefact that is never shipped — the shipped build below
#                      always carries `--options runtime` (kept for notarization,
#                      which needs it; measured shape S shows the entitlement
#                      itself does not).
#
# The identity's password is never handled here: the caller imports the p12.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=""
IDENTITY=""
PROFILE=""
KEYCHAIN=""
BINARY=""
SOURCE="$HERE/lop-keyagent/se-keyagent.c"
ASAN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --identity) IDENTITY="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --keychain) KEYCHAIN="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --asan) ASAN=1; shift ;;
    *) echo "assemble_keyagent_bundle.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "$OUT" ] || [ -z "$IDENTITY" ] || [ -z "$PROFILE" ]; then
  echo "assemble_keyagent_bundle.sh: --out, --identity and --profile are required" >&2
  exit 2
fi
[ -f "$PROFILE" ] || { echo "no provisioning profile at $PROFILE" >&2; exit 2; }

APP="$OUT"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/lop-keyagent-stage.$$.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

# --- 1. the binary ----------------------------------------------------------
# BOTH SLICES, because the wheel is tagged macosx_11_0_universal2 and an arm64
# runner would otherwise ship an arm64-only helper under a universal2 tag. The
# SDK cross-compiles both; -mmacosx-version-min=11 matches Info.plist's floor.
if [ -z "$BINARY" ]; then
  BINARY="$STAGE/lop-keyagent"
  if [ "$ASAN" = "1" ]; then
    clang -fsanitize=address -g -O1 -arch arm64 -arch x86_64 -mmacosx-version-min=11 \
      -o "$BINARY" "$SOURCE" -framework Security -framework CoreFoundation
  else
    clang -O2 -arch arm64 -arch x86_64 -mmacosx-version-min=11 \
      -o "$BINARY" "$SOURCE" -framework Security -framework CoreFoundation
  fi
fi
[ -f "$BINARY" ] || { echo "no binary at $BINARY" >&2; exit 2; }

# --- 2. the bundle ----------------------------------------------------------
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp "$HERE/lop-keyagent/Info.plist" "$APP/Contents/Info.plist"
cp "$BINARY" "$APP/Contents/MacOS/lop-keyagent"
chmod 0755 "$APP/Contents/MacOS/lop-keyagent"
# LOAD-BEARING, see the file header: without this the entitled bundle is killed
# by the kernel rather than refused, so it is copied here and never optional.
cp "$PROFILE" "$APP/Contents/embedded.provisionprofile"
chmod 0644 "$APP/Contents/embedded.provisionprofile"

# --- 3. the profile must authorize exactly what we claim --------------------
# Read with python3 + plistlib rather than `plutil -extract Entitlements.
# com.apple.application-identifier`: that key contains dots, which plutil reads
# as path separators, so the obvious one-liner silently extracts nothing.
python3 - "$PROFILE" "$HERE/keyagent.entitlements" "$APP/Contents/Info.plist" <<'PY'
import plistlib, subprocess, sys

profile_path, ent_path, plist_path = sys.argv[1:4]
profile = plistlib.loads(subprocess.run(
    ["security", "cms", "-D", "-i", profile_path], capture_output=True, check=True
).stdout)
entitlements = plistlib.loads(open(ent_path, "rb").read())
info = plistlib.loads(open(plist_path, "rb").read())

profile_app_id = profile.get("Entitlements", {}).get("com.apple.application-identifier")
claimed_app_id = entitlements.get("com.apple.application-identifier")
bundle_id = info.get("CFBundleIdentifier")

if not profile_app_id:
    sys.exit("the profile carries no com.apple.application-identifier")
if claimed_app_id != profile_app_id:
    sys.exit(
        f"the profile authorizes {profile_app_id!r} but the bundle is signed with "
        f"{claimed_app_id!r}; the kernel would SIGKILL this bundle on first use"
    )
# The bundle id is the app id without its team prefix, and the profile's app id
# must be exactly <team>.<bundle id> — anything else and the signature's
# identifier and the profile's authorization describe two different apps.
team, _, suffix = profile_app_id.partition(".")
if suffix != bundle_id:
    sys.exit(
        f"Info.plist's CFBundleIdentifier is {bundle_id!r} but the profile's app id "
        f"{profile_app_id!r} names {suffix!r}"
    )
print(f"profile authorizes {profile_app_id} for bundle {bundle_id} (team {team})")
PY

# --- 4. sign ---------------------------------------------------------------
# `--options runtime` is kept for the SHIPPED build: notarization requires it and
# notarization is the next step on this path, while the entitlement itself does
# not need it (measured shape S: bundle + profile, no hardened runtime, PASS).
CODE_KW=()
[ -n "$KEYCHAIN" ] && CODE_KW=(--keychain "$KEYCHAIN")
# macOS ships bash 3.2, where "${arr[@]}" on an EMPTY array trips `set -u`; the
# ${arr[@]+"${arr[@]}"} idiom is the portable spelling (same as run_probe.sh).
if [ "$ASAN" = "1" ]; then
  codesign --force ${CODE_KW[@]+"${CODE_KW[@]}"} \
    --entitlements "$HERE/keyagent.entitlements" --sign "$IDENTITY" "$APP"
else
  codesign --force --options runtime ${CODE_KW[@]+"${CODE_KW[@]}"} \
    --entitlements "$HERE/keyagent.entitlements" --sign "$IDENTITY" "$APP"
fi

# --- 5. verify -------------------------------------------------------------
codesign --verify --strict --verbose=2 "$APP"
echo "assembled and signed $APP"
codesign -d --entitlements - --verbose=2 "$APP" 2>&1 | sed 's/^/  /'
