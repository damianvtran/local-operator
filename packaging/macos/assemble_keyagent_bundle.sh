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

# --- 1b. BOTH SLICES, ASSERTED ON THE FILE ----------------------------------
# The wheel is tagged ``macosx_11_0_universal2``, and a tag is not a binary: the
# ``--binary`` option above skips the compile entirely, so a single-arch helper could be
# placed under that tag and an Intel Mac would install a wheel it cannot execute. The
# compile path can silently degrade too, if a future SDK or flag combination stops
# cross-building. ``lipo`` is the only thing that reads the Mach-O headers, so it is the
# check — and it runs AFTER the compile/``--binary`` branch, so it covers both.
#
# ``-archs`` prints the architectures in no guaranteed order (measured: ``x86_64 arm64``
# from this SDK), so membership is asserted per arch rather than as a string.
ARCHS="$(lipo -archs "$BINARY" 2>/dev/null || true)"
for arch in x86_64 arm64; do
  case " $ARCHS " in
    *" $arch "*) ;;
    *)
      echo "the helper is not universal2: lipo -archs reports '$ARCHS', which has no $arch slice." >&2
      echo "The wheel tag claims macosx_11_0_universal2, so this would ship a bundle an Intel" >&2
      echo "Mac cannot execute. Rebuild with '-arch arm64 -arch x86_64' (the default path)." >&2
      exit 2
      ;;
  esac
done
echo "  binary arches: $ARCHS"

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
import datetime, plistlib, subprocess, sys

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

# ---- IS THE PROFILE USABLE, NOT MERELY PRESENT? (agent review round 1, R1-2) ----
# The checks above prove the blob names the right app. A ROTATED-OUT-OF-STEP OR LAPSED
# profile names the right app too, and it ships: at runtime it passes the pre-flight's
# string test and then meets the operator as a refused or killed key agent. Both facts
# below are free here — the profile is already decoded — and neither was checked anywhere
# (``grep -rn Expiration`` found the topic only in the design document's prose). The
# certificate pairing needs the SIGNATURE rather than the identity name, so it is checked
# in step 5b, after signing; the header above says so rather than leaving it unstated.
expiry = profile.get("ExpirationDate")
if not isinstance(expiry, datetime.datetime):
    sys.exit("the profile carries no ExpirationDate")
if expiry.tzinfo is None:
    expiry = expiry.replace(tzinfo=datetime.timezone.utc)
now = datetime.datetime.now(datetime.timezone.utc)
if expiry <= now:
    sys.exit(
        f"the profile expired on {expiry.date().isoformat()} — a lapsed profile ships a key "
        "agent the OS refuses, so this must not be notarized or released"
    )
if not profile.get("DeveloperCertificates"):
    sys.exit("the profile carries no DeveloperCertificates, so nothing authorizes a signature")

team_ids = profile.get("TeamIdentifier") or []
if team_ids and team not in team_ids:
    sys.exit(
        f"the profile's TeamIdentifier is {team_ids}, which does not include the signing "
        f"team {team!r}"
    )
print(f"profile authorizes {profile_app_id} for bundle {bundle_id} (team {team})")
print(f"profile is current to {expiry.date().isoformat()}")
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

# --- 5b. THE PROFILE AND THE SIGNATURE MUST NAME THE SAME CERTIFICATE --------
# THE PAIRING HALF OF THE PROFILE CHECK, and it runs here because it needs the
# SIGNATURE: step 3 has only the profile, so it can establish that the profile is
# current and authorizes this assignment, but not that the certificate Apple will match
# it against is the certificate this bundle was signed with. A bundle whose profile
# authorizes a different certificate is exactly the shape that ships and then fails on
# the operator's first verb.
#
# Read from the SIGNATURE rather than looked up by identity NAME: the job may pass
# ``--identity`` as a SHA-1 hash, and comparing names would compare printable forms of the
# fact instead of the certificate itself.
python3 - "$APP" "$PROFILE" "$STAGE" <<'PY'
import plistlib, subprocess, sys

app, profile_path, stage = sys.argv[1:4]
profile = plistlib.loads(subprocess.run(
    ["security", "cms", "-D", "-i", profile_path], capture_output=True, check=True
).stdout)
certificates = profile.get("DeveloperCertificates") or []
if not certificates:
    sys.exit("the profile carries no DeveloperCertificates, so nothing authorizes a signature")
prefix = f"{stage}/signature-chain"
subprocess.run(
    ["codesign", "--display", f"--extract-certificates={prefix}", app],
    capture_output=True, check=True,
)
leaf = open(f"{prefix}0", "rb").read()
if leaf not in certificates:
    sys.exit(
        "the profile does not name the certificate that signed this bundle: it authorizes "
        f"{len(certificates)} certificate(s), none of which is the signing leaf "
        f"({len(leaf)} DER bytes) — the kernel would refuse the bundle on first use"
    )
print(f"profile names the signing certificate ({len(leaf)} DER bytes)")
PY

echo "assembled and signed $APP"
codesign -d --entitlements - --verbose=2 "$APP" 2>&1 | sed 's/^/  /'
