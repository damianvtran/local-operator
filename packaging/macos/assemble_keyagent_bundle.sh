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
#     thing is verified before it can ship;
#   * a Developer ID signature WITHOUT a secure timestamp stops validating the day
#     the signing certificate expires, which the design document's rotation plan
#     (§8) explicitly does not assume: an already-installed wheel would start being
#     refused on that date. `codesign`'s own default already requests one for a
#     Developer ID identity, and relying on that silently is exactly how the
#     property goes unasserted — a TSA that is unreachable, or a default that
#     changes, leaves an artefact that looks fine here and dies in 2031. So the mode
#     is named (`--timestamp`, default `secure`) and the SIGNED RESULT is read back
#     for the timestamp rather than trusted; the one mode that skips that read-back
#     refuses unless a named, deliberate non-release override is set (R3-1).
#
# usage:
#   assemble_keyagent_bundle.sh --out APP --identity ID --profile FILE [options]
#
# options:
#   --binary FILE      prebuilt universal binary (default: compile from source)
#   --source FILE      the C source (default: lop-keyagent/se-keyagent.c)
#   --keychain FILE    a throwaway keychain holding the identity, for a caller that
#                      has one (the local probe rigs under tools/lop-signing). WHAT
#                      IT DOES AND DOES NOT DO, because assuming it is sufficient is
#                      what broke the v0.62.39 release: it narrows the IDENTITY
#                      (certificate) lookup to FILE. The signing KEY is resolved
#                      through the user keychain SEARCH LIST, so FILE must ALSO be
#                      listed there (`security list-keychains -d user -s FILE …`).
#                      Passing --keychain alone fails at the codesign step below —
#                      measured here as errSecInternalComponent, the same condition
#                      the CI runner's macOS 15 reports as errSecItemNotFound. See
#                      the failure help there for the check that tells the two
#                      failure modes apart. The release job does not pass it at all:
#                      it pins the identity with --identity and reaches the key
#                      through the search list.
#   --timestamp MODE   ``secure`` (default) or ``none``. ``none`` is NOT for a
#                      release: it is for a local build on a host whose timestamp
#                      authority cannot be reached. It REFUSES unless the caller
#                      sets LOP_KEYAGENT_ALLOW_UNTIMESTAMPED_BUILD=1 (no release
#                      job sets it), is then warned about loudly, and the
#                      timestamp assertion below is skipped for it.
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
TIMESTAMP_MODE="secure"

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --identity) IDENTITY="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --keychain) KEYCHAIN="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --timestamp) TIMESTAMP_MODE="$2"; shift 2 ;;
    --asan) ASAN=1; shift ;;
    *) echo "assemble_keyagent_bundle.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "$OUT" ] || [ -z "$IDENTITY" ] || [ -z "$PROFILE" ]; then
  echo "assemble_keyagent_bundle.sh: --out, --identity and --profile are required" >&2
  exit 2
fi
[ -f "$PROFILE" ] || { echo "no provisioning profile at $PROFILE" >&2; exit 2; }
case "$TIMESTAMP_MODE" in
  secure) ;;
  none)
    # REFUSED UNLESS A DELIBERATE NON-RELEASE OVERRIDE IS SET (agent review round 3,
    # R3-1). The step-5c warning below is a message, not a mechanism: it scrolls past,
    # and it leaves an untimestamped artefact one copy-pasted argv away from a release
    # job — which is the exact property this whole file exists to make impossible. A
    # warning alone was judged insufficient for that reason: it reports the state
    # rather than preventing it, and "must never reach a release" stayed prose. A named
    # override no release job sets makes reaching `none` a deliberate act, so a release
    # that took this path can be seen to have taken it. The `secure` default and the
    # read-back assertion below are untouched.
    if [ "${LOP_KEYAGENT_ALLOW_UNTIMESTAMPED_BUILD:-}" != "1" ]; then
      echo "assemble_keyagent_bundle.sh: refusing --timestamp=none. A signature with no" >&2
      echo "secure timestamp stops validating when the Developer ID certificate expires," >&2
      echo "for every wheel already installed, so it must never reach a release. It is for" >&2
      echo "a local build on a host whose timestamp authority cannot be reached: set" >&2
      echo "LOP_KEYAGENT_ALLOW_UNTIMESTAMPED_BUILD=1 to build it anyway. Never set that in" >&2
      echo "a release job." >&2
      exit 2
    fi
    ;;
  *)
    echo "assemble_keyagent_bundle.sh: --timestamp takes 'secure' or 'none', not '$TIMESTAMP_MODE'" >&2
    exit 2
    ;;
esac

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
# EXPLICIT, not left to codesign's default for the certificate type — see the header and
# step 5c. macOS ships bash 3.2, where "${arr[@]}" on an EMPTY array trips `set -u`; the
# ${arr[@]+"${arr[@]}"} idiom is the portable spelling (same as run_probe.sh).
if [ "$TIMESTAMP_MODE" = "secure" ]; then
  TS_KW=(--timestamp)
else
  TS_KW=(--timestamp=none)
fi

# WHY THIS FAILURE GETS ITS OWN HELP TEXT. The v0.62.39 release died here with
# `error: The specified item could not be found in the keychain.` and NOTHING in the
# log named the command or what to check, so the same cause cost a second
# investigation. The command is named below — but the text hands the reader a check
# that DISCRIMINATES rather than an asserted cause, because this call's two failure
# modes are not distinguishable from the message:
#
#   errSecInternalComponent   an identity WAS found and its KEY could not be used
#   errSecItemNotFound /      a LOOKUP returned nothing — which is equally what the
#   "no identity found"        IDENTITY lookup says when nothing matches, or when
#                             the certificate chain cannot be evaluated
#
# So the help names the measurement instead:
#
#   security find-identity -p codesigning "$KEYCHAIN"   # is the identity THERE
#   security find-identity -p codesigning               # can the SEARCH LIST reach it
#
# `-v` is deliberately NOT used in either form. It lists only identities whose
# certificate chain VALIDATES, so it is a validity report and not a presence check —
# and validity here depends on what else the search list can see, so neither reading
# of it generalises: on the host measured below it reports 0 in the failing state,
# while the v0.62.39 release's own log reports 1 in that same state (the throwaway
# keychain absent from the list). The non-`-v` form is the presence check, and it is
# what the two commands above ask for.
#
# MEASURED 2026-09-25 on macOS 27, trusted Developer ID identity present only in the
# throwaway keychain (the CI runner's condition — on this host the login keychain
# holds the same identity, which masks the bug), p12 imported, partition list set,
# G2 intermediate imported, same identity SHA throughout:
#
#   keychain NOT in the user search list:
#     find-identity -p codesigning "$KC"  -> 1   (the identity IS in the keychain)
#     find-identity -v -p codesigning "$KC" -> 0 here; 1 in the release's own log
#     find-identity -p codesigning        -> 0   (the search list cannot reach it)
#     codesign ... --keychain "$KC"       -> errSecInternalComponent
#     codesign ... (no --keychain)        -> "<sha>: no identity found"
#   keychain APPENDED to the user search list:  all four report the identity, and
#     both argvs sign and verify (`valid on disk`, `satisfies its Designated
#     Requirement`).
#
# The rig's string is errSecInternalComponent, NOT the release's errSecItemNotFound:
# those are one condition spelled differently by two macOS versions, and what
# establishes the cause is the RUNNER ladder, not this host's string — appending the
# keychain to the search list turned the failing release job green on a macOS 15
# runner while `--keychain` was still being passed (run 36178040698, head 77ed7beca),
# and dropping that flag later changed nothing there (run 36179085672). Set it as the
# DEFAULT keychain instead and it still fails, as it does with the Apple root
# certificate imported to silence the chain warning: both were measured and ruled out
# rather than argued away.
key_unreachable_help() {
  cat >&2 <<'EOM'
assemble_keyagent_bundle.sh: THE STEP THAT FAILED IS THE `codesign` CALL ABOVE.

`codesign` fails here in two different situations and the message does not tell
them apart: `errSecInternalComponent` means an identity WAS found and its key
could not be used, while `errSecItemNotFound` ("The specified item could not be
found in the keychain") or "no identity found" means a LOOKUP returned nothing —
which is also what the IDENTITY lookup says when the certificate chain cannot be
evaluated. Measure rather than guess. In the shell that created the keychain, and
WITHOUT `-v` — it reports only chain-validating identities, a validity report rather
than a presence check (the v0.62.39 log shows it returning a VALID identity in this
very state, so it cannot be relied on in either direction):

  security find-identity -p codesigning "$KEYCHAIN"   # is the identity THERE?
  security find-identity -p codesigning               # can the SEARCH LIST reach it?

The second count says what it says only on a host where NOTHING ELSE holds this
identity. On a machine whose login keychain already has it — the operator's Mac, or
anywhere the bundle has been signed before — that copy is what the second count
reports, so it cannot see the misconfiguration at all. That masking is what hid this
bug locally for as long as it lasted.

* An entry for that identity in the first and none in the second: the keychain is not
  in the user keychain SEARCH LIST. That is what broke the v0.62.39 release — `codesign
  --keychain FILE` narrows only the IDENTITY lookup, while the signing KEY resolved
  through the SEARCH LIST. Add it, and restore the list afterwards:

    security list-keychains -d user -s "$KEYCHAIN" \
      $(security list-keychains -d user | tr -d '"')

* NONE in both: the identity or its chain is not usable where it was imported —
  check `--identity` against the p12, that the import landed, and that the
  Developer ID G2 intermediate is present.

* An identity in BOTH and signing still fails: the key access itself is refused —
  look at the partition list and the key's ACL, not the search list.

Making the keychain the DEFAULT keychain is NOT a substitute for listing it, and
neither is importing the Apple root certificate to silence the chain-to-self-signed
root warning: both were measured to still fail (2026-09-25).
EOM
}

if [ "$ASAN" = "1" ]; then
  codesign --force ${CODE_KW[@]+"${CODE_KW[@]}"} ${TS_KW[@]+"${TS_KW[@]}"} \
    --entitlements "$HERE/keyagent.entitlements" --sign "$IDENTITY" "$APP" || {
      key_unreachable_help
      exit 1
    }
else
  codesign --force --options runtime ${CODE_KW[@]+"${CODE_KW[@]}"} ${TS_KW[@]+"${TS_KW[@]}"} \
    --entitlements "$HERE/keyagent.entitlements" --sign "$IDENTITY" "$APP" || {
      key_unreachable_help
      exit 1
    }
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

# --- 5c. THE TIMESTAMP MUST BE ON THE SIGNATURE, AND IS READ BACK ------------
# WHY AN ASSERTION RATHER THAN AN ASSUMPTION (QA round 2). The entitlement is checked
# against the signature at run time, so a Developer ID signature carrying no secure
# timestamp stops validating when the certificate expires — for every wheel already
# installed (design doc §8, which the rotation plan depends on). QA round 2 reported
# that this host could not obtain a timestamp for a universal2 binary and signed its own
# artefact with `--timestamp=none` through a shim. That did NOT reproduce on 2026-09-25,
# under the same conditions — three fresh universal2 binaries and this very bundle all
# signed with a real `Timestamp=` at 08:43-08:44 — so a universal2 binary is NOT
# untimestampable. The MECHANISM of round 2's failure is NOT established and is not
# asserted here (QA round 3, the timestamp section): three fat-binary failures in a row
# while thin binaries succeeded on the same host is not what a TSA outage looks like, and
# nothing further was measured. Only what is measured is claimed: the failure did not
# reproduce under the same conditions later. This assertion is what makes the class moot
# — it fails closed, so "could not obtain one" can never ship silently whatever the cause.
if [ "$TIMESTAMP_MODE" = "none" ]; then
  echo "  WARNING: signed WITHOUT a secure timestamp (--timestamp=none), under the" >&2
  echo "  LOP_KEYAGENT_ALLOW_UNTIMESTAMPED_BUILD=1 override this run was started with." >&2
  echo "  This signature stops validating when the Developer ID certificate expires, so" >&2
  echo "  this artefact must NOT be released or published. It is for a local build on a" >&2
  echo "  host whose timestamp authority cannot be reached." >&2
else
  SIGNED_TS="$(codesign -d --verbose=4 "$APP" 2>&1 | grep -c '^Timestamp=' || true)"
  if [ "${SIGNED_TS:-0}" -lt 1 ]; then
    echo "the signature on $APP carries NO secure timestamp, and one is required: a" >&2
    echo "Developer ID signature without it stops validating when the certificate" >&2
    echo "expires, for every wheel already installed. This is what an unreachable Apple" >&2
    echo "timestamp authority looks like. Retry when it is reachable — do not release" >&2
    echo "this artefact, and do not switch to --timestamp=none to get past this check." >&2
    exit 1
  fi
  codesign -d --verbose=4 "$APP" 2>&1 | grep '^Timestamp=' | sed 's/^/  secure /'
fi

echo "assembled and signed $APP"
codesign -d --entitlements - --verbose=2 "$APP" 2>&1 | sed 's/^/  /'
