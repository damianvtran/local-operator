# The macOS presence-gated operator key: the signed key agent

**Status:** implemented. **Scope:** the macOS half of the operator key's presence
tier — the one the operator answers with a Touch ID gesture.
**Code:** `packaging/macos/` (the helper, its entitlements, the bundle assembly, the
wheel retag), `local_operator/operator/macos/` (the client), the
`SecureEnclaveBackend` in `local_operator/operator/keychain.py`, and the
`keyagent-macos` job in `.github/workflows/publish.yml`.

This document is the context a future reader needs and the measurements behind the
decisions. Every number in it was taken on this machine unless it says otherwise:
macOS 27.0 (26A428), arm64, uid 501, signed with this project's `Developer ID
Application` identity (team `SHA2U6KT7V`) — named by TEAM ID rather than by the
certificate's common name, because this document is meant to be shareable.

---

## 1. The problem, in one paragraph

A Secure Enclave key can only live in the **data-protection** keychain, and that
keychain admits a caller only when the caller's **code signature** carries the
keychain entitlement *and* an **embedded provisioning profile** authorizes it. A
Python interpreter cannot be signed that way: the runtime is installed by
`uv tool install` / `pip`, on hosts and at paths that never see this repository. So
before this change the tier could not work on **any** Mac, however correct the
CoreFoundation calls were, and `lop operator init` failed with a precise message that
named no remedy that could work.

## 2. The measured shape matrix — why the artefact looks the way it does

One row per calling shape, same binary and same entitlement unless the row says
otherwise (`packaging/macos/lop-keyagent/se-keyagent.c`, signed by
`assemble_keyagent_bundle.sh`):

| # | Shape | Result |
|---|---|---|
| a | ad-hoc signed bare tool, no entitlement | `errSecMissingEntitlement` (-34018) |
| b | Developer ID signed bare tool, no entitlement | -34018 |
| c | Developer ID bare tool + `keychain-access-groups` | **SIGKILL (137)** |
| d | app-like bundle + entitlement + embedded profile | **PASS** |
| e | bare tool + `application-identifier` only | -34018 |
| f | bundle + `application-identifier` + embedded profile | **PASS** |
| g | bundle + both entitlements + embedded profile | **PASS** |
| d' | bundle + `keychain-access-groups` + profile + hardened runtime | **PASS** |
| Q | bundle + `keychain-access-groups`, **no profile** | **SIGKILL (137)** |
| Q2 | bundle + `application-identifier`, **no profile** | -34018 |
| S | bundle + profile, **no hardened runtime** | **PASS** |
| T | bundle + `application-identifier` + profile | **PASS** |

Three decisions fall out of that table, and each is load-bearing:

1. **The bundle is required.** A bare Mach-O carrying the entitlement is refused (b,
   e) or killed (c). Apple's TN3137 says a command-line tool needs an app-like bundle
   to embed a profile; the table is the same statement measured in both directions.
2. **The embedded profile is required, and its absence is not an error.** Without it
   an *entitled* bundle is **SIGKILLed by the kernel** (Q), so the failure has no
   error code and no message: nothing the process can catch, print or exit with. That
   is why the runtime verifies the profile **before exec**
   (`keyagent.verify_bundle`), because a signal is unreportable.
3. **Sign with `com.apple.application-identifier` only.** `keychain-access-groups` is
   unnecessary (f and T pass without it) *and* it is the entitlement that turns a
   missing profile into a SIGKILL (Q) instead of the fully explained -34018 (Q2).
   Fewer claims, and the degraded failure mode is a diagnosable refusal.

Hardened runtime is not required (S). It is kept because notarization needs it and
notarization is the next step on this path; when it lands, nothing else changes.

`kSecUseDataProtectionKeychain` is **not** in the helper: it is not in Apple's
documented Enclave generation dictionary and it is not load-bearing — the passing
shape does not set it, and the Python implementation that used it measured the round
trip identical with and without (recorded in `keychain.py`'s history and pinned there
as an unvalidated no-op before that path was deleted).

## 3. The finding that sizes the whole design: the entitlement gates READING too

A Secure Enclave key created by an entitled process is **invisible to an unsigned
one**, and the answer the unsigned process gets is *not found*, not *refused*:

```
create  (entitled process, test tag)      -> OK, 65-byte P-256 point
query   (this runtime's own interpreter)  -> -25300 errSecItemNotFound
delete  (entitled process, afterwards)    -> 0   (the key was there the whole time)
```

Consequences, and they are the rules the Python side is built around:

* **every verb runs inside the entitled process** — create, load, sign, and "is there
  a key at all";
* **the runtime may never conclude "no operator key" from a query of its own**.
  `SecureEnclaveBackend.load()` returning `None` means the *helper* said -25300;
  a helper that is missing, unverifiable or killed **raises** instead, because
  "broken install" and "no key yet" have different remedies
  (`test_no_runtime_path_queries_the_keychain_for_the_operator_key` asserts that no
  `SecItem*` call remains anywhere in the package, reading the AST rather than the
  text so the docstrings may keep explaining it);
* **the presence sheet is raised inside the helper**, because nothing else can reach
  the key. The sheet cannot carry this project's copy — `SecKeyCreateSignature` takes
  no parameters dictionary and `kSecUseOperationPrompt` was deprecated in macOS 11 —
  so what names the session and the effect is the *caller's* line on stderr
  (`sign.py`). That is why the helper has **no internal timeout**: it blocks for as
  long as the sheet is up, and the caller's bound (`lop operator sign --timeout`,
  default 180 s) is what ends the wait.

## 4. The interface

```
<site-packages>/local_operator/operator/macos/lop-keyagent.app/
    Contents/Info.plist                CFBundleIdentifier = com.local-operator.lopkeyagent
                                       CFBundleExecutable = lop-keyagent, LSUIElement = true
    Contents/MacOS/lop-keyagent        universal2 Mach-O, signed, hardened runtime
    Contents/embedded.provisionprofile the Developer ID profile (load-bearing, §2)
    Contents/_CodeSignature/           the signature
```

Invoked **directly** by `subprocess`, never through LaunchServices (`open`): the
entitlement comes from the bundle's signature and embedded profile, so there is no
window, no Dock icon and no activation for a one-shot CLI call. One process per
operation: no daemon, no socket, no shared state, and any leak is bounded by a process
measured in milliseconds.

The verbs, and what each answers:

| Verb | Prompts | stdout |
|---|---|---|
| `create` | no (creation does not raise user presence) | `{"ok":true,"protocol":1,"spki":"<b64url>","reused":false,"rung":"…"}` |
| `public` | no | `{"ok":true,"protocol":1,"spki":"…"}` |
| `exists` | no | `{"ok":true,"protocol":1,"present":true}` |
| `sign` | **yes, every call** | `{"ok":true,"protocol":1,"signature":"<b64url DER>"}` |
| `doctor` | no (it creates ONE item under `<tag>.<pid>` and deletes it again) | `{"ok":true,…,"rung":"…","keychain":"ok","keychain_status":0,"profile":"ok","generation":"ok","generation_status":0}` |
| `selftest` | no | `{"ok":true,…,"checks":[{"name":…,"expected":…,"actual":…}]}` |
| `purge` | no | `{"ok":true,"protocol":1,"deleted":<OSStatus>}` |

* **`create` is idempotent** (`reused`), which is what makes a second
  `lop operator init` a report rather than a second key — and a second key would
  invalidate every device certificate signed under the first anchor. The **tag is
  consulted BEFORE the key is generated**, because `SecKeyCreateRandomKey` SUCCEEDS
  against a tag that already holds an item (measured, QA round 1): attempting the create
  first meant the duplicate branch was never reached, three calls returned three
  different points with `reused:false`, and `public` kept resolving the tag to the first
  one — so an anchor staged from `create()`'s handle pinned a public half the agent would
  never sign with. `emit_key_reply` is the single place the reply is built, so `create`
  and `public` cannot describe two different keys.
* **`doctor` MEASURES the entitlement instead of inspecting it.** The keychain query is
  answered `errSecItemNotFound` to an unentitled process as well as to an entitled one
  (§3's trap), so it cannot distinguish a working install from one whose entitlement the
  OS will not honour — a bundle of the latter shape reported `keychain":"ok"` and looked
  healthy (QA round 1). Generation is the operation the entitlement gates, so `doctor`
  performs it once under a throwaway tag and deletes it immediately; `ok` requires it.
* **`exists` exists because of §3**: the Python side may not answer "is there a key?"
  itself. A query is a question, so "there is none" is its *success* reply
  (`present:false`); `public` is where -25300 becomes exit 3.
* **`sign` signs exactly the bytes it is handed on stdin.** Not argv: the message is
  arbitrary bytes, and a session id and a single-use challenge have no business in
  `ps`. No framing is added, because domain separation is `verify.signed_message` and
  a second definition of the wire format here would be a defect.
* **`selftest`** is §6's C-side ownership check (see below). **`purge`** deletes the
  item under a **test** tag and refuses the operator's own tag outright: deletion is
  gated by the same entitlement as creation, so without an entitled delete the item a
  test or a QA run creates could never be removed — and "leave nothing behind" is not
  optional. It cannot delete the operator's key, which is why the product still has no
  delete verb.

## 5. Failure semantics

Exit codes: `0` success, `2` cancelled (-128), `3` no key (-25300), `4` refused,
`5` protocol/usage error, killed by signal (`-9`/`137`) for the kernel refusing the
entitlement. Failures carry `site` (`access control` / `key generation` / `signature`
/ `key lookup`) as well as `status`, because `errSecParam` means different things at
different sites and the Python diagnosis table
(`keychain._SECURE_ENCLAVE_DIAGNOSES`) must not guess which one it was.

`init` and `status` then say, in one vocabulary:

| State | `init` | `status` |
|---|---|---|
| helper absent | not installed: an sdist install, or a wheel for another platform — reinstall, or `--backend file-only` and here is what that costs | `key agent : the macOS key agent is not installed (broken install)`, plus `fix :` naming the reinstall; the `reason` line keeps the ANCHOR's own sentence |
| verification failed | its signature does not verify as ours, or it carries no profile | `key agent : the key agent failed verification` + `fix :` |
| killed by signal | the kernel refused its entitlement: the profile is missing, stale or does not authorize its application identifier | `key agent : the key agent was killed: bad or missing embedded profile` + `fix :` |
| keychain call refused | the key agent ran but the OS refused its keychain call, or the helper declined the request itself | `key agent : the key agent refused its keychain call: <the helper's own sentence naming the cause>` + `fix :` |
| no key stored | no operator key on this host: run `lop operator init` | `reason : no key on this host` |
| key stored | the level it achieved, with the protection class `[kSecAttrAccessible…]` | `private-half backend : operator-secure-enclave`, `presence per signature : True` |

**The key agent's fault STANDS BESIDE the authority reason, never in place of it**
(design round 1). The level is `spawn-capability-only` because no anchor is installed, and
the key agent's state does not move the level at all; collapsing the two into one field
read as "my key vanished and my installation is broken" one command after a successful
`init`, and routed the reader to a reinstall that neither installs the anchor nor changes
the level. `status` names a remedy that can work, which is why the `loosening:` block
keeps its `lop operator init` sentence only in the state where that verb is the next step.

**Nothing silently downgrades.** On macOS the key agent is part of the *install*, not
a host capability, so a missing or unverifiable helper makes `auto` and
`--backend secure-enclave` **hard-fail** rather than fall back to a file-backed key
that any process running as you can read. That is the `CngBackend` precedent read in
the other direction: there the *build* cannot do it (`supported()` is False), here the
build can and the install is wrong. `choose_backend("auto")` therefore returns
`SecureEnclaveBackend` on macOS **whether or not** its helper is usable, and its
docstring says so; `--backend file-only` remains the only route to the weaker level.
When the anchor names the presence backend and the helper cannot run, `status` prints a
`key agent` line rather than letting the two fields above promise a gate that cannot be
answered.

**The file-only level line is split by platform, and that is a fix, not a regression**
(design round 2, D1). Round 1 removed "this host has no presence store" because it is
false on macOS, and replaced it with "Run `lop operator init` after reinstalling for a
presence-gated key" — true on macOS and false on the platform that prints the line most
often: Linux has no OS-mediated per-signature presence store and the Windows tier is
unimplemented (§10), so there a reinstall installs the same file-only wheel and
`lop operator init` is the idempotent report that replaces nothing. The reader is sent
around a loop that cannot end, which is the same failure class round 1 rated MAJOR in D2
and D3. `describe_level` therefore names the reinstall route only on darwin, and gives
every other platform the one lever that works from a file-only host — a paired phone
authorises a session from another device. The cost sentence ("ANY process running as you
can sign for you … NOT a boundary") is platform-independent and unchanged.

**The cause is printed beside the kind** (design round 2, R2-1). One kind, `refused`,
covers an entitlement the OS will not honour, a helper that declined the request itself,
and any failed generation. A one-line field that named only the entitlement told an
operator whose keychain was locked — or whose helper had merely refused a flag — that
their entitlement was not in effect, and offered the reinstall that follows that
diagnosis. `SecureEnclaveBackend.health` prints the helper's own sentence as the cause for
that kind, and for that kind only: the other kinds' copies already ARE their specific
cause, and an `unverified` detail can be a tool's stderr tail, which must not be pasted
into a one-line field.

## 6. Ownership, and what replaced the ASan pass

The Python implementation of the native sequence was **deleted**, not kept as a
fallback: it could never work (§2, §3), and two implementations of one native
sequence, one of which no test on any host can exercise, is a second way of doing
things. The sequence now exists once, in C, and the two things that used to crash it
are structurally impossible there:

* the **application tag** is created once, by `main`, and the ladder only borrows it —
  the use-after-free that shipped (a released `CFDataRef` handed to the next
  iteration) cannot be expressed;
* every dictionary is built with the **typed** CoreFoundation callbacks, because NULL
  callbacks retain nothing and a generation dictionary so built SIGSEGVs inside
  `SecKeyCreateRandomKey` (measured, 6 of 6 runs).

The C-side check the design asked for was an **ASan/`leaks` pass in CI. Neither is an
instrument on this macOS**, measured:

* `clang -fsanitize=address` links `libclang_rt.asan_osx_dynamic.dylib`, and dyld
  refuses to load it into a signed process — *"Sanitizer load violates platform
  policy"*, SIGABRT, exit 134. There is no ASan reading to be had.
* `leaks --atExit` runs and prints a report — and says **"0 leaks for 0 total leaked
  bytes"** for a control binary that deliberately leaks 4 KiB. A dead instrument that
  returns a reading is worse than no instrument.

So the invariant is asserted **in the shipped binary**, with `CFGetRetainCount`,
against the very constructors `create` runs (`make_access_control`,
`make_private_attrs`, `make_generation_dict`), which is what makes the check and the
shipped path impossible to diverge without editing both in one place. Two properties
of it are deliberate: the assertions are *invariants* (a dictionary retains what it is
handed, and releasing it returns that reference) rather than absolute counts, and the
release half is asserted too — after the program gives up its own reference, the
count must be exactly the dictionary's, which is how a *missing* release would be
caught. It needs no entitlement, no keychain and no writes, so CI runs it on the
built binary; a mutation (deleting one release) makes it report
`"expected":1,"actual":2` and exit 4, which is recorded in the PR's evidence.

One measurement from that work is worth keeping: a small **integer** `CFNumber` is an
**immortal** tagged value (`CFGetRetainCount` returns `LONG_MAX` for
`kCFNumberSInt32Type 256`, for `kCFNumberLongType 2^40`, and even for
`kCFNumberDoubleType 256.0`), so the `CFNumberRef` the old ladder never released could
not in fact have leaked; a *fractional* double is an ordinary object, and that is what
the selftest uses to exercise the dictionary's ownership of a number.

## 7. The release path

`keyagent-macos` (in `.github/workflows/publish.yml`) is a **pinned** `macos-15` job,
deliberately unlike the `*-latest` runners elsewhere in that file: the signing
toolchain must not move under the artefact the entitlement depends on.

1. the p12 from `secrets.MACOS_SIGNING_P12_BASE64` decoded at `0600` under
   `$RUNNER_TEMP`, imported into a **throwaway keychain** (the operator's login
   keychain is never touched);
2. `security set-key-partition-list` — without it the identity imports and `codesign`
   still cannot use the key;
3. the **Developer ID G2 intermediate** fetched from Apple and pinned by sha256
   (`f16cd3c5…df3a`): a runner without it imports an identity that is not valid;
4. compile universal2 (`-arch arm64 -arch x86_64`), **assert with `lipo -archs` that the
   assembled binary really carries both slices** (the tag is not a binary, and `--binary`
   skips the compile), assemble the bundle, assert the profile authorizes exactly what the
   entitlements claim AND that it is still current and names the signing certificate, sign
   with the identity addressed **by hash** (so the workflow carries nobody's name), and
   `codesign --verify --strict`;
5. run the helper's `selftest`;
6. retag the pure wheel into `local_operator-<v>-py3-none-macosx_11_0_universal2.whl`
   with the signed bundle injected at the path the runtime resolves, its execute bit
   preserved, and `RECORD` rebuilt from the bytes written — then re-assert `lipo -archs`
   on the helper EXTRACTED FROM THE WHEEL, because the wheel is what an Intel Mac
   downloads and a single-arch helper under a universal2 tag installs cleanly and cannot
   execute;
7. upload that wheel as `release-dists-macos`; `pypi-publish` needs **both** jobs, so a
   release that cannot build the key agent fails rather than shipping the pure wheel
   alone;
8. the throwaway keychain and the decoded identity move are deleted in an `if:
   always()` step, and the p12 is never uploaded as an artefact.

**The signature carries a secure timestamp, and the build READS IT BACK** — stated
because the reason is not notarization. The entitlement is checked against the signature
at run time, so a Developer ID signature with no secure timestamp stops validating when
the certificate expires, for every wheel already installed; that is exactly what §8's
rotation plan assumes does not happen. `assemble_keyagent_bundle.sh` therefore names the
mode (`--timestamp`, default `secure`) instead of depending on `codesign`'s default for
this certificate type, and step 5c FAILS the build if the signed result carries no
`Timestamp=` line — an unreachable Apple timestamp authority fails the release rather
than shipping an artefact that stops working in 2031. `--timestamp=none` exists for a
local build on a host that cannot reach the TSA: it REFUSES unless the caller sets
`LOP_KEYAGENT_ALLOW_UNTIMESTAMPED_BUILD=1`, which no release job sets — a warning alone
was judged insufficient (agent review round 3, R3-1), because a message reports the state
instead of preventing it, and "must never reach a release" was prose. Set deliberately,
it then warns, skips the assertion, and is for a local build only. QA round 2 reported
that a universal2 binary could not obtain a timestamp on this host and signed its own
artefact through a shim; that did not reproduce on 2026-09-25 under the same conditions
(three fresh universal2 binaries and this very bundle all carried a real `Timestamp=`),
which refutes the property — a universal2 binary is not untimestampable — but leaves the
MECHANISM unknown, and unknown is what this document records: three fat-binary failures
in a row while thin binaries succeeded on the same host does not fit an outage, and
nothing further was measured. What the release relies on is measured, not inferred: step
5c reads the timestamp back and fails the build closed, so an artefact without one cannot
ship silently whatever the cause.

**The two wheels are OFFERED, never passed as two paths** (QA round 1, Q5).
`uv pip install --find-links <dir> local-operator==<v>` lets the resolver choose, and
the choice is measured: it selects
`local_operator-<v>-py3-none-macosx_11_0_universal2.whl`. Passing both files
explicitly — `uv pip install <pure.whl> <macos.whl>` — is refused outright with
`Requirements contain conflicting URLs for package 'local-operator'`, so nothing in a
document, a script or an operator's habit should imply that form works.

**The profile is a secret, the app id is not.** The entitlement must name a real
application identifier for `codesign` to accept it, and an app id is public — it is in
the signature of every wheel we publish and `codesign -d` prints it from any installed
helper. The *profile* carries the certificate, is bound to this team, and must be
refreshable, so it travels as `secrets.MACOS_SIGNING_PROFILE_BASE64`; committing it
would put one team's identity into every fork's build. A fork substitutes its own app
id, profile and `Info.plist` together — and the runtime pins the identifier and team it
expects, so a wheel signed for another identity is **refused** rather than trusted.

**Attestations.** `pypa/gh-action-pypi-publish` generates and uploads PEP 740
attestations by default for a Trusted Publishing flow (its `action.yml` declares
`attestations` with `default: 'true'`), and the only permission that requires is the
`id-token: write` the job already has. The workflow sets `attestations: true`
explicitly so that turning it off has to be deliberate. Verify a wheel with the
Integrity API —
`curl -sH 'Accept: application/vnd.pypi.integrity.v1+json'
https://pypi.org/integrity/local-operator/<version>/<wheel-filename>/provenance` — or
by reading the Provenance section of the file's page on PyPI. Neither pip nor uv
verifies attestations at install time today (uv's PEP 740 verification is an open
request, astral-sh/uv#9122), so this is audit evidence rather than an install-time
guarantee.

**The bundle is not `package-data`.** It enters a wheel only through
`packaging/macos/make_macos_wheel.py`, which also retags the wheel and sets
`Root-Is-Purelib: false`. Declaring it in `pyproject.toml` would let a plain
`python -m build` produce a wheel that contains a Mach-O binary while still claiming
`py3-none-any` — a wheel every platform would install, carrying code only one can run.

## 8. Renewal and rotation

* **The Developer ID certificate expires after 5 years.** Rotate before it lapses and
  keep the old one importable until wheels signed by it have aged out of use: the
  entitlement is checked against the signature at run time, so a wheel signed by an
  expired certificate stops working on the day the certificate expires — for users who
  already installed it.
* **The profile expires 2044-09-20, but a team admin can revoke the App ID or the
  profile at any time**, and a revoked profile turns `create` into a **SIGKILL** (§2,
  Q). That is why the signal case has its own diagnosis and its own copy: the failure
  tells the operator their key agent was killed for a bad or missing profile, not that
  the key is absent.
* Rotating means: new profile in `MACOS_SIGNING_PROFILE_BASE64`, updated app id in
  `packaging/macos/keyagent.entitlements` and `Info.plist` if they change,
  `TEAM_IDENTIFIER`/`BUNDLE_IDENTIFIER` in `local_operator/operator/macos/keyagent.py`,
  and a release. The runtime deliberately **refuses** a helper it does not recognize,
  so a half-rotated state fails loudly at the pre-flight instead of silently signing
  with something unexpected.

## 9. What is deliberately not proven here

* **The presence sheet itself.** Nothing in this work raised one: key *creation* does
  not require user presence, and a signing call would have put a Touch ID sheet on the
  operator's screen while they were working. So "the sheet appears for a subprocess of
  an unsigned runtime, it can be cancelled, and a cancel maps to -128" is designed,
  unit-tested through a fake helper, and **not** measured. The test that settles it,
  once, with the operator at the keyboard: `lop operator init`, then
  `lop operator sign --challenge <hex> --purpose loosen --session <id>` — read the
  sheet, cancel once, confirm the runtime reports *cancelled* and signs nothing.
* Whether the sheet dismisses itself when the helper is killed while it is up (the
  180 s bound), and what the OS does with a prompt whose requesting process has exited.
* **Notarization.** Deliberately not in this landing: files written by pip/uv carry no
  `com.apple.quarantine` attribute, so Gatekeeper does not evaluate them, and
  notarization would add a third credential and another failure mode to the release
  path for no functional gain. The signing semantics are kept such that adding it
  changes nothing else (`--options runtime` is already there, which is what
  notarization requires).
* **The `keyagent-macos` job has never executed.** It first runs on the next release, so
  §7's steps 1-8 — the identity import, the pinned G2 intermediate, the timestamp the
  check reads back, the wheel retag and the `lipo` re-read on the extracted helper — are
  designed and locally measured but unmeasured in CI. Stated so the release owner treats
  that job's first run as the test of it.
* An **Intel/T2 Mac**: that the universal2 slice runs, that the entitlement passes
  there, and that the ladder picks the passcode-set rung.
* That `uv tool install` leaves `Contents/embedded.provisionprofile` in place — it is a
  plain file copy out of the wheel, and it was verified for a `uv pip install` of the
  macOS wheel into a clean venv (the round trip is in the PR's evidence) but not for
  the tool-install layout specifically.
* Whether another Apple-platform-signed process can see this key at all. Measured only
  that *our* unsigned runtime cannot (-25300, §3); the key is in this app's access
  group, so it should not be visible elsewhere — unmeasured.

## 10. Windows and Linux, for contrast

The tiers are asymmetric and the reason is worth keeping in view. On **Windows** the
presence gate is `NCRYPT_UI_POLICY` set on a persistent key at generation time and
enforced **in-process** by the KSP: no entitlement, no profile, no Authenticode
signature, and therefore no signed helper to build. What is missing there is the
implementation (`CngBackend.supported()` returns False and `create()` raises), and it
is the cheap tier because GitHub Actions has Windows runners, so it can be *tested*
rather than written blind. On **Linux** there is no OS-mediated per-signature presence
store: `fprintd` proves a finger touched a reader rather than that a human consented to
*this* signature, `polkit` authenticates to the system with a password that can be
prompt-spammed, and a TPM2 PIN is a secret rather than a gesture. `file-only` stays the
honest Linux level, and the report keeps saying so — with the level line's remedy split by
platform (§5, design round 2 D1), because "reinstall for a presence-gated key" names a
store neither of these platforms can reach.
