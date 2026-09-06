# Browser extension release record

Section 11 of `submission-checklist.md` requires that every Chrome Web Store
release be recorded with the facts needed to reproduce and audit it: the exact
artifact, the commit it was built from, and how it reached the store. This file
is that record.

**Add a new section for every release, newest first.** Never edit a shipped
entry except to append its approval timestamp or a post-release incident note —
the value of this file is that it says what was actually shipped, not what we
intended to ship.

## How to audit a release from these fields

The store package is not committed (`extension/local-operator-extension.zip` is
in `.gitignore`), so the artifact cannot be recovered from the repository. It is
rebuilt by checking out the recorded source commit and running
`pnpm --dir extension build:zip`. Recording only the version would not be
enough — `main` moves, and a version number does not pin a tree.

**The archive SHA-256 identifies the uploaded file; it is NOT reproducible.**
`build.mjs` shells out to `zip`, which stamps every entry with its current
filesystem mtime, so two builds of one tree seconds apart produce different
archive hashes. Comparing a rebuild's hash against the recorded one will always
mismatch, on a perfectly clean release. Do not read that as tampering.

**Before rebuilding anything, copy the retained artifact aside.**
`pnpm build:zip` writes to `extension/local-operator-extension.zip` and deletes
any file already there, which is the same path the uploaded artifact sits at.
That file is gitignored and the store will not give it back, so a rebuild run
without this step destroys the only evidence the audit depends on:

```console
$ cp extension/local-operator-extension.zip /tmp/uploaded-<version>.zip
```

Audit a release this way, in order of strength:

1. **`extension/` tree hash** — `git rev-parse <commit>:extension`. This is the
   field that deterministically pins the build input, and it is the one to
   trust when asking "what source produced this release?" It requires no
   rebuild and cannot destroy anything.
2. **Extracted contents** — with the copy safely aside, rebuild, then `unzip`
   both archives to separate directories and `diff -r` them. Byte-identical
   contents with differing archive hashes is the expected result, because only
   zip metadata differs.
3. **Archive SHA-256** — use it only to confirm the *copy you set aside* is the
   file that was uploaded, never against a rebuild.

Making `build:zip` deterministic would collapse these three into one hash
comparison; that is tracked as a follow-up (see the note under v0.1.5).

---

## v0.1.8 — submitted 2026-09-06, pending review as of 2026-09-06

| Field | Value |
| --- | --- |
| Extension version | 0.1.8 (submitted; **not yet the approved version**) |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `ff9b16b5` (`feat(extension): scoped Allow (domain/site/once) and dangerous allow-all setting (0.1.8)`, PR #672, squash-merged to `main`) |
| `extension/` tree hash | `0605d9a5c34a2681c49665a8b9e8aaaada89c50b` (the deterministic input pin — see the audit note above) |
| Artifact SHA-256 | *not recoverable — see note below* |
| Artifact size | 12 files, no source maps (byte size not recorded — see note below) |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Automated** — `chrome-web-store.yml`, [run 34000911184](https://github.com/damianvtran/local-operator/actions/runs/34000911184), dispatched with `ref=main` `version=0.1.8` |
| Store state | `PENDING_REVIEW`, 100% deployment |
| State last checked | 2026-09-06 |
| Approval timestamp | *pending — append when review completes* |
| Previously published | v0.1.7, live at 100% during review (promoted 2026-09-04, run 33926643637) |

**Second release to go out through the automated path**; v0.1.7 was the first
(staged run 33815585846, promoted run 33926643637). Every release *before v0.1.7*
was a manual dashboard upload. The workflow validated and submitted in one run:
`validated Chrome Web Store package v0.1.8 (12 files, no source maps)`, then
`submitted ... v0.1.8 with STAGED_PUBLISH (PENDING_REVIEW)`.

**Why there is no artifact hash for this entry.** The automated path builds the
zip on an ephemeral GitHub runner, uploads it straight to the store, and retains
nothing — the run publishes no build artifact and logs no digest, so no local
copy of the uploaded file exists to hash. Do **not** fill this row in by running
`pnpm --dir extension build:zip` here: as the audit note above explains, `zip`
stamps entries with current mtimes, so a rebuild's hash would be a *different*
number that never identified the uploaded file, and the rebuild would also
overwrite `extension/local-operator-extension.zip`. Audit this release by its
`extension/` tree hash (step 1 above), which pins the build input exactly and
needs no artifact. Recording the digest in the workflow output is the durable
fix, and it is a natural companion to the deterministic-`build:zip` follow-up
noted under v0.1.5.

**Permissions: byte-identical to the v0.1.7 base.** QA verified that the only
diff in the built manifest is the version string and re-indentation — no
permission added, removed, or changed. That is why the automated path applied:
the standing rule in `submission-checklist.md` sends any permission-adding
package to a human in the dashboard, because the Chrome Web Store API cannot set
permission justifications.

---

## v0.1.7 — submitted 2026-09-03, published 2026-09-04

| Field | Value |
| --- | --- |
| Extension version | 0.1.7 (**the live published version** as of 2026-09-06) |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `63d1f175` (`fix(release): make store environment verification reachable from the workflow token`, PR #589) |
| `extension/` tree hash | `c8b694bb2a06213cff4786fe6f844538986afbaa` (the deterministic input pin — see the audit note above) |
| Artifact SHA-256 | *not recoverable — see note below* |
| Artifact size | 12 files, no source maps (byte size not recorded — see note below) |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Automated** — `chrome-web-store.yml`, [run 33815585846](https://github.com/damianvtran/local-operator/actions/runs/33815585846), dispatched with `ref=main` `version=0.1.7` |
| Promotion route | **Automated** — `chrome-web-store-promote.yml`, [run 33926643637](https://github.com/damianvtran/local-operator/actions/runs/33926643637) |
| Store state | `PUBLISHED`, 100% deployment |
| State last checked | 2026-09-06 |
| Approval timestamp | *exact time not available — bounded below* |
| Previously published | v0.1.0 (v0.1.5 was submitted but superseded before it went live) |

**Recorded retrospectively on 2026-09-06**, while adding the v0.1.8 entry. This
entry was missed at release time, which is the gap it exists to close: the file
had gone straight from v0.1.8 to v0.1.5 while v0.1.7 was the version actually
live. Every field above is derived from the workflow logs and the git object DB
rather than from memory; the fields those sources cannot establish say so
instead of carrying a plausible number.

**This was the first release to go out through the automated path**, both
halves of it — submitted by `chrome-web-store.yml` and published by
`chrome-web-store-promote.yml`, with no dashboard step. The logs read:

```
EXPECTED_VERSION: 0.1.7
validated Chrome Web Store package v0.1.7 (12 files, no source maps)
submitted Chrome Web Store extension omibaecbjdhgbbcedbnnnmjpmopfheof v0.1.7 with STAGED_PUBLISH (PENDING_REVIEW)
promoted Chrome Web Store extension omibaecbjdhgbbcedbnnnmjpmopfheof v0.1.7 to PUBLISHED
```

**Why the source commit is the *staged* run's `headSha`, not the promote run's.**
The promote run (33926643637) reports `headSha` `5cbea141` — a later `main`
commit that merely still carried 0.1.7 in the manifest. Promotion publishes the
already-uploaded revision and builds nothing, so its checkout is not the build
input. The commit that produced the artifact is the staged run's `headSha`,
`63d1f175`, whose `extension/manifest.json` and `package.json` both read `0.1.7`
(verified with `git show`). Use the tree hash above when auditing.

**Why there is no artifact hash for this entry.** Same reason as v0.1.8: the
automated path builds on an ephemeral runner and retains nothing. Confirmed for
this release specifically — the artifacts API reports `total_count = 0` for both
run 33815585846 and run 33926643637, and neither log contains a digest. Audit by
the `extension/` tree hash, and do not rebuild `build:zip` to manufacture a hash
(see the audit note above for why a rebuild's hash is a different number).

**Approval timestamp: exact time unknown, bounded to a ~24-hour window.** The
Chrome Web Store does not report an approval time in either workflow log, and no
`fetchStatus` call was made at the time. What the run history does establish:

- Submitted 2026-09-03T22:59:35Z (run 33815585846).
- A promote attempted 2026-09-03T23:01:50Z, ~2 minutes later, **failed** with
  `Chrome Web Store publish failed: only an approved STAGED revision can be
  promoted` (run 33815763092) — so the revision was still unapproved then.
- The promote succeeded 2026-09-04T22:42:40Z, so it was approved by then.

Approval therefore landed between 2026-09-03T23:01:50Z and 2026-09-04T22:42:40Z.
That failed promote is worth knowing operationally: the promote workflow is not
idempotent against an unapproved revision and fails closed rather than waiting.

**Permissions: unchanged from the v0.1.5 base.** `git diff 37289774 63d1f175
-- extension/manifest.json` shows only the version string; the permission array
(`debugger`, `tabs`, `tabGroups`, `scripting`, `storage`, `alarms`,
`webNavigation`, `notifications`) and host `<all_urls>` are identical. That is
why this one could take the automated path — no justification field to fill in
by hand.

**What shipped in it** (`37289774..63d1f175`, extension-affecting commits):
session-named tab groups (#555), a browser bridge that stays usable when the
heartbeat writer dies (#563), and two release-workflow fixes (#585, #589).
Note **v0.1.6 never shipped** — #555 bumped the manifest to 0.1.6 and #563
superseded it with 0.1.7 before any submission, so no 0.1.6 entry is owed.

---

## v0.1.5 — submitted 2026-09-02, pending review as of 2026-09-02

| Field | Value |
| --- | --- |
| Extension version | 0.1.5 (submitted; **not yet the approved version**) |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `37289774` (`chore(release): bump version to 0.44.38`) |
| `extension/` tree hash | `9fe12bde271cc84f060922d8d254d22d046b7e6a` (the deterministic input pin — see the audit note above) |
| Artifact SHA-256 | `724f3f91117f166263c462801a9a73c27d0a4787bd4e5c5dcb6374a2d56d2d0a` (uploaded file only; not reproducible) |
| Artifact size | 45,262 bytes, 12 files, no source maps |
| Bridge protocol version | `PROTO_VERSION = 1` |
| Submission route | **Manual dashboard upload** |
| Store state | `PENDING_REVIEW`, 100% deployment |
| State last checked | 2026-09-02 |
| Approval timestamp | *pending — append when review completes* |
| Previously published | v0.1.0, live at 100% during review |

**Refresh the state rows** with a `fetchStatus` call — the same read-only API
`extension/scripts/chrome-web-store.sh` uses — rather than trusting the values
above. "Pending review" is a snapshot from the submission date; Chrome review
usually resolves within days, so if that date is well in the past, assume the
row is stale and re-check before relying on it. Append the approval timestamp
and promote this heading when it lands. Note there is no status-only workflow to
dispatch — both store workflows write (`stage` uploads and submits, `promote`
publishes), so a read-only status check has to be the hand-rolled API call, and
that needs a temporary IAM grant which must be revoked afterwards. Read the
warning under "Release-automation verification" before making it. The dashboard
shows the same state with no grant at all, which is the cheaper check when you
only need to eyeball it.

**Follow-up: make `build:zip` deterministic.** `build.mjs` invokes `zip` without
normalising timestamps, which is why the archive hash above cannot be
regenerated. Normalising mtimes and passing `-X` (verified locally to produce
identical hashes across repeated builds) would make a rebuild hash-comparable
and let this record drop the three-step audit procedure for a single check.

**Why this one was uploaded by hand.** 0.1.5 adds the `tabGroups` permission,
which 0.1.0 did not request. Permission justifications exist only in the
dashboard — the Chrome Web Store API v2 cannot set them, and Chrome forbids
scripting the extensions gallery, so no automation can fill that field. The
justification pasted was the `tabGroups` entry in `permissions.md`, verbatim.

**Permissions declared** (matches the built `dist/manifest.json`, not the
source list): `debugger`, `tabs`, `tabGroups`, `scripting`, `storage`, `alarms`,
`webNavigation`, `notifications`, plus host `<all_urls>`. The only delta from
the published v0.1.0 is `tabGroups`.

**Pre-upload validation.** Built from a clean `extension/` working tree at the
recorded commit: `pnpm typecheck` clean, 67/67 tests passing, and
`scripts/validate-store-zip.sh` confirming the archive matches the reviewed
`store-package-files.txt` allowlist, that manifest and package versions agree,
and that no source maps are present.

**User-visible changes since the published v0.1.0** (`5cbb91e1..37289774`):
multi-tab surfaces so parallel sessions each own a tab, session-based tab
grouping (what `tabGroups` is for), site approval as a first-class agent-legible
flow with queued concurrent approvals and explicit loopback all-port grants, an
MV3 reconnect alarm so a suspended worker wakes, owned tabs closing before the
final response, and fixes to snapshot ref resolution, AX wrapper traversal,
hidden-tab scrolling, and popup pairing feedback.

### Release-automation verification performed on this date

The automated path was audited against the live GitHub and GCP APIs so the next
release can use it. All checks passed.

- **Both protected environments** (`chrome-web-store`,
  `chrome-web-store-production`): one required reviewer, custom deployment
  branch policy allowing exactly `main`, and all four release variables defined
  at environment scope with identical values. Verified by running the real
  `extension/scripts/verify-release-environment.sh` against the live API — the
  same script the workflows run before authenticating.
- **`CWS_EXTENSION_ID`** matches the permanent ID hardcoded in
  `chrome-web-store.sh`; no repository- or organization-scoped copies of any
  release variable exist, which the script would reject.
- **WIF provider** `local-operator-main` in pool `github-releases`
  (project `pivotal-tower-456213-u5`, number `778402241192`): ACTIVE, issuer
  `https://token.actions.githubusercontent.com`, the documented three-entry
  attribute mapping, and attribute condition
  `assertion.repository_id == "922327641" && assertion.ref == "refs/heads/main"`.
  The numeric repository ID was confirmed to be this repository's real ID.
- **Service account** `cws-publisher@pivotal-tower-456213-u5.iam.gserviceaccount.com`
  exists and is enabled, carries exactly one binding —
  `roles/iam.workloadIdentityUser` to the WIF principalSet scoped by that
  repository ID — and holds no service-account key. `chromewebstore`,
  `iamcredentials`, and `sts` APIs are enabled on the project.
- **End-to-end authorization proven.** The service account was added under
  Developer Dashboard → Account, and a read-only `fetchStatus` call made as that
  service account returned HTTP 200 with the correct item ID. This is the one
  link that cannot be checked from configuration alone, because it lives only in
  the dashboard. **Reproducing this call requires a temporary
  `roles/iam.serviceAccountTokenCreator` grant that MUST be revoked immediately
  afterwards — see the warning below before running it.**

**Warning — read before calling `fetchStatus` by hand.** A human cannot mint a
token for the publishing service account by design: the only binding on it is
`roles/iam.workloadIdentityUser` for the GitHub principalSet, so nobody at a
keyboard can produce a store credential. Testing the call therefore requires
temporarily granting yourself `roles/iam.serviceAccountTokenCreator`. When this
audit did that, the grant was removed immediately and the resulting IAM policy
was diffed against a pre-test capture to confirm it was byte-identical.

**Never leave that role in place.** A standing human token-creator binding
defeats the entire reason this release path uses workload identity federation
instead of a stored key: it recreates the durable human-usable credential that
WIF exists to eliminate. Capture the policy before granting, revoke straight
after, and diff to prove the revert — do not rely on remembering.

---

## v0.1.0 — first public release

| Field | Value |
| --- | --- |
| Extension version | 0.1.0 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `5cbb91e1` (`feat: Local Operator browser extension and browser bridge`) |
| `extension/` tree hash | `2d6fa2b3d0665a241071fa8e74e91184530d5be3` (derived from the source commit while backfilling this entry) |
| Bridge protocol version | `PROTO_VERSION = 1` (at `5cbb91e1`) |
| Artifact SHA-256 | *never recorded — see note below* |
| Artifact size | ~31 KB (the only surviving fingerprint, from the original checklist) |
| Submission route | Manual dashboard upload (first publication) |
| Store state | `PUBLISHED`, 100% deployment |
| State last checked | 2026-09-02 |

First publication, gated on Radient, Inc. business verification and the EEA
trader declaration rather than on anything in the package. Declared the original
seven permissions plus `<all_urls>` — `tabGroups` did not yet exist.

Artifact SHA-256 was not recorded at the time; this file was created during the
0.1.5 release. The approximate size above is the only surviving fingerprint. The
source commit still pins the tree, so the contents can be rebuilt and inspected,
but nothing ties them to the specific file that was uploaded. This gap is
exactly what the fields above exist to prevent — note that even a recorded
SHA-256 would only have identified the uploaded artifact, not enabled a
hash-comparable rebuild, until `build:zip` is made deterministic.
