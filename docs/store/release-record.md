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
and promote this heading when it lands. Calling that API **by hand** needs a
temporary IAM grant that must be revoked afterwards — read the warning under
"Release-automation verification" before you do, or just dispatch the workflow,
which needs no grant at all.

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
