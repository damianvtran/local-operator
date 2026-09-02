# Browser extension release record

Section 11 of `submission-checklist.md` requires that every Chrome Web Store
release be recorded with the facts needed to reproduce and audit it: the exact
artifact, the commit it was built from, and how it reached the store. This file
is that record.

**Add a new section for every release, newest first.** Never edit a shipped
entry except to append its approval timestamp or a post-release incident note —
the value of this file is that it says what was actually shipped, not what we
intended to ship.

## Why a release can only be reproduced from these fields

The store package is not committed (`extension/local-operator-extension.zip` is
in `.gitignore`), so the artifact cannot be recovered from the repository. It is
reproduced by checking out the recorded source commit and running
`pnpm --dir extension build:zip`; the recorded SHA-256 is what proves the
rebuild matches what was uploaded. Recording only the version would not be
enough — `main` moves, and a version number does not pin a tree.

---

## v0.1.5 — submitted 2026-09-02, pending review

| Field | Value |
| --- | --- |
| Extension version | 0.1.5 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Source commit | `37289774` (`chore(release): bump version to 0.44.38`) |
| `extension/` tree hash | `9fe12bde271cc84f060922d8d254d22d046b7e6a` |
| Artifact SHA-256 | `724f3f91117f166263c462801a9a73c27d0a4787bd4e5c5dcb6374a2d56d2d0a` |
| Artifact size | 45,262 bytes, 12 files, no source maps |
| Bridge protocol version | `PROTO_VERSION = 1` |
| Submission route | **Manual dashboard upload** |
| Store state at submission | `PENDING_REVIEW`, 100% deployment |
| Previously published | v0.1.0, still live at 100% during review |

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
  the dashboard.

The `fetchStatus` test required temporarily granting a human
`roles/iam.serviceAccountTokenCreator` on the service account, since the
principalSet binding deliberately lets only GitHub Actions mint a token. The
grant was removed immediately and the resulting IAM policy was diffed against
the pre-test policy to confirm it was byte-identical. **Do not leave that role
in place**: a standing human token-creator binding would defeat the reason this
release path uses workload identity federation instead of a stored key.

---

## v0.1.0 — first public release

| Field | Value |
| --- | --- |
| Extension version | 0.1.0 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Source commit | `5cbb91e1` (`feat: Local Operator browser extension and browser bridge`) |
| Submission route | Manual dashboard upload (first publication) |
| Store state | `PUBLISHED`, 100% deployment |

First publication, gated on Radient, Inc. business verification and the EEA
trader declaration rather than on anything in the package. Declared the original
seven permissions plus `<all_urls>` — `tabGroups` did not yet exist.

Artifact SHA-256 was not recorded at the time; this file was created during the
0.1.5 release. The artifact is reproducible from the source commit, but a
rebuild today is not byte-verifiable against what was uploaded. This gap is
exactly what the fields above exist to prevent.
