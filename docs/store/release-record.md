# Browser extension release record

Every Chrome Web Store release is recorded here with the facts needed to
reproduce and audit it: the exact artifact, the commit it was built from, and
how it reached the store. This file is that record, and **this header is the
standing requirement's source — not `submission-checklist.md` §11.** §11 is the
*post-approval* section: all it asks of this file is the approval half of a
table (approved version, approval timestamp, item ID, listing URL, artifact
SHA-256, source commit, protocol version), while the rule that an *unapproved*
submission is recorded here with its route is this header's — which is why the
0.1.20 submission entry below has no §11 row to point at. (Correction owed to
PR #1352's review, which caught §11 cited for the whole rule.)

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

## v0.1.20 — submitted 2026-09-23, in review as of 2026-09-23

This is the submission of the tree the entry below pins. That entry was written
on 2026-09-22, before anything had been dispatched, and it stands as the record
of the renumber; the rows here are the submission's own state.

| Field | Value |
| --- | --- |
| Extension version | 0.1.20 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` (the same item; a revision of it) |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `dad3b92f` — the merge commit of PR #1435, the renumber that **set** this version (`git rev-parse dad3b92ff6ef0e166f5e0caed0ffa8dcd3539402:extension` returns the tree hash below). The **build** ran from `1392324b` instead — this workflow dispatches `ref=main`, and `main` was `1392324b…` at 2026-09-23T13:54:40Z — and the pair is not a contradiction: `git log --oneline dad3b92f..1392324b -- extension/` is empty, so both refs resolve `extension/` to the same tree, and the tree hash is the field that pins the input. Both commits are named so an auditor does not have to work out which one a given sentence means |
| `extension/` tree hash | `c1b5780a214cb795a19bfe5b0c160ebc904e2d15` |
| Artifact SHA-256 | *not recoverable — the same automated-path limitation as v0.1.18, v0.1.17 and earlier. The upload is the half that succeeded and the `STAGED_PUBLISH` call is the half that was refused, so a revision was uploaded and no copy of the file is retained* |
| Artifact size | 13 files, no source maps (`validated Chrome Web Store package v0.1.20 (13 files, no source maps)`) |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Refused by the API, then submitted from the Dashboard** — `chrome-web-store.yml`, [run 35870383291](https://github.com/damianvtran/local-operator/actions/runs/35870383291), dispatched by `damianvtran` on `main` at 2026-09-23T13:54:40Z with `version=0.1.20` (`headSha=1392324b…`). Its build/validate step reports **296 tests, 0 fail** and `validated Chrome Web Store package v0.1.20 (13 files, no source maps)`; the submit call was then answered **HTTP 400**, `status: INVALID_ARGUMENT`, `reason: INVALID_ITEM_METADATA`, *"Your submission does not meet the requirements to be published in the store. Check the Developer Dashboard for steps to resolve."* |
| Promotion route | **Pending** — dispatch `chrome-web-store-promote.yml -f version=0.1.20` once the store reports the revision `STAGED` |
| Store state | `PENDING_REVIEW` for this revision, with **0.1.18 still the live version** — the store's own fields, read by promote run [35874866430](https://github.com/damianvtran/local-operator/actions/runs/35874866430) at 2026-09-23T14:33:16Z: `submitted state=PENDING_REVIEW distributionChannels=[crxVersion=0.1.20 deployPercentage=100]; published state=PUBLISHED distributionChannels=[crxVersion=0.1.18 deployPercentage=100]` |
| State last checked | 2026-09-23T14:33:16Z — the timestamp of that run's own output line, i.e. the run's own reading |
| Approval timestamp | ***Not applicable — the revision is in review.*** There is no approval instant to record and none is invented; see the note below on what that promote run does and does not evidence |
| Previously published | v0.1.18, `PUBLISHED` at 100% — see that entry, whose publication this file records with the successful promote run behind it |

**The refusal was a new class, and the uploaded revision survived it.** The store
refusals this file records before this one are all queue-state answers —
`FAILED_PRECONDITION` / `NOT_UPDATEABLE`, *"You may not edit or publish an item
that is in review."* — which say an item is already queued and nothing about the
package. This one is `INVALID_ARGUMENT` / `INVALID_ITEM_METADATA`: the package
built, validated and uploaded, and the *submission* was refused on the item's own
metadata. Per the operator, the outstanding field was the `downloads` permission
justification — the manifest change most likely to be it, and the one completed
before the Dashboard submission — needed because this tree's manifest gained
`"optional_permissions": ["downloads"]` (`git show c1b5780a…:manifest.json`; the
v0.1.18 tree `3341f432…` declares no `optional_permissions` at all). The
operator completed it in the Developer Dashboard and submitted from there, and
the store then reported the revision `PENDING_REVIEW` at `crxVersion=0.1.20` —
the reading quoted in the table, taken by the next promote dispatch. **The
Dashboard step is the operator's account of the route, not a run's record**; what
a run holds is the state that followed it, which is why that state is quoted raw
and dated.

**What the promote run does and does not evidence.** Run 35874866430 was
dispatched for `0.1.20` and refused at its own gate — *"only an approved STAGED
revision can be promoted"* — printing the store's fields on the way out. That is
what makes it the useful read: the refusal is the probe, and the `submitted` half
is the evidence that 0.1.20 is the queued revision. It is **not** evidence of any
approval, and no `STAGED` reading for 0.1.20 exists yet, so the approval row
stays not-applicable until a promote succeeds.

**0.1.19 never reached the store.** The entry below carries the three refused
dispatches and the eight trees that version came to name; 0.1.20 is the repair
for that ambiguity, and it is the number the store now holds.

**Post-submission addendum (appended 2026-09-24): approved and published,
without a promote run behind it.** The rows above are left as written, because
this file corrects a shipped entry by appending; the approval is recorded here.
**v0.1.20 is live and the queue is empty**, and both halves of that statement are
the store's own fields rather than a workflow's exit code:

- **The queue is empty.** Promote run
  [36008806075](https://github.com/damianvtran/local-operator/actions/runs/36008806075),
  dispatched by `damianvtran` on `main` at 2026-09-24T13:53:24Z
  (`headSha=bc1e247f…`, `EXPECTED_VERSION=0.1.20`), was refused at its own gate at
  2026-09-24T13:56:45Z and printed its store read on the way out: `Chrome Web Store publish failed: only an approved STAGED revision can be promoted (store said: submitted <absent>; published state=PUBLISHED distributionChannels=[crxVersion=0.1.20 deployPercentage=100])`. `submitted
  <absent>` is **not** "still in review" — a revision in review reads
  `PENDING_REVIEW`, which is what this run's own predecessor printed a day
  earlier — it is the store saying there is nothing pending, so there was nothing
  left to promote. As with that predecessor, the refusal is the probe and the
  fields it carried on the way out are the answer.
- **The public listing agrees, and it is the independent half** (checked
  2026-09-24): **"Version 0.1.20"**, **"Updated September 24, 2026"**,
  **132KiB**, and the manifest the page serves reads `"version": "0.1.20"`. The
  0.1.18 entry's own check on 2026-09-23 read 0.1.18, so what moved on the public
  page is this publication. Both version sites on `main`
  (`extension/manifest.json`, `extension/package.json`) read `0.1.20` as well,
  which is what an empty queue means for the next bump rather than a claim about
  the store.

**The approval instant is bounded, not known, and none is invented.** The last
evidence that this revision was still unpublished is the 2026-09-23 probe's own
read — `submitted state=PENDING_REVIEW distributionChannels=[crxVersion=0.1.20 deployPercentage=100]`, printed by run
[35874866430](https://github.com/damianvtran/local-operator/actions/runs/35874866430)
at 2026-09-23T14:33:16Z — and the first evidence that it was live is the
2026-09-24 run above at 13:56:45Z, with the listing read of the same day. The
approval instant therefore lies **between 2026-09-23T14:33:16Z and
2026-09-24T13:56:45Z**, and no run we hold narrows it further, which is why the
`Approval timestamp` row above is left as written rather than rewritten. Four
rows above are consequently stale — `Promotion route` (which still instructs a
developer to dispatch a promote once the store reports the revision `STAGED`, a
state this revision has left), `Store state`, `State last checked` and
`Approval timestamp` — and the heading's "in review as of 2026-09-23" is
superseded by this addendum; the rows and the heading are left standing, as this
file leaves them. The listing carries the only
fine-grained reading inside that window — a timestamp field beside the version and
size in the page's own payload, `1790254705` seconds and `513623000` nanos, i.e.
2026-09-24T12:58:25.513623Z, the value behind the rendered "Updated September 24,
2026" — and it is recorded as what it is, a listing field consistent with a
publication that morning, **not** as the approval instant, which no store read
exposes.

**The route, precisely, because it is the unusual half.** The submission did not
travel this repository's API path: `chrome-web-store.yml` run
[35870383291](https://github.com/damianvtran/local-operator/actions/runs/35870383291)
uploaded and validated `v0.1.20` and was then refused `INVALID_ARGUMENT` /
`INVALID_ITEM_METADATA`, so the `publish` call that carries `STAGED_PUBLISH`
never created a submission for this revision (see `Submission route` above). The
operator completed the outstanding `downloads` justification in the Developer
Dashboard and submitted from there, so this revision's publish semantics are the
Dashboard's, not our script's — and not this record's to assert. What is measured
is the consequence: **no `chrome-web-store-promote.yml` run published this
version.** The last successful promote anywhere is still 0.1.18's (run
[35736288300](https://github.com/damianvtran/local-operator/actions/runs/35736288300),
2026-09-22T13:51:32Z), and both dispatches since — this entry's, on 2026-09-23 and
2026-09-24 — failed. That must not be folded into an
approval-publishes-by-itself reading: Google's publish reference defines
`STAGED_PUBLISH` as "After approval the submission will be staged and can then be
published by the developer", against `DEFAULT_PUBLISH`, "The submission will be
published immediately after being approved", and this revision never went in
through either call. So the publication was a developer-initiated action — the
Dashboard's publish path, or an out-of-band API call — that this checkout holds
no record of: **the fourth occurrence of the gap this file records under v0.1.12,
v0.1.15 and v0.1.17.** Unlike those three, the reason our promote is not behind
it is visible rather than inferred: the API half of the route was refused for
this revision, so there was no `STAGED` submission for the documented "wait for
`STAGED`, then promote" sequence to act on. The runbook question the earlier three
left open is not closed by this entry.

**The queue is clear.** `submitted <absent>` means no revision is queued ahead of
the next one, so this entry closes the 0.1.20 queue item: the next extension
change carries the next bump, and a fresh `chrome-web-store.yml` dispatch meets an
empty queue rather than the `NOT_UPDATEABLE` refusal a revision already in review
draws.

---

## v0.1.20 — on `main`, NOT submitted (as of 2026-09-22)

The number moved so that the extension version pins exactly one tree again.
**`0.1.19` named eight trees — by this file's own counting convention — and no
revision was ever created under it.**
`97753b4c` (`feat(browser): serve downloads from the extension, behind operator
switches`) is the commit that set the number — its parent reads `0.1.18` — and
the tree it named is `8a49f477b0e33969524de6fdc98d19150a1f6ac1`
(`git rev-parse 97753b4c:extension`). **Seven** commits then landed under
`extension/` without the bump the rule requires: `97250981`, `bd74a1fc`,
`f53cbdce`, `ab05f3da`, `594db769`, `5300a59b`, `e8ed28a1` — all file-transfer
and consent work, and all of it now inside what ships. Under this file's
counting convention — every tree that landed under the number counts, which is
what makes the 0.1.14 entry's mess *three* trees and not two — that is
**eight** trees: the tree `97753b4c` named (`8a49f477…`), plus one per landed
commit (`791a8f14…`, `a7477713…`, `de1ee574…`, `f9af392c…`, `d3664516…`,
`c66a89da…`, `270538e3…`). By 2026-09-22 `origin/main:extension` was
`270538e3dd15870212a4ef35b06b15e4e83f6f21`, a different tree reading the same
`0.1.19`. That is the ambiguity AGENTS.md forbids, and the same shape as the
0.1.14 mess below — three trees there, eight here, five more. So the repair is
the renumber the 0.1.15 entry already establishes, not a recall: `0.1.20` names
the tree this bump lands.

| Field | Value |
| --- | --- |
| Extension version | 0.1.20 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` (the same item; this is a revision of it) |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | the bump commit on `chore/extension-version-0.1.20` (SHA in that PR's thread; the tree hash below is the field that pins the input) |
| `extension/` tree hash | `c1b5780a214cb795a19bfe5b0c160ebc904e2d15` (`git rev-parse <bump commit>:extension`) |
| Artifact SHA-256 | *not applicable — not uploaded* |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Not dispatched.** Three dispatches aimed at `0.1.19` were refused by the store before this number was taken — runs [35531388197](https://github.com/damianvtran/local-operator/actions/runs/35531388197) (2026-09-20T19:09:46Z), [35604532647](https://github.com/damianvtran/local-operator/actions/runs/35604532647) (2026-09-21T13:16:17Z) and [35734051298](https://github.com/damianvtran/local-operator/actions/runs/35734051298) (2026-09-22T13:31:44Z, `headSha=0caf7a32`), each of which built and validated `v0.1.19` and was then answered `FAILED_PRECONDITION` with `reason: NOT_UPDATEABLE` and `"You may not edit or publish an item that is in review."` — 0.1.18 holds the queue |
| Promotion route | **Not dispatched** |
| Store state | Not submitted. `0.1.18` is `PENDING_REVIEW`; a fresh dispatch would read the same refusal |
| State last checked | 2026-09-22 |
| Approval timestamp | *not applicable* |
| Previously published | v0.1.17 (`PUBLISHED` at 100% — see that entry; confirmed on the public listing 2026-09-20) |

**So nothing shipped under the ambiguous number, and that is measured, not
assumed.** Every `chrome-web-store.yml` run for `0.1.19` failed, the store's own
body is the reason, and the item's revision is still 0.1.18 — no user holds a
build that reports `0.1.19`, and no revision exists that a reader could confuse
with the other tree of that name. The seven commits' contents do reach the
store, but as `0.1.20`, which every version site in the tree reads.

**Why the bump is in this commit rather than the one that changed behaviour.**
The rule is that a behaviour change carries its version bump in the same PR
(AGENTS.md). The seven commits above broke it, and this is the repair: the
version now names the tree that is actually on `main`. The runtime's
`EXPECTED_EXTENSION_VERSION` moves with it (`extension/manifest.json` and
`extension/package.json` are the source of truth; a unit test pins the constant
to them), and both generated targets — `extension/src/protocol.gen.ts` and the
`extension/ui-vendor/` bundle, whose headers carry an INPUT hash over
`protocol.py` — are regenerated with `gen_ts`, not hand-edited.

**What moved, and what deliberately did not.** Seven sites are the *curated
identity set* that carries the tree's version: the two manifests,
`EXPECTED_EXTENSION_VERSION`, the two generated
targets, and the two synthetic performance fixtures that report this tree's
version on the wire (`extension/scripts/popup-performance-live.mjs`,
`extension/tests/fixtures/worker-performance.mjs` — both had been left at
`0.1.17` through the 0.1.18 and 0.1.19 renumbers). The seven are curated, not an
inventory: the tree carries version literals in more than a dozen other fixtures
and scripts by design (`popup-states-shot.mjs`, `bridge-wedge.integration.test.mjs`,
`popup-render.integration.test.mjs`, `worker-multi-identity.integration.test.mjs`,
…), so read this list as the identity sites a renumber must move — the two
fixtures above came to be two renumbers stale precisely because a curated list
was taken as complete — never as everything that names a version.
`PROTO_VERSION` stays 1, and so does
`CAPABILITY_MIN_EXTENSION_VERSION["download"] = "0.1.19"`: that table
names the first extension **tree** that can serve a capability-gated method —
the tree `97753b4c` introduced — so it is a statement about git history rather
than about the name this tree carries, and moving it would change the refusal
copy this runtime gives an older peer. A version-pin repair changes no
behaviour — with one caveat that makes the sentence exact rather than
approximately true: `EXPECTED_EXTENSION_VERSION` is advisory-only, so nothing is
refused, but a peer reporting `0.1.19` (a locally built copy of the pre-bump
tree — nothing shipped under it) begins to see the update advisory, and the
`worker-performance.mjs` fixture moves from `0.1.17` to `0.1.20`, crossing the
`0.1.19` capability floor; both are intended and unobservable to any user. It is
checkable rather than asserted: diffing the two
`extension/` trees (`git diff 270538e3dd15870212a4ef35b06b15e4e83f6f21
c1b5780a214cb795a19bfe5b0c160ebc904e2d15`) yields **version literals and
generated header stamps and nothing else** — **17 files**, 38 changed lines,
every one of them a `"version"` literal, an `EXPECTED_EXTENSION_VERSION`, or an
`Inputs sha256:` line (the 17th, `ui-vendor/driver/file-transfer.tables.gen.ts`,
is the one file in the diff carrying two `Inputs sha256:` headers).

**A flake observed while verifying the pin, and why it cannot be
version-related.** Three wall-clock-sensitive bridge tests failed in one wider
local run and passed in another on the same host minutes apart, at load average
108-197 — PR #1435's body recorded it, and this paragraph corrects the argument
that body made, which said the two files contain "no version literal". The first
half of that claim is true and the second is not: `test_bridge_wedge.py` carries
`0.1.0`, `0.1.8` and `0.1.10`, and `test_multi_identity.py` carries `0.1.10` and
`0.1.13`. The conclusion holds on the stronger argument that replaces it: every
literal in those two files sits below both versions in play — `0.1.19`, the
capability floor that did not move, and `0.1.20`, the expected version that did
— so the predicate these tests exercise cannot flip on any literal they carry,
and no version string can reach a wall-clock path. (The two test files *call*
`extension_older` nowhere; the call sites are the daemon's advisory checks at
`daemon.py:1754` and `:4071`, against `EXPECTED_EXTENSION_VERSION`,
`resources.py:677` against `OWNERSHIP_MIN_EXTENSION_VERSION`, and
`backend.py:749` against `CAPABILITY_MIN_EXTENSION_VERSION` — so the literals the
tests carry are what decides, and all of them sit below both floors.) Recorded
rather than dropped because a
checkably-false "no literal exists" is what a later reader leans on to dismiss
a genuinely version-related flake.

## v0.1.18 — submitted 2026-09-20, pending review as of 2026-09-20

| Field | Value |
| --- | --- |
| Extension version | 0.1.18 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` (the same item; a revision of it) |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `deb6231b` — the merge commit of #1343, i.e. `main` itself at dispatch time. `main` has since moved on (#1341's mobile-bundle fix, no `extension/` changes: `git log --oneline deb6231b..origin/main -- extension/` is empty), so `git rev-parse origin/main:extension` still returns the tree hash below |
| `extension/` tree hash | `3341f4324e8368648213a22d585b636de0e0692b` (`git rev-parse deb6231bc9ee5bd18377d876a2ef5612ec8c799e:extension`) |
| Artifact SHA-256 | *not recoverable — same automated-path limitation as v0.1.17, v0.1.12, v0.1.10 and earlier* |
| Artifact size | 13 files, no source maps (`validated Chrome Web Store package v0.1.18`) — the built package, not a count of the 21-file source tree below |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged — the news is the additive `capabilities` frame, so no already-released daemon is closed by this build) |
| Submission route | **Automated** — `chrome-web-store.yml`, [run 35514131365](https://github.com/damianvtran/local-operator/actions/runs/35514131365), dispatched by `damianvtran` at 2026-09-20T13:39:09Z with `ref=main` `version=0.1.18` (`headBranch=main`, `headSha=deb6231b…`, conclusion `success`, ended 13:40:13Z). Its own build/validate step reports **283 tests, 0 fail** and `validated Chrome Web Store package v0.1.18 (13 files, no source maps)` |
| Promotion route | **Pending** — dispatch `chrome-web-store-promote.yml -f version=0.1.18` once the store reports the revision `STAGED`. No promote dispatch has been made since this submission (`gh run list --workflow chrome-web-store-promote.yml`, checked 2026-09-20); the last promote anywhere, [run 35289401051](https://github.com/damianvtran/local-operator/actions/runs/35289401051) (2026-09-18T00:01:40Z), failed like every promote since 0.1.10's success — the gap recorded under v0.1.12, v0.1.15 and v0.1.17, live for this entry too |
| Store state | `PENDING_REVIEW` — the submission's own raw response: `submitted Chrome Web Store extension omibaecbjdhgbbcedbnnnmjpmopfheof v0.1.18 with STAGED_PUBLISH (PENDING_REVIEW)` |
| State last checked | 2026-09-20T13:40:10Z — the timestamp of that response line, i.e. the run's own reading. No later probe of the queue state exists as of this entry |
| Approval timestamp | ***Not applicable — the revision is in review.*** There is no approval instant to record, so none is invented. Google publishes no SLA, and a `debugger` + `<all_urls>` extension has drawn extended manual review before (v0.1.8 ≈ 4.5 days). When it is approved this row is **not** rewritten: the correction is appended below, which is what this file requires of a shipped entry |
| Previously published | v0.1.17, `PUBLISHED` at 100% — store fields `submitted <absent>; published state=PUBLISHED distributionChannels=[crxVersion=0.1.17 deployPercentage=100]` (run [35289401051](https://github.com/damianvtran/local-operator/actions/runs/35289401051), read 2026-09-18T00:04:25Z — the last status reading before this submission), and independently confirmed on the public listing (checked 2026-09-20): **"Version 0.1.17"**, **"Updated September 16, 2026"**, **119KiB** |

**What this number names, measured against the published 0.1.17 tree rather
than summarised.** `git diff bc3cfba6c13623b036588afd67cb109426821056
3341f4324e8368648213a22d585b636de0e0692b` — the two `extension/` trees, so the
comparison needs no commit — is **21 files, +1192/-28**, across five commits
(`git log --oneline ae560d4d8..deb6231b -- extension/`: `6d0176db`,
`5ecc5569`, `83fbc7cf`, `d697c8e0`, `7d0d11a0`). The headline is **browser
file transfer**: the extension now advertises the wire methods it actually
serves (a new additive `capabilities` frame, which is why `PROTO_VERSION`
stays 1) and serves **`upload`** — attaching local files to a page's file
input through `DOM.setFileInputFiles` on the tab-scoped debugger session it
already holds, which the existing `debugger` permission covers, so no new
`chrome.*` permission (and none of the permission-driven re-review one would
bring) is added. The new modules are `src/commands/upload.ts` (+237),
`src/driver/file-transfer-policy.ts` (+200, the host-free name and credential
check, generated-from-Python tables beside it in
`src/driver/file-transfer.tables.gen.ts`, +118) and
`tests/file-transfer.test.mjs` (+134); `src/commands/input.ts` and
`src/worker.ts` change with them, and every `driver/` module is vendored into
`ui-vendor/` for the desktop app to consume, which is where **11 of those 21
paths** come from. The same window carries #1338's additive console/surface
error-code vocabulary and the version move itself; the count needs both halves,
because #1338 accounts for **11** of the new members while **this entry's own
tree diff adds 12** (the generated `src/protocol.gen.ts` enum goes 18 → 30
between `bc3cfba6…` and `3341f432…`), and an auditor who adds up the diff will
otherwise land on 12 and distrust the 11. The design of record is
`docs/design/browser-file-transfer.md`.

**`download` is in the vocabulary and served by nothing — deliberately.**
`EXTENSION_CANNOT_SERVE = ['download']` is generated into the protocol for the
same reason `capabilities` is: Chrome denies a tab-scoped `chrome.debugger`
session both CDP primitives that could answer "where does this download go"
(`Page.setDownloadBehavior` returns `-32000`, `Browser.setDownloadBehavior`
`-32601`, and no browser target is attachable — measured on Chrome 153.0.8010.53,
`docs/design/browser-file-transfer.md` §17.1). So the request is answered
`capability_unsupported`, naming the desktop app's browser tab as the host that
can serve it, rather than failing as a missing handler. The daemon side gates the
other half on this version and no other:
`CAPABILITY_MIN_EXTENSION_VERSION = {"upload": "0.1.18"}` in
`local_operator/browser_bridge/protocol.py`.

**This submission is not the #1169 renumber, and the difference is the diff.**
The paragraph above records #1169 as history: it proposed moving *this number*
onto the tree the store **already had in review**, where staging it would have
re-submitted *identical* content for another multi-day review, and it was closed
unmerged for exactly that reason. That paragraph stands unedited. What was
submitted here is the opposite shape, and it fails the renumber test on the one
measure this file already rests on: a renumber has **no diff at all** against the
tree it borrows — that is what made #1169 a renumber, and why the note says
"identical content" — whereas this number landed on a tree that differs from the
published 0.1.17 tree by **21 files, +1192/-28**, four of them new source modules
with their tests. So 0.1.18 pins exactly one tree, that tree is not the one the
store shipped as 0.1.17, and nothing here re-opens, amends or contradicts the
#1169 note. It supersedes it as a description of *this* submission, whose
justification is the diff rather than the number.

**How this entry's fields were derived.** Every value above was re-read on
2026-09-20 from `gh` and `git` at the moment of writing rather than carried over
from the dispatch order: the version sites (`manifest.json`, `package.json` and
`EXPECTED_EXTENSION_VERSION` in both `protocol.gen.ts` files) all read `0.1.18`
on `main`; the tree hash is `git rev-parse` against the recorded source commit
and separately against `origin/main`; the run's `headBranch`, `headSha`,
timestamps and both raw lines come from `gh run view 35514131365`; and the
`PUBLISHED` half of `Previously published` is confirmed on the public listing,
not from a workflow's success — a workflow exit code is not evidence that a
version is live.

**Post-submission addendum (appended 2026-09-23): approved and published, by the
promote route.** The `Approval timestamp` row above is left as written — this
file corrects a shipped entry by appending, which is what that row itself says —
and the approval is recorded here. 0.1.18 went live on **2026-09-22**, and for
the first time since 0.1.10 the documented route is what put it there:
`chrome-web-store-promote.yml`, [run
35736288300](https://github.com/damianvtran/local-operator/actions/runs/35736288300),
dispatched by `damianvtran` on `main` at 2026-09-22T13:51:32Z, completed
`success` at 13:52:11Z and printed `promoted Chrome Web Store extension
omibaecbjdhgbbcedbnnnmjpmopfheof v0.1.18 to PUBLISHED` at 2026-09-22T13:52:08Z.
The public listing is the independent half and agrees: **"Version 0.1.18"**,
**"Updated September 22, 2026"**, **122KiB** (checked 2026-09-23).

**The shape of that evidence, since it is not a timestamp.** The successful
promote run *is* the approval evidence — it is the store's own answer and the
first successful promote since 0.1.10's, against every promote dispatch in
between failing — but it is not the approval *instant*: the run refuses unless
the revision is already `STAGED`, so approval landed before 13:52:08Z and
appears in no run we hold. It is bounded from this file's own history instead:
the 0.1.19 dispatch recorded above was refused at 2026-09-22T13:33:04Z because
*"You may not edit or publish an item that is in review."* — i.e. 0.1.18 was
still *in* review then — and the promote succeeded at 13:52:08Z, so the approval
instant lies **between 2026-09-22T13:33:04Z and 13:52:08Z**. Two rows above are
consequently stale — `Store state` and `Promotion route` — and the heading's
"pending review as of 2026-09-20" is superseded by this addendum; the heading is
left standing, as this file leaves it.

---

## v0.1.17 — submitted 2026-09-15, pending review as of 2026-09-15

| Field | Value |
| --- | --- |
| Extension version | 0.1.17 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` (the same item; a revision of it) |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `ae560d4d8` — the merge commit of #1167, i.e. `main` itself at dispatch time. `main` has since moved on (0.55.9 runtime release, no `extension/` changes), so `git rev-parse origin/main:extension` still returns the tree hash below |
| `extension/` tree hash | `bc3cfba6c13623b036588afd67cb109426821056` (`git rev-parse ae560d4d8:extension`) |
| Artifact SHA-256 | *not recoverable — same automated-path limitation as v0.1.12, v0.1.10 and earlier* |
| Artifact size | 13 files, no source maps (`validated Chrome Web Store package v0.1.17`) |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Automated** — `chrome-web-store.yml`, [run 35036966759](https://github.com/damianvtran/local-operator/actions/runs/35036966759), dispatched by `damianvtran` at 2026-09-15T23:43:57Z with `ref=main` `version=0.1.17` |
| Promotion route | **Pending** — dispatch `chrome-web-store-promote.yml -f version=0.1.17` once the store reports the revision `STAGED` |
| Store state | `PENDING_REVIEW` at 100% — the store reports `submittedItemRevisionStatus` `state=PENDING_REVIEW`, `crxVersion=0.1.17`, `deployPercentage=100` |
| State last checked | 2026-09-15T23:47:46Z (the run's own timestamp) — promote run [35037244157](https://github.com/damianvtran/local-operator/actions/runs/35037244157), created 23:47:46Z, ended 23:50:05Z having refused at gate 1 and printed the store's own fields |
| Approval timestamp | ***Not directly observable — no run in our history records it.*** The version went live inside the bounded window **2026-09-17T00:33:44Z–2026-09-17T06:33:08Z** (see the post-release addendum below); the approval instant itself appears in no run we hold, so it is recorded as a bound rather than a timestamp. The outcome is verified independently on the public listing: the URL above reads **"Version 0.1.17"**, **"Updated September 16, 2026"** (checked 2026-09-17) |
| Previously published | v0.1.15, `PUBLISHED` at 100% (state confirmed by the probe above; the *publication date* is not evidenced by any run — see the promotion-route note) |

**One number, one tree — but read this before citing a SHA for 0.1.17.** The rule
is that a submitted version names exactly one tree; what makes 0.1.17 clean is the
**submission**, not the renumber that first took the number. `ef7f761c2`
("chore(extension): renumber the tree to 0.1.17") named its own tree, and two
commits afterwards changed **bundled extension source** without a bump:

- **`724fd6fea`** — `extension/src/state.ts`, which is imported by all three build
  entries (`src/worker.ts`, `src/popup/popup.ts`, `src/options/options.ts`)
- **`a6f980e49`** — `extension/src/driver/deadline.ts`, re-exported by `src/settle.ts`

So the tree `ef7f761c2` named **was never submitted**, and the store's `0.1.17` is
`ae560d4d8` — the two commits above are *inside* what shipped. Anyone auditing from
the renumber commit alone will pin the wrong tree: use the table above.

**A renumber to 0.1.18 was opened (#1169) and closed unmerged, deliberately.**
Merging it would have left `main` at 0.1.18 while the store's in-review revision is
0.1.17 built from 0.1.17's tree — `main` ahead of the queue with no submission
behind it — and the obvious follow-through, staging 0.1.18, would have re-submitted
identical content for another ~4.5-day review of a `debugger` + broad-host
extension. The branch remains the template if a future submission needs a number
above 0.1.17: it moved all seven version sites and regenerated **both** generated
targets (`extension/src/protocol.gen.ts` and `extension/ui-vendor/`), with
`gen_ts --check` green. That generator's two-target behaviour is worth knowing when
the next bump happens — the vendored copy is not optional, because the generated
header carries an input hash over `protocol.py`, and editing only the extension
target leaves that hash stale.

**Promotion route, second occurrence of the same gap.** As with v0.1.12, the
publication of v0.1.15 has **no successful `chrome-web-store-promote.yml` run**
behind it — the last successful promote anywhere remains 0.1.10's (run 34610983340,
2026-09-11), and every promote dispatch since has failed. Google's publish
reference says `STAGED_PUBLISH` stages on approval and is then published *by the
developer* (against `DEFAULT_PUBLISH`, which publishes on approval), so an approved
staged revision does not go live by itself: the publication was an explicit
developer action taken from the dashboard or an out-of-band API call, by an
operator or session this checkout holds no record of. Our documented sequence
("wait for `STAGED`, then promote") is not implicated, but it is also not what
happened, twice — worth resolving in the runbook rather than re-deriving per
release.

**Post-release addendum (appended 2026-09-17).** This entry was written while the
version sat in review. It has since gone live **without any successful
`chrome-web-store-promote.yml` run behind it** — the third occurrence of the gap
already recorded under v0.1.12 and v0.1.15 — and the store now reports it
**`PUBLISHED` at 100%**: `publishedItemRevisionStatus` `state=PUBLISHED`,
`crxVersion=0.1.17`, `deployPercentage=100`, with `submittedItemRevisionStatus`
`<absent>`. Five rows above and the heading are consequently stale — `Promotion
route`, `Store state`, `State last checked`, `Source commit` and `Artifact size` —
and the heading's "pending review" is superseded by this addendum. The heading is
left standing rather than rewritten, because this file does not rewrite a shipped
entry: what the entry records is what we intended to ship at submission time, and
the corrections belong here.

**The promotion route, for the third time, was not the documented one.** Eight
promote dispatches landed between 2026-09-15T23:47:46Z (minutes after the submission)
and 2026-09-17T06:32:41Z — both run-creation times, as `gh run list` reports them —
and every one of them failed; the last of them, run
[35190293922](https://github.com/damianvtran/local-operator/actions/runs/35190293922),
being the very probe whose output shows the version already `PUBLISHED`. The last
successful promote anywhere remains 0.1.10's (run 34610983340, 2026-09-11). Google's
publish reference says `STAGED_PUBLISH` stages on approval and is then published *by
the developer* (against `DEFAULT_PUBLISH`, which publishes on approval), so an
approved staged revision does not go live by itself: the publication was an explicit
developer action taken from the dashboard or an out-of-band API call, by an operator
or session this checkout holds no record of. Our documented sequence ("wait for
`STAGED`, then promote") is not implicated, but it is also not what happened, three
times now — worth resolving in the runbook rather than re-deriving per release. So
the `Promotion route` row above must not be read as "our promote workflow published
this": it did not, and the record is the audit artifact that has to survive that
question.

**The publication window, and why the listing's date reads a day earlier.**
Three probes on 2026-09-17 print the store's fields. Every timestamp in this
paragraph is that of the workflow's own log line — the instant the store was
actually read, a few tens of seconds after each run was created — so that no
reader has to guess which basis a given time is on. Two of them, with 0.1.17
still in review, printed the same fields — run
[35164942781](https://github.com/damianvtran/local-operator/actions/runs/35164942781)
at 00:04:01Z and run
[35167032053](https://github.com/damianvtran/local-operator/actions/runs/35167032053)
at 00:33:44Z:

```text
submitted state=PENDING_REVIEW distributionChannels=[crxVersion=0.1.17 deployPercentage=100]; published state=PUBLISHED distributionChannels=[crxVersion=0.1.15 deployPercentage=100]
```

and the 06:33:08Z reading (run
[35190293922](https://github.com/damianvtran/local-operator/actions/runs/35190293922),
the one recorded in `State last checked`) printed:

```text
submitted <absent>; published state=PUBLISHED distributionChannels=[crxVersion=0.1.17 deployPercentage=100]
```

So 0.1.17 went live **between 2026-09-17T00:33:44Z and 2026-09-17T06:33:08Z** — the
latest reading still showing `crxVersion=0.1.15` published, and the first showing
`0.1.17` — and the approval instant itself appears in no run we hold: it is recorded
as a bound, not invented as a timestamp. The listing's own `Updated` string says
**September 16, 2026**, which is not a contradiction as far as we can tell: that
whole window falls on September 16 in US Pacific time (00:33:44–06:33:08Z =
17:33:44–23:33:08 PDT), so the store's date and our UTC bound agree once the store's
day boundary is allowed for. That reading is an inference from the two sources, not
a field the store exposes, and it is written here so a later reader does not read
the pairing as an error. The listing's own page is the check that stands behind the
live claim — it reads **"Version 0.1.17"** and **"Updated September 16, 2026"**
(checked 2026-09-17) — not a workflow's success.

**Two row corrections, recorded here rather than applied to the rows.** `Source
commit` named the runtime release that had landed since dispatch, which was true when
written; what matters to a reader is the invariant behind it — nothing under
`extension/` has been committed since `ae560d4d8`
(`git log --oneline ae560d4d8..origin/main -- extension/` is empty at this head), so
`git rev-parse origin/main:extension` still returns the recorded tree hash
`bc3cfba6c13623b036588afd67cb109426821056`. And `Artifact size` omits the size the
store listing now reports: **119KiB**, against the 13 files the zip contains. Both
statements supersede the rows above; the rows themselves are left as written.

## v0.1.15 — on `main`, NOT submitted (as of 2026-09-14)

The number moved so that the extension version pins exactly one tree again.
**Three** trees landed on `main` under the never-submitted `0.1.14`: the
version-skew tree (`d44560d3…`), the popup-collapse tree (`7b6a0c65…`, PR #1095)
and the driver-move tree (`a1a53479…`, PR #1104) — the latter two changed
`extension/` without a bump, which is the rule AGENTS.md states and the reason
this number exists. `0.1.15` names the tree this bump lands: the version-skew
tree with the range/guard work of #1038, #1095 and #1104 underneath it.

| Field | Value |
| --- | --- |
| Extension version | 0.1.15 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` (the same item; this is a revision of it) |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | the bump commit on `chore/extension-version-0.1.15` (SHA in that PR's thread; the tree hash below is the field that pins the input) |
| `extension/` tree hash | `41c61cc4198bab327eac49108aa1e7b60d295d6a` (`git rev-parse <bump commit>:extension`, rebased onto `main` so it is the tree this version actually lands with — a hash measured before the rebase would pin a tree nobody merges) |
| Artifact SHA-256 | *not applicable — not uploaded* |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Not dispatched.** Running `chrome-web-store.yml` requires a clear queue; the live listing is still `v0.1.10` with `v0.1.12` in review (`PENDING_REVIEW`), and the store refuses uploads while an item is queued |
| Promotion route | **Not dispatched** |
| Store state | Not submitted. `v0.1.12` was still `PENDING_REVIEW` when this number was taken |
| State last checked | 2026-09-14 |
| Approval timestamp | ***Not directly observable — no run in our history records it.*** The store's own fields bound it instead, and the publication **date is 2026-09-15** on both our UTC log-line base and the store's own Pacific day: promote run [35032253219](https://github.com/damianvtran/local-operator/actions/runs/35032253219) read the store at 2026-09-15T22:42:34Z with 0.1.15 already approved (`submitted state=STAGED …[crxVersion=0.1.15 deployPercentage=100]`) and 0.1.12 still live, and run [35035927938](https://github.com/damianvtran/local-operator/actions/runs/35035927938) — dispatched for 0.1.15 — read it at 2026-09-15T23:31:02Z with 0.1.15 live and nothing left submitted (`submitted <absent>; published state=PUBLISHED …[crxVersion=0.1.15 deployPercentage=100]`). The approval **instant** itself appears in no run we hold, so it is recorded as a bound rather than a timestamp, and no time is inferred: our promote route did not publish this version — the dispatch aimed at 0.1.15 was refused (`only an approved STAGED revision can be promoted`) because it was already live — and the public listing reports a date, not a timestamp. See the addendum below. |
| Previously published | v0.1.10 (0.1.12 in review) |

**Why the bump is in this commit rather than the one that changed behaviour.**
The rule is that a behaviour change carries its version bump in the same PR
(AGENTS.md). #1095 broke it, and this is the repair: the version now names the
tree that is actually on `main`. The runtime's `EXPECTED_EXTENSION_VERSION`
moves with it (`extension/manifest.json` and `extension/package.json` are the
source of truth; a unit test pins the constant to them), so the update advisory
keeps telling users a newer extension exists without ever implying the store can
serve it yet.

**Post-release addendum (appended 2026-09-15).** This entry was written when the
number had been taken but nothing had been dispatched. It **was** subsequently
submitted: `chrome-web-store.yml` run 34886613253, 2026-09-14T19:23:41Z,
`STAGED_PUBLISH` → `PENDING_REVIEW`; the store now reports it **`PUBLISHED` at
100%** (`publishedItemRevisionStatus` `crxVersion=0.1.15`, `deployPercentage=100`,
read 2026-09-15). Two rows below are consequently stale — `Store state` and
`Promotion route` — and the heading's "NOT submitted" is superseded by this
addendum. The heading is left standing rather than rewritten, because this file
does not rewrite a shipped entry. As with v0.1.12, no successful
`chrome-web-store-promote.yml` run exists behind that publication; see the v0.1.17
entry for the full reading.

## v0.1.13 and v0.1.14 — two trees landed on `main`, NEITHER submitted

Two numbers, two trees, one story, recorded together so neither reads as an
unexplained footnote to the other: 0.1.13 is #1038's tree, 0.1.14 is the
version-skew tree, and **neither has ever been sent to the store.**

| Field | Value |
| --- | --- |
| Extension versions | **0.1.13** — #1038's tree (`ced1828f4`, `feat(bridge): pair several extension identities at once`); **0.1.14** — the version-skew tree (`fix/extension-version-skew-does-not-break-bridge`) |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` (one item; the versions are revisions of it) |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | 0.1.13: `ced1828f4`. 0.1.14: the bump commit on `fix/extension-version-skew-does-not-break-bridge` (SHA in that PR's thread — recording it inside the commit that creates it is impossible, which is why the tree hash below is the field that actually pins the input) |
| `extension/` tree hash | 0.1.13: `8c351b2525b97da9bdd3075e3e2359886c3a8596` (`git rev-parse ced1828f4:extension`). 0.1.14: `d44560d35d98d05b6e7bb8f3f89261bd45468962` (`git rev-parse <0.1.14 bump commit>:extension`, computed from the staged tree) |
| Artifact SHA-256 | *not applicable — neither was uploaded* |
| Artifact size | *not applicable — neither was built for upload* |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged); the runtime now accepts the WINDOW `MIN_SUPPORTED_PROTO..PROTO_VERSION` |
| Submission routes | **None for either number.** Deliberately NOT dispatched — see below |
| Promotion routes | **None.** Deliberately NOT dispatched |
| Store state | **Neither submitted.** The live listing is still `v0.1.10`; `v0.1.12` is `PENDING_REVIEW` |
| State last checked | 2026-09-13 |
| Approval timestamp | *not applicable* |
| Previously published | v0.1.10 (0.1.12 still in review) |

**Why two entries became one.** Both numbers moved on `main` without a
submission, and they moved for the same reason: `extension/manifest.json` and
`extension/package.json` track the extension CODE, not the review queue
(AGENTS.md), so a behaviour change has to carry a bump in the same commit or the
version stops pinning exactly one tree (the 0.1.9 lesson). #1038 bumped to 0.1.13
for the multi-identity worker, popup and pairing work; the version-skew change
then had to bump again rather than share the number, because two different trees
reading `0.1.13` is exactly the ambiguity that lesson forbids. So: **`0.1.13`
means #1038's tree (`ced1828f4`), `0.1.14` means the version-skew tree**, and the
two tree hashes above are how a reader tells them apart without a rebuild.

**The store dispatch is deliberately withheld for both.** The store refuses
uploads while an item is in review (`HTTP 400 FAILED_PRECONDITION /
NOT_UPDATEABLE`), and the recorded rule is never to cancel a pending review to
force a submission. Whichever of the two trees is promoted next — the one the
next submission window picks, after 0.1.12 clears — needs its own workflow run,
API response and tree hash appended here before promotion; until then this
section claims nothing is live.

**Post-script (2026-09-14): 0.1.14 came to name THREE trees.** Two PRs landed
under `extension/` with no version bump, so both manifests kept reading `0.1.14`
while the tree moved twice underneath them: PR #1095
(`fix(extension): stop the popup clamping its card to its own window`, merge
`d383e6bfe`) took it to `7b6a0c65901ba5a94ae5ab1cfbaac3e501a766b5`, and PR #1104
(`refactor(browser-driver): extract the host-free modules under src/driver/`,
merge `3905e0d2a`, merged 13:33:21Z on 2026-09-14) took it to
`a1a5347974003cf77e20837c5e7e7fd634ab5356`. That is the 0.1.9 ambiguity again,
caught before either number was submitted; the version was therefore moved on to
**0.1.15** (see the section above) rather than letting `0.1.14` stand for three
trees. `d44560d3…` above remains the tree `0.1.14` named at the moment this
section was written. **Any further change under `extension/` must carry its own
bump in the same PR** — this section records what happens when it does not.

**Nothing about either version is required for the version-skew fix to work.**
That defect was fixed on the RUNTIME side (an older extension is driven, not
refused), and the advisory only reads the version the extension reports — so the
live 0.1.10 build benefits without any upload at all.

## v0.1.12 — submitted 2026-09-13, published 2026-09-14

| Field | Value |
| --- | --- |
| Extension version | 0.1.12 |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `7c78b6eda` (merge commit of PR #1026 on `main`) |
| `extension/` tree hash | `3069532b9f2ac82fb6603f2ec5e2f453639ac008` (the deterministic input pin — see the audit note above) |
| Artifact SHA-256 | *not recoverable — same automated-path limitation as v0.1.10, v0.1.8 and earlier* |
| Artifact size | 113KiB as reported by the store listing; 13 files, no source maps |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Automated** — `chrome-web-store.yml`, [run 34731860480](https://github.com/damianvtran/local-operator/actions/runs/34731860480), dispatched with `ref=7c78b6eda` `version=0.1.12` |
| Promotion route | **Developer-initiated, not our workflows** — no `chrome-web-store-promote.yml` run for 0.1.12 ever succeeded (the last successful promote anywhere is 0.1.10's, run 34610983340, 2026-09-11); see the note below |
| Store state | **`PUBLISHED` at 100%** — the store reports `publishedItemRevisionStatus.state = PUBLISHED`, `distributionChannels = [crxVersion 0.1.12, deployPercentage 100]` |
| State last checked | 2026-09-15T22:42Z — store `fetchStatus`, read through the diagnostics added in PR #1160 (promote run 35032253219) |
| Approval timestamp | *not recorded by any workflow* — see the note below |
| Previously published | v0.1.10, live during this review |

**First submitted revision carrying both halves of the stale-worker defect.** The
store queued exactly one version for the whole defect area, deliberately:

- **#996** (`99e81a1c5`, merged 2026-09-12T23:48:28Z) — the daemon/extension
  wedge work: an unresponsive extension no longer wedges every session, and the
  popup's `#unresponsive` card states the manual OFF/ON remedy.
- **#1026** (`7c78b6eda`) — the operator-reported follow-up from this defect's
  own symptom ("clicking the extension icon sometimes does nothing, 2-3 tries"):
  the MV3 worker's remaining uncaught throws are contained, the popup's
  `decide()` is bounded and failure-safe so a dead worker can never leave
  Allow/Deny disabled, the reload remedy is reachable from the consent card, and
  the first-paint pins are state-aware.

**`0.1.11` was never submitted.** #996 carried a `0.1.11` bump on its branch, but
both PRs agreed to hold the store dispatch so the operator paid one review wait
rather than two for the same defect. That version therefore never existed on the
store, and reading it as a skipped release is correct.

**How it was published: developer-initiated, and not by these workflows.** The
Chrome Web Store API v2 `publishers.items.publish` reference settles the branch
that matters — `STAGED_PUBLISH`: "After approval the submission will be staged and
can then be published by the developer", against `DEFAULT_PUBLISH`: "The
submission will be published immediately on approval"
(https://developer.chrome.com/docs/webstore/api/reference/rest/v2/publishers.items/publish).
An approved staged revision therefore cannot go live by itself, so the publication
was an explicit developer action. What our records cannot say is *whose*: no
`chrome-web-store-promote.yml` run for 0.1.12 ever succeeded, the last successful
promote anywhere is 0.1.10's (run 34610983340, 2026-09-11), and between the last
gate-1 refusal of the other session's pending submission (`state != STAGED`,
2026-09-15T05:31Z) and the first gate-2 refusal (2026-09-15T17:31Z) the only
store-workflow activity was three refused *stage* attempts for 0.1.17
(08:38:31Z, 11:40:38Z, 14:43:01Z). It was published from the publisher dashboard,
or by an out-of-band API call, by an operator or session this checkout holds no
record of. The documented process is not implicated: staging defers publication,
exactly as this file assumes.

**A correction to an earlier revision of this entry, kept visible rather than
quietly dropped.** Two claims were wrong and are fixed here: the gate-2 refusal
sequence begins 2026-09-15T17:31Z, not 05:31Z (05:31Z was still a gate-1 refusal,
so the submitted revision became `STAGED` between the two), and a claim that no
other store workflow ran in the window was false for the three 0.1.17 stage
attempts listed above.

**The queue as of 2026-09-15T22:42Z**, the only recorded store read: the
*submitted* revision is a later **0.1.15** (submitted 2026-09-14, run 34886613253,
with `STAGED_PUBLISH`), and `fetchStatus` reports it `STAGED` at 100%. Five
subsequent 0.1.17 stage attempts were refused with
`FAILED_PRECONDITION`/`NOT_UPDATEABLE` — "you may not edit or publish an item that
is in review". Two cautions for whoever acts on it: a promote dispatch must name
the version that is actually staged (naming a published version fails safely on
the version mismatch, but before PR #1160 said nothing about why), and `main` has
since been renumbered to **0.1.17**, so promoting the staged **0.1.15** would put
a tree older than `main` on the store. The slot is holding a superseded revision,
which is a decision for whoever owns the renumber rather than a mechanical
promote. Note also that the `v0.1.15` entry above still reads "NOT submitted",
which its own 2026-09-14 submission falsifies: this file protects *shipped*
entries, that entry is not mine, and it is flagged here for its owner rather than
edited.

**Live now.** The store build is 0.1.12, so the earlier instruction to exercise
the fix from an unpacked build of the merge commit no longer applies — the
installed extension updates in place under the same item id, with no re-pairing.
The port-4099 hazard is still worth remembering for anyone running the capture
harness against a local build.

## v0.1.10 — submitted 2026-09-10, published 2026-09-11

| Field | Value |
| --- | --- |
| Extension version | 0.1.10 (**the live published version**) |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `7ad53a5e1` (`chore(extension): bump version to 0.1.10 and add store publishing guidance`, PR #907, squash-merged to `main`) |
| `extension/` tree hash | `7d5585a6d430ae57b4cc0948f148ffa4c9c8fa5b` (the deterministic input pin — see the audit note above) |
| Artifact SHA-256 | *not recoverable — same automated-path limitation as v0.1.8 below* |
| Artifact size | 13 files, no source maps; 104KiB as reported by the store listing |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Automated** — `chrome-web-store.yml`, [run 34489484307](https://github.com/damianvtran/local-operator/actions/runs/34489484307), dispatched with `ref=7ad53a5e1` `version=0.1.10` |
| Promotion route | **Automated** — `chrome-web-store-promote.yml`, [run 34610983340](https://github.com/damianvtran/local-operator/actions/runs/34610983340), dispatched with `version=0.1.10` |
| Store state | `PUBLISHED` (promoted 2026-09-11 ~14:35 UTC) |
| State last checked | 2026-09-11 |
| Approval timestamp | 2026-09-11 — review completed and the approved revision was promoted the same day; confirmed live on the listing as "Updated September 11, 2026" |
| Previously published | v0.1.8, live during review (promoted 2026-09-10, run 34482703148) |

**Third release through the automated path, both halves of it.** The stage run
validated and submitted in one pass — `validated Chrome Web Store package v0.1.10
(13 files, no source maps)`, then `submitted ... v0.1.10 with STAGED_PUBLISH
(PENDING_REVIEW)` — and the promote run later reported `promoted Chrome Web Store
extension ... v0.1.10 to PUBLISHED`. v0.1.7 was the first release to go out that
way on both halves (see its entry below); the releases before it involved a
dashboard step.

**What this version carries.** Four merged PRs landed in the `extension/` tree
between v0.1.8's source commit (`ff9b16b5`) and this one:

- **#766** (`e3cb1ebc8`) — removes the deterministic jitter from the pairing
experience; the user-reported "jittery pairing code" fix. This is why the
release matters to users.
- **#798** (`ee146fb73`) — makes tab allocation exception-safe and ownership
durable. Adds `OWNER_REFUSED` and four `owner_*` methods to `protocol.gen.ts`.
- **#782** (`be2546ec0`) — release tooling only: the store script surfaces the
store's own rejection reason instead of a bare `curl: (22)`. Not shipped in the
store package (it is absent from the `extension/store-package-files.txt`
allowlist, which is what defines the thirteen zipped files).
- **#907** (`7ad53a5e1`) — the version bump itself, plus the store-publishing
guidance now carried in `AGENTS.md`.

**Why this version is 0.1.10 and not 0.1.9.** 0.1.9 was never submitted. It was
bumped in the tree, but #798 then changed nine `extension/` files — including
`protocol.gen.ts` — **without** bumping the version, so "0.1.9" no longer named a
single tree and could not be pinned by this record. #907 bumped to 0.1.10 to
restore the version-to-tree correspondence before submitting. There is therefore
no v0.1.9 entry in this file and none should be added: no artifact with that
version ever reached the store.

**Permissions unchanged from v0.1.8.** No permission was added, removed, or
altered, which is why the automated path applied without a dashboard step. The
standing rule in `submission-checklist.md` sends any permission-adding package to
a human, because the Chrome Web Store API cannot set permission justifications.

**Review duration: bounded to a ~12-hour window, not measured.** The store
reports no approval time in either workflow log, so this is bounded from run
history the same way the v0.1.7 entry below is:

- Submitted 2026-09-10T14:31Z (run 34489484307).
- A promote attempted 2026-09-11T02:35Z, ~12 hours later, **failed** with
  `Chrome Web Store publish failed: only an approved STAGED revision can be
  promoted` (run 34555183445) — so the revision was still unapproved then.
- The promote succeeded 2026-09-11T14:34Z (run 34610983340), so it was approved
  by then.

Approval therefore landed between 2026-09-11T02:35Z and 2026-09-11T14:34Z —
between ~12 and ~24 hours after submission, against ~4.5 days for v0.1.8. Both
are ordinary for an extension carrying `debugger` plus broad host access; do not
read the difference as a trend from two samples.

---

## v0.1.8 — submitted 2026-09-06, published 2026-09-10

| Field | Value |
| --- | --- |
| Extension version | 0.1.8 (published 2026-09-10; **superseded by v0.1.10**) |
| Item ID | `omibaecbjdhgbbcedbnnnmjpmopfheof` |
| Listing URL | https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof |
| Source commit | `ff9b16b5` (`feat(extension): scoped Allow (domain/site/once) and dangerous allow-all setting (0.1.8)`, PR #672, squash-merged to `main`) |
| `extension/` tree hash | `0605d9a5c34a2681c49665a8b9e8aaaada89c50b` (the deterministic input pin — see the audit note above) |
| Artifact SHA-256 | *not recoverable — see note below* |
| Artifact size | 12 files, no source maps (byte size not recorded — see note below) |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Automated** — `chrome-web-store.yml`, [run 34000911184](https://github.com/damianvtran/local-operator/actions/runs/34000911184), dispatched with `ref=main` `version=0.1.8` |
| Promotion route | **Automated** — `chrome-web-store-promote.yml`, [run 34482703148](https://github.com/damianvtran/local-operator/actions/runs/34482703148), dispatched with `version=0.1.8` |
| Store state | `PUBLISHED`, 100% deployment |
| State last checked | 2026-09-10 |
| Approval timestamp | 2026-09-10 — appended when review completed; superseded by v0.1.10 on 2026-09-11 |
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
