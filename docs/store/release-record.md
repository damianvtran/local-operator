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
| Approval timestamp | *not applicable* |
| Previously published | v0.1.10 (0.1.12 in review) |

**Why the bump is in this commit rather than the one that changed behaviour.**
The rule is that a behaviour change carries its version bump in the same PR
(AGENTS.md). #1095 broke it, and this is the repair: the version now names the
tree that is actually on `main`. The runtime's `EXPECTED_EXTENSION_VERSION`
moves with it (`extension/manifest.json` and `extension/package.json` are the
source of truth; a unit test pins the constant to them), so the update advisory
keeps telling users a newer extension exists without ever implying the store can
serve it yet.

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
| Artifact size | 13 files, no source maps (size as reported by the store listing once published) |
| Bridge protocol version | `PROTO_VERSION = 1` (unchanged) |
| Submission route | **Automated** — `chrome-web-store.yml`, [run 34731860480](https://github.com/damianvtran/local-operator/actions/runs/34731860480), dispatched with `ref=7c78b6eda` `version=0.1.12` |
| Promotion route | **Unresolved — see the note below.** No `chrome-web-store-promote.yml` dispatch for 0.1.12 ever ran successfully after submission |
| Store state | **`PUBLISHED` at 100%** — the store reports `publishedItemRevisionStatus.state = PUBLISHED`, `distributionChannels = [crxVersion 0.1.12, deployPercentage 100]` |
| State last checked | 2026-09-15 (store `fetchStatus`, read through the diagnostics added in PR #1160) |
| Approval timestamp | *not recorded by any workflow* — see the promotion note below |
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

**Review queue expectations.** Google publishes no SLA for review, and an
extension using `debugger` with `<all_urls>` routinely draws extended manual
review: 0.1.8 took ~4.5 days, 0.1.10 cleared the next day. Do not cancel the
pending review to force a resubmission — cancelling forfeits the accrued queue
position with no visibility into how close it was.

**How it was published — developer-initiated, and our workflows did not do it.**
Google's publish reference settles the branch that matters: `STAGED_PUBLISH` means
the submission "will be staged and can then be published by the developer" once
approved, while `DEFAULT_PUBLISH` is the variant that publishes immediately on
approval. So an approved staged revision cannot go live by itself, and the
publication was an explicit developer action. What our own records cannot say is
*whose*: the last successful `chrome-web-store-promote.yml` run anywhere is
34610983340 (0.1.10, 2026-09-11), every promote dispatch since has failed, and the
only store workflow that ran in the window between the last gate-1 refusal
(2026-09-15T05:31Z) and the first gate-2 refusal (2026-09-15T17:31Z) was the
0.1.15 *stage*. The revision was therefore published from the publisher dashboard
or by an out-of-band API call, by an operator or session this checkout has no
record of. The documented process is not implicated: staging defers publication,
exactly as this file assumes.

**The record's own misreadings, corrected.** Two claims in an earlier revision of
this entry were wrong and are fixed here rather than quietly dropped: the gate-2
sequence begins at 2026-09-15T17:31Z, not 05:31Z (05:31Z was still a gate-1
refusal, so the submitted revision became `STAGED` between the two); and "no other
store or release workflow ran in that window" was false — the 0.1.15 stage ran
inside it.

**The queue as of 2026-09-15T23:00Z.** The *submitted* revision is a later
**0.1.15** (submitted 2026-09-14T19:24:54Z with `STAGED_PUBLISH`; `fetchStatus`
now reports it `STAGED` at 100%), and five 0.1.17 stage attempts were refused
with `FAILED_PRECONDITION`/`NOT_UPDATEABLE` — "you may not edit or publish an
item that is in review" — the earliest three of them while 0.1.15 was still in
review. Two cautions for whoever acts on it: a promote dispatch must name the
version that is actually staged (naming a published version fails safely on the
version mismatch, but before PR #1160 said nothing about why), and `main` has
since been renumbered to **0.1.17**, so promoting the staged **0.1.15** would put
a tree older than `main` on the store — the queue slot is holding a superseded
revision, which is a decision for whoever owns the renumber rather than a
mechanical promote. Note also that the `v0.1.15` entry elsewhere in this file
still reads "NOT submitted", which its own submission on 2026-09-14 falsifies;
that entry is not mine to edit, so it is flagged here for its owner.

**Live now.** The store build is 0.1.12, so the earlier instruction to
exercise the fix from an unpacked build of this merge commit no longer applies —
the installed extension auto-updated in place under the same item id, with no
re-pairing. The port-4099 hazard is still worth remembering for anyone running
the capture harness against a local build.

