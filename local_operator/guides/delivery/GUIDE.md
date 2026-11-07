---
name: delivery
description: "Allocate bounded engineering delivery, brief so a child can land, preflight costly evidence, and converge review and QA without duplicating full-suite gates."
---

# Bounded delivery

AGENTS.md remains the authority for isolation, commands, security, merge/release,
approvals and public notes; this changes scope/allocation, not validation depth.
Keep internal gate evidence out of user-facing notes.

## Allocate once, inspect narrowly

- A manager owns routine bounded technical allocations inside the authorized
  task. Choose a reversible implementation and a bounded test lease; do not
  escalate ordinary library, layout, test or lane choices. Escalate genuine
  access/security decisions, destructive actions, operator preferences and
  material scope changes. Authorization to fix is not authorization to delete.
- Give each slice one owner identified by stable session ID, scope, acceptance
  criteria and a durable receipt (project record or linked task/PR). Address the
  owner directly; no status hunts or unchanged resends after a durable ACK.
  Update only material deltas; verify completion from primary evidence.
- Prefer metadata/narrow indexes before logs. Verify surprising first reads one
  step further; cite relayed sources. Root recovery restores only prerequisites;
  cross-repository hardening is separate scope.

## Delegate so the child can land

Briefs name file:line, failure and acceptance check; do not delegate rediscovery.
Coder children edit within ~10 calls; after ~40 without an edit, return the
blocker. A repeat "explored, nothing landed" needs a smaller brief after checking
for a harness cut-off (cut children end named).

## Preflight before expensive dispatch

Before native/browser/hardware dispatch, plan changed surfaces,
commands, dependencies, source/binary identity, isolation, expected outputs and a
negative control. Verify compiled source identity and success/failure detection;
source-only checks are not native proof. Failed setup stops dispatch: report the
prerequisite and bounded repair. Use the harness browser, no second stack; follow
AGENTS.md's HOME/config, CMUX stripping and exact-process-group teardown rules.

**Budget costly evidence.** A native, simulator, device or hosted E2E lane gets
3 failed-or-cancelled attempts OR 6 h from first dispatch — the same budget
measured two ways: PR #156 (local-operator-mobile) saw 36 failures at a 118 min
median cadence. Attempts lost to our own harness count. Spent, stop dispatching
and escalate with each option's consequence: merge on unit/contract tests plus
manual device QA; split the harness into its own PR; drop the row; or extend by a
stated amount for a stated reason. No new mechanism and no retry while the
escalation is open. A lane never once succeeding in 3 attempts is a harness
problem until shown otherwise — 20 of those 36 failures, 41.4 h, came after the
last product commit.

**Measure before you cap; name the invariant.** No new timeout, cap, threshold or
retry count without the distribution it bounds (n, median, max, source) and a
setting above the observed max; every mechanism (cache, reuse, skip, fixture,
gate) and shortcut names its invariant and the check proving it — no check, no
shortcut. PR #156 capped hierarchy reads at 20 s over a measured 66–173 s, and
reused a candidate app on a fixture port that was ephemeral: a round-36 blocker.

## Worktrees

The manager (or lead coder) owns ONE PR worktree in `<repo>/.worktrees/` from
latest origin/main, briefs its absolute path and SHA, and freezes the head while
readers work. Reviewer/QA/UX/design/copy/i18n reuse it read-only every round
(`git checkout --detach <head>`, `git diff/show`, targeted tests, shared deps):
no clone, install or new worktree, and no full-tree copies into scratch. Python:
`PYTHONPATH=<worktree>` with the primary `.venv`; `python -c 'import
local_operator;print(local_operator.__file__)'` must print a worktree path.
Exception, with a manager lease: a conflicting commit author or an isolated
mutable build collision gets a detached-SHA worktree on shared deps, bounded and
creator-removed. Keep isolated HOME/config. Remove clean
worktrees on merge/close; the manager verifies (`scripts/worktree_janitor.py`).

## Never use a live peer as the revoke test

Never remove a live peer to test revocation. Use a disposable peer; if none can
be created, exercise revocation at the grant/token layer instead. Before any
membership change, list the peer's sessions and worktree state, confirm a
reachable head and console path, get explicit operator approval naming the
peer, and record it. Authorization to test is not authorization to delete.

## Spend coverage once

Iterate with targeted tests/lints; heavy runs bounded/serial. Full coverage once
at the terminal head: when CI runs the same suite, use it, not a duplicate local
run. Run omitted required surfaces bounded and record them. Merge needs actual
checks and independent review/QA; CI never replaces real-path QA. Resource
holds forbid hold-class runs, not evidence. Read CI once at the terminal merge
gate, never sleep-poll it mid-cycle; investigate even out-of-slice fails.

## Converge, do not restart

**Converge the base once, at the merge gate.** Do not merge or rebase the base
into a PR mid-review; the one exception is GitHub reporting DIRTY where a required
run cannot start — prefer `workflow_dispatch` on the branch ref, because a
`pull_request` workflow gets no merge ref on a conflicting PR and never starts.
At the gate: converge, rebase proof, ONE round on files upstream also touched, CI.
PR #156 merged main 13 times, each a proof, a pin recompute and often a round.

**Harness-only delta: no numbered round.** After a clean round on head R, delta
R..H takes no `### Agent review — round N` when
`scripts/harness_delta_proof.py R H` exits 0: same merge-base;
`.github/harness-paths` present and unchanged; every changed path (added,
modified, deleted, both sides of a rename, mode) under a `harness:` glob; nothing
ship-reachable. That proves the one invariant: *no byte the shipping pipeline
consumes differs between R and H*. Manifest `gate:` entries (checks that ship no bytes) are
product if edited and seed no closure, so the probes they run stay exempt-able; a harness
file only a gate names is listed `gate-run` and gets the FULL review lens at the merge gate.
Instead, ONE `### Harness delta check — R..H`
at the merge gate by an independent reviewer reading for gate weakening (deleted
or loosened assertions, skips, widened tolerances, `continue-on-error`, a probe
that cannot fail) AND for any harness change that runs with release triggers,
secrets, signing or publishing, plus the script output the merger re-runs; QA
retained, shipped bytes unchanged. Anything unreadable, unresolvable or
unclassifiable is a round or exit 2, never a pass.

Honest yield: it exempts only manifest-declared harness files nothing in the ship
closure names — a reachability classifier made safe by failing closed, not a proof
that a harness change cannot alter behaviour.

**Split when the harness outgrows the feature.** At each review request compare
added lines under harness globs with all others (`--split` prints both); harness
above product moves the harness to its own PR (review plus its own negative
control, no product QA), leaving the feature a minimal proof. PR #156 crossed at
round 1 (2,787 vs 2,764 lines) and stayed crossed from round 11 (9,780 vs 4,730).

Complete every applicable initial stream (code, independent real-path QA,
design, UX, security) across its full in-scope surface before remediation.
Early BLOCKER alerts must say incomplete. Review runs asynchronously with CI.

Manager collects all reports into ONE deduplicated severity/evidence ledger and
batches clarifications before edits. Implementer verifies claims, then fixes
accepted IDs in ONE pass, answering each with fixed SHA, rejection evidence, or
deferral reason. Default ONE commit; no per-finding commit/review cycles.

Follow up on `<previous_reviewed_head>..<new_head>` plus direct regression
neighbors, never unchanged re-audits. Late BLOCKER/MAJOR needs a concrete code
path/reproducer in changed/direct-neighbor scope. Retain evidence only with
byte/source compatibility proof; backend response-shape changes may need rendered
proof. Provenance-only addenda reset nothing.

After two unsuccessful cycles, freeze severity/evidence ledger, root-transition/
authority matrix (failing state, intended transition, authoritative owner/source)
and exit criteria BEFORE more patches. An unknown persisting twice escalates to
its owner. This checkpoint never waives unresolved BLOCKER/MAJOR or evidence gates.

A moved base (rebase OR merge of main) is NOT a blind re-pin: post
`scripts/rebase_proof.py`'s `### Rebase proof` (range-diff `=`, byte-identical
patch, no file overlap). MECHANICAL keeps the last clean code and QA rounds; CI
on the new head guards cross-file breaks. Otherwise run one `old-head..new-head`
round on the files it names. Follow `AGENTS.md`'s SHA-freshness rules and PR
comment format. Fix real BLOCKER/MAJOR before merge; delta scope never waives
correctness, security or regression risk. MINOR/NIT do not force another round.

## Review economy

- **Context contract.** Every review/QA/design/UX brief carries the operator's
  request (verbatim or close paraphrase), intent, acceptance criteria, explicit
  non-goals, the prior-round ledger (round 2+) and one field line:
  `Worktree: <abs path> @ <sha> · Round: <N> · Scope: <prev>..<head> · Retest:
  <FAIL ids>|none`. Missing request: ask once via hub, or state assumed scope.
- **Scope.** Judge the diff against the stated request. Pre-existing defects not
  introduced or worsened are listed `out of scope — pre-existing` and never gate.
  No redesigns or extra features. No speculative findings: without a code path
  or repro a "might" is a lead, never BLOCKER/MAJOR.
- **Waves.** All applicable streams run round 1 on the SAME pushed head and
  report before remediation; ONE ledger, ONE remediation commit, ONE follow-up
  wave on that delta. Target <= 2 waves; a third needs a named open BLOCKER/MAJOR.
- **Minors.** Fold accepted MINOR/NIT into the majors' commit or defer them.
  Never a minors-only commit after all streams are terminal: it re-stales the head.
- **Freshness.** A delta round is `### Agent review — round <N>`, `Scope:
  <prev-head>..<head>`, by an independent reviewer. MECHANICAL rebase/merge of
  main (`scripts/rebase_proof.py`) keeps rounds fresh on its proof, and so does
  a harness-only delta (`scripts/harness_delta_proof.py`, above) — that is the
  ONLY extra exemption. A clean stream is NOT re-run when the delta misses its
  surface: a test-only, docs-only, CI-fix or comment-only commit after clean
  rounds gets at most a delta code check, no QA/design/UX round. `tests/` is
  product here, so a test change never takes the harness-only carve-out —
  weakening a test weakens the gate — and it still needs only that delta code
  check, not a numbered round. Executable code needs a delta round plus QA
  re-testing prior FAIL rows and the surfaces it moved; other rows stay
  `retained (<sha>)`.
- **Hygiene.** Rounds only on pushed heads; one comment per stream per round; no
  `re-pin`/`addendum` comments. Amend your own round (`gh api -X PATCH`) only
  for provenance/typos, before any remediation reply; never change `Scope:`.
- **Tone.** Collaborative: problem, evidence, suggested fix. No retraction
  theatre; disagree once, with evidence, then settle.
- **Stop.** Past 3 waves or ~2x review-driven diff growth, split follow-on scope
  into another PR; never an open BLOCKER/MAJOR.

## Model provenance

Use supported fallback and disclose the actual served model, including changes.
Preference is not an approval gate; never rewrite role/default pins or tools to
work around it. A missing required independent round blocks; never invent one.

## Rolling out role policy

Seeds update only unedited, install-recorded rows. Pilot via `agent op=show
<role>` plus a fresh `task(agent=<role>)`, not `load_seed`. Merge into local
prompts: back up, keep edits, security/approval policy, tools, model/effort;
read back.
