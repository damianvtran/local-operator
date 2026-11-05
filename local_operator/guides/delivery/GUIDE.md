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
  main (`scripts/rebase_proof.py`) keeps rounds fresh on its proof. A clean
  stream is NOT re-run when the delta misses its surface: a test-only, CI-fix or
  comment-only commit after clean rounds gets at most a delta code check, no
  QA/design/UX round. Executable code needs a delta round plus QA re-testing prior
  FAIL rows and the surfaces it moved; other rows stay `retained (<sha>)`.
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
