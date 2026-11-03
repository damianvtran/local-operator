---
name: delivery
description: "Allocate bounded engineering delivery, brief so a child can land, preflight costly evidence, and converge review and QA without duplicating full-suite gates."
---

# Bounded delivery

For allocation, costly evidence and review convergence. AGENTS.md remains the
authority for isolation, commands, security, merge/release and approvals; this
changes scope/allocation, not validation depth. Public notes follow
`.github/RELEASE_NOTES_TEMPLATE.md` and AGENTS.md, including claim/bump PR and
publication file. Keep internal gate evidence out of user-facing notes.

## Allocate once, inspect narrowly

- A manager owns routine bounded technical allocations inside the authorized
  task. Choose a reversible implementation and a bounded test lease; do not
  escalate ordinary library, layout, test or lane choices. Escalate genuine
  access/security decisions, destructive actions, operator preferences and
  material scope changes. Authorization to fix is not authorization to delete.
- Give each slice one owner identified by stable session ID, scope, acceptance
  criteria and a durable receipt (project record or linked task/PR). Address the
  owner directly; do not broadcast a status hunt. One confirmed durable ACK
  means no unchanged resend or repeated approval request. Update only material deltas;
  verify completion from primary evidence, not an agent's assertion.
- Prefer metadata/narrow indexes before logs. Verify surprising first reads one
  step further; cite relayed sources. Root recovery restores only prerequisites;
  cross-repository hardening is separate scope.

## Delegate so the child can land

Briefs name file:line, failure and acceptance check; do not delegate rediscovery.
Coder children edit within ~10 calls; after ~40 without an edit, return the
blocker. A repeat "explored, nothing landed" needs a smaller brief after checking
for a harness cut-off (cut children end named).

## Preflight before expensive dispatch

Measure raw/loaded prompt budgets and source winners so clipping cannot fake
compliance. Before native/browser/hardware dispatch, plan changed surfaces,
commands, dependencies, source/binary identity, isolation, expected outputs and a
negative control. Verify compiled source identity and success/failure detection;
source-only checks are not native proof. Failed setup stops dispatch: report the
prerequisite and bounded repair. Use the harness browser, no second stack; follow
AGENTS.md's HOME/config, CMUX stripping and exact-process-group teardown rules.

## Worktrees

Prefer `<repo>/.worktrees/<name>` sharing primary deps; sibling `../<repo>-<topic>` only with cloned deps.
Never scratchpads, /tmp, `$HOME` top level. `git worktree remove` (clean only) once the PR merges or
the lane closes; `scripts/worktree_janitor.py` lists stragglers.

## Never use a live peer as the revoke test

Never remove a live peer to test revocation. Use a disposable peer; if none can
be created, exercise revocation at the grant/token layer instead. Before any
membership change, list the peer's sessions and worktree state, confirm a
reachable head and console path, get explicit operator approval naming the
peer, and record it. Authorization to test is not authorization to delete.

## Spend coverage once

Iterate with targeted tests/lints; heavy runs bounded/serial. Full coverage once
at the terminal head: when CI runs the same suite, use it, not a duplicate local
run. Run omitted required surfaces bounded locally/in QA and record them. Scoped
is not full coverage. Merge requires actual checks and independent review/QA;
CI never replaces independent real-path QA. Resource holds forbid hold-class
runs, not evidence. Review CI asynchronously; investigate even out-of-slice fails.

## Converge, do not restart

Complete every applicable initial stream (code, independent real-path QA,
design, UX, security) across its full in-scope surface before remediation.
Early BLOCKER alerts must say incomplete. Review runs asynchronously with CI.

Manager collects all reports into ONE deduplicated severity/evidence ledger,
triages conflicts and batches necessary clarifications before edits. Implementer
verifies claims, then fixes accepted IDs in ONE coordinated pass, answering each
with fixed SHA, rejection evidence, or deferral reason/location. Default ONE
batched commit; separate commits need justified independently revertible safety
or stream boundaries in the SAME round. No per-finding commit/review cycles.

Follow up on `<previous_reviewed_head>..<new_head>` plus direct regression
neighbors, never unchanged re-audits. Late BLOCKER/MAJOR needs a concrete code
path/reproducer in changed/direct-neighbor scope. Retain evidence only with byte/source compatibility proof; backend response-shape
changes may need rendered proof. Model-header/provenance-only addenda do not reset
product/suites; executable/user-visible changes are not metadata.

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
  non-goals, base..head and, round 2+, the prior-round ledger. Missing request:
  ask once via hub, or state the assumed scope atop the round.
- **Scope.** Judge the diff against the stated request. Pre-existing defects not
  introduced or worsened are listed `out of scope — pre-existing` and never gate.
  No redesigns or extra features. No speculative findings: without a code path
  or repro a "might" is a lead, never BLOCKER/MAJOR.
- **Waves.** All applicable streams run round 1 on the SAME pushed head and
  report before remediation; ONE ledger, ONE remediation commit, ONE follow-up
  wave on that delta. Target <= 2 waves; a third needs a named open BLOCKER/MAJOR.
- **Minors.** Fold accepted MINOR/NIT into the same commit as the majors, or
  defer them in the remediation comment. Never a commit just for minors after
  all streams are terminal: that re-stales the head.
- **Freshness.** A delta round is `### Agent review — round <N>`, `Scope:
  <prev-head>..<head>`, by an independent reviewer. MECHANICAL rebase/merge of
  main (`scripts/rebase_proof.py`) keeps rounds fresh on its proof; docs,
  comments or tests need only a delta round. Executable code, minor fixes
  included, needs a delta round plus QA if a QA'd surface moved.
- **Hygiene.** Rounds only on pushed heads; one comment per stream per round; no
  `re-pin`/`addendum` comments. Amend your own round (`gh api -X PATCH`) only
  for provenance/typos, before any remediation reply; never change `Scope:`.
- **Tone.** Collaborative: state the problem, evidence and a suggested fix; note
  what works in a line. No retraction/correction theatre; disagree once, with
  evidence, then settle.
- **Stop.** Past 3 waves, or review-driven growth past ~2x the original diff,
  the manager splits follow-on scope into a separate PR; an open BLOCKER/MAJOR
  is never split out of the PR being merged.

## Model provenance

Use supported fallback and disclose the actual served model, including changes.
Preference is not an approval gate; never rewrite role/default pins or tools to
work around it. A missing required independent round blocks; never invent one.

## Rolling out role policy

Seeds update only unedited, install-recorded rows. Pilot via `agent op=show
<role>` plus a fresh `task(agent=<role>)`, not `load_seed`: it reads this guide,
keeps scope, escalates a destructive step. Keep prompts, responses, served model
in scratch. Merge into local prompts: back up, keep edits, security/approval
policy, tools, model/effort; read back.
