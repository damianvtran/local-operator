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
design, UX, security) across its full in-scope surface before remediation, not
just the first finding. Early BLOCKER alerts must say incomplete. Review runs
asynchronously with CI; merge still needs all gates.

Manager collects all reports into ONE deduplicated severity/evidence ledger,
triages conflicts and batches necessary clarifications before edits. Implementer
verifies claims, then fixes accepted IDs in ONE coordinated pass, answering each
with fixed SHA, rejection evidence, or deferral reason/location. Default ONE
batched commit; separate commits need justified independently revertible safety
or stream boundaries in the SAME round. No per-finding commit/review cycles.

Follow up on `<previous_reviewed_head>..<new_head>` plus direct regression
neighbors, never unchanged re-audits or MINOR/NIT loops. Late BLOCKER/MAJOR needs
a concrete code path/reproducer and changed/direct-neighbor scope. Target delta
and direct-regression tests; preserve independent QA/design/UX/security gates.
Retain evidence only with byte/source compatibility proof; backend response-shape
changes may need rendered proof. Model-header/provenance-only addenda do not reset
product/suites; executable/user-visible changes are not metadata.

After two unsuccessful cycles, freeze severity/evidence ledger, root-transition/
authority matrix (failing state, intended transition, authoritative owner/source)
and exit criteria BEFORE more patches. An unknown persisting twice escalates to
its owner. This checkpoint never waives unresolved BLOCKER/MAJOR or evidence gates.

A moved base is NOT a blind re-pin: post `scripts/rebase_proof.py`'s
`### Rebase proof` (range-diff `=`, byte-identical patch, no file overlap).
MECHANICAL keeps the last clean code and QA rounds; CI on the new head guards
cross-file breaks. Otherwise run one `old-head..new-head` round on the files
it names: if that proof fails, review and test the affected content. A test-
or docs-only commit after a clean round needs an independent
`<prev-head>..<head>` delta review; no QA unless a QA'd surface moved. Follow
`AGENTS.md`'s SHA-freshness rules and the standing PR comment format. Fix real BLOCKER/MAJOR before merge; delta scope never waives correctness,
security or regression risk. MINOR/NIT do not force another round.

## Model provenance

Use supported fallback and disclose the actual served model, including changes.
Preference is not an approval gate; never rewrite role/default pins or tools to
work around it. A missing required independent round blocks; never invent one.

## Evidence and pilots

Before shipping policy, run fresh-agent pilots at the loaded role/team boundary,
including a genuine destructive/access escalation. Keep prompts, actual responses
and model provenance in scratch. Preserve custom security/release/roster/approval
policy in live overrides: back up and read back; never force-sync away additions.
