---
name: delivery
description: "Allocate bounded engineering delivery, brief so a child can land, preflight costly evidence, and converge review and QA without duplicating full-suite gates."
---

# Bounded delivery

Read this when allocating engineering work, planning costly evidence, or closing
review rounds. `AGENTS.md` remains the repository authority for isolation,
commands, security, merge and release gates. This policy changes allocation and
scope, not the depth of validation or the operator's approval requirements.
For this repository's public release copy, follow
`.github/RELEASE_NOTES_TEMPLATE.md` and the release steps in `AGENTS.md`.
The claim/bump PR's public-notes section and publication file follow that same
contract; internal gate evidence remains in separate PR comments, never in
user-facing release notes.

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
- Prefer metadata and narrow indexes before reading logs or artifacts. Follow
  a surprising first read one step further; cite the source of relayed results.
  Root recovery restores the minimal prerequisite; cross-repository hardening
  is a separate scope, not an incidental repair.

## Delegate so the child can land

Measured: the median first write/edit lands on call 29;
rediscovering a diagnosis you hold costs a child twice — budget, then the
resume brief.

- Briefs carry the change site (file:line), the failing behaviour, the
  acceptance check; "investigate and fix X" redoes your work.
- Coder-class children land early: first edit within ~10 tool calls; after ~40
  with nothing landed, return the precise blocker, not more exploration.
- A repeat "explored, nothing landed" gets a smaller brief — after ruling out
  a harness cut-off (cut children end named).

## Preflight before expensive dispatch

Inventory the actual loaded briefs and source winners first; measure raw and
loaded prompt budgets so clipping cannot masquerade as compliance.
Write one test plan before handing work to native, browser or hardware lanes:
changed surfaces and direct neighbors, commands, dependencies, source/binary
identity, isolation, expected outputs and at least one negative control. Confirm
that the compiled artifact contains the changed source and that the harness can
observe success AND failure. A source-only test is not native execution proof.
A setup failure stops costly dispatch: report the failed prerequisite and its
bounded repair, rather than sending reviewers to a broken rig. Repeat a costly
probe only for a material delta or new discriminating hypothesis; retire obsolete
probes. Use the harness
browser, never install a second browser stack; obey the isolated HOME/config,
CMUX stripping and exact-process-group teardown rules in `AGENTS.md`.

## Never use a live peer as the revoke test

A membership revocation test must run against a purpose-built disposable peer,
never a live peer. `member rm` burns the device id, bumps the epoch, wipes
credential copies and forces a re-key; it is not reversible by the agent, and
re-pairing may need a person at the joining console or a head address the peer
can actually reach. Removing a working peer can take its sessions and
uncommitted work offline even when no file is deleted.

- If the product cannot create a disposable peer, test revocation at the
  grant/token layer (revoke a share, expire or reject a token) and leave the
  live peer untouched.
- Before ANY membership-changing command — `member rm`, identity rotation,
  invite minting, epoch/re-key, security-group changes — preflight the peer:
  list its sessions, check every worktree's HEAD and `git status`, identify
  uncommitted or active work, and confirm a reachable head and a human/console
  path for re-joining. Get explicit operator approval naming the peer and record
  the preflight in the task receipt. Authorization to test is not authorization
  to delete.
- If the peer cannot be shown idle and committed, stop and ask; a revoke test
  that leaves the live peer intact is a better outcome than a correct test that
  strands a lane.

## Spend coverage once

Iterate with affected tests and lints, bounded and serial for heavy work. At the
terminal frozen head require full coverage once: when CI runs the same suite,
CI supplies that pass, not an additional mandatory local full suite. If CI
omits a required surface, run that missing coverage with a bounded local or QA
command and record it. A scoped pass is not a full pass; merge still waits for
the actual required checks and independent review/QA. CI replaces duplicate
automated coverage, never the independent real-path QA matrix. Resource holds forbid
hold-class local gates, not the evidence requirement. Read CI asynchronously
while review proceeds; investigate failures, including ones outside the slice.

## Converge, do not restart

Collect all initial code, QA, design and UX reports before ONE remediation
commit. On same-history round 2+, review the remediation delta and direct
regression neighbors, with the previous head as a negative comparison when
needed. Keep unchanged code/design/QA evidence when byte/source compatibility
is proved and the changed surface cannot invalidate it. A model-header or
provenance-only metadata correction is an addendum, not a reason to rerun the
suite. A new executable or user-visible change is not metadata.

A moved base is NOT a blind re-pin: prove patch compatibility with
`git range-diff` and byte-identical added/removed line sets, then run one
`old-head..new-head` convergence round for semantic conflicts in files upstream
also changed. If that proof fails, review and test the affected content. Follow
`AGENTS.md`'s SHA-freshness rules and the standing PR comment/remediation format.
A genuine late BLOCKER/MAJOR with a demonstrated code path stays admissible;
delta scope never waives correctness, security or regression risk.
Always fix real BLOCKER/MAJOR findings before merge; MINOR/NIT findings do not
force another gate round and must be answered with a reason if not fixed.

## Model provenance

Use supported fallback when available and disclose the actual served model,
including a mid-round change. A model preference is not a new approval gate;
never rewrite role/default model pins or tool permissions to work around it.
Inability to execute the required independent round remains a blocker, not a
reason to fabricate a verdict.

## Evidence and pilots

Structural prompt-budget tests cannot establish model behavior. Before shipping
policy changes, run fresh-agent pilots against the actually loaded role/team
boundary: normal bounded allocation, failed compiled/native preflight, genuine
destructive/access escalation, remediation-only scope with a real major, and
stable-ID ownership/material-only status. Keep prompts and actual responses in
session scratch, including negative cases and model provenance. Retain custom
security, release, roster and approval policy when patching live overrides;
announce edits, back up and read them back. Never force-sync away local additions.
