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
  task: a reversible implementation, a bounded test lease, ordinary library,
  layout, test or lane choices. Escalate genuine access/security decisions,
  destructive actions, operator preferences and material scope changes.
  Authorization to fix is not authorization to delete.
- Give each slice one owner (stable session ID), scope, acceptance criteria and
  a durable receipt (project or linked task/PR). Address the owner directly; no
  status hunts or resends after an ACK. Update only material deltas; verify
  completion from primary evidence.
- Prefer metadata/narrow indexes before logs. Verify surprising first reads one
  step further; cite relayed sources. Root recovery restores only prerequisites;
  cross-repository hardening is separate scope.

## Delegate so the child can land

Briefs name file:line, failure and acceptance check; do not delegate rediscovery.
Coder children edit within ~10 calls; after ~40 without an edit, return the
blocker. A repeat "explored, nothing landed" needs a smaller brief after checking
for a harness cut-off (cut children end named).

## Budgets

Classify each lane at its start; record class and budget in the project.

| Class | hotfix | fix/patch | feature | very significant | codebase refactor |
|---|---|---|---|---|---|
| Budget | <=30 min | <2 h | <=3 h impl. | 3-4 h | 8 h+, rare, reason stated |

At budget, report elapsed, critical path, what remains and options to the
delegator, then continue the critical path unless redirected; re-report at 2x;
never stall awaiting an ACK.

## Preflight before expensive dispatch

Before native/browser/hardware dispatch, plan changed surfaces, commands,
dependencies, compiled source identity, isolation, expected outputs, success/
failure detection and a negative control; source-only checks are not native
proof. Failed setup stops dispatch: report the prerequisite and bounded repair.
Harness browser only; AGENTS.md owns HOME/config isolation, CMUX stripping and
exact-process-group teardown.

**Budget costly evidence.** A native, simulator, device or hosted E2E lane gets
3 failed-or-cancelled attempts OR 6 h from first dispatch (local-operator-mobile
PR #156: 36 failures, 118 min median cadence); harness-lost attempts count.
Spent, stop and escalate with each option's consequence: merge on unit/contract
tests plus manual device QA; split the harness out; drop the row; or extend by a
stated amount for a stated reason. No new mechanism or retry meanwhile. A lane
never succeeding in 3 attempts is a harness problem until shown otherwise (41.4 h
of failures came after the last product commit).

**Measure before you cap; name the invariant.** No new timeout, cap, threshold or
retry count without the distribution it bounds (n, median, max, source) and a
setting above the observed max; every mechanism (cache, reuse, skip, fixture,
gate) and shortcut names its invariant and the check proving it — no check, no
shortcut (PR #156 capped hierarchy reads at 20 s over a measured 66–173 s).

## Worktrees

The manager (or lead coder) owns ONE PR worktree in `<repo>/.worktrees/` from
latest origin/main, briefs its absolute path and SHA, and freezes the head while
readers work. Reviewer/QA/UX/design/copy/i18n reuse it read-only every round
at the briefed head (targeted tests, shared deps): no clone, install or new
worktree, and no full-tree copies into scratch. Never move it off that head;
base/before reads use `git show <base>:<path>` or `git archive` into scratch. Python:
`PYTHONPATH=<worktree>` with the primary `.venv`; `python -c 'import
local_operator;print(local_operator.__file__)'` must print a worktree path.
Exception, with a manager lease: a conflicting commit author or an isolated
mutable build collision gets a detached-SHA worktree on shared deps, bounded and
creator-removed, with isolated HOME/config. Remove clean
worktrees on merge/close; the manager verifies (`scripts/worktree_janitor.py`).

## Never use a live peer as the revoke test

Use a disposable peer, else exercise revocation at the grant/token layer. Before any
membership change, list the peer's sessions and worktree state, confirm a
reachable head and console path, get explicit operator approval naming the
peer, and record it. Authorization to test is not authorization to delete.

## Spend coverage once

**Local test lease.** The coder runs only the test files covering changed code
and lints changed files, each run bounded; reviewers and QA run the same tests
covering the diff. Whole-directory, whole-suite and `make check-changed` runs are
CI's when CI runs the same suite (#3575: ~3.7 h of rc=124 local pytest).

**CI is concurrent** with review/QA on every push. Read it once per wave,
non-blocking, folding failures into that wave's remediation. No `--watch`, no
waiting before the merge gate, and never sleep-poll it mid-cycle; at the gate
required checks must be green (#3575 read CI after 3 clean rounds: 4 failures,
+2.3 h). Run omitted required surfaces bounded and record them; CI never
replaces real-path QA. Resource holds forbid hold-class runs, not evidence.
Investigate even out-of-slice fails.

## Converge, do not restart

**Converge the base once, at the merge gate.** Do not merge or rebase the base
into a PR mid-review, except when GitHub reports DIRTY and a required run cannot
start — prefer `workflow_dispatch` on the branch ref (a conflicting PR's
`pull_request` workflow gets no merge ref).
At the gate: converge, rebase proof, ONE round on files upstream also touched, CI
(PR #156 merged main 13 times).

**Harness-only delta: no numbered round.** After a clean round on head R, delta
R..H takes no `### Agent review — round N` when
`scripts/harness_delta_proof.py R H` exits 0 (same merge-base,
`.github/harness-paths` present and unchanged, every changed path under a
`harness:` glob): *no byte the shipping pipeline consumes differs between R and H*.
Instead, ONE `### Harness delta check — R..H` at the merge gate by an independent
reviewer reading for gate weakening (loosened assertions, skips, widened
tolerances, `continue-on-error`, a probe that cannot fail) and harness changes
running with release triggers, secrets, signing or publishing; QA retained. Anything
unreadable, unresolvable or unclassifiable is a round or exit 2, never a pass: a
reachability classifier made safe by failing closed.

**Split when the harness outgrows the feature.** At each review request compare
added lines under harness globs with all others (`--split` prints both); harness
above product moves the harness to its own PR (review plus its own negative
control, no product QA). PR #156 crossed at round 1 (2,787 vs 2,764 lines).

Manager collects all reports into ONE deduplicated severity/evidence ledger and
batches clarifications before edits. Implementer verifies claims, then fixes
accepted IDs in ONE pass, answering each with fixed SHA, rejection evidence, or
deferral reason. Default ONE commit, no per-finding cycles. Brief
a FRESH coder child with the ledger; never resume a long-idle big-context coder
(#3565: 13 resumes at 226k context, $17.8 cold-cache). Every fix ships a
regression test where the defect is testable; a fix in a bypass-prone area
(parsers, denylists) checks its own new code for the same class before pushing.

Follow up on `<prev-head>..<head>` plus direct regression
neighbors; a late BLOCKER/MAJOR needs a code path/repro there.
Retain evidence only with
byte/source compatibility proof (response-shape changes may need rendered proof).

After two unsuccessful cycles, freeze the ledger, a root-transition/authority
matrix (failing state, intended transition, owner) and exit criteria first. An
unknown persisting twice escalates to its owner. This never waives BLOCKER/MAJOR
or evidence gates.

A moved base (rebase OR merge of main) is NOT a blind re-pin: post
`scripts/rebase_proof.py`'s `### Rebase proof` (range-diff `=`, byte-identical
patch, no file overlap). MECHANICAL keeps the last clean code and QA rounds; CI
on the new head guards cross-file breaks. Otherwise run one `old-head..new-head`
round on the files it names, per `AGENTS.md`'s SHA-freshness rules. Delta scope
never waives correctness, security or regression risk.

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
- **Waves.** Every applicable stream (code, real-path QA, design, UX, copy,
  security) starts round 1 on the SAME pushed head and completes its full
  in-scope surface before remediation (a round-2 joiner forced a round 3:
  +70 min); early BLOCKER alerts say incomplete. ONE ledger, ONE
  remediation commit, ONE follow-up wave. Target <= 2 waves; a third needs a
  named open BLOCKER/MAJOR.
- **Clean is terminal.** Fold accepted MINOR/NIT into the majors' commit or
  defer them in-thread. Once all streams are clean, a new commit lands only for
  a red required check, a real BLOCKER/MAJOR or a base-merge/harness-only/
  test-docs-only path below (own rules); never minors folded into new code
  (#3565: clean at round 7, rounds 8-12 reopened so, +3.9 h).
- **Non-shipping artifacts** (not packaged or run: design notes, PR evidence,
  unpackaged docs) freeze once no BLOCKER/MAJOR is open; minors accepted,
  further rounds only for one (a note took 13 rounds).
- **Freshness.** A delta round is `### Agent review — round <N>`, `Scope:
  <prev-head>..<head>`, by an independent reviewer. MECHANICAL rebase/merge of
  main (`scripts/rebase_proof.py`) keeps rounds fresh on its proof, and so does
  a harness-only delta (`scripts/harness_delta_proof.py`, above) — the ONLY
  extra exemption. A clean stream is NOT re-run when the delta misses its
  surface: a test-only, docs-only, CI-fix or comment-only commit after clean
  rounds gets at most a delta code check, no QA/design/UX round. `tests/` is
  product here (never harness-only), yet gets just that delta code check.
  Executable code needs a delta round plus QA re-testing prior FAIL rows and the
  surfaces it moved; other rows stay `retained (<sha>)`.
- **Hygiene.** Rounds only on pushed heads; one comment per stream per round; no
  `re-pin`/`addendum` comments. Amend your own round (`gh api -X PATCH`) only
  for provenance/typos, before any remediation reply; never change `Scope:`.
- **Tone.** Collaborative: problem, evidence, suggested fix. Disagree once, with
  evidence, then settle.
- **Stop.** Past 3 waves or ~2x review-driven diff growth, split follow-on scope
  into another PR; never an open BLOCKER/MAJOR.

## Model provenance

Use supported fallback and disclose the actual served model, including changes.
Preference is not an approval gate; never rewrite role/default pins or tools to
work around it. A missing required independent round blocks; never invent one.

## Rolling out role policy

Seeds update only unedited, install-recorded rows. Pilot via `agent op=show
<role>` plus a fresh `task(agent=<role>)`, not `load_seed`. Merge into local
prompts: back up, keep edits, policy, tools, model/effort; read back.
