---
name: responding-to-review
description: "Answering review findings: verify each claim before implementing, no performative agreement, per-item fix/reject/defer with evidence, one batched remediation round. Use when responding to review or remediation rounds."
---

# Responding to review

Review feedback is input, not instruction. Verify before implementing; answer per item; ship one batched round.

## Per finding, in order

1. **Verify the claim.** Reproduce it, or read the code path it names. A finding can be wrong, and implementing a wrong fix adds a bug. If the finding is about evidence (a run, a frame, a number), the fix is to regenerate the evidence, not to edit the sentence.
2. **Decide**: fix, reject, or defer.
3. **Answer with the decision and its support.** No performative agreement ("Great catch!"). Say what happens to the finding; agreement without verification commits you to work you have not assessed.

## Clarify all first

- Read the whole round before editing. If an item is genuinely unclear (you cannot state the change it asks for), ask once, for all unclear items together, before the remediation edits. One batch of questions beats three round trips.

## One batched round

- Collect all accepted fixes into ONE remediation pass. Do not commit-and-reply per finding.
- Group by stream where the project does (code, tests, docs); keep it one round.
- Where several reviewers ran, answer each report's items in one combined response pass.

## The reply format

For each item, in the thread where the round was posted:

- `fixed: <what changed> (<commit SHA>)`
- `rejected: <why, with the check that shows it>`
- `deferred: <where it is recorded and why it cannot be now>`

Never mark fixed without the commit; never leave an item unanswered. "Not addressed" with no reason is a rejection nobody can audit.

## When you disagree

- Say so with the evidence: the test output, the code path, the spec line. Silence is the worst option; fix nothing, say nothing, and the item stays open forever.
- Once the round resolves, let it go. Re-litigating a settled item in the next round burns a round for everyone.

## Red flags

- Implementing every finding unverified because "the reviewer is usually right".
- Arguing from preference with no check either way.
- Marking items fixed in a summary while the code is unchanged.
- One comment per finding instead of one batched response.
- A "fixed" whose commit does not contain the fix.
