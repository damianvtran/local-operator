---
name: code-review
description: "Reviewing code for defects: diff-first reading, severity classification, evidence over inference, finding caps, and terminal verdicts. Use when reviewing a diff, PR, or patch, or auditing someone else's work."
---

# Code review

The same practice for any agent: reviewing a diff, a patch, or another agent's work.

## Read the diff first

- `git diff <base>..<head>` (or the PR's diff) IS the review. Read the tree only where the diff raises a question you can state.
- Open a file when you can name the question it answers; re-reading files you already read is waste.
- Review what changed plus what it touches: callers, shared state, and the tests near the change.

## Classify every finding

| Severity | Means |
|---|---|
| BLOCKER | Wrong, unsafe, or loses data. Ship it and something breaks. |
| MAJOR | A real defect or missing case that will bite, but not stop-ship. |
| MINOR | Correctness-preserving improvement. |
| NIT | Style, naming, wording. |

- Caps: at most 5 MINOR and 5 NIT per round, the highest-value ones. A long tail of nits buries the blockers and costs a remediation round to answer.
- Severity is about consequence, not effort to fix. Do not inflate; do not deflate to avoid an awkward round.

## Evidence over inference

- Every finding carries a location: file, line, and the code path. Without one it is a lead, not a finding.
- Reproduce cheaply where possible: the targeted test, the query, a tiny probe. A finding you reproduced outranks one you reasoned to.
- Where you cite a number or status, derive it or name its source. A value read from a summary is a lead.
- Say plainly when nothing blocks. Padding a report to look thorough is a failure mode; so is agreeing to be agreeable.

## Remediation rounds (round 2 and later)

- Scope to the delta: `git diff <previous_reviewed_head>..<current_head>`. Verify the fixes; do not re-audit unchanged, previously approved files.
- No new out-of-scope findings; note them as follow-ups instead.
- A fix that unexpectedly touches more is in scope again: check its blast radius.

## Verdict

- End with a terminal verdict: with no BLOCKER and no MAJOR left, the verdict is `clean` and you state the round is terminal; remaining minors and nits are follow-ups, not a reason for another round.
- Language is precise: "blocked on X", "clean with follow-ups". Never "LGTM" with no checks behind it.

## Red flags

- Reviewing the tree instead of the diff.
- Speculative findings ("there might be a race") with no code path named.
- Silent scope creep into unrelated code.
- Style preferences dressed as defects.
- A finding the author cannot act on because it names no change.
