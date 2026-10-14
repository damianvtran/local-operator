---
name: verification-before-completion
description: "Proving work works before claiming it does: exercise the real path, read the actual output, gather fresh evidence per claim, and say what remains unverified. Use before reporting done, handing off, or posting results."
---

# Verification before completion

No claim ships without evidence you produced yourself, against the current state of the work.

## Claims need evidence

| Claim | Counts as evidence | Does not count |
|---|---|---|
| "The fix works" | Rerun of the original reproduction, output shown | "Should work now" reasoning |
| "Tests pass" | The test command and its actual output | A remembered pass from before the change |
| "The endpoint works" | Real request, real response (status, body) | A unit test that mocks the call |
| "The UI shows X" | A rendered frame you captured and looked at | Reading the component source |
| "It fails safely" | The failure case exercised (unauthorized, empty, invalid) | Presence of a try/catch |
| "Deployed" | Success output plus a request against the new version | The commit that should have triggered it |
| "The subagent finished" | Its output read, its central claim spot-checked | The task marked complete |

## Run it, don't say it

- Exercise the real path: the actual command, the actual endpoint, the actual page.
- Read the output. A non-zero exit or an error string is not success; read what came back.
- Freshness: evidence must come from the current state. A run from before the last edit proves nothing about the edit.
- One claim, one piece of evidence. Where you cite a number or status, derive it or name its source.

## When you cannot verify

Say so, in this shape: `verified: X (evidence); unverified: Y; reason: Z`. Never let an unverified item pass silently as done. An explicit "unverified" is an honest result; an implied pass is not.

## Verifying other people's reports

The same bar applies to summaries, subagent reports, and CI statuses:

- A "done" from another agent is a lead, not evidence. Spot-check the load-bearing claim, cheaply.
- A summary that cites its sources can be trusted one level deeper only where the source exists and says what the summary claims.

## Red flags

- "It compiles" offered as proof of behaviour.
- Quoting a previous test run instead of running the suite now.
- Screenshots or output from before the change under review.
- A long list of "should" statements.
- Reporting a percentage, count, or status from memory.
- Tests passing treated as the feature working; a green suite is the code doing what the tests expect, and the tests may not cover the change.

## Checklist before your final message

- [ ] Every claim in the message traces to output you produced now.
- [ ] The original symptom was re-exercised, not just "the new test".
- [ ] At least one failure case was tried, not only the happy path.
- [ ] Anything unchecked is listed as `unverified` with its reason.
