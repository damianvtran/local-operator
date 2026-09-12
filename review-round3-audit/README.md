# Round 3 evidence — the AUDIT phase, and what a pasted notice now gets

Remediation round 2 for PR #993, after QA round 2's Q1 (major): shedding the
eight CARRIED notice copies also removed their ids from the hoisted suppression
set, so the four plain STORED rows a pre-stamp build wrote (journal lines
13952-13955, no `provider_payload` at all) came back into view in the attached
viewer's AUDIT phase. The context phase stayed clean; the mirror opened.

Everything below runs against a **byte-identical copy** of the operator's session
`835fbcafdc27` (APFS clone; the source is never opened for writing), in an
isolated `HOME` + `LOCAL_OPERATOR_CONFIG_DIR`, with no `CMUX_*` in the
environment.

## The two phases, measured with each tree's own rule

`scripts/audit_notice_census.py <session-dir>` replays both phases and counts, in
each: the user rows, the rows the TREE's rule identifies as notices, and — the
figure that is the symptom — the rows the phone fold actually PAINTS whose text
opens with a notice head. `census/phases-before-after.log` is the raw output:

| phase | round-2 head `9992708fc` | this head |
|---|---|---|
| context | painted **0** of 5 notice rows | painted **0** of 5 |
| audit | painted **4** of 8 switch-text rows | painted **0** of 8 |

The audit rows are the ones the attached viewer serves: `_capture_audit_window`
pages PRE-COMPACTION history by journal index, and its own docstring says it
mirrors `replay_entries(..., mode="audit")` — the function this census replays.

## The pixels

`frames/audit-before-round2-head.png` and `frames/audit-after-fixed.png` are the
same window of the audit-phase rows (the notices and their neighbours, chosen by
TEXT so both checkouts frame the same rows) presented through the assembled
application — real `OperatorApp`, `local_operator.tcss`, `screen.size ==
screen.virtual_size` on both, `css_path = ['local_operator.tcss']` in each
`.geometry.json`. Before: four `[model switch]` notices sit behind the user
gutter with the compaction marker and the conversation's own rows around them.
After: the same window with them gone.

**What these frames are not.** They are not a capture of the attached viewer's
own scroll into its audit pages: driving that headlessly to the audit cursor did
not converge in a bounded time here (the chain stalls ~36 rows pending — see
`census/attached-viewer-attempt.log`, 400 wheel-ups, `pending=36`), so the
attached path is evidenced at the ROW level above rather than in pixels. The rows
in the frames are the real audit-phase rows; only the act of scrolling to them is
not reproduced.

## What a pasted notice gets, stated plainly

A person who pastes a notice verbatim loses their DISPLAY row. The row is still in
the journal, still in the model's context, still in the export — only every human
surface drops it, because the audit phase serves stored rows whose sole surviving
evidence is the text. `is_harness_chrome` has always traded the same way for the
three continuation prompts, which a person can equally paste; the limit is now
pinned by tests on all three surfaces (`test_a_pasted_notice_is_hidden_on_display_
but_kept_in_the_transcript`, the D12 phone case, the panel case).

## The legacy pair, re-anchored (design round 2's D1)

`frames/legacy-before-anchored.png` and `frames/legacy-after-anchored.png` replace
the round-2 legacy pair, which had been landed at the same NUMERIC content offset
in two differently-anchored windows: after the notices were removed, that offset
pointed at the rows that used to sit above them, so the pair compared positions
rather than the change. Both frames now land on the same ROW — the operator's
own "Ok in that case then, let's continue. Fix the reported image issue…" prompt,
immediately above the batch — found by text in each tree, at content offsets 108
and 138 respectively. Same window, same anchor; the notices are the only
difference.
