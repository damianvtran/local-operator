# Round 2 evidence — the LEGACY notice copies, and the receipts

Remediation round 1 for PR #993, after QA's Q1 (major): the operator's own
session still showed the complaint, because eight of the notices it carries were
written BEFORE the `harness_injected` stamp existed and a compaction pass had
lifted them into its marker's preserved block as "user turns". The stamped-row
fix cannot see them.

All of this runs against a **byte-identical copy** of the operator's session
`835fbcafdc27` (`cp -c`, APFS clone; the source is never opened for writing), in
an isolated `HOME` + `LOCAL_OPERATOR_CONFIG_DIR`, with no `CMUX_*` in the
environment. The frames come from the assembled app: a real `/resume` through the
app's own path (`_resume_session`) followed by the reader's scroll-up gesture.

## What the frames show

| frame | tree | what it is |
|---|---|---|
| `frames/legacy-before-round1-head.png` | `b8d98b2f1` (PR head before this round) | the reported rows: four `[model switch]` notices painted behind the user gutter in the operator's own conversation, at content offset 116 |
| `frames/legacy-after-fixed.png` | this round's head | the same offset: the notices are gone, the operator's real prompts are there |
| `frames/stamped-before-base.png` | `17138cf20` (this PR's base) | the four STAMPED leaked rows, painted as user bubbles — the original report's shape |
| `frames/stamped-after-fixed.png` | this round's head | those rows mount nothing; the fallover receipt and the band are unchanged |

## What the app was asked, and what it answered

`scripts/legacy_notice_resume_shot.py` prints a census after every page-back —
painted user rows, and how many of them are harness notices (the switch notice,
or the elision notice). `census/` holds the raw logs:

| page-back depth | base `17138cf20` | round-1 head `b8d98b2f1` | this head |
|---|---|---|---|
| 4 | 4 notice rows | 0 | 0 |
| 5 | 4 | 8 | 0 |
| 6 | 12 | 8 | 0 |
| 7 | 12 | 8 | 0 |

The round-1 head's 8 are the eight carried copies; the base adds the four stamped
ones on top. On this head the number is zero at every depth.

## The replay itself, measured

`census/fixed-head-replay-probe.log` is the same session read through
`build_llm_history` and `display_window` (the row-level numbers behind the
frames):

| reading | base | round-1 head | this head |
|---|---|---|---|
| context replay rows | 310 | 310 | 302 |
| rows whose text is a switch notice | 12 (4 stamped + 8 carried) | 12 | 4 (all stamped, hidden by the fold) |
| `opener_text` | the elision notice | the elision notice | the operator's first real prompt |
| `theme_turn_count` | 170 | 170 | 157 |
| display pages | 175 | 175 | 175 |

`total_message_count` is unchanged from the canonical replay's row count, which
is the invariant the reader's paging is checked against.

## The receipt, stated correctly

The fallover receipt in the stamped pair is the `NoticeEvent`
`configure.py::_on_route_change` emits (`"<reason> — falling back to <selector>"`,
with `_on_route_settle`'s `"back to <primary>"` on recovery) — **not** a retry
notice: nothing in this tree raises `RetryStartEvent`, and the previous revision
of this script posted that message by hand, which is corrected here. It is also
the account of the event only **while the session is live**: reopening the
conversation carries no trace of it by design.

## Geometry

Both stamped frames and both legacy frames report
`css_path = ['local_operator.tcss']` and `screen.virtual_size` equal to
`screen.size` (118x44 for the stamped pair, 118x42 for the legacy pair), so
neither the stylesheet nor a scrollbar changed between any pair — the
`.geometry.json` beside each frame holds the full readings.
