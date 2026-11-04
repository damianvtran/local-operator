# es / tui.small — round 4 (w6a4): closing audit of S1

Auditor: i18n-auditor (lopdev), session 65ddc2e6e89d, model anthropic/claude-sonnet-5-5 — no provider failure, fallback or model switch occurred in this run.
Audited revision: `215c87822c7c3d2b3e52a9acde39b41a570851d9` (PR #3188 head). Remediation commit: `3209592c2b1`.
Verdict: **passed** (0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT open). S1 **closed**.

Scope is deliberately narrow: the 137 values were passed in w6a1 and w6a2 and were not re-audited. This round answers one question — is S1 (w6a3) genuinely closed on the head?

## 1. Bytes and value (verified, not assumed)
- `es/tui.small.json` sha256 `e374959d70bb379a37e3f02e8c69edeea168b7edbd11107e4131f41fe116b9af` — equals the round-2 and w6a3 hash. Git blob is identical at `3209592c2b1` and `215c87822c7` (`cb094e1eeceb…`).
- 137 keys, same key set as English.
- `tui.small.goal_panel.done_suffix` = `"  — hecho"`: two ASCII spaces, U+2014, space, `hecho`; 9 cells (English 8). Matches the sidecar rule the w6a3 finding cited, so the catalogue was correctly left alone.

## 2. S1 closed in code — driven through the real `build_goal_body`
Locale `es`, status `done`, a 252-char clipped goal, content box = terminal columns − `PANEL_PADDING_CELLS` (4), rendered through a Rich `Console` of exactly that width (the same wrapping the card suffers):

| cols | fixed code (HEAD) | pre-fix `4912975a4b5` (control) |
|---|---|---|
| 60 | 1 line, ends `… — hecho`, max 56/56 | 2 lines, 2nd line `hecho` (orphan) |
| 80 | 1 line, ends `… — hecho`, max 76/76 | 2 lines, 2nd line `hecho` (orphan) |
| 100 | 1 line, ends `… — hecho`, max 96/96 | 2 lines, 2nd line `hecho` (orphan) |

- Control: the pre-fix source came from `git show 4912975a4b5:…goal_panel.py` into session scratch and was loaded under another module name; nothing was written into the worktree. The probe fails there at all three widths, so it discriminates.
- Sweep: 3 widths × 150 goal lengths. Fixed code: 0 orphans, 0 overflowing lines. Pre-fix: 282 orphaning cases. So the bug is not an artefact of the one goal string.
- English output is byte-identical pre vs fixed (the `max(_GOAL_TAG_CELLS, …)` floor keeps 8), so the fix cannot have moved English.
- The new unit test `test_a_longer_localised_done_tag_does_not_orphan_itself_on_a_second_line` runs and passes (1 passed) in an isolated HOME.
- The fix itself: `done_suffix = _t(_x.goal_panel_done_suffix()) if done else ""`; `tag_cells = max(_GOAL_TAG_CELLS, cell_len(done_suffix)) if done else 0`. It measures the string actually appended, so it is also right for any other locale, not just Spanish.

## 3. Other code-fixed columns (read-only look, goal_panel / info_panel / network_panel / wake_panel / session_picker)
Method: listed every tui.small value in those five panels whose es width exceeds English, then read each consumer.
- `goal_panel`: history rows budget from `len(entry_status)` + stamp (`_HISTORY_STAMP_CELLS`), and the status words are wire values, not catalogue strings. `settled_heading`, `none_yet`, `judge_run`, `more_settled*` are appended without a subtraction budget — they are flow text that may wrap, not a clip budget with a fixed English reserve. No second constant-vs-locale mismatch.
- `wake_panel`: headers and rows are `Text(no_wrap=True, overflow="ellipsis")`, so wider Spanish titles truncate, never wrap. No fixed reserve for the title.
- `network_panel`: fixed columns (`{mark:7}`, device id `:12`, role `:5`, `_ID_CELLS`) hold wire values, not catalogue words. Localised tags (`stale_tag`, `suspect_tag`, `trust_field`) are appended after, no subtracted budget.
- `info_panel`: `busy_word`, `needs_pending`, `not_answering`, `last_heartbeat_ago` go into a `metas` ladder that is built from the strings (measured, with a shed ladder), not a constant.
- `session_picker`: fixed columns are `AGE_CELLS`, `STATE_COL_CELLS`, `GUTTER_CELLS`; the catalogue strings sit in the free message line / key row and are truncated by `truncate_cells`.
- Observation, not a finding: at the 46-column panel floor (content 42) the Spanish empty-state hint `sin objetivo — /goal <text> para definir uno` (44 cells) wraps to a second line (`uno`). It is flow text, the card counts wrapped lines, the hint row stays present, and English fits. Not a code-fixed column; not raised.
- Honest limit: this is 15 minutes of reading plus a width table, not a render of every panel at every width. I found no other value in a code-fixed column; I cannot prove a negative beyond that.

## 4. Refresh-merge check
`215c87822c7` has parents `3209592c2b1` and `0034441080b`. Against `3209592c2b1`, the blobs of `es/tui.small.json`, `es/tui.analytics.json`, `goal_panel.py`, `test_goal_panel.py` and `es.md` are identical. The only ledger change is one appended `cli.team_bridge` row (+20 lines), unrelated to tui.small. The tui.small ledger row is unchanged (passed, source `16d49c4ba2a5…`).

## Findings
None open. S1 — closed by `3209592c2b1` (code), confirmed on `215c87822c7`.

Evidence: `checks.py` (incl. negative control), `audit.log` (captured commands/output), `record.json`.
