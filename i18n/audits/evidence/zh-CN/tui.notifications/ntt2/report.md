# Simplified Chinese audit — tui.notifications — ROUND 2

Auditor: session 0603506819c6 · model radient/auto (switch trail below) · independent of the
translator and of the coder.
Locale: zh-CN · Namespace: tui.notifications · Audit id: **ntt2** · Date: 2026-10-25
Round 1: `ntt1` (verdict not passed — one major, `N1`, a renderer defect).

**Verdict: PASSED — 72/72 keys read; N1 resolved and re-measured; no blocker, no major.**
The copy is byte-identical to round 1 (target sha256 unchanged), so every copy-level result
stands on the same bytes; what changed is the renderer, and it now paints every row exactly.

## Pins

| Input | Value | Check |
|---|---|---|
| Code + English source | commit `8976104b83b1f4af2fa6014ae5c73d9bbcdebc0a` (`fix(notifications): isolate in-sentence commands and reserve the TUI's columns by cells`) | exported with `git archive` into the auditor's scratchpad |
| English source | sha256 `b415bc703840afabc1c3ecaeecdad24a84d9a6ab3367833cad4f17f0e0e70b68` | unchanged from ntt1 |
| Context sidecar | sha256 `0a9ee26060dcc5a5dc44c784fcdff2d3b64c6457899d7b3c382c0c7e32d85907` | unchanged from ntt1 |
| Target (untracked) | sha256 `cd60e55f3d532f44848fe44a88ee87b879192f1b5149545526b7ded829768156` | unchanged from ntt1, and re-checked against the worktree after the run |
| Renderer | sha256 `64ca3084716c99b33823a1123efe0879764af1573a464b3b8a092057b98cad72` | differs from the pre-fix module, as intended |

The commit touches `local_operator/tui/{notifications_render.py,notifications_model.py,
widgets/notifications_view.py,widgets/session_sidebar.py,local_operator.tcss}` and
`tests/unit/tui/test_notifications_page.py`. **It contains no catalogue file** — the claim that
the zh copy is untouched is verified by inspection of the diff and by the identical hash.

## N1 — the measurement, executed on both trees

Same script, same catalogue, same 78-cell list width (what `_paint` hands `row_line` for an
80-column window), run once against the pre-fix tree and once against the fix.

| # | Case | Before (cells) | Δ | After (cells) | Δ | Exact = 78 |
|---|---|---|---|---|---|---|
| (a) | zh-CN just-now row, ASCII conversation title | **80** | +2 | **78** | 0 | yes |
| (b) | zh-CN just-now row, Chinese conversation title | **84** | +6 | **78** | 0 | yes |
| (c) | zh-CN 2h-old row, ASCII title | 78 | 0 | 78 | 0 | yes |
| (d) | **English** just-now row, Chinese conversation title | **82** | +4 | **78** | 0 | yes |
| (e) | control: normal en row, long ASCII title | 78 | 0 | 78 | 0 | yes |
| (f) | control: 30-char ASCII conversation title | 78 | 0 | 78 | 0 | yes |
| (g) | zh-CN just-now, CJK title, high + repeat chips | **84** | +6 | **78** | 0 | yes |

`cells-78` is read directly from `cell_len(row_line(...).plain)` — see `audit.log`. Round 1's
numbers reproduce exactly (80/84/78/82), so the before column is not a re-description: it is the
same bug measured on the old bytes.

**Structural proof (fail-closed).** A second verifier asserts properties that do not name the
fix: total cells == width; the last 5 cells are exactly the right-aligned clock from
`row_time`; the 12 cells before the 1-cell separator are the right-aligned
`truncate_cells(source_name, 12)`; a source longer than 12 cells is truncated, never overrun.

| Tree | Result |
|---|---|
| pre-fix `d25392f3553` | **4 of 7 cases FAIL** (a, b, d, g — the CJK/just-now rows), `ALL_PASS=False` |
| post-fix `8976104b83b` | **7 of 7 PASS**, `ALL_PASS=True` |

That asymmetry is the point: the verifier is capable of failing on the defect it was written
for, so its pass on the fix is evidence rather than a tautology.

No row clips or shifts: every case fills its width **exactly** (`cells == width`), so
`overflow="crop"` never engages and nothing wraps. The time column is right-aligned in its 5
cells (`| 刚刚|`, `|  now|`) and the source column in 12 (`|Release tri…|`, `|    发布排查|`).

## Independent re-run of the whole gate at the new pin

`audit.py` (the round-1 harness, pointed at the fixed tree) re-ran unchanged: **51 checks PASS,
0 FAIL**, including key parity, `check_catalogues()` on an isolated root, argument/kind/order
parity on all 72, plural categories and renders at 0/1/2/5/21/1000, literal tokens, NFC, bidi,
register, and the ten negative controls (renamed argument, wrong plural category `one`, dropped
`{source}`, plural downgrade, dropped count, bidi control, 您, over-budget label, dropped
`/unread read`, transliterated `/通知`) — all still caught.

Composite render at 40/80/120 columns with the zh-CN catalogue active (`list_text`,
`header_line`, `detail_text`, `hint_row`): every zh list line now fits (before: 40/80/120 all
overran). The remaining failures in that log are **not** this namespace's and are unchanged
from round 1:
- `en@40` header (54 cells) and `en@40` detail top (44 cells) — pre-existing English clipping at
  40 columns; zh passes both.
- the hint tie-breaks recorded as `N2` below.

## Round-1 findings, disposition

| id | severity | status in round 2 |
|---|---|---|
| N1 | major | **fixed** — measured above on both trees |
| N2 | minor | unchanged, non-gating — see below |
| N3 | nit | unchanged, recorded — see below |

**N2 (minor, unchanged, no fix requested).** `hint_read` `标为已读` / `hint_unread` `标为未读` are
8 cells each, exactly their `maxCells: 8` budget (re-measured this round: 8/8, and
`hint_clear` `取消选择` 8/8 — all three still pass). Because they sit at budget, the whole-hint
fitter drops one more hint in zh than in en at ≤80 columns (at 80, multi-select: zh drops
`u 标为未读`, en drops `o conversation`). Hints are dropped **whole**, never clipped. `已读`/
`未读` would collide with the read-state labels and break the desktop mapping, so no shorter
correct form exists; recorded, not gated.

**N3 (nit, unchanged, recorded).** `group_time` `按时间` / `group_conversation` `按对话` vs the
desktop's bare `时间` / `对话`. The TUI paints them as bare values on the header edge with no
chip label; the glossary term is intact. Keep.

## New observation this round (programme-level, not a zh finding)

`_pad_cells` is documented as "never truncating", and measured as such: a value wider than its
field overruns (`_pad_cells('123456789', 5)` → 9 cells). The **source** column is safe because
`truncate_cells` runs first, so it is exact in every locale. The **time** column is not
truncated, so the row stays exact only while the localised `now` label is ≤ 5 cells — zh `刚刚`
is 4, so zh is safe; a future locale whose "now" is longer would reintroduce the overrun. Cheap
guard for the rest of the six-locale wave: assert `cell_len(row_now()) <= TIME_CELLS` per
locale. Not a gate for zh-CN.

## Verified otherwise unchanged (same bytes as round 1)

Accuracy, idiom, register (你, no 您/！/请/particles), glossary consistency with
`ui.notifications` and `wire.slash*`/`wire.errors.notifications`, argument integrity, literal
tokens (`/notifications`, `/unread`, `/unread read`, `read`, `y/n`, `f`, `a`), plurals
(`other`-only, `#` in every branch, 0/1/2/5/21/1000), NFC, no bidi controls, 66 `maxCells`
budgets. Nothing in the copy moved, and the fix confirms the round-1 judgement that `刚刚` was
inside its budget and the padding was what was wrong.

Bonus effect of the same commit, verified (code-side, not copy): the zh detail header used to
paint the raw English priority token — ` ← 通知 … ▁▃ normal · 更新 · 1/3 ` before, and
` ← 通知 … ▁▃ · 更新 · 1/3 ` (normal) / `▁▃▅ 高 · 更新 · 1/3 ` (high) after. A sentinel control
(chip_high patched to `CHIPMARK` / `标记X` in an isolated catalogue copy) proves the header now
reads the **catalogued** value rather than the data token, in both locales.

## Verdict

- **Namespace `tui.notifications`, zh-CN: PASSED** on target sha256
  `cd60e55f3d532f44848fe44a88ee87b879192f1b5149545526b7ded829768156` at code
  `8976104b83b1f4af2fa6014ae5c73d9bbcdebc0a`.
- 72/72 keys read. 0 blocker, 0 major, 1 minor (N2), 1 nit (N3).
- **The zh-CN ledger row may ship at this hash.**

## Limits

- LLM semantic judgement, not native-speaker attestation.
- Renderer-level cell audit and real `render`/`list_text` composites; no Textual frame capture,
  no release eligibility claim, no CI claim (the coder's own regressions in
  `tests/unit/tui/test_notifications_page.py` were read, not run — my evidence is the
  measurement above).
- **Nothing here concerns the programme-level zh-CN slash namespace security HOLD.** This audit
  covers `tui.notifications` only.
