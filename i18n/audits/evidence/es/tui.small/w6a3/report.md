# es · tui.small · round 3 (w6a3) — class-sweep finding

Status: **changes_requested** — 1 major, 0 blocker. Independent auditor (not the translator).
Model as run: anthropic/claude-sonnet-5-5, no fallback. Session 13ac587c68a7.
Scope: ONE finding from the code-fixed-column sweep run in the tui.analytics w6a3 bundle.
This is NOT a re-audit of the 137 keys of tui.small.

## S1 — `tui.small.goal_panel.done_suffix` overruns a code-fixed budget (MAJOR)

- Current: `  — hecho` = **9 cells**; en `  — done` = 8.
- Code: `_GOAL_TAG_CELLS = 8` (`goal_panel.py:79`) is the width the clip reserves for the tag;
  the tag is appended after the clip (`goal_panel.py:~268-272`). Sidecar declares no `maxCells`.
- Rendered (real `GoalPanel`, 100x40, `d` pressed; audit.log in the analytics bundle):

      ▌ Migrar el servicio de pagos al nuevo proveedor y verificar cada cobro con e…  —
      hecho

  English on the same goal: one line, 86/86 cells. Pure-function check at content widths 66 and 76:
  es 67/77 cells, 2 painted lines; en 66/76, 1 line.
- Why major: the tag that states the goal is *done* is split across two lines and the card gains a
  row — the exact D2 defect the constant's own comment records for the English `— done`.
- Fix: raise `_GOAL_TAG_CELLS` to 9 (one line, code) — recommended; or `  —hecho` (8 cells) in
  the catalogue, which breaks the sidecar's 'two leading spaces and the em dash' rule.

Checked and fine in tui.small: `session_sidebar` suffixes (ladder, first rung that fits),
`todo_panel.more_below*` (measured against the widest remainder), `info_panel` metas (`_fit_pair`).
