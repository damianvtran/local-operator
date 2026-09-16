# Ledger row left inset — the peer/wake rows were one cell left of the tool rows

## Defect

Reported: *"sometimes the peer message doesn't have the proper left padding
like the other tool cards."*

In a transcript, a `PeerMessageBlock` (and the same-shaped `WakeBlock`) drew
its icon flush against the card's own left wall, while a neighbouring
`ToolCard` drew its icon one cell in. Icon, name column and summary were each
one cell left of the tool rows they sat between, so the ledger's column
visibly broke wherever an inbound receipt or a wake landed.

`annotated-collapsed.png` and `annotated-wake.png` stack the before frame over
the after frame at 3x with a red rule on the shared icon column: in both "before"
panes the `bash`/`read`/`send` icons stand on the rule while `peer`/`wake`
stand left of it; in both "after" panes every row is on the rule.

## Root cause

`ToolCard._build_row` (`local_operator/tui/widgets/tool_card.py`) draws a
one-cell left inset (`ROW_INDENT`) and takes it off the row's width budget
before anything else is measured; `ToolCard._row_indent()` derives the same
value from the built width and feeds `copy_gutter`, so the copy gutter and the
painted row cannot disagree.

The two other ledger row types in `local_operator/tui/widgets/transcript.py`
never got it:

- `WakeBlock._build_row` / `PeerMessageBlock._build_row` did
  `width = max(width - 2, 10)` and prepended nothing.
- `WakeBlock.copy_gutter` / `PeerMessageBlock.copy_gutter` hardcoded
  `2 if index == 0 else OUTPUT_INDENT`, i.e. they assumed indent 0.

## Fix

Give both rows the same derivation `ToolCard` uses, imported from `tool_card`
so the constant stays single-sourced: a `_row_indent()` helper reading
`ROW_INDENT if self._built_width >= ROW_INDENT_MIN_WIDTH else 0`, used by both
`_build_row`s (which now subtract the indent from the width before the rest of
the arithmetic, and prepend it to the row — including the degraded
`name_budget < 2` early return) and by both `copy_gutter`s
(`self._row_indent() + ToolCard.ICON_COLS` for index 0). The narrow-width
ladder is unchanged: below `ROW_INDENT_MIN_WIDTH` the indent is 0 for all
three row types, so nothing is pushed past the `⟨∅⟩` answer rung.

## Geometry (the numbers behind the stills)

Cell width 8px; columns are the cell the glyph run starts in. `geometry-*.txt`
carry the full dumps; both frames are 100x30.

| row | field | before | after |
|---|---|---|---|
| `ToolCard` (`bash`) | icon / name / summary | c3 / c5 / c14 | c3 / c5 / c14 |
| `PeerMessageBlock` (`peer`) | icon / name / summary | **c2** / **c4** / **c13** | c3 / c5 / c14 |
| `WakeBlock` (`wake`) | icon / name / summary | **c2** / **c4** / **c13** | c3 / c5 / c14 |

`probe.py` re-derives the table from any frame's SVG (`<tspan x=...>` is one
per grapheme cluster, so the reported x is the exact cell the compositor was
told to draw at).

## Frames

| file | shape |
|---|---|
| `before-collapsed.svg` / `.png`, `after-collapsed.*` | `scripts/peer_message_shot.py out.svg 100x30 collapsed` |
| `before-expanded.*`, `after-expanded.*` | same script, `expanded` |
| `before-wake-collapsed.*`, `after-wake-collapsed.*` | `scripts/wake_shot.py out.svg 100x30 collapsed` |
| `annotated-collapsed.png`, `annotated-wake.png` | the before/after crops at 3x with a red rule on the shared icon column |

Produced with an isolated HOME/config (`scripts/visual_capture.isolate_capture`)
and the faithful capture profile (`scripts/visual_capture.save_capture`), then
rasterised with `rasterise.py` (librsvg via `rsvg-convert -z 2`, with
`xml:space="preserve"` asserted on every `<text>` node — a run of leading
spaces collapses under `xml:space="default"`, which is exactly the spacing the
frames prove).

The two sides of a pair are one-variable comparisons, and that is enforced
rather than hoped for:

- The shot scripts call `scripts.visual_capture.settle_status_line` before
  saving. It waits on the band's own state — comparing against the
  `MODEL_PENDING` sentinel the app pushes, not merely against an empty label —
  and it reports on stderr if a band it can read never settles. (QA round 1 on
  PR #972, Q2: the committed peer pair used to carry that unrelated footer
  difference. Review round 2, M1: the first version of the helper tested the
  label for truthiness, which the sentinel satisfies, so it waited one frame
  and proved nothing.)
- The BEFORE frames are the pre-fix builders run with the same harness and the
  same process cwd as the AFTER frames: the base worktree's copy of the
  scripts, invoked from this worktree, so the status bar's cwd segment reads
  identically on both sides. Verified by pixel-diffing the rasterised pairs —
  the only differing bands are the ledger rows themselves. For the collapsed
  peer pair those are `y 278-305` and `y 415-441` (the two peer rows); the wake
  pair `278-305` and `414-441`; the expanded pair `687-713` (the summary row,
  since the peer receipt scrolled its collapsed summary off-frame).

Reproducing a pair from scratch, with `$BASE` as the throwaway worktree path
(any path you like — it is never the cwd, so it only shows up inside the base
tree):

```
# 1. The pre-fix tree, cut at the last commit before this change. `origin/main`
#    plus these scripts reproduces the defect; 5dd5e57e3 is the SHA the frames
#    here were cut at.
BASE=/tmp/lo-base972
git worktree add --detach "$BASE" 5dd5e57e3
ln -s "$PWD/.venv" "$BASE/.venv"          # throwaway only, see AGENTS.md
cp scripts/visual_capture.py scripts/peer_message_shot.py scripts/wake_shot.py "$BASE/scripts/"

# 2. BEFORE: pre-fix builders, identical harness, and the SAME cwd as AFTER
#    (the status band paints the cwd, and the cwd is what the process started in).
env -u NO_COLOR TERM=xterm-256color .venv/bin/python "$BASE/scripts/peer_message_shot.py" before.svg 100x30 collapsed
env -u NO_COLOR TERM=xterm-256color .venv/bin/python "$BASE/scripts/wake_shot.py" before-wake.svg 100x30 collapsed

# 3. AFTER: this branch, same cwd.
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/peer_message_shot.py after.svg 100x30 collapsed
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/wake_shot.py after-wake.svg 100x30 collapsed

python3 rasterise.py before.svg before.png 2
python3 probe.py before.svg
```

The symlinked venv is the one AGENTS.md allows: a throwaway worktree used to
take a before-frame, where the parent's code is what you want — every script
here self-corrects with `sys.path.insert(0, ...)` at the top, so it reads the
tree it lives in.

The expanded frames confirm the expansion's own two-cell body indent
(`OUTPUT_INDENT`) is untouched — its rows start on the same cell in both
frames — so the change moves the collapsed summary row only.

A first paint compared against the settled frame is byte-identical (after
normalising the per-run generated ids), so the ledger does not reflow after
paint: no motion is introduced.
