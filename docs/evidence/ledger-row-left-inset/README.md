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

```
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/peer_message_shot.py before.svg 100x30 collapsed
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/wake_shot.py before-wake.svg 100x30 collapsed
python3 rasterise.py before.svg before.png 2
python3 probe.py before.svg
```

The expanded frames confirm the expansion's own two-cell body indent
(`OUTPUT_INDENT`) is untouched — its rows start on the same cell in both
frames — so the change moves the collapsed summary row only.

A first paint compared against the settled frame is byte-identical (after
normalising the per-run generated ids), so the ledger does not reflow after
paint: no motion is introduced.
