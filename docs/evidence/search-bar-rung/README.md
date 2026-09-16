# Search-spend bar threshold — measured from each ladder's shortest rung (#1073)

## Defect

One ledger, two screens, two different bar thresholds. At terminal width 100
`/session` drew the per-provider bars (`· bars: operations` in the block header)
while `/analytics` drew none, and its counts sat ten cells left of `/session`'s:

```
/session    body 83   ▌ Search spend   this session · live · bars: operations
/analytics  body 83   ▌ Search spend   process-wide · live
```

The threshold was `_base_cells + 2 + bar + 2 + max(widest rung over every row)`.
The longest rung in the block belongs to a *reference* row — `/analytics`'
`This session … 100% of search spend`, 43 cells — which `/session` never draws,
so `/analytics` alone was held back. Measured on the same ledger at the builder
level, before this change:

| screen shape | narrowest body width that draws a bar (before) | after |
|---|---|---|
| `/session` (`session=None`) | 67 | 67 |
| `/analytics` (`session=snapshot`) | **90** | **67** |

## Root cause

`row()` in `search_spend_section` pushes a count onto a continuation line only
when **no** rung fits, so a rung that is too long for the row simply yields to a
shorter one and the row still paints one line. The wrap is therefore conditioned
on the ladder's *shortest* rung, and measuring the *widest* one suppressed the
bar in frames where every row would in fact have fitted. The same wrong rung
family sat in the per-row guard (`if len(notes[0]) > with_bar: row_bar = 0`),
which zeroed the bar — and, on the reference rows, the blank gutter their counts
share with the providers' (round-2 design D11) — for a row that would have
rendered on one line.

## Fix

`local_operator/tui/widgets/analytics_panel.py`: `[0]` → `[-1]` at the
threshold's collection site (the ladders are widest-first, so `[-1]` is the
shortest rung) and in the per-row guard, so the threshold and the check that
backs it up measure the same rung family. Comments record why the shortest rung
is the wrap condition.

## Numbers

Ledger used for every frame and the test: 5 free `duckduckgo` searches, 3 free
`deepseek:read` reads, 1 priced `brave` search ($0.0069) — every ladder bottoms
out at `6 searches · 3 reads` (20 cells), and `/analytics`' reference row ends
its ladder on `6 searches · 3 reads · 100% of search spend` (43 cells).

| quantity | value |
|---|---|
| shortest-rung threshold | `2 + 22 + 11 + 2 + 8 + 2 + 20` = **67** |
| widest-rung threshold (the old rule) | `2 + 22 + 11 + 2 + 8 + 2 + 43` = **90** |
| section width at terminal 80 / 83 / 100 | 65 / 67 / 83 — identical on both screens |
| column the counts start at, bar drawn | **47** (both screens) |
| column the counts start at, no bar | **37** (both screens) |

The count column moves by exactly the bar's ten cells (8 + its two-cell gutter),
which is what makes it the frame-level measurement of "is the bar drawn here".

The two panels measure the *same* section width at these terminals (65/65, 67/67,
83/83 through the real app), so the pair is one-variable: the only difference
between the frames is the bar policy. At terminal 83 the width is exactly the 67
threshold; at 100 it is the width the issue measured the split at.

`/session`'s frames from the two trees are identical apart from the `Last 1
request` clock (the painted rows compare equal), so the change reaches
`/analytics` and nothing else.

## Frames

`before/` is the pre-fix tree (`e854b684c`, this branch's base) and `after/` is
this branch, both captured with the same harness, the same ledger and the same
`shot_search_block.py`.

| file | shape |
|---|---|
| `before/analytics-100x72.svg` / `.png` | the reported symptom: no bar, no `bars:` clause, counts at 37 |
| `after/analytics-100x72.svg` / `.png` | same width: `· bars: operations`, counts at 47, bars drawn |
| `after/analytics-83x72.svg` / `.png` | the threshold width itself — both bodies are exactly 67 |
| `before/analytics-83x72.svg` / `.png` | the same width pre-fix, still no bar |
| `before/` `after/` `session-{83,100}x72.*` | `/session`, unchanged by the fix (bars at both widths in both trees) |

`*.crop.png` are the Search-spend block cut out at 1.6× for reading the marks;
the full-frame `.png` and the `.svg` carry the whole screen, and the
`.geometry.json` next to each SVG carries the widget regions.

## Reproducing

`$BASE` is a throwaway worktree at the pre-fix commit (it is never the cwd, so
it shows up only inside the base tree):

```
BASE=/tmp/lo-before-1073
git worktree add --detach "$BASE" e854b684c
ln -s "$PWD/.venv" "$BASE/.venv"       # throwaway only, see AGENTS.md

for tree in before:"$BASE" after:"$PWD"; do
  name=${tree%%:*}; root=${tree#*:}
  for screen in session analytics; do
    for width in 100 83; do
      env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        docs/evidence/search-bar-rung/shot_search_block.py "$root" "$name" "$screen" "$width" 72
    done
  done
done
rsvg-convert -z 1.6 -o before/analytics-100x72.png before/analytics-100x72.svg
```

The script takes the tree explicitly because the BEFORE frames come from another
checkout: it inserts that root at `sys.path[0]` and imports
`scripts.probe_isolation` before any `local_operator` module (the guard between
the two import blocks keeps isort from sorting the probe after them, which would
re-home the operator's live config). Height 72 is the shortest frame that keeps
`/session`'s block — which sits at painted row 47 — inside the viewport.
