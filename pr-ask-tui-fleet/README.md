# Frames — TUI ask fleet scope, three-way filter, per-session marks

Evidence for the `/asks` TUI change (design `docs/design/ask-nonblocking.md`
§4/§5, amendment §11). Captured with:

```
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/ask_queue_shot.py OUT.svg 100x30 MODE        # list/bar/card frames
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/ask_fleet_shot.py OUT.svg 100x30 marks|none|fleet
```

`before/` is the same capture set on the base commit (7d21201f3), for the
before/after pairs. `*.geometry.json` alongside each frame is the capture's own
geometry dump (widget regions, virtual sizes, resolved font).

## The frames

| file | shows |
|---|---|
| `list-100x30-dark.svg` / `before/list-100x30.svg` | the list, before/after: the AFTER frame's header carries the three segments |
| `filter-all-100x30-dark.svg` | `All` pressed over a mixed queue — four rows, two halves |
| `filter-outstanding-100x30-dark.svg` | `Waiting or moved on` pressed: only the answerable/moved-on rows |
| `filter-settled-100x30-dark.svg` | `Settled` pressed: two rows, status WORDS in place of the glyph, no bullet |
| `list-settled-100x30-dark.svg` | a queue with nothing outstanding at all |
| `empty-outstanding-100x30-dark.svg` | the middle half empty over a queue that plainly has rows — its own sentence |
| `empty-settled-100x30-dark.svg` | the third half empty — its own sentence |
| `truncated-100x30-dark.svg` | a capped wire frame: the header states the backend tally and withholds the split |
| `list-fleet-100x30-dark.svg`, `list-fleet-190x50-dark.svg` | the ONE list on `All conversations`, rows from two sessions, each naming its own |
| `list-fleet-empty-100x30-dark.svg` | the fleet scope with nothing outstanding |
| `sidebar-marks-100x30-dark.svg` | the operator's report, answered: a `?` mark on EVERY session with an outstanding ask (two non-current ones) and `esc return · asks: 4` in the footer |
| `sidebar-none-100x30-dark.svg` | no outstanding ask anywhere: no mark, no note — absence is not emptiness |
| `sidebar-fleet-100x30-dark.svg` | the footer note's door: the fleet list opened from it |
| `bar-*`, `card-*`, `response*`, `late`, `timeout` | the unchanged surfaces, re-captured on this head |

## Geometry (from the frames' own geometry dumps)

* `filter-all-100x30-dark`: `ask-queue-list` region `98x7`, content region `96x5`,
  virtual `98x7` — that is `padding(2) + HEADER_ROWS(1) + 4 asks`: **one painted
  header line and exactly one painted line per ask**, which is the invariant the
  pointer hit test rests on.
* `list-100x30-dark`: virtual `98x6` = `2 + 1 + 3` asks (the base three).
* `list-fleet-190x50-dark`: virtual `188x5` = `2 + 1 + 2` fleet asks.
* The three segments are 47 cells at their widest (`All · 3` + `  ` +
  `Waiting or moved on · 3` + `  ` + `Settled · 0`) and fit the 96-cell content
  line with the drawer clause in front of them at 100x30; at 130 columns the
  hints follow, at 60 the control is alone.

## What was viewed

Every frame above was rendered to PNG with `rsvg-convert` and looked at. The two
that matter most: `sidebar-marks` shows the `?` mark on the two NON-current
sessions beside the current one and the footer total of 4 (2 + 1 from the index,
+ the current session's live 1), and `filter-outstanding`/`filter-settled` show
the pressed segment in the accent weight with the two halves genuinely showing
different rows.

## Invariant

`ask_shot.py` and `approval_shot.py` are byte-identical between `before/` and the
head (modulo the process-random `terminal-<id>` CSS prefix and the working
directory the footer prints) — the ask-long-descriptions invariant. The two runs
were made from one shared cwd so the footer's cwd cell matches too.
