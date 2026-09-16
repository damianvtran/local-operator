# The expanded tool-card body wraps its failure reason

Evidence behind the fix for #1066: the expanded body cropped its failure
sentence at the card's measure instead of wrapping, so on the canonical
80-column terminal a failure whose remedy fits lost its cause entirely — and
the collapsed status cap (`max(8, width // 3)`, shared with the outcome glyph
and the clock) can never carry it in any state.

**What licenses the wrap is REACHABILITY, not authorship.** The guarded line is
the body's leading line when it is also the line the collapsed row's status
leads with: the status cap is a fraction of a row it shares with the glyph and
the clock, so that text is cut in every width the one-line row has, and the
expansion is its only home. On the bash surfaces that line is the tool's OWN
first output line (`app.py` settles a failed call with
`mark_failed(_first_line(result.text), result.text, details)`), which is why the
guard is a text comparison and why the wrap is budgeted by
`REASON_MAX_CELLS`/`REASON_MAX_ROWS` rather than by who composed the line. The
round-1 review (R2) was about a comment that claimed provenance; the code and
this document now say what the test actually is.

## The card under test

Every case drives the app's own settle shape, and every sentence is composed
from shipped builders or a fixture rather than re-typed, so a wording change
moves the frame instead of leaving the artifact arguing about a sentence the app
no longer produces.

| case | card | sentence | measured |
|---|---|---|---|
| `fail` | failed `web_search` | `providers._perplexity_authwall` + `service`'s `Web search failed: ` prefix | **201 cells** |
| `long_reason` | failed `bash` (`curl -X POST`) | a one-line provider failure, worded like a classifier's | **879 cells** |
| `long_token` | failed `bash` | the same shape, carrying a real-shaped webhook URL longer than the measure | **241 cells** |
| `stdout` | succeeding `bash` (`curl \| jq -c .`) | 40 captured lines of `payload=…`, **397 cells** each | 15,880 cells |

## Reproduce / re-capture

```sh
# frames + geometry for one case at one width (writes <stem>-collapsed.svg too)
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/tool_card_wrap_shot.py OUT.svg 80x30 [fail|long_reason|long_token|stdout]

# the committed sets were captured this way
for size in 40x30 45x30 60x30 80x30 100x30 200x30; do
  env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
      scripts/tool_card_wrap_shot.py after/fail-$size.svg $size fail \
      > after/fail-$size.geometry.txt 2>&1
  rsvg-convert after/fail-$size.svg -o after/fail-$size.png
done
```

`before/` is a throwaway worktree at `acbc4d72b` (the branch's base) and
`round1/` one at `ed837ce66` (the head the design, QA and agent rounds reviewed)
— both running the *same script*, which is why the three trees differ in the
source and nothing else. The script chdirs to a fixed directory because the
status band paints `os.getcwd()`, and pins the card's duration because the
settled clock is wall time; without both, the trees would differ in two columns
this work never touches.

Rasterise to look at it (an SVG is not something to eyeball as markup):

```sh
rsvg-convert after/fail-80x30.svg -o /tmp/after-80.png
```

## Geometry: the cause was unreachable, and now is not

`probe` = the row/column where each phrase of the sentence appears in the
*built* body; `ABSENT` means the crop ate it. `body_rows` counts the card's
built rows (summary + argument row + body). Every body row carries
`OUTPUT_INDENT` (2), and every row of the reason block additionally carries the
two-cell lead below — the row figures in these tables are PAINTED cells, so the
indent and the lead are in them (subtract 4 for the sentence's own cells).

| width | tree | rows | the reason block | cause reachable |
|---|---|---|---|---|
| 80 | `before` | 3 | 1 cropped row (74 cells) | **no** — `keyed Sonar`, `refused this search`, `only provider tried` all ABSENT |
| 80 | `round1`/`after` | 5 | 3 rows (75 / 61 / 75 cells) | **yes** — every probe present |
| 100 | `before` | 3 | 1 cropped row | **no** — `refused this search`, `only provider tried` ABSENT |
| 100 | `after` | 5 | 3 rows (92 / 95 / 24) | **yes** |
| 200 | `before` | 3 | 1 cropped row | **no** — `only provider tried` ABSENT |
| 200 | `after` | 4 | 2 rows (189 / 19) | **yes** |
| 60 | `round1`/`after` | 7 | 5 rows (52 / 53 / 34 / 54 / 24) | **yes** |
| 45 | `round1` | 8 | 6 rows, last `…the only provider…` | **no** — `tried)` ABSENT: the flat six-row cap cut the tail again |
| 45 | `after` | 9 | 7 rows (35 / 39 / 39 / 29 / 32 / 39 / 10) | **yes** |
| 40 | `round1` | 8 | 6 rows, last `…LOGIN) ('…` | **no** |
| 40 | `after` | 10 | 8 rows | **yes** |

The collapsed row is **byte-identical** across all three trees at every width
and for every case (the geometry transcripts' collapsed row text compares equal,
and the SVGs differ only in librsvg's random element ids): this work does not
touch the one-line guarantee, the status cap, or any ink. The card is 1 row
taller at 45 columns and 2 at 40 — the price of carrying the sentence whole
there — and identical at 60/80/100/200.

## The lead: how a reader tells our sentence from the tool's bytes (D1)

The plain body paints its reason in `tool.output.error`, which on an error card
IS the captured rows' ink (both resolve to `tint-danger`), so before this round
a 42-row failure card gave no visual answer to “which of these rows is our
sentence?”.

Every row of the reason block now carries a two-cell lead **after** the indent:
`✗` on the first row and two blanks on the continuations.

- It is a **glyph** because a glyph is this card's monochrome-safe state
  vocabulary (`ICON_ERROR`'s own note: the three outcome glyphs are the only
  thing distinguishing success from failure in a still, colourless frame). An
  ink step could not do the job here, and the alternative the review offered —
  repainting the reason in the outcome ink — is the ink the reason already has.
- It is the **same glyph the collapsed row paints its own status with**, which
  is the door the fetch card already opens for its own prose (the promoted
  `⚠ ` lead reflowing at the same indent).
- It is drawn *inside* the width the wrap already used, so the row count and
  every row's text are unchanged from `round1/`: compare `after/fail-80x30.svg`
  with `round1/fail-80x30.svg` — same three rows of 71/57/71 text cells, now
  leaded and indented — and no row exceeds the card's 76-cell lane at any width.

## The two budgets, and the marker that says what went (D2, Q1)

The reason block is bounded **twice**, because one bound cannot do both jobs:

- **`REASON_MAX_CELLS` (432) bounds the CONTENT**, and it is what binds on a
  normal frame: six rows at the canonical 72-cell measure, over twice the 201-cell
  sentence this codebase builds, and independent of the frame — a pathological
  one-line payload paints the same amount at 200 columns as at 80.
- **`REASON_MAX_ROWS` (8) bounds the SHAPE**, and it is what binds below ~62
  columns: a cell budget alone would let a 16-column terminal (an 8-cell measure)
  turn 432 cells into 54 rows. Eight is the smallest backstop that carries the
  201-cell sentence whole down to 40 columns (8 rows at that frame's 32-cell
  measure); it is exactly at the limit there, and that is stated rather than
  padded.

Measured on the 879-cell `long_reason` — the shape both bounds exist for. Cells
are the sentence's own, lead and indent excluded:

| width | sentence rows | cells painted | marker |
|---|---|---|---|
| 200 | 2 | 379 | `… 3 more lines` (cell budget bound) |
| 80 | 6 | 418 | `… 7 more lines` (cell budget bound) |
| 60 | 8 | 392 | `… 10 more lines` (row backstop bound) |
| 45 | 8 | 271 | `… 17 more lines` |
| 40 | 8 | 228 | `… 22 more lines` |

The cut is announced in the surface's own vocabulary — `… N more lines`, the
same marker the captured crop and the live body use — rather than the bare `…`
an in-sentence elision would use, and the count is the rows this card dropped
(the tests assert it against the wrap the card painted). `round1/` shows the
defect for the pair: `round1/long_reason-80x30.svg` simply stops mid-sentence
(`…attempt 2 aft…`) with nothing said, and `round1/long_reason-40x30.svg` stops
at the sixth row.

## The crop is still load-bearing (the half a blanket wrap would break)

The `stdout` case is the measurement: a `bash` card whose 40 output lines are 397
cells each (a minified payload).

| | `before` | `after` |
|---|---|---|
| body rows (80 cols) | 40 | **40** |
| card rows | 42 | **42** |
| frame | — | **byte-identical to `before`** (every row's text compares equal) |

Wrapping every body line instead would have spent **6 rows on each** of those
397-cell lines at the 80-column 72-cell measure: 240 rows where 40 stand today,
**+200 rows (6x)** on one tool call — a scroll trap in a transcript whose
expansion exists to be a receipt. The wrap budget reaches exactly one line per
card, and only when it is the line the collapsed status leads with.

## Deferred: a long unbreakable token (D3)

`long_token` records the one shape this round does **not** fix: a token longer
than the measure is broken mid-word by `wrap_cells`, leaving an orphan fragment
(`…/receip` / `ts to compare the digests`). That is the house convention the
card's ARGUMENT row already uses for the same shape, and the fetch card's prose
path deliberately takes the other one — clip, never break. Deferred on the PR
with the reason recorded there: taking it means a second wrap mode for prose,
and the clip would silently drop the tail of the very sentence #1066 is about.
The frame is otherwise unchanged from `round1/` — the same five rows with the
same break points (`…/receip` / `ts to compare the digests`, and the same
mid-token split of the ID on the row above), now leaded and indented.

## Frame inventory

`before/`, `round1/` and `after/` hold, per case and width, the expanded frame
(`<name>.svg`), its PNG rasterisation (`<name>.png` — an SVG is not something to
eyeball as markup), its `scripts/visual_capture` geometry sidecar
(`<name>.geometry.json`), and the probe transcript (`.geometry.txt`). The
collapsed frame is kept as SVG + geometry only: its byte-identity is checked by
diffing it, not by eye — and the collapsed row is one row.

- `after/` — the current head, at 40/45/60/80/100/200 for `fail`, 40/45/60/80
  plus 200 (the cell budget's end of the range) for `long_reason`, and 80 for
  `long_token` and `stdout`.
- `round1/` — the comparison tree, holding only the pairs the findings turn on:
  `fail` at 40/45/80 and `long_reason` at 40/80, plus `long_token-80x30`.
- `before/` — the base revision's four committed families, unchanged.

Changed as intended vs `round1/`: every expanded failure frame (the lead, the
narrow-frame tail, and the marker). Byte-identical: every collapsed frame, the
`stdout` frame, and the `long_token` frame's wrap — which is the deferred
finding, not a regression.
