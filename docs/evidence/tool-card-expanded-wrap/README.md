# The expanded tool-card body wraps its failure reason

Evidence behind the fix for #1066: the expanded body cropped its failure
sentence at the card's measure instead of wrapping, so on the canonical
80-column terminal a failure whose remedy fits lost its cause entirely — and
the collapsed status cap (`max(8, width // 3)`, shared with the outcome glyph
and the clock) can never carry it in any state.

## The card under test

A **failed `web_search`**, settled through the app's own call shape
(`mark_failed(_first_line(result.text), result.text, details)` in `app.py`).
The sentence is composed from the shipped builders — `providers._perplexity_authwall`
plus `service`'s `Web search failed: ` prefix — never re-typed, so a wording
change moves the frame instead of leaving the artifact arguing about a sentence
the app no longer produces. It measures **201 cells**:

```
Web search failed: Fetch a page directly, or set PERPLEXITY_API_KEY for keyed
Sonar: the anonymous tier refused this search (wall fraud_authwall_upsell/LOGIN)
('perplexity' was the only provider tried)
```

## Reproduce / re-capture

```sh
# frames + geometry for one case at one width (writes <stem>-collapsed.svg too)
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/tool_card_wrap_shot.py OUT.svg 80x30 [fail|stdout]

# the saved evidence (before/ and after/ were captured this way)
for size in 80x30 100x30 200x30; do
  env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
      scripts/tool_card_wrap_shot.py after/fail-$size.svg $size fail \
      > after/fail-$size.geometry.txt 2>&1
done
```

The `before/` tree is a throwaway worktree at `acbc4d72b` (the branch's base)
running the *same* script — the pair differs in the source and nothing else.
The script chdirs to a fixed directory because the status band paints
`os.getcwd()`, and pins the card's duration because the settled clock is wall
time; without both, the pair would differ in two columns this work never
touches.

Rasterise to look at it (an SVG is not something to eyeball as markup):

```sh
rsvg-convert after/fail-80x30.svg -o /tmp/after-80.png
```

## Geometry: the cause was unreachable, and now is not

`probe` = the row/column where each phrase of the sentence appears in the
*built* body; `ABSENT` means the crop ate it. `body_rows` counts the card's
built rows (summary + argument row + body), and every body row carries
`OUTPUT_INDENT` (2) in front of it.

| width | state | rows | the crop point | cause reachable |
|---|---|---|---|---|
| 80 | before | 3 | `…_API_KEY for…` (74 cells) | **no** — `keyed Sonar`, `refused this search`, `only provider tried` all ABSENT |
| 80 | after | 5 | — | **yes** — 3 body rows (73 / 59 / 73 cells), every probe present |
| 100 | before | 3 | `…keyed Sonar: the an…` (94 cells) | **no** — `refused this search`, `only provider tried` ABSENT |
| 100 | after | 5 | — | **yes** — 3 body rows (90 / 93 / 22 cells) |
| 200 | before | 3 | `…was the only provi…` (194 cells) | **no** — `only provider tried` ABSENT |
| 200 | after | 4 | — | **yes** — 2 body rows (187 / 17 cells) |

The collapsed row is byte-identical before and after at all three widths
(the only difference between the two SVG sets is librsvg's random element ids,
normalised away for the comparison) — this fix does not touch the one-line
guarantee, the status cap, or any ink.

## The budget, measured rather than assumed

The crop is load-bearing for **captured output**, so the wrap is scoped to the
card's own reason sentence and not to the tool's bytes. The `stdout` case is
the measurement: a `bash` card whose 40 output lines are 397 cells each (a
minified payload — the shape a blanket wrap punishes hardest).

| | before | after |
|---|---|---|
| body rows (80 cols) | 40 | 40 |
| card rows | 42 | 42 |
| frame | — | **byte-identical to before** |

Wrapping every body line instead would have spent **6 rows on each** of those
397-cell lines at the 80-column 72-cell body measure: 240 rows where 40 stand
today, **+200 rows (6x)** on one tool call — a scroll trap in a transcript whose
expansion exists to be a receipt. So the wrap budget is:

- **one line per card** — the failure reason, and only when the body's leading
  line IS that reason, so no card grows a synthetic row restating what its
  collapsed row already carries;
- **at most `REASON_MAX_ROWS` (6) rows** — 432 cells at the 80-column measure,
  twice this 201-cell sentence, and still unable to open an unbounded body from
  a pathological single-line payload (an HTTP body echoed back).

Continuation rows keep `OUTPUT_INDENT`, which is what stops a wrapped fragment
reading as a stray row — the defect `session_panel._Body.note` was fixed for.

## Frame inventory

`before/` and `after/` each hold, per case and width: the collapsed frame
(`<name>-collapsed.svg`), the expanded frame (`<name>.svg`), a PNG rasterisation
of each (`<name>.png`, via `rsvg-convert` — an SVG is not something to eyeball as
markup), their `scripts/visual_capture` geometry sidecars (`.geometry.json`), and
the probe transcript (`.geometry.txt`).

Byte-identical before ↔ after (modulo SVG element ids): every `*-collapsed`
frame, and `stdout-80x30`. Changed as intended: `fail-{80,100,200}x30` expanded.
