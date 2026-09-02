# Design: line-granular windowing for the `ask` picker — descriptions and scroll that COEXIST

Status: proposal for implementation. Author: architect.
Base: worktree `lo-ask-scroll`, branch `fix/ask-long-text`, HEAD `18f9131f`.
Every `file:line` below is against that tree; oh-my-pi (OMP) references are
against the local clone at `~/workspace/repos/oh-my-pi` (HEAD `18781d8`).

Supersedes the **row-granular decision** in the three prior designs — the
argument, not the code they shipped:

- `ask-long-descriptions.md` §1 "Where I diverge from the framing" (`:117-133`),
  which took OMP's wrapping and rejected its visual-row scrolling;
- `ask-omp-scroll.md` §0 gap **G3** (`:53`) and §4.2 "Row-granular, argued with
  the cost" (`:273-314`), which closed line-granularity "by argument, not by
  code".

It keeps everything else those designs shipped: the pooled continuation-line
wrap, the `▸ RECOMMENDED` badge, the `ctrl+e` reveal, the display-only thumb,
`PageUp`/`PageDown`, the position row, and the D1 collapse ladder. This is a
change to the **windowing unit** — from option ROWS to visual LINES — and to
`_allocate`'s step 9, and to nothing else in the priority order.

Every number here was measured in this worktree against the real `OperatorApp`
(dock 5), via the probes named in §11. `_AskHost` reserves 0 dock rows where the
app reserves 5 (`test_ask_picker.py:1756-1771` uses the real app for exactly
this reason), so no geometry claim is taken from it.

---

## 0. The gap, measured

The three prior rounds all shipped. The card today wraps descriptions to a
2-line cap, reveals the rest on `ctrl+e`, and — when the list is too tall —
windows whole option ROWS with a thumb. The designer rendered ours against OMP
and found the one thing that still diverges:

**Descriptions and scroll are mutually exclusive on our card. They coexist on
OMP's.**

In OMP every option keeps its 2-line clamped description *while* the list
scrolls a thumb through them (`ask-dialog.ts:333-341`, `renderRowLabel` pushes
`label + description.slice(0,2)` for every row unconditionally; `:868-887`
`renderRows` concatenates all of them into one `allLines` list;
`scroll-view.ts:201-214` windows that list a visual LINE at a time). Ours drops
**all** descriptions the instant the list windows.

Measured on a 12-option question (13 rows with the free-text row),
`/tmp/probe_lg.py`:

```
size     rc page pos desc rev budget  grants
190x50   13   13  0    0    0    27   {all 0}   list fits, no descriptions bought (budget-starved)
150x40   13   13  0    0    0    20   {all 0}   list fits, still no descriptions
130x30   13    8  1    0    0    13   {all 0}   WINDOWS — grants all 0
120x24   13    3  1    0    0     8   {all 0}   WINDOWS — grants all 0
100x30   13    7  1    0    0    13   {all 0}   WINDOWS — grants all 0
100x20   13    2  1    0    0     6   {all 0}   WINDOWS — grants all 0
100x16   13    1  0    0    0     3   {   0}    WINDOWS, D1 collapse (pos dropped)
```

Every windowed frame has `desc_rows == {}`. That is not a coincidence the prior
designs relied on — it is the mechanism they built. `_allocate` buys the
description column all-or-nothing at step 9, and only when `remaining >=
row_count` *after* every option row is bought (`ask_picker.py:1977`,
`descriptions = rows >= self.row_count and remaining >= self.row_count`). When
the list is long enough to force `page < row_count`, `remaining` is 0 before
step 9, so the column is never bought and a scrolling list shows bare
single-line labels.

`ask-omp-scroll.md` §4.2 was right that this made line- and row-granular scroll
*identical in the frames the card could then reach*. The user has now decided
those are the wrong frames: **make descriptions and scroll coexist like OMP.**

### What the target frame costs

The OMP-style list — every row a label plus its 2-line clamp — is far taller
than the budget wherever the list matters. Measured (`/tmp/probe_appr.py`, the
12-option question, target = `sum(1 + min(2, wrap))` over all rows):

```
size     budget  target visual lines   verdict
190x50     27           26             fits — no window
150x40     20           38             WINDOWS with descriptions (was: all dropped)
130x30     13           38             windows with descriptions
120x24      8           38             windows with descriptions
100x30     13           38             windows with descriptions
100x20      6           38             windows with descriptions
100x16      3           38             windows, D1 collapse
```

So at 150x40 — the size the user works at and reported — the target list is 38
visual lines in a 20-row body: it must window *and* keep every row's 2-line
clamp. That is exactly the OMP frame the designer liked, and exactly the frame
our current allocator cannot produce.

---

## 1. Recommendation

**Adopt OMP's model directly: render every row at its 2-line clamp into one tall
line list, and window that list in VISUAL LINES. `_offset`, `_window`,
`_move_to`, the thumb and paging all reinterpret in lines. `_allocate` stops
dropping the description column when it windows; instead it fixes a body height
and lets the line list scroll under it.**

Concretely:

1. **Fixed body height, OMP-style, replacing the shrink-to-fit budget for the
   list region.** The card's body height is still `_body_rows(...)` (§3), but
   the option-list *sub-region* becomes a fixed-height viewport the line list
   scrolls under, rather than a count of rows the allocator sizes to content.
2. **Every drawn-or-not row contributes `1 + min(2, wrap)` visual lines to a
   `line list`.** This is `renderRowLabel`'s output shape (`ask-dialog.ts:329-341`)
   in our cell model. Descriptions are no longer all-or-nothing across the
   window; they are per-row and uniform at the 2-line cap.
3. **`_offset` counts visual lines. `_window` returns a line slice. `_move_to`
   keeps the SELECTED ROW'S WHOLE LINE-SPAN visible** using a `line_start_by_row`
   map — OMP's `lineStartByRow` (`ask-dialog.ts:868-912`, `#scrollOffsetForCursor`
   `:974-993`).
4. **Partial rows at the window edges are ALLOWED (line-granular clip), not
   snapped to row boundaries** (§2.3). This is what OMP does and what makes the
   thumb honest.
5. **The thumb tracks visual lines**: `total = len(line_list)`, `budget = body
   visual lines`, `offset = _offset` (§5).
6. **The reveal is re-resolved from scratch (§4). It becomes a per-row cap
   LIFT, not a competing block** — the single most important correctness
   decision here.

### Why not the alternatives

| option | verdict | reason, measured |
| --- | --- | --- |
| keep row-granular, drop descriptions on window (today) | **rejected — this is the reported bug** | the user saw OMP keep descriptions while scrolling and asked for it; §0 shows ours drops them at every windowing size |
| row-granular but draw the 2-line clamp on each windowed row (a "cheap" middle) | **rejected** | a row is 1-3 visual lines tall, so windowing whole rows makes the drawn line count jump as the window slides (a 3-line row entering the bottom either overflows the body or is refused, shrinking the page). The card's height is fixed, so the overflow clips the footer — the R11/D1 clip this file exists to prevent. OMP avoids it precisely by clipping mid-row; a row-granular card cannot, so it must either reflow or clip. Both are defects here |
| line-granular, but snap the window to row boundaries (no partial rows) | **rejected** (§2.3) | reintroduces the same jump: a body of N lines that must start and end on a row boundary draws a variable number of lines, so either the last row is dropped (page shrinks, thumb lies) or the footer clips. Snapping is row-granularity wearing a line-granular map |
| do nothing / argue the user out of it | **rejected** | §0 is a real, reported divergence from the model the user chose. The prior "buys zero frames" argument (`ask-omp-scroll.md:299`) was true *for the frames the old allocator could reach* and is false the moment the allocator keeps descriptions while windowing, which is the whole change |

### The honest risk statement, up front

This is the largest change to this widget yet and it touches the allocator core
that took four rounds to stabilise (R9-R15, D1-D9). §10 is a risk register with
a prevention for each past bug. The design is *buildable and correct*, but it is
not small, and it is not splittable (§9). If the team wants OMP fidelity — and
the user asked for it — this is the way to get it; the cost is a real
re-verification of the allocator's exact-budget proof in visual-line terms
(§3.5), not a paint tweak.

---

## 2. The windowing model: rows → visual lines

### 2.1 The line list and the row→line map

Today the card renders each windowed row on the fly inside `_card_text`
(`ask_picker.py:2433-2494`). The new model builds an intermediate the way OMP's
`renderRows` does (`ask-dialog.ts:868-887`):

```
line_list: list[_VisualLine]           # every visual line of every row, in order
line_start_by_row: list[int]           # line_start_by_row[i] = first line index of row i
                                        #   (len row_count + 1; the last entry = len(line_list))
```

Each row `i` contributes, in order:

- its **label line** (1 line), plus label continuation lines if the label
  itself wraps (rare; OMP allows it, `ask-dialog.ts:330-332`; we truncate labels
  today and may keep doing so — see §2.5);
- up to `DEFAULT_DESC_CAP` (= 2) **description lines**, `_description_lines(i,
  width)[:2]`, the last cut with `…` if the wrap is longer.

This is `_description_lines` (`:1378-1451`) sliced to the cap and the badge
already charged to the first line — no new wrap logic. The line list is a flat
`list[(row_index, kind, Text)]`; `line_start_by_row` is filled exactly as OMP's
`lineStartByRow.push(allLines.length)` before each row (`:872`).

**`line_start_by_row` is the `lineStartByRow` analogue** the brief asks for. It
is the sole source of truth for three things: `_move_to`'s cursor-visibility
math (§2.2), the thumb's `total` (§5), and the hit-test's line→row resolution
(§6). It is recomputed per paint, like `_line_rows` already is (`:2509`).

### 2.2 `_move_to`: keep the selected row's WHOLE span visible

`_offset` is now a visual-line offset. The body draws
`line_list[_offset : _offset + body_lines]`. `_move_to` scrolls just far enough
to keep the *entire* line-span of the selected row on screen — not just its
first line, which would leave the selected row's description clipped under the
cursor.

This is OMP's `#scrollOffsetForCursor` (`ask-dialog.ts:974-993`) for the
non-manual case, transcribed:

```python
def _scroll_offset_for_cursor(self, offset, cur_start, cur_end, body_lines, total_lines):
    max_off = max(0, total_lines - body_lines)
    if max_off == 0:
        return 0
    span = cur_end - cur_start                       # lines this row occupies
    if cur_start < offset or cur_end > offset + body_lines:
        # off screen either end: if it FITS, pull its bottom to the body's
        # bottom (cur_end - body_lines); if it is taller than the body, pin its
        # TOP (cur_start) so the label is the anchor, not the last desc line.
        offset = cur_end - body_lines if span <= body_lines else cur_start
    return max(0, min(offset, max_off))
```

`cur_start = line_start_by_row[selected]`, `cur_end =
line_start_by_row[selected + 1]` — OMP's `:902-903`. The `span <= body_lines`
branch is load-bearing: a row taller than the body (a very tall label plus its
clamp on a tiny card) anchors on its LABEL so the user sees what they are on,
matching OMP `:990` (`cursorRows <= rows ? cursorEnd - rows : cursorStart`).

Wheel and click keep their current shape (`on_mouse_scroll_*:1094-1100`,
`on_click:1102-1133`) — they move the CURSOR one row, and `_move_to` recomputes
the line offset. Per AGENTS.md:604 the wheel/scrollbar "move the VIEWPORT and
leave the cursor alone" is the `/settings` full-page rule; this card is an
overlay whose wheel moves the selection (the shipped, signed-off behaviour,
`ask-omp-scroll.md:106-108`), and we keep it. The cursor is never off-screen
here because every gesture routes through `_move_to`.

### 2.3 Partial rows at the edges: CLIP, do not snap — decided

**A row may be clipped mid-description at the top or bottom of the body. The
window does NOT snap to row boundaries.**

This is OMP's behaviour (`scroll-view.ts:201-214` slices `lines[offset+row]`
with no row awareness at all) and it is the correct choice here, for a reason
that is not aesthetic:

The body height is FIXED (§3, the anti-reflow invariant, `ask-long-descriptions.md`
D3). A line list of `total` lines drawn into a `body_lines`-tall viewport draws
*exactly* `body_lines` lines at every offset (padded blank only at the very end
when `total < body_lines`). If instead the window snapped to whole rows, the
number of lines drawn would vary with which rows are in view (a mix of 1-, 2-
and 3-line rows), so the body would either:

- draw fewer than `body_lines` lines (a ragged bottom edge that moves on every
  scroll — the churn D3 killed), or
- draw more and clip the footer (R11).

Clipping mid-row keeps the body a rigid rectangle, which is the whole point of
the fixed height. The partial row at the top shows its tail; the partial row at
the bottom shows its head. A reader scrolling sees a description slide up out of
view a line at a time — exactly the OMP frame the user pointed at.

What the reader must never lose is **which row the cursor is on**, and §2.2
guarantees the selected row's whole span is unclipped. Only non-selected rows
are ever partial, and a partially-visible non-selected description reads as
"there is more below", which is what the thumb confirms.

### 2.4 `_window` and the clamp

`_window` returns the line slice and clamps `_offset` into `[0, total_lines -
body_lines]` — structurally identical to today (`:2031-2037`) but in lines:

```python
def _window_lines(self):
    total = len(self._line_list)
    body = self._body_line_budget()          # §3
    offset = max(0, min(self._offset, max(0, total - body)))
    self._offset = offset
    return self._line_list[offset : offset + body]
```

The row-oriented `_window(page)` and its `page` argument disappear from the
paint path. `page` (option-row count) survives only where a *row* count is still
wanted: the position row's `showing X–Y of N` (§7) and `action_page`'s step
(§2.6). Both are derived from `line_start_by_row`, not stored as the windowing
unit.

### 2.5 Labels that wrap

OMP wraps labels (`ask-dialog.ts:327-332`). We truncate them today
(`_row_text`, single line). **Keep truncating labels** — a wrapped label is a
second, unrelated change with its own golden blast radius, and the approval
gate's labels ("Allow"/"Deny"/"Allow all") never wrap. So a row is `1 + min(2,
wrap)` lines: 1 label + 0/1/2 description lines. This bounds a row at 3 visual
lines, which keeps `span <= body_lines` true at every size with a drawable body
(`MIN_BODY_ROWS = 3`, but the list sub-region can be smaller — see §3's floor
handling).

### 2.6 Paging in the line model

`action_page` (`:798-812`) currently steps the cursor by `max(1, page - 1)`
ROWS. Keep it row-based for the CURSOR (paging moves the selection), but derive
the step from how many WHOLE ROWS fit in the body at the current offset, so a
page still means "about a screenful":

```python
def action_page(self, delta):
    body = self._body_line_budget()
    # rows whose whole span fits in one body-height, from the current top —
    # OMP pages the offset by bodyRows-1 (ask-dialog.ts:702-708); we page the
    # CURSOR by the equivalent row count so the window follows via _move_to.
    step = max(1, self._rows_per_body(body) - 1)
    target = self.state.selected + delta * step
    self._move_to(max(0, min(self.row_count - 1, target)))
```

`_rows_per_body` counts rows from `line_start_by_row` whose spans sum to
`<= body`. This preserves the "keep one row of context" property (`:803-804`)
without a fixed `page`. CLAMPED, not wrapped, unchanged (`:801`).

---

## 3. `_allocate` in visual lines

### 3.1 The body height is unchanged; step 9 changes

`_body_rows` (`:1484-1626`) still decides the card's total body height from the
three limits (available / anchored share / transcript floor) and `wanted`. The
`PROMPT_HEIGHT_SHARE = 0.7` anchoring stays — this is our analogue of OMP's
`DIALOG_HEIGHT_RATIO = 0.7` (`ask-dialog.ts:48`), and it already fixes the card
height the way OMP fixes its box (`ask-long-descriptions.md:52-54`). The card
height does NOT change with selection today and must not after this change
(§10, reflow-on-arrow).

What changes is `wanted`'s `described` term and step 9. Today `described`
sums `max(1, min(2, wrap))` per row (`:1594-1597`) — which is *already* the
OMP-style "every row's clamped height". Good: `wanted` already asks for the full
line list's height, so a roomy terminal that can afford it draws every
description with no window (matches OMP; matches §0's 190x50 target=26 ≤ budget
27). The bug is only in `_allocate` step 9's all-or-nothing gate.

### 3.2 The new step 9

Steps 1-8 are UNCHANGED (footer, question first line, one option row, windowing
line, rest of question, title, rest of option rows, reveal block §4, spacers).
They buy CHROME and the OPTION-ROW COUNT, in visual lines, exactly as today.
After step 8, `remaining` is the number of body lines left for the option list
region — call it `body_line_budget`.

The list sub-region is then a **fixed-height line viewport** of
`body_line_budget` lines. Step 9 no longer decides "descriptions: all or none".
It builds the line list at the uniform 2-line cap and lets the viewport window
it:

```
# step 9 (new): the option list is a line viewport, not a row window.
line_list, line_start_by_row = build_line_list(window_rows, width, cap=DEFAULT_DESC_CAP)
#   window_rows = the option rows step 3+7 bought the right to draw = range(row_count)
#                 (every row is in the list; the VIEWPORT hides the off-screen ones)
body_line_budget = remaining         # all remaining lines go to the viewport
show_position = len(line_list) > body_line_budget
```

Two departures from today, both deliberate:

1. **The description column is no longer conditional on `remaining >=
   row_count`.** Every row in the list carries its 2-line clamp; the viewport
   decides how many are visible. This is the whole change — it is what makes
   descriptions and scroll coexist.
2. **`page` (rows) is derived, not chosen.** `show_position` becomes "the line
   list is taller than the viewport" (`len(line_list) > body_line_budget`),
   which is OMP's `#shouldRenderScrollbar` (`scroll-view.ts:222-227`,
   `rowCount > height`).

### 3.3 What the free-text row does

The free-text row (`other_row`) has no description (`_reveal_wrap` returns `[]`
for it, `:1470-1471`; `_description_lines` returns `[]` when there is no
description). So it contributes exactly 1 visual line, always. It stays last
(`:707-724`), so its position never moves, and it windows like any other row.

### 3.4 The C1-C4 invariants, re-established in visual lines

The prior C-invariants (`ask-long-descriptions.md:296-308`) restated for the
line model:

- **C1 (exact budget).** Today: every line the plan implies is bought from
  `remaining`, one at a time, forward-only, no overdraw (`:2010`, verified over
  351,648 plans). New: steps 1-8 are unchanged and still buy line-by-line. Step
  9 spends the *entire* `remaining` on the viewport and draws EXACTLY
  `min(body_line_budget, len(line_list))` lines — the viewport pads blank past
  the list's end and clips past its budget, so it draws neither more nor fewer
  than `body_line_budget` lines when the list is at least that tall. The
  invariant becomes: **the plan's implied line count is `budget - remaining +
  min(body_line_budget, len(line_list))`, and `body_line_budget = remaining`, so
  it never exceeds `budget`.** This is C1's line-granular form and it is what
  keeps the footer (bought first, step 1) on the tail. §3.5 discharges it as a
  property test.
- **C2 (priority order).** Steps 1-8 keep their order; the change is entirely
  *within* the option-list region step 9 owns. The list region cannot take a
  footer, a question line, the windowing line, the title, a spacer, or the
  reveal block, because those are bought in steps 1-8 before the viewport is
  sized. Preserved.
- **C3 (question outranks options — the `rm -rf`/cursor-on-Allow safety case).**
  Steps 1-5 buy the question's first line (step 2) before any option beyond the
  first (step 3 buys ONE), and the rest of the question (step 5) before the rest
  of the rows (step 7). Untouched — the change is downstream of step 8. The D1
  collapse and the `_layout` position-retry guard (`:1705-1742`) are unchanged.
- **C4 (footer bought first).** Step 1, untouched.

**C5 is retired, not merely amended.** C5 was "descriptions' first lines are
all-or-nothing across the window" (`:1974-1977`) — the exact rule that made
descriptions vanish on window. Under line-granular windowing there is no
all-or-nothing decision to make: every row carries its clamp and the viewport
clips lines uniformly. The visual-consistency worry C5 encoded ("a list where
only some rows have prose reads as broken", `:1975-1976`) does not arise,
because EVERY row has its clamp — the asymmetry C5 prevented cannot occur. This
is the cleanest part of the change: the invariant that fought the feature is
removed by making the feature the default.

`show_descriptions` (`_CardLayout`, `:462-486`) becomes `True` whenever the list
region is drawn at all (any row visible), since every visible row carries prose.
Its two readers — the badge placement and `_reveal_is_useful` — are re-resolved
in §4.

### 3.5 Discharging C1 as a property test

The forward-pass proof (`ask-long-descriptions.md:296-303`) does not transfer
unchanged because step 9 is no longer a per-line loop — it is a viewport
`min(budget, len)`. Replace the exhaustive-sweep proof with a property test over
the real `_allocate` (§8, new test): for every `(width, budget, row_count,
wrap-profile, reveal-state, offset)` in a sweep, assert

```
implied_lines(plan) == budget - plan.remaining_after_step_8 + min(body_line_budget, len(line_list))
implied_lines(plan) <= budget
```

and render the frame and assert no line is clipped off the bottom (the footer is
present in `_line_rows`). This is `ask-long-descriptions.md`'s
`test_the_card_never_draws_more_lines_than_its_budget` (`:504-505`) re-derived
for the viewport. It is the single most important test in the change.

---

## 4. The reveal/scroll interaction — resolved from scratch

**This is the crux.** The prior designs proved reveal and windowing were
DISJOINT because windowing forced `remaining == 0`, hence `desc_rows == {}` and
`reveal_rows == 0` (`ask-omp-scroll.md:303-309`, `ask-long-descriptions.md`
§5). **That disjointness is gone** — a windowed list now draws descriptions, so
the two regimes overlap and the old "they never both fire" argument no longer
holds. The interaction must be redesigned, and it is where the "two mechanisms
fighting for the viewport" bug (twice-seen, AGENTS.md:595-608) will reappear if
we are careless.

### 4.1 What `ctrl+e` MEANT, and what it can mean now

`ctrl+e` today swaps the whole list's 2-line-capped view for a single tall block
showing the SELECTED row's full description, in a constant-height reservation
(`_reveal_text:2925`, `reveal_rows`). It exists because a capped list cuts every
description at 2 lines and the reveal is the only way to read the rest
(`_reveal_is_useful:3230`).

Under line-granular windowing there is a cleaner meaning available, and it is
the OMP-consistent one:

**`ctrl+e` LIFTS the per-row description cap for the SELECTED row from
`DEFAULT_DESC_CAP` (2) to `REVEAL_MAX_ROWS` (8), inside the same scrolling line
list.** The selected row grows from 3 lines to up to 9 lines *in place*; the
list scrolls to keep the now-taller selected row's span visible (§2.2 already
does this — it keys on `line_start_by_row`, which reflects the lifted cap). No
separate block, no constant-height reservation, no competing viewport.

This is strictly better than a block and it kills the fight:

- **There is only ONE viewport.** The reveal does not open a second region that
  scrolls independently of the list; it makes one row taller and the SAME line
  viewport scrolls it into view. AGENTS.md:595 ("one gesture owns the viewport")
  is satisfied by construction — there is one viewport and `_move_to` owns it.
- **`ctrl+e` cannot fight the scroll**, because it does not scroll. It changes
  the selected row's line count; `_move_to` then does the one scroll needed to
  keep the row visible, exactly as an arrow press does. Pressing `ctrl+e`,
  arrowing away, and pressing it again are all just `_move_to` calls against a
  line list whose one row is taller.
- **It is reversible and per-question**, exactly as today (`_QuestionState.revealed`,
  `:834-867`), and moving between questions does not carry the mode
  (`test_the_reveal_mode_does_not_follow_the_user_to_the_next_question:1674`).

### 4.2 Why not keep the constant-height block

The block's whole reason to exist was the anti-reflow property: a block sized to
the tallest description in the list keeps the CARD height stable as the cursor
moves (`_CardLayout.reveal_rows` docstring, `:450-460`; D3). Under line-granular
windowing the CARD height is already fixed by `_body_rows` and the viewport
(§3.1) — the list scrolls inside a rigid rectangle, so a taller selected row
does NOT change the card's height; it changes how many OTHER rows' lines are
visible in the viewport. The property the block bought is now free.

So the block is redundant machinery. Retiring it removes `reveal_rows`,
`_reveal_text`'s constant-height padding, the step-7a reveal purchase
(`:1857-1945`, ~90 lines including the `affords_column` search), and the
spacer/title yield (`:1946-1972`). **This is a large simplification** — the most
intricate, most-revised part of the allocator (the reveal-vs-column fight,
BLOCKER 1) is deleted, not ported. The reveal becomes a one-line cap change.

### 4.3 The exact mechanism

```
cap_for_row(i) = REVEAL_MAX_ROWS if (self.state.revealed and i == self.state.selected)
                 else DEFAULT_DESC_CAP
```

`build_line_list` uses `cap_for_row(i)` instead of a flat `DEFAULT_DESC_CAP`.
That is the entire reveal. The last kept line of a capped description is `…`-cut
(as `_description_lines` already marks, `:1392`); at the lifted cap the selected
row shows up to 8 lines and marks a cut only if the wrap exceeds 8.

`_reveal_is_useful` (`:3230-3286`) simplifies to: the selected row's wrap
exceeds `DEFAULT_DESC_CAP` (there is more to show) AND lifting its cap would draw
at least one more of its lines given the viewport. The elaborate "would the block
cost the column" refusal (`:3277-3284`, BLOCKER 1) is DELETED — there is no
column to cost, because lifting one row's cap never removes another row's
description; it only pushes other rows further out of the viewport, which the
thumb honestly reports.

**One new question the block never had: can `ctrl+e` push the cursor's own row
partly off-screen?** No. §2.2 anchors a row taller than the body on its LABEL
(`span <= body_lines` false → `offset = cur_start`), so a revealed row taller
than the viewport shows its label and as many description lines as fit, and the
thumb shows there is more. That is the honest degradation and it matches OMP's
manual-page clamp intent (`ask-dialog.ts:986-988`).

### 4.4 The safety case: approval gate

On the approval gate every consequence is 1 line at every width down to 44
columns (`ask-long-descriptions.md:68-70`, re-measured `/tmp/probe_appr2.py`:
grants `{0:1,1:1,2:1}`, target total 6 lines at 100/130/150). So `ctrl+e` lifts
a cap on a description that has nothing beyond line 1 to show — `_reveal_is_useful`
returns False and the footer does not offer it, exactly as today
(`test_the_approval_gate_reveal_never_strips_a_consequence:3955` must still
pass, now trivially: there is no block to strip anything). The gate is a no-op
for the reveal wherever it is already correct, which is the safety property.

---

## 5. The thumb

The thumb already exists (`_scrollbar_thumb:2039-2058`, painted in
`_card_text:2430-2453`) and its maths is the standard proportional model. Three
changes:

1. **`total` becomes `len(line_list)` (visual lines), `budget` becomes
   `body_line_budget` (visual lines), `offset` stays `_offset` (now lines).**
   `_scrollbar_thumb(total, budget)` is unchanged arithmetic — it never cared
   what the units were (`:2053-2058`). OMP's `#thumbRange` is the same model
   (`scroll-view.ts:229-238`), and it operates on `totalRows`/`height` which are
   its visual-line counts — so this brings us to byte-parity with OMP's thumb,
   which the row-granular version could not have.
2. **The thumb is painted per BODY LINE, not per option row.** Today it overlays
   the `page` one-line option rows (`:2433-2453`). Now it overlays every one of
   the `body_line_budget` viewport lines: cut each viewport line to `width - 1`,
   append track/thumb glyph in the freed column. `on_thumb = thumb_top <=
   line_position < thumb_top + thumb_len`. This is `scroll-view.ts:201-214`
   exactly — the bar spans the full body height, and one thumb cell maps to one
   visual line (not one option row). A description line under the thumb gets the
   glyph in its last column like any other.
3. **Keyed on `show_position`, unchanged.** `show_position` is still the
   allocator's overflow decision (§3.2: `len(line_list) > body_line_budget`),
   and the D1 collapse still drops it at tight heights to protect the question
   (`_layout:1720-1742`). The thumb and the position row appear and vanish
   together (the invariant `test_the_thumb_appears_only_when_the_list_windows:4928`
   pins). The coupling survives verbatim because both read `show_position`.

### 5.1 Approval byte-identity survives

The thumb is drawn only when `show_position`, and `show_position` is
`len(line_list) > body_line_budget`. At the approval gate's pinned sizes
(100/130/150), the line list is 6 lines against a 13-line budget
(`/tmp/probe_appr2.py`), so `show_position` is False and NO thumb is drawn.
The three consequence lines are drawn at their 1-line clamp exactly as today.
**The approval frame at 100x30/130x30/150x40 is byte-identical** — the change
does not reach it, because it never windows. This is the same argument
`ask-omp-scroll.md:167-197` made, and it is preserved because the *condition*
(never windows at pinned sizes) is unchanged; only the definition of "windows"
moved from rows to lines, and at the gate both agree (3 rows = 6 lines, both
under budget).

`test_the_cap_leaves_the_approval_gate_byte_identical` (`:4351`) and
`test_the_approval_gate_is_byte_identical_with_the_thumb` (`:5210`) must both
still pass unchanged; §8 adds a `line_list` length assertion so a future
regression that starts windowing the gate fails with a readable cause.

### 5.2 AMENDMENT (implementation): the gate's BOOT layout DOES overflow — labels must all fit

The byte-identity argument above measured only the SEEDED gate (dock 5, a
populated transcript). The **BOOT** gate (dock 8, empty transcript — an approval
can be the first thing a session shows) has a much tighter body budget:
measured at 100x30 the option-list viewport is `body_line_budget = 4`, and the
three consequences at their 2-line-eligible clamp make a 6-line list that
**overflows** it. Windowing there scrolls *Allow all* off the first frame
(`visible_labels == ['Allow', 'Deny']`), which is the exact **C3/D1 safety
failure** the priority order exists to prevent: an authorisation prompt that
does not NAME every answer, with a permissive option in view. This is strictly
worse than the pre-rework frame, which dropped the descriptions and kept all
three labels.

The fix is a surface policy, `AskPickerScreen._labels_must_all_fit()`:

- **False on the ask picker** — a long scannable list SHOULD scroll its labels
  with descriptions kept (this document's whole point). Unchanged behaviour.
- **True on `ApprovalPrompt`** — when the full 2-line-clamp list overflows,
  step 9 drops the uniform description cap `2 → 1 → 0`, keeping the LARGEST cap
  whose list still fits every LABEL, and windows only if even the labels-only
  (cap 0) list overflows. Descriptions are the lowest-priority content; an
  option's label is not. This *is* the (inverted-but-real) "drops descriptions
  before it drops options" ordering the prior rounds shipped.

Why this cannot be a single global ladder: at 150x40 the 12-option ask card has
`body_line_budget = 13` and its **labels-only list is also 13** — it fits
exactly. A global "reduce cap until all labels fit" ladder would therefore
collapse the headline coexist frame to labels-only-no-scroll. Both surfaces have
labels-only fitting at the geometries that matter, so the distinguishing factor
is the surface's *contract*, not the numbers: the gate must name every answer;
the picker is a list the user scrolls. The hook lives beside `ApprovalPrompt`'s
other contract overrides (Escape-denies, the `y`/`n`/`A` answer keys, the
`esc deny` exit hint).

Verified (real `OperatorApp`): the seeded gate stays byte-identical at
100/130/150 (both transcript states); the boot gate names Allow/Deny/Allow all
at 100/130/150 with no scroll; the 150x40 12-option ask card still windows WITH
2-line descriptions and a thumb. `^e` is not offered on the gate (1-line
consequences have nothing past the clamp), so it can never strip a consequence —
the §4.4 safety property holds by construction.

---

## 6. Mouse hit-testing under partial rows

`_index_at` (`:1146-1178`) resolves a click through `_line_rows` — the
body-relative line → row map recorded while painting (`:2509`). Under
line-granular windowing this map is built from the SAME structure as the thumb
and `_move_to`: `_line_rows[k]` is the row index of the `k`-th DRAWN body line
(the viewport slice's rows), or `None` for chrome (spacers, position row,
footer). A partial row at the top or bottom contributes fewer than its full span
of lines to the viewport, but every line it DOES contribute maps to it.

So a click on a partially-visible row's visible lines resolves to that row —
correct. A click on the thumb's column still resolves to the row that owns that
line (the glyph replaces the row's last cell but does not change which row the
line belongs to), which
`test_a_click_under_the_thumb_still_answers_the_row_it_lands_on:4419` already
pins by re-deriving the target from `_line_rows`. The reveal's lifted-cap lines
map to the SELECTED row (they ARE that row's description now, not a separate
block), so a click on them selects/answers the selected row — harmless and
correct, unlike today's block lines which map to `None` (`:2493`) precisely
because the block was chrome. **This is a simplification**: there is no longer a
block whose blank padding must be defended from becoming a click target
(`:2488-2492`); every line in the viewport belongs to a real row or to chrome.

One guard to keep: the free-text row's single line maps to `other_row`, and a
click there selects without answering (`on_click:1132`, `index != other_row`).
Unchanged.

---

## 7. The position row

`_position_row` (`:2512-2532`) says `showing 2–3 of 6`. Today the numbers are
ROW indexes derived from the window. Under line-granular windowing the honest
statement is still about ROWS, not lines — "showing options 2–3 of 6" is what a
user wants, not "showing lines 14–27 of 38". So:

- compute the visible ROW RANGE from `line_start_by_row` and `_offset`: the
  first row whose span intersects `[_offset, _offset + body_line_budget)` and the
  last such row. A partially-visible row counts as visible (the user can see part
  of it).
- `span = f"{first_visible_row + 1}–{last_visible_row + 1}"`, `total =
  row_count`. Same rendering as today (`:2524-2531`).

This keeps the position row an OPTION count, matching OMP's `#clipIndicator`
which is also about content presence (`↕`/`↑`/`↓`, `ask-dialog.ts:995-1002`)
not line numbers. `DEFAULT_DESC_CAP`, the badge, and the position row's styling
are untouched.

---

## 8. Tests: break, add, amend

All geometry via the real `OperatorApp` (`_real_app_card`, `:1756`), never
`_AskHost`. Visual-validation per AGENTS.md:221-373 (before-frame from a
throwaway worktree, never `git stash`).

### Breaks (must be rewritten, not deleted)

| test | line | why it breaks | new pin |
| --- | --- | --- | --- |
| `test_a_short_terminal_drops_descriptions_before_it_drops_options` | `:747` | its whole premise — descriptions dropped on window — is exactly what this change removes. A short terminal now WINDOWS with descriptions, not drops-then-windows | invert it: a short terminal keeps each visible row's 2-line clamp and windows the LIST; assert `desc` present on visible rows AND `show_position` True. The ORDER it protected (rows before descriptions) is retired with C5; the new order is "the viewport clips lines uniformly" |
| the row-granular thumb tests `test_the_thumb_tracks_the_scroll_offset` | `:4978` | `track = max(1, page)` and `_scrollbar_thumb(row_count, page)` are ROW counts; the thumb now spans body LINES over `len(line_list)` | re-pin on `_scrollbar_thumb(len(line_list), body_line_budget)`; the drawn-thumb readback (`_thumb_span`, `_thumb_column`) still works but the column is now `body_line_budget` cells tall, not `page` cells |
| `_thumb_column` helper | `:3487` | it collects "the rightmost cell of every OPTION-ROW line" via `_line_rows is not None` — but chrome lines in the viewport (none exist now inside the list, but partial-row desc lines DO map to rows) | redefine as "rightmost cell of every DRAWN BODY line in the viewport", i.e. the `body_line_budget` lines between the pre-list chrome and the position row. The thumb spans all of them |

### Amend

| test | line | change |
| --- | --- | --- |
| `test_the_thumb_appears_only_when_the_list_windows` | `:4928` | premise unchanged (thumb iff `show_position`), but `show_position` now = `len(line_list) > body_line_budget`. Add: at a size where the list fits WITH descriptions, no thumb; at a size where descriptions make it overflow, a thumb — the new reachable state |
| `test_the_thumb_never_widens_the_card` | `:5032` | unchanged in intent; now every viewport line (incl. description lines) is cut to `width-1` under the thumb. Re-assert `cell_len(line) <= width` over the whole body |
| `test_page_down_and_up_move_a_page_and_clamp` | `:5056` | step is now `_rows_per_body - 1`, not `page - 1`; re-derive expected step from `line_start_by_row` |
| `test_a_click_on_a_row_selects_and_answers_with_it` | `:387` | already re-derives target from `_line_rows` — passes unchanged (the map is the contract) |
| `test_a_click_under_the_thumb_still_answers_the_row_it_lands_on` | `:4419` | already `_line_rows`-derived — passes unchanged; strengthen by clicking a DESCRIPTION line of a windowed row and asserting it answers that row |
| `test_the_cap_leaves_the_approval_gate_byte_identical` | `:4351` | passes unchanged (gate never windows); add assertion `len(card._line_list) <= body_line_budget` at each pinned size, so a regression that starts windowing the gate fails here readably |
| `test_the_approval_gate_is_byte_identical_with_the_thumb` | `:5210` | passes unchanged (no thumb on the gate); keep as the thumb-specific guard |
| `test_the_position_row_is_only_drawn_when_the_list_is_windowed` | `:820` | `show_position` condition moved rows→lines; re-derive the windowing size. The row RANGE it reports is still option indexes (§7) |
| the 14 reveal tests | `:1282-4842` | **these change the most.** They assume a constant-height block. Re-pin each against the lifted-cap model: `test_ctrl_e_reveals_the_selected_rows_full_consequence` asserts the selected row grows to `min(REVEAL_MAX_ROWS, wrap)` lines IN THE LIST; `test_the_revealed_card_is_the_same_height_for_every_selection` asserts the CARD (not a block) is stable height (now trivially true — the viewport is fixed); `test_the_reveal_block_is_drawn_under_the_row_it_explains` becomes "the revealed lines ARE the selected row's description, in place"; `test_the_reveal_never_takes_the_last_option_row` is retired (there is no block to take a row). See §4 — this is the largest test-surface change |

### Add (new)

1. `test_descriptions_and_scroll_coexist` — the headline. A 12-option question at
   150x40: assert `show_position` True (windows) AND every VISIBLE row carries a
   2-line clamp AND the thumb is drawn. This is the frame the user asked for; it
   is currently unreachable (§0). The regression guard for the whole change.
2. `test_the_selected_rows_whole_span_stays_visible` — arrow through the list;
   after every press assert `line_start_by_row[selected] >= _offset` AND
   `line_start_by_row[selected+1] <= _offset + body_line_budget` (or, for a row
   taller than the body, `line_start_by_row[selected] == _offset` — the
   label-anchor branch, §2.2).
3. `test_a_partial_row_is_clipped_not_snapped` — at a size where the viewport
   boundary falls mid-row, assert the body draws EXACTLY `body_line_budget` lines
   (no ragged edge, no clipped footer) and that a non-selected row is
   partially visible.
4. `test_the_card_never_draws_more_lines_than_its_budget` — C1 property test
   (§3.5): sweep sizes/offsets/reveal-states, assert `implied_lines(plan) <=
   budget` and the footer is present in `_line_rows`.
5. `test_ctrl_e_does_not_fight_the_scroll` — the crux guard. Press `ctrl+e`,
   assert the selected row grew and its span is visible; arrow down twice, assert
   the list scrolled and the cursor's span is still visible; `ctrl+e` off, assert
   the row shrank back and no offset was left stranded. Pins §4's "one viewport"
   claim against the twice-seen fighting bug.
6. `test_the_reveal_is_a_noop_on_the_approval_gate` — replaces the block-strip
   test's intent: on the gate, `_reveal_is_useful` False at every pinned size,
   footer offers no `^e`, frame byte-identical revealed vs not.

---

## 9. Slicing

**One indivisible slice, one owner, one file (`ask_picker.py`).** It reworks
`_allocate` step 9, `_window`/`_move_to`/`_offset` (rows→lines),
`build_line_list`/`line_start_by_row` (new), `_card_text`'s paint loop, the
thumb paint, `_position_row`'s range computation, `action_page`'s step, and the
reveal (cap-lift replacing the block). These are not separable: the line list is
the shared data structure every one of them reads, and shipping half of them
leaves the card in a state where the window and the paint disagree about the
unit. Two coders in this file at once is the defect, not the parallelism
(`AGENTS.md` collaboration brief; `ask-omp-scroll.md:466-474`).

Sequence it INTERNALLY, so each step is verifiable before the next:

1. **Build the line list + map, unused.** Add `build_line_list` and
   `line_start_by_row`, compute them in `_layout`, assert their shape in a test.
   No behaviour change yet (the paint still uses the old row window).
2. **Switch the paint and window to lines.** `_card_text` draws the viewport
   slice; `_window`/`_move_to`/`_offset` go line-granular; thumb spans body
   lines. Descriptions now coexist with scroll — the headline test (§8 add 1)
   goes green here.
3. **Retire the reveal block, add the cap-lift.** Delete step-7a and
   `_reveal_text`'s padding; `ctrl+e` lifts the selected row's cap. The reveal
   tests are re-pinned here.
4. **Position row range, paging step, C1 property test.** The polish and the
   proof.

Then, CONCURRENTLY:

- **reviewer** reads the diff for C1-C4 in line terms (§3.4), the partial-row
  clip vs the fixed body height (§2.3), the reveal/scroll one-viewport claim
  (§4), and that the hit-test resolves partial and description lines correctly
  (§6).
- **qa** exercises the running app: the 150x40 coexist frame; arrow and page
  through a 21-option list and confirm the cursor's span stays visible and the
  thumb tracks; `ctrl+e` on a long description mid-list then arrow away and back;
  the approval gate at 100/130/150 byte-identical and thumbless.
- **designer** looks at rendered frames (the recipe, AGENTS.md:221) side by side
  with OMP: the coexist frame, a mid-scroll partial row, the revealed selected
  row in place. This is the only surface that can say it now reads like OMP.

DevOps: nothing — no pipeline/container/release surface changes. `lop-update`
is a separate final step only if the user asks to make it live.

---

## 10. Risk register

This touches the allocator core across R9-R15 and D1-D9. Each past bug, whether
this change can reintroduce it, and the prevention:

| past bug | can it come back? | prevention in this design |
| --- | --- | --- |
| **clipped footer** (R11, R15) — plan implies more lines than the body has, Textual drops the tail | **highest risk here** — the line viewport is new arithmetic | C1 in line terms (§3.4): `body_line_budget = remaining` and the viewport draws `min(budget, len)` lines, so it can never exceed budget. §3.5's property test renders the frame and asserts the footer is in `_line_rows` across a sweep. This is the test that must be written first and must be green before anything else lands |
| **reflow on arrow press** (D3) — card height changes as the cursor moves | **prevented by construction** — the card height is fixed by `_body_rows`/`PROMPT_HEIGHT_SHARE` (§3.1) and the list scrolls inside a rigid viewport. A taller selected row (reveal) changes which OTHER lines are visible, NOT the card height. §8 add-2/add-3 pin it |
| **reveal fighting scroll** (the twice-seen "two mechanisms, one viewport" bug, AGENTS.md:595) | **the crux; prevented by §4** — the reveal is a per-row cap lift inside the ONE line viewport, not a second scrolling region. There is nothing to fight. §8 add-5 is the dedicated guard. If the coder is tempted to keep the block "just in case", that reintroduces the exact fight — the block MUST be deleted |
| **approval frame shift** (byte-identity, `:4351`) | **prevented** — the gate never windows at pinned sizes (6 lines < 13 budget, measured), so no thumb, no viewport clipping, descriptions at their 1-line clamp exactly as today. §5.1, §8 amend. The two golden tests stay unrelaxed |
| **cursor off-screen** (the `/settings` wheel bug, AGENTS.md:600) | **prevented** — every gesture (arrow, wheel, click, page, ctrl+e) routes through `_move_to`, which scrolls the selected row's whole span into view (§2.2). The cursor is never left off-screen because the card is an overlay whose gestures move the selection, not a full-page list whose wheel moves the viewport independently |
| **descriptions silently lost** (the `max(1, ...)` floor bug, `_CardLayout` docstring `:478-486`) | **low** — `wanted`'s `described` term already counts `max(1, min(2, wrap))` per row (`:1594`), which is the line list's height; the viewport never asks for a budget below what step 9 needs because step 9 no longer has a conditional column to under-fund | 
| **D1 collapse divergence** (thumb keyed on raw `page<row_count`) | **prevented** — thumb keyed on `show_position`, which the D1 collapse still controls (`_layout:1720-1742`, unchanged). §5.3 |

The residual risk that is NOT a past bug: the line list is recomputed per paint
and `_layout` runs three times per paint (`_body_rows` comment, `:1590`). The
description wraps are already memoised (`_description_wraps`, `:1409`), so
`build_line_list` is `O(row_count)` slicing over cached wraps — cheap. But
verify no accidental un-memoised `_description_lines` call enters a hot loop; the
21-option thumb tests exercise the worst case.

---

## 11. Probes

Measured against the real `OperatorApp`, this worktree, HEAD `18f9131f`:

- `/tmp/probe_lg.py` — 12-option question across 190x50…100x16, plus an arrow
  sweep at 100x30. Proves **every windowed frame has `desc_rows == {}`** today
  (§0) and that `_offset`/`_window`/`_move_to` track the cursor row-granularly.
- `/tmp/probe_appr.py` — the same question with the **target visual-line total**
  (`sum(1 + min(2, wrap))`): 38 lines at 150x40 against a 20-row budget, i.e. the
  target must window WITH descriptions (§0, "What the target frame costs").
- `/tmp/probe_appr2.py` — the approval gate at 100/130/150/60/44. Confirms the
  gate's target total is **6 lines under a 13-line budget at the pinned sizes**
  (never windows, no thumb, byte-identity safe) and windows only below ~60
  columns where no golden exists (§5.1).
- `/tmp/current-150x40.svg` — the reported gap frame rendered from
  `scripts/ask_user_repro.py 150x40 0`: the 4-option long-description card whose
  windowing drops descriptions.

Re-run any with `env -u NO_COLOR TERM=xterm-256color .venv/bin/python <probe>`.

---

## 12. One-paragraph summary for the manager

The user is right and the fix is real work, not a talk-them-out-of-it. Our card
drops every description the instant the list scrolls (measured: `desc_rows=={}`
at every windowing size); OMP keeps each row's 2-line clamp and scrolls the list
a visual line at a time. Closing the gap means switching the windowing unit from
option ROWS to visual LINES — a `line_list` + `line_start_by_row` map (OMP's
`lineStartByRow`), `_offset`/`_window`/`_move_to`/thumb/paging all in lines, and
`_allocate` step 9 stops gating the description column on the list fitting.
Partial rows clip at the edges (not snap) so the fixed body height stays a rigid
rectangle and the footer never clips. The reveal is **redesigned**: `ctrl+e`
becomes a per-row cap lift inside the one viewport instead of a competing block —
which deletes the most-revised, most-fragile part of the allocator and removes
the "two mechanisms fighting for the viewport" bug by construction. The approval
gate never windows at its pinned sizes, so its byte-identical frames survive
untouched. One indivisible slice in `ask_picker.py`, sequenced internally in
four verifiable steps, with a C1 line-budget property test that must be green
before anything else lands. This is the biggest change to this widget yet; §10
maps every past bug to its prevention.
