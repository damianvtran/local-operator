# Design: making the `ask` picker card scroll like oh-my-pi

Status: proposal for implementation. Author: architect.
Base: worktree `lo-ask-scroll`, branch `fix/ask-long-text`, HEAD `3b1281b5`.
Every `file:line` below is against that tree; oh-my-pi (OMP) references are
against the local clone at `~/workspace/repos/oh-my-pi`.

Supersedes nothing. It ADDS to the two shipped changes — the pooled wrap
(`docs/design/ask-long-descriptions.md` §1(A)) and the `ctrl+e` reveal (§1(B),
kept verbatim) — a scroll affordance the user asked for after seeing OMP's
picker side by side with ours. It does not touch the allocator's priority
order, the reveal, or the byte-identical approval frames.

Every number here was measured in this worktree against the real `OperatorApp`
(the probes are named in §9), not estimated from the code. `_AskHost` was used
for nothing: it reserves **0** dock rows where the app reserves **5**
(`test_ask_picker.py:4251-4268`), so a geometry claim taken from it is a claim
about a card that does not exist.

---

## 0. Verdict up front

**The card already scrolls the way OMP does. Three quarters of the requested
change is already shipped, and the honest design is to add the two pieces that
are missing and change nothing else.**

Measured against the real app (`/tmp/probe2.py`, `/tmp/probe3.py`), a
12-option question at 120×24:

```
page=3  row_count=13  offset=0  sel=0   window=[0,1,2]
                       (press ↓ ×4)
page=3  row_count=13  offset=2  sel=4   window=[2,3,4]
                       (press ↓ ×8)
page=3  row_count=13  offset=6  sel=8   window=[6,7,8]
```

That is OMP's autoscroll model — arrows move the selection, the viewport
follows to keep the cursor drawn — running today through `_move_to`
(`ask_picker.py:1998-2012`) and `_window` (`:1989-1995`). The wheel already
scrolls it (`on_mouse_scroll_down/up`, `:1052-1058`). The overflow is already
reported, textually, by the position row `showing 2–3 of 6`
(`_position_row`, `:2415-2435`), bought at step 4 and drawn from
`layout.show_position` (`:2406-2408`).

What OMP has that we do **not**:

| id | severity | gap |
| --- | --- | --- |
| G1 | requested | **No scrollbar thumb.** OMP paints a proportional `█` on a `│` track (`scroll-view.ts:229-238`); we show only the textual count. |
| G2 | requested | **No `PageUp`/`PageDown`.** OMP pages the viewport by `bodyRows-1` (`ask-dialog.ts:701-712`); this card has no page binding at all (`BINDINGS`, `:525-550`). |
| G3 | rejected | **Line-granular scrolling.** OMP scrolls by visual ROWS through a `ScrollView`; we window whole option rows. §4 shows the two are provably identical wherever scrolling actually happens, so this gap is closed by argument, not by code. |

So the deliverable is small: **add a display-only thumb painted only while the
list windows (G1), add two page bindings (G2), and keep everything else.** The
rest of this document is mostly about what NOT to touch and why touching it
would break the approval gate.

---

## 1. Recommendation

**One slice, one owner, in `ask_picker.py`, sequenced internally B → A:**

- **(B) `PageUp`/`PageDown`** — two `BINDINGS` entries and one `action_page`
  that moves the cursor a page and lets `_move_to` autoscroll. No allocator
  change, no byte-identity risk. Tiny; land it first.
- **(A) The scrollbar thumb** — copy `usage_panel.py`'s thumb maths, paint it
  over the rightmost column of the windowed body rows, and **only when
  `page < row_count`**. This carries all the byte-identity risk, so it lands
  second with the full visual-validation recipe and a golden re-run.

**Row-granular, not line-granular** (§4). **Display-only thumb, no drag** (§3.4).
**No new footer glyph** — the position row already is the overflow indicator
(§3.3). **The reveal is untouched** and never coexists with scroll because the
two live in disjoint size regimes (§5).

### Why not the alternatives

| option | verdict | reason, measured |
| --- | --- | --- |
| port OMP's `ScrollView` + line-granular budgeting | **rejected** | it is the exact change `docs/design/ask-long-descriptions.md:126-133` rejected: converting `page` from ROWS to LINES makes the number of ANSWERS on screen depend on prose length, inverting C2/C6. And it buys nothing — §4 shows windowing only ever happens when every row is one line, so line- and row-granularity coincide in every scrolling state |
| reserve a persistent scrollbar gutter (usage_panel's model) | **rejected** | usage_panel reserves the column in EVERY state (`SCROLLBAR_GUTTER_CELLS`, `usage_panel.py:76`, `_body_content_width:993-1004`) because its right-aligned NUMBERS must not slide when the bar toggles. Our rows are left-aligned labels with nothing at the right edge to keep stable, so the reservation buys nothing — and it would shift the approval frame one column at 100/130/150, breaking `test_the_cap_leaves_the_approval_gate_byte_identical` (`:4227`). §3.2 |
| a footer `↕`/`↑`/`↓` glyph (OMP's `#clipIndicator`, `ask-dialog.ts:995-1002`) | **rejected as an ADDITION** | the position row (`:2415`) already says the same thing with a count instead of a glyph, and it appears on exactly the same condition. Two indicators for one fact is the redundancy this file refuses elsewhere. §3.3 |
| a distinct "scroll without moving the selection" arrow gesture | **rejected — and the user settled this** | OMP has no such gesture; paging and the wheel already move the viewport. Adding one would leave the cursor off-screen, which `_move_to` exists to prevent |
| thumb drag (grab-and-drag like usage_panel) | **rejected** | a 1-cell drag target on a card that can authorise a tool call is a misclick onto a row, and a click on a row ANSWERS (`on_click`, `:1060`). usage_panel needed `SCROLLBAR_GRAB_PAD` gymnastics (`:78-93`) precisely because 1-cell targets are missable; here a miss authorises. The wheel and the page keys already cover the gesture. §3.4 |
| do nothing | **rejected** | G1/G2 are real gaps against the model the user chose, and the thumb in particular is the at-a-glance "where am I in the list" the count only gives numerically |

---

## 2. What is already there, and stays untouched

The windowing machinery this design builds on, all verified against the real
app, none of it changed by this slice:

- `_offset` — the scroll position (init `:607`, reset in `_move_to`/reveal).
- `_window(page)` (`:1989-1995`) — clamps `_offset` and returns the drawn
  indexes. Read-and-writes-back the offset; the same shape as
  `session_picker.py:1010-1014`.
- `_move_to(index)` (`:1998-2012`) — moves the cursor and scrolls **only far
  enough to keep it drawn**. This IS OMP's `#scrollOffsetForCursor`
  (`ask-dialog.ts:974-993`) for the non-manual case: `cursorStart < offset`
  scrolls up, `cursorEnd > offset + rows` scrolls down, otherwise nothing
  moves. We do it per row; OMP does it per line (§4).
- `on_mouse_scroll_down/up` (`:1052-1058`) — the wheel moves the cursor one
  row, CLAMPED (a wheel gesture that wrapped would read as the card resetting).
  This is the mouse-wheel behaviour the user asked to keep.
- `_position_row` (`:2415-2435`) — `showing 2–3 of 6`, bought as
  `show_position` at step 4 and drawn at `:2406-2408`. The overflow indicator,
  already conditioned on windowing.
- `_line_rows`/`_index_at` (`:1104-1136`) — the body-line → row map the
  hit-test reads. Already handles a windowed paint; a thumb overlaid on a row's
  last column does not change which row a line maps to.

`_allocate`'s priority order (`:1763-1987`) and its `page`-counts-ROWS invariant
(C6) are **not** touched — that is the whole force of the row-granular decision
in §4.

---

## 3. (A) The scrollbar thumb

### 3.1 What to add, and where it comes from

Copy the thumb maths from `usage_panel.py` rather than porting OMP's
TypeScript or inventing a third one. The two agree on the standard model, and
usage_panel's is already in this codebase, tested, and in this repo's idiom:

- `SCROLLBAR_TRACK = "│"`, `SCROLLBAR_THUMB = "█"` (`usage_panel.py:102-103`) —
  reuse the glyphs and the `edge`/`muted` styling so the card matches the rest
  of the app. (OMP uses the same two glyphs, `scroll-view.ts:5-6`.)
- `_scrollbar_thumb(total, budget) -> (thumb_top, thumb_len)`
  (`usage_panel.py:1088-1103`) — `thumb = clamp(round(track·budget/total),1,track)`,
  `top = round((track-thumb)·offset/max_off)`. This is byte-for-byte OMP's
  `#thumbRange` (`scroll-view.ts:229-238`) apart from `floor` vs `round` on the
  size; either is fine, and matching the in-repo one keeps a single
  implementation. Here `total = row_count`, `budget = page`, `offset = _offset`.
- The paint step is a trimmed `_paint_scrollbar` (`usage_panel.py:1119-1147`):
  for each drawn body row, cut the row to `width - 1` and append the track or
  thumb glyph in the freed column.

New constant on the widget: none needed beyond the two glyphs — the gutter is
**not** reserved (§3.2), so there is no `SCROLLBAR_GUTTER_CELLS` analogue.

Wiring into `_card_text` (`:2353-2357`): after the window is drawn, if
`layout.page < self.row_count`, overlay the thumb on the `page` option-row
lines. The rows are drawn one visual line each in this state (§4), so the
overlay maps one thumb cell to one option row — the thumb's position IS the
window's position in the list.

### 3.2 Why the thumb is painted only on overflow, and never reserved — the approval-gate constraint

**Paint the thumb only when `page < row_count`, and reserve no gutter when it
is absent.** This is the decision the scout flagged, and it is correct. The
justification is a measured property of the approval gate.

`ApprovalPrompt` (`approval.py:980`) subclasses this card, mounts it with
`allow_free_text=False`, and its three consequence strings are the description
that decides an authorisation. `test_the_cap_leaves_the_approval_gate_byte_
identical` (`:4227-4300`) pins the EXACT frame at 100×30, 130×30 and 150×40.
A persistent gutter would shift every row one column left at all three sizes
and break all three goldens — and that test says in its own docstring that
needing to relax it is "a stop-and-escalate, not an expectation to update"
(`:4248`).

The reason a conditional thumb is safe is not luck. **Windowing and the
approval gate's byte-identical frames never coincide.** Measured
(`/tmp/probe3.py`):

```
approval, 100×30: page=4  row_count=4  windowing=False   (no thumb)
approval, 130×30: page=4  row_count=4  windowing=False   (no thumb)
approval, 150×40: page=4  row_count=4  windowing=False   (no thumb)
approval,  60×20: page=1  row_count=4  windowing=True    (thumb — no golden here)
approval,  44×20: page=1  row_count=4  windowing=True    (thumb — no golden here)
```

The approval card has four rows (three consequences + no free-text) and windows
only below ~44 columns, which agrees with `ask-long-descriptions.md:68-70`
(consequences wrap `[1,1,1]` down to 44 columns). Every pinned golden is at a
width where `page == row_count`, so the thumb is never drawn on a frame anyone
has written down. The thumb is keyed on `show_position`, NOT on a fresh
`page < row_count` — an implementation refinement over this section's original
claim that the two are equivalent. They are NOT: at tight heights the D1
collapse in `_layout` drops the position row while the list is still windowed
(measured at 100x16, 22 options: `page < row_count` is True but `show_position`
is False). Keying the thumb on `show_position` keeps the thumb
and the count appearing and disappearing together — one overflow signal, two
renderings, never one without the other.

usage_panel reserves its gutter unconditionally because toggling it would
slide right-aligned numbers (`_body_content_width:993-1004`, "toggling
scrollable/not-scrollable never slides a column sideways"). This card has no
right-edge data — rows are left-aligned labels — so there is nothing to keep
stable and nothing the reservation buys. We take usage_panel's *maths* and
reject its *gutter policy*, and the difference is load-bearing.

**One byte-identity subtlety to hand the coder.** When the thumb IS drawn, the
row under it is cut to `width - 1` (OMP does the same: `contentWidth = width -
(showScrollbar ? 1 : 0)`, `scroll-view.ts:198`). That is a change to *windowed*
frames only. Any golden pinned on a windowed frame shifts by that one column
and must be re-pinned; the non-windowed goldens (approval, short-description)
are untouched because no thumb is drawn there. §7 lists the tests to check.

### 3.3 Why no separate footer glyph

The user's settled decision names "a scrollbar thumb and a footer overflow
indicator." The footer overflow indicator **already exists**: the position row
`showing 2–3 of 6` (`:2415-2435`) sits directly above the footer and is drawn
only when the list windows — the same semantics as OMP's `↕`/`↑`/`↓`
(`ask-dialog.ts:995-1002`), with a count where OMP has a glyph. Adding OMP's
glyph on top would be two indicators for one fact.

Recommend: **keep the position row as the overflow indicator, add the thumb for
the at-a-glance proportional read the count cannot give, and add no footer
glyph.** If literal OMP parity on the glyph is later wanted, it is a one-line
change to `_footer_hints` — but it should REPLACE the position row, not sit
beside it, and that is a separate change with its own windowed-frame blast
radius. Out of scope here.

### 3.4 Display-only, no drag

usage_panel's thumb is draggable (`on_mouse_down/move/up`, `_scrollbar_hit`,
`:835-880`). **Do not copy the drag.** A click on an option row ANSWERS the
question (`on_click`, `:1060`; on the approval gate it authorises the tool
call), and usage_panel needed the `SCROLLBAR_GRAB_PAD` blank-tail gymnastics
(`:78-93`) precisely because a 1-cell drag target is easy to miss — here a miss
lands on a row and answers. The wheel (`:1052-1058`) and the new page keys
(§4) already move the viewport, so the drag adds a misclick hazard for a
gesture already covered. OMP's `ScrollView` thumb is likewise display-only in
the ask dialog; only `handleScrollKey`/wheel move it (`scroll-view.ts:153-187`).

---

## 4. (B) PageUp/PageDown, and the row-granular decision

### 4.1 The gesture

Two `BINDINGS` entries and one action:

```python
Binding("pageup", "page(-1)", "Page up", show=False),
Binding("pagedown", "page(1)", "Page down", show=False),

def action_page(self, delta: int) -> None:
    """PageUp/PageDown move the cursor a page and let the window follow.

    CLAMPED, not wrapped: paging past the end lands on the end, unlike
    action_move which wraps a discrete keypress. The step matches OMP's
    `Math.max(1, bodyRows - 1)` (ask-dialog.ts:702-708) so a page keeps one
    row of context across the jump.
    """
    step = max(1, self._layout().page - 1)
    target = self.state.selected + delta * step
    self._move_to(max(0, min(self.row_count - 1, target)))
```

This routes through `_move_to`, so the viewport autoscrolls to keep the cursor
drawn exactly as the arrows do. It "pages the viewport" (the window jumps a
page because the cursor does) AND keeps the cursor visible — which is precisely
the settled decision, and it does so **without** adding the distinct
"scroll-without-moving-selection" gesture the user ruled out. OMP separates the
two (`manualScroll=true` on a page, `false` on an arrow, `ask-dialog.ts:701-724`)
and then re-clamps the cursor into view on the next arrow via
`#scrollOffsetForCursor`'s manual branch (`:986-988`). Collapsing them onto our
cursor-locked model is simpler, has no `manualScroll` state to carry, and can
never leave the cursor off-screen.

The step is in ROWS (`page - 1`), which is C6-consistent and needs no allocator
change.

### 4.2 Row-granular, argued with the cost

The scout asks whether row-granular windowing (cheaper, keeps `_allocate`
intact) is acceptable or whether line-granular is required. **Row-granular, and
not as a compromise — the two are provably identical in every state where
scrolling occurs.**

The claim rests on one measured fact: **the card windows only when every drawn
row is exactly one visual line.** `_allocate` buys descriptions all-or-nothing
at step 9, and only when `remaining >= row_count` after every row is bought
(`:1935`). When the budget is tight enough to force `page < row_count`,
`remaining` is 0 before step 9, so `description_rows` is `{}`. Measured across
the windowed sizes (`/tmp/probe2.py`, `/tmp/probe3.py`):

```
120×24, 12 opts:  page=3  desc_rows={}   (windowing)
100×28, 13 opts:  page=5  desc_rows={}   (windowing)
100×20, 13 opts:  page=1  desc_rows={}   (windowing)
 60×20, approval: page=1  desc_rows={}   (windowing)
```

Every windowed plan measured has `desc_rows={}`. In that regime a drawn row IS
one visual line, so "window N rows" and "window N visual lines" produce the
identical frame. Line-granular scrolling can only differ from row-granular
where a row spans more than one line — and those are exactly the states where
`page == row_count`, the whole list fits, and nothing scrolls at all.
Therefore line-granular buys **zero** additional reachable frames, while
costing the `_allocate` rewrite and the C2/C6 inversion
`ask-long-descriptions.md:126-133` already rejected.

There is one apparent exception to rule out: the reveal block adds lines under
the selected row. Could the card window *and* reveal at once, giving a
multi-line row inside a windowed list? No — the reveal is bought at step 7a
from `remaining` (`:1815-1878`), and in the windowing regime `remaining` is 0,
so `reveal_rows = 0`. Measured: every windowed probe reports `reveal_rows=0`.
The two regimes are disjoint (§5), so the "multi-line row in a windowed list"
state does not exist and line-granularity has nothing to fix there either.

**Cost of row-granular: nothing.** `_allocate`, `_window`, `_move_to`,
`_index_at` are all unchanged. The paging action reads `layout.page` (rows) and
moves the cursor. The thumb reads `(row_count, page, _offset)` — all row
counts. No line arithmetic enters the design.

---

## 5. How the retained `ctrl+e` reveal coexists with scroll

**They do not fight, because they occupy disjoint size regimes.** This is the
interaction the scout warns about ("we already had that bug"), and the
resolution is that the two mechanisms are never both live:

- **Reveal** answers "the list fits but a description is CUT" — a card roomy
  enough to draw every label (`page == row_count`) but not every wrapped
  description. `_reveal_is_useful` (`:3133-3160`) requires the selected row's
  prose to exceed its grant AND the plan to afford a reveal line, i.e.
  `remaining > 0` after the rows.
- **Scroll** answers "the list itself does not fit" — `page < row_count`,
  which by §4 means `remaining == 0` and `desc_rows == {}`.

`remaining > 0` and `remaining == 0` cannot both hold, so a windowed card never
offers the reveal and a card offering the reveal never windows. Measured: in
every windowed probe `reveal_rows=0`, and `_offers_reveal` (`:3113-3131`)
returns False whenever `_reveal_is_useful` is False. No new guard is required —
the existing budget arithmetic already makes them mutually exclusive.

Two consequences to state so the coder does not re-introduce the old bug:

1. **The selected row is always drawn**, so the reveal always targets a visible
   row. `_move_to` guarantees `_offset <= selected < _offset + page`
   (`:2008-2011`), so `_reveal_text` (`:2828`, targets `self.state.selected`)
   can never point at an off-screen row. Paging preserves this because it goes
   through `_move_to`.
2. **Do not make `ctrl+e` scroll, and do not make paging reveal.** They are
   answers to different questions and the footer advertises each only where it
   works (`_reveal_hint`, `:3085-3111`; page keys are always live but no-op on
   a list that fits, like the arrows). Keeping them separate is what avoids the
   "two overlapping mechanisms fighting for one viewport" defect
   (`AGENTS.md:577-607`, "one gesture owns the viewport at a time").

The 14 reveal tests (§7) all exercise the non-windowing regime and are
unaffected.

---

## 6. The degradation ladder

**The change adds no rung to the priority order.** The thumb is a paint-time
overlay on rows the allocator already bought; paging is pure navigation.
Neither spends a body row, so the ladder from
`ask-long-descriptions.md:396-421` stands verbatim and C1–C6 are preserved by
construction:

- **C1 (exact budget):** untouched — no `remaining -= n` is added. The thumb
  overlays existing rows; it draws no new line.
- **C2 (priority order) / C3 (question outranks options) / C4 (footer first):**
  untouched — `_allocate` is not edited.
- **C5 (first line all-or-nothing):** untouched — descriptions are not touched,
  and in the windowing regime there are none.
- **C6 (`page` counts ROWS):** the load-bearing one, and the reason the thumb
  and paging are cheap. `page` stays a row count; the thumb reads it as one;
  paging steps by it. This is the invariant the whole slice is built to
  preserve.

The ladder, with the thumb noted at each rung:

- **≥190×50, ≥150×40** (roomy): `page == row_count`, list fits, **no thumb**,
  reveal available where a description is cut. Byte-identical to today.
- **descriptions dropped but list fits** (~140×36): `page == row_count`, **no
  thumb**, reveal recovers the selected row. Unchanged.
- **list windows** (tight budget / long list): `page < row_count`,
  `desc_rows={}`, **thumb drawn** over the `page` one-line rows, position row
  bought, reveal not offered (`remaining == 0`). This is the only new-looking
  frame, and the thumb costs no row.
- **`page == 1`** (very tight): thumb is a 1-row track with a 1-row thumb —
  honest ("you are somewhere in a taller list"), harmless. OMP paints here too
  (`scroll-view.ts` has no minimum-height suppression).
- **`budget < MIN_BODY_ROWS`** (`:1289-1327`): the collapse branch is
  untouched; no window, no thumb.
- **approval gate:** never windows ≥44 columns (§3.2), so **no thumb on any
  pinned golden**; below 44 columns it windows and gets a thumb, where there is
  no golden to break.

One-sentence rule: **the thumb appears exactly when, and only when, the
position row does — on `page < row_count` — and costs nothing that the window
did not already cost.**

---

## 7. Tests to add, amend, and keep

All geometry via the real `OperatorApp`, never `_AskHost` (the dock-0 fiction,
`:4251-4268`). Follow the visual-validation recipe in `AGENTS.md:221-373`:
capture `before.svg` from a throwaway worktree (§AGENTS.md:330-336, **never
`git stash`**), then `after.svg`, render with `qlmanage -t -s 1600`, and look.

**Keep, unchanged (14 reveal tests, all non-windowing regime):**
`test_ctrl_e_reveals_the_selected_rows_full_consequence` (`:1223`),
`test_the_revealed_card_is_the_same_height_for_every_selection` (`:1277`),
`test_the_reveal_never_takes_the_last_option_row` (`:1320`),
`test_the_footer_offers_the_reveal_only_where_it_does_something` (`:1391`),
`test_the_reveal_block_is_drawn_under_the_row_it_explains` (`:1432`),
`test_the_reveal_is_advertised_from_the_selected_row_not_any_drawn_row` (`:1495`),
`test_the_reveal_mode_does_not_follow_the_user_to_the_next_question` (`:1615`),
`test_the_reveal_stays_live_when_the_default_view_stops_capping` (`:3503`),
`test_the_reveal_never_strips_the_other_rows_prose` (`:3607`),
`test_the_reveal_never_shows_less_than_the_default_view` (`:3704`),
`test_the_approval_gate_reveal_never_strips_a_consequence` (`:3831`),
`test_the_reveal_says_so_when_it_is_still_holding_text_back` (`:4604`),
`test_a_silently_truncated_reveal_is_caught` (`:4700`). Add nothing to these —
scroll never enters their regime (§5).

**Amend:**

| test | line | change |
| --- | --- | --- |
| `test_the_cap_leaves_the_approval_gate_byte_identical` | `:4227` | Passes **unchanged** — the thumb is not drawn at 100/130/150 (`page==row_count`). Add ONE assertion that `card._layout().page == card.row_count` at each pinned size, so a future regression that starts windowing the approval gate (and thus draws a thumb over it) fails HERE with a readable cause rather than as a mystery column shift. Do not relax the golden. |
| `test_the_cap_leaves_the_short_description_card_byte_identical` | `:4318` | Check whether either pinned size windows. If both are `page==row_count`, unchanged + the same `page` assertion. If either windows, its golden shifts one column (row cut to `width-1` under the thumb) and must be re-pinned from a looked-at frame. |
| `test_the_position_row_is_only_drawn_when_the_list_is_windowed` | `:767` | The thumb and the position row share the `page < row_count` condition. Extend this test (or pair it with a sibling) to assert the thumb column is present iff `show_position`, so the two overflow signals can never drift apart. |
| `test_a_click_on_a_row_selects_and_answers_with_it` | `:378` | If it is exercised in a windowing state, the row is now cut to `width-1` under the thumb; re-derive the click target from `_line_rows`/`_index_at` (the map the hit-test actually reads) rather than from column arithmetic, so the thumb's column cannot land a click on the wrong row. |

**Add (new):**

1. `test_the_thumb_appears_only_when_the_list_windows` — sweep sizes for one
   long list; assert the thumb glyph is present exactly when
   `page < row_count`, absent otherwise, and that its presence tracks
   `show_position` one-for-one.
2. `test_the_thumb_tracks_the_scroll_offset` — page/arrow through a windowed
   list and assert `_scrollbar_thumb(row_count, page)` top/len move
   monotonically with `_offset`, top at 0 when `_offset==0`, top at
   `track-thumb` when `_offset==max_offset`.
3. `test_the_thumb_never_widens_the_card` — the frame width with the thumb
   equals the card width without it (row cut to `width-1`, glyph in the freed
   column); assert no body line exceeds `layout.width`, reusing the property in
   `test_no_row_overflows_the_card_at_any_width` (`:443`).
4. `test_page_down_and_up_move_a_page_and_clamp` — `PageDown` moves the cursor
   `max(1, page-1)` rows and clamps at the last row (no wrap); `PageUp`
   mirrors it and clamps at 0.
5. `test_paging_keeps_the_cursor_visible` — after every page and arrow the
   selected row is within `[_offset, _offset+page)`; pin the autoscroll
   invariant `_move_to` provides.
6. `test_the_wheel_scrolls_the_windowed_list` — the existing wheel handler
   moves the cursor and scrolls the window on a windowed list, and the thumb
   follows. (If a wheel test already exists, extend it to assert the thumb
   moves; if not, add it.)
7. `test_the_approval_gate_is_byte_identical_with_the_thumb` — the strongest
   guard for §3.2: mount `ApprovalPrompt` via the real app at 100/130/150,
   assert the frame equals the pre-change frame AND that no thumb column is
   present. This is distinct from `:4227` (which pins against the CAP); this one
   pins against the THUMB, and both are kept for the same reason the two wrap/cap
   goldens are (`:4243-4246`).

---

## 8. Slicing plan

**One slice, one owner, one file.** The whole change lives in
`ask_picker.py`. It could be described as two strands — the thumb (render +
new methods) and paging (bindings + one action) — that touch largely disjoint
regions, but they are the same file, and the team rule is one owner per file
(slices touching the same file run in sequence, `AGENTS.md` collaboration
brief). Two coders in `ask_picker.py` at once is the defect, not the
parallelism. So this is not a two-coder job.

Sequence it **internally**, B before A:

1. **B — paging** (`BINDINGS:525-550` + `action_page`). No allocator change, no
   byte-identity risk, small. Land and verify with tests 4–5 first so the
   navigation is solid before the render change.
2. **A — thumb** (constants + `_scrollbar_thumb` copied from
   `usage_panel.py:1088-1103` + a trimmed `_paint_scrollbar` + the `page <
   row_count` overlay in `_card_text:2353-2357`). This carries the
   byte-identity risk, so it lands second with the full before/after
   visual-validation recipe and the golden re-run named in §7.

Then, **concurrently:**

- **reviewer** reads the diff for C1–C6, the `page < row_count` guard, the
  no-gutter decision, and that the wheel/click hit-test still resolves rows
  correctly under the thumb column.
- **qa** exercises the running app: page through a 20-option question, confirm
  autoscroll keeps the cursor drawn, confirm the thumb tracks and disappears
  when the list fits, and — the safety case — confirm the approval gate at
  100/130/150 is byte-identical and thumbless.
- **designer** looks at rendered frames (the recipe, `AGENTS.md:221`): the thumb
  on a windowed ask card, its absence on a fitting card, and the approval gate
  before/after. This is the only surface that can say the thumb reads as a
  scrollbar and not as a border.

DevOps: nothing — no pipeline, container, or release surface changes. Runtime
publication (`lop-update`, `AGENTS.md:160-186`) is a separate final step only if
the user asks to make it live.

---

## 9. Probes

Measured against the real `OperatorApp`, this worktree, HEAD `3b1281b5`:

- `/tmp/probe.py` — the committed repro (`scripts/ask_user_repro.py` fixture,
  4 options) at 190×50 / 150×40 / 130×30 / 100×30: `page == row_count == 4` at
  every size (never windows), `desc_rows` non-empty where the budget affords it,
  `reveal_wrap` lengths, footer ladder. Confirms the 4-option repro does not
  window and so exercises the reveal regime, not the scroll regime.
- `/tmp/probe2.py` — a 12/13-row question at 100×20 and 100×28, arrowed 0/3/6/9/11:
  `_offset` and `_window` track the cursor (autoscroll), `desc_rows == {}` in
  every windowed frame.
- `/tmp/probe3.py` — the windowed sweep (12 options at 120×24) and the approval
  shape (3 consequences, `allow_free_text=False`) at 100/130/150/60/44: proves
  windowing ⇒ `desc_rows=={}` and `reveal_rows==0`, and that the approval gate
  windows only below ~44 columns, so the thumb never touches a pinned golden.

Re-run any of them with `env -u NO_COLOR TERM=xterm-256color
.venv/bin/python /tmp/probe*.py` before trusting a number here.

---

## 10. Risks to watch during rollout

- **A windowed golden shifting one column.** The only byte-identity change is
  windowed rows cut to `width-1` under the thumb. Any golden pinned on a
  windowed frame must be re-pinned from a looked-at frame, not regenerated
  blind. The approval and short-description goldens are the ones to check
  first (§7); both are believed non-windowed at their pinned sizes, but that is
  a check, not an assumption.
- **The `page < row_count` guard drifting from `show_position`.** If a later
  change makes one true without the other, the card shows a thumb with no count
  or a count with no thumb. Test 1 pins them together; keep it.
- **A future change that windows the approval gate.** Today it cannot above 44
  columns, and the whole no-gutter safety argument (§3.2) rests on that. The
  `page == row_count` assertion added to `:4227` is the tripwire — if someone
  lengthens a consequence string or adds a fourth option, that test fails and
  says why before a mystery column-shift reaches a golden.
- **The 1-cell thumb inviting a drag request.** It is deliberately display-only
  (§3.4). If a drag is later asked for, it must not be a bare 1-cell target on
  a card that answers on click — it needs usage_panel's blank-tail grab
  discipline (`:78-93`) or it will authorise tool calls by misclick.
