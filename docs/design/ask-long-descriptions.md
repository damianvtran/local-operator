# Design: reaching an option's full description on the `ask` card

Status: proposal for implementation. Author: architect.
Base: worktree `lo-ask-scroll`, branch `fix/ask-long-text`, on `origin/main`
v0.44.23 (`b64eb01c`). Every `file:line` below is against that tree.

Every number in this document was measured in this worktree, with the probes
named at the end of §11 — not estimated from the code.

---

## 0. The problem, as the frames actually show it

`_description_text` (`ask_picker.py:2003-2047`) builds each description as one
`Text(no_wrap=True, overflow="ellipsis")` and `truncate_cells`-es it into
`width - indent`. One line per row, always. The allocator buys those lines
all-or-nothing, one per row (`:1361-1364`).

Measured on the committed repro (`scripts/ask_long_shot.py`, four long options
plus the free-text row):

| size | card width | body budget | what the descriptions do |
| --- | --- | --- | --- |
| 190x50 | 186 | 20 | one line each, all four end in `…` |
| 150x40 | 146 | 20 | one line each, all four end in `…` |
| 140x36 | 136 | 17 | **gone entirely** |
| 130x30 | 126 | 13 | **gone entirely** |
| 100x30 | 96 | 13 | gone; list also windows to 2 of 5 |

The full descriptions need 4, 3, 2, 2, 1 wrapped lines at 146 cells — 12 lines
against the 5 the card spends. The prose the user is being asked to compare is
unreachable at every terminal size this app supports.

Meanwhile the QUESTION wraps and is never truncated (`_question_lines`,
`:1085-1091`, "Never truncated: it is what is being asked"). The inconsistency
is real, and the descriptions are on the wrong side of it.

### 0.1 The "blank space on the card" is not spare room — this is the crux

The user's report says there is blank space left at 150x40 while the text is
being cut. There is blank space on the SCREEN; there is none on the CARD.

Composited frame at 150x40 (`/tmp/frame.py`): rows 1-10 are the transcript,
rows 11-32 are the card, rows 33-39 are the composer and status band. The card
draws 20 body lines into a budget of exactly 20:

```
budget = min(available=28, anchored=20, wanted=21) = 20
used   = 2 title + 5 question + 1 spacer + 5 rows + 5 descriptions + 1 spacer + 1 footer = 20
```

The card is at its ceiling. `anchored=20` comes from `PROMPT_HEIGHT_SHARE`
(`:136`) — the anchoring rule that exists precisely so the conversation the
question is ABOUT stays on screen (`:113-136`, and
`test_the_conversation_stays_readable_behind_a_question` at
`test_ask_picker.py:1166`). The blank the user sees is the transcript's
reserved share plus the dock's own rows.

**Consequence, and it is the whole design:** a fix that simply wraps every
description needs 12 lines where 5 are spent, and the seven extra rows can only
come from the conversation. That is the modal behaviour this surface was
rewritten to remove. Wrap-everything is not free at any size; §2 costs it.

### 0.2 The approval gate is not affected by the bug, and constrains the fix

`ApprovalPrompt` (`approval.py:980`) subclasses this card, and its consequence
text is the description. Measured: those three strings are 37, 36 and 28 cells
(`APPROVAL_CHOICES`, `approval.py:920-924`) and wrap to `[1, 1, 1]` at every
width down to 44 columns. **The approval card never truncates a description
today.** It reaches `[2, 2, 1]` only at 40 columns and `[3, 3, 2]` at 24.

So the approval gate is not what we are fixing — it is what we must not break.
It is the surface where a description "decides an authorisation"
(`:2014-2019`), and it is the reason two otherwise-attractive options are
rejected below: anything that stops drawing a non-selected row's consequence,
or that makes the card's height move as the cursor moves, degrades the frame
that today is correct.

---

## 1. Recommendation

**Two changes, in this order of importance:**

**(A) Spend leftover budget on continuation lines — a pooled, deterministic
wrap that costs nothing at sizes where the budget is tight.** Descriptions
still get their first line all-or-nothing (C5 survives, restated in §3.3);
lines beyond the first are bought from whatever `remaining` is left after every
other step, selected row first. No new keys, no new state, no reflow on cursor
movement.

**(B) An explicit, keyboard-reachable reveal on `ctrl+e`** that trades the
one-line-per-row list for the full capped description of the selected row,
inside a CONSTANT reserved block so the footer never moves.

(A) alone fixes 190x50 completely and 150x40 partially (the first description
goes from 1 line to 3 of its 4). (A) alone does nothing at 130x30, where the
budget cannot even buy first lines — which is why (B) is needed. (B) alone
would leave a roomy terminal drawing five ellipsised lines with rows to spare,
which is the reported bug.

**Ship (A) first and independently.** It is small, it is pure allocator, it
changes no keymap, and it removes the ellipsis at the sizes with room. (B) is
the second slice and carries all the new interaction risk.

### Why not the alternatives

| option | verdict | cost, measured |
| --- | --- | --- |
| wrap every description unconditionally | **rejected** | needs 12 lines where 5 exist at 150x40; the 7 extra rows come from the transcript, breaking `PROMPT_HEIGHT_SHARE` and `test_the_conversation_stays_readable_behind_a_question` (`:1166`). At 100x30 it needs 16 against a budget of 13, so the list windows to 2 of 5 — it trades unreadable prose for hidden ANSWERS, which the priority order (`:1206-1220`) ranks strictly worse |
| wrap-selected-only / expand-on-select (height follows the cursor) | **rejected** | the card's height would change on every `↑`/`↓` (measured: 17/19/24 lines for different selections at 190x50 under scheme S1). The dock re-lays out and the transcript moves under the user mid-answer. This is exactly what `settings_view._paint_detail` (`settings_view.py:3044-3056`) refuses to do — "a detail line that appeared and disappeared would move the footer and the whole body with it on every cursor move" — and AGENTS.md:622 ("Rows are load-bearing"). It is also the option with the termination risk in §4.3 |
| full-width cursor-following detail row | **rejected** | buys ~148 cells at 150 columns against the ~141 a description already gets (`width - GUTTER_CELLS - NUMBER_CELLS`) — a 5% gain against a 4x deficit. It solves nothing here, and it is a second mechanism beside the description line rather than a fix to it |
| hover-only reveal (the user's suggestion) | **rejected as the MECHANISM** | gates the text that decides an approval behind a mouse. No comparable system does this; scout confirmed oh-my-pi's hover only applies a style band. A keyboard user, a user over ssh without mouse reporting, and every screen reader lose the consequence text. See §6 for what hover may do as a supplement |
| explicit reveal keybinding | **adopted, as (B)** | one key, one line of footer, no reflow. Precedent: Claude Code's `ctrl+e` on its permission prompt |
| do nothing | **rejected** | the text is unreachable at every size; the user's only workaround is full-screening the terminal |

### Where I diverge from the framing

The brief asks whether to adopt oh-my-pi's model wholesale: wrap onto
continuation rows and scroll the list in VISUAL ROWS rather than items. I
recommend **taking the wrapping and NOT the visual-row scrolling**, and the
reason is a measured property of this card rather than a preference.

oh-my-pi's `select-list.ts` is a list that owns its viewport. This card is a
band whose height is capped by the conversation behind it (`_body_rows`,
`:1093-1193`), and whose `page` is already row-count-agnostic (`:270-272`).
Converting `page` from ROWS to LINES means the number of ANSWERS on screen
starts depending on how long their prose is — at 100x30 the same question would
show 5 rows or 2 rows depending only on description length. The priority order
(`:1206-1220`) ranks option rows strictly above descriptions, and visual-row
budgeting inverts that ranking silently. Keeping `page` in rows preserves C2,
C6 and every existing windowing test; the pool in §3.4 gets the wrapping
benefit without the inversion.

---

## 2. Finding: `_layout`'s one-step settle is ALREADY unsound

This is a pre-existing defect, found while checking whether the proposal's
extra allocator step could oscillate. It is not caused by this change, it
blocks a clean proof for it, and it should be fixed in slice 1.

`_layout` (`:1252-1268`) reasons:

> Taking a row back can only shrink the page, so this settles in one step
> rather than looping

That is false. Exhaustive sweep over the REAL `_allocate` (`/tmp/mono.py`,
2-7 rows x budgets 0-25 x 1-5 question lines, with and without the free-text
row): **60 cases where `page` GROWS when the position row is bought.**

Mechanism: `show_title` is a step-6 purchase of TWO rows (`:1349-1351`). Buying
the position line takes one row from `remaining`, which drops `remaining` below
2, so the title is no longer affordable — and the two rows the title gives back
buy MORE option rows than the one the position line cost.

Rendered symptom, at 100x12 with a two-option question (`/tmp/mono3.py`):

```
Ship it?
❯ 1. Yes
  2. No
showing 1–2 of 2          <-- both rows are drawn; nothing is hidden
↑↓ move · 1-9 jump · enter answer · esc skip
```

`show_position` is bought on the `page < row_count` trial, then the retry
returns `page == row_count`, and the renderer draws a row that says the card is
hiding answers it is not hiding. That is the same class of defect as R11
(`:1749-1753`), inverted.

**Fix (slice 1, ~4 lines):** after the retry, keep `windowed` only if it is
still short of the list.

```python
if windowed.question or not plan.question:
    # ...and only if the retry is still windowing. Buying the count can make
    # the title unaffordable, and the two rows the title gives back buy more
    # option rows than the count cost — so the retry can come back showing the
    # WHOLE list while still carrying `showing 1–2 of 2` (measured at 100x12,
    # 60 cases across budgets 0-25). The count is only honest about a window.
    if windowed.page < self.row_count:
        plan = windowed
```

With that guard the one-step settle is sound, and it stays sound for the plans
this design adds (§4.3).

---

## 3. The allocator change

### 3.1 What is added to `_CardLayout`

Replace the single `show_descriptions: bool` with the flag PLUS a per-row line
grant:

```python
#: Lines each DRAWN row's description may use, by row index. Absent or 0 means
#: the row draws no description at all. Replaces the old all-or-nothing bool
#: for lines BEYOND the first: the first line is still all-or-none across the
#: window (C5), because a list where only some rows have any prose reads as
#: broken. Beyond that, a row is allowed to be taller than its siblings — the
#: continuation lines are indented under the row they belong to, so they read
#: as one paragraph rather than as a second row.
description_rows: dict[int, int]
```

Keep `show_descriptions` as a derived property (`any(description_rows.values())`)
so `_row_text`'s recommended-badge branch (`:1988`) and the tests reading it do
not change shape.

### 3.2 The per-row line-count function

```python
#: The most lines one description may ever take, however much room there is.
#:
#: Six, not unbounded: the cap is what stops ONE verbose option from pushing
#: every other option's prose off the card. Measured on the repro at 24
#: columns, the four descriptions want 34, 25, 19 and 18 lines — an uncapped
#: grant would spend a 13-row budget on a third of one option's prose and show
#: nothing about the other three. Six lines is about 140 words at 100 columns,
#: which is longer than any description in this repo (the approval gate's are
#: one line at every width down to 44 columns) and long enough that hitting the
#: cap means the model wrote an essay, not a consequence.
DESC_MAX_ROWS = 6

def _description_lines(self, index: int, width: int) -> list[str]:
    """One description wrapped into the card's own cell model, capped.

    `wrap_cells` and NOT a wrappable `Text`: `Content.from_rich_text` discards
    `no_wrap`/`overflow` when a `Text` crosses into a widget (see
    command_picker.py:31-39, and why `_cut_row` exists at :2202), so handing
    Textual a wrappable Text would let the card set its own width. Every line
    is cut to the card's width by `_fit_row` exactly as the single line is
    today.
    """
```

- indent = `GUTTER_CELLS + NUMBER_CELLS + (cell_len(CHECK_ON) if multi else 0)`
  — identical to `_description_text:2032`, so continuation lines sit under the
  label, not under the number.
- the recommended row's first line carries `RECOMMENDED_TAG · ` as it does
  today (`:2037-2044`), so its budget for prose is `room - len(tag) - 3`; the
  tag is charged once, to the first line only.
- when the wrap is longer than the grant, the LAST KEPT line is ellipsised via
  `truncate_cells` — the same "say that it continues" discipline the question
  already uses (`:1337-1344`). Only the last line is marked, never every line.

Line count for row `i` given grant `g`: `min(g, len(wrap), DESC_MAX_ROWS)`.

### 3.3 What happens to C5

C5 (all-or-nothing, one line per row, `:1361-1364`) is **kept for the first
line and dropped for the rest**, and the rationale in the comment survives
intact. Its stated reason is visual consistency — "a list where only some
entries have their second line reads as broken rather than as abbreviated."
That failure is about a row having NO prose while its sibling has some. A row
whose paragraph is three lines next to one whose paragraph is one line is not
that failure; it is what every wrapped list looks like, and the indent makes
the grouping unambiguous.

So the invariant becomes: **every drawn row has a first description line, or
none of them do.** The step-9 test is unchanged; steps 10 and 11 are new.

### 3.4 The new priority order

Steps 1-8 are untouched. The order gains two rungs at the bottom and, for
reveal, one in the middle:

1. the footer
2. the first line of the question
3. one option row
4. the windowing line
5. the rest of the question
6. the title and its rule
7. the rest of the option rows
   - **7a. (reveal only) the reveal block** — see §4
8. the blank spacers
9. **the descriptions' FIRST lines, all of them or none** *(C5, unchanged)*
10. **continuation lines for the SELECTED row, up to its wrap and `DESC_MAX_ROWS`**
11. **continuation lines for the remaining drawn rows, in window order**

Steps 10 and 11 spend only what steps 1-9 left. They cannot take a row, a
spacer, the title, the question or the footer, because those are already
bought. This is what makes the change safe at small sizes: **where the budget
is tight, `remaining` is 0 after step 9 and the frame is byte-identical to
today's.**

Selected-row-first (step 10 before 11) rather than round-robin: measured at
150x40 both grant `[4,3,2,2,1]`, but where the pool is short the user is
reading the row under the cursor, and a round-robin spreads the shortfall
across four rows so that none of them is complete. Deterministic in the
cursor's position only, which the paint already depends on.

### 3.5 The exact-budget invariant (C1)

Every step above is a `remaining -= n` guarded by `remaining >= n`, in one
forward pass, exactly as `:1332-1364` already is. Verified exhaustively
(`/tmp/mono.py`, 351,648 plans over budgets 0-39, 1-11 question lines, 1-8
rows, wrap profiles up to 34 lines, both reveal states, cursor at every row,
offsets at both ends): **0 overdraws.** The renderer's implied line count
equals `budget - remaining` in every case.

C2 (priority order) is preserved — the new rungs are at the bottom. C3
(question outranks options) is untouched — steps 1-5 are not modified. C4
(footer bought first) is untouched. C6 (`page` counts ROWS) is untouched, which
is why `_window`, `_move_to` and `_index_at` need no change at all.

---

## 4. The reveal (B)

### 4.1 The gesture

`ctrl+e`, bound on the card, named in the footer as `^e more` / `^e less`.

- **`ctrl+e` and not hover:** §1. Hover may ADD to it (§6), never replace it.
- **`ctrl+e` and not `enter`-to-expand** (the `settings_view:1042-1078`
  pattern): `enter` ANSWERS here. On the approval card it authorises a tool
  call. Overloading it is not available.
- **`ctrl+e` is free:** audited. It is not in `app.py`'s bindings
  (`app.py:1620-1717`); the only mention is a comment noting `TextArea` binds
  `end,ctrl+e` (`app.py:1707`) — and the composer does not have focus while
  this card does (`:296-301`). `on_key` (`:768-792`) only swallows printable
  single characters, so it does not intercept it.
- It is a per-question toggle held in `_QuestionState` (`:236-246`), so moving
  between questions in a multi-question ask does not silently carry the mode
  over, and moving back restores it.

### 4.2 Constant reserved height — the property that makes it safe

The reveal block reserves `E = min(DESC_MAX_ROWS, max(wrap lengths over the
whole list), remaining)` — **the tallest capped description in the list, not
the selected row's.** The selected row draws up to `E` lines and the remainder
is padded blank.

This costs a few blank rows on the shorter options and buys the property that
matters: **the card's height does not change as the cursor moves.** Measured
across every selection, at every size in the sweep, the revealed card has
exactly ONE height (`/tmp/sim3.py`, `stable=True` on all 15 ask sizes and all
15 approval sizes). Without this, the same sweep gives 3 different heights at
190x50 — the footer and the whole dock moving on every arrow press.

This is `_paint_detail`'s rule (`settings_view.py:3044-3056`) applied to a
block instead of a line, and AGENTS.md:622-625 ("Animated content must reserve
its row even when it has nothing to show").

### 4.3 Termination

The reveal is bought in ONE forward pass at step 7a, from `remaining`, like
every other rung. There is no feedback loop from the drawn height back into the
budget, so there is nothing to oscillate: `_body_rows` reads the screen and
`wanted`, never the previous plan.

The one place a fixed point is computed is `_layout`'s position-line retry
(`:1252-1268`), and that is the pre-existing unsoundness in §2. With the §2
guard, the sweep confirms the retry never grows the page under any reveal
state. **This is the "prove termination or use a bounded loop" requirement
discharged by making the pass forward-only rather than iterative**, which is
strictly stronger than a bounded loop and matches what the file already does.

Note the reveal is bought AFTER the option rows (step 7a, not 6a). Buying it
earlier was measured and rejected: at 130x30 it collapsed the list from 5 rows
to 1 (`/tmp/sim2.py`). Revealing one row's prose must never take another row's
LABEL off the card — that is C2's ranking, and it is a safety property on the
approval gate.

### 4.4 What the reveal achieves, measured

`/tmp/sim3.py`, repro question, `shown/want` for the selected row:

| size | budget | default frame | revealed |
| --- | --- | --- | --- |
| 190x50 | 25 | 3,2,2,2,1 — **already complete via (A)** | 5/5 rows complete |
| 150x40 | 20 | 1,1,1,1,1 (pool exhausted) | 5/5 complete, height 19 ≤ 20 |
| 140x36 | 17 | no descriptions | 5/5 complete |
| 130x30 | 13 | no descriptions | 4/5 complete; list windows to 1 row + position line |
| 120x30 | 13 | no descriptions | 2 of 4 lines shown |
| 100x30 | 13 | no descriptions | 1 of 5 lines shown |
| 100x24 and below | ≤8 | no descriptions | **reveal buys nothing; frame unchanged** |

At 130x30 and below the reveal trades option ROWS for prose. That is a
deliberate trade the user made with an explicit keypress, it is honest (the
position line is bought and says `showing 1–1 of 5`), and it is reversible with
the same key. It is not something the card does on its own.

**On the approval card the reveal is a no-op wherever the frame is already
correct** — `[1,1,1]` at every width to 44 columns, so `E = 1` and the revealed
frame equals the default frame. It only does anything at 24-44 columns, where
it recovers a consequence line the default frame has dropped. This is the
safety case satisfied: the gate never gets worse and sometimes gets better.

---

## 5. The degradation ladder

In the file's own idiom — bought in priority order, and what goes first:

- **≥190 cols, ≥50 rows** (budget 25): every description complete via (A)
  alone. No ellipsis anywhere. `^e` is a no-op the footer still advertises.
- **150x40** (budget 20): first lines for every row, then continuations to the
  cursor's row until the pool runs out. `^e` completes all of them.
- **140x36** (budget 17): the pool is 2 and cannot buy 5 first lines, so C5
  drops all descriptions — as today. `^e` recovers the selected row in full.
- **130x30** (budget 13): no descriptions; `^e` windows the list to 1 row and
  shows 3 of 4 lines, with `showing 1–1 of 5` bought.
- **100x30 / 100x24 / 100x20** (budget 13/8/6): the list itself is windowing.
  Descriptions are unaffordable in both modes below budget 13, and `^e` buys
  E = 1 or 0. The frames are IDENTICAL to today's at 100x24 and below.
- **100x13** (budget 1) and **100x12** (budget 0): `budget < MIN_BODY_ROWS`;
  the collapse branch (`:1289-1327`) is untouched. Question-then-exit, then
  nothing.
- **24-190 wide sweep** (`test_no_row_overflows_the_card_at_any_width:443`):
  every continuation line goes through `_fit_row` (`:2217`), so it is cut to
  the card's width and padded in the row's own ground exactly as the current
  single line is. Verified in the sweep: no plan exceeds its budget at any
  width.

The one-sentence rule: **continuations are the last thing bought and the first
thing lost, and the reveal outranks nothing except the OTHER rows' prose.**

---

## 6. Scroll affordance, and hover

**No scroll affordance for descriptions. Do not add one.**

With `DESC_MAX_ROWS = 6` and the reveal, the only text still unreachable is a
description longer than six wrapped lines on a terminal too small to show six
lines. At that point the card is already windowing its ANSWERS (measured:
budget ≤ 8 at 100x24 and below), and a second scrollable region inside a card
that is itself windowing is two gestures competing for one viewport — the
defect AGENTS.md:577-607 documents at length ("One gesture owns the viewport at
a time"). The wheel is already bound here and moves the CURSOR
(`:818-824`); making it mean something else inside a sub-region would break
that rule on the surface that can authorise a tool call.

Where text is still cut, the last kept line ends in `…` and the phone shows it
in full (the mobile relay reads the raw model, PR #266). That is an honest
degradation with an existing escape hatch.

**How the user learns `^e` exists:** it goes in the footer ladder
(`_footer_hints:2049-2167`) as `("^e", "more")` / `("^e", "less")`, inserted
before the exit hint. Footer budget, measured:

- single-select today: `↑↓ move · 1-9 jump · enter answer · esc skip` = 44 cells;
  with `^e more` = 54 cells.
- approval today: `↑↓ move · enter answer · esc deny` = 33; with `^e more` = 43.

So the hint is free above ~56 columns and enters the shed ladder below it. It
must be placed in the `ladder` list **immediately before the exit** — i.e. it
sheds its word, then itself, before `esc` loses anything — preserving the D3 /
D16 / D19 rule that `esc skip` survives to the narrowest card
(`:2103-2118`). Concretely: `ladder = ["↑↓", jump, "enter", "^e", "esc"]`.

**Offer it only where it does something.** The footer already refuses to
advertise keys that are dead (`:2151-2164`, `1-9 jump` on a single-row window;
`:2073-2083`, the whole keymap while the composer has focus). `^e` is dead
when no drawn row's description exceeds the lines it is already granted. So:
show the hint only when `any(len(wrap_i) > granted_i)` in the current plan, or
when the reveal is already on (to advertise `less`). That is one predicate, and
it keeps the roomy-terminal frame from advertising a key that changes nothing.

**Hover:** the user asked for hover-to-reveal. Adopt hover as a SUPPLEMENT
only, and only in slice 2 if it is free: `on_mouse_move` already repaints on
hover change (`:859-863`), and `_row_ground` already tints the hovered row
(`:1848-1866`). Hovering a row may set it as the reveal target when the reveal
is ON — it must never be what turns the reveal on, and it must never be the
only way to read the text. If it costs more than a few lines, drop it; the
keyboard path is the deliverable.

---

## 7. Tests to amend

Measured first: the three "blocking" fixtures all use descriptions that wrap to
ONE line at their test widths (`/tmp/testsim.py` — `nothing reads the column`,
`why a`, `holds a lock for forty minutes` are all `[1,1,1]` at width 96). **The
default frame at those sizes is therefore byte-identical, and the full file is
green today (53 passed).** The amendments below are smaller than the brief
feared. Each still needs re-deriving, not deleting.

| test | line | why it must be touched |
| --- | --- | --- |
| `test_a_short_terminal_drops_descriptions_before_it_drops_options` | `:494`, esp. `:520` | **Amend the docstring and add an assertion; do not weaken it.** C5's first-line rule still holds, so `why a` is still absent at 100x20 and every label is still drawn — the test passes unchanged. It must gain a note that "descriptions" now means "first lines", and a new assertion that the reveal cannot buy prose at 100x20 either (E = 0), so the ORDER the test names is not circumventable by the new key. Its own comment says the number is not the contract, the ORDER is — the order is unchanged |
| `test_a_click_on_a_row_selects_and_answers_with_it` | `:211`, `:223-224` | Hard-codes `first_row_line = 2 + 1 + 1` and `+ 2` for the second row — two-lines-per-row arithmetic. It passes unchanged for this fixture, but it is now arithmetic that is only true when every grant is 1. **Re-derive it from `_line_rows`**: find the first index where `card._line_rows[i] == 1` and click that. That is the map `_index_at` actually uses (`:870-904`), so the test then pins the hit-test rather than a layout coincidence |
| `test_no_row_overflows_the_card_at_any_width` | `:443` | Add a long-description fixture to the width sweep. Currently every row is one line, so the sweep never exercises a continuation line through `_fit_row` |
| `test_the_conversation_stays_readable_behind_a_question` | `:1166` | Passes unchanged — (A) spends only leftover budget and cannot raise `_body_rows`. **Add** the revealed state to it: press `ctrl+e`, then re-assert the same proportion. This is the guard that the reveal cannot eat the transcript, and it is the assertion I most want in the tree |
| `test_the_badge_no_longer_shortens_the_option_it_promotes` | `:1115` | Positional — asserts the tag is on the line right after the label. Still true (the tag is on the description's FIRST line). Passes unchanged; verify, do not edit |
| `test_a_secret_question_masks_the_typed_value_and_returns_it` | `:113-121` | Calls `_row_text` with 7 positional args including a `_CardLayout`. `_row_text`'s signature is unchanged by this design, and `_CardLayout` is constructed by `card._layout()` in the test rather than by hand — so it passes unchanged. **Only touched if a coder changes `_row_text`'s signature, which this design says not to do** |

## 8. Tests to add

In `tests/unit/tui/test_ask_picker.py` (QA owns the file):

1. `test_a_long_description_wraps_instead_of_ending_in_an_ellipsis` — at
   190x50 with the repro's descriptions, no drawn description line ends in `…`
   and the full text of option 1 is present across its lines. **This is the
   bug; it must fail on `main`.**
2. `test_a_wrapped_description_never_costs_an_option_row` — at every size in
   24-190 x 12-50, `page` and the set of drawn labels are identical with and
   without the wrapping change. Pins that steps 10-11 are additive only.
3. `test_the_card_never_draws_more_lines_than_its_budget` — C1 as a property
   test over the plan: `implied_lines(plan) <= _body_rows(...)` across a sweep.
   This is the regression guard for the clipped footer.
4. `test_ctrl_e_reveals_the_selected_rows_full_consequence` — press `ctrl+e` at
   150x40, assert option 1's complete description text is present.
5. `test_the_revealed_card_is_the_same_height_for_every_selection` — the §4.2
   property. Iterate the cursor over every row in the revealed state and assert
   one distinct line count. This is the "footer does not move" guard.
6. `test_the_reveal_never_takes_the_last_option_row` — at 130x30 and 100x20,
   `page >= 1` and the position line is bought whenever `page < row_count`.
7. `test_the_approval_cards_consequences_are_unchanged_by_the_wrap` — the
   `ApprovalPrompt` frame at 100x30, 130x30 and 150x40 is byte-identical before
   and after. The safety regression guard.
8. `test_the_position_row_is_only_drawn_when_the_list_is_windowed` — §2, at
   100x12 with a two-option question. **Must fail on `main`** (it does today:
   `showing 1–2 of 2`).
9. `test_the_footer_offers_the_reveal_only_where_it_does_something` — the hint
   is absent at 190x50 (nothing left to reveal) and at 100x20 (E = 0), present
   at 150x40.
10. `test_a_secret_question_still_draws_one_row` — the secret path has
    `row_count == 1` and `SECRET_HINT` as its description (`:1703-1706`);
    assert the wrap and the reveal do not grow it.

Per AGENTS.md:484-494, tests 1 and 8 must be shown to FAIL on the pre-fix tree.

---

## 9. Visual validation (AGENTS.md §"Visual validation")

Mandatory, and it is what actually closes this bug — the report came from
rendered frames and the verdict has to as well.

Capture BEFORE frames first, from a throwaway worktree (AGENTS.md:302-318 —
**never `git stash`**, other agents hold uncommitted work in this checkout):

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/ask_long_shot.py /tmp/before-150x40.svg 150x40 0
```

Sizes to capture, before and after, ask and approval: 190x50, 150x40, 140x36,
130x30, 100x30, 100x20. Plus, for the reveal, 150x40 at rows 0 and 2 in both
states — and consecutive frames at the same size to confirm the card settles
(AGENTS.md:343-349): the first painted frame and the settled frame must be
identical, or the reveal is animating something.

`scripts/ask_long_shot.py` needs a fourth argument for the reveal state; that
is slice 2's work.

---

## 10. Slicing plan

**This is not two parallel slices in the same file, and pretending otherwise
would put two coders in `ask_picker.py` at once.** The allocator, `_CardLayout`,
the renderer and `_footer_hints` are one organ; splitting them across two
owners guarantees a conflict on `_allocate`. So: SEQUENTIAL slices with one
owner each, plus one genuinely independent parallel strand.

**Slice 1 — allocator + wrapping (coder A), `ask_picker.py` only.**
- §2 position-row guard in `_layout` (independent bug, smallest diff, do it
  first so the invariant is sound before anything is built on it)
- `DESC_MAX_ROWS`, `_description_lines`, `description_rows` on `_CardLayout`
- steps 10-11 in `_allocate`; `_card_text` emits N lines per row via
  `newline(index)` per line (`_line_rows` is already multi-line tolerant by
  design, `:421-425` — no change to `_index_at`, `_window` or hover)
- ships alone and fixes the reported bug at 190x50 and partially at 150x40

**Slice 2 — the reveal (coder A, after slice 1 merges), `ask_picker.py` +
`approval.py` if the footer needs it.**
- `ctrl+e` binding, `_QuestionState.revealed`, step 7a, footer hint + ladder
  placement, the shot script's fourth argument
- same owner as slice 1 because it edits the same three methods

**Parallel strand — QA, from the moment slice 1 opens.** `tests/unit/tui/`
only, which no coder touches. QA owns the amendments in §7 and the new tests in
§8, and can write tests 1, 2, 3, 7, 8 against slice 1 immediately (8 and part of
1 fail on `main` today, which is the point).

**Reviewer** reads the diff for C1-C4 and the priority order; **designer**
reads the §9 frames, which is the only surface that can say whether this looks
right.

Realistic order: `slice 1 → (review ∥ QA ∥ designer) → slice 2 → (review ∥ QA ∥
designer)`. The second coder is genuinely idle on `ask_picker.py`; if they must
be used, the honest independent work is the shot-script and docs strand, not a
second editor in this file.

---

## 11. Risks to watch during rollout

1. **The approval gate is the blast radius.** Its consequence text decides an
   authorisation (`:2014-2019`). Test 7 pins its frames byte-identical; if that
   test needs relaxing, stop and escalate rather than updating the expectation.
2. **`_body_rows`'s `wanted` cap (`:1185-1192`) must be updated with the line
   counts**, or the budget is capped below what the plan can now spend and
   every description silently disappears — the exact failure recorded at
   `:1178-1184`. It becomes
   `2 + question_lines + 2 + row_count + sum(min(DESC_MAX_ROWS, wrap_i)) + 1 + 1`.
   This is the single most likely way to ship a regression here.
3. **Cost of wrapping on every paint.** `_layout` is called 3x per paint
   (`:1380`, `:1395`, `:1713`) and `_repaint` runs on every keystroke. Wrapping
   N descriptions per call is new work on the paint path. Memoise
   `_description_lines` on `(index, width)`, invalidated on resize and on
   question advance. Watch `test_tui_responsiveness.py` and calibrate from CI,
   never from a laptop (AGENTS.md:456-482).
4. **`Content.from_rich_text` discards `no_wrap`/`overflow`**
   (`command_picker.py:31-39`). Every continuation line must go through
   `wrap_cells` + `_fit_row`. A coder who "simplifies" this by handing Textual a
   wrappable `Text` will produce a card that sets its own width — the condition
   AGENTS.md calls always a bug.
5. **Grapheme clusters.** `wrap_cells` already measures the finished head once
   for this reason (`transcript.py:187-196`); descriptions are model-authored
   and can carry emoji. The width sweep (test 2) must include one.
6. **Footer shed ladder.** Adding a hint to a row already described as "the
   tightest on the card" (`:190-192`) risks re-opening D3/D16/D19. Assert
   `esc skip` and `esc deny` survive at 18-26 columns.
7. **`_line_rows` and the mobile relay** are both fine by construction —
   `_line_rows` is rebuilt per paint and already tolerates multi-line rows
   (`:421-425`), and the relay reads the model, never the renderer. No action;
   listed so nobody "fixes" them.

### What would change my recommendation

- If measurement shows `_description_lines` on the paint path costs more than
  ~1 ms at 12 options and 190 columns even memoised, step 11 (continuations for
  non-selected rows) should be dropped and only the selected row wrapped — half
  the benefit, a quarter of the wrapping work.
- If the designer's frames show the reveal's blank padding (§4.2) reads as a
  rendering fault rather than as reserved space, the alternative is to reserve
  the block only when `max(wrap) > 1` and accept a height change on the one
  transition into and out of reveal — a single deliberate keypress, not every
  arrow. I would not accept a height change per cursor move.

### Probes used (all re-runnable, none committed)

`/tmp/ask_probe.py` (frames + plan at 11 sizes), `/tmp/ask_budget.py` (budget
arithmetic), `/tmp/frame.py` (composited screen), `/tmp/sim.py`, `/tmp/sim2.py`,
`/tmp/sim3.py` (candidate allocators), `/tmp/mono.py` (351,648-plan invariant
sweep), `/tmp/mono2.py` / `/tmp/mono3.py` (the §2 defect), `/tmp/testsim.py`
(which fixtures change). The committed repro is `scripts/ask_long_shot.py`.
