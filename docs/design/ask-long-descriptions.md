# Design: readable long descriptions on the `ask` card

Status: implemented. Base: `origin/main` @ `b64eb01c`; all `file:line` are
against `local_operator/tui/widgets/ask_picker.py` unless stated.

This is the consolidated record. The work ran through several rounds against
user feedback and the intermediate proposals are in the PR's commit history;
what follows is the design that shipped and the reasons that still hold.

## 0. The problem

`_description_text` rendered every option description as one
`Text(no_wrap=True, overflow="ellipsis")` line. Measured on a four-option
question with paragraph-length descriptions:

| size | what the descriptions did |
| --- | --- |
| 190x50 | one line each, all four end in `…` |
| 150x40 | one line each, all four end in `…` |
| 130x30 | **dropped entirely** — the allocator bought them all-or-nothing |

The prose the user is being asked to compare was unreachable at every terminal
size the app supports. The question itself already wrapped and was never
truncated (`_question_lines`), so descriptions were the inconsistency.

**The constraint that shaped everything below:** `ApprovalPrompt`
(`approval.py:980`) subclasses this card, and there an option's description
*is* the consequence of authorising a possibly destructive tool call. Its
frames are pinned byte-identical at 100x30 / 130x30 / 150x40, and it must never
hide an answer.

## 1. What shipped

**A two-line clamp, `ctrl+e` to expand, and a list that scrolls.**

1. **`DEFAULT_DESC_CAP = 2`.** Every description clamps to two wrapped lines in
   the list. An earlier round wrapped descriptions into all the leftover budget
   and produced a wall of text — 24 body rows at 190x50, no blank line between
   an option's paragraph and the next option's label, labels unfindable. It was
   rejected on a rendered frame. A picker's first job is to be a list.

2. **`ctrl+e` lifts the selected row's cap** from `DEFAULT_DESC_CAP` to
   `REVEAL_MAX_ROWS`, in place, inside the one viewport (`_cap_for_row`). It is
   not a separate panel: an earlier design reserved a constant-height block for
   the revealed text, which meant two regions competing for one viewport — the
   failure mode this file had already hit twice. The cap-lift deletes that
   machinery (~90 lines including the `affords_column` search) rather than
   porting it. `_move_to` owns the single scroll, so there is nothing to fight.

3. **Line-granular windowing.** The viewport is measured in visual *lines*, not
   whole option rows (`_build_line_list`, `line_start_by_row`). This is what
   lets a long list scroll *and* keep every visible option's description —
   previously the two were mutually exclusive, because windowing forced the
   description column off entirely.

4. **A scrollbar thumb and `PageUp`/`PageDown`.** The thumb's arithmetic is
   `usage_panel`'s (`_scrollbar_thumb`), reused rather than reinvented. It is
   keyed on `show_position`, the same overflow decision that draws
   `showing X–Y of N`, so the two signals cannot disagree — they diverge from a
   raw `page < row_count` test at tight heights where the D1 collapse drops the
   count row.

5. **`MAX_QUESTION_ROWS = 4`.** A long question could wrap far enough to leave
   the option list a one- or two-line viewport, in which a clamped description
   was cut *and* `ctrl+e` was correctly refused — text unreachable by any
   gesture. The question is now bounded and its cut marked. The safety ordering
   (the question is shown first and outranks the options) is untouched; only
   how many rows it may take before the options compete.

6. **`▸ RECOMMENDED`** in bold `fg`. Every chromatic token failed the WCAG AA
   floor on the light theme against the card's `overlay` ground (amber 3.97,
   violet 3.62, blue 3.54), so the badge earns its salience from weight, case
   and glyph rather than hue, leaving the accent's one-thing rule intact.

## 2. The invariants

These are the properties the tests pin, and the reasons they exist.

**C1 — exact budget.** Every line the plan implies is bought from `remaining`,
and the viewport draws `min(budget, len)`. Overdrawing is how the footer — the
only statement of how to leave a card the turn is parked on — was once clipped
off the tail. Property-tested across a size sweep.

**C2/C3 — the priority order.** The footer is bought first, then the question,
then one option row. The question outranking the options is a *safety*
property, not a preference: an approval prompt that names `Allow` without
naming what it would allow looks answerable and is worse than no card.

**Never hide an answer.** `_labels_must_all_fit()` is `False` on the ask picker
(a long list *should* scroll) and `True` on `ApprovalPrompt`: there the
description cap drops 2→1→0 to keep every option's label on screen, and the
list windows only if even the bare labels overflow. Without it the boot-layout
gate (dock 8, an approval can be the first thing a session shows) scrolled
*Deny* and *Allow all* off the first frame.

**Never truncate in silence.** Every cut is marked — the default two-line
clamp, the revealed row, and the bounded question all end in `…` when text
continues. `_mark_clipped` restamps the last *visible* line when the viewport
cuts a row below, and only ever runs on a description line: run on a label it
would strip the cursor glyph and the number gutter and repaint the label as
unselected prose.

**Where the text is still unreachable**, the terminal is physically too short
for any expansion; the mobile relay ships the full description regardless, and
that escape hatch is asserted.

## 3. Two things worth knowing before changing this file

**Measure against the real dock.** The lightweight test hosts declare no
`#input-shell`, so `_dock_reserved_rows` returns 0 where the real app returns
5–8, and they render a card several rows taller than the app ever draws. A
byte-identity guard on such a host is measuring a frame that does not exist.
Every geometry claim runs through `OperatorApp`.

**A green suite is not evidence here.** Four defects shipped past a fully green
suite during this work and were caught by rendering the frame: descriptions
losing their tail behind the scrollbar column, a revealed row clipped with no
marker, the approval gate scrolling a label away, and the clip marker
corrupting a label. The visual-validation recipe in `AGENTS.md` is the check
that matters; `scripts/ask_user_repro.py` and `scripts/ask_scroll_shot.py`
reproduce the frames.
