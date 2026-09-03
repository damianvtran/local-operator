# Design: reaching the text in a long `/btw` aside

**Status:** proposal (design only — no source changed by this document)
**Author:** architect (delivery team)
**Base:** `origin/main` @ `093cf9f4` (v0.44.x)
**Scope of ground truth read:** `AGENTS.md`; the full module docstring and body
of `local_operator/tui/widgets/aside_panel.py`; `local_operator/tui/app.py`
(`BINDINGS` 1740-1791, `_cmd_copy` 7002-7076, `_put_on_clipboard` 6941,
`on_editor_submitted_inline` 7580-7609, `_open_aside` 15480-15530,
`_aside_can_fork` 15744-15757, `action_fork_aside` 15765, `_help_block`
16730/16930-16943, `_scroll_todos` 13772); `local_operator/tui/copy_targets.py`;
`local_operator/tui/widgets/copy_picker.py`;
`local_operator/tui/widgets/usage_panel.py`;
`local_operator/tui/widgets/settings_view.py:1-20`;
`local_operator/tui/widgets/org_chart_view.py:1-9`;
`local_operator/tui/widgets/subagent_view.py:1000-1015`;
`local_operator/tui/local_operator.tcss:1514-1526`;
`tests/unit/tui/test_aside_reachability_investigation.py`;
`tests/unit/tui/test_app_pilot.py:9670-9700`;
`docs/evidence/cmd-chords/MEASURED.md`.

Three new measurements were taken for this proposal and are reported in §2.4,
§3 and §5.3. Everything else is cited to a line.

---

## 1. Problem, as found

The `/btw` aside renders its exchange into a card that is sized to the ground
above the dock. When one turn's own rows exceed that budget, `_body`
(`aside_panel.py:445-492`) keeps the question row plus the newest rows and
**silently drops the middle** (486-492). Nothing in the widget can bring those
rows back, because the only scroll state it has, `_scroll_back`, is a **turn
index**, not a row offset: `_max_scroll_back()` is `len(turn_groups) - 1`
(362-364).

Measured by QA's reachability oracle (`test_aside_reachability_investigation.py`,
`_exhaust_the_wheel`, which drives `_scroll_by` to its clamp in both directions):

| case | reachable | unreachable |
|---|---|---|
| 1 turn × 200 rows, 80×24, no band | rows 184-199 | **184 of 200 (92%)**, `_max_scroll_back() == 0` |
| 8 turns × 30 rows | all 8 questions; last 14 rows of each | **128 of 248** |
| 8 turns × 3 rows (control) | everything | none |

The control is what makes this diagnosable: the wheel works exactly as designed,
and the defect is confined to **a turn whose own rows exceed the body budget**.

These are three distinct defects with different severities, and they do not all
have the same fix. Separating them is the main analytical claim of this
document.

### D1 — Unreachable rows inside one turn (severity: high)

The row model has no unit smaller than a turn, so no gesture — present or
future — can address a row inside one. This is the user's actual complaint and
it is a **correctness** defect: the card displays a subset of the answer and
offers no path to the rest.

### D2 — No overflow marker in the single-turn case (severity: high, cost: trivial)

`aside_panel.py:483`: when there is only one turn, `hidden == 0`, so `_body`
returns the bare `lines[-budget:]` slice and never emits the
`↑ N earlier questions · scroll` marker. **The user is not told anything was
cut.** QA isolated this precisely: the identical 60-row answer *does* announce
itself as soon as a second short turn exists.

This is worse than D1. With D1 alone the user knows text is missing and hunts
for it; with D2 they read a truncated answer as a complete one. It is also the
same failure class the `/copy` work already classified as `MAJOR-1` — "the user
got a half sentence that reads as a complete short answer"
(`app.py:7047-7050`). That precedent was decided; this instance was missed.

### D3 — No keyboard path at all (severity: medium, conditional)

The wheel is the only gesture (`aside_panel.py:346-352`). In-app dispatch is
correct — mouse events route by position, not focus, so `can_focus = False`
does not block them — but the wheel depends on terminal mouse reporting being
negotiated. Under `tmux set -g mouse off`, on terminals with mouse reporting
disabled, under `screen(1)`, and on non-SGR terminals, the wheel never arrives.
In those setups the `↑ N earlier questions · scroll` marker names content with
**literally zero way to reach it**, because the card has no `BINDINGS` and
`can_focus = False` (195).

This is exactly the situation `ctrl+up`/`ctrl+down` were added for on the todo
panel, in a comment that states the general rule (`app.py:1757-1769`):

> …its overflow scrolls inside a non-focusable region (`#todo-scroll`,
> `can_focus = False` for U3), so a focus-then-arrow gesture cannot reach it and
> the hidden todos would be **mouse-only** while the footer implies `ctrl+t`
> reveals everything.

### What makes it worse, and what does not

**The dock band is real but working as designed.** Measured end-to-end through
the real app, a populated todo band costs exactly 6 body rows at both 80×24 and
120×40: budget 9 → 3 at 80×24 (a 200-row answer renders **three rows**), 25 → 19
at 120×40. `overlay.rows_above_dock` (`overlay.py:47-84`) deliberately reads
whatever is docked, and a previous design review that called its disagreement
with `composer_column` a bug **was wrong** (rationale at `overlay.py:62-73`).
This proposal does not touch it. The band makes D1 more acute; it is not a
defect.

**Streaming makes it worse and the anchor rule has nothing to anchor.**
`append_answer` (284-293) repaints on every delta and `_body` is recomputed per
chunk with no caching. `_scroll_back` is deliberately not reset mid-answer
(comment 288-292, mirroring `TailAnchor` at `transcript.py:75-98`) — but since
it counts *turns* and a streaming answer is *one* turn, max scroll-back is 0
during a stream. **The state that comment protects is unreachable within one
answer.** A running turn also appends a trailing `…` row (528-530), costing one
more budget row exactly when the answer is largest.

**`^F` is a real escape hatch that costs the feature's contract.**
`fork_messages` (321-327) returns the full `turn.answer` off the dataclass and
never consults `_body`/`_fit`/`_scroll_back`, so `^F` genuinely recovers the
complete text. But `forkable` requires `state == "done"` (165-168) and
`_aside_can_fork` refuses while `session.is_streaming` (`app.py:15744-15757`) —
precisely when the aside is most used — and forking **permanently writes the
exchange into context and the transcript**, which is the one thing the module
docstring (1-38) promises the surface will not do. `close`/esc discards `_turns`
outright (254-260). Nothing else persists the answer.

### 1.1 Two things the brief asked me to check

**Drag-select reaches the card, but only what is painted.** I measured this
rather than reasoning about it. `AsidePanel` inherits `ALLOW_SELECT = True` from
`Static` (it sets no override, unlike `Chrome` at `app.py:1516`), so the
selection walk *does* include it. Driving a real mouse-down/`MouseMove`/mouse-up
across the card at 120×40 over a 200-row answer, `screen.get_selected_text()`
returned **25 answer rows — exactly rows 175-199, the painted budget — plus the
card's title and rule as chrome**. So drag-select is not a workaround for D1: it
can only ever copy what is already on screen, which is the subset the user can
already read, and it additionally sweeps up chrome the `Chrome.ALLOW_SELECT`
comment (`app.py:1507-1516`) exists to keep out of the clipboard.

**`/copy` has landed, and it cannot reach the aside — twice over.**
`SlashCommand("copy", …)` is at `app.py:502` with `_cmd_copy` at 7002;
`HANDOFF-copy-command.md` in the repo root is **stale** and should not be built
from. But `/copy` is not an escape hatch here for two independent reasons:

1. `build_copy_targets` walks transcript blocks for **assistant answers**
   (`copy_targets.py:383-440`, `_is_assistant_answer`). An `AsideTurn` is a
   dataclass on the card, never a transcript block, so it is invisible to the
   tree.
2. **Even if it were listed, the command is unreachable while the aside is
   open.** `_open_aside` calls `editor.set_commands([])` (`app.py:15508`) and
   `on_editor_submitted` routes a slash-shaped line to the aside *as a
   question* — the comment at 15509-15513 states it: "`/model` became a question
   ABOUT `/model`". The inline path guards the same way (`app.py:7604-7607`).

So a copy escape hatch for the aside needs **a key, not a slash command**. That
materially changes its cost, and §4.4 prices it accordingly.

---

## 2. Options

Five options, including the one I think is wrong and the one the user proposed.
Each states what it fixes, what it does not, cost, risk, the recorded decision
it touches, and whether it survives a no-wheel terminal.

### 2.1 Option A — Row-offset scroll model + a keyboard chord (the user's "just make it scrollable")

Replace `_scroll_back`-as-turn-index with a **row offset** over the flattened
body, the way `UsagePanel` already does it (`_offset` 589, `_max_offset`
1050-1060). Keep the wheel driving it; add `ctrl+pageup`/`ctrl+pagedown` at app
level so the gesture exists without a mouse.

- **Fixes:** D1 completely (every row addressable), and D3 (a keyboard path
  that works with mouse reporting off). Makes the streaming anchor comment
  (288-292) *true* for the first time — a reader who scrolls back mid-answer
  now has somewhere to be.
- **Does not fix:** D2 on its own — the marker logic at 482-492 still has to
  change, though under a row model it becomes simpler, not harder. Does not
  make the text persist after esc.
- **Cost:** moderate, and concentrated in one file. `_body` (445-492),
  `_scroll_by` (354-360), `_max_scroll_back` (362-364), plus two `BINDINGS`
  entries, two actions, and one `/help` row in `app.py`.
- **Risk:** the turn-group structure exists for a stated reason — cutting by row
  "left the top of the card showing a mid-sentence continuation at the answer
  indent with no question above it, which reads as the start of a new answer"
  (425-432). A row model must keep the owning question pinned, which
  `_body:486-491` already does for the single-turn case; the fix generalises
  that rule rather than discarding it. Second risk: `_scroll_back` is reset in
  three places (`open` 249, `close` 259, `ask` 274) whose meaning ("snap to the
  tail") must be preserved under the new unit.
- **Recorded decision touched:** the wheel-only note at 338-342, which rejected
  **↑/↓ specifically** because they belong to the composer's prompt history. A
  free chord *applies* that reasoning rather than overriding it. See §5.
- **No-wheel terminal:** yes, with the chord. Without it, no.

### 2.2 Option B — Promote the aside to a bigger surface (mode / full page)

Expand the card into a full-page mode when the content overflows, following
`SubagentView` / `OrgChartView` / `SettingsView`.

- **Fixes:** D1 and D3 by inheriting a real focusable `ScrollableContainer`.
- **Does not fix:** D2 in the docked state, which is where the user still spends
  most of their time.
- **Cost:** **high**, and the highest-risk option here.
- **Risk — this is the honest problem with it.** Every existing full-page mode
  sets `_set_composer_read_only(True)` (`app.py:11562`). The aside's entire
  premise is that the composer stays live — the module docstring's "No second
  composer" (34-37) and `_open_aside`'s "the aside is a place to keep typing"
  (15525-15528). This would be **the first writable mode in the app**: a real
  design decision, not a free reuse. It also collides with the stated
  mode-vs-modal rule (`settings_view.py:8-18`, `local_operator.tcss:1518-1522`):
  a modal is acceptable only for a surface that never mutates anything, and this
  one takes input. And it inverts a decision made in the opposite direction only
  recently — the subagent page **yields to the aside** for exactly this reason
  (`app.py:15486-15493`).
- **Recorded decision touched:** the composer-stays-live contract, the
  mode-vs-modal rule, and the subagent-yields-to-aside decision. Three, all
  pointing away from this.
- **No-wheel terminal:** yes.

I do not recommend this. It is a large change that trades a recorded contract
for an outcome Option A reaches inside one widget. Worth reconsidering only if
the aside later grows genuinely page-scale content (tables, diffs), which it
does not have today.

### 2.3 Option C — `⟨expand⟩ N more lines` in-place disclosure

Reuse `InstructionBlock`'s idiom (`subagent_view.py:967-1091`): show the tail,
and an affordance row stating the withheld cost.

- **Fixes:** D2 completely and idiomatically — the affordance row *is* the
  overflow marker, and the app's own comment says why it states a number:
  "`⟨expand⟩` alone does not distinguish two more lines from fifty, and that is
  the whole difference between clicking and not bothering"
  (`subagent_view.py:1008-1011`).
- **Does not fix:** D1. **Expanding into what?** The card is bounded by
  `rows_above_dock`; there is no room to expand *into* without either covering
  the composer (which `_fit`/`stack_on_dock` exist to prevent, 383-398, 686) or
  becoming Option B. On the measured 80×24-with-band case the budget is 3 rows —
  expansion has nowhere to go. Without a row-offset model underneath it, expand
  is a button that cannot deliver.
- **Cost:** low.
- **Risk:** low, but it risks being a *worse* lie than D2 — an affordance that
  names 184 hidden rows and then shows you 3 more.
- **Recorded decision touched:** none adversely; it extends an existing idiom.
- **No-wheel terminal:** the affordance yes, the content no.

**Verdict:** valuable as a *presentation* of D2, not viable as a fix for D1. If
Option A lands, the marker it already renders (485) covers the same ground more
cheaply, so C's real contribution is the **"state the cost" principle**, which
§4.2 folds into slice 1.

### 2.4 Option D — Card sizes to content (rejected on measurement)

I costed "let the card grow to content up to the available ground rather than
always tail-cutting", and then read the code: **it already does this.**
`_repaint` pins `styles.height` to `len(rows) + gutter` (664-674) and the
comment at 634-638 states the rule — "the card is sized to its CONTENT and rests
on the composer, so unspent budget is not padding to be printed". The card is
already as tall as its content up to `_fit()`'s budget, and the budget is the
ground above the dock, which is the ceiling that cannot be raised without
covering the composer.

**This option does not exist.** I include it because it was on the brief's
candidate list and a reader will otherwise re-derive it; the answer is that the
behaviour is already implemented and the remaining gap is the budget ceiling,
which is Option A or B.

### 2.5 Option E — A copy escape hatch for the aside (`ctrl+y`), off the record

Extend the copy machinery so the **full** text of an aside answer can be lifted
to the clipboard without forking it into context. Because `/copy` is
unreachable while the aside is open (§1.1), this must be a **key**, and the
write must go through `_put_on_clipboard` (`app.py:6941`) — the single clipboard
write the transcript drag and the composer already share, whose docstring exists
to stop a third gesture drifting in what it writes or claims (`app.py:7009-7014`).

- **Fixes:** "the text is lost". It is the only escape hatch that **preserves
  the off-the-record contract** — the clipboard is outside the session, so
  nothing joins the context or the transcript, unlike `^F`. It also works while
  streaming, where `^F` is refused by `_aside_can_fork` (15744-15757), if it
  takes the settled-or-partial text off the dataclass rather than the painted
  rows.
- **Does not fix:** D1 or D3 at all. The text is still unreachable **on screen**;
  the user copies it out and reads it somewhere else. That is a real limitation,
  not a quibble — "paste it into another window to read the answer" is a poor
  primary answer to "I cannot see the text".
- **Cost:** low-moderate. One binding, one action, one `/help` row, and a
  decision about whether the payload is the last turn or all forkable turns.
  Notably it does **not** need `copy_targets.py` touched: the aside has at most
  a handful of turns and no transcript blocks, so a picker is over-engineering.
- **Risk:** a new global key (see §5.3 for the audit), and the off-the-record
  contract deserves a moment's thought — but the clipboard is the user's own
  deliberate act, and `/copy` already establishes that lifting text out is a
  supported gesture. It must **not** reuse the drag path (§1.1: that copies
  chrome and only the painted subset).
- **Recorded decision touched:** the module docstring's "the one door out is the
  user's" (21-28). This adds a second door — but one that goes to the clipboard,
  not to the context, so the docstring's actual claim ("the aside READS the
  conversation and never writes to it", 18-19) is untouched. The proposal must
  say so in the docstring rather than leaving the reader to notice.
- **No-wheel terminal:** yes — it is a key, and it takes the text off the
  dataclass, not off the screen.

---

## 3. Recommendation

**A combination: Option A as the fix, plus D2's marker, plus Option E as a
cheap and genuinely complementary follow-on. Not Option B.**

The user's instinct is right, and I want to say that plainly rather than
manufacture complexity: **"just allow the aside to be scrollable" is the correct
fix.** The reason is not that it is easy — it is that the defect is a *missing
unit* in the row model, and a scroll model is precisely the thing that supplies
it. Every other option either works around the missing unit (E), presents it
more honestly (C), or replaces the whole surface to inherit someone else's
(B). Only A fixes the thing that is actually wrong.

Two refinements to the naive version:

1. **The scroll model alone is not enough — it needs a keyboard chord**, or D3
   survives the fix and the app ships a second `↑ N earlier questions · scroll`
   marker naming content a no-mouse user cannot reach. The todo panel already
   made this exact mistake and fixed it the same way (`app.py:1757-1769`).
2. **D2 must be fixed first and separately.** It is a two-line change of far
   higher severity-per-cost than anything else here, it is independently
   shippable, and it is the difference between a truncated answer that lies and
   one that admits it. If only one thing lands, it should be this.

Option E is worth doing because it fixes a different problem — *persistence*,
not reachability — with the only mechanism that does not breach the
off-the-record contract. It is not a substitute for A, and the plan sequences it
last so it can be dropped without leaving anything half-built.

**Uncertainty I want on the record.** The one thing I could not settle from the
code is whether `ctrl+pageup`/`ctrl+pagedown` are *delivered by the target
terminals*. I measured that they work in-app (§5.3), which is necessary and not
sufficient. §4.5 makes the terminal probe a **gate on slice 3**, not an
assumption, following the precedent this codebase already set for `Cmd+V`
(`docs/evidence/cmd-chords/MEASURED.md`, `editor.py:989-1001`). If the probe
comes back negative on a target terminal, slice 3 changes key and slices 1-2
are unaffected — which is why the plan is sliced this way.

---

## 4. Slicing plan

Four slices. **Slice 1 and slice 4 own disjoint files and can run in
parallel from the start.** Slice 2 owns `aside_panel.py` and must follow slice 1
(same file). Slice 3 follows slice 2 (it needs a row model to drive) and is
gated on the probe.

| # | slice | owns | depends on |
|---|---|---|---|
| 1 | Overflow marker on a single turn (D2) | `aside_panel.py` | — |
| 2 | Row-offset scroll model (D1) | `aside_panel.py` | slice 1 (same file) |
| 3 | Keyboard chord (D3) | `app.py`, `tests/.../test_app_pilot.py` | slice 2 + probe |
| 4 | Aside copy key (E) | `app.py` | — (file-shares with 3; see note) |

**Parallelism, concretely.** Two developers: one takes slices 1→2 in
`aside_panel.py`; the other takes slice 4 in `app.py`. Slice 3 also lands in
`app.py`, so it must be taken by **whoever owns slice 4**, sequentially, not by
the first developer — otherwise two agents edit `app.py`'s `BINDINGS` block at
once. If slice 4 is dropped, slice 3 returns to either developer once slice 2
is merged.

### 4.1 Slice 1 — the overflow marker (D2)

**Change:** `aside_panel.py:482-483`. The `hidden == 0` branch must still
announce a cut when `len(lines) > budget`. The marker's noun must change — the
existing string counts *questions* (`AsideBody.hidden_turns`, whose docstring at
171-178 explains why questions and not lines: "a user remembers asking three
things; they never counted the rows an answer wrapped to"). That reasoning holds
for *earlier turns* and does not extend to rows dropped inside one turn, so this
case needs its own wording. Suggest a second marker naming the situation rather
than a count of questions that is zero — and per `subagent_view.py:1008-1011`,
**state the cost**.

**Acceptance criteria (headless):**
- A single turn whose rows exceed the budget renders a marker row; assert on
  `render_lines_for_test()`.
- The marker states a non-zero quantity of withheld content.
- The existing multi-turn marker (`↑ N earlier questions · scroll`) is
  unchanged in wording and count for the ≥2-turn case.
- The card's settled height still never exceeds `_fit()[0] - PANEL_HEIGHT_MARGIN`
  (the marker comes out of the budget, not the card's height — `_fit`'s
  docstring, 383-394).

### 4.2 Slice 2 — row-offset scroll model (D1)

**Change:** `_scroll_back` becomes a row offset from the tail; `_max_scroll_back`
computes from total flattened rows minus budget; `_body` windows by row while
**keeping the owning question pinned** (the rule already at 486-491, generalised)
so the 425-432 rationale is preserved. The three reset sites (249, 259, 274)
keep their "snap to the tail" meaning in the new unit. Consider naming the
attribute for what it now is.

**Acceptance criteria (headless):**
- QA's `_exhaust_the_wheel` oracle returns **100% of rows** for: 1 turn × 200
  rows at 80×24 with no band; 8 turns × 30 rows; and 1 turn × 200 rows at 80×24
  **with a populated todo band** (budget 3 — the worst measured case).
- The control case (8 turns × 3 rows) stays 100% reachable — no regression.
- Every window that shows a fragment of a turn also shows that turn's question
  row, or a marker identifying it. No orphaned continuation at the answer
  indent.
- Scroll position clamps at both ends; `_scroll_by` past the end is a no-op.
- `ask()` snaps to the tail; a mid-stream `append_answer` **does not** move a
  reader who has scrolled back (this is the first time `288-292` is testable —
  assert it).
- Card height is unchanged by scroll position at every offset.

### 4.3 Slice 3 — the keyboard chord (D3)

**Change:** two `BINDINGS` entries in `app.py` near the existing todo chords
(1770-1771), two actions beside `action_scroll_todos_*` (13740-13746), and a
`_key_row` in `_help_block` (~16939). Follow the todo chords exactly: `show=False`,
**bubble rather than `priority`** so a focused picker keeps first refusal, and
**no-op unless the aside is open and overflowing** so they shadow nothing on the
common path.

**Constraint:** `test_app_pilot.py:9676-9696` matches `/help`'s **key gutter**
(first 20 columns of each row), not the whole text. A chord added without a
`_key_row` will not be caught by that test unless it is added to `required` —
the slice should add it there, which is what makes the gutter row a contract
rather than a courtesy.

**Acceptance criteria (headless):**
- With the aside open and the **Editor focused**, the chord fires the action and
  **the composer's text and caret are unchanged**.
- The chord moves the card's scroll offset and reaches rows unreachable at
  offset 0.
- With the aside closed, the chord is a no-op and does not steal from the todo
  or transcript chords.
- With the aside open but not overflowing, it is a silent no-op (the
  `_scroll_todos` rule, 13772-13781).
- `/help` shows the chord in the key gutter; it is added to `required` in
  `test_app_pilot.py`.

### 4.4 Slice 4 — aside copy key (Option E)

**Change:** one binding + action in `app.py`. Payload comes off `AsideTurn`
(the dataclass, like `fork_messages` 321-327), **not** off the painted rows and
**not** via the drag path. Write through `_put_on_clipboard` (6941) so the
receipt stays the one shared toast. Must work while streaming (where `^F` is
refused). Add a `_key_row` to `/help` and a line to the aside's hint row
(`_hint_row` 579-620) only if it fits the shed-right-to-left budget — `esc` is
never dropped (582-583).

**Acceptance criteria (headless):**
- With a settled 200-row aside answer, the key puts **all 200 rows** on the
  clipboard — assert against the payload, not the screen.
- The payload contains **no card chrome** (no title row, no `─` rule, no hint
  row) — the `Chrome.ALLOW_SELECT` rule (`app.py:1507-1516`) applied to this
  path.
- It works while `session.is_streaming` is true.
- Nothing is appended to the session context or the transcript — assert the
  message list is byte-identical before and after.
- The write goes through `_put_on_clipboard` (assert the shared toast fired, not
  a bespoke one).

### 4.5 The gate on slice 3 — terminal delivery probe

**Before slice 3 is written**, run a raw-mode PTY probe for
`ctrl+pageup`/`ctrl+pagedown` on **Ghostty, iTerm2, Terminal.app and cmux**,
reusing the existing harness at `docs/evidence/cmd-chords/` (`probe2.py.txt`,
`run.sh`, `rung.sh`) and its two load-bearing properties: `select()` rather than
non-blocking polling, and a **typed-character control run first** so a zero-byte
result means "the terminal sent nothing", not "the harness missed it"
(`MEASURED.md:9-19`).

Record the byte captures in `docs/evidence/aside-scroll/MEASURED.md` in the same
table format. **If a target terminal delivers nothing, slice 3 changes key
rather than shipping a chord that silently does not exist there** — which is
precisely the conclusion the `Cmd+V` probe forced (`editor.py:989-1001`: "With
an image-only pasteboard, Terminal.app delivers ZERO bytes on `Cmd+V` and
beeps"). This is a gate, not a formality, and it is the one open question in
this proposal.

### 4.6 What happens to the investigation test file

`tests/unit/tui/test_aside_reachability_investigation.py` is a **QA artifact,
not a regression guard**, and says so in its own docstring: "Every test here
asserts the CURRENT (defective) behaviour, so a fix for the defect will turn
these red on purpose. Delete the file with the fix, or rewrite each assertion to
the intended behaviour — **do not 'repair' it by loosening the numbers**."

**Plan:** delete it as the final step of slice 2, and carry its two durable
assets forward into a real regression test:

- `_exhaust_the_wheel` — the reachability oracle. It is the acceptance criterion
  for slice 2 and must survive, retargeted to assert **100%** reachability.
- `_long_answer` — the fixture that avoids wrap-dependent row counts.

The end-to-end pilot test (283-340) should be rewritten rather than deleted: it
is the only test that drives the real app the way a user does, and its
`assert present == set(range(200 - budget, 200))` becomes
`assert present == set(range(200))`. Its comment about keys that "deliberately
declined to bind" needs updating once slice 3 lands.

Loosening a number in that file instead of deleting it would be the single worst
outcome of this work, so it is called out here and belongs in the review
checklist.

---

## 5. Recorded decisions this proposal touches

### 5.1 Wheel-only, and the rejection of scroll keys

> "The WHEEL is the one gesture that acts. Scroll KEYS were rejected because
> ↑/↓ belong to the focused composer's prompt history and the aside's whole
> premise is that the user keeps typing there — but the wheel costs no key, and
> without it the `↑ N earlier questions` marker names content with no way to
> reach it." — `aside_panel.py:338-342`

**Answer: this decision is upheld, not overridden.** It rejects **↑/↓
specifically**, and for a reason that is still true — those keys belong to the
composer's prompt history (confirmed at `editor.py:2363-2376`). It does not
reject scroll keys in general; its actual principle is "do not take a key the
composer needs". A chord the composer does not bind satisfies that principle.

Note also that the comment's own closing clause — a marker naming content with
no way to reach it — is **exactly the state a no-wheel terminal is in today**
(D3). The decision anticipated this failure mode and solved it for the
mouse-having case only.

### 5.2 The off-the-record contract, and the one door out

> "The one door out is the user's. `^F` forks the exchange into the chat… the
> user decides when a side thread becomes part of the record" —
> `aside_panel.py:21-28`; "the aside READS the conversation and never writes to
> it" — 18-19.

**Answer: untouched by A, and consciously extended by E.** Options A, B and C
change only what is *displayed*. Option E adds a second door, but it opens onto
the **clipboard**, not the context or the transcript: nothing is written to the
session, so the docstring's load-bearing claim survives intact. The distinction
is real and worth stating in the docstring if slice 4 lands, because a reader
who sees two doors and one sentence saying "the one door" will assume the
sentence is stale.

### 5.3 Free-key discipline

> "`ctrl+up`/`ctrl+down` because `TextArea` binds neither (audited: it binds
> up/down/pageup/pagedown but not the ctrl variants), so the composer keeps
> every cursor key it had, and they bubble (not `priority`) so a focused picker
> keeps first refusal. They no-op unless the panel is expanded AND overflowing,
> so they never shadow anything on the common path." — `app.py:1764-1769`

**Answer: followed exactly.** The audit for this proposal:

- **Taken:** `shift+up/down` (TextArea selection + Editor chord targets),
  `pageup`/`pagedown` (TextArea + model picker), `ctrl+u`/`ctrl+d`
  (**destructive** — delete-to-start-of-line / delete-right + exit-on-empty),
  `alt+up/down` (rewritten to plain up/down by `Editor.VERTICAL_CHORD_KEYS`;
  binding them caused a real past defect, `editor.py:1070-1078` — **do not
  propose**), `ctrl+up`/`ctrl+down` (todos), `ctrl+home`/`ctrl+end`
  (transcript).
- **Free:** `ctrl+pageup` / `ctrl+pagedown`. Only
  `ScrollableContainer.page_left/right` claims them, and only when a scrollable
  has focus, which never happens while the aside is open. `grep` for both across
  `local_operator/` returns **zero hits**.

**Measured, not assumed (in-app half).** I subclassed `OperatorApp` with probe
bindings on both chords, opened a real aside through the composer via `/btw` at
120×40, and pressed them with the `Editor` focused:

```
aside open: True | focused: Editor
ACTIONS FIRED: ['ctrl+pageup', 'ctrl+pagedown']
composer text unchanged: True ''
```

Both chords reach app level with the composer focused and the aside open, and
neither perturbs the composer buffer. **This settles in-app dispatch only.**
Terminal delivery remains open and is gated at §4.5.

### 5.4 `overlay.rows_above_dock` reads the live dock

> rationale at `overlay.py:62-73`; a previous design review (D1) called its
> disagreement with `composer_column` a bug and **was wrong**.

**Answer: not touched.** The dock band's 6-row cost is working as designed. It
makes D1 more acute, which is why slice 2's acceptance criteria include the
band-populated 80×24 case (budget 3) rather than only the comfortable one.

### 5.5 Turn-grouped cutting

> "Grouped rather than flattened because the card sheds whole TURNS when it
> overflows — cutting by row left the top of the card showing a mid-sentence
> continuation at the answer indent with no question above it, which reads as
> the start of a new answer." — `aside_panel.py:425-432`

**Answer: this is the real constraint on slice 2, and it is preserved.** The
fix is not "flatten and forget the groups" — that is the exact bug this comment
records. It is "window by row, keep the owning question pinned", which `_body`
**already does** for the single-turn case at 486-491. Slice 2 generalises an
existing rule rather than reversing a decision, and the acceptance criterion
"every window showing a fragment also shows its question" is how QA holds that
line.

### 5.6 The card is sized to its content

> "the card is sized to its CONTENT and rests on the composer, so unspent budget
> is not padding to be printed — a two-line answer in a card padded to the full
> budget is thirty rows of empty overlay covering the conversation it is about."
> — `aside_panel.py:634-638`, implemented at `_repaint` 664-674.

**Answer: this is why Option D does not exist** (§2.4). The behaviour is already
shipped; the ceiling is the ground above the dock, and raising *that* is Option
B, with the costs §2.2 sets out.

---

## 6. Risks to watch during rollout

1. **Repaint cost under streaming.** `_body` is recomputed on every delta with
   no caching (284-293). A row-offset model does more work per call (flatten +
   window). On a 200-row answer streaming token by token this is a hot path.
   Watch for perceptible lag; if it appears, cache the flattened rows against a
   `(turn count, answer lengths, width)` fingerprint — the same shape
   `_layout_fingerprint` (417-419) already uses. **Do not** pre-optimise this in
   slice 2; measure first, per `AGENTS.md`'s rule that a number copied from an
   older comment is not evidence (`AGENTS.md:472`).
2. **The trailing `…` row during streaming** (528-530) costs a budget row
   exactly when the answer is largest. Under a row model it should not be part
   of the scrollable content — it is a status indicator, not an answer row.
   Easy to get wrong; worth an explicit assertion.
3. **The `-squeezed` class and the gutter** (`_fit` 396, `_repaint` 672). A
   scroll offset must not change the card's height, or the card will breathe by
   a row as the user scrolls. Called out as an acceptance criterion in §4.2.
4. **Chord shadowing under a picker.** Bubble, never `priority` — the todo
   chords' stated reason (1766-1767). A `priority` binding here would steal the
   key from a focused picker.
5. **The investigation test file being "repaired" rather than replaced** (§4.6).
   Its own docstring forbids it. This is a review checklist item.
6. **`/help` gutter drift.** `test_app_pilot.py:9676-9696` matches the first 20
   columns; a chord whose gutter text exceeds that, or which is not added to
   `required`, ships undocumented and untested.
7. **`HANDOFF-copy-command.md` is stale and untracked in the repo root.** It
   describes building `/copy`, which has shipped. Anyone picking up slice 4 will
   find it and may build from it. Recommend deleting it — as a separate,
   explicitly-approved change, since it is the user's own untracked file and not
   this proposal's to remove.
