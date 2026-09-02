# Ruling: D7 — reveal reachability when the QUESTION eats the option budget

Status: architect ruling. Author: architect.
Base: worktree `lo-ask-scroll`, branch `fix/ask-long-text`, HEAD `720dc444`.
Every `file:line` below is against that tree; oh-my-pi (OMP) references are
against the local clone at `~/workspace/repos/oh-my-pi` (HEAD `18781d8`).

Routed here as the one open item on the line-granular ask-picker change: the QA
verdict was PASS with `test_a_cut_description_is_never_unreachable_by_every_gesture`
pinned **xfail-strict** pending this ruling
(`tests/unit/tui/test_ask_picker.py:3969-4047`).

---

## TL;DR

**FIX, and the fix is not in the reveal.** Refusing `ctrl+e` on a 2-line
viewport is *correct* — a cap-lift genuinely cannot draw a third line, so
offering the key would be a dead-key lie. The real defect is one step upstream:
**our card lets a long QUESTION consume the entire option budget, which OMP
structurally forbids.** OMP caps the in-body question header at `MAX_HEADER_ROWS
= 4` and floors the option body at `MIN_BODY_ROWS = 5`
(`ask-dialog.ts:50,62,519-526`). We cap the question at nothing
(`ask_picker.py:1328-1334`, "Never truncated"). Adopt OMP's header cap. It is
the smallest change that restores fidelity, it is the mechanism OMP already uses
for exactly this case, and it dissolves D7 at every realistic size rather than
papering over the one fixture that trips it.

The recommendation touches **one invariant deliberately**: "the question is
never truncated" becomes "the question is bounded to `MAX_QUESTION_ROWS`, with
its cut marked." That invariant was never a safety property — the *ordering*
(question outranks options) is the safety property, and it is untouched.

---

## 1. Is D7 a real problem or a pathological corner?

**It is real, and wider than the fixture QA pinned.** The framing that "only
canary 100x30 bites" is an artefact of the two fixtures in the test file, not a
property of the card. Measured on the real `OperatorApp` (dock 5, seeded
transcript), via `/tmp/probe_d7b.py`:

```
question       size     q-lines  body_budget  ^e     description reachable?
repro (74ch)   100x50   1        12           live   partial (162/1023 on screen, ^e reaches rest)
repro (74ch)   100x30   1        6            live   partial, ^e live
repro (74ch)   100x28   1        4            live   partial, ^e live
repro (74ch)   100x24   1        3            REFUSED   cut, no ^e   <-- D7
repro (74ch)   100x20   1        1            REFUSED   cut, no ^e   <-- D7
```

So D7 is **not** unique to a 592-char question. A perfectly ordinary 74-char,
one-line question hits the identical wall at **100x24 and 100x20** — the reveal
is refused while a description is cut, at 120 and 160 columns too (same probe).
The canary fixture is not what causes D7; it just moves the trigger *up* to a
taller terminal (100x30) because its 7-line question steals five extra rows.

What decides D7 is a single quantity: **`body_line_budget` after the question is
paid for.** Whenever that drops to 1-2 lines with an option list still present
and a description longer than the budget, the description is cut and the reveal
cannot help — a 2-line viewport cannot draw a 3rd line even with the cap lifted
(`_reveal_is_useful`, `ask_picker.py:3157-3200`; the §4.3 label-anchor
degradation). That is a correct refusal (see §3), but the *state that forces it*
is reachable two ways:

- **the terminal is genuinely tiny** (≤24 rows). Here 1-2 body lines is honest —
  there is no budget to give anyone, and the escape hatch is the mobile relay
  (§4). This is an accepted limit for both us and OMP.
- **the question ate the budget on a NOT-tiny terminal.** This is the canary at
  100x30 (30 rows is not tiny; a 7-line question left `body_budget=2`) and it is
  a genuine defect: rows exist, but the question monopolised them. **This is the
  half D7 should fix.**

The measurement that proves the split is the header-cap simulation
(`/tmp/probe_d7fix.py`, capping the question to OMP's 4 lines):

```
canary 100x30   BEFORE: q=7 budget=2 ^e=REFUSED reach=65/434   D7
                AFTER:  q=4 budget=5 ^e=LIVE    reach=153/434   fixed
canary 100x44   BEFORE: q=7 budget=9                             (ok)
                AFTER:  q=4 budget=12 reach 153->153, ^e live    better
canary 100x24   BEFORE: q=5 budget=1 ^e=REFUSED                 D7
                AFTER:  q=4 budget=2 ^e=REFUSED                 STILL D7 (genuinely tiny)
```

The cap converts every *non-tiny* D7 case to a live reveal, and leaves only the
genuinely-tiny ≤24-row cases refusing — which is the accepted limit OMP also
lives with.

---

## 2. What OMP actually does in this situation (fidelity check)

The user's north star is OMP fidelity and reachability of the text that was the
original bug. So the ruling is anchored on what OMP does when a long question
meets a long description in a short box. Three findings from `ask-dialog.ts`,
all load-bearing:

1. **OMP caps the question header.** `MAX_HEADER_ROWS = 4` (`:62`). A question
   that wraps past four lines is truncated to `4-1` lines plus a `…`-marked line
   (`renderQuestionTitle`, `:153-159`), and the box height is measured with the
   header already clamped to that (`:565-566`). **A long question cannot eat the
   option viewport in OMP.**

2. **OMP floors the body.** `MIN_BODY_ROWS = 5` (`:50`); the body is
   `Math.max(MIN_BODY_ROWS, totalRows - fixedRows)` (`:526`) and the panel is
   measured to guarantee it (`:576`, `:582`). The option list is always handed
   at least five lines. Combined with (1), the D7 state — a 1-2 line body while
   an option list is present — is **structurally unreachable** in OMP's design.

3. **OMP has no description reveal at all.** Descriptions are hard-clamped to two
   lines with `wrapped.slice(0, 2)` (`:338`) and there is no key to see more of
   a *description*. OMP's only reveal is `Ctrl+O`, which expands the truncated
   *question header* (`toggleQuestionExpansion`, `:461-471`; `#expanded`
   drives `titleRows` at `:565`). Our `ctrl+e` description-reveal is an *ours*
   feature with no OMP analogue (the line-granular design says as much,
   `ask_picker.py:1709-1711`).

The fidelity conclusion is sharp: **OMP does not solve D7 by making the reveal
smarter — it never has a description reveal to make smarter. OMP solves it by
never letting the question starve the body in the first place.** Porting the
header cap is therefore not a new invention; it is closing the one place our
port diverged from OMP's layout contract. The line-granular design already
ported OMP's line list, `lineStartByRow`, thumb, and `renderRows(width-1)`
scrollbar-column trick — the header cap and body floor are the pieces of the
same layout it left on the table.

---

## 3. Is refusing `ctrl+e` correct? Yes — leave the reveal alone.

`_reveal_is_useful` (`ask_picker.py:3157-3200`) returns False at
`body_budget=2` because lifting the selected row's cap from 2 to 8 draws *zero*
additional lines of that row: the row is already clipped by the viewport, and a
taller row clips at the same viewport edge. `_visible_row_lines(revealed) ==
_visible_row_lines(default)`, so the honest answer is "the key would change
nothing." Offering it would re-introduce exactly the dead-key lie the footer is
built to refuse (the digits on a one-row window, the keymap while the composer
holds focus — `_reveal_hint`, `:3110-3137`).

So **do not touch `_reveal_is_useful`, `_reveal_hint`, `_cap_for_row`, or the
cap-lift model.** They are correct. Making the reveal "try harder" at
`body_budget=2` would mean drawing a 3rd line into a 2-line viewport, which is
the R11/D1 footer-clip class this whole file exists to prevent. The reveal is
the wrong layer; the question budget is the right one.

This also disposes of the tempting-but-wrong option **(b)** from the routing
brief — a modal full-description mode where `ctrl+e` collapses the question to
one line and gives the selected row the whole body. It is more machinery than
the problem needs, it re-opens the "two mechanisms fighting for the viewport"
bug the line-granular design deleted by construction
(`ask-line-granular-scroll.md:429-488`; AGENTS.md:595), it has no OMP analogue,
and it fixes only the description-reveal symptom while leaving the deeper defect
(a 7-line question over a 2-line body is a bad frame *before* anyone presses a
key) in place. Rejected.

Option **(a)** — let the question scroll/truncate so options get budget back —
is the *right instinct pointed at the wrong control*. "Let the question
truncate" is correct; "let the question *scroll*" is not (a scrolling question
is a second viewport, the same fight). OMP's header cap is exactly (a) done the
safe way: a bounded, `…`-marked, non-scrolling header. That is the recommendation.

---

## 4. Is the mobile relay a sufficient escape hatch for the residual?

**Partly — necessary, not sufficient on its own, but it covers the genuinely-tiny
tail.** After the header cap, the only surviving D7 cases are terminals ≤24 rows
where even a 4-line question leaves a 1-2 line body. There, no in-TUI gesture can
help (there is no budget to give), and the honest fallback is that the phone
holds the full text. Verified:

- the ask projection ships each option's **full** description to the phone
  (`mobile/owned.py:222-232` builds through the shared `ask_pending_request`
  seam; the mobile card renders `opt.description` verbatim,
  `mobile/web/src/components/pending-card.tsx:206-210`);
- the only bound is `FRAME_CAP_PENDING_DETAIL_CHARS = 2000`
  (`mobile/projection.py:166,490`), and the D7 descriptions are 434 (canary) and
  1023 (repro) characters — **both well under the cap**, so the phone receives
  the whole description uncut.

So the escape hatch is real and it is the correct thing to *document* as the
residual-limit answer. It is not a licence to skip the header cap: relying on it
for the 100x30 canary case (a non-tiny terminal) would be shipping a known bad
in-TUI frame and pointing at a phone, which is not fidelity. Fix what can be
fixed in the TUI (the header cap), and let the relay cover the irreducible
tiny-terminal tail.

---

## 5. Recommendation — the precise fix for a coder

Adopt OMP's `MAX_HEADER_ROWS` as a question-line cap. Smallest change that
solves it; mirrors OMP `ask-dialog.ts:62,153-159`.

### 5.1 The change

**File `local_operator/tui/widgets/ask_picker.py`.**

1. **Add a module constant** next to `MIN_BODY_ROWS` (`:170`) /
   `DEFAULT_DESC_CAP` (`:356`):

   ```python
   #: Max wrapped lines the QUESTION header may occupy, mirroring OMP's
   #: MAX_HEADER_ROWS (ask-dialog.ts:62). A question longer than this is
   #: truncated to (cap - 1) whole lines plus a `…`-marked line, so a long
   #: question cannot starve the option list of its budget (GAP D7). The
   #: ordering "question outranks options" (the safety property) is unchanged;
   #: this only bounds how many rows the question may TAKE before the options
   #: start competing.
   MAX_QUESTION_ROWS = 4
   ```

   Use `4` to match OMP exactly. (It sits comfortably above the realistic
   one-line question and the canary's natural 4-5 lines at working sizes.)

2. **Cap the wrap in `_question_lines`** (`:1328-1334`). Replace the body so it
   truncates past the cap and marks the cut with the SAME idiom `_allocate`
   already uses for a question the *budget* cannot fit
   (`ask_picker.py:1892-1895` — `truncate_cells` to `width-2`, then append
   ` …`, avoiding a double ellipsis):

   ```python
   def _question_lines(self, width: int) -> list[str]:
       """The question, wrapped and bounded to MAX_QUESTION_ROWS.

       Bounding (not "never truncated") so a long question cannot consume the
       whole body and leave the option list a 1-2 line viewport in which a cut
       description is unreachable by any gesture (GAP D7; OMP MAX_HEADER_ROWS,
       ask-dialog.ts:62,153-159). The cut is MARKED, like every other
       abbreviation this card makes, so the reader can tell the question
       continues. The ordering that makes the question outrank the options
       (the safety property, _layout steps 1-2) is untouched: this bounds how
       many rows the question may take, not whether it is shown first.
       """
       lines = wrap_cells(self.question.question, width) or [""]
       if len(lines) <= MAX_QUESTION_ROWS:
           return lines
       kept = lines[: MAX_QUESTION_ROWS - 1]
       tail = truncate_cells(lines[MAX_QUESTION_ROWS - 1], max(1, width - 2))
       tail = tail[:-1].rstrip() if tail.endswith("…") else tail
       return [*kept, f"{tail} …"]
   ```

   `truncate_cells` is already imported and used at `:1892`. Do **not** collapse
   the remaining wrapped lines into the tail the way OMP's
   `renderQuestionTitle` does (`join(" ")`, `:158`) — our `_allocate` already
   marks a budget-cut question the same way (single `…`), so a plain
   `…`-mark on the 4th line keeps the two truncation paths visually identical and
   avoids a re-wrap.

### 5.2 What this touches, and what it must not

- **Do NOT add a `MIN_BODY_ROWS` floor to the body.** OMP floors the body
  because its dialog is a fixed-height modal box that can afford to; our card is
  a dock band whose whole design is `height: auto` with three anchoring ceilings
  and *no floor* (`_body_rows`, `ask_picker.py:1476-1481` — "There is no floor
  under the result. Zero is a legitimate answer"). A floor here would re-open the
  R9-R11 clip the design explicitly closed. The header cap alone is sufficient
  (§1 measurements): capping the question is what frees the body; we do not also
  need to force a floor into it.

- **Do NOT touch the reveal** (`_reveal_is_useful`, `_reveal_hint`,
  `_cap_for_row`, the cap-lift). §3 — they are correct.

- **The `ctrl+e less/more` interaction is unchanged.** With more body budget the
  reveal simply becomes live at more sizes (the AFTER column in §1); it is the
  same mechanism reading a bigger `body_line_budget`.

### 5.3 Interaction with the approval gate — verified safe

The approval gate does NOT regress. Its question is one line at every width
(`Allow bash? <target>`), so it never wraps past `MAX_QUESTION_ROWS` and the cap
is a no-op there. Verified on the real boot (dock 8) and seeded (dock 5)
geometries via `/tmp/probe_d7c.py`: `q-lines=1` at every size 190x50 down to
80x20, D7 never fires (the gate's consequences are 1 line, so nothing is ever
silently cut; at tight sizes it drops to labels-only via `_labels_must_all_fit`,
`approval.py:1029-1048`). The header cap and the `_labels_must_all_fit` hook are
orthogonal — one bounds the question, the other protects the option labels — and
neither changes the gate's frames.

### 5.4 Invariants to re-verify after the change (the rollout watch-list)

1. **No-reflow / stable height.** `_question_lines` feeds `_body_rows`'
   `question_lines` count (`ask_picker.py:1657-1658`) and the `wanted` natural
   height (`:1575`). Capping it changes both consistently, so the card stays a
   single settled height per size — but re-run the `_settle` three-frame check
   (the visual-validation recipe, AGENTS.md:361-367) on the canary at 100x30 to
   confirm no post-paint reflow.
2. **The coexist headline** (descriptions + scroll together at 150x40) is
   unaffected: the canary already draws 4 question lines at 150x40, so the cap is
   a no-op there. Confirm `grants` still `{0:2,1:2,2:2,3:2,4:1}` at 150x40.
3. **C1 (viewport never overdraws its budget).** Unchanged — the cap only
   changes how many lines the *question* claims before step 9; `_allocate`'s
   line-by-line spend (`:1927-1941`) is untouched.
4. **The question-outranks-options ordering** (`_layout` steps 1-2,
   `:1596-1642`; `test` at `:1688`) is untouched: the question is still bought
   first; it is merely bounded in how much it may buy.

---

## 6. How the test should be pinned

`test_a_cut_description_is_never_unreachable_by_every_gesture`
(`test_ask_picker.py:3969-4047`) is currently `xfail(strict=True)`. Once the
header cap lands:

- **The canary 100x30 case flips to reachable** (`^e` live, 153/434 on screen
  and the rest reachable via reveal). This is the case the test was written to
  catch, so it should now PASS at that size — **remove the xfail and let the test
  run live** for the size grid it already sweeps, EXCEPT the genuinely-tiny tail.
- **The residual is the ≤24-row cases** (100x24, 100x20 in the sweep), where D7
  survives by physics, not by defect. Do **not** relax the assertion to hide
  them (the D5 "silent-truncation-pinned-as-correct" discipline the test's own
  docstring cites, `:4011-4013`). Instead, **narrow the swept sizes to those
  where a reveal is physically possible** — drop `(100, 24)`-and-tighter from the
  grid at `:4025`, and add a *separate, explicit* documented-limit assertion for
  the tiny tail:

  ```python
  # At <=24 rows a one-line question already leaves a 1-2 line body, so no
  # cap-lift can draw a 3rd line: the reveal is correctly refused and the
  # residual text lives on the phone (full description, mobile relay,
  # projection.py:166 cap 2000 >> 434/1023). This is the accepted limit OMP
  # shares (MIN_BODY_ROWS=5 body floor + MAX_HEADER_ROWS=4 header cap still
  # cannot conjure budget on a 20-row box). Asserted as a DOCUMENTED LIMIT,
  # not xfailed: the behaviour is correct, so it gets a live guard that pins
  # it, and the mobile-relay reach is asserted alongside so the escape hatch
  # is a tested property rather than a comment.
  ```

  That converts the pin from "a red flag we are ignoring" to "a green guard that
  states the boundary and proves the escape hatch," which is the honest end
  state: the fixable half is fixed and guarded live; the irreducible half is
  documented, bounded, and its mobile fallback is tested.

The net: after this change there is **no size where a NON-tiny terminal cuts a
description unreachably**, which is the property the whole feature exists to
guarantee.

---

## 7. Risks

- **Someone reads a truncated question and misses a constraint.** Mitigated by
  the `…` mark (the reader can tell it continues) and by the cap being generous
  (4 lines ≈ 300-400 chars at working widths). OMP has run this exact cap in
  production. If we want the OMP parity in full, a follow-up could add a
  question-expand key (OMP's `Ctrl+O`), but that is **out of scope for D7** — the
  gap is reachability of the *description*, and the cap fixes that without it.
  Flag it as a possible future item, not a blocker.
- **The cap value (4) is a judgement call.** 4 matches OMP; it is measured to
  clear D7 at every non-tiny size (§1). If a future question genuinely needs more
  header room the constant is one number to change, and `MAX_QUESTION_ROWS`
  names itself.
- **A hidden dependence on `_question_lines` being unbounded.** Grep shows two
  callers: `_layout` (`:1657`) and `_body_rows`'s `wanted` (`:1575` via the same
  count) — both *want* the bounded count. No caller relies on receiving every
  wrapped line. Re-confirm with a full `env -u NO_COLOR TERM=xterm-256color
  .venv/bin/python -m pytest tests/unit/tui -q` run plus the four gates before
  the MR.

---

## Appendix: reproduction

All numbers above are from the real `OperatorApp` (never the dockless
`_AskHost`), seeded transcript, dock 5 — probes at `/tmp/probe_d7b.py` (realistic
question row-sweep), `/tmp/probe_d7c.py` (approval-gate immunity),
`/tmp/probe_d7fix.py` (header-cap before/after). The D7 fixture and the pinned
test are `tests/unit/tui/test_ask_picker.py:3969-4047`; the canary/repro
fixtures are `:185-234` and `:3440-3498`.
