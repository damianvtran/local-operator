# Design: the multi-level tool-call fold in the TUI transcript

Status: IMPLEMENTED (slices 1–5 on `feat/tui-tool-fold`). Base: `origin/main` @
`bd7bf7520`. Author: coder (slice 5), 2026-11-22, from the frozen spec and
consult in the build brief; the visual truth is the interactive prototype
(`tui-tool-collapse-prototype.html`, screenshots `shot-01`…`shot-18`).

**What this is.** Ben asked the TUI transcript to "match more closely to the lop
desktop where it has multiple levels of collapse", with a setting to render by
turn or by response, and "subsequent tool calls collapsed into each other". The
answer is a TUI-only fold layer over the existing tool rows: turn → run →
same-tool sub-group → call row → the call's own expanded detail. It does not
touch the receipt fold (`QuietGroupBlock`), the quiet-group contract, or the
cross-client parity fixture.

---

## 1. Levels

| level | row | hides |
|---|---|---|
| turn bar | `FoldBarBlock`, kind `turn` | everything a settled answered turn did before its answer (By turn only) |
| run bar | `FoldBarBlock`, kind `run` | a maximal span of 3+ back-to-back tool calls with no visible row between them |
| sub bar | `FoldBarBlock`, kind `sub` | 4+ consecutive calls of ONE tool inside a run (`browser ×16`) |
| call row | the existing `ToolCard` | nothing; its own row is the receipt |
| call detail | the card's existing expansion | nothing |

Never two bars saying the same thing: a turn holding exactly ONE run is *flat*,
and opening its turn bar reveals that run's children directly (no run bar in
between). A turn bar is only drawn when it would hide at least
`TURN_BAR_MIN_ROWS = 3` rows.

**What never folds.** A call awaiting approval (`waiting`), an `ask`, and a
generated image split a run and stay visible. A row a fold would hide is always
exempted first; the layer never has to ask a row whether it may hide.

## 2. Defaults

- A settled answered turn is CLOSED under its turn bar (By turn).
- A run of 3+ calls is CLOSED.
- A same-tool sub-group of 4+ calls is CLOSED.
- The run a LIVE turn is still growing stays OPEN, and so does the newest
  sub-group inside it: the reader is watching it.
- The reader's own open/close choice is never undone by a later append; it is
  keyed by the bar's stable key (`tf:turn:<first-row-id>`, `tf:run:<first-tool-id>`,
  `tf:sub:<first-tool-id>`) and lives for the view's lifetime (no persistence;
  `/clear` resets it).
- A `waiting` call keeps its row; nothing open or closed changes that.

## 3. Keys (on a focused fold bar only)

| key | action |
|---|---|
| enter / space | toggle the bar |
| up / down | walk the visible ledger, bars included |
| left | close; when already closed, step OUT to the parent bar |
| right | open; when already open, step INTO the first child (bar, else row) |
| `f` / `F` | next / previous failed call in document order, opening every bar over it and focusing the call |
| `+` | open every level of the focused bar's turn |
| `-` | that turn back to its defaults (drop the turn's explicit choices) |
| esc | never closes a fold — it keeps its app-wide meaning |

**The printable keys are claimed on `FoldBarBlock` and nowhere else.**
`ExpandableActionBlock.on_key` forwards every unbound printable to the composer,
so binding `f`/`F`/`+`/`-` higher in the hierarchy would swallow the first
letter of a sentence typed while a tool row had focus. A focused bar is a
navigation surface, and there the design deliberately spends those characters.
`c` was not chosen for anything: it is taken app-wide. `v` is not a key either —
the layout is the `display.transcript_display` setting plus `/transcript
by-turn|by-response`.

`f`/`F` land the keyboard on the CALL (that is where the reader acts). A
ToolCard does not bind those characters, so the next jump is taken from a bar
again (Up from the call returns to it); the layer keeps a failing-call cursor so
that press continues the ladder rather than restarting from the bar's first row.

## 4. Summary grammar

`N actions · kind, kind +M · ✗ K failed` — one grammar on every bar.

- Kinds use the existing shortened nouns (bash→shell, task→subagents,
  hub/send→messages), in order of first appearance, at most 3 then `+M`.
- A same-tool sub bar says `tool ×N` in place of the count, because its name
  column already names the tool.
- `✗ K failed` is muted on a turn that ANSWERED (failures are history) and
  danger+bold only when the turn DIED.
- Time is right-aligned in the ledger's duration column. It is the SUM of the
  calls' known durations (`ToolCard.duration_s`), not a wall clock; a call with
  no recorded duration contributes nothing, and a span with none shows no time.
- That clock speaks the tool row's OWN grammar (`format_clock`): `<0.1s` under
  50 ms, one decimal under ten seconds, whole seconds under a minute, then
  `format_duration` (`1m57s`). `format_duration` alone floored a 0.5 s run to
  `0s` beside its own member rows reading `0.1s`, and a 12.95 s foot to `12s`
  where the calls said `13s` — the app disagreeing with itself in one column.
- Narrow widths drop kinds first, then the kinds clause; the failure count and
  the time are never dropped. A live bar replaces the clock with `◐ <tool>
  <summary> <elapsed>` and repaints at 1 Hz.
- Every width in that ladder is counted in CELLS (`rich.cells.cell_len`), never
  code points: a bar's clause can be a translated count, a count word or a
  tool's own name. Counting characters let a wide-script summary believe it fit
  and reach the ellipsis cut with the kinds still in front of the tally.

## 5. Why a separate layer (and not one more `QuietGroupBlock`)

The receipt fold groups delivery receipts by an ownership projection: members
share an owner, an interval, a quiet verdict. A tool fold groups by SHAPE
(consecutive calls, same tool, no visible row between). Extending
`QuietGroupBlock` would have put two incompatible group definitions on one
widget, and the receipt contract is consumed by the mobile web projection:
`local_operator/mobile/web/src/lib/quiet-groups.parity.json` and
`tests/unit/tui/test_quiet_groups.py` hash-pin today's behaviour across clients.
That fixture is untouched, and so is the receipt's own visibility machinery.

Instead:

- one pure model, `local_operator/tui/tool_outline.py` (rows in → outline of
  turn > run > sub nodes; pure, so live and replay agree);
- one widget pair, `FoldBarBlock` and `TurnFootBlock`, in
  `widgets/transcript.py`, sharing the one-row bar plumbing (`_SingleRowBarBlock`)
  with the receipt bar;
- one owner, `local_operator/tui/tool_fold.py::ToolFoldLayer`, deriving once per
  dirty frame from the view's own rows — never from the event stream, which is
  why a replayed transcript folds exactly like a live one;
- one visibility owner: `TranscriptBlock.fold_vetoes` + `sync_display()`, the
  same veto channel the receipt fold uses, so two folds over one row cannot
  overwrite each other's `display`.

The tool-only `work` bar of `quiet_groups.py` was RETIRED for this
(`TOOL_ONLY_WORK_BAR = False`, a one-line revert). Receipt groups with tools
inside keep today's behaviour, and the layer skips ToolCards inside an open
receipt bar (they fold with the receipt, not with their neighbours).

## 6. Failure policy

- A failed call is counted on its bar (`✗ K failed`) and stays INSIDE the folded
  run. Nothing is released into the transcript.
- The ONE exemption: a turn that died (no answer, or a turn-failed notice) pins
  its LAST failed call below the collapsed fold, so the reason is reachable
  without unfolding a sixteen-call batch. `f`/`F` open whatever is needed for any
  failure.
- On a turn that answered, the tally is muted — history, not alarm (desktop
  issue #6's ruling, kept).
- A turn-failed notice keeps its stored text; the By-response FOOT renders the
  reason cleaned (no `[session incident …]` head, no "Take it into account
  before repeating…" trailer) and withholds the notice row's paint so the reason
  is stated once. Stored rows are never rewritten for a layout.

## 7. By turn vs By response

**By turn (default).** A settled answered turn folds under one turn bar; its
pre-answer rows are hidden until opened. Tool rows and bars sit one step
(two cells) deeper than prose, in this layout too. There is no foot line — the
turn bar already closes the turn with the same facts.

**By response (layout B, "compact steps").** No turn bar. Runs fold; mid-turn
narration is drawn in the muted ink at the prose column while the final answer
stays text ink; no blank rows between the steps of one turn — one blank after
the prompt and one before the answer, and nothing else. A settled turn that had
actions ends with a one-row foot:

- `✓ N actions · 12.9s` (success glyph, dim) — the whole turn's `Turn.facts`,
  clock summed like a bar's;
- `✗ turn failed · <reason>` (danger) when the turn died. It WRAPS over rows with
  a four-cell hanging indent and never truncates: the reason is the value.

The foot mounts only when the turn settles (a live total would be a number still
moving) and only when the turn had at least one action. A plain answer has
nothing to total. By turn: no foot.

## 8. Indent rule

Two columns, not a ruler:

- prose (user, narration, answer, foot) draws at the transcript's own padding;
- fold bars and the call rows of a step draw **two cells deeper** (the ledger's
  one-cell spine plus one). A STEP is the turn, not the bar: a turn that holds
  any tool call puts its rows in that column in BOTH modes, whether or not it
  grew a bar (By response closes such a turn with its foot; By turn may simply
  have too few calls for a bar). Two insets for one kind of row in one frame —
  an unfolded call at col 3 beside a row revealed at col 6 — read as two kinds
  of row;
- a row REVEALED under an open bar, and a pinned failure below a collapsed one,
  draws **two cells deeper again** (four total);
- a call's own DETAIL rows (`action:`/`target:`/`error:`, output, args) stay one
  cell RIGHT of the row at every level, as they always did at the plain spine
  (`detail_indent`). They used to sit on the fixed `OUTPUT_INDENT`, so a row
  revealed at col 6 painted its own body at col 4 — the row walked right and
  left its content behind.

A receipt card (scheduled wake, inbound peer message) keeps the ledger spine it
has always drawn: it is not a step of a turn the fold condenses. The inset is NOT
CSS padding: it is threaded through `row_indent(width, depth)` and read by the row
builder AND the copy gutter, so a selection of a fold-covered row cannot eat a
character of its tool name; `detail_indent` does the same for the body rows and
`copy_gutter` reads it back. A bar does not nest further when it is inside an
open bar: the grid has two columns, and a run bar stepping in beside its turn
bar read as a second kind of row.

## 9. Colour

`local_operator/tui/bindings.py` carries one element per new ink: bar glyph,
bar name, summary, the failure tally in its two states (muted when answered,
danger+bold when dead), bar time, the live reading, the foot's success glyph,
foot text, foot failure and By-response narration. `muted` is the ramp's
secondary ink and clears the palette gate's 4.5:1 floor on both grounds in every
registered theme (`tests/unit/tui/test_palette_contrast.py`).

## 10. Known limits

- **The clock is a SUM of call durations, not wall clock.** Parallel calls are
  added, not overlapped, and a turn whose calls have no recorded durations shows
  no time. The design chose the sum because the numbers the bars already show
  are per-call intervals; a wall clock would disagree with them.
- **A page-seam turn gets no turn bar and no foot.** Its prompt row is outside
  the loaded window, so the bar would label a turn whose start the reader cannot
  see and hide exactly the rows they resumed to read. Its runs still fold.
- **`f`/`F` need a bar under the keyboard.** The jump focuses the call
  (deliberately); the next jump is taken from a bar, and a ToolCard does not
  claim the characters because it must keep forwarding typing to the composer.
- **Narration muting is a re-tint of the flattened message.** Inline code, links
  and headings in a narration block take the muted ink too; emphasis, fences and
  list structure are untouched.
- **A dead turn's notice is hidden in By response** because its foot restates the
  reason once. The notice remains in the transcript model (and in what the model
  is shown); only its paint is withheld.

## 11. The "9 browser actions over ~19 rows" report

Ben's first screenshot showed a bar counting 9 browser actions over a body that
looked like about 19 painted rows. Likely cause from code reading: the old
tool-only `work` bar RELEASED its tail when a member failed — `_release_splitter`
closed the group at the failure, so rows behind it painted outside the bar the
count had been taken over, and the reader saw more work than the bar admitted.
This was not reproduced live. The new layer removes the class of defect by
design: failures stay inside the run, the tally counts every call the bar hides,
and the bar and its coverage are derived from the same outline in the same pass,
so a count and its body cannot disagree.
