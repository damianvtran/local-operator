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
between). A loose narration line does not make a flat turn non-flat; a turn bar
is only drawn when it would hide at least `TURN_BAR_MIN_ROWS = 3` rows.

**The turn bar ends at the answer.** A row appended after the final answer — a
slash receipt, a system notice, an inbound peer message or a scheduled wake that
arrives while the agent is idle — is not work the answer condenses. It stays
visible, it is not counted in the bar's tally, and opening the bar cannot reveal
a second bar over the same calls.

**What never folds.** A call awaiting approval (`waiting`; a LIVE call's card is
`running`, so the session's pending ids decide), an `ask`, and a generated image
split a run and stay visible. A turn bar covers only tool rows and narration:
an error/failed notice, an interactive notice, a receipt (peer, wake, monitor), a
slash receipt, a streaming row and every other kind the outline does not name
stay visible with every bar closed. Rows a fold could hide are exempted first;
the layer never has to ask a row whether it may hide.

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
- The STORED mode is applied where the view is created: at boot next to the
  composer flags, and in the full-page subagent transcript. A setting is not a
  value only a change can deliver; the view's class default is a fallback for an
  unreadable file, not the boot layout.

## 3. Keys (on a focused fold bar only)

| key | action |
|---|---|
| enter / space | toggle the bar |
| up / down | walk the visible ledger, bars included |
| left | close; when already closed, step OUT to the parent bar |
| right | open; when already open, step INTO the first child (bar, else row) |
| `f` / `F` | next / previous failed call in the focused TURN, opening every bar over it, focusing the call and scrolling it into view |
| `+` | open every level of the focused bar's turn; keep the bar on screen |
| `-` | that turn back to its defaults (drop the turn's explicit choices, even when the keyboard is on a child bar) |
| esc | never closes a fold — it keeps its app-wide meaning |

**The printable keys are claimed on `FoldBarBlock` and nowhere else — plus the
one card the ladder is standing on.** `ExpandableActionBlock.on_key` forwards
every unbound printable to the composer, so binding `f`/`F`/`+`/`-` higher in
the hierarchy would swallow the first letter of a sentence typed while a tool
row had focus. A focused bar is a navigation surface, and there the design
deliberately spends those characters. The jump then lands the keyboard on the
CALL; `f`/`F` continue the ladder only while that card is the layer's failed-call
cursor, and on every other card the characters reach the composer as before, so
typing `found` on a row still works.
`c` was not chosen for anything: it is taken app-wide. `v` is not a key either —
the layout is the `display.transcript_display` setting plus `/transcript
by-turn|by-response`.

`f`/`F` land the keyboard on the CALL (that is where the reader acts). The
ladder is SCOPED to the turn the reader is standing in and circular inside it: a
clean turn reports `No failed calls in this turn` in the APP'S OWN toast slot
(`#toast-host`'s `Toast` — the surface every other gesture receipt uses, and the
one a headless test can read), never `App.notify`, whose separate rack this app
does not place and every test harness disables; and never a transcript row, which
the reader's own keystrokes would otherwise append. Pressing `f` twice from the
same card keeps stepping instead of typing an `f` into the draft.

## 4. Summary grammar

`N actions · kind, kind +M · ✗ K failed` — one grammar on every bar.

- Kinds use the shortened nouns in order of first appearance, at most 3 then
  `+M`. The three builtin nouns are catalogue copy (bash→`shell`,
  task→`subagents`, hub/send→`messages`), so the seven shipped locales read
  them; an MCP tool folds through the row's own display-name path
  (`mcp__linear_create_issue` → `create_issue`), never the wire identifier.
- A run bar is named `work` in By turn's ledger and has **no name** in By
  response's layout B: four identical cells on every run bar of a skimmable
  step list say nothing. A turn bar always names the turn; a sub bar names its
  tool.
- A same-tool sub bar says `tool ×N` in place of the count, because its name
  column already names the tool.
- `✗ K failed` is muted on a turn that ANSWERED (failures are history) and
  danger+bold only when the turn DIED.
- Time is right-aligned in the ledger's ONE duration column, at every depth: the
  cell is derived from the ledger's own right edge (`row_body_width(width, 0)`),
  never from the row's depth-reduced body width, so a sub-group's members, a
  sibling call in the same run and the bar over them all end their clock on the
  same cell (D12 — measuring per depth painted one column of durations as
  three). Focusing a bar does not move it.
- Time is right-aligned in the ledger's duration column. It is the SUM of the
  calls' known durations (`ToolCard.duration_s`), not a wall clock; a call with
  no recorded duration contributes nothing, and a span with none shows no time.
- That clock speaks the tool row's OWN grammar (`format_clock`): `<0.1s` under
  50 ms, one decimal under ten seconds, whole seconds under a minute, then
  `format_duration` (`1m57s`). `format_duration` alone floored a 0.5 s run to
  `0s` beside its own member rows reading `0.1s`, and a 12.95 s foot to `12s`
  where the calls said `13s` — the app disagreeing with itself in one column.
- **Narrow widths degrade in this order:** the bar's NAME (dropped whole — a
  truncated `wo…` names nothing), then the kinds clause, then the tally's WORD
  (`✗ 3 failed` → `✗3`). The failure count and the time are never dropped, the
  row never wraps, and past the count's cells the tally stands alone. A live bar
  replaces the clock with `◐ <tool> <summary> <elapsed>`, CLAMPED to the cells
  left after the count and a compact tally (it sheds the summary, then the tool
  name, then becomes the bare clock), and repaints at 1 Hz from mount.
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
- **"No answer" is not "failed".** A turn reads as dead only when the harness's
  own notice says so — a `turn failed` / `Stopped with an error` row — or when
  the last settled turn ends on an error notice. A steer, a deliberate stop and
  a `turn cut off` notice stay neutral: no danger tally, no `✗ turn failed`
  foot. The cut-off's own warning row stays visible.
- The ONE exemption: a turn that died or is the last settled turn with no answer
  pins its LAST failed call below the collapsed fold, so the reason is reachable
  without unfolding a sixteen-call batch. A mid-turn steer segment does not pin.
  `f`/`F` open whatever is needed for any failure.
- On a turn that answered, the tally is muted — history, not alarm (desktop
  issue #6's ruling, kept).
- A turn-failed notice keeps its stored text; the By-response FOOT renders the
  reason cleaned (no `[session incident …]` head, no `suggested action:` advice
  line addressed to the model, no "Take it into account…" trailer) and withholds
  the notice row's paint so the reason is stated once. Stored rows are never
  rewritten for a layout.
- **A turn's death is read from its TAIL, not its last row.** The harness says a
  turn failed at its close, and a transcript keeps appending afterwards (a slash
  command's receipt, a system notice, a peer message that arrived while the
  agent was idle). Those rows cannot unsay it: the failure is the non-neutral
  error notice anywhere after the turn's last WORK row — a call, a remark or the
  answer — so a receipt below it neither withdraws the foot nor repaints the raw
  failure row it withheld (Q13). A failure the turn recovered from and worked
  past is not in the tail and does not count.

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
nothing to total. A turn that neither answered nor failed (a steer segment, a
stop, a cut-off) gets NO foot: `✓` would claim it succeeded and `✗` would state
an outcome nobody diagnosed. By turn: no foot.

A dead turn's foot also REPLACES every stored row that restates the same
failure — the `turn failed` stop notice and the raw `[session incident …]`
record both, when a resumed transcript carries both — so the reason is painted
once, not three times.

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
  draws **two cells deeper again** (four total); a sub bar nests **two cells
  deeper than the run it belongs to** (its own level 2), and the calls revealed
  under it take the level below that (six total), because the prototype's
  `browser ×16` bar must read as owned by the run, not as its sibling. A pinned
  failure that is a MEMBER of such a sub-group takes the group's own column, not
  the run's: the calls above it are revealed at six, and a failure one step out
  of its own group read as a different kind of row (D11);
- a call's own DETAIL rows (`action:`/`target:`/`error:`, output, args) stay one
  cell RIGHT of the row at every level, as they always did at the plain spine
  (`detail_indent`). They used to sit on the fixed `OUTPUT_INDENT`, so a row
  revealed at col 6 painted its own body at col 4 — the row walked right and
  left its content behind.

A receipt card (scheduled wake, inbound peer message) steps with the tool rows
around it: the three are ONE ledger, and a receipt left on the plain spine while
the tool rows beside it stepped in is the column defect the operator reported
long before the fold existed ("the peer message doesn't have the proper left
padding like the other tool cards"). Receipt bodies follow their row like a
card's. Notices, prose and the fold's own bars are not ledger rows and keep
their column.

The inset is NOT CSS padding: it is threaded through `row_indent(width, depth)`
and read by the row builder AND the copy gutter, so a selection of a fold-covered
row cannot eat a character of its tool name; `detail_indent` does the same for
the body rows and `copy_gutter` reads it back. A bar does not nest further when it is inside an
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
- **`f`/`F` continue from the card they landed on.** The card claims those two
  characters only while it is the ladder's cursor; every other row forwards them
  to the composer as before, and the ladder is scoped to the turn the keyboard
  is standing in.
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
