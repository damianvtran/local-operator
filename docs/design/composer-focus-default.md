# Composer focus: the input is where the keyboard lives

Design for two reproduced focus defects and one they are both instances of.
Proposal only — no production code here.

Baseline: `origin/main` @ `23d0e171`, Textual 8.2.8, measured in
`~/workspace/repos/lo-focus-input` at 120x40 unless a size is named.

---

## 1. The problem as I found it

The user reported three things. They are one thing.

> "sometimes there's an issue where focus is lost or when you click on a tool
> call and click back to the text input it doesn't appear focused in style"

> "if a btw or ask or something is not open then the input is the focus"

> "if I were to hit tab it would just continue hovering/selecting and cycling
> the next tool call and/or message from the context history"

The app has exactly one text input and a ledger of rows that are focus stops
so the keyboard can expand them. Every reported symptom is the same defect
class: **focus can come to rest somewhere that is not the composer, and there
is no cheap way back.** The chevron is not lying about that; it is reporting
it accurately.

`_sync_composer_focus` (`app.py:14867-14891`) is correct and stays correct.
It reads `editor.has_focus`, it is idempotent, it fires on both
`on_descendant_focus` (`app.py:14682`) and `on_descendant_blur`
(`app.py:14685`). Confirmed in every probe below: the dock class tracked
`Editor.has_focus` exactly, in all seventeen states I put it through. Nothing
in this design touches it.

### 1.1 Defect A — the composer has a dead frame around it

Reproduced. Clicking the shell's padding while a `ToolCard` holds focus does
nothing at all:

```
[1] after clicking tool card:              focused=ToolCard  dock class=False
[2] after clicking shell PADDING at (2,34): focused=ToolCard  dock class=False  <-- BUG
[3] after clicking editor body:            focused=Editor    dock class=True
```

Mechanism, and it is entirely Textual's: `Screen._forward_event` calls
`get_focusable_widget_at` on `MouseDown` (`screen.py:1933-1939`), which walks
`ancestors_with_self` for the first node with `focusable` true
(`screen.py:709-712`). `#input-shell` is a plain `Container`
(`app.py:6736`) and `#input-dock` likewise (`app.py:6706`); neither is
focusable, and the walk terminates at the screen with `None`. Textual then
does **not** call `set_focus(None)` on that path — it only does that on
`NoWidget` — so focus stays exactly where it was. The click is swallowed
silently.

**The dead zone is much larger than the 1-cell padding.** I mapped every
coordinate of the dock at 120x40 (`get_widget_at` × `get_focusable_widget_at`
per cell). Only ONE of five dock rows contains any focusable cell:

```
row 34 (top pad)    x=1..118  -> #input-shell   / focusable: None   DEAD
row 35 (input row)  x=1       -> #input-shell   / focusable: None   DEAD
                    x=2..3    -> #prompt-chevron/ focusable: None   DEAD
                    x=4..117  -> Editor         / focusable: Editor  live
                    x=118     -> #input-shell   / focusable: None   DEAD
row 36 (band)       x=2..117  -> #status-band   / focusable: None   DEAD
row 37 (band)       x=2..117  -> #status-band   / focusable: None   DEAD
row 38 (bottom pad) x=1..118  -> #input-shell   / focusable: None   DEAD
```

That is 4 of 5 rows fully dead, plus 4 dead columns on the live row. The
chevron — the app's own focus affordance, the thing that goes bright when the
composer is focused — is itself a dead cell. A user clicking the visual
target that means "focused" gets nothing.

It is worse in the boot layout, where `#input-shell` is clamped to a centred
card (`local_operator.tcss:1078`) and `#input-dock` spans the full width
behind it. Measured at 120x40 on boot: dock `Region(x=1, y=29, w=118, h=10)`,
shell `Region(x=19, y=29, w=82, h=5)`. Every cell in the 18-column gutter
either side, and the five dock rows below the card, hit `#input-dock`
directly — also dead.

So the report is literal and the user is describing it precisely. Focus never
came back. They clicked the input, the input did not take focus, and the
chevron correctly stayed dim.

### 1.2 Defect B — Tab walks the ledger

Reproduced. From a focused tool card, with three cards in the transcript:

```
tab #1 -> ToolCard    tab #2 -> ToolCard    tab #3 -> Editor
```

`Screen.BINDINGS` binds `tab` to `app.focus_next` (`screen.py:269`). The
focus chain in the real app with three cards is:

```
TranscriptView#transcript, ToolCard, ToolCard, ToolCard, Editor
```

Every focusable row is a tab stop, so the number of presses needed to reach
the composer scales with the length of the conversation. In a real session
that is tens to hundreds. The user's "it would just continue hovering/
selecting and cycling" is exactly this.

### 1.3 What I found that was not reported — and it matters

Three findings, each of which changes the design.

**(i) The documented Shift+Tab route into the ledger is already dead.**
`tool_card.py:872-877` says `can_focus` exists so "Shift+Tab out of the
composer lands on the last action". `test_shift_tab_out_of_the_composer_lands_on_the_last_action`
(`test_tool_card.py:1732`) still passes — but it runs against `_ComposerApp`,
a stripped harness, not `OperatorApp`. In the real app:

```
focused before: Editor
after shift+tab: Editor        <-- cycle_effort took it
```

`app.py:2439` binds `shift+tab` to `cycle_effort` with `priority=True`, and a
priority binding is matched before the focused widget sees the key. **The
keyboard has had no way into the transcript since that binding landed.** The
comment at `app.py:2436-2438` even acknowledges the cost — "It costs reverse
focus cycling, which nothing in this app needs" — without noticing that the
tool card's whole justification for being focusable is that reverse cycling.

This is a BRIEF-level finding about the codebase, not about any coder's work:
`test_tool_card.py:1732` documents a behaviour the shipped app does not have.
It must be corrected as part of this work, or it will keep asserting a false
claim about the product.

**(ii) `TranscriptView` is a focus stop that silently eats every keystroke.**

```
click empty transcript area -> focused=TranscriptView  dock class=False
typed 'x' on TranscriptView -> editor.text=''  focused=TranscriptView
```

Clicking any blank area of the conversation moves focus to the container, and
from there every printable key vanishes. `ExpandableActionBlock.on_key`
(`transcript.py:778-808`) bounces printables to the composer — but that
handler is on the ROWS, and `TranscriptView` is their parent, so it has no
such guard. This is the identical defect `todo_panel.py:409-419` already
names in its own comment ("the app looked focused while every keystroke
vanished into a widget that does nothing with them") and fixed there by
dropping `can_focus`. It is also the state that `set_interactive`
(`session_presentation.py:195`) deliberately lands focus on, calling it "on
screen and answers `enter` with nothing" — which is true, and is the bug.

`TranscriptView` does not need focus for scrolling. Its scroll keys work when
focused (up/down/pageup/pagedown/home/end all measured moving `scroll_y`),
but the app already binds `ctrl+home`/`ctrl+end` to `transcript_home`/
`transcript_end` (`app.py:2543-2544`) which drive it without focus — measured
working with the Editor focused. Mouse wheel does not require focus either.

**(iii) `action_stop` (Esc) does not restore composer focus.** Measured:
card focused, `action_stop()`, still `focused=ToolCard`. The app's one "get
me out of here" key leaves the user in the ledger.

---

## 2. Decisions

### Q1 — Defect A: forward dock clicks to the composer

**Decision: option (a). An `on_click` handler on `#input-shell` AND
`#input-dock` that focuses the editor when the click did not land on
something else that wanted it.**

I prototyped it (`Container` subclass with `on_click` swapped onto the live
widget, real `OperatorApp`, real pilot clicks). Results:

```
top-pad         @(60,34)  -> focused=Editor  dockclass=True  handler fired=1
left-of-chevron @(1,35)   -> focused=Editor  dockclass=True  handler fired=1
chevron         @(2,35)   -> focused=Editor  dockclass=True  handler fired=1
band-row        @(60,36)  -> focused=Editor  dockclass=True  handler fired=1
editor-body               -> focused=Editor  handler fired=1  (bubbled, harmless)
caret after pad click: Selection(start=(0,5), end=(0,5))  text='hello world'
read-only pad click -> focused=ToolCard  can_focus=False  fired=1
```

Every property the brief asked me to watch holds:

- **Caret and draft survive.** The handler calls `editor.focus()` and nothing
  else. The caret stayed at column 5 of `hello world`. It does not place the
  caret from the click position, which is right: a click on the padding is
  "put me back in the input", not "put the caret at this coordinate", and
  there is no coordinate in the padding that maps to a document position.
- **Read-only is respected for free.** `_set_composer_read_only` drops
  `editor.can_focus` (`app.py:22118`); the handler guards on
  `editor.can_focus` and the click became a no-op on the subagent page, which
  is correct — that page's whole argument is that the dock is not where you
  are.
- **The editor's own mouse handling is untouched.** A click on the editor
  body is handled by `Editor._on_mouse_down`/`_on_mouse_up`/`_on_click`
  (`editor.py:4391/4405/4438`) first and Textual has already focused it via
  `get_focusable_widget_at` before the `Click` bubbles. The shell handler
  then runs and finds `editor.has_focus` already true, so it does nothing.
  Marker selection and drag-selection measured intact.

Implementation shape (not code): a small `Container` subclass — one class,
used for both `#input-shell` and `#input-dock` — whose `on_click` focuses the
editor if `editor.can_focus and not editor.has_focus`. It must NOT call
`event.stop()`: `Click` bubbles up from children, and stopping it here would
be stopping an event that has already been handled by whoever owned it. The
guard is "focus only if nothing already took it", not "claim the event".

**Why both widgets, not just the shell.** The boot layout's gutter and the
rows below the card hit `#input-dock` directly (§1.1). Handling only the
shell leaves the boot splash — the first frame a new user ever sees — with a
dead gutter on both sides of the card.

**Rejected — (b) make the shell focusable and delegate.** Adding
`can_focus = True` to `#input-shell` puts a second composer-ish node in the
Screen's focus chain, which makes Defect B measurably worse (one more tab
stop between the ledger and the editor) and creates a state where the shell
holds focus while the chevron is dark. Delegation would then need
`allow_focus` overrides plus a focus-forwarding hook, which is more machinery
for a strictly worse focus chain. Rejected.

**Rejected — (c) remove the padding.** The padding is load-bearing and
documented: `local_operator.tcss:487-489` records that the editor text used
to sit flush against the panel edge and "read as an unfinished box rather
than a deliberate borderless field". It is the reason "no border" looks
chosen. Removing it trades a visual rule the team paid for against a defect
that has a direct fix — and it would not even work, because the band rows
and the boot gutter are dead for a different reason (a non-focusable sibling
widget, not padding). Rejected on both counts.

**Is there a Textual-idiomatic route?** I looked. `Widget.focus_on_click()`
(`widget.py:720`) only gates whether an ALREADY-focusable widget takes focus
from a click; it cannot redirect. There is no `focus_delegate` in 8.2.8. The
`on_click`-forwarding pattern is what the codebase already does elsewhere for
non-focusable chrome (`model_picker.py:1050`, `session_presentation.py:203`),
so this IS the local idiom.

### Q2 — Defect B: Tab on a transcript row returns to the composer

**Decision: bind `tab` on the focusable-row base class to an action that
focuses the composer. Keep the rows in the focus chain. Do not touch the
editor's Tab.**

Prototyped on a real `ToolCard` subclass in the real app:

```
start:            ToolCard
after tab:        Editor   dockclass=True   text=''      <-- no stray whitespace
up from last:     True     (focus_neighbour still walks)
enter expands:    True     (row keys intact)
read-only tab ->  ToolCard                                <-- correctly refused
```

A widget binding beats the Screen binding — Textual resolves the focused
widget's bindings before bubbling to the Screen — so this overrides
`tab -> app.focus_next` for exactly the rows that need it and nowhere else.

Placement: on **`TranscriptBlock`** (`transcript.py:2843`'s neighbour, the
base at `transcript.py`'s `class TranscriptBlock(Static)`), not on
`ExpandableActionBlock`. `TranscriptBlock.BINDINGS` is currently `[]`, and
the focusable rows are not all `ExpandableActionBlock`s: the interactive
notices (`HistoryPageNotice`, `OlderHistoryNotice`, `DraftRecoveryNotice` —
`session_presentation.py:108/128/214`) descend from `NoticeBlock` →
`TranscriptBlock`. I verified the binding reaches them: a
`DraftRecoveryNotice` subclass with the binding merged to `['enter', 'tab']`
and Tab from it landed on the Editor with the dock class set. Binding on
`ExpandableActionBlock` would leave three notice types still trapping Tab.

Two details that the prototype settled:

- **`_BOUND_KEYS` needs `tab` added, and it happens automatically.**
  `ExpandableActionBlock._BOUND_KEYS` (`transcript.py:694-698`) is derived
  from `BINDINGS` at class-definition time and is used by `on_key` to exclude
  the row's own keys from the printable-bounce. Tab is not printable
  (`Key("tab","\t").is_printable` is `False` — measured), so `on_key` returns
  early on it regardless and the two paths cannot collide. But if the binding
  lands on `TranscriptBlock` while `_BOUND_KEYS` is computed on
  `ExpandableActionBlock` from its own `BINDINGS`, the derivation must be
  reviewed so it still sees the inherited binding. Measured on the prototype:
  a subclass declaring `BINDINGS = [tab...]` merged correctly to
  `['down','enter','space','tab','up']` via `_merged_bindings`, but
  `_BOUND_KEYS` read `['down','enter','space','up']` — it iterates the class's
  OWN `BINDINGS`, not the merged map. **The brief must state that
  `_BOUND_KEYS` should be derived from `_merged_bindings` or the new binding
  added to the same list `_BOUND_KEYS` reads.** This is the single most
  likely way this slice ships subtly wrong.
- **Read-only guard.** The action must check `editor.can_focus` — measured
  refusing correctly on the subagent page. Without it, Tab on a row would
  focus a composer that refuses every key.

**What accessibility affordance is lost, and how it is preserved.** Strictly:
Tab-forward through the ledger. That is not a loss, because —

- **Up/Down already do it, and do it better.** `focus_neighbour`
  (`transcript.py:3508-3535`) walks actionable rows, SKIPS inert prose,
  clamps to transcript-top and composer-bottom. Tab walks everything
  including the container. The correctly-scoped key already exists; Tab was
  the badly-scoped duplicate.
- **The entry point must be repaired.** Shift+Tab into the ledger is
  currently dead (§1.3-i), so today there is no keyboard route in at all.
  This design does not fix that — see §5, Deferred — but it must be recorded,
  and the false test corrected, in the same MR.

**Rejected — remove transcript rows from the Tab chain entirely** (the
`todo_panel.py:420` treatment, `can_focus = False`). That kills
click-to-expand-by-keyboard, kills `focus_neighbour`, and directly reverses
the documented decision at `tool_card.py:872-877` which exists so the ledger
is reachable "in a terminal without mouse reporting". A fix that removes an
accessibility affordance to fix a navigation annoyance is the wrong trade.
Rejected.

**Rejected — make the transcript a single focus stop.** Appealing (one tab
stop, rows reached by arrows from it), but it requires `TranscriptView` to be
the stop — and `TranscriptView` is the widget that silently eats keystrokes
(§1.3-ii). It would make the swallowing state the DEFAULT landing place.
Rejected. Slice 3 removes that stop instead.

**Not touched: the editor's Tab.** TUI-013 is `tab_behavior="indent"`
(`editor.py:2074`) — Tab inside the composer completes a picker row or
indents, and never moves focus. Pinned by `test_app_pilot.py:3654`. Nothing
in this design changes it. `keymap.py:180` keeps `tab` reserved and
unbindable; this changes what Tab MEANS contextually, not who may bind it.

### Q3 — "if a btw or ask is not open then the input is the focus"

This is the request that can do the most damage if taken literally, so the
policy is deliberately narrow.

**Decision: name the predicate `_focus_is_claimed()`. Reassert composer focus
on EVENTS, never on a condition being observed true.**

#### The predicate

`OperatorApp._focus_is_claimed() -> bool` — "some surface has a legitimate
claim on the keyboard that the composer must not take." It lives in `app.py`
beside `_live_prompt()` (`app.py:14848-14865`), which is its seed and which it
should call rather than duplicate.

It must return True for every one of these, each of which I verified is
tracked somewhere today:

| Claimant | Probe | Source |
|---|---|---|
| unanswered attached approval | `self._approval is not None and not answered and is_attached` | `app.py:14853-14856` |
| unsettled ask picker | `self._ask_screen is not None and not settled and is_attached` | `app.py:14859-14862` |
| `/btw` aside card | `self._aside_is_open()` | `app.py:4160` and 5 other call sites |
| full-page subagent view | `self._subagent_view is not None` | `app.py:3500` |
| org chart view | `self._org_chart_view is not None` | `app.py:3550` |
| settings view | `self._settings_view is not None` | `app.py:3555` |
| credential/login key prompt | `self._key_prompt is not None` | `app.py:3398` |
| session sidebar focused | `self._session_sidebar.has_focus` | `app.py:6295` |
| a pushed Screen above the default | `len(self.screen_stack) > 1` | catch-all |
| composer is read-only | `not self._editor().can_focus` | `app.py:22118` |

The last two are the important ones. `len(self.screen_stack) > 1` is the
generic guard for any modal route not enumerated above (`/resume`'s
`session_picker` is a pushed Screen — `conftest.py:352` records this), so a
future overlay is safe by default rather than by remembering to update a
list. And the read-only check makes the predicate subsume the
`_set_composer_read_only` constraint instead of leaving it to each call site.

The predicate is READ-ONLY and must not itself move focus. Every branch is
wrapped so a stripped harness with no composer degrades to True ("something
might be claiming it") rather than raising out of a focus path — refusing to
steal is always the safe direction.

#### The policy: three triggers, all of them EVENTS

**Never a timer. Never a poll. Never "if the composer is empty".** The
`route_key_to_live_prompt` docstring (`app.py:14690-14714`) records what that
costs, and it is the most expensive lesson in this file: an earlier revision
"bounced focus onto the prompt whenever the composer was empty, and the
composer is empty *exactly* when the user is about to start typing". Measured
through the real gate, typing `yes do it` at a live `rm -rf` prompt
**authorised the call** and left `es do it` in the buffer (F3, review round
2). And `app.py:14741-14746` records the second attempt (F9, D18): inferring
"the user has finished typing" from the buffer cost them a message, twice,
and the conclusion was that only a NAMED GESTURE can carry that meaning.

The generalisation this design adopts: **focus moves when the user does
something, or when a surface they were using goes away. It never moves
because a condition became true while they were sitting still.** A trigger
tied to an event has a user action behind it; a trigger tied to a state does
not, and will fire between two keystrokes.

The three triggers:

**T1 — on overlay close.** Where a close path has no focus restore, or
restores to a widget that is gone. Three of the five already do the right
thing: `_close_org_chart_view` (`app.py:21583-21586`),
`_close_settings_view` (`app.py:21795-21797`) and `_close_subagent_view`
(`app.py:21491-21497`) all fall back to `self._editor().focus()`;
`_close_aside` ends on `editor.focus()` (`app.py:28545`). **The work here is
mostly an audit, not new code**: confirm each close path lands on the
composer when its restore target is stale, and add the fallback only where it
is missing. This trigger is safe by construction — an overlay closing IS a
user action.

**T2 — on Esc (`action_stop`).** Measured: Esc from a focused card leaves
focus on the card. Esc is the app's one "stop / get me out" key and the
docstring (`app.py:15438-15460`) is explicit that it means one thing wherever
focus happens to be. It should end with the composer focused when
`_focus_is_claimed()` is False — after every existing branch, so the aside's
`esc close`, the subagent page's exit and the approval denial all still
consume the key first and are unaffected.

**T3 — on a click that hits nothing focusable.** Q1's handler is this trigger
for the dock. The same reasoning covers a click on `TranscriptView`'s blank
area, which Slice 3 handles by removing that focus stop entirely (§Q2, and
below) — after which a click on blank transcript hits a non-focusable
container and, per §1.1's mechanism, leaves focus alone. That is not enough:
it leaves focus on the card. So Slice 3 also gives `TranscriptView` an
`on_click` that returns focus to the composer when the click landed on no
block. Same guard, same shape as Q1.

**Explicitly rejected as a trigger: "on turn end."** The brief lists it as a
candidate. It is the `route_key_to_live_prompt` failure with extra steps: a
turn ends on the AGENT's schedule, not the user's, and the user may be
mid-keystroke on a tool card (reading output, about to press Enter to expand)
when it lands. The one thing that must never happen is focus moving under a
key that is already in flight. **Rejected.**

#### What must NEVER be stolen from

A hard list. Each of these holds focus legitimately and reassertion must
stand down for all of them — this is what `_focus_is_claimed()` is FOR:

1. **A live approval or ask prompt** that has pulled focus deliberately via
   `_prompt_wants_the_keyboard` (`app.py:14834-14846`) — the multi-select
   case, answered by Space and Enter, the one question the routed keys cannot
   reach. Taking focus back would make it unanswerable.
2. **Any pushed Screen** — the `/resume` session picker and anything modal.
3. **The `/btw` aside**, the subagent page, org chart, settings — each owns
   its own keyboard while up.
4. **The session sidebar** when the user pressed F9 to focus it
   (`app.py:6295`). It has its own restore path (`_restore_sidebar_focus`,
   `app.py:6305-6310`) which already falls back to the editor.
5. **A read-only composer.** `_set_composer_read_only(True)` drops
   `can_focus` and deliberately blurs (`app.py:22118-22127`); focusing it
   would paint a caret in a field that refuses every key.
6. **A key prompt** (`self._key_prompt`) — it is taking a credential.
7. **Any moment there is a keystroke in flight.** Structural, not a state
   check: this is why T1/T2/T3 are all events with a user action behind them
   and why "on turn end" is out.

### Q4 — Slicing

Four slices. Three are independent with disjoint file ownership; the fourth
is sequenced behind two of them. Two coders.

```
        parallel                        then
  ┌─────────────────┐            ┌──────────────────┐
  │ S1  dock clicks │───────┐    │ S4  Esc + audit  │
  │   app.py + tcss │       ├───▶│   app.py         │
  └─────────────────┘       │    │   (needs S1's    │
  ┌─────────────────┐       │    │    predicate)    │
  │ S2  Tab on rows │───────┘    └──────────────────┘
  │   transcript.py │
  └─────────────────┘
  ┌─────────────────┐
  │ S3  TranscriptView stop      │  ← can run parallel with S1;
  │   transcript.py              │    SHARES transcript.py with S2
  └──────────────────────────────┘
```

**The honest answer on disjointness: S2 and S3 both own
`local_operator/tui/widgets/transcript.py`. They cannot run in parallel.**
Merge them into one brief for one coder, or sequence them. I recommend
**merging S2 and S3 into a single slice** — they are the same subject (what
the transcript does with focus), they share the test file, and splitting them
buys nothing but a merge conflict.

Revised, and this is the plan I recommend:

#### Slice A — "the dock is clickable" (coder 1)

**Owns:** `local_operator/tui/app.py`,
`local_operator/tui/local_operator.tcss` (comment only, if any),
`tests/unit/tui/test_composer_focus.py` (new).

**Does:**
1. A `Container` subclass with the forwarding `on_click`, used for both
   `#input-dock` and `#input-shell` (`app.py:6706`, `app.py:6736`).
2. `_focus_is_claimed()` beside `_live_prompt()` (`app.py:14865`), with every
   claimant in §Q3's table.
3. The handler consults it.

**Does NOT:** touch `_sync_composer_focus`, the editor, or the transcript.

#### Slice B — "the transcript hands focus back" (coder 2)

**Owns:** `local_operator/tui/widgets/transcript.py`,
`tests/unit/tui/test_transcript_focus.py` (new), and the docstring correction
in `tests/unit/tui/test_tool_card.py:1732`.

**Does:**
1. `tab` binding on `TranscriptBlock` → return to composer, guarded on
   `editor.can_focus`.
2. Fix `_BOUND_KEYS` derivation so it sees inherited bindings
   (`transcript.py:694-698`) — see §Q2.
3. `TranscriptView.can_focus = False` plus an `on_click` returning focus to
   the composer when the click hit no block, with the comment recording why
   (the `todo_panel.py:409-419` precedent, and the `session_presentation.py:195`
   comment that must be updated because its stated landing place no longer
   exists).
4. Correct `test_shift_tab_out_of_the_composer_lands_on_the_last_action`'s
   claim (§1.3-i): it must either move to `OperatorApp` and assert what
   actually happens, or state in its docstring that it pins the harness
   behaviour and that the real app's `shift+tab` is `cycle_effort`.

**Does NOT:** touch `app.py`, the editor, or `focus_neighbour`'s logic.

**Blast radius warning for the brief:** step 3 changes where
`NoticeBlock.set_interactive`'s blur lands (`session_presentation.py:186-195`).
That comment documents a measured failure — blurring after clearing
`can_focus` sent focus ~770 rows up to the topmost `ToolCard`. With
`TranscriptView` no longer focusable, `Screen._reset_focus` will pick a
different neighbour. **The coder must re-measure that path**, not assume it
still lands safely. This is the highest-risk change in the design.

#### Slice C — "Esc comes home" (coder 1, after Slice A)

**Owns:** `local_operator/tui/app.py` (`action_stop` at `app.py:15438`, plus
the close-path audit), `tests/unit/tui/test_composer_focus.py` (extends
Slice A's file).

Sequenced behind A because it calls `_focus_is_claimed()` and touches the
same file. Same coder, so no ownership conflict.

**Does:** T1's audit of the four close paths; T2's Esc restoration.

#### Sequencing note

Slice B's step 3 and Slice A's dock handler interact at exactly one point: a
click on blank transcript. Neither breaks the other (different widgets,
different handlers), but the integration test for "click anywhere sane and
the composer is focused" belongs to whoever lands second. The lead should put
that test in Slice C.

### Q5 — Test plan

Repo pilot style throughout: `app.run_test(size=...)`, `pilot.click(...)`,
`pilot.press(...)`, then assert `app.focused`,
`dock.has_class(COMPOSER_FOCUSED_CLASS)`, and where the caret is the claim,
`composer_cells(app)` / `caret_cells(...)` from `tests/unit/tui/conftest.py:340`.

**One measurement discipline this design depends on:** do not hardcode click
coordinates. My first prototype run failed with `OutOfBounds` on
`(118, 35)` and `(60, 38)` because the dock's geometry differs between the
boot layout and the settled layout. Every click offset must be derived from
`shell.region` / `editor.region` at test time.

#### Slice A tests — `tests/unit/tui/test_composer_focus.py`

1. `test_clicking_the_shell_padding_returns_focus_to_the_composer` — the
   Defect A repro, promoted. Card focused → click `(shell.region.x + w//2,
   shell.region.y)` → `app.focused is editor` and dock class True.
2. `test_every_dead_cell_of_the_dock_returns_focus` — parametrised over the
   five sites I measured (top pad, left of chevron, the chevron itself, a
   band row, the boot gutter). The chevron case is the one that matters most
   to the report: the affordance that says "focused" must be clickable.
3. `test_a_dock_click_does_not_move_the_caret` — type text, put the caret
   mid-buffer, focus a card, click the padding: caret and text unchanged.
   Asserted via `composer_cells`.
4. `test_a_dock_click_is_refused_while_the_composer_is_read_only` —
   `_set_composer_read_only(True)`, click padding, focus unchanged,
   `editor.can_focus` still False.
5. `test_clicking_the_editor_body_still_places_the_caret` — the regression
   guard for the forwarding handler stealing the editor's own mouse work.
6. `test_focus_is_claimed_covers_every_overlay` — parametrised over the
   §Q3 table, each state opened through its real path.
7. Boot-layout variant of (2), since the dock gutter only exists there.

**Existing tests at risk:** `test_boot_layout.py` (19 references to
`#input-shell`/`#input-dock`; it queries by id and reads `.region`, so a
`Container` subclass keeping the same id and CSS is transparent — but the
CSS selector assertion at `test_boot_layout.py:303` parses the tcss by regex
for `Screen.boot.boot-card #input-shell`, so the selector must not change);
`test_composer_seam.py` (measures dock fill row by row — a subclass with no
style change must not perturb it).

#### Slice B tests — `tests/unit/tui/test_transcript_focus.py`

1. `test_tab_from_a_tool_card_returns_to_the_composer` — Defect B repro
   promoted: ONE press, from the first of three cards, lands on the Editor
   with the dock class set and `editor.text == ""` (no stray tab character).
2. `test_tab_returns_from_every_focusable_row_kind` — parametrised over
   `ToolCard`, `WakeBlock` (`transcript.py:1466`), `PeerMessageBlock`
   (`transcript.py:1801`), `HistoryPageNotice`, `OlderHistoryNotice`,
   `DraftRecoveryNotice`. This is the test that catches the binding being put
   on `ExpandableActionBlock` instead of `TranscriptBlock`.
3. `test_up_and_down_still_walk_the_ledger` — `focus_neighbour` unaffected.
4. `test_enter_and_space_still_expand_a_focused_row` — `_BOUND_KEYS` intact.
5. `test_typing_on_a_focused_row_still_reaches_the_composer_intact` — the
   existing `test_tool_card.py` guarantee, re-asserted here because
   `_BOUND_KEYS` is being changed.
6. `test_tab_on_a_row_is_refused_while_the_composer_is_read_only`.
7. `test_clicking_blank_transcript_leaves_focus_on_the_composer`.
8. `test_a_notice_losing_interactivity_does_not_land_focus_off_screen` — the
   `session_presentation.py:186` path, re-measured with `TranscriptView` no
   longer focusable. Assert the landing widget is on screen.

**Existing tests at risk:**
`test_tool_card.py:1732` (`test_shift_tab_...`) — must be corrected, not
just kept green; `test_tool_card.py:1756`
(`test_typing_on_a_focused_row_reaches_the_composer_intact`) — direct
`_BOUND_KEYS` dependency; anything in `test_app_pilot.py`,
`test_ask_picker.py`, `test_ghost_text.py`, `test_command_picker.py`,
`test_settings_view.py`, `test_slash_echo.py`, `test_inline_credential.py`,
`test_team_chart.py` that presses `tab` (all nine files do) — none should be
affected since none presses Tab on a transcript row, but the full TUI suite
is the gate; `test_bindings.py`; and any snapshot in
`tests/unit/tui/__snapshots__/` that captures a focused transcript row.

#### Slice C tests

1. `test_esc_returns_focus_to_the_composer_from_a_tool_card`.
2. `test_esc_does_not_steal_focus_from_a_live_prompt` — the critical
   negative. Parametrised over the multi-select approval that legitimately
   holds focus.
3. `test_closing_each_overlay_lands_on_the_composer` — parametrised over
   aside, subagent view, org chart, settings.
4. `test_closing_an_overlay_whose_restore_target_is_gone_lands_on_the_composer`.
5. Integration: `test_the_composer_is_focused_after_any_ordinary_gesture`.

---

## 3. What I recommend NOT doing

- **Do not touch `_sync_composer_focus`.** Verified correct in seventeen
  states. The chevron has been telling the truth throughout.
- **Do not change the editor's Tab.** TUI-013 stands.
- **Do not add a "restore focus" timer, poll, or idle check**, and do not
  gate any focus move on the buffer being empty. See §Q3.
- **Do not make transcript rows non-focusable.** That is the `todo_panel`
  treatment and it is wrong here for the reasons at `tool_card.py:872-877`.
- **Do not fix the dead Shift+Tab in this MR.** See §5.

---

## 4. Risks to watch during rollout

**R1 — HIGH. `TranscriptView.can_focus = False` moves where a blurring
notice lands.** `session_presentation.py:176-198` documents a measured
failure (focus jumping ~770 rows to the topmost ToolCard, with the user's
next Enter expanding a card they cannot see). That comment's fix depends on
`TranscriptView` being the safe landing place. Removing it invalidates the
stated remedy. Must be re-measured, not reasoned about. Slice B test 8.

**R2 — MEDIUM. `_BOUND_KEYS` is derived from a class's own `BINDINGS`, not
the merged map** (measured). A `tab` binding inherited from `TranscriptBlock`
will not appear in `ExpandableActionBlock._BOUND_KEYS` unless the derivation
changes. Tab is not printable so `on_key` is not affected today — but the
list is a correctness invariant ("the row's own keys") that would silently
become false.

**R3 — MEDIUM. The `Click` event bubbles through the dock on every editor
click.** The forwarding handler runs on every composer interaction. It must
be a cheap `has_focus` check and must not `event.stop()`. Watch for
double-handling with the editor's marker selection (`editor.py:4405`) and
with `model_picker.on_click` (`model_picker.py:1050`), both of which mount
inside `#input-shell`.

**R4 — MEDIUM. The predicate's claimant list is a maintenance surface.** Ten
entries, scattered across six mechanisms. A new overlay that forgets to
register will have focus stolen from it. The
`len(self.screen_stack) > 1` catch-all covers pushed Screens; a new
in-place overlay (the `_settings_view` pattern) is the exposed case. Consider
whether those three view attributes can be unified behind one accessor — but
that is a refactor and belongs in its own change, not this one.

**R5 — LOW. Boot-layout geometry.** The dock is 10 rows and the shell 5 in
the boot layout versus 5 and 5 settled. Any test with a hardcoded offset will
be flaky or `OutOfBounds`. Derive from `.region`. I hit this twice.

**R6 — LOW. Snapshot churn.** A focus-band repaint on a row that no longer
takes focus may move ink in `tests/unit/tui/__snapshots__/`.

**R7 — Reported-but-unfixed.** The user said "sometimes focus is lost". Slice
A and B explain and fix every loss I could reproduce. If a report survives
this change, the remaining suspects are the session-swap path
(`draft.focus_id`, `app.py:5805`) and `_sidebar_focus_restore`
(`app.py:6305`), neither of which I found broken but neither of which I
exercised under a real session swap.

---

## 5. Deferred, with a recommendation

**The keyboard has no way into the transcript.** `shift+tab` is
`cycle_effort` at `priority=True` (`app.py:2439`) and the documented entry
route (`tool_card.py:872-877`) does not exist in the shipped app. After this
design lands, the ledger is reachable by mouse and by Up/Down once you are
already in it — and there is no way to get in from the keyboard.

I recommend **not** fixing it in this MR: it is a keymap decision (which key
becomes the door) with its own trade-offs, and `keymap.py`'s reserved-key
rules apply. It should be its own ticket. What this MR MUST do is stop
`test_tool_card.py:1732` from asserting a behaviour the product does not
have — Slice B step 4.

The evidence that would settle the keymap question: whether any user has ever
reached the ledger by keyboard since the `cycle_effort` binding landed. `git
log -S 'cycle_effort'` against the binding line dates it; usage telemetry, if
the analytics session table records key actions, answers the rest.
