# In-app session sidebar

The sidebar is an opt-in terminal view over existing session runtimes, not another
runtime or a new scheduler. `Ctrl+B` and `/sidebar` toggle visibility without moving
the editor caret. `F9` and `/sidebar focus` enter the list; F9 returns focus while
leaving it open, and Escape dismisses it and returns to the last usable surface.
A pointer press on the list enters it too, and typed text always lands in the
composer — see "Click-to-focus and typing-home" below.
`Ctrl+Shift+↑`/`Ctrl+Shift+↓` attach the previous/next conversation directly — the
one-press form of F9-then-arrow-then-Enter — in the list's own ranking, wrapping
at the ends and without moving the caret; with the list closed the catalog is read
fresh first so "next" is what the list would show, never a stale snapshot.
Settings control visibility and left/right placement. Narrow layouts use a drawer
that ends above the input dock and closes after any valid selection.

## Click-to-focus and typing-home

A pointer press on the list — a row, the pin cell, or the panel's chrome — moves
the keyboard to it and wears the panel's focus ground (issue #1357 principles
1–3; decided in `### lopdev — design decision: click-to-focus` on the issue).
The row press still does what it always did — attach/open, with the cursor moving
to the pressed row — and the pin cell still pins without opening. A press on the
header, the `+N more pinned` line, the dead space below the rows or the footer
takes the keyboard without acting on any row. Pinning keeps its cell and `F10`;
there is no plain-key mnemonic for it, so `Space` types like every other
printable key.

The list never keeps the keyboard *for text*: any printable character, or a
bracketed paste, delivered to it is handed to the composer and the composer takes
the keyboard back. This holds however the list's keyboard mode was entered — a
press, `F9` or `/sidebar focus` — so typed input can never go nowhere. The
delivery goes through the composer's own handler (a fresh key/paste is posted to
it after focus moves, the same forwarding the transcript uses), so the draft,
caret, shell mode and paste machinery see exactly what a focused composer would.

Two guards bound the press:

* **A hard claimant keeps its keys.** A live approval, an ask picker, the aside,
  a full-page mode (subagent view, org chart, settings, login prompt) or a pushed
  screen, and a read-only composer, all refuse the press's keyboard move — the
  press may still act on the row, but focus moves nowhere. This is the same
  predicate the composer's own focus routes consult (`_focus_is_claimed()`), held
  from `Screen._forward_event`'s click-to-focus walk before any handler runs.
* **A closed panel never holds the keyboard.** In the narrow drawer placement a
  valid selection closes the drawer and the keyboard returns to the composer.

The one shipped behaviour this decision changes: pressing the *already-attached*
session's row keeps the keyboard on the list rather than returning it to the
composer — every row press behaves alike, and typing-home makes the cost a
single keystroke, because the first one is typed rather than discarded.

Key meanings follow the keyboard, structurally. With the list focused: `enter`
opens the cursor row, `esc` dismisses the panel and returns the keyboard to where
it was before the panel opened (else to the composer), `f9` toggles the list's
keyboard mode without closing it, `f10` pins, `up`/`down`/`PageUp`/`PageDown`/
`Home`/`End` move the list's cursor window, and `ctrl+b` hides the panel. With the
composer focused they keep their composer meanings unchanged.

## Sections, pins and the subagent layer

The list is drawn in up to four sections, in this order: **★ Pinned**, **Active
Sessions**, **Previous Sessions**, **⌥ Subagent Runs**. An empty section costs
no heading. `rank_entries` still decides the ranking — every row's tier, and
pins never reorder it — which is what keeps the terminal and the mobile relay
agreeing on the same active/previous partition. The sections are also the
scroll order: each peer's rows form one contiguous block after **Previous
Sessions** and before **⌥ Subagent Runs**, so the window the list scrolls by
and the frame it paints are one order, and rows enter and leave at the frame's
edges. Page keys move that window a page at a time: the next page starts where
the previous ended (overlap is allowed, a gap never), and a press whose
landing reaches the tail settles bottom-aligned, on the same window the wheel
clamp and `End` land on, rather than on a remainder stub.

The phone draws the same partition and the same ★ Pinned section, and it does
so off the SAME key: the mobile daemon ranks through `session.catalog.entry_for`
(see `_rank_row` in `mobile/daemon.py`) rather than re-deriving an order from
live state, so the two surfaces agree about a row's tier (within a section) and
its place. Two known, deliberate asymmetries:

* the wake band — the phone's rows carry no wake data, so `wake_rank` is a
  constant there (`CatalogEntry.rank` names this);
* the phone-woken window — a `/wake` accepted but not yet discovered is ranked
  as a live `idle` row so it lands in Active at once, where the sidebar has no
  equivalent window and so no equivalent tier.

A pin made on the phone — the list's long-press or the session view's ☆/★ header
control — is written to this same store below, and a pin made here reaches an
open phone list within one mobile-daemon discovery pass (about 2 s), because
that pass fingerprints the pin file on every tick.

The phone pins in two phases, which is worth knowing when comparing the two
surfaces (the measured figures are in `docs/mobile.md`): the press paints the ★
at once from an optimistic mark, and the row is re-sectioned only when a list
frame from the mobile daemon confirms the pin. So a confirmed pin moves the
phone's rows by up to one row height, and a refused one clears the mark and
moves nothing. The phone renders no refusal text in this release — the daemon
still sends its reason in the 409 body, and a follow-up PR is what puts it back
on screen.

### Pins

`F10` pins or unpins the session under the pointer; with no pointer on the list
and the list focused, it acts on the cursor row. Hover wins over the cursor
because the pointer resting on a row says which session is meant, while the
cursor persists invisibly when the list is unfocused. A pinned row leaves
whatever section it ranked into and appears under ★ Pinned wearing `★` in its
own leading cell.

The pin is also a direct pointer action (issue #1357): every row opens with a
two-cell pin cell ahead of the caret. A pinned row shows `★` there on every
frame — the caret keeps its own cell beside it, so a pinned row that is also
the cursor shows both facts — and an unpinned row shows `☆` while the pointer
rests on it. That cell is what a click toggles; clicking anywhere else on the
row keeps the behaviour it always had (open/select), and a click meant for the
star never opens or selects the row. Like every other press on the panel, a
pin-cell press takes the keyboard (see "Click-to-focus and typing-home").

Pinned rows are not lifted into view: a pin is a display lift, and the list
still scrolls over the slot the row would occupy in its own section, so a
pinned session whose slot is below the page is not on screen. When that
happens the `★ Pinned` section ends with a `+N more pinned` line (at the top of
the page when no pinned row is on it), so the heading never claims to be
showing the whole pinned set. That line is chrome — clicking it does nothing.

Pins live in `sidebar-pins.json` in the configuration directory
(`~/.local-operator` unless `LOCAL_OPERATOR_CONFIG_DIR` says otherwise), newest
pin first, capped at 50. They are durable: a pin survives quitting and
relaunching. A pin to a session that no longer exists simply disappears — the
list is pruned against the session store when it is read, so deleting a session
needs to know nothing about pins.

`F10` is a function key rather than `Ctrl+P` because Textual's `App` binds
`Ctrl+P` to its command palette at `priority`, so an app-level binding there
never fires. It also continues the series `F8` (aside) and `F9` (focus
sessions) for app-level gestures that must survive a focused composer.

### The ⌥ subagent layer

Subagent runs are hidden from the list by default. `Ctrl+A` toggles them on for
the current session, and `Ctrl+O` jumps the cursor to the first one. Both are
**sidebar-scoped**: they fire only while the list owns the focus chain (F9
mode), because both are also composer keys — `Ctrl+A` is the caret's line-start
and `Ctrl+O` expands a collapsed paste.

`tui.sidebar_show_subagents` sets the startup default. `Ctrl+A` never writes it:
the flip applies to the session you pressed it in, and a write would fan out
through the config watcher to every running `lop` process. A `/settings` change
does apply live to an already-painted sidebar.

The footer's `⌥N` chip is also a control, not only a marker: a pointer press on
it flips the layer for the current session with no `F9` at all — the route for
a user who has not learned the chord, and the one that needs no keyboard mode.
It runs the same flip `Ctrl+A` runs and holds the same discipline: per-session,
and **never a write to `tui.sidebar_show_subagents`**, for the same fan-out
reason. Hovering the chip underlines it; the count itself is never replaced or
moved. The chip is the ladder's last rung, so its column moves when the footer's
shape changes — a flip that pages the list can shift it a few cells, and a press
aimed where it used to sit is inert; the underline returns on the next hover,
wherever the chip now sits. The chord and the pointer are two ways to one flip,
never two flips.

Turning the layer on costs screen space before it shows a single row. Each
section spends a heading plus the blank line beneath it, and every heading after
the first takes a separating blank as well, so going from two sections to four
takes section chrome from 5 lines to 11. On a 30-row terminal that is enough to
turn a list that fitted into a paged one: the footer then gains a position
counter (`1–16/21`), and `ctrl+b hide` becomes the first hint to yield when the
full form no longer fits the width. This is accepted and expected, not a defect
— the blank above a heading is what keeps it from sitting flush against the
previous group's last row.

When the list pages, the footer is a fitted ladder, not a fixed string. It tries
the full form (`{position} · {lead} · ctrl+b hide · {chip}`) and drops whichever
fact does not fit, least-load-bearing first: `ctrl+b hide` yields before the
position counter. The lead key is focus-aware — `esc return` when the list holds
the keyboard, `f9 focus` when it does not — and never yields, because a page
counter may never be the reason the only named exit disappears (main's D4 rule).
The `⌥N` chip never yields either: the position is recoverable by scrolling (the
cursor row is painted), but the hidden population is recoverable from nowhere
else on the frame. `/help` still lists `ctrl+b`.

When the list holds the keyboard the ladder re-ranks: `f10 pin` rides it from
27 cells of list width up (the pin cell's `☆` is invisible until hovered, so
the footer is its one standing teacher; between 17 and 26 the chip outranks it,
and a 30-column terminal lands there), and `ctrl+a ⌥` renders exactly where the
position form does not fit — on an unpaged list at 38 cells or more, and on a
paged one in the band from 38 up to one cell under `{position} · esc return ·
f10 pin · ⌥N`. That band is empty for a position string of 8 cells or fewer
(`1–18/21`, whose form is 38 and wins at 38) and opens for longer ones: a
152-entry list on a deep page (`128–152/152`, its form 41) shows
`esc return · f10 pin · ctrl+a ⌥ · ⌥N` at content 38–40 and the position form
returns from 41. Every width here moves with the position string and the chip,
so the code's fit test is the authority, not a quoted figure (design round 1
D1, corrected by review round 2). The position yields BEFORE the pin —
recoverable by scrolling, while `f10 pin` is not taught anywhere else on the
frame — so the focused rungs are `esc return · f10 pin [· ctrl+a ⌥]{ · ⌥N}`
above the floor fallbacks. `ctrl+o` has no spelling that fits a real content
width beside the chip and the pin, so it stays a documented chord rather than
an invisible candidate.

The layer is capped at 40 rows and does not page. A sub row is labelled by what
it was delegated to do (`label · role`, degrading to whichever half exists),
carries `⌥` where a state glyph would be, and never shows live state or a
completion mark — the hidden population is not polled for one.

A pinned subagent row stays in ★ Pinned even with the layer off, since pinning
is an explicit request for that row.

### When the completion mark can appear

A row's completion mark, and its place in **Active Sessions**, both read the
durable attention row for that conversation, so a completion that could not be
published yet is a mark that cannot be drawn yet. `attention.db` adopts WAL on
its write path, so on a converted store readers never block a writer's `COMMIT`
at all; what remains, and stays rare, is writer-against-writer contention, under
which a publish gives up after its bounded budget (~11 s) and is DEFERRED rather
than lost — the outcome is already durable
in the transcript journal. The owning session then republishes it in-process on a
bounded ladder (four rungs, ~86 s), and a viewer already polling fires the parked
rung early on its next tick — the tick itself never writes — so the mark normally
appears within about a second of the store clearing, at the ladder's own attempt
count rather than one store write per tick. Nothing here needs a restart or a
`/resume`, and no test loop has to watch for it; if the whole ladder is exhausted
the mark waits for that session's next boot, which the log says once. A newer
completion is never outranked by an older republished one: this session
serialises the journal-read-then-insert against its own turn-end publication.
Subagent rows are never polled for a completion mark at all (see above).

### The footer count

When hidden subagent runs exist, the footer gains a `· ⌥N` chip on its existing
line — never a second line, which would cost a session row at every terminal
height. The count is capped at `1k+` above 999 (`⌥999` still renders exactly) so
the footer's width stays predictable, and it is refreshed every 15 polls (about
30 s) plus whenever the sidebar is opened, rather than on every poll: reading it
is a second full scan of the session store.

It counts the store's hidden (subagent) population rather than "what the layer
currently hides", so the chip remains on the line — same cells, same count —
while the layer is shown: the control that raised the rows is the one that puts
them away. It is also the toggle's press target (see §The ⌥ subagent layer):
those cells, and only those, flip the layer on a press; the rest of the footer
stays inert to the pointer, as it always was.

### One behaviour change to know about

`Ctrl+Shift+↑`/`Ctrl+Shift+↓` traverse the list's own ranking, so **with the ⌥
layer on, subagent rows join that traversal**. This follows the shortcut's
existing contract — what the user sees is what they traverse — but it is the
change most likely to surprise. With the layer off they are not in the list and
cannot be reached.

## Ownership and readiness

`SessionInteraction` owns each source's turns, loop, shell, compaction, draft,
approval policy, gate input and accounting. A prepared/retained view owns widgets,
not work. Switching does not answer a gate, cancel a turn, or redirect its eventual
result.

That guarantee belongs to this navigation, **not** to `/resume`, which ends the
turn it leaves and so denies its queued approvals. Anything that switches on the
user's behalf therefore goes through `_select_sidebar_session` — the one entry
point both the keystroke and a notification click use — rather than issuing a
`/resume` that merely resembles it. A click routed the second way answered a
parked approval "no" in the session being left, with no receipt.

Preparation never acknowledges completion attention. The current binding
changes only after canonical attachment and prepared replay; navigation stays
pending until a real displayed frame also has the correct scroll/gate surface.

`SessionNavigation` is latest-wins and bounds preparation. Retained presentations
are separately budgeted from data-only source contexts and drafts. Private draft
spill files are temporary, not session journals; secret gate text never enters
those files. Focus restoration uses weak references so a dismissed login widget
and its pasted secret cannot be retained by sidebar bookkeeping.

## Opt-in durable display history

An upgraded runtime advertises `display-history-window-v1`. Sidebar connections ask
for `AttachedSession.connect(..., display_window=True)`; ordinary connections retain
full-history initialization. The window rides the existing atomic frontend sync:
runtime snapshot, durable cursor, window selection and subscription share one
no-yield authoritative-loop boundary.

`Transcript.build_llm_history(through_id=...)` selects the journal cut before
interpreting compaction and prunes. The window reuses that canonical replay,
including custom roles, attachments and preserved user turns. Compaction markers
reuse their durable entry ID. Tool call/result groups stay together, including
results separated by custom messages. There is no second transcript index or
projection of live model context.

Pages are capped at 120 messages and 512 KiB, including actual JSON escaping, with
space reserved for metadata. The entire sync still respects the transport's
1 MiB frame ceiling. Required prose or a tool group that cannot fit is **not
truncated**: an explicit `full_required` result selects the existing off-thread
local full replay at the captured cut. That fallback does not promise low latency.

Signed tokens bind conversation, runtime epoch, replay generation, durable cut and
message position; clients cannot provide filesystem paths or byte offsets.
Appends preserve a captured cut. Compaction, pruning and file folding invalidate
its generation. Paging returns a typed reset rather than mixing generations;
the viewer obtains a fresh canonical sync without restarting or prompting the
runtime. A missing saved anchor resets to the new canonical tail, never a different
row presented as the old anchor.

`display_history_window()` is explicitly partial. `history_message_count`,
`history_theme_turn_count` and `history_opener_text` carry whole-cut metadata.
`history_page()` and `load_older_display_page()` read older pages as needed.
`ensure_display_anchor()` validates a saved message/tool anchor and hydrates a
contiguous interval through it for the existing two-direction renderer.
`materialize_history()` is the explicit full-trajectory API used by background
naming. Calling `history()` on an unhydrated window raises, rather than silently
returning a partial conversation. Runtime/model/idempotency full-history callers
are unchanged.

Loaded IDs, painted IDs and pre-cut live-seed IDs are distinct. Canonical live
messages and tool outcomes are retained separately from the durable window so a
source that leaves the screen during a gate does not lose its initiating prompt
or the later result. Reconnect pages back through the previous durable frontier;
a replay-changing generation produces an explicit presentation reset.

## Parked-source event mute

A prewarmed source stays subscribed so its conversation is warm, and everything
it receives while nobody is looking is discarded app-side. The DELIVERY of those
frames is not free — the owner serialises each frame once per connection, and the
viewer decodes and deserialises it before the drop, with `message_update` frames
carrying the accumulated message — so a parked viewer asks its owner to stop
sending them at all.

An upgraded runtime advertises `event-mute-v1`. The viewer sends **`event_mute`**
when its controller parks and **`event_unmute`** on reveal; the owner then skips
exactly `EVENT_MUTE_DROP_TYPES` (`message_update`, `tool_execution_update`,
`subagent_progress`) for that one connection, and nothing else: turn boundaries,
gates, notices, tool start/end, compaction and model changes keep flowing, since
a parked source's state must still be right when it is revealed. The mute is per
connection, idempotent, and re-asserted on reconnect (a fresh socket starts
unmuted); a viewer whose owner does not advertise the capability sends nothing
and reads the record's capability list as the pre-dial gate instead. Reveal is
unchanged: the presentation rebuilds from history plus the canonical live seed
and the remaining deltas resume, with the settled row's authoritative text
healing anything that streamed during the parked window.

## Local workflows and compatibility

The invoking TUI schedules `/loop`, while each iteration prompts its captured
runtime. Another conversation's stop cannot cancel it. Bang commands execute in the
invoking terminal and forward their receipt to the captured runtime. Stable IDs
from the submitted tool-call ID make duplicate receipt delivery idempotent. A
busy runtime queues the receipt behind its turn; acceptance is not a claim that
queued persistence is already durable. Offscreen completion retains its result.

Already-running older owners remain visible and selectable through authenticated
full-history attachment. They are never restarted, stopped, prompted, or marked
read to accelerate selection. An immediate first click before legacy preparation
finishes can still require a full journal parse. Older owners without the shell
receipt operation report a save failure explicitly; the terminal cannot retrofit
that capability into another running process. No universal sub-100 ms guarantee
is implied, particularly for large required content or unprepared legacy owners.
