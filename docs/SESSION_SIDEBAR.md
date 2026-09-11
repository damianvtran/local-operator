# In-app session sidebar

The sidebar is an opt-in terminal view over existing session runtimes, not another
runtime or a new scheduler. `Ctrl+B` and `/sidebar` toggle visibility without moving
the editor caret. `F9` and `/sidebar focus` enter the list; F9 returns focus while
leaving it open, and Escape dismisses it and returns to the last usable surface.
`Ctrl+Shift+↑`/`Ctrl+Shift+↓` attach the previous/next conversation directly — the
one-press form of F9-then-arrow-then-Enter — in the list's own ranking, wrapping
at the ends and without moving the caret; with the list closed the catalog is read
fresh first so "next" is what the list would show, never a stale snapshot.
Settings control visibility and left/right placement. Narrow layouts use a drawer
that ends above the input dock and closes after any valid selection.

## Sections, pins and the subagent layer

The list is drawn in up to four sections, in this order: **★ Pinned**, **Active
Sessions**, **Previous Sessions**, **⌥ Subagent Runs**. An empty section costs
no heading. Sectioning is display-only — `rank_entries` still decides the
ranking, and pins never reorder it — which is what keeps the terminal and the
mobile relay agreeing on the same active/previous partition.

### Pins

`F10` pins or unpins the session under the pointer; with no pointer on the list
and the list focused, it acts on the cursor row. Hover wins over the cursor
because the pointer resting on a row says which session is meant, while the
cursor persists invisibly when the list is unfocused. A pinned row leaves
whatever section it ranked into and appears under ★ Pinned with a `★` in place
of its state mark.

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

The layer is capped at 40 rows and does not page. A sub row is labelled by what
it was delegated to do (`label · role`, degrading to whichever half exists),
carries `⌥` where a state glyph would be, and never shows live state or a
completion mark — the hidden population is not polled for one.

A pinned subagent row stays in ★ Pinned even with the layer off, since pinning
is an explicit request for that row.

### The footer count

When hidden subagent runs exist, the footer gains a `· ⌥N` chip on its existing
line — never a second line, which would cost a session row at every terminal
height. The count is capped at `999+` so the footer's width stays predictable,
and it is refreshed every 15 polls (about 30 s) plus whenever the sidebar is
opened, rather than on every poll: reading it is a second full scan of the
session store.

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
