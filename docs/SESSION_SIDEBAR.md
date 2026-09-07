# In-app session sidebar

The sidebar is an opt-in terminal view over existing session owners, not another
owner or a new scheduler. `Ctrl+B` and `/sidebar` toggle visibility without moving
the editor caret. `F9` and `/sidebar focus` enter the list; F9 returns focus while
leaving it open, and Escape dismisses it and returns to the last usable surface.
`Ctrl+Shift+↑`/`Ctrl+Shift+↓` attach the previous/next conversation directly — the
one-press form of F9-then-arrow-then-Enter — in the list's own ranking, wrapping
at the ends and without moving the caret; with the list closed the catalog is read
fresh first so "next" is what the list would show, never a stale snapshot.
Settings control visibility and left/right placement. Narrow layouts use a drawer
that ends above the input dock and closes after any valid selection.

## Ownership and readiness

`SessionInteraction` owns each source's turns, loop, shell, compaction, draft,
approval policy, gate input and accounting. A prepared/retained view owns widgets,
not work. Switching does not answer a gate, cancel a turn, or redirect its eventual
result. Preparation never acknowledges completion attention. The current binding
changes only after canonical attachment and prepared replay; navigation stays
pending until a real displayed frame also has the correct scroll/gate surface.

`SessionNavigation` is latest-wins and bounds preparation. Retained presentations
are separately budgeted from data-only source contexts and drafts. Private draft
spill files are temporary, not session journals; secret gate text never enters
those files. Focus restoration uses weak references so a dismissed login widget
and its pasted secret cannot be retained by sidebar bookkeeping.

## Opt-in durable display history

An upgraded owner advertises `display-history-window-v1`. Sidebar connections ask
for `RemoteSession.connect(..., display_window=True)`; ordinary connections retain
full-history initialization. The window rides the existing atomic frontend sync:
owner snapshot, durable cursor, window selection and subscription share one
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

Signed tokens bind conversation, owner epoch, replay generation, durable cut and
message position; clients cannot provide filesystem paths or byte offsets.
Appends preserve a captured cut. Compaction, pruning and file folding invalidate
its generation. Paging returns a typed reset rather than mixing generations;
the viewer obtains a fresh canonical sync without restarting or prompting the
owner. A missing saved anchor resets to the new canonical tail, never a different
row presented as the old anchor.

`display_history_window()` is explicitly partial. `history_message_count`,
`history_theme_turn_count` and `history_opener_text` carry whole-cut metadata.
`history_page()` and `load_older_display_page()` read older pages as needed.
`ensure_display_anchor()` validates a saved message/tool anchor and hydrates a
contiguous interval through it for the existing two-direction renderer.
`materialize_history()` is the explicit full-trajectory API used by background
naming. Calling `history()` on an unhydrated window raises, rather than silently
returning a partial conversation. Owner/model/idempotency full-history callers
are unchanged.

Loaded IDs, painted IDs and pre-cut live-seed IDs are distinct. Canonical live
messages and tool outcomes are retained separately from the durable window so a
source that leaves the screen during a gate does not lose its initiating prompt
or the later result. Reconnect pages back through the previous durable frontier;
a replay-changing generation produces an explicit presentation reset.

## Local workflows and compatibility

The invoking TUI schedules `/loop`, while each iteration prompts its captured
owner. Another conversation's stop cannot cancel it. Bang commands execute in the
invoking terminal and forward their receipt to the captured owner. Stable IDs
from the submitted tool-call ID make duplicate receipt delivery idempotent. A
busy owner queues the receipt behind its turn; acceptance is not a claim that
queued persistence is already durable. Offscreen completion retains its result.

Already-running older owners remain visible and selectable through authenticated
full-history attachment. They are never restarted, stopped, prompted, or marked
read to accelerate selection. An immediate first click before legacy preparation
finishes can still require a full journal parse. Older owners without the shell
receipt operation report a save failure explicitly; the terminal cannot retrofit
that capability into another running process. No universal sub-100 ms guarantee
is implied, particularly for large required content or unprepared legacy owners.
