# Notifications inbox

Status: core data plane (lane L1), banner ladder and relay wire (L1b) and the
TUI page (L2) are implemented on stacked branches; the relay screens (L3), the
desktop page (L4) and native (L5/L6) follow.
Companions: [`../ATTENTION.md`](../ATTENTION.md) "Inbox items" (rung 0 and
in-view auto-read), [`../DESKTOP_API.md`](../DESKTOP_API.md) "The Notifications
inbox" (routes and the feed frame). Section numbers match the architect's
contract, which code comments cite.

## Why

A push or banner is gone once seen. The operator asked for one place that holds
everything pending — the questions agents are blocked on, and the findings
agents decide the operator must see (an outside contributor's PR waiting for
review, a fraud attempt against an agent server, a campaign's numbers) — with
mark read / mark all read, dismiss (one, several, all from a session), and a
detail that shows the item's context, its conversation, related projects and
that conversation's asks. Watchers on 5-minute wakes are the volume case: an
item must update in place rather than pile up.

## Decisions this lane implements

| # | Decision | Where |
|---|---|---|
| D1 | The reel shows OPEN asks only ("Waiting on you"); settled asks appear only in an item's detail. | `notifications/items.py` `open_asks`, `detail` |
| D2 | `/notifications` is the new page; the old completion-receipt listing is `/unread` (`/unread read` clears), behaviour unchanged. Its receipt lines now name `/unread`: the old last line said `/notifications read`, which now clears the inbox instead. | `slash_commands.py`, `tui/app.py` |
| D3 (superseded) | Operator presence addendum: creation is always unread; only item-specific evidence acknowledges it. | `Session._notify_inbox`, `NotificationStore.notify`, `seen` |
| D10 | Sidebar id `notifications` appended to the wire vocabulary; default layout puts it under Aida and folds Schedules into More; `read_layout` drops ids it does not know instead of discarding the whole layout. | `server/sidebar_layout.py` |
| D11 | Aida's seed: raise a `notify` for findings the operator must act on. | `agent_seeds/aida.md` |

## 1. Store: `<config_dir>/notifications.db`

A separate SQLite file, NOT tables in `attention.db` (the most contended store on
the machine) and NOT the asks log pattern (an item has no model-delivery leg,
and every inbox operation is cross-session: a machine-wide unread count, paging,
"mark all read through seq N", an atomic per-revision banner claim). Same
connection discipline as `AttentionStore`: WAL, `BEGIN IMMEDIATE`, the shared
contention classifier (`session/store_failures.py`), read-only readers.
`notifications/store.py` is stdlib-only because the feed, the relay and the TUI
poll all read it.

Tables: `items` (one row per item; `seq` is machine-wide and re-stamped on every
write; `revision` bumps on a content change and is THE token for read/ack;
unread ⇔ `read_rev < revision`; `delivered_rev` is the banner watermark, separate
from the read mark — notifying is not reading), `item_history` (earlier versions,
capped at 20), `ref_state` (reel-side read marks for asks this store does not
own), `meta`.

### 1.3 Collapse and limits

| Call | Store effect | Tool result |
|---|---|---|
| `key` empty, or new for this session | insert | `Notified (id n_…).` |
| `key` exists, not dismissed, same content | no write | `Unchanged — n_… already says this.` |
| `key` exists, not dismissed, changed | update in place; `revision+1`, previous version to history, unread again | `Updated n_… (Nth update).` |
| `key` exists and was dismissed | absorbed silently: content updated, stays dismissed, no banner | tells the model the operator dismissed it |
| …and the new priority is ABOVE the stored one | un-dismissed (escalation) | `Raised n_… again at <priority> priority` |

The dismissed row is what keeps a 5-minute wake from flooding the reel. At most
30 NEW rows per session per rolling hour (keyed updates do not count); over the
cap the write is refused and the refusal names the remedy (pass a `key`).

### 1.4 Retention and privacy

Retention prunes on write, at most hourly: dismissed after 14 d since its
last activity (`max(dismissed_at, updated_at)`, so a still-ticking watcher keeps
its tombstone), read after 30 d, unread never by age; a 10,000-row cap. An
unreadable store never invalidates the desktop feed: its state stays unknown,
the previous state is retained and other frame families continue. A corrupt
SQLite file is quarantined by the next write and recreated (CD1). Deleting a conversation deletes its
items, their history and its asks' read marks (`session/cleanup.py`). Title,
body and link labels are scrubbed through the session's redaction before they
reach the store: the loop redacts the public tool-start event, not the raw
arguments.

## 2. What the reel shows

Two sources, never copied: `notify` rows, and open asks read at request time
from the ask queue's derived index. One normalized item shape for both
(`notifications/items.py`). An open ask always reports `high`. The reel never
settles one: dismissing it — by id, by session or all — answers
`skipped: live_ask` and leaves the ask log byte-identical. Settling goes through
the existing reasoned "Dismiss ask…" doors, which tell the agent. Related
projects come from the session's project links at read time.

## 3. The `notify` tool

### 3.1 Schema

`title` (≤120), `body` (markdown, ≤4000; metrics go in a table here),
`priority` (`normal` default, `high`, `low`), `key` (≤120), `links` (≤5, http(s)
only). `approval_tier="read"` (the operator's own inbox, the `no_reply`
reasoning), shared concurrency.

**Title quality — the contract.** The list shows the title ALONE, so a title
that identifies no event is a row the operator must read and dismiss to learn
nothing. `NotifyParams` therefore refuses a degenerate title as
`invalid arguments` (`local_operator/notifications/title_quality.py`):

- **Information floor** — fewer than three alphanumerics after dropping
everything else. Three admits what people really write (`PR #7`, `CI green`),
rejects `x`/`--`/`ok`, and is script-independent: `修复完成` counts four,
`🚀` counts zero.
- **Placeholder vocabulary** — the WHOLE title, casefolded and stripped of
non-alphanumerics, against a fixed set (`x`, `noop`/`no-op`, `n/a`, `test`,
`todo`, `tbd`, `update`, `unused`, `placeholder`, `wip`, `done`, `ok`, …).
Whole-equality only: `Update: 3 PRs merged` and `Tests failing on shard 3`
pass, because they are statements that happen to open with a placeholder word.
- **Nothing at all** — a whitespace/punctuation-only title is refused HERE. It
used to reach the store, whose bare `ValueError` surfaced as "the inbox could
not be written": a dead end naming neither the cause nor a way out.

The boundary is deliberately one-sided: **refuse provably content-free titles,
never judge a short-but-real one.** Over-refusing a legitimate notification is
the worse failure, because the model's cheapest way to obey a guard is to post
nothing. The body is never consulted — it is not on the list row, so a
substantial body behind `x` still renders as an opaque row.

The refusal names the silent alternative in its own words: a turn that produced
nothing the operator needs ends with `no_reply` and posts no notification.
`no_reply` itself is untouched, so a quiet turn stays zero-friction. The guard
lives on the tool params, NOT in `NotificationStore.notify`: server-side writers
(an approval gate titled with the tool name, tests) use short titles on purpose
and must keep working. The 25 legacy placeholder rows were retired by hand;
this guard is what stops new ones.

### 3.2 Who gets it (createIf rung 3)

Only a top-level conversation the operator's own listings show. The session
binds `ToolContext.notify_inbox` to `None` for a delegated child (it reports
through `hub`; its parent decides), a session `resume.is_user_session` hides, a
one-shot host, an output-contract session, and under `LOP_NOTIFY=0`.

### 3.3 Footprint: published after a failed deferred pilot

The completed live deferred pilot verified actual tool inventory, loaded nudge
and effective deferral, then made 20 wrong `todo` calls and falsely claimed it
had notified; no durable row existed. Its activation plus wake used 22 provider
requests. This is not an adoption percentage. The earlier purported published
control was INVALID: top-level `tools` config was ignored (settings belong
under `values`). It provides no published/deferred comparison.

The manager rejected deferred discovery. Only notify is now published;
top-level createIf and the kill switch remain. Same-tree measurement:
93,021 → 94,615 chars, +1,594 (+1,624 schema, −30 inventory) ≈ +573 billed.
The top ratchet rises 33,466 → 34,039, preserving five tokens of headroom;
the child arm is unchanged. No user tool settings or role pins change.

The obsolete 20×2 deferred/published comparison is replaced by a bounded
published scenario matrix: three meaningful outcomes and three routine/no-action
cases, durable rows/tool receipts required. Run one positive and one negative
first, stop on failure, and have independent QA repeat the discriminating pair.
No fabricated statistical claim or positive-only flood-prevention claim.

### 3.4 Quiet ends

A wake or monitor turn that called `notify` may end with `no_reply` — including
a `notify:true` delivery: the operator asked to be told, and the item told them
(`Session._run_notified`). A person waiting on words still gets words.

## 4. The turn-end suggestion

One `system.md` sentence gated on `has_notify` (tool membership), so sessions
without the tool — children included — render nothing new: "Before ending a
turn, `notify` anything the operator should see on return and might miss in
chat — results, metrics, notable events; never routine progress." Rejected: an
envelope clause on every wake delivery (paid per delivery and persisted in
history), a yield-time injection (an extra model round trip per turn).

## 5. APIs

Full shapes: `../DESKTOP_API.md` "The Notifications inbox".

### 5.1 Capability

`features.notifications: 1`. Absent it, a client offers no page, row, slash
destination or feed branch. `features.notifications_seen: 1` is the native
receipt door (D11): the phone turns its in-view receipt feed on only when it is
present, because a core that predates it refuses every receipt.

### 5.2 Desktop routes

Five cold routes: list, count, detail, the token-bound PATCH (`read`, `unread`,
`dismiss`, `undismiss` over `ids[{id, revision}]`, or `session_id`+`through_seq`,
or `all`+`through_seq`, with exact `observed` tokens for either bulk target), and the `seen` evidence door. A write naming a revision
the store has moved past is `skipped: stale_revision`, never applied.

`seen` accepts the surfaces `desktop`, `tui` and `native` (`store.SEEN_SURFACES`),
each through its own door: the desktop route takes pointer/key/wheel, the TUI's
key/mouse-down, the relay's touch/press/scroll (D11, and the relay answers any
other surface with `relay_surface_mismatch`). A focus gain is never an input edge
(D12). Every accepted row records `read_via: in_view` plus the `read_surface`
that produced it.

### 5.3 Feed frame

The `notifications` invalidation frame on `GET /v1/desktop/events`,
`{revision, unread_count, waiting_count, waiting_unread_count}`, at most once per tick, when either
doorbell moved: the store's stat fingerprint (`notifications.db` and its two
sidecars, on the 100 ms tick) or the asks index directory (the 1 s probe — the
badge counts open asks and nothing else watches that directory). Invalidation
only: the client refetches.

## 6. Presence

### 6.2 Rung 0 at write time

See `../ATTENTION.md` "Inbox items". The operator's presence addendum
supersedes the initial creation-time predicate: matching session and fresh
activity do NOT prove a newly written callout rendered. Scrolled-up, child,
covered and not-yet-rendered cases all stay unread, even with fresh OS input.
Persist unread → render the anchor → per-item revision-bound receipt with
fresh post-update input. L1b owns bounded banner grace and re-checking the
read watermark before delivery. Presence remains an advisory signal only.

## 7. TUI and 8. Desktop (core side)

The `/notifications` slash row carries `desktop_destination="notifications"`, so
the desktop navigates instead of sending. The receipt listing is `/unread` (D2).
The sidebar id is appended at the end of the wire vocabulary (D10).

### 7.1 The TUI page (lane L2)

`/notifications` opens a full-page mode with the `/library` contract (transcript
hidden, dock greyed, the page owns the keyboard until `esc`). `/notifications
read` opens it with the mark-all confirm armed, so the typed form and `A` share
one guard. Code: `tui/widgets/notifications_view.py` (state and keys),
`tui/notifications_render.py` (cells), `tui/notifications_model.py` (every rule
about what a gesture touches; no Textual), `tui/notifications_attention.py`
(rung 3 and in-view reads).

- **Layout.** Under 120 columns the detail replaces the list (`↵` in, `esc` out,
  the cursor kept, the detail pinned to the row it opened so a row that leaves
  the Unread filter does not swap the detail under the reader). At 120+ the page
  splits 55/45 with an `edge` rule and the detail follows the cursor after
  150 ms. The wide preview never marks anything read; `↵` marks THAT revision.
- **Keys** are design §8.4's table. `E` dismisses every update in the row's
  conversation and `A` marks the whole Unread scope; both confirm on one `y/n`
  line. `z` undoes the last write for 6 s by applying the inverse op to exactly
  the ids it changed.
- **Bulk writes.** `enumerate_scope` walks every page under the FIRST page's
  `high_water` as `through_seq`, keeps the page filters (a filtered gesture
  never widens), and `write` sends the observed tokens in batches of at most
  500. A row written after the confirm was armed stays unread.
- **Asks.** `e` on an open ask says it stays until answered; `a`/`↵` closes the
  page and opens the existing fleet ask list on that ask.
- **Sidebar note.** `notifications N` (`unread_count + waiting_unread_count`,
  the same figure every surface badges) sits on the footer after `asks: N`,
  hidden at 0, and yields before every other tail fact. It is fed from the
  existing slow-cadence poll and repainted at once after a page write.
- **Rung 3.** The running TUI's 1 s attention poll claims unclaimed, unread,
  undismissed `high` items (`claim_delivery(..., "tui")`), raises them through
  the existing `detached_notify` (3 per tick, the rest as one digest), and
  releases a claim whose spawn reported nothing. It stands down while a desktop
  advertises `update`. The runtime's rung 4 already stands down for a running
  TUI (`serving._tui_viewer_running`).
- **In-view reads.** See `../ATTENTION.md` "Inbox items".

## Not in this lane

- Banners for `high` items (L1b) are specified in `../../DESKTOP_API.md` and
  implemented on this branch: the claim is copy-once per revision machine-wide,
  a receipt before the claim cancels the banner it makes redundant, a claim whose
  delivery produced nothing is released, and the desktop's `update` branch is the
  consumer. The TUI announcer leg and the `error`-exempt completion suppression
  are core code covered by `tests/unit/notifications/test_delivery.py`.
- Historical note (superseded): the claim primitive (`claim_delivery`) and the
  `delivered_rev` watermark exist; nothing raises yet, so the completion-banner
  suppression for a run that raised an item banner lands with it.
- The `activity` control op that would let an attached terminal satisfy rung 0
  is not built; under the presence addendum rung 0 no longer creates read, so
  nothing depends on it.
- Phone push: blocked on the cloud transport (S7). Off-desk reach in v1 is the
  relay/native inbox and its badge.

## Round-one observation refinements

Bulk actions echo list `observed` id/revision tokens (max 500 per PATCH).
An explicit All action enumerates the full intended filtered scope, freezing
first-page `high_water` as GET `through_seq`, unioning tokens across pages,
then batching writes with progress/count feedback. It never means only the
first page, and never discovers later asks at write time. `waiting_count`
retains all pending asks; `waiting_unread_count` drives the ask half of the
unread badge. See DESKTOP_API for exact request shapes.

Release sequencing: L1 is not independently exposed before shipped read
surfaces. D-C1 (interim TUI help versus receipt handler) is resolved in L2: the
help line describes the page that ships.
