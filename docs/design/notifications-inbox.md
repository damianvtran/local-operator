# Notifications inbox

Status: core data plane shipped (lane L1). Banner ladder (L1b), the TUI page
(L2), the relay (L3), the desktop page (L4) and native (L5/L6) follow.
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
| D2 | `/notifications` is the new page; the old completion-receipt listing is `/unread` (`/unread read` clears), behaviour and receipt copy unchanged. Until the TUI page lands (L2) the terminal routes `/notifications` to the same receipt handler. | `slash_commands.py`, `tui/app.py` |
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

Retention prunes on write, at most hourly: dismissed after 14 d, read after
30 d, unread never by age; a 10,000-row cap. Deleting a conversation deletes its
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

### 3.2 Who gets it (createIf rung 3)

Only a top-level conversation the operator's own listings show. The session
binds `ToolContext.notify_inbox` to `None` for a delegated child (it reports
through `hub`; its parent decides), a session `resume.is_user_session` hides, a
one-shot host, an output-contract session, and under `LOP_NOTIFY=0`.

### 3.3 Footprint: deferred

Published, the schema would cost ~584 billed tokens against a start-context
ceiling with ~76 to spare. Deferred, the measured top-arm delta is +196 chars
≈ +70.5 billed (the §4 clause, plus the purpose phrase on the schema-on-demand
line); the child arm does not move. The adoption probe guards the choice: 20
scripted wake turns per arm on a live model; under 80% correct calls, un-defer
and raise the ceiling in the same PR.

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
destination or feed branch.

### 5.2 Desktop routes

Five cold routes: list, count, detail, the token-bound PATCH (`read`, `unread`,
`dismiss`, `undismiss` over `ids[{id, revision}]`, or `session_id`+`through_seq`,
or `all`+`through_seq`), and the `seen` evidence door. A write naming a revision
the store has moved past is `skipped: stale_revision`, never applied.

### 5.3 Feed frame

The `notifications` invalidation frame on `GET /v1/desktop/events`,
`{revision, unread_count, waiting_count}`, at most once per tick, when either
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
the desktop navigates instead of sending. The receipt listing is `/unread` (D2);
until the TUI page lands (L2) the terminal routes `/notifications` to the same
handler. The sidebar id is appended at the end of the wire vocabulary (D10).

## Not in this lane

- Banners for `high` items (L1b): the claim primitive (`claim_delivery`) and the
  `delivered_rev` watermark exist; nothing raises yet, so the completion-banner
  suppression for a run that raised an item banner lands with it.
- The TUI page, its sidebar note and the `activity` control op that would let an
  attached terminal satisfy rung 0 (L2). Until then a terminal never does.
- Phone push: blocked on the cloud transport (S7). Off-desk reach in v1 is the
  relay/native inbox and its badge.
