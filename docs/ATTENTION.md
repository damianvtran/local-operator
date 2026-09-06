# Viewed completion receipts

A mounted frontend, open terminal, watch lease or SSE subscription is not a
read. Completion state lives in the private `attention.db` under the configured
Local Operator config root. Pending questions and approvals have their own
lifecycle and are never answered or removed by a completion receipt.

## Identity and durability

Canonical identities distinguish `session/<durable-id>` from legacy persistent
`agent/<agent-id>` conversations. Selecting an agent profile does not alias an
ordinary session to that profile's persistent conversation. Followers acknowledge
through their authenticated owner connection, not through their own PID.

The logical run journals its token before execution. A settled eligible outcome
is journaled after durable message persistence and imported idempotently into
SQLite. Error and interrupted outcomes can have an explicit outcome marker even
when no assistant message exists. Resuming an unfinished journaled run records an
interruption rather than treating its output as unknown pre-upgrade history.
Copied fork journals cannot reuse another conversation's token.

A receipt advances through the supplied token's sequence using a monotonic
watermark. Delayed or duplicate acknowledgement of A cannot acknowledge newer B.
Owner epoch, transcript mtime, heartbeat time and stream sequence are not
completion clocks. The SQLite engine serializes writers across processes. All
schema objects initialize in one transaction; readers of a positively identified
empty, not-yet-initialized database see no published completion. Corrupt bytes or
missing tables in an established schema remain errors, never a false read state.

## Read APIs and transports

`AttentionStore.state_many(conversations)` returns a consistent map keyed by
canonical identity on one read-only connection. Run it in a worker and merge the
returned map on the UI loop; do not open one connection per list row. The state
contains:

- `conversation_id`
- `completion_token`, `anchor_id`, `kind` (`complete`, `error`, `interrupted`)
- `unseen`
- `revision: [completion_sequence, acknowledged_sequence]`

The canonical frontend and mobile projection carry additive `attention` state.
Runtime capability `completion-ack-v1` enables `acknowledge_attention` with
`completion_token`. Mobile `POST /api/sessions/{id}/seen` takes the same token in
a JSON object: missing legacy bodies return422, unknown/foreign tokens409,
unknown sessions404, and unauthenticated callers401. Reads and subscriptions do
not mutate the receipt store.

The relay maintains its existing projection ordering while alive. A new,
authenticated and source-fenced SSE connection starts with an authoritative
snapshot; its first projection may have a lower counter after daemon restart.
Retired sources cannot publish callbacks or close the current connection.

## What a frontend can acknowledge

The selected result must actually be rendered, uncovered, and visible in a
focused foreground interface. Old scrollback, loading, a covering screen or a
child-agent page does not qualify. Terminal startup's default Textual focus
value is not positive evidence. On macOS cmux, the bounded off-loop probe also
checks the frontmost application, the same socket's kernel peer PID, and the
key visible window's selected workspace and terminal surface.

Mobile transcript rows carry `text_complete`. `final` means streaming settled;
it does not prove transport retained the final row's ending. Both runtime and
relay serialization preserve `text_complete=false` when clipping a row. Missing
metadata from an older owner is unknown, not permission to acknowledge. Only a
rendered anchor with both flags true qualifies. Unrelated degraded rows do not
prevent acknowledgement of a complete result. If a capped result cannot be
hydrated in full on the phone, it remains unread until a full surface views it;
this contract does not add a new full-text mobile viewer.

## Upgrade boundaries

Initial historical bootstrap compares known legacy `mobile-seen.json` stamps
against actual final-message timestamps, not metadata mtimes. Unknown old
history retains the historical no-flood baseline. Relay startup imports its
bounded recent100 retained user conversations; other histories import when a
Session loads them. Running old owners and relays must restart to adopt the new
capability; they cannot acquire new behavior from an updated file on disk.

Desktop adoption belongs on its canonical session API and existing native IPC
boundary. It must not equate a watch/notification lease with a durable read.
Previously delivered native OS/cmux notifications have no common token-specific
withdrawal API: this contract synchronizes viewed/unread state, not notification
center history or notification-click side effects.
