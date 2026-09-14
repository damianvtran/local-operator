# Viewed completion receipts

A mounted frontend, open terminal, watch lease or SSE subscription is not a
read. Completion state lives in the private `attention.db` under the configured
Local Operator config root. Pending questions and approvals have their own
lifecycle and are never answered or removed by a completion receipt.

## Identity and durability

Canonical identities distinguish `session/<durable-id>` from legacy persistent
`agent/<agent-id>` conversations. Selecting an agent profile does not alias an
ordinary session to that profile's persistent conversation. Followers acknowledge
through their authenticated runtime connection, not through their own PID.

The logical run journals its token before execution. A settled eligible outcome
is journaled after durable message persistence and imported idempotently into
SQLite. Error and interrupted outcomes can have an explicit outcome marker even
when no assistant message exists; an outcome marker carries an additive `cause`
(a machine token, from the harness's own cut-off vocabulary) and `reason` (one
operator-facing sentence) whenever it knows why the turn ended.

**The default flipped for an unfinished run.** Resuming an unfinished journaled
run records an **error** naming the cause, not an interruption: a cut-off the
harness cannot explain is not a stop. Only POSITIVE evidence of a deliberate act
— a recorded stop marker, or a deliberate rung's own `user-stop` cause — records
an `interrupted`. "Deliberate rung" means every route a person's stop takes:
the `stop` control op (`/stop`, `lop stop`), the in-process dispose a bare
`/stop` performs on a TUI-owned session, the phone's `abort` button and a
supervisor's `cancel`. An INVOLUNTARY teardown (a reload, an unmount, a session
swap) is a cut-off and keeps saying so.

`cause` is empty when the harness has no evidence at all — the reason then says
the cause could not be determined, and nothing names a mechanism nobody
observed. A cause is set only where one was established.

The one exception is a run that published its own outcome before the process
went away: that marker is replayed verbatim, so a deliberate stop that settled is
still an interruption. A marker that reports a CUT-OFF is also journaled at that
boot (`session_incident`), once per token, because the process that published it
could not journal it itself — `journal_incident` refuses once the session is
disposed, and the dispose rung sets that flag before the turn's `finally`
publishes.
Copied fork journals cannot reuse another conversation's token.

A receipt names the completion the caller actually OBSERVED, and is accepted
only while it can be observed as that conversation's read state: the supplied
token must be the conversation's CURRENT completion, or the conversation must
already be read. A superseded token — a real completion of this conversation
that a newer one has replaced — is refused with 409 (`superseded_completion_token`,
carried as `detail.code` on the desktop plane and as `code` in the mobile
body) instead of being answered with a success that moves a watermark no
surface can see. The caller's remedy is in its own hands: re-read the attention
state and acknowledge the token it now names. Delayed or duplicate
acknowledgement of A therefore cannot acknowledge newer B — while B is unread it
is refused rather than recorded, and once B has been read it converges to the
same read state. **A 2xx from `/seen` means the conversation is read** (`unseen`
false); no client may treat a resolved call as proof of a read, which is how a
no-op acknowledgement used to strand a completion checkmark forever.
Runtime epoch, transcript mtime, heartbeat time and stream sequence are not
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
- `cause`, `reason` — the machine token and the operator-facing sentence for a
  non-`complete` outcome, both `""` when there is nothing to say (every
  completion, and every row written before this vocabulary existed)
- `unseen`
- `revision: [completion_sequence, acknowledged_sequence]`

The canonical frontend and mobile projection carry additive `attention` state.
Runtime capability `completion-ack-v1` enables `acknowledge_attention` with
`completion_token`. Mobile `POST /api/sessions/{id}/seen` takes the same token in
a JSON object: missing legacy bodies return422, unknown/foreign tokens409,
superseded tokens409 carrying `code: superseded_completion_token`, unknown
sessions404, and unauthenticated callers401. Reads and subscriptions do not
mutate the receipt store.

The relay maintains its existing projection ordering while alive. A new,
authenticated and source-fenced SSE connection starts with an authoritative
snapshot; its first projection may have a lower counter after daemon restart.
Retired sources cannot publish callbacks or close the current connection.

The desktop app is a `claim_delivery` claimant like any other, with
`backend="desktop"`. It reaches the primitive through `POST
/v1/desktop/sessions/{id}/notified` (`{completion_token}` →`{claimed}`), which
is cold: no bridge is acquired and no runtime is started, because a completion
worth announcing usually has no owner alive. With the desktop window unfocused
a session's `live_state` is `idle`, so a TUI observer and the desktop are both
eligible for one completion and the watermark picks exactly one — a losing claim
is the arbitration working, not a fault. The `backend` column is diagnostics
only; no decision reads it, since a claim consulting anything beyond the
monotonic sequence would stop being clock-free.

**The claim never advances the read watermark.** `claim_delivery` writes
`deliveries` alone, so `unseen` and the sidebar's mark survive a banner
untouched and a conversation can be delivered-and-unread indefinitely. The
desktop route therefore does not reuse `/seen` and must not be routed through
any foreground-receipt guard: that guard demands a focused window, which is the
exact opposite of when a notification fires. Claim-then-deliver also means the
claimant must BE the deliverer — the app claims immediately before constructing
the OS notification and after its focus gate, because a claim taken for a banner
it then suppresses would mark the completion delivered to nobody, for good.

## What a frontend can acknowledge

The selected result must actually be rendered, uncovered, and visible in a
focused foreground interface. Old scrollback, loading, a covering screen or a
child-agent page does not qualify. Terminal startup's default Textual focus
value is not positive evidence. On macOS cmux, the bounded off-loop probe also
checks the frontmost application, the same socket's kernel peer PID, and the
key visible window's selected workspace and terminal surface.

Focus evidence is obtained, not assumed — and where it can be MEASURED, it may
be re-taken on a bounded cadence rather than only on the terminal's focus
report. A terminal that was already focused when Textual enabled focus reports
sends none, so an app that starts focused would otherwise never acknowledge what
it displays; on macOS cmux the probe's verdict is that same evidence, so a
poller with no report may ask again at most once every 30 s. Terminals where the
probe answers from the environment rather than from a measurement (a plain
terminal, where its True means only that no `CMUX_*` variable is set) keep the
stricter rule: they wait for a real focus report.

Mobile transcript rows carry `text_complete`. `final` means streaming settled;
it does not prove transport retained the final row's ending. Both runtime and
relay serialization preserve `text_complete=false` when clipping a row. Missing
metadata from an older runtime is unknown, not permission to acknowledge. Only a
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
