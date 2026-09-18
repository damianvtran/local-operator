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
`completion_token`. Its successful operation ack frame retains the legacy string
`detail` and adds the owner's resulting `attention` state; followers must apply
that answer before returning rather than waiting for a later projection push.
A refused operation carries no replacement attention state. Mobile `POST /api/sessions/{id}/seen` takes the same token in
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

## Who raises the banner: the eligibility ladder

`claim_delivery` arbitrates among surfaces that are ALREADY eligible. Eligibility
itself is a separate decision, first match wins, and it is a gate rather than a
race — a surface that is not eligible never takes a claim, so the watermark stays
free for the one that will actually deliver:

1. **A surface is WATCHING the session** — a TUI attached to it, a phone, or a
   desktop window genuinely displaying it. The card is in band; no OS banner is
   raised. The predicate is VISIBILITY (`RuntimeServer.watching_surfaces()`),
   never `notification_surfaces()`: the latter answers "could a banner reach
   somebody somewhere", and using reachability to suppress meant "this machine
   can banner" read as "a human is reading X" — with the panel on X and the
   window behind another app, every OS surface went quiet while nobody looked.
2. **A notify-capable desktop app** on this host claims the completion kind. The
   machine-wide feed (`docs/DESKTOP_API.md`) composes and publishes it, so the
   runtime and a TUI stay silent.
3. **A TUI is running anywhere on this machine** — its 1 s background announcer
   raises it. Its viewer record is the signal, so a crashed TUI does not hold
   this rung forever.
4. **Nothing** — the session RUNTIME raises the banner itself. This rung did not
   exist before: with no TUI and no app, a finished turn was announced by
   nobody. It claims with `backend="runtime"` and hands the claim back through
   `release_delivery` when the spawn reports nothing went out, because a
   watermark asserting a banner nobody received is the silent hole that
   primitive exists to close.

Rung 4 is a GATE and not a claim race, deliberately: the runtime learns about a
completion at turn settle, EARLIER than the feed's 100 ms poll or the TUI's 1 s
tick, so an arm that announced unconditionally would win every completion and
make both richer paths dead. Eligibility first, claim second.

**Rungs 2 and 3 are narrowed by KIND**, and the gate path is not touched. The
machine-wide presence advertises `can_notify_kinds` (`["complete","error"]`)
because the feed carries completions only; a parked `ask`/`approval` therefore
keeps the per-session lease and the per-session backend toast it has today.
Widening that suppression to the machine-wide lease would have silenced a
background session's parked question with nothing to replace it.

**A burst is capped.** At most `BURST_LIMIT` (3) individual banners per poll, on
both transports (the feed's frames and the TUI's own announcer, asserted equal by
a test). The remainder is announced as ONE digest naming the count rather than
dropped, and the feed's digest carries the member session ids so a click can land
on the catalogue instead of on an arbitrary member.

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

### A gesture may acknowledge what a surface ENUMERATES (R10)

The rule above is per result. Clearing a whole pile is the one place it is
deliberately relaxed, and the relaxation is narrow enough to state:

**An explicit user gesture may acknowledge the completions a surface
enumerates, token-bound; no automatic path may acknowledge anything.** The
surface sends the completions it actually rendered — one `(conversation,
token)` pair each — and the store compares every pair against that
conversation's CURRENT completion inside one write transaction. A completion
published after the render is not in the batch, so it stays unread, and the
caller learns which items did not clear (`superseded`, `unknown`) rather than a
success that moved a watermark no surface can see. Clearing is therefore not a
sweep of "everything unread": it is the same observed-token rule applied to a
list, and `acknowledge_all()` does not exist on the store for exactly that
reason.

The explicit half is load-bearing too. A timer, a poll, a subscription or a
focus change may never reach the batch operation, and nothing here changes the
per-result rule for the surfaces that acknowledge one at a time. The TUI's
`/notifications read` renders the set it is about to clear — one catalogue read,
painted before the write — and acknowledges exactly that read's pairs, so the
rows a user clears are rows they were shown, including the one that finished
while they were typing; the desktop clears the rows its sidebar holds, and only
with its window in the foreground (`guardForegroundReceipts` covers the op by
name). Reading is still not notifying: `deliveries`, the supersede log and the
completion rows are untouched by a bulk acknowledgement, so an already-delivered
banner stays delivered and nothing read can be resurrected as unread.

Two more things this relaxation depends on, both stated because they are easy to
re-derive differently:

- **A store that could not be READ is not an empty pile.** The empty states
  (`No unread completions.`, `Nothing unread.`) are findings, and a failed read
  supports neither. Both surfaces therefore branch on ONE classification —
  `session/store_failures.py`, consumed by the desktop ladder and by the TUI —
  whose codes and log levels decide the condition and the ink (contention is
  retryable; a full disk and an unopenable store are not). The SENTENCE is
  composed per surface, deliberately: the desktop's strings are the send path's
  ("the message could not be written", "send it again") and are false about a
  receipt clear, which has no message in it. Both surfaces do say the same three
  things — which condition they met, what could not happen, and whether retrying
  is the remedy. A failed read also means no write: an acknowledgement that cannot be
  verified is not a receipt.
- **The word is `unread` on both surfaces, and `unseen` is the store's field.**
  The TUI's sidebar tooltip for the same mark says "Unseen completion"
  (`CatalogEntry.status`), so one app spells the state twice. That is recorded
  rather than fixed here: `unread` is the word the desktop half ships in its
  control, its receipts and its row tooltip (and the word this document uses for
the watermark a human read), while `unseen` is the column an `AttentionState`
  carries. Two words with one meaning, owned by two layers, is a smaller defect
  than three surfaces renaming a status string — but it is a defect, and this is
  where a future round should look before it moves either one.

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
