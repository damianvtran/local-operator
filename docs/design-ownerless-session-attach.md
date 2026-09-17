# Design: a session attaches without an owner, and a read never needs one

Status: **proposal (architect)**. Base: `origin/main` `bf67bf699` in
`~/local-operator` (read in the worktree `~/local-operator-worktrees/ownerless-attach`,
branch `fix/desktop-attach-without-owner`) and `origin/main` `da9e75a61` in
`~/local-operator-ui`. Scope: **two backend PRs and one UI PR** (§6). No
`pyproject.toml` bump in any of them — the window's release owner handles that.

Operator's requirement, verbatim:

> Running into some strange issues via the UI, can you fix?

> Also there shouldn't be a 'session owner' make sure that we properly support
> detached runtimes, a session/runtime should be able to be connected to without
> needing to reach any 'owner'. Fix robustly and make sure that the ui is
> properly able to connect to any session

Every `file:line` below was read on `bf67bf699` (backend) or `da9e75a61` (UI).
One throwaway probe was run against an **isolated config root** with a synthetic
12-hex session id (§1.1); nothing was run against the operator's own sessions or
their config dir, and the product code was not modified.

---

## 1. The problem as I found it

### 1.1 The measurement: the same session reads in 0.02 s, or fails after 15.27 s, depending only on whether a silent process is alive

The route a read goes through is four steps and the last one is the only one
that can fail:

`GET /v1/desktop/sessions/{id}` (`routes/desktop_sessions.py:1326`) →
`DesktopSessions.session` (the pool door, `utils/desktop_sessions.py:2829`) →
`bridge.acquire()` (`:993`) → `AttachedSession.attach_existing()`
(`session/attached.py:2248`) → `_bind_to(record, sync_timeout=15.0)` (`:3222`).

A throwaway probe (isolated config root; a synthetic session directory with two
durable transcript rows; a fake owner that **welcomes the dial and then answers
nothing**) produced this, and it is the whole problem in four lines:

```
=== case 1: a live owner that welcomes and never syncs ===
[owner] accepted connection #1; reading the auth frame
[owner] sent the welcome; now staying silent about frontend_sync
[live-silent] acquire() RAISED after 15.27s: OwnerAckTimeout: owner did not
              answer 'desktop_watch' within 15s
[live-silent] history() offline: entries=2

=== case 2: no owner at all (record removed) ===
[no-owner] acquire() returned after 0.02s
[no-owner] snapshot with no remote: cold=True
[no-owner] history() offline: entries=2
```

Two facts, both load-bearing:

- **The durable answer was available in both cases.** `bridge.history()` served
  both rows with no owner at all, in the same process, on the same session
  directory. Nothing about a read needs a runtime. `AttachedSession.cold`
  (`attached.py:1415`) already parses the whole transcript into the facade
  (`:1455`), and the bridge's own `snapshot()`/`history()`
  (`utils/desktop_sessions.py:1411`, `:1476`) read from disk.
- **The only difference between the two cases is a live process that does not
  answer.** Case 1 is not an error state: the runtime is alive, its transcript
  lease is held, and it will answer again as soon as its loop is free. The read
  is refused anyway — and this matches the live reproduction the manager
  recorded (HTTP 503 after 17.33 s on `/snapshot` and on `/history`, with the
  same session's runtime alive at pid 29037).

### 1.2 A desktop read's first blocking need is a PRESENCE RPC (F1)

The probe's exception names the wait, and it is not the canonical sync:

`_dial` (`attached.py:3311`) opens the socket and, because the desktop facade is
built with `surface="desktop"`, then does:

```python
if self._surface == "desktop":
    ...
    await client.desktop_watch(visible=..., can_notify=...)   # attached.py:3423-3437
    except BaseException:
        client.close(); self._client = None; raise
```

`desktop_watch` is `AttachClient._request("desktop_watch", ...)`
(`mobile/attach_client.py:1404`) with `ACK_TIMEOUT_S = 15.0` (`:64`), so a silent
owner turns it into `OwnerAckTimeout` (`:164`) after 15 s.

**A read has no use for that RPC at all.** It asserts the *renderer's presence
lease* — the thing that keeps a runtime resident and lets the daemon raise a
notification (`server.py::attach_clients`, `DESKTOP_WATCH_LEASE_S`) — and its own
contract is TTL-bounded and self-correcting: the renderer re-beats every 15 s
and the lease expires after 45 s. The code already treats its sibling on the
same dial path exactly that way:

```python
if self._event_mute_requested:
    try:
        await asyncio.wait_for(client.set_event_muted(True), timeout=5.0)
    except Exception:  # a lost re-assert is a cost, not a defect
        logger.debug("event mute re-assert failed", exc_info=True)
```

(`attached.py:3414-3422`.) Two best-effort re-asserts of the same kind sit five
lines apart, and the one that is *not* best-effort is the one that fails a read.

### 1.3 The failure throws away a socket it already has (F2)

The client is already authenticated by the time that wait runs — the welcome was
read and the identity checked (`attach_client.py:1000-1010`). On the failure,
`_dial` closes it (`:3435`) and `_bind_to`'s `except BaseException` runs
`_discard_rejected_client()` (`attached.py:3259`, `:3267`). Both are right for a
*refused* dial and wrong for a *slow* one: the sync the owner pushes at
connect time (`server.py:2038-2105`) may arrive moments later, and by then
nothing is listening for it.

`_on_frontend_sync` (`attached.py:4698`) resolves **only** the in-flight future;
once a dial has been discarded there is no path by which a late sync installs
state. So "the owner was slow" and "the owner refused" are collapsed into one
outcome, and the collapse is irreversible.

### 1.4 A write needs a socket, and is made to demand a sync (F3)

`admit_prompt` (`attached.py:2277`) needs exactly one thing from the owner:

```python
await self._ensure_bound()
client = self._client
if client is None or not client.connected:
    raise ConnectionError(self._unavailable_reason())
return await client.request_ack_with_duplicate(...)
```

The prompt is acked on the durable transcript append (`serving.py:1890`), which
is an *independent* round trip from the canonical sync. But `_ensure_bound`
(`:2863`) returns early only when `not self.is_cold` (`:2912`), and `is_cold`
(`:2176`) is true whenever `_ready_for_events` is false — a state `_bind_to`
clears only after the sync **and** the history page have landed (`:3251-3258`).
So a write waits out the same 15 s sync envelope, and on expiry
`_bind_under_lock`'s retry loop raises the last error (`:3178`), which the route
ladder turns into the 503 at `routes/desktop_sessions.py:1022`.

The code comments state the same conclusion from the other end:

> The one case a retry cannot help is a genuinely silent owner, which is also
> the only case that consumes the envelope.
> — `attached.py:306-310`

### 1.5 A drained runtime refuses work while holding the lease, and the drain has no clock bound (F4)

`process._drain_for` (`:1188`) says it in one line:

> Nothing in flight is aborted: the wait is bounded by the work, never by a
> clock. — `process.py:1199-1200`

and the gate it waits on, `_idle_for_refresh` → `may_refresh`
(`serving.py:1460`), is true only when `is_busy()` is false, where `is_busy()`
covers *"a live turn, a parked gate, a running goal loop, live subagents and
background jobs"* (`serving.py:1470-1472`).

While that wait runs:

- every admission is refused — `prompt` raises `_retiring_refusal()`
  (`serving.py:1863-1868`), which renders as
  `"This session is leaving; it will not start a new turn. The message was not
  admitted — send it again once the session is running again."`
  (`session/errors.py:130-132`, joined at `:163`);
- **no successor can start**, because the draining runtime still holds the
  transcript lease and `engage_runtime` will not spawn while a live pid holds it
  (`session/runtime/launch.py:23-27`);
- and the sentence's promise — *"send it again once the session is running
  again"* — has nothing behind it until the drain ends.

The signal path bounds this at `SIGNAL_DRAIN_S = 120 s`
(`runtime/types.py:341`, `process.py:1366`); the **build** path, which is the one
that fires for the whole fleet after a `lop-update`, imposes no bound at all
(`process.py:1285`: *"the build drain waits on its work alone"*). The manager's
reading of the live box — ~17 runtimes alive whose `leaving` column reads
"leaving for the build on disk when its turn ends" — is consistent with that
state being reachable and persistent. I have **not** proved why those particular
runtimes have not converged (a live background job and a stuck busy bit are both
plausible; see §7), and the design below does not depend on which it is.

### 1.6 The renderer's three misclassifications (F5, F6, F7) — all UI, all today

- **F5 — a provably-unadmitted refusal is treated as unknowable.**
  `isRefusedBeforeAdmission` (`canonical-sessions-store.ts:414`) answers true only
  for `413`, `422`, and the read-window code. The retiring refusal arrives as
  **409** with `code: "runtime_retiring"` (`routes/desktop_sessions.py:176-181`,
  `:983-998`), so `chat-page.tsx:1075` returns `SEND_HELD`, the echo is kept,
  and the composer offers *Restore unsent message* (`message-input.tsx:3637`)
  over a message the backend has just said it never accepted. That is the held
  draft the operator reported.
- **F6 — a session-scoped 503 is classified as `unreachable`**, i.e. "the Local
  Operator server is not running" (`backend-error.ts:56-58`). The session panel
  reaches the same conclusion by a different route: `/events` acquires the
  bridge **before** returning headers (`routes/desktop_sessions.py:2249-2267`), so
  the stream 503s; `use-canonical-session.ts:338-339` then spends
  `STREAM_RETRY_DELAYS_MS` (23.5 s of delays plus per-attempt latency, and every
  attempt pays the 15.27 s of §1.1) before painting
  *"Lost the connection to this conversation"* (`:1476-1483`,
  `desktop-stream-notice.ts`).
- **F7 — the guard read treats any failure as "the session does not exist".**
  `openSession` commits first and validates behind the commit with
  `sessions.get` (`canonical-sessions-store.ts:1597`); any throw rolls the view
  back to the previous conversation (`:1616-1631`). A 503 — a backend that
  answered and said "not yet" — is therefore indistinguishable from a 404, and
  clicking a row whose runtime is merely busy bounces the user back after
  15-17 s with no explanation.

### 1.7 What is already right, so that the fix is small

Worth stating plainly, because it is why this is a coupling problem and not a
missing-mechanism problem:

- **Discovery already attaches to a runtime ANY surface started.** The owner's
  pid comes from the session directory's `.session.pid`
  (`resume.py:1135`), never from the caller's memory, and the record is
  published per runtime (`registry.py::publish`). A viewer of a TUI-, phone-,
  peer- or supervisor-started runtime is the ordinary case
  (`launch.py:1-31`).
- **Reads already never spawn.** `attach_existing` (`attached.py:2248`) only
  adopts a record that exists; the documented invariant is *"a cold read or
  stream attaches only to an already-live runtime"* (`docs/DESKTOP_API.md:625`).
- **A cold facade already serves everything a read needs**, including the
  durable turn-end checkpoint (roster, todos, title, spend) at `cold()`:
  `attached.py:1455-1474`.
- **The wire already reports cold** (`utils/desktop_sessions.py:1472`) and the
  renderer already consumes it, including an epoch rollover when a cold viewer
  later attaches (`use-canonical-session.ts:1213-1214`, `:1240-1252`).
- **At-most-once is already durable and survives process replacement.**
  `CommandReservations.reserve` consults `session.has_admitted_command`
  (`mobile/command_reservation.py:68`), and the transcript entry id **is** the
  caller's id (`Transcript.append_message` → `Message.user(id=message_id)`,
  `session/session.py:5172`). Re-admitting the same `command_id` to a successor
  is therefore safe by construction, which is what makes §3 D6 possible.
- **Handover of work across a drain already exists** for peer messages:
  `_spool_for_successor` → `inbox.jsonl` → drained by the successor *before its
  socket listens* (`serving.py:1428-1458`, `process.py:1446-1500`, `:1663`).
- **The desktop app already warms off the request path** (first keystroke,
  `use-warm-session.ts`) and already holds a presence lease
  (`use-desktop-watch-lease.ts`).

---

## 2. The invariant that decides everything

**One writer per transcript, and no second writer as a remedy.** The lease
(`.session.pid`, `live_runtime_pid`) and `engage_runtime`'s lease arbitration
exist so two processes never append to one journal (`launch.py:9-18`). Every
decision below preserves that. What changes is only *who is asked, how long, and
what happens when the answer does not come* — never the number of writers.

Corollary, and the operator's actual requirement: **no surface may need to know
that an "owner" exists.** An owner is an implementation fact of the lease; it is
not a party the UI can be asked to reach. What the UI needs is a per-session
*reachability* answer, which it must get without ever failing a read.

---

## 3. Decisions

Each decision names the option taken, the rejected alternative, and the files
that change. Budgets are chosen against the constants already in the tree:
`COLD_FALLBACK_S = 8.0` (`attached.py:174`), `FRONTEND_SYNC_FOREGROUND_S = 15.0`
(`:277`), `FRONTEND_SYNC_BACKSTOP_S = 120.0` (`:267`), `ACK_TIMEOUT_S = 15.0`
(`attach_client.py:64`), `_ADMISSION_ACK_BOUND_S = 2.0`
(`routes/desktop_sessions.py:99`), `SIGNAL_DRAIN_S = 120.0` (`runtime/types.py:341`).

### D1 — Reads are served cold, on a 2 s live attempt, and never raise

**Decision.** A read route acquires the bridge in a new **read mode**: it makes
one bounded attempt to attach to an existing record and, if the owner has not
delivered canonical state inside `READ_ATTACH_BUDGET_S = 2.0`, it serves the cold
facade and returns. It never raises `ConnectionError`/`TimeoutError` for a
session that exists on disk.

Which routes are reads (read mode): `snapshot` (`routes:1326`), `history`
(`:1331`), `events` (`:2241`), `seen` (`:1783`), `notified` (`:1791`),
`child_transcript` (`:1346`), `child_attachment` (`:1385`), `attachment`
(`:1423`), and the bridge's own `watch`/`events` (`utils:1512`, `:2097`).
Everything else keeps the existing control envelope: `messages` (`:1501`),
`commands` (`:1611`), `answers` (`:1749`), `interrupt` (`:1975`),
`working-directory` (`:2119`), `warm` (`:1818`).

**Why 2.0 s.** Not `COLD_FALLBACK_S = 8.0`: that is the point at which a *viewer*
concludes owner loss and ends an in-flight turn locally with a named
`owner-lost` cut-off (`attached.py:183-190`), and paying it on every read would
put 8 s in front of every panel open. Not `FRONTEND_SYNC_FOREGROUND_S = 15.0`:
that is the envelope for an action a user is waiting on, which is exactly what a
read must not become (the measured failure is 15.27 s, and the route adds the
rest of the observed 17.33 s). 2.0 s is the number this codebase already uses for
*"one socket round trip plus the leg's own work"* against a 15 s caller deadline
— `_ADMISSION_ACK_BOUND_S`, whose comment records the legs at *"single-digit
milliseconds"* (`routes:275-289`). A healthy owner lands the sync and the
welcome inside it by three orders of magnitude; a stalled one does not, and the
read does not care.

**Which surfaces observe the difference.** The app's panel open
(`useCanonicalSessionStream`) and its guard read (`sessions.get`,
`canonical-sessions-store.ts:1597`) go from 15.27 s + 503 to ≤2 s + a cold
paint; `lop`/CLI callers of the same routes are unaffected because they do not
read these routes; the TUI is unaffected (it is not an HTTP client).

**Rejected alternatives.**
- *Serve cold immediately, with no live attempt* (0 s). Simplest and fastest,
  and rejected: it would make a healthy owner's canonical state — the whole
  reason the sync exists, including a turn in flight and any pending gate —
  a second-class answer on the panel's first paint, when today it arrives in
  ~10-40 ms. The bounded attempt keeps the live answer for the case that
  actually works and gives it up only for the case that does not.
- *Keep the 503 but shorten it to 2 s.* Rejected: the read still fails, and
  §1.1 shows the failure is unnecessary — the durable answer is right there.
- *Have the renderer retry harder.* Rejected: F7 already shows a retry of a 503
  is indistinguishable from "the session is gone", and the panel's retry budget
  is already 23.5 s of delays.

### D2 — Make the dial survive a late answer, and stop letting a presence RPC fail a read

**Decision.** Two changes inside `_dial`/`_bind_to`, both of which D1 depends on:

1. The desktop watch re-assert becomes **best-effort, exactly like the mute
   re-assert five lines above it** (`attached.py:3414-3422`): bounded
   (`await asyncio.wait_for(..., timeout=5.0)`) and swallowed on failure. It is
   a lease assertion whose own TTL (45 s) and the renderer's next beat (15 s)
   repair it; a read must not be refused because a presence hint went
   unacknowledged.
2. A dial that authenticates is **retained past the sync envelope**. When the
   canonical sync has not landed inside the caller's budget, the facade records
   "socketed, unsynced" and returns; the sync that lands later installs the
   state and publishes the rollover frame the renderer already handles
   (`_install_frontend(..., publish=True)`, `attached.py:4746`; consumer at
   `use-canonical-session.ts:1240-1252`). A hard landing deadline
   (`SYNC_LANDING_DEADLINE_S = 30.0`) abandons the client if nothing ever
   comes — because an attach socket is a *residency* term of the runtime's own
   exit predicate (`process._should_exit`, viewer term 3), and a viewer that has
   given up must not hold an 82 MB process up.

**The semantics are already established in this file, which is why this is a
small change rather than a new contract.** `_await_frontend_preemptible`
(`attached.py:3521-3576`) already means *"shorten the wait, never discard the
dial"*: it never cancels the future, and `test_a_preempted_wait_still_adopts_a_sync_that_lands`
(`tests/unit/session/test_frontend_sync_liveness.py:881`) pins that a sync landing
inside the shortened window *"is still adopted"*. What D2.2 changes is only the
end of that same arc: when the **total** envelope expires, `_await_frontend`
raises `RuntimeUnresponsiveError` (`:3519`) and `_bind_to`'s handler closes the
socket (`:3259-3265`), so the sync that would have been adopted a moment later
has nowhere to land.

**Rejected alternative.** *Discard the client on a sync timeout (today's
behaviour) and let the next read redial.* Rejected because it can never recover:
each read pays the full envelope, and a busy owner is then unreadable for as long
as it is busy — the failure is self-perpetuating.

### D3 — What the wire reports, so the renderer can tell "live" from "served cold"

**Decision.** `snapshot.payload` keeps `cold` (`utils:1472`) and gains
`cold_reason`, one of `"no-runtime" | "owner-silent" | "owner-leaving"`, plus
`"attaching"` on the frame that announces a retained dial that has not synced
yet. `frontend.replace` carries the same `cold`/`cold_reason` pair
(`utils:969-991`). The three values are distinguishable from facts the bridge
already has: no record at all; a live record that has not produced a sync; a
record whose `leaving` phrase is set (`runtime/types.py:636-645`).

Rules for the vocabulary:
- **Additive and defaulted.** A backend that omits `cold_reason` is read as
  `cold ? "no-runtime" : null` — every existing renderer path keeps working.
- **No prose for the renderer to parse.** Copy stays in the app; the wire carries
  a token, the same discipline `code` already follows in the error ladder.
- **The refusal codes carry the disposition the renderer branches on**
  (§3 D6): `runtime_retiring` (the session exists; the message was **not**
  admitted; retry the same id), `runtime_unreachable` (transient), and
  `unknown_session` on the existing 404.

**Rejected alternative.** *Report "live" with a degraded flag.* Rejected: a
degraded live frame cannot be acted on — the renderer would have to guess
whether the missing state is a slow sync or an absent runtime, which is the same
conflation §1.2-1.3 exists to remove.

### D4 — A read does not spawn (and the two things that do are unchanged)

**Decision.** No read path calls `_ensure_bound`/`engage_runtime`, so D1 adds no
spawn. "The next action finds a live runtime" stays the job of the two
mechanisms that already do it and are already driven by the app:

- `POST /warm` (`routes:1818`), fired on the first keystroke into an empty
  composer (`use-warm-session.ts`), and
- the live **visible** watch lease (`utils:1533-1604`), which creates residency
  for a session the user is looking at, with the existing backoff
  (`_LEASE_WARM_BACKOFF_S = 30`, cap 120 s) so a session that cannot start is
  not respawned per heartbeat.

**Rejected alternative.** *Have the read kick a warm when it finds no runtime.*
Rejected: it breaks the documented side-effect-free property of a GET
(`DESKTOP_API.md:625`), and on a 100-row sidebar sweep would spawn for rows
nobody clicked. What D1 does *not* do is abandon the dial it already opened
(D2) — that is a read completing its own work, not starting new work.

### D5 — A write needs a socket, not a sync

**Decision.** The bind a *write* requires is an authenticated connection.
`_ensure_bound` gains the distinction: if a client is connected and
authenticated, it returns without demanding `_ready_for_events`; the canonical
state arrives as a delta/rollover when the owner gets there, and until then the
facade reports `cold`/`attaching` on reads while accepting the write. This is
sound because the prompt ack is the durable append (`serving.py:1890`), not the
sync: `.request_ack_with_duplicate` (`attach_client.py`) rides the same socket
independently.

**What the user sees in each case**, and this is the part the UI PR must render:

| the owner is | the write's outcome | what the renderer shows |
|---|---|---|
| healthy | `admitted` in ~10-40 ms | unchanged today |
| starved but alive (loop busy) | the request is on the socket; the ack may miss `_ADMISSION_ACK_BOUND_S` (2 s) and the receipt answers **`pending`** | the echo stays painted, marked *not yet acknowledged*; **no held draft**; the durable row arrives when the owner appends it and coalesces with the echo by id |
| gone (no record) | `_ensure_bound` engages a fresh runtime (unchanged behaviour, `launch.py:20-31`); if it cannot start, `ActionableConnectionError` carries the vetted reason (`:3062`) | today's message, unchanged |
| draining for a new build | the refusal (D6) | D6 |

The second row is the honest end for a starved owner: the text **is** with the
owner (it is in the socket's buffer and will be read), so calling it failed
would be a lie, and calling it delivered would be a second one. `pending` is
already the documented disposition (`DESKTOP_API.md:573-591`) and the backend
already publishes `admission.failed` when a pending admission later fails
(`routes:145`, `:681`).

**Gap to close with this decision:** the renderer currently ignores both the
`admission.status` and the `admission.failed` frame — `admitChatDraft` calls
`store.finishDraft` on any resolved response
(canonical-sessions-store.ts:802), and no UI file reads
`admission.failed`. So a `pending` write is painted as a success. The UI PR
consumes both: `pending` ⇒ keep the echo + *not yet acknowledged* state;
`admission.failed` ⇒ the same treatment F5 gives a refusal.

### D6 — A message that provably was not admitted is not the operator's problem to solve by hand

**Decision, three parts.**

1. **The refusal keeps its type and gains its disposition.**
   `_retiring_refusal` (`serving.py:1320`) already carries
   `RuntimeRetiring.code = "runtime_retiring"` (`errors.py:66`) and the routes
   already answer 409 with `{code, message}` (`routes:983-998`). Add
   `"retryable": true` to that body — the message **was not admitted**, which the
   sentence itself says (`errors.py:131`), so this is a statement of fact, not a
   promise.
2. **The bridge re-admits it, once, under the same `command_id`.** On a drain
   refusal the bridge waits for a live runtime within
   `ADMISSION_DEFER_BUDGET_S = 30.0` (the engage deadline the same code already
   allows a foreground bind, `launch.py::DEFAULT_DEADLINE_S`) and re-issues the
   admission. Safety is structural: the message provably was not admitted, the
   caller's id is the durable row id, and the successor's
   `CommandReservations.reserve` refuses an id already in the index
   (`command_reservation.py:68`). At most once, either way. The common drain ends
   in ~1 s, so the ordinary case becomes invisible to the operator.
3. **When the window expires, the refusal is terminal-for-this-attempt and the
   renderer owns the retry** — under the same id, with the echo still painted,
   retried when the session next reports live, and *never* as a held draft.
   `runtime_retiring` joins the "nothing was admitted" family in
   `isRefusedBeforeAdmission` (`canonical-sessions-store.ts:414`) as a third
   disposition (`deferred`), distinct from the 413/422 case: the text does not go
   back in the box (the user pressed Send once), the echo stays, and the row says
   *waiting for the session to restart on the newer build*.

**Rejected alternatives.**
- *Reuse `_spool_for_successor` / `inbox.jsonl` for the user's own prompt.*
  Rejected, and it is worth writing down because it looks like the elegant
  answer: the successor delivers a spooled row by calling
  `receive_peer_message(mode="mailbox")` (`process.py:1487-1495`), so the
  operator's message would land as a **peer card**, not their own turn, and
  `InboxLine` carries no `command_id` (`inbox.py:82-98`) — the renderer's
  already-painted echo could never coalesce with the durable row, and `mode` is
  deliberately not honoured at delivery (`serving.py:1415-1421`), so a steer
  intent would be lost. The mechanism is right for peer traffic and wrong for
  the composer.
- *Answer 200 with `admission.status = "failed"` (what `/commands` does via
  `admit_receipt_request`, `routes:1714`).* Rejected for `/messages` as the
  renderer stands: a 200 resolves the send and `finishDraft` deletes the draft,
  so the user would lose the message silently. Unify the *codes* across the two
  routes; keep the status semantics.
- *Wait indefinitely in the request.* Rejected: a drain can legitimately last
  tens of minutes (D7), and a parked HTTP request is not a place to hold a
  user's message.

### D7 — The build drain gets a generous clock bound, because "unwritable forever" is its own defect

**Decision.** `_drain_for` (`process.py:1188`) gains a maximum dwell
(`BUILD_DRAIN_MAX_S`, default **1800 s**), after which the runtime exits as the
signal path does at `SIGNAL_DRAIN_S` (`process.py:1366-1389`): the exit is logged
loudly, the turn in flight is disposed with its existing label, and the lease is
released so a successor can boot.

**Why generous, and why it is not the fix for the operator's flow.** The bound
must not cut legitimate long work — that is precisely the incident the drain was
built for (`process.py:1289-1302`, PR #1141), and a short bound would be worse
than the disease. Its job is the class §1.5 describes: a drain that can never
converge because the work it waits for cannot be reached or finished by anyone
(a live background job nobody can stop, a `busy` probe that answers from a state
no client can clear). Today that state is terminal for the session — unwritable
*and* unhandable-off — and no surface can repair it. With the bound, it always
ends, and the D6 retry is the thing that makes the wait feel short.

**Rejected alternative.** *Bound it short (e.g. `SIGNAL_DRAIN_S`).* Rejected:
it converts a 3-minute legitimate turn into an aborted turn on every
`lop-update`, which is the regression #1141 fixed.

### D8 — What "no owner" means, precisely

**Keep:**
- the **transcript lease** (`.session.pid`, `live_runtime_pid`, `engage_runtime`'s
  arbitration). This is the single-writer invariant and it is not negotiable
  (§2). Nothing in this design weakens it, and no path here spawns a second
  writer.

**Change:**
- **`_no_takeover` stays, its sentence does not.** `_no_takeover`
  (`utils:144-145`) is correct: an HTTP backend must never become the runtime for
  a session (it would be a second execution host and a second writer). But its
  message — *"Desktop viewers cannot own a runtime"* — leaks internal vocabulary
  if it ever surfaces, and it is reachable only through the legacy
  takeover-on-owner-loss contract, which a viewer facade never takes: `cold`
  sets `_can_go_cold = True` (`attached.py:1449`) and `surface="desktop"` sets it
  at construction (`:904`), so `_give_up_recovery` goes cold instead
  (`:6104`). Make that explicit with a typed internal error and a test that the
  viewer path cannot reach the factory.
- **The user-visible sentence loses the word "owner".** The 503 read
  (`routes:1022`) is replaced by D1 for reads and by
  `code: "runtime_unreachable"` for the control paths that keep a transient
  refusal. **This is a two-repo change and must ship as one intent:** the app
  matches that sentence by prefix today — `MCP_SESSION_UNAVAILABLE =
  "Session owner is unavailable."`, matched at `mcp-failure.ts:222` — so
  renaming it without the UI change silently changes the MCP row's copy. The
  renderer should key on `detail.code` and keep the prefix only as a fallback
  for older backends.
- **What a client does about ownership: nothing, ever.** A viewer attaches to a
  runtime that exists, whoever started it. There is no "whose is it" question in
  the protocol, and after this design there is no read that can fail because the
  answer to that question is inconvenient.

### D9 — The `(None, owner)` trap gets one answer, and it is not "spawn"

`find_runtime_record` (`attach_client.py:686`) returns `(None, pid)` for *"an
owner exists but no usable record does"* — which today covers three different
facts: a v1 record, a record for a pid still stamped with the previous
`session_id` (the rebind race), and a **wedged** record (`scan`'s third state,
`registry.py:476-479`, heartbeat older than `HEARTBEAT_TIMEOUT_S = 45`). Only
`dialable_record_exists` (`:737`) separates them, and its only caller is the TUI
(`tui/app.py:14180`).

What each caller must do, and what it does instead:

| caller | today | required |
|---|---|---|
| `attach_existing` (`attached.py:2266-2270`) | `record is None` ⇒ cold | already right for a read (D1), keep |
| `_bind_under_lock` (`:3116-3138`) | `record is None` after engage ⇒ `ConnectionError("could not start a runtime for this session")` | with a live pid holding the lease, do **not** claim "no runtime": dial the owner's own record if one exists at all (live **or** wedged), let the welcome's identity check arbitrate, and only then report unreachable |
| `saved_preview` (`:1390-1392`) | `(None, owner)` ⇒ `FileNotFoundError` | unchanged (it is a display fallback) |
| wake supervisor (`wakes/supervisor.py:572`) | filters to live | unchanged |

The wrong answer to `(None, owner)` is the one that spawns: `engage_runtime`
already refuses to spawn while a live pid holds the lease (`launch.py:23-27`),
so a spawner would only build a candidate doomed to lose the race — while
telling the user their session has no runtime.

---

## 4. The client-visible state machine

One session, as the desktop renderer sees it. `cold_reason` is D3's field; the
two "transient" rows are the ones that must never hold a draft.

| state | how the renderer learns it | reads | sends | subscriptions |
|---|---|---|---|---|
| `unknown` | nothing fetched yet | not attempted | composer gated (§F7's read window) | not open |
| `connecting` | stream opened, no snapshot yet | — | gated until the first snapshot or `404` | open, `open` frame pending |
| `cold(no-runtime)` | `cold: true`, `cold_reason: "no-runtime"` | served from disk; page is empty ⇒ `/history` reconcile (already implemented) | allowed; the send engages a runtime inline (unchanged) | open; frames arrive once a runtime is engaged |
| `attaching(owner-silent)` | `cold: true`, `cold_reason: "owner-silent"`, `attaching: true` | served from disk, live state lands later as a rollover | allowed (D5); the ack may answer `pending` | open; deltas buffered until the sync lands, then applied in order |
| `live` | `cold: false` | live canonical state | `admitted` | full stream |
| `leaving` | `cold_reason: "owner-leaving"` (record's `leaving` phrase set) | served from disk and still live-capable until the exit | refused with `runtime_retiring`; the bridge re-admits within 30 s (D6) or the renderer retries the same id | open; the `retiring` frame (`draining: true`) is the announcement |
| `gone` | no record, lease free, session dir present | cold | allowed; engages a fresh runtime | open, cold |
| `unavailable` | `404` on the session, or `SessionStoreUnavailable` | none | none | closed |

**Transitions the renderer must handle (and the UI work they imply):**

1. `cold → live`: an epoch rollover in a `frontend.update`. Already handled
   (`use-canonical-session.ts:1240-1252`); D2 makes it the *normal* path for a
   starved owner rather than a rare one, so it needs a test that asserts the
   paint sequence rather than only the reducer's behaviour in isolation.
2. `connecting → cold`: already handled (the snapshot is the first frame, and
   cold is a field of it).
3. `live → attaching`: this is the one that does not exist today — an owner that
   goes silent *while attached* is `_recovering`, and after `COLD_FALLBACK_S = 8.0`
   the facade gives up and goes cold (`_give_up_recovery`, `attached.py:6104` →
   `_go_cold`, `:5437`). The renderer sees a cold snapshot either way; the new
   part is that the **reads keep answering** across it, and that the retained
   dial (D2.2) can still upgrade it.
4. `any → unavailable`: **only** `404` may paint this. A 503 must not (F6/F7):
   the guard read keeps the target, and the panel keeps its retry.
5. `leaving → live`: the successor's rollover. The renderer must keep the
   composer's text (D6) across it, which is exactly what the echo + same-id
   retry buys.
6. `live → leaving`: the `retiring` frame. The renderer must show the drain
   state rather than treat an exit as `gone` — the runtime is alive and still
   serving reads, and (per §1.5) may stay that way for a long time.

---

## 5. Risks

1. **A retained unsynced socket holds a runtime resident** (D2). The runtime's
   exit predicate counts an attach client
   (`process.py:570-602`, term 3 `_viewer_attached`), so a socket we keep "just
   in case" delays an idle reap by up to the process's own policy. Mitigation is the hard
   `SYNC_LANDING_DEADLINE_S` in D2 plus the existing `ATTACH_MAX_CLIENTS` LRU
   cap; watch it in QA by measuring an idle runtime's residency with the panel
   open against one with it closed.
2. **Two dials for one viewer.** With `is_cold` still true while unsynced, a
   `warm`/lease-warm arriving in that window will dial again
   (`_bind_under_lock`'s entry guard reads `is_cold`, `:3101`). Bounded (the
   loser is discarded, the LRU cap holds), but it should not happen: gate the
   redial on "no client connected" rather than on `is_cold`.
3. **The re-admit races the retiring runtime's own exit.** If the drain ends
   while the bridge is re-admitting, the socket dies mid-request; the admission
   is then either `pending` (it was written) or a transport error (it was not).
   Both are handled by D6's dispositions, but the *interaction* is the highest-
   risk code in the change and needs a test that kills the owner between the
   refusal and the re-admit.
4. **`BUILD_DRAIN_MAX_S` is a new way to cut a turn** (D7). It fires only after
   30 minutes of a drain that has not converged, which is a state no operator
   can currently escape at all; the test must assert the existing label on the
   cut turn (`runtime-shutdown` semantics) so a bounded exit is never mistaken
   for a user's stop.
5. **The wire rename in D8 breaks a UI string match** if the two repos move
   out of step — see the note in D8. This is the most likely way for a correct
   backend change to produce a *new* UI bug, and it is why the code, not the
   prose, has to become the contract.
6. **Cold-first paint changes what "the panel is showing" means** in a
   screenshot-level sense: a starved owner's first frame now comes from the
   durable checkpoint (last turn end) instead of the live state. Designer
   review should confirm the cold affordance is legible and that the `attaching`
   row does not read as an error.

---

## 6. The PR split

Two backend PRs and one UI PR. B1 is the reported failure (reads); U1 is the
renderer half; B2 is the write path. B1 and U1 can go in parallel — U1's
`runtime_retiring` branch keys on a code that already exists — with B2 last.
The alternative (one backend PR) is rejected because it would put the bind
semantics (D2/D5), the bridge's acquire modes (D1) and the drain (D7) in one
review round, and each is separately testable.

### B1 — `fix(desktop): serve a session read without an answering owner`

- `attached.py`: best-effort desktop-watch re-assert (D2.1); retained dial +
  late-sync install + `SYNC_LANDING_DEADLINE_S` (D2.2); `attach_existing` gains
  a budget and never raises (D1); the "socketed, unsynced" flag and the
  `is_cold`-vs-connected gate on redial (risk 2).
- `server/utils/desktop_sessions.py`: `acquire()` splits into read mode and
  control mode; the read mode calls the bounded attach and reports
  `cold`/`cold_reason`/`attaching`.
- `server/routes/desktop_sessions.py`: the read routes take read mode; the
  ladder's 503 sentence is replaced by `{code: "runtime_unreachable", ...}` for
  the control paths that still need it.
- `docs/DESKTOP_API.md`: the cold/`cold_reason` contract and the read-mode list.
- Tests: the probe of §1.1 becomes a unit test (a fake owner that welcomes and
  goes silent, asserting the read answers cold inside the budget); a late-sync
  test asserting the rollover frame is published; a test that a lost
  `desktop_watch` ack does not fail a bind. Two notes for the implementer:
  `test_settling_is_bounded_so_a_silent_owner_still_fails`
  (`tests/unit/session/test_frontend_sync_liveness.py:264`) pins today's
  behaviour for the **control** envelope and stays valid — add the read-mode
  sibling rather than rewriting it; and the runtime-side lease test
  (`tests/unit/session/runtime/test_server.py:1686`) must keep passing, because
  D2.1 changes only what the *viewer* does when the ack does not come.

### U1 — `fix(chat): a session that is not answering is not a session that is gone`

- `canonical-sessions-store.ts:414` (`isRefusedBeforeAdmission`): add the
  `runtime_retiring` code as a third disposition (`deferred`), distinct from
  the "text back in the box" case; retry the same id when the session next
  reports live.
- `canonical-sessions-store.ts:1597-1631` (the guard read): only `404` rolls the
  view back; a 503 keeps the target and retries behind the commit.
- `backend-error.ts:56-58`: classify by `detail.code` before status, so a
  session-scoped 503 is not "the Local Operator server is not running".
- `use-canonical-session.ts`: consume `admission.status` (`pending`) and the
  `admission.failed` frame; render `cold_reason` as a status row rather than
  the generic lost-connection notice.
- `mcp-failure.ts:129/222`: match the new code, keep the old prefix as a
  fallback.
- Evidence: rendered frames for cold / attaching / leaving / live, with the
  before/after pair, driven through the real app.

### B2 — `fix(session): a send that was not admitted is retried, not held`

- `routes/desktop_sessions.py`: `retryable: true` on the `runtime_retiring`
  409; the same code on the `/commands` `failed` receipt.
- `utils/desktop_sessions.py`: the bounded re-admit (`ADMISSION_DEFER_BUDGET_S`)
  with the same `command_id`, publishing `admission.failed` if it must give up.
- `process.py`: `BUILD_DRAIN_MAX_S` with a loud log and the existing exit rungs.
- Tests: a drain refusal followed by a re-admit onto the successor, asserting
  one durable row; a re-admit that gives up, asserting the typed refusal and the
  frame; a drain that never converges, asserting the bound ends it and the lease
  is released.

### Evidence to attach to each PR (house rule: on the PR, not in the repo)

B1: the probe output of §1.1 before and after, plus the route-level timing.
B2: an end-to-end run that reproduces the held draft first (a live draining
runtime refusing a `/messages` send) and shows the admitted row after the fix.
U1: before/after screenshots of the panel in cold, attaching, leaving and live
states, with the geometry numbers for anything that moves.

---

## 7. What would settle the remaining uncertainty

- **Why ~17 runtimes sit in `leaving`.** The record already carries the
  diagnostic, in fields a read-only pass can print without touching a session:
  `leaving` (the phrase), `pending` (`"approval"` / `"ask"` / `None` — a gate
  parked on a person), `busy`, `subagents_running`, `subagents_queued`
  (`runtime/types.py:605`, `:629`, `:645`, `:702`, `:706`; `jq` over
  `~/.local-operator/run/mobile/*.json`
  for the pids whose `leaving` is set), alongside the drain's log lines
  (`session runtime: … exiting cleanly`, `work arrived as the drain closed`).
  The three readings and what each means:
  - `pending` set ⇒ the gate is a **parked gate**, which is the one the operator
    can clear — the four `_retiring_refusal` sites are `prompt`
    (`serving.py:1868`), `receive_peer_message` (`:2291`) and the spool fallback
    (`:1439`, `:1456`), and the gate-answer op is none of them (it does need a
    *synced* facade: `attached.py:2317-2340` reads `pending_gate` off the
    facade's installed state). Answering it lets the drain converge on its own.
  - `busy` with no `pending` ⇒ genuine in-flight work (a turn, a job, a
    subagent); D7's bound is what ends that if it never will.
  - neither ⇒ the drain's gate is answering "busy" from a state no client can
    clear, which is the class the autorefresh design already recorded once as
    *"a session refused forever, holding the lease so that no successor could
    boot"* (`serving.py:1500-1504`). This is the state that most needs D7.
- **Whether a `pending` receipt is reachable in practice for a starved owner.**
  The probe in §1.1 reproduces the read half exactly; the write half needs the
  same fake owner plus a `/messages` call, which is a two-hour job for QA rather
  than an architect's read, and it is written as a B2 test above.
- **Whether the retained dial (D2.2) delays an idle reap** beyond the policy's
  own expectations. Measure residency with and without an unsynced client; the
  number decides whether `SYNC_LANDING_DEADLINE_S` can be relaxed upward or must
  tighten.
