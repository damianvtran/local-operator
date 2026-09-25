# Desktop stream gap storm — backend design

Status: design for review. No product code in this document; every line number was
verified in this worktree (`fix/desktop-stream-gap-storm`, `0133fe4e5` = 0.62.37),
where `git diff v0.62.35 HEAD` over
`local_operator/server/utils/desktop_sessions.py`,
`local_operator/server/routes/desktop_sessions.py`,
`local_operator/server/utils/desktop_feed.py`,
`local_operator/session/attached.py` and
`local_operator/server/utils/sse.py` is **empty** — so the running 0.62.35 daemon,
`origin/main` and this branch execute the same code on this path, and there is no
upstream fix to point at.

> **Baseline re-checked 2026-09-25 18:20 (task-8).** The app's daemon has since
> moved to **0.62.38** (`Registered this app's own daemon … pid 74822, v0.62.38`,
> log 17:49:43) and `origin/main` is now **`ae0667aa5` = 0.62.41**. The premise
> holds: across `v0.62.35 → v0.62.38`, and again across `0133fe4e5 → origin/main`,
> the only change touching any of these five files is an unrelated refactor of
> `_draft_model_spec` in the routes module (model-selection validation); the
> `/events` handler, `DesktopSessionBridge` and the feed are byte-identical. The
> daemon also restarts often — **10 registrations across the two log files**, each
> an app or `uv-tool` update — and each one resets the bridge pool, every epoch,
> every runtime and every socket at once (§11.1, §11.6).

Evidence base: the analyst's measured report (`~/lo-osc/findings/oscillation-backend.md`)
plus the code below, re-read here. Where the report and this document disagree, this
document's line numbers are this tree's.

---

## 1. The defect, in one paragraph

The desktop session stream is the bridge's only reference
(`server/routes/desktop_sessions.py:3269` acquires, `:3277` subscribes;
`desktop_sessions.py:4973`/`:4984` acquire/release). Every stream end therefore
drops `users` to 0, `release()` (`:1430-1435`) runs `_detach()` (`:1437-1522`),
which disposes the facade (`:1499-1501`) and removes the bridge's subscriptions,
and the next `acquire()` takes the cold path (`:1369-1427`) which **rotates the
epoch, zeroes the sequence and clears the replay** (`:1381-1384`). A client that
reconnects with the cursor it learned from the previous `open` can never match
that epoch, so `gap = epoch != self.epoch` (`:3109`) is true *by construction*:
every reopen is a mandatory full repaint. Separately, the stream itself is being
ended by the server: the subscriber is registered before the response headers
exist, but nothing reads its queue until `events` reaches `:3133` — the whole
`await self.snapshot()` at `:3113` happens in between — and `publish` treats a
queue that fills in that window as a broken subscriber and disconnects it
(`:1577-1580` → `_disconnect` `:1530-1536` → the `gap` frame and `return` at
`:3137-3139`). Those two mechanisms compose into a closed loop: the burst that a
cold engage publishes during the handshake overflows the 256-slot queue of the
subscriber that caused the cold engage, the server closes that stream, the close
guarantees the next open is a gap, and the gap's own snapshot is the next burst.

## 2. What the code does today, precisely

### 2.1 The handshake has no reader

1. `GET /v1/desktop/sessions/{id}/events` builds the pool context and awaits
   `context.__aenter__()` **before returning headers**
   (`routes/desktop_sessions.py:3265-3269`). That runs `bridge.acquire(read=True)`
   (`desktop_sessions.py:4973`), which — on a cold facade — builds the facade,
   installs the bridge's two subscriptions (`:1424-1427`) and starts the read
   attach (`:1264-1265`, bounded by `READ_FIRST_FRAME_GRACE_S = 0.05`,
   `session/attached.py:214`).
2. `bridge.subscribe(...)` registers the subscriber (`routes:3277`,
   `desktop_sessions.py:3078-3101`). Its queue is
   `asyncio.Queue(maxsize=REPLAY_COUNT)` = **256 frames** (`:985-987`, `:137`) and
   its byte budget is `REPLAY_BYTES` = **8 MiB** (`:138`).
3. Only when the ASGI layer starts iterating `stream()` (`routes:3305-3310`) does
   `events` (`:3103-3150`) run: it computes `cutoff` (`:3107`), `first`
   (`:3108`), `gap` (`:3109`) and the replay list (`:3110-3112`) — **then** awaits
   `snapshot()` (`:3113`), which awaits the shared attention refresh (bounded at
   `ATTENTION_SNAPSHOT_WAIT_S = 0.05`, `:235`) and, for a facade with a cursor or a
   cold one, `history()` (`:2381`) — a durable journal read.
4. The FIRST `queue.get()` is at `:3133`, after `open`, the replay and the
   snapshot have been yielded. Everything `publish` put on that queue in between
   was written with no consumer.
5. `publish` (`:1557-1583`) appends to `replay` and then, per subscriber, either
   enqueues or — if `sub.queue.full() or sub.queued_bytes + size > REPLAY_BYTES`
   (`:1577`) — calls `_disconnect` (`:1530-1536`): the queue is drained, a `None`
   sentinel is pushed, `overflow` is set and presence is revoked
   (`sub.visible = sub.can_notify = False`, `:1532`). `events` turns the sentinel
   into `{"type":"gap"}` and **returns** (`:3137-3139`), ending the stream.

The window is therefore not a race that has to be caught; it is the ordinary
shape of every open, and its length is the length of the snapshot build. The
analyst measured ended streams of ~1.5-2.5 s against a cycle of 1.8-4.7 s, and
256 frames at the measured 100+ frames/s is 2.5 s — the quantitative fit is the
strongest single indication that this path, rather than a client-side close, is
what ends these streams. (It is an indication, not an observation: see §8.1.)

### 2.2 The gap is unavoidable, and it is not only the epoch

Three terms decide `gap` at `:3109`:

| term | condition | today |
|---|---|---|
| `epoch != self.epoch` | cursor epoch vs bridge epoch | **always true on reopen** — the cold path rotates the epoch (`:1381`) |
| `after_seq < first - 1` | cursor older than the oldest RETAINED replay frame (`:3108`, trimmed at `:1569-1573`) | true whenever the outage produced more than 256 frames / 8 MiB |
| `after_seq > cutoff` | cursor ahead of the bridge | a consequence of the rotation |

A *fresh* subscription (no `epoch`) is `gap=true` by design and is harmless: the
client has no painted state to drop. The damage is on the *reopen*, where the
client holds painted canonical state, and the UI's answer to `gap=true` is to
drop it: `frontend: null`, `history: null`,
`dropLiveRecords(transcript)`, `status: "reconnecting"`
(`local-operator-ui`, `src/renderer/src/shared/hooks/use-canonical-session.ts:1113-1121`,
and the `gap`-frame arm at `:1080-1096`) — "loading" readouts until the snapshot
lands. That is the reported oscillation.

### 2.3 Why the cold path is reached every time, and what it costs

`_detach()` unsubscribes the bridge's two handlers (`:1496-1497`) and calls
`remote.dispose()` (`:1499-1501`), which closes the viewer's socket
(`session/attached.py:8903-8908`). The runtime's residency predicate counts a
desktop viewer only while its watch lease is fresh **and** it is visible or
notifiable (`session/runtime/server.py:4273-4299`, `DESKTOP_WATCH_LEASE_S` =
45 s, `session/runtime/types.py:72`); the socket is gone, so term 3 of
`_should_exit` (`session/runtime/process.py:1085`, `:1138`) falls and the runtime
exits after `DEFAULT_GRACE_S = 3.0` (`process.py:102`). The client's own retry
schedule is `[500, 1000, 2000, 4000, 8000, 8000] ms`
(`use-canonical-session.ts:302-303`), so it comes back to a **dead owner** and
pays a full cold engage — which is the burst that started the loop.

Two further consequences of the same detachment, both load-bearing for the
design below:

* **The bridge stops learning.** `FrontendStateStore.subscribe` returns a
  `FrontendSync` to the *caller* and pushes it to nobody
  (`session/frontend_state.py:5666-5719`); the bridge discards that object and
  keeps only `.unsubscribe` (`desktop_sessions.py:1426`). So while the bridge is
  detached, the deltas the store publishes are never recorded anywhere the client
  can reach. A cursor that survives a detach is **not** by itself a correct
  resumption.
* **The replay stops growing.** `publish` is the only writer of `replay` and
  `sequence`, and a detached bridge publishes nothing, so the 256-frame window
  does not advance during a detached interval — it simply cannot cover what the
  client missed, because nothing was captured.

---

## 3. D1 — stop the server killing a stream it caused to overflow

### 3.1 The invariant to protect

The comment at `publish:1578` states it: *"Never silently discard a semantic
event."* The handshake's own ordering contract is the other one, argued at
`:3125-3127`: replay receipts precede the authoritative snapshot, so a cumulative
record update cannot repaint newer snapshot text with old deltas.

Neither has to be weakened, because the handshake has a third property the
current code does not exploit: **a subscriber that has not yet received its
snapshot has nothing to be inconsistent with.** Every frame published before the
snapshot's own state read is, by construction, already reflected in that
snapshot; delivering it is redundant, and dropping it is not a silent discard of
a semantic event — it is the snapshot doing its job.

### 3.2 Recommendation: pre-open supersession, with the snapshot's watermark read last

Two edits, one contract.

**(a) `snapshot()` captures `state`/`seq` LAST.** Today `state = self.state()` is
read at `:2327`, `seq, epoch = self.sequence, self.epoch` at `:2328`, and only
then does the method await `history()` at `:2381` — so the frame's envelope `seq`
is a watermark that is *older* than the last await in its own construction. Move
the capture to immediately before the dict is composed at `:2382`, and compute
the page gate (`:2329`, `:2349`) from a cheap field read instead of a full state
clone:

```python
        # the gate keeps exactly today's predicate, from the store's own field
        remote = self.remote
        cursor = (
            remote.frontend_state.history_cursor if remote is not None else None
        )                                                       # was: state["snapshot"]["history_cursor"]
        ...
        if cursor or (remote is not None and remote.is_cold):
            history = await self.history()
        # THE WATERMARK IS READ WITH THE STATE IT DESCRIBES, and nothing awaits
        # between them: every frame with seq <= this value is already inside
        # `frontend`, so a subscriber may drop it without losing anything.
        state = self.state()
        seq, epoch = self.sequence, self.epoch
        return {...}
```

(`self.state()` already asserts `remote is not None` at `:2274`, so the guard
above reproduces today's reachable behaviour exactly; it is written out only
because the gate is now the earlier of the two reads.)

`self.remote.frontend_state` is the store object itself
(`session/attached.py:7607-7610`) and `history_cursor` is a real field on it
(`session/frontend_state.py:2787`, used at `:5709`), so the gate is unchanged.
No await may be introduced between `self.state()` and the `return`: that is the
property the next edit depends on, and it is the thing to pin with a test
(§6, T1-4).

**(b) A subscriber that has not opened is never disconnected for overflow.**
Add one field to `DesktopSubscription` (`:982-996`):

```python
    #: False until ``events`` has finished the OPEN handshake, i.e. until the
    #: snapshot (and the frames published after its watermark) have been handed
    #: to the transport. Before that the queue holds only frames the snapshot
    #: supersedes, so a full queue is eviction, not failure.
    opened: bool = False
```

`publish`'s overflow arm becomes two policies:

```python
        for sub in self.subscribers.values():
            if sub.overflow or sub.dwelling:
                continue
            if sub.queue.full() or sub.queued_bytes + size > REPLAY_BYTES:
                if sub.opened:
                    # A READER that is behind: backpressure, unchanged. Closing
                    # forces an authoritative gap snapshot on reconnect.
                    self._disconnect(sub)
                else:
                    # A subscriber that has not been given its snapshot yet.
                    # Every frame it holds is <= the watermark that snapshot
                    # will carry, so the oldest one is superseded, not lost.
                    self._evict_oldest(sub)
                    sub.queue.put_nowait((frame, size))
                    sub.queued_bytes += size
            else:
                sub.queue.put_nowait((frame, size))
                sub.queued_bytes += size
```

`_evict_oldest(sub)` pops from the head while the queue is full or the byte budget
is exceeded, subtracting each popped frame's size from `sub.queued_bytes`. It can
never see the `None` sentinel, because a disconnected sub has `overflow = True`
and is skipped.

`events` then finishes the handshake with one synchronous drain, and only then
sets `opened`:

```python
        snapshot = await self.snapshot()
        # ATOMIC AGAINST publish: snapshot()'s state capture is its last await, so
        # no frame can be published between that capture and this drain.
        disconnected = False
        pending: list[dict[str, Any]] = []
        while True:
            try:
                item = sub.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:                      # close()/_disconnect ran mid-build
                disconnected = True
                break
            frame, size = item
            sub.queued_bytes -= size
            if frame["seq"] > snapshot["seq"]:
                pending.append(frame)              # newer than the snapshot: must be delivered
            # else: already inside snapshot["payload"]["frontend"] — superseded.
        sub.opened = True
        yield {"type": "open", ...}                # unchanged shape, gap unchanged
        for frame in replay:
            yield frame
        yield snapshot
        for frame in pending:
            yield frame
        if disconnected:
            yield {"type": "gap", "session_id": self.session_id}
            return
        while True:                                # unchanged from :3131
            ...
```

Why the pre-open eviction is provably lossless: at the moment `snapshot()`
returns, the queue can only hold frames published before the state capture (no
await follows it), and the snapshot frame's `seq` is that capture's sequence. The
evicted frames are therefore all `seq <= snapshot["seq"]`, and each is reflected
in the snapshot's `frontend` — which is `self.state()` under the same clone the
`open`/`snapshot` handshake has always served (`:2273-2283`). The frames that
must *not* be dropped are those published after the capture, and none can exist
before the drain.

### 3.3 What this does not do, on purpose

* **`_disconnect` stays for an opened subscriber.** A subscriber that has its
  snapshot and still cannot take 256 frames / 8 MiB is genuinely behind; the
  existing relief valve is correct there, and it is now cheap (see D2).
* **The handshake's yield window is not protected.** While `events` is suspended
  yielding `open`/`snapshot`, a burst can still fill the queue and disconnect.
  This is accepted rather than patched: the frames then are *newer* than the
  snapshot, so dropping or evicting them would be a real loss, and the honest
  relief valve is the one that exists. In practice this window is quiet on the
  cold open that matters most — the engage's burst lands during the build
  (§2.1), not after the state capture.
* **`publish_to_subscription` (`:1585-1637`) is untouched.** Its frame is
  addressed by `subscription_id`, which a client can only know after it has
  received `open`; its overflow arm therefore only ever sees an open subscriber.
* **The machine feed is untouched** (`server/utils/desktop_feed.py:791-792`,
  `:716-723`): same bounds, same shape of window (subscribe at
  `routes:3368`, first drain after `_open_frame` at `desktop_feed.py:707`), but a
  poll-cadence publisher rather than a 100+ fps engage burst. The same
  supersession rule would apply if a defect is ever measured there; there is no
  measurement today, so nothing is added (see §7.2).

### 3.4 Rejected for D1

* **Subscribe after the snapshot** (move `bridge.subscribe()` into the
  generator). It moves the capacity refusal (`SUBSCRIBER_COUNT = 32`, `:3086`)
  and the move fence (`LegacySubscriberDuringMove`, `:3088-3098`) past the
  response start, destroying the property the route states at `:3265-3266` and
  `:3285-3287`: invalid identity/capacity must be JSON status, never a 200
  followed by a broken stream. It also loses the frames published during the
  build outright rather than dropping them under proof.
* **Coalesce on overflow (keep-latest per frame kind).** Coalescing needs a
  frame-kind taxonomy that the replay does not need and the client does not
  share, and it still discards events. Supersession is the same idea with a proof
  attached and no new taxonomy.
* **Bound what a cold engage publishes.** The engage's deltas are how the facade
  builds its state; suppressing them at the source breaks the reconstruction
  that the snapshot is derived from. The supersession rule bounds what is
  *delivered* without touching what is *learned*.
* **Grow the queue.** Unbounded memory on the hottest path in the app; and a
  bigger queue only moves the wall (22 605-frame epochs were measured).

---

## 4. D2 — the epoch, the release, and what a correct resumption means

### 4.1 The three candidates

* **(A) A reconnect dwell — hold the bridge across its last viewer's transport.**
* **(B) An epoch that survives a detach** (delete the rotation at `:1381-1384`
  and let `gap` be decided by the replay-window predicates alone).
* **(C) Keep the current contract** and rely on D1: the stream stops ending, so
  the mandatory gap stops firing.

### 4.2 Why (B) does not work on its own

Keeping the epoch across a detach looks like the smaller change, and it is not
sound. §2.3 is the reason: while detached, the bridge is unsubscribed
(`:1496-1497`), so it publishes nothing and its replay does not advance. On
re-acquire the facade re-dials — likely a **new** runtime, whose cold sync
publishes hundreds to thousands of deltas through the bridge — and the 256-frame
window trims `first` past the client's cursor, so `after_seq < first - 1` makes
`gap` true anyway (`:3109`). The measured reopens carry `after_seq` values from 0
to 22 605 against epochs that publish thousands of frames, so (B) would flip
`gap` to false only in the calm cases and leave the storm's cases unchanged —
while permanently widening what `gap=false` claims, because the detached interval
was never captured. It would also require the client to reconcile a snapshot it
currently ignores when it already has painted state (§5.3) — a coupling to the
parallel UI change that this design does not need.

### 4.3 Recommendation: a bounded reconnect dwell

The property to restore is not "the cursor survives a detach"; it is **"the
bridge keeps working while its viewer is away"**. If the bridge is still attached
and still subscribed, then:
`sequence` keeps advancing, `replay` keeps covering the interval, the facade is
not disposed, the runtime is not orphaned, the re-acquire is warm, and `gap=false`
becomes a *provable* statement — the client really is patched, frame by frame,
through the replay it missed — instead of a promise the bridge cannot keep.

#### 4.3.1 The constant

```python
#: How long a bridge outlives its last viewer's transport, in seconds.
#:
#: THE PROBLEM IT SOLVES: the desktop stream is the bridge's only reference
#: (`session()` :4973/:4984), so a transport break is a detach, and a detach is a
#: disposed runtime plus a rotated epoch — i.e. a full repaint on a reconnect the
#: client makes ~500 ms later. This window keeps the bridge working across that
#: break, which is what makes `gap=false` a statement the replay can honour.
#:
#: SIZED AGAINST THE CLIENT'S OWN BUDGET, not chosen: the renderer retries at
#: 500, 1000, 2000, 4000, 8000, 8000 ms (`local-operator-ui`
#: `use-canonical-session.ts:302-303`), and 20 s covers its first five attempts
#: (cumulatively 15.5 s) with margin while staying strictly inside `WATCH_TTL`
#: (45 s), so the dwell can never outlive the lease vocabulary it borrows.
#:
#: ZERO MEANS "NO DWELL", AND IT IS CONTRACT RATHER THAN A TEST HOOK. At zero the
#: subscription is popped and the last release detaches, in the same call, byte
#: for byte the old behaviour — no task is created, `dwelling` is never set, and
#: there is no window in which `_evictable` says no. Every existing test that
#: asserts "the last release detaches" is written against this value (§8.4).
RECONNECT_DWELL_S = 20.0

#: How often the dwell re-reads the module clock before it lets go. The dwell is
#: deadline-driven rather than a single `asyncio.sleep` precisely so its expiry is
#: observable from an injected clock: `tests/unit/server/test_desktop_sessions.py`
#: already replaces `module.time` with a `SimpleNamespace(monotonic=...)`
#: (`:172`), and that is the switch a test uses to move a dwell into the past
#: instead of waiting one out. Shrinking this constant with it keeps such a test
#: to a few milliseconds.
DWELL_TICK_S = 0.25
```

`WATCH_TTL` is at `:202`; put both beside it. **`RECONNECT_DWELL_S` must stay
strictly below `WATCH_TTL`** (45 s): one `update_desktop_watch` at the dwell's
start carries the presence assertion for the runtime's whole lease
(`_desktop_lease_live`, `session/runtime/server.py:4295-4299`), and a dwell
longer than the lease would let presence lapse mid-window. A test pins the
inequality (§6, T2-7).

#### 4.3.2 Who dwells

One new field on `DesktopSubscription`:

```python
    #: True once this subscription has lost its transport but is still holding
    #: the bridge open (see `_arm_dwell`). A dwelling subscription has no reader,
    #: so it is skipped by `publish`, keeps its place in `subscribers` (it is what
    #: `refresh_watch` asserts residency from), and is not evictable.
    dwelling: bool = False
    #: When the dwell lets go, on the module's own clock (`time.monotonic()`
    #: offset). Carried on the subscription rather than inside the task so the
    #: deadline is INSPECTABLE — a test asserts it and moves it, instead of
    #: waiting `RECONNECT_DWELL_S` out.
    dwell_until: float = 0.0
```

`events`'s `finally` (`:3144-3150`) changes from "pop the subscriber" to:

```python
        finally:
            if RECONNECT_DWELL_S > 0 and not sub.overflow and sub.id in self.subscribers:
                sub.dwelling = True
                self._arm_dwell(sub)
            else:
                self.subscribers.pop(sub.id, None)
            with CancelScope(shield=True), contextlib.suppress(ConnectionError, RuntimeError):
                await self.refresh_watch()
```

Three properties of that condition, each load-bearing:

* **`RECONNECT_DWELL_S > 0`** — zero is the documented off switch (§4.3.1).
* **`not sub.overflow`** — a subscription the BRIDGE revoked does not dwell.
  `_disconnect` (`:1530-1536`) is the bridge's own decision: the reader is behind
  (`:1577`) or the bridge is closing (`close()`, `:1524-1528`). Holding a runtime
  open for a reader the bridge just told to resync would invert the relief valve,
  and a *closing* bridge must never dwell at all. It also keeps
  `test_slow_subscriber_overflow_is_explicit_and_bounded`
  (`tests/unit/server/test_desktop_sessions.py:138-156`) meaningful: its final
  `assert not bridge.subscribers` stays true. What does dwell is a subscription
  that lost its transport while the bridge had nothing against it — a dropped
  socket, a cancelled ASGI scope, a relay watchdog.
* **`sub.id in self.subscribers`** — an already-removed subscription has nothing
  to hold.

Only a **stream** subscription dwells. A read route's `release()` still detaches
on the spot. That scoping is deliberate: the dwell exists to cover a *transport*
break, and applying it to every `session()` read would silently extend residency
across the whole read surface (and change every test that asserts a read detaches).

#### 4.3.3 The dwell task, and release

```python
    async def _arm_dwell(self, sub: DesktopSubscription) -> None:
        """Hold the bridge open across one viewer's lost transport. See RECONNECT_DWELL_S."""
        sub.dwell_until = time.monotonic() + RECONNECT_DWELL_S
        self._dwell_tasks[sub.id] = asyncio.create_task(self._end_dwell(sub))

    async def _end_dwell(self, sub: DesktopSubscription) -> None:
        try:
            # Deadline-driven, on the module clock, so an injected clock can move
            # the expiry instead of a test sleeping one out (DWELL_TICK_S).
            while True:
                remaining = sub.dwell_until - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(remaining, DWELL_TICK_S))
            async with self.lock:
                if self.subscribers.get(sub.id) is sub:
                    self.subscribers.pop(sub.id, None)
                if self.users == 0 and not self._dwelling:
                    await self._detach()
            with contextlib.suppress(ConnectionError, RuntimeError):
                await self.refresh_watch()
        finally:
            self._dwell_tasks.pop(sub.id, None)
```

`release()` (`:1430-1435`) gains the dwell guard:

```python
            if self.users == 0 and not self._dwelling:
                await self._detach()
```

where `self._dwelling` is `any(s.dwelling for s in self.subscribers.values())`
(and is False when the dwell is disabled). The same guard applies to `acquire()`'s
failure arm (`:1260-1263`).

**A re-acquire does not cancel the dwell.** `_ensure_facade()` finds
`self.remote is not None` and reuses the same facade, so `acquire` is warm by
construction; the dwell task then wakes, finds `users > 0`, drops the stale
subscription and does not detach. No cancellation race, no new cancellation path.
The dwell is inert rather than harmful while a viewer is back: a dwelling
subscription asserts nothing a live subscription does not already assert.

#### 4.3.4 What the dwell asserts to the runtime — residency, not attention

The runtime's term 3 is `attach_clients() > 0`
(`session/runtime/process.py:1138`, `session/runtime/server.py:4273-4293`), which
for a desktop connection requires a fresh lease **and**
`desktop_visible or desktop_can_notify` (`:4291`, `:4295-4299`). So a dwell that
merely keeps the socket open changes nothing — the runtime exits after its 3 s
drain and the dwell holds a dead facade. The dwell must therefore keep asserting
presence, and this is the one place the design has to choose what it claims:

**During the dwell the bridge asserts `visible=False, can_notify=True`.** It
keeps the runtime resident through the `can_notify` half, and it does **not** tell
the notification ladder that a person is reading a session they have walked away
from: `_visible_attach_surfaces` (`server.py:4301-4309`) stays empty, so rung 1
is untouched and a turn that completes during the dwell still notifies. That
matters here more than anywhere else, because "a turn finished while my window was
reconnecting" is exactly the event a user must not lose.

Implementation, in `_live_leases` (`:2447-2456`) and `refresh_watch`
(`:2474-2513`). `_live_leases` gains a flag, because its two callers want
different answers from the same list:

```python
    def _live_leases(self, *, include_dwelling: bool = False) -> list[DesktopSubscription]:
        """The subscriptions holding a LIVE lease. Caller holds ``watch_lock``.

        A DWELLING SUBSCRIPTION IS EXCLUDED BY DEFAULT, and the flag is the whole
        reason this is a question rather than a list. ``refresh_watch`` asks "who
        is this bridge still asserting to the owner", and during a dwell the
        answer must include the returning viewer, or the runtime exits under the
        window the dwell exists to cover. ``in_flight_reason`` (:3710-3717) asks
        "may this daemon leave", and a viewer that has ALREADY lost its transport
        may not pin a build update for RECONNECT_DWELL_S: the successor daemon
        serves the reconnect, and the announcement is what the client is already
        reacting to. Counting the dwell there would make every update wait up to
        the dwell out, per session — and `server/retire.py`'s drain is exactly
        what a dwell would then be holding shut.
        """
        now = time.monotonic()
        return [
            s for s in self.subscribers.values()
            if not s.overflow and (s.expires > now or (include_dwelling and s.dwelling))
        ]
```
```python
            live = self._live_leases(include_dwelling=True)     # refresh_watch only
            visible = any(s.visible for s in live if not s.dwelling)
            can_notify = any(s.can_notify for s in live if not s.dwelling) or any(
                s.dwelling for s in live
            )
```

`in_flight_reason` (`:3710-3717`) is otherwise **unchanged**: the `users` term
already excludes a dwelling bridge (its last release dropped `users` to 0), and
the `warm_task` term is never armed during a dwell because `visible` is False and
`refresh_watch` returns before `_arm_lease_warm` (`:2503-2513`). That matters for
`tests/unit/server/test_serve_retire.py:874-936`, which cancels the app's own
relay and then requires the drain to empty within 2 s — pinning
`tests/unit/server/test_serve_retire.py:518` and `:526-559` (the `users` and
lease terms) as the tests that must keep passing unchanged.

One `update_desktop_watch` at the dwell's first `refresh_watch` is enough: the
runtime's lease is 45 s (`DESKTOP_WATCH_LEASE_S`), well past the 20 s dwell. The
dwell's expiry asserts the truth again (`visible=False`, nothing notifying), term
3 falls, and the runtime exits on its ordinary 3 s drain.

#### 4.3.5 Eviction, teardown and the pool

* `_evictable` (`:4598-4608`) gains `and not bridge.dwelling` — without it, a
  dwelling bridge has `users == 0`, so `BRIDGE_COUNT` pressure (`:4958-4963`)
  could delete it from the pool while the dwell still holds a facade, a presence
  assertion and a runtime that nothing can reach. That is the same class of leak
  `_handouts` (`:4586-4596`) exists to prevent, and it must be prevented the same
  way.
* `close()` (`:1524-1528`) and `_detach()` (`:1437`) cancel every dwell task,
  pop every dwelling subscription and clear `_dwell_tasks` before disposing — a
  forgotten bridge must leave no subscription asserting presence on its behalf;
  `forget()` (`:4986-5018`, on session delete) and `DesktopSessions.close()`
  (`:5020-5022`, pool shutdown) already route through them.
* `publish` skips `sub.dwelling` (see §3.2b). Without that skip the dwell is
  self-defeating: its queue has no reader, so it would overflow, `_disconnect`
  would revoke presence (`:1532`) and the runtime would die inside the window the
  dwell exists to cover.

### 4.3.6 The dwell's observable contract

Everything below is assertable from a test without waiting 20 s.

| state | how it is entered | observable |
|---|---|---|
| armed | `events`'s generator finishes — by end of stream, `aclose()`, or a cancelled ASGI scope — while `RECONNECT_DWELL_S > 0` and the sub is still registered | `sub.dwelling is True`, `sub` still in `bridge.subscribers`, `sub.dwell_until > 0`, one task in `bridge._dwell_tasks[sub.id]` |
| held | while any sub is dwelling | `bridge.dwelling is True`; `pool._evictable(bridge) is False`; `_live_leases()` (default) is empty while `_live_leases(include_dwelling=True)` contains the sub; `refresh_watch` writes `visible=False, can_notify=True` |
| released by the viewer's return | a new `acquire()` + `subscribe()` before the deadline | `bridge.remote` is the *same object* (never re-created, no epoch rotation), `sub.dwelling` stays True until its own deadline, and `release()` at `users == 0` does **not** detach while it holds |
| expired | `sub.dwell_until` passes | sub popped, `bridge.dwelling` False, `_dwell_tasks` empty, `refresh_watch` re-asserts `visible=False, can_notify=False`, and `_detach()` runs iff `users == 0` |
| torn down | `close()`, `forget()`, pool shutdown | dwell tasks cancelled, dwelling subs popped, then the ordinary `_detach()` |
| disabled | `RECONNECT_DWELL_S = 0` | `dwelling` is never set, no task is created, and the last `release()` detaches in the same call — today's behaviour exactly |

**Driving the expiry from an injected clock, not a sleep.** The deadline lives on
the subscription and is read through the module's `time`, which is the file that
tests already patch (`tests/unit/server/test_desktop_sessions.py:172` replaces
`module.time` with a `SimpleNamespace(monotonic=...)`):

```python
now = [1000.0]
monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
monkeypatch.setattr(module, "DWELL_TICK_S", 0.01)
...                      # run the stream to its end
assert sub.dwelling and sub.dwell_until == 1000.0 + module.RECONNECT_DWELL_S
now[0] += module.RECONNECT_DWELL_S + 1          # move the deadline into the past
await _until(lambda: not bridge.subscribers, why="the dwell did not let go")
```

`_until` is this suite's own load-tolerant poller
(`tests/unit/server/test_desktop_sessions.py:46-62`) and its docstring already
states the convention this relies on: it reads the *real* clock while the module's
`time` is patched. The fallback recipe — patch `RECONNECT_DWELL_S` to a few
milliseconds and await the real expiry — is the same test with no fake clock.

### 4.4 What a "correct resumption" then means

With the dwell in place, a reopen inside the window is not a resumption *of a
broken connection* at all — it is the same bridge, the same runtime, the same
epoch, and a replay that has been recording the whole time. The client's cursor
is honoured and `gap=false` is true because the frames between the cursor and the
head are all in `replay`, in order, and are delivered before the snapshot
(`:3128-3129`). The two-way statement is:

* **`gap=false`** — "every frame the bridge published after your cursor is in
  the replay that follows, and the snapshot that follows it is authoritative for
  the state." The client may keep painted state (nothing was lost) and will still
  be correct if it repaints from the snapshot.
* **`gap=true`** — "the replay cannot cover your cursor." That is now the
  *honest* case and only fires when it should: a dwell that expired, an outage
  that produced more than 256 frames / 8 MiB, a cursor from a previous epoch, a
  genuinely new bridge (pool eviction, daemon restart), or a fresh attach with no
  epoch at all.

This is the answer to D2's "either refute the comment or keep it". The comment at
`:1379-1380` ("A detached interval has no receipt feed. A new epoch makes that gap
explicit even when the runtime itself never died") is **kept, and made true**:
after the dwell expires there really is no receipt feed, and the epoch really does
rotate. What changes is that a 500 ms transport blip no longer creates a detached
interval — so the sentence stops describing the common case and starts describing
the exceptional one.

---

## 5. D3 — what a client sees, exactly

Frame sequences below are the wire as `events` composes it. `open`'s `seq` is
`cutoff` (`:3107`), the snapshot's `seq` is the watermark (§3.2a), and every
frame carries `{session_id, epoch, seq, type, payload}`.

### 5.1 Case matrix

| # | case | frame sequence | UI today | honest? |
|---|---|---|---|---|
| 1 | fresh attach (no cursor) | `open{gap:true}` → replay(none) → `snapshot` → live | nothing is painted yet; the `gap` arm is a no-op | yes — unchanged |
| 2 | **reconnect inside the dwell** | `open{gap:false, seq:S}` → replayed frames `S+1..M` → `snapshot{seq:M}` → live | keeps painted state, applies the replayed deltas, ignores the snapshot (see §5.3) | **yes** — the replay is complete and ordered, so ignoring the snapshot loses nothing |
| 3 | reconnect after the replay window was exceeded | `open{gap:true, seq:M}` → `snapshot{seq:M}` → live | clears live projections, repaints from the snapshot | yes — unchanged, and now rare |
| 4 | cold open under burst | `open{gap:true}` → `snapshot` → frames newer than the watermark | as case 1 | yes — the stream no longer ends |
| 5 | real detach (dwell expired, delete, restart) | epoch rotated; next open is `gap:true` | full repaint | yes — the frame space really did break |
| 6 | subscriber slower than 256 frames *after* its snapshot | `… frames …` → `gap` → end | clears, repaints | yes — unchanged backpressure |

The change the user sees is case 2 moving from "gap + full repaint" to "no gap +
a handful of replayed frames", and case 4 ceasing to exist as a stream end.

### 5.2 Where the frames in the pre-open window go (D1's wire view)

Nothing new appears on the wire. Frames published between `subscribe` and the
snapshot's watermark are dropped instead of queued; frames published after the
watermark are delivered after the snapshot, exactly as today's drain loop delivers
them (`:3140-3143`). The client's cursor moves `open.seq` → (replay) →
`snapshot.seq` → live — monotone in every case, because the watermark is now read
where the state is.

### 5.3 The UI, and the fix landing in parallel

The UI's existing handling is: `gap` frame → drop `frontend`/`history`/live
transcript rows, `status: "reconnecting"` (`use-canonical-session.ts:1080-1096`);
`open{gap:true}` → drop `frontend`/`history`/live rows, `snapshotted = false`
(`:1113-1121`); the `snapshot` frame is applied **only while `snapshotted` is
false** (`:1075`, `:1124-1205`), otherwise it falls through unhandled.

That last detail is why this design deliberately does not depend on the UI change:
under case 2 the client keeps painted state, so `snapshotted` stays true and the
snapshot frame is ignored — which is *safe only because the replay delivered
every frame the client missed*. That is the property the dwell buys. The parallel
UI fix ("holds painted readings across a gap") stays honest under this design for
the same reason, and becomes strictly safer: if it also applies the snapshot as a
reconciliation on a gap-free reopen, that is idempotent (durable-wins on
`history`, wholesale on `frontend`) and cannot regress a client that the replay
already patched.

The one thing the UI change must **not** do is keep painted readings on a
`gap=true` *without* applying the snapshot: with a real gap the replay is
incomplete by definition, and the snapshot is the only reconciliation. That is
unchanged from today's contract, and worth stating in the UI PR.

---

## 6. D4 — the test plan

Fixtures to reuse (all in `tests/unit/server/test_desktop_sessions.py`): the
module-level `_until` poller (`:46-62`), the pool construction pattern
`pool = DesktopSessions(tmp_path)` + `async with pool.session(sid) as bridge`
(`:91-104`), and the app fixture `move_api` (`:4173-4204`) for anything that needs
the route. `bridge.events(...)` is a plain async generator and can be driven with
`anext` — the "algebra" block at `:90-156` is exactly that.

### T1 — pre-open supersession (`tests/unit/server/test_desktop_sessions.py`)

1. `test_a_subscriber_that_has_not_opened_survives_the_burst` — subscribe, publish
   `REPLAY_COUNT + 100` frames **before the first `anext`**, then drive `events`:
   assert `not sub.overflow`, assert `sub.queue` drained, and assert that every
   frame yielded after the snapshot has `seq > snapshot["seq"]` while no frame
   with `seq <= snapshot["seq"]` is yielded at all.
2. `test_the_pre_open_drop_is_exactly_the_watermark` — patch `snapshot()` to a
   wrapper that records `bridge.sequence` around its own body, publish a frame
   from inside the `history()` read, and assert the dropped set is exactly the
   frames with `seq <= snapshot["seq"]` and that the frame published during the
   read is delivered (it is newer than the watermark).
3. `test_a_slow_reader_after_the_handshake_is_still_disconnected` — the existing
   `test_slow_subscriber_overflow_is_explicit_and_bounded` (`:138-156`) with
   `sub.opened` set by one `anext`, asserting `overflow`, the `gap` frame and
   `StopAsyncIteration`. **Kept as-is in spirit; the burst variant above is its
   twin** and the pair is what pins the boundary.
4. `test_nothing_awaits_between_the_snapshot_state_and_its_watermark` — the guard
   for §3.2a: monkeypatch `history()` to publish a frame, and assert the snapshot
   frame's `seq` is greater than that frame's seq (i.e. the capture is last).
   A regression that moves the capture back above the read fails this.

### T2 — the reconnect dwell (`tests/unit/server/test_desktop_stream_dwell.py`, new)

1. `test_a_reconnect_inside_the_dwell_keeps_the_epoch_and_replays_the_outage` —
   open a stream, `aclose()` it, publish N frames, re-subscribe with the *same*
   `epoch` and the client's `after_seq`: assert `open.payload.gap is False`, that
   the N frames are replayed in order before the snapshot, and that the facade was
   never re-created (`bridge.remote` is the same object).
2. `test_the_dwell_expiry_detaches_and_rotates_the_epoch` — the injected-clock
   recipe of §4.3.6: assert `sub.dwelling`/`sub.dwell_until` at the arm, move the
   clock past the deadline, then assert the sub is gone, the epoch changed and
   `bridge.remote is None` — and that none of that is true *before* the deadline.
3. `test_zero_disables_the_dwell` — with `RECONNECT_DWELL_S = 0`: `dwelling` is
   never set, `_dwell_tasks` stays empty, and the last release detaches in the
   same call. This is the switch §8.4's existing tests need, and this test is
   what makes it a contract rather than a convention.
4. `test_a_dwelling_bridge_is_not_evictable` — `pool._evictable(bridge)` is False
   while dwelling; `close()` cancels the dwell, pops the sub and detaches.
5. `test_the_dwell_holds_residency_without_claiming_attention` — the
   `update_desktop_watch` spy (the pattern at `:160-175`) sees
   `visible=False, can_notify=True` during the dwell, and the last pre-dwell
   `visible=True` is not what is held.
6. `test_an_outage_longer_than_the_replay_window_still_gaps` — `REPLAY_COUNT`
   patched to 2, three frames published during the dwell, reconnect with the old
   cursor: `gap is True`. The honest fallback is part of the contract.
7. `test_a_dwelling_subscription_does_not_hold_the_daemon` — the dwell's own
   regression guard for §4.3.4: end a stream, then assert
   `pool.in_flight_reason()` does not name that session (with `_live_leases()`'s
   default), while `refresh_watch`'s aggregate still counts the dwelling sub.
   `tests/unit/server/test_serve_retire.py:874-936` is the end-to-end form of
   this and must keep passing unchanged.
8. `test_the_dwell_is_inside_the_watch_lease` — `0 < RECONNECT_DWELL_S <
   WATCH_TTL`, because one presence assertion must carry the whole window
   (§4.3.1). A constant-only test, and the reason a future raise of the dwell
   cannot silently outlive the lease.

### T3 — the handshake end to end (`tests/unit/server/test_desktop_stream_handshake.py`, new)

The D4 fixture the brief asks for, at route level: the `move_api` app
(`test_desktop_sessions.py:4173-4204`) with the engage stubbed, a `snapshot`
patched to publish a burst and `await asyncio.sleep`, and a subscriber driven
through the **route's** ordering (`acquire` before headers, `subscribe`, then the
generator). Assert `sub.overflow is False`, that the response status is 200 and
that the stream is still live after the snapshot; and the negative twin — the
same fixture with the burst moved to *after* the handshake, which must still end
in `gap` and close. `ASGITransport` buffers a response until the app returns
(`test_desktop_sessions.py:5903-5909`), so drive `bridge.events` directly and use
the route only for the two hand-offs on either side of the response.

### T4 — neighbours (`tests/unit/server/test_desktop_feed.py`)

`test_the_feed_overflow_policy_is_unchanged` — the machine feed still emits
`gap{reason:"overflow"}` and closes on its own bound
(`desktop_feed.py:791-792`, `:716-723`), i.e. this change did not reach it. Plus,
in the dwell file, an assertion that a *read*-only release still detaches on the
spot (the dwell is scoped to streams).

---

## 7. D5 — blast radius

### 7.1 Touched

Single product file: `local_operator/server/utils/desktop_sessions.py`.

| range | what changes |
|---|---|
| `:137-140` | constants block gains nothing; `RECONNECT_DWELL_S` goes beside `WATCH_TTL` at `:202` |
| `:982-996` | `DesktopSubscription` gains `opened`, `dwelling` |
| `:1220-1296` | `acquire` — failure arm routes through the dwell guard |
| `:1369-1427` | cold path unchanged (the epoch rotation stays; it is now reached only by a real detach) |
| `:1430-1436` | `release` — `users == 0 and not self._dwelling` |
| `:1437-1522` | `_detach` cancels dwell tasks; `close()` clears them |
| `:1530-1583` | `_disconnect` unchanged; `publish` gains the two-policy overflow arm and the `dwelling` skip; new `_evict_oldest` |
| `:2303-2392` | `snapshot` — state/watermark captured last; gate from `frontend_state.history_cursor` |
| `:2437-2456`, `:2458-2513` | `_live_leases(*, include_dwelling=False)` and `refresh_watch(include_dwelling=True)` — the dwelling arm |
| `:3710-3717` | `in_flight_reason` — **unchanged**, and pinned as such: the retirement drain must not count a dwelling subscription (§4.3.4, §8.4) |
| `:3057-3101` | no `subscribe` change needed; `_arm_dwell`/`_end_dwell` live beside it |
| `:3103-3150` | `events` — the drain, `opened`, the dwell in `finally` |
| `:4586-4608` | `_evictable` gains `and not bridge.dwelling` |

Docs: `docs/DESKTOP_API.md` — applied in this branch. "Stream ordering and
lifecycle" (`:1177-1236`, items 1-5 renumbered) now states what `gap:false`
claims, that a pre-snapshot frame is superseded rather than delivered, and that
the snapshot's `seq` is the watermark; the overflow paragraph (`:1376-1400`) now
says an overflow is only reachable by an opened subscriber and carries the dwell's
contract. The implementing PR inherits both.

### 7.2 Neighbours, and why they are safe

* **Machine feed** (`server/utils/desktop_feed.py`, route `:3365-3368`): same
  256/8 MiB bounds, same subscribe-then-drain shape, different publisher (a poll
  cadence, not a 100+ fps engage). Untouched; T4 pins it.
* **Mobile attach** (`local_operator/mobile/attach_client.py:1230`, `:1239`): a
  different transport with its own sequence check; nothing in this design changes
  its frames. It is affected only indirectly — the dwell can keep a runtime
  resident ~20 s longer — and that is the same term the 45 s watch lease already
  grants.
* **TUI attach** (`kind == "attach"`, `surface != "desktop"`): counts
  unconditionally for term 3 (`server.py:4288-4292`); untouched.
* **`_handouts` / eviction**: see §4.3.5. The interplay is the one place where a
  dwell could leak, and the `_evictable` term closes it.
* **`WATCH_TTL` presence**: the dwell borrows the lease vocabulary and is strictly
  inside it (20 s < 45 s). `_expire_watches` (`:3057-3076`) is not modified; a
  dwelling subscription keeps its `expires` fresh through `refresh_watch`'s new
  arm rather than through a re-beat that will never come.
* **The daemon's retirement drain** (`server/retire.py`, `in_flight_reason`
  `:3710-3717`): the dwell's presence hold is deliberately *not* a term in it, or
  every announced build update would wait up to the dwell out, per session, on a
  viewer that has already lost its transport — and the successor daemon serves
  the reconnect anyway. `tests/unit/server/test_serve_retire.py:874-936` is the
  end-to-end guard, and §4.3.4 carries the flag that keeps it green.
* **Mobile/desktop notification ladder**: rung 1 is untouched by construction
  (§4.3.4). Worth one QA cell: a turn completing during a dwell still notifies.
* **`docs/DESKTOP_API.md`**: `gap=false` on a reopen is new, and the doc is the
  contract of record for both clients.

---

## 8. Risks I could not close

### 8.1 Which side issues the final close is still unobserved

The analyst's §6.1 stands: the uvicorn access line is emitted at response
*start*, so the logs carry no close time, and mechanism A is identified by
elimination. This design removes the *cause* of the only code-visible server-side
closer (the un-drained window) and makes any remaining close cheap and gap-free,
but it does not, on its own, prove which side closed.

**The fix is its own discriminator, and that is the reading to take — no shipped
probe is proposed for this, in either repository.** The two mechanisms are
independent, so the post-landing observation decides the question without new
instrumentation:

* **Streams stop ending.** Then the subscriber queue was the closer, mechanism A
  is confirmed, and the cycle is closed at its cause.
* **Streams still end, but the reopen is `gap=false` and no runtime is
  re-engaged.** Then D2 is doing its job: the user sees no repaint and pays no
  spawn, and the remaining closer is a *separate* finding with its own
  measurement — not a reason to reopen this design.
* **Streams still end and the reopen is `gap=true`.** Then the outage exceeded
  the replay window or the bridge really was detached, and the new code's own
  state (`bridge.dwelling`, `epoch`, `first`, `after_seq`) plus the existing log
  line at `desktop_sessions.py:3712` name which term fired.

The measurement is therefore the existing one, re-taken after rollout: the
per-session stream-start intervals from the daemon's access log (the analyst's
§2 method — 289 starts, 1.8-4.7 s cycles, and the epoch/`after_seq` chain that
proves a rotation) against the same window after the change. What would settle
the residual cases *definitively* is a client-side frame trace, and that is
recorded here as available rather than requested: the relay's `data:` branch
(`local-operator-ui` `src/main/desktop-stream.ts` ~`:283`) is the one place that
sees frame type, size and the `gap` flag together. It should be reached for only
if the re-measurement above lands in the third row.

### 8.2 A single frame over 8 MiB still cannot be delivered

`publish`'s byte test (`:1577`) disconnects an opened subscriber for one oversized
frame, the replay trims it away (`:1569-1573`), and the relay aborts its buffer
above the same 8 MiB (`desktop-stream.ts:269-276`). The **snapshot frame has no
byte budget at all**: `snapshot()` composes a full `frontend` state plus a ≤100-row
history page (`:2382-2392`) and yields it directly, bypassing the subscriber
queue. If a session's snapshot ever exceeds 8 MiB, that session is permanently
unopenable in the app, and neither D1 nor D2 helps.

**MEASURED ON THE OPERATOR'S REAL STORE, read-only (2026-09-25). The risk is not
realized: the worst snapshot found is ~3.9 MB, about 2.1x inside the 8 MiB frame
cap.**

* The `history` term is the only unbounded one, and a page is at most its file, so
  the sweep is exhaustive over just the journals that could reach the cap: **40 of
  8 437** `transcript.jsonl` files exceed 8 MiB (1 530 exceed 1 MiB). Across all
  40, the last-100-line page measures **88 KB - 2 833 947 B** (widest:
  `74f2d12c7b66`, an 8.21 MB journal) and the largest single row is **836 065 B**.
  The 257 MB journal (`bda7b76d34e0`) — the one this module's own comments cite —
  has a **163 196 B** page: page size tracks the *recent* rows, not the journal.
* The `frontend` term is bounded by construction rather than by luck:
  `USAGE_COMPONENT_CAP = 200` (`session/frontend_state.py:156`, ~55 KB),
  `LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS = 56 000` (`:438`) with
  `LIVE_EVENT_END_ROWS_MAX = 100` (`:500`), and the checkpoint strips
  `live_events` and job trajectories (`:143`). The measured proxy — the largest
  `frontend_state_checkpoint_v1` row in the tail of 7 candidate journals — is
  **≤ 1 095 868 B**.
* The eight sessions the app actually churned on (the analyst's §2 table) measure
  pages of **186 KB - 1 122 632 B** each.
* Nothing *enforces* the bound, and that is the part worth watching: the page is
  capped at 100 rows, not at a byte count, and rows of 836 KB exist, so 100
  consecutive such rows would be ~83 MB. What would justify a follow-up is a
  family of sessions whose *last hundred* rows are each hundreds of KB — the
  probe is `tail -n 100 transcript.jsonl | wc -c`, and it is cheap enough to run
  against the store whenever this area is touched again.

**The sharper sub-case is not the snapshot but the resolved image.** The desktop
bridge is the one surface that carries image bytes INLINE: the runtime
externalizes any block at or above `_ATTACHMENT_FLOOR_BYTES = 1024`
(`session/transcript.py:397`, `:333-364`) into
`~/.local-operator/attachments`, and the desktop viewer resolves the digest back
to base64 in its wire callback, ahead of validation and therefore ahead of the
bridge's `model_dump` (`session/attached.py:5781-5788`; the shape is pinned by
`tests/unit/server/test_desktop_sessions.py:1809`). `AttachmentStore.put` has **no
size cap**. Measured: across **31 592** stored attachments the largest is
**1 046 884 B** (~1.40 MB as base64), so one image leaves about **6x** headroom
against the frame cap: it would take roughly six in a single frame to cross it —
not reachable in this store today, and the ingest ladder (`IMAGE_MAX_BYTES = 1 MiB`,
`local_operator/imaging.py:229`) is what keeps it that way for images this app
creates. A non-resized blob arriving through some other path would not be bounded
by that ladder, which is the case a byte budget would have to cover.

**Disposition: recorded, not fixed.** Neither measurement justifies a change in
this PR, and fixing the second would be a frame-budget change to the richest
payload on the desktop surface — out of scope for a stream-stability fix.

### 8.3 The dwell's residency cost is unmeasured on this host

One idle runtime is ~82 MB by this module's own measurement (`:2600`). The dwell
holds it for up to 20 s after the last viewer leaves, for every conversation left
in that window. The module's aggregate argument ("one lease-bearing view per
focused window", `:2593-2606`) bounds the steady state, not the churn.

**What to watch after rollout**: the count and RSS of session-runtime children
while navigating several conversations inside 20 s, and the engage count in
`backend-service.log` (the analyst's method: the daemon prints its generation
tree). If the aggregate is a problem, the constant is the lever and it is one
number.

### 8.4 Test-surface cost of the dwell, verified rather than estimated

The dwell holds a subscription past the end of its stream, so any existing test
that ends a stream and *then* asserts the bridge detached has to turn it off with
`monkeypatch.setattr(module, "RECONNECT_DWELL_S", 0)` (§4.3.1). Every test that
drives a stream was enumerated (`grep -rn "\.events(" tests/` — 16 sites in 6
files, plus the `/events` route in `tests/unit/server/test_serve_retire.py` and
the e2e files) and each was read for a post-stream assertion:

| site | verdict |
|---|---|
| `tests/unit/server/test_desktop_sessions.py:103-104` (`test_replay_receipts_precede_snapshot_even_when_snapshot_is_newer`) | **breaks** — `stream.aclose()` inside the block, `assert bridge.remote is None and bridge.users == 0` after it. Needs `RECONNECT_DWELL_S = 0`. **The only existing assertion that breaks.** |
| `tests/unit/server/test_desktop_sessions.py:123-134` (`test_reopening_after_last_detach_invalidates_receipt_epoch`) | **does not break**: the detach it asserts happens at the end of a block in which no stream was ever opened (`subscribe`/`events` only appear in the *second* block, `:132-134`, after the assertion). An interim report of mine claimed otherwise; this is the corrected reading. |
| `tests/unit/server/test_desktop_sessions.py:138-156` (slow subscriber) | **does not break** — because the dwell declines an `overflow`ed subscription (§4.3.2); `assert not bridge.subscribers` stays true |
| `tests/unit/server/test_desktop_sessions.py:232-259` (unconsumed generator) | **does not break** — the generator is never iterated, so its `finally` never runs and no dwell is armed; only the background task's `release_once` runs |
| `tests/unit/server/test_desktop_sessions.py:98, 116, 145` | no post-block assertion |
| `tests/unit/server/test_desktop_admission_ack.py:129, 259` | no detach/epoch assertion; the harness cancels its pump and releases the pool hold |
| `tests/unit/server/test_desktop_read_without_owner.py:343, 371, 860` | `remote is None or remote.cold_reason is None` runs while the block still holds the bridge; the dwell does not cold or dispose a held facade |
| `tests/unit/server/test_desktop_attention.py:428`, `test_desktop_mcp_catalog_routes.py:89` | no stream in either test |
| `tests/unit/server/test_serve_retire.py:874-936` | **does not break, and depends on §4.3.4's flag**: the relay is cancelled, the dwell arms, and the drain must still empty within 2 s — which holds only because `in_flight_reason` reads `_live_leases()` *without* dwelling subscriptions |
| `tests/e2e/*`, `tests/unit/server/test_desktop_aside_route.py`, `tests/unit/server/test_desktop_feed.py` | no stream-detach assertions (`grep` for `remote is None` / `epoch !=` / `users == 0` is empty in them) |

One residual, not a failure: a test that ends a stream and never closes its pool
leaves the dwell task pending until the loop tears down (which cancels it, running
`_end_dwell`'s `finally` only). That is why the deadline lives on the
subscription and the task holds no cleanup of its own — and why the pool fixtures
that *do* close (`move_api`, `tests/unit/server/test_desktop_sessions.py:4202-4203`)
exercise the cancellation path for free.

The wider point stands: `0` must mean "detach immediately, in the same call",
because that is what makes this a one-line patch at a single existing site rather
than a rewrite of the suite's teardown assumptions.

### 8.5 `snapshot()`'s reorder is a small semantic change with a wide read surface

Moving the state capture below `history()` makes the snapshot's `frontend` fresher
by the length of the journal read, and its envelope `seq` larger. Both are what
the frame claims to be, so the change is a correction — but the snapshot frame is
read by both clients and by `frontend.replace`'s cursor ordering
(`use-canonical-session.ts:1276-1305`, `acceptFrontendReplace(priorCursor, …)`),
which is ordered against the receipt cursor this frame advances. T1-4 pins the
watermark; the `frontend.replace` ordering is worth an explicit QA cell on the
real flow (a move performed on a session whose stream just reconnected).

---

## 9. Ordered task breakdown

Each item names the one test file that carries it.

1. **`publish`/`events` pre-open supersession** — `DesktopSubscription.opened`,
   `_evict_oldest`, the two-policy overflow arm, the handshake drain, and
   `snapshot()`'s last-position watermark. Test file:
   `tests/unit/server/test_desktop_sessions.py` (T1-1..T1-4; T1-3 is the existing
   `test_slow_subscriber_overflow_is_explicit_and_bounded`, `:138-156`, whose
   meaning is narrowed to post-open).
2. **The reconnect dwell** — `RECONNECT_DWELL_S` / `DWELL_TICK_S`,
   `dwelling` / `dwell_until`, `_arm_dwell` / `_end_dwell`, the `release` guard,
   the `publish` skip, `_live_leases(*, include_dwelling)` and `refresh_watch`,
   `_evictable`, dwell cancellation in `_detach` / `close`, and the `0`-means-off
   contract. Test file: `tests/unit/server/test_desktop_stream_dwell.py` (new;
   T2-1..T2-8). This item also carries the **one** existing test that must gain
   `RECONNECT_DWELL_S = 0`: `tests/unit/server/test_desktop_sessions.py:103-104`
   (§8.4), and must leave `tests/unit/server/test_serve_retire.py:874-936` green
   without editing it.
3. **Route-level handshake regression** — the injected-snapshot-delay fixture and
   its post-handshake negative twin, proving the route's acquire-before-headers
   ordering is unchanged. Test file:
   `tests/unit/server/test_desktop_stream_handshake.py` (new).
4. **Neighbour guard and contract docs** — the feed's own overflow policy pinned,
   the read-route detach pinned, and the `docs/DESKTOP_API.md` edits (already in
   this branch at `:1177-1236` and `:1376-1400`) reviewed against the code that
   lands. Test file: `tests/unit/server/test_desktop_feed.py`.

Then, unchanged and non-negotiable: the whole-tree gates
(`flake8 .`, `black --check .`, `isort --check`, `pyright`, the unit suite, and
`tests/e2e -m e2e -n0`), an independent QA round driving the real app, and the
review round on head. The QA matrix should carry the four cases of §5.1 plus the
two watch-items in §8.1 and §8.5.

---

## 10. Rejected alternatives, consolidated

| option | rejected because |
|---|---|
| Subscribe after the snapshot | moves the capacity refusal and the move fence past the response start, breaking the route's documented JSON-vs-200 promise (`routes:3265-3266`, `:3285-3287`), and loses the build's frames outright instead of dropping them under proof |
| Coalesce on overflow (keep-latest per kind) | needs a frame-kind taxonomy neither the replay nor the client has, and still discards events |
| Bound what a cold engage publishes | the engage deltas are how the facade's state is built; suppressing them breaks the snapshot's own source |
| Enlarge the queue / the replay window | unbounded memory on the hottest path, and it moves the wall rather than the policy |
| Keep the epoch across a detach (no dwell) | the detached interval is not captured by anything (§4.2), so `gap=false` would be a claim the bridge cannot honour, and the re-engage's own burst still makes `gap` true in the cases that matter |
| Dwell with both presence flags preserved | holds `visible` on a viewer that has gone, suppressing rung-1 notifications for up to 20 s — the exact window in which a completed turn most needs to notify |
| Dwell on every `release()`, not only a stream's | silently extends residency across the whole read surface, and changes every test (and every claim) about a read detaching |
| Dwell a subscription the bridge itself revoked (`overflow`) | inverts the relief valve — the bridge has just told that reader to resync — and makes a *closing* bridge dwell; it also keeps `test_slow_subscriber_overflow_is_explicit_and_bounded` (`tests/unit/server/test_desktop_sessions.py:138-156`) meaningful |
| Do nothing (D1 alone) | the stream stops ending, but every genuine break — sleep/wake, daemon re-exec, a slow reader — still costs a runtime teardown and a full repaint |

---

## 11. What accumulates with uptime? (task-8)

The operator's report — *"after restarting the UI it seems to go away for a bit,
but I think it probably happens after some long amount of time of use"* — says the
trigger is state that accumulates, not a steady-state property. Five candidates
were tested against the live daemon and its two logs (read-only: `ps`, the log
files, the store's `stat`s; no request was made to the daemon, nothing was
written, nothing was restarted).

**Headline: none of the five is the uptime switch, and the one that is real needs
a change the design did not have.** The rest of this section is the evidence and
the ranked disposition.

### 11.1 The bridge pool (`BRIDGE_COUNT = 64`) — latent, and D2 makes it reachable

**Mechanism, as claimed.** `DesktopSessions.session` evicts at the cap with
`del self.bridges[oldest.session_id]` (`:4958-4963`) and never `close()`s what it
drops, so the dropped bridge keeps its facade, its runtime, its subscriptions and
its presence until something else reaps them — and a later open of that session
builds a *second* bridge with its own epoch.

**Measured.** Distinct sessions opened per daemon lifetime, counted from every
session-scoped route in both logs and segmented at each
`Registered this app's own daemon` line:

| log | lifetime | distinct sessions opened |
|---|---|---|
| old | 00:43 → 08:42 · 08:42 → 10:47 · 10:47 → 17:59 · 17:59 → 23:05 | 17 · 17 · 12 · 25 |
| current | 08:13 → 12:13 · 12:13 → 13:36 · 13:36 → 15:14 · 15:14 → 17:49 · 17:49 → now | 24 · 14 · 25 · 17 · **10** |

The maximum over ten lifetimes is **25 against a cap of 64**, and
`Too many active desktop sessions` appears **0 times** in both logs. The pool is
not full today, has never been full across the ten daemon lifetimes these two logs
cover, and cannot fill while the daemon restarts every few hours on `uv-tool`
updates — each restart resets `self.bridges` outright.

**So: not realized, and not the cause of this report.** The defect is real but
latent, and it is *bounded* rather than permanent: the dropped bridge stays alive
through its own `attention_task` and its facade's callbacks, so it keeps its
runtime attached until its watch lease lapses and the `DEFAULT_GRACE_S = 3 s`
drain reaps it (~48 s), after which nothing references it. What that window buys
is a session that can have **two bridges with two epochs** (and its runtime held
by the orphan), and it stops being merely latent under D2, which is why it becomes
item 3 of §12: a **dwelling** bridge is deliberately not evictable (§4.3.5), so the
set of evictable bridges shrinks and the
`ValueError("Too many active desktop sessions")` arm of `:4961` becomes reachable
in a state that previously always had an idle bridge to take. The shape of the
change is in §12.

### 11.2 Per-bridge subscribers (`SUBSCRIBER_COUNT = 32`) — not realized

The stream's own teardown pops its subscriber as the generator finishes (`:3145`),
so an ordinary disconnect reaps promptly; `_expire_watches` (`:3057-3076`) is a
*presence* refresh, not a subscriber reaper, and has nothing to do with the table.
Measured: `Too many event subscribers` appears **0 times** in both logs.

One latent leak exists and is worth naming because it is in the handshake:
the route subscribes at `:3277`, and if the response body is never iterated —
the client disconnects between headers and body — the generator's `finally` never
runs, so that subscription is never popped (`release_once` releases the *bridge*,
not the subscription). Reaching the cap needs **32 of those on one bridge**, on a
bridge that stays reachable; nothing in either log suggests it. D2 does not make
it worse: a dwelling subscription is popped by its own dwell task after
`RECONNECT_DWELL_S` at the latest, so the dwell bounds the slot rather than
holding it forever.

**No change needed for this report.** If a reviewer wants it closed anyway, the
shape is a `sub`-aware `release_once` that pops only when `not sub.dwelling` —
not worth spending task-4's budget on.

### 11.3 Per-session runtime residency — bounded; one unreaped child

`_detach()` unsubscribes the bridge (`:1496-1497`) and calls `remote.dispose()`
(`:1499-1501`), which closes the viewer's socket
(`session/attached.py:8903-8908`); the runtime's term 3 then falls and it exits
after `DEFAULT_GRACE_S = 3.0` (`session/runtime/process.py:102`, `:1138`).

Measured: the app's daemon (`pid 74822`) has exactly **three children** — one live
session runtime (`id=0c1358c8`, 34 MB RSS, 18 min), one **unreaped zombie**
(`<defunct>`, ppid 74822), and a runtime that started 11 minutes ago. It is not
proportional to the 44 sessions the log shows being opened, and the 28
`session.runtime.process` processes alive on this host belong to the ~25 agent
sessions this machine runs concurrently, not to the app.

Ten `ppid=1` runtimes *do* carry session ids the app has opened
(`112979d4`, `790d870e`, `a81ceec0`, `6011712f`, …), but they cannot be attributed
to the app: the agent fleet shares the same session store and the same ids, and
`lop exec` runtimes are orphaned to pid 1 in exactly the same way. What can be
said is the bound that matters: **the daemon's own child count is 1-2**, so
residency does not grow with the number of conversations viewed.

The zombie child is a small hygiene finding of its own (the daemon does not always
reap), unrelated to this defect and out of scope.

### 11.4 Store growth and the length of the un-drained window — refuted, and covered anyway

The concern is that a 678 MB `analytics.db` (it was 659 MB earlier today) and
growing journals stretch `await self.snapshot()` (`:3113`), the window D1 is about.

Measured proxy for that window, from the log's own two routes: the delay between a
stream's `GET …/events` (logged at response *start*) and the client's next
`POST …/watch` (sent only after it has the `open` frame), per hour:

| hour | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p50 (s) | 0.34 | 0.61 | 2.79 | 1.08 | 0.34 | 0.13 | 0.14 | 0.74 | 0.61 | 0.61 | 0.41 |
| p90 (s) | 6.48 | 3.47 | 6.60 | 6.17 | 2.26 | 0.16 | 0.46 | 2.07 | 1.24 | 2.70 | 1.95 |

Over 332 pairs: p50 **0.64 s**, p90 **3.95 s**, max **17.3 s**. It does **not**
grow with uptime — the worst hours are the morning's, the best are the afternoon's
— which fits what the code says the window is made of: a cold attach, not the
store. The store's own terms are bounded or independent of size: the attention
read is capped at `ATTENTION_SNAPSHOT_WAIT_S = 0.05` (`:235`) and serves the last
known state; `history()` is a *backward* tail page whose cost the reader's own
docstring measures at 1.5-6 ms at the tail, and it is served through a
process-wide LRU with explicit bounds (`PAGE_CACHE_ENTRIES = 16`,
`PAGE_CACHE_BYTES = 24 MiB`, `local_operator/session/page_cache.py:113-114`) — no
unbounded cache accumulates across sessions.

**And D1 removes the window's duration from the equation entirely**: before the
handshake completes, a full queue evicts a superseded frame instead of
disconnecting, so a snapshot that takes 20 s can no longer end a stream. A longer
window costs a longer cold attach and more dropped-and-superseded frames, never a
gap storm. **Fully covered.**

### 11.5 Renderer-side accumulation — not measurable from here, and probably not uptime

Read (not run): the main relay keeps one `AbortController` per stream in a `Map`
and deletes it on end, on abort and on close
(`local-operator-ui` `src/main/desktop-stream.ts:89,158,189,315-321`); the hook's
buffer is drained every flush (`pending.current = []`), its label-gap set is
bounded (`LABEL_GAP_MAX_TRACKED = 512`), and the other per-flush sets are locals.
The one structure that grows without a constant is the transcript and its index
(`paintedIds`, `use-canonical-session.ts:729,1358`), which grows with the
**conversation's** length, not with app uptime, and resets when the operator
navigates to another conversation. That is a plausible "one very long conversation
gets slower" effect; it is not a per-hour accumulator and it cannot be measured
from this position.

**What would settle it:** the renderer-driver DOM capture task-2 already built,
run against one long-lived conversation at two points an hour apart, comparing
flush duration and the painted row count. No backend item either way.

### 11.6 What the operator's own observation most likely is

"Restarting the UI fixes it for a bit" is exactly what a **daemon restart** does
here, and the app performs one when it is relaunched: the current daemon
registered at **17:49:43** (`pid 74822, v0.62.38`), the previous one at 15:14:05,
before that 13:36:42, 12:13:03 and 08:13:52 — five in one working day. Each
restart drops every bridge (so every epoch rotates), every stream socket and every
runtime at once, and the storm needs all three of its ingredients to recur: a
long *cold* engage that publishes a burst (measured: the window's p90 is ~4 s,
max 17 s), a subscriber with no reader for that burst, and a reconnect after the
epoch rotated. D1 removes the second ingredient and D2 removes the third's cost;
neither is a function of uptime, which is consistent with the report — the
*building blocks* are always present, and what "some long amount of time of use"
supplies is simply more chances to hit them.

### 11.7 Ranked disposition

| # | mechanism | realized here today? | covered by D1/D2? | needs a change? |
|---|---|---|---|---|
| 1 | **Bridge-pool eviction drops a bridge without closing it** (`:4958-4963`), and D2 makes dwelling bridges unevictable | **no** — max occupancy 25 of 64 across ten daemon lifetimes; the refusal string never appears | **no** — D2 *adds* a state where the pool can have no evictable bridge | **yes — item 3, §12** |
| 2 | Subscriber table filling to `SUBSCRIBER_COUNT` | no — the refusal string appears 0 times; a leak path exists but needs 32 hits on one bridge | yes in effect (the dwell bounds a slot rather than holding it) | no |
| 3 | Runtime residency growing with sessions viewed | no — the daemon's child count is 1-2 regardless of the 44 sessions opened | n/a | no (a separate unreaped-zombie hygiene note) |
| 4 | Store growth lengthening the snapshot window | **refuted** — the window is 0.64 s p50 / 3.95 s p90 and does not trend with uptime; its terms are bounded or size-independent | **yes, fully** — D1 makes window *length* irrelevant | no |
| 5 | Renderer-side accumulation | unknown from here; the only unbounded structure grows with conversation length, not uptime | n/a | no (task-2's DOM capture settles it) |

---

## 12. Item 3 for task-4: eviction closes what it drops, and the dwelling bridge is the last resort

Recommended addition to the implementing PR, in the same function task-4 already
edits (`_evictable` `:4598-4608`, and its one caller `session()` `:4930-4984`).

```python
    def _evict_one(self) -> DesktopSessionBridge | None:
        """The bridge to drop for pool room, or ``None``. Caller holds the pool lock.

        TWO TIERS, and the order is the point. An ordinary idle bridge
        (``users == 0``, no handout, not dwelling) has nothing to lose. A
        DWELLING bridge is the last resort — it is holding a runtime for a viewer
        that may be reconnecting this second — so it is taken only when nothing
        else is available, and taking it ends its dwell: its viewer gets an
        honest gap on reconnect instead of being refused the open outright.
        """
        idle = [b for b in self.bridges.values() if self._evictable(b)]
        if not idle:
            idle = [b for b in self.bridges.values() if b.dwelling and not b.users]
        if not idle:
            return None
        victim = min(idle, key=lambda b: b.touched)
        del self.bridges[victim.session_id]
        return victim
```

and in `session()`, the victim is closed **after the pool lock is released** —
and not one line earlier:

```python
        finally:
            if taken:
                self._end_handout(session_id)
        if evicted is not None:
            # AFTER the lock, deliberately: ``close()`` awaits the bridge's own
            # lock and the owner connection's teardown, and holding the pool lock
            # across either is the 7.9 s open the docstring at :4796-4812 exists
            # to prevent. Closing is also what the bare ``del`` never did: without
            # it the dropped bridge keeps its facade, its runtime, its subscribers
            # and its presence, and a later open of that session builds a SECOND
            # bridge with its own epoch while the first is still attached.
            with contextlib.suppress(ConnectionError, RuntimeError):
                await evicted.close()
        try:
            yield bridge
```

**Why the second tier is not optional once D2 lands.** Before D2, `users == 0`
implied evictable, so a pool at the cap always had a victim unless 64 sessions
were streaming at once. After D2 a bridge can be unevictable *and* idle for 20 s,
so the `ValueError("Too many active desktop sessions")` arm (`:4961`) becomes
reachable in a state that used to be impossible — the buyer of a 20 s reconnect
window would be a new refusal to open. The fallback trades that refusal for one
honest gap on a viewer that had already lost its transport.

**Why it is still not the fix for this report.** Measured occupancy is 25 of 64 at
its worst, and every daemon restart resets the pool (§11.1, §11.6). This item
protects the change from introducing a new refusal and closes a pre-existing leak;
it must not be allowed to complicate or delay D1/D2 verification.

**Tests** (both in `tests/unit/server/test_desktop_stream_dwell.py`, beside T2-4):

* `test_eviction_closes_what_it_drops` — fill the pool to `BRIDGE_COUNT`, open one
  more session: the evicted bridge has `remote is None`, no subscribers, no live
  dwell task, and the new session has a working bridge.
* `test_the_dwelling_bridge_is_the_last_resort` — with one ordinary idle bridge
  and one dwelling bridge present, eviction takes the idle one; with only dwelling
  bridges, it takes the oldest and that bridge's dwell ends (its viewer's next
  open is a plain `gap=true`).
* `test_eviction_never_awaits_inside_the_pool_lock` — the regression guard for the
  ordering: hold the pool lock (or assert via a spy that `close()` is not entered
  while `pool.lock.locked()`), because this is the one way the item can be
  implemented and still be wrong.

**Regression risk, stated.** The eviction path is reached only at the cap, which
this machine never reaches, so the change cannot alter any measured behaviour
here — and the `_evict_one` extraction must leave the non-full path byte-identical
(`_evictable`'s meaning, `touched`'s role and the handout reservation untouched).
The one new await lands on the cap path only, where the pool is already paying a
bridge lookup and a handout anyway.
