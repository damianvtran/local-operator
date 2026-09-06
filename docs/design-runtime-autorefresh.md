# Design: idle runtimes refresh themselves, and viewers never fake an abort

Status: proposal (architect). Base: `origin/main` b0db51d9c. Two PRs (§8).
No `pyproject.toml` bump in either — the window's release owner handles that.

Operator's requirement, verbatim:

> make sure that runtimes that are inactive automatically refresh and update
> and/or bring down the runtime on inactive sessions so that resuming would do
> the update. In practice, on resume we should never see this message, and we
> shouldn't show it if resuming an active session regardless. The user should
> never need to run /stop to refresh or update a runtime.

> make sure the TUI can properly stay connected during waits and there's
> nothing that's killing or interrupting waiting sessions.

Every line reference below is against b0db51d9c.

## 1. The problems as found in the code

### 1.1 A stale runtime is immortal while somebody is looking at it

`process.py::_should_exit` (`session/runtime/process.py:121-152`) is the only
exit policy a runtime has, and it is three terms: not busy, no wake within
`WARM_WINDOW_S`, **no attach client**. Term 3 is why runtime pid 57079
(0.49.8@46a4e9b, last turn ended 05:01) was still resident at 10:00 on a host
whose disk had been 0.49.9@f4a70b99 since ~05:00: `lop --resume a6a8186ea42d`
(pid 55933) held an `attach` socket to it. Nothing in the runtime compares
`installed_build()` against what it loaded — the stamp is read exactly once,
at record construction (`server.py:411-443`), and never again.

The only thing that notices is the *viewer*: `OperatorApp._check_build_skew`
(`tui/app.py:14870-15011`) compares `owner_version/owner_source_ref` off the
record with its own `_loaded_build` and paints notice C —
`“<name>” is running 0.49.8@46a4e9b but this window is 0.49.9@f4a70b99 — some
commands may not work. /stop, then send again to restart it.` — at
`adopt`, `engage`, `engage-bound`, and `slash-engage` (`app.py:2953, 8513,
8553, 8687`). It is advisory and hands the user a chore. That chore is what
the operator is refusing.

### 1.2 The record's `busy` bit sticks at True after any wake/peer turn — PROVEN

Hypothesis in the task: only `_observe_prompt_drain` (`owned.py:1058-1074`)
publishes `busy` *after* the final `AgentEndEvent`, because that event is
emitted while `Session._is_streaming` is still True (cleared in the `finally`
at `session.py:6047`, after `_flush_held_end` at `:5766`). Every turn that
does not run through the prompt queue therefore ends with the record saying
`busy: true`.

Verified on this worktree with a throwaway e2e probe (real `Session` from
`tests/e2e/harness.build_session`, production `OwnedSessionHandle` +
`RuntimeServer`, `handle.subscribe(server._schedule_push)` as `_serve` does):

| path | `handle.is_busy()` after turn | `server._busy` (record) |
|---|---|---|
| `handle.prompt(...)` | False | **False** |
| `receive_peer_message(mode="mailbox", wake=True)` | False | **True — stuck** |
| `receive_peer_message(mode="steer")` on an idle session | False | **True — stuck** |

The four `_spawn_background(self._prompt_messages(...))` callers in
`session.py` all share the shape: peer wake `:4889/:4897`, background-job
result delivery `:6607`, resume catch-up `:9472`, scheduled wake `:10201`.
Also the goal loop: `_loop_driver` (`owned.py:1010-1037`) *does* go through
`self.prompt(..., wait_complete=True)`, so it is covered by the drain
observer — but only for its last iteration, which is fine.

Consequence today: `lop sessions` shows a running marker on an idle session
(the 57079 record said `busy: true` for five hours), the desktop/phone
pickers do the same, and any future "retire when idle" policy that trusted
the record would never fire for a session whose last turn was a wake. The
runtime's own `is_busy()` is correct; only the *published* bit is wrong.

### 1.3 A transient viewer disconnect paints "interrupted" on a turn that never stopped

`RemoteSession._on_disconnected` (`session/remote.py:1824-1857`):

```
self._recovering = True
self._owner_ready.clear()
self._end_turn_locally()          # ← synthesises AgentEndEvent(aborted=True, generation=0)
self._recovery_task = asyncio.create_task(self._recover_owner())
```

`_end_turn_locally` (`remote.py:1859-1918`) exists for three genuinely
terminal outcomes (killed owner, stopped owner, going cold) but it is called
*before* recovery has learned which of those — if any — applies. The
synthesised end reaches `EventController._handle_agent_end` and the app's
turn-end path, which calls `_retire_live_tool_cards` (`app.py:24152`, each
live card → `⊘ interrupted`) and, when no card was live, appends
`NoticeBlock("interrupted", "warning")` (`app.py:23732-23742`). Then
`_recover_owner` (`remote.py:2037-2170`) finds the *same* record (same pid),
re-dials within ~100 ms, and `_finish_sync` (`remote.py:1436-1468`) seeds a
fresh `AgentStartEvent` because `frontend_state.streaming` is still True.
The band recovers; the ledger keeps the `⊘ interrupted` cards and the notice.
That is exactly "interrupted painted mid-wait": the `wait` tool was still
running on the runtime the whole time.

Why the socket drops in the first place — every one of these is silent
(`_drop_client`, `server.py:1022-1088`, logs nothing; only cap eviction at
`:928` does):

1. `_send_to` (`server.py:2110-2119`): any frame whose `writer.drain()`
   exceeds `_SEND_TIMEOUT_S = 1.0` drops the client. `_push` (`:2053-2067`)
   sends the **full capped projection to every client including TUI attach
   viewers**, ~20×/s while streaming (`_push_later` coalesces to 50 ms). The
   TUI's `AttachClient` was constructed with `lambda _projection: None`
   (`remote.py:1257`) — it throws every one away. The projection is
   100–900 KB on this host's resident runtimes. A TUI event-loop stall of
   >1 s (a large transcript reflow, a 60 MB parse on a sibling path, the
   compositor under load) leaves the kernel socket buffer full of projection
   bytes the viewer never wanted, `drain()` times out, and the viewer is
   dropped from a runtime that is perfectly healthy.
2. `_enqueue_client_frame` (`server.py:1892-1913`): the 64-deep event FIFO
   overflows after `_compact_event_queue` fails to free a slot → drop.
3. `_relay_frontend_to_on_loop` (`server.py:1881-1887`): >64 canonical
   updates pending before `frontend_ready` → drop.
4. Reader EOF / reset (`:1010, :1017`) and shutdown (`:779-781`).

Also `RemoteSession.abort` (`remote.py:2662-2664`) is
`asyncio.create_task(self._client.abort())` with no done-callback; when the
client is mid-recovery `_request_frame` raises `ConnectionError("not
attached")` (`attach_client.py:461`) and the loop logs "Task exception was
never retrieved" — 12 such rows in the operator's log today (10:28).

### 1.4 What is NOT broken

- The `wait` tool's own cancel paths (`tools/builtin.py`) — the task's
  evidence stands: steer-driven cancels say "the wait was cancelled", they
  never paint "interrupted". Not touched here.
- `retire_if_pristine` (`server.py:1450-1517`) and the `stopping` frame
  (`:545-615`) — the design below reuses both rather than adding a second
  stop path.
- `_should_exit` terms 1 and 2 — correct and kept.

## 2. Options

### 2.1 Where the refresh decision lives

**A. Viewer-side only.** `_check_build_skew` sends `stop` when the owner is
stale and idle, then re-engages. Rejected as the *primary* mechanism: a
runtime with no viewer attached (the phone's SSE daemon attach is `daemon`
kind; a `send`-woken session) is never looked at by a TUI, so it would stay
stale until someone opened it — precisely the "on resume we should never see
this" case, because the first thing the resume does is bind to the stale
process. Viewer-side stays as a *belt* (§4.2), not the policy.

**B. Runtime-side self-retirement (recommended).** The runtime compares its
boot stamp with disk on the reaper's tick; when they differ and the runtime is
idle, it announces and exits cleanly. The viewer treats that announcement as
"a fresh runtime is owed, re-engage" rather than "the session was stopped".
Covers every runtime including unwatched ones; the only new wire item is one
additive field on an existing frame.

**C. Runtime self-re-exec** (`os.execv` a new `process.py` in place, same
pid). Rejected: the lease, the record, the socket, and the attach clients all
key on the pid and port, and a re-exec would have to hand every one of them
over across the exec boundary. `engage_runtime` already knows how to spawn a
successor on demand (`launch.py:376-540`); reusing it costs a ~1.2 s cold
start on the *next* message, which the residency design already accepts as
the price of an idle exit (`process.py:57-66`).

### 2.2 Should an attached viewer block the refresh?

The operator says no, and the code agrees it need not: a viewer already
survives owner exit (`_recover_owner` → `_go_cold` after `COLD_FALLBACK_S`,
`remote.py:2056-2066`) and a cold viewer already re-engages on the next
prompt (`_ensure_bound`, `remote.py:1074-1144`). What is missing is only
(a) an announcement so the viewer does not spend 8 s chasing a record and
then paint "runtime exited" for what was a planned refresh, and (b) that the
announcement not be confused with `stopping` (which parks the viewer in the
"this session was stopped; /resume reopens it" state, `app.py:3111-3180`).

So: **an attached viewer does not block retirement**; it receives a
`retiring` frame with `reason: "stale-build"` and re-engages eagerly.

### 2.3 Eager vs lazy re-engage after a refresh retirement

- *Lazy*: viewer goes cold quietly; next prompt spawns. Cheapest, and the
  band would show the cold state ("no runtime") until the user types.
- *Eager* (recommended): viewer calls `_start_runtime_engage(reason=
  "refresh")` as soon as the retiring runtime's socket closes. The mount path
  already engages eagerly for exactly the band-completeness reason
  (`app.py:8427-8455`), and the engage is idempotent behind `_bind_lock`.
  Cost: one process spawn per stale runtime per `lop-update`, which is the
  same count the lazy path would pay one message later — but with a band that
  never goes half-empty. Debounce (§3.4) keeps N viewers from spawning at the
  same instant.

### 2.4 Disconnect handling: synthesise the abort now, or defer?

- *Now* (status quo): honest for owner death, false for transient drops.
- *Defer until recovery's verdict* (recommended): `_on_disconnected` marks
  the turn *suspect* and starts recovery; the synthesised end fires only when
  recovery goes cold, learns the stop, or re-binds to a snapshot whose
  `streaming` is False / whose generation moved past the one that was live.
  On a successful re-bind to the same live turn, nothing is synthesised and
  `_finish_sync`'s seeded `AgentStartEvent` is the *only* thing the app sees
  — which its generation guard (`tui/events.py:536-565`) already treats as
  "same turn, still open" when the generation is unchanged.

The deferral cannot hang the UI: every path out of `_recover_owner` already
ends in `_go_cold` (which calls `_end_turn_locally(direct=True)`,
`remote.py:1999`), `_notify_stopped` (§4 keeps the end there), or a
successful bind. The 8 s cold deadline bounds the wait.

## 3. PR A — self-refresh, busy-bit, notices

### 3.1 Busy bit: publish from the session's turn boundary, not the drain task

Fix at the source: the handle needs a signal that fires **after**
`_is_streaming` is cleared, for every turn regardless of who opened it.

`session/session.py` — `_prompt_messages` (`:5694-5710`) and
`Session.prompt`'s lock hold both release `_turn_lock` after
`_run_turn_pipeline`'s `finally` has already run, but `_is_streaming = False`
lives one level deeper in `_run_turn`'s `finally` (`:6047`). Add ONE hook,
called from `_run_turn_pipeline`'s `finally` **after** `_flush_held_end`
(`:5766`) and **after** the loop's own `_is_streaming = False` has run
(i.e. as the last statement of that `finally`, guarded so it cannot raise):

```python
# session.py, Session.__init__
#: Fired once per turn, after the turn's terminal event has been emitted
#: AND ``is_streaming`` has been cleared. The OwnedSessionHandle uses it to
#: republish the record's ``busy`` bit; the per-event path cannot, because
#: the final AgentEndEvent is emitted while the flag is still True.
self.on_turn_settled: Callable[[], None] | None = None
```

and in `_run_turn_pipeline`'s `finally` (after `self._flush_context_journal()`):

```python
settled = self.on_turn_settled
if settled is not None:
    try:
        settled()
    except Exception:  # noqa: BLE001 — a publish failure is not a turn failure
        logger.debug("on_turn_settled hook failed", exc_info=True)
```

Ordering matters and must be pinned by the test: `_run_turn`'s `finally`
(`:6030-6047`) runs on the way out of `await self._run_turn(...)` at `:5757`,
so by the time `_run_turn_pipeline`'s `finally` executes, `_is_streaming` is
already False and `_held_end` has been flushed. `running_subagents()` may
still be >0 (a `task` left children running); that is correct — the handle's
`is_busy()` then still says True and the bit stays True until the next
`_notify` (a subagent event) republishes it. The hook only guarantees the
*turn's* contribution is observed.

`session/runtime/owned.py` — `OwnedSessionHandle.__init__`: 
`session.on_turn_settled = self._publish_busy` (probe `hasattr` so reduced
sessions in tests keep working). `_observe_prompt_drain` keeps its
`_publish_busy()` — it is harmless and covers legacy sessions without the
hook. Also call `self._publish_busy()` at the tail of `receive_peer_message`
(`owned.py:1273` already calls `_notify()` which does it — no change needed
there; the missing publish is the *end* of the turn, which the hook covers).

Belt: `RuntimeServer._heartbeat_loop` (`server.py:826-845`) adds
`self.set_busy(bool(is_busy()))` from a probed `self._handle.is_busy` before
`_republish_detached()`. A 15 s-stale bit is still wrong for the picker, so
the hook is the fix and this is only the floor.

### 3.2 Runtime self-refresh: a fourth reaper term, one new frame field

**Boot stamp.** `server.py:411-443` already computes `build = installed_build()`
for the record; keep it on the server as `self._boot_build: BuildStamp`.

**Detection.** `process.py` gains:

```python
#: How often an idle runtime re-reads the install on disk. Cheap (one
#: dist-info lookup + one 60-byte file), but there is no reason to do it on
#: every 250 ms reaper tick: a runtime that is already idle can afford to
#: notice an update within a few seconds, and a busy one never checks.
BUILD_CHECK_S = 5.0

#: A freshly written install is not a stable one: ``lop-update`` runs
#: ``uv tool install --force`` (which rewrites site-packages over several
#: seconds) and THEN writes ``.lop-source``. Retiring against a half-written
#: tree would spawn a successor that imports a mix of two builds. Require
#: the marker's mtime to be at least this old before acting on it. Also the
#: natural jitter base for the stagger below.
BUILD_SETTLE_S = 10.0
```

`_build_changed(boot: BuildStamp) -> bool`: `installed_build() != boot` AND
`.lop-source` (when present) has `mtime <= now - BUILD_SETTLE_S`. When
`.lop-source` is absent (PyPI/pipx install) compare on version alone — that
is what `BuildStamp` already does — and use the `dist-info` directory mtime
for the settle check. Editable checkouts have no `.lop-source` and a
constant version: they never trip this, by design (matches
`design-build-skew.md` §6.5).

**Policy.** Not a fourth term inside `_should_exit` — that predicate answers
"may I exit *quietly*", and a refresh must *announce*. Add beside it:

```python
def _should_refresh(handle, runtime, boot: BuildStamp) -> bool:
    """Retire so the next engage spawns from the build now on disk.

    Only when the runtime is doing NOTHING it would lose: ``is_busy()``
    False (the same authority as term 1 — never the record's bit, which
    §3.1 fixes but which is a derived copy) and no wake inside the warm
    window (a wake about to fire would just be paid twice). An attached
    viewer does NOT hold: the viewer re-engages a fresh runtime on its own
    (``retiring`` frame below); holding for it is exactly what kept a
    5-hour-stale runtime resident. ``is_pristine()`` runtimes are retired
    too — a pristine stale runtime is the cheapest possible refresh.
    """
```

The reaper loop (`process.py:172-201`) gets a second branch checked every
`BUILD_CHECK_S`: if `_should_refresh(...)`, re-check once after a jittered
stagger (see §3.4), then `await runtime.announce_retiring("stale-build")`,
re-check `is_busy()` **again** (the announce is an await; a `peer_message`
can open a turn in that gap — same shape as `_retire_if_pristine`'s re-check
at `server.py:1502-1510`), and if still idle run `_clean_exit` and set
`stop`. Log at INFO:
`session runtime: build on disk is 0.49.9@f4a70b9 but this process loaded
0.49.8@46a4e9b; idle, retiring so the next engage runs the new build`.

Mid-turn is impossible by construction: `is_busy()` is checked immediately
before `_clean_exit` on the same loop step, and `_clean_exit` →
`handle.dispose()` is the path a `stop` op already takes.

**Wire.** The `stopping` frame (`server.py:575`) is *the* "deliberate
disconnect follows" signal and every viewer already reads it. Do **not**
overload it: a viewer that reads `stopping` parks in the stopped state and
tells the user `/resume` reopens it (`app.py:3162-3180`), and
`_session_was_stopped` (`remote.py:1920-1947`) would then look for a
`stopped_at` marker that a refresh never writes. Add a new frame:

```python
{"op": "retiring", "session_id": ..., "reason": "stale-build",
 "from": "0.49.8@46a4e9b", "to": "0.49.9@f4a70b9"}
```

Emitted by `RuntimeServer.announce_retiring(reason)`, a sibling of
`announce_stop` that reuses `_write_now`/`_broadcast` exactly as `:576-615`
do. Additive: an old viewer ignores the unknown op, sees EOF, runs the
existing `_recover_owner`, and goes cold after 8 s — the pre-PR behaviour, so
no `PROTOCOL_VERSION` bump. Not sent to `daemon` clients (the phone's
projection path stays byte-identical; the daemon already handles owner exit
by re-adopting on the next record).

`attach_client.py::_pump` (`:373-381`): add `elif op == "retiring": reason =
RETIRING_REASON` with `RETIRING_REASON = "owner retired for a newer build"`
next to `STOPPED_REASON` (`:61`). `_ClientConn` needs nothing new.

**Viewer.** `RemoteSession._on_disconnected` (`remote.py:1824`):

```python
if _reason == RETIRING_REASON:
    # A planned refresh, not owner death and not a stop: the runtime
    # left so the next engage runs the build now on disk. Nothing to
    # recover — the successor does not exist yet — and nothing was
    # interrupted (the runtime retires only when idle). Go cold NOW
    # rather than chasing a record for 8 s, and tell the app so it can
    # re-engage eagerly.
    self._go_cold(refresh=True)
    return
```

`_go_cold(refresh: bool = False)` skips `_end_turn_locally` when `refresh`
(the runtime was idle by contract; `_streaming` is already False — assert
that in the test) and calls a new `_refresh_callback` instead of
`_went_cold_callback`. `set_refresh_callback` follows the
`set_went_cold_callback` shape (`remote.py:2011-2017`).

`tui/app.py`: in `_adopt_session`'s `is_remote` block (`:2812-2830`) install
`set_refresh(self._on_runtime_refreshed)`:

```python
def _on_runtime_refreshed(self) -> None:
    """The bound runtime retired itself for a newer build. Re-engage now so
    the band never shows the cold state for a refresh the user did not ask
    for; the next prompt would have done this anyway (``_ensure_bound``)."""
    self._warm_engage_started = False
    self._start_runtime_engage(reason="refresh")
```

`_start_runtime_engage` (`app.py:8501-8555`) already re-runs
`_check_build_skew` after the bind; with a fresh runtime the owner stamp
equals the disk stamp and notice C is silent. If the *window* itself is stale
(notice A), the fresh runtime is newer than the window — that is the
incident class `design-build-skew.md` §4.1 (B) already repairs, and A's
`/reload` copy stays.

**No notice is painted for a refresh.** Reasoning: the operator's bar is
"never see this message"; the band's `starting…` state (`_set_starting`,
`app.py:8528`) already covers the ~1 s the re-engage takes, and a second
line of prose for a background housekeeping event is the pattern
`_system_notice`'s own contract calls noise. A DEBUG log line suffices
(`runtime for %s retired for %s → %s; re-engaging`). Design review can
overrule; if it does, the copy is one dim line:
`updated to 0.49.9@f4a70b9` — never a warning.

### 3.3 Viewer-side belt: resume/bind against a stale idle owner

`_check_build_skew`'s C branch (`app.py:14957-15011`) becomes:

```
owner stale?
  ├─ owner idle (frontend_state.streaming False AND no running jobs AND
  │  no pending gate)  → request refresh; NO notice
  └─ owner busy       → notice C′ (copy below), once per session
```

"Request refresh" = `session.request_refresh()` → new `AttachClient.
request_refresh()` → op `refresh_if_idle`, dispatched next to
`retire_if_pristine` (`server.py:1332`, and add it to the no-push list at
`:1425-1432`). Handler: same `_should_refresh` predicate as the reaper (share
it: move the predicate into `owned.py` as `OwnedSessionHandle.may_refresh()`
and have `process.py` call that, so both sides use one function). Answers
`"retiring"` or `"kept: <reason>"`; a `kept` answer because the runtime is
busy is when C′ paints. An old runtime answers unknown-op → treat as `kept`
and paint C′ (the honest state for a runtime that predates the op).

Is the belt worth having if the reaper already does this within
`BUILD_CHECK_S + BUILD_SETTLE_S + stagger` (~15–40 s)? Yes, for one case:
`lop --resume` in the seconds after `lop-update`, before the reaper has
noticed. Without the op the viewer binds, sees skew, and either waits or
warns; with it the refresh happens in the bind path itself. Small (one op,
~40 lines) and it turns a race into a determinism.

### 3.4 Debounce across many runtimes

This host runs ~16 resident runtimes. `lop-update` rewrites the install over
several seconds, and all 16 reapers would otherwise see the change on the
same tick and spawn 16 successors within ~1 s (the eager re-engage). Three
layers:

1. **Settle**: `BUILD_SETTLE_S` (§3.2) — no runtime acts on a marker
   younger than 10 s, so nothing retires mid-rewrite.
2. **Stagger**: after `_should_refresh` first holds, sleep
   `random.uniform(0, BUILD_STAGGER_S)` with `BUILD_STAGGER_S = 20.0`, then
   re-check (work may have arrived). Sixteen runtimes spread over ~20 s; the
   spawn cost of a successor is ~1.2 s CPU-light, so ≤1 spawn/s average.
3. **Successor only on demand or viewer**: an unwatched idle runtime that
   retires spawns *nothing* — the next `send`/wake/resume engages from the
   new build. Only viewer-attached runtimes trigger an eager successor.

`retiring` is announced *after* the stagger, immediately before exit, so a
viewer never waits on a runtime that is "about to" leave.

### 3.5 Notice copy

- **C′ (stale AND busy)**, replaces both C variants at `app.py:14990-15011`:
  `“<name>” is running 0.49.8@46a4e9b (this window is 0.49.9@f4a70b9) — it
  will move to the new version when its current work finishes.` Absent
  version: `“<name>” is running an older version than this window — it will
  move to the new version when its current work finishes.` Kind: `"info"`
  not `"warning"` — nothing is wrong and nothing is asked of the user. The
  `/stop` sentence is gone from every notice. (`design-build-skew.md` §4.3's
  old-runtime `noop team_mutate` degradation notice at `app.py:14629` keeps
  its `/stop` wording for now: it fires only against pre-#624 runtimes, none
  of which can exist after this ships — drop the sentence there too, cheap.)
- **A** (window stale) unchanged. Self-reexec of the window when idle
  (`reexec.py`, `_cmd_reload` at `app.py:5145`) was considered: the machinery
  exists and `_turn_is_live` (`:5158`) is the guard. **Recommend not** in this
  PR: a TUI relaunch drops composer draft, scrollback position and any
  in-progress `ask` picker, and the operator did not ask for it. Leave A as
  the one notice with a manual remedy; revisit if it becomes the next
  complaint.

### 3.6 File-by-file (PR A)

| file | change |
|---|---|
| `session/session.py` | `on_turn_settled` hook; call in `_run_turn_pipeline` `finally` (`:5764-5776`) |
| `session/runtime/owned.py` | wire hook in `__init__`; `may_refresh()` predicate + `refresh_if_idle` handle method |
| `session/runtime/process.py` | `BUILD_CHECK_S/SETTLE_S/STAGGER_S`; `_build_changed`; reaper refresh branch |
| `session/runtime/server.py` | `_boot_build`; `announce_retiring`; `refresh_if_idle` op; heartbeat republishes busy |
| `mobile/attach_client.py` | `RETIRING_REASON`; `retiring` op in `_pump`; `request_refresh()` |
| `session/remote.py` | `_on_disconnected` refresh branch; `_go_cold(refresh=)`; `set_refresh_callback`; `request_refresh()`; `owner_idle` helper |
| `tui/app.py` | `_on_runtime_refreshed`; `_check_build_skew` C→C′ + belt; copy |
| `update.py` | `build_marker_age_s()` (mtime of `.lop-source` or dist-info) |
| `cli.py` | nothing (record fields unchanged) |

## 4. PR B — disconnect and relay hardening

### 4.1 Defer the synthesised abort

`RemoteSession._on_disconnected` (`remote.py:1854-1857`) becomes:

```python
self._recovering = True
self._owner_ready.clear()
# Do NOT end the turn here. A dropped socket says nothing about the
# turn: the runtime is usually still running it (a send timeout under a
# stalled TUI loop is the common cause). Recovery decides — see
# _settle_suspect_turn.
self._suspect_generation = self._generation if self._streaming else None
self._recovery_task = asyncio.create_task(self._recover_owner())
```

`_recover_owner` outcomes and what each does with the suspect turn:

| outcome | today | proposed |
|---|---|---|
| re-bind, snapshot `streaming` True and `generation == suspect` | seeds `AgentStartEvent` on top of an already-ended turn | seed as today; **no end synthesised**; ledger untouched |
| re-bind, snapshot `streaming` False or `generation > suspect` | (already ended, as aborted) | the turn ended while we were away. The durable replay (`_replay_durable_suffix`, `:1470`) paints its real rows; the real `AgentEndEvent` is NOT in `live_events` (the store clears them at turn end, `frontend_state.py:2870-2871`, `agent_end` empties the seed), so the viewer must synthesise one — but it cannot tell aborted from completed today. Add `last_turn_outcome: Literal["completed","aborted","error",""]` to `FrontendSessionState` (additive; set from `_run_turn_pipeline`'s `finally` off the flushed end's `aborted`/`error`), and synthesise `AgentEndEvent(aborted=(outcome=="aborted"), generation=0, error=...)`. An old runtime without the field → `""` → synthesise `aborted=True` (today's behaviour) |
| `_go_cold` | ends turn (aborted) | unchanged — a runtime that is gone did abort |
| `_notify_stopped` | ends turn (aborted) | unchanged |
| `retiring` (PR A) | n/a | never streaming by contract |

`_streaming` stays True during recovery (it is what routes the next message
to the steer branch, `remote.py:2608`); the app's band keeps `working` —
correct, since it is.

`_apply_frontend_facades` (`:1631-1633`) already overwrites `_streaming` and
`_generation` from the snapshot, so the second row's comparison is a
two-line check right after `_install_frontend` in `_recover_owner`
(`:2111`).

### 4.2 Stop pushing projections to full-TUI attach clients

`_push` (`server.py:2053-2067`) and `_push_to` send the projection to
every client. A client that declared `events=True` **and**
`frontend_state=True` is a full-TUI viewer (`remote.py:1259-1261`); it
consumes `frontend_sync`/`frontend_update`/`event` and discards
projections. The welcome projection (`_on_connection:954`) is still needed —
`AttachClient.connect` (`attach_client.py:259-263`) reads it as the identity
check. So:

```python
def _projection_recipients(self):
    return [c for c in self._clients.values()
            if not (c.wants_events and c.wants_frontend)]
```

used by `_push` only; `_push_to` (welcome) unchanged. The daemon (phone) and
legacy v3/v4-only attach clients keep receiving repaints byte-for-byte.
Measured motivation: 100–900 KB × ~20 Hz on a socket the viewer needs for
events; nothing on the TUI side reads it (`_dial:1257`). Justification for
*not* deferring this: it is the single largest contributor to the
`_SEND_TIMEOUT_S` drop and it is a pure subtraction.

Desktop surface (`surface == "desktop"`) declares the same two flags
(`remote.py` constructs the client identically with `surface=self._surface`)
and also reads nothing from projections — confirm with the desktop tests
(`tests/e2e/test_desktop_*.py`) that nothing keys on `op == "projection"`
after the welcome; the coder must grep `desktop` for `_on_projection` before
relying on this.

### 4.3 Log every drop with its reason and client kind

`_drop_client(conn, *, reason: str)` — every caller names one:

| caller | reason |
|---|---|
| `_send_to` timeout (`:2118`) | `send timeout (%.1fs)` |
| `_send_to` reset/pipe/OSError | `send failed: <exc type>` |
| `_enqueue_client_frame` overflow (`:1912`) | `event queue overflow (%d frames)` |
| `_relay_frontend_to_on_loop` pending overflow (`:1886`) | `frontend pending overflow before ready` |
| reader EOF (`:1010`) | `reader eof` |
| reader reset (`:1017`) | `reader reset` |
| daemon replacement (`:918`) | `daemon replaced` |
| cap eviction (`:929`) | `attach cap` (existing INFO line folds into this) |
| no `subscribe_frontend` (`:958`) | `frontend requested but unsupported` |
| shutdown (`:781`) | `runtime shutdown` |

One INFO line: `session runtime: dropped %s client %s (events=%s
frontend=%s surface=%s): %s`. The second, no-op call from a reader loop's
`finally` after a send-path drop logs at DEBUG only (guard on
`was_registered`, which the method already computes at `:1035`).

### 4.4 Raise the send bound for full-TUI clients, once projections are gone

With projections off the viewer socket, the remaining traffic is events
(~1–10 KB) and canonical updates. Keep `_SEND_TIMEOUT_S = 1.0` for the daemon
and legacy paths; for `wants_events and wants_frontend` clients use
`_TUI_SEND_TIMEOUT_S = 5.0`. Rationale: the bound exists to protect
"authority-bearing ACKs for every healthy front end" (`server.py:101-103`),
and sends are per-connection under `send_lock` — one slow TUI cannot block
another client's writes. A 5 s stall on a TUI loop is rare but real (a
60 MB transcript parse was measured at ~90 ms *per* threaded read; a
`--resume` reflow is seconds); the cost of a false drop is the whole §1.3
sequence. Watch it in rollout (§7).

### 4.5 `RemoteSession.abort` unretrieved exception

`remote.py:2662-2664`:

```python
def abort(self, reason: str = "interrupted") -> None:
    client = self._client
    if client is None or not client.connected:
        return  # nothing to abort on; the local end is what the app shows
    task = asyncio.create_task(client.abort())
    task.add_done_callback(_log_abort_failure)   # ConnectionError → DEBUG
```

### 4.6 File-by-file (PR B)

| file | change |
|---|---|
| `session/remote.py` | deferred end (`_suspect_generation`, `_settle_suspect_turn`); `abort` done-callback |
| `session/frontend_state.py` | `last_turn_outcome` field (additive) |
| `session/session.py` | set `last_turn_outcome` in `_run_turn_pipeline` finally |
| `session/runtime/server.py` | `_drop_client(reason=)` + logs; `_projection_recipients`; per-kind send timeout |
| `session/runtime/types.py` | nothing (no protocol bump) |

## 5. Test plan

### 5.1 PR A

Unit — `tests/unit/session/runtime/test_busy_settles.py`:
1. Real `Session` + `OwnedSessionHandle` + `RuntimeServer` (the probe in
   §1.2, promoted): after `prompt`, `receive_peer_message(wake=True)`,
   `receive_peer_message(mode="steer")` on idle, and a `_spawn_background(
   _prompt_messages(...))` driven directly — `server._busy is False` within
   500 ms of `handle.is_busy()` going False. **This test fails on
   b0db51d9c** (verified: 2 of 3 params fail).
2. `on_turn_settled` fires after `is_streaming` is False (assert inside the
   hook) and after the `AgentEndEvent` has been delivered to subscribers.
3. Hook raising does not fail the turn.

Unit — `tests/unit/session/runtime/test_process_refresh.py` (same fakes as
`test_process_reaper.py`):
4. `_should_refresh`: busy → False; wake in window → False; attached viewer
   → **True** (the operator's rule, pinned); pristine → True.
5. `_build_changed`: same stamp → False; different ref, marker mtime < settle
   → False; > settle → True; no marker, version differs → True.
6. Reaper: stamp flips mid-loop → `announce_retiring("stale-build")` called,
   then `_clean_exit`, `stop` set; a turn starting between announce and exit
   → kept (inject busy on the second probe).
7. Stagger bounded by `BUILD_STAGGER_S` (monkeypatch `random.uniform`).

Unit — `tests/unit/session/test_remote_refresh.py`:
8. `_on_disconnected(RETIRING_REASON)` → `_go_cold(refresh=True)`;
   `_end_turn_locally` NOT called; refresh callback fired; `is_cold` True;
   no recovery task.
9. `_on_disconnected(STOPPED_REASON)` unchanged (regression guard).

Unit — `tests/unit/tui/test_build_skew.py` (extend):
10. Stale + idle owner → `request_refresh` called, no notice.
11. Stale + busy owner → C′ info notice, no `/stop` substring anywhere in
    the ledger (`assert "/stop" not in transcript_text(app)`).
12. Refresh callback → `_start_runtime_engage(reason="refresh")` and
    `_warm_engage_started` reset.

E2E — `tests/e2e/test_runtime_refresh_e2e.py` (production
`process.py` in a subprocess like `test_cold_wake_e2e.py`, isolated
`HOME`+config dir, all `CMUX_*` unset, `LOP_SESSION_GRACE_S` small,
`BUILD_SETTLE_S`/`BUILD_STAGGER_S` overridable via env for the test only —
add `LOP_BUILD_SETTLE_S` / `LOP_BUILD_STAGGER_S` read the way
`LOP_SESSION_GRACE_S` is at `process.py:69-75`):
13. Boot a runtime with `installed_build` monkeypatched via a fake
    `.lop-source` in a temp prefix (`update.installed_build(prefix=)`
    already takes one — pass it through an env var `LOP_BUILD_PREFIX` the
    runtime honours only under test); attach a headless `OperatorApp` via
    `--resume`; flip the marker; assert within 10 s: old pid gone, record
    names a new pid, viewer `is_cold` False, `transcript_text(app)` contains
    no `"/stop"`, no `"interrupted"`, no `"stopped"`; the band shows the
    model.
14. Same, runtime busy (scripted stream parked on a `wait`-shaped tool) →
    still the old pid after 10 s; C′ present; then the turn ends → old pid
    exits, new pid appears, no further notice.
15. Unwatched: no viewer; flip; old pid exits; **no** new record appears
    (no eager spawn without a viewer); `lop send` to it engages a new pid
    from the new stamp.

### 5.2 PR B

Unit — `tests/unit/session/test_remote_disconnect.py`:
16. Mid-turn socket close → no `AgentEndEvent` emitted; `_streaming` True;
    re-bind with `streaming=True, generation==suspect` → seeded start only,
    still no end.
17. Re-bind with `generation > suspect` / `streaming False` → exactly one
    end, `aborted` per `last_turn_outcome`.
18. Recovery goes cold → one aborted end (unchanged).
19. `abort()` while detached → no task, no "never retrieved".

Unit — `tests/unit/session/runtime/test_server.py` (extend):
20. `_push` skips clients with `wants_events and wants_frontend`; welcome
    still delivered; daemon and events-only clients still receive.
21. Every drop path logs `dropped ... : <reason>` (caplog), exactly once at
    INFO per connection.
22. Send timeout for TUI-class client is `_TUI_SEND_TIMEOUT_S`.

E2E — extend `tests/e2e/test_viewer_attach_e2e.py`:
23. Viewer attached mid-turn (scripted `wait`); close the viewer's socket
    from the server side (`_drop_client(conn, reason="test")`); assert the
    viewer re-binds to the same pid and `transcript_text(app)` never
    contains `"interrupted"`; the turn then completes normally and the app
    paints the assistant row once.

QA matrix (qa-tester, real execution, isolated `HOME` per cell; never the
operator's live `~/.local-operator`): cell A = e2e 13 by hand with two `uv
tool install --force` refs in an isolated `UV_TOOL_DIR` and a real
`.lop-source` rewrite; cell B = e2e 14; cell C = e2e 15 with `lop send`;
cell D = e2e 23 with `kill -STOP` on the TUI pid for 3 s (the real stall)
and confirm no `interrupted` and one `dropped attach client ... send
timeout` line in `mobile.log`; cell E = phone daemon still receives
projections after PR B (`lop mobile status` + a phone attach).

## 6. Interactions and edge cases

- **Refresh while a gate is parked**: `is_busy()` returns True for
  `_pending_futures` (`owned.py:657-660`) — never retires under a question.
- **Refresh while subagents run but the parent turn ended**:
  `running_subagents() > 0` → busy → no refresh. Correct: the children die
  with `dispose`.
- **Goal loop between iterations**: `_goal_loop.running` → busy.
- **Wake due in 60 s**: term 2 holds; the refresh happens after the wake's
  turn settles. A schedule with a 60 s recurrence never refreshes while it
  fires — acceptable; the alternative (retire and let the supervisor re-spawn
  for the wake) is what `WARM_WINDOW_S` exists to avoid, and the stale
  runtime is only stale until the recurrence pauses.
- **Two viewers on one runtime**: both get `retiring`; both re-engage;
  `engage_runtime`'s lease arbitration (`launch.py:447-462`) makes one the
  winner and the other binds to it — the existing N-contender path.
- **`lop-update` to a build that fails to construct**: old runtime is gone;
  `engage_runtime` reports the child's own reason (`launch.py:504-516`) as
  today for any spawn. No worse than a cold start after the update.
- **Refresh racing `/stop`**: `stop` op → `request_stop` → `stop.set()`;
  reaper's `stop.is_set()` check short-circuits. `retiring` never sent after
  `stopping`.
- **Old viewer + new runtime**: viewer ignores `retiring`, chases the record
  for 8 s, goes cold, paints "runtime exited" — today's behaviour for any
  idle exit while attached. Self-limiting: the viewer is stale, and notice A
  told it to `/reload`.

## 7. Risks to watch during rollout

1. **Spawn storms.** 16 runtimes × eager re-engage. Settle + stagger bound
   it to ~1/s; watch `mobile.log` for `engage: spawning` bursts after the
   first `lop-update` post-merge. If it is still lumpy, `BUILD_STAGGER_S`
   is the knob (not env-tunable on purpose, like `WARM_WINDOW_S`).
2. **A half-written install passing the settle check.** `uv tool install
   --force` on this host takes 3–8 s; `BUILD_SETTLE_S = 10` plus the marker
   being written *after* the install (`lop-update:291-292`) gives margin. A
   successor that imports a torn tree fails at construction and the viewer
   shows the engage error — loud, not silent. If observed, raise the
   settle.
3. **Deferred abort hides a real death for up to 8 s.** During recovery the
   band says `working` for a runtime that may be gone. Today it says
   `interrupted` for one that is not. The 8 s bound is `COLD_FALLBACK_S`;
   the go-cold path still paints the abort. Accept.
4. **Desktop surface reading projections.** §4.2 assumes it does not; the
   coder verifies with grep + `test_desktop_*` before landing. If it does,
   scope the recipient filter to `surface == "terminal"`.
5. **`last_turn_outcome` semantics** for a turn that ended in compaction
   continuation: set in `_run_turn_pipeline`'s finally, which is after the
   whole logical turn (start/end pair) — one value per user prompt. Pin it.
6. **Busy hook and `_disposing`**: `is_busy()` returns False under
   `_disposing` (`owned.py:633-636`), so a publish from the hook during
   dispose writes `busy: false` right before the record is unpublished —
   harmless, and `set_busy` de-dupes.

## 8. PR split

**PR A — `fix(runtime): idle runtimes refresh themselves onto the build on
disk`**: §3 entirely. Files in §3.6. Tests 1–15. Notice copy is the only
user-visible surface (design round required: D-findings on C′ and on the
no-notice refresh decision). No band/layout change. Risk: medium (a new exit
path for runtimes) — QA cells A–C.

**PR B — `fix(runtime): keep viewers bound across transient socket drops`**:
§4 entirely. Files in §4.6. Tests 16–23. No user-visible copy change
(removes a false notice; adds none) — UX round on the flow (mid-wait
disconnect no longer paints `interrupted`), no design round. Risk: low-medium
(recipient filtering touches the phone path; test 20 and QA cell E cover
it).

Order: **B first** if only one can ship in the window — it removes the
"interrupted" false paint the operator saw in several sessions and is
independent of A. A depends on nothing in B, but its e2e 13 asserts no
`interrupted` text, which B makes robust against a viewer stall during the
re-engage. Both PRs: gates per AGENTS.md (flake8 / black==26.1.0 /
isort==5.13.2 / pyright over the whole tree, unit suite via
`.venv/bin/python`, `tests/e2e -m e2e -n0` with `env -u NO_COLOR
TERM=xterm-256color`), then the standing reviewer + QA rounds; every fork or
TUI boot in tests unsets inherited `CMUX_*`.

## 9. What I would NOT do

- Bump `PROTOCOL_VERSION` — every frame and op here is additive and
  ignorable by an older peer.
- Add a fourth term to `_should_exit` — a quiet exit and an announced
  retirement are different contracts; mixing them makes an old viewer read
  a refresh as death.
- Overload `stopping` for the refresh — it parks the viewer in "stopped".
- Make the TUI self-reexec for notice A in this PR.
- Touch the `wait` tool.
