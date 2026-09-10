# Design: give `/btw` a deadline of its own, and stop calling a timeout a disconnect

Status: proposal (architect). Scope: two shippable slices, one deferred.
No `pyproject.toml` version bump — the release owner handles that.
All line references are against `origin/main` @ `036d9e76`.

## 1. The problem as found in the code

A user opens `/btw` during a long turn and the card paints
`! owner connection lost:` — trailing colon, no reason. Two independent
mechanisms meet:

**A. An ack timeout is misclassified as a lost connection.** Both request
helpers wrap their wait in an `OSError` arm:

* `mobile/attach_client.py:508` — `return await asyncio.wait_for(future, timeout=ACK_TIMEOUT_S)`
* `mobile/attach_client.py:531` — the same wait in `_request_payload`
* `mobile/attach_client.py:509-511` / `532-534` —
  `except (ConnectionResetError, BrokenPipeError, OSError) as exc: raise ConnectionError(f"owner connection lost: {exc}")`

On Python 3.11+ `asyncio.TimeoutError is TimeoutError` and `TimeoutError`
subclasses `OSError`, so the timeout is swallowed by the arm meant for dead
sockets. Confirmed on this interpreter:

```
asyncio.TimeoutError is TimeoutError: True
str(TimeoutError()): ''
```

An empty `str(exc)` interpolated into `f"owner connection lost: {exc}"` is the
observed string, character for character.

**B. The deadline is sized for the wrong kind of work.** `ACK_TIMEOUT_S = 15.0`
(`attach_client.py:53-57`) documents itself as long enough for "a turn-boundary
op (prompt acquires the turn lock)". It is a **control-plane** budget, and one
shared by all 19 ops that route through `_request`.

`complete_aside` is not a control-plane op. It is a full provider round trip
carrying the whole conversation:

* `tui/app.py:28660` — the aside worker awaits `session.complete_aside(...)`
* `session/remote.py:4987-5000` — `RemoteSession.complete_aside` → `client.complete_aside`
* `mobile/attach_client.py:765-766` — `return await self._request("complete_aside", turns=turns)`
* `session/runtime/server.py:2141-2148` — dispatch awaits the handle
* `mobile/tui_handle.py:528-536` — `asyncio.run_coroutine_threadsafe(session.complete_aside(messages), owner_loop)`, unbounded, no timeout, no cancellation

For calibration, the provider layer's own budget for *silence between chunks of
a stream already in flight* is `STREAM_READ_TIMEOUT_S = 180.0`
(`providers/clients.py:1874`), and its docstring notes a reasoning model "emits
nothing at all until it has finished thinking, which is legitimately minutes".
The client gives this whole round trip **15 seconds**. Exceeding it is normal
operation, not a fault — which is why the bug is "intermittent, reproducible
under load" rather than rare.

Blast radius is every terminal: since v0.45.0 every interactive `lop` TUI is a
`RemoteSession` viewer, so every `/btw` everywhere takes this path.

### 1.1 Two corrections to the reported mechanism

Both change what the fix should be, so they are stated rather than buried.

**Finding 5's file does not exist.** There is no `local_operator/mobile/server.py`
on `origin/main`. The serial reader loop is
`session/runtime/server.py:1372-1382`:

```python
while not self._closed.is_set():
    line = await reader.readline()          # 1373
    ...
    await self._on_request(frame, conn)     # 1382
```

The finding itself is correct — dispatch is awaited *inline* in the read loop,
so a running `complete_aside` blocks every other op on that connection. Only
the citation was wrong.

**The head-of-line block is not caused by the client's timeout, and a longer
deadline does not extend it.** This is the load-bearing correction. When the
client gives up at 15 s it pops its future (`attach_client.py:512-513`,
`535`) and returns — but it sends the owner nothing. The owner is still parked
on line 1382 for the *entire* provider call regardless. So today the socket is
already blocked for the full duration; the 15 s timeout buys no unblocking
whatever, it only guarantees the user sees a false error while the block
continues.

The brief's objection — "a longer timeout alone leaves the socket head-of-line
blocked for a minute" — is true, but it is equally true *today*, at 15 s. The
block is owned by the owner's inline await, and it is a real but **separate,
pre-existing** defect. Raising the client deadline is therefore not a trade
against head-of-line; it is strictly better than the status quo on both axes:

| | socket blocked for | user sees |
|---|---|---|
| today | full provider call | false error at 15 s |
| with a deadline | full provider call | the answer |

**`deadline_s` at `attach_client.py:834` is not a per-op seam.** It is a
parameter of the module-level `continue_command`, forwarded to
`engage_runtime`. `_request` has no deadline parameter today.

### 1.2 `/btw` is not the only op with this shape

`/compact` from a viewer routes `client.slash("compact", "")`
(`session/remote.py:5191`) through the same `_request` 15 s budget. The owner's
summarization is a provider call the codebase itself measures at **"20-50 s on
a large context"** (`session/session.py:447`).

Its failure is quieter than `/btw`'s: `compact_now` catches `ConnectionError`
(`remote.py:5192`) and returns
`CompactionOutcome(False, "unavailable", self._unavailable_reason())`, so a
timeout is reported to the user as *the owner being unavailable* — no dangling
colon, but a wrong diagnosis of a healthy owner.

This is **not** in scope for slice 1: it is a different op with a different
surface and needs its own verification of the owner-side path (an
advisor-triggered pass runs in the background against a snapshot, so the inline
`/compact` cost needs measuring before a number is chosen). Slice 1 makes it a
one-line fix later by putting the `deadline_s` seam on `_request` — which is
the argument for adding the parameter rather than hard-coding a branch for
`complete_aside`.

## 2. Error taxonomy

### What must be preserved

`ConnectionError` from `_request` is load-bearing: callers read it to decide
whether to redial. Auditing every catch that can see it:

* `session/remote.py:2132` — `except (ConnectionError, OSError, TimeoutError)` (bind retry)
* `session/remote.py:4418`, `4506` — `except (ConnectionError, OSError, TimeoutError)` (redial / takeover)
* `session/remote.py:4571`, `4601`, `4647` — `except (ConnectionError, RuntimeError)` (capability absent → `return False`)
* `session/remote.py:5183`, `5192` — `except ConnectionError` **alone** (routed slash, `compact_now`)
* `mobile/attach_client.py:424` — `except ConnectionError` alone (pump)
* `server/routes/desktop_sessions.py:214` — `except ConnectionError` alone

The two `ConnectionError`-only sites at `remote.py:5183/5192` are the
constraint. Re-raising an ack timeout as a plain `TimeoutError` would sail past
both, and a timeout on `/compact` would escape as a raw exception instead of
`CompactionOutcome(False, "unavailable", ...)`. **Changing the exception type to
`TimeoutError` is not a safe refactor.**

The aside call site itself is permissive — `tui/app.py:28825` catches bare
`Exception` and renders `str(error)` — so the *message* is free there, but the
*type* is not free repo-wide.

### Recommendation: a dual-inheritance exception

```python
class OwnerAckTimeout(ConnectionError, TimeoutError):
    """The owner is alive; it did not answer THIS request in time."""
```

Verified on this interpreter — MRO
`OwnerAckTimeout → ConnectionError → TimeoutError → OSError → Exception`, and
`isinstance` is `True` for all four. Every catch site above keeps its current
behaviour with **zero** caller edits, while any code that wants to tell the two
apart can ask `isinstance(exc, OwnerAckTimeout)`.

User-visible text, both non-empty by construction (no dangling colon is
*possible*):

* ack timeout → `owner did not answer 'complete_aside' within 180s`
* real disconnect → `owner connection lost: <errno text>`, unchanged

Rejected: **message-only fix** (keep `ConnectionError`, just interpolate
something non-empty). It hides the dangling colon without making the two
conditions distinguishable, so the next reader still cannot tell a wedged owner
from a slow one, and finding 9's test gap gets a test asserting on prose. It is
smaller by about four lines and worse.

Rejected: **plain `TimeoutError`**. Cleanest taxonomy, breaks
`remote.py:5183/5192` as shown above.

## 3. Deadline policy

Four candidates, weighed against the corrected finding 5.

**(a) Per-op deadline.** `_request` grows `deadline_s`, defaulting to
`ACK_TIMEOUT_S`; `complete_aside` passes a provider-shaped one. Client-only,
no protocol change, no owner change. The other 18 ops are untouched because the
default is unchanged.

**(b) Real streaming.** `remote.py:4998-5003` already fakes it — it calls
`on_delta` once with the settled answer — and the card is built to stream. Best
end-state UX, and progress on the socket. But it needs an out-of-band
owner→client frame, owner-side plumbing of `session.complete_aside(on_delta=)`,
capability negotiation, and skew handling. Not a now-fix.

**(c) Heartbeat / in-progress frame.** Resets the client deadline. Solves less
than (b) for nearly the same protocol cost, and invents a second progress
mechanism beside the `event`/`frontend_update` channel that already exists.
Rejected as a second way of doing an existing thing.

**(d) PR #678 shape — ack admission, answer out-of-band.** The right *final*
architecture, and the one that genuinely fixes head-of-line: `prompt` already
returns as soon as the turn is admitted (`mobile/tui_handle.py:340-355`,
`admitted` future). Applying it to `complete_aside` means a new op contract, a
delivery frame, a correlation id and a skew story.

**Recommend (a) now, (d)+(b) as the follow-up.** (a) fixes 100% of the reported
symptom, touches one file, and — per §1.1 — costs nothing in head-of-line terms
because that block already lasts the full provider call. (d) is the correct fix
for the *socket* defect, which is real but is not what the user reported.

**The number: 180 s**, named `ASIDE_DEADLINE_S`, grounded on
`STREAM_READ_TIMEOUT_S = 180.0` rather than picked. Shorter would fire while
the provider layer still considers the stream healthy — the client would kill
requests the layer below is deliberately waiting on. Much longer would out-wait
the layer whose job is detecting the stall, so a genuinely wedged owner would
stop surfacing. Matching it means the client gives up when, and only when, the
layer that can actually see the stream has also given up.

## 4. Cancellation and the orphaned reply

Today: the client pops its future in `finally` (`attach_client.py:512-513`),
the owner finishes and sends its answer, and the pump finds no future
(`attach_client.py:411-414`) and drops a paid-for completion in silence.

**What the minimum actually is.** With a 180 s deadline the orphan stops
happening in the reported scenario — the request completes and is delivered.
The orphan that remains is the user closing the card
(`tui/app.py:28658-28666`, `_close_aside` cancels the worker group), and that
one is *unchanged from today*: the owner was never told then either. So the
minimum that avoids paying for an unread call is **not** in this fix's path —
it is the same follow-up as head-of-line, because the owner-side handle you
need in order to cancel is exactly the handle slice (d) creates.

What is worth doing now is one line of diagnosability: the pump's silent drop
becomes a `logger.warning` naming the orphaned `req`. A completed provider
answer being discarded should not be invisible.

Deferred, deliberately: a `cancel_aside` op. It is correct, it is not free
(new op, capability negotiation, an owner-side task handle that does not exist
yet), and it protects spend rather than the user's reported symptom.

## 5. Compatibility

**Slice 1 is client-only.** No new op, no new frame, no new field, no
`PROTOCOL_VERSION` bump. The wire is byte-identical. A patched viewer against
any owner build — older or newer — behaves exactly as before except that it
waits longer before giving up, and names the reason correctly when it does.
There is no skew case to degrade *to*. This is the single strongest argument
for this scope.

For the deferred slice, the repo's established seams apply and should be used
rather than reinvented: an additive field on a known op is tolerated by
`validate_control_frame` (`mobile/types.py:187-190` validates only `turns`); a
new op gets `error: unknown op` from an old owner
(`session/runtime/server.py:2257`, `2492`), which callers already handle
(`remote.py:4571` treats it as capability-absent); and a capability string in
`record.capabilities` is the pre-dial probe, as `completion-ack-v1` does at
`session/runtime/server.py:668-671` / `attach_client.py:218`.

## 6. Slicing plan

Slices 1 and 2 ship now and share no files. Slice 3 is scoped, not scheduled.

### Slice 1 — the transport fix (ship now)

Files (exclusive owner):
* `local_operator/mobile/attach_client.py`
* `tests/unit/mobile/test_attach_client.py`

Changes:
1. Add `OwnerAckTimeout(ConnectionError, TimeoutError)` with a docstring saying why both bases (§2).
2. Add `ASIDE_DEADLINE_S = 180.0` next to `ACK_TIMEOUT_S:53-57`, with the `STREAM_READ_TIMEOUT_S` derivation in the comment.
3. `_request_frame:491` and `_request_payload:514` grow `deadline_s: float = ACK_TIMEOUT_S`; `_request` forwards it.
4. In both, catch `TimeoutError` **before** the `OSError` arm and raise `OwnerAckTimeout(f"owner did not answer {op!r} within {deadline_s:.0f}s")`. Ordering is the fix — an `OSError` arm placed first re-swallows it.
5. `complete_aside:765` passes `deadline_s=ASIDE_DEADLINE_S`. No other op changes.
6. Pump `:411-414`: log a warning when a reply arrives for an unknown `req`.
7. Drop the redundant `self._pending.pop(req, None)` at `:533` — `finally:535` already does it.

Evidence this slice must produce:
* A test that drives a fake owner which **never answers** a request, and asserts the raised exception is `isinstance` of all of `OwnerAckTimeout`, `ConnectionError`, `TimeoutError`; that `str(exc)` is non-empty; and that it does **not** contain `owner connection lost` (finding 9's gap, closed).
* A test that a real reset still raises `owner connection lost: <something non-empty>` — proving the disconnect path is untouched.
* A test that `complete_aside` waits past `ACK_TIMEOUT_S` and that another op still gives up at 15 s — proving the default is unchanged for the other 18.
* A test that a late reply for a popped `req` is logged, not silently dropped.
* `.venv/bin/python -m pytest tests/unit/mobile tests/unit/session -q` green, with the command and counts pasted.

### Slice 2 — never render an empty error (ship now)

Independent defence in depth for finding 8: any exception with an empty `str()`
renders as a dangling colon or a blank error, and slice 1 only fixes the one
that is known to produce one.

Files (exclusive owner):
* `local_operator/tui/app.py` (`:28828` `panel.fail_answer(generation, str(error))`, `:28833` `target.fail(str(error))`)
* `local_operator/tui/widgets/settings_view.py` (`:1966`, `:2530`)
* the matching tests under `tests/unit/tui/`

Change: one shared helper that falls back to the exception's class name when
`str(exc)` is blank, used at all four sites.

Evidence: a test that raising an exception whose `str()` is `''` through the
aside worker renders a non-empty message and no trailing `:` — asserted on the
rendered card text, not on the helper in isolation.

Note the file boundary: slice 1 owns `mobile/attach_client.py`, slice 2 owns
`tui/*`. Neither touches `session/remote.py`, which needs no change at all.

### Slice 3 — the socket fix (follow-up, NOT this change)

Files: `local_operator/session/runtime/server.py`, `local_operator/mobile/tui_handle.py`, `local_operator/mobile/types.py`, plus tests.

Scope: dispatch `complete_aside` off the reader loop (PR #678 shape) so it
stops head-of-line blocking `server.py:1382`; deliver the answer out of band;
add `cancel_aside`; then real streaming (§3b) reusing the delta channel
`remote.py:4998-5003` currently fakes. Behind a capability string per §5.

Deferred honestly: this is the *correct* fix for the socket, and it is not what
the user reported. It changes the protocol, so it carries skew risk that slices
1 and 2 provably do not.

## 7. Risks to watch during rollout

1. **A genuinely wedged owner now takes 180 s to surface on `/btw`.** Accepted
   and bounded to one op — every other op still fails at 15 s, so a dead owner
   is still detected in 15 s by the next thing the TUI does. Worth watching in
   feedback anyway, because it is the one behaviour that gets *slower*.
2. **The `except` ordering is the whole fix.** A later edit that reorders those
   arms, or adds an `OSError` arm above the `TimeoutError` one, silently
   restores the bug. The test in slice 1 must fail if the arms are swapped —
   that is what makes it a regression test rather than a unit test.
3. **Head-of-line stays.** Anyone reading the shipped fix may believe the
   socket defect was addressed. It was not; §1.1 and slice 3 say so in the
   repo, which is why this document is part of the deliverable.
4. **`/compact` still misreports** (§1.2) — a timeout there reads as "owner
   unavailable". Untouched by this change, and now cheap to fix.
5. **`OwnerAckTimeout` inherits `OSError`,** so a future `except OSError`
   anywhere on this path will catch it. That is a deliberate inherited property
   (it is what preserves today's callers), not an oversight.
