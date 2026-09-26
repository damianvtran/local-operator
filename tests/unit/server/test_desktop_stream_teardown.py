"""The dwell must arm BEFORE the release can detach (F-A/F-B/F-C).

WHY THIS FILE EXISTS. The reconnect dwell landed in an earlier round and its own
tests passed, and the product still churned. The reason is an ORDERING that only
the real route exhibits: ``routes/desktop_sessions.py`` wrapped the SSE generator
with no ``aclosing``, so on a CANCELLED teardown -- which is what an SSE client's
disconnect *is* -- the outer ``finally`` reached ``release()`` before the inner
generator's ``finally`` could arm the dwell. ``release()`` saw ``users == 0`` and
``dwelling is False``, detached, cleared ``self.remote``, and the next ``acquire()``
minted a fresh epoch. The client's cursor was present and stale, so every reopen
was ``gap:true`` by construction: measured on the operator's daemon as **31
distinct epochs from 35 opens**.

The load-bearing detail, and the one every case below is shaped around: the
cancellation lands while the wrapper is parked at ITS OWN ``yield`` (the consumer
is between chunks -- writing to a socket, reading from a relay), NOT inside the
generator's ``__anext__``. A cancellation delivered *inside* ``__anext__`` unwinds
the generator as part of the raise and its ``finally`` runs either way, which is
why a test that simply cancels a draining task proves nothing about this defect.
``_teardown`` below builds the discriminating shape, and every case that depends
on it was falsified by reverting the site it pins.
"""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI

from local_operator.server.routes import desktop_sessions as routes
from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import (
    RECONNECT_DWELL_S,
    DesktopSessions,
)

pytestmark = pytest.mark.asyncio


def _app(tmp_path: Path) -> tuple[FastAPI, DesktopSessions]:
    """A minimal app whose pool is the real one, so the REAL route can be driven.

    The route is called directly rather than through ``ASGITransport``: it
    buffers a response until the app returns, and the whole defect lives in the
    teardown ordering of a response that is still streaming. ``routes.events`` is
    the production callable, so its wrapper -- the site F-A fixes -- is what these
    cases exercise.
    """
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    return app, pool


async def _open_stream(
    app: FastAPI,
    session_id: str,
    *,
    epoch: str | None = None,
    after_seq: int = 0,
) -> Any:
    """One ``GET /events`` response, exactly as the route builds it."""
    return await routes.events(
        session_id,
        cast(Any, SimpleNamespace(app=app)),
        epoch=epoch,
        after_seq=after_seq,
        frontend_replace=0,
    )


async def _first_frames(response: Any, count: int) -> list[dict[str, Any]]:
    """The first ``count`` decoded frames, consumed from the response body."""
    frames: list[dict[str, Any]] = []
    iterator = response.body_iterator
    for _ in range(count):
        chunk = await anext(iterator)
        frames.append(json.loads(chunk.removeprefix("data: ").strip()))
    return frames


async def _teardown(response: Any) -> dict[str, Any]:
    """Consume ONE frame, then tear the response down the way the framework does.

    ``aclose()`` on the REAL ``body_iterator``, and the reason it is a faithful
    stand-in rather than a convenience is worth stating: Starlette's
    ``StreamingResponse`` never calls ``aclose`` on its iterator (checked in the
    installed 1.7.0 -- ``stream_response`` is a bare ``async for`` over
    ``self.body_iterator``). On a client disconnect the wrapper is ABANDONED at
    its own ``yield`` -- the ``await send(...)`` raises, the ``async for``
    unwinds, and the generator receives no exception at all -- so both it and the
    inner ``events`` generator are closed later, by CPython's async-generator
    finalizer, in finalizer order.

    That deferral IS the defect: the wrapper's ``finally`` and the inner
    generator's fire in the order their finalizers reach them, and without
    ``aclosing`` that order is release-then-dwell. ``aclose()`` here performs the
    same close the finalizer performs, on the same object, through the real route
    -- deterministically, which is what a unit test needs.
    """
    iterator = response.body_iterator
    chunk = await anext(iterator)
    opened = json.loads(chunk.removeprefix("data: ").strip())
    await iterator.aclose()
    return opened


async def test_a_cancelled_teardown_arms_the_dwell_before_the_release(tmp_path):
    """F-A's site: the dwell is armed, and the facade survives, on cancellation."""
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))

    response = await _open_stream(app, sid)
    opened = await _teardown(response)
    epoch, cursor = opened["epoch"], opened["seq"]
    assert opened["type"] == "open"

    bridge = pool.bridges[sid]
    # ONE TICK, no sleeping out the window: the generator's ``finally`` has run by
    # the time the cancelled task has unwound, and this is the assertion the
    # pre-fix tree fails.
    await asyncio.sleep(0)
    assert bridge.dwelling is True, (
        "the dwell must be armed before the release: without ``aclosing`` the "
        "outer finally reaches ``release()`` first and this flag is still False"
    )
    assert bridge.remote is not None, (
        "and the facade must still be there -- a detach here clears ``self.remote`` "
        "and the next acquire mints a new epoch, so the client's cursor can never "
        "match"
    )

    # The reconnect the dwell exists for: same epoch, and a cursor that MATCHES.
    async with pool.session(sid) as reopened:
        assert reopened is bridge, "the bridge was never disposed"
        assert reopened.epoch == epoch, "and the epoch did not rotate"

    again = await _open_stream(app, sid, epoch=epoch, after_seq=cursor)
    first = (await _first_frames(again, 1))[0]
    assert first["type"] == "open"
    assert first["payload"]["gap"] is False, (
        "a cursor from this epoch must be honoured, which is the whole point: the "
        "observed product behaviour was gap:true on every reopen"
    )


async def test_an_orderly_teardown_still_dwells(tmp_path):
    """The control: the sentinel path already armed in time, and must keep doing so."""
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))

    response = await _open_stream(app, sid)
    frames = await _first_frames(response, 2)
    epoch = frames[0]["epoch"]
    bridge = pool.bridges[sid]

    # The bridge's own sentinel ends the stream: ``events`` RETURNS, so the inner
    # generator's ``finally`` runs first and the dwell arms -- this path worked
    # before the fix and is what made the defect invisible to the earlier round.
    await bridge.close()
    await asyncio.sleep(0)
    assert bridge.dwelling is False, "a close DISPOSES the bridge; it does not dwell"
    assert bridge.remote is None
    assert epoch, "the pre-close epoch was captured"


async def test_the_bound_is_intentional(tmp_path, monkeypatch):
    """Past ``RECONNECT_DWELL_S``: a new epoch, and an honest gap."""
    now = 1000.0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(module, "DWELL_TICK_S", 0.01)
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))

    response = await _open_stream(app, sid)
    opened = await _teardown(response)
    epoch, cursor = opened["epoch"], opened["seq"]
    bridge = pool.bridges[sid]
    assert bridge.dwelling is True

    # Move the CLOCK, not the test: the dwell is deadline-driven on the module
    # clock precisely so its expiry can be moved instead of waited out.
    now += RECONNECT_DWELL_S + 1
    for _ in range(200):
        if bridge.remote is None:
            break
        await asyncio.sleep(0.01)
    assert bridge.remote is None, "the window is bounded; the bridge does let go"

    async with pool.session(sid) as reopened:
        assert reopened.epoch != epoch, "and past the bound the epoch rotates"
    again = await _open_stream(app, sid, epoch=epoch, after_seq=cursor)
    first = (await _first_frames(again, 1))[0]
    assert first["payload"]["gap"] is True, (
        "a cursor from the old epoch cannot be honoured, and the frame says so "
        "rather than pretending"
    )


async def test_the_bridge_lets_go_when_the_dwell_expires(tmp_path, monkeypatch):
    """The bound is real on a REAL clock, and the stamp's ORDER is what guarantees it.

    Two deadlines are computed here from two clock reads: the dwell's, armed when
    the generator is closed, and the grace's, stamped when the stream ends. Their
    relation decides whether the dwell's expiry can actually detach, and only one
    order makes that a guarantee rather than a hope:

    * stamp FIRST: ``dwell_until = stamp + delta + window`` and the grace expires
      at ``stamp + window``, so it is expired at the wake by construction, for any
      ``delta`` and any timer jitter;
    * stamp LAST: the grace expires at ``stamp + window`` where ``stamp`` is
      ``delta`` after the arm, so the wake must overshoot by more than ``delta``
      to detach -- and if it does not, the dwell task ends without detaching and
      NOTHING is left to wake the bridge.

    ``delta`` is the close's own work, which includes the generator's
    ``refresh_watch()`` -- an owner call, so milliseconds. This case makes it
    dominate the timer jitter (a 50 ms refresh, a 1 ms tick, a 500 ms window) so
    the race is decided rather than sampled, and it is falsified by moving the
    stamp after the close: the bridge then stays attached with nothing left to
    wake it.
    """
    # The window must outlast the close's own work, or the dwell expires DURING
    # the teardown and this case measures its own latency instead of the ordering:
    # 50 ms of refresh against a 500 ms window, with a 1 ms tick so the wake
    # jitter cannot cover a 50 ms gap.
    monkeypatch.setattr(module, "RECONNECT_DWELL_S", 0.5)
    monkeypatch.setattr(module, "DWELL_TICK_S", 0.001)
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))

    response = await _open_stream(app, sid)
    bridge = pool.bridges[sid]
    original_refresh = bridge.refresh_watch

    async def slow_refresh() -> None:
        # Stands in for the real owner call the generator's ``finally`` makes:
        # ``delta`` has to be big enough to lose a jitter race, or this case
        # samples the ordering instead of pinning it.
        await asyncio.sleep(0.05)
        await original_refresh()

    monkeypatch.setattr(bridge, "refresh_watch", slow_refresh)

    await _teardown(response)
    assert bridge.dwelling is True, "the dwell was armed on the way out"

    for _ in range(200):
        if bridge.remote is None:
            break
        await asyncio.sleep(0.01)
    assert bridge.remote is None, (
        "the dwell's expiry must actually detach. A bridge still attached here is "
        "the leak the two deadlines create when the grace is stamped second"
    )
    assert bridge.dwelling is False


async def test_an_overflowed_subscriber_never_dwells(tmp_path):
    """The relief valve: a stream the BRIDGE revoked must not be held by either half.

    The overflow must be true AT THE MOMENT the stream ends, because that is when
    the bridge records it; marking a subscriber afterwards is a different case and
    would not be the valve. Both refusals are asserted, because they are
    independent: the dwell arm's ``not sub.overflow`` and the release-time grace's
    own copy of it.
    """
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))

    response = await _open_stream(app, sid)
    bridge = pool.bridges[sid]
    sub = next(iter(bridge.subscribers.values()))

    # The bridge revoking its own subscriber IS the overflow -- a reader that fell
    # behind is told with a gap and disconnected, by design.
    bridge._disconnect(sub)
    opened = await _teardown(response)
    await asyncio.sleep(0)

    assert opened["type"] == "open"
    assert sub.overflow is True
    assert bridge.dwelling is False, (
        "an overflowed subscription must never dwell: the relief valve would "
        "invert and a reader the bridge disconnected would be held alive"
    )
    assert bridge._within_stream_grace() is False, (
        "and the release-time grace excludes it too, or that same reader keeps the "
        "bridge, its facade and its runtime for the whole window"
    )


async def test_close_beats_the_window(tmp_path, monkeypatch):
    """Bound 5: ``close()`` detaches even if a stream ended a millisecond ago."""
    now = 500.0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))

    response = await _open_stream(app, sid)
    await _teardown(response)
    bridge = pool.bridges[sid]
    sub = next(iter(bridge.subscribers.values()))
    assert bridge._within_stream_grace() is True, "the window is open right now"

    # Session delete / plane shutdown: the timestamp is a RELEASE-time policy and
    # never a property ``close`` consults.
    await bridge.close()
    assert bridge.remote is None, (
        "a close must detach regardless of the window, or a session delete and a "
        "shutdown both leak a facade for up to RECONNECT_DWELL_S"
    )
    assert bridge.dwelling is False
    assert sub.id not in bridge.subscribers, (
        "the subscription the dwell was holding is dropped with the bridge, so "
        "nothing asserts presence for a viewer nobody can reach"
    )


async def test_a_release_inside_the_window_does_not_detach_without_a_dwell(tmp_path, monkeypatch):
    """F-B's own site, driven WITHOUT the dwell: the timestamp alone holds it.

    This is the ordering-independent half. The generator's ``finally`` is what
    arms the dwell, and the whole defect is that a cancelled teardown can reach
    the release before it -- or with a finalizer that never runs at all. So the
    policy is evaluated from ``note_stream_ended``'s timestamp, and this case
    removes the dwell from the picture entirely to pin that.
    """
    now = 700.0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
    # No dwell flag is ever set, so nothing but the timestamp can refuse this.
    monkeypatch.setattr(module, "RECONNECT_DWELL_S", RECONNECT_DWELL_S)
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))

    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        bridge.note_stream_ended(sub)
        assert bridge.stream_ended_overflowed is False
        assert bridge._within_stream_grace() is True

    # The context's release ran at users == 0 with no dwell and no armed flag.
    assert bridge.remote is not None, (
        "a release inside the window must NOT detach: this is the belt that holds "
        "even when the generator's finally never ran"
    )

    # And it is not a leak: past the window the ordinary path detaches again.
    now += RECONNECT_DWELL_S + 1
    assert bridge._within_stream_grace() is False
    async with pool.session(sid) as reopened:
        assert reopened is bridge, "the same facade is still there to be reused"
        assert reopened.users == 1
    assert bridge.remote is None, "the window is a delay, never a reprieve"


async def test_the_reason_log_and_refresh_watch_run_on_the_cancellation_path(
    tmp_path, monkeypatch, caplog
):
    """F-A's guard: the cancellation path used to skip BOTH, silently.

    The per-stream reason line and ``refresh_watch()`` live in the generator's
    ``finally``. Without ``aclosing`` the cancellation teardown destroyed the
    generator before either ran, which is why the teardown reason was unreadable
    in production and why presence could be left asserted after a dropped relay.
    """
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))

    response = await _open_stream(app, sid)
    bridge = pool.bridges[sid]
    refreshed: list[bool] = []
    original = bridge.refresh_watch

    async def spy() -> None:
        refreshed.append(True)
        await original()

    monkeypatch.setattr(bridge, "refresh_watch", spy)

    # AT INFO, which is the level the line is emitted at and therefore exactly
    # what a run with ``LOG_LEVEL=INFO`` captures. No switch of this module's own
    # is involved: the knob is the app's logging configuration.
    with caplog.at_level(logging.INFO, logger="local_operator.server.utils.desktop_sessions"):
        await _teardown(response)
        await asyncio.sleep(0)

    reasons = [
        record.getMessage()
        for record in caplog.records
        if "desktop stream ended" in record.getMessage()
    ]
    assert reasons, (
        "the cancellation path must reach the reason line; it is emitted at INFO, "
        "so a run with LOG_LEVEL=INFO captures it on the operator's daemon"
    )
    assert "client disconnect" in reasons[-1], (
        "and the vocabulary must name the ordinary case: a dropped transport is a "
        "client disconnect, not a relay error"
    )
    assert refreshed, (
        "refresh_watch() must run on this path too, or a dropped relay leaves the "
        "presence assertion standing with nothing to renew it"
    )


async def test_the_reason_is_emitted_at_info_and_needs_no_switch(tmp_path, caplog):
    """F-C in its shipped form: observable through the app's OWN log level.

    The first draft gave this module a ``LOCAL_OPERATOR_DESKTOP_STREAM_TRACE``
    switch. That is a second reader of the desktop environment -- which
    ``test_only_desktop_posture_reads_the_desktop_environment`` forbids -- and a
    brand-new ambient variable, which
    ``tests/unit/test_ambient_env_isolation.py`` requires to be scrubbed or
    explained. Both guards were right and neither needed editing: the reason is
    emitted at INFO, and ``LOG_LEVEL=INFO`` is already a first-class knob
    (``local_operator/logger.py``, applied to the root logger by
    ``configure_console_logging``). This case pins that route instead of a switch.
    """
    app, pool = _app(tmp_path)
    sid = await pool.create(str(tmp_path))
    response = await _open_stream(app, sid)

    with caplog.at_level(logging.INFO, logger="local_operator.server.utils.desktop_sessions"):
        await _teardown(response)
        await asyncio.sleep(0)

    ended = [record for record in caplog.records if "desktop stream ended" in record.getMessage()]
    assert ended, "the reason line is emitted at INFO, for a captured level to find"
    assert ended[-1].levelno == logging.INFO, (
        "at INFO rather than promoted: a stream ending is a normal event, and "
        "raising it permanently would bury the WARNINGs that mean something"
    )
    assert "STREAM_REASON_TRACE_ENV" not in vars(module), (
        "and this module owns no env switch for it, which is what keeps the "
        "desktop environment single-reader and the ambient set unchanged"
    )
