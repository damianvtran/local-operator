"""The desktop stream's OPEN handshake, at the seam D1 changes.

T3 of ``docs/design/desktop-stream-gap-storm.md`` §6, and the fixture the brief
asks for.

WHY THE ROUTE'S ORDERING IS PINNED AS SOURCE RATHER THAN DRIVEN. ``ASGITransport``
buffers a response until the app returns, so an SSE route cannot be driven
through it: the whole point of the ordering is that the bridge is acquired BEFORE
any response exists, and a client that only sees the completed response cannot
observe that. The route's two hand-offs on either side of the response are
therefore asserted as source text -- the same instrument the route's own comment
names -- and the handshake BETWEEN them is driven directly on the bridge, which is
where D1's change lives.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import DesktopSessions

pytestmark = pytest.mark.asyncio

ROUTE = (
    Path(__file__).resolve().parents[3]
    / "local_operator"
    / "server"
    / "routes"
    / "desktop_sessions.py"
)


def test_the_route_acquires_before_any_response_exists() -> None:
    """The ordering D1 depends on, and the promise it protects.

    `host(request).session(...)` -- which ACQUIRES the bridge -- runs before the
    route returns anything, and `bridge.subscribe(...)` runs after it and still
    inside the `async with errors(request)` block. So an invalid identity or a
    full subscriber table is a JSON status rather than a 200 followed by a broken
    stream, which is why D1 did NOT take the "subscribe after the snapshot"
    route: that would move both refusals past the response start.
    """
    source = ROUTE.read_text()
    entry = source.index('@router.get("/v1/desktop/sessions/{session_id}/events")')
    body = source[entry : source.index("\n@router.", entry + 1)]

    acquire = body.index("host(request).session(session_id, read=True)")
    subscribe = body.index("bridge.subscribe(")
    assert acquire < subscribe, (
        "the bridge must be acquired before the subscription is taken; the "
        "capacity refusal and the move fence both depend on it happening while a "
        "JSON status is still possible"
    )
    assert "await context.__aenter__()" in body[:subscribe], (
        "and the acquire must be AWAITED before the subscribe, not merely ordered "
        "before it in the source"
    )


async def _snapshot_that_bursts(bridge: Any, original: Any, count: int) -> Any:
    """A `snapshot()` that publishes a cold engage's burst while it builds.

    This is the window D1 exists for: the subscriber is registered, nothing has
    read its queue yet, and the frames a cold engage publishes all land before the
    snapshot that supersedes them exists.
    """

    async def snapshot_and_burst() -> dict[str, Any]:
        for n in range(count):
            bridge.publish("event", {"burst": n})
        return await original()

    return snapshot_and_burst


async def test_the_handshake_survives_a_burst_published_while_it_builds(
    tmp_path, monkeypatch
) -> None:
    """The end-to-end form of D1: the stream is STILL LIVE after the snapshot."""
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        original = bridge.snapshot
        monkeypatch.setattr(
            bridge,
            "snapshot",
            await _snapshot_that_bursts(bridge, original, module.REPLAY_COUNT + 50),
        )

        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
        assert (await anext(stream))["type"] == "open"
        snapshot = await anext(stream)
        assert snapshot["type"] == "snapshot"

        assert sub.overflow is False, (
            "a burst published during the handshake must not disconnect the "
            "subscriber whose own engage produced it"
        )
        assert sub.opened is True, "and the window is closed behind it"

        # STILL LIVE: the post-handshake loop answers a new frame rather than
        # having already ended in `gap`.
        bridge.publish("event", {"after": "the handshake"})
        delivered = await asyncio.wait_for(anext(stream), timeout=5)
        assert delivered["payload"] == {"after": "the handshake"}
        await stream.aclose()


async def test_the_same_burst_after_the_handshake_still_ends_in_a_gap(
    tmp_path, monkeypatch
) -> None:
    """The negative twin, and the boundary's other side.

    The same burst, moved to AFTER the handshake, is a slow READER rather than a
    cold subscriber: those frames are newer than the snapshot, evicting them would
    be a real loss, and the honest answer is the relief valve -- `gap`, then
    close.
    """
    monkeypatch.setattr(module, "REPLAY_BYTES", 400)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0)
        assert (await anext(stream))["type"] == "open"
        assert (await anext(stream))["type"] == "snapshot"

        for _ in range(20):
            bridge.publish("event", {"text": "x" * 200})

        assert sub.overflow is True, "an OPENED subscriber that is behind is disconnected"
        assert (await anext(stream))["type"] == "gap"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert not bridge.subscribers, "and the relief valve revoked its subscription"


def test_the_queue_bound_is_replay_count() -> None:
    """One number, both sides of the comparison.

    The pre-open policy says a full queue is EVICTION because every frame in it is
    superseded; the queue's size is `REPLAY_COUNT`, and the replay window the
    reopen is answered from is the same constant. A queue that grew past it -- or
    a replay window that shrank below it -- would make "too much to hold" a
    different quantity from "too old to replay", and the proof attached to the
    eviction would stop being about the frames actually in the queue.
    """
    from local_operator.server.utils.desktop_sessions import DesktopSubscription

    assert DesktopSubscription().queue.maxsize == module.REPLAY_COUNT
    assert json.dumps({"seq": 1})  # the frame-size accounting the bound measures
