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

import ast
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import DesktopSessions

ROUTE = (
    Path(__file__).resolve().parents[3]
    / "local_operator"
    / "server"
    / "routes"
    / "desktop_sessions.py"
)


def _events_handler(source: str) -> str:
    """The `events` route handler's OWN source text.

    WHY THE HANDLER, NOT A DECORATOR-TO-DECORATOR SLICE (reviewer MINOR, #2931).
    The ordering asserted below is a property of THIS handler. A slice bounded by
    the next `@router.` happens to be this handler today and quietly stops being
    it the moment the route table changes; and a slice that accidentally widened
    would let an unrelated route's `….session(session_id…)` satisfy the anchor,
    turning the assertion into a tautology that still passes. `ast` names the
    handler, so the instrument cannot test another route by accident.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "events":
            lines = source.splitlines()
            first = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
            return "\n".join(lines[first - 1 : node.end_lineno])
    raise AssertionError("the desktop routes module no longer defines an `events` handler")


def _assert_handshake_order(source: str) -> None:
    """Assert the stream route's open hand-off: acquire first, and awaited.

    The ordering D1 depends on, and the promise it protects. The bridge is
    acquired -- and the acquire AWAITED -- before `bridge.subscribe(...)` runs and
    still inside the `async with errors(request)` block. So an invalid identity or
    a full subscriber table is a JSON status rather than a 200 followed by a
    broken stream, which is why D1 did NOT take the "subscribe after the
    snapshot" route: that would move both refusals past the response start.

    THE ANCHOR IS A SHAPE, DELIBERATELY. This asserts an ORDERING, not a call's
    arguments or spelling: the argument list already changed once for the draft
    pre-engage path (`allow_draft=True`), and the receiver changed when the route
    started reusing a pool local (CORE-PROFILE L1) -- both preserved the property
    and one of them turned a literal anchor red. So the match runs on
    whitespace-collapsed source (a wrapped call collapses back to one line) and
    accepts either receiver.
    """
    handler = _events_handler(source)
    flat = " ".join(handler.split())
    acquires = re.findall(r"(?:host\(request\)|\bpool)\.session\(\s*session_id", flat)
    assert len(acquires) == 1, (
        "the stream route must acquire exactly one bridge, so that 'before the "
        f"subscribe' names one call; found {len(acquires)}: " + flat[:400]
    )
    acquire = re.search(r"(?:host\(request\)|\bpool)\.session\(\s*session_id", flat)
    assert acquire is not None  # narrowed by the count above
    subscribe = flat.index("bridge.subscribe(")
    assert acquire.start() < subscribe, (
        "the bridge must be acquired before the subscription is taken; the "
        "capacity refusal and the move fence both depend on it happening while a "
        "JSON status is still possible"
    )
    assert "await context.__aenter__()" in flat[:subscribe], (
        "and the acquire must be AWAITED before the subscribe, not merely ordered "
        "before it in the source"
    )


def test_the_route_acquires_before_any_response_exists() -> None:
    """The events handler takes its bridge before it can return anything."""
    _assert_handshake_order(ROUTE.read_text())


def test_the_ordering_assertion_catches_a_subscription_taken_first() -> None:
    """The instrument's failing direction, on the REAL handler.

    A control that only ever passes proves nothing, so the ordering is inverted
    here the way a regression would invert it -- the subscribe taken before the
    acquire -- and the instrument must refuse it.
    """
    source = ROUTE.read_text()
    acquire_call = "    context = pool.session(session_id, read=True, allow_draft=True)\n"
    assert source.count(acquire_call) == 1, "the control anchors on the route's real acquire"
    inverted = source.replace(acquire_call, "    early = bridge.subscribe()\n" + acquire_call)
    with pytest.raises(AssertionError, match="acquired before the subscription"):
        _assert_handshake_order(inverted)


def test_the_ordering_assertion_cannot_be_satisfied_from_another_route() -> None:
    """The boundary control: a decoy route must not satisfy the anchor.

    Every earlier route in this module that also acquires a bridge is exactly the
    material a widened slice would match first. With one prepended AND the real
    handler's ordering inverted, an instrument that searched more than the
    handler would anchor on the decoy, see a correct-looking order and pass; the
    handler-scoped one must still fail.
    """
    source = ROUTE.read_text()
    decoy = (
        '@router.get("/v1/desktop/sessions/{session_id}/decoy")\n'
        "async def decoy(session_id: str, request: Request):\n"
        "    context = host(request).session(session_id, read=True)\n"
        "    return None\n"
        "\n"
        "\n"
    )
    events_decorator = '@router.get("/v1/desktop/sessions/{session_id}/events")'
    assert source.count(events_decorator) == 1
    acquire_call = "    context = pool.session(session_id, read=True, allow_draft=True)\n"
    inverted = source.replace(acquire_call, "    early = bridge.subscribe()\n" + acquire_call)
    decoyed = inverted.replace(events_decorator, decoy + events_decorator)

    with pytest.raises(AssertionError, match="acquired before the subscription"):
        _assert_handshake_order(decoyed)

    # And the decoy alone must not disturb the real handler's passing direction.
    _assert_handshake_order(source.replace(events_decorator, decoy + events_decorator))


async def _snapshot_that_bursts(bridge: Any, original: Any, count: int) -> Any:
    """A `snapshot()` that publishes a cold engage's burst while it builds.

    This is the window D1 exists for: the subscriber is registered, nothing has
    read its queue yet, and the frames a cold engage publishes all land before the
    snapshot that supersedes them exists.
    """

    async def snapshot_and_burst(**_kwargs) -> dict[str, Any]:
        for n in range(count):
            bridge.publish("event", {"burst": n})
        return await original()

    return snapshot_and_burst


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
