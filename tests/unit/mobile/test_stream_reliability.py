"""Phone stream recovery uses snapshots and bounded residency, not event replay."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.requests import Request

from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import (
    MobileDaemon,
    SessionEntry,
    _LatestProjectionQueue,
    build_app,
)
from local_operator.mobile.types import SessionProjection, SessionRecord


def fixture_daemon():
    daemon = MobileDaemon(password="pw", dial_registrants=False)
    record = SessionRecord(
        pid=900301,
        session_id="reliability-fixture",
        control_port=1,
        control_key="synthetic",
        kind="tui",
        conversation_name="fixture",
        cwd="/synthetic",
        model_label="fixture",
    )
    entry = SessionEntry(record)
    entry.projection = SessionProjection(session_id=record.session_id, pid=record.pid, version=1)
    daemon.table.entries[record.pid] = entry
    return daemon, entry


async def response_for(app, path):
    route = next(route for route in app.routes if route.path == path)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [(b"cookie", f"{COOKIE_NAME}={sign_cookie('pw')}".encode())],
            "query_string": b"",
            "path_params": {"session_id": "reliability-fixture"},
        }
    )
    return await route.endpoint(request)


def version(frame):
    return json.loads(frame.split("data: ", 1)[1])["version"]


@pytest.mark.asyncio
async def test_all_projection_producers_share_one_latest_wins_slot():
    queue = _LatestProjectionQueue()
    for number in range(100):
        queue.put_nowait({"version": number, "pending": {"id": "approval"}})
    assert queue.qsize() == 1
    # A late durable rebuild cannot replace a newer live snapshot.
    queue.put_nowait({"version": 1})
    assert queue.get_nowait() == {"version": 99, "pending": {"id": "approval"}}


@pytest.mark.asyncio
async def test_projection_burst_sends_only_trailing_snapshot(monkeypatch):
    from local_operator.mobile import daemon as module

    monkeypatch.setattr(module, "PROJECTION_FRAME_INTERVAL_S", 0.01)
    daemon, entry = fixture_daemon()
    stream = (
        await response_for(build_app(daemon), "/api/sessions/{session_id:str}/events")
    ).body_iterator
    assert version(await anext(stream)) == 1
    queue = next(iter(daemon.table.session_subscribers[entry.record.session_id]))
    for number in range(2, 102):
        queue.put_nowait({"version": number, "session_id": entry.record.session_id})
    assert version(await anext(stream)) == 101
    assert queue.empty()
    await stream.aclose()
    await daemon.close_phone_views()


@pytest.mark.asyncio
async def test_rotation_keeps_attach_watch_and_epoch_until_grace_expires(monkeypatch):
    daemon, entry = fixture_daemon()
    viewer = Mock(connected=True)
    daemon._phone_attaches[entry.record.session_id] = viewer
    daemon.notify_watch_transition = Mock()
    daemon._prune_projection_generation = Mock()
    app = build_app(daemon)
    first = (await response_for(app, "/api/sessions/{session_id:str}/events")).body_iterator
    await anext(first)
    assert daemon.notify_watch_transition.call_args.kwargs == {"watching": True}
    daemon.notify_watch_transition.reset_mock()
    await first.aclose()
    assert daemon.phone_viewing(entry.record.session_id)
    assert daemon.phone_viewer(entry.record.session_id) is viewer
    viewer.close.assert_not_called()
    daemon.notify_watch_transition.assert_not_called()
    daemon._prune_projection_generation.assert_not_called()
    timer = daemon._phone_view_graces[entry.record.session_id]
    second = (await response_for(app, "/api/sessions/{session_id:str}/events")).body_iterator
    await anext(second)
    assert timer.cancelled()
    daemon.notify_watch_transition.assert_not_called()  # no duplicate watch
    assert daemon.phone_viewer(entry.record.session_id) is viewer
    await second.aclose()
    # Invoke the actual scheduled callback: no 20-second sleep or flaky latency assertion.
    grace = daemon._phone_view_graces[entry.record.session_id]
    callback, args = grace._callback, grace._args
    grace.cancel()
    callback(*args)
    viewer.close.assert_called_once()
    assert daemon.notify_watch_transition.call_args.kwargs == {"watching": False}
    daemon._prune_projection_generation.assert_called_once_with(entry.record.session_id)
    assert not daemon.phone_viewing(entry.record.session_id)
    await daemon.close_phone_views()


@pytest.mark.asyncio
async def test_list_burst_dedupes_but_keeps_comment_liveness(monkeypatch):
    from local_operator.mobile import daemon as module

    monkeypatch.setattr(module, "LIST_FRAME_INTERVAL_S", 0.001)
    monkeypatch.setattr(module, "SSE_KEEPALIVE_S", 0.02)
    daemon, entry = fixture_daemon()
    daemon.table.summaries = AsyncMock(return_value=[])
    stream = (await response_for(build_app(daemon), "/api/sessions/events")).body_iterator
    first = await anext(stream)
    assert first.startswith("event: sessions\n")
    for _ in range(100):
        daemon.table.notify_list_changed()
    # The frame rebuilt after a hundred rings is identical. It is suppressed,
    # but the keepalive still arrives; otherwise the byte watchdog would fire.
    assert await asyncio.wait_for(anext(stream), 2) == ": keepalive\n\n"
    assert daemon.table.summaries.call_count == 2
    await stream.aclose()
    assert not daemon.table.list_subscribers
    await daemon.close_phone_views()


STREAM_ROUTES = [
    "/api/sessions/{session_id:str}/events",
    "/api/sessions/events",
    "/api/config/events",
]


async def cancel_parked(path, park_on: str):
    """Drive one stream's response with ``send`` parked on ``park_on``, then cancel."""
    daemon, entry = fixture_daemon()
    response = await response_for(build_app(daemon), path)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"cookie", f"{COOKIE_NAME}={sign_cookie('pw')}".encode())],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 1),
        "root_path": "",
    }
    parked = asyncio.Event()

    async def receive():
        await asyncio.Future()

    async def send(message):
        if message["type"] == park_on:
            parked.set()
            await asyncio.Future()

    task = asyncio.create_task(response(scope, receive, send))
    await asyncio.wait_for(parked.wait(), 2)
    task.cancel()
    outcome: BaseException | None = None
    try:
        await task
    except BaseException as exc:  # noqa: BLE001 - the test asserts the exact type
        outcome = exc
    return daemon, entry, outcome


@pytest.mark.parametrize("path", STREAM_ROUTES)
@pytest.mark.asyncio
async def test_a_cancellation_parked_on_the_headers_leaves_no_attachment(path):
    """A stream cancelled before its body starts must leave nothing behind.

    The routes register their subscriber on the first pull, so the parked case
    has nothing to detach; this pins that property together with the response
    owning the release, so it cannot regress into a route-time attach that the
    generator's ``finally`` would then never undo.
    """
    daemon, entry, outcome = await cancel_parked(path, "http.response.start")
    assert isinstance(outcome, asyncio.CancelledError)
    assert entry.record.session_id not in daemon.table.session_subscribers
    assert not daemon.table.list_subscribers
    assert not daemon.config_subscribers
    await daemon.close_phone_views()


@pytest.mark.asyncio
async def test_a_failing_stream_cleanup_does_not_replace_the_cancellation(monkeypatch):
    """The request's own cancellation outranks a cleanup failure.

    The generator's ``finally`` runs inside ``aclose()``, i.e. inside the
    response's ``finally``, where the in-flight exception is the
    ``CancelledError``. A detach that raises there used to escape and become
    the request's outcome.
    """
    daemon, entry = fixture_daemon()
    monkeypatch.setattr(
        daemon, "leave_phone_view", Mock(side_effect=RuntimeError("release failed"))
    )
    path = "/api/sessions/{session_id:str}/events"
    response = await response_for(build_app(daemon), path)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"cookie", f"{COOKIE_NAME}={sign_cookie('pw')}".encode())],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 1),
        "root_path": "",
    }
    first_body = asyncio.Event()

    async def receive():
        await asyncio.Future()

    async def send(message):
        if message["type"] == "http.response.body":
            first_body.set()
            await asyncio.Future()

    task = asyncio.create_task(response(scope, receive, send))
    await asyncio.wait_for(first_body.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert entry.record.session_id not in daemon.table.session_subscribers
    await daemon.close_phone_views()
