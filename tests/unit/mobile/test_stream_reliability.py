"""Phone stream recovery uses snapshots and bounded residency, not event replay."""

import asyncio
import json
import logging
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


def http_scope(path):
    """The bare ASGI scope these stream responses are driven with."""
    return {
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


async def cancel_parked(path, park_on: str, fixture=None):
    """Drive one stream's response with ``send`` parked on ``park_on``, then cancel.

    ``fixture`` is one of these scenarios with its own patched daemon; it is
    also the reason the masking test below reads as this scenario plus a raising
    detach rather than as a second copy of the same twenty lines.
    """
    daemon, entry = fixture if fixture is not None else fixture_daemon()
    response = await response_for(build_app(daemon), path)
    scope = http_scope(path)
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
    """Property, not a regression guard: a parked start leaves nothing attached.

    This passes on the pre-fix module too — every route registers its
    subscriber on the first pull, so a stream cancelled before its body starts
    has nothing to detach. It is stated here so the invariant sits beside the
    response-side release that now owns the detach: that release is what stops
    the guarantee from depending on where a route's attach happens to sit.
    """
    daemon, entry, outcome = await cancel_parked(path, "http.response.start")
    assert isinstance(outcome, asyncio.CancelledError)
    assert entry.record.session_id not in daemon.table.session_subscribers
    assert not daemon.table.list_subscribers
    assert not daemon.config_subscribers
    await daemon.close_phone_views()


@pytest.mark.asyncio
async def test_the_response_release_runs_when_the_body_never_starts():
    """The response-side release is the only close a never-started body can reach.

    Built directly from the class the routes return, with a spy release and a
    body nobody pulls: the generator's ``finally`` cannot run, so a
    ``call_count`` of one can only come from ``SessionEventResponse`` itself.
    Delete that branch and this test fails while every route-level test above
    stays green — which is the whole reason it exists (review round 5, R5-1).
    """
    daemon, _ = fixture_daemon()
    response_type = type(await response_for(build_app(daemon), "/api/sessions/events"))
    pulled: list[str] = []

    async def body():
        pulled.append("pulled")
        yield "data: never\n\n"

    release = Mock()
    response = response_type(body(), release=release)
    path = "/api/sessions/events"
    parked = asyncio.Event()

    async def receive():
        await asyncio.Future()

    async def send(message):
        if message["type"] == "http.response.start":
            parked.set()
            await asyncio.Future()

    task = asyncio.create_task(response(http_scope(path), receive, send))
    await asyncio.wait_for(parked.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert release.call_count == 1
    assert pulled == []


@pytest.mark.asyncio
async def test_a_raising_release_is_logged_at_warning_and_keeps_the_cancellation(caplog):
    """A release that raises is a WARNING, not the request's outcome.

    ``caplog`` rather than live log output: under the suite's default ``-n
    auto``, worker ``--log-cli-level`` records are dropped, so the level has to
    be asserted from inside the worker.
    """
    daemon, _ = fixture_daemon()
    response_type = type(await response_for(build_app(daemon), "/api/sessions/events"))
    release = Mock(side_effect=RuntimeError("release failed"))

    async def body():
        yield "data: never\n\n"

    response = response_type(body(), release=release)
    path = "/api/sessions/events"
    parked = asyncio.Event()

    async def receive():
        await asyncio.Future()

    async def send(message):
        if message["type"] == "http.response.start":
            parked.set()
            await asyncio.Future()

    with caplog.at_level(logging.WARNING, logger="local_operator.mobile.daemon"):
        task = asyncio.create_task(response(http_scope(path), receive, send))
        await asyncio.wait_for(parked.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert release.call_count == 1
    assert [record.getMessage() for record in caplog.records] == ["stream cleanup failed"]


@pytest.mark.asyncio
async def test_a_failing_stream_cleanup_does_not_replace_the_cancellation(monkeypatch, caplog):
    """The request's own cancellation outranks a cleanup failure.

    The generator's ``finally`` runs inside ``aclose()``, i.e. inside the
    response's ``finally``, where the in-flight exception is the
    ``CancelledError``. A detach that raises there used to escape and become the
    request's outcome; it is now logged at WARNING, pinned here with ``caplog``
    for the same worker-record reason as the test above.
    """
    fixture = fixture_daemon()
    daemon, entry = fixture
    monkeypatch.setattr(
        daemon, "leave_phone_view", Mock(side_effect=RuntimeError("release failed"))
    )
    with caplog.at_level(logging.WARNING, logger="local_operator.mobile.daemon"):
        _, _, outcome = await cancel_parked(
            "/api/sessions/{session_id:str}/events", "http.response.body", fixture
        )
    assert isinstance(outcome, asyncio.CancelledError)
    assert [record.getMessage() for record in caplog.records] == ["stream cleanup failed"]
    assert entry.record.session_id not in daemon.table.session_subscribers
    await daemon.close_phone_views()
