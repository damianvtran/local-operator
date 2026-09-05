"""A real child runtime, lease and control socket behind the phone's start API.

Only session construction is replaced: the fixture has no provider, tools or
credentials. Its residency predicate and reaper are the actual product code.
Assertions inspect that predicate and await process termination/publication;
no sleep followed by a timing assertion stands in for readiness.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid

import httpx
import pytest
from starlette.requests import Request

from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import MobileDaemon, build_app

CHILD = """
import asyncio, contextlib, json, os, signal
from local_operator.paths import config_dir
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.process import _reaper, _should_exit
from local_operator.session_lease import acquire_session_lease
from local_operator.session.transcript import Transcript
from tests.unit.mobile.test_daemon import FakeHandle

async def main():
    handle = FakeHandle()
    handle._projection.session_id = os.environ['LOP_MOBILE_CHILD_RESUME']
    handle._projection.cwd = os.environ['LOP_MOBILE_CHILD_CWD']
    handle._projection.model_label = os.environ.get('LOP_MOBILE_CHILD_MODEL', 'fixture/model')
    handle.is_busy = lambda: False
    async def dispose(): pass
    handle.dispose = dispose
    runtime = RuntimeServer(handle, kind='daemon')
    async def prompt(*args, **kwargs):
        transcript = Transcript(config_dir() / 'sessions' / handle._projection.session_id)
        await transcript.append_custom('fixture_prompt', {})
        return json.dumps({'idle_exit_eligible': _should_exit(handle, runtime),
            'viewers': runtime.attach_clients(),
            'all_remote': all(c.locality == 'remote' for c in runtime._clients.values()),
            'cwd': handle._projection.cwd, 'model': handle._projection.model_label})
    handle.prompt = prompt
    lease = acquire_session_lease(config_dir() / 'sessions' / handle._projection.session_id)
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    await runtime.start_in_process()
    reaper = asyncio.create_task(_reaper(handle, runtime, stop))
    try:
        await stop.wait()
    finally:
        reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError): await reaper
        await runtime.aclose()
        lease.release()
asyncio.run(main())
"""


@pytest.fixture
def fixture_children(monkeypatch):
    """Keep the real spawn API/env but run an inert session in each child."""
    actual_spawn = asyncio.create_subprocess_exec
    children = []

    async def spawn(*args, **kwargs):
        assert args == (sys.executable, "-m", "local_operator.session.runtime.process")
        assert kwargs["start_new_session"] is True
        assert "LOP_RUNTIME_DEFER_MATERIALISE" not in kwargs["env"]
        child = await actual_spawn(sys.executable, "-c", CHILD, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return children


async def stop_fixture(daemon, children):
    await daemon.close_phone_views()
    for child in children:
        if child.returncode is None:
            child.terminate()
        await asyncio.wait_for(child.wait(), 10)
    for task in daemon._dial_tasks.values():
        task.cancel()
    await asyncio.gather(*daemon._dial_tasks.values(), return_exceptions=True)


async def session_stream_response(app, session_id):
    path = f"/api/sessions/{session_id}/events"
    endpoint = next(
        route.endpoint
        for route in app.routes
        if route.path == "/api/sessions/{session_id:str}/events"
    )
    response = await endpoint(
        Request(
            {
                "type": "http",
                "method": "GET",
                "path": path,
                "path_params": {"session_id": session_id},
                "query_string": b"",
                "headers": [
                    (b"host", b"fixture"),
                    (b"cookie", f"{COOKIE_NAME}={sign_cookie('pw')}".encode()),
                ],
                "scheme": "http",
                "server": ("fixture", 80),
            }
        )
    )
    return response


async def open_session_stream(app, session_id):
    response = await session_stream_response(app, session_id)
    frame = await anext(response.body_iterator)
    assert f'"session_id": "{session_id}"' in frame
    return response.body_iterator


@pytest.mark.asyncio
async def test_start_is_ready_by_durable_id_and_phone_viewers_hold_actual_child(
    tmp_path, monkeypatch, fixture_children
):
    monkeypatch.setenv("LOP_RUNTIME_DEFER_MATERIALISE", "1")
    daemon = MobileDaemon(password="pw")
    app = build_app(daemon)
    try:
        async with (
            asyncio.timeout(20),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://fixture",
                cookies={COOKIE_NAME: sign_cookie("pw")},
            ) as client,
        ):
            response = await client.post(
                "/api/sessions/start",
                json={
                    "cwd": str(tmp_path),
                    "provider": "radient",
                    "model_id": "fixture/chosen",
                },
            )
            assert response.status_code == 200, response.text
            receipt = response.json()
            session_id = receipt["session_id"]
            assert re.fullmatch(r"[0-9a-f]{12}", session_id)
            assert receipt["pid"] == fixture_children[0].pid
            assert daemon.table.entries[receipt["pid"]].ready.is_set()
            assert daemon.session_projections[session_id].model_label == "fixture/chosen"
            assert session_id in daemon._phone_handoffs

            first = await open_session_stream(app, session_id)
            second = await open_session_stream(app, session_id)
            assert session_id not in daemon._phone_handoffs
            await first.aclose()
            assert daemon._phone_attaches[session_id].connected
            # Inspect the ACTUAL child reaper's predicate after the response
            # and one viewer disconnect. False holds for any idle dwell,
            # including a phone taking longer than the 3 s drain to type.
            command = await client.post(
                f"/api/sessions/{session_id}/command",
                json={
                    "op": "prompt",
                    "text": "inspect residency",
                    "images": [],
                    "command_id": str(uuid.uuid4()),
                },
            )
            assert command.status_code == 200, command.text
            assert json.loads(command.json()["detail"]) == {
                "idle_exit_eligible": False,
                "viewers": 1,
                "all_remote": True,
                "cwd": str(tmp_path),
                "model": "fixture/chosen",
            }
            # Reopening a still-live conversation returns its verified owner,
            # with no second child and no process-ID route substitution.
            resumed = await client.post("/api/sessions/resume", json={"session_id": session_id})
            assert resumed.status_code == 200, resumed.text
            assert resumed.json() == receipt
            assert len(fixture_children) == 1
            await second.aclose()
            assert session_id not in daemon._phone_attaches
            # The real reaper publishes termination after the LAST viewer.
            # This wait is a hang guard, not an elapsed-time assertion.
            assert await fixture_children[0].wait() == 0
    finally:
        await stop_fixture(daemon, fixture_children)


@pytest.mark.asyncio
async def test_abandoned_browser_handoff_releases_child(tmp_path, fixture_children):
    daemon = MobileDaemon(password="pw")
    try:
        async with asyncio.timeout(15):
            pid = await daemon.spawn_session(str(tmp_path), resume="handoff12345")
            assert pid == fixture_children[0].pid
            timer = daemon._phone_handoffs["handoff12345"]
            # Invoke the scheduled lease expiry itself, no 60-second sleep.
            timer._callback(*timer._args)
            assert timer.cancelled()
            assert "handoff12345" not in daemon._phone_attaches
            assert await fixture_children[0].wait() == 0
    finally:
        await stop_fixture(daemon, fixture_children)


@pytest.mark.asyncio
async def test_child_exit_before_registration_is_not_acknowledged(tmp_path, monkeypatch):
    actual_spawn = asyncio.create_subprocess_exec
    children = []

    async def fail(*args, **kwargs):
        child = await actual_spawn(sys.executable, "-c", "raise SystemExit(2)", **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    daemon = MobileDaemon(password="pw")
    try:
        async with asyncio.timeout(10):
            with pytest.raises(RuntimeError, match="exited before becoming ready"):
                await daemon.spawn_session(str(tmp_path), resume="failed123456")
        assert children[0].returncode == 2
        assert not daemon._phone_attaches
        assert not daemon._phone_handoffs
    finally:
        await stop_fixture(daemon, children)


@pytest.mark.asyncio
async def test_concurrent_resume_and_cancelled_http_share_one_bounded_handoff(
    tmp_path, fixture_children
):
    daemon = MobileDaemon(password="pw")
    first = asyncio.create_task(daemon.spawn_session(str(tmp_path), resume="shared123456"))
    second = asyncio.create_task(daemon.spawn_session(str(tmp_path), resume="shared123456"))
    try:
        async with asyncio.timeout(15):
            # Let both callers enter spawn_session, with no elapsed-time bet.
            await asyncio.sleep(0)
            owned = daemon._session_starts["shared123456"]
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            pid = await second
            assert await owned == pid
            assert len(fixture_children) == 1
            assert daemon._phone_attaches["shared123456"].connected
            assert "shared123456" in daemon._phone_handoffs
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await stop_fixture(daemon, fixture_children)


@pytest.mark.asyncio
async def test_different_owner_is_never_acknowledged_as_spawned_pid(
    tmp_path, monkeypatch, fixture_children
):
    from local_operator.mobile import attach_client
    from local_operator.mobile.types import SessionRecord

    queries = 0

    def swapped_owner(_config, session_id):
        nonlocal queries
        queries += 1
        # No owner before spawn; another owner wins its lease before our
        # candidate publishes. The response may not mix these two identities.
        if queries == 1:
            return None, None
        return (
            SessionRecord(
                pid=1,
                session_id=session_id,
                kind="daemon",
                control_port=1,
                control_key="unused",
                conversation_name="",
                cwd="",
                model_label="",
            ),
            1,
        )

    monkeypatch.setattr(attach_client, "find_owner_record", swapped_owner)
    daemon = MobileDaemon(password="pw")
    try:
        async with asyncio.timeout(10):
            with pytest.raises(RuntimeError, match="another runtime acquired"):
                await daemon.spawn_session(str(tmp_path), resume="swapped12345")
        assert not daemon._phone_attaches
        assert not daemon._phone_handoffs
    finally:
        await stop_fixture(daemon, fixture_children)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_message", ["http.response.start", "http.response.body"])
async def test_sse_send_failure_cannot_retain_a_phantom_viewer(
    tmp_path, fixture_children, failed_message
):
    daemon = MobileDaemon(password="pw")
    app = build_app(daemon)
    try:
        async with asyncio.timeout(15):
            await daemon.spawn_session(str(tmp_path), resume="sendfail1234")
            response = await session_stream_response(app, "sendfail1234")

            async def send(message):
                if message["type"] == failed_message:
                    raise OSError("fixture disconnected")

            with pytest.raises(OSError, match="fixture disconnected"):
                await response.stream_response(send)
            assert not daemon.table.session_subscribers.get("sendfail1234")
            if failed_message == "http.response.start":
                # Headers failed: viewing never began and the original
                # bounded startup handoff still owns readiness.
                assert "sendfail1234" in daemon._phone_handoffs
                daemon.release_phone_view("sendfail1234")
            assert "sendfail1234" not in daemon._phone_attaches
            assert await fixture_children[0].wait() == 0
    finally:
        await stop_fixture(daemon, fixture_children)


@pytest.mark.asyncio
async def test_failed_start_cannot_signal_an_owner_that_accepted_work(
    tmp_path, monkeypatch, fixture_children
):
    daemon = MobileDaemon(password="pw")
    prepare = daemon._prepare_phone_record
    admitted = asyncio.Event()

    async def fail_after_work(record):
        await prepare(record)
        reply = await daemon.request(record.pid, "prompt", text="actual fixture admission")
        assert reply["op"] == "ack"
        child = fixture_children[0]
        monkeypatch.setattr(child, "terminate", lambda: pytest.fail("signalled a published owner"))
        monkeypatch.setattr(child, "kill", lambda: pytest.fail("killed a published owner"))
        admitted.set()
        raise RuntimeError("fixture readiness failure after admission")

    monkeypatch.setattr(daemon, "_prepare_phone_record", fail_after_work)
    try:
        async with asyncio.timeout(15):
            with pytest.raises(RuntimeError, match="readiness failure after admission"):
                await daemon.spawn_session(str(tmp_path), resume="admitted1234")
            assert admitted.is_set()
            assert not daemon._phone_attaches
            # Its own idle reaper may retire the inert fixture; the relay may
            # only ask the runtime to decide and must never send a process kill.
            assert await fixture_children[0].wait() == 0
    finally:
        monkeypatch.undo()
        await stop_fixture(daemon, fixture_children)


@pytest.mark.asyncio
async def test_phone_callbacks_cannot_follow_rebind_or_overwrite_replacement(monkeypatch):
    from local_operator.mobile import attach_client
    from local_operator.mobile.types import SessionProjection, SessionRecord

    clients = []

    class CapturedClient:
        def __init__(self, repaint, _disconnected, *, locality):
            assert locality == "remote"
            self.repaint = repaint
            self.connected = False
            clients.append(self)

        async def connect(self, record, session_id):
            self.connected = True
            self.repaint(
                SessionProjection(
                    session_id=session_id,
                    pid=record.pid,
                    kind="daemon",
                    conversation_name=record.control_key,
                    cwd="",
                    model_label="",
                )
            )

        def close(self):
            self.connected = False

    monkeypatch.setattr(attach_client, "AttachClient", CapturedClient)
    daemon = MobileDaemon(password="pw")
    daemon.table.session_subscribers["view12345678"] = {asyncio.Queue()}

    def record(key):
        return SessionRecord(
            pid=10001,
            session_id="view12345678",
            kind="daemon",
            control_key=key,
            control_port=1,
            conversation_name="",
            cwd="",
            model_label="",
        )

    try:
        first = daemon._schedule_phone_attach(record("old"))
        assert first is not None
        await first
        second = daemon._schedule_phone_attach(record("replacement"))
        assert second is not None
        await second
        assert len(clients) == 2
        assert not clients[0].connected
        entry = daemon.table.entries[10001]
        assert entry.record.control_key == "replacement"
        assert entry.projection is not None
        assert entry.projection.conversation_name == "replacement"
        clients[0].repaint(
            SessionProjection(
                session_id="view12345678",
                pid=10001,
                kind="daemon",
                conversation_name="late stale callback",
                cwd="",
                model_label="",
            )
        )
        assert entry.projection.conversation_name == "replacement"
        clients[1].repaint(
            SessionProjection(
                session_id="another12345",
                pid=10001,
                kind="daemon",
                conversation_name="wrong conversation",
                cwd="",
                model_label="",
            )
        )
        assert not clients[1].connected
        assert "view12345678" not in daemon._phone_attaches
        assert entry.projection.session_id == "view12345678"
    finally:
        await daemon.close_phone_views()
