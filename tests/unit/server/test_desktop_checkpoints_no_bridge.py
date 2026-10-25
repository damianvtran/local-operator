"""``GET .../checkpoints`` without a bridge, and the ``/events`` paint hold.

CORE-PROFILE L1 + L2 (project ``conversation-first-paint``, 2026-10-25):

* L1: the manifest route used to enter ``DesktopSessions.session``, which builds
  the whole cold facade (``AttachedSession.cold``) and holds the bridge lock, for
  an answer the index module gives by itself. These cells pin that it no longer
  does, and that every answer the door used to give -- the manifest bytes, the
  404s, the draft's empty manifest, the peer's ``unsupported``, the latch's
  503 -- is still given.
* L2: ``/events`` holds the session's paint hold until its snapshot frame is
  written, so a cold unpayable index scan starts after the first frame. These
  cells pin that the hold is released on every exit path (a disconnect before
  the snapshot included), so the scan starts promptly and never at the ceiling.

Everything runs under the test's own ``tmp_path`` root with synthetic ids.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.resume import SessionRow
from local_operator.server import retire
from local_operator.server.models.desktop_sessions import CheckpointManifest
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import transcript_index as ti
from local_operator.session.attached import AttachedSession

TOKEN = "synthetic-checkpoints-token"
SID = "abcabc000001"
PEER_SID = "e" * 12


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    ti._reset_for_tests()
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.serve_retiring = False
    app.state.desktop_sessions = DesktopSessions(
        tmp_path,
        retiring=lambda: bool(getattr(app.state, retire.RETIRING_STATE_ATTR, False)),
    )
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app, tmp_path
    await app.state.desktop_sessions.close()
    ti._reset_for_tests()


def _write_session(root: Path, *, big: bool = False) -> None:
    """A canonical session: marker plus one completed run, in the journal's row shape."""
    session = root / "sessions" / SID
    session.mkdir(parents=True, exist_ok=True)
    (session / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    token = "t1"
    conv = f"session/{SID}"
    rows: list[dict[str, Any]] = [
        {
            "id": "s1",
            "ts": 1.0,
            "type": "custom",
            "payload": {
                "custom_type": "attention_started",
                "data": {"conversation_id": conv, "token": token},
            },
        },
        {
            "id": "u1",
            "ts": 1.1,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": "hi"}]},
        },
        {
            "id": "a1",
            "ts": 1.2,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "assistant",
                "content": [{"text": "x" * (6 << 20) if big else "hello"}],
            },
        },
        {
            "id": "m1",
            "ts": 1.3,
            "type": "custom",
            "payload": {
                "custom_type": "completion_attention",
                "data": {
                    "conversation_id": conv,
                    "token": token,
                    "anchor": "a1",
                    "kind": "complete",
                },
            },
        },
    ]
    with (session / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _count_cold(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = AttachedSession.cold.__func__  # type: ignore[attr-defined]

    async def counting(cls, *args: Any, **kwargs: Any) -> Any:
        calls.append("cold")
        return await real(cls, *args, **kwargs)

    monkeypatch.setattr(AttachedSession, "cold", classmethod(counting))
    return calls


async def _ready(client: AsyncClient, sid: str = SID) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 30
    while True:
        response = await client.get(f"/v1/desktop/sessions/{sid}/checkpoints")
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        if result["index"]["state"] == "ready":
            return response.json()
        assert asyncio.get_running_loop().time() < deadline, result
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_the_manifest_builds_no_facade_and_takes_no_bridge(api, monkeypatch) -> None:
    """Acceptance: cold ``/checkpoints`` alone calls ``AttachedSession.cold`` 0 times."""
    client, app, root = api
    _write_session(root)
    cold = _count_cold(monkeypatch)
    body = await _ready(client)
    assert cold == [], "the manifest route built the cold facade"
    assert app.state.desktop_sessions.bridges == {}, "the manifest route built a bridge"
    assert [c["id"] for c in body["result"]["checkpoints"]] == ["u1", "a1"]


@pytest.mark.asyncio
async def test_the_answer_is_byte_identical_to_the_bridge_path(api) -> None:
    """Same bytes as the door + ``bridge.checkpoints()`` gave, for the same session."""
    client, app, root = api
    _write_session(root)
    body = await _ready(client)
    pool: DesktopSessions = app.state.desktop_sessions
    async with pool.session(SID, read=True, allow_draft=True) as bridge:
        via_bridge = await bridge.checkpoints()
    # Through the route's own response model, which is what the door path's
    # answer was serialised by on the wire.
    expected = CheckpointManifest.model_validate(via_bridge).model_dump(mode="json")
    assert body["result"] == expected
    # And with a RESIDENT bridge the bridge-free route still answers the same.
    async with pool.session(SID, read=True, allow_draft=True):
        again = await client.get(f"/v1/desktop/sessions/{SID}/checkpoints")
    assert again.json() == body


@pytest.mark.asyncio
async def test_unknown_and_malformed_ids_are_the_shared_404(api) -> None:
    client, _app, _root = api
    for bad in ("0123456789AB", "ffffffff9999"):
        response = await client.get(f"/v1/desktop/sessions/{bad}/checkpoints")
        assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_a_draft_answers_the_empty_ready_manifest(api) -> None:
    client, app, root = api
    pool: DesktopSessions = app.state.desktop_sessions
    draft_id = await pool.mint_draft(str(root))
    response = await client.get(f"/v1/desktop/sessions/{draft_id}/checkpoints")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["index"]["state"] == "ready" and result["checkpoints"] == []
    assert pool.bridges == {}


@pytest.mark.asyncio
async def test_a_peer_row_still_answers_unsupported(api, monkeypatch) -> None:
    """Invariant: a peer conversation is ``unsupported``, cold or resident."""
    from local_operator.session import remote_open

    client, _app, _root = api
    row = SessionRow(
        id=PEER_SID,
        mtime=1789400000.0,
        name="peer chat",
        live_state="idle",
        locality="remote",
        owner_device="d_" + "7" * 32,
        owner_device_name="build-box",
        reachable=True,
        unreachable_reason="",
    )
    asked: list[str] = []

    def fake(session_id: str, root: Any = None) -> tuple[SessionRow | None, tuple[Any, ...]]:
        asked.append(session_id)
        return (row if session_id == PEER_SID else None), ()

    monkeypatch.setattr(remote_open, "remote_row_and_silence", fake)
    response = await client.get(f"/v1/desktop/sessions/{PEER_SID}/checkpoints")
    assert response.status_code == 200, response.text
    # The exact bytes the bridge's peer branch produced through the route model.
    assert response.json()["result"] == {
        "session_id": PEER_SID,
        "index": {"state": "unsupported", "built_at": None},
        "checkpoints": [],
    }
    assert asked == [PEER_SID]


@pytest.mark.asyncio
async def test_a_retiring_daemon_still_refuses(api, monkeypatch) -> None:
    """Invariant: the latch refuses a known id; an unknown id stays the 404."""
    client, app, root = api
    _write_session(root)
    cold = _count_cold(monkeypatch)
    app.state.serve_retiring = True
    response = await client.get(f"/v1/desktop/sessions/{SID}/checkpoints")
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "daemon-retiring"
    unknown = await client.get("/v1/desktop/sessions/ffffffff9999/checkpoints")
    assert unknown.status_code == 404
    assert cold == []


@pytest.mark.asyncio
async def test_events_does_not_wait_on_the_manifest(api, monkeypatch) -> None:
    """The parallel burst: a manifest read in flight holds no bridge lock, so the
    snapshot path can take the lock while the manifest is still running."""
    client, app, root = api
    _write_session(root)
    gate = asyncio.Event()
    entered = asyncio.Event()
    real_view = ti.checkpoints_view

    async def slow_view(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await gate.wait()
        return await real_view(*args, **kwargs)

    monkeypatch.setattr(module.transcript_index, "checkpoints_view", slow_view)
    manifest = asyncio.create_task(client.get(f"/v1/desktop/sessions/{SID}/checkpoints"))
    await asyncio.wait_for(entered.wait(), 10)
    pool: DesktopSessions = app.state.desktop_sessions
    # The door the /events route takes: it must not queue behind the manifest.
    async with asyncio.timeout(10):
        async with pool.session(SID, read=True, allow_draft=True) as bridge:
            assert not bridge.lock.locked()
    gate.set()
    response = await manifest
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_a_client_gone_before_the_snapshot_releases_the_hold(api, monkeypatch) -> None:
    """Every exit releases the paint hold: an open whose body is never consumed
    (the client left between headers and snapshot) still lets the deferred scan
    start promptly -- far inside the ceiling, which is raised here to prove it."""
    client, app, root = api
    _write_session(root, big=True)
    monkeypatch.setattr(ti, "_PAINT_HOLD_CEILING_S", 60.0)

    async def no_frames(*_args: Any, **_kwargs: Any):
        if False:  # pragma: no cover - keeps this an async generator
            yield {}
        raise asyncio.CancelledError  # the client disconnected mid-open

    monkeypatch.setattr(module.DesktopSessionBridge, "events", no_frames)
    with pytest.raises(BaseException):
        await client.get(f"/v1/desktop/sessions/{SID}/events")
    assert ti._PAINT_HOLDS == {}, "a disconnected open leaked its paint hold"
    task, _ = ti.start_refresh(root, SID)
    assert not ti.refresh_deferred(root, SID)
    assert await asyncio.wait_for(task, 30) is not None


@pytest.mark.asyncio
async def test_a_failed_open_releases_the_hold(api) -> None:
    """An acquire refusal (unknown id) is an exit too."""
    client, _app, _root = api
    response = await client.get("/v1/desktop/sessions/ffffffff9999/events")
    assert response.status_code == 404
    assert ti._PAINT_HOLDS == {}


@pytest.mark.asyncio
async def test_the_snapshot_frame_releases_the_hold_before_the_stream_ends(
    api, monkeypatch
) -> None:
    """The release point is the snapshot frame, not stream teardown: while the
    stream is still open after its snapshot, a held scan is already running."""
    client, app, root = api
    _write_session(root, big=True)
    after_snapshot = asyncio.Event()
    finish = asyncio.Event()
    seen: dict[str, Any] = {}

    async def frames(self, *_args: Any, **_kwargs: Any):
        task, _ = ti.start_refresh(self.root, self.session_id)
        seen["deferred_before"] = ti.refresh_deferred(self.root, self.session_id)
        seen["task"] = task
        yield {"type": "open", "session_id": self.session_id, "payload": {}}
        yield {"type": "snapshot", "session_id": self.session_id, "payload": {}}
        # The route releases the hold right after it serialises the snapshot;
        # the next resumption of this generator happens after that.
        seen["holds_after"] = dict(ti._PAINT_HOLDS)
        after_snapshot.set()
        await finish.wait()

    monkeypatch.setattr(module.DesktopSessionBridge, "events", frames)

    async def consume() -> None:
        async with client.stream("GET", f"/v1/desktop/sessions/{SID}/events") as response:
            async for _line in response.aiter_lines():
                if after_snapshot.is_set():
                    break

    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(after_snapshot.wait(), 30)
    assert seen["deferred_before"] is True
    assert seen["holds_after"] == {}, "the hold outlived the snapshot frame"
    assert await asyncio.wait_for(seen["task"], 30) is not None
    finish.set()
    consumer.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await consumer
    assert ti._PAINT_HOLDS == {}


@pytest.mark.asyncio
async def test_cancellation_at_response_start_releases_before_generator_entry(api, monkeypatch):
    """ASGI can cancel the headers send before the generator ever starts.

    This drives the app's real HTTP request scope and aborts the response-start
    send, rather than pretending an unentered generator's finally will run.
    """
    _client, app, root = api
    _write_session(root, big=True)
    entered: list[str] = []
    real_snapshot = module.DesktopSessionBridge.snapshot

    async def snapshot(self, **kwargs):
        entered.append(self.session_id)
        return await real_snapshot(self, **kwargs)

    monkeypatch.setattr(module.DesktopSessionBridge, "snapshot", snapshot)

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.start":
            raise asyncio.CancelledError

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": f"/v1/desktop/sessions/{SID}/events", "query_string": b"",
        "headers": [(b"authorization", f"Bearer {TOKEN}".encode())],
        "server": ("localhost", 80), "client": ("127.0.0.1", 1234),
    }
    with pytest.raises(asyncio.CancelledError):
        await app(scope, receive, send)
    assert entered == [], "the test cancelled after, not before, generator entry"
    assert ti._PAINT_HOLDS == {}
    task, _ = ti.start_refresh(root, SID)
    assert not ti.refresh_deferred(root, SID)
    assert await asyncio.wait_for(task, 30) is not None


@pytest.mark.asyncio
async def test_snapshot_exception_releases_its_deferred_scan(api, monkeypatch):
    """A real /events request whose snapshot raises still releases promptly."""
    client, _app, root = api
    _write_session(root, big=True)
    monkeypatch.setattr(ti, "_PAINT_HOLD_CEILING_S", 60.0)
    tasks = []

    async def broken_snapshot(self, **kwargs):
        task, _ = ti.start_refresh(self.root, self.session_id)
        tasks.append(task)
        assert ti.refresh_deferred(self.root, self.session_id)
        raise RuntimeError("synthetic snapshot failure")

    monkeypatch.setattr(module.DesktopSessionBridge, "snapshot", broken_snapshot)
    with pytest.raises(RuntimeError, match="synthetic snapshot failure"):
        await client.get(f"/v1/desktop/sessions/{SID}/events")
    assert ti._PAINT_HOLDS == {}
    assert await asyncio.wait_for(tasks[0], 30) is not None


@pytest.mark.asyncio
async def test_two_http_opens_cancelled_mid_snapshot_release_only_their_own_hold(api, monkeypatch):
    """Two HTTP opens share one deferred scan; cancelling one cannot release
    the other's hold. Cancelling the second starts the scan before the ceiling.
    """
    client, _app, root = api
    _write_session(root, big=True)
    monkeypatch.setattr(ti, "_PAINT_HOLD_CEILING_S", 60.0)
    both_entered = asyncio.Event()
    tasks = []

    async def blocked_snapshot(self, **kwargs):
        task, _ = ti.start_refresh(self.root, self.session_id)
        tasks.append(task)
        if len(tasks) == 2:
            both_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(module.DesktopSessionBridge, "snapshot", blocked_snapshot)
    opens = [
        asyncio.create_task(client.get(f"/v1/desktop/sessions/{SID}/events")) for _ in range(2)
    ]
    try:
        await asyncio.wait_for(both_entered.wait(), 30)
        assert tasks[0] is tasks[1], "concurrent opens started two index scans"
        opens[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await opens[0]
        assert ti._PAINT_HOLDS[ti._key(root, SID)] == 1
        assert ti.refresh_deferred(root, SID) and not tasks[0].done()
        opens[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await opens[1]
        assert ti._PAINT_HOLDS == {}
        assert await asyncio.wait_for(tasks[0], 30) is not None
    finally:
        for request in opens:
            if not request.done():
                request.cancel()
        await asyncio.gather(*opens, return_exceptions=True)
