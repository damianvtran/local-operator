"""Interactive-first admission: a background cold build yields to an ``/events`` open.

WHY (2026-11-25, trace on v0.113.1): the slowest sidebar opens waited while a
DIFFERENT, non-resident session cold-built to serve a background
``GET .../history`` read; the build holds the GIL, so the click's first frame
stretched by 1-1.5 s. ``DesktopSessions.hold_interactive_open`` marks an
``/events`` open in flight from route entry to its snapshot frame, and a
``read=True`` acquire that would cold-build waits (bounded) for it.

These cells pin: the cold build starts after the open's snapshot; a resident
read is never delayed; the wait is bounded; an ``/events`` open never waits on
another; and the hold is released on error and disconnect.

Everything runs under the test's own ``tmp_path`` root with synthetic ids.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server import retire
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import transcript_index as ti

TOKEN = "synthetic-admission-token"
CLICKED = "abcabc000001"
BACKGROUND = "abcabc000002"


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
    for sid in (CLICKED, BACKGROUND):
        _write_session(tmp_path, sid)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app, tmp_path
    await app.state.desktop_sessions.close()
    ti._reset_for_tests()


def _write_session(root: Path, sid: str) -> None:
    session = root / "sessions" / sid
    session.mkdir(parents=True, exist_ok=True)
    (session / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    rows = [
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
            "payload": {"kind": "message", "role": "assistant", "content": [{"text": "hello"}]},
        },
    ]
    with (session / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _record_builds(monkeypatch: pytest.MonkeyPatch, log: list[tuple[str, str]]) -> None:
    """Log ``("build", sid)`` whenever an acquire would construct a cold facade."""
    real = module.DesktopSessionBridge._ensure_facade

    async def recording(self, *args: Any, **kwargs: Any) -> Any:
        if self.remote is None:
            log.append(("build", self.session_id))
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(module.DesktopSessionBridge, "_ensure_facade", recording)


def _gated_events(
    monkeypatch: pytest.MonkeyPatch,
    log: list[tuple[str, str]],
    gates: dict[str, asyncio.Event],
) -> None:
    """``bridge.events`` that waits on its session's gate before the snapshot."""

    async def frames(self, *_args: Any, **_kwargs: Any):
        yield {"type": "open", "session_id": self.session_id, "payload": {}}
        gate = gates.get(self.session_id)
        if gate is not None:
            await gate.wait()
        log.append(("snapshot", self.session_id))
        yield {"type": "snapshot", "session_id": self.session_id, "payload": {}}
        await asyncio.Event().wait()  # a live stream stays open

    monkeypatch.setattr(module.DesktopSessionBridge, "events", frames)


async def _open_until_snapshot(client: AsyncClient, sid: str, seen: asyncio.Event) -> None:
    # httpx's ASGITransport may buffer a streamed body, so tests observe the
    # snapshot through the server-side ``log`` rather than through ``seen``.
    async with client.stream("GET", f"/v1/desktop/sessions/{sid}/events") as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if '"snapshot"' in line:
                seen.set()


async def _cancel(task: asyncio.Task[Any]) -> None:
    task.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await task


@pytest.mark.asyncio
async def test_background_cold_read_builds_only_after_the_open_snapshot(api, monkeypatch):
    client, app, _root = api
    log: list[tuple[str, str]] = []
    gates = {CLICKED: asyncio.Event()}
    _record_builds(monkeypatch, log)
    _gated_events(monkeypatch, log, gates)
    snap = asyncio.Event()
    opener = asyncio.create_task(_open_until_snapshot(client, CLICKED, snap))
    await asyncio.wait_for(_until(lambda: ("build", CLICKED) in log), 10)
    reader = asyncio.create_task(client.get(f"/v1/desktop/sessions/{BACKGROUND}/history"))
    # Long enough for an unheld read to build: on base it does, here it waits.
    await asyncio.sleep(0.4)
    assert ("build", BACKGROUND) not in log, f"background built during the open: {log}"
    gates[CLICKED].set()
    response = await asyncio.wait_for(reader, 10)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["entries"], "the held read still answers rows"
    assert log.index(("build", BACKGROUND)) > log.index(("snapshot", CLICKED)), log
    await _cancel(opener)
    assert app.state.desktop_sessions._interactive_opens == 0


@pytest.mark.asyncio
async def test_a_resident_read_is_never_delayed(api, monkeypatch):
    client, app, _root = api
    monkeypatch.setattr(module, "INTERACTIVE_ADMISSION_WAIT_S", 5.0)
    log: list[tuple[str, str]] = []
    gates = {CLICKED: asyncio.Event()}  # never set: the open hangs before its snapshot
    _gated_events(monkeypatch, log, gates)
    pool: DesktopSessions = app.state.desktop_sessions
    async with pool.session(BACKGROUND, read=True) as bridge:
        assert bridge.remote is not None  # resident: its facade is built
        opener = asyncio.create_task(_open_until_snapshot(client, CLICKED, asyncio.Event()))
        await asyncio.wait_for(_until(lambda: pool._interactive_opens == 1), 10)
        started = time.monotonic()
        response = await client.get(f"/v1/desktop/sessions/{BACKGROUND}/history")
        elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    assert elapsed < 2.0, f"a resident read waited {elapsed:.2f}s on an open"
    await _cancel(opener)


@pytest.mark.asyncio
async def test_the_wait_is_bounded_when_an_open_hangs(api, monkeypatch):
    client, app, _root = api
    monkeypatch.setattr(module, "INTERACTIVE_ADMISSION_WAIT_S", 0.5)
    log: list[tuple[str, str]] = []
    gates = {CLICKED: asyncio.Event()}  # never set
    _gated_events(monkeypatch, log, gates)
    pool: DesktopSessions = app.state.desktop_sessions
    opener = asyncio.create_task(_open_until_snapshot(client, CLICKED, asyncio.Event()))
    await asyncio.wait_for(_until(lambda: pool._interactive_opens == 1), 10)
    started = time.monotonic()
    response = await asyncio.wait_for(client.get(f"/v1/desktop/sessions/{BACKGROUND}/history"), 10)
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    assert 0.45 <= elapsed < 5.0, f"held {elapsed:.2f}s against a 0.5 s bound"
    await _cancel(opener)


@pytest.mark.asyncio
async def test_an_events_open_never_waits_on_another_open(api, monkeypatch):
    client, app, _root = api
    monkeypatch.setattr(module, "INTERACTIVE_ADMISSION_WAIT_S", 5.0)
    log: list[tuple[str, str]] = []
    gates = {CLICKED: asyncio.Event()}  # the first open hangs; the second has no gate
    _gated_events(monkeypatch, log, gates)
    pool: DesktopSessions = app.state.desktop_sessions
    first = asyncio.create_task(_open_until_snapshot(client, CLICKED, asyncio.Event()))
    await asyncio.wait_for(_until(lambda: pool._interactive_opens == 1), 10)
    started = time.monotonic()
    second = asyncio.create_task(_open_until_snapshot(client, BACKGROUND, asyncio.Event()))
    await asyncio.wait_for(_until(lambda: ("snapshot", BACKGROUND) in log), 10)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"a cold /events open waited {elapsed:.2f}s on another open"
    await _cancel(second)
    await _cancel(first)


@pytest.mark.asyncio
async def test_a_failed_open_releases_the_admission_hold(api):
    client, app, _root = api
    response = await client.get("/v1/desktop/sessions/ffffffff9999/events")
    assert response.status_code == 404
    pool: DesktopSessions = app.state.desktop_sessions
    assert pool._interactive_opens == 0
    assert pool._idle_event().is_set()


@pytest.mark.asyncio
async def test_a_client_gone_before_the_snapshot_releases_the_admission_hold(api, monkeypatch):
    client, app, _root = api

    async def no_frames(*_args: Any, **_kwargs: Any):
        if False:  # pragma: no cover - keeps this an async generator
            yield {}
        raise asyncio.CancelledError  # the client disconnected mid-open

    monkeypatch.setattr(module.DesktopSessionBridge, "events", no_frames)
    with pytest.raises(BaseException):
        await client.get(f"/v1/desktop/sessions/{CLICKED}/events")
    pool: DesktopSessions = app.state.desktop_sessions
    assert pool._interactive_opens == 0, "a disconnected open leaked its admission hold"
    started = time.monotonic()
    response = await client.get(f"/v1/desktop/sessions/{BACKGROUND}/history")
    assert response.status_code == 200
    assert time.monotonic() - started < 1.5


async def _until(predicate: Any) -> None:
    while not predicate():
        await asyncio.sleep(0.01)
