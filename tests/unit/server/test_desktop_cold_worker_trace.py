"""Exercise the actual desktop cold SSE first paint without a listener or owner."""

from __future__ import annotations

import asyncio
import json
import re
import socket
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.types import Message as ASGIMessage
from starlette.types import Scope

from local_operator.config import ConfigManager
from local_operator.harness.types import Message
from local_operator.model import configure
from local_operator.model.registry import ModelInfo
from local_operator.providers.auth_store import AuthStore
from local_operator.secrets import client as secrets_client
from local_operator.server.early_paint_hold import EarlyPaintHoldMiddleware
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import open_trace
from local_operator.session.frontend_state import (
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendSessionState,
    JobState,
)
from local_operator.session.transcript import (
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    Transcript,
    TranscriptEntry,
)

_SID = "abcdef012345"
_TITLE = "synthetic-checkpoint-title"
_TEXT = "synthetic-private-journal-text"
_ACCOUNT = "synthetic-offline-account-value"
_BEARER = "synthetic-desktop-only-bearer"


async def _first_frame(
    app: FastAPI, *, bearer: bool = True, origin: bool = False, query: bytes = b"open_frame=1"
) -> tuple[int, list[dict[str, Any]], bytes]:
    done = asyncio.Event()
    reads = 0
    statuses: list[int] = []
    bodies: list[bytes] = []
    frames: list[dict[str, Any]] = []
    path = f"/v1/desktop/sessions/{_SID}/events"
    headers = [(b"host", b"localhost")]
    if bearer:
        headers.append((b"authorization", f"Bearer {_BEARER}".encode()))
    if origin:
        headers.append((b"origin", b"https://untrusted.example.invalid"))
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query,
        "root_path": "",
        "client": ("127.0.0.1", 1234),
        "server": ("localhost", 80),
        "headers": headers,
    }

    async def receive() -> ASGIMessage:
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message: ASGIMessage) -> None:
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
        if message["type"] != "http.response.body":
            return
        body: bytes = message.get("body", b"")
        bodies.append(body)
        for line in body.splitlines():
            if line.startswith(b"data: "):
                frame = json.loads(line[6:])
                frames.append(frame)
                if frame["type"] == "snapshot":
                    # Disconnect the real streaming response after first paint;
                    # a buffering ASGI client would wait forever for SSE EOF.
                    done.set()

    await asyncio.wait_for(app(scope, receive, send), timeout=15)
    (status,) = statuses
    return status, frames, b"".join(bodies)


async def _seed(root: Path) -> tuple[ConfigManager, bytes]:
    config = ConfigManager(root)
    config.set_config_value("hosting", "openai")
    config.set_config_value("model_name", "gpt-5.1")
    with closing(AuthStore(root / "auth.db")) as auth:
        auth.upsert_credential(
            "openai",
            {
                "type": "oauth",
                "access": _ACCOUNT,
                "refresh": "synthetic-unused-refresh",
                "expires": 4_102_444_800_000,
                "account_id": "synthetic-account-id",
                "org_id": "synthetic-organisation-id",
            },
        )
    directory = root / "sessions" / _SID
    directory.mkdir(parents=True)
    user = Message.user(_TEXT).model_copy(update={"id": "user-fixed"})
    answer = Message.assistant("Synthetic answer").model_copy(update={"id": "answer-fixed"})
    checkpoint = FrontendSessionState(
        session_id=_SID,
        epoch="old-owner",
        conversation_title=_TITLE,
        jobs=[JobState(id="synthetic-job", type="task", label="check", status="succeeded")],
        cumulative_parent_cost=12.5,
    )
    with closing(Transcript(directory)) as journal:
        # A real seeded journal with fixed timestamps makes the two independent
        # cold opens byte-comparable, not two reads of a warm resident facade.
        await journal.append_imported(
            [
                TranscriptEntry(user.id, 1_790_000_000.0, ENTRY_MESSAGE, {}),
                TranscriptEntry(answer.id, 1_790_000_001.0, ENTRY_MESSAGE, {}),
                TranscriptEntry(
                    "checkpoint-fixed",
                    1_790_000_002.0,
                    ENTRY_CUSTOM,
                    {
                        "custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                        "details": {
                            "checkpoint_id": "checkpoint-fixed",
                            "state": checkpoint.model_dump(mode="json"),
                        },
                    },
                ),
            ],
            messages={user.id: user, answer.id: answer},
        )
        journal.flush()
        return config, journal.path.read_bytes()


@pytest.mark.asyncio
async def test_cold_worker_trace_joins_real_sse_without_altering_the_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None]
) -> None:
    listing: list[bool] = []
    network_attempts: list[str] = []
    disabled_calls: list[str] = []
    responses: dict[bool, tuple[bytes, bytes]] = {}
    statuses: list[int] = []
    trace_file = tmp_path / "open.jsonl"

    def offline_listing(
        provider: str, model_id: str, *, credential: tuple[str, ...] | None = None, **_: Any
    ) -> ModelInfo:
        matched = provider == "openai" and bool(credential and credential[0] == _ACCOUNT)
        listing.append(matched)
        return ModelInfo(
            id=model_id,
            name=model_id,
            description="",
            context_window=150_000 if matched else 128_000,
            max_tokens=8192,
        )

    def forbidden_network(*_: object, **__: object) -> None:
        network_attempts.append("external transport")
        raise AssertionError(network_attempts[-1])

    def forbidden_probe(*_: object) -> None:
        disabled_calls.append("new diagnostic on the disabled path")
        raise AssertionError(disabled_calls[-1])

    monkeypatch.setattr(configure, "resolve_model_info", offline_listing)
    monkeypatch.setattr(secrets_client, "PEER_AUTHENTICATION_SUPPORTED", False)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket.socket, "bind", forbidden_network)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", _BEARER)
    for variable in ("OPENAI_API_KEY", "OPENAI_OAUTH_TOKEN", "LOCAL_OPERATOR_OPENAI_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    for enabled in (True, False):
        home = tmp_path / ("on" if enabled else "off")
        root = home / "config"
        root.mkdir(parents=True)
        with monkeypatch.context() as m:
            m.setenv("HOME", str(home))
            m.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
            m.setattr(open_trace, "_PATH", str(trace_file) if enabled else None)
            if not enabled:
                m.setattr(
                    open_trace,
                    "time",
                    SimpleNamespace(
                        time=time.time,
                        monotonic=time.monotonic,
                        monotonic_ns=forbidden_probe,
                        thread_time_ns=forbidden_probe,
                    ),
                )
                m.setattr(open_trace.ColdOpenTrace, "__init__", forbidden_probe)
                m.setattr(open_trace.ColdOpenTrace, "worker", forbidden_probe)
            config, journal = await _seed(root)
            auth_disk = (root / "auth.db").read_bytes()
            app = FastAPI()
            app.state.config_manager = config
            pool = app.state.desktop_sessions = DesktopSessions(root)
            app.include_router(desktop_sessions.router)
            app.add_middleware(EarlyPaintHoldMiddleware)
            try:
                if enabled:
                    refusals = (
                        (await _first_frame(app, bearer=False), 401),
                        (await _first_frame(app, origin=True), 403),
                        (await _first_frame(app, query=b"open_frame=not-an-integer"), 422),
                    )
                    for (denied, no_frames, problem), expected in refusals:
                        statuses.append(denied)
                        assert denied == expected and not no_frames
                        assert "detail" in json.loads(problem)
                    assert not pool.bridges and not trace_file.exists()
                status, frames, wire = await _first_frame(app)
                statuses.append(status)
                assert status == 200 and [frame["type"] for frame in frames] == ["open", "snapshot"]
                bridge = pool.bridges[_SID]
                assert bridge.users == 0 and bridge.remote is not None
                state = bridge.remote.frontend_state
                assert state.conversation_title == _TITLE
                assert [job.id for job in state.jobs] == ["synthetic-job"]
                assert state.selected_model is not None
                assert state.selected_model.context_window == 150_000
                assert (root / "auth.db").read_bytes() == auth_disk
                opened, snapshot = frames
                epoch = opened["epoch"]
                sub = opened["payload"]["subscription_id"]
                assert epoch == snapshot["epoch"] and re.fullmatch(r"[0-9a-f]{32}", epoch)
                assert re.fullmatch(r"[0-9a-f]{32}", sub)
                assert _TITLE.encode() in wire and _TEXT.encode() in wire
                # Normalize only independently minted transport IDs and the
                # marker-less fixture's cwd, then compare actual SSE bytes.
                wire = wire.replace(epoch.encode(), b"<epoch>").replace(sub.encode(), b"<sub>")
                responses[enabled] = (wire.replace(str(home).encode(), b"<home>"), journal)
            finally:
                await pool.close()
    assert responses[True] == responses[False]
    assert not network_attempts and not disabled_calls and listing.count(True) >= 2

    rows = [json.loads(line) for line in trace_file.read_text().splitlines()]
    (ingress,) = [row for row in rows if row["stage"] == "desktop.events.acquire"]
    (acquire,) = [row for row in rows if row["stage"] == "a.acquire"]
    (summary,) = [row for row in rows if row["stage"] == "cold.summary"]
    workers = [row for row in rows if row["stage"] == "cold.worker"]
    assert len(workers) == 3
    assert {row["phase"] for row in workers} == {
        "model_synthesis",
        "account_context",
        "durable_validate",
    }
    assert ingress["acquire_id"] == acquire["acquire_id"] == summary["span"]
    assert all(row["acquire_id"] == acquire["acquire_id"] for row in workers)
    assert all(rows.index(summary) < rows.index(row) < rows.index(acquire) for row in workers)
    for private in (_SID, _TITLE, _TEXT, _ACCOUNT, _BEARER, str(tmp_path)):
        assert private not in json.dumps(workers)
    record_property("desktop_sse_statuses", statuses)
    record_property("sse_first_frames", [frame["type"] for frame in frames])
    record_property("normalized_equal_wire_bytes", len(responses[True][0]))
    record_property("cold_worker_phases", [row["phase"] for row in workers])
