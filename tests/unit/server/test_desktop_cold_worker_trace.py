"""Exercise the actual desktop cold SSE first paint without a listener or owner."""

from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.types import Message as ASGIMessage
from starlette.types import Scope

from local_operator.config import ConfigManager
from local_operator.harness.types import Message
from local_operator.mobile import attach_client
from local_operator.model import configure
from local_operator.model.registry import ModelInfo
from local_operator.providers.auth_store import AuthStore
from local_operator.secrets import client as secrets_client
from local_operator.server.early_paint_hold import EarlyPaintHoldMiddleware
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import open_trace, transcript_index
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendSessionState,
    JobState,
)
from local_operator.session.owner import LocalOwner
from local_operator.session.runtime.types import SessionRecord
from local_operator.session.transcript import (
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    Transcript,
    TranscriptEntry,
)
from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE

_SID = "abcdef012345"
_TITLE = "synthetic-checkpoint-title"
_TEXT = "synthetic-private-journal-text"
_ACCOUNT = "synthetic-offline-account-value"
_BEARER = "synthetic-desktop-only-bearer"
_OPEN_QUERY = b"open_frame=1&supplements=1&frontend_replace=1"


async def _first_frame(
    app: FastAPI,
    *,
    bearer: bool = True,
    origin: bool = False,
    query: bytes = _OPEN_QUERY,
    sid: str = _SID,
) -> tuple[int, list[dict[str, Any]], bytes]:
    done = asyncio.Event()
    reads = 0
    statuses: list[int] = []
    bodies: list[bytes] = []
    frames: list[dict[str, Any]] = []
    path = f"/v1/desktop/sessions/{sid}/events"
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
                    "supplement-fixed",
                    1_790_000_001.5,
                    ENTRY_CUSTOM,
                    {
                        "custom_type": SUPPLEMENT_CUSTOM_TYPE,
                        "details": {
                            "anchor": answer.id,
                            "job": "3f9c1a7e5b20",
                            "version": 1,
                            "state": "done",
                        },
                    },
                ),
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


def _app(root: Path, config: ConfigManager) -> tuple[FastAPI, DesktopSessions]:
    app = FastAPI()
    app.state.config_manager = config
    pool = app.state.desktop_sessions = DesktopSessions(root)
    app.include_router(desktop_sessions.router)
    app.add_middleware(EarlyPaintHoldMiddleware)
    return app, pool


def _rows(trace_file: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace_file.read_text().splitlines()]


@pytest.mark.asyncio
async def test_cold_worker_trace_joins_real_sse_without_altering_the_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None]
) -> None:
    listing: list[bool] = []
    network_attempts: list[str] = []
    disabled_calls: list[str] = []
    responses: dict[bool, tuple[bytes, bytes, bytes]] = {}
    statuses: list[int] = []
    lookups: list[Path] = []
    real_locate = LocalOwner.locate
    trace_file = tmp_path / "open.jsonl"

    def count_locate(owner: LocalOwner):
        lookups.append(owner._config_dir)
        return real_locate(owner)

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

    monkeypatch.setattr(LocalOwner, "locate", count_locate)
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
                m.setattr(open_trace.ReadAttachOwnerLookupTrace, "__init__", forbidden_probe)
            config, journal = await _seed(root)
            auth_disk = (root / "auth.db").read_bytes()
            app, pool = _app(root, config)
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
                # Both arms need the same index residency; don't compare a
                # deadline-driven fallback to an index that happened to land.
                index_task, _ = transcript_index.start_refresh(root, _SID)
                assert await asyncio.wait_for(index_task, 10) is not None
                before = len(lookups)
                status, frames, wire = await _first_frame(app)
                assert lookups[before:] == [root]
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
                assert b"supplement-fixed" in wire
                # Normalize only independently minted transport IDs and the
                # marker-less fixture's cwd, then compare actual SSE bytes.
                wire = wire.replace(epoch.encode(), b"<epoch>").replace(sub.encode(), b"<sub>")
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://localhost",
                    headers={"Authorization": f"Bearer {_BEARER}"},
                ) as client:
                    history = await client.get(
                        f"/v1/desktop/sessions/{_SID}/history?open_frame=1&supplements=1"
                    )
                assert history.status_code == 200 and _TEXT.encode() in history.content
                assert b"supplement-fixed" in history.content
                responses[enabled] = (
                    wire.replace(str(home).encode(), b"<home>"),
                    history.content,
                    journal,
                )
            finally:
                await pool.close()
    assert responses[True] == responses[False]
    assert not network_attempts and not disabled_calls and listing.count(True) >= 2
    rows = _rows(trace_file)
    (ingress,) = [row for row in rows if row["stage"] == "desktop.events.acquire"]
    acquisitions = [row for row in rows if row["stage"] == "a.acquire"]
    (acquire,) = [row for row in acquisitions if row["acquire_id"] == ingress["acquire_id"]]
    (history_acquire,) = [row for row in acquisitions if row is not acquire]
    assert "owner_lookup_result_at_acquire" not in history_acquire
    (summary,) = [row for row in rows if row["stage"] == "cold.summary"]
    (lookup,) = [row for row in rows if row["stage"] == "a.attach.owner_lookup"]
    assert lookup == {
        "stage": "a.attach.owner_lookup",
        "at": lookup["at"],
        "pid": lookup["pid"],
        "acquire_id": ingress["acquire_id"],
        "result": "none",
    }
    assert acquire["owner_lookup_result_at_acquire"] == "none"
    assert acquire["draft_bridge"] is False and acquire["remote_branch"] is False
    assert rows.index(lookup) < rows.index(acquire)
    record_property("owner_lookup", lookup["result"])
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
    record_property("equal_history_bytes", len(responses[True][1]))
    record_property("cold_worker_phases", [row["phase"] for row in workers])


@pytest_asyncio.fixture
async def owner_trace_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[FastAPI, DesktopSessions, Path]]:
    home = tmp_path / "home"
    root = home / "config"
    root.mkdir(parents=True)
    trace_file = tmp_path / "open.jsonl"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", _BEARER)
    monkeypatch.setattr(secrets_client, "PEER_AUTHENTICATION_SUPPORTED", False)
    monkeypatch.setattr(open_trace, "_PATH", str(trace_file))
    monkeypatch.setattr(
        configure,
        "resolve_model_info",
        lambda _provider, model_id, **_: ModelInfo(
            id=model_id, name=model_id, description="", context_window=128_000, max_tokens=8192
        ),
    )
    config, _ = await _seed(root)
    app, pool = _app(root, config)
    try:
        yield app, pool, trace_file
    finally:
        await pool.close()


def _acquire_observations(rows: list[dict[str, Any]]) -> dict[str, str]:
    # Reject a stray result on /history, warm, or any flight that did not
    # belong to one of these ingress requests.
    opens = [row for row in rows if row["stage"] == "desktop.events.acquire"]
    open_ids = {row["acquire_id"] for row in opens}
    assert len(open_ids) == len(opens)
    acquisitions = {
        row["acquire_id"]: row["owner_lookup_result_at_acquire"]
        for row in rows
        if row["stage"] == "a.acquire"
    }
    assert acquisitions.keys() == open_ids
    return {row["acquire_id"]: acquisitions[row["acquire_id"]] for row in opens}


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", ["record", "pid_only"])
async def test_lookup_is_attributed_before_recovery_or_dial(
    owner_trace_app: tuple[FastAPI, DesktopSessions, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    expected: str,
) -> None:
    app, _, trace_file = owner_trace_app
    record = SessionRecord(876543210, "tui", _SID, _TITLE, "/synthetic-cwd", "model", 1, _BEARER)
    calls: list[str] = []

    def locate(self: LocalOwner) -> tuple[SessionRecord | None, int | None]:
        calls.append("locate")
        return (record if expected == "record" else None), record.pid

    def recover(root: Path, pid: int) -> SessionRecord:
        calls.append("recover")
        assert pid == record.pid
        return record

    async def refuse_dial(self: AttachedSession, selected: SessionRecord, **_: Any) -> None:
        calls.append("dial")
        assert selected is record
        [lookup] = [row for row in _rows(trace_file) if row["stage"] == "a.attach.owner_lookup"]
        assert lookup["result"] == expected
        raise ConnectionRefusedError("synthetic: no socket opened")

    monkeypatch.setattr(LocalOwner, "locate", locate)
    monkeypatch.setattr(attach_client, "dialable_owner_record", recover)
    monkeypatch.setattr(AttachedSession, "_bind_to", refuse_dial)
    status, frames, _ = await _first_frame(app)
    assert status == 200 and [frame["type"] for frame in frames] == ["open", "snapshot"]
    expected_calls = ["locate", "recover", "dial"] if expected == "pid_only" else ["locate", "dial"]
    assert calls == expected_calls
    rows = _rows(trace_file)
    [(acquire_id, observed)] = _acquire_observations(rows).items()
    assert observed == expected
    [lookup] = [row for row in rows if row["stage"] == "a.attach.owner_lookup"]
    assert lookup["acquire_id"] == acquire_id and lookup["result"] == expected
    assert set(lookup) == {"at", "stage", "pid", "acquire_id", "result"}
    assert str(record.pid) not in json.dumps(lookup) and _BEARER not in json.dumps(lookup)
    record_property("asgi_result", {"status": status, "lookup": observed, "calls": calls})


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["delayed", "cancelled"])
async def test_shared_open_never_inherits_the_original_lookup(
    owner_trace_app: tuple[FastAPI, DesktopSessions, Path],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    cancel: bool,
) -> None:
    app, pool, trace_file = owner_trace_app
    started, release, returned = threading.Event(), threading.Event(), threading.Event()
    looked_up: list[bool] = []

    def locate(self: LocalOwner) -> tuple[None, None]:
        looked_up.append(True)
        started.set()
        try:
            assert release.wait(10), "test must release the blocked owner lookup"
            return None, None
        finally:
            returned.set()

    monkeypatch.setattr(LocalOwner, "locate", locate)
    first_open = asyncio.create_task(_first_frame(app))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        replies = await asyncio.gather(first_open, _first_frame(app))
        assert [reply[0] for reply in replies] == [200, 200]
        assert all([f["type"] for f in reply[1]] == ["open", "snapshot"] for reply in replies)
        assert looked_up == [True]
        rows = _rows(trace_file)
        observations = _acquire_observations(rows)
        original, shared = observations
        assert list(observations.values()) == ["unknown", "unknown"]
        assert not [row for row in rows if row["stage"] == "a.attach.owner_lookup"]
        flight = pool.bridges[_SID].read_attach_task
        assert flight is not None and not flight.done()
        if cancel:
            flight.cancel()
            with pytest.raises(asyncio.CancelledError):
                await flight
        release.set()
        assert await asyncio.to_thread(returned.wait, 5)
        if not cancel:
            await asyncio.wait_for(asyncio.shield(flight), timeout=5)
            # The no-owner memo skips locate entirely: it must not inherit
            # the first request's actual result, even though the tuple is known.
            assert (await _first_frame(app))[0] == 200
        rows = _rows(trace_file)
        observations = _acquire_observations(rows)
        assert list(observations.values()) == ["unknown"] * (2 if cancel else 3)
        results = [row for row in rows if row["stage"] == "a.attach.owner_lookup"]
        assert looked_up == [True]
        if cancel:
            assert not results
        else:
            [lookup] = results
            assert lookup["acquire_id"] == original and lookup["acquire_id"] != shared
            assert lookup["result"] == "none"
            headers = [row for row in rows if row["stage"] == "desktop.events.response_start"]
            assert all(rows.index(lookup) > rows.index(row) for row in headers[:2])
        record_property("request_results_at_acquire", list(observations.values()))
        record_property("attempt_results", [row["result"] for row in results])
        record_property("lookup_calls", len(looked_up))
    finally:
        release.set()
        if not first_open.done():
            first_open.cancel()
            await asyncio.gather(first_open, return_exceptions=True)


@pytest.mark.asyncio
async def test_draft_scope_uses_the_minted_bridge(
    owner_trace_app: tuple[FastAPI, DesktopSessions, Path],
    record_property: Callable[[str, object], None],
) -> None:
    app, pool, trace_file = owner_trace_app
    sid = await pool.mint_draft(str(pool.root))
    status, frames, _ = await _first_frame(app, sid=sid)
    assert status == 200 and [frame["type"] for frame in frames] == ["open", "snapshot"]
    [acquire] = [row for row in _rows(trace_file) if row["stage"] == "a.acquire"]
    assert acquire["draft_bridge"] is True and acquire["remote_branch"] is False
    assert acquire["owner_lookup_result_at_acquire"] == "none"
    record_property("draft_asgi_result", {"status": status, "lookup": "none"})
