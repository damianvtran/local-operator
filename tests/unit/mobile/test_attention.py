"""Exercise the real relay route, including stale receipt and passive SSE cases."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, _projection_frame, build_app
from local_operator.mobile.types import SessionProjection
from local_operator.paths import config_dir
from local_operator.session.attention import AttentionStore


def test_mobile_requires_exact_observed_token_and_never_reads_on_subscription() -> None:
    sid = "abcdef123456"
    directory = config_dir() / "sessions" / sid
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("")
    store = AttentionStore()
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    store.publish(f"session/{sid}", a, "answer-a", "complete")
    daemon = MobileDaemon(port=0, password="isolated-test")
    row = SimpleNamespace(mtime=1e30)
    daemon.table._attention_states = store.state_many([f"session/{sid}"])
    assert daemon.table._is_unseen(sid, row, None)
    daemon.table.session_subscribers[sid] = {asyncio.Queue(maxsize=1)}
    assert daemon.table._is_unseen(sid, row, None)
    before = (config_dir() / "attention.db").stat().st_mtime_ns
    _projection_frame(SessionProjection(session_id=sid, pid=12345))
    assert (config_dir() / "attention.db").stat().st_mtime_ns == before
    client = TestClient(build_app(daemon), follow_redirects=False)
    route = f"/api/sessions/{sid}/seen"
    assert client.post(route, json={"completion_token": a}).status_code == 401
    client.post("/login", data={"password": "isolated-test"})
    assert client.post(route).status_code == 422
    assert client.post(route, json={"completion_token": 123}).status_code == 422
    assert client.post(route, json={"completion_token": str(uuid.uuid4())}).status_code == 409
    assert (
        client.post("/api/sessions/deadbeef1234/seen", json={"completion_token": a}).status_code
        == 404
    )
    store.publish(f"session/{sid}", b, "answer-b", "complete")
    response = client.post(route, json={"completion_token": a})
    assert response.status_code == 200
    assert response.json()["attention"]["unseen"]
    assert response.json()["attention"]["completion_token"] == b
    assert client.post(route, json={"completion_token": b}).json()["attention"]["unseen"] is False
    assert client.post(route, json={"completion_token": a}).json()["attention"]["unseen"] is False
    assert not AttentionStore(config_dir() / "attention.db").state(f"session/{sid}")["unseen"]


@pytest.mark.asyncio
async def test_summary_receipts_are_one_batch_off_the_event_loop(monkeypatch) -> None:
    from local_operator.mobile.daemon import SessionTable

    store = AttentionStore()
    ids = [f"session-{index}" for index in range(40)]
    for session_id in ids:
        store.publish(f"session/{session_id}", str(uuid.uuid4()), "result", "complete")
    table = SessionTable()

    async def durable_rows():
        return dict.fromkeys(ids)

    monkeypatch.setattr(table, "_refresh_durable_rows", durable_rows)
    monkeypatch.setattr(
        table, "_merge_summaries", lambda rows: list(table._attention_states.values())
    )
    original = sqlite3.connect
    threads: list[int] = []

    def connect(*args, **kwargs):
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    rows = await table.summaries()
    assert len(rows) == 40
    assert len(threads) == 1
    assert threads[0] != threading.get_ident()
    for session_id in ids:
        assert table._is_unseen(session_id, None, None)
    assert len(threads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["file_created", "partial_transaction"])
async def test_first_publication_is_safe_for_batch_revision_and_authenticated_list(
    monkeypatch, stage
) -> None:
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.transcript import Transcript

    sid = "fedcba654321"
    transcript = Transcript(config_dir() / "sessions" / sid)
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="Completed result")])
    )
    path = config_dir() / "attention.db"
    created, release = threading.Event(), threading.Event()
    original_touch, original_connect = Path.touch, sqlite3.connect

    def pause() -> None:
        created.set()
        if not release.wait(10):
            raise RuntimeError("test did not release the first publisher")

    def touch(self: Path, *args: Any, **kwargs: Any) -> None:
        original_touch(self, *args, **kwargs)
        if self == path and stage == "file_created":
            pause()

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        if stage == "partial_transaction" and threading.current_thread().name.startswith(
            "attention-writer"
        ):

            def trace(sql: str) -> None:
                if "CREATE TABLE" in sql.upper() and "receipts" in sql:
                    pause()

            conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(Path, "touch", touch)
    monkeypatch.setattr(sqlite3, "connect", connect)
    daemon = MobileDaemon(port=0, password="isolated-first-publication")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "isolated-first-publication"})
    store = AttentionStore(path)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="attention-writer") as pool:
        publisher = pool.submit(
            store.publish, f"session/{sid}", str(uuid.uuid4()), "result", "complete"
        )
        try:
            assert created.wait(10)
            assert store.state_many([f"session/{sid}"])[f"session/{sid}"]["unseen"] is False
            assert store.revision() == (0, 0)
            response = client.get("/api/sessions")
            assert response.status_code == 200
            assert response.json()["sessions"][0]["unseen"] is False
        finally:
            release.set()
            publisher.result(timeout=10)
    assert store.state(f"session/{sid}")["unseen"] is True


@pytest.mark.parametrize(
    "final_size,neighbor_size,complete",
    [(1_000_000, 0, False), (5000, 1_000_000, False), (20, 1_000_000, True)],
)
def test_completion_end_completeness_survives_runtime_and_relay_caps(
    final_size, neighbor_size, complete
) -> None:
    from local_operator.mobile.types import TranscriptEntry, _projection_from_json
    from local_operator.session.runtime.server import RuntimeServer

    ending = "TRUE_FINAL_RESULT_END"
    original = "x" * final_size + ending
    projection = SessionProjection(
        session_id="cap-session",
        pid=0,
        attention={
            "conversation_id": "session/cap-session",
            "completion_token": str(uuid.uuid4()),
            "anchor_id": "final",
            "kind": "complete",
            "unseen": True,
            "revision": [1, 0],
        },
        transcript=[
            TranscriptEntry(id="neighbor", kind="assistant", text="n" * neighbor_size),
            TranscriptEntry(id="final", kind="assistant", text=original),
        ],
    )
    runtime = RuntimeServer(
        cast(Any, SimpleNamespace(session_projection_seed=projection)), kind="tui"
    )
    payload = runtime._projection_payload()["data"]
    received = _projection_from_json(payload, runtime._record)
    relayed = _projection_frame(received)
    row = next(row for row in relayed["transcript"] if row["id"] == "final")
    assert row["text_complete"] is complete
    assert (ending in row["text"]) is complete
    assert relayed["attention"]["anchor_id"] == row["id"]
    assert projection.transcript[-1].text == original
    assert projection.transcript[-1].text_complete


@pytest.mark.asyncio
async def test_cold_receipt_repaint_keeps_the_existing_relay_epoch(monkeypatch) -> None:
    daemon = MobileDaemon(port=0, password="isolated-cold-repaint")
    daemon._attention_bootstrapped = True
    sid = "cold-session"
    previous = daemon.capture_subagent_details(SessionProjection(session_id=sid, pid=0, version=80))
    prior_version = previous.version
    cold = SessionProjection(session_id=sid, pid=0, version=1)
    monkeypatch.setattr("local_operator.mobile.daemon._durable_projection", lambda session_id: cold)
    monkeypatch.setattr("local_operator.mobile.daemon.registry.scan", lambda *args, **kwargs: [])
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
    daemon.table.session_subscribers[sid] = {queue}
    store = AttentionStore()
    token = str(uuid.uuid4())
    store.publish(f"session/{sid}", token, "result", "complete")
    store.acknowledge(f"session/{sid}", token)
    await daemon._scan_once()
    frame = queue.get_nowait()
    assert frame["version"] == prior_version
    assert frame["attention"]["completion_token"] == token
    assert not frame["attention"]["unseen"]


def test_relay_only_cap_and_unknown_old_owner_never_claim_complete() -> None:
    from local_operator.mobile.types import TranscriptEntry, _projection_from_json
    from local_operator.session.runtime.server import RuntimeServer

    projection = SessionProjection(
        session_id="cap-session",
        pid=0,
        transcript=[TranscriptEntry(id="final", kind="assistant", text="f" * 5000 + "TRUE_END")],
    )
    runtime = RuntimeServer(
        cast(Any, SimpleNamespace(session_projection_seed=projection)), kind="tui"
    )
    payload = runtime._projection_payload()["data"]
    assert payload["transcript"][0]["text_complete"]
    received = _projection_from_json(payload, runtime._record)
    received.transcript.insert(
        0, TranscriptEntry(id="neighbor", kind="assistant", text="n" * 1_000_000)
    )
    row = _projection_frame(received)["transcript"][-1]
    assert not row["text_complete"] and "TRUE_END" not in row["text"]
    payload["transcript"][0].pop("text_complete")
    legacy = _projection_from_json(payload, runtime._record)
    assert not legacy.transcript[0].text_complete
    assert not _projection_frame(legacy)["transcript"][0]["text_complete"]


def test_legacy_metadata_mtime_is_not_a_completion(tmp_path: Path) -> None:
    daemon = MobileDaemon(port=0, password="isolated-test")
    assert not daemon.table._is_unseen("no-outcome", SimpleNamespace(mtime=1e30), None)
