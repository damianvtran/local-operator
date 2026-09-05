"""Exercise the real relay route, including stale receipt and passive SSE cases."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

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


def test_legacy_metadata_mtime_is_not_a_completion(tmp_path: Path) -> None:
    daemon = MobileDaemon(port=0, password="isolated-test")
    assert not daemon.table._is_unseen("no-outcome", SimpleNamespace(mtime=1e30), None)
