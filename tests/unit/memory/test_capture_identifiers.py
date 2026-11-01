"""Capture identifiers are structure, never content for the secret-shape guard."""

import asyncio
import json
import sqlite3
import uuid

import pytest

from local_operator import memory_capture
from local_operator.memory import (
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    guard,
    validate_fact,
)

COLLISION = "cc57d53b05cb48ac9d2740b56736f266"
CONTROL = "cc57d53b05cb48ac9d2740b56736f267"


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "cfg", MemoryIdentity("capture-test", "t", "a"), None)


@pytest.mark.parametrize("marker", [CONTROL, COLLISION])
def test_capture_identifier_is_not_content(store, marker):
    rows = memory_capture.extract("", "## Key Decisions\n- Use SQLite for the local index")
    assert rows
    assert memory_capture.persist(store, marker, rows, "ok") == ("ok", None)
    with store.connection() as db:
        assert tuple(db.execute("SELECT compaction, status, n FROM captures").fetchone()) == (
            marker,
            "ok",
            len(rows),
        )


@pytest.mark.parametrize("scan_fails", [False, True])
@pytest.mark.parametrize("marker", [CONTROL, COLLISION])
def test_failure_recorder_keeps_terminal_outcome(store, marker, scan_fails, monkeypatch):
    calls = []
    original = memory_capture.persist

    def traced(*args, **kwargs):
        calls.append((args[1], args[3]))
        return original(*args, **kwargs)

    monkeypatch.setattr(memory_capture, "persist", traced)

    async def run():
        async def scan():
            if scan_fails:
                raise RuntimeError("synthetic scan failure")
            return memory_capture.extract("", "## Key Decisions\n- Use SQLite for the local index")

        return await memory_capture.finish(store, marker, asyncio.create_task(scan()), "")

    assert asyncio.run(run()) == ("failed" if scan_fails else "ok")
    assert calls == [(marker, "failed" if scan_fails else "ok")]
    with store.connection() as db:
        assert tuple(db.execute("SELECT compaction, status, n FROM captures").fetchone()) == (
            marker,
            "failed" if scan_fails else "ok",
            0 if scan_fails else 1,
        )


@pytest.mark.parametrize("body", [COLLISION, "DE89370400440532013000", "GB82WEST12345698765432"])
def test_content_guard_is_unchanged(store, body):
    with pytest.raises(MemoryError):
        guard(body)
    rows = [{"title": "Account detail", "body": body, "tags": ["decision"]}]
    assert memory_capture.persist(store, CONTROL, rows, "ok") == ("failed", "write-refused")


@pytest.mark.parametrize("marker", [None, [], "", ".", "../escape", "has space", "x" * 129])
def test_malformed_marker_is_refused(store, marker):
    with pytest.raises(MemoryError):
        memory_capture.persist(store, marker, [], "failed")


@pytest.mark.parametrize("field", ["session", "entry"])
@pytest.mark.parametrize(
    "value", [COLLISION, "DE89370400440532013000", "GB82WEST12345698765432", "4111111111111111"]
)
def test_untrusted_source_identifiers_cannot_bypass_content_guard(store, field, value):
    # These are the reviewer's exact put payloads: source is caller-controlled,
    # not trusted provenance merely because a field is called session or entry.
    fact = {"title": "Audit", "body": "Use SQLite", "source": {"kind": "reported", field: value}}
    with pytest.raises(MemoryError):
        validate_fact(fact)
    with pytest.raises(MemoryError):
        store.put("session", fact)
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM ops").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM current").fetchone()[0] == 0


def test_caller_cannot_override_host_session(store):
    prepared = store.prepare_fact(
        {"title": "Audit", "body": "Use SQLite", "source": {"kind": "reported", "session": "other"}}
    )
    assert prepared["source"]["session"] == store.identity.session_id


def test_host_minted_capture_keeps_marker_out_of_fact_content(store):
    marker = uuid.uuid4().hex
    rows = memory_capture.extract("", "## Key Decisions\n- Use SQLite for the local index")
    assert memory_capture.persist(store, marker, rows, "ok") == ("ok", None)
    with store.connection() as db:
        fact = json.loads(db.execute("SELECT fact FROM current").fetchone()[0])
        assert "entry" not in fact["source"]
        assert marker not in json.dumps(fact)
        assert tuple(db.execute("SELECT compaction, status, n FROM captures").fetchone()) == (
            marker,
            "ok",
            1,
        )
    assert memory_capture.capture_detail(store, marker) is not None


def test_persist_fault_records_failed_with_same_collision_marker(store, monkeypatch):
    calls = []
    original = memory_capture.persist

    def fail_first(*args, **kwargs):
        calls.append((args[1], args[3]))
        if len(calls) == 1:
            raise sqlite3.OperationalError("synthetic transient store fault")
        return original(*args, **kwargs)

    monkeypatch.setattr(memory_capture, "persist", fail_first)

    async def run():
        async def scan():
            return memory_capture.extract("", "## Key Decisions\n- Use SQLite for the local index")

        return await memory_capture.finish(store, COLLISION, asyncio.create_task(scan()), "")

    assert asyncio.run(run()) == "failed"
    assert calls == [(COLLISION, "ok"), (COLLISION, "failed")]
    with store.connection() as db:
        assert tuple(db.execute("SELECT compaction, status, n FROM captures").fetchone()) == (
            COLLISION,
            "failed",
            0,
        )
        assert db.execute("SELECT count(*) FROM current").fetchone()[0] == 0


@pytest.mark.parametrize("field", ["author", "uri", "tool"])
def test_source_content_remains_guarded(field):
    fact = {
        "title": "Index",
        "body": "Use SQLite",
        "source": {"kind": "compaction", field: COLLISION},
    }
    with pytest.raises(MemoryError):
        validate_fact(fact)
