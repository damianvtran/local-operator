"""Capture identifiers are structure, never content for the secret-shape guard."""

import asyncio

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
def test_source_identifiers_are_structural(field):
    fact = {
        "title": "Index",
        "body": "Use SQLite",
        "source": {"kind": "compaction", field: COLLISION},
    }
    assert validate_fact(fact)["source"][field] == COLLISION
    fact["source"][field] = "../escape"
    with pytest.raises(MemoryError):
        validate_fact(fact)


@pytest.mark.parametrize("field", ["author", "uri", "tool"])
def test_source_content_remains_guarded(field):
    fact = {
        "title": "Index",
        "body": "Use SQLite",
        "source": {"kind": "compaction", field: COLLISION},
    }
    with pytest.raises(MemoryError):
        validate_fact(fact)
