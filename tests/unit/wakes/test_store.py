"""The wake index store: one JSON file per wake-carrying session, derived
from the transcript, absent ⇒ no wakes."""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.harness.wake import WakeSchedule
from local_operator.wakes import store


def _schedule(sid: str = "w1", due: int = 1_700_000_060_000) -> WakeSchedule:
    return WakeSchedule(id=sid, message="check in", next_due_at=due, created_at=1_700_000_000_000)


def test_absent_directory_reads_as_empty_index(tmp_path: Path) -> None:
    assert store.read_index(tmp_path) == {}
    assert store.read_entry(tmp_path, "nope") is None


def test_write_entry_shape_matches_transcript_dump(tmp_path: Path) -> None:
    schedule = _schedule()
    path = store.write_entry(tmp_path, "s1", cwd="/work", schedules=[schedule])
    assert path == tmp_path / "wakes" / "s1.json"
    assert path is not None and path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == store.INDEX_SCHEMA
    assert data["session_id"] == "s1"
    assert data["cwd"] == "/work"
    assert isinstance(data["updated_at"], int)
    # The row is exactly the transcript's persisted form.
    assert data["schedules"] == [schedule.model_dump()]
    # No staged temp file survives.
    assert [p.name for p in (tmp_path / "wakes").iterdir()] == ["s1.json"]


def test_write_accepts_pre_dumped_dicts(tmp_path: Path) -> None:
    rows = [_schedule().model_dump()]
    store.write_entry(tmp_path, "s1", cwd="/work", schedules=rows)
    assert store.read_entry(tmp_path, "s1")["schedules"] == rows  # type: ignore[index]


def test_empty_schedules_removes_the_entry(tmp_path: Path) -> None:
    store.write_entry(tmp_path, "s1", cwd="/work", schedules=[_schedule()])
    assert store.write_entry(tmp_path, "s1", cwd="/work", schedules=[]) is None
    assert not (tmp_path / "wakes" / "s1.json").exists()
    assert store.read_index(tmp_path) == {}


def test_remove_entry_is_idempotent(tmp_path: Path) -> None:
    assert store.remove_entry(tmp_path, "s1") is False
    store.write_entry(tmp_path, "s1", cwd="/work", schedules=[_schedule()])
    assert store.remove_entry(tmp_path, "s1") is True
    assert store.remove_entry(tmp_path, "s1") is False


def test_read_index_scans_directory_and_keys_by_filename(tmp_path: Path) -> None:
    store.write_entry(tmp_path, "s1", cwd="/a", schedules=[_schedule("w1")])
    store.write_entry(tmp_path, "s2", cwd="/b", schedules=[_schedule("w2")])
    # Staged temp files, dotfiles and non-json strays are never entries.
    (tmp_path / "wakes" / ".s3.abc.tmp").write_text("{}", encoding="utf-8")
    (tmp_path / "wakes" / "README").write_text("x", encoding="utf-8")
    index = store.read_index(tmp_path)
    assert sorted(index) == ["s1", "s2"]
    assert index["s1"]["cwd"] == "/a"
    assert index["s2"]["schedules"][0]["id"] == "w2"


def test_unreadable_or_foreign_schema_entries_are_skipped(tmp_path: Path) -> None:
    store.write_entry(tmp_path, "good", cwd="/a", schedules=[_schedule()])
    (tmp_path / "wakes" / "torn.json").write_text('{"schema": 1, "sched', encoding="utf-8")
    (tmp_path / "wakes" / "future.json").write_text(
        json.dumps({"schema": 99, "session_id": "future", "schedules": []}), encoding="utf-8"
    )
    index = store.read_index(tmp_path)
    assert sorted(index) == ["good"]
    assert store.read_entry(tmp_path, "torn") is None
    assert store.read_entry(tmp_path, "future") is None


def test_preserve_keeps_unknown_keys_and_clear_drops_them(tmp_path: Path) -> None:
    store.write_entry(tmp_path, "s1", cwd="/a", schedules=[_schedule()])
    path = tmp_path / "wakes" / "s1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stopped_at"] = 1_700_000_100_000
    data["future_key"] = {"nested": True}
    path.write_text(json.dumps(data), encoding="utf-8")

    # A persist-time rewrite preserves what it does not know about.
    existing = store.read_entry(tmp_path, "s1")
    store.write_entry(tmp_path, "s1", cwd="/a", schedules=[_schedule("w2")], preserve=existing)
    after = store.read_entry(tmp_path, "s1")
    assert after is not None
    assert after["stopped_at"] == 1_700_000_100_000
    assert after["future_key"] == {"nested": True}
    assert after["schedules"][0]["id"] == "w2"

    # The open-time rewrite clears stopped_at and nothing else.
    existing = store.read_entry(tmp_path, "s1")
    store.write_entry(
        tmp_path,
        "s1",
        cwd="/a",
        schedules=[_schedule("w2")],
        preserve=existing,
        clear=("stopped_at",),
    )
    after = store.read_entry(tmp_path, "s1")
    assert after is not None
    assert "stopped_at" not in after
    assert after["future_key"] == {"nested": True}


def test_preserve_cannot_override_authoritative_fields(tmp_path: Path) -> None:
    stale = {"schema": 0, "session_id": "other", "cwd": "/stale", "schedules": [{"id": "old"}]}
    store.write_entry(tmp_path, "s1", cwd="/fresh", schedules=[_schedule()], preserve=stale)
    after = store.read_entry(tmp_path, "s1")
    assert after is not None
    assert after["schema"] == store.INDEX_SCHEMA
    assert after["session_id"] == "s1"
    assert after["cwd"] == "/fresh"
    assert after["schedules"][0]["id"] == "w1"


def test_next_due_at_is_the_earliest_valid_row() -> None:
    entry = {
        "schedules": [
            {"id": "a", "next_due_at": 300},
            {"id": "b", "next_due_at": 100},
            {"id": "c", "next_due_at": "soon"},
            {"id": "d", "next_due_at": True},
            "garbage",
        ]
    }
    assert store.next_due_at(entry) == 100
    assert store.next_due_at({"schedules": []}) is None
    assert store.next_due_at({}) is None


def test_write_is_staged_then_replaced(tmp_path: Path, monkeypatch) -> None:
    """A failure mid-write leaves the previous entry intact and no temp file."""
    store.write_entry(tmp_path, "s1", cwd="/a", schedules=[_schedule("w1")])
    before = (tmp_path / "wakes" / "s1.json").read_bytes()

    def boom(*args, **kwargs):  # noqa: ANN001, ANN202
        raise OSError("disk full")

    monkeypatch.setattr(store.os, "replace", boom)
    try:
        store.write_entry(tmp_path, "s1", cwd="/a", schedules=[_schedule("w2")])
    except OSError:
        pass
    else:  # pragma: no cover - the monkeypatch guarantees the raise
        raise AssertionError("expected the staged write to raise")
    assert (tmp_path / "wakes" / "s1.json").read_bytes() == before
    assert [p.name for p in (tmp_path / "wakes").iterdir()] == ["s1.json"]
