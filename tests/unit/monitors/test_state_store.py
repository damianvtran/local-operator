"""State files and the derived index: shapes, tolerance, self-healing seams."""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store


def test_counters_round_trip_and_unknown_schema_reads_as_absent(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    monitor_state.write_counters(root, "sess", "m1", {"schema": 1, "checks": 3})
    assert monitor_state.read_counters(root, "sess", "m1") == {"schema": 1, "checks": 3}

    path = monitor_state.counters_path(root, "sess", "m1")
    path.write_text(json.dumps({"schema": 99, "checks": 3}), encoding="utf-8")
    assert monitor_state.read_counters(root, "sess", "m1") is None  # never raises


def test_a_corrupt_counters_file_reads_as_absent(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    path = monitor_state.counters_path(root, "sess", "m1")
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert monitor_state.read_counters(root, "sess", "m1") is None


def test_snapshot_round_trip_and_truncation_flag(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    monitor_state.write_snapshot(root, "sess", "m1", "01234", truncated=True)
    blob = monitor_state.read_snapshot(root, "sess", "m1")
    assert blob == {
        "schema": 1,
        "monitor_id": "m1",
        "snapshot": "01234",
        "snapshot_truncated": True,
    }
    monitor_state.write_snapshot(root, "sess", "m1", "full", truncated=False)
    blob = monitor_state.read_snapshot(root, "sess", "m1")
    assert blob is not None and blob["snapshot_truncated"] is False


def test_writes_are_atomic_and_leave_no_temp_files(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    monitor_state.write_counters(root, "sess", "m1", {"schema": 1})
    children = sorted(child.name for child in monitor_state.state_dir(root, "sess").iterdir())
    assert children == ["m1.json"]  # no .tmp leftovers


def test_remove_monitor_state_and_session_state(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    monitor_state.write_counters(root, "sess", "m1", {"schema": 1})
    monitor_state.write_snapshot(root, "sess", "m1", "x", truncated=False)
    monitor_state.remove_monitor_state(root, "sess", "m1")
    assert monitor_state.read_counters(root, "sess", "m1") is None
    assert monitor_state.read_snapshot(root, "sess", "m1") is None

    monitor_state.write_counters(root, "sess", "m1", {"schema": 1})
    monitor_state.write_counters(root, "sess", "m2", {"schema": 1})
    monitor_state.remove_session_state(root, "sess")
    assert not monitor_state.state_dir(root, "sess").exists()


def test_the_index_round_trips_rows(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    rows = [
        {
            "id": "m1",
            "name": "watch",
            "tool": "bash",
            "arguments": {"command": "date"},
            "every_ms": 60_000,
            "next_due_at": 123,
        }
    ]
    monitor_store.write_entry(root, "sess", cwd="/w", monitors=rows)
    entry = monitor_store.read_entry(root, "sess")
    assert entry is not None
    assert entry["schema"] == monitor_store.INDEX_SCHEMA
    assert entry["cwd"] == "/w"
    assert entry["monitors"][0]["id"] == "m1"
    index, error = monitor_store.read_index_report(root)
    assert error is False
    assert index["sess"]["session_id"] == "sess"


def test_an_empty_list_removes_the_entry(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    monitor_store.write_entry(root, "sess", cwd="/w", monitors=[{"id": "m1"}])
    assert monitor_store.entry_path(root, "sess").exists()
    monitor_store.write_entry(root, "sess", cwd="/w", monitors=[])
    assert not monitor_store.entry_path(root, "sess").exists()
    assert monitor_store.read_entry(root, "sess") is None


def test_the_index_scan_skips_state_temp_and_dotfiles(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    monitor_store.write_entry(root, "sess", cwd="/w", monitors=[{"id": "m1"}])
    state_dir = root / "monitors" / "state" / "sess"
    state_dir.mkdir(parents=True)
    (state_dir / "m1.json").write_text("{}", encoding="utf-8")  # never an index entry
    (root / "monitors" / ".tmp-x.json").write_text("{}", encoding="utf-8")
    index = monitor_store.read_index(root)
    assert list(index) == ["sess"]


def test_preserve_keeps_stopped_at_and_clear_drops_it(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    monitor_store.write_entry(root, "sess", cwd="/w", monitors=[{"id": "m1"}])
    entry = monitor_store.read_entry(root, "sess")
    assert entry is not None
    monitor_store.write_entry(
        root, "sess", cwd="/w", monitors=[{"id": "m1"}], preserve=dict(entry, stopped_at=42)
    )
    held = monitor_store.read_entry(root, "sess")
    assert held is not None and held["stopped_at"] == 42
    monitor_store.write_entry(
        root,
        "sess",
        cwd="/w",
        monitors=[{"id": "m1"}],
        preserve=monitor_store.read_entry(root, "sess"),
        clear=("stopped_at",),
    )
    cleared = monitor_store.read_entry(root, "sess")
    assert cleared is not None and "stopped_at" not in cleared


def test_has_armed_counts_only_what_can_fire() -> None:
    now = 1_000_000
    assert monitor_store.has_armed(None, now) is False
    assert monitor_store.has_armed({"monitors": []}, now) is False
    assert monitor_store.has_armed({"monitors": [{"id": "m1"}]}, now) is True
    assert monitor_store.has_armed({"monitors": [{"disabled": True}]}, now) is False
    assert monitor_store.has_armed({"monitors": [{"until_at": now - 1}]}, now) is False
    assert monitor_store.has_armed({"monitors": [{"until_at": now + 1}]}, now) is True
    assert monitor_store.has_armed({"monitors": ["junk"]}, now) is False


def test_is_held_reads_the_stop_marker() -> None:
    assert monitor_store.is_held({"stopped_at": 1}) is True
    assert monitor_store.is_held({"stopped_at": 0}) is False
    assert monitor_store.is_held({}) is False
    assert monitor_store.is_held(None) is False


def test_next_due_at_reports_the_earliest_live_row() -> None:
    entry = {
        "monitors": [
            {"next_due_at": 50},
            {"next_due_at": 10},
            {"next_due_at": 20, "disabled": True},
            {"next_due_at": "junk"},
        ]
    }
    assert monitor_store.next_due_at(entry) == 10
    assert monitor_store.next_due_at({"monitors": []}) is None


def test_an_unreadable_directory_reports_the_read_error(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    blocked = root / "monitors"
    blocked.mkdir(parents=True)
    blocked.chmod(0)
    try:
        index, error = monitor_store.read_index_report(root)
    finally:
        blocked.chmod(0o700)
    if not error:  # running as root: permissions do not bind
        return
    assert index == {}
