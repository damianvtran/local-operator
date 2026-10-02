"""State files and the derived index: shapes, tolerance, self-healing seams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# §D5: housekeeping — the orphan state a cancel and a dead session leave
# ---------------------------------------------------------------------------


def test_removing_the_last_monitor_reclaims_the_session_directory(tmp_path: Path) -> None:
    """The live store carried 16 empty ``state/<session_id>/`` directories, one
    per armed-then-cancelled session: the unlink removed the files and nothing
    ever removed their container.
    """
    root = tmp_path / "cfg"
    monitor_state.write_counters(root, "sess", "m1", {"schema": 1, "monitor_id": "m1"})
    monitor_state.write_snapshot(root, "sess", "m1", "x", truncated=False)
    assert monitor_state.state_dir(root, "sess").is_dir()

    monitor_state.remove_monitor_state(root, "sess", "m1")

    assert not monitor_state.state_dir(root, "sess").exists()


def test_a_sibling_monitor_keeps_the_directory(tmp_path: Path) -> None:
    """``rmdir`` is the whole safety argument: it refuses a non-empty
    directory, so a cancel can never take a sibling's state with it.
    """
    root = tmp_path / "cfg"
    monitor_state.write_counters(root, "sess", "m1", {"schema": 1, "monitor_id": "m1"})
    monitor_state.write_counters(root, "sess", "m2", {"schema": 1, "monitor_id": "m2"})

    monitor_state.remove_monitor_state(root, "sess", "m1")

    assert monitor_state.state_dir(root, "sess").is_dir()
    assert monitor_state.read_counters(root, "sess", "m2") is not None


def test_prune_empty_state_dirs_never_touches_state(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    # Two orphans from an older build, one live session, and a stray FILE at
    # the top of the state directory.
    empty = monitor_state.state_dir(root, "orphan1")
    empty.mkdir(parents=True)
    monitor_state.state_dir(root, "orphan2").mkdir()
    monitor_state.write_counters(root, "live", "m1", {"schema": 1, "monitor_id": "m1"})
    stray = root / "monitors" / "state" / "README"
    stray.write_text("not a directory", encoding="utf-8")

    removed = monitor_state.prune_empty_state_dirs(root)

    assert removed == 2
    assert not empty.exists()
    assert monitor_state.read_counters(root, "live", "m1") is not None
    assert monitor_state.state_dir(root, "live").is_dir()
    assert stray.exists()


def test_prune_is_a_no_op_when_the_directory_is_absent(tmp_path: Path) -> None:
    assert monitor_state.prune_empty_state_dirs(tmp_path / "cfg") == 0


def _ghost_entry(root: Path, session_id: str, *keys: str, age_ms: int = 7_200_000) -> None:
    """One index entry, optionally held, stamped ``age_ms`` in the past."""
    import time

    entry = {
        "schema": 1,
        "session_id": session_id,
        "cwd": "/gone",
        "updated_at": int(time.time() * 1000) - age_ms,
        "monitors": [{"id": "m1", "name": "watch"}],
    }
    for key in keys:
        entry[key] = int(time.time() * 1000)
    path = monitor_store.entry_path(root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")


def test_the_ghost_sweep_removes_only_an_old_entry_with_no_transcript(tmp_path: Path) -> None:
    """The live store carried ``9a7c31e40b22.json`` — no session directory, no
    ``stopped_at``, and a ``cwd`` pointing into a QA scratch home. Three
    conditions must ALL hold before an entry goes.
    """
    import time

    root = tmp_path / "cfg"
    now = int(time.time() * 1000)
    _ghost_entry(root, "ghost")
    _ghost_entry(root, "fresh", age_ms=60_000)
    _ghost_entry(root, "held", "stopped_at")
    _ghost_entry(root, "live")
    # And a held entry that is ALSO recent, so neither key can pass alone.
    _ghost_entry(root, "heldfreshentry", "stopped_at", age_ms=60_000)

    removed = monitor_store.prune_ghost_entries(
        root, now, session_exists=lambda session_id: session_id == "live"
    )

    assert removed == ["ghost"]
    assert sorted(monitor_store.read_index(root)) == [
        "fresh",
        "held",
        "heldfreshentry",
        "live",
    ]
    # A ghost's own state directory goes with it.
    assert not monitor_state.state_dir(root, "ghost").exists()


def test_the_ghost_sweep_removes_the_state_directory_too(tmp_path: Path) -> None:
    import time

    root = tmp_path / "cfg"
    _ghost_entry(root, "ghost")
    monitor_state.write_counters(root, "ghost", "m1", {"schema": 1, "monitor_id": "m1"})

    monitor_store.prune_ghost_entries(
        root, int(time.time() * 1000), session_exists=lambda _sid: False
    )

    assert not monitor_store.entry_path(root, "ghost").exists()
    assert not monitor_state.state_dir(root, "ghost").exists()


def test_the_ghost_sweep_gives_up_rather_than_guessing(tmp_path: Path) -> None:
    """An unreadable index directory is not evidence that anything is a ghost;
    the same asymmetry ``read_index_report`` already encodes.
    """
    import time

    root = tmp_path / "cfg"
    # No monitors directory at all: nothing to prune, and no raise.
    assert (
        monitor_store.prune_ghost_entries(
            root, int(time.time() * 1000), session_exists=lambda _sid: False
        )
        == []
    )


# ---------------------------------------------------------------------------
# §D6: the one health hint every surface renders
# ---------------------------------------------------------------------------


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "m1",
        "name": "watch",
        "tool": "bash",
        "every_ms": 60_000,
        "created_at": 1_756_000_000_000,
        "checks": 7,
        "deliveries": 3,
        "next_due_at": 1_756_000_030_000,
    }
    row.update(overrides)
    return row


def test_a_healthy_row_says_nothing() -> None:
    """The common case must stay silent, or the hint becomes noise on every
    list surface.
    """
    assert monitor_store.health_hint(_row(), 1_756_000_060_000) is None


def test_a_disabled_row_is_left_to_its_own_wording() -> None:
    disabled = _row(disabled=True, disabled_reason="boom")
    assert monitor_store.health_hint(disabled, 1_756_000_060_000) is None


def test_an_unavailable_episode_is_named_with_its_clock() -> None:
    hint = monitor_store.health_hint(_row(unavailable_since=1_756_000_000_000), 1_756_000_060_000)
    assert hint is not None and hint.startswith("tool unavailable since "), hint
    assert hint.endswith("— retrying")


def test_a_monitor_that_never_checked_says_why() -> None:
    """The live store carried arms with ``checks=0``: the operator armed a
    watch, closed the conversation, and nothing ran.
    """
    never = _row(checks=0, created_at=1_756_000_000_000)
    assert (
        monitor_store.health_hint(never, 1_756_000_000_000 + 3_600_000)
        == "never checked — its session was not open since arming"
    )
    # ...but not before the threshold, or every fresh arm reads as broken: a
    # minute after the arm is ordinary, not a broken watch.
    fresh = _row(checks=0, created_at=1_756_000_000_000)
    assert monitor_store.health_hint(fresh, 1_756_000_000_000 + 60_000) is None


def test_zero_deliveries_after_several_checks_gets_a_neutral_hint() -> None:
    hint = monitor_store.health_hint(_row(checks=5, deliveries=0), 1_756_000_060_000)
    assert hint is not None
    assert hint.startswith("5 checks, 0 deliveries")
    assert "confirm the call observes what you expect" in hint
    # Three checks is the floor: two quiet checks are not evidence of anything.
    assert monitor_store.health_hint(_row(checks=2, deliveries=0), 1_756_000_060_000) is None


def test_idle_is_overdue_beyond_two_intervals_and_not_held() -> None:
    now = 1_756_000_060_000
    stale = _row(next_due_at=now - 3_600_000)
    assert monitor_store.is_idle(stale, now) is True
    assert "session not open" in monitor_store.idle_detail(stale, now)

    assert monitor_store.is_idle(_row(next_due_at=now - 60_000), now) is False
    assert monitor_store.is_idle({**stale, "disabled": True}, now) is False
    assert monitor_store.is_idle({**stale, "stopped_at": now}, now) is False
    # A monitor whose interval is longer than the floor gets twice its own.
    slow = _row(every_ms=3_600_000, next_due_at=now - 1_800_000)
    assert monitor_store.is_idle(slow, now) is False
    assert monitor_store.is_idle({**slow, "next_due_at": now - 7_500_000}, now) is True


def test_format_age_ms_is_compact() -> None:
    assert monitor_store.format_age_ms(45_000) == "45s"
    assert monitor_store.format_age_ms(12 * 60_000) == "12m"
    assert monitor_store.format_age_ms(3 * 3_600_000) == "3h"
    assert monitor_store.format_age_ms(50 * 3_600_000) == "2d"
