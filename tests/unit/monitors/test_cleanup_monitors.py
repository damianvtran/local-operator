"""The park, pristine and cleanup guards (§10.5, §11.5)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.session import cleanup


def _entry(tmp_path: Path, monitors: list[dict[str, Any]]) -> None:
    monitor_store.write_entry(tmp_path, "sess", cwd=str(tmp_path), monitors=monitors)


def test_an_armed_monitor_refuses_the_delete(tmp_path: Path) -> None:
    _entry(tmp_path, [{"id": "m1", "next_due_at": int(time.time() * 1000) + 60_000}])
    assert cleanup._has_armed_monitor(tmp_path, "sess") is True
    # ... and through the guard, which is what the delete consults.
    assert cleanup._guard(tmp_path / "sess", tmp_path, now=time.time()) == "has an armed monitor"


def test_a_durable_monitor_refuses_the_delete(tmp_path: Path) -> None:
    # No until_at at all == durable: the strongest reason to keep the session.
    _entry(tmp_path, [{"id": "m1"}])
    assert cleanup._has_armed_monitor(tmp_path, "sess") is True


def test_disabled_or_expired_monitors_do_not_refuse(tmp_path: Path) -> None:
    now = int(time.time() * 1000)
    _entry(tmp_path, [{"id": "m1", "disabled": True, "disabled_reason": "boom"}])
    assert cleanup._has_armed_monitor(tmp_path, "sess") is False
    _entry(tmp_path, [{"id": "m1", "until_at": now - 1}])
    assert cleanup._has_armed_monitor(tmp_path, "sess") is False
    # One armed row among disarmed ones still refuses.
    _entry(tmp_path, [{"id": "m1", "disabled": True}, {"id": "m2", "until_at": now + 60_000}])
    assert cleanup._has_armed_monitor(tmp_path, "sess") is True


def test_a_dormant_entry_does_not_refuse(tmp_path: Path) -> None:
    _entry(tmp_path, [{"id": "m1"}])
    entry = monitor_store.read_entry(tmp_path, "sess")
    assert entry is not None
    monitor_store.write_entry(
        tmp_path,
        "sess",
        cwd=str(tmp_path),
        monitors=[{"id": "m1"}],
        preserve=dict(entry, stopped_at=int(time.time() * 1000)),
    )
    assert cleanup._has_armed_monitor(tmp_path, "sess") is False


def test_an_unreadable_entry_is_fail_closed(tmp_path: Path) -> None:
    path = monitor_store.entry_path(tmp_path, "sess")
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert cleanup._has_armed_monitor(tmp_path, "sess") is True


def test_a_missing_entry_is_not_armed(tmp_path: Path) -> None:
    assert cleanup._has_armed_monitor(tmp_path, "sess") is False


def test_mark_monitors_dormant_stamps_and_counts(tmp_path: Path) -> None:
    from local_operator.session.runtime.control import _mark_monitors_dormant
    from local_operator.session.runtime.registry import SessionRecord

    _entry(tmp_path, [{"id": "m1"}, {"id": "m2"}])
    record = SessionRecord(
        pid=1,
        kind="tui",
        session_id="sess",
        conversation_name="c",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="",
    )
    assert _mark_monitors_dormant(record, tmp_path) == 2
    entry = monitor_store.read_entry(tmp_path, "sess")
    assert entry is not None and entry["stopped_at"] > 0
    # No entry: the park is a no-op, not an error.
    assert _mark_monitors_dormant(record, tmp_path / "empty") == 0


def test_the_stopped_line_names_both_families() -> None:
    from local_operator.session.runtime.control import _stopped_line
    from local_operator.session.runtime.registry import SessionRecord

    record = SessionRecord(
        pid=1,
        kind="tui",
        session_id="s",
        conversation_name="c",
        cwd="/w",
        model_label="",
        control_port=0,
        control_key="",
    )
    assert (
        _stopped_line(record, "socket", 0, monitors=2)
        == 'stopped "c" — 2 monitors dormant until you reopen it'
    )
    assert (
        _stopped_line(record, "socket", 1, monitors=2)
        == 'stopped "c" — 1 wake and 2 monitors dormant until you reopen it'
    )
    # The pre-monitor sentence is byte-identical when no monitor is parked.
    assert _stopped_line(record, "socket", 1) == 'stopped "c" — 1 wake dormant until you reopen it'


def test_forget_removes_the_entry_and_the_state(tmp_path: Path) -> None:
    from local_operator.session.cleanup import _forget_monitor_entry

    _entry(tmp_path, [{"id": "m1"}])
    monitor_state.write_counters(tmp_path, "sess", "m1", {"schema": 1})
    monitor_state.write_snapshot(tmp_path, "sess", "m1", "x", truncated=False)
    _forget_monitor_entry(tmp_path, "sess")
    assert monitor_store.read_entry(tmp_path, "sess") is None
    assert not monitor_state.state_dir(tmp_path, "sess").exists()


def test_the_refusal_sentence_names_the_entry_file(tmp_path: Path) -> None:
    message = cleanup._guard_refusal("has an armed monitor", "sess")
    assert "monitors/sess.json" in message
    assert "cancel the monitor" in message
