"""Discovery records: publication is staged, permissions are the security
model, and scan classifies liveness the way the daemon's adoption loop
depends on."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

from local_operator.mobile import registry
from local_operator.mobile.types import HEARTBEAT_TIMEOUT_S, SessionRecord


def make_record(pid: int | None = None) -> SessionRecord:
    return SessionRecord(
        pid=pid or os.getpid(),
        kind="tui",
        session_id="s1",
        conversation_name="demo",
        cwd="/tmp",
        model_label="anthropic/claude-opus-5",
        control_port=12345,
        control_key="k" * 64,
    )


def test_publish_creates_0700_dir_and_0600_record(tmp_path: Path) -> None:
    record = make_record()
    path = registry.publish(record, root=tmp_path)
    dir_mode = stat.S_IMODE(path.parent.stat().st_mode)
    file_mode = stat.S_IMODE(path.stat().st_mode)
    assert dir_mode == 0o700
    assert file_mode == 0o600


def test_scan_classifies_live_wedged_and_stale(tmp_path: Path) -> None:
    live = make_record()
    registry.publish(live, root=tmp_path)

    results = {r.pid: state for r, state in registry.scan(root=tmp_path)}
    assert results[live.pid] == "live"


def test_scan_marks_a_stale_heartbeat_wedged(tmp_path: Path) -> None:
    record = make_record()
    record.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 1
    directory = registry.run_dir(tmp_path)
    import json

    # Written directly rather than through publish(), which stamps a fresh
    # heartbeat by design — a wedged record is exactly one whose heartbeat
    # stopped arriving.
    (directory / f"{record.pid}.json").write_text(json.dumps(record.to_json()))

    results = {r.pid: state for r, state in registry.scan(root=tmp_path)}
    assert results[record.pid] == "wedged"


def test_scan_reaps_dead_pid_records(tmp_path: Path) -> None:
    dead = make_record(pid=2**22 - 3)  # a pid that does not exist
    path = registry.publish(dead, root=tmp_path)
    results = registry.scan(root=tmp_path)
    assert [(r.pid, s) for r, s in results] == [(dead.pid, "stale")]
    assert not path.exists()  # reaped


def test_scan_tolerates_torn_records(tmp_path: Path) -> None:
    directory = registry.run_dir(tmp_path)
    (directory / "999999.json").write_text("{not json")
    assert registry.scan(root=tmp_path) == []
    assert not (directory / "999999.json").exists()


def test_unpublish_is_best_effort(tmp_path: Path) -> None:
    record = make_record()
    registry.publish(record, root=tmp_path)
    registry.unpublish(record.pid, root=tmp_path)
    assert registry.scan(root=tmp_path) == []
    registry.unpublish(record.pid, root=tmp_path)  # twice: no raise


def test_record_round_trips_and_ignores_unknown_keys(tmp_path: Path) -> None:
    record = make_record()
    data = record.to_json()
    data["future_field"] = "from a newer binary"
    restored = SessionRecord.from_json(data)
    assert restored.control_key == record.control_key
    assert not hasattr(restored, "future_field")
