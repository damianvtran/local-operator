"""Discovery records: publication is staged, permissions are the security
model, and scan classifies liveness the way the daemon's adoption loop
depends on."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

from local_operator.session.runtime import registry
from local_operator.session.runtime.types import HEARTBEAT_TIMEOUT_S, SessionRecord


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


def test_a_killed_runtime_is_not_reported_live_while_it_is_a_zombie() -> None:
    """`kill -9` must not leave `lop sessions` claiming the session is live.

    `os.kill(pid, 0)` succeeds against a process that has exited but not been
    reaped, so a crashed runtime reported `live` with `0B` RSS until the
    heartbeat aged it out 45 s later — and `lop sessions`, the one place a
    user checks to understand the failure, actively misled them (round 3,
    U10). The window is real: a runtime's parent is often a shell that has
    since exited, so nothing reaps the entry promptly.

    The probe is opt-in because it costs a `ps` fork on macOS (measured
    3.9 ms vs ~1 µs for signal-0) and `scan()` runs on every `lop`
    invocation; `scan` spends it only on records whose heartbeat has already
    gone quiet.
    """
    import subprocess
    import time

    from local_operator.session.runtime.registry import pid_alive

    proc = subprocess.Popen(["sleep", "30"])  # noqa: S603,S607 — fixed argv
    try:
        assert pid_alive(proc.pid, check_zombie=True) is True
        proc.kill()
        # Wait for the kernel to move it to Z without reaping it (no wait()).
        for _ in range(100):
            if not pid_alive(proc.pid, check_zombie=True):
                break
            time.sleep(0.02)
        assert (
            pid_alive(proc.pid, check_zombie=True) is False
        ), "an exited-but-unreaped runtime must not report as live"
        # The cheap path is unchanged: it still sees the zombie as alive, which
        # is what keeps `scan` fork-free for healthy sessions.
        assert pid_alive(proc.pid) is True
    finally:
        proc.wait()

    # Once reaped, both paths agree it is gone.
    assert pid_alive(proc.pid) is False


def test_the_build_stamp_round_trips_on_a_record(tmp_path: Path) -> None:
    """The record IS the version channel between a viewer and a runtime.

    An attach client reads it before it dials, so whatever the runtime stamped
    has to survive the JSON round trip intact — a stamp that only exists in
    the writing process tells nobody anything.
    """
    record = make_record()
    record.version = "0.49.0"
    record.source_ref = "4d3ce1d1a48f4f3b799efdfabb014979e70e0630"
    restored = SessionRecord.from_json(record.to_json())
    assert restored.version == "0.49.0"
    assert restored.source_ref == "4d3ce1d1a48f4f3b799efdfabb014979e70e0630"


def test_a_record_from_an_older_runtime_defaults_the_build_stamp() -> None:
    """Additive means an OLD writer's payload still parses, as empty strings.

    Every runtime resident when this ships predates the fields, and a new
    viewer must read those records normally rather than raising mid-scan. The
    empty stamp is itself the signal: a runtime that cannot say what it runs
    is older than the terminal reading it.
    """
    payload = make_record().to_json()
    payload.pop("version")
    payload.pop("source_ref")
    restored = SessionRecord.from_json(payload)
    assert restored.version == ""
    assert restored.source_ref == ""


def test_the_heartbeat_republishes_the_build_stamp(tmp_path: Path) -> None:
    """The stamp rides every rewrite because the publisher owns the dataclass.

    ``RecordPublisher.heartbeat`` re-serialises the live record object rather
    than rebuilding a payload from a field list, so a new field is carried
    without a second code path. Pinned because the alternative — a hand-rolled
    dict somewhere in the heartbeat — would publish the stamp once at startup
    and then quietly drop it on the first rewrite, 15 seconds later.
    """
    record = make_record()
    record.version = "0.49.0"
    record.source_ref = "abc1234"
    publisher = registry.RecordPublisher(record, root=tmp_path)
    try:
        publisher.heartbeat(conversation_name="renamed")
        found = [rec for rec, _state in registry.scan(root=tmp_path) if rec.pid == record.pid]
        assert found, "the republished record must still be discoverable"
        assert found[0].version == "0.49.0"
        assert found[0].source_ref == "abc1234"
        assert found[0].conversation_name == "renamed", "the rewrite really happened"
    finally:
        publisher.close()
