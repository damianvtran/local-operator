"""Discovery records: publication is staged, permissions are the security
model, and scan classifies liveness the way the daemon's adoption loop
depends on."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

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
    """A proven-dead record leaves discovery, and does NOT leave the machine.

    ``lop sessions`` must stop showing the session — but the record is also the
    only artifact that says which runtime died, so it is moved to the sidecar
    instead of unlinked. Both halves are asserted here because either alone is
    a bug: a discovery loop that keeps listing a dead pid, or a reap that
    destroys the answer to "why did this die".
    """
    dead = make_record(pid=2**22 - 3)  # a pid that does not exist
    path = registry.publish(dead, root=tmp_path)
    results = registry.scan(root=tmp_path)
    assert [(r.pid, s) for r, s in results] == [(dead.pid, "stale")]
    assert not path.exists()  # gone from discovery
    sidecar = tmp_path / "run" / "mobile" / registry.REAPED_DIRNAME / f"{dead.pid}.json"
    assert sidecar.exists(), "the reap must move the record, not delete it"
    # And it is evidence only: a second sweep must not resurrect it as a session.
    assert registry.scan(root=tmp_path) == []


def test_scan_reader_mode_moves_nothing(tmp_path: Path) -> None:
    """``reap=False``: the same verdicts, with the run directory untouched.

    The desktop feed's status probe calls this once a SECOND, and it is a reader:
    a sweep on that poller would move another process's evidence aside behind its
    back (the reason :data:`registry.REAPED_DIRNAME` exists at all), and would
    delete a record it could not parse. The verdicts do not depend on the sweep —
    reaping is a side effect of the classification, never an input to it — so a
    reader that does not reap sees exactly what a reaper saw, and this asserts
    both halves: the same answer, and nothing removed.
    """
    dead = make_record(pid=2**22 - 3)  # a pid that does not exist
    path = registry.publish(dead, root=tmp_path)
    directory = registry.run_dir(tmp_path)
    torn = directory / "999998.json"
    torn.write_text("{not json", encoding="utf-8")

    results = registry.scan(root=tmp_path, reap=False)
    assert [(r.pid, s) for r, s in results] == [(dead.pid, "stale")]
    assert path.exists(), "reader mode moved a proven-dead record aside"
    assert torn.exists(), "reader mode deleted a file it could not parse"
    assert not (directory / registry.REAPED_DIRNAME).exists()


def test_scan_passes_the_zombie_policy_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``check_zombie`` reaches :func:`classify`, and ``False`` spends no fork.

    A structural spy rather than a timing bound, for the reason the timing section
    gives: the property is WHERE the code ran (a ``ps`` fork on macOS, measured at
    3.9 ms against ~1 µs for signal-0), not how long it took. The desktop feed
    reads with ``False`` on the record doorbell, where a record that just moved
    was demonstrably written by a live owner, and with the derived policy on its
    1 s probe, where its verdicts must match ``decorate_rows``'.
    """
    import json

    probes: list[int] = []
    monkeypatch.setattr(registry, "is_zombie", lambda pid: probes.append(pid) or False)
    record = make_record()  # this process: alive, so the zombie branch is reachable
    record.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 1
    directory = registry.run_dir(tmp_path)
    (directory / f"{record.pid}.json").write_text(json.dumps(record.to_json()), encoding="utf-8")

    assert [state for _record, state in registry.scan(root=tmp_path, check_zombie=False)] == [
        "wedged"
    ]
    assert probes == [], "check_zombie=False still probed for a zombie"
    # The derived policy (the default) is the one that spends the probe, because
    # this record's heartbeat has gone quiet — which is what makes the assertion
    # above a decision rather than a spy that never fires.
    assert [state for _record, state in registry.scan(root=tmp_path)] == ["wedged"]
    assert probes == [record.pid]


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


def test_the_subagent_counts_round_trip_and_absent_means_none_not_zero() -> None:
    """``None`` and ``0`` are DIFFERENT FACTS and the record must keep them apart.

    A runtime that predates these fields has not told us it has no subagents.
    Defaulting the absent case to 0 would let ``/info`` sum a fleet total whose
    missing terms are invisible — the caveat line ("the total is a lower bound")
    exists precisely because this distinction survives the wire.
    """
    record = make_record()
    record.subagents_running = 3
    record.subagents_queued = 1
    payload = record.to_json()
    assert SessionRecord.from_json(payload).subagents_running == 3
    assert SessionRecord.from_json(payload).subagents_queued == 1

    payload.pop("subagents_running")
    payload.pop("subagents_queued")
    restored = SessionRecord.from_json(payload)
    assert restored.subagents_running is None, "absent must not become 0"
    assert restored.subagents_queued is None


def test_the_heartbeat_republishes_the_subagent_counts(tmp_path: Path) -> None:
    """They ride every rewrite, like the build stamp, for the same reason.

    A field published once at startup and dropped by the first rewrite would be
    wrong 15 seconds later — and these change far more often than the stamp.
    """
    record = make_record()
    record.subagents_running = 2
    record.subagents_queued = 1
    publisher = registry.RecordPublisher(record, root=tmp_path)
    try:
        publisher.heartbeat(conversation_name="renamed")
        found = [rec for rec, _ in registry.scan(root=tmp_path) if rec.pid == record.pid]
        assert found and found[0].subagents_running == 2
        assert found[0].subagents_queued == 1
        assert found[0].conversation_name == "renamed", "the rewrite really happened"
    finally:
        registry.unpublish(record.pid, root=tmp_path)


def test_the_publisher_rewrites_the_file_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publisher's directory is decided when it publishes, not per call.

    ``root=None`` means "whatever ``config_dir()`` says", and ``config_dir()``
    reads the environment on every call on purpose. Resolving it again inside
    ``heartbeat`` therefore let a runtime that outlived its own config dir
    rewrite its record into whatever directory was current at that moment — a
    different session's file, because records are keyed by pid alone, and an
    xdist worker keeps one pid for every test it runs. This is the shape a
    shard runner produced: a runtime from the previous test, whose heartbeat
    had not been cancelled yet, landed its own ``started=False`` copy on the
    next test's record.
    """
    started_in = tmp_path / "started-in"
    moved_to = tmp_path / "moved-to"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(started_in))
    record = make_record()
    publisher = registry.RecordPublisher(record)
    own = registry.run_dir(started_in) / f"{record.pid}.json"
    try:
        assert publisher.path == own, "the publisher's own record is the one it names"
        # The config dir moves under a live runtime. Every test boundary in this
        # suite does exactly this to HOME, which is what makes it the shape that
        # bit, rather than a hypothetical.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(moved_to))
        publisher.heartbeat(conversation_name="renamed")
        assert own.exists(), "the record must stay where this publisher put it"
        stray = registry.run_dir(moved_to) / f"{record.pid}.json"
        assert not stray.exists(), (
            "a heartbeat must not write into a directory this publisher never "
            "published into — that filename belongs to another session"
        )
    finally:
        publisher.close()


def test_the_publisher_removes_only_the_file_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit path, where the damage inverts from clobbering to deleting.

    Needs its own test because the two directions fail differently: a stray
    ``heartbeat`` overwrites a stranger's record, a stray ``close`` unlinks it,
    and the suite's own runs showed the second — a leaked runtime's shutdown
    resolving the NEXT test's directory and taking that test's record with it.
    """
    started_in = tmp_path / "started-in"
    moved_to = tmp_path / "moved-to"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(started_in))
    publisher = registry.RecordPublisher(make_record())
    own = publisher.path
    assert own.exists()
    # Another session's record in the directory the config dir moves to, keyed
    # by the SAME pid — the pid keying is what makes this a wrong-FILE hazard
    # rather than a missing-file one.
    other = make_record()
    registry.publish(other, root=moved_to)
    other_path = registry.run_dir(moved_to) / f"{other.pid}.json"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(moved_to))
    try:
        publisher.close()
        assert not own.exists(), "a publisher removes the record it created"
        assert other_path.exists(), "and must leave a different session's record alone"
    finally:
        registry.unpublish(other.pid, root=moved_to)


def test_the_protocol_version_did_not_move_for_the_subagent_counts() -> None:
    """Pinned WITH ITS REASON, because the tempting change is to bump it.

    ``subagents_running``/``subagents_queued`` are purely additive: an older
    reader drops them in ``from_json`` and behaves exactly as before, and a
    newer reader sees ``None`` for a record an older runtime wrote. Meanwhile
    peers read ``record.protocol`` as a pre-dial promise about which FRAMES a
    runtime speaks (``attach_client`` refuses below 2, the TUI takeover path
    and ``session_factory`` below 4, ``remote``'s canonical attach below 5).
    Two JSON integers that ride in no frame change nothing those readers ask
    about, so bumping would spend the one number that carries that promise and
    leave nothing to mark a build whose frames really did change.
    """
    from local_operator.session.runtime.types import PROTOCOL_VERSION

    assert PROTOCOL_VERSION == 5


def test_a_record_with_unknown_future_keys_still_parses() -> None:
    """Forward-compat in the other direction: a NEWER runtime's record.

    The same tolerance the counts rely on, asserted from the far side so a
    future field cannot be added in a way that breaks this build mid-upgrade.
    """
    payload = make_record().to_json()
    payload["a_field_from_the_future"] = {"nested": True}
    assert SessionRecord.from_json(payload).pid == make_record().pid


def test_the_reaped_sidecar_is_bounded_by_count_and_age(tmp_path: Path) -> None:
    """Evidence with an expiry, so the sidecar is not a second unbounded store.

    Retention is by AGE first (a burst cannot evict today's death in favour of
    yesterday's) and by COUNT after, keeping the newest ``REAPED_MAX_FILES``.
    Without this the directory grows one file per session death forever, on a
    host whose whole point is that it stays up for months.
    """
    sidecar = registry.reaped_dir(tmp_path)
    old = time.time() - registry.REAPED_MAX_AGE_S - 3600
    stale = sidecar / "1.json"
    stale.write_text("{}")
    os.utime(stale, (old, old))
    # Distinct mtimes, oldest first, so "the newest survive" is decided by AGE
    # and not by whatever order the filesystem hands the glob back.
    base = time.time() - 3600
    for pid in range(2, registry.REAPED_MAX_FILES + 12):
        path = sidecar / f"{pid}.json"
        path.write_text("{}")
        os.utime(path, (base + pid, base + pid))

    registry._prune_reaped(sidecar)

    survivors = {int(p.stem) for p in sidecar.glob("*.json")}
    assert not stale.exists(), "an entry past the age bound is dropped"
    assert len(survivors) == registry.REAPED_MAX_FILES
    # The 10 OLDEST (the smallest pids here) are the ones spent, and the age
    # bound is applied before the count bound so a burst cannot evict today's
    # evidence in favour of yesterday's.
    assert 2 not in survivors
    assert registry.REAPED_MAX_FILES + 11 in survivors


def test_the_stop_marker_round_trips_0600_beside_the_transcript(tmp_path: Path) -> None:
    """The durable stop marker: readable, private, and where the reader looks.

    0600 like every artifact that can name a process, staged-write so a reader
    never sees half a marker, and read back through the same directory the
    classifier starts from — the two sides disagreeing is how a deliberate stop
    would silently go back to reading as an unexplained death.
    """
    conversation = tmp_path / "sessions" / "sid"
    conversation.mkdir(parents=True)
    payload = {"session_id": "sid", "pid": 4242, "rung": "sigkill", "deliberate": True}
    path = registry.write_stop_marker(conversation, payload)

    assert path == registry.stop_marker_path(conversation)
    assert path.name == registry.STOP_MARKER_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert registry.read_stop_marker(conversation) == payload
    # No marker, a torn marker, or a JSON scalar all mean "no usable evidence".
    assert registry.read_stop_marker(tmp_path / "sessions" / "elsewhere") is None
    path.write_text("{torn")
    assert registry.read_stop_marker(conversation) is None
    path.write_text("3")
    assert registry.read_stop_marker(conversation) is None


def test_a_missing_conversation_directory_is_not_conjured_by_the_marker(
    tmp_path: Path,
) -> None:
    """A failed marker write must not create a phantom session.

    Every session listing walks ``<root>/sessions``, so a writer that invented
    the conversation directory would make an empty session appear in the
    picker. The caller swallows the failure; this pins the absence.
    """
    conversation = tmp_path / "sessions" / "never-existed"
    with pytest.raises(OSError):
        registry.write_stop_marker(conversation, {"session_id": "never-existed"})
    assert not conversation.exists()


def test_classify_reports_a_quiet_owner_as_wedged_with_the_measured_age() -> None:
    """The one owner of the vocabulary, and the whole of what it claims.

    A stale beat with a live pid is ``wedged`` — degraded-responsiveness
    evidence, NOT a death certificate. The beat is authored by the runtime's own
    event loop, so a long turn or a starved scheduler produces this exact
    reading on a session that is demonstrably working (measured on this host at
    105.8 s and 205.8 s against the 45 s timeout), which is why the surfaces say
    "not answering" rather than naming a cause. The AGE comes back beside the
    word so those surfaces do not each re-derive the clamp.
    """
    record = make_record()
    record.heartbeat_at = time.time() - 300

    verdict = registry.classify(record, check_zombie=False)
    assert verdict.state == "wedged"
    # The pid exists, and that fact travels WITH the verdict: it is the
    # difference between "quiet" and "gone", and every caller that words this
    # state needs it.
    assert verdict.pid_alive is True
    assert 299 <= verdict.heartbeat_age_s <= 302

    # A fresh beat is live, and that word claims only that the owner reported.
    record.heartbeat_at = time.time()
    fresh = registry.classify(record, check_zombie=False)
    assert fresh.state == "live"
    assert fresh.heartbeat_age_s < 1.0

    # The pid decides ``stale``, never the stamp: a record whose process is gone
    # is reaped by the caller whatever its heartbeat says.
    dead = make_record(pid=2**22 - 3)  # a pid that does not exist
    dead.heartbeat_at = time.time()
    assert registry.classify(dead, check_zombie=False).state == "stale"

    # Clock skew can only make the register QUIETER: a stamp dated in the future
    # is not evidence against the process.
    skewed = make_record()
    skewed.heartbeat_at = time.time() + 60
    assert registry.classify(skewed, check_zombie=False).heartbeat_age_s == 0.0
