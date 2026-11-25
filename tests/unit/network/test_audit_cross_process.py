"""Audit appends stay uniquely numbered across writers in different PROCESSES.

WHY THIS FILE EXISTS. ``AuditLog`` used to seed ``seq`` once at open and append
with only a ``threading.RLock``, so two OS processes (the relay plus an approval
transition's short-lived writer, ``approvals._audit``) stamped the same number on
different rows. #4563's P3 reader (``record_mentioned_file_required``) refuses a
non-unique number, so that race turned an approval into refused reads. These tests
use REAL processes (``subprocess``) released together by a start file so the
appends actually interleave, then assert the numbering is unique and contiguous.
"""

from __future__ import annotations

import errno
import gzip
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

import local_operator
from local_operator.network import approvals, audit
from local_operator.wakes import lock as wake_lock
from tests.unit.network.test_mentioned_file_read_audit import identity

_REPO = str(Path(local_operator.__file__).resolve().parent.parent)

#: Each worker waits for ``go`` so both are past import (~1 s) before appending;
#: without it the first process usually finishes before the second starts.
_WORKER = r"""
import os, sys, time
from pathlib import Path
root, go, mode, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
from local_operator.network.audit import AuditEvent, AuditLog
from local_operator.network import approvals
log = AuditLog(root) if mode == "writer" else None
while not go.exists():
    time.sleep(0.001)
for i in range(n):
    if mode == "writer":
        log.record(AuditEvent(event="handshake_refused", detail={"reason": "x"}))
        log.flush()
        # Paced so this writer's appends span the approvals writer's (which opens
        # a fresh log per event and is much slower); unpaced it finishes first.
        if mode == "writer" and sys.argv[5:] == ["paced"]:
            time.sleep(0.01)
    else:
        # The real approvals path: a fresh writer per event, record, close.
        approvals._audit("onboard_requested", actor="self", subject="s", root=root)
if log is not None:
    log.close()
"""


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _race(root: Path, modes: list[str], n: int, *extra: str) -> list[dict[str, Any]]:
    go = root / "go"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(root), str(go), mode, str(n), *extra],
            env=_env(),
            stderr=subprocess.PIPE,
            text=True,
        )
        for mode in modes
    ]
    time.sleep(2.0)  # both processes past import, spinning on the start file
    go.write_text("")
    for proc in procs:
        _, err = proc.communicate(timeout=240)
        assert proc.returncode == 0, err
    path = root / "network" / "audit.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_unique_contiguous(rows: list[dict[str, Any]], expected: int) -> None:
    seqs = [row["seq"] for row in rows]
    dupes = sorted({s for s in seqs if seqs.count(s) > 1})
    assert not dupes, f"{len(dupes)} duplicated seq values, e.g. {dupes[:10]}"
    assert sorted(seqs) == list(range(1, expected + 1))
    assert seqs == sorted(seqs), "file order must follow numbering"


def test_two_process_writers_never_share_a_seq(tmp_path: Path) -> None:
    rows = _race(tmp_path, ["writer", "writer"], 300)
    _assert_unique_contiguous(rows, 600)


def test_approvals_writer_and_long_lived_writer_never_share_a_seq(tmp_path: Path) -> None:
    rows = _race(tmp_path, ["writer", "approvals"], 150, "paced")
    assert sum(row["event"] == "onboard_requested" for row in rows) == 150
    _assert_unique_contiguous(rows, 300)


# ---------------------------------------------------------------------------
# In-process: two AuditLog instances share only the file and its sibling lock,
# exactly the relationship between two processes' writers.
# ---------------------------------------------------------------------------


#: Rendered by the PRE-FIX writer (``git show a9ecde898:local_operator/network/audit.py``,
#: ``AuditLog._render`` of the event in the test below). The lock must not change a byte.
_GOLDEN_ROW = (
    '{"actor":"peer-1","actor_kind":"unknown","cause":"","event":"onboard_requested",'
    '"network_id":"net-1","outcome":"ok","schema":"lop.mesh.audit.v1","seq":1,'
    '"ts":1764000000.123,"ts_iso":"2025-11-24T16:00:00.123Z"}\n'
)


def _event(**kw):
    # A BATCHED event (not durable, not stream lifecycle), so rows really sit in
    # the buffer until the test flushes - the state renumbering exists for.
    return audit.AuditEvent(event="onboard_requested", **kw)


def _all_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "network" / "audit.jsonl"
    rows = []
    for gen in sorted(path.parent.glob("audit.jsonl.*.gz")):
        with gzip.open(gen, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    if path.exists():
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def _quiet(log: audit.AuditLog) -> audit.AuditLog:
    log.TICK_S = 3600.0  # nothing flushes until the test says so
    return log


def test_row_bytes_match_the_pre_fix_renderer_including_a_renumbered_row(tmp_path: Path) -> None:
    ev = _event(ts=1764000000.123, actor="peer-1", network_id="net-1")
    log = _quiet(audit.AuditLog(tmp_path))
    log.record(ev)
    log.flush()
    assert log._path.read_text() == _GOLDEN_ROW
    # A buffered row renumbered past another writer differs ONLY in its seq digits.
    other = _quiet(audit.AuditLog(tmp_path))
    mine = _quiet(audit.AuditLog(tmp_path))
    mine.record(ev)  # provisional seq 2
    other.record(_event())
    other.flush()  # takes seq 2
    mine.flush()  # renumbered to 3
    assert log._path.read_text().splitlines()[2] + "\n" == _GOLDEN_ROW.replace('"seq":1', '"seq":3')


def test_buffered_rows_are_renumbered_after_another_writers_flush(tmp_path: Path) -> None:
    a, b = _quiet(audit.AuditLog(tmp_path)), _quiet(audit.AuditLog(tmp_path))
    for _ in range(3):
        a.record(_event(actor="a"))
    assert a.recorded_through == 3
    for _ in range(2):
        b.record(_event(actor="b"))
    b.flush()
    a.flush()
    rows = _all_rows(tmp_path)
    assert [(r["seq"], r["actor"]) for r in rows] == [
        (1, "b"),
        (2, "b"),
        (3, "a"),
        (4, "a"),
        (5, "a"),
    ]
    assert a.published_through == a.recorded_through == 5


def test_rotation_by_one_writer_keeps_the_other_numbering_after_it(tmp_path: Path) -> None:
    a = _quiet(audit.AuditLog(tmp_path))
    b = _quiet(audit.AuditLog(tmp_path))
    a.record(_event())
    a.flush()
    a._max_bytes = a._path.stat().st_size + 1  # a's next flush rotates
    a.record(_event(actor="a"))
    a.flush()
    assert list(a._path.parent.glob("audit.jsonl.*.gz")), "rotation did not happen"
    b.record(_event(actor="b"))  # b opened before the rotation
    b.flush()
    seqs = [r["seq"] for r in _all_rows(tmp_path)]
    # 1 (first), 2 (a's batch) + 3 (audit_rotated), then b's row follows both.
    assert sorted(seqs) == [1, 2, 3, 4] and len(seqs) == len(set(seqs)), seqs


def test_mentioned_file_admission_after_an_approvals_write_is_not_refused(tmp_path: Path) -> None:
    """Q1-1 of #4563: a long-lived writer's P3 admission right after an approval."""
    relay = audit.AuditLog(tmp_path)
    relay.record(audit.AuditEvent(event="panic_raised"))  # relay has a counter at 1
    approvals._audit("onboard_requested", actor="self", subject="s", root=tmp_path)
    receipt = relay.record_mentioned_file_required(identity())
    assert receipt is not None, "P3 admission refused after an approvals-path write"
    seqs = [r["seq"] for r in _all_rows(tmp_path)]
    assert sorted(seqs) == [1, 2, 3] and receipt.sequence == 3


def _hold_lock(root: Path) -> int:
    path = root / "network" / "audit.jsonl.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    assert wake_lock._try_lock(fd)
    return fd


def test_lock_timeout_keeps_record_buffered_and_refuses_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "AUDIT_LOCK_WAIT_S", 0.05)
    held = _hold_lock(tmp_path)  # a separate open file description: real contention
    log = _quiet(audit.AuditLog(tmp_path))
    try:
        log.record(_event())
        log.flush()
        assert not log._path.exists(), "a row was written without the lock"
        assert log.lock_timeouts >= 1 and not log.degraded  # nothing lost: still buffered
        assert log.publication_of(1) == "buffered"
        assert log.record_required(audit.AuditEvent(event="panic_raised")) is False
        assert log.record_mentioned_file_required(identity()) is None
        assert not log._path.exists()
    finally:
        wake_lock._unlock(held)
        os.close(held)
    log.flush()  # the retry publishes what the timeout kept
    assert [r["seq"] for r in _all_rows(tmp_path)] == [1]
    assert log.record_required(audit.AuditEvent(event="panic_raised")) is True


def test_no_lock_primitive_refuses_loudly_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "fcntl", None)  # `import fcntl` -> ImportError
    log = audit.AuditLog(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised"))
    assert not log._path.exists()
    assert log.degraded and "no file-lock primitive" in log.degraded_reason
    assert "nothing was written" in capsys.readouterr().err
    assert log.record_required(audit.AuditEvent(event="panic_raised")) is False


def _windows(
    monkeypatch: pytest.MonkeyPatch, contended: bool, log_path: list[Path]
) -> list[tuple[str, int]]:
    """Drive ``wakes.lock``'s ``os.name == "nt"`` branch with a fake ``msvcrt``.

    Each call records the log's size at that moment, so a test can prove the lock
    was taken before the append and released after it.
    """
    calls: list[tuple[str, int]] = []

    def locking(fd: int, mode: int, nbytes: int) -> None:
        size = log_path[0].stat().st_size if log_path and log_path[0].exists() else 0
        calls.append(("lock" if mode == fake.LK_NBLCK else "unlock", size))
        if contended and mode == fake.LK_NBLCK:
            raise OSError(errno.EACCES, "locked")

    fake = types.SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    real_os = os

    class _NtOs:
        name = "nt"

        def __getattr__(self, attr: str):
            return getattr(real_os, attr)

    monkeypatch.setattr(wake_lock, "os", _NtOs())
    return calls


def test_windows_branch_locks_around_the_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = audit.AuditLog(tmp_path)
    calls = _windows(monkeypatch, contended=False, log_path=[log._path])
    assert log.record_required(audit.AuditEvent(event="panic_raised")) is True
    assert calls[0] == ("lock", 0), calls  # taken before any byte was appended
    assert calls[-1][0] == "unlock" and calls[-1][1] > 0, calls  # released after it
    assert [c[0] for c in calls].count("lock") == 1, "re-entry must not re-lock"


def test_windows_branch_contention_retries_then_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "AUDIT_LOCK_WAIT_S", 0.05)
    log = audit.AuditLog(tmp_path)
    calls = _windows(monkeypatch, contended=True, log_path=[log._path])
    assert log.record_required(audit.AuditEvent(event="panic_raised")) is False
    assert len([c for c in calls if c[0] == "lock"]) > 1, "no retry before refusing"
    assert not log._path.exists()


def test_one_shot_writers_lose_no_row_to_a_lock_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED-FIRST (round-1 remediation, L3): the REAL per-event writers time out on a
    held lock, the holder releases, and the next writer's flush must publish both
    rows exactly once with unique numbers - a timeout may delay a row, never drop it."""
    monkeypatch.setattr(audit, "AUDIT_LOCK_WAIT_S", 0.05)
    held = _hold_lock(tmp_path)  # a separate open file description: real contention
    try:
        approvals._audit("onboard_approved", actor="self", subject="sess-a", root=tmp_path)
        audit.write_event(tmp_path, event="member_removed", subject="dev-b")
        assert not (tmp_path / "network" / "audit.jsonl").exists()
    finally:
        wake_lock._unlock(held)
        os.close(held)
    later = audit.AuditLog(tmp_path)
    later.record(_event(subject="later"))
    later.close()
    rows = _all_rows(tmp_path)
    events = [(r["event"], r.get("subject")) for r in rows]
    assert events.count(("onboard_approved", "sess-a")) == 1, events
    assert events.count(("member_removed", "dev-b")) == 1, events
    assert sorted(r["seq"] for r in rows) == list(range(1, len(rows) + 1)), rows


def test_batched_record_makes_one_non_blocking_attempt_and_rearms_the_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L2 (S1-2/R1-2/Q1-1): with the lock wedged, a batched ``record`` past the tick
    must never wait the deadline. Structural, not wall-clock: a deadline wait would
    count a ``lock_timeouts``; the single attempt counts ``lock_busy`` and re-arms."""
    monkeypatch.setattr(audit, "AUDIT_LOCK_WAIT_S", 30.0)  # any wait would be obvious
    held = _hold_lock(tmp_path)
    log = audit.AuditLog(tmp_path)
    log.TICK_S = 0.0  # every record is past the tick
    try:
        started = time.monotonic()
        for index in range(20):
            log.record(_event(subject=f"s{index}"))
        assert time.monotonic() - started < 10.0  # one 30 s wait would blow this
        assert log.lock_timeouts == 0 and log.lock_busy == 20 and not log.degraded
        assert not log._path.exists() and log.publication_of(20) == "buffered"
        log.TICK_S = 3600.0
        before = log._last_flush
        log.record(_event(subject="tick-not-due"))
        assert log.lock_busy == 20 and log._last_flush == before  # no attempt at all
    finally:
        wake_lock._unlock(held)
        os.close(held)
    log.close()
    rows = _all_rows(tmp_path)
    assert [r["seq"] for r in rows] == list(range(1, 22))
    assert [r["subject"] for r in rows][:2] == ["s0", "s1"]


def _parked(root: Path) -> list[str]:
    directory = root / "network" / "audit.jsonl.pending"
    return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []


def test_close_parks_rows_0600_and_a_drain_keeps_their_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "AUDIT_LOCK_WAIT_S", 0.05)
    held = _hold_lock(tmp_path)
    try:
        log = _quiet(audit.AuditLog(tmp_path))
        log.record(_event(subject="parked", ts=1700000000.0))
        expected = log._buffer[0]
        log.close()
        assert log.rows_spilled == 1 and log.rows_dropped == 0 and not log.degraded
        (spill,) = _parked(tmp_path)
        assert spill.endswith(".jsonl")
        path = tmp_path / "network" / "audit.jsonl.pending" / spill
        assert oct(path.stat().st_mode & 0o777) == "0o600"
        assert path.read_text() == expected
    finally:
        wake_lock._unlock(held)
        os.close(held)
    relay = audit.AuditLog(tmp_path)
    relay.flush()  # an empty flush does not take the lock; a record does
    relay.record(audit.AuditEvent(event="panic_raised"))  # durable: locks, drains
    lines = (tmp_path / "network" / "audit.jsonl").read_text().splitlines(keepends=True)
    assert lines[0] == expected  # seq 1 on both sides: byte-identical
    assert [json.loads(line)["seq"] for line in lines] == [1, 2]
    assert relay.rows_drained == 1 and _parked(tmp_path) == []


def test_a_crash_after_the_claim_neither_loses_nor_duplicates(tmp_path: Path) -> None:
    """The claim names the numbers it will take; recovery reads the tail against it."""
    pending = tmp_path / "network" / "audit.jsonl.pending"
    pending.mkdir(parents=True)
    seed = audit.AuditLog(tmp_path)
    seed.record(audit.AuditEvent(event="panic_raised"))  # seq 1 on disk
    row = lambda seq, subject: json.dumps(  # noqa: E731
        {"event": "onboard_requested", "seq": seq, "subject": subject}, sort_keys=True
    )
    # Claimed 2-3 and fully appended before the crash: must not be appended again.
    (pending / "00000000000000000001-1-aa.2-3.draining").write_text(
        row(2, "a") + "\n" + row(3, "b") + "\n"
    )
    with (tmp_path / "network" / "audit.jsonl").open("a") as handle:
        handle.write(row(2, "a") + "\n" + row(3, "b") + "\n")
    # Claimed 4-5 but the crash came before the append: must be appended once.
    (pending / "00000000000000000002-1-bb.4-5.draining").write_text(
        row(4, "c") + "\n" + row(5, "d") + "\n"
    )
    audit.AuditLog(tmp_path).record(audit.AuditEvent(event="panic_raised"))
    rows = _all_rows(tmp_path)
    assert [r["seq"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert [r.get("subject") for r in rows][1:5] == ["a", "b", "c", "d"]
    assert _parked(tmp_path) == []


def test_a_failed_spill_is_the_one_loss_and_it_is_degraded_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(audit, "AUDIT_LOCK_WAIT_S", 0.05)
    (tmp_path / "network").mkdir(parents=True)
    (tmp_path / "network" / "audit.jsonl.pending").write_text("not a directory")
    held = _hold_lock(tmp_path)
    try:
        log = _quiet(audit.AuditLog(tmp_path))
        log.record(_event())
        log.close()
    finally:
        wake_lock._unlock(held)
        os.close(held)
    assert log.rows_dropped == 1 and log.degraded and not log.admits_required()
    assert "were NOT written" in capsys.readouterr().err
    assert "onboard_requested" in log.degraded_reason or "1 audit rows lost" in log.degraded_reason


def test_an_unopenable_lock_degrades_the_log(tmp_path: Path) -> None:
    """L4 (S1-3): a permission/placement fault recurs on every call - degraded."""
    (tmp_path / "network").mkdir(parents=True)
    target = tmp_path / "elsewhere"
    target.write_text("")
    (tmp_path / "network" / "audit.jsonl.lock").symlink_to(target)  # ELOOP
    log = audit.AuditLog(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised"))
    assert log.degraded and "could not be opened" in log.degraded_reason
    assert not log.admits_required() and not log._path.exists()
    assert target.read_text() == ""


def test_an_existing_lock_leaf_is_checked_and_reset_to_0600(tmp_path: Path) -> None:
    """L5 (S1-4): mirrors ``credentials.placement._open_lock``."""
    lock = tmp_path / "network" / "audit.jsonl.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    os.chmod(lock, 0o644)
    audit.AuditLog(tmp_path).record(audit.AuditEvent(event="panic_raised"))
    assert oct(lock.stat().st_mode & 0o777) == "0o600"
    other = tmp_path / "b"
    (other / "network").mkdir(parents=True)
    os.mkfifo(other / "network" / "audit.jsonl.lock")  # opens O_RDWR without blocking
    log = audit.AuditLog(other)
    log.record(audit.AuditEvent(event="panic_raised"))
    assert log.degraded and "not a regular file" in log.degraded_reason
    assert not log._path.exists()


# -- round 2: a failed drain refuses the lock (R2-1), hostile entries (R2-3/Q2-2) ----


def _parsed(root: Path) -> tuple[list[dict[str, Any]], int]:
    """Rows of the live file, plus how many lines did NOT parse (a torn fragment)."""
    rows, torn = [], 0
    for line in (root / "network" / "audit.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            torn += 1
    return rows, torn


def _park_one(root: Path, subject: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(audit, "AUDIT_LOCK_WAIT_S", 0.05)
    held = _hold_lock(root)
    try:
        audit.write_event(root, event="member_removed", subject=subject)
    finally:
        wake_lock._unlock(held)
        os.close(held)
    (spill,) = _parked(root)
    return spill


def test_an_eio_on_the_claim_fsync_refuses_the_lock_and_undoes_the_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-1 (a): nothing of the claim reached the log, so the rename is undone.

    Before the fix the acquisition was GRANTED after the failed drain: the relay
    numbered its own row 1 over the claim ``.1-1.draining``, and the next writer read
    that claim as already appended and deleted it - the parked row was gone.
    """
    spill = _park_one(tmp_path, "PARKED", monkeypatch)
    real = audit._fsync_directory
    calls = {"n": 0}

    def flaky(path: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EIO, "EIO (injected)")
        real(path)

    monkeypatch.setattr(audit, "_fsync_directory", flaky)
    relay = _quiet(audit.AuditLog(tmp_path))
    relay.record(audit.AuditEvent(event="panic_raised", subject="relay"))
    monkeypatch.setattr(audit, "_fsync_directory", real)
    assert calls["n"] == 2  # the failed claim fsync, then the undo's own
    # Refused: the relay's durable row stays buffered, nothing was numbered.
    assert not (tmp_path / "network" / "audit.jsonl").exists()
    assert len(relay._buffer) == 1 and relay.drain_failures == 1
    assert _parked(tmp_path) == [spill], "the claim rename was not undone"
    audit.write_event(tmp_path, event="member_removed", subject="later")
    relay.close()
    rows, torn = _parsed(tmp_path)
    assert torn == 0
    assert [r.get("subject") for r in rows].count("PARKED") == 1
    assert sorted(str(r.get("subject")) for r in rows) == ["PARKED", "later", "relay"]
    assert [r["seq"] for r in rows] == [1, 2, 3]
    assert _parked(tmp_path) == []


def test_an_enospc_mid_append_refuses_and_recovery_appends_only_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-1 (b) + R2-4: a torn drain keeps its claim; later writers finish it once.

    Two torn drains in a row: the first leaves P1 plus half of P2 in the log, the
    second (recovery) appends only P2 before failing again. Before the fix the first
    holder numbered over the claim; and a recovery re-claim renamed the WHOLE file to
    a narrower range, so the third writer re-appended rows already on disk. The
    newline written before a recovery append keeps P2 off the torn fragment's line.
    """
    pending = tmp_path / "network" / "audit.jsonl.pending"
    pending.mkdir(parents=True)

    def row(subject: str) -> str:
        record = {"schema": 1, "event": "member_removed", "seq": 0, "subject": subject}
        return json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"

    (pending / "00000000000000000001-1-aa.jsonl").write_text(row("P1") + row("P2") + row("P3"))
    cuts: list[Any] = []
    real_open = Path.open

    def torn_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, mode, *args, **kwargs)
        if self.name == "audit.jsonl" and "a" in mode and cuts:
            cut = cuts.pop(0)
            write = handle.write

            def torn(payload: str) -> int:
                write(payload[: cut(payload)])
                handle.flush()
                raise OSError(errno.ENOSPC, "ENOSPC (injected)")

            handle.write = torn  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", torn_open)
    # First drain: P1 whole and half of P2.
    cuts.append(lambda payload: len(payload.splitlines(keepends=True)[0]) + 10)
    first = _quiet(audit.AuditLog(tmp_path))
    first.record(audit.AuditEvent(event="panic_raised", subject="first"))
    assert len(first._buffer) == 1 and first.drain_failures == 1, "the lock was granted"
    rows, torn = _parsed(tmp_path)
    assert [r.get("subject") for r in rows] == ["P1"] and torn == 1
    # Second drain (recovery): only the leading newline and P2 land.
    cuts.append(lambda payload: payload.index("\n", 1) + 1)
    second = _quiet(audit.AuditLog(tmp_path))
    second.record(audit.AuditEvent(event="panic_raised", subject="second"))
    assert len(second._buffer) == 1 and second.drain_failures == 1
    monkeypatch.setattr(Path, "open", real_open)
    audit.write_event(tmp_path, event="member_removed", subject="later")
    first.close()
    second.close()
    rows, torn = _parsed(tmp_path)
    subjects = [r.get("subject") for r in rows]
    for subject in ("P1", "P2", "P3", "first", "second", "later"):
        assert subjects.count(subject) == 1, (subject, subjects)
    assert subjects[:3] == ["P1", "P2", "P3"]
    assert [r["seq"] for r in rows] == list(range(1, 7)) and torn == 1
    assert _parked(tmp_path) == []


@pytest.mark.parametrize("kind", ["nondict", "badjson", "symlink", "fifo", "dir", "huge"])
def test_a_hostile_parked_entry_is_set_aside_and_the_lock_still_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    """R2-3/Q2-1/Q2-2: never followed, never blocks, never wedges or degrades the log."""
    pending = tmp_path / "network" / "audit.jsonl.pending"
    pending.mkdir(parents=True)
    entry = pending / "00000000000000000001-1-aa.jsonl"
    forged = tmp_path / "secret.txt"
    forged.write_text(json.dumps({"event": "forged", "seq": 1}) + "\n")
    if kind == "nondict":
        entry.write_text("[1, 2]\n")
    elif kind == "badjson":
        entry.write_text("not json\n")
    elif kind == "symlink":
        entry.symlink_to(forged)
    elif kind == "fifo":
        os.mkfifo(entry)
    elif kind == "dir":
        entry.mkdir()
    else:
        monkeypatch.setattr(audit, "_PENDING_MAX_BYTES", 64)
        entry.write_text(json.dumps({"event": "x", "seq": 0, "pad": "y" * 100}) + "\n")
    log = audit.AuditLog(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised", subject="own"))
    assert log._file_lock_depth == 0
    assert not log.degraded and log.admits_required() and log.pending_rejected == 1
    assert [(r["seq"], r["event"]) for r in _all_rows(tmp_path)] == [(1, "panic_raised")]
    assert _parked(tmp_path) == [entry.name + ".rejected"]
    assert "set aside" in capsys.readouterr().err
    assert forged.read_text().startswith('{"event": "forged"')
    audit.write_event(tmp_path, event="member_removed", subject="next")
    assert [r["seq"] for r in _all_rows(tmp_path)] == [1, 2]


def test_a_symlinked_pending_directory_is_not_followed(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "00000000000000000001-1-aa.jsonl").write_text(
        json.dumps({"event": "forged", "seq": 0}) + "\n"
    )
    (tmp_path / "network").mkdir()
    (tmp_path / "network" / "audit.jsonl.pending").symlink_to(elsewhere)
    log = audit.AuditLog(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised"))
    assert [r["event"] for r in _all_rows(tmp_path)] == ["panic_raised"]
    assert sorted(os.listdir(elsewhere)) == ["00000000000000000001-1-aa.jsonl"]


def _fail_open_once(monkeypatch: pytest.MonkeyPatch, name: str, err: int) -> list[str]:
    """Arm ONE transient ``err`` on the next ``os.open`` of the pending entry ``name``."""
    real_open = os.open
    armed = [name]

    def flaky(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if armed and Path(path).name == armed[0]:
            armed.pop()
            raise OSError(err, f"{os.strerror(err)} (injected)")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(audit.os, "open", flaky)
    return armed


@pytest.mark.parametrize("err", [errno.EMFILE, errno.EIO])
def test_a_transient_error_reading_a_spill_refuses_and_keeps_it_for_the_next_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int
) -> None:
    """R3-1: an EMFILE/EIO is not a hostile entry; set aside, the parked row was lost."""
    spill = _park_one(tmp_path, "PARKED", monkeypatch)
    armed = _fail_open_once(monkeypatch, spill, err)
    first = _quiet(audit.AuditLog(tmp_path))
    first.record(audit.AuditEvent(event="panic_raised", subject="own"))
    assert not armed, "the injected fault never fired"
    assert len(first._buffer) == 1 and first.drain_failures == 1, "the lock was granted"
    assert first.pending_rejected == 0 and first._file_lock_depth == 0
    assert not (tmp_path / "network" / "audit.jsonl").exists() or _all_rows(tmp_path) == []
    assert _parked(tmp_path) == [spill]  # left in place, not renamed
    audit.write_event(tmp_path, event="member_removed", subject="later")
    first.close()
    rows = _all_rows(tmp_path)
    subjects = [r.get("subject") for r in rows]
    for subject in ("PARKED", "own", "later"):
        assert subjects.count(subject) == 1, (subject, subjects)
    assert subjects[0] == "PARKED"
    assert [r["seq"] for r in rows] == [1, 2, 3]
    assert _parked(tmp_path) == []


@pytest.mark.parametrize("err", [errno.EMFILE, errno.EIO])
def test_a_transient_error_reading_a_torn_claim_refuses_and_numbers_nothing_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int
) -> None:
    """R3-1: a granted lock here numbered ``own`` as seq 2, inside claim 1-3 (R2-1 class)."""
    pending = tmp_path / "network" / "audit.jsonl.pending"
    pending.mkdir(parents=True)
    row = lambda seq, subject: json.dumps(  # noqa: E731
        {"event": "onboard_requested", "seq": seq, "subject": subject}, sort_keys=True
    )
    claim = "00000000000000000001-1-aa.1-3.draining"
    (pending / claim).write_text("".join(row(n, f"P{n}") + "\n" for n in (1, 2, 3)))
    live = tmp_path / "network" / "audit.jsonl"
    live.write_text(row(1, "P1") + "\n")  # torn: only P1 reached the log
    armed = _fail_open_once(monkeypatch, claim, err)
    first = _quiet(audit.AuditLog(tmp_path))
    first.record(audit.AuditEvent(event="panic_raised", subject="own"))
    assert not armed, "the injected fault never fired"
    assert len(first._buffer) == 1 and first.drain_failures == 1, "the lock was granted"
    assert first.pending_rejected == 0 and first._file_lock_depth == 0
    assert [r.get("subject") for r in _all_rows(tmp_path)] == ["P1"]  # nothing numbered
    assert _parked(tmp_path) == [claim]  # the claim keeps its name and its numbers
    audit.write_event(tmp_path, event="member_removed", subject="later")
    first.close()
    rows = _all_rows(tmp_path)
    subjects = [r.get("subject") for r in rows]
    for subject in ("P1", "P2", "P3", "own", "later"):
        assert subjects.count(subject) == 1, (subject, subjects)
    assert subjects[:3] == ["P1", "P2", "P3"]
    assert [r["seq"] for r in rows] == [1, 2, 3, 4, 5]
    assert _parked(tmp_path) == []
