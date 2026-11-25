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
        (1, "b"), (2, "b"), (3, "a"), (4, "a"), (5, "a")
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
