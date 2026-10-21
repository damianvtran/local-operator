"""The runtime's durable account of its own departure (``runtime-exit.json``).

WHY THESE TESTS EXIST. On 2026-10-21 the operator asked "when did it stop" about a
session that had left through the runtime's OWN residency policy — the 3 s idle
exit — and no artifact anywhere in the store could answer. ``_clean_exit`` wrote
one INFO line, withdrew the boot record, and that was all: in the store the
session was simply ``stored``, exactly as a SIGKILLed runtime is. These tests pin
the record that closes that gap, its schema, and the two guards the exit path
depends on (best-effort writes, and no record invented for a runtime with no
conversation).

They are unit tests by contract: no real process, no wall clock beyond the
records they write. The end-to-end reproduction — a real runtime idling out and
the listing rendering its stop age — lives in ``scripts/repro_exit_record.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import journal, process, registry

# --- builders ---------------------------------------------------------------


class _Transcript:
    def __init__(self, directory: Path) -> None:
        self.directory = directory


class _Session:
    def __init__(self, session_id: str, directory: Path) -> None:
        self.session_id = session_id
        self._transcript = _Transcript(directory)


class _Record:
    def __init__(self, session_id: str, started_at: float) -> None:
        self.session_id = session_id
        self.started_at = started_at


class _Runtime:
    """The two reads ``_clean_exit`` makes: the run key and the boot build."""

    def __init__(self, session_id: str, started_at: float) -> None:
        self._record = _Record(session_id, started_at)
        self._boot_build = None


class _Handle:
    """A reduced session handle: dispose/aclose are no-ops, as in the real host."""

    def __init__(self, session_id: str, directory: Path) -> None:
        self._session = _Session(session_id, directory)
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True

    async def aclose(self) -> None:
        return None


def _session_dir(tmp_path: Path, session_id: str = "abc123") -> Path:
    directory = tmp_path / "sessions" / session_id
    directory.mkdir(parents=True)
    return directory


# --- the artifact -----------------------------------------------------------


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    """Every schema field survives the write, and the file is 0600."""
    directory = _session_dir(tmp_path)
    payload = {
        "session_id": "abc123",
        "conversation_dir": directory,
        "pid": 4242,
        "started_at": 1_700_000_000.25,
        "reason": "idle-exit",
        "detail": "no work, no viewer, no wake within 90s",
        "requested": False,
        "build": "0.84.4@abc1234",
    }

    path = journal.write_exit_record(**payload)

    assert path is not None
    assert path == directory / "runtime-exit.json"
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    stored = json.loads(path.read_text())
    # THE SCHEMA IS THE CONTRACT, key by key, in the order a reader may rely on
    # nothing about — so this asserts the SET and the values rather than a dump.
    assert stored["v"] == 1
    assert stored["kind"] == "runtime-exit"
    assert stored["session_id"] == "abc123"
    assert stored["pid"] == 4242
    assert stored["started_at"] == 1_700_000_000.25
    assert stored["reason"] == "idle-exit"
    assert stored["detail"] == "no work, no viewer, no wake within 90s"
    assert stored["requested"] is False
    assert stored["build"] == "0.84.4@abc1234"
    # ``pressure`` is always present with two keys, and an unanswerable reading is
    # null rather than 0 (0 would read as "the disk was full").
    assert set(stored["pressure"]) == {"free_disk_bytes", "load_1m"}
    assert isinstance(stored["at"], float)

    record = journal.read_exit_record(directory)
    assert record is not None
    assert record.reason == payload["reason"]
    assert record.detail == payload["detail"]
    assert record.requested is False
    assert record.pid == 4242
    assert record.started_at == 1_700_000_000.25


def test_the_write_replaces_atomically_and_leaves_no_temp(tmp_path: Path) -> None:
    """A second departure replaces the first in ONE rename, staging in-dir."""
    directory = _session_dir(tmp_path)
    journal.write_exit_record(
        "abc123", conversation_dir=directory, started_at=1.0, reason="idle-exit"
    )

    seen: list[tuple[str, str]] = []
    real_replace = os.replace

    def _spy(src: Any, dst: Any) -> None:
        seen.append((str(src), str(dst)))
        # The source is a sibling temp file (same filesystem, so the rename
        # cannot cross a mount) and the destination is the artifact itself.
        assert Path(src).parent == directory
        assert Path(src).name.startswith(".runtime-exit.json.")
        assert Path(src).name.endswith(".tmp")
        assert Path(dst) == directory / "runtime-exit.json"
        real_replace(src, dst)

    original = registry.os.replace
    registry.os.replace = _spy
    try:
        journal.write_exit_record(
            "abc123", conversation_dir=directory, started_at=2.0, reason="socket-stop"
        )
    finally:
        registry.os.replace = original

    assert len(seen) == 1
    assert sorted(p.name for p in directory.iterdir()) == ["runtime-exit.json"]
    record = journal.read_exit_record(directory)
    assert record is not None and record.reason == "socket-stop"


def test_best_effort_on_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write that fails answers ``None`` and never raises out of the exit path."""
    directory = _session_dir(tmp_path)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk is gone")

    monkeypatch.setattr(registry, "_staged_write", _boom)
    assert (
        journal.write_exit_record(
            "abc123", conversation_dir=directory, started_at=1.0, reason="idle-exit"
        )
        is None
    )
    assert not (directory / "runtime-exit.json").exists()


def test_a_missing_conversation_is_not_created(tmp_path: Path) -> None:
    """No directory, no record — an instrument must not conjure a session."""
    missing = tmp_path / "sessions" / "nobody"

    assert (
        journal.write_exit_record(
            "nobody", conversation_dir=missing, started_at=1.0, reason="idle-exit"
        )
        is None
    )
    assert not missing.exists()


def test_untrustworthy_payloads_read_as_none(tmp_path: Path) -> None:
    """A torn, future or reason-less row is "no evidence", never an exception."""
    directory = _session_dir(tmp_path)
    path = directory / "runtime-exit.json"

    assert journal.read_exit_record(directory) is None
    path.write_text("{not json")
    assert journal.read_exit_record(directory) is None
    path.write_text(json.dumps({"v": 99, "pid": 1, "reason": "idle-exit"}))
    assert journal.read_exit_record(directory) is None
    path.write_text(json.dumps({"v": 1, "pid": 1, "at": 1.0}))
    assert journal.read_exit_record(directory) is None


def test_covers_run_refuses_a_previous_life(tmp_path: Path) -> None:
    """The run key is what keeps an old departure from explaining a new death."""
    directory = _session_dir(tmp_path)
    journal.write_exit_record(
        "abc123", conversation_dir=directory, pid=10, started_at=100.0, reason="idle-exit"
    )
    record = journal.read_exit_record(directory)
    assert record is not None

    assert record.covers_run(10, 100.0)
    assert record.covers_run(10, 100.5)
    assert not record.covers_run(11, 100.0)
    assert not record.covers_run(10, 500.0)
    # A missing run key on either side is no evidence to refuse on: the pid match
    # is what is left, and refusing there would delete an explanation rather than
    # protect one.
    assert record.covers_run(10, 0.0)


# --- the writer on the exit path -------------------------------------------


def test_clean_exit_writes_the_reason_and_detail_it_was_given(tmp_path: Path) -> None:
    """``_clean_exit`` is the one place every planned exit converges."""
    directory = _session_dir(tmp_path)
    handle = _Handle("abc123", directory)
    runtime = _Runtime("abc123", 1_700_000_000.0)

    asyncio.run(
        process._clean_exit(
            handle,
            runtime,
            reason="idle-exit",
            detail="no work, no viewer, no wake within 90s",
        )
    )

    record = journal.read_exit_record(directory)
    assert record is not None
    assert record.reason == "idle-exit"
    assert record.detail == "no work, no viewer, no wake within 90s"
    assert record.requested is False
    assert record.pid == os.getpid()
    assert record.session_id == "abc123"
    assert handle.disposed is True


def test_clean_exit_marks_a_requested_departure(tmp_path: Path) -> None:
    """A covering stop marker is what makes ``requested`` True, and it is run-scoped."""
    directory = _session_dir(tmp_path)
    uid = os.getpid()
    started_at = 1_700_000_000.0
    registry.write_stop_marker(
        directory,
        {
            "session_id": "abc123",
            "pid": uid,
            "started_at": started_at,
            "at": time.time(),
            "rung": "socket",
            "deliberate": True,
        },
    )
    handle = _Handle("abc123", directory)
    runtime = _Runtime("abc123", started_at)

    asyncio.run(process._clean_exit(handle, runtime, reason="socket-stop", detail="stop op"))

    record = journal.read_exit_record(directory)
    assert record is not None and record.requested is True

    # A MARKER FROM AN EARLIER LIFE DOES NOT COVER THIS ONE: the same session,
    # a later run, and the marker still names the old pid — so the departure is
    # NOT reported as the user's own act.
    later = _Handle("abc123", directory)
    later_runtime = _Runtime("abc123", started_at + 500.0)
    asyncio.run(process._clean_exit(later, later_runtime, reason="idle-exit"))
    record = journal.read_exit_record(directory)
    assert record is not None and record.requested is False


def test_clean_exit_without_identity_writes_nothing(tmp_path: Path) -> None:
    """A reduced handle (no conversation) leaves no record — and pays no write.

    This is the guard that keeps the 40 ms exit-path timing budget untouched for
    in-process hosts and the reaper's own test doubles.
    """

    class _Bare:
        disposed = False

        async def dispose(self) -> None:
            self.disposed = True

        async def aclose(self) -> None:
            return None

    handle = _Bare()
    runtime = _Runtime("", 0.0)
    asyncio.run(process._clean_exit(handle, runtime, reason="idle-exit"))

    assert handle.disposed is True
    assert list(tmp_path.rglob("runtime-exit.json")) == []


def test_a_non_boolean_requested_is_never_a_requested_stop(tmp_path: Path) -> None:
    """The one direction this field must not err in.

    A payload carrying the STRING ``"false"`` is truthy under ``bool()``. Read
    that as a request and a crash would be narrated as the user's own act — the
    wrong-verdict class the attribution path was fixed for. Only a JSON ``true``
    counts.
    """
    directory = _session_dir(tmp_path)
    journal.exit_record_path(directory).write_text(
        json.dumps({"v": 1, "pid": 1, "at": 1.0, "reason": "idle-exit", "requested": "false"})
    )

    record = journal.read_exit_record(directory)
    assert record is not None
    assert record.requested is False


# --- what the socket rung knows and the marker cannot (round 1, BLOCKER) ----


def test_a_socket_stop_is_recorded_as_requested_without_waiting_for_the_marker(
    tmp_path: Path,
) -> None:
    """``_clean_exit(requested=True)`` states a fact the marker cannot yet carry.

    ``control._graceful_stop`` deliberately stages its stop marker AFTER the
    target's ack, so the runtime starts exiting in the same tick the marker is
    written — measured 3/3 as ``requested: false`` for a stop a client asked for.
    The caller that knows says so, and the durable-evidence path is not consulted.
    """
    directory = _session_dir(tmp_path)
    handle = _Handle("abc123", directory)
    runtime = _Runtime("abc123", 1_700_000_000.0)

    asyncio.run(
        process._clean_exit(handle, runtime, reason="socket-stop", detail="stop op", requested=True)
    )

    record = journal.read_exit_record(directory)
    assert record is not None
    assert record.reason == "socket-stop"
    assert record.requested is True


def test_the_record_is_on_disk_before_dispose_runs(tmp_path: Path) -> None:
    """The write precedes the dispose, so a kill DURING dispose still explains itself.

    Dispose is the stage measured at minutes (it awaits every viewer's writer);
    a record written after it would be absent for exactly the slow exits a reader
    is most likely to find.
    """
    directory = _session_dir(tmp_path)
    seen: list[bool] = []

    class _WatchingHandle(_Handle):
        async def dispose(self) -> None:
            seen.append(journal.exit_record_path(directory).exists())
            self.disposed = True

    handle = _WatchingHandle("abc123", directory)
    runtime = _Runtime("abc123", 1_700_000_000.0)
    asyncio.run(process._clean_exit(handle, runtime, reason="idle-exit"))

    assert seen == [True]


# --- refusals the reader owes a listing --------------------------------------


def test_a_departure_stamp_a_reader_cannot_use_is_refused(tmp_path: Path) -> None:
    """Zero and non-finite stamps are no evidence, never an exception.

    A listing renders an age from ``at``; ``inf`` used to render a fabricated
    ``20747d`` and a zero a departure before the epoch. ``int('inf')`` raises
    ``OverflowError``, which must be a refusal rather than an escape.
    """
    directory = _session_dir(tmp_path)
    path = journal.exit_record_path(directory)
    base = {"v": 1, "session_id": "abc123", "pid": 1, "started_at": 1.0, "reason": "idle-exit"}

    for bad in (
        {**base, "at": 0},
        {**base, "at": -1.0},
        {**base, "at": float("inf")},
        {**base, "at": float("nan")},
        {**base, "at": 1.0, "pid": float("inf")},
        {**base, "at": 1.0, "pid": 0},
        {**base, "at": 1.0, "started_at": float("nan")},
        {**base, "at": 1.0, "started_at": -5.0},
    ):
        path.write_text(json.dumps(bad))
        assert journal.read_exit_record(directory) is None, bad


def test_a_record_from_another_conversation_is_refused(tmp_path: Path) -> None:
    """The directory is the authority: a mismatched ``session_id`` is no evidence."""
    directory = _session_dir(tmp_path)
    journal.write_exit_record(
        "someone-else", conversation_dir=directory, pid=1, started_at=1.0, reason="idle-exit"
    )

    assert journal.read_exit_record(directory) is None


# --- the run key at its OTHER end (round 1, MAJOR-1) ------------------------


def test_the_run_record_round_trips_and_only_its_own_key_clears_it(tmp_path: Path) -> None:
    """Boot writes it; only that run's clean exit removes it."""
    directory = _session_dir(tmp_path)
    path = journal.write_run_record(
        "abc123", conversation_dir=directory, pid=4242, started_at=900.0
    )
    assert path is not None
    assert oct(path.stat().st_mode & 0o777) == "0o600"

    record = journal.read_run_record(directory)
    assert record is not None
    assert record.covers_run(4242, 900.0)
    assert not record.covers_run(4242, 500.0)

    # A different run must not clear this one's marker: deleting a successor's
    # record would make its eventual death read as a clean exit.
    journal.clear_run_record(directory, pid=4242, started_at=500.0)
    assert journal.read_run_record(directory) is not None

    journal.clear_run_record(directory, pid=4242, started_at=900.0)
    assert journal.read_run_record(directory) is None


def test_a_clean_exit_removes_its_run_record(tmp_path: Path) -> None:
    """The pair "exit record written, run record gone" is a completed departure."""
    directory = _session_dir(tmp_path)
    journal.write_run_record("abc123", conversation_dir=directory, pid=os.getpid(), started_at=42.0)
    handle = _Handle("abc123", directory)
    runtime = _Runtime("abc123", 42.0)

    asyncio.run(process._clean_exit(handle, runtime, reason="idle-exit"))

    assert journal.read_run_record(directory) is None
    assert journal.read_exit_record(directory) is not None
