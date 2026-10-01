"""The queued move's durable record and its driver (design note §5.4).

WHAT THESE PIN, in the order the record lives it:

* the record's file discipline — 0600, atomic, no litter, and a read-modify-write
  that is serialised ACROSS PROCESSES by the per-record flock (F3, §2.3: atomic
  write is not concurrency control);
* the phase machine — monotone transitions, terminal write-once, ``claim_pause``
  as the cancel race's fulcrum, and ``cancel`` refusing past it with the phase
  named rather than a word that claims the move started when it did not;
* the driver — wait-for-park (a lease holder is a writer), one delivery per
  runtime process (the re-arm contract), the copy start exactly when no writer
  remains, and refusal folds that keep the sentence verbatim.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import move_queue as mq

SID = "9f3ac1e0b7d2"


def _enqueue(root: Path, request_id: str = "rq1", **overrides: object) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "to_device": "d_b",
        "to_name": "dev-b",
        "request_id": request_id,
        "requested_by": "d_a",
    }
    payload.update(overrides)
    record, _created = mq.enqueue(root, SID, **payload)  # type: ignore[arg-type]
    return record


def _phases(root: Path) -> list[str]:
    record = mq.read_record(root, SID) or {}
    return [str(stamp.get("phase") or "") for stamp in record.get("phases") or []]


def _wait_for(predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class _FakeRuntime:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        # The driver fails CLOSED on a runtime that does not advertise the queued
        # move (`not_implemented`, reload first); a fake that forgot this would be
        # testing the fold instead of the delivery cadence.
        from local_operator.session.runtime.types import QUEUED_MOVE_CAPABILITY

        self.capabilities = (QUEUED_MOVE_CAPABILITY,)


class _FakeServer:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)


class TestRecord:
    def test_enqueue_is_idempotent_and_a_second_request_sees_the_first(
        self, tmp_path: Path
    ) -> None:
        record, created = mq.enqueue(
            tmp_path, SID, to_device="d_b", to_name="dev-b", request_id="rq1", requested_by="d_a"
        )
        assert created is True and record["phase"] == "queued"
        same, created_again = mq.enqueue(
            tmp_path, SID, to_device="d_b", to_name="dev-b", request_id="rq1", requested_by="d_a"
        )
        assert created_again is False and same["request_id"] == "rq1"
        other, _ = mq.enqueue(
            tmp_path, SID, to_device="d_b", to_name="dev-b", request_id="rq2", requested_by="d_a"
        )
        # The caller distinguishes same-vs-different by the record it gets back;
        # two live requests for one session must never both look accepted.
        assert other["request_id"] == "rq1"

    def test_transitions_are_monotone_and_terminal_is_write_once(self, tmp_path: Path) -> None:
        _enqueue(tmp_path)
        finished = mq.transition(tmp_path, SID, "finishing", writer="relay")
        assert finished is not None and finished["phase"] == "finishing"
        # An out-of-order phase is refused silently (no rewrite, no error).
        assert mq.transition(tmp_path, SID, "queued", writer="relay") is None
        # (and the phase is untouched — the refusal above proves it)
        assert (mq.read_record(tmp_path, SID) or {})["phase"] == "finishing"
        mq.transition(tmp_path, SID, "paused", writer="relay")
        mq.transition(tmp_path, SID, "copying", writer="relay")
        mq.transition(tmp_path, SID, "resumed", writer="relay")
        # Terminal is write-once: a later failure fold cannot rewrite a commit.
        mq.transition(tmp_path, SID, "failed", detail="too late", code="x", writer="relay")
        assert (mq.read_record(tmp_path, SID) or {})["phase"] == "resumed"
        assert _phases(tmp_path) == ["queued", "finishing", "paused", "copying", "resumed"]

    def test_claim_pause_is_the_cancel_point(self, tmp_path: Path) -> None:
        _enqueue(tmp_path)
        _, outcome = mq.claim_pause(tmp_path, SID)
        assert outcome == "claimed"
        assert (mq.read_record(tmp_path, SID) or {})["phase"] == "paused"
        # And a cancel arriving after the claim is refused, naming the phase.
        record, cancel_outcome = mq.cancel(tmp_path, SID)
        assert cancel_outcome == "too_late"
        assert (record or {})["phase"] == "paused"

    def test_cancel_honours_queued_and_finishing(self, tmp_path: Path) -> None:
        _enqueue(tmp_path, request_id="rq-q")
        record, outcome = mq.cancel(tmp_path, SID)
        assert outcome == "cancelled" and record is not None and record["phase"] == "cancelled"
        assert _phases(tmp_path) == ["queued", "cancelled"]

        root2 = tmp_path / "second"
        _enqueue(root2, request_id="rq-f")
        mq.transition(root2, SID, "finishing", writer="relay")
        record2, outcome2 = mq.cancel(root2, SID)
        assert outcome2 == "cancelled" and record2 is not None and record2["phase"] == "cancelled"

    def test_a_second_cancel_reads_as_already_not_too_late(self, tmp_path: Path) -> None:
        _enqueue(tmp_path)
        mq.cancel(tmp_path, SID)
        record, outcome = mq.cancel(tmp_path, SID)
        assert outcome == "already"
        assert (record or {})["phase"] == "cancelled"
        assert mq.cancel(tmp_path / "nothing-here", SID)[1] == "absent"

    def test_the_file_is_0600_with_no_staging_litter(self, tmp_path: Path) -> None:
        _enqueue(tmp_path)
        path = mq.record_path(tmp_path, SID)
        assert path.is_file()
        assert os.stat(path).st_mode & 0o777 == 0o600
        leftovers = [p.name for p in path.parent.iterdir() if p.name.startswith(".")]
        assert leftovers == [], leftovers

    def test_sweep_removes_only_old_terminal_records(self, tmp_path: Path) -> None:
        _enqueue(tmp_path)
        assert mq.sweep_terminal(tmp_path) == []
        assert mq.read_record(tmp_path, SID) is not None
        mq.cancel(tmp_path, SID)
        assert mq.sweep_terminal(tmp_path) == []
        assert mq.sweep_terminal(tmp_path, max_age_s=-1.0) == [SID]
        assert mq.read_record(tmp_path, SID) is None

    def test_a_read_modify_write_serialises_across_processes(self, tmp_path: Path) -> None:
        """F3's point: the flock is the concurrency control, not the atomic write.

        A child process takes the record's lock, writes ``finishing``, holds the lock
        for a beat, and exits. The parent's cancel — which read the record before the
        child's write, had it been unserialised — must block, see the child's state,
        and land its terminal decision on top of it. Without the lock the parent's
        write would be a lost update of a record it believed was ``queued``.
        """
        _enqueue(tmp_path, request_id="rq-x")
        # THE CHILD IS DELIBERATELY A RAW WRITER. Calling ``transition`` inside the
        # held lock would test the lock with the lock: on this platform ``flock``
        # is per open-file-description, so the inner acquisition blocks and the
        # child deadlocks instead of exercising the parent. Raw read-modify-write
        # under the held lock is exactly what a competing writer does, so this is
        # the honest shape of "another process is mid-write".
        child = "\n".join(
            [
                "import sys, time",
                "sys.path.insert(0, sys.argv[1])",
                "from pathlib import Path",
                "from local_operator.network import move_queue as mq",
                "root = Path(sys.argv[2])",
                "with mq.lock_for(root, sys.argv[3]):",
                "    record = mq._record_or_none(root, sys.argv[3])",
                "    mq._write_record_raw(",
                "        root, sys.argv[3], mq._stamp(record, 'finishing', writer='relay')",
                "    )",
                "    time.sleep(0.8)",
            ]
        )
        repo_root = str(Path(__file__).resolve().parents[3])
        proc = subprocess.Popen(
            [sys.executable, "-c", child, repo_root, str(tmp_path), SID],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # Wait until the child has landed its write (the lock is held now).
            assert _wait_for(
                lambda: (mq.read_record(tmp_path, SID) or {}).get("phase") == "finishing"
            )
            started = time.monotonic()
            record, outcome = mq.cancel(tmp_path, SID)
            waited = time.monotonic() - started
        finally:
            proc.wait(timeout=10)
        assert outcome == "cancelled", (outcome, record)
        assert record is not None and record["phase"] == "cancelled"
        assert waited >= 0.3, f"the cancel did not wait for the child's lock ({waited:.2f}s)"
        assert _phases(tmp_path) == ["queued", "finishing", "cancelled"]


class TestDriver:
    @pytest.fixture(autouse=True)
    def _faster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mq, "DRIVER_POLL_S", 0.02)
        monkeypatch.setattr(mq, "DRIVER_WAIT_S", 0.02)

    def test_a_lease_holder_is_a_writer_and_the_driver_waits_for_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = _FakeServer(tmp_path)
        _enqueue(tmp_path)
        held = {"value": True}
        copied: list[str] = []

        monkeypatch.setattr(mq, "_find_runtime", lambda root, sid: None)
        monkeypatch.setattr(mq, "_lease_holder_present", lambda root, sid: held["value"])

        def fake_copy(srv: object, sid: str) -> str:
            copied.append(sid)
            mq.transition(tmp_path, sid, "copying", writer="relay")
            mq.transition(tmp_path, sid, "resumed", writer="relay")
            return "resumed"

        monkeypatch.setattr(mq, "_drive_copy", fake_copy)
        monkeypatch.setattr(mq, "DRIVER_WAIT_S", 0.05)
        worker = threading.Thread(target=mq._driver_main, args=(server, SID), daemon=True)
        worker.start()
        assert _wait_for(lambda: "finishing" in _phases(tmp_path))
        assert copied == [], "the copy started while a lease holder was present"
        held["value"] = False
        assert _wait_for(lambda: copied == [SID], timeout_s=10.0)
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert copied == [SID]
        assert _phases(tmp_path)[:2] == ["queued", "finishing"]
        assert _phases(tmp_path)[-2:] == ["copying", "resumed"]

    def test_the_intent_is_delivered_once_per_runtime_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = _FakeServer(tmp_path)
        _enqueue(tmp_path)
        current = {"record": _FakeRuntime(pid=4242)}
        deliveries: list[str] = []

        monkeypatch.setattr(mq, "_find_runtime", lambda root, sid: current["record"])
        monkeypatch.setattr(mq, "_lease_holder_present", lambda root, sid: False)
        monkeypatch.setattr(
            mq, "_deliver_intent", lambda srv, sid: deliveries.append(sid) or "delivered"
        )
        monkeypatch.setattr(mq, "_drive_copy", lambda srv, sid: "stopped")
        monkeypatch.setattr(mq, "DRIVER_POLL_S", 0.05)
        monkeypatch.setattr(mq, "DRIVER_WAIT_S", 0.05)

        worker = threading.Thread(target=mq.drive, args=(server, SID), daemon=True)
        worker.start()
        assert _wait_for(lambda: len(deliveries) == 1)
        time.sleep(0.4)
        assert deliveries == [SID], "the same runtime process was delivered to twice"
        # A REPLACED runtime (a new pid) gets the intent again — the re-arm
        # contract; the same object identity is not enough, the pid is the fact.
        current["record"] = _FakeRuntime(pid=5150)
        assert _wait_for(lambda: len(deliveries) == 2, timeout_s=10.0)
        # End the driver deterministically: the record going terminal is its own
        # stand-down path, so the thread is gone before the monkeypatches are.
        mq.transition(tmp_path, SID, "failed", detail="test", code="x", writer="relay")
        worker.join(timeout=5)
        assert not worker.is_alive()

    def test_a_cancel_while_finishing_ends_the_driver_without_a_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = _FakeServer(tmp_path)
        _enqueue(tmp_path)
        copied: list[str] = []
        monkeypatch.setattr(mq, "_find_runtime", lambda root, sid: None)
        monkeypatch.setattr(mq, "_lease_holder_present", lambda root, sid: True)
        monkeypatch.setattr(mq, "_drive_copy", lambda srv, sid: copied.append(sid) or "nope")

        worker = threading.Thread(target=mq.drive, args=(server, SID), daemon=True)
        worker.start()
        assert _wait_for(lambda: "finishing" in _phases(tmp_path))
        assert mq.cancel(tmp_path, SID)[1] == "cancelled"
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert copied == []
        assert (mq.read_record(tmp_path, SID) or {})["phase"] == "cancelled"

    def test_a_failed_copy_folds_with_the_sentence_verbatim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = _FakeServer(tmp_path)
        _enqueue(tmp_path)
        monkeypatch.setattr(mq, "_find_runtime", lambda root, sid: None)
        monkeypatch.setattr(mq, "_lease_holder_present", lambda root, sid: False)
        sentence = "the destination is not answering right now; nothing was changed"

        def failing(srv: object, sid: str) -> str:
            mq.transition(
                tmp_path, sid, "failed", detail=sentence, code="unreachable", writer="relay"
            )
            return "failed"

        monkeypatch.setattr(mq, "_drive_copy", failing)
        worker = threading.Thread(target=mq.drive, args=(server, SID), daemon=True)
        worker.start()
        assert _wait_for(lambda: (mq.read_record(tmp_path, SID) or {}).get("phase") == "failed")
        record = mq.read_record(tmp_path, SID) or {}
        assert record["detail"] == sentence
        assert record["code"] == "unreachable"


class TestCliSurface:
    """The two verb shapes the parser must offer, and the dispatch's reading of them.

    Parse-level deliberately: the wire behaviour behind both calls is covered
    where it lives (``TestDriver`` above, the e2e cell), and what can rot HERE is
    the surface — a renamed dest, a cancel verb that stopped carrying the id, or
    ``--queue`` failing to travel as the request flag.
    """

    def test_the_parser_accepts_the_queue_flag_and_the_cancel_verb(self) -> None:
        from local_operator.cli import build_cli_parser

        parser = build_cli_parser()
        queued = parser.parse_args(["sessions", "move", "abc123", "--to", "peer", "--queue"])
        assert queued.queue is True
        assert queued.session == "abc123" and queued.to == "peer"
        cancelled = parser.parse_args(["sessions", "move", "--cancel-queued", "abc123"])
        assert cancelled.cancel_queued == "abc123"

    def test_the_dispatch_sends_the_queue_request_and_the_cancel_by_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator import cli
        from local_operator.network import mobility

        calls: list[tuple[str, dict[str, Any]]] = []

        def fake_move(session_id: str, **kwargs: object) -> dict[str, Any]:
            calls.append(("move", {"session_id": session_id, **kwargs}))
            return {"ok": True, "code": "", "message": "", "session_id": session_id}

        def fake_cancel(session_id: str, **kwargs: object) -> dict[str, Any]:
            calls.append(("cancel", {"session_id": session_id, **kwargs}))
            return {"ok": True, "code": "", "message": "", "session_id": session_id}

        monkeypatch.setattr(mobility, "request_move", fake_move)
        monkeypatch.setattr(mobility, "request_move_cancel", fake_cancel)
        parser = cli.build_cli_parser()
        args = parser.parse_args(
            ["sessions", "move", "abc123", "--to", "peer", "--queue", "--json"]
        )
        assert cli.sessions_move_command(args) == 0
        assert calls[0][0] == "move" and calls[0][1]["queue"] is True, calls
        args = parser.parse_args(["sessions", "move", "--cancel-queued", "abc123", "--json"])
        assert cli.sessions_move_command(args) == 0
        assert calls[-1][0] == "cancel" and calls[-1][1]["session_id"] == "abc123", calls
