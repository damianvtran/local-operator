"""The evidence writer keeps per-lane receipts off the session loop.

WHY. ``subagent_ledger.note_lane_step`` and ``WipCheckpointer.note_step`` run
from a child's ``on_tool_end`` relay, i.e. ON THE PARENT SESSION'S LOOP; their
synchronous ``mkstemp``/``json.dump``/``os.replace`` measured up to 315 ms per
call and ~1.8 s of loop wall per 40 s on a 12-lane manager, and every control
op (a model switch included) waited behind it. These cells pin the four
properties the move must keep: the write is not on the calling thread; bursts
coalesce latest-wins; ``force`` (a new recovery ref) survives coalescing; and a
settle's unlink can never be undone by a deferred write (crash-safety and
ordering).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from local_operator.harness.wip_checkpoint import WipCheckpointer
from local_operator.session import subagent_ledger as ledger
from local_operator.session.coalescing_writer import EVIDENCE_WRITER, CoalescingWriter


def _seed_lane(child: Path, job_id: str = "job-1") -> Path:
    child.mkdir(parents=True, exist_ok=True)
    path = ledger.write_lane_receipt(child, ledger.build_lane_payload(job_id=job_id))
    assert path is not None and path.exists()
    return path


@pytest.fixture(autouse=True)
def _drained():
    yield
    assert EVIDENCE_WRITER.flush(timeout=5.0), "the evidence writer never drained"


def test_ops_run_off_the_calling_thread_and_in_submission_order() -> None:
    writer = CoalescingWriter("test-writer")
    seen: list[tuple[str, int]] = []
    caller = threading.get_ident()

    for index in range(3):
        writer.submit(
            f"key-{index}", lambda i=index: seen.append((f"key-{i}", threading.get_ident()))
        )

    assert writer.flush(timeout=5.0)
    assert [key for key, _ in seen] == ["key-0", "key-1", "key-2"]
    assert all(thread != caller for _, thread in seen), "an op ran on the submitting thread"


def test_a_burst_on_one_key_coalesces_to_the_latest_op() -> None:
    writer = CoalescingWriter("test-writer")
    gate = threading.Event()
    ran: list[str] = []
    # Park the writer on an unrelated key so the burst below queues behind it.
    writer.submit("blocker", lambda: gate.wait(5))
    for value in ("a", "b", "c"):
        writer.submit("same", lambda v=value: ran.append(v))
    gate.set()

    assert writer.flush(timeout=5.0)
    assert ran == ["c"], "a coalesced key ran a superseded op"


def test_note_lane_step_does_not_write_on_the_calling_thread(tmp_path: Path, monkeypatch) -> None:
    path = _seed_lane(tmp_path / "child")
    writers: list[int] = []
    real = ledger._staged_write

    def recording(target: Path, payload: object) -> None:
        writers.append(threading.get_ident())
        real(target, payload)

    monkeypatch.setattr(ledger, "_staged_write", recording)

    ledger.note_lane_step(tmp_path / "child", "job-1", "bash: Building", 10.0, force=True)
    assert EVIDENCE_WRITER.flush(timeout=5.0)

    assert writers and threading.get_ident() not in writers
    assert json.loads(path.read_text())["last_step"] == "bash: Building"


def test_a_forced_step_survives_being_coalesced_with_a_throttled_one(
    tmp_path: Path, monkeypatch
) -> None:
    """A NEW recovery ref must be durable even when a later throttled step
    replaces its queued write: the merge keeps ``force``, so the inner 5 s
    throttle cannot drop the ref the forced write was carrying."""
    child = tmp_path / "child"
    path = _seed_lane(child)
    # A fresh stamp puts every unforced write inside the 5 s throttle window.
    payload = json.loads(path.read_text())
    payload["step_written_at"] = time.time()
    path.write_text(json.dumps(payload))
    gate = threading.Event()
    EVIDENCE_WRITER.submit("blocker", lambda: gate.wait(5))
    try:
        ledger.note_lane_step(child, "job-1", "checkpoint", 11.0, "refs/lop/wip/x", force=True)
        ledger.note_lane_step(child, "job-1", "bash: later", 12.0, "refs/lop/wip/x")
    finally:
        gate.set()
    assert EVIDENCE_WRITER.flush(timeout=5.0)

    written = json.loads(path.read_text())
    assert written["wip_ref"] == "refs/lop/wip/x"
    assert written["last_step"] == "bash: later"


def test_a_withdrawn_lane_receipt_is_never_resurrected_by_a_queued_step(tmp_path: Path) -> None:
    """Ordering + crash-safety: the settle's unlink drops the pending step, so a
    settled lane can never be re-reported as "ran here and never settled"."""
    child = tmp_path / "child"
    path = _seed_lane(child)
    gate = threading.Event()
    EVIDENCE_WRITER.submit("blocker", lambda: gate.wait(5))
    try:
        ledger.note_lane_step(child, "job-1", "bash: last", 13.0, force=True)
        ledger.withdraw_lane_receipt(child, "job-1")
    finally:
        gate.set()
    assert EVIDENCE_WRITER.flush(timeout=5.0)

    assert not path.exists()


def test_withdraw_waits_out_a_step_already_mid_write(tmp_path: Path, monkeypatch) -> None:
    """The narrower race: the step write is INSIDE its rename when the settle
    arrives. ``exclusive`` waits for it, so the unlink is the last word."""
    child = tmp_path / "child"
    path = _seed_lane(child)
    entered, release = threading.Event(), threading.Event()
    real = ledger._staged_write

    def slow(target: Path, payload: object) -> None:
        entered.set()
        release.wait(5)
        real(target, payload)

    monkeypatch.setattr(ledger, "_staged_write", slow)
    ledger.note_lane_step(child, "job-1", "bash: racing", 14.0, force=True)
    assert entered.wait(5)
    withdrawn = threading.Thread(target=ledger.withdraw_lane_receipt, args=(child, "job-1"))
    withdrawn.start()
    time.sleep(0.05)  # let the withdraw reach the lock; the assert below is structural
    assert withdrawn.is_alive(), "the unlink did not wait for the in-flight write"
    release.set()
    withdrawn.join(5)

    assert not path.exists()


def test_wip_note_step_defers_the_receipt_and_clear_is_final(tmp_path: Path, monkeypatch) -> None:
    child = tmp_path / "child"
    child.mkdir()
    cp = WipCheckpointer(tmp_path, child, "job-w", "coder")
    writers: list[int] = []
    real = cp._store

    def recording(payload: dict) -> None:
        writers.append(threading.get_ident())
        real(payload)

    monkeypatch.setattr(cp, "_store", recording)
    monkeypatch.setattr(cp, "_git", lambda *a, **k: "")

    cp.note_step("bash", "Writing", 100.0)
    assert EVIDENCE_WRITER.flush(timeout=5.0)
    assert writers and threading.get_ident() not in writers
    assert json.loads(cp.path.read_text())["step"] == "bash: Writing"

    # A step queued just before ``clear`` must not land after it.
    gate = threading.Event()
    EVIDENCE_WRITER.submit("blocker", lambda: gate.wait(5))
    try:
        cp.last_receipt = 0.0
        cp.note_step("write", "Racing the settle", 101.0)
        cp.clear()
    finally:
        gate.set()
    assert EVIDENCE_WRITER.flush(timeout=5.0)
    assert not cp.path.exists()


def test_a_failing_op_does_not_stop_the_writer() -> None:
    writer = CoalescingWriter("test-writer")
    ran: list[str] = []

    def boom() -> None:
        raise OSError("disk full")

    writer.submit("bad", boom)
    writer.submit("good", lambda: ran.append("ok"))

    assert writer.flush(timeout=5.0)
    assert ran == ["ok"]
