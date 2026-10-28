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


def _wait(gate: threading.Event) -> None:
    # Ops are ``() -> None``; ``Event.wait`` returns a bool.
    gate.wait(5)


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
    writer.submit("blocker", lambda: _wait(gate))
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
    EVIDENCE_WRITER.submit("blocker", lambda: _wait(gate))
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
    EVIDENCE_WRITER.submit("blocker", lambda: _wait(gate))
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
    real = cp._write_payload

    def recording(payload: dict[str, object]) -> None:
        writers.append(threading.get_ident())
        real(payload)

    monkeypatch.setattr(cp, "_write_payload", recording)
    monkeypatch.setattr(cp, "_git", lambda *a, **k: "")

    cp.note_step("bash", "Writing", 100.0)
    assert EVIDENCE_WRITER.flush(timeout=5.0)
    assert writers and threading.get_ident() not in writers
    assert json.loads(cp.path.read_text())["step"] == "bash: Writing"

    # A step queued just before ``clear`` must not land after it.
    gate = threading.Event()
    EVIDENCE_WRITER.submit("blocker", lambda: _wait(gate))
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


def test_the_notify_path_rereads_the_ask_log_at_most_once_per_floor(monkeypatch) -> None:
    """``_refresh_ask_projection`` runs on EVERY ``_notify``; its disk re-read
    of ``asks.jsonl`` was ~950 loop syscalls per 40 s on a busy manager. A burst
    of notifies reads once, an install from the queue's own sink counts as
    fresh, and once the floor elapses the next notify reads again (so another
    process's change still lands, at most one floor late)."""
    from local_operator.session import frontend_state
    from local_operator.session.runtime import serving

    reads: list[int] = []
    installed: list[object] = []

    def fake_ask_wire(session: object) -> tuple[list[dict[str, object]], int]:
        reads.append(1)
        return [], 0

    class _Fold:
        def set_asks(self, rows: object, count: object) -> None:
            installed.append((rows, count))

    monkeypatch.setattr(frontend_state, "ask_wire", fake_ask_wire)
    handle = serving.ServingSessionHandle.__new__(serving.ServingSessionHandle)
    handle._session = object()  # type: ignore[attr-defined]
    handle._fold = _Fold()  # type: ignore[attr-defined]

    for _ in range(50):
        handle._refresh_ask_projection()
    assert len(reads) == 1, "a notify burst re-read the ask log more than once"

    monkeypatch.setattr(serving, "_ASK_REFRESH_MIN_S", 0.0)
    handle._refresh_ask_projection()
    assert len(reads) == 2, "an elapsed floor did not re-read"

    monkeypatch.setattr(serving, "_ASK_REFRESH_MIN_S", 3600.0)
    handle._ask_fold_installed_at = float("-inf")  # type: ignore[attr-defined]
    handle._publish_pending_gate = lambda: None  # type: ignore[method-assign]
    handle._on_projection = None  # type: ignore[attr-defined]
    handle._ask_state_changed([{"ask_id": "a"}], 1)
    handle._refresh_ask_projection()
    assert len(reads) == 2, "the queue's own fresh install did not count as fresh"
    assert installed[-1] == ([{"ask_id": "a"}], 1)


def test_the_channel_sidecar_is_read_at_most_once_per_floor_on_the_tick_path(
    tmp_path: Path, monkeypatch
) -> None:
    """``pending_channel_records`` runs on every jobs tick; its sidecar read is
    throttled there, while a dedicated money publish still reads fresh."""
    from local_operator.session import session as session_module

    reads: list[int] = []

    def fake_read(directory: object) -> list[dict[str, object]]:
        reads.append(1)
        return []

    monkeypatch.setattr(session_module, "read_pending_records", fake_read)

    class _Transcript:
        directory = tmp_path

    class _Channels:
        def has(self, record_id: str) -> bool:
            return False

    host = session_module.Session.__new__(session_module.Session)
    host._transcript = _Transcript()  # type: ignore[attr-defined]
    host.channels = _Channels()  # type: ignore[attr-defined]
    for _ in range(20):
        host.pending_channel_records()
    assert len(reads) == 1, "a burst of ticks re-read the sidecar"

    host._publish_channel_spend()  # no store: only the cache drop matters here
    host.pending_channel_records()
    assert len(reads) == 2, "a money edge did not read the sidecar fresh"


def test_clear_is_final_even_against_an_op_already_dequeued(tmp_path: Path, monkeypatch) -> None:
    """Review round 1, R1-1: hold the writer BETWEEN dequeue and execution.

    The writer used to pop the op under the state lock and take the I/O lock
    only afterwards; ``clear()`` arriving in that gap found nothing pending,
    unlinked, and the popped step write then re-created the receipt. The gap
    is held open deliberately here: the writer parks right after the pop until
    ``clear`` has had every chance to run.
    """
    child = tmp_path / "child"
    child.mkdir()
    cp = WipCheckpointer(tmp_path, child, "job-gap", "coder")
    monkeypatch.setattr(cp, "_git", lambda *a, **k: "")
    cp.note_step("bash", "First", 100.0)
    assert EVIDENCE_WRITER.flush(timeout=5.0)
    assert cp.path.exists()

    popped = threading.Event()
    release = threading.Event()
    real_take = EVIDENCE_WRITER._take

    def take_then_park():
        op = real_take()
        if op is not None and not popped.is_set():
            popped.set()
            release.wait(5)
        return op

    monkeypatch.setattr(EVIDENCE_WRITER, "_take", take_then_park)
    cp.last_receipt = 0.0
    cp.note_step("bash", "step before settle", 101.0)
    assert popped.wait(5), "the writer never dequeued the step"
    clearing = threading.Thread(target=cp.clear)
    clearing.start()
    time.sleep(0.05)  # give an unguarded clear() time to unlink inside the gap
    release.set()
    clearing.join(5)
    assert EVIDENCE_WRITER.flush(timeout=5.0)

    assert not cp.path.exists(), "a step dequeued before clear() resurrected the receipt"


def test_a_throttled_ask_read_still_lands_when_the_session_goes_quiet(monkeypatch) -> None:
    """Review round 1, R1-2 / QA Q1 (``rig.py tail``), reproduced on a real loop.

    An event reads the fold; ANOTHER process then appends an ask; one more
    event arrives inside the floor (skipped) and the session goes quiet. The
    ask must reach the fold — and a republish must go out — within about one
    floor, not wait for an unrelated event that may never come.
    """
    import asyncio

    from local_operator.session import frontend_state
    from local_operator.session.runtime import serving

    disk: list[dict[str, object]] = []
    installed: list[object] = []
    notified: list[int] = []

    monkeypatch.setattr(frontend_state, "ask_wire", lambda session: (list(disk), len(disk)))
    monkeypatch.setattr(serving, "_ASK_REFRESH_MIN_S", 0.2)

    class _Fold:
        def set_asks(self, rows: object, count: object) -> None:
            installed.append((rows, count))

    async def scenario() -> None:
        handle = serving.ServingSessionHandle.__new__(serving.ServingSessionHandle)
        handle._session = object()  # type: ignore[attr-defined]
        handle._fold = _Fold()  # type: ignore[attr-defined]
        handle._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
        handle._disposing = False  # type: ignore[attr-defined]

        def notify() -> None:
            # ``_notify``'s ask leg: ``_publish_pending_gate`` refreshes the fold.
            notified.append(1)
            handle._refresh_ask_projection()

        handle._notify = notify  # type: ignore[method-assign]

        handle._refresh_ask_projection()  # an event stamps the fold
        disk.append({"ask_id": "from-another-process", "state": "queued"})
        await asyncio.sleep(0.02)
        handle._refresh_ask_projection()  # trailing event, inside the floor: skipped
        handle._refresh_ask_projection()  # a burst still arms ONE re-check
        assert installed[-1] == ([], 0)
        await asyncio.sleep(0.5)  # quiet: no further events

    asyncio.run(scenario())
    assert installed[-1] == (
        [{"ask_id": "from-another-process", "state": "queued"}],
        1,
    ), "an external ask change skipped by the throttle never reached the fold"
    assert len(notified) == 1, "the trailing re-check did not coalesce to one republish"


def test_a_throttled_sidecar_read_still_lands_when_the_session_goes_quiet(
    tmp_path: Path, monkeypatch
) -> None:
    """R1-2 for the voice-spend sidecar: a marker written by another process
    just after a read is published within one floor even with no later tick;
    an unchanged sidecar publishes nothing extra."""
    import asyncio

    from local_operator.session import session as session_module

    disk: list[dict[str, object]] = []
    published: list[list[dict[str, object]]] = []
    monkeypatch.setattr(session_module, "read_pending_records", lambda directory: list(disk))
    monkeypatch.setattr(session_module, "_CHANNEL_PENDING_MIN_S", 0.2)

    class _Transcript:
        directory = tmp_path

    class _Channels:
        def has(self, record_id: str) -> bool:
            return False

    host = session_module.Session.__new__(session_module.Session)
    host._transcript = _Transcript()  # type: ignore[attr-defined]
    host.channels = _Channels()  # type: ignore[attr-defined]

    class _Store:
        def refresh_spend_channels(self, session: object) -> None:
            published.append(host.pending_channel_records())

    host._frontend_state_store = _Store()  # type: ignore[attr-defined]

    async def scenario() -> None:
        assert host.pending_channel_records() == []  # a tick reads
        host.pending_channel_records()  # inside the floor, unchanged: arms a re-check
        await asyncio.sleep(0.5)
        assert published == [], "an unchanged sidecar published on the trailing edge"
        host.pending_channel_records()  # floor elapsed: reads
        disk.append({"record_id": "voice-1"})  # another process writes just after
        host.pending_channel_records()  # inside the floor: cached, arms a re-check
        await asyncio.sleep(0.5)  # quiet

    asyncio.run(scenario())
    assert published and published[0] == [
        {"record_id": "voice-1"}
    ], "a sidecar marker skipped by the throttle never reached the band"
