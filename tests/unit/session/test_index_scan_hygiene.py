"""Index-scan hygiene: no synchronous builds, one scan per journal, pid-stamped traces.

Three defects the transcript-index ADR's caller map found, independent of the
off-process decision (PR #4219):

1. ``peer_cycle_display``'s readers called ``refresh_index`` synchronously --
   a whole-journal scan on the TUI's and the runtime's event loops -- although
   the module promised "never a scan".
2. Two opens of one journal from different loops (the relay's page loop and
   another loop) or different processes (``lop serve`` and the mesh relay)
   each scanned it.
3. ``open_trace`` rows carried no pid, so serve rows and runtime rows sharing
   one ``LOP_OPEN_TRACE`` file could not be told apart.

Synthetic journals only; nothing here touches a live session.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness import peer_cycle_display as pcd
from local_operator.session import open_trace
from local_operator.session import transcript_index as ti

SID = "aabbccddee02"


@pytest.fixture(autouse=True)
def _clean_module_state():
    ti._reset_for_tests()
    yield
    ti._reset_for_tests()


def _journal(root: Path) -> Path:
    path = root / "sessions" / SID / "transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "u1",
            "ts": 1.0,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": "hi"}]},
        },
        {
            "id": "a1",
            "ts": 2.0,
            "type": "message",
            "payload": {"kind": "message", "role": "assistant", "content": [{"text": "yo"}]},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _count_scans(monkeypatch) -> dict[str, int]:
    calls = {"full": 0, "incremental": 0}
    real_full, real_incremental = ti._scan_full, ti._scan_incremental

    def full(*args, **kwargs):
        calls["full"] += 1
        return real_full(*args, **kwargs)

    def incremental(*args, **kwargs):
        calls["incremental"] += 1
        return real_incremental(*args, **kwargs)

    monkeypatch.setattr(ti, "_scan_full", full)
    monkeypatch.setattr(ti, "_scan_incremental", incremental)
    return calls


# --- 1. peer_cycle_display never builds -------------------------------------


def _in_thread(fn):
    """Run ``fn`` on a plain thread with no event loop: a ``to_thread`` worker."""
    out: dict[str, Any] = {}

    def run() -> None:
        try:
            out["value"] = fn()
        except BaseException as error:  # noqa: BLE001 — re-raised on the test thread
            out["error"] = error

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(60)
    assert not worker.is_alive(), "off-loop reader hung"
    if "error" in out:
        raise out["error"]
    return out["value"]


def _persisted_cycle(root: Path) -> tuple[Path, list[Any], dict[str, Any]]:
    """A real peer cycle with its display record archived in a journal, then the
    module's memory dropped so the next reader meets a COLD index."""
    from local_operator.harness.peer_cycle_display import persist_record
    from local_operator.session.transcript import Transcript
    from tests.unit.session.test_peer_cycle_display_archive import make_cycle

    _, history, record = make_cycle()
    directory = root / "sessions" / "sess"

    async def write() -> None:
        transcript = Transcript(directory)
        await transcript.append_messages(history[:-1])
        assert await persist_record(transcript, record)
        await transcript.append_message(history[-1])

    asyncio.run(write())
    ti._reset_for_tests()
    return directory, history, record


def test_off_loop_reader_builds_the_cold_index_itself(tmp_path, monkeypatch):
    """Review round 1, M1. A ``to_thread`` worker has no loop to arm a refresh on,
    so "not ready" would be its permanent answer; it is already off every loop,
    so it builds directly. Failed on f6759db with ``status: building`` and no
    records, every time."""
    directory, history, record = _persisted_cycle(tmp_path)
    scans = _count_scans(monkeypatch)

    response = _in_thread(lambda: pcd.discover_records(directory, history, "sess"))

    assert response is not None and "status" not in response
    assert response["records"] == [record]
    assert not response["unavailable"] and not response["remaining"]
    assert sum(scans.values()) == 1
    assert not ti._IN_FLIGHT, "an off-loop build owns no loop task"
    # Built once: the next off-loop read is served from memory, no rescan.
    _in_thread(lambda: pcd.discover_records(directory, history, "sess"))
    assert sum(scans.values()) == 1


def test_durable_fold_in_a_worker_keeps_peer_cycle_grouping_on_a_cold_index(tmp_path):
    """M1 at the fold that caches its answer: ``mobile/durable.py`` folds in a
    worker and keeps the rows until the journal changes, so a cold-index read
    must still group the peer cycle. Failed on f6759db with zero rows."""
    from local_operator.mobile.durable import _fold

    directory, history, _record = _persisted_cycle(tmp_path)

    rows = _in_thread(lambda: _fold(history, directory))

    grouped = [row for row in rows if row.details.get("peer_cycle")]
    assert grouped, "the durable fold lost the peer-cycle grouping"


def test_peer_cycle_display_on_a_loop_arms_the_shared_refresh_without_waiting(
    tmp_path, monkeypatch
):
    """On a loop: answer not-ready at once, arm ONE single-flight refresh, and the
    next read is served from the resident index the refresh left behind."""
    _journal(tmp_path)
    directory = tmp_path / "sessions" / SID
    scans = _count_scans(monkeypatch)
    monkeypatch.setattr(pcd, "candidate_record_ids", lambda *_: ["rec1"])
    messages = [SimpleNamespace(id="a1")]

    async def drive() -> None:
        first = pcd.discover_records(directory, messages, SID)
        # Nothing scanned synchronously: the build is armed, not run.
        assert scans == {"full": 0, "incremental": 0}
        assert first is not None and first["unavailable"] == ["rec1"]
        entry = ti._IN_FLIGHT.get(ti._key(tmp_path, SID))
        assert entry is not None, "the shared single-flight refresh must be armed"
        # A second read while it runs joins it rather than starting another.
        pcd.discover_records(directory, messages, SID)
        assert ti._IN_FLIGHT.get(ti._key(tmp_path, SID)) is entry
        await entry[1]
        # Ready: ``rec1`` is not in the journal, so the page owes no proof (None),
        # which is the READY answer, not the not-ready shape.
        assert pcd.discover_records(directory, messages, SID) is None

    asyncio.run(drive())
    assert scans == {"full": 1, "incremental": 0}


def test_peer_cycle_display_uses_this_process_parse_of_a_current_publication(tmp_path, monkeypatch):
    """A publication this process already parsed (and still current) is served
    without a scan or a re-parse, even when nothing is resident."""
    _journal(tmp_path)
    assert ti.refresh_index(tmp_path, SID) is not None  # publishes + memoises
    scans = _count_scans(monkeypatch)
    monkeypatch.setattr(
        ti, "_read_raw", lambda *_: pytest.fail("a cold parse on the caller's thread")
    )
    index = ti.current_without_build(tmp_path, SID)
    assert index is not None and "a1" in index.journal_rows
    assert scans == {"full": 0, "incremental": 0}


# --- 2. one scan per journal across loops and processes ---------------------


def test_relay_and_desktop_opens_on_two_loops_start_exactly_one_scan(tmp_path, monkeypatch):
    """The desktop's loop starts the refresh; the relay's page loop (another
    thread, another loop) asks for the same journal while it runs and JOINS it."""
    _journal(tmp_path)
    calls: list[str] = []
    release = threading.Event()
    real = ti.refresh_index

    def slow_refresh(config_dir, session_id):
        calls.append(session_id)
        release.wait(30)
        return real(config_dir, session_id)

    monkeypatch.setattr(ti, "refresh_index", slow_refresh)

    desktop_loop = asyncio.new_event_loop()
    desktop_thread = threading.Thread(target=desktop_loop.run_forever, daemon=True)
    desktop_thread.start()
    try:

        async def desktop_open():
            task, started = ti.start_refresh(tmp_path, SID)
            return task, started

        desktop_task, desktop_started = asyncio.run_coroutine_threadsafe(
            desktop_open(), desktop_loop
        ).result(10)
        assert desktop_started

        async def relay_open():
            task, started = ti.start_refresh(tmp_path, SID)
            assert started is False, "the relay's loop must join, not start"
            release.set()
            return await asyncio.wait_for(task, 30)

        relay_index = asyncio.run(relay_open())
        desktop_index = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(asyncio.shield(desktop_task), 30), desktop_loop
        ).result(30)
    finally:
        release.set()
        desktop_loop.call_soon_threadsafe(desktop_loop.stop)
        desktop_thread.join(10)
        desktop_loop.close()
    assert calls == [SID]
    assert relay_index is not None and relay_index.sig == desktop_index.sig


def test_a_stopped_loops_entry_is_replaced_not_joined(tmp_path):
    """A recorded task whose loop no longer runs can never finish: start fresh."""
    _journal(tmp_path)
    dead = asyncio.new_event_loop()
    pending = dead.create_future()
    ti._IN_FLIGHT[ti._key(tmp_path, SID)] = (dead, pending)  # type: ignore[assignment]
    try:

        async def open_():
            task, started = ti.start_refresh(tmp_path, SID)
            return started, await task

        started, index = asyncio.run(open_())
    finally:
        dead.close()
    assert started and index is not None


_CHILD = """
import json, sys
from local_operator.session import transcript_index as ti
calls = {"n": 0}
real = ti._scan_full
def counting(*a, **k):
    calls["n"] += 1
    return real(*a, **k)
ti._scan_full = counting
print("ready", flush=True)
index = ti.refresh_index(sys.argv[1], sys.argv[2])
print(json.dumps({"scans": calls["n"], "rows": len(index.journal_rows)}), flush=True)
"""


def test_relay_and_serve_processes_scan_one_journal_once(tmp_path):
    """``lop serve`` and the mesh relay are separate processes. While one holds
    the session's build lock (mid-build), the other must WAIT and then take the
    published cache instead of scanning the same bytes itself."""
    _journal(tmp_path)
    lock = ti.index_path(tmp_path, SID).with_suffix(".build.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    with ti._publish_lock(lock):
        child = subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(tmp_path), SID],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "ready"
            # Give an unlocked child every chance to scan first. With the lock it
            # blocks here however long this is, so the pass is not timing-bound;
            # without it the child scans and reports ``scans: 1``.
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # This process is "the other open": it builds and publishes while
            # holding the lock, exactly as refresh_index does. (``getattr`` so the
            # test runs -- and fails on the child's scan -- against a tree
            # without the lock.)
            build = getattr(ti, "_refresh_index_locked", ti.refresh_index)
            assert build(tmp_path, SID) is not None
        except BaseException:
            child.kill()
            child.wait()
            raise
    out, _ = child.communicate(timeout=60)
    assert child.returncode == 0
    assert json.loads(out.strip().splitlines()[-1]) == {"scans": 0, "rows": 2}


def test_no_journal_takes_no_lock_and_creates_nothing(tmp_path):
    assert ti.refresh_index(tmp_path, SID) is None
    assert not ti.index_path(tmp_path, SID).parent.exists()


# --- 3. trace rows carry the pid ---------------------------------------------


def test_every_trace_row_carries_the_writing_process_pid(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(open_trace, "_PATH", str(path))
    open_trace.mark("index.refresh_start", session=SID)
    open_trace.mark("b.frame_index", 1.5, session=SID, fresh=False)
    rows: list[dict[str, Any]] = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["pid"] for row in rows] == [os.getpid(), os.getpid()]
    assert rows[1]["stage"] == "b.frame_index" and rows[1]["ms"] == 1.5


def test_trace_stays_inert_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(open_trace, "_PATH", None)
    monkeypatch.setattr(os, "getpid", lambda: pytest.fail("disabled trace read the pid"))
    open_trace.mark("index.refresh_start", session=SID)
    assert list(tmp_path.iterdir()) == []


# --- review/QA round 1 follow-ons ------------------------------------------


def _owner_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _stop(loop, thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(10)
    loop.close()


def test_a_cross_loop_join_is_visible_to_in_flight_refresh_and_arm_refresh(tmp_path, monkeypatch):
    """m1: the joiner's loop finds the proxy, and ``arm_refresh`` hands it back,
    so the phone binder's re-ask fires for a build another loop owns. Before,
    ``in_flight_refresh`` answered None after the join."""
    _journal(tmp_path)
    release = threading.Event()
    real = ti.refresh_index

    def slow(config_dir, session_id):
        release.wait(30)
        return real(config_dir, session_id)

    monkeypatch.setattr(ti, "refresh_index", slow)
    owner, thread = _owner_loop()
    try:

        async def start():
            return ti.start_refresh(tmp_path, SID)[0]

        owner_task = asyncio.run_coroutine_threadsafe(start(), owner).result(10)

        async def joiner():
            armed = ti.arm_refresh(tmp_path, SID)
            assert armed is not None and armed is not owner_task
            assert ti.in_flight_refresh(tmp_path, SID) is armed
            release.set()
            index = await asyncio.wait_for(armed, 30)
            await asyncio.sleep(0)  # let the proxy's done-callback drop its entry
            assert ti.in_flight_refresh(tmp_path, SID) is None
            return index

        assert asyncio.run(joiner()) is not None
        assert not ti._JOINED
    finally:
        release.set()
        _stop(owner, thread)


def test_an_owner_loop_teardown_is_a_failure_not_a_borrowed_cancellation(tmp_path, monkeypatch):
    """m2: the relay page loop cancels its pending tasks on stop; a joiner on
    another loop must see that as a failed refresh, not as its own cancel."""
    _journal(tmp_path)
    release = threading.Event()
    monkeypatch.setattr(ti, "refresh_index", lambda *_: release.wait(30))
    owner, thread = _owner_loop()
    try:

        async def start():
            return ti.start_refresh(tmp_path, SID)[0]

        owner_task = asyncio.run_coroutine_threadsafe(start(), owner).result(10)

        async def joiner():
            proxy, started = ti.start_refresh(tmp_path, SID)
            assert started is False
            owner.call_soon_threadsafe(owner_task.cancel)
            with pytest.raises(RuntimeError, match="owner loop shut down"):
                await asyncio.wait_for(proxy, 30)
            current = asyncio.current_task()
            assert current is not None and current.cancelling() == 0

        asyncio.run(joiner())
    finally:
        release.set()
        _stop(owner, thread)


def test_a_frozen_build_lock_holder_bounds_the_wait_then_scans_independently(tmp_path, monkeypatch):
    """QA round 1, Q1: a live holder that never releases (SIGSTOPped serve)
    blocks a waiter only up to ``_BUILD_LOCK_WAIT_S``; then the waiter scans on
    its own, the pre-lock behaviour. Both lock halves are exercised: another
    open file description holding the flock (what a second process looks like),
    and this process's thread lock."""
    import fcntl
    import time

    _journal(tmp_path)
    monkeypatch.setattr(ti, "_BUILD_LOCK_WAIT_S", 0.3)
    scans = _count_scans(monkeypatch)
    lock = ti.index_path(tmp_path, SID).with_suffix(".build.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)

    with lock.open("a") as foreign:
        fcntl.flock(foreign.fileno(), fcntl.LOCK_EX)
        began = time.monotonic()
        assert ti.refresh_index(tmp_path, SID) is not None
        waited = time.monotonic() - began
    assert 0.25 <= waited < 5, waited
    assert scans["full"] == 1

    thread_lock = ti._PUBLISH_THREADS.setdefault(str(lock), threading.Lock())
    with thread_lock:
        began = time.monotonic()
        assert _in_thread(lambda: ti.refresh_index(tmp_path, SID)) is not None
        assert time.monotonic() - began < 5

    # Released: the lock is taken normally again (no leaked half).
    with ti._build_lock(lock, 0.1) as held:
        assert held


def test_sweep_prunes_lock_files_of_deleted_sessions_only(tmp_path):
    """n1: ``.build.lock``/``.publish.lock`` of a deleted session go with its cache."""
    _journal(tmp_path)
    cache = ti._index_dir(tmp_path)
    cache.mkdir(parents=True, exist_ok=True)
    names = [f"{owner}.{kind}.lock" for owner in (SID, "gone0000") for kind in ("build", "publish")]
    for name in names:
        (cache / name).touch()

    ti._sweep_missing(tmp_path)

    assert sorted(p.name for p in cache.glob("*.lock")) == [
        f"{SID}.build.lock",
        f"{SID}.publish.lock",
    ]
