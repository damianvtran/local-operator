"""Pin the attribution and lifecycle of the acquire-scoped worker trace."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ParamSpec, TypeVar

import pytest

from local_operator.session import open_trace
from local_operator.session.open_trace import ColdOpenTrace, ColdWorkerPhase

_WAIT = 10  # Deadlock backstop, not a performance assertion.
_ID = "0123456789abcdef0123456789abcdef"
_SECRET = "synthetic-data-must-not-enter-the-worker-row"
_P = ParamSpec("_P")
_T = TypeVar("_T")
_PHASE = ColdWorkerPhase.MODEL_SYNTHESIS


def _hold_until(event: threading.Event) -> None:
    assert event.wait(_WAIT), "worker test barrier did not release"


@pytest.fixture
def output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(open_trace, "_PATH", str(path))
    return path


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class _GatedPool(ThreadPoolExecutor):
    """One worker, with independent witnesses for enqueue and Future delivery."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.watch = False
        self.queued = asyncio.Event()
        self.done = threading.Event()
        self.target_running: Callable[[], bool] | None = None

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        future = super().submit(fn, *args, **kwargs)
        if self.watch:
            self.watch = False
            self.target_running = future.running
            future.add_done_callback(lambda _: self.done.set())
            self.queued.set()
        return future


async def _shape(output: Path, *, cpu_work: bool) -> tuple[dict[str, Any], dict[str, int]]:
    before = _rows(output) if output.exists() else []
    loop = asyncio.get_running_loop()
    context = ContextVar("probe-input", default="not copied")
    context.set(_SECRET)
    loop_thread = threading.get_ident()
    release_pool, release_work = threading.Event(), threading.Event()
    blocker_entered, target_entered = asyncio.Event(), asyncio.Event()
    points: dict[str, int] = {}
    trace = ColdOpenTrace("do-not-include-session", _ID)

    def blocker() -> None:
        loop.call_soon_threadsafe(blocker_entered.set)
        _hold_until(release_pool)

    def work(value: str, *, phase: str, func: str) -> object:
        assert threading.get_ident() != loop_thread
        assert value == phase == func == context.get() == _SECRET
        points["called"] = time.monotonic_ns()
        loop.call_soon_threadsafe(target_entered.set)
        _hold_until(release_work)
        cpu_start = time.thread_time_ns()
        if cpu_work:
            result = sum(i * i for i in range(1_000_000))
        else:
            time.sleep(0.03)  # Deliberate wall-only work, not a readiness wait.
            result = 17
        points["body_cpu"] = time.thread_time_ns() - cpu_start
        return result

    # Install a private default executor on only this test's event loop. The
    # wrapper itself must keep using that default; no product pool is changed.
    pool = _GatedPool()
    loop.set_default_executor(pool)
    prior = asyncio.create_task(asyncio.to_thread(blocker))
    target: asyncio.Task[object] | None = None
    try:
        await asyncio.wait_for(blocker_entered.wait(), _WAIT)
        pool.watch = True
        target = asyncio.create_task(
            trace.worker(_PHASE, work, _SECRET, phase=_SECRET, func=_SECRET)
        )
        await asyncio.wait_for(pool.queued.wait(), _WAIT)
        assert pool.target_running is not None and not pool.target_running()
        points["queue_release"] = time.monotonic_ns()
        release_pool.set()
        await asyncio.wait_for(target_entered.wait(), _WAIT)
        # Hold this loop until the real concurrent future completes. A sleep
        # is not needed: readiness comes from the executor, not the loop.
        release_work.set()
        _hold_until(pool.done)
        points["loop_release"] = time.monotonic_ns()
        assert not target.done()
        expected = 333_332_833_333_500_000 if cpu_work else 17
        assert await asyncio.wait_for(target, _WAIT) == expected
    finally:
        release_pool.set()
        release_work.set()
        await prior
        if target is not None and not target.done():
            await target
        pool.shutdown()

    # Nothing writes from the worker or at the individual await. Only finish
    # appends one bounded row on the acquire's caller.
    assert (_rows(output) if output.exists() else []) == before
    trace.finish("ok")
    written = _rows(output)[len(before) :]
    assert [row["stage"] for row in written] == ["cold.summary", "cold.worker"]
    return written[1], points


@pytest.mark.asyncio
async def test_worker_times_discriminate_queue_work_and_caller_delivery(output: Path) -> None:
    cpu, cpu_points = await _shape(output, cpu_work=True)
    wait, wait_points = await _shape(output, cpu_work=False)
    for row, points in ((cpu, cpu_points), (wait, wait_points)):
        assert set(row) == {
            "at",
            "pid",
            "stage",
            "acquire_id",
            "phase",
            "submit_ns",
            "worker_start_ns",
            "worker_end_ns",
            "resume_ns",
            "worker_cpu_ns",
        }
        assert (row["stage"], row["acquire_id"], row["phase"]) == ("cold.worker", _ID, _PHASE)
        assert isinstance(row["at"], float) and type(row["pid"]) is int
        assert all(type(value) is int for key, value in row.items() if key.endswith("_ns"))
        assert (
            row["submit_ns"]
            < points["queue_release"]
            <= row["worker_start_ns"]
            <= points["called"]
            <= row["worker_end_ns"]
            <= points["loop_release"]
            <= row["resume_ns"]
        )
        assert row["worker_cpu_ns"] >= points["body_cpu"] >= 0
    # Compare controlled workloads, not a CPU/wall ratio: host scheduling can
    # stretch even the known CPU-heavy body's wall time considerably.
    assert cpu_points["body_cpu"] > wait["worker_cpu_ns"]
    assert wait["worker_end_ns"] - wait["worker_start_ns"] > wait["worker_cpu_ns"]
    assert _SECRET not in output.read_text()
    assert all("do-not-include-session" not in json.dumps(row) for row in (cpu, wait))


@pytest.mark.asyncio
async def test_worker_preserves_error_and_running_worker_cancellation(output: Path) -> None:
    trace = ColdOpenTrace("synthetic", _ID)
    error = RuntimeError(_SECRET)

    def fail() -> None:
        raise error

    with pytest.raises(RuntimeError) as caught:
        await trace.worker(_PHASE, fail)
    assert caught.value is error
    trace.finish("failed")
    assert [r["stage"] for r in _rows(output)] == ["cold.summary", "cold.worker"]
    assert _SECRET not in output.read_text()

    loop = asyncio.get_running_loop()
    entered, exited, release = asyncio.Event(), asyncio.Event(), threading.Event()
    cancelled = ColdOpenTrace("synthetic", "different-acquire")

    def running() -> None:
        loop.call_soon_threadsafe(entered.set)
        _hold_until(release)
        loop.call_soon_threadsafe(exited.set)

    task = asyncio.create_task(cancelled.worker(_PHASE, running))
    try:
        await asyncio.wait_for(entered.wait(), _WAIT)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not exited.is_set(), "cancel must not join or release a running worker"
        cancelled.finish("cancelled")
    finally:
        release.set()
        await asyncio.wait_for(exited.wait(), _WAIT)
    assert [r["stage"] for r in _rows(output)] == ["cold.summary", "cold.worker", "cold.summary"]


@pytest.mark.asyncio
async def test_worker_trace_fails_open_and_does_not_sample_unscoped_opens(
    output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad_clock() -> int:
        raise OSError("synthetic dead clock")

    for clock in ("monotonic_ns", "thread_time_ns"):
        trace = ColdOpenTrace("synthetic", _ID)
        proxy = SimpleNamespace(
            time=time.time,
            monotonic=time.monotonic,
            monotonic_ns=time.monotonic_ns,
            thread_time_ns=time.thread_time_ns,
        )
        setattr(proxy, clock, bad_clock)
        with monkeypatch.context() as m:
            m.setattr(open_trace, "time", proxy)
            assert await trace.worker(_PHASE, lambda: 23) == 23
            trace.finish("ok")
    unscoped = ColdOpenTrace("synthetic")
    assert await unscoped.worker(_PHASE, lambda: 27) == 27
    unscoped.finish("ok")
    assert [r["stage"] for r in _rows(output)] == ["cold.summary"] * 3

    real_mark = open_trace.mark

    def broken_sink(stage: str, *args: Any, **kwargs: Any) -> None:
        if stage == "cold.worker":
            raise OSError("synthetic unwritable sink")
        real_mark(stage, *args, **kwargs)

    trace = ColdOpenTrace("synthetic", _ID)
    monkeypatch.setattr(open_trace, "mark", broken_sink)
    assert await trace.worker(_PHASE, lambda: 29) == 29
    trace.finish("ok")
    assert [r["stage"] for r in _rows(output)] == ["cold.summary"] * 4
