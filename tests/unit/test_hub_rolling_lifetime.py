"""Actual worker lifetime negatives; no real profile service/account is invoked.

A cancelled async waiter is deliberately NOT the completion oracle. The tiny
thread's own event/side effect discriminates real completion from lock/task state.
These regressions do not replace normal API/controller/runtime acceptance.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from local_operator.hub_sync.runner import HubSyncRunner


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_actual_offload_owned(tmp_path):
    runner = HubSyncRunner(config_manager=SimpleNamespace(config_dir=tmp_path))
    started = threading.Event()
    finish = threading.Event()

    def work():
        started.set()
        assert finish.wait(3)
        (tmp_path / "actual-worker-completed").write_text("done")
        return "done"

    caller = asyncio.create_task(runner._offload(work))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not runner._lock.locked()
        assert runner.reload_blocker() is not None
        assert not (tmp_path / "actual-worker-completed").exists()
    finally:
        finish.set()
        await asyncio.wait_for(asyncio.gather(*runner._offloads), 3)
    assert (tmp_path / "actual-worker-completed").read_text() == "done"
    assert runner.reload_blocker() is None


@pytest.mark.asyncio
async def test_manual_disconnect_preserves_dispatch_and_reload_parks_next(tmp_path):
    runner = HubSyncRunner(
        config_manager=SimpleNamespace(config_dir=tmp_path), build_ctx=lambda: object()
    )
    started = threading.Event()
    finish = threading.Event()

    def apply(_ctx):
        started.set()
        assert finish.wait(3)
        (tmp_path / "manual-side-effect").write_text("applied")
        return "applied"

    caller = asyncio.create_task(runner.run_exclusive(apply))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        runner.begin_reload()
        assert runner.reload_blocker() is not None
        next_call = asyncio.create_task(runner.run_exclusive(lambda _ctx: "next"))
        await asyncio.sleep(0)
        assert not next_call.done()
        finish.set()
        await asyncio.wait_for(asyncio.gather(*runner._dispatches), 3)
        assert (tmp_path / "manual-side-effect").read_text() == "applied"
        runner.end_reload()
        assert await asyncio.wait_for(next_call, 3) == "next"
    finally:
        finish.set()
        runner.end_reload()
        if runner._dispatches:
            await asyncio.wait_for(asyncio.gather(*runner._dispatches), 3)
