"""Acquire diagnostics are per-call observations, never attach policy.

Synthetic clocks isolate phase attribution from a busy host's scheduling. The
real bridge lock, facade construction seam, wait and release still execute;
only birth/cold/attach dependencies are controlled. No live session is opened.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

from local_operator.server.utils import desktop_sessions as module


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=10.0)
    rows: list[dict[str, Any]] = []
    handles: list[asyncio.Handle] = []
    callbacks = []
    loop = asyncio.get_running_loop()

    def call_soon(callback):
        # Keep the actual one-shot handle so cancellation is observable; invoke
        # its callback explicitly when the test injects scheduling starvation.
        handle = loop.call_soon(callback)
        handles.append(handle)
        callbacks.append(callback)
        return handle

    class AsyncioProxy:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        def get_running_loop(self):
            return SimpleNamespace(call_soon=call_soon)

    def mark(stage, ms=None, **fields):
        rows.append({"stage": stage, "ms": ms, **fields})

    monkeypatch.setattr(module, "asyncio", AsyncioProxy())
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(module.open_trace, "enabled", lambda: True)
    monkeypatch.setattr(module.open_trace, "mark", mark)
    bridge = module.DesktopSessionBridge(tmp_path, "synthetic-acquire", str(tmp_path))
    # Do not start attention I/O; the diagnostics must not alter its ownership.
    bridge.attention_task = asyncio.current_task()
    return SimpleNamespace(
        clock=clock, rows=rows, handles=handles, callbacks=callbacks, bridge=bridge
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["lock", "birth", "cold", "attach", "loop"])
async def test_injected_phases_are_distinct(rig, monkeypatch, phase):
    bridge, clock = rig.bridge, rig.clock
    remote = SimpleNamespace(
        is_cold=True,
        set_refresh_callback=lambda callback: None,
        set_local_cwd_callback=lambda callback: None,
        subscribe=lambda callback: lambda: None,
        subscribe_frontend=lambda callback: SimpleNamespace(unsubscribe=lambda: None),
    )

    async def birth(*args):
        if phase == "birth":
            clock.now += 1
        return None

    async def cold(*args, **kwargs):
        assert kwargs["deferred_history"] is True
        # The cold summary must join this exact acquisition, not just another
        # overlapping open of the same session.
        assert kwargs["_trace_span"] == rig.rows[0]["acquire_id"]
        if phase == "cold":
            clock.now += 1
        if phase == "loop":
            # Model synchronous loop blocking before its queued witness runs.
            clock.now += 1
            rig.callbacks[0]()
        return remote

    monkeypatch.setattr(module.asyncio, "to_thread", birth)
    monkeypatch.setattr(module.AttachedSession, "cold", cold)
    gate = asyncio.Event()

    async def attach():
        await gate.wait()
        return False

    task = asyncio.create_task(attach())
    monkeypatch.setattr(bridge, "_start_read_attach", lambda remote: task)
    real_wait = asyncio.wait

    async def wait(tasks, *, timeout):
        assert tasks == {task}
        assert timeout == module.READ_FIRST_FRAME_GRACE_S
        if phase == "attach":
            clock.now += 1
        gate.set()
        return await real_wait(tasks, timeout=timeout)

    monkeypatch.setattr(module.asyncio, "wait", wait)
    if phase == "lock":
        await bridge.lock.acquire()
    acquiring = asyncio.create_task(bridge.acquire(read=True))
    if phase == "lock":
        await asyncio.sleep(0)
        assert not acquiring.done()
        clock.now += 1
        bridge.lock.release()
    assert await acquiring is remote
    assert bridge.users == 1
    end = rig.rows[-1]
    assert end["stage"] == "a.acquire"
    assert end["completed"] is True
    assert end["attach_landed"] is True
    assert end["attach_task_completed"] is True  # completion != False result
    assert end["selected_grace_ms"] == module.READ_FIRST_FRAME_GRACE_S * 1000
    expected = {
        "lock_wait_ms": phase == "lock",
        "birth_await_ms": phase == "birth",
        "cold_ms": phase in ("cold", "loop"),
        "grace_elapsed_ms": phase == "attach",
    }
    for field, injected in expected.items():
        assert end[field] == (1000 if injected else 0)
    assert end["facade_ms"] == (1000 if phase in ("birth", "cold", "loop") else 0)
    assert end["lock_held_ms"] == end["facade_ms"]
    if phase == "loop":
        assert end["loop_queue_delay_ms"] == 1000
    assert rig.rows[0]["acquire_id"] == end["acquire_id"]
    assert rig.rows[0]["session"] == end["session"] == bridge.session_id
    assert all(handle.cancelled() for handle in rig.handles)
    monkeypatch.setattr(type(bridge), "dwelling", property(lambda self: True))
    await bridge.release()
    assert bridge.users == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", [False, True])
@pytest.mark.parametrize("outcome", ["pending", "false", "error", "cancelled"])
async def test_grace_branch_completion_and_task_ownership(rig, monkeypatch, peer, outcome):
    bridge = rig.bridge
    remote = SimpleNamespace(is_cold=True)
    bridge.remote = remote
    if peer:
        bridge.remote_row = {"synthetic": True}
    gate = asyncio.Event()

    async def attach():
        await gate.wait()
        if outcome == "error":
            raise ValueError("synthetic failure")
        return False

    task = asyncio.create_task(attach())
    if outcome == "cancelled":
        task.cancel()
    if outcome != "pending":
        gate.set()
        await asyncio.sleep(0)
    monkeypatch.setattr(bridge, "_start_read_attach", lambda remote: task)
    real_wait = asyncio.wait

    async def wait(tasks, *, timeout):
        expected = (
            module.REMOTE_READ_FIRST_FRAME_GRACE_S if peer else module.READ_FIRST_FRAME_GRACE_S
        )
        assert timeout == expected
        # No clock ceiling: model timeout with a ready-loop wait while checking
        # the product passed the unchanged local/peer policy to asyncio.wait.
        rig.clock.now += timeout
        return await real_wait(tasks, timeout=0)

    monkeypatch.setattr(module.asyncio, "wait", wait)
    try:
        assert await bridge.acquire(read=True) is remote
        row = rig.rows[-1]
        assert row["remote_branch"] is peer
        assert row["grace_elapsed_ms"] == pytest.approx(row["selected_grace_ms"])
        assert row["attach_landed"] is (outcome != "pending")
        assert row["attach_task_completed"] is (outcome != "pending")
        assert (task in bridge.read_attach_outran) is (outcome == "pending")
        assert task.cancelled() is (outcome == "cancelled")
        assert all(handle.cancelled() for handle in rig.handles)
    finally:
        gate.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_disabled_trace_has_no_witness_metadata_or_marks(rig, monkeypatch):
    monkeypatch.setattr(module.open_trace, "enabled", lambda: False)
    remote = SimpleNamespace(is_cold=False)

    async def facade():  # no diagnostic keyword when disabled
        return remote

    monkeypatch.setattr(rig.bridge, "_ensure_facade", facade)
    assert await rig.bridge.acquire(read=True) is remote
    assert rig.bridge.users == 1
    assert rig.handles == rig.rows == []


@pytest.mark.asyncio
async def test_concurrent_calls_have_distinct_ids_and_local_phases(rig, monkeypatch):
    rig.bridge.remote = SimpleNamespace(is_cold=False)
    await asyncio.gather(rig.bridge.acquire(read=True), rig.bridge.acquire(read=True))
    rows = [row for row in rig.rows if row["stage"] == "a.acquire"]
    assert len({row["acquire_id"] for row in rows}) == 2
    assert all("birth_await_ms" not in row and "cold_ms" not in row for row in rows)
    assert rig.bridge.users == 2
    assert all(handle.cancelled() for handle in rig.handles)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["facade", "lock_cancel", "wait_cancel"])
async def test_failure_preserves_refcounts_and_cancels_witness(rig, monkeypatch, failure):
    bridge = rig.bridge
    monkeypatch.setattr(type(bridge), "dwelling", property(lambda self: True))
    if failure == "facade":

        async def fail(**kwargs):
            raise ValueError("synthetic facade failure")

        monkeypatch.setattr(bridge, "_ensure_facade", fail)
        with pytest.raises(ValueError, match="synthetic facade failure"):
            await bridge.acquire(read=True)
    else:
        bridge.remote = SimpleNamespace(is_cold=True)
        entered = asyncio.Event()
        gate = asyncio.Event()

        async def attach():
            entered.set()
            await gate.wait()
            return False

        task = asyncio.create_task(attach())
        monkeypatch.setattr(bridge, "_start_read_attach", lambda remote: task)
        if failure == "lock_cancel":
            await bridge.lock.acquire()
        acquiring = asyncio.create_task(bridge.acquire(read=True))
        if failure == "wait_cancel":
            await entered.wait()
            # attach has started, and acquire is suspended in its real wait.
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0)
        acquiring.cancel()
        with pytest.raises(asyncio.CancelledError):
            await acquiring
        if failure == "lock_cancel":
            bridge.lock.release()
        assert not task.cancelled()
        gate.set()
        await task
    assert bridge.users == 0
    assert rig.rows[-1]["completed"] is False
    assert all(handle.cancelled() for handle in rig.handles)
