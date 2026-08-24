"""The child's self-reaper: the truth table, the grace window, and the
clean-exit ordering."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.mobile import child as child_mod
from local_operator.mobile.child import _clean_exit, _reaper, _should_exit


class FakeRegistrant:
    def __init__(self, *, supported: bool = False, watchers: int = 0, attaches: int = 0) -> None:
        self.watch_supported = supported
        self.phone_watchers = watchers
        self._attaches = attaches
        self.closed = False

    def attach_clients(self) -> int:
        return self._attaches

    async def aclose(self) -> None:
        self.closed = True


class FakeHandle:
    def __init__(self, *, busy: bool = False) -> None:
        self._busy = busy
        self.disposed = False
        self.denied = False
        self.dispose_order: list[str] = []

    def is_busy(self) -> bool:
        return self._busy

    def _deny_pending_gates(self) -> None:
        self.denied = True

    async def dispose(self) -> None:
        self.dispose_order.append("dispose")
        self.disposed = True


@pytest.mark.parametrize(
    "reg",
    [
        FakeRegistrant(supported=False),
        FakeRegistrant(supported=True, watchers=1),
        FakeRegistrant(supported=True, watchers=5, attaches=3),
    ],
)
def test_viewer_counts_never_pin_idle_host(reg: FakeRegistrant) -> None:
    assert _should_exit(FakeHandle(), reg) is True


def test_active_work_holds() -> None:
    assert _should_exit(FakeHandle(busy=True), FakeRegistrant()) is False


@pytest.mark.asyncio
async def test_clean_exit_orders_gates_dispose_unpublish(monkeypatch) -> None:
    order: list[str] = []
    handle = FakeHandle()
    reg = FakeRegistrant(supported=True)

    async def dispose() -> None:
        order.append("dispose")

    async def aclose() -> None:
        order.append("unpublish")

    handle._deny_pending_gates = lambda: order.append("deny")  # type: ignore[method-assign]
    handle.dispose = dispose  # type: ignore[method-assign]
    reg.aclose = aclose  # type: ignore[method-assign]
    await _clean_exit(handle, reg)
    assert order == ["dispose", "unpublish"]


@pytest.mark.asyncio
async def test_grace_elapses_then_clean_exit(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.05)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.2")
    reg = FakeRegistrant(supported=True)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        if stop.is_set():
            break
        await asyncio.sleep(0.05)
    assert stop.is_set()
    assert handle.disposed and not handle.denied and reg.closed
    await task


@pytest.mark.asyncio
async def test_viewers_do_not_change_idle_timing(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.08")
    elapsed: list[float] = []
    for watchers, attaches in ((0, 0), (1, 0), (5, 3)):
        reg = FakeRegistrant(supported=bool(watchers), watchers=watchers, attaches=attaches)
        handle = FakeHandle()
        stop = asyncio.Event()
        started = asyncio.get_running_loop().time()
        await _reaper(handle, reg, stop)
        elapsed.append(asyncio.get_running_loop().time() - started)
    assert max(elapsed) - min(elapsed) < 0.04


@pytest.mark.asyncio
async def test_busy_session_defers_grace_start(monkeypatch) -> None:
    """A turn mid-flight when the last front end leaves must NOT start the
    clock; grace begins at turn end and outlives the turn by construction."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.05)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.2")
    reg = FakeRegistrant(supported=True)
    handle = FakeHandle(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.5)  # past the whole grace had it started at t=0
    assert not stop.is_set()
    handle._busy = False  # the turn ends NOW
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        if stop.is_set():
            break
        await asyncio.sleep(0.05)
    assert stop.is_set()
    await task


@pytest.mark.asyncio
async def test_new_activity_resets_idle_drain(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.12")
    reg = FakeRegistrant(watchers=3, attaches=2)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.08)
    handle._busy = True
    await asyncio.sleep(0.1)
    handle._busy = False
    await asyncio.sleep(0.08)
    assert not stop.is_set()
    await task
    assert stop.is_set()


def test_grace_env_override_and_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LOP_SESSION_GRACE_S", raising=False)
    assert child_mod._grace_seconds() == child_mod.DEFAULT_GRACE_S == 3.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "10")
    assert child_mod._grace_seconds() == 10.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "not-a-number")
    assert child_mod._grace_seconds() == 3.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "-5")
    assert child_mod._grace_seconds() == 3.0
