"""The child's self-reaper: the truth table, the grace window, and the
clean-exit ordering."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.mobile import child as child_mod
from local_operator.mobile.child import _clean_exit, _reaper, _should_exit


class FakeRegistrant:
    def __init__(
        self, *, supported: bool = False, watchers: int = 0, attaches: int = 0
    ) -> None:
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

    def is_busy(self) -> bool:
        return self._busy

    def _deny_pending_gates(self) -> None:
        self.denied = True

    async def dispose(self) -> None:
        self.disposed = True


def test_unknown_watchers_never_exit() -> None:
    """The mixed-version guard: an old daemon never sends watch/unwatch, so
    watchers are UNKNOWN and _should_exit must read as present."""
    reg = FakeRegistrant(supported=False, watchers=0)
    assert _should_exit(FakeHandle(), reg) is False


def test_known_zero_watchers_idle_session_exits() -> None:
    reg = FakeRegistrant(supported=True, watchers=0)
    assert _should_exit(FakeHandle(), reg) is True


def test_present_watchers_hold() -> None:
    reg = FakeRegistrant(supported=True, watchers=1)
    assert _should_exit(FakeHandle(), reg) is False


def test_attach_client_holds() -> None:
    reg = FakeRegistrant(supported=True, watchers=0, attaches=1)
    assert _should_exit(FakeHandle(), reg) is False


def test_busy_session_holds() -> None:
    reg = FakeRegistrant(supported=True, watchers=0)
    assert _should_exit(FakeHandle(busy=True), reg) is False


def test_latch_never_resets() -> None:
    """Once watch-capable, silence means zero — but the latch itself must
    survive a watchers count that rises again (it gates on the flag, not the
    count's history)."""
    reg = FakeRegistrant(supported=True, watchers=2)
    assert _should_exit(FakeHandle(), reg) is False
    reg.phone_watchers = 0
    assert _should_exit(FakeHandle(), reg) is True


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
    assert order == ["deny", "dispose", "unpublish"]


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
    assert handle.disposed and handle.denied and reg.closed
    await task


@pytest.mark.asyncio
async def test_front_end_returning_mid_grace_cancels(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.05)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.5")
    reg = FakeRegistrant(supported=True)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    # Inside the grace window, a phone comes back.
    await asyncio.sleep(0.15)
    reg.phone_watchers = 1
    await asyncio.sleep(0.8)  # well past the original grace
    assert not stop.is_set()
    assert not handle.disposed
    task.cancel()


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


def test_grace_env_override_and_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LOP_SESSION_GRACE_S", raising=False)
    assert child_mod._grace_seconds() == child_mod.DEFAULT_GRACE_S == 120.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "10")
    assert child_mod._grace_seconds() == 10.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "not-a-number")
    assert child_mod._grace_seconds() == 120.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "-5")
    assert child_mod._grace_seconds() == 120.0
