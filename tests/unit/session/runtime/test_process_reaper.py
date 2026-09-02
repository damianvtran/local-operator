"""The runtime's self-reaper: the three-term residency predicate (design
§6.1), the grace window, and the clean-exit ordering."""

from __future__ import annotations

import asyncio
import time

import pytest

from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.process import _clean_exit, _reaper, _should_exit


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
    def __init__(self, *, busy: bool = False, next_wake_ms: int | None = None) -> None:
        self._busy = busy
        self._next_wake_ms = next_wake_ms
        self.disposed = False
        self.denied = False
        self.dispose_order: list[str] = []

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return self._next_wake_ms

    def _deny_pending_gates(self) -> None:
        self.denied = True

    async def dispose(self) -> None:
        self.dispose_order.append("dispose")
        self.disposed = True


def _now_ms() -> int:
    return int(time.time() * 1000)


@pytest.mark.parametrize(
    "reg",
    [
        FakeRegistrant(supported=False),
        FakeRegistrant(supported=True, watchers=1),
        FakeRegistrant(supported=True, watchers=5),
    ],
)
def test_daemon_and_phone_watchers_never_pin_idle_runtime(reg: FakeRegistrant) -> None:
    # Term 3 counts INTERACTIVE viewers only. The phone daemon's adoption
    # connection and its SSE watcher count are daemon-class signals: the
    # daemon adopts every session on the machine, so if they held runtimes
    # warm nothing would ever exit.
    assert _should_exit(FakeHandle(), reg) is True


@pytest.mark.parametrize("attaches", [1, 3])
def test_attached_viewer_holds_idle_runtime(attaches: int) -> None:
    # Term 3: a follower terminal (ClientKind "attach") is the user's
    # attention, and the next thing they do is type — hold the process warm.
    assert _should_exit(FakeHandle(), FakeRegistrant(attaches=attaches)) is False


def test_active_work_holds() -> None:
    assert _should_exit(FakeHandle(busy=True), FakeRegistrant()) is False


def test_wake_inside_warm_window_holds() -> None:
    # Term 2: a wake due in 60 s (the tightest recurrence the wake layer
    # allows) is inside WARM_WINDOW_S, so the runtime stays to fire it itself
    # rather than exiting and cold-starting a minute later.
    assert child_mod.WARM_WINDOW_S > 60.0
    handle = FakeHandle(next_wake_ms=_now_ms() + 60_000)
    assert _should_exit(handle, FakeRegistrant()) is False
    overdue = FakeHandle(next_wake_ms=_now_ms() - 1_000)
    assert _should_exit(overdue, FakeRegistrant()) is False


def test_wake_beyond_warm_window_does_not_hold() -> None:
    # A wake an hour out is cheaper to leave to a cold spawn than to hold
    # ~283 MB for.
    handle = FakeHandle(next_wake_ms=_now_ms() + 3_600_000)
    assert _should_exit(handle, FakeRegistrant()) is True


def test_warm_window_exceeds_min_wake_interval() -> None:
    # The constant pairs with MIN_WAKE_INTERVAL_MS: the window must be wider
    # than the tightest allowed recurrence or a 1-minute wake thrashes
    # exit → spawn → exit forever.
    from local_operator.harness.wake import MIN_WAKE_INTERVAL_MS

    assert child_mod.WARM_WINDOW_S * 1000 > MIN_WAKE_INTERVAL_MS


def test_predicate_tolerates_reduced_handles_and_runtimes() -> None:
    # Older hosts and reduced test doubles lack the accessors; each missing
    # or broken term reads as "does not hold" rather than crashing the reaper
    # or pinning the process.
    class Bare:
        pass

    class Broken:
        def is_busy(self) -> bool:
            return False

        def next_wake_due_at(self) -> int:
            raise RuntimeError("scheduler gone")

        def attach_clients(self) -> int:
            raise RuntimeError("registry gone")

    assert _should_exit(Bare(), Bare()) is True
    assert _should_exit(Broken(), Broken()) is True


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
async def test_phone_watchers_do_not_change_idle_timing(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.08")
    elapsed: list[float] = []
    for watchers in (0, 1, 5):
        reg = FakeRegistrant(supported=bool(watchers), watchers=watchers)
        handle = FakeHandle()
        stop = asyncio.Event()
        started = asyncio.get_running_loop().time()
        await _reaper(handle, reg, stop)
        elapsed.append(asyncio.get_running_loop().time() - started)
    assert max(elapsed) - min(elapsed) < 0.04


@pytest.mark.asyncio
async def test_attached_viewer_holds_then_release_starts_drain(monkeypatch) -> None:
    """The TUI-closes-while-idle path (design §6.1): the attach count drops
    to zero, the drain starts THEN, and the runtime exits one grace later."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.15")
    reg = FakeRegistrant(attaches=1)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.4)  # well past the grace had the viewer not held it
    assert not stop.is_set() and not handle.disposed
    reg._attaches = 0  # the terminal closed
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        if stop.is_set():
            break
        await asyncio.sleep(0.02)
    assert stop.is_set() and handle.disposed and reg.closed
    await task


@pytest.mark.asyncio
async def test_viewer_attaching_mid_drain_cancels_it(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.15")
    reg = FakeRegistrant()
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.08)  # inside the drain
    reg._attaches = 1
    await asyncio.sleep(0.3)
    assert not stop.is_set() and not handle.disposed
    reg._attaches = 0
    await task
    assert stop.is_set()


@pytest.mark.asyncio
async def test_wake_in_window_holds_until_it_passes(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.1")
    reg = FakeRegistrant()
    handle = FakeHandle(next_wake_ms=_now_ms() + 30_000)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.3)
    assert not stop.is_set()
    handle._next_wake_ms = None  # the wake was cancelled (or retired)
    await task
    assert stop.is_set() and handle.disposed


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
    # Phone watchers only: an attach client would (correctly) hold forever.
    reg = FakeRegistrant(supported=True, watchers=3)
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
