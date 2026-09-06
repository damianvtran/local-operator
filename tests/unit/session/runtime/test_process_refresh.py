"""The runtime's self-refresh: an idle runtime whose build has moved under it
retires, announced, so the next engage runs the build on disk
(design-runtime-autorefresh §3.2/§3.4).

Same fakes as ``test_process_reaper.py``. The predicate under test is
``OwnedSessionHandle.may_refresh`` reached through ``process._should_refresh``;
the fakes carry a ``may_refresh`` of their own so the reaper's branch can be
driven without a Session.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.process import (
    _build_changed,
    _reaper,
    _should_exit,
    _should_refresh,
)
from local_operator.update import BuildStamp

OLD = BuildStamp(version="0.49.8", source_ref="46a4e9b1234567")
NEW = BuildStamp(version="0.49.9", source_ref="f4a70b991234567")


class FakeRegistrant:
    def __init__(self, *, attaches: int = 0, boot: BuildStamp | None = OLD) -> None:
        self.watch_supported = False
        self.phone_watchers = 0
        self._attaches = attaches
        self.closed = False
        self._boot_build = boot
        self.retiring: list[tuple[str, str]] = []

    def attach_clients(self) -> int:
        return self._attaches

    async def announce_retiring(self, reason: str, *, to: str = "") -> None:
        self.retiring.append((reason, to))

    async def aclose(self) -> None:
        self.closed = True


class FakeHandle:
    """Carries the SAME two-term predicate the real handle does."""

    def __init__(self, *, busy: bool = False, next_wake_ms: int | None = None) -> None:
        self._busy = busy
        self._next_wake_ms = next_wake_ms
        self.disposed = False
        self.probes = 0
        self.on_probe: Any = None

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return self._next_wake_ms

    def may_refresh(self) -> str:
        self.probes += 1
        if self.on_probe is not None:
            self.on_probe(self.probes)
        if self.is_busy():
            return "busy"
        if child_mod._wake_within_window(self):
            return "wake due within the warm window"
        return ""

    def _deny_pending_gates(self) -> None:
        pass

    async def dispose(self) -> None:
        self.disposed = True


def _now_ms() -> int:
    return int(time.time() * 1000)


@pytest.fixture
def disk(monkeypatch):
    """Control what ``installed_build``/``build_marker_age_s`` report."""
    state = {"build": NEW, "age": 999.0}
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: state["build"])
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: state["age"])
    monkeypatch.delenv("LOP_BUILD_SETTLE_S", raising=False)
    monkeypatch.delenv("LOP_BUILD_STAGGER_S", raising=False)
    monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)
    return state


# -- 4. the predicate -----------------------------------------------------------


def test_busy_never_refreshes(disk) -> None:
    assert _should_refresh(FakeHandle(busy=True), OLD) is None


def test_wake_inside_warm_window_never_refreshes(disk) -> None:
    handle = FakeHandle(next_wake_ms=_now_ms() + 30_000)
    assert _should_refresh(handle, OLD) is None


def test_wake_beyond_warm_window_refreshes(disk) -> None:
    handle = FakeHandle(next_wake_ms=_now_ms() + 3_600_000)
    assert _should_refresh(handle, OLD) == NEW


def test_attached_viewer_does_not_hold_a_refresh(disk) -> None:
    """THE operator's rule, pinned: term 3 of ``_should_exit`` holds a quiet
    exit, and must NOT hold a refresh — that is exactly what kept a
    five-hour-stale runtime resident while a `--resume` looked at it."""
    handle = FakeHandle()
    reg = FakeRegistrant(attaches=2)
    assert _should_exit(handle, reg) is False
    assert _should_refresh(handle, OLD) == NEW


def test_pristine_runtime_refreshes(disk) -> None:
    # Pristineness is not a term at all; an idle pristine runtime refreshes.
    assert _should_refresh(FakeHandle(), OLD) == NEW


def test_a_handle_without_the_probe_never_refreshes(disk) -> None:
    class Bare:
        pass

    class Broken:
        def may_refresh(self) -> str:
            raise RuntimeError("scheduler gone")

    assert _should_refresh(Bare(), OLD) is None
    assert _should_refresh(Broken(), OLD) is None


# -- 5. build change + settle ---------------------------------------------------


def test_same_stamp_is_no_change(disk) -> None:
    disk["build"] = OLD
    assert _build_changed(OLD) is None


def test_no_boot_stamp_is_no_change(disk) -> None:
    assert _build_changed(None) is None


def test_different_ref_inside_settle_waits(disk) -> None:
    disk["age"] = child_mod.BUILD_SETTLE_S / 2
    assert _build_changed(OLD) is None


def test_different_ref_past_settle_changes(disk) -> None:
    disk["age"] = child_mod.BUILD_SETTLE_S + 1
    assert _build_changed(OLD) == NEW


def test_unknown_marker_age_waits(disk) -> None:
    disk["age"] = None
    assert _build_changed(OLD) is None


def test_version_only_change_without_marker(disk) -> None:
    # A PyPI install: no ``.lop-source``; BuildStamp compares on version.
    disk["build"] = BuildStamp(version="0.50.0")
    assert _build_changed(BuildStamp(version="0.49.9")) == BuildStamp(version="0.50.0")


def test_unreadable_stamp_is_no_change(disk, monkeypatch) -> None:
    def boom(*_a, **_k):
        raise OSError("dist-info gone")

    monkeypatch.setattr(update_mod, "installed_build", boom)
    assert _build_changed(OLD) is None


def test_settle_and_stagger_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("LOP_BUILD_SETTLE_S", raising=False)
    assert child_mod._build_settle_seconds() == child_mod.BUILD_SETTLE_S == 10.0
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "0.2")
    assert child_mod._build_settle_seconds() == 0.2
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "0")
    assert child_mod._build_settle_seconds() == 10.0, "zero is the torn-tree race; refused"
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "nope")
    assert child_mod._build_settle_seconds() == 10.0
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.5")
    assert child_mod._build_stagger_seconds() == 0.5


# -- 6. the reaper branch -------------------------------------------------------


async def _run_until(stop: asyncio.Event, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline and not stop.is_set():
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_stamp_flip_announces_then_exits(disk, monkeypatch) -> None:
    """Flip mid-loop → ``retiring`` announced with both stamps, clean exit,
    ``stop`` set. A viewer being attached does not matter."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.03)
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.05")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")  # the quiet exit must not win
    disk["build"] = OLD
    reg = FakeRegistrant(attaches=1)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.1)
    assert not stop.is_set() and reg.retiring == []
    disk["build"] = NEW  # lop-update ran
    await _run_until(stop)
    assert stop.is_set()
    assert reg.retiring == [("stale-build", NEW.label())]
    assert handle.disposed and reg.closed
    await task


@pytest.mark.asyncio
async def test_work_arriving_after_the_announce_keeps_the_runtime(disk, monkeypatch) -> None:
    """The re-check after ``announce_retiring`` is load-bearing: a
    ``peer_message`` can open a turn in that await."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.02")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    reg = FakeRegistrant()
    handle = FakeHandle()

    async def announce(reason: str, *, to: str = "") -> None:
        reg.retiring.append((reason, to))
        if len(reg.retiring) == 1:
            handle._busy = True  # a turn starts between announce and exit, ONCE

    reg.announce_retiring = announce  # type: ignore[method-assign]
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.25)
    assert reg.retiring, "the announce must have gone out"
    assert not stop.is_set() and not handle.disposed, "busy after the announce keeps it"
    handle._busy = False  # the turn ends; the next check retires
    await _run_until(stop)
    assert stop.is_set() and handle.disposed
    await task


@pytest.mark.asyncio
async def test_work_arriving_during_the_stagger_keeps_the_runtime(disk, monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.3")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    monkeypatch.setattr(child_mod.random, "uniform", lambda a, b: b)
    reg = FakeRegistrant()
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.1)  # inside the stagger
    handle._busy = True
    await asyncio.sleep(0.4)
    assert reg.retiring == [], "no announce while busy"
    assert not stop.is_set()
    handle._busy = False
    await _run_until(stop, timeout=3.0)
    assert stop.is_set()
    await task


@pytest.mark.asyncio
async def test_a_stop_during_the_stagger_yields_to_it(disk, monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "1.0")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    monkeypatch.setattr(child_mod.random, "uniform", lambda a, b: b)
    reg = FakeRegistrant()
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.1)
    stop.set()  # a /stop landed
    await asyncio.wait_for(task, timeout=2)
    assert reg.retiring == [] and not handle.disposed, "the stop path owns the exit"


@pytest.mark.asyncio
async def test_a_runtime_without_a_boot_stamp_never_refreshes(disk, monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    reg = FakeRegistrant(attaches=1, boot=None)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.2)
    assert not stop.is_set() and reg.retiring == []
    stop.set()
    await task


# -- 7. the stagger -------------------------------------------------------------


@pytest.mark.asyncio
async def test_stagger_is_bounded_by_build_stagger_s(disk, monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.25")
    draws: list[tuple[float, float]] = []

    def uniform(a: float, b: float) -> float:
        draws.append((a, b))
        return b

    monkeypatch.setattr(child_mod.random, "uniform", uniform)
    reg = FakeRegistrant()
    handle = FakeHandle()
    stop = asyncio.Event()
    started = asyncio.get_running_loop().time()
    await asyncio.wait_for(_reaper(handle, reg, stop), timeout=3)
    elapsed = asyncio.get_running_loop().time() - started
    assert draws == [(0, 0.25)], "one draw, over [0, BUILD_STAGGER_S]"
    assert elapsed >= 0.25, "the retirement waited out the full stagger"
    assert stop.is_set()
