"""The bound on build staleness, and the files-gone probe (design §4 F1).

``_should_refresh`` acts only on an instant at which nothing would be lost, so a
session busy for hours never reaches one — while ``lop-update``'s ``uv tool
install --force`` replaces the install tree WHOLESALE underneath it. Measured on
the reporting host: eight runtimes still executing 0.54.33 and two on 0.54.35
while the install had moved to 0.54.39 across six generations, and no retire
line for the last replacement because those runtimes were never idle.

These tests pin the two ways that ends without ever aborting work:

* the bound — a newer stamp declined ``BUILD_MAX_STALE_GENERATIONS`` times, or
  for ``BUILD_MAX_STALENESS_S``, drains the runtime: it stops admitting work,
  ANNOUNCES while it is still busy, and leaves only once the work in flight is
  done;
* the probe — the module tree the process loaded having disappeared from disk
  drains it too, and an editable/worktree install is never retired by it (the
  negative control: a developer's checkout legitimately looks stale by mtime).

Fakes in the style of ``test_process_refresh.py``, with the two latches the real
handle grew (``begin_drain``, ``begin_retire``) so the reaper's branch can be
driven without a Session.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.process import _BuildWatch, _reaper, _should_refresh
from local_operator.update import BuildStamp

#: The two builds the reporting host's gap spanned, verbatim.
OLD = BuildStamp(version="0.54.33", source_ref="7fe8b1005")
NEW = BuildStamp(version="0.54.39", source_ref="dec7933a6")


class FakeRegistrant:
    def __init__(self, *, boot: BuildStamp | None = OLD) -> None:
        self.watch_supported = False
        self.phone_watchers = 0
        self._attaches = 0
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
    """The real handle's shape: an idle gate plus the two retirement latches.

    ``begin_drain`` latches whatever the gate says; ``begin_retire`` refuses
    while work would be lost. The recording of WHICH one was used is the point —
    the whole bound is about the first being reachable while busy.
    """

    def __init__(self, *, busy: bool = False, next_wake_ms: int | None = None) -> None:
        self._busy = busy
        self._next_wake_ms = next_wake_ms
        self.disposed = False
        self.drained = False
        self.drain_cause = ""
        self.drain_detail = ""
        self.retired = False

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return self._next_wake_ms

    def may_refresh(self) -> str:
        if self.is_busy():
            return "busy"
        if child_mod._wake_within_window(self):
            return "wake due within the warm window"
        return ""

    def begin_drain(self, cause: str, detail: str = "") -> bool:
        self.drained = True
        self.drain_cause = cause
        self.drain_detail = detail
        return True

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        if self.may_refresh():
            return False
        self.retired = True
        return True

    def _deny_pending_gates(self) -> None:
        pass

    async def dispose(self) -> None:
        self.disposed = True


@pytest.fixture
def disk(monkeypatch):
    """Control what the build stamp on disk reports, as ``test_process_refresh`` does."""
    state: dict[str, Any] = {"build": NEW, "age": 999.0}
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: state["build"])
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: state["age"])
    for name in (
        "LOP_BUILD_SETTLE_S",
        "LOP_BUILD_STAGGER_S",
        "LOP_BUILD_PREFIX",
        "LOP_SESSION_GRACE_S",
    ):
        monkeypatch.delenv(name, raising=False)
    return state


async def _run_until(stop: asyncio.Event, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline and not stop.is_set():
        await asyncio.sleep(0.01)


async def _wait_for(predicate, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


# -- the bound ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_busy_runtime_drains_once_the_bound_trips(disk, monkeypatch) -> None:
    """THE pinning test (§4 F1): a permanently busy runtime declines a newer
    build, and past the bound it announces, refuses admissions and still does
    not exit until its work is done."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(child_mod.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    disk["build"] = NEW  # the install has already moved on
    reg = FakeRegistrant(boot=OLD)
    handle = FakeHandle(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))

    assert await _wait_for(lambda: handle.drained), "the drain latch never engaged"
    assert reg.retiring == [("stale-build", NEW.label())], "announced once, at drain start"
    assert handle.drain_cause == "runtime-retired"
    assert "declined" in handle.drain_detail, handle.drain_detail
    assert not stop.is_set() and not handle.disposed, "in-flight work must never be aborted"
    assert not handle.retired, "the exit rung is not reached while busy"

    handle._busy = False  # the turn ends; NOW the drain may take the exit
    await _run_until(stop)
    assert stop.is_set() and handle.disposed and handle.retired
    assert reg.closed
    await task


@pytest.mark.asyncio
async def test_the_bound_is_not_reached_while_the_stamp_is_settling(disk, monkeypatch) -> None:
    """The unsettled window is not a decline: a marker written mid-install is
    not a build this runtime can be said to have refused."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    disk["age"] = child_mod.BUILD_SETTLE_S / 2  # mid-install: torn tree
    reg = FakeRegistrant(boot=OLD)
    handle = FakeHandle(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.2)
    assert not handle.drained and reg.retiring == []
    assert not stop.is_set()
    stop.set()
    await task


@pytest.mark.asyncio
async def test_the_age_bound_catches_a_stamp_that_keeps_moving(disk, monkeypatch) -> None:
    """The count is per stamp, so a rebuild-per-check would reset it forever.
    The clock is the bound for that shape, and it is keyed on the FIRST decline."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(child_mod, "BUILD_MAX_STALE_GENERATIONS", 10_000)
    monkeypatch.setattr(child_mod, "BUILD_MAX_STALENESS_S", 0.15)
    monkeypatch.setattr(child_mod.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    counter = {"n": 0}

    def moving(*_a: Any, **_k: Any) -> BuildStamp:
        counter["n"] += 1
        return BuildStamp(version=f"0.54.{40 + counter['n']}", source_ref=f"deadbee{counter['n']}")

    monkeypatch.setattr(update_mod, "installed_build", moving)
    reg = FakeRegistrant(boot=OLD)
    handle = FakeHandle(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))

    assert await _wait_for(lambda: handle.drained), "the clock never tripped the bound"
    # The count never got past its first observation — every check saw a
    # DIFFERENT stamp, which is exactly what a per-stamp counter cannot bound —
    # and the clock tripped anyway. That is the shape the age bound exists for.
    assert "declined 1x" in handle.drain_detail, handle.drain_detail
    assert counter["n"] >= 2, "the stamp really did keep moving under the counter"
    assert not stop.is_set() and not handle.disposed
    stop.set()
    await task


@pytest.mark.asyncio
async def test_an_idle_runtime_keeps_todays_soft_refresh(disk, monkeypatch) -> None:
    """The negative control's other half: nothing changed for a runtime with
    nothing to lose — it retires on the FIRST observation through the soft rung,
    and the drain latch is never engaged."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.05")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    disk["build"] = NEW
    reg = FakeRegistrant(boot=OLD)
    handle = FakeHandle()
    stop = asyncio.Event()
    await asyncio.wait_for(_reaper(handle, reg, stop), timeout=5)
    assert stop.is_set() and handle.disposed
    assert not handle.drained, "an idle runtime never needs the hard-stale latch"
    assert reg.retiring == [("stale-build", NEW.label())]


@pytest.mark.asyncio
async def test_a_handle_without_the_drain_latch_keeps_serving(disk, monkeypatch) -> None:
    """An older host cannot refuse admissions, so it must not pretend to leave:
    the bound keeps its old behaviour rather than walking away mid-stream."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    disk["build"] = NEW

    class Bare(FakeHandle):
        begin_drain = None  # type: ignore[assignment]

    reg = FakeRegistrant(boot=OLD)
    handle = Bare(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.3)
    assert reg.retiring == [], "no announcement a handle cannot honour"
    assert not stop.is_set() and not handle.disposed
    stop.set()
    await task


# -- the files-gone probe -------------------------------------------------------


def test_the_probe_is_disarmed_for_an_editable_tree(tmp_path: Path) -> None:
    """THE negative control: a worktree install is not retired when its files
    move. Every development checkout is this case, and an editor's atomic save
    or a branch switch is not a reason to kill a session."""
    gone = tmp_path / "local_operator" / "session" / "runtime" / "process.py"
    watch = _BuildWatch(OLD, paths=(gone,), armed=False)
    handle = FakeHandle(busy=True)
    now = time.monotonic()
    assert watch.poll(handle, now=now).files_gone is False
    assert watch.poll(handle, now=now + 10_000).files_gone is False


def test_the_probe_is_disarmed_for_an_unknown_layout(tmp_path: Path, monkeypatch) -> None:
    """An install kind no updater owns keeps today's behaviour (no probe),
    which is also why the e2e stage's fake prefix cannot trip it."""
    from local_operator.update import InstallKind

    monkeypatch.setattr(update_mod, "install_kind", lambda **_k: InstallKind.UNKNOWN)
    monkeypatch.setattr(child_mod, "_build_prefix", lambda: None)
    assert child_mod._tree_is_replaceable() is False
    monkeypatch.setattr(update_mod, "install_kind", lambda **_k: InstallKind.UV_TOOL)
    assert child_mod._tree_is_replaceable() is True


def test_a_present_tree_is_never_gone(tmp_path: Path) -> None:
    present = tmp_path / "process.py"
    present.write_text("x", encoding="utf-8")
    watch = _BuildWatch(OLD, paths=(present,), armed=True)
    assert watch.poll(FakeHandle(busy=True), now=time.monotonic()).files_gone is False


def test_the_probe_waits_out_the_install_settle(tmp_path: Path, monkeypatch) -> None:
    """One missing observation is the shape of a NORMAL in-place upgrade (the
    installer rewrites site-packages over several seconds), so the probe needs
    the absence to persist: a runtime that retired inside that window would send
    its viewer to spawn against a half-written tree."""
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "5")
    gone = tmp_path / "local_operator" / "__init__.py"
    watch = _BuildWatch(OLD, paths=(gone,), armed=True)
    handle = FakeHandle(busy=True)
    first = time.monotonic()
    assert watch.poll(handle, now=first).files_gone is False, "first sighting is not a verdict"
    assert watch.poll(handle, now=first + 1).files_gone is False, "inside the settle"
    assert watch.poll(handle, now=first + 6).files_gone is True, "sustained past the settle"
    # And a tree that comes back clears the clock rather than poisoning it.
    gone.parent.mkdir(parents=True, exist_ok=True)
    gone.write_text("x", encoding="utf-8")
    assert watch.poll(handle, now=first + 7).files_gone is False


@pytest.mark.asyncio
async def test_files_gone_drains_even_with_no_newer_stamp(disk, monkeypatch, tmp_path) -> None:
    """The case no stamp comparison can see: the tree is gone, so the stamp
    read degrades to \"nothing to do\" forever. The probe is the only signal,
    and it must drain a BUSY runtime — that runtime is the one that never
    reaches an idle instant on its own."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(child_mod.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "0.02")
    disk["build"] = OLD  # the stamp reads exactly what this process loaded
    monkeypatch.setattr(child_mod, "_tree_is_replaceable", lambda: True)
    monkeypatch.setattr(
        child_mod,
        "_loaded_tree_paths",
        lambda: (tmp_path / "site-packages" / "local_operator" / "__init__.py",),
    )
    reg = FakeRegistrant(boot=OLD)
    handle = FakeHandle(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))

    assert await _wait_for(lambda: handle.drained), "the probe never drained the runtime"
    assert "gone" in handle.drain_detail, handle.drain_detail
    assert not stop.is_set() and not handle.disposed, "still no abort of in-flight work"
    handle._busy = False
    await _run_until(stop)
    assert stop.is_set() and handle.disposed
    await task


def test_should_refresh_still_refuses_a_busy_runtime(disk) -> None:
    """The soft rung is unchanged: the bound lives beside it, not in it."""
    assert _should_refresh(FakeHandle(busy=True), OLD) is None
