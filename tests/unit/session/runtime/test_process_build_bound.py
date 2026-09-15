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
from types import SimpleNamespace
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
        #: ``(reason, to, draining)`` — the third term is the frame's own
        #: verdict that admissions are about to be refused, which is what a
        #: viewer's notice is gated on. Recorded here because the CALLER is
        #: what decides it: a double that dropped the keyword would make the
        #: announce fail into the caller's `except` and read as "not
        #: announced" rather than "announced without the fact".
        self.retiring: list[tuple[str, str, bool]] = []

    def attach_clients(self) -> int:
        return self._attaches

    async def announce_retiring(self, reason: str, *, to: str = "", draining: bool = False) -> None:
        self.retiring.append((reason, to, draining))

    async def aclose(self) -> None:
        self.closed = True


class FakeHandle:
    #: Set by the handover test; the reaper reads it through ``getattr``.
    _session: Any = None
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
    assert reg.retiring == [
        ("stale-build", NEW.label(), True)
    ], "announced once, at drain start, and the frame SAYS a refusal is in force"
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
async def test_the_age_bound_survives_the_settle_windows_of_its_own_installs(
    disk, monkeypatch
) -> None:
    """MINOR 1 pinned: the belt must not be reset by the event it bounds.

    Every install that moves the stamp also produces observations with NO
    settled stamp at all — ``build_changed`` answers ``None`` while
    ``.lop-source`` is younger than ``BUILD_SETTLE_S``, about two checks per
    install at the shipped cadence. Clearing the clock on those made it the age
    of the last uninterrupted run of declines and left the "stamp that keeps
    moving" shape to the per-stamp count, which a fresh stamp every settled
    check resets by construction. On the head this was found on: 40 checks, a
    fresh stamp every other one, one decline each, never hard-stale.
    """
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

    def settling(*_a: Any, **_k: Any) -> float:
        # Every second check is INSIDE the settle window, which is what an
        # install in progress looks like: a stamp on disk, nothing this runtime
        # can be said to have refused yet.
        return 999.0 if counter["n"] % 2 else child_mod.BUILD_SETTLE_S / 2

    monkeypatch.setattr(update_mod, "installed_build", moving)
    monkeypatch.setattr(update_mod, "build_marker_age_s", settling)
    reg = FakeRegistrant(boot=OLD)
    handle = FakeHandle(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))

    assert await _wait_for(lambda: handle.drained), "the belt never tripped across settle windows"
    # The count cannot be what tripped it: every decline was a DIFFERENT stamp,
    # so it never got past one.
    assert "declined 1x" in handle.drain_detail, handle.drain_detail
    assert counter["n"] >= 4, "the install really did keep moving under the counter"
    assert not stop.is_set() and not handle.disposed, "in-flight work must never be aborted"
    stop.set()
    await task


@pytest.mark.asyncio
async def test_a_latched_drain_wins_over_the_soft_rung_at_the_exit(disk, monkeypatch) -> None:
    """MINOR 2 pinned: a latched drain is consulted BEFORE ``refreshable()``.

    The drain latches while the runtime is BUSY by construction, so the first
    build check after the work ends finds it idle-and-newer. Taking the soft
    rung there draws a SECOND ``BUILD_STAGGER_S`` slice at the exit — the delay
    the drain draws its own stagger to avoid — and announces ``retiring`` twice
    for one departure.
    """
    # A build check due on EVERY tick, so "which branch runs first" is decided
    # by the code rather than by the tick that happens to land first: the soft
    # rung is only ever taken inside ``refresh_check``, which the loop runs
    # before its own drain branch.
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.05)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.001)
    draws: list[tuple[float, float]] = []
    monkeypatch.setattr(child_mod.random, "uniform", lambda a, b: draws.append((a, b)) or 0.0)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    disk["build"] = NEW
    reg = FakeRegistrant(boot=OLD)
    handle = FakeHandle(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))

    assert await _wait_for(lambda: handle.drained), "the drain latch never engaged"
    handle._busy = False  # the turn ends and the exit path opens up
    assert await _wait_for(lambda: stop.is_set()), "the drain never took the exit"
    assert draws == [(0, child_mod.BUILD_STAGGER_S)], "exactly one stagger, drawn at drain start"
    assert reg.retiring == [("stale-build", NEW.label(), True)], "one departure, one announce"
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
    assert reg.retiring == [
        ("stale-build", NEW.label(), False)
    ], "an idle refresh refuses nothing, so the frame must not say it does"


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


def test_the_probe_is_disarmed_for_an_editable_tree(tmp_path: Path, monkeypatch) -> None:
    """THE negative control, through the REAL probe (review round 1, NIT 1).

    The earlier form injected ``armed=False``, which proved only that the flag
    gates the probe — never that an editable install answers False, which is the
    half every development worktree depends on. A worktree's files legitimately
    move under a running session (an editor's atomic save, a branch switch) and
    its stamp is constant, so retiring on either signal would kill a session for
    being edited.
    """
    from local_operator.update import InstallKind

    gone = tmp_path / "local_operator" / "session" / "runtime" / "process.py"
    monkeypatch.setattr(update_mod, "install_kind", lambda **_k: InstallKind.EDITABLE)
    monkeypatch.setattr(child_mod, "_build_prefix", lambda: None)
    assert child_mod._tree_is_replaceable() is False, "an editable tree must not arm the probe"

    # And the verdict the runtime acts on is the PROBE's, not the file's
    # absence: this deleted path is real and the watch still says nothing.
    watch = _BuildWatch(OLD, paths=(gone,), armed=child_mod._tree_is_replaceable())
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


# -- the inbox handover (the successor's half of the drain) ---------------------


@pytest.mark.asyncio
async def test_the_exit_hands_the_drains_wakes_to_a_successor(disk, monkeypatch) -> None:
    """The WIRING, not just the pieces: ``_drain_for`` calls the session's
    handover on the way out, after the commit and while the session is still
    alive to write it.

    A wake whose fire retired its schedule has no other route to a successor —
    nothing raises an errand for a schedule that has already fired (MINOR 3) —
    so a drain that exited without this call would leave the occurrence in the
    log and nowhere else.
    """
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(child_mod.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    disk["build"] = NEW
    handed: list[str] = []

    class StubSession:
        async def hand_wakes_to_successor(self) -> int:
            handed.append("disposed=%s" % handle.disposed)
            return 1

    reg = FakeRegistrant(boot=OLD)
    handle = FakeHandle(busy=True)
    handle._session = StubSession()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))

    assert await _wait_for(lambda: handle.drained), "the drain latch never engaged"
    handle._busy = False
    assert await _wait_for(lambda: stop.is_set()), "the drain never took the exit"
    assert handed == ["disposed=False"], "handed over once, before the dispose"
    await task


@pytest.mark.asyncio
async def test_the_boot_drain_runs_a_spooled_wake(tmp_path: Path) -> None:
    """MINOR 3, the sender's half: a wake row is RUN by the successor, not read.

    ``send --wake`` asked for a turn, and a row spooled by a runtime that was
    leaving a replaced build carries that ask. The boot drain used to hardcode
    ``wake=False``, so the field was written and then ignored and every spooled
    wake could only ever be read. A row with no wake stays the quiet note it
    always was.
    """
    from local_operator.session.runtime.inbox import InboxLine, append_inbox

    append_inbox(tmp_path, InboxLine(text="run the report", sender={}, mode="mailbox", wake=True))
    append_inbox(tmp_path, InboxLine(text="fyi", sender={}, mode="mailbox", wake=False))
    seen: list[tuple[str, str, bool]] = []

    class Handle:
        _session = SimpleNamespace(transcript=SimpleNamespace(directory=tmp_path))

        async def receive_peer_message(self, text, *, mode="mailbox", wake=False, sender=None):
            seen.append((text, mode, wake))
            return "recorded"

    assert await child_mod._drain_inbox_into(Handle()) == 2
    assert seen == [("run the report", "mailbox", True), ("fyi", "mailbox", False)]
