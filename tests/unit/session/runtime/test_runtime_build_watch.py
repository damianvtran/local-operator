"""The runtime's own build watch retires it, and no third party kills it.

WHY THIS FILE EXISTS (I2 of the mass-kill work). Three fleet-wide deaths are
documented in this repo — the 2026-09-15 19:41 sweep of 36 runtimes, the
`libpython` dylib pin that killed 113 processes the same day, and the 2026-09-18
vanishing of 25 runtimes inside 13 seconds — and the runtime's build watch is the
one path that LOOKS like all three from the outside: a runtime disappears
mid-fleet, on a build move, without an exit record of the ordinary kind. It is not
any of them. It calls itself: the ladder below retires ONE runtime, at ITS OWN
turn boundary, and leaves through the graceful disposal, logging
``retiring for <build>`` and then ``exiting cleanly``.

So the fix for the two hazards (I1) must not touch this path, and the pin is the
test: what the path does, in order, and the two things it must never do — take a
turn with it, or signal anybody. The ``lop serve`` daemon's half (announce only,
never exit, production supplies no callback) is pinned in
``tests/unit/server/test_serve_retire.py``; this is the session-runtime half.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from local_operator import buildwatch
from local_operator import update as update_mod
from local_operator.session.runtime import process
from local_operator.update import BuildStamp

BOOT = BuildStamp(version="0.51.0", source_ref="abc1234567890")
NEW = BuildStamp(version="0.52.0", source_ref="def1234567890")


@pytest.fixture()
def moved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer, SETTLED build on disk, and no stagger to sleep out.

    The readers are the shared ones (``buildwatch``/``update``) because both build
    watchers must obey one rule; the stagger is shortened because it is jitter for
    a fleet, not a behaviour under test.
    """
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)
    monkeypatch.delenv("LOP_BUILD_SETTLE_S", raising=False)
    monkeypatch.delenv("LOP_BUILD_STAGGER_S", raising=False)
    monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)
    monkeypatch.setattr(process, "_build_stagger_seconds", lambda: 0.0)


class IdleHandle:
    """The two seams the retire path uses, and a tripwire for the one it must not.

    ``may_refresh`` is the product's own idle predicate (``ServingSessionHandle``);
    ``begin_retire`` is the LATCH that commits the runtime to leaving in one
    synchronous step. ``request_stop`` exists to raise: nothing here may stop the
    runtime from the outside, and a signal sent by this path would be the
    third-party kill this whole change set exists to rule out.
    """

    def __init__(self, *, admits: bool = True) -> None:
        self.admits = admits
        self.latched: list[tuple[str, str]] = []

    def may_refresh(self) -> str:
        return ""

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        self.latched.append((cause, detail))
        return self.admits

    def request_stop(self) -> None:
        raise AssertionError("the build watch must never signal anyone")


class IdleRuntime:
    """The session under the handle: the announce seam and the boot build."""

    def __init__(self) -> None:
        self._boot_build = BOOT
        self.announced: list[tuple[str, str]] = []

    async def announce_retiring(self, cause: str, *, to: str = "") -> None:
        self.announced.append((cause, to))


@pytest.mark.asyncio
async def test_an_idle_runtime_retires_itself_gracefully_on_a_build_move(
    moved: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Announce, latch, then leave through the GRACEFUL disposal — in that order.

    Every assertion here is a different promise, and the order is the one the
    reference investigation could not reconstruct afterwards: the announcement is
    what makes a viewer re-engage instead of reading the exit as owner death, the
    latch is what makes the decision atomic against a turn arriving during the
    announce, and the reason string is what makes the exit self-explaining in the
    log (``retiring for 0.52.0`` — the sentence design §1.6 asks for, because an
    exiting runtime that logged nothing about itself is how a refresh retirement,
    a SIGTERM and a torn install became one story).
    """
    handle, runtime, stop = IdleHandle(), IdleRuntime(), asyncio.Event()
    exits: list[str] = []

    async def recording_exit(_handle: object, _runtime: object, *, reason: str) -> None:
        exits.append(reason)

    monkeypatch.setattr(process, "_clean_exit", recording_exit)

    with caplog.at_level(logging.INFO, logger=process.__name__):
        retired = await process._refresh_for(NEW, handle, runtime, stop)

    assert retired is True
    assert handle.latched == [("runtime-retired", " (0.51.0@abc1234 → 0.52.0@def1234)")]
    assert runtime.announced == [("stale-build", NEW.label())]
    assert exits == [f"retiring for {NEW.label()}"], exits
    assert stop.is_set(), "the exit ends the run: amain's wait() must return"
    assert any("retiring for 0.52.0" in record.getMessage() for record in caplog.records), [
        record.getMessage() for record in caplog.records
    ]


@pytest.mark.asyncio
async def test_a_turn_arriving_during_the_announce_keeps_the_runtime(
    moved: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The latch refusing is the whole safety property, and nothing else may proceed.

    ``begin_retire`` returns False when any work is in flight or any admission
    would be refused, so the refresh must keep the runtime rather than abort a turn
    it had just decided not to disturb — the shape a "sample the predicate, then
    exit" watcher gets wrong. Nothing was disposed and nothing was exited.
    """
    handle, runtime, stop = IdleHandle(admits=False), IdleRuntime(), asyncio.Event()
    exits: list[str] = []

    async def recording_exit(_handle: object, _runtime: object, *, reason: str) -> None:
        exits.append(reason)

    monkeypatch.setattr(process, "_clean_exit", recording_exit)

    assert await process._refresh_for(NEW, handle, runtime, stop) is False
    assert exits == [], "a refused latch must not leave"
    assert not stop.is_set(), "the runtime is still serving"


@pytest.mark.asyncio
async def test_a_stop_landing_during_the_stagger_leaves_the_exit_to_the_stop(
    moved: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stagger is an await, and a stop inside it owns the exit rather than this.

    A stop has already staged its own evidence (``control._write_stop_marker``) and
    chosen its rung; a second exit path completing here would attribute the death
    twice, to two different parties. The watcher stands down instead.
    """
    handle, runtime, stop = IdleHandle(), IdleRuntime(), asyncio.Event()
    stop.set()
    exits: list[str] = []

    async def recording_exit(_handle: object, _runtime: object, *, reason: str) -> None:
        exits.append(reason)

    monkeypatch.setattr(process, "_clean_exit", recording_exit)

    assert await process._refresh_for(NEW, handle, runtime, stop) is False
    assert handle.latched == [], "a stopter's runtime is not retired by the watcher too"
    assert exits == []


def test_the_serve_daemon_can_only_ever_exit_with_an_injected_callback() -> None:
    """The other graceful retirement, pinned at ITS seam: production only announces.

    ``retirement_poll`` writes the handover into the daemon's record and keeps
    serving; the only thing that can make it exit is an ``exit_process`` callback,
    which production never supplies — the live ``run/serve/61225.json`` still
    reading ``retiring_from 0.59.0`` hours later is exactly that behaviour and not a
    stuck daemon. ``tests/unit/server/test_serve_retire.py`` pins the sequence in
    full (announce while serving, never request shutdown, no callback); this pins the
    default that makes it true, so a future signature that defaults the callback to
    something exit-shaped reddens here.
    """
    import inspect

    from local_operator.server import retire

    parameter = inspect.signature(retire.retirement_poll).parameters["exit_process"]
    assert parameter.default is None
    assert "must not exit on marker drift" in (retire.retirement_poll.__doc__ or "")


def test_the_watchers_share_one_build_rule() -> None:
    """Both watchers read ``buildwatch``, so neither can drift onto its own copy.

    The session runtime's retire and the serve daemon's announcement must agree on
    what "the build moved" means, or one of them fires where the other refuses —
    and the disagreement shows up as a fleet behaviour rather than as a defect.
    """
    from local_operator.server import retire

    assert process._buildwatch is buildwatch
    assert retire.buildwatch is buildwatch
    assert process.BUILD_CHECK_S == buildwatch.BUILD_CHECK_S
    assert process.BUILD_SETTLE_S == buildwatch.BUILD_SETTLE_S
    assert retire.buildwatch.BUILD_CHECK_S == buildwatch.BUILD_CHECK_S


def test_a_handle_without_the_idle_predicate_is_never_retired() -> None:
    """Unknown state is not an invitation to leave (the reduced-host contract).

    A handle that cannot answer ``may_refresh`` — an older host, a stripped test
    handle — must never retire: the whole path is decoration on a process listing
    until the runtime can prove it would lose nothing.
    """

    class NoProbe:
        pass

    assert process._idle_for_refresh(NoProbe()) is False
    assert process._should_refresh(NoProbe(), BOOT) is None


def test_the_watcher_never_signals_anyone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A structural pin for the symptom this change set is about: nothing kills.

    ``IdleHandle.request_stop`` raises, and the fall-through of every guard above
    is the graceful disposal, so the only way this path can end a runtime is by the
    runtime leaving itself. Asserted by name rather than by behaviour because the
    count of kill sites is what an investigation reads: ``grep -rn 'os.kill'`` over
    this module must never grow one.
    """
    from pathlib import Path

    source = (Path(process.__file__)).read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert "signal.SIGKILL" not in source
    assert "_signal_and_confirm" not in source
