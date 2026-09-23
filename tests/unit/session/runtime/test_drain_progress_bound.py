"""The red-then-green CELL for the drain's progress bound (agent review R4).

This file is deliberately the smallest thing that is **red on a tree without the
bound and green on one with it**, and it uses nothing a tree without the bound does
not already have: the production ``_reaper``, the production build watch, a fake
handle whose work is permanently in flight, and the bound set with
``raising=False`` — on a tree that never reads that constant, setting it is a no-op,
the drain holds, and the release this file asserts never happens.

WHY IT IS SEPARATE from ``test_buildwatch_progress.py``, which covers the mechanism
in detail: that file imports the new vocabulary, so on the older tree it fails at
COLLECTION — an ImportError proves nothing about behaviour, and the reviewer was
right to file it (round 1, R4). Everything asserted here has to be expressible on
both trees, which is also why the frame is only checked for *differing* from the
ordinary build phrase rather than for its exact new wording.

The observation window is one second of wall time at 4 Hz reaper ticks, against a
bound of 50 ms: it is the SHAPE (silent work is released by a clock, not by going
idle) and not the shipped 15 min, which is pinned in the other file with an
injected clock.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.process import _reaper
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import LEAVING_FOR_BUILD
from local_operator.update import BuildStamp

OLD = BuildStamp(version="0.59.7", source_ref="7fe8b1005")
NEW = BuildStamp(version="0.59.8", source_ref="dec7933a6")


class _MovingLane:
    """A session double whose LANE keeps stepping: never silent, never idle.

    One of ``_work_motion``'s five signals and the one the incident's runtime kept
    moving — the roster generation is bumped by every completed assistant message, model
    change and lifecycle event a child reports (``Session._schedule_subagent_persist``),
    so a stepping child keeps the parent's movement clock reset while the parent's own
    transcript stays frozen.

    ``step_forever`` is driven as a real task rather than by poking the movement clock,
    because the property under test is precisely that the staleness clock cannot be left
    alone by work of this shape.
    """

    def __init__(self) -> None:
        self._subagent_roster_generation = 0
        self.steps = 0

    async def step_forever(self) -> None:
        while True:
            self._subagent_roster_generation += 1
            self.steps += 1
            await asyncio.sleep(0.002)


class _Handle:
    """Permanently busy, with no session to report movement from."""

    _session: Any = None

    def __init__(self) -> None:
        self.drains = 0
        self.releases = 0
        self.disposed = False
        self.denials = 0
        self.retired = False
        # THE LATCH IS THE PRODUCTION HANDLE'S FIELD, because
        # ``begin_drain``/``end_drain`` below call that handle's own methods:
        # ``_draining`` and the retiring cause it latches are what it writes and
        # what the arms read back.
        self._draining = False
        self._retiring_cause = ""
        self._retiring_detail = ""
        self._disposing = False
        self.update_failed: str | None = None
        #: Permanently busy until a test says otherwise — the incident's own shape
        #: (a lane parked behind a child process), and the one state the abandon arm
        #: exists for. Flipped by the cell that asserts the COMMITMENT outlives the
        #: abandon: the departure must still happen when the work finally ends.
        self.busy_forever = True

    def is_busy(self) -> bool:
        return self.busy_forever

    def next_wake_due_at(self) -> None:
        return None

    def may_refresh(self) -> str:
        return "busy" if self.busy_forever else ""

    def attach_clients(self) -> int:
        return 0

    def begin_drain(self, cause: str, detail: str = "") -> bool:
        """The PRODUCTION latch, called through, with this file's counter beside it.

        ``ServingSessionHandle.begin_drain`` decides whether the latch closes and is
        the only thing that writes the state the arms read back, so it is CALLED
        rather than modelled — a double that re-implemented it would keep these cells
        green while the production handle stopped latching at all. ``drains`` is this
        rig's bookkeeping for the cell that asserts the latch is not taken twice.
        """
        latched = ServingSessionHandle.begin_drain(
            cast("ServingSessionHandle", self), cause, detail
        )
        if latched:
            self.drains += 1
        return latched

    def end_drain(self) -> bool:
        """The PRODUCTION release, called not modelled, for ``begin_drain``'s reason.

        The assertion these cells are about is that the latch comes OFF, so the release
        they measure has to be the one the production handle performs; ``releases``
        counts those, and a double that could fabricate one would let the arm pass while
        the handle kept refusing admissions forever.
        """
        released = ServingSessionHandle.end_drain(cast("ServingSessionHandle", self))
        if released:
            self.releases += 1
        return released

    @property
    def draining(self) -> bool:
        """The production latch's own field, under the name these cells read."""
        return self._draining

    def note_update_failed(self, pair: str, bound: float = 0.0) -> None:
        self.update_failed = pair

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        if self.may_refresh():
            return False
        self.retired = True
        return True

    def _deny_pending_gates(self) -> None:
        self.denials += 1

    async def dispose(self) -> None:
        self.disposed = True


class _Runtime:
    _boot_build = OLD

    def __init__(self) -> None:
        self.retiring: list[tuple[str, str, bool, str]] = []
        self.failures: list[tuple[str, float]] = []

    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        self.retiring.append((reason, to, draining, leaving))

    async def note_update_failed(self, pair: str, bound: float = 0.0) -> None:
        self.failures.append((pair, bound))

    async def aclose(self) -> None:
        pass


async def _wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.mark.asyncio
async def test_a_drain_with_MOVING_work_is_abandoned_at_the_dwell(monkeypatch) -> None:
    """THE OPERATOR'S EIGHT HOURS: work that keeps reporting is now bounded too.

    The cell above is the SILENT hold. This is the one the fleet actually produced,
    and it is the reason a second bound exists at all: a lane that keeps STEPPING
    resets the movement clock on every read (:func:`process._work_motion` reads the
    subagent roster generation among its five signals), so ``stalled_s`` never
    reaches ``BUILD_DRAIN_PROGRESS_S`` and, before this arm, nothing else in the
    process ended the hold — measured on the reporting host as eight hours latched,
    with the session refusing admissions for the whole of it.

    Red on a tree without the dwell, green with it, and the discrimination is by the
    PUBLISHED BOUND: the staleness arm cannot have fired here, because its own bound
    is fifteen minutes and this cell's run is a fraction of a second — so a tree that
    reached the abandon through silence would have to publish 900 s, while the dwell
    arm publishes the dwell it ran out of (patched here to 50 ms).

    The lane is driven for real — a task stepping the roster every few milliseconds,
    the shape its own comment describes — rather than by moving the clock, because the
    WHOLE POINT is that no injected clock can stand in for this: the clock the
    staleness bound reads is pushed forward by the work itself.
    """
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(child_mod, "BUILD_DRAIN_DWELL_S", 0.05, raising=False)
    monkeypatch.setattr(child_mod.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)

    lane = _MovingLane()
    handle, runtime, stop = _Handle(), _Runtime(), asyncio.Event()
    handle._session = lane
    stepping = asyncio.ensure_future(lane.step_forever())
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))
    try:
        assert await _wait_for(lambda: handle.drains == 1), "the drain latch never engaged"
        assert await _wait_for(lambda: handle.releases == 1, timeout=2.0), (
            "a lane that keeps stepping held the drain past the dwell: this is the "
            "state that stayed latched for eight hours, and nothing else ends it"
        )
        assert lane.steps > 2, "the lane has to have stepped for this cell to mean anything"
        assert handle.draining is False, "the latch was counted but never taken off"
        assert handle.retired is False, "the moving hold is abandoned, not retired"
        assert (
            not stop.is_set() and not handle.disposed
        ), "an abandoned handover keeps serving; it never ends the process or its turn"
        assert runtime.failures, "the abandoned handover was never published"
        assert runtime.failures[0][1] == pytest.approx(0.05), (
            "the failure was published with the STALENESS bound, so this cell was "
            "passed by the wrong arm rather than by the dwell"
        )
        assert handle.drains == 1, "the drain was latched a second time"
    finally:
        stepping.cancel()
        stop.set()
        task.cancel()


@pytest.mark.asyncio
async def test_a_drain_with_silent_work_is_ABANDONED_and_the_runtime_keeps_serving(
    monkeypatch,
) -> None:
    """The bound no longer cuts the turn: it gives up the handover and says so.

    THIS CELL USED TO ASSERT THE OPPOSITE, and the change is the operator's: a
    build move may not end a runtime with a turn in flight, because the answer to
    "this work has not reported anything for fifteen minutes" is a person looking
    at it, not a silent cut. The shape it asserts now is the whole arm — the latch
    is RELEASED (the double CALLS the production ``end_drain`` — see ``_Handle`` — so a
    tree that keeps refusing admissions fails here), nothing is disposed and ``stop`` stays
    clear (the process keeps serving), the failure is PUBLISHED, and the ordinary
    build phrase is still the one on the record rather than a forced-handover
    phrase that no longer describes what happens.

    The retry is asserted too, and it is a property of the SHAPE rather than of a
    number: the watch's trip is monotone for the process's life, so the next check
    re-commits a fresh drain. A tree that dropped the latch without re-arming the
    watch would hold this at one.
    """
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(child_mod, "BUILD_DRAIN_PROGRESS_S", 0.05, raising=False)
    monkeypatch.setattr(child_mod.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)

    handle, runtime, stop = _Handle(), _Runtime(), asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))
    try:
        assert await _wait_for(lambda: handle.drains == 1), "the drain latch never engaged"
        assert not stop.is_set() and not handle.disposed, "in-flight work is never aborted here"

        # The stall: nothing this runtime can observe moves again, ever.
        assert await _wait_for(lambda: handle.releases == 1, timeout=2.0), (
            "the drain's work went silent past the bound and the handover was not "
            "abandoned: the refusal is still holding and this handle is busy forever"
        )
        assert handle.draining is False, "the latch was counted but never taken off"
        assert not stop.is_set(), "the process must keep serving, not stop"
        assert not handle.disposed, "a runtime that keeps its build is never disposed"
        assert handle.retired is False, "the silent hold is abandoned, not quietly retired"
        assert runtime.failures, "the failed handover was never published"
        assert runtime.failures[0][1] == pytest.approx(0.05), (
            "the failure was published without the bound it ran out of, so no surface "
            "can say why the update did not happen"
        )
        assert handle.update_failed is None, (
            "the handle remembered the pair, which is the WINDOW rung's memo for not "
            "burning its bound twice — here it would be the opposite of correct, "
            "because the drain rung keeps asking"
        )
        assert [leaving for _reason, _to, _draining, leaving in runtime.retiring] == [
            LEAVING_FOR_BUILD
        ], "the abandoned handover wears a phrase for a departure it did not take"
        # THE COMMITMENT SURVIVES THE ABANDON, and the two halves are asserted apart
        # because they are what a re-latch would break: the drain object stays (a
        # second ``begin_drain`` would re-run ``retire_wakes_to_inbox`` and discard
        # the wakes this drain already swallowed), and the departure still happens at
        # the first idle instant.
        assert handle.drains == 1, "the drain was latched a second time"
        assert [reason for reason, _to, _d, _l in runtime.retiring] == [
            "stale-build"
        ], "the drain announced its departure twice"
        handle.busy_forever = False
        assert await _wait_for(lambda: handle.disposed, timeout=2.0), (
            "the work finished and the runtime never left, so the abandonment turned a "
            "stalled handover into a permanent one"
        )
    finally:
        stop.set()
        task.cancel()
