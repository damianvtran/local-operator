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
from typing import Any

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.process import _reaper
from local_operator.session.runtime.types import LEAVING_FOR_BUILD
from local_operator.update import BuildStamp

OLD = BuildStamp(version="0.59.7", source_ref="7fe8b1005")
NEW = BuildStamp(version="0.59.8", source_ref="dec7933a6")


class _Handle:
    """Permanently busy, with no session to report movement from."""

    _session: Any = None

    def __init__(self) -> None:
        self.drains = 0
        self.disposed = False
        self.denials = 0
        self.retired = False

    def is_busy(self) -> bool:
        return True

    def next_wake_due_at(self) -> None:
        return None

    def may_refresh(self) -> str:
        return "busy"

    def attach_clients(self) -> int:
        return 0

    def begin_drain(self, cause: str, detail: str = "") -> bool:
        self.drains += 1
        return True

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

    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        self.retiring.append((reason, to, draining, leaving))

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
async def test_a_drain_with_silent_work_is_released_by_its_bound(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    # BOTH ARMS OF THE RED/GREEN: on a tree that has never heard of this constant
    # the assignment below simply creates it, nothing reads it, and the drain holds
    # for the whole window — which is the failure this cell reproduces.
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
        assert await _wait_for(lambda: handle.disposed, timeout=2.0), (
            "a drained runtime whose work reports nothing was never released: the "
            "drain waits for `is_busy()` to clear and this handle is busy forever"
        )
        assert stop.is_set(), "the reaper's release must stop the process, not just dispose"
        assert handle.retired is False, "the silent hold is cut, not quietly retired"
        assert [leaving for _reason, _to, _draining, leaving in runtime.retiring][-1] != (
            LEAVING_FOR_BUILD
        ), "the forced handover must not wear the ordinary build phrase"
    finally:
        stop.set()
        task.cancel()
