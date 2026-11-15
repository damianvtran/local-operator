"""The desktop snapshot builds its frame OFF the serving event loop.

WHY THIS FILE EXISTS (serve stall, 2026-11-15). A ``lop serve`` on the
operator's Mac stalled for 70-195 s at a time, 13+ episodes, until its
watchdog restarted it; ``/health`` timed out and the desktop app was unusable.
The loop-watchdog dump (``serve-stall-<pid>.log``) showed the events stream's
snapshot doing ALL of its state work on MAIN: ``bridge.state()`` ->
``FrontendStateStore._public_state`` (a deep clone of every job and usage row)
-> ``sync_wire_payload`` -> pydantic ``model_dump`` through the per-job usage
serializer. Proportional to the session, on the one thread every route shares.

Pinned here:

* the build no longer holds the loop: a ticker coroutine keeps running while a
  snapshot of a large frozen roster is built (fails on the pre-fix code, which
  held the loop for the whole build);
* the ORDERING the stream depends on survives the move: the state and its
  ``seq`` watermark are captured together, frames published while the frame is
  being built are above that watermark and are delivered after the snapshot
  exactly once, and a burst bigger than the queue during the build ends in an
  explicit ``gap`` rather than a silent eviction;
* concurrent opens of one unchanged state share one build, each with its own
  copy of the parts a caller writes.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import store as asks_store
from local_operator.harness.types import Usage
from local_operator.server.utils import desktop_sessions as module
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendUsage,
    JobState,
)

#: The roster size the guard builds. Measured on the dev host (2026-11-15): a
#: 2,000-job roster with usage costs ~0.2 s to clone and ~0.5 s to dump, i.e.
#: the pre-fix snapshot held the loop ~0.75 s -- far past the bound below, and
#: still cheap enough to build in a unit test.
ROSTER = 2_000
#: The loop-lag bound. Measured on the dev host (2026-11-15, 8 cores, CPython
#: 3.12) with the build already off the loop: 130-180 ms worst tick, and it is
#: NOT the build running on the loop -- with ``gc.disable()`` the same probe
#: reads ~6 ms for the clone and ~47 ms for the dump. The remainder is a
#: generational GC pass the worker's allocations trigger (it holds the GIL for
#: the whole collection, whatever thread runs it) plus pydantic-core's
#: Rust-side ``model_dump`` stretches between Python callbacks. The pre-fix
#: code held the loop 743-859 ms on this fixture (the roster clone for
#: ``history_cursor`` plus the inline build), so 300 ms discriminates by ~2.5x
#: while leaving ~1.7x headroom over the fixed code on a loaded CI runner. The
#: deterministic half of the guard is the thread assertion beside it.
MAX_LAG_S = 0.3
TICK_S = 0.01


def _large_state(session_id: str, cwd: str) -> FrontendSessionState:
    """A frozen roster shaped like the stalled session's: many jobs, each with usage."""
    usage = Usage(input_tokens=1000, output_tokens=200)
    components = [
        FrontendUsage(input_tokens=1000, output_tokens=200, account_ref=f"acct-{index}")
        for index in range(5)
    ]
    jobs = [
        JobState(
            id=f"job-{index:05d}",
            type="task",
            status="completed",
            result_text="r" * 200,
            usage=usage,
            descendant_usage=components,
        )
        for index in range(ROSTER)
    ]
    return FrontendSessionState(session_id=session_id, epoch="e-large", cwd=cwd, jobs=jobs)


def _bridge(tmp_path: Path, state: FrontendSessionState | None = None) -> Any:
    """The bridge over a real ``AttachedSession`` holding ``state``.

    Real rather than a double for the same reason ``test_desktop_cold_asks``
    gives: the snapshot reads the facade's own store, and the off-loop path's
    safety rests on that store's install-new-object discipline.
    """
    bridge = module.DesktopSessionBridge(tmp_path, "s1", str(tmp_path))
    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=module._no_takeover,
        surface="desktop",
    )
    remote._install_frontend(
        state or FrontendSessionState(session_id="s1", epoch="e-small", cwd=str(tmp_path))
    )
    bridge.remote = remote
    (tmp_path / asks_store.SESSIONS_DIRNAME / "s1").mkdir(parents=True, exist_ok=True)
    return bridge


async def _max_loop_lag(work: Any) -> tuple[Any, float]:
    """Run ``work`` while a ticker measures the longest the loop went unscheduled."""
    stop = asyncio.Event()
    worst = 0.0

    async def ticker() -> None:
        nonlocal worst
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(TICK_S)
            now = time.perf_counter()
            worst = max(worst, now - last - TICK_S)
            last = now

    task = asyncio.create_task(ticker())
    # Let the ticker take its first timestamp before the work starts.
    await asyncio.sleep(0)
    try:
        result = await work
    finally:
        stop.set()
        await task
    return result, worst


@pytest.mark.asyncio
async def test_a_large_snapshot_does_not_hold_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stall's MAIN-thread stack, as a bound: the loop keeps ticking during the build."""
    bridge = _bridge(tmp_path, _large_state("s1", str(tmp_path)))

    threads: list[str] = []
    original = module._wire_state_of

    def recorded(capture: Any) -> dict[str, Any]:
        threads.append(threading.current_thread().name)
        return original(capture)

    monkeypatch.setattr(module, "_wire_state_of", recorded)
    gc.collect()

    snapshot, lag = await _max_loop_lag(bridge.snapshot())

    jobs = snapshot["payload"]["frontend"]["snapshot"]["jobs"]
    assert len(jobs) == ROSTER, "the guard must measure the real build, not an empty frame"
    assert threads and all(
        name.startswith("lop-desktop-snapshot") for name in threads
    ), f"the frame was built on {threads}, not the snapshot pool"
    assert lag < MAX_LAG_S, (
        f"the snapshot held the event loop for {lag * 1000:.0f} ms; the state clone "
        "and sync_wire_payload must run off the loop (serve stall, 2026-11-15)"
    )


@pytest.mark.asyncio
async def test_the_off_loop_frame_is_the_frame_state_builds(tmp_path: Path) -> None:
    """Moving the build changed where it runs, not a byte of what it serves."""
    bridge = _bridge(tmp_path, _large_state("s1", str(tmp_path)))

    snapshot = await bridge.snapshot()

    assert snapshot["payload"]["frontend"] == bridge.state()


class _HeldBuild:
    """Holds the off-loop build open so the test can publish while it runs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.builds = 0
        original = module._wire_state_of

        def held(capture: Any) -> dict[str, Any]:
            self.builds += 1
            self.started.set()
            assert self.release.wait(10), "the test never released the build"
            return original(capture)

        monkeypatch.setattr(module, "_wire_state_of", held)

    async def wait_started(self) -> None:
        assert await asyncio.to_thread(self.started.wait, 10), "the build never started"


@pytest.mark.asyncio
async def test_frames_published_during_the_build_arrive_once_after_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering argument, end to end through ``events``.

    A frame published BEFORE the capture is inside the snapshot's state and is
    superseded (never delivered); frames published WHILE the frame is built off
    the loop are above its watermark and must each be delivered once, in order,
    after the snapshot.
    """
    bridge = _bridge(tmp_path)
    held = _HeldBuild(monkeypatch)
    sub = bridge.subscribe()
    bridge.publish("event", {"marker": "before"})
    stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
    first = asyncio.ensure_future(anext(stream))
    await held.wait_started()
    assert sub.captured and not sub.opened
    for index in range(3):
        bridge.publish("event", {"marker": f"during-{index}"})
    held.release.set()

    frames = [await first]
    for _ in range(4):
        frames.append(await asyncio.wait_for(anext(stream), 5))
    await stream.aclose()

    assert [frame["type"] for frame in frames] == ["open", "snapshot", "event", "event", "event"]
    snapshot = frames[1]
    delivered = [frame for frame in frames if frame["type"] == "event"]
    assert [frame["payload"]["marker"] for frame in delivered] == [
        "during-0",
        "during-1",
        "during-2",
    ], "a frame published during the build was lost, duplicated or reordered"
    assert all(frame["seq"] > snapshot["seq"] for frame in delivered)
    assert held.builds == 1


@pytest.mark.asyncio
async def test_a_burst_during_the_build_is_a_gap_not_a_silent_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above the watermark, eviction would LOSE frames: overflow must end in ``gap``.

    Before the capture a pre-open queue evicts its oldest frame because every
    frame it holds is superseded by the coming snapshot. Once the watermark is
    read, that is no longer true, so a full queue takes the opened reader's
    policy -- fold, else disconnect with an explicit gap the client reconciles.
    """
    monkeypatch.setattr(module, "REPLAY_COUNT", 8)
    bridge = _bridge(tmp_path)
    held = _HeldBuild(monkeypatch)
    sub = bridge.subscribe()
    # Fill the pre-open queue BEFORE the capture: these are superseded, and the
    # capture's drain must free the queue for what follows.
    for index in range(sub.queue.maxsize):
        bridge.publish("event", {"marker": f"before-{index}"})
    stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
    first = asyncio.ensure_future(anext(stream))
    await held.wait_started()
    assert sub.queue.empty(), "the capture must drain the superseded frames"
    for index in range(sub.queue.maxsize + 3):
        bridge.publish("event", {"marker": f"during-{index}"})
    held.release.set()

    frames = [await first]
    while frames[-1]["type"] != "gap":
        frames.append(await asyncio.wait_for(anext(stream), 5))
    await stream.aclose()

    markers = [frame["payload"]["marker"] for frame in frames if frame["type"] == "event"]
    assert not any(marker.startswith("before-") for marker in markers)
    # What was delivered is an unbroken PREFIX of what was published, then the gap.
    assert markers == [f"during-{index}" for index in range(len(markers))]


@pytest.mark.asyncio
async def test_concurrent_opens_of_one_state_share_one_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N windows on one huge session pay one build, and cannot see each other's edits."""
    bridge = _bridge(tmp_path)
    held = _HeldBuild(monkeypatch)
    first = asyncio.ensure_future(bridge.snapshot())
    await held.wait_started()
    second = asyncio.ensure_future(bridge.snapshot())
    # Let the second open reach the in-flight build before releasing it: it
    # awaits the attention glance (<= ATTENTION_SNAPSHOT_WAIT_S) and the cold
    # history/asks reads first.
    await asyncio.sleep(0.5)
    held.release.set()
    one, two = await asyncio.gather(first, second)

    assert held.builds == 1
    front_one = one["payload"]["frontend"]
    front_two = two["payload"]["frontend"]
    assert front_one == front_two
    assert front_one is not front_two and front_one["snapshot"] is not front_two["snapshot"]
    assert bridge._wire_inflight is None, "a settled build must not stay pinned on the bridge"
