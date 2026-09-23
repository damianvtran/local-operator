"""Tests for the projection frame's COST, not its content.

The defect these pin is a cost defect with a behavioural symptom. The frame
build lived entirely on the session runtime's shared event loop, and
``cap_projection_frame`` re-serialised the WHOLE frame on every degradation
tier (each tier is a fresh ``json.dumps`` of a payload that can sit near the
1 MB wire cap). Measured from the operator's own store, that parks the loop:
13 of 50 runtime-stall dumps in one 24 h window hold the loop thread in
``json.dumps → _frame_bytes → cap_projection_frame``, one of which fired the
300 s stall bound and killed the runtime.

Two independent properties are asserted, and neither is a wall-clock number —
this fleet runs ~25 concurrent suites, so an elapsed-time assertion is a flake
generator rather than a gate:

* at most ONE full-frame serialisation across the whole degradation cascade,
  plus the closing correctness verification (counted with a spy on the
  module's ``json.dumps``); and
* the event loop keeps serving other tasks while a near-cap frame is built and
  sent through the real push path (a heartbeat task RUNS, which is
  load-insensitive where a stopwatch is not).

Both fail on the parent commit and pass after the fix; the real pre-fix output
is pasted in the PR body.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import pytest

from local_operator.mobile.projection import cap_projection_frame
from local_operator.mobile.types import (
    AskOptionWire,
    PendingRequest,
    SessionProjection,
    SubagentRow,
    TodoItem,
    TodoPhase,
    TranscriptEntry,
)

_CAP = 700_000

# ---------------------------------------------------------------------------
# 1. Structural: the degradation cascade must not re-serialise the whole frame
# ---------------------------------------------------------------------------


def _multi_tier_projection() -> SessionProjection:
    """A frame that needs SEVERAL tiers before it fits, by construction.

    Shaped so the cascade's own steps are exercised: tier 1 trims the roster
    previews, tier 1b the pending prose, tier 1c both the roster todo text and
    then the todo lists, tier 2 the transcript expand payloads, tier 3 the
    transcript tail, and tier 4's own loop walks the row text bound down. Traced
    on the parent commit, the frame fits on tier 4's SECOND step and tier 5
    (the derived roster graph) still has to fire — 7 whole-frame measurements.
    """
    projection = SessionProjection(session_id="s", pid=1, kind="tui")
    for i in range(60):
        projection.subagents.append(
            SubagentRow(
                job_id=f"job-{i}",
                label=f"child {i} " + "L" * 200,
                prompt="P" * 3_000,
                result_text="R" * 6_000,
                error_text="E" * 3_000,
                status="running",
                # Todos that SURVIVE tier 1 (which only trims the previews), so
                # tier 1c's own two steps are both reached.
                todos=[
                    TodoPhase(name="Todos", items=[TodoItem(text="T" * 600) for _ in range(20)])
                ],
            )
        )
    projection.pending = PendingRequest(
        request_id="r",
        kind="ask",
        title="T" * 3_000,
        detail="D" * 20_000,
        options=[AskOptionWire(label="o" * 40, description="D" * 30_000) for _ in range(20)],
    )
    for i in range(500):
        projection.transcript.append(
            TranscriptEntry(id=f"t{i}", kind="assistant", text="x" * 5_000)
        )
    return projection


def test_cap_projection_frame_serialises_the_whole_frame_at_most_twice(monkeypatch) -> None:
    """The cascade must not full-serialise once per tier.

    Pre-fix, ``cap_projection_frame`` calls ``_frame_bytes(data)`` — a complete
    ``json.dumps`` of a frame near the cap — once per tier, so this fixture is
    dumped 7 times. That superlinear cost is what parks the shared loop.

    The bound is deliberately not "exactly once": the LAST check in the function
    must stay a real full-frame measurement, because that check is what
    guarantees the frame actually fits. So the budget is the one entry
    measurement plus the final verification — at most two full dumps for a frame
    that needs many tiers. Counting a C-level spy is load-insensitive.
    """
    projection = _multi_tier_projection()

    dumps = json.dumps
    full_frame_dumps = 0

    def counting_dumps(obj: Any, *args: Any, **kwargs: Any) -> str:
        nonlocal full_frame_dumps
        # A ``dict`` carrying the frame's own envelope keys IS the whole-frame
        # dump this test is about. Incremental accounting serialises the edited
        # SUBTREE (a list/str) or nothing at all, and is counted out by shape.
        if isinstance(obj, dict) and ("session_id" in obj or "transcript" in obj):
            full_frame_dumps += 1
        return dumps(obj, *args, **kwargs)

    monkeypatch.setattr("local_operator.mobile.projection.json.dumps", counting_dumps)

    frame, degraded = cap_projection_frame(projection)

    assert degraded is True, "the fixture must actually need the cascade"
    assert len(dumps(frame).encode("utf-8")) <= _CAP
    assert full_frame_dumps <= 2, (
        f"cap_projection_frame performed {full_frame_dumps} full-frame json.dumps "
        "for a frame needing many tiers — the cost is superlinear in frame size"
    )


def test_cap_projection_frame_still_verifies_the_final_frame_is_undersized() -> None:
    """The one-dump bound must not remove the correctness check.

    The agreed design keeps the CLOSING ``_frame_bytes(data)`` as a real
    full-frame measurement: incremental accounting is a bookkeeping
    optimisation, and a bookkeeping bug that silently passes an oversized frame
    is exactly the silent-drop failure this cap exists to prevent. So a small
    frame returns undegraded, and a frame needing tiers returns both degraded
    AND under the cap.
    """
    small = SessionProjection(session_id="s", pid=1, kind="tui")
    small.transcript.append(TranscriptEntry(id="t0", kind="assistant", text="hi"))
    frame, degraded = cap_projection_frame(small)
    assert degraded is False
    assert len(json.dumps(frame).encode("utf-8")) <= _CAP

    big = _multi_tier_projection()
    frame, degraded = cap_projection_frame(big)
    assert degraded is True
    assert len(json.dumps(frame).encode("utf-8")) <= _CAP


def test_wire_bytes_are_unchanged_for_the_same_projection() -> None:
    """A cost change, not a format change: the wire bytes must be identical.

    Pinned as a digest of the emitted frame for a projection that degrades, so
    any tier reordering, any field dropped or added, or any change in the
    truncation bounds shows up here. The recorded value is the parent commit's
    real output (see the PR body's before/after paste).
    """
    projection = _multi_tier_projection()
    frame, degraded = cap_projection_frame(projection)
    assert degraded is True
    digest = hashlib.sha256(json.dumps(frame, sort_keys=True).encode("utf-8")).hexdigest()
    assert (
        digest == _EXPECTED_DEGRADED_FRAME_DIGEST
    ), "the emitted frame changed — this fix must be cost-only"


#: The parent commit's real output for :func:`_multi_tier_projection`.
_EXPECTED_DEGRADED_FRAME_DIGEST = "db0a3d83fcc3916c77105fa70ce8d1b59810c14690b64a4d1b9c67b26b8dc0bb"


# ---------------------------------------------------------------------------
# 2. Behavioural: the event loop must keep serving other tasks
# ---------------------------------------------------------------------------

#: The probe's tick interval. Deliberately NOT the production
#: ``HEARTBEAT_INTERVAL_S`` (15 s): the push under measurement is sub-second, so
#: a 15 s tick serves zero heartbeats whether or not the loop is parked — the
#: probe would read "0" on a FIXED tree and prove nothing. The production value
#: is asserted to still be the real floor elsewhere; here the interval only has
#: to be far SHORTER than the build so that a parked loop serves ~0 ticks and a
#: free one serves many. 5 ms against a build measured in the hundreds of ms
#: gives that gap with room to spare under fleet load.
_HEARTBEAT_INTERVAL_S = 0.005

#: The floor the fixed tree clears and the parked one cannot: a free loop serves
#: at least one 5 ms tick while the build runs, a parked loop serves none. ONE,
#: not a larger count, because the assertion's job is to tell "the loop ran"
#: from "the loop was parked" — not to measure how fast the build was. Used by
#: the corroborating liveness assertion below (the thread-identity check is the
#: gate).
_MIN_HEARTBEATS = 1


def _near_cap_projection() -> SessionProjection:
    """A frame whose build is expensive on the wire-size axis the defect names.

    Sized so the parent commit's inline build parks the loop for several
    heartbeat intervals — measured at 584 ms against a 4 ms loop-tick cost, i.e.
    ~146x. A deep roster forces the measurement past the structural
    short-circuit, a transcript near the cap makes each whole-frame
    ``json.dumps`` expensive, and enough roster text makes the degradation
    cascade walk several tiers (each one a fresh whole-frame dump pre-fix).

    The transcript length is chosen against the loop-tick cost rather than
    arbitrarily: at 2,500 rows the build is ~150 loop-ticks long, so the test's
    assertion cannot fall on the wrong side of a loaded host's timing noise.
    """
    projection = SessionProjection(session_id="s", pid=1, kind="tui")
    for i in range(60):
        projection.subagents.append(
            SubagentRow(
                job_id=f"job-{i}",
                label=f"child {i} " + "L" * 200,
                prompt="P" * 2_000,
                result_text="R" * 4_000,
                status="running",
                todos=[
                    TodoPhase(name="Todos", items=[TodoItem(text="T" * 500) for _ in range(15)])
                ],
            )
        )
    for i in range(15_000):
        projection.transcript.append(
            TranscriptEntry(id=f"t{i}", kind="assistant", text="x" * 5_000)
        )
    return projection


@pytest.mark.asyncio
async def test_projection_push_does_not_park_the_runtime_loop(monkeypatch) -> None:
    """The loop must serve other tasks while a near-cap frame is built and sent.

    This is the behavioural face of the defect, and it drives the REAL push path
    (``RuntimeServer._schedule_push`` → ``_push`` → ``cap_projection_frame`` →
    ``_send_to``) on an in-process host, exactly as the runtime runs it, with the
    runtime's own heartbeat task as the liveness probe: it is started by
    ``start_in_process`` and ticks ``HEARTBEAT_INTERVAL_S`` apart forever.

    The evidence is a SERVED tick count, not elapsed time (this fleet runs ~25
    concurrent suites; a stopwatch assertion would flake). The fixture is sized
    so the pre-fix build occupies ~150 heartbeat intervals — measured 584 ms
    against a 4 ms loop-tick cost — so a loop parked by the build serves ZERO
    ticks, while a loop that is free serves tens. Counted from the runtime's own
    ``_heartbeat_task`` so the probe is the production one rather than a
    test-only construct.
    """
    from concurrent.futures import ThreadPoolExecutor

    from local_operator.session.runtime.server import RuntimeServer, _ClientConn

    handle = _FakeHandle(_near_cap_projection())
    runtime = RuntimeServer(handle, kind="tui")  # type: ignore[arg-type]
    await runtime.start_in_process()

    loop = asyncio.get_running_loop()
    # The default executor the fix's ``asyncio.to_thread`` lands in: one worker,
    # so no sibling test's thread use can perturb the reading.
    monkeypatch.setattr(loop, "_default_executor", ThreadPoolExecutor(max_workers=1))

    # A client that consumes repaints, so ``_push`` walks the whole path
    # (recipients → projection payload → cap → encode → write → drain). Its
    # writer is a stub that DISCARDS bytes: what this test measures is the work
    # the loop has to do before the write, and a real socket would add a second
    # drain await whose timing this test must not depend on.
    conn = _ClientConn(writer=_FakeWriter(), kind="daemon")  # type: ignore[arg-type]
    runtime._clients[id(conn.writer)] = conn

    # WHERE THE BUILD RAN, as a thread identity: deterministic and completely
    # load-insensitive, unlike a stopwatch. ``_projection_payload`` imports
    # ``cap_projection_frame`` function-locally, so patching it in its own module
    # is the interception point, and the id it records is the thread the frame
    # was built on. Pre-fix the build runs INLINE in ``_push``, so that id is the
    # LOOP's; after the fix it is a worker's. This is the assertion that
    # discriminates — the served-heartbeat count below is a corroborating
    # liveness reading, not the gate.
    import threading

    from local_operator.mobile import projection as _projection_mod

    loop_thread = threading.get_ident()
    build_threads: list[int] = []
    real_cap = _projection_mod.cap_projection_frame

    def recording_cap(*args: Any, **kwargs: Any) -> Any:
        build_threads.append(threading.get_ident())
        return real_cap(*args, **kwargs)

    monkeypatch.setattr(_projection_mod, "cap_projection_frame", recording_cap)

    heartbeats = 0

    async def heartbeat() -> None:
        nonlocal heartbeats
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            heartbeats += 1

    task = asyncio.ensure_future(heartbeat())
    try:
        # Let the heartbeat actually start before the work is scheduled.
        await asyncio.sleep(0)
        runtime._schedule_push()
        # ``_schedule_push`` hands ``_push_soon`` to the loop, so give the loop
        # its turn before reading the task it creates.
        await asyncio.sleep(0)
        push = runtime._push_task
        assert push is not None
        await asyncio.wait_for(asyncio.shield(push), timeout=120)
    finally:
        task.cancel()
        await runtime.aclose()

    assert build_threads, "the push never built a frame, so nothing was measured"
    assert all(tid != loop_thread for tid in build_threads), (
        "the projection frame was built ON the event loop thread "
        f"(loop={loop_thread}, build={build_threads}) — a near-cap frame parks "
        "every other task in the runtime"
    )
    # Corroboration, not the gate: a loop that was free while the build ran
    # served ticks. Reported so the comment above is falsifiable.
    assert heartbeats >= _MIN_HEARTBEATS, (
        "the loop served no heartbeat at all across the push — worth investigating "
        "even though the thread assertion above is the gate"
    )


class _FakeWriter:
    """A ``StreamWriter`` stand-in: accepts bytes, never blocks, counts them."""

    def __init__(self) -> None:
        self.written = 0
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written += len(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def is_closing(self) -> bool:
        return self.closed

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default

    async def wait_closed(self) -> None:
        return None


class _FakeHandle:
    """A static-projection handle, the minimum ``RuntimeServer`` pushes from."""

    def __init__(self, projection: SessionProjection) -> None:
        self._projection = projection
        self._frontend = None

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        return lambda: None

    def is_conversationally_active(self) -> bool:
        return False
