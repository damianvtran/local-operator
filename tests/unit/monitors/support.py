"""Deterministic waits for the monitor suites.

WHY THIS EXISTS. These suites used to wait for a due check with
``for _ in range(N): await asyncio.sleep(0)``. That bounds *event-loop turns*,
not *work*: it is only right while the awaited chain never leaves the loop.
Since ``Session._deliver_monitor`` began awaiting ``asyncio.to_thread(mark_dirty)``
before it emits the ``MonitorDeltaEvent`` (code-requests PR1b, 9385a7658f), the
chain hops to a worker thread and back, and how many turns that takes depends on
the host's scheduling, not on the code. On a loaded CI runner the emit landed
after the assertion and
``test_arm_kill_reopen_delivers_one_consolidated_delta`` failed with
``assert 0 == 1`` on linux 3.12 shards at least five times.

THE FIX IS TO WAIT ON THE WORK. A delivery is awaited *inside* the scheduler's
check task (``MonitorScheduler._run_check`` -> ``_deliver``), so awaiting the
scheduler's in-flight check tasks to completion covers the check, the gate, the
counters write and the delivery sink, however many thread hops the sink makes.
The deadline below is a backstop that turns a genuine wedge into a failure
carrying state, not a timing assumption; the success path never consults a
clock. Same shape as ``_wait_for_delivery`` in ``tests/e2e/test_ask_queue_e2e.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

#: Backstop for one drain, in seconds. The healthy path takes milliseconds; this
#: exists only so a check that never finishes fails the test instead of the CI
#: job's ``timeout-minutes``. Generous on purpose: raising it is never the fix
#: for a failure here, a wedge is.
DRAIN_BACKSTOP_S = 60.0


def _pending(tasks: Any) -> list[asyncio.Task[Any]]:
    return [task for task in list(tasks) if not task.done()]


def _state(scheduler: Any, session: Any | None) -> str:
    lines = [
        f"scheduler in-flight monitors: {sorted(scheduler._inflight)}",
        f"scheduler pending check tasks: {[repr(t) for t in _pending(scheduler._check_tasks)]}",
    ]
    if session is not None:
        lines.append(
            "session pending background tasks: "
            f"{[repr(t) for t in _pending(session._background_tasks)]}"
        )
    return "\n".join(lines)


async def drain_checks(scheduler: Any, session: Any | None = None) -> None:
    """Await every in-flight check (and, given a session, its spawned turns).

    Loops until a pass finds nothing pending, so a check whose task lands
    while the drain is running is awaited too. A check parked on the
    semaphore is in that set -- its task exists from the pump, and the drain
    awaits it as it takes the slot the finishing check frees. A due entry the
    pump DEFERRED at pump time (no task created -- the semaphore was closed
    when it came due) is NOT covered: it starts on a later, timer-driven
    pump, armed no sooner than ``MIN_ARM_MS`` after the freeing check
    (agent review round 1 narrowed this sentence).
    With ``session``, the background tasks the delivery spawned
    (``_send_monitor_message`` -> ``_spawn_background``) are drained as well, so
    the test ends with no delivery work still racing ``dispose``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + DRAIN_BACKSTOP_S
    while True:
        pending = _pending(scheduler._check_tasks)
        if session is not None:
            pending += _pending(session._background_tasks)
        if not pending:
            return
        remaining = deadline - loop.time()
        done, still = await asyncio.wait(pending, timeout=max(remaining, 0))
        if still:
            raise AssertionError(
                f"timed out after {DRAIN_BACKSTOP_S}s draining monitor work\n"
                + _state(scheduler, session)
            )


async def wait_set(event: asyncio.Event, what: str) -> None:
    """Await a test-owned event with the same backstop as :func:`drain_checks`.

    For "the check has started and is parked on my gate" waits, where the event
    is the effect itself. A bare ``event.wait()`` would hang the CI job (there is
    no ``pytest-timeout`` in this suite) if the code under test never got there.
    """
    try:
        await asyncio.wait_for(event.wait(), DRAIN_BACKSTOP_S)
    except asyncio.TimeoutError:
        raise AssertionError(f"timed out after {DRAIN_BACKSTOP_S}s waiting for {what}") from None
