"""The publication latch: opened from the publishing loop, or from any thread.

The latch is what keeps a runtime child's deferred MCP wiring off its
pre-publication path, and its ONE non-obvious requirement is that opening it
must wake a task parked on another loop's thread. A bare ``asyncio.Event``
satisfies every test that opens it on the same loop and silently fails the one
case that matters — ``Event.set()`` from a foreign thread sets the flag and never
wakes the waiter — so the cross-thread case is pinned here directly rather than
left to a comment.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from local_operator.session.runtime.publication import PublicationGate

#: Upper bound on an awaited event, never a budget to sleep through.
GUARD_S = 20.0


@pytest.mark.asyncio
async def test_opening_on_the_waiting_loop_releases_the_parked_task() -> None:
    gate = PublicationGate()
    released = asyncio.Event()

    async def waiter() -> None:
        await gate.wait()
        released.set()

    task = asyncio.create_task(waiter())
    try:
        for _ in range(50):
            await asyncio.sleep(0)
        assert not gate.is_set(), "the latch opened before anything published"
        assert not released.is_set()
        gate.set()
        assert await asyncio.wait_for(released.wait(), timeout=GUARD_S)
    finally:
        await task


@pytest.mark.asyncio
async def test_opening_from_another_thread_wakes_the_parked_task() -> None:
    """The thread-mode case: a bare ``asyncio.Event`` fails exactly here.

    ``RuntimeServer.start()`` runs ``_serve`` on the runtime's own thread while
    the deferred task waits on this process's main loop, so the open arrives from
    a foreign thread with no running loop of its own. Setting the flag without
    hopping would leave the wiring parked for the session's life — MCP silently
    never wired — which is the failure the latch exists to prevent.
    """
    gate = PublicationGate()
    released = asyncio.Event()

    async def waiter() -> None:
        await gate.wait()
        released.set()

    task = asyncio.create_task(waiter())
    try:
        for _ in range(50):
            await asyncio.sleep(0)
        assert not released.is_set()

        opener = threading.Thread(target=gate.set, name="gate-opener")
        opener.start()
        opener.join(timeout=GUARD_S)
        assert not opener.is_alive()

        assert await asyncio.wait_for(released.wait(), timeout=GUARD_S), (
            "the open arrived from another thread and the parked task was never "
            "woken; a flag-only set (a bare asyncio.Event) is the failure this "
            "class exists to prevent"
        )
    finally:
        await task


async def _build_gate_on_a_fresh_loop() -> PublicationGate:
    return PublicationGate()


def test_opening_a_closed_loop_is_dropped_rather_than_raised() -> None:
    """Called from a ``finally`` on the way out of a failing boot, so no raise.

    A raise here would replace the boot failure it is running underneath, and
    the process is going away with the task it would have woken anyway. Built on
    a loop this test then closes, so the gate's bound loop is genuinely closed
    when the open arrives from a thread with no loop of its own.
    """
    loop = asyncio.new_event_loop()
    try:
        gate = loop.run_until_complete(_build_gate_on_a_fresh_loop())
    finally:
        loop.close()

    gate.set()
    assert not gate.is_set(), (
        "a dropped open must leave the latch shut rather than appear to have "
        "released a task that no longer exists"
    )
