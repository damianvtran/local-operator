"""The latch that keeps a runtime's deferred MCP wiring off its boot path.

WHAT PROBLEM THIS SOLVES. A runtime child defers its MCP wiring
(``serving.spawn_owned_session`` passes ``defer_mcp_wiring=True``) so no
integration configuration can sit between the user and a bound session. The
deferred task's first instruction is a *synchronous* import of the MCP SDK,
though, and a task cannot run until the loop is free — which in
``process.amain`` is the inbox drain BEFORE ``RecordPublisher`` publishes the
record. So the import ran to completion inside the pre-publication window
anyway: measured 2.3 s on a machine with a server declared, in 12 of 12
MCP-declaring runs.

A task that *waits* on this latch instead parks before that import, and
``RuntimeServer._serve`` opens the latch the moment the record exists. The work
then happens where the design always said it should — after publication, riding
the record.

WHY A CLASS AND NOT A BARE ``asyncio.Event``. The latch is created on the
SESSION's loop (``spawn_owned_session``, where the deferred task is dispatched)
and opened by the RUNTIME. In the production path those are the same loop
(``process.amain`` -> ``start_in_process``), and a plain ``Event`` would do. They
are NOT the same thread under ``RuntimeServer.start()`` — thread mode runs
``_serve`` on the runtime's own thread — and ``Event.set()`` from another thread
sets the flag without waking the waiting loop, so the parked task would never
run and MCP would silently never be wired. That is the exact failure this latch
exists to prevent, so the loop is bound here and a cross-thread open hops with
``call_soon_threadsafe``. The contract is structural rather than a comment.

Nothing in-tree pairs a gated handle with thread mode today
(``spawn_owned_session`` is the only source of a gated handle, and its only user
is ``process.amain``). The point is that a future caller cannot break it by
choosing ``start()``.
"""

from __future__ import annotations

import asyncio


class PublicationGate:
    """A one-shot latch, opened from whichever thread publishes the record.

    Duck-typed for its two callers on purpose: the deferred wiring task only
    needs ``wait()``, and ``RuntimeServer._open_mcp_wiring_gate`` only needs
    ``set()``. ``is_set()`` exists for tests and for a reader asking whether the
    record has been published yet.
    """

    __slots__ = ("_event", "_loop")

    def __init__(self) -> None:
        #: The loop the waiting task runs on. Bound at construction, which is
        #: where the task is dispatched, so nothing later has to guess.
        self._loop = asyncio.get_running_loop()
        self._event = asyncio.Event()

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def set(self) -> None:
        """Open the latch, from the waiting loop or from any other thread.

        Same loop: set directly, which is the production path and costs nothing
        extra. Anywhere else — a runtime thread, or a plain thread with no loop
        at all — hop onto the owner loop so the parked task is actually woken
        rather than left parked with its flag already set. A closed loop means
        the process is going away and the task with it, so the open is dropped
        rather than raised: this is called from a ``finally`` on the way out of
        a failing boot, and a raise there would replace the real failure.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._event.set()
            return
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._event.set)
