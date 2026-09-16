"""Generation-fenced preparation followed by one synchronous presentation commit.

The coordinator never knows how to stop a runtime. Its release callback owns
only speculative viewer resources, so cancellation, failure and rapid clicks
cannot inherit /resume's preference to stop the outgoing owner.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

Prepared = TypeVar("Prepared")
logger = logging.getLogger(__name__)


class PreparationInvalidated(RuntimeError):
    """The authoritative replay changed before presentation ownership moved."""


class SurfaceNotReady(RuntimeError):
    """A committed session's input surface never painted a usable frame.

    A PAINT failure, not a connection failure, and the distinction is the whole
    reason this type exists rather than a bare ``RuntimeError``. The session on
    the far side is bound and reachable; what did not happen is local. So it
    must NOT be retried the way an unreachable owner is: the readiness gate runs
    its own 15 s timer to expiry on every attempt, so folding it into a bounded
    reconnect budget multiplies one 15 s failure by the attempt count — measured
    at 8 commits, ~120 s of "Connecting…", and 8 forced full-screen arming
    relayouts, where the honest behaviour is a single 15 s failure.

    Subclasses ``RuntimeError`` because that is what ``_await_sidebar_frame``
    raised before it was named, and every existing handler that catches it —
    ``_commit_sidebar_session``'s callers, the navigation coordinator's
    ``failed`` hook — must keep catching it unchanged. Naming it only lets the
    ONE handler that needs to tell paint from connectivity do so, instead of
    matching on the message text.
    """


class OwnerWentCold(ConnectionError):
    """A LIVE commit whose owner went cold before its first frame could paint.

    The other half of the split above, and the reason both are named rather
    than matched on text. ``SurfaceNotReady`` is a LOCAL paint failure of a
    session that is bound and reachable, so it is terminal on the first
    occurrence (#883). This one is the TRANSIENT shape and must spend the
    reconnect budget instead: the frame was armed against a session that had
    passed the bind postcondition, and the owner was lost in the window after
    it — the facade is ``_recovering`` and clears itself within
    ``COLD_FALLBACK_S``, so a reselect heals in milliseconds while a latch
    makes the user perform that reselect by hand.

    Raised from ``post_display_hook`` rather than by a timer because the gate's
    FIRST check is ``is_cold``: for as long as the session is cold the gate
    can never pass, so every frame it waits out is spent buying full-screen
    relayouts for a verdict that is already known (measured pre-fix: 1,820
    refusals, every one of them a relayout, over the 15 s timer).

    Subclasses ``ConnectionError`` — the same type the bind postcondition in
    ``_connect_sidebar_source`` raises for this exact condition, carrying the
    same sentence — so the arms that already classify "the runtime is not
    responding" as connectivity keep classifying this as connectivity, and no
    second terminal wording is introduced.
    """


#: The one sentence both arms that detect an unreachable owner report, so the
#: bind postcondition and the cold-frame failure are indistinguishable to the
#: user and neither can drift into a second vocabulary for one condition.
UNREACHABLE_OWNER_MESSAGE = "the runtime is not responding"


class SessionNavigation(Generic[Prepared]):
    def __init__(
        self,
        *,
        prepare: Callable[[str], Awaitable[Prepared]],
        commit: Callable[[str, Prepared, int], Awaitable[None] | None],
        release: Callable[[Prepared], Awaitable[None]],
        pending: Callable[[str], None],
        failed: Callable[[str, Exception], None],
    ) -> None:
        self._prepare = prepare
        self._commit = commit
        self._release = release
        self._pending = pending
        self._failed = failed
        self.generation = 0
        self.requested_id = ""
        #: Where the user is HEADED, published SYNCHRONOUSLY — before the
        #: ``Selected`` message that starts the navigation has been dispatched.
        #: ``requested_id`` only becomes true inside :meth:`select`, which runs
        #: on message dispatch; a held key auto-repeats into ONE event batch,
        #: so a second press read the pre-press origin and both presses
        #: computed the same target (round 5, U7). Anything choosing a target
        #: relative to "where I am going" must read this, not ``requested_id``.
        self.intent_id = ""
        self.committed_id = ""
        self._task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._preparation_lock = asyncio.Lock()
        self._closed = False

    def intend(self, session_id: str) -> None:
        """Publish the target synchronously, before ``select`` can dispatch.

        Separate from :meth:`select` because a caller that posts a message to
        reach ``select`` has already decided; the decision must be readable in
        the same event batch, or the next press in an auto-repeat burst steps
        from a stale origin. Deliberately does NOT raise the input boundary or
        touch ``generation``: intent is not a commitment to prepare anything.
        """
        self.intent_id = session_id

    def select(self, session_id: str) -> asyncio.Task[None]:
        if self._closed:
            raise RuntimeError("session navigation is closed")
        self.generation += 1
        generation = self.generation
        self.requested_id = session_id
        self.intent_id = session_id
        if self._task is not None:
            self._task.cancel()
        # The boundary is raised before yielding: a following Enter cannot
        # accidentally submit to the conversation the user just left.
        self._pending(session_id)
        self._task = asyncio.create_task(self._navigate(session_id, generation))
        self._tasks.add(self._task)
        self._task.add_done_callback(self._settled)
        return self._task

    def _settled(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("session navigation cleanup failed", exc_info=task.exception())

    async def _navigate(self, session_id: str, generation: int) -> None:
        # One preparation owns sockets/history/widgets at a time. A rapid
        # burst replaces the desired ID, not a queue of expensive cold reads.
        async with self._preparation_lock:
            if self._closed or generation != self.generation:
                return
            while not self._closed and generation == self.generation:
                if not await self._prepare_and_commit(session_id, generation):
                    break

    async def _prepare_and_commit(self, session_id: str, generation: int) -> bool:
        prepared: Prepared | None = None
        transferred = False
        retry = False
        try:
            prepared = await self._prepare(session_id)
            if self._closed or generation != self.generation or session_id != self.requested_id:
                return False
            # There is deliberately no await between the final identity check
            # and commit. The app swaps every source-bound presentation field
            # together, then enables input on that exact authoritative facade.
            ready = self._commit(session_id, prepared, generation)
            transferred = True
            if ready is not None:
                # Ownership already moved atomically. The requested boundary
                # stays raised until its actual frame and input gates exist.
                await ready
            if not self._closed and generation == self.generation:
                self.committed_id = session_id
        except asyncio.CancelledError:
            raise
        except PreparationInvalidated:
            # Keep the requested/input boundary raised while the stale widgets
            # are released and the same latest intent prepares a canonical cut.
            retry = True
        except Exception as error:
            if not self._closed and generation == self.generation:
                self._failed(session_id, error)
        finally:
            try:
                if prepared is not None and not transferred:
                    await self._release(prepared)
            finally:
                if not retry and not self._closed and generation == self.generation:
                    self.requested_id = ""
                    self.intent_id = ""
                    self._pending("")
        return retry

    def cancel(self) -> None:
        self.generation += 1
        if self._task is not None:
            self._task.cancel()
        self.requested_id = ""
        self.intent_id = ""
        self._pending("")

    def abandon(self, task: "asyncio.Task[None]") -> bool:
        """Stop ONE navigation the caller started, if it is still in flight.

        :meth:`cancel` stops whatever is current, which is right for a user who
        just changed their mind and wrong for a caller giving up on its own
        request: a notification click that overruns its bound must not cancel a
        switch the user began in the meantime. Returns whether this call is the
        one that stopped ``task``, so a caller can tell "I stopped it" from "it
        was already finished or already superseded" — both of which mean the
        caller must read the outcome rather than assume one.

        THE GENERATION BUMP IS THE FENCE, and it is why this is not merely a
        polite ``task.cancel()``. Cancellation is delivered when the task next
        yields, which is unbounded; the bump is immediate and
        :meth:`_prepare_and_commit` re-reads it directly before ``_commit``
        with no await in between. So a navigation that outruns a caller's bound
        either commits before the bump — the caller reads a completed switch —
        or is fenced out of committing at all. It cannot commit *after* the
        caller has been told it failed, which is the state that gave a click
        both a session switch and a duplicate window.
        """
        if self._task is not task or task.done():
            return False
        self.cancel()
        return True

    async def close(self) -> None:
        self._closed = True
        self.generation += 1
        tasks, self._task = tuple(self._tasks), None
        for task in tasks:
            task.cancel()
        if tasks:
            # _settled records non-cancellation failures. Joining all retired
            # preparations prevents callbacks touching an already closed app.
            await asyncio.gather(*tasks, return_exceptions=True)
        self.requested_id = ""
        self.intent_id = ""
