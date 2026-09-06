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
            await self._prepare_and_commit(session_id, generation)

    async def _prepare_and_commit(self, session_id: str, generation: int) -> None:
        prepared: Prepared | None = None
        transferred = False
        try:
            prepared = await self._prepare(session_id)
            if self._closed or generation != self.generation or session_id != self.requested_id:
                return
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
        except Exception as error:
            if not self._closed and generation == self.generation:
                self._failed(session_id, error)
        finally:
            try:
                if prepared is not None and not transferred:
                    await self._release(prepared)
            finally:
                if not self._closed and generation == self.generation:
                    self.requested_id = ""
                    self.intent_id = ""
                    self._pending("")

    def cancel(self) -> None:
        self.generation += 1
        if self._task is not None:
            self._task.cancel()
        self.requested_id = ""
        self.intent_id = ""
        self._pending("")

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
