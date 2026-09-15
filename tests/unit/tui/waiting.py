"""Readiness waits driven by Textual's existing test message publication.

Pass ``messages.on_message`` to ``App.run_test(message_hook=...)`` before boot.
This observes the real message/worker path; it neither pumps idle frames nor
replaces startup. A ready session is NOT a settled layout: geometry, animation
and parser-timer tests still need their own frame or timer barriers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from textual.message import Message


class MessageWaiter:
    """Recheck observable state on messages, with a hang guard, not a time bet."""

    def __init__(self) -> None:
        self._pending: set[asyncio.Event] = set()

    def on_message(self, message: Message) -> None:
        # One event per await lets independent consumers cancel without stealing
        # each other's wake. The hook has no subscription or task to outlive the
        # run_test context that owns it, and is inert between waits.
        for changed in tuple(self._pending):
            changed.set()

    async def wait_for(
        self,
        predicate: Callable[[], bool],
        *,
        description: str,
        timeout: float = 30.0,
    ) -> None:
        """Return on readiness, including readiness published before this wait.

        The hook runs before a message handler: publication means "recheck",
        not "complete". Never consume a wake after testing the predicate, since
        the predicate can itself arrange another publication. The timeout only
        turns a missing edge into a diagnostic instead of hanging the shard.
        """
        changed = asyncio.Event()
        self._pending.add(changed)
        guard = asyncio.timeout(timeout)
        try:
            async with guard:
                while True:
                    changed.clear()
                    if predicate():
                        return
                    await changed.wait()
        except TimeoutError as error:
            if not guard.expired():
                raise
            raise AssertionError(f"never became ready: {description}") from error
        finally:
            self._pending.remove(changed)
