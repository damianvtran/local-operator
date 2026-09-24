"""Bounded, ordered delivery for observers that must not pace a model stream.

Durability reducers continue to use Session.subscribe. Presentation consumers
may lag and reconnect from the canonical frontend snapshot / durable transcript;
if lifecycle events alone exhaust their budget, disconnect explicitly rather
than silently dropping boundaries or retaining an unbounded queue.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from local_operator.harness.types import (
    AgentEvent,
    EventHandler,
    MessageUpdateEvent,
    ReasoningDeltaEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class _Pending:
    event: AgentEvent
    deltas: list[str] = field(default_factory=list)
    #: The coalescing key this item was queued under, or ``None`` for an event
    #: that carries no delta. Held rather than re-derived from ``event`` so a
    #: later fragment's key can be compared against the QUEUED item's without
    #: unwrapping a possibly-mutated event.
    key: tuple[str, str] | None = None


def _coalesce_key(event: AgentEvent) -> tuple[str, str] | None:
    """The key adjacent delta events may share, or ``None`` when the event is
    not a delta fragment.

    Two families carry fragments of something in flight and are indistinguishable
    from each other by field name alone, so the key names the FAMILY as well as
    the stream: an assistant's visible text (``message_update``, keyed by the
    message it accumulates into) and the model's private reasoning
    (``reasoning_delta``, keyed by the message it belongs to). Folding one into
    the other would splice the model's thinking into the answer being painted,
    which is transcript corruption, so the family is part of the key and the two
    can never merge.

    A tuple rather than one joined string because this runs once per streamed
    token on the busiest path in the app: a tuple compare costs no formatting,
    and the family is a constant so the string part is the message id the event
    already holds.
    """
    if isinstance(event, MessageUpdateEvent):
        return ("message_update", event.message.id)
    if isinstance(event, ReasoningDeltaEvent):
        return ("reasoning_delta", event.message_id)
    return None


class PresentationSubscription:
    """One async observer, one bounded FIFO, with adjacent text coalescing.

    The byte budget also bounds a single coalesced item: queue length alone
    cannot constrain a long response arriving while the observer is suspended.
    No state reducer or tool decision is permitted to rely on this observer.

    Coalescing covers the reasoning channel too, and has to: reasoning arrives
    once per token and a long-thinking turn emits thousands of fragments, so a
    256-deep queue of uncoalesced reasoning would close the observer and drop
    the very stream this subscription exists to carry.
    """

    def __init__(
        self,
        handler: EventHandler,
        *,
        max_pending: int = 256,
        max_delta_chars: int = 65536,
        on_overflow: Callable[[], None] | None = None,
    ) -> None:
        if max_pending < 1 or max_delta_chars < 1:
            raise ValueError("presentation queue limits must be positive")
        self._handler = handler
        self._max_pending = max_pending
        self._max_delta_chars = max_delta_chars
        self._on_overflow = on_overflow
        self._queue: deque[_Pending] = deque()
        self._delta_chars = 0
        self._task: asyncio.Task[None] | None = None
        self.closed = False

    def enqueue(self, event: AgentEvent) -> None:
        if self.closed:
            return
        key = _coalesce_key(event)
        delta = str(getattr(event, "delta", "") or "") if key is not None else ""
        coalesce = key is not None and bool(self._queue) and self._queue[-1].key == key
        if (not coalesce and len(self._queue) >= self._max_pending) or (
            self._delta_chars + len(delta) > self._max_delta_chars
        ):
            self.close()
            logger.warning("presentation observer overflowed; reconnect from the session snapshot")
            if self._on_overflow is not None:
                self._on_overflow()
            return
        # Streaming messages mutate as generation advances; copy the envelope
        # and message now so later mutation cannot reorder this observer's view.
        snapshot = event.model_copy(deep=True)
        if coalesce:
            self._queue[-1].event = snapshot
            self._queue[-1].deltas.append(delta)
        else:
            self._queue.append(_Pending(snapshot, [delta] if delta else [], key))
        self._delta_chars += len(delta)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while self._queue and not self.closed:
            pending = self._queue.popleft()
            event = pending.event
            if pending.key is not None:
                # One ``delta`` per event, on the model copy, for both families:
                # ``model_copy(update=...)`` leaves a message_update's
                # accumulated ``message`` (the newest fragment's) alone, and a
                # reasoning_delta has no other content to preserve.
                delta = "".join(pending.deltas)
                self._delta_chars -= len(delta)
                event = event.model_copy(update={"delta": delta})
            try:
                outcome = self._handler(event)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                logger.warning("presentation observer failed for %s", event.type, exc_info=True)

    async def flush(self) -> None:
        """Explicit observer-side drain; the model's critical path never calls it."""
        if self._task is not None:
            await asyncio.shield(self._task)

    def close(self) -> None:
        self.closed = True
        self._queue.clear()
        self._delta_chars = 0
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def aclose(self) -> None:
        self.close()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
