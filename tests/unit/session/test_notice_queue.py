"""When a queued notice reaches the transcript, and when it is thrown away.

``Session.queue_notice`` exists so a line raised while a turn's PROMPT is being
built — which is where the classification layer raises its resource line — does not
land in the answer's slot (design round 1, D1). Two properties make that work, and
neither was pinned by a test until round 3: the line goes out AFTER the answer has
been persisted, and a turn that dies before producing an answer discards its queue
rather than leaving it for the next message to inherit (review round 3, MINOR 2 —
the stale line arrived ahead of the next turn's own, attributed to a message that
delivered nothing).

These drive the REAL ``Session`` through the real turn path; the harness comes from
``test_config_live``, which builds one over a stub stream.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import StreamEndEvent, StreamTextDelta
from tests.unit.session.test_config_live import make_session

LINE = "Suggestion added for this message: skill://alpha"
STALE = "Suggestion added for your previous message: skill://beta"


class _StubStream:
    """Text, then an end — or a stall, so the turn can be aborted mid-flight."""

    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.entered = asyncio.Event()

    def __call__(self, request: Any, signal: Any = None) -> Any:
        async def gen() -> Any:
            yield StreamTextDelta(delta="ok")
            self.entered.set()
            if self.block:
                # Long enough that the abort below is what ends the turn; the session
                # cancels the pump, so nothing waits this out.
                await asyncio.sleep(30)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _kinds(events: list[Any]) -> list[str]:
    return [str(event.type) for event in events]


def _texts(session: Any) -> list[str]:
    return [getattr(event, "text", "") for event in _EVENTS[id(session)] if event.type == "notice"]


#: Captured per session, keyed by id — the same shape ``test_config_live`` uses.
_EVENTS: dict[int, list[Any]] = {}


def _capture(session: Any) -> list[Any]:
    events: list[Any] = []
    _EVENTS[id(session)] = events
    session.subscribe(events.append)
    return events


def _queue_once(session: Any, line: str) -> Any:
    """A ``system_blocks_provider`` that queues ONE notice on its first call.

    Queued from the prompt build on purpose: that is the sync seam the knowledge hook
    runs in, inside the turn, with ``_turn_lock`` held — exactly where
    ``_select_knowledge_block`` emits the classification notice. A line queued from
    outside a turn would take the immediate branch instead and prove nothing.
    """
    state = {"queued": False}

    def provider() -> list[str]:
        if not state["queued"]:
            state["queued"] = True
            asyncio.get_running_loop().create_task(session.queue_notice(line))
        return ["stable"]

    return provider


@pytest.mark.asyncio
async def test_a_queued_notice_lands_after_the_answer(tmp_path: Path) -> None:
    """The answer's events come first — the flush point IS the contract."""
    session = make_session(tmp_path, _StubStream())
    session._system_blocks_provider = _queue_once(session, LINE)  # type: ignore[method-assign]
    events = _capture(session)

    await session.prompt("hello")

    assert LINE in _texts(session)
    kinds = _kinds(events)
    # The answer must be on the wire BEFORE the line that annotates it, or the line
    # reads as the first thing the model said (D1). ``MessageEndEvent`` is the
    # assistant message landing; the notice must follow it.
    assert kinds.index("message_end") < kinds.index("notice"), kinds
    assert session._queued_notices == []

    await session.dispose()


@pytest.mark.asyncio
async def test_a_queued_notice_is_discarded_when_the_turn_is_cancelled(tmp_path: Path) -> None:
    """A cancelled turn throws its queue away, and the next message carries only its own.

    This is the sequence the review reproduced: the cancelled turn's line survived it,
    so the NEXT message painted two notices — its own, and a stale one attributed to a
    message that delivered nothing. Cancelling the turn's own task is how that happens:
    the ``CancelledError`` unwinds ``_run_turn`` past its flush, which is why the
    discard belongs in the ``finally``.

    (An ABORT is a different ending and is not asserted the same way: the run returns
    normally from an aborted stream, persists the partial answer and DOES reach the
    flush, so its line is delivered — correctly, the answer landed.)
    """
    stream = _StubStream(block=True)
    session = make_session(tmp_path, stream)
    session._system_blocks_provider = _queue_once(session, LINE)  # type: ignore[method-assign]
    _capture(session)

    task = asyncio.create_task(session.prompt("hello"))
    await asyncio.wait_for(stream.entered.wait(), timeout=10)
    # The prompt build has run, so the line is queued and held: it is waiting for an
    # answer that this turn will never produce.
    assert session._queued_notices, "the line must be held, or this tests nothing"

    task.cancel()
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(task, timeout=10)

    assert session._queued_notices == [], "a cancelled turn's queue must not survive it"

    # …and the NEXT message must not inherit it: it queues its own line, and that is
    # the only notice the user sees. Pre-fix this list held the stale line FIRST.
    session._stream_fn = _StubStream()
    session._system_blocks_provider = _queue_once(session, STALE)  # type: ignore[method-assign]
    await session.prompt("second")

    assert _texts(session) == [STALE]

    await session.dispose()
