"""Compaction and the continuation nudge, which must not collide.

The todo reminder is the ONE injection nothing persists: ``_todo_continuation``
hands it to the loop as a follow-up, no event is emitted and no transcript entry
is written. Compaction, on the other hand, is built on the rendered history
being persisted history — it matches ``kept[0]`` against transcript ids so a
resume can replay the window it keeps, and it rebuilds the live context out of
that same rendered list.

Rendered into that history, one reminder broke compaction at both ends, and both
failures were silent:

- a reminder AT the cut point failed the replayability guard, so the pass was
  refused as ``cut_not_replayable`` — measured at 30/30 refusals with one open
  todo against 25/30 committed with none — and it recurred every turn, because
  the next reminder lands at the same structural offset. The automatic gate has
  nobody to tell, so the context simply stopped being compacted.
- a reminder inside the KEPT window was rebuilt as a plain ``Message(role="user")``,
  which neither expiry guard can see (both match ``CustomMessage``), so it went
  on asserting "these todo items are still open" after the items were done.

Both are asserted through the outcomes a host observes — the CompactionOutcome
and the text of the next request — never through a planner flag.
"""

from __future__ import annotations

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: Small enough that a handful of short turns leaves history outside the kept
#: window, so every turn past the first has something to summarize — and small
#: enough that the cut lands in the tail, where the reminder is.
KEEP_RECENT = 40


class ReplyingStream:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="reply " * 20)
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    def reminder_blocks(self) -> list[str]:
        """The ``<system-reminder>`` texts in the LAST request the model got."""
        return [
            message.text
            for message in self.requests[-1].messages
            if "<system-reminder>" in (message.text or "")
        ]


def make_session(tmp_path, stream, session_id: str, keep_recent: int = KEEP_RECENT) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / session_id),
        session_id=session_id,
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=keep_recent),
    )


async def committed_passes(
    tmp_path, session_id: str, todos: list[dict[str, str]], turns: int
) -> int:
    """How many of ``turns`` compaction passes actually committed.

    ``compact_now`` rather than the automatic gate so the trigger is not what is
    under test: both share one ``_plan_compaction``/``_run_compaction`` pair, and
    the manual entry point only skips the threshold check.
    """
    builtin.TODO_STORE.pop(session_id, None)
    if todos:
        builtin.TODO_STORE[session_id] = todos
    stream = ReplyingStream()
    session = make_session(tmp_path, stream, session_id)
    committed = 0
    refusals: list[str | None] = []
    try:
        for index in range(turns):
            await session.prompt(f"question {index} " + "detail " * 30)
            outcome = await session.compact_now()
            if outcome.ran:
                committed += 1
            else:
                refusals.append(outcome.reason)
    finally:
        await session.dispose()
        builtin.TODO_STORE.pop(session_id, None)
    # Named in the assertion message rather than asserted on: the point is the
    # rate, and a refusal for an honest reason (an empty first turn) is fine.
    assert "cut_not_replayable" not in refusals, refusals
    return committed


@pytest.mark.asyncio
async def test_an_open_todo_does_not_disable_compaction(tmp_path) -> None:
    """The regression, as a rate: a session with work still open compacts as
    often as one with none. Before the fix this was 0 committed against 5,
    every refusal reading ``cut_not_replayable`` and nothing said to the user.
    """
    turns = 6
    with_todos = await committed_passes(
        tmp_path,
        "todo-compaction-open",
        [{"text": "add the columns", "status": "pending"}],
        turns,
    )
    without_todos = await committed_passes(tmp_path, "todo-compaction-clear", [], turns)

    assert with_todos == without_todos
    assert with_todos >= turns - 1  # only the opening turn has nothing to compact


@pytest.mark.asyncio
async def test_a_committed_pass_leaves_no_reminder_to_go_stale(tmp_path) -> None:
    """The kept window must not bake the nudge in.

    A rebuilt context is rendered history, and a rendered reminder is a plain
    user message: invisible to ``_live_todo_reminders`` and to the renderer's
    newest-only rule, so it kept insisting the work was open after it was done.
    """
    session_id = "todo-compaction-stale"
    builtin.TODO_STORE[session_id] = [{"text": "add the columns", "status": "pending"}]
    stream = ReplyingStream()
    # A kept window several messages wide (these turns run ~600 tokens in total)
    # puts the live reminder INSIDE the window rather than at the cut, which is
    # the other half of the same defect.
    session = make_session(tmp_path, stream, session_id, keep_recent=200)
    try:
        for index in range(8):
            await session.prompt(f"question {index} " + "detail " * 30)
        # The reminder is doing its job while the item is open...
        assert stream.reminder_blocks()

        outcome = await session.compact_now()
        assert outcome.ran, outcome.reason

        builtin.TODO_STORE[session_id] = [{"text": "add the columns", "status": "done"}]
        await session.prompt("anything else?")

        # ...and says nothing once it is done. Before the fix one baked copy
        # survived here, asserting open work for the rest of the session.
        assert stream.reminder_blocks() == []
    finally:
        await session.dispose()
        builtin.TODO_STORE.pop(session_id, None)


@pytest.mark.asyncio
async def test_the_nudge_still_reaches_the_model_on_a_compacting_session(tmp_path) -> None:
    """The fix must not buy compaction by disarming the guardrail: excluding the
    reminder from the COMPACTION render leaves the request path untouched, so a
    turn that ends with work open is still re-entered with the items in front of
    the model."""
    session_id = "todo-compaction-nudge"
    builtin.TODO_STORE[session_id] = [{"text": "add the columns", "status": "pending"}]
    stream = ReplyingStream()
    session = make_session(tmp_path, stream, session_id)
    try:
        await session.prompt("go")

        assert len(stream.requests) == 2  # the loop re-entered
        blocks = stream.reminder_blocks()
        assert len(blocks) == 1 and "add the columns" in blocks[0]
    finally:
        await session.dispose()
        builtin.TODO_STORE.pop(session_id, None)
