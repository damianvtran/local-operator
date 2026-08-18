"""Do subagents actually make progress while the parent is inside a tool call?

The reported symptom was a parent that had to YIELD — finish its turn, or stop
calling tools — before a child would advance, which would make ``task`` a
sequential API wearing a concurrent one's clothes. That would be a real bug in
the shape of the loop rather than a slow model, so it is asserted here against
real ``Session`` objects and the real ``AsyncJobManager`` rather than argued
from the code.

Two independent claims, because they fail for different reasons and only one
of them is about the event loop:

- a child registered by ``task`` runs while the parent sits in a long
  ``await``, which is what "background" has to mean; and
- the parent can put a question to that child mid-tool-call and get the answer
  back, which additionally requires the child's aside boundary to be reached
  while the parent is blocked.

The parent's blocking work here is ``asyncio.sleep``, not a CPU spin: the
harness's own tools are async or ``to_thread``-offloaded, so a sleep models
them faithfully. A tool that blocked the loop synchronously would stall the
child, and that is a property of THAT tool, not of the subagent machinery.

These lock in the fix from #133 ("keep subagents running while the parent and
siblings work"), which found the per-turn compaction ruler running tiktoken
synchronously on the one loop that serves the parent, every child and the TUI
— 116 of 121 blocking samples, worst stall 860 ms — and moved it to a worker
thread. The first test was checked to genuinely discriminate: replacing its
``await asyncio.sleep(0.5)`` with a synchronous ``time.sleep(0.5)``, which is
what that stall looked like, fails it with the child frozen at one turn. So a
regression that puts blocking work back on the loop is caught here rather than
being rediscovered as "subagents only run when I yield".
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


async def wait_for(predicate, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


class CountingChild:
    """A child provider that records every turn it is asked for.

    The turn COUNT is the evidence: a child that never advances while the
    parent is busy produces exactly the turns it managed before the parent
    blocked, and one that runs concurrently keeps producing them.
    """

    def __init__(self) -> None:
        self.turns = 0

    def __call__(self, request: ChatRequest, signal=None):
        self.turns += 1
        from local_operator.harness.types import (
            StreamEndEvent,
            StreamTextDelta,
            StreamToolCallDelta,
        )

        body = "\n".join(
            message.text
            for message in request.messages
            if isinstance(message, Message) and message.role == "user"
        )

        async def stream(req, sig=None):
            if "stop now" in body:
                yield StreamTextDelta(delta="stopping")
                yield StreamEndEvent(stop_reason="stop")
                return
            # Keep working: a short sleep per turn so the child is genuinely
            # mid-flight rather than completing in one scheduler pass.
            yield StreamToolCallDelta(
                index=0, id=f"call-{self.turns}", name="bash", argument_delta='{"command": "true"}'
            )
            yield StreamEndEvent(stop_reason="toolUse")

        return stream(request, signal)


def make_session(tmp_path, provider, name: str) -> Session:
    return Session(
        model=MODEL,
        stream_fn=provider,
        tools=[],
        transcript=Transcript(tmp_path / name),
        system_blocks_provider=lambda: ["parent", "env"],
        cwd=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_a_child_advances_while_the_parent_is_inside_a_tool_call(tmp_path, monkeypatch):
    """The core claim of ``task``: launching is not the same as blocking.

    The parent registers a child and then sits in a long await, exactly as it
    would inside a slow tool. If the loop only advanced children between
    parent turns, the child's turn count would be frozen for the whole sleep.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    child_provider = CountingChild()
    parent = make_session(tmp_path, child_provider, "parent")
    await parent.async_init()

    parent._launch_subagent(label="worker", prompt="Keep working.")
    await wait_for(lambda: child_provider.turns >= 1)
    before = child_provider.turns

    # The parent is now doing something slow and awaits it, the way every
    # async tool in this harness does.
    await asyncio.sleep(0.5)

    assert child_provider.turns > before, (
        "the child made no progress while the parent was awaiting; subagents are "
        "not running concurrently with the parent's tool calls"
    )
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_parent_can_question_a_busy_child_and_get_an_answer(tmp_path, monkeypatch):
    """The round trip the hub exists for, while the child is mid tool loop.

    Stronger than the test above: the answer only arrives if the child reaches
    an aside-injection boundary and takes a turn to reply while the parent is
    parked in ``await comms.ask`` — i.e. the parent blocking does not stop the
    child from being scheduled.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    class Answering:
        """Works forever until asked something, then answers with its hub tool."""

        def __call__(self, request: ChatRequest, signal=None):
            from local_operator.harness.types import (
                StreamEndEvent,
                StreamTextDelta,
                StreamToolCallDelta,
            )

            body = "\n".join(
                message.text
                for message in request.messages
                if isinstance(message, Message) and message.role == "user"
            )

            async def stream(req, sig=None):
                if "Answer it now" in body and "answered" not in body:
                    yield StreamToolCallDelta(
                        index=0,
                        id="reply-1",
                        name="hub",
                        argument_delta='{"message": "still going, no blockers"}',
                    )
                    yield StreamEndEvent(stop_reason="toolUse")
                    return
                if "background job" in body:
                    yield StreamTextDelta(delta="ack")
                    yield StreamEndEvent(stop_reason="stop")
                    return
                yield StreamToolCallDelta(
                    index=0, id="tick", name="bash", argument_delta='{"command": "sleep 0.1"}'
                )
                yield StreamEndEvent(stop_reason="toolUse")

            return stream(request, signal)

    parent = make_session(tmp_path, Answering(), "parent")
    await parent.async_init()
    job_id = parent._launch_subagent(label="worker", prompt="Keep working.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)

    reply = await comms.ask(job_id, "are you stuck?", 20_000)

    assert reply.error is None, reply.error
    assert reply.timed_out is False, "the busy child never answered; it was not scheduled"
    assert "no blockers" in (reply.text or "")
    await parent.dispose()
