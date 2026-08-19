"""A parent talks to a genuinely busy child, through the real ``hub`` tool.

The reported symptom was a parent that had to YIELD before a child would
advance, and a ``hub ask`` that would not reach a child mid-work. Both were
root-caused and fixed by #133 ("keep subagents running while the parent and
siblings work"): the per-turn compaction ruler ran tiktoken synchronously on
the one loop serving the parent, every child and the TUI (116 of 121 blocking
samples, worst stall 860 ms), and ``ask`` refused a child the loop had
scheduled but not yet entered.

**Those two halves are already guarded, and not here.** The loop-stall half is
held by ``test_the_loop_stays_responsive_while_several_subagents_run`` in
``tests/unit/session/test_launch_subagent.py``, whose watchdog measures loop
lateness under a calibrated multi-child workload — the only honest way to
catch a synchronous stretch, and one this file cannot improve on. The
``ask``-grace half is held by ``test_ask_waits_for_a_child_the_loop_has_not
_entered_yet`` and its neighbours in ``test_comms.py``. Reverting either half
of #133 fails four of those tests.

An earlier version of this file claimed to guard both and did not: reverting
#133 outright left it green. The tests looked right and bit nothing, because
a two-turn fixture history never reaches ``OFFLOAD_MIN_CHARS`` (20 000) so the
ruler stayed inline either way, and awaiting ``session_dir_of`` before asking
meant the child was always already attached. A test that cannot fail is worse
than no test: it advertises coverage that is not there.

So this file keeps only what the others do NOT cover — the full round trip
through the real ``hub`` tool against a child that is genuinely mid tool loop.
That path crosses the tool layer, the aside-injection boundary and the child's
reply watcher, and it is the exact user-facing action ("check in on a working
subagent") that was reported broken.

Be precise about what this does and does not prove, because the previous
version of this file was not. It is an INTEGRATION test of the ask round trip,
not a regression test for #133: reverting #133 leaves it green, since by the
time it asks, the child is attached and the fixture history is far too small
to reach the offload threshold. The #133 guards are the four tests named
above, and they are where a loop-blocking regression is caught. What this adds
is that the tool-layer path from ``execute_hub`` to a working child's reply
holds together end to end — which none of those four exercise.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import execute_hub

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


async def wait_for(predicate, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


class BusyChild:
    """A child that works in a tool loop forever until the parent asks it
    something, answers with its ``hub`` tool, and then carries on working.

    The unbounded loop is deliberate: the child must still be MID-WORK when
    the question arrives, so the answer proves an aside reached a busy agent
    rather than one that had already gone idle.
    """

    def __call__(self, request: ChatRequest, signal=None):
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
                # A settled child's result re-wakes the parent; acknowledge it
                # so the parent's own turn can end.
                yield StreamTextDelta(delta="ack")
                yield StreamEndEvent(stop_reason="stop")
                return
            yield StreamToolCallDelta(
                index=0, id="tick", name="bash", argument_delta='{"command": "sleep 0.05"}'
            )
            yield StreamEndEvent(stop_reason="toolUse")

        return stream(request, signal)


@pytest.mark.asyncio
async def test_the_hub_tool_questions_a_busy_child_and_returns_its_answer(tmp_path, monkeypatch):
    """The user-facing round trip: `hub op='ask'` against a working child.

    Everything downstream of the tool call has to work for this to pass — the
    question is queued as an aside, the child reaches an injection boundary
    while still in its tool loop, answers with its own child-shaped ``hub``,
    and the reply resolves the parent's pending future. A child that could
    only be reached while idle, or a parent that could not be woken by the
    answer, fails here.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = Session(
        model=MODEL,
        stream_fn=BusyChild(),
        tools=[],
        transcript=Transcript(tmp_path / "parent"),
        system_blocks_provider=lambda: ["parent", "env"],
        cwd=str(tmp_path),
    )
    await parent.async_init()
    job_id = parent._launch_subagent(label="worker", prompt="Keep working.")
    comms = parent.subagent_comms
    await wait_for(lambda: comms.session_dir_of(job_id) is not None)

    result = await execute_hub(
        "call-1",
        {"op": "ask", "to": ["worker"], "message": "are you stuck?", "timeout_ms": 20_000},
        None,
        None,
        ToolContext(cwd=".", subagent_comms=comms),
    )

    block = result.content[0]
    assert isinstance(block, TextContent)
    assert not result.is_error, block.text
    assert "no blockers" in block.text, f"the busy child never answered; got: {block.text}"
    # The child is still working: the answer came from a mid-flight agent, not
    # from one that had finished and gone quiet.
    job = parent.jobs.get(job_id)
    assert job is not None and job.status == "running"
    await parent.dispose()
