"""The aside and compaction-advisor requests are bounded like the turn's.

WHY THIS FILE EXISTS
====================
``harness/loop.py`` bounds what one turn re-sends. A REVIEW round 1 found that
claim was only half true: ``Session._read_only_prompt`` builds a SECOND
conversation request — the aside helper and the compaction advisor both go
through it — and it never crossed that seam. Measured on a real session, the
same tool row was 8,190 chars on the turn request and 123,780 here.

That was not only a missed bound. ``complete_aside`` is deliberately
byte-identical to the turn so it READS THE TURN'S PROVIDER CACHE, and a bound
applied at one site and not the other diverges the two at the turn's LAST
message — exactly where the cached prefix ends. So the property worth pinning
is not "the aside is bounded"; it is "the aside's messages are the SAME bounded
view the turn sends", which is what keeps the cache line intact.

The negative control at the end is what makes the positive assertion evidence
about the bound rather than about the fixture: with the bound switched to the
identity, the same call sends the full payload, exactly as it did before.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.harness.replay_bound import bound_replay_payloads
from local_operator.harness.types import Message, TextContent, ToolCall
from local_operator.session.session import Session


def lines(count: int, prefix: str = "output") -> str:
    return "".join(f"{prefix} {index}: value-{index}\n" for index in range(count))


def build_history() -> list[Message]:
    """One assistant call with oversized arguments and its oversized result."""
    argument_text = lines(4000)
    result_text = lines(4000)
    call = Message(
        role="assistant",
        content=[TextContent(text="writing the file")],
        tool_calls=[
            ToolCall(
                id="c1",
                name="write",
                arguments={"path": "big.py", "content": argument_text},
                raw_arguments=json.dumps({"path": "big.py", "content": argument_text}),
            )
        ],
    )
    result = Message(
        role="tool",
        content=[TextContent(text=result_text)],
        tool_call_id="c1",
        tool_name="write",
    )
    return [Message.user("go"), call, result]


class StubSession:
    """The smallest object ``_read_only_prompt`` needs to reach its return.

    Deliberately not a real ``Session``: constructing one costs a config dir, a
    credential store and a model registry, and none of that is what this
    property is about. Every method here is one ``_read_only_prompt`` actually
    calls, implemented as the identity it needs, so a failure can only come
    from the bound.
    """

    _frozen_system_blocks: list[str] = []

    def __init__(self, history: list[Message]) -> None:
        self._history = history

    def _system_blocks(self) -> list[str]:
        return ["you are a helper"]

    def _reconcile_tool_inventory(self, blocks: list[str]) -> list[str]:
        return blocks

    def _wire_legal_snapshot(self) -> list[Message]:
        return list(self._history)

    def _system_state_delta(self, desired: list[str]) -> tuple[list[str], Any]:
        return [], None

    def _render_history(self, messages: Any) -> list[Message]:
        return list(messages)


async def read_only_prompt(stub: StubSession, turns: list[Message]) -> list[Message]:
    _blocks, messages = await Session._read_only_prompt(stub, turns)  # type: ignore[arg-type]
    return messages


@pytest.mark.asyncio
async def test_the_aside_request_is_bounded():
    """The oversized row does not ride out at full size on the aside path."""
    history = build_history()
    messages = await read_only_prompt(StubSession(history), [Message.user("and now?")])

    result = next(m for m in messages if m.role == "tool")
    text = next(b for b in result.content if isinstance(b, TextContent)).text

    assert len(text) < len(history[2].content[0].text)  # type: ignore[union-attr]
    assert "elided" in text.lower(), "the elision marker must be visible"


@pytest.mark.asyncio
async def test_the_aside_is_the_same_bounded_view_the_turn_sends():
    """Messages-axis equality — the property that protects the provider cache.

    ``bound_replay_payloads`` is applied independently here, exactly as the
    loop's seam applies it to the turn. If the aside ever stops applying the
    same deterministic function, or applies it to a different list, these two
    stop being equal and the aside no longer reads the turn's cached prefix.
    """
    history = build_history()
    turns = [Message.user("and now?")]
    expected = bound_replay_payloads([*history, *turns])

    messages = await read_only_prompt(StubSession(history), turns)

    assert [m.model_dump() for m in messages] == [m.model_dump() for m in expected]


@pytest.mark.asyncio
async def test_with_the_bound_switched_off_the_full_payload_goes_out(monkeypatch):
    """The negative control: without the bound this test's fixture is unbounded.

    Same call, same fixture, one substitution — which is what makes the
    assertions above evidence about the bound rather than about the fixture.
    """
    monkeypatch.setattr(
        Session.__module__ + ".bound_replay_payloads",
        lambda messages: messages,
    )
    history = build_history()
    messages = await read_only_prompt(StubSession(history), [Message.user("and now?")])

    result = next(m for m in messages if m.role == "tool")
    text = next(b for b in result.content if isinstance(b, TextContent)).text

    assert text == history[2].content[0].text  # type: ignore[union-attr]
    assert "elided" not in text.lower()
