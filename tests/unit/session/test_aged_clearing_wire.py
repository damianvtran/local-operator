"""Aged tool-result clearing at the WIRE: the request the provider receives.

No provider: a scripted stream records every ``ChatRequest`` a real
``Session`` builds while one prompt drives ~45 tool calls of ~2k tokens each.
That is the shape the pass exists for — a delegated child's single launch
turn, which never reaches the compaction threshold — and what matters is the
request bytes, not the in-memory list:

(a) the first request after the clear carries stubs for results older than K
    calls and full text for newer ones;
(b) everything before the first cleared index is byte-identical to the
    previous request (the cached prefix survives up to the rewrite);
(c) the same run with the knob off carries no stubs (negative control);
(d) a fresh ``Session`` over the same transcript replays the same stubs.
"""

from __future__ import annotations

import json

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    AgentTool,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

from .test_session import ScriptedStream

# A window big enough that ~90k of history never approaches the compaction
# trigger: anything this test sees is the aged pass, not a compaction pass.
MODEL = ModelSpec(provider="test", model_id="m", context_window=1_000_000)
CALLS = 45
AFTER = 30
STUB_PREFIX = "[Older result cleared to save context"


def _view_tool() -> AgentTool:
    """A ``read``-named tool returning ~2k tokens of distinct text per path."""

    async def execute(tool_call_id, args, signal, on_update, context):
        path = args["path"]
        body = "\n".join(f"{path}:{line} " + "token " * 40 for line in range(48))
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="read",
            content=[TextContent(text=body)],
            details={"path": path, "range": "1-48"},
        )

    return AgentTool(
        name="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )


def _script() -> ScriptedStream:
    turns = [
        [
            StreamToolCallDelta(
                index=0, id=f"c{i}", name="read", argument_delta=json.dumps({"path": f"/r/f{i}.py"})
            ),
            StreamEndEvent(stop_reason="toolUse"),
        ]
        for i in range(CALLS)
    ]
    turns.append([StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")])
    return ScriptedStream(turns)


def _session(directory, stream, *, enabled: bool) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[_view_tool()],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable", "env"],
        compaction_settings=CompactionSettings(
            aged_clear_enabled=enabled,
            aged_clear_after_calls=AFTER,
            aged_clear_min_tokens=500,
            aged_clear_batch_tokens=10_000,
        ),
    )


def _wire(request) -> list[str]:
    """Each message of a request as the bytes a provider adapter serialises."""
    return [
        json.dumps(
            {
                "role": m.role,
                "tool_call_id": m.tool_call_id,
                "content": [b.model_dump(mode="json") for b in m.content],
                "tool_calls": [c.model_dump(mode="json") for c in m.tool_calls],
            },
            sort_keys=True,
        )
        for m in request.messages
    ]


def _tool_texts(request) -> dict[str, str]:
    return {m.tool_call_id: m.text for m in request.messages if m.role == "tool"}


@pytest.mark.asyncio
async def test_the_request_after_a_clear_carries_stubs_and_keeps_its_prefix(tmp_path):
    stream = _script()
    session = _session(tmp_path / "sess", stream, enabled=True)
    await session.prompt("survey the repo")

    stubbed = [i for i, r in enumerate(stream.requests) if STUB_PREFIX in "".join(_wire(r))]
    assert stubbed, "no request ever carried a stub"
    first = stubbed[0]
    request, previous = stream.requests[first], stream.requests[first - 1]

    # (a) Results >= AFTER calls old are stubs that name the re-fetch; the
    # rest are the full text. Request ``k`` is built after assistants
    # ``c0..c<k-1>``, so the result of ``c<i>`` has ``k - 1 - i`` assistant
    # messages after it — the pass's definition of age.
    texts = _tool_texts(request)
    for call_id, text in texts.items():
        age = first - 1 - int(call_id[1:])
        if age >= AFTER:
            path = f"/r/f{call_id[1:]}.py"
            assert text == (
                f"{STUB_PREFIX} \u2014 read {path} 1-48. Re-run it if you still need it.]"
            ), call_id
        else:
            assert text.startswith(f"/r/f{call_id[1:]}.py:0 "), call_id

    # (b) The cached prefix: byte-identical up to the first cleared index.
    now, before = _wire(request), _wire(previous)
    cut = next(i for i, row in enumerate(now) if STUB_PREFIX in row)
    assert cut > 0
    assert now[:cut] == before[:cut]
    assert now[cut] != before[cut]

    # And it happened mid-turn: the prompt is a single user turn, so every
    # stubbed request came from a tool-loop boundary, before the run ended.
    assert first < CALLS

    # (d) Resume: a fresh Session over the same transcript replays the stubs.
    session_texts = {
        m.tool_call_id: m.text
        for m in session._context.messages
        if getattr(m, "role", "") == "tool"
    }
    await session.dispose()
    resumed = _session(tmp_path / "sess", ScriptedStream([]), enabled=True)
    replayed = {
        m.tool_call_id: m.text
        for m in resumed._render_history(list(resumed._context.messages))
        if m.role == "tool"
    }
    assert replayed == session_texts
    assert sum(STUB_PREFIX in t for t in replayed.values()) >= CALLS - AFTER
    await resumed.dispose()


@pytest.mark.asyncio
async def test_negative_control_with_the_knob_off_sends_no_stub(tmp_path):
    stream = _script()
    session = _session(tmp_path / "sess", stream, enabled=False)
    await session.prompt("survey the repo")
    assert len(stream.requests) == CALLS + 1
    assert not any(STUB_PREFIX in "".join(_wire(r)) for r in stream.requests)
    # Every view is still there, full, in the last request.
    texts = _tool_texts(stream.requests[-1])
    assert len(texts) == CALLS
    assert all(t.startswith(f"/r/f{k[1:]}.py:0 ") for k, t in texts.items())
    await session.dispose()
