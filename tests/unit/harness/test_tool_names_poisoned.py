"""A poisoned transcript row comes out of replay clean on the request path.

The durable record keeps the garbled name forever; this proves the wire view
— the exact list the turn request carries — does not re-send it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.harness.replay_bound import bound_replay_payloads
from local_operator.harness.tool_names import (
    format_malformed_notice,
    normalise_replay_tool_names,
)
from local_operator.harness.types import Message, TextContent, ToolCall, ToolResult
from local_operator.session.transcript import Transcript, replay_entries


@pytest.mark.asyncio
async def test_poisoned_transcript_replays_clean_through_the_request_path(tmp_path: Path):
    transcript = Transcript(tmp_path)
    call = Message(
        role="assistant",
        content=[TextContent(text="running it")],
        tool_calls=[ToolCall(id="c1", name="bashLecture", arguments={"command": "echo hi"})],
    )
    result = Message.tool_result(
        ToolResult(
            tool_call_id="c1",
            tool_name="bashLecture",
            content=[TextContent(text="Tool not found: bashLecture")],
            is_error=True,
        )
    )
    await transcript.append_messages([Message.user("go"), call, result])
    transcript.flush()

    # The durable row still holds the original garble (audit path).
    raw_all = replay_entries(transcript.entries(), None, mode="context")
    raw = [m for m in raw_all if isinstance(m, Message)]
    assert any(c.name == "bashLecture" for m in raw for c in m.tool_calls)

    # The request path the turn builds: bind, then sanitise.
    wire = normalise_replay_tool_names(bound_replay_payloads(raw), ["bash"])
    names = [c.name for m in wire for c in m.tool_calls]
    texts = [block.text for m in wire for block in m.content if isinstance(block, TextContent)]
    assert names == ["bash"]
    assert format_malformed_notice("bash") in texts
    assert "Lecture" not in " ".join([*names, *texts])
    # The untouched replay still holds the garble for audit.
    assert any(c.name == "bashLecture" for m in raw for c in m.tool_calls)
