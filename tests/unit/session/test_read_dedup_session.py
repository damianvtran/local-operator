"""Read dedup through a real Session: live history, the loop's result append,
and a real compaction pass — no provider, a scripted stream drives the calls.

This is the end-to-end proof the ledger is wired where it must be: the
Session owns it, ``_build_tool_context`` hands it to the real ``read`` tool,
residency is read off ``Session._context.messages`` (the list the next request
is built from), and ``compact_now`` replacing that list invalidates it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import build_read_tool
from local_operator.tools.read_dedup import DEDUP_DETAILS_KEY

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


class ReadScript:
    """Each user prompt is answered by one ``read`` call (args from the queue)
    and then a closing text reply; compaction summaries get an inert reply."""

    def __init__(self) -> None:
        self.queue: list[dict] = []
        self.calls = 0

    def __call__(self, request, signal):
        purpose = getattr(request, "purpose", "")
        last = request.messages[-1] if request.messages else None
        wants_tool = (
            purpose != "compaction"
            and self.queue
            and isinstance(last, Message)
            and last.role == "user"
        )
        if wants_tool:
            self.calls += 1
            args = json.dumps(self.queue.pop(0))
            call_id = f"toolu_{self.calls:04d}"

            async def gen():
                yield StreamToolCallDelta(index=0, id=call_id, name="read", argument_delta=args)
                yield StreamEndEvent(stop_reason="tool_use")

            return gen()

        async def done():
            yield StreamTextDelta(delta="SUMMARY" if purpose == "compaction" else "ok")
            yield StreamEndEvent(stop_reason="stop")

        return done()


def read_results(session: Session) -> list[Message]:
    return [
        m
        for m in session._context.messages
        if isinstance(m, Message) and m.role == "tool" and m.tool_name == "read"
    ]


def body(message: Message) -> str:
    return "".join(b.text for b in message.content if isinstance(b, TextContent))


def details(message: Message) -> dict:
    payload = message.provider_payload or {}
    return payload.get("details") or {}


@pytest.mark.asyncio
async def test_session_dedups_then_rereads_after_edit_and_after_compaction(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    target = work / "notes.txt"
    target.write_text("".join(f"row {i}: the quick brown fox\n" for i in range(1, 41)))

    script = ReadScript()
    session = Session(
        model=MODEL,
        stream_fn=script,
        tools=[build_read_tool()],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        cwd=str(work),
        yolo=True,
    )

    script.queue.append({"path": "notes.txt"})
    await session.prompt("read it")
    script.queue.append({"path": "notes.txt", "range": "5-9"})
    await session.prompt("read it again")
    first, second = read_results(session)
    assert "row 20: the quick" in body(first)
    assert DEDUP_DETAILS_KEY in details(second), body(second)
    assert body(second).startswith(f"[unchanged since your earlier read of {target} lines 5-9")

    # An edit (same size, different bytes) makes the third read return content.
    target.write_text(target.read_text().replace("row 7: the quick", "row 7: THE QUICK"))
    # Whole file, so the entry it records is far longer than a pointer: the
    # post-compaction read below can then only return content because the
    # entry is no longer resident, never because the pointer would be no
    # shorter (verified by forcing ``resident`` true: that read then points).
    script.queue.append({"path": "notes.txt"})
    await session.prompt("and again")
    third = read_results(session)[-1]
    assert DEDUP_DETAILS_KEY not in details(third)
    assert "row 7: THE QUICK" in body(third)

    # After a compaction pass that summarises those results away, the same
    # unchanged span is read in full again.
    # The cut snaps to a turn boundary and keeps the newest turn whole, so a
    # plain filler turn puts the read turns on the summarised side.
    await session.prompt("filler " + "detail " * 200)
    session._compaction_settings = CompactionSettings(keep_recent_tokens=50)
    outcome = await session.compact_now()
    assert outcome.ran is True, outcome
    assert not any(m.tool_call_id == third.tool_call_id for m in read_results(session))
    script.queue.append({"path": "notes.txt", "range": "5-9"})
    await session.prompt("after compaction")
    fourth = read_results(session)[-1]
    assert DEDUP_DETAILS_KEY not in details(fourth), body(fourth)
    assert "row 7: THE QUICK" in body(fourth)
