"""Transcript.append_imported: source timestamps, anchors, attachments, validation."""

from __future__ import annotations

import json

import pytest

from local_operator.harness.types import ImageContent, Message, TextContent, ToolCall
from local_operator.session.transcript import (
    ENTRY_COMPACTION,
    ENTRY_MESSAGE,
    Transcript,
    TranscriptEntry,
)


def _user(entry_id: str, text: str, ts: float) -> Message:
    return Message(role="user", content=[TextContent(text=text)], id=entry_id)


@pytest.mark.asyncio
async def test_append_imported_preserves_source_time(tmp_path):
    transcript = Transcript(tmp_path / "sess")
    message = _user("entry-a", "hello", 1700000000.25)
    await transcript.append_imported(
        [TranscriptEntry("entry-a", 1700000000.25, ENTRY_MESSAGE, {"kind": "message"})],
        messages={"entry-a": message},
    )
    row = json.loads(transcript.path.read_text().splitlines()[0])
    assert row["ts"] == 1700000000.25
    assert row["id"] == "entry-a"
    assert transcript.build_llm_history()[0].content[0].text == "hello"


@pytest.mark.asyncio
async def test_anchor_later_in_same_batch_is_valid(tmp_path):
    transcript = Transcript(tmp_path / "sess")
    rows = [
        TranscriptEntry(
            "compaction-1",
            100.0,
            ENTRY_COMPACTION,
            {
                "summary": "summary",
                "first_kept_entry_id": "entry-b",
                "tokens_before": 10,
            },
        ),
        TranscriptEntry("entry-b", 101.0, ENTRY_MESSAGE, {"kind": "message"}),
    ]
    await transcript.append_imported(
        rows, messages={"entry-b": _user("entry-b", "kept", 101.0)}
    )
    history = transcript.build_llm_history()
    assert len(history) == 2
    assert history[1].content[0].text == "kept"


@pytest.mark.asyncio
async def test_missing_or_duplicate_ids_are_refused(tmp_path):
    transcript = Transcript(tmp_path / "sess")
    with pytest.raises(ValueError):
        await transcript.append_imported(
            [TranscriptEntry("entry-a", 1.0, ENTRY_MESSAGE, {"kind": "message"})] * 2
        )
    with pytest.raises(ValueError):
        await transcript.append_imported(
            [
                TranscriptEntry(
                    "compaction-1",
                    1.0,
                    ENTRY_COMPACTION,
                    {"summary": "s", "first_kept_entry_id": "missing", "tokens_before": 1},
                )
            ]
        )
    await transcript.append_imported(
        [TranscriptEntry("entry-a", 1.0, ENTRY_MESSAGE, {"kind": "message"})],
        messages={"entry-a": _user("entry-a", "one", 1.0)},
    )
    with pytest.raises(ValueError):
        await transcript.append_imported(
            [TranscriptEntry("entry-a", 2.0, ENTRY_MESSAGE, {"kind": "message"})]
        )


@pytest.mark.asyncio
async def test_inline_image_is_externalised(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    transcript = Transcript(tmp_path / "sess")
    message = Message(
        role="user",
        content=[TextContent(text="look"), ImageContent(data="aGk=" * 400, mime_type="image/png")],
        id="entry-img",
    )
    await transcript.append_imported(
        [TranscriptEntry("entry-img", 1.0, ENTRY_MESSAGE, {"kind": "message"})],
        messages={"entry-img": message},
    )
    payload = json.loads(transcript.path.read_text().splitlines()[0])["payload"]
    image = payload["content"][1]
    assert "data" not in image
    assert image["attachment"]


@pytest.mark.asyncio
async def test_tool_pair_stays_orderable(tmp_path):
    transcript = Transcript(tmp_path / "sess")
    call = ToolCall(id="call-1", name="bash", arguments={"command": "ls"})
    assistant = Message(
        role="assistant", content=[TextContent(text="running")], tool_calls=[call], id="a"
    )
    result = Message(
        role="tool",
        content=[TextContent(text="ok")],
        tool_call_id="call-1",
        tool_name="bash",
        id="r",
    )
    await transcript.append_imported(
        [
            TranscriptEntry("a", 1.0, ENTRY_MESSAGE, {"kind": "message"}),
            TranscriptEntry("r", 2.0, ENTRY_MESSAGE, {"kind": "message"}),
        ],
        messages={"a": assistant, "r": result},
    )
    history = transcript.build_llm_history()
    assert history[0].tool_calls[0].id == "call-1"
    assert history[1].tool_call_id == "call-1"
