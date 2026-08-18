"""Transcript tests: append-only JSONL, replay, and the compaction boundary."""

from __future__ import annotations

import json

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.session.transcript import ENTRY_MESSAGE, Transcript, TranscriptEntry


@pytest.fixture
def transcript(tmp_path):
    return Transcript(tmp_path / "sess")


@pytest.mark.asyncio
async def test_append_message_writes_jsonl(transcript):
    message = Message.user("hello")
    entry = await transcript.append_message(message)
    assert entry.type == ENTRY_MESSAGE
    assert entry.id == message.id  # entry id IS the message id

    lines = transcript.path.read_text().splitlines()
    assert len(lines) == 1
    raw = json.loads(lines[0])
    assert raw["type"] == "message"
    assert raw["payload"]["role"] == "user"
    assert raw["payload"]["content"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_reloads_from_disk(tmp_path):
    directory = tmp_path / "sess"
    first = Transcript(directory)
    await first.append_message(Message.user("one"))
    await first.append_message(Message.assistant("two"))

    reopened = Transcript(directory)
    assert len(reopened.entries()) == 2
    history = reopened.build_llm_history()
    assert [m.text for m in history if isinstance(m, Message)] == ["one", "two"]


@pytest.mark.asyncio
async def test_malformed_lines_dropped_individually(tmp_path):
    directory = tmp_path / "sess"
    store = Transcript(directory)
    await store.append_message(Message.user("good"))
    with store.path.open("a") as handle:
        handle.write("{not json\n")
        handle.write(
            json.dumps({"id": "x", "ts": 1, "type": "message"}) + "\n"
        )  # missing payload ok

    reopened = Transcript(directory)
    assert len(reopened.entries()) == 2  # corrupt line dropped, rest survives


@pytest.mark.asyncio
async def test_custom_entries_ignored_by_replay(transcript):
    await transcript.append_message(Message.user("hi"))
    await transcript.append_custom("wake_schedules", {"schedules": []})
    history = transcript.build_llm_history()
    assert len(history) == 1
    assert transcript.latest_custom("wake_schedules") == {"schedules": []}


@pytest.mark.asyncio
async def test_latest_custom_backward_scan(transcript):
    await transcript.append_custom("wake_schedules", {"v": 1})
    await transcript.append_custom("wake_schedules", {"v": 2})
    await transcript.append_custom("other", {"v": 99})
    assert transcript.latest_custom("wake_schedules") == {"v": 2}
    assert transcript.latest_custom("missing") is None


@pytest.mark.asyncio
async def test_compaction_replay_boundary(transcript):
    """Latest compaction wins: summary marker + entries from first_kept onward;
    nothing before the cut replays."""
    m1 = Message.user("early one")
    m2 = Message.user("early two")
    m3 = Message.user("kept")
    await transcript.append_message(m1)
    await transcript.append_message(m2)
    entry3 = await transcript.append_message(m3)
    await transcript.append_compaction("SUMMARY-TEXT", entry3.id, tokens_before=5000)
    m4 = Message.assistant("after")
    await transcript.append_message(m4)

    history = transcript.build_llm_history()
    assert len(history) == 3
    marker = history[0]
    assert isinstance(marker, CustomMessage)
    assert marker.custom_type == "compaction_summary"
    assert marker.details["summary"] == "SUMMARY-TEXT"
    assert isinstance(history[1], Message) and history[1].text == "kept"
    assert isinstance(history[2], Message) and history[2].text == "after"
    # Nothing before the cut leaked in.
    assert all(not (isinstance(m, Message) and "early" in m.text) for m in history)


@pytest.mark.asyncio
async def test_latest_compaction_wins(transcript):
    """Two compactions: only the newest boundary applies."""
    m1 = Message.user("a")
    e1 = await transcript.append_message(m1)
    m2 = Message.user("b")
    e2 = await transcript.append_message(m2)
    m3 = Message.user("c")
    await transcript.append_message(m3)

    await transcript.append_compaction("FIRST", e2.id, 100)
    await transcript.append_compaction("SECOND", e1.id, 200)

    history = transcript.build_llm_history()
    marker = history[0]
    assert isinstance(marker, CustomMessage)
    assert marker.details["summary"] == "SECOND"
    texts = [m.text for m in history[1:]]
    assert texts == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_compaction_with_missing_first_kept_replays_full_history(transcript):
    """If first_kept_entry_id cannot be found, replay falls back to the FULL
    history. The old fallback (compaction_index + 1) pointed past the kept
    window and silently dropped every message compaction promised to keep;
    replaying too much is recoverable at the next compaction, amnesia is not."""
    await transcript.append_message(Message.user("before"))
    await transcript.append_compaction("S", "no-such-entry", 10)
    await transcript.append_message(Message.user("after"))
    history = transcript.build_llm_history()
    assert [m.text for m in history if isinstance(m, Message)] == ["before", "after"]


@pytest.mark.asyncio
async def test_message_round_trip_keeps_provider_payload(transcript):
    """Tool-result metadata rides in provider_payload and must survive replay."""
    tool_msg = Message(
        role="tool",
        tool_call_id="c1",
        tool_name="read",
        provider_payload={"details": {"path": "/tmp/x"}, "useless": False},
    )
    await transcript.append_message(tool_msg)
    history = transcript.build_llm_history()
    assert len(history) == 1
    restored = history[0]
    assert isinstance(restored, Message)
    assert restored.provider_payload == {"details": {"path": "/tmp/x"}, "useless": False}
    assert restored.tool_call_id == "c1"


@pytest.mark.asyncio
async def test_custom_message_round_trip(transcript):
    custom = CustomMessage(custom_type="skill_prompt", details={"name": "s"})
    await transcript.append_message(custom)
    history = transcript.build_llm_history()
    assert len(history) == 1
    restored = history[0]
    assert isinstance(restored, CustomMessage)
    assert restored.custom_type == "skill_prompt"


def test_entry_from_json_rejects_bad_rows():
    assert TranscriptEntry.from_json("nonsense") is None
    assert (
        TranscriptEntry.from_json('{"id": "a", "type": "message"}') is not None
    )  # payload defaults
    assert TranscriptEntry.from_json('{"ts": 1, "type": "message"}') is None  # missing id


@pytest.mark.asyncio
async def test_append_recreates_a_directory_that_vanished_mid_session(tmp_path):
    """A deleted session directory must cost history, not the session.

    Retention sweeps this store, and an operator may clear it by hand. Before
    the recovery path, the append raised ``FileNotFoundError`` and kept raising
    it on every following turn, so one deleted directory left a session that
    could not take another turn at all.
    """
    import shutil

    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("before"))

    shutil.rmtree(directory)

    await transcript.append_message(Message.assistant("after"))

    assert transcript.path.is_file()
    lines = transcript.path.read_text().splitlines()
    # The WHOLE history, not just the line that triggered the recovery: a file
    # beginning part way through a conversation would replay as one.
    assert len(lines) == 2
    assert json.loads(lines[0])["payload"]["content"][0]["text"] == "before"
    assert json.loads(lines[1])["payload"]["content"][0]["text"] == "after"

    replayed = [m.content[0].text for m in Transcript(directory).build_llm_history()]
    assert replayed == ["before", "after"]
