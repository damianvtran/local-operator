"""Transcript tests: append-only JSONL, replay, and the compaction boundary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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
async def test_append_recreates_a_vanished_directory(tmp_path):
    """The session must survive its directory being deleted underneath it.

    A sibling process's startup sweep (or a user tidying ``sessions/`` by
    hand) can remove a directory that still looks empty — the gap between
    Session construction and the first turn is as long as the user takes to
    type. The old behaviour was fatal: the first append raised
    ``FileNotFoundError: .../transcript.jsonl`` and the whole session died.
    The append now recreates the directory and rebuilds the file from the
    in-memory entries, so nothing already appended is lost either.
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("before the deletion"))

    shutil.rmtree(directory)  # what the racing sweep used to do

    await transcript.append_message(Message.assistant("after the deletion"))

    lines = transcript.path.read_text().splitlines()
    assert len(lines) == 2  # rebuilt complete, not truncated to the new row
    texts = [json.loads(line)["payload"]["content"][0]["text"] for line in lines]
    assert texts == ["before the deletion", "after the deletion"]


@pytest.mark.asyncio
async def test_append_rebuilds_when_only_the_file_vanished(tmp_path):
    """The quieter variant of the vanished-directory wound (review R1-1).

    Deleting just ``transcript.jsonl`` with the directory intact never
    raises: ``"a"`` mode recreates the file, so the append "succeeds" while
    the file silently holds one row and memory holds the whole session — a
    resume would then replay a single message as if the rest never
    happened. The append must notice the file is gone and rebuild it
    complete from the in-memory entries.
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("one"))
    await transcript.append_message(Message.assistant("two"))

    transcript.path.unlink()  # the user tidied the file, not the directory

    await transcript.append_message(Message.user("three"))

    lines = transcript.path.read_text().splitlines()
    assert len(lines) == 3  # rebuilt complete, not restarted at one row
    texts = [json.loads(line)["payload"]["content"][0]["text"] for line in lines]
    assert texts == ["one", "two", "three"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["write", "flush"])
async def test_failed_append_never_enters_memory_index_or_later_rebuild(
    tmp_path, monkeypatch, failure_point
):
    """A rejected producer row must not resurrect through a later rebuild."""
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(
        Message.user("admitted", id="admitted"),
        producer_command_id="admitted",
    )
    real_open = Path.open
    fail_once = True

    class FailingHandle:
        def __init__(self, handle):  # noqa: ANN001
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):  # noqa: ANN002, ANN202
            return self._handle.__exit__(*args)

        def write(self, value):  # noqa: ANN001, ANN201
            if failure_point == "write":
                self._handle.write(value[: max(1, len(value) // 2)])
                raise OSError("injected write failure")
            return self._handle.write(value)

        def flush(self):
            if failure_point == "flush":
                raise OSError("injected flush failure")
            return self._handle.flush()

        def fileno(self):
            return self._handle.fileno()

    def failing_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
        nonlocal fail_once
        handle = real_open(path, *args, **kwargs)
        if fail_once and path == transcript.path and args and args[0] == "a":
            fail_once = False
            return FailingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match=failure_point):
        await transcript.append_message(
            Message.user("failed", id="failed"),
            producer_command_id="failed",
        )

    assert [entry.id for entry in transcript.entries()] == ["admitted"]
    assert not transcript.has_admitted_command("failed")
    assert [entry.id for entry in Transcript(directory).entries()] == ["admitted"]

    transcript.path.unlink()
    await transcript.append_message(
        Message.user("later", id="later"),
        producer_command_id="later",
    )
    reopened = Transcript(directory)
    assert [entry.id for entry in reopened.entries()] == ["admitted", "later"]
    assert not reopened.has_admitted_command("failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("command_kind", ["prompt", "steer"])
async def test_failed_first_rebuild_is_retryable_without_resurrection(
    tmp_path, monkeypatch, command_kind
):
    """An open failure before the first row leaves no disk or memory claim."""
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    real_open = Path.open
    fail_once = True

    def failing_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
        nonlocal fail_once
        if fail_once and path == transcript.path and args and args[0] == "w":
            fail_once = False
            raise OSError("injected open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    command_id = f"retry-{command_kind}"
    failed = Message.user("failed", id=command_id)
    with pytest.raises(OSError, match="open failure"):
        await transcript.append_message(failed, producer_command_id=command_id)

    assert transcript.entries() == []
    assert not transcript.path.exists()
    assert not transcript.has_admitted_command(command_id)

    await transcript.append_message(failed, producer_command_id=command_id)
    reopened = Transcript(directory)
    assert [entry.id for entry in reopened.entries()] == [command_id]
    assert reopened.has_admitted_command(command_id)


def test_only_valid_user_message_rows_claim_producer_markers(tmp_path) -> None:
    directory = tmp_path / "sess"
    directory.mkdir()
    valid_payload = {
        "kind": "message",
        "role": "user",
        "content": [{"text": "valid"}],
        "producer_command_id": "valid",
    }
    invalid_payloads = [
        {**valid_payload, "producer_command_id": "missing-kind", "kind": None},
        {**valid_payload, "producer_command_id": "custom-kind", "kind": "custom"},
        {**valid_payload, "producer_command_id": "assistant", "role": "assistant"},
        {**valid_payload, "producer_command_id": "system", "role": "system"},
        {**valid_payload, "producer_command_id": "malformed-content", "content": "text"},
        {**valid_payload, "producer_command_id": "malformed-block", "content": [42]},
        {**valid_payload, "producer_command_id": "   "},
    ]
    entries = [
        TranscriptEntry(id="valid-row", ts=1, type=ENTRY_MESSAGE, payload=valid_payload),
        *[
            TranscriptEntry(
                id=f"invalid-{index}", ts=2 + index, type=ENTRY_MESSAGE, payload=payload
            )
            for index, payload in enumerate(invalid_payloads)
        ],
        TranscriptEntry(
            id="import-collision",
            ts=20,
            type="custom",
            payload={**valid_payload, "producer_command_id": "import-collision"},
        ),
    ]
    (directory / "transcript.jsonl").write_text(
        "".join(entry.to_json() + "\n" for entry in entries), encoding="utf-8"
    )

    transcript = Transcript(directory)

    assert transcript.has_admitted_command("valid")
    for command_id in [
        "missing-kind",
        "custom-kind",
        "assistant",
        "system",
        "malformed-content",
        "malformed-block",
        "import-collision",
    ]:
        assert not transcript.has_admitted_command(command_id)


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
