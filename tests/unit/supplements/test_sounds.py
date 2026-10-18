"""Generated audio reuses durable turn artifacts all the way to Highlights."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.types import (
    AgentTool,
    Message,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    SupplementProgressEvent,
    TextContent,
    ToolResult,
)
from local_operator.session.open_frame import strip_entry
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.supplements.contract import reader_disposition
from local_operator.supplements.decision import Decision
from local_operator.supplements.persistence import build_details, next_version
from local_operator.supplements.trigger import snapshot_messages, turn_sounds
from tests.unit.session.test_session import ScriptedStream
from tests.unit.supplements.test_runner import _await_row, _make_session


def artifact(n=0, **extra):
    return dict(
        digest=f"{n:032x}",
        kind="audio",
        content_type="audio/mpeg",
        name="Voiceover",
        duration_s=2.5,
        **extra,
    )


def result(artifacts, tool="generate_voiceover"):
    return ToolResult(
        tool_call_id="audio-call",
        tool_name=tool,
        content=[TextContent(text="Generated audio")],
        details={"artifacts": artifacts},
    )


def test_audio_snapshots_reuse_bounded_facts_not_text_or_other_tools():
    audio = result([artifact(n) for n in range(8)])
    message = Message.tool_result(audio)
    items = snapshot_messages([message])
    assert len(items[0].artifacts) == 3
    message.provider_payload["details"]["artifacts"][0]["name"] = "mutated"
    sounds = turn_sounds(items + items)
    assert len(sounds) == 3 and sounds[0] == {
        "attachment": "0" * 32,
        "mime": "audio/mpeg",
        "title": "Voiceover",
        "duration_s": 2.5,
    }
    assert turn_sounds(snapshot_messages([Message.tool_result(result([artifact()], "bash"))])) == []
    assert turn_sounds(snapshot_messages([Message.assistant("Generated audio")])) == []


def test_unknown_duration_and_missing_title_are_safe_and_invalid_digests_are_dropped():
    raw = artifact()
    raw.update(name=None, duration_s=None)
    items = snapshot_messages(
        [Message.tool_result(result([raw, {"kind": "audio", "digest": "bad"}]))]
    )
    assert turn_sounds(items)[0]["title"] == "Generated audio"
    assert turn_sounds(items)[0]["duration_s"] == 0.0


@pytest.mark.parametrize(
    "state,live,expected",
    [
        ("done", False, "block"),
        ("decided", True, "files_preparing"),
        ("failed", False, "files_failed_retry"),
    ],
)
def test_sound_only_row_is_nonempty_and_survives_open_frame_and_versions(state, live, expected):
    sounds = turn_sounds(snapshot_messages([Message.tool_result(result([artifact()]))]))
    row = build_details(
        anchor="answer",
        job="audio-job",
        version=1,
        state="decided",
        decision=Decision(),
        sounds=sounds,
    )
    final = next_version(row, state=state)
    assert final["sounds"] == sounds
    assert reader_disposition(final, job_live=live) == expected
    kept = strip_entry(
        {
            "id": "row",
            "type": "custom",
            "payload": {
                "custom_type": "supplement_v1",
                "details": {**final, "sounds": sounds * 5},
            },
        },
        supplements=True,
    )
    assert kept["payload"]["details"]["sounds"] == sounds * 3
    event = SupplementProgressEvent(
        anchor="answer", job="audio-job", version=1, state="done", sounds=sounds
    )
    assert event.model_dump()["sounds"] == sounds
    empty = build_details(anchor="answer", job="job", version=1, state="done", decision=Decision())
    assert "sounds" not in empty and reader_disposition(empty, job_live=False) == "nothing"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["generate_voiceover", "generate_sound_effect"])
async def test_real_turn_audio_result_journals_sound_only_done_row_and_live_beat(
    tmp_path, tool_name
):
    async def execute(tool_call_id, args, signal, on_update, context):
        return result([artifact()], tool_name).model_copy(update={"tool_call_id": tool_call_id})

    tool = AgentTool(
        name=tool_name,
        parameters={"type": "object", "properties": {}},
        interruptible=True,
        execute=execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="audio-call", name=tool_name, argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Here is your audio."), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = _make_session(tmp_path / "session", stream, [tool], tmp_path)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    events = []
    session.subscribe(events.append)
    handle.subscribe(lambda: None)
    try:
        await session.prompt("Generate a short audio clip")
        row = await _await_row(session.transcript)
        assert row["state"] == "done" and row["files"] == row["components"] == []
        assert row["sounds"] == [
            {"attachment": "0" * 32, "mime": "audio/mpeg", "title": "Voiceover", "duration_s": 2.5}
        ]
        assert reader_disposition(row, job_live=False) == "block"
        beats = [event for event in events if event.type == "supplement_progress"]
        assert beats[-1].sounds == row["sounds"]
    finally:
        await handle.dispose()
        await session.dispose()


@pytest.mark.asyncio
async def test_graphics_decided_and_terminal_rows_restate_the_same_turn_sounds(
    tmp_path, monkeypatch
):
    from tests.unit.supplements.test_runner_generator import (
        BLOCK,
        DATASETS,
        StubRunner,
        _decision,
        _drain,
        _handle,
        _rows,
        _settings,
    )

    session = _make_session(tmp_path / "session", ScriptedStream([]), [], tmp_path)
    handle = await _handle(session, tmp_path)
    StubRunner(handle._supplements, answers=[BLOCK]).install(monkeypatch)
    sounds = turn_sounds(snapshot_messages([Message.tool_result(result([artifact()]))]))
    try:
        handle._supplements._render(
            "answer", _decision(), DATASETS, "chart", "facts", _settings(), sounds=sounds
        )
        await _drain(handle)
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "done"]
        assert all(row["sounds"] == sounds for row in rows)
    finally:
        await handle.dispose()
        await session.dispose()
