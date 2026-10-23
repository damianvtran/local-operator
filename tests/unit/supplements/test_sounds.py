"""Generated audio reuses durable turn artifacts all the way to Highlights."""

from __future__ import annotations

import asyncio
from typing import Any, cast

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


def artifact(n: int = 0, **extra: Any) -> dict[str, Any]:
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
    # EVERY artifact, not a head slice: an 8-item batch is 8 sounds.
    assert len(items[0].artifacts) == 8
    payload = message.provider_payload
    assert payload is not None  # tool_result builds it from ToolResult.details
    payload["details"]["artifacts"][0]["name"] = "mutated"
    sounds = turn_sounds(items + items)
    assert len(sounds) == 8 and sounds[0] == {
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
    assert final.get("sounds") == sounds
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
    assert kept is not None
    # The wire strip carries the row's sounds; only the row's own bound truncates.
    assert kept["payload"]["details"]["sounds"] == sounds * 5
    # The event field is wire-typed dict[str, Any]; the stored row type is the
    # stricter TypedDict. Same dicts at runtime, so the cast is serialization-safe.
    event = SupplementProgressEvent(
        anchor="answer",
        job="audio-job",
        version=1,
        state="done",
        sounds=cast("list[dict[str, Any]]", sounds),
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


# ---------------------------------------------------------------------------
# Sound parity: every generated sound is an artifact, at every collapse state
# ---------------------------------------------------------------------------

#: The distinguishing fixture: one ``generate_sound_effect`` batch of 14, which is
#: the shape the operator reported. The old three-place cap (snapshot head slice,
#: ``turn_sounds`` early return, row write) turned it into 3 visible sounds.
BATCH_SOUNDS = 14


def sound_batch(count: int = BATCH_SOUNDS) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for n in range(count):
        # Distinct digests (``artifact``'s hex index) with a title the test can order by.
        raw = artifact(n)
        raw["name"] = f"Storm swell {n}"
        items.append(raw)
    return items


def batch_sounds() -> list[Any]:
    items = snapshot_messages([Message.tool_result(result(sound_batch(), "generate_sound_effect"))])
    return turn_sounds(items)


def test_fourteen_sounds_from_one_batch_reach_the_row_in_generation_order():
    items = snapshot_messages([Message.tool_result(result(sound_batch(), "generate_sound_effect"))])
    assert len(items[0].artifacts) == BATCH_SOUNDS
    sounds = turn_sounds(items)
    assert [sound["title"] for sound in sounds] == [f"Storm swell {n}" for n in range(BATCH_SOUNDS)]
    row = build_details(
        anchor="answer",
        job="audio-job",
        version=1,
        state="done",
        decision=Decision(),
        sounds=sounds,
    )
    assert [sound["attachment"] for sound in row.get("sounds", [])] == [
        f"{n:032x}" for n in range(BATCH_SOUNDS)
    ]
    assert "sounds_more" not in row

    # 15 entries = the 14 plus one continuation repeat of an already-seen digest:
    # dedupe by digest is not a cap, so the answer is still 14.
    repeated = turn_sounds(
        snapshot_messages(
            [Message.tool_result(result(sound_batch() + [artifact(0)], "generate_sound_effect"))]
        )
    )
    assert len(repeated) == BATCH_SOUNDS

    # The ONE bound is a bound, and it discloses itself instead of vanishing.
    from local_operator.supplements.contract import SOUNDS_MAX

    over = turn_sounds(
        snapshot_messages(
            [Message.tool_result(result(sound_batch(SOUNDS_MAX + 1), "generate_sound_effect"))]
        )
    )
    capped = build_details(
        anchor="answer", job="audio-job", version=1, state="done", decision=Decision(), sounds=over
    )
    assert len(capped.get("sounds") or []) == SOUNDS_MAX and capped.get("sounds_more") == 1


@pytest.mark.asyncio
async def test_fourteen_sounds_survive_the_live_beat_and_the_mobile_fold(tmp_path):
    from local_operator.mobile.projection import ProjectionFold
    from local_operator.mobile.types import SessionProjection

    sounds = batch_sounds()
    row = build_details(
        anchor="a" * 32,
        job="audio-job",
        version=1,
        state="done",
        decision=Decision(),
        sounds=sounds,
    )
    session = _make_session(tmp_path / "session", ScriptedStream([]), [], tmp_path)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    events: list[Any] = []
    session.subscribe(events.append)
    handle.subscribe(lambda: None)
    try:
        await handle._supplements._progress(row, "done")
    finally:
        await handle.dispose()
        await session.dispose()
    beats = [event for event in events if event.type == "supplement_progress"]
    assert len(beats) == 1
    live = beats[0]
    # The live wire carries all 14 -- the emit site must not re-slice either.
    assert [sound["title"] for sound in live.sounds] == [
        f"Storm swell {n}" for n in range(BATCH_SOUNDS)
    ]
    assert len(live.model_dump()["sounds"]) == BATCH_SOUNDS

    # The phone's live fold keeps them (the relay projection dropped the key before).
    fold = ProjectionFold(SessionProjection(session_id="s1", pid=1))
    fold.fold_event(live)
    held = fold.projection.supplements["a" * 32]
    assert [sound["attachment"] for sound in held["sounds"]] == [
        f"{n:032x}" for n in range(BATCH_SOUNDS)
    ]
    # A beat with no audio keeps the old bytes: the key is absent, not empty.
    silent = live.model_copy(update={"sounds": []})
    quiet = ProjectionFold(SessionProjection(session_id="s1", pid=1))
    quiet.fold_event(silent)
    assert "sounds" not in quiet.projection.supplements["a" * 32]


def test_a_fourteen_sound_row_survives_the_durable_cold_open(tmp_path, monkeypatch) -> None:
    from local_operator.mobile.daemon import _durable_projection
    from local_operator.session.transcript import Transcript
    from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE

    cfg = tmp_path / "config"
    root_dir = cfg / "sessions" / "root-session"
    root_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    row = build_details(
        anchor="a" * 32,
        job="audio-job",
        version=1,
        state="done",
        decision=Decision(),
        sounds=batch_sounds(),
    )
    transcript = Transcript(root_dir)
    asyncio.run(transcript.append_custom(SUPPLEMENT_CUSTOM_TYPE, dict(row)))
    projection = _durable_projection("root-session")
    assert projection is not None
    assert [sound["title"] for sound in projection.supplements["a" * 32]["sounds"]] == [
        f"Storm swell {n}" for n in range(BATCH_SOUNDS)
    ]
