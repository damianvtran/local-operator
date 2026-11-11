"""Settled background music reaches its PRODUCING turn's highlights row.

The defect (UI native run 42949407400): highlights are decided at turn end, when a
``generate_music`` result is only ``{"stage": "queued", "job_id": ...}``; the tracks land
minutes later in the job's ``music_receipt`` and never reached ``sounds[]``. These tests
drive the real session + runtime handle: the turn records ``music_jobs`` on its row, and the
settle (``Session._on_job_completed``) merges the tracks in, idempotently, with a live beat.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentTool,
    Message,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.supplements import SupplementRunner
from local_operator.supplements.contract import (
    SUPPLEMENT_CUSTOM_TYPE,
    reader_disposition,
)
from local_operator.supplements.trigger import (
    music_sounds,
    snapshot_messages,
    turn_music_jobs,
)
from tests.unit.session.test_session import ScriptedStream, wait_for
from tests.unit.supplements.test_runner import _await_row, _make_session

JOB = "a1b2c3d4"


def track_artifact(n: int, name: str) -> dict[str, Any]:
    """One ``artifact_details`` entry exactly as ``music/jobs.py::_register`` writes it."""
    return {
        "digest": f"{n:032x}",
        "kind": "audio",
        "name": name,
        "content_type": "audio/mpeg",
        "size_bytes": 2_400_000,
        "width": None,
        "height": None,
        "duration_s": 61.4,
        "material": "music",
        "track_id": f"trk_{n}",
        "artifact_id": f"art_{n}",
        "role": "primary",
        "variant_index": n,
        "title": "Rainy Lo-fi",
        "operation": "compose",
        "attribution": "",
    }


def receipt(*artifacts: dict[str, Any], job: str = JOB) -> dict[str, Any]:
    """The runner's terminal ``music_receipt``: two tracks, one failed upstream."""
    return {
        "job": job,
        "material": "music",
        "model": "radient/music",
        "provider": "radient",
        "tracks": [
            {"id": "trk_0", "status": "completed", "settled": True, "variant_index": 0},
            {"id": "trk_1", "status": "failed", "settled": True, "error_type": "provider_error"},
        ],
        "artifacts": list(artifacts),
        "attribution": "",
    }


def queued(tool_call_id: str, job: str = JOB) -> ToolResult:
    """``music_tool._start``'s real result shape: no bytes yet, only the job id."""
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name="generate_music",
        content=[TextContent(text=f"Music job {job} started.")],
        details={
            "job_id": job,
            "stage": "queued",
            "material": "music",
            "model": "radient/music",
            "tracks": [],
            "cancel_support": "queued_only",
        },
    )


def _rows(transcript: Any) -> list[dict[str, Any]]:
    rows = []
    for entry in transcript.entries():
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        if entry.type == "custom" and payload.get("custom_type") == SUPPLEMENT_CUSTOM_TYPE:
            rows.append(payload["details"])
    return rows


def _settled_job(data: dict[str, Any]) -> Any:
    # ``consumed`` = the model already collected it with ``wait``: the highlights merge
    # must still run (it sits before that gate), and no delivery turn starts.
    return SimpleNamespace(type="music", consumed=True, latest_details={"music_receipt": data})


def test_receipt_tracks_become_sounds_failed_and_foreign_entries_excluded():
    data = receipt(
        track_artifact(0, "rainy-lo-fi-1.mp3"),
        track_artifact(0, "rainy-lo-fi-1.mp3"),  # same bytes reported twice: one sound
        {"kind": "video", "digest": "f" * 32, "name": "clip.mp4"},
        {"kind": "audio", "digest": "not-a-digest", "name": "x.mp3"},
    )
    assert music_sounds(data) == [
        {
            "attachment": "0" * 32,
            "mime": "audio/mpeg",
            "title": "rainy-lo-fi-1.mp3",
            "duration_s": 61.4,
            "origin": "music",
        }
    ]
    assert music_sounds(receipt()) == []  # every track failed: nothing playable


def test_turn_music_jobs_reads_queued_results_and_skips_errors():
    ok = Message.tool_result(queued("c1"))
    again = Message.tool_result(queued("c2"))  # a continuation re-reporting the same job
    failed = Message.tool_result(queued("c3", job="zzz").model_copy(update={"is_error": True}))
    assert turn_music_jobs(snapshot_messages([ok, again, failed])) == [JOB]


def _music_turn(tmp_path, on_execute=None):
    async def execute(tool_call_id, args, signal, on_update, context):
        if on_execute is not None:
            await on_execute()
        return queued(tool_call_id)

    tool = AgentTool(
        name="generate_music",
        parameters={"type": "object", "properties": {}},
        interruptible=True,
        execute=execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(
                    index=0, id="music-call", name="generate_music", argument_delta="{}"
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Your track is composing."), StreamEndEvent(stop_reason="stop")],
        ]
    )
    return _make_session(tmp_path / "session", stream, [tool], tmp_path)


@pytest.mark.asyncio
async def test_settled_music_merges_into_the_producing_turn_row_with_a_live_beat(tmp_path):
    session = _music_turn(tmp_path)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    events: list[Any] = []
    session.subscribe(events.append)
    handle.subscribe(lambda: None)
    try:
        await session.prompt("Make me a rainy lo-fi track")
        first = await _await_row(session.transcript)
        # The link row: written though it shows nothing yet, and renders nothing.
        assert first["music_jobs"] == [JOB] and "sounds" not in first
        assert reader_disposition(first, job_live=False) == "nothing"
        anchor = first["anchor"]

        settled = receipt(track_artifact(0, "rainy-lo-fi-1.mp3"))
        await session._on_job_completed(JOB, "ready", _settled_job(settled))
        await wait_for(lambda: len(_rows(session.transcript)) == 2, timeout=10)
        row = _rows(session.transcript)[-1]
        assert row["anchor"] == anchor and row["state"] == "done"
        assert row["version"] == first["version"] + 1
        assert [(s["attachment"], s["title"]) for s in row["sounds"]] == [
            ("0" * 32, "rainy-lo-fi-1.mp3")
        ]
        assert reader_disposition(row, job_live=False) == "block"
        # LIVE: the same family a surface already paints from, carrying the row's sounds.
        beats = [e for e in events if e.type == "supplement_progress"]
        assert beats[-1].anchor == anchor and beats[-1].sounds == row["sounds"]

        # Replay / reconnect / recovered observer: identical receipt writes nothing.
        await handle._supplements.on_music_settled(settled)
        # A failed-only job writes nothing either.
        await handle._supplements.on_music_settled(receipt())
        assert len(_rows(session.transcript)) == 2

        # After a restart the in-memory map is empty: the journal's link still finds the
        # anchor, and a second job's track is appended without duplicating the first.
        fresh = SupplementRunner(session, cwd=str(tmp_path))
        await fresh.on_music_settled(
            receipt(track_artifact(0, "rainy-lo-fi-1.mp3"), track_artifact(1, "rainy-lo-fi-2.mp3"))
        )
        last = _rows(session.transcript)[-1]
        assert [s["attachment"] for s in last["sounds"]] == ["0" * 32, f"{1:032x}"]
    finally:
        await handle.dispose()
        await session.dispose()


@pytest.mark.asyncio
async def test_music_settling_before_the_turn_ends_lands_in_its_first_row(tmp_path):
    handles: list[ServingSessionHandle] = []

    async def settle_mid_turn() -> None:
        # The job settles while the producing turn is still streaming: no row exists to
        # merge into, so the turn-end build must pick the parked tracks up itself.
        await handles[0]._supplements.on_music_settled(
            receipt(track_artifact(0, "rainy-lo-fi-1.mp3"))
        )

    session = _music_turn(tmp_path, settle_mid_turn)
    handles.append(
        ServingSessionHandle(
            session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
        )
    )
    handles[0].subscribe(lambda: None)
    try:
        await session.prompt("Make me a rainy lo-fi track")
        row = await _await_row(session.transcript)
        assert row["music_jobs"] == [JOB]
        assert [s["title"] for s in row["sounds"]] == ["rainy-lo-fi-1.mp3"]
        assert len(_rows(session.transcript)) == 1
    finally:
        await handles[0].dispose()
        await session.dispose()


@pytest.mark.asyncio
async def test_graphics_path_links_music_and_carries_merged_tracks_to_its_terminal_row(
    tmp_path, monkeypatch
):
    from tests.unit.supplements.test_runner_generator import (
        BLOCK,
        DATASETS,
        StubRunner,
        _decision,
        _drain,
        _handle,
        _settings,
    )

    session = _make_session(tmp_path / "session", ScriptedStream([]), [], tmp_path)
    handle = await _handle(session, tmp_path)
    StubRunner(handle._supplements, answers=[BLOCK]).install(monkeypatch)
    runner = handle._supplements
    try:
        # Parked before the row exists, as when the job settles mid-turn.
        await runner.on_music_settled(receipt(track_artifact(0, "rainy-lo-fi-1.mp3")))
        runner._render(
            "answer", _decision(), DATASETS, "chart", "facts", _settings(), music_jobs=(JOB,)
        )
        await _drain(handle)
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "done"]
        for row in rows:
            assert row["music_jobs"] == [JOB]
            assert [s["title"] for s in row["sounds"]] == ["rainy-lo-fi-1.mp3"]
    finally:
        await handle.dispose()
        await session.dispose()
