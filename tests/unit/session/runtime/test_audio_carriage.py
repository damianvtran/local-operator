"""The recording's carriage through the serving handle: wire dicts -> blocks.

``ServingSessionHandle.prompt``/``steer`` decode the wire's shape
(``[{"data_b64", "mime_type"}]``) with ``audio_blocks`` — the ingest images
already use — and hand ``Session`` decoded blocks. These tests pin that seam
against a REAL session: a LYING declared mime is corrected from the bytes (the
OpenAI-compatible part keys its ``input_audio.format`` token on the stored
mime), the durable row carries the corrected block and the daemon annotation,
and the ``ChatRequest`` the turn builds carries the ``AudioContent`` — the
"Prompt -> durable row -> live event -> ChatRequest" round trip of design §5,
minus only the HTTP hop QA exercises separately.

The steer case is the same carriage on the other op: a mid-turn recording lands
on the queued row when the boundary drain persists it, so nothing is lost at a
hop whose signature takes the keyword.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AudioContent,
    Message,
    ModelSpec,
    StreamEndEvent,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_session import ScriptedStream, make_session

AUDIO_MODEL = ModelSpec(
    provider="test",
    model_id="m",
    context_window=100_000,
    supports_audio_input=True,
)

#: EBML magic — a webm capture, the browser recorder's native container.
WEBM = base64.b64encode(b"\x1a\x45\xdf\xa3" + b"\x00" * 64).decode("ascii")


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the forked sidecar's auth store inside the test's tmp tree."""
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


def _message_rows(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "sess" / "transcript.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line).get("type") == "message"
    ]


@pytest.mark.asyncio
async def test_a_wire_recording_reaches_the_row_and_the_request_with_its_sniffed_mime(
    tmp_path: Path,
) -> None:
    """One submit, four readings: receipt, durable row, live event, request.

    The declared mime is a LIE (wav for webm bytes) on purpose: the correction
    is the property under test, because the model wire's ``format`` token comes
    from the block's mime and a lie that survives admission is a provider 400.
    """
    stream = ScriptedStream(
        [
            [StreamEndEvent(stop_reason="stop")],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, model=AUDIO_MODEL)
    events: list[Any] = []
    session.subscribe(events.append)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))

    receipt = await handle.prompt(
        "listen",
        audio=[{"data_b64": WEBM, "mime_type": "audio/wav"}],
        command_id="audio-1",
    )
    assert receipt, "the admission receipt must be the durable append's answer"
    await _wait_for(lambda: not session.is_streaming and bool(stream.requests))

    user_row = next(
        row["payload"] for row in _message_rows(tmp_path) if row["payload"].get("role") == "user"
    )
    block = user_row["content"][-1]
    assert block["type"] == "audio"
    assert block["mime_type"] == "audio/webm", "the row must carry the SNIFFED container"
    assert user_row["input_path"] == "model_audio_sidecar"

    starts = [
        event
        for event in events
        if getattr(event, "type", "") == "message_start"
        and getattr(getattr(event, "message", None), "role", "") == "user"
    ]
    assert starts, "the live stream must carry the user row"
    live_audio = [b for b in starts[-1].message.content if isinstance(b, AudioContent)]
    assert [b.mime_type for b in live_audio] == ["audio/webm"]

    # The TURN's request, not simply request zero: the handle may also have
    # produced a naming call, and identifying the request by the recording it
    # carries is what keeps this assertion about the feature rather than about
    # call ordering.
    carrying = [
        request
        for request in stream.requests
        if any(
            isinstance(block, AudioContent)
            for message in request.messages
            if isinstance(message, Message)
            for block in message.content
        )
    ]
    assert carrying, "the turn's request must carry the recording"
    sent = carrying[0].messages[-1]
    assert isinstance(sent, Message)
    assert sent.input_path == "model_audio_sidecar"
    assert [b.mime_type for b in sent.content if isinstance(b, AudioContent)] == ["audio/webm"]

    await session.dispose()


@pytest.mark.asyncio
async def test_a_steered_recording_lands_on_the_queued_row(tmp_path: Path) -> None:
    """The steer op carries the recording too, through the same decode.

    Steers persist at the next boundary drain rather than at the ack; the drain
    is called directly here because that is exactly the call the loop makes, and
    it keeps this test off a tool-scripted turn.
    """
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, model=AUDIO_MODEL)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))

    assert (
        await handle.steer(
            "mid-turn recording",
            audio=[{"data_b64": WEBM, "mime_type": "audio/webm"}],
            command_id="steer-audio-1",
        )
        == "steering queued"
    )

    delivered = await session._drain_steering()
    assert [m.text for m in delivered if isinstance(m, Message)] == ["mid-turn recording"]

    user_rows = [
        r["payload"] for r in _message_rows(tmp_path) if r["payload"].get("role") == "user"
    ]
    assert len(user_rows) == 1
    block = user_rows[0]["content"][-1]
    assert block["type"] == "audio"
    assert block["mime_type"] == "audio/webm"
    assert user_rows[0]["input_path"] == "model_audio_sidecar"

    await session.dispose()
