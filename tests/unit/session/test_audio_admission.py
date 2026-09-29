"""The recording door at admission: gate, annotation, fork ordering, dispose.

Phase 1 of the STT cascade wires ``audio=`` from the wire to ``Session.prompt``
(``Message.user``'s third attachment slot). These tests hold the four facts a
later edit can break independently:

* the CAPABILITY GATE refuses a recording the selected model cannot take, with
  the typed refusal (``AudioInputUnsupported``) that names the model and carries
  the resolver's report — the honest error — before any paid work;
* the FORMAT GATE (review round 1 remediation) refuses, at the same point, a
  capture the capable model's WIRE cannot carry (OQ-3: no transcoding) — a
  webm capture must never become the sticky every-request wedge QA reproduced
  — while wav/mp3 on the chat wire and Gemini's documented containers pass;
* BOTH DOORS fork the sidecar once per recording — the prompt loop and the
  steer drain — and only after their own append made the row durable;
* the ANNOTATION is daemon-derived: an audio row carries
  ``input_path="model_audio_sidecar"`` whatever a surface asserted, and a
  non-audio row keeps the surface-asserted value (the carriage's own contract,
  unchanged);
* the SIDECAR forks ONLY after the row is durable (``has_entry`` is True at
  fork time) and a disposed session cancels it;
* the durable row round-trips: the block, the annotation, the live
  ``message_start`` event and the ``ChatRequest`` the loop builds.

The session rig is ``tests.unit.session.test_session.make_session`` — a real
``Session`` with a scripted provider stream; only the stream is fake.
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
    MessageStartEvent,
    ModelSpec,
    StreamEndEvent,
)
from local_operator.session.errors import AudioInputUnsupported

from .test_session import MODEL, ScriptedStream, make_session

#: The tape-model spec: ``supports_audio_input`` False (the field's safe
#: default), so every gate test below is about the DEFAULT, not a configuration.
AUDIO_MODEL = ModelSpec(
    provider="test",
    model_id="m",
    context_window=100_000,
    supports_audio_input=True,
)

#: A capable model on the OpenAI-compatible CHAT wire — the wire whose
#: ``input_audio`` part takes wav|mp3 only (OQ-3). Used to pin the admission
#: format gate against the real registry wire lookup.
OPENAI_AUDIO_MODEL = ModelSpec(
    provider="openai",
    model_id="gpt-audio",
    context_window=128_000,
    supports_audio_input=True,
)

#: A capable model on the Google wire, whose inline audio part takes Gemini's
#: documented mime list (stt/audio.py, verified against the vendor page).
GOOGLE_AUDIO_MODEL = ModelSpec(
    provider="google",
    model_id="gemini-3.8-flash",
    context_window=1_048_576,
    supports_audio_input=True,
)

#: A minimal RIFF/WAVE capture — the sniffer-verifiable shape, small on purpose.
WAV = base64.b64encode(b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 32).decode("ascii")


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sidecar builds its own auth store from ``config_dir()``.

    Point that at the test's tmp tree (created up front — the resolver and the
    sidecar both open a store at that path) so a real session's fork can never
    touch the developer's (or CI runner's) live config, the same isolation the
    resolver docstring describes.
    """
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))


def _audio() -> list[AudioContent]:
    return [AudioContent(data=WAV, mime_type="audio/wav")]


def _message_rows(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "sess" / "transcript.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line).get("type") == "message"
    ]


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_a_recording_for_an_incapable_model_is_refused_before_any_work(tmp_path):
    """The rung-6 honest error: typed, named, and free.

    ``MODEL`` carries the field's default (False), so this is the backstop path
    a surface reaches by sending audio without asking the resolver first. The
    refusal must name the model and carry the resolver's report, and it must
    cost NOTHING: no provider request, no durable row.
    """
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, model=MODEL)

    with pytest.raises(AudioInputUnsupported) as excinfo:
        await session.prompt("listen", audio=_audio())

    sentence = str(excinfo.value)
    assert "test/m" in sentence
    assert "Resolver report:" in sentence
    assert "does not accept audio input" in sentence
    assert stream.requests == [], "the refused send must not spend a provider call"
    assert _message_rows(tmp_path) == [], "the refused send must not leave a row"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_container_the_wire_cannot_carry_is_refused_at_admission(tmp_path):
    """The OQ-3 rung-unavailable error, at the door (agent review round 1, M2).

    QA reproduced the wedge with a webm capture — the browser recorder's
    default — on a capable model: admission wrote the row, the chat renderer
    then raised ``WireCannotCarryAudio`` on the first send AND every request
    after it (zero HTTP calls, sticky), because nothing evicts a durable
    block. Admission now asks the renderers' own question BEFORE the write, so
    the refusal is typed, names the model and the constraint, and costs
    nothing — and no row can be the start of a wedge.
    """
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, model=OPENAI_AUDIO_MODEL)

    webm = [AudioContent(data=WAV, mime_type="audio/webm")]
    with pytest.raises(AudioInputUnsupported) as excinfo:
        await session.prompt("listen", audio=webm)

    sentence = str(excinfo.value)
    assert "openai/gpt-audio" in sentence, sentence
    assert "wav or mp3" in sentence, sentence
    assert "audio/webm" in sentence, sentence
    assert "No transcoding" in sentence, sentence
    assert stream.requests == [], "the refused send must not spend a provider call"
    assert _message_rows(tmp_path) == [], "the refused send must not leave a row"
    await session.dispose()


@pytest.mark.asyncio
async def test_wav_and_mp3_sail_through_the_chat_wire_gate(tmp_path):
    """The control for the format gate: both containers the wire takes pass.

    Without this, a gate that refused everything would look identical to a
    working one from the webm test alone.
    """
    stream = ScriptedStream(
        [[StreamEndEvent(stop_reason="stop")], [StreamEndEvent(stop_reason="stop")]]
    )
    session = make_session(tmp_path, stream, model=OPENAI_AUDIO_MODEL)

    await session.prompt("wav", audio=_audio())
    await session.prompt("mp3", audio=[AudioContent(data=WAV, mime_type="audio/mpeg")])
    await _wait_for(lambda: not session.is_streaming and len(_message_rows(tmp_path)) >= 2)

    user_rows = [
        r["payload"] for r in _message_rows(tmp_path) if r["payload"].get("role") == "user"
    ]
    # ``exclude_defaults`` drops the wav block's default ``mime_type`` and the
    # text blocks' default ``type`` from the durable row, so read both with
    # their model defaults (the row is still byte-identical to a pre-audio one
    # for text-only sends — that contract is pinned elsewhere).
    mimes = [
        block.get("mime_type", "audio/wav")
        for block in user_rows[0]["content"]
        if block.get("type") == "audio"
    ]
    assert mimes == ["audio/wav"]
    mimes = [
        block.get("mime_type", "audio/wav")
        for block in user_rows[1]["content"]
        if block.get("type") == "audio"
    ]
    assert mimes == ["audio/mpeg"]
    await session.dispose()


@pytest.mark.asyncio
async def test_the_google_wire_takes_its_documented_list(tmp_path):
    """Gemini's inline part: webm is ON the documented list; audio/mp4 is not.

    The vendor's page spells the M4A container ``audio/m4a`` and not
    ``audio/mp4``, and admission mirrors the page rather than guessing family
    aliases (v1 renames no declarations) — the m4a family still reaches Gemini
    through the STT rungs.
    """
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, model=GOOGLE_AUDIO_MODEL)

    await session.prompt("webm", audio=[AudioContent(data=WAV, mime_type="audio/webm")])
    await _wait_for(lambda: not session.is_streaming)

    with pytest.raises(AudioInputUnsupported) as excinfo:
        await session.prompt("m4a", audio=[AudioContent(data=WAV, mime_type="audio/mp4")])

    sentence = str(excinfo.value)
    assert "audio/mp4" in sentence, sentence
    assert "does not take" in sentence, sentence
    await session.dispose()


@pytest.mark.asyncio
async def test_the_audio_door_annotates_the_row_and_round_trips_to_the_request(tmp_path):
    """Prompt → durable row → live event → ChatRequest, on the real session.

    The row is read from disk (not from the message object), the live event is
    what every subscriber paints, and the ChatRequest is what the provider
    actually receives — the three carriers the annotation contract names.
    """
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, model=AUDIO_MODEL)
    events: list[Any] = []
    session.subscribe(events.append)

    await session.prompt("listen", audio=_audio())
    await _wait_for(lambda: not session.is_streaming)

    rows = _message_rows(tmp_path)
    user = [r for r in rows if r["payload"].get("role") == "user"]
    assert len(user) == 1, rows
    payload = user[0]["payload"]
    assert payload["input_path"] == "model_audio_sidecar"
    block = payload["content"][-1]
    assert block["type"] == "audio", "the discriminant rides the durable row"
    assert block["data"] == WAV

    started = [
        e
        for e in events
        if isinstance(e, MessageStartEvent)
        and isinstance(e.message, Message)
        and any(isinstance(b, AudioContent) for b in e.message.content)
    ]
    assert started, "the user row must be announced with its recording"
    announced = started[0].message
    assert isinstance(announced, Message)
    assert announced.input_path == "model_audio_sidecar"

    sent = stream.requests[0].messages[-1]
    assert sent.role == "user"
    assert sent.input_path == "model_audio_sidecar"
    assert [b.mime_type for b in sent.content if isinstance(b, AudioContent)] == ["audio/wav"]
    await session.dispose()


@pytest.mark.asyncio
async def test_the_daemon_stamp_wins_and_plain_sends_keep_the_client_value(tmp_path):
    """``model_audio_sidecar`` is daemon-derived; other values stay client-asserted.

    An audio row's route slot is a fact the daemon is holding (the blocks are on
    the message), so a client assertion cannot overwrite it. A send WITHOUT
    audio is untouched: its ``input_path`` stays whatever the surface said,
    which is the carriage's contract for text-from-speech sends.
    """
    stream = ScriptedStream(
        [[StreamEndEvent(stop_reason="stop")], [StreamEndEvent(stop_reason="stop")]]
    )
    session = make_session(tmp_path, stream, model=AUDIO_MODEL)

    await session.prompt("listen", audio=_audio(), input_path="provider_stt_radient")
    await session.prompt("typed", input_path="provider_stt_elevenlabs")
    await _wait_for(lambda: not session.is_streaming and len(_message_rows(tmp_path)) >= 2)

    user_rows = [
        r["payload"] for r in _message_rows(tmp_path) if r["payload"].get("role") == "user"
    ]
    assert user_rows[0]["input_path"] == "model_audio_sidecar"
    assert user_rows[1]["input_path"] == "provider_stt_elevenlabs"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_sidecar_forks_only_after_the_row_is_durable(tmp_path, monkeypatch):
    """The ordering proof: at fork time, the row is already on disk.

    ``fork_audio_sidecar`` is spied, not faked, so the assertion is about WHEN
    the call fires: a record for a message id the transcript does not have would
    dangle, and a prompt that dies before admission must leave nothing behind.
    The non-audio send proves the fork is not on some broader path.
    """
    seen: list[dict[str, Any]] = []

    def spy(session, *, message_id, audio, config_dir, store):
        seen.append(
            {
                "durable": session.transcript.has_entry(message_id),
                "message_id": message_id,
                "mime": audio.mime_type,
            }
        )

    monkeypatch.setattr("local_operator.stt.sidecar.fork_audio_sidecar", spy)
    stream = ScriptedStream(
        [[StreamEndEvent(stop_reason="stop")], [StreamEndEvent(stop_reason="stop")]]
    )
    session = make_session(tmp_path, stream, model=AUDIO_MODEL)

    await session.prompt("listen", audio=_audio())
    await session.prompt("no recording here")
    await _wait_for(lambda: not session.is_streaming)

    assert len(seen) == 1, "one fork for one recording, none for the text send"
    assert seen[0]["durable"] is True, "the fork ran before the append landed"
    assert seen[0]["mime"] == "audio/wav"
    assert session.transcript.has_entry(seen[0]["message_id"])
    await session.dispose()


@pytest.mark.asyncio
async def test_dispose_cancels_a_forked_sidecar(tmp_path, monkeypatch):
    """The fork rides the session's own tracked spawn, so dispose owns it.

    ``_model_transcript`` is held open (no network, no clock), which is the only
    way to observe the task WHILE it is pending; dispose must then cancel it
    rather than leave work running past the session it records for.
    """
    started = asyncio.Event()
    never = asyncio.Event()

    async def hang(session, audio, *, store, timeout_s):
        started.set()
        await never.wait()

    monkeypatch.setattr("local_operator.stt.sidecar._model_transcript", hang)
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, model=AUDIO_MODEL)

    await session.prompt("listen", audio=_audio())
    await asyncio.wait_for(started.wait(), 5)
    pending = [task for task in session._background_tasks if not task.done()]
    assert pending, "the forked sidecar must be a tracked background task"

    await session.dispose()

    assert all(
        task.cancelled() or task.done() for task in pending
    ), "dispose must cancel and settle every task the fork registered"
