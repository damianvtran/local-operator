"""``/v1/desktop/voice*`` — the frozen contract of docs/design/session-voice.md.

Driven the way the renderer drives it: the REAL voice and sessions routers over
an ASGI transport with a bearer token, an isolated config root, and a double
for the session pool (the one seam that would otherwise need a live runtime).
The utterance goes through the REAL ``/messages`` handler, which is the point
of V5, so the doubles below are shaped like ``DesktopSessions``/the bridge/the
facade that handler reads.

V7 is pinned twice, deliberately of two kinds:

* STATIC: ``desktop_voice`` neither imports nor calls ``answer_gate`` nor the
  ``/answers`` handler, read off the module's AST. A later edit that adds a
  spoken "yes" fails here before it can be reviewed past.
* BEHAVIOURAL (V13): with an approval pending, an utterance is refused 409
  ``approval_pending`` before STT and before any owner op, and the facade
  records zero gate answers. The runtime half, against a real ``bash`` card, is
  ``tests/unit/voice/test_utterance_during_approval.py``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_sessions, desktop_voice
from local_operator.server.utils.desktop_feed import DesktopFeed
from local_operator.tts import RungAvailability as TtsRung
from local_operator.tts import TtsOutcome, VoicePath, VoicePathResolution
from local_operator.voice import slot as voice_slot

TOKEN = "desktop-voice-token"
SESSION_A = "a1a1a1a1a1a1"
SESSION_B = "b2b2b2b2b2b2"
ASSISTANT_ID = "assistant-row-1"
USER_ID = "user-row-1"
WAV_HEAD = b"RIFF\x24\x00\x00\x00WAVEfmt "


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class FakeRemote:
    """The facade: records every owner op, and has NO way to be told an answer
    except ``answer_gate``, which records the call so V7 can assert it never ran."""

    def __init__(self) -> None:
        self.is_streaming = False
        self.pending_gate: Any = None
        self.admissions: list[dict[str, Any]] = []
        self.gate_answers: list[Any] = []

    async def admit_prompt(self, text: str, *, command_id: str, steer: bool = False, **fields):
        self.admissions.append({"text": text, "command_id": command_id, "steer": steer, **fields})
        return ("steering queued" if steer else "prompt admitted", False)

    async def answer_gate(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - V7
        self.gate_answers.append((args, kwargs))
        return "answered"


class FakeBridge:
    def __init__(self, remote: FakeRemote) -> None:
        self.remote = remote
        self.published: list[tuple[str, dict[str, Any]]] = []

    async def prepare_peer_images(self, images: list[dict[str, str]]) -> list[dict[str, str]]:
        return images

    async def refresh_watch(self) -> None:
        return None

    async def history(self, *, around_id: str, **_: Any) -> dict[str, Any]:
        rows = {
            ASSISTANT_ID: {
                "id": ASSISTANT_ID,
                "type": "message",
                "ts": 1.0,
                "ts_source": "entry",
                "payload": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Here are the files. ```\nls\n``` Done."},
                    ],
                },
            },
            USER_ID: {
                "id": USER_ID,
                "type": "message",
                "ts": 1.0,
                "ts_source": "entry",
                "payload": {"role": "user", "content": [{"type": "text", "text": "secret"}]},
            },
        }
        row = rows.get(around_id)
        return {"entries": [row] if row else [], "has_more": False, "cursor_missing": not row}

    def publish(self, kind: str, payload: dict[str, Any], *, replay: bool = True) -> None:
        self.published.append((kind, payload))


class FakePool:
    """``DesktopSessions``-shaped: the door both routes take."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.remote = FakeRemote()
        self.bridge = FakeBridge(self.remote)
        self.doors: list[tuple[str, bool]] = []
        self.deleted: list[str] = []

    @contextlib.asynccontextmanager
    async def session(self, session_id: str, *, read: bool = False, allow_draft: bool = False):
        self.doors.append((session_id, read))
        if session_id not in (SESSION_A, SESSION_B):
            raise KeyError("Unknown session")
        yield self.bridge

    async def announce_admission(self, *_: Any, **__: Any) -> bool:
        return False

    async def announce_admission_failure(self, *_: Any, **__: Any) -> bool:
        return False

    async def delete(self, session_id: str) -> dict[str, Any]:
        import shutil

        shutil.rmtree(self.root / "sessions" / session_id)
        self.deleted.append(session_id)
        return {"session_id": session_id, "deleted": True}


def _resolution(path: VoicePath, servable: bool) -> VoicePathResolution:
    return VoicePathResolution(
        path=path,
        reason="probe" if servable else "Nothing can speak here.",
        rungs=(TtsRung(path, servable, "probe"),),
        servable=servable,
    )


@pytest_asyncio.fixture
async def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for sid in (SESSION_A, SESSION_B):
        (tmp_path / "sessions" / sid).mkdir(parents=True)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.env_config = SimpleNamespace(radient_api_base_url="http://radient.invalid")
    app.state.instance_id = "inst-test"
    pool = FakePool(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(desktop_voice.router)
    app.include_router(desktop_sessions.router)

    async def no_store():
        return SimpleNamespace(close=lambda: None)

    from local_operator.server.dependencies import get_provider_auth_store

    app.dependency_overrides[get_provider_auth_store] = no_store

    state: dict[str, Any] = {"tts": _resolution(VoicePath.LOCAL_TTS_SYSTEM, True)}

    async def fake_resolve_tts(*_: Any, **__: Any) -> VoicePathResolution:
        return state["tts"]

    async def fake_resolve_stt(*_: Any, **__: Any):
        from local_operator.stt import AudioPath, AudioPathResolution

        return AudioPathResolution(
            path=AudioPath.NONE,
            reason="No transcription provider.",
            rungs=(),
            model_audio_capable=False,
        )

    spoken: list[str] = []

    async def fake_synthesize(text: str, **kwargs: Any) -> TtsOutcome:
        assert kwargs["include_local"] is True, "the voice route must opt into the say rung"
        spoken.append(text)
        return TtsOutcome(audio=_wav_bytes(), path=VoicePath.LOCAL_TTS_SYSTEM, attempts=())

    spend: list[dict[str, Any]] = []
    monkeypatch.setattr(desktop_voice, "_resolve_tts", fake_resolve_tts)
    monkeypatch.setattr(desktop_voice, "_resolve_stt", fake_resolve_stt)
    monkeypatch.setattr("local_operator.tts.cascade.synthesize_speech", fake_synthesize)
    monkeypatch.setattr(desktop_voice, "emit_voice_spend", lambda **kw: spend.append(kw))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield SimpleNamespace(
            client=client, pool=pool, root=tmp_path, state=state, spoken=spoken, spend=spend
        )


async def _enable(rig: Any, session_id: str) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice", json={"op": "enable", "session_id": session_id}
    )


async def _utter(rig: Any, session_id: str, text: str) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": session_id, "request_id": str(uuid.uuid4()), "text": text},
    )


# -- the contract --------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_reports_the_slot_paths_and_cap(rig: Any) -> None:
    result = (await rig.client.get("/v1/desktop/voice")).json()["result"]
    assert result == {
        "active": None,
        "stt": {"path": "none", "available": False, "reason": "No transcription provider."},
        "tts": {"path": "local_tts_system", "available": True, "reason": "probe"},
        "settings": {"max_spoken_chars": 800},
    }


@pytest.mark.asyncio
async def test_enabling_b_replaces_a(rig: Any) -> None:
    first = (await _enable(rig, SESSION_A)).json()["result"]
    assert first["replaced"] is None and first["active"]["session_id"] == SESSION_A
    second = await _enable(rig, SESSION_B)
    assert second.status_code == 200
    body = second.json()["result"]
    assert body["replaced"] == SESSION_A
    assert body["active"]["session_id"] == SESSION_B
    assert isinstance(body["active"]["enabled_at"], int)
    status = (await rig.client.get("/v1/desktop/voice")).json()["result"]
    assert status["active"] == body["active"]


@pytest.mark.asyncio
async def test_disable_is_scoped_to_the_holder(rig: Any) -> None:
    await _enable(rig, SESSION_B)
    stale = (
        await rig.client.post("/v1/desktop/voice", json={"op": "disable", "session_id": SESSION_A})
    ).json()["result"]
    assert stale["active"]["session_id"] == SESSION_B and stale["replaced"] is None
    cleared = (
        await rig.client.post("/v1/desktop/voice", json={"op": "disable", "session_id": SESSION_B})
    ).json()["result"]
    assert cleared == {"active": None, "replaced": None}


@pytest.mark.asyncio
async def test_unknown_session_and_no_tts_are_refused_with_codes(rig: Any) -> None:
    unknown = await _enable(rig, "c3c3c3c3c3c3")
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["code"] == "unknown_session"
    traversal = await _enable(rig, "..")
    assert traversal.status_code == 404
    rig.state["tts"] = _resolution(VoicePath.NONE, False)
    refused = await _enable(rig, SESSION_A)
    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert detail["code"] == "voice_unavailable"
    assert "Nothing can speak here." in detail["text"]
    assert voice_slot.read_slot(rig.root) is None


@pytest.mark.asyncio
async def test_a_held_slot_lock_is_voice_store_busy(rig: Any, monkeypatch) -> None:
    from local_operator.wakes.lock import WakeLockBusy

    def busy(*_: Any, **__: Any) -> Any:
        raise WakeLockBusy("held")

    monkeypatch.setattr(voice_slot, "enable", busy)
    refused = await _enable(rig, SESSION_A)
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "voice_store_busy"
    assert refused.json()["detail"]["text"] == voice_slot.BUSY_SENTENCE


@pytest.mark.asyncio
async def test_every_route_refuses_a_session_not_in_the_slot(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    utter = await _utter(rig, SESSION_B, "hello")
    speak = await rig.client.post(
        "/v1/desktop/voice/speech",
        json={"session_id": SESSION_B, "source": {"kind": "message", "message_id": ASSISTANT_ID}},
    )
    for response in (utter, speak):
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "voice_not_enabled"
    assert rig.pool.remote.admissions == []
    assert rig.spoken == []


@pytest.mark.asyncio
async def test_the_utterance_is_admitted_as_a_dictated_prompt(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    response = await _utter(rig, SESSION_A, "  [bash:1] list files  ")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["text"] == "[bash:1] list files"
    assert result["path"] is None
    assert result["admission"]["status"] == "admitted"
    assert result["admission"]["detail"] == "prompt admitted"
    (admitted,) = rig.pool.remote.admissions
    assert admitted["steer"] is False
    assert admitted["text"] == "[bash:1] list files"
    assert admitted["input_mode"] == "dictated"
    assert admitted["input_path"] is None


@pytest.mark.asyncio
async def test_the_utterance_steers_while_the_turn_is_streaming(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    rig.pool.remote.is_streaming = True
    result = (await _utter(rig, SESSION_A, "also check the tests")).json()["result"]
    assert result["admission"]["detail"] == "steering queued"
    assert rig.pool.remote.admissions[-1]["steer"] is True


@pytest.mark.asyncio
async def test_a_slash_command_is_refused_as_on_messages(rig: Any) -> None:
    """Same rule as ``/messages``, but a DECLARED refusal naming the problem (B1)."""
    await _enable(rig, SESSION_A)
    response = await _utter(rig, SESSION_A, "/login openai")
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_voice_request"
    # The validator's own sentence, not the "exactly one of" body-shape copy.
    assert "/login is a command, not a message" in detail["text"], detail
    assert "exactly one" not in detail["text"]
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_malformed_utterances_are_refused_before_anything_runs(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    neither = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4())},
    )
    both = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4()), "text": "x"},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    bad_id = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": "not-a-uuid", "text": "x"},
    )
    blank = await _utter(rig, SESSION_A, "   ")
    for response, code, words in (
        (neither, "invalid_voice_request", "exactly one of a recording or a text"),
        (both, "invalid_voice_request", "exactly one of a recording or a text"),
        (bad_id, "invalid_voice_request", "dashed UUID"),
        (blank, "empty_transcript", "Nothing was heard"),
    ):
        assert response.status_code == 422, response.text
        assert response.json()["detail"]["code"] == code
        assert words in response.json()["detail"]["text"]
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_audio_with_no_stt_rung_is_stt_unavailable(rig: Any, monkeypatch) -> None:
    from local_operator.stt import AudioPath, AudioPathResolution
    from local_operator.stt.cascade import SttUnavailable

    async def no_rung(*_: Any, **__: Any):
        resolution = AudioPathResolution(
            path=AudioPath.NONE,
            reason="No transcription provider.",
            rungs=(),
            model_audio_capable=False,
        )
        raise SttUnavailable("none", resolution=resolution, attempts=())

    monkeypatch.setattr("local_operator.stt.cascade.transcribe_audio", no_rung)
    await _enable(rig, SESSION_A)
    response = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4())},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "stt_unavailable"
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_transcribed_audio_carries_its_path_and_spend(rig: Any, monkeypatch) -> None:
    from local_operator.stt import AudioPath, SttOutcome

    async def transcribe(path: Path, **kwargs: Any) -> SttOutcome:
        assert path.read_bytes()[:4] == b"RIFF"
        assert kwargs["correlation_session_id"] == SESSION_A
        return SttOutcome(text="list files", path=AudioPath.PROVIDER_STT_OPENAI, attempts=())

    monkeypatch.setattr("local_operator.stt.cascade.transcribe_audio", transcribe)
    await _enable(rig, SESSION_A)
    response = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4())},
        # The declared type lies; the container is sniffed from the bytes.
        files={"file": ("a.mp3", _wav_bytes(), "audio/mpeg")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["path"] == "provider_stt_openai"
    assert rig.pool.remote.admissions[-1]["input_path"] == "provider_stt_openai"
    (record,) = rig.spend
    assert record["channel"] == "stt" and record["session_id"] == SESSION_A
    assert record["provider"] == "openai"


@pytest.mark.asyncio
async def test_unsupported_audio_is_refused_by_its_bytes(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    response = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4())},
        files={"file": ("a.wav", b"definitely not audio", "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_voice_request"


@pytest.mark.asyncio
async def test_speech_reads_the_agents_message_cleaned_with_its_headers(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    response = await rig.client.post(
        "/v1/desktop/voice/speech",
        json={"session_id": SESSION_A, "source": {"kind": "message", "message_id": ASSISTANT_ID}},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-voice-path"] == "local_tts_system"
    (text,) = rig.spoken
    assert "ls" not in text.split() and "(code block omitted)" in text
    assert response.headers["x-voice-spoken-chars"] == str(len(text))
    assert response.content[:4] == b"RIFF"
    (record,) = rig.spend
    assert record["channel"] == "tts" and record["session_id"] == SESSION_A
    assert record["units"] == float(len(text))


@pytest.mark.asyncio
async def test_each_spoken_reply_gets_its_own_correlation_id(rig: Any, monkeypatch) -> None:
    """B4: two replies in the same millisecond must not share a spend id."""
    monkeypatch.setattr(desktop_voice.time, "time", lambda: 1_792_000_000.0)
    await _enable(rig, SESSION_A)
    for _ in range(2):
        response = await rig.client.post(
            "/v1/desktop/voice/speech",
            json={
                "session_id": SESSION_A,
                "source": {"kind": "message", "message_id": ASSISTANT_ID},
            },
        )
        assert response.status_code == 200, response.text
    first, second = (record["request_id"] for record in rig.spend)
    assert first != second
    assert first.startswith(f"voice-{SESSION_A}-1792000000000-")


@pytest.mark.asyncio
async def test_speech_honours_the_configured_cap(rig: Any) -> None:
    from local_operator import settings_io

    settings_io.write_setting(
        ConfigManager(rig.root), settings_io.BY_KEY["voice.max_spoken_chars"], 100
    )
    await _enable(rig, SESSION_A)
    status = (await rig.client.get("/v1/desktop/voice")).json()["result"]
    assert status["settings"]["max_spoken_chars"] == 100


@pytest.mark.asyncio
async def test_speech_never_reads_a_non_assistant_or_unknown_source(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    for source in (
        {"kind": "message", "message_id": USER_ID},
        {"kind": "message", "message_id": "missing"},
        {"kind": "gate", "request_id": "no-gate"},
        {"kind": "ask", "ask_id": "q1"},
    ):
        response = await rig.client.post(
            "/v1/desktop/voice/speech", json={"session_id": SESSION_A, "source": source}
        )
        assert response.status_code == 404, (source, response.text)
        assert response.json()["detail"]["code"] == "unknown_source"
    free_text = await rig.client.post(
        "/v1/desktop/voice/speech",
        json={"session_id": SESSION_A, "source": {"kind": "text", "text": "say anything"}},
    )
    assert free_text.status_code == 422
    assert rig.spoken == []


@pytest.mark.asyncio
async def test_speech_announces_a_pending_approval_from_its_card_text(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    rig.pool.remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="bash", detail="rm -rf build/"
    )
    response = await rig.client.post(
        "/v1/desktop/voice/speech",
        json={"session_id": SESSION_A, "source": {"kind": "gate", "request_id": "gate-1"}},
    )
    assert response.status_code == 200, response.text
    assert rig.spoken == ["Approval needed: bash. rm -rf build/"]
    assert rig.pool.remote.gate_answers == []


# -- V7 --------------------------------------------------------------------------


def test_desktop_voice_has_no_import_or_call_path_to_answer_gate() -> None:
    """V7, statically: no name in the module reaches a gate answer."""
    source = Path(desktop_voice.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"answer_gate", "answer", "approval_answer", "ask_answer", "Answer"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            offenders += [f"import {alias.name}" for alias in node.names if alias.name in forbidden]
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            offenders.append(f"attribute .{node.attr} at line {node.lineno}")
        elif isinstance(node, ast.Name) and node.id in forbidden:
            offenders.append(f"name {node.id} at line {node.lineno}")
    assert offenders == [], offenders
    # And no route on the router is the answers path.
    assert all(
        "/answers" not in getattr(route, "path", "") for route in desktop_voice.router.routes
    )


@pytest.mark.asyncio
async def test_an_utterance_during_a_pending_approval_is_refused(rig: Any, monkeypatch) -> None:
    """V7 + V13: "yes" spoken at a parked approval card is refused, text or audio."""
    stt_calls: list[Any] = []

    async def transcribe(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - V13
        stt_calls.append((args, kwargs))
        raise AssertionError("STT ran while an approval was pending")

    monkeypatch.setattr("local_operator.stt.cascade.transcribe_audio", transcribe)
    await _enable(rig, SESSION_A)
    remote = rig.pool.remote
    remote.is_streaming = True  # a parked approval sits inside a streaming turn
    remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="bash", detail="rm -rf build/"
    )
    spoken = await _utter(rig, SESSION_A, "yes")
    recorded = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4())},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    for response in (spoken, recorded):
        assert response.status_code == 409, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "approval_pending"
        assert detail["text"] == "Answer the question or approval on screen first."
    assert stt_calls == [], "a refused recording was transcribed"
    assert rig.spend == [], "a refused recording was billed"
    assert remote.admissions == [], "the utterance reached the owner"
    assert remote.gate_answers == [], "an utterance reached answer_gate"
    assert remote.pending_gate is not None

    # N4: a BLOCKING ask's card (``LOP_ASK_NONBLOCKING=0``) is interruptible too,
    # so it is refused the same way. A queued ask parks no card at all.
    remote.pending_gate = SimpleNamespace(request_id="ask-1", kind="ask", title="", detail="")
    asked = await _utter(rig, SESSION_A, "the blue one")
    assert asked.status_code == 409, asked.text
    assert asked.json()["detail"]["code"] == "approval_pending"
    assert remote.admissions == []
    # And with no card at all, the same words are an ordinary steer.
    remote.pending_gate = None
    free = await _utter(rig, SESSION_A, "the blue one")
    assert free.status_code == 200, free.text
    assert remote.admissions[-1]["steer"] is True


@pytest.mark.asyncio
async def test_a_card_that_parks_during_stt_still_refuses(rig: Any, monkeypatch) -> None:
    """V13's second check: STT takes seconds and a card can park inside them."""
    from local_operator.stt import AudioPath, SttOutcome

    remote = rig.pool.remote

    async def transcribe(path: Path, **kwargs: Any) -> SttOutcome:
        remote.is_streaming = True
        remote.pending_gate = SimpleNamespace(
            request_id="gate-2", kind="approval", title="bash", detail="ls"
        )
        return SttOutcome(text="yes", path=AudioPath.PROVIDER_STT_OPENAI, attempts=())

    monkeypatch.setattr("local_operator.stt.cascade.transcribe_audio", transcribe)
    await _enable(rig, SESSION_A)
    response = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4())},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "approval_pending"
    assert remote.admissions == []


# -- B3: a deleted conversation leaves the slot ----------------------------------


@pytest.mark.asyncio
async def test_a_slot_naming_a_removed_conversation_reads_as_empty(rig: Any) -> None:
    """Deleted by another surface: the slot file stays, but nothing treats it as live."""
    import shutil

    await _enable(rig, SESSION_A)
    shutil.rmtree(rig.root / "sessions" / SESSION_A)
    status = (await rig.client.get("/v1/desktop/voice")).json()["result"]
    assert status["active"] is None
    speak = await rig.client.post(
        "/v1/desktop/voice/speech",
        json={"session_id": SESSION_A, "source": {"kind": "message", "message_id": ASSISTANT_ID}},
    )
    utter = await _utter(rig, SESSION_A, "hello")
    for response in (speak, utter):
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "voice_not_enabled"
    # Enabling another conversation does not report the vanished one as replaced.
    moved = (await _enable(rig, SESSION_B)).json()["result"]
    assert moved["replaced"] is None and moved["active"]["session_id"] == SESSION_B


@pytest.mark.asyncio
async def test_deleting_the_slot_holder_clears_the_slot(rig: Any) -> None:
    """The delete route clears the file, which is what moves the feed's frame."""
    await _enable(rig, SESSION_A)
    response = await rig.client.request(
        "DELETE", f"/v1/desktop/sessions/{SESSION_A}", json={"confirmed": True}
    )
    assert response.status_code == 200, response.text
    assert rig.pool.deleted == [SESSION_A]
    assert voice_slot.read_slot(rig.root) is None
    # Deleting a conversation that does NOT hold the slot leaves the slot alone.
    await _enable(rig, SESSION_B)
    (rig.root / "sessions" / SESSION_A).mkdir(parents=True)
    other = await rig.client.request(
        "DELETE", f"/v1/desktop/sessions/{SESSION_A}", json={"confirmed": True}
    )
    assert other.status_code == 200, other.text
    current = voice_slot.read_slot(rig.root)
    assert current is not None and current.session_id == SESSION_B


@pytest.mark.asyncio
async def test_a_delete_never_creates_voice_state_when_voice_was_never_used(rig: Any) -> None:
    """N3: the hook reads first, so a machine with no voice gets no ``voice/`` dir."""
    assert not (rig.root / "voice").exists()
    response = await rig.client.request(
        "DELETE", f"/v1/desktop/sessions/{SESSION_A}", json={"confirmed": True}
    )
    assert response.status_code == 200, response.text
    assert not (rig.root / "voice").exists(), sorted(p.name for p in rig.root.iterdir())


@pytest.mark.asyncio
async def test_a_delete_does_not_wait_on_a_slot_lock_another_process_holds(rig: Any) -> None:
    """N3: deleting a NON-holder must not stall behind the slot lock (up to 5 s)."""
    import fcntl
    import os
    import time

    await _enable(rig, SESSION_B)
    lock_path = rig.root / "voice" / voice_slot.LOCK_NAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    # flock is per open file description, so this fd contends with the route's
    # own open exactly as another process's would.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        started = time.monotonic()
        response = await rig.client.request(
            "DELETE", f"/v1/desktop/sessions/{SESSION_A}", json={"confirmed": True}
        )
        elapsed = time.monotonic() - started
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert response.status_code == 200, response.text
    assert elapsed < voice_slot.LOCK_WAIT_S / 2, f"the delete waited {elapsed:.2f}s on the lock"
    current = voice_slot.read_slot(rig.root)
    assert current is not None and current.session_id == SESSION_B


# -- the machine-feed frame ------------------------------------------------------


def test_the_feed_publishes_the_slot_as_a_voice_frame(tmp_path: Path) -> None:
    import asyncio

    feed = DesktopFeed(tmp_path, bridged=lambda: frozenset())
    feed._take_baseline()
    subscription = feed.subscribe()

    def tick() -> None:
        feed._voice_probed_at = 0.0
        asyncio.run(feed._maybe_emit_voice())

    def frames() -> list[dict[str, Any]]:
        out = []
        while not subscription.queue.empty():
            frame = subscription.queue.get_nowait()
            if frame is not None and frame["type"] == "voice":
                out.append(frame)
        return out

    tick()
    assert frames() == [], "an unchanged slot published"
    voice_slot.enable(tmp_path, SESSION_A, now_ms=11)
    tick()
    (enabled,) = frames()
    assert set(enabled) == {"epoch", "seq", "type", "payload"}
    assert enabled["payload"] == {"active": {"session_id": SESSION_A, "enabled_at": 11}}
    voice_slot.enable(tmp_path, SESSION_B, now_ms=12)
    tick()
    assert [f["payload"]["active"]["session_id"] for f in frames()] == [SESSION_B]
    voice_slot.disable(tmp_path, SESSION_B)
    tick()
    assert [f["payload"] for f in frames()] == [{"active": None}]
    asyncio.run(feed.close())
