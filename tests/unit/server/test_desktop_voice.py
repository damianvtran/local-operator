"""``/v1/desktop/voice*`` — the frozen contract of docs/design/qwen-voice.md.

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
  ``approval_pending`` before any owner op, and the facade records zero gate
  answers. The runtime half, against a real ``bash`` card, is
  ``tests/unit/voice/test_utterance_during_approval.py``.

Speech goes through the harness cascades (session-voice-groq D1, which replaced
Q10): the route module owns no rung or client, and every cascade call passes
``include_groq=True``. The cascades are stubbed here; no key and no network.
"""

from __future__ import annotations

import ast
import contextlib
import uuid
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
from local_operator.voice import slot as voice_slot

TOKEN = "desktop-voice-token"
SESSION_A = "a1a1a1a1a1a1"
SESSION_B = "b2b2b2b2b2b2"
ASSISTANT_ID = "assistant-row-1"
USER_ID = "user-row-1"
TOOL_ID = "tool-row-1"
CUSTOM_ID = "custom-row-1"


class FakeRemote:
    """The facade: records every owner op, and has NO way to be told an answer
    except ``answer_gate``, which records the call so V7 can assert it never ran."""

    def __init__(self) -> None:
        self.is_streaming = False
        self.pending_gate: Any = None
        #: True when the runtime runs the queued-ask engine (``asks``/``asks_open``
        #: are on its state), so an ask-kind ``pending_gate`` is only the legacy mirror.
        self.queued_engine_live = False
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
                    # No ``type`` key: the journal stores a text block WITHOUT it
                    # (``exclude_defaults``), and a reader keyed on ``type`` must
                    # fail here, not only on a live session (review M1).
                    "content": [
                        {"text": "Here are the files. ```\nls\n``` Done."},
                    ],
                },
            },
            TOOL_ID: {
                "id": TOOL_ID,
                "type": "message",
                "ts": 1.0,
                "ts_source": "entry",
                "payload": {"role": "tool", "content": [{"type": "text", "text": "exit 0"}]},
            },
            CUSTOM_ID: {
                "id": CUSTOM_ID,
                "type": "custom",
                "ts": 1.0,
                "ts_source": "entry",
                "payload": {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
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
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield SimpleNamespace(client=client, pool=pool, root=tmp_path, app=app)


def _choose(rig: Any, provider: str | None) -> None:
    """Write ``speech.provider`` the way the settings page does (``None`` clears it)."""
    from local_operator import settings_io

    settings_io.write_setting(
        ConfigManager(rig.root), settings_io.BY_KEY["speech.provider"], provider
    )


async def _enable(rig: Any, session_id: str) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice", json={"op": "enable", "session_id": session_id}
    )


async def _utter(rig: Any, session_id: str, text: str, **fields: str) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": session_id, "request_id": str(uuid.uuid4()), "text": text, **fields},
    )


async def _text(rig: Any, session_id: str, source: dict[str, Any]) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice/text", json={"session_id": session_id, "source": source}
    )


# -- the contract --------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_reports_the_slot_the_cap_the_choice_and_no_key_material(
    rig: Any,
) -> None:
    """The slot and cap, plus which provider is chosen and whether each can serve:
    booleans and the resolver's sentences, and no credential of any kind."""
    result = (await rig.client.get("/v1/desktop/voice")).json()["result"]
    assert result["active"] is None
    assert result["settings"] == {"max_spoken_chars": 800}
    assert result["provider"] is None, "nothing is chosen until the user picks"
    assert set(result["providers"]) == {"radient", "elevenlabs"}
    for state in result["providers"].values():
        assert set(state) == {"ready", "reason"}
        assert state["ready"] is False, "the test rig stores no credential"
        assert isinstance(state["reason"], str) and state["reason"]
    # Kept for the app that predates the provider choice.
    assert set(result["byok"]) == {"provider", "ready", "reason"}
    assert result["byok"]["provider"] == "elevenlabs"
    assert set(result) == {"active", "settings", "byok", "provider", "providers"}


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
async def test_unknown_session_is_refused_and_enable_needs_no_speech_path(rig: Any) -> None:
    unknown = await _enable(rig, "c3c3c3c3c3c3")
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["code"] == "unknown_session"
    traversal = await _enable(rig, "..")
    assert traversal.status_code == 404
    assert voice_slot.read_slot(rig.root) is None
    # Q10: no TTS precondition. Nothing is stubbed in this rig, and enable works.
    enabled = await _enable(rig, SESSION_A)
    assert enabled.status_code == 200, enabled.text
    # The removed ``front_end`` field is refused, not silently ignored.
    stale = await rig.client.post(
        "/v1/desktop/voice",
        json={"op": "enable", "session_id": SESSION_A, "front_end": "local"},
    )
    assert stale.status_code == 422


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
    speak = await _text(rig, SESSION_B, {"kind": "message", "message_id": ASSISTANT_ID})
    for response in (utter, speak):
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "voice_not_enabled"
    assert rig.pool.remote.admissions == []


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


async def _utter_with_id(rig: Any, request_id: str, text: str) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": request_id, "text": text},
    )


@pytest.mark.asyncio
async def test_a_retry_after_the_turn_started_replays_instead_of_conflicting(rig: Any) -> None:
    """The mode is the SERVER's pick (steer while streaming), so it must not be
    part of what the receipt journal compares: a timed-out client retrying the
    same ``request_id`` after its own prompt started the turn is a replay, not
    "Request ID was already used with different input"."""
    await _enable(rig, SESSION_A)
    request_id = str(uuid.uuid4())
    first = await _utter_with_id(rig, request_id, "list files")
    assert first.status_code == 200, first.text
    assert first.json()["result"]["admission"]["detail"] == "prompt admitted"

    rig.pool.remote.is_streaming = True
    retry = await _utter_with_id(rig, request_id, "list files")
    assert retry.status_code == 200, retry.text
    assert retry.json()["result"]["admission"]["replayed"] is True
    assert len(rig.pool.remote.admissions) == 1, "a replay never reaches the owner twice"


@pytest.mark.asyncio
async def test_a_retry_after_the_steered_turn_ended_replays_instead_of_conflicting(
    rig: Any,
) -> None:
    await _enable(rig, SESSION_A)
    request_id = str(uuid.uuid4())
    rig.pool.remote.is_streaming = True
    first = await _utter_with_id(rig, request_id, "also check the tests")
    assert first.status_code == 200, first.text
    assert first.json()["result"]["admission"]["detail"] == "steering queued"

    rig.pool.remote.is_streaming = False
    retry = await _utter_with_id(rig, request_id, "also check the tests")
    assert retry.status_code == 200, retry.text
    assert retry.json()["result"]["admission"]["replayed"] is True
    assert len(rig.pool.remote.admissions) == 1


@pytest.mark.asyncio
async def test_a_reused_request_id_with_other_words_is_still_a_conflict(rig: Any) -> None:
    """The mode-blind replay must not weaken the journal's real protection."""
    await _enable(rig, SESSION_A)
    request_id = str(uuid.uuid4())
    assert (await _utter_with_id(rig, request_id, "list files")).status_code == 200
    rig.pool.remote.is_streaming = True
    other = await _utter_with_id(rig, request_id, "delete everything")
    assert other.status_code == 409, other.text
    assert len(rig.pool.remote.admissions) == 1


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
    # A fixed keyed sentence (no validator prose, which would be English in every
    # locale), and not the "exactly one of" body-shape copy.
    assert detail["text"] == (
        "That message was not sent: it is too long, or it reads as a command."
    ), detail
    assert "params" in detail and detail["params"] == {}
    assert "exactly one" not in detail["text"]
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_malformed_utterances_are_refused_before_anything_runs(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    no_text = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4())},
    )
    bad_id = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": "not-a-uuid", "text": "x"},
    )
    blank = await _utter(rig, SESSION_A, "   ")
    for response, code, words in (
        (no_text, "invalid_voice_request", "No words came"),
        (bad_id, "invalid_voice_request", "dashed UUID"),
        (blank, "empty_transcript", "Nothing was heard"),
    ):
        assert response.status_code == 422, response.text
        assert response.json()["detail"]["code"] == code
        assert words in response.json()["detail"]["text"]
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_a_recording_and_text_together_are_refused(rig: Any) -> None:
    """Exactly one door per request: both is as invalid as neither."""
    await _enable(rig, SESSION_A)
    response = await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4()), "text": "list files"},
        files={"file": ("a.wav", b"RIFF\x24\x00\x00\x00WAVEfmt ", "audio/wav")},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_voice_request"
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_text_origin_qwen_s2s_is_recorded_as_input_path(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    response = await _utter(rig, SESSION_A, "list files", text_origin="qwen_s2s")
    assert response.status_code == 200, response.text
    assert response.json()["result"]["path"] == "qwen_s2s"
    (admitted,) = rig.pool.remote.admissions
    assert admitted["input_path"] == "qwen_s2s"
    assert admitted["input_mode"] == "dictated"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["free text", "gemini_live_transcript", "provider_stt_openai"])
async def test_an_unknown_text_origin_is_refused(rig: Any, origin: str) -> None:
    await _enable(rig, SESSION_A)
    response = await _utter(rig, SESSION_A, "list files", text_origin=origin)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_voice_request"
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_a_controls_field_is_not_a_grammar(rig: Any) -> None:
    """The R6 controls grammar is gone (Q2): "stop" is relayed like any take."""
    await _enable(rig, SESSION_A)
    response = await _utter(rig, SESSION_A, "Stop.", controls="1")
    assert response.status_code == 200, response.text
    assert "control" not in response.json()["result"]
    (admitted,) = rig.pool.remote.admissions
    assert admitted["text"] == "Stop."


# -- POST /v1/desktop/voice/text -------------------------------------------------


@pytest.mark.asyncio
async def test_text_reads_the_agents_message_through_speakable(rig: Any) -> None:
    from local_operator.voice import speakable

    await _enable(rig, SESSION_A)
    response = await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert set(result) == {"text", "spoken_chars", "truncated", "segments", "full_text"}
    assert "".join(result["segments"]) == result["text"]
    assert result["full_text"] == speakable.display_text("Here are the files. ```\nls\n``` Done.")
    assert "(code block omitted)" in result["full_text"] and "```" not in result["full_text"]
    expected = speakable.speakable("Here are the files. ```\nls\n``` Done.").text
    assert result["text"] == expected
    assert "ls" not in result["text"].split() and "```" not in result["text"]
    assert result["spoken_chars"] == len(result["text"])
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_text_honours_the_configured_cap_read_from_disk(rig: Any) -> None:
    from local_operator import settings_io
    from local_operator.voice import speakable

    long_reply = " ".join(f"Sentence number {n} is here." for n in range(60))
    rows = rig.pool.bridge.history

    async def history(*, around_id: str, **kwargs: Any) -> dict[str, Any]:
        page = await rows(around_id=around_id, **kwargs)
        for entry in page["entries"]:
            entry["payload"]["content"] = [{"type": "text", "text": long_reply}]
        return page

    rig.pool.bridge.history = history
    settings_io.write_setting(
        ConfigManager(rig.root), settings_io.BY_KEY["voice.max_spoken_chars"], 100
    )
    await _enable(rig, SESSION_A)
    status = (await rig.client.get("/v1/desktop/voice")).json()["result"]
    assert status["settings"]["max_spoken_chars"] == 100
    result = (await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})).json()[
        "result"
    ]
    assert len(result["text"]) <= 100
    assert result["text"].endswith(speakable.TRUNCATION_TAIL)
    assert result["truncated"] is True
    assert result["text"] == speakable.speakable(long_reply, max_chars=100).text
    # The audio stays capped; the screen text is the whole display-cleaned reply.
    assert result["full_text"] == speakable.display_text(long_reply)
    assert len(result["full_text"]) > 100
    assert speakable.TRUNCATION_TAIL not in result["full_text"]
    assert result["spoken_chars"] == len(result["text"])


@pytest.mark.asyncio
async def test_text_full_text_keeps_a_table_as_lines_while_text_stays_flat(rig: Any) -> None:
    from local_operator.voice import speakable

    table = (
        "Status:\r\n\r\n| Hook | Fires | Fixes today? |\r\n|---|---|---|\r\n"
        "| **Pre** | yes | no |\r\n| Post | no | yes |\r\n\r\n- one\r\n- two"
    )
    rows = rig.pool.bridge.history

    async def history(*, around_id: str, **kwargs: Any) -> dict[str, Any]:
        page = await rows(around_id=around_id, **kwargs)
        for entry in page["entries"]:
            entry["payload"]["content"] = [{"type": "text", "text": table}]
        return page

    rig.pool.bridge.history = history
    await _enable(rig, SESSION_A)
    result = (await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})).json()[
        "result"
    ]
    assert result["full_text"] == (
        "Status:\n\nHook · Fires · Fixes today?\nPre · yes · no\nPost · no · yes\n\n- one\n- two"
    )
    # The spoken fields are the flat string they always were.
    assert result["text"] == speakable.speakable(table).text
    assert "\n" not in result["text"] and "|" not in result["text"]
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_text_full_text_uses_the_active_locales_code_placeholder(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``display_text`` is handed the locale's stand-in, as ``text`` is."""
    from local_operator.i18n.text import set_active_locale

    reply = "Before.\n```\nx = 1\n```\nAfter."
    monkeypatch.setattr(FakeBridge, "history", _history_with(ASSISTANT_ID, reply), raising=True)
    await _enable(rig, SESSION_A)
    english = desktop_voice._spoken_words("en")["code_omitted"]
    set_active_locale("es")
    try:
        spanish = desktop_voice._spoken_words("es")["code_omitted"]
        assert spanish != english
        result = (
            await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})
        ).json()["result"]
    finally:
        set_active_locale("en")
    assert result["full_text"] == f"Before.\n{spanish}\nAfter."
    assert spanish in result["text"] and english not in result["full_text"]


@pytest.mark.asyncio
async def test_text_a_reply_carrying_the_display_sentinel_is_still_a_200(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The private-use pair ``display_text`` parks spans behind was a 500 (IndexError)."""
    reply = "before \ue000999\ue001 https://a.com \ue0000\ue001 after `my_var_name`"
    monkeypatch.setattr(FakeBridge, "history", _history_with(ASSISTANT_ID, reply), raising=True)
    await _enable(rig, SESSION_A)
    response = await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})
    assert response.status_code == 200, response.text
    assert response.json()["result"]["full_text"] == "before 999 https://a.com 0 after my_var_name"


@pytest.mark.asyncio
async def test_text_never_reads_a_user_tool_or_unknown_source(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    for source in (
        {"kind": "message", "message_id": USER_ID},
        {"kind": "message", "message_id": TOOL_ID},
        {"kind": "message", "message_id": CUSTOM_ID},
        {"kind": "message", "message_id": "missing"},
        {"kind": "gate", "request_id": "no-gate"},
    ):
        response = await _text(rig, SESSION_A, source)
        assert response.status_code == 404, (source, response.text)
        assert response.json()["detail"]["code"] == "unknown_source"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        {"kind": "text", "text": "say anything"},
        {"kind": "phrase", "key": "passing_on"},
        {"kind": "ask", "ask_id": "q1"},
        {"kind": "message", "message_id": ASSISTANT_ID, "text": "say anything"},
        {"kind": "gate", "request_id": "gate-1", "text": "say anything"},
    ],
)
async def test_text_accepts_no_free_text_and_no_other_source_kind(
    rig: Any, source: dict[str, Any]
) -> None:
    await _enable(rig, SESSION_A)
    response = await _text(rig, SESSION_A, source)
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_text_has_no_top_level_text_field(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    response = await rig.client.post(
        "/v1/desktop/voice/text",
        json={
            "session_id": SESSION_A,
            "source": {"kind": "message", "message_id": ASSISTANT_ID},
            "text": "say anything",
        },
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_text_announces_a_pending_approval_from_its_card_and_answers_nothing(
    rig: Any,
) -> None:
    await _enable(rig, SESSION_A)
    rig.pool.remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="bash", detail="rm -rf build/"
    )
    response = await _text(rig, SESSION_A, {"kind": "gate", "request_id": "gate-1"})
    assert response.status_code == 200, response.text
    assert response.json()["result"]["text"] == "Approval needed: bash. rm -rf build/"
    stale = await _text(rig, SESSION_A, {"kind": "gate", "request_id": "gate-0"})
    assert stale.status_code == 404
    rig.pool.remote.pending_gate = SimpleNamespace(
        request_id="ask-1", kind="ask", title="", detail="which one?"
    )
    ask = await _text(rig, SESSION_A, {"kind": "gate", "request_id": "ask-1"})
    assert ask.status_code == 404
    assert rig.pool.remote.gate_answers == []
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_the_live_token_route_is_gone_and_the_router_is_the_four_routes(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    response = await rig.client.post("/v1/desktop/voice/live-token", json={})
    assert response.status_code in (404, 405), response.text
    assert {getattr(route, "path", "") for route in desktop_voice.router.routes} == {
        "/v1/desktop/voice",
        "/v1/desktop/voice/utterance",
        "/v1/desktop/voice/text",
        "/v1/desktop/voice/speech",
    }


def test_desktop_voice_adds_no_speech_client_of_its_own() -> None:
    """D1, statically: the module reaches the cascades, not a rung or a client."""
    tree = ast.parse(Path(desktop_voice.__file__).read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if ("stt" in parts or "tts" in parts) and "clients" in parts:
                offenders.append(node.module)
            offenders += [
                f"{node.module}.{alias.name}"
                for alias in node.names
                if ("stt" in parts or "tts" in parts) and alias.name.startswith("_run_")
            ]
    assert offenders == [], offenders


def test_desktop_voice_has_no_import_or_call_path_to_answer_gate() -> None:
    """V7, statically: no name in the module reaches a gate answer.

    It matches NAMES, so a dynamic ``getattr(remote, "answer_" + kind)`` would
    evade it; review covers that shape, and this pin catches the plain one.
    """
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
async def test_an_utterance_during_a_pending_approval_is_refused(rig: Any) -> None:
    """V7 + V13: "yes" spoken at a parked approval card is refused."""
    await _enable(rig, SESSION_A)
    remote = rig.pool.remote
    remote.is_streaming = True  # a parked approval sits inside a streaming turn
    remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="bash", detail="rm -rf build/"
    )
    spoken = await _utter(rig, SESSION_A, "yes")
    assert spoken.status_code == 409, spoken.text
    detail = spoken.json()["detail"]
    assert detail["code"] == "approval_pending"
    assert detail["text"] == "Answer the question or approval on screen first."
    assert remote.admissions == [], "the utterance reached the owner"
    assert remote.gate_answers == [], "an utterance reached answer_gate"
    assert remote.pending_gate is not None

    # N4: a BLOCKING ask's card (``LOP_ASK_NONBLOCKING=0``) is interruptible too,
    # so it is refused the same way. A QUEUED ask is not a card (see the tests below).
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
async def test_a_queued_asks_mirrored_card_does_not_refuse_speech(rig: Any) -> None:
    """A queued ask is mirrored onto ``pending_gate`` for old clients, but it parks
    nothing: the agent is continuing, so words are an ordinary message or steer.

    Found live (2026-11-17): with an ask open, every spoken request was refused
    ``approval_pending`` until the ask was answered. An APPROVAL still refuses.
    """
    await _enable(rig, SESSION_A)
    remote = rig.pool.remote
    remote.queued_engine_live = True
    remote.pending_gate = SimpleNamespace(request_id="a-7df9.0", kind="ask", title="", detail="")
    spoken = await _utter(rig, SESSION_A, "what about digimon cards")
    assert spoken.status_code == 200, spoken.text
    assert remote.admissions[-1]["text"] == "what about digimon cards"

    # An approval is a real parked card whatever else is queued beside it.
    remote.admissions.clear()
    remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="bash", detail="rm -rf build/"
    )
    refused = await _utter(rig, SESSION_A, "yes")
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "approval_pending"
    assert remote.admissions == []


@pytest.mark.asyncio
async def test_a_blocking_ask_still_refuses_speech_when_the_queue_is_off(rig: Any) -> None:
    """``LOP_ASK_NONBLOCKING=0``: no ``asks`` on the state, the ask-kind gate is a
    real parked card and speech must not withdraw it (V13)."""
    await _enable(rig, SESSION_A)
    remote = rig.pool.remote
    remote.queued_engine_live = False
    remote.pending_gate = SimpleNamespace(request_id="ask-1", kind="ask", title="", detail="")
    refused = await _utter(rig, SESSION_A, "the blue one")
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "approval_pending"
    assert remote.admissions == []


# -- one chosen provider: speech.provider drives every voice route ----------------


async def _recording_with(rig: Any, **fields: str) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4()), **fields},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )


async def _phrase_speech(rig: Any, **extra: Any) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice/speech",
        json={
            "session_id": SESSION_A,
            "source": {"kind": "phrase", "key": "voice_off"},
            "segment": 0,
            **extra,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chosen", "stt_rung", "tts_rung"),
    [
        ("elevenlabs", "PROVIDER_STT_ELEVENLABS", "PROVIDER_TTS_ELEVENLABS"),
        ("radient", "PROVIDER_STT_RADIENT", "PROVIDER_TTS_RADIENT"),
    ],
)
async def test_the_chosen_provider_pins_hearing_and_speaking_to_one_rung(
    rig: Any, cascades: SimpleNamespace, chosen: str, stt_rung: str, tts_rung: str
) -> None:
    from local_operator.stt import AudioPath
    from local_operator.tts import VoicePath

    _choose(rig, chosen)
    await _enable(rig, SESSION_A)
    assert (await _recording_with(rig)).status_code == 200
    ((_head, stt_kwargs),) = cascades.stt
    assert stt_kwargs["only"] == AudioPath[stt_rung]

    assert (await _phrase_speech(rig)).status_code == 200
    assert cascades.resolve[-1]["only"] == VoicePath[tts_rung]
    assert cascades.tts[-1][1]["only"] == VoicePath[tts_rung]


@pytest.mark.asyncio
async def test_unset_refuses_a_recording_and_costs_nothing(
    rig: Any, cascades: SimpleNamespace
) -> None:
    _choose(rig, None)
    await _enable(rig, SESSION_A)
    response = await _recording_with(rig)
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "voice_provider_unset"
    assert "Radient" in detail["text"] and "ElevenLabs" in detail["text"]
    assert cascades.stt == [] and cascades.resolve == [] and cascades.spend == []
    assert rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_unset_refuses_speech_before_the_phrase_cache_or_any_cascade(
    rig: Any, cascades: SimpleNamespace
) -> None:
    _choose(rig, None)
    await _enable(rig, SESSION_A)
    response = await _phrase_speech(rig)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "voice_provider_unset"
    assert cascades.tts == [] and cascades.resolve == [] and cascades.spend == []


@pytest.mark.asyncio
async def test_a_client_provider_does_not_stand_in_for_an_unset_choice(
    rig: Any, cascades: SimpleNamespace
) -> None:
    """The old app pinned ElevenLabs itself. With nothing chosen, that is still
    nothing chosen: the setting is the one place the choice lives."""
    _choose(rig, None)
    await _enable(rig, SESSION_A)
    heard = await _recording_with(rig, provider="elevenlabs")
    spoke = await _phrase_speech(rig, provider="elevenlabs")
    assert heard.status_code == spoke.status_code == 409
    assert (
        heard.json()["detail"]["code"] == spoke.json()["detail"]["code"] == "voice_provider_unset"
    )
    assert cascades.stt == [] and cascades.tts == []


@pytest.mark.asyncio
async def test_the_text_door_needs_no_provider(rig: Any, cascades: SimpleNamespace) -> None:
    """Typed words involve no hearing and no speaking, so there is nothing to refuse."""
    _choose(rig, None)
    await _enable(rig, SESSION_A)
    assert (await _utter(rig, SESSION_A, "list files")).status_code == 200
    assert cascades.stt == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chosen", "client_says", "expected_stt", "expected_tts"),
    [
        ("radient", "elevenlabs", "PROVIDER_STT_RADIENT", "PROVIDER_TTS_RADIENT"),
        ("elevenlabs", "radient", "PROVIDER_STT_ELEVENLABS", "PROVIDER_TTS_ELEVENLABS"),
        ("elevenlabs", "elevenlabs", "PROVIDER_STT_ELEVENLABS", "PROVIDER_TTS_ELEVENLABS"),
    ],
)
async def test_the_setting_wins_over_the_old_client_provider_field(
    rig: Any,
    cascades: SimpleNamespace,
    chosen: str,
    client_says: str,
    expected_stt: str,
    expected_tts: str,
) -> None:
    from local_operator.stt import AudioPath
    from local_operator.tts import VoicePath

    _choose(rig, chosen)
    await _enable(rig, SESSION_A)
    assert (await _recording_with(rig, provider=client_says)).status_code == 200
    assert cascades.stt[-1][1]["only"] == AudioPath[expected_stt]
    assert (await _phrase_speech(rig, provider=client_says)).status_code == 200
    assert cascades.tts[-1][1]["only"] == VoicePath[expected_tts]


@pytest.mark.asyncio
@pytest.mark.parametrize("not_pinnable", ["openai", "groq", "qwen"])
async def test_a_client_provider_outside_the_two_is_a_validation_error(
    rig: Any, cascades: SimpleNamespace, not_pinnable: str
) -> None:
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)
    heard = await _recording_with(rig, provider=not_pinnable)
    spoke = await _phrase_speech(rig, provider=not_pinnable)
    assert heard.status_code == 422 and spoke.status_code == 422
    assert cascades.stt == [] and cascades.tts == [] and rig.pool.remote.admissions == []


@pytest.mark.asyncio
async def test_the_choice_is_read_per_request_from_disk(
    rig: Any, cascades: SimpleNamespace
) -> None:
    """No restart: a settings write between two requests changes the second."""
    from local_operator.stt import AudioPath

    await _enable(rig, SESSION_A)
    _choose(rig, "elevenlabs")
    assert (await _recording_with(rig)).status_code == 200
    _choose(rig, "radient")
    assert (await _recording_with(rig)).status_code == 200
    assert [kw["only"] for _h, kw in cascades.stt] == [
        AudioPath.PROVIDER_STT_ELEVENLABS,
        AudioPath.PROVIDER_STT_RADIENT,
    ]


# -- the real cascades, real credentials: a pin never reaches the other provider ---


@pytest.fixture
def real_cascades(rig: Any, monkeypatch: pytest.MonkeyPatch):
    """The REAL cascades over a real credential store; only the two network legs
    are replaced, by recorders. ``legs`` says which provider was actually called."""
    from datetime import datetime

    from local_operator.providers.auth_store import AuthStore
    from local_operator.server.dependencies import get_provider_auth_store
    from local_operator.stt import cascade as stt_cascade
    from local_operator.tts import cascade as tts_cascade

    store = AuthStore(db_path=rig.root / "auth.db", config_dir=rig.root)

    async def provide() -> AuthStore:
        return store

    rig.app.dependency_overrides[get_provider_auth_store] = provide
    # A stored Radient login only counts for the canonical Radient base URL, so
    # the real resolver needs it here (the stubbed-cascade tests need no URL).
    rig.app.state.env_config = SimpleNamespace(radient_api_base_url="https://api.radienthq.com/v1")
    legs = SimpleNamespace(
        called=[], formats=[], radient_error=None, spend=[], sign_in=None, elevenlabs_key=None
    )

    def sign_in_radient(access: str = "a") -> None:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "refresh": "r",
                "access": access,
                "expires": int(datetime.now().timestamp() * 1000) + 3_600_000,
            },
        )

    def store_elevenlabs_key(key: str = "el-test-key") -> None:
        store.upsert_credential("elevenlabs", {"type": "api_key", "source": "login", "key": key})

    legs.sign_in = sign_in_radient
    legs.elevenlabs_key = store_elevenlabs_key

    async def tts_radient(*_a: Any, **_k: Any) -> Any:
        legs.called.append("tts:radient")
        legs.formats.append(_k.get("response_format"))
        raise legs.radient_error

    async def tts_elevenlabs(*_a: Any, **_k: Any) -> Any:
        legs.called.append("tts:elevenlabs")
        legs.formats.append(_k.get("response_format"))
        return (_mp3_bytes(), "elevenlabs", [], [])

    async def stt_radient(*_a: Any, **_k: Any) -> Any:
        legs.called.append("stt:radient")
        raise legs.radient_error

    async def stt_elevenlabs(*_a: Any, **_k: Any) -> Any:
        legs.called.append("stt:elevenlabs")
        raise AssertionError("not reached unless the pin leaks")

    monkeypatch.setattr(tts_cascade, "_run_radient_rung", tts_radient)
    monkeypatch.setattr(tts_cascade, "_run_elevenlabs_rung", tts_elevenlabs)
    monkeypatch.setattr(stt_cascade, "_run_radient_rung", stt_radient)
    monkeypatch.setattr(stt_cascade, "_run_elevenlabs_rung", stt_elevenlabs)
    monkeypatch.setattr(desktop_voice, "emit_voice_spend", lambda **kw: legs.spend.append(kw))
    yield legs
    store.close()


def _mp3_bytes() -> bytes:
    """A tagged MP3 head: all the sniffer (and so the response label) reads."""
    return b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 8


def _credit_exhausted() -> Any:
    from local_operator.clients._http import APIError

    return APIError("insufficient credits", status_code=402)


@pytest.mark.asyncio
async def test_a_radient_402_on_speech_says_so_and_never_reaches_elevenlabs(
    rig: Any, real_cascades: SimpleNamespace
) -> None:
    real_cascades.sign_in()
    real_cascades.elevenlabs_key()  # a usable key sits right there; the pin forbids it
    real_cascades.radient_error = _credit_exhausted()
    _choose(rig, "radient")
    await _enable(rig, SESSION_A)
    response = await _phrase_speech(rig)
    assert response.status_code == 402, response.text
    assert response.json()["detail"]["code"] == "voice_credit_low"
    assert response.json()["detail"]["text"] == (
        "Your Radient credit balance is too low for speech. "
        "Add credits in the Radient Console to continue."
    )
    assert real_cascades.called == ["tts:radient"]
    assert real_cascades.spend == []


@pytest.mark.asyncio
async def test_a_radient_402_on_hearing_says_so_admits_nothing_and_never_reaches_elevenlabs(
    rig: Any, real_cascades: SimpleNamespace
) -> None:
    real_cascades.sign_in()
    real_cascades.elevenlabs_key()
    real_cascades.radient_error = _credit_exhausted()
    _choose(rig, "radient")
    await _enable(rig, SESSION_A)
    response = await _recording_with(rig)
    assert response.status_code == 402, response.text
    # The status is the cascade's; the CODE is the one hearing failure this API
    # documents, so a client branches on one name whatever went wrong upstream.
    assert response.json()["detail"]["code"] == "stt_unavailable"
    assert "Radient credit balance is too low" in response.json()["detail"]["text"]
    assert real_cascades.called == ["stt:radient"]
    assert rig.pool.remote.admissions == [] and real_cascades.spend == []


@pytest.mark.asyncio
async def test_signed_out_of_radient_refuses_with_radients_reason_and_calls_nobody(
    rig: Any, real_cascades: SimpleNamespace
) -> None:
    real_cascades.elevenlabs_key()  # not signed in to Radient, but a key IS stored
    _choose(rig, "radient")
    await _enable(rig, SESSION_A)
    spoke = await _phrase_speech(rig)
    heard = await _recording_with(rig)
    assert spoke.status_code == heard.status_code == 409
    assert spoke.json()["detail"]["code"] == "voice_unavailable"
    assert heard.json()["detail"]["code"] == "stt_unavailable"
    assert "Not signed in to Radient." in spoke.json()["detail"]["text"]
    assert "Not signed in to Radient." in heard.json()["detail"]["text"]
    assert real_cascades.called == []


@pytest.mark.asyncio
async def test_the_elevenlabs_pin_never_reaches_radient(
    rig: Any, real_cascades: SimpleNamespace
) -> None:
    real_cascades.sign_in()  # signed in to Radient, NO ElevenLabs key
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)
    spoke = await _phrase_speech(rig)
    heard = await _recording_with(rig)
    assert spoke.status_code == heard.status_code == 409
    assert "No ElevenLabs API key is stored." in spoke.json()["detail"]["text"]
    assert "No ElevenLabs API key is stored." in heard.json()["detail"]["text"]
    assert real_cascades.called == []

    # With the key stored the same pin speaks, and still never touches Radient.
    real_cascades.elevenlabs_key()
    assert (await _phrase_speech(rig)).status_code == 200
    assert real_cascades.called == ["tts:elevenlabs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["radient", "elevenlabs"])
async def test_a_pinned_speech_request_asks_the_leg_for_mp3_never_wav(
    rig: Any, real_cascades: SimpleNamespace, provider: str
) -> None:
    real_cascades.sign_in()
    real_cascades.elevenlabs_key()
    real_cascades.radient_error = _credit_exhausted()
    _choose(rig, provider)
    await _enable(rig, SESSION_A)
    await _phrase_speech(rig)
    assert real_cascades.formats == ["mp3"]


def _hub_refusing_wav(legs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Radient leg that answers as the real hub does: ``wav`` is a 400, ``mp3`` is audio."""
    from local_operator.clients._http import APIError
    from local_operator.tts import cascade as tts_cascade

    async def tts_radient(*_a: Any, **kwargs: Any) -> Any:
        legs.called.append("tts:radient")
        if kwargs.get("response_format") != "mp3":
            raise APIError(
                "[bad_request] Invalid request parameters: Key: 'SpeechRequest.ResponseFormat' "
                "Error:Field validation for 'ResponseFormat' failed on the 'oneof' tag",
                status_code=400,
            )
        return (_mp3_bytes(), "radient", [], [])

    monkeypatch.setattr(tts_cascade, "_run_radient_rung", tts_radient)


@pytest.mark.asyncio
async def test_radient_speech_is_served_as_audio_mpeg_where_wav_was_a_400(
    rig: Any, real_cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_cascades.sign_in()
    _hub_refusing_wav(real_cascades, monkeypatch)
    _choose(rig, "radient")
    await _enable(rig, SESSION_A)
    response = await _phrase_speech(rig)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == _mp3_bytes()
    assert response.headers["x-voice-path"] == "provider_tts_radient"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["radient", "elevenlabs"])
async def test_a_pinned_upstream_failure_carries_the_providers_own_sentence(
    rig: Any, real_cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from local_operator.clients._http import APIError

    real_cascades.sign_in()
    real_cascades.elevenlabs_key()
    real_cascades.radient_error = APIError(
        "[bad_request] Invalid request parameters: ResponseFormat", status_code=400
    )
    _choose(rig, provider)
    await _enable(rig, SESSION_A)
    real_cascades.called.clear()
    if provider == "elevenlabs":
        # The ElevenLabs leg is the one that fails here; the sentence is ITS own.
        from local_operator.tts import cascade as tts_cascade

        async def refused(*_a: Any, **_k: Any) -> Any:
            raise APIError("quota_exceeded: detail from ElevenLabs", status_code=500)

        monkeypatch.setattr(tts_cascade, "_run_elevenlabs_rung", refused)
    response = await _phrase_speech(rig)
    assert response.status_code == 502, response.text
    text = response.json()["detail"]["text"]
    assert ("ResponseFormat" if provider == "radient" else "detail from ElevenLabs") in text


@pytest.mark.asyncio
async def test_the_audio_is_labelled_with_what_it_actually_is(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ElevenLabs answers MP3, Groq answers WAV: the label follows the bytes, so
    the page never decodes an MP3 believing it is PCM."""
    from local_operator.tts import TtsOutcome, VoicePath

    mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 8

    async def synthesize(text: str, **kwargs: Any) -> TtsOutcome:
        return TtsOutcome(
            audio=mp3, path=VoicePath.PROVIDER_TTS_ELEVENLABS, attempts=(), provider="elevenlabs"
        )

    monkeypatch.setattr("local_operator.tts.cascade.synthesize_speech", synthesize)
    await _enable(rig, SESSION_A)
    body = {"session_id": SESSION_A, "source": {"kind": "phrase", "key": "voice_off"}}
    response = await rig.client.post("/v1/desktop/voice/speech", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["x-voice-path"] == "provider_tts_elevenlabs"


@pytest.mark.asyncio
async def test_status_says_which_provider_is_chosen_and_whether_each_can_serve(
    rig: Any, real_cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readiness comes from stored credentials alone: no network call, no spend.
    Both legs are replaced by recorders that fail the test if they are ever hit."""
    from local_operator.stt import cascade as stt_cascade
    from local_operator.tts import cascade as tts_cascade

    async def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("status must not call a provider")

    monkeypatch.setattr(tts_cascade, "synthesize_speech", never)
    monkeypatch.setattr(stt_cascade, "transcribe_audio", never)

    async def status() -> dict[str, Any]:
        return (await rig.client.get("/v1/desktop/voice")).json()["result"]

    nothing = await status()
    assert nothing["provider"] is None
    assert nothing["providers"] == {
        "radient": {"ready": False, "reason": "Not signed in to Radient."},
        "elevenlabs": {"ready": False, "reason": "No ElevenLabs API key is stored."},
    }
    assert nothing["byok"]["ready"] is False

    real_cascades.sign_in()
    _choose(rig, "radient")
    signed_in = await status()
    assert signed_in["provider"] == "radient"
    assert signed_in["providers"]["radient"] == {"ready": True, "reason": None}
    assert signed_in["providers"]["elevenlabs"]["ready"] is False

    real_cascades.elevenlabs_key()
    _choose(rig, "elevenlabs")
    both = await status()
    assert both["provider"] == "elevenlabs"
    assert both["providers"]["radient"]["ready"] is True
    assert both["providers"]["elevenlabs"] == {"ready": True, "reason": None}
    assert both["byok"]["ready"] is True
    assert both["active"] is None, "the slot half of the status is unchanged"
    assert real_cascades.called == []


@pytest.mark.asyncio
async def test_status_reports_a_provider_ready_only_when_it_can_both_hear_and_speak(
    rig: Any, real_cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One half missing is not ready, and the reason names the half that is."""
    from local_operator.stt import AudioPath, AudioPathResolution, RungAvailability
    from local_operator.stt import cascade as stt_cascade

    real_cascades.sign_in()

    async def deaf(**_k: Any) -> AudioPathResolution:
        rung = RungAvailability(AudioPath.PROVIDER_STT_RADIENT, False, "Hearing is down.")
        return AudioPathResolution(
            path=AudioPath.NONE, reason="x", rungs=(rung,), model_audio_capable=False
        )

    monkeypatch.setattr(stt_cascade, "resolve_audio_path", deaf)
    radient = (await rig.client.get("/v1/desktop/voice")).json()["result"]["providers"]["radient"]
    assert radient == {"ready": False, "reason": "Hearing is down."}


# -- B3: a deleted conversation leaves the slot ----------------------------------


@pytest.mark.asyncio
async def test_a_slot_naming_a_removed_conversation_reads_as_empty(rig: Any) -> None:
    """Deleted by another surface: the slot file stays, but nothing treats it as live."""
    import shutil

    await _enable(rig, SESSION_A)
    shutil.rmtree(rig.root / "sessions" / SESSION_A)
    status = (await rig.client.get("/v1/desktop/voice")).json()["result"]
    assert status["active"] is None
    speak = await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})
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


# -- B2: the speech cascades behind the voice routes (session-voice-groq D1/D3/D4) --


def _wav_bytes() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


@pytest.fixture
def cascades(rig: Any, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Both cascades stubbed at the module the route imports them from."""
    from local_operator.stt import AudioPath, SttOutcome
    from local_operator.tts import (
        RungAvailability,
        TtsOutcome,
        VoicePath,
        VoicePathResolution,
    )

    calls = SimpleNamespace(
        tts=[],
        stt=[],
        resolve=[],
        spend=[],
        tts_error=None,
        stt_error=None,
        text="list files",
        tts_audio=None,
        tts_path=VoicePath.PROVIDER_TTS_GROQ,
    )

    async def synthesize(text: str, **kwargs: Any) -> TtsOutcome:
        calls.tts.append((text, kwargs))
        if calls.tts_error is not None:
            raise calls.tts_error
        return TtsOutcome(
            audio=calls.tts_audio or _wav_bytes(),
            path=calls.tts_path,
            attempts=(),
            provider="groq",
        )

    async def resolve(**kwargs: Any) -> VoicePathResolution:
        calls.resolve.append(kwargs)
        rung = RungAvailability(calls.tts_path, True, "A key is stored.")
        return VoicePathResolution(path=calls.tts_path, reason="ok", rungs=(rung,), servable=True)

    async def transcribe(path: Path, **kwargs: Any) -> SttOutcome:
        calls.stt.append((path.read_bytes()[:4], kwargs))
        if calls.stt_error is not None:
            raise calls.stt_error
        return SttOutcome(text=calls.text, path=AudioPath.PROVIDER_STT_GROQ, attempts=())

    monkeypatch.setattr("local_operator.tts.cascade.synthesize_speech", synthesize)
    monkeypatch.setattr("local_operator.tts.cascade.resolve_voice_path", resolve)
    monkeypatch.setattr("local_operator.stt.cascade.transcribe_audio", transcribe)
    monkeypatch.setattr(desktop_voice, "emit_voice_spend", lambda **kw: calls.spend.append(kw))
    # Voice mode refuses until a provider is chosen; these tests are about what
    # the cascades do once one is, so they choose ElevenLabs (the stubs answer
    # whatever rung they like). Tests of the unset state clear it.
    _choose(rig, "elevenlabs")
    return calls


async def _speech(rig: Any, source: dict[str, Any], segment: int = 0) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice/speech",
        json={"session_id": SESSION_A, "source": source, "segment": segment},
    )


async def _recording(rig: Any, data: bytes | None = None) -> Any:
    return await rig.client.post(
        "/v1/desktop/voice/utterance",
        data={"session_id": SESSION_A, "request_id": str(uuid.uuid4())},
        # The declared type lies; the container is sniffed from the bytes.
        files={"file": ("a.mp3", data or _wav_bytes(), "audio/mpeg")},
    )


@pytest.mark.asyncio
async def test_a_recording_is_transcribed_through_the_cascade_and_admitted(
    rig: Any, cascades: SimpleNamespace
) -> None:
    await _enable(rig, SESSION_A)
    response = await _recording(rig)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["text"] == "list files" and result["path"] == "provider_stt_groq"
    # The rung token is the message's input_path; the words steer as dictation.
    assert rig.pool.remote.admissions[-1]["input_path"] == "provider_stt_groq"
    ((head, kwargs),) = cascades.stt
    assert head == b"RIFF" and kwargs["include_groq"] is True
    assert kwargs["correlation_session_id"] == SESSION_A
    (record,) = cascades.spend
    assert record["channel"] == "stt" and record["provider"] == "groq"
    assert record["session_id"] == SESSION_A


@pytest.mark.asyncio
async def test_a_retried_recording_is_a_second_spend_record(
    rig: Any, cascades: SimpleNamespace
) -> None:
    """A same-``request_id`` retry pays STT again; the spend record id is derived from
    the id handed to ``emit_voice_spend``, so it must differ per paid call."""
    await _enable(rig, SESSION_A)
    request_id = str(uuid.uuid4())
    for _ in range(2):
        response = await rig.client.post(
            "/v1/desktop/voice/utterance",
            data={"session_id": SESSION_A, "request_id": request_id},
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        )
        assert response.status_code == 200, response.text
    first, second = cascades.spend
    assert first["request_id"] != second["request_id"]
    assert first["request_id"].startswith(request_id) and second["request_id"].startswith(
        request_id
    )


@pytest.mark.asyncio
async def test_unsupported_and_oversized_recordings_cost_nothing(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _enable(rig, SESSION_A)
    bad = await _recording(rig, b"definitely not audio")
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_voice_request"
    monkeypatch.setattr(desktop_voice, "MAX_UPLOAD_BYTES", 64)
    big = await _recording(rig, _wav_bytes())
    assert big.status_code == 413
    assert big.json()["detail"]["code"] == "voice_upload_too_large"
    assert cascades.stt == [] and cascades.spend == []
    assert rig.pool.remote.admissions == []
    assert desktop_voice.MAX_UPLOAD_BYTES == 64  # the patched ceiling, restored on teardown


@pytest.mark.asyncio
async def test_an_oversized_content_length_is_refused_before_the_form_is_parsed(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _enable(rig, SESSION_A)
    from starlette.requests import Request

    async def _form_must_not_parse(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the multipart form was parsed")

    monkeypatch.setattr(Request, "_get_form", _form_must_not_parse)
    response = await rig.client.post(
        "/v1/desktop/voice/utterance",
        content=b"x",
        headers={
            "content-type": "multipart/form-data; boundary=b",
            "content-length": str(desktop_voice.MAX_UPLOAD_BYTES + 1),
        },
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "voice_upload_too_large"
    assert cascades.stt == []


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", ["10", None])
async def test_a_lying_or_missing_content_length_is_caught_by_the_read_backstop(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, declared: str | None
) -> None:
    await _enable(rig, SESSION_A)
    monkeypatch.setattr(desktop_voice, "MAX_UPLOAD_BYTES", 64)
    boundary = b"BND"
    sep = b"--" + boundary + b"\r\n"
    body = (
        sep
        + b'Content-Disposition: form-data; name="session_id"\r\n\r\n'
        + SESSION_A.encode()
        + b"\r\n"
        + sep
        + b'Content-Disposition: form-data; name="request_id"\r\n\r\n'
        + str(uuid.uuid4()).encode()
        + b"\r\n"
        + sep
        + b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        + b"Content-Type: audio/wav\r\n\r\n"
        + _wav_bytes()
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )

    async def chunks() -> Any:
        yield body

    headers = {"content-type": "multipart/form-data; boundary=BND"}
    if declared is not None:
        headers["content-length"] = declared  # a lie: far below the real size
    response = await rig.client.post(
        "/v1/desktop/voice/utterance",
        content=body if declared is not None else chunks(),
        headers=headers,
    )
    assert response.status_code == 413, response.text
    assert response.json()["detail"]["code"] == "voice_upload_too_large"
    assert cascades.stt == []


def test_the_upload_ceiling_is_16_mib() -> None:
    import importlib

    assert importlib.reload(desktop_voice).MAX_UPLOAD_BYTES == 16 * 1024 * 1024


@pytest.mark.asyncio
async def test_a_recording_is_refused_while_a_card_is_pending_before_stt(
    rig: Any, cascades: SimpleNamespace
) -> None:
    """V7 + V13 on the file door: the refusal comes before any STT spend."""
    await _enable(rig, SESSION_A)
    remote = rig.pool.remote
    remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="bash", detail="rm -rf build/"
    )
    response = await _recording(rig)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "approval_pending"
    assert cascades.stt == [] and cascades.spend == []
    assert remote.admissions == [] and remote.gate_answers == []


@pytest.mark.asyncio
async def test_a_card_that_parks_during_stt_still_refuses(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _enable(rig, SESSION_A)
    remote = rig.pool.remote
    real = cascades.stt

    async def transcribe(path: Path, **kwargs: Any):
        from local_operator.stt import AudioPath, SttOutcome

        real.append((b"", kwargs))
        remote.pending_gate = SimpleNamespace(
            request_id="g", kind="approval", title="bash", detail="x"
        )
        return SttOutcome(text="yes", path=AudioPath.PROVIDER_STT_GROQ, attempts=())

    monkeypatch.setattr("local_operator.stt.cascade.transcribe_audio", transcribe)
    response = await _recording(rig)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "approval_pending"
    assert remote.admissions == [] and remote.gate_answers == []


@pytest.mark.asyncio
async def test_no_stt_rung_is_stt_unavailable(rig: Any, cascades: SimpleNamespace) -> None:
    from local_operator.stt import AudioPath, AudioPathResolution
    from local_operator.stt.cascade import SttUnavailable

    cascades.stt_error = SttUnavailable(
        "none",
        resolution=AudioPathResolution(
            path=AudioPath.NONE, reason="No provider.", rungs=(), model_audio_capable=False
        ),
    )
    await _enable(rig, SESSION_A)
    response = await _recording(rig)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "stt_unavailable"
    assert rig.pool.remote.admissions == []


def _limited(status: int = 429, **details: Any) -> Any:
    from local_operator.clients._http import APIError

    return APIError("Too many requests.", status_code=status, details=details)


@pytest.mark.parametrize(
    ("details", "scope", "retry", "remaining"),
    [
        ({"retry_after_s": 7.0, "remaining_requests": 12}, "minute", 7.0, 12),
        ({"retry_after_s": 41000.0, "remaining_requests": 0}, "day", 41000.0, 0),
        # No remaining count: the size of retry-after decides (D4's fallback).
        ({"retry_after_s": 8.0}, "minute", 8.0, None),
        ({"retry_after_s": 3600.0}, "day", 3600.0, None),
    ],
)
@pytest.mark.asyncio
async def test_a_tts_429_maps_to_voice_rate_limited_with_retry_after(
    rig: Any,
    cascades: SimpleNamespace,
    details: dict[str, Any],
    scope: str,
    retry: float | None,
    remaining: int | None,
) -> None:
    from local_operator.tts import VoicePath
    from local_operator.tts.cascade import TtsUnavailable as Unavailable

    cascades.tts_error = Unavailable(
        "x",
        resolution=None,  # type: ignore[arg-type]
        error=_limited(**details),
        failed_path=VoicePath.PROVIDER_TTS_GROQ,
    )
    await _enable(rig, SESSION_A)
    response = await _speech(rig, {"kind": "message", "message_id": ASSISTANT_ID})
    assert response.status_code == 429, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "voice_rate_limited"
    assert detail["scope"] == scope
    assert detail["retry_after_s"] == retry
    assert detail["remaining_requests"] == remaining
    assert detail["rung"] == "provider_tts_groq"
    assert detail["text"]
    if retry is not None:
        assert response.headers["retry-after"] == str(int(-(-retry // 1)))


@pytest.mark.parametrize(
    "details",
    [
        {"retry_after_s": float("nan")},
        {"retry_after_s": float("inf")},
        {"retry_after_s": -5.0},
        {"remaining_requests": float("inf")},
        {"remaining_requests": float("nan")},
        {"remaining_requests": "inf"},
        {"retry_after_s": float("inf"), "remaining_requests": float("nan")},
    ],
)
@pytest.mark.asyncio
async def test_a_non_finite_limit_header_is_dropped_not_a_crash(
    rig: Any, cascades: SimpleNamespace, details: dict[str, Any]
) -> None:
    """``nan``/``inf`` parse as floats but are not seconds or counts: they crash
    ``math.ceil`` and are not valid JSON, so the refusal must be the unknown-scope
    429 with a strict-JSON body and no ``Retry-After``."""
    import json as _json

    from local_operator.tts import VoicePath
    from local_operator.tts.cascade import TtsUnavailable as Unavailable

    cascades.tts_error = Unavailable(
        "x",
        resolution=None,  # type: ignore[arg-type]
        error=_limited(**details),
        failed_path=VoicePath.PROVIDER_TTS_GROQ,
    )
    await _enable(rig, SESSION_A)
    response = await _speech(rig, {"kind": "message", "message_id": ASSISTANT_ID})
    assert response.status_code == 429, response.text

    def _no_constants(token: str) -> Any:
        raise AssertionError(f"not valid JSON: {token}")

    detail = _json.loads(response.text, parse_constant=_no_constants)["detail"]
    assert detail["code"] == "voice_rate_limited"
    assert detail["scope"] is None
    assert detail["retry_after_s"] is None and detail["remaining_requests"] is None
    assert "retry-after" not in response.headers


@pytest.mark.asyncio
async def test_an_stt_429_is_voice_rate_limited_and_admits_nothing(
    rig: Any, cascades: SimpleNamespace
) -> None:
    from local_operator.stt import AudioPath, AudioPathResolution, SttAttempt
    from local_operator.stt.cascade import SttUnavailable

    cascades.stt_error = SttUnavailable(
        "x",
        resolution=AudioPathResolution(
            path=AudioPath.NONE, reason="r", rungs=(), model_audio_capable=False
        ),
        attempts=(SttAttempt(path=AudioPath.PROVIDER_STT_GROQ, outcome="failed"),),
        error=_limited(retry_after_s=5.0, remaining_requests=0),
        status_code=429,
        detail="rate limited",
    )
    await _enable(rig, SESSION_A)
    response = await _recording(rig)
    assert response.status_code == 429, response.text
    detail = response.json()["detail"]
    assert (detail["code"], detail["scope"], detail["retry_after_s"]) == (
        "voice_rate_limited",
        "day",
        5.0,
    )
    assert detail["rung"] == "provider_stt_groq"
    assert rig.pool.remote.admissions == [] and cascades.spend == []


@pytest.mark.asyncio
async def test_a_non_429_failure_keeps_the_speech_refusal_path(
    rig: Any, cascades: SimpleNamespace
) -> None:
    from local_operator.tts import VoicePath
    from local_operator.tts.cascade import TtsUnavailable

    cascades.tts_error = TtsUnavailable(
        "x",
        resolution=None,  # type: ignore[arg-type]
        error=_limited(status=402),
        failed_path=VoicePath.PROVIDER_TTS_GROQ,
    )
    await _enable(rig, SESSION_A)
    response = await _speech(rig, {"kind": "message", "message_id": ASSISTANT_ID})
    assert response.status_code == 402, response.text
    assert response.json()["detail"].get("code") != "voice_rate_limited"


@pytest.mark.parametrize(
    ("upstream", "status", "code"),
    [
        # A rejected provider key is NOT a 401 from this API: that status is what
        # ``require_desktop`` answers for a bad app token.
        (401, 502, "voice_provider_rejected"),
        (402, 402, "voice_credit_low"),
        (500, 502, "voice_upstream_failed"),
        (503, 503, "voice_temporarily_unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_a_speech_refusal_carries_a_machine_code_and_never_a_bare_401(
    rig: Any, cascades: SimpleNamespace, upstream: int, status: int, code: str
) -> None:
    from local_operator.tts import VoicePath
    from local_operator.tts.cascade import TtsUnavailable

    cascades.tts_error = TtsUnavailable(
        "x",
        resolution=None,  # type: ignore[arg-type]
        error=_limited(status=upstream),
        failed_path=VoicePath.PROVIDER_TTS_ELEVENLABS,
    )
    await _enable(rig, SESSION_A)
    response = await _speech(rig, {"kind": "message", "message_id": ASSISTANT_ID})
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code and detail["text"]


@pytest.mark.asyncio
async def test_a_facade_that_cannot_say_whether_a_card_waits_refuses_instead_of_steering(
    rig: Any,
) -> None:
    """Fail CLOSED: a wrong "no card" lets a steer cancel an approval."""

    class Unsynchronised(FakeRemote):
        @property
        def pending_gate(self) -> Any:
            raise RuntimeError("frontend state has not synchronized")

        @pending_gate.setter
        def pending_gate(self, _value: Any) -> None:
            return None

    remote = Unsynchronised()
    rig.pool.remote = remote
    rig.pool.bridge = FakeBridge(remote)
    await _enable(rig, SESSION_A)
    response = await _utter(rig, SESSION_A, "also check the tests")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "session_syncing" and detail["text"]
    assert remote.admissions == [] and remote.gate_answers == []


@pytest.mark.asyncio
async def test_the_spoken_stand_ins_come_from_the_locale_not_the_module(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``speakable`` is pure: the route hands it the active locale's words."""
    text = "See https://example.com now.\n```\nx = 1\n```\nDone."
    monkeypatch.setattr(FakeBridge, "history", _history_with(ASSISTANT_ID, text), raising=True)
    monkeypatch.setattr(
        desktop_voice,
        "_spoken_words",
        lambda _locale: {"tail": "TAIL.", "code_omitted": "CODE-GONE", "link": "ENLACE"},
    )
    await _enable(rig, SESSION_A)
    result = (await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})).json()[
        "result"
    ]
    assert "ENLACE" in result["text"] and "CODE-GONE" in result["text"]
    assert "link" not in result["text"] and "code block omitted" not in result["text"]


@pytest.mark.asyncio
async def test_the_gate_sentence_comes_from_the_catalogue(rig: Any) -> None:
    await _enable(rig, SESSION_A)
    rig.pool.remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="bash", detail="run ls"
    )
    source = {"kind": "gate", "request_id": "gate-1"}
    titled = (await _text(rig, SESSION_A, source)).json()["result"]["text"]
    assert titled == "Approval needed: bash. run ls"
    rig.pool.remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="", detail=""
    )
    assert (await _text(rig, SESSION_A, source)).json()["result"]["text"] == "Approval needed."


@pytest.mark.asyncio
async def test_speech_returns_one_segment_with_the_cascade_flags(
    rig: Any, cascades: SimpleNamespace
) -> None:
    await _enable(rig, SESSION_A)
    text = await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})
    segments = text.json()["result"]["segments"]
    response = await _speech(rig, {"kind": "message", "message_id": ASSISTANT_ID}, 0)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert response.headers["x-voice-path"] == "provider_tts_groq"
    assert response.headers["x-voice-spoken-chars"] == str(len(segments[0]))
    ((spoken, kwargs),) = cascades.tts
    assert spoken == segments[0]
    assert kwargs["include_groq"] is True
    # Never "wav": Radient's hub refuses it, and MP3 is what every pinned rung answers.
    assert kwargs["response_format"] == "mp3"
    assert "include_local" not in kwargs
    (record,) = cascades.spend
    assert record["channel"] == "tts" and record["unit"] == "chars"
    assert record["units"] == float(len(segments[0])) and record["session_id"] == SESSION_A


@pytest.mark.asyncio
async def test_every_segment_is_addressable_and_one_past_the_end_is_404(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    long = "Here is a sentence about files. " * 20
    monkeypatch.setattr(FakeBridge, "history", _history_with(ASSISTANT_ID, long), raising=True)
    await _enable(rig, SESSION_A)
    source = {"kind": "message", "message_id": ASSISTANT_ID}
    segments = (await _text(rig, SESSION_A, source)).json()["result"]["segments"]
    assert len(segments) > 2
    for index, expected in enumerate(segments):
        assert (await _speech(rig, source, index)).status_code == 200
        assert cascades.tts[-1][0] == expected
    past = await _speech(rig, source, len(segments))
    assert past.status_code == 404
    assert past.json()["detail"]["code"] == "unknown_segment"
    assert len(cascades.tts) == len(segments), "the 404 reached the cascade"


def _history_with(message_id: str, text: str):
    async def history(self: Any, **kwargs: Any) -> dict[str, Any]:
        row = {
            "id": message_id,
            "type": "message",
            "ts": 1.0,
            "ts_source": "entry",
            "payload": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        }
        return {"entries": [row], "has_more": False, "cursor_missing": False}

    return history


@pytest.mark.asyncio
async def test_the_configured_cap_applies_before_segmenting(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator import settings_io
    from local_operator.voice import speakable

    long = "Another fairly ordinary sentence here. " * 60
    monkeypatch.setattr(FakeBridge, "history", _history_with(ASSISTANT_ID, long), raising=True)
    cm = ConfigManager(rig.root)
    settings_io.write_setting(cm, settings_io.BY_KEY["voice.max_spoken_chars"], 300)
    await _enable(rig, SESSION_A)
    result = (await _text(rig, SESSION_A, {"kind": "message", "message_id": ASSISTANT_ID})).json()[
        "result"
    ]
    assert result["truncated"] is True and result["spoken_chars"] <= 300
    assert "".join(result["segments"]) == result["text"]
    assert result["segments"][-1] == speakable.TRUNCATION_TAIL
    assert all(len(part) <= 200 for part in result["segments"])


def _unhashed(path: Path) -> str:
    """A cache file's name without the trailing rendering hash (``-<8 hex>``)."""
    import re

    return re.sub(r"-[0-9a-f]{8}(\.\w+)$", r"\1", path.name)


@pytest.mark.asyncio
async def test_a_voice_setting_change_misses_the_phrase_cache(
    rig: Any, cascades: SimpleNamespace
) -> None:
    """The cache name carries a hash of the whole descriptor: after a change to a
    ``speech.voice.*`` setting the old rendering must not be replayed."""
    from local_operator import settings_io

    await _enable(rig, SESSION_A)
    source = {"kind": "phrase", "key": "on_screen_first"}
    assert (await _speech(rig, source)).status_code == 200
    assert (await _speech(rig, source)).headers["x-voice-cache"] == "hit"
    assert len(cascades.tts) == 1

    # The app's own manager: the route reads ITS in-memory config, per call.
    settings_io.write_setting(
        rig.app.state.config_manager, settings_io.BY_KEY["speech.voice.tone"], "calm"
    )
    changed = await _speech(rig, source)
    assert changed.status_code == 200
    assert "x-voice-cache" not in changed.headers, "an old rendering was replayed"
    assert len(cascades.tts) == 2
    assert (await _speech(rig, source)).headers["x-voice-cache"] == "hit"
    names = sorted(p.name for p in (rig.root / "voice" / "phrase-cache").iterdir())
    assert len(names) == 2 and len({_unhashed(Path(n)) for n in names}) == 1


@pytest.mark.asyncio
async def test_a_phrase_hit_makes_no_cascade_call_and_a_miss_stores_it(
    rig: Any, cascades: SimpleNamespace
) -> None:
    await _enable(rig, SESSION_A)
    source = {"kind": "phrase", "key": "on_screen_first"}
    first = await _speech(rig, source)
    assert first.status_code == 200, first.text
    assert len(cascades.tts) == 1 and cascades.tts[0][0] == "Answer the approval on screen first."
    cached = list((rig.root / "voice" / "phrase-cache").glob("*.wav"))
    assert [_unhashed(path) for path in cached] == [
        "provider_tts_groq-female-en-on_screen_first.wav"
    ]

    second = await _speech(rig, source)
    assert second.status_code == 200
    assert second.content == first.content and second.headers["x-voice-cache"] == "hit"
    assert second.headers["content-type"] == "audio/wav"
    assert len(cascades.tts) == 1, "a cache hit reached the cascade"
    assert len(cascades.spend) == 1, "a cache hit was billed"


@pytest.mark.asyncio
async def test_a_cached_mp3_phrase_is_stored_and_served_as_audio_mpeg(
    rig: Any, cascades: SimpleNamespace
) -> None:
    from local_operator.tts import VoicePath

    cascades.tts_path = VoicePath.PROVIDER_TTS_RADIENT
    cascades.tts_audio = _mp3_bytes()
    await _enable(rig, SESSION_A)
    source = {"kind": "phrase", "key": "on_screen_first"}
    first = await _speech(rig, source)
    assert first.status_code == 200, first.text
    assert first.headers["content-type"] == "audio/mpeg"
    cached = list((rig.root / "voice" / "phrase-cache").iterdir())
    assert [_unhashed(path) for path in cached] == [
        "provider_tts_radient-female-en-on_screen_first.mp3"
    ]

    second = await _speech(rig, source)
    assert second.status_code == 200 and second.headers["x-voice-cache"] == "hit"
    assert second.content == first.content
    assert second.headers["content-type"] == "audio/mpeg"
    assert len(cascades.tts) == 1, "a cache hit reached the cascade"


@pytest.mark.asyncio
async def test_a_truncated_cache_file_is_replaced_not_served(
    rig: Any, cascades: SimpleNamespace
) -> None:
    await _enable(rig, SESSION_A)
    source = {"kind": "phrase", "key": "on_screen_first"}
    first = await _speech(rig, source)
    (cached,) = (rig.root / "voice" / "phrase-cache").glob("*.wav")
    cached.write_bytes(first.content[:6])  # cut inside the header
    again = await _speech(rig, source)
    assert again.status_code == 200 and "x-voice-cache" not in again.headers
    assert again.content == first.content
    assert len(cascades.tts) == 2
    assert cached.read_bytes() == first.content, "the bad file was not replaced"


@pytest.mark.asyncio
async def test_the_truncation_tail_segment_is_served_from_the_phrase_cache(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator import settings_io

    long = "Another fairly ordinary sentence here. " * 60
    monkeypatch.setattr(FakeBridge, "history", _history_with(ASSISTANT_ID, long), raising=True)
    settings_io.write_setting(
        ConfigManager(rig.root), settings_io.BY_KEY["voice.max_spoken_chars"], 300
    )
    await _enable(rig, SESSION_A)
    source = {"kind": "message", "message_id": ASSISTANT_ID}
    segments = (await _text(rig, SESSION_A, source)).json()["result"]["segments"]
    last = len(segments) - 1
    assert (await _speech(rig, source, last)).status_code == 200
    calls = len(cascades.tts)
    assert (await _speech(rig, source, last)).headers["x-voice-cache"] == "hit"
    assert len(cascades.tts) == calls


@pytest.mark.asyncio
async def test_the_phrase_cache_is_keyed_by_locale(rig: Any, cascades: SimpleNamespace) -> None:
    """A phrase cached in one language is never served after a language switch.

    The phrase's text follows the active locale, so a cache name without it
    would replay the old language's audio for the new language's sentence.
    """
    from local_operator.i18n.text import set_active_locale

    await _enable(rig, SESSION_A)
    source = {"kind": "phrase", "key": "on_screen_first"}
    english = await _speech(rig, source)
    assert english.status_code == 200, english.text
    assert cascades.tts[0][0] == "Answer the approval on screen first."

    set_active_locale("es")
    spanish = await _speech(rig, source)
    assert spanish.status_code == 200, spanish.text
    assert "x-voice-cache" not in spanish.headers, "a Spanish request was served English audio"
    assert len(cascades.tts) == 2 and cascades.tts[1][0] != cascades.tts[0][0]
    assert (await _speech(rig, source)).headers["x-voice-cache"] == "hit"
    assert len(cascades.tts) == 2

    set_active_locale("en")
    back = await _speech(rig, source)
    assert back.headers["x-voice-cache"] == "hit" and len(cascades.tts) == 2
    names = sorted(_unhashed(p) for p in (rig.root / "voice" / "phrase-cache").iterdir())
    assert names == [
        "provider_tts_groq-female-en-on_screen_first.wav",
        "provider_tts_groq-female-es-on_screen_first.wav",
    ]


@pytest.mark.asyncio
async def test_a_pre_locale_cache_file_is_never_served(rig: Any, cascades: SimpleNamespace) -> None:
    """Files named before the locale joined the key are left alone and never matched."""
    await _enable(rig, SESSION_A)
    first = await _speech(rig, {"kind": "phrase", "key": "voice_off"})
    (cached,) = (rig.root / "voice" / "phrase-cache").glob("*.wav")
    legacy = cached.with_name("provider_tts_groq-female-failed.wav")
    legacy.write_bytes(first.content)
    again = await _speech(rig, {"kind": "phrase", "key": "failed"})
    assert again.status_code == 200 and "x-voice-cache" not in again.headers
    assert legacy.exists()


@pytest.mark.asyncio
async def test_the_truncation_tail_follows_the_active_locale(
    rig: Any, cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The closing sentence is the active locale's phrase, spoken and cached as such."""
    from local_operator import settings_io
    from local_operator.i18n.text import set_active_locale

    long = "Another fairly ordinary sentence here. " * 60
    monkeypatch.setattr(FakeBridge, "history", _history_with(ASSISTANT_ID, long), raising=True)
    settings_io.write_setting(
        ConfigManager(rig.root), settings_io.BY_KEY["voice.max_spoken_chars"], 300
    )
    await _enable(rig, SESSION_A)
    set_active_locale("es")
    tail = desktop_voice._wv.phrase_truncated().envelope()["text"]
    assert tail != "That's where I'll stop \u2014 the rest is on screen."
    source = {"kind": "message", "message_id": ASSISTANT_ID}
    result = (await _text(rig, SESSION_A, source)).json()["result"]
    assert result["segments"][-1] == tail and result["text"].endswith(tail)
    assert result["spoken_chars"] <= 300
    assert "".join(result["segments"]) == result["text"]
    last = len(result["segments"]) - 1
    assert (await _speech(rig, source, last)).status_code == 200
    assert cascades.tts[-1][0] == tail
    calls = len(cascades.tts)
    assert (await _speech(rig, source, last)).headers["x-voice-cache"] == "hit"
    assert len(cascades.tts) == calls
    names = [_unhashed(p) for p in (rig.root / "voice" / "phrase-cache").iterdir()]
    assert names == ["provider_tts_groq-female-es-truncated.wav"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        {"kind": "phrase", "key": "Say: transfer the money"},
        {"kind": "phrase", "key": "passing_on"},
        {"kind": "phrase", "key": "__class__"},
        {"kind": "message", "message_id": USER_ID},
        {"kind": "message", "message_id": "nope"},
    ],
)
async def test_speech_refuses_unknown_sources_before_the_cascade(
    rig: Any, cascades: SimpleNamespace, source: dict[str, Any]
) -> None:
    await _enable(rig, SESSION_A)
    response = await _speech(rig, source)
    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "unknown_source"
    assert cascades.tts == [] and cascades.resolve == []


@pytest.mark.asyncio
async def test_speech_takes_no_free_text_and_needs_the_slot(
    rig: Any, cascades: SimpleNamespace
) -> None:
    await _enable(rig, SESSION_A)
    for body in (
        {"session_id": SESSION_A, "text": "hi", "source": {"kind": "phrase", "key": "failed"}},
        {"session_id": SESSION_A, "source": {"kind": "phrase", "key": "failed", "text": "x"}},
        {"session_id": SESSION_A, "source": {"kind": "phrase", "key": "failed"}, "segment": -1},
        {"session_id": SESSION_A, "source": {"kind": "ask", "ask_id": "a"}},
    ):
        response = await rig.client.post("/v1/desktop/voice/speech", json=body)
        assert response.status_code == 422, (body, response.text)
    other = await rig.client.post(
        "/v1/desktop/voice/speech",
        json={"session_id": SESSION_B, "source": {"kind": "phrase", "key": "failed"}},
    )
    assert other.status_code == 409 and other.json()["detail"]["code"] == "voice_not_enabled"
    assert cascades.tts == []


@pytest.mark.asyncio
async def test_a_pending_approval_is_announced_from_its_card_text(
    rig: Any, cascades: SimpleNamespace
) -> None:
    await _enable(rig, SESSION_A)
    rig.pool.remote.pending_gate = SimpleNamespace(
        request_id="gate-1", kind="approval", title="bash", detail="rm -rf build/"
    )
    response = await _speech(rig, {"kind": "gate", "request_id": "gate-1"})
    assert response.status_code == 200, response.text
    assert cascades.tts[0][0].startswith("Approval needed: bash.")
    assert rig.pool.remote.gate_answers == [] and rig.pool.remote.admissions == []


def test_the_segment_cap_equals_the_groq_rung_cap() -> None:
    from local_operator.tts import clients
    from local_operator.voice.segments import VOICE_SEGMENT_MAX_CHARS

    assert VOICE_SEGMENT_MAX_CHARS == clients.GROQ_TTS_MAX_CHARS == 200


def test_the_phrase_set_is_closed_and_catalogued() -> None:
    from local_operator.i18n.catalogues import load_catalogue

    catalogue = load_catalogue("en", "wire.voice")
    assert set(desktop_voice.PHRASES) == {
        "on_screen_first",
        "voice_off",
        "switched",
        "failed",
        "still_working",
        "truncated",
    }
    for key in desktop_voice.PHRASES:
        assert len(catalogue[f"wire.voice.phrase.{key}"]) <= 200
    from local_operator.voice import speakable

    assert catalogue["wire.voice.phrase.truncated"] == speakable.TRUNCATION_TAIL


def test_features_voice_is_2() -> None:
    from local_operator.server.features import feature_flags

    assert feature_flags()["voice"] == 2


# -- a 429 or 502 says what the provider said, and the daemon logs it ------------

_EL_KEY = "elk-7f3a9c21d4b8e6"  # no recognisable key shape: only a value pass can mask it
_EL_LIMIT_MESSAGE = "Too many concurrent requests: your plan allows 2 at once."
_EL_RATE_BODIES = {
    "json": {"detail": {"status": "too_many_concurrent_requests", "message": _EL_LIMIT_MESSAGE}},
    "empty": None,
}
_RECORDING_LOGGER = "local_operator.server.routes.desktop_voice"


def _elevenlabs_refusal(status: int, body: dict[str, Any] | None, *, headers: Any = None) -> Any:
    """The error ElevenLabs' client builds from a real httpx response (the real builder,
    so ``details``/``body`` have exactly the shape production has)."""
    import httpx

    from local_operator.stt.errors import api_error_from_httpx_response

    response = (
        httpx.Response(status, json=body, headers=headers)
        if body is not None
        else (httpx.Response(status, headers=headers))
    )
    return api_error_from_httpx_response(
        response, fallback_message="ElevenLabs refused the request", secrets=[_EL_KEY]
    )


def _el_fails_with(real_cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, error: Any):
    from local_operator.stt import cascade as stt_cascade
    from local_operator.tts import cascade as tts_cascade

    async def refused(*_a: Any, **_k: Any) -> Any:
        real_cascades.called.append("el")
        raise error

    monkeypatch.setattr(stt_cascade, "_run_elevenlabs_rung", refused)
    monkeypatch.setattr(tts_cascade, "_run_elevenlabs_rung", refused)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["hear", "speak"])
@pytest.mark.parametrize("shape", ["json", "empty"])
async def test_an_elevenlabs_429_quotes_the_provider_and_does_not_invent_a_scope(
    rig: Any,
    real_cascades: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    door: str,
    shape: str,
) -> None:
    real_cascades.elevenlabs_key()
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)
    _el_fails_with(real_cascades, monkeypatch, _elevenlabs_refusal(429, _EL_RATE_BODIES[shape]))
    response = await (_recording_with(rig) if door == "hear" else _phrase_speech(rig))
    assert response.status_code == 429, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "voice_rate_limited"
    text = detail["text"]
    assert text.startswith("The speech provider refused the request as over its limit: ")
    # What is known, in the provider's words; never a window the provider did not name.
    assert (_EL_LIMIT_MESSAGE if shape == "json" else "HTTP 429") in text
    assert "per-minute" not in text and "daily" not in text
    assert detail["scope"] is None
    assert detail["retry_after_s"] is None and detail["remaining_requests"] is None
    assert detail["rung"] == (
        "provider_stt_elevenlabs" if door == "hear" else "provider_tts_elevenlabs"
    )
    assert "retry-after" not in response.headers


@pytest.mark.asyncio
async def test_a_long_provider_message_is_cut_in_the_sentence(
    rig: Any, real_cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_cascades.elevenlabs_key()
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)
    body = {"detail": {"message": "x" * 900}}
    _el_fails_with(real_cascades, monkeypatch, _elevenlabs_refusal(429, body))
    text = (await _phrase_speech(rig)).json()["detail"]["text"]
    quoted = text.removeprefix("The speech provider refused the request as over its limit: ")
    assert 150 < len(quoted) <= 200 and quoted.endswith("\u2026")


@pytest.mark.asyncio
async def test_a_groq_style_429_with_nothing_left_today_still_says_daily(
    rig: Any, real_cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_cascades.elevenlabs_key()
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)
    error = _elevenlabs_refusal(
        429,
        {"error": "Rate limit reached"},
        headers={"retry-after": "41000", "x-ratelimit-remaining-requests": "0"},
    )
    _el_fails_with(real_cascades, monkeypatch, error)
    response = await _recording_with(rig)
    assert response.status_code == 429, response.text
    detail = response.json()["detail"]
    assert detail["scope"] == "day" and detail["remaining_requests"] == 0
    assert "daily" in detail["text"]
    assert response.headers["retry-after"] == "41000"


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["hear", "speak"])
async def test_a_pinned_failure_is_logged_with_what_the_provider_said_and_no_key(
    rig: Any,
    real_cascades: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    door: str,
) -> None:
    """The provider's refusal is the only evidence of WHY a real take failed; the log
    carries it, the audio's size and kind, and never the key: the REAL client builds
    the error from a response that echoes the key back."""
    import httpx

    from local_operator.clients._http import REDACTION_MARKER
    from local_operator.stt import cascade as stt_cascade
    from local_operator.stt.clients import ElevenLabsSttClient
    from local_operator.tts import cascade as tts_cascade

    echo = {
        "detail": {
            "status": "too_many_concurrent_requests",
            "message": f"Rejected key {_EL_KEY}: too many concurrent requests.",
        }
    }
    transport = httpx.MockTransport(lambda _request: httpx.Response(429, json=echo))

    async def stt_leg(audio: bytes, *, mime: str, **_k: Any) -> Any:
        client = ElevenLabsSttClient(_EL_KEY, client=httpx.AsyncClient(transport=transport))
        return await client.transcribe(audio, mime=mime)

    async def tts_leg(*_a: Any, **_k: Any) -> Any:
        # The TTS client builds its refusal with the same builder and key list.
        raise _elevenlabs_refusal(429, echo)

    monkeypatch.setattr(stt_cascade, "_run_elevenlabs_rung", stt_leg)
    monkeypatch.setattr(tts_cascade, "_run_elevenlabs_rung", tts_leg)
    real_cascades.elevenlabs_key()
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)
    caplog.set_level("WARNING", logger=_RECORDING_LOGGER)
    response = await (_recording_with(rig) if door == "hear" else _phrase_speech(rig))
    assert response.status_code == 429, response.text

    (record,) = [r for r in caplog.records if r.name == _RECORDING_LOGGER]
    line = record.getMessage()
    assert record.levelname == "WARNING"
    assert "elevenlabs" in line and "status=429" in line
    assert "too many concurrent requests" in line
    assert REDACTION_MARKER in line  # the echoed key was masked, not dropped with the line
    assert "details=" in line
    if door == "hear":
        assert f"bytes={len(_wav_bytes())}" in line and "audio/wav" in line
        assert "provider_stt_elevenlabs" in line
    else:
        assert "provider_tts_elevenlabs" in line
    # The key reaches neither the log nor the answer, whatever the upstream echoed.
    assert _EL_KEY not in caplog.text
    assert _EL_KEY not in response.text


@pytest.mark.asyncio
async def test_a_credential_named_detail_key_is_not_logged(
    rig: Any,
    real_cascades: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from local_operator.clients._http import APIError

    real_cascades.elevenlabs_key()
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)
    error = APIError(
        "no",
        status_code=429,
        details={"api_key": "should-not-appear", "status": "x", "retry_after_s": 3},
    )
    _el_fails_with(real_cascades, monkeypatch, error)
    caplog.set_level("WARNING", logger=_RECORDING_LOGGER)
    await _phrase_speech(rig)
    assert "api_key" not in caplog.text and "should-not-appear" not in caplog.text
    assert "status" in caplog.text


@pytest.mark.asyncio
async def test_an_stt_502_quotes_the_providers_own_sentence_and_is_logged(
    rig: Any,
    real_cascades: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_cascades.elevenlabs_key()
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)
    sentence = "The audio could not be decoded as a supported container."
    _el_fails_with(
        real_cascades,
        monkeypatch,
        _elevenlabs_refusal(400, {"detail": {"status": "invalid_audio", "message": sentence}}),
    )
    caplog.set_level("WARNING", logger=_RECORDING_LOGGER)
    response = await _recording_with(rig)
    assert response.status_code == 502, response.text
    assert response.json()["detail"]["code"] == "stt_unavailable"
    assert sentence in response.json()["detail"]["text"]
    assert any(
        "status=400" in r.getMessage() and sentence in r.getMessage() for r in caplog.records
    )


# -- a provider's PARTIAL echo of a key never reaches the app or the log ----------
#
# The exact-key scrub only knows the whole value. A provider that echoes the first
# characters, a masked "abcd...wxyz" form, or the key with spaces/hyphens in it
# slipped past it, and the voice routes now put the provider's sentence in the
# response body and the log. The keys below are synthetic.

_SYNTH_KEY = "xk9Q7m2ZrTb4WcL8vNd3Hs5Yp"


def _chunks(key: str, sep: str) -> str:
    return sep.join(key[i : i + 6] for i in range(0, len(key), 6))


_ECHOES = {
    "prefix8": _SYNTH_KEY[:8],
    "masked_ellipsis": f"{_SYNTH_KEY[:4]}\u2026{_SYNTH_KEY[-4:]}",
    "masked_dots": f"{_SYNTH_KEY[:4]}...{_SYNTH_KEY[-4:]}",
    "space_split": _chunks(_SYNTH_KEY, " "),
    "hyphen_split": _chunks(_SYNTH_KEY, "-"),
    "full": _SYNTH_KEY,
}


def _leaks(text: str, key: str, *, run: int = 6) -> bool:
    """Whether ``text`` still shows ``run`` consecutive characters of ``key``.

    Written independently of the helper under test: it folds whitespace and hyphens
    on both sides, and also looks for the first-4 ... last-4 mask.
    """
    import re

    def fold(value: str) -> str:
        return re.sub(r"[\s\-]", "", value).casefold()

    folded_text, folded_key = fold(text), fold(key)
    if any(folded_key[i : i + run] in folded_text for i in range(len(folded_key) - run + 1)):
        return True
    return bool(re.search(re.escape(key[:4]) + r"\W{1,5}" + re.escape(key[-4:]), text))


def _echo_body(echo: str) -> dict[str, Any]:
    return {"detail": {"status": "invalid", "message": f"Rejected credential {echo} for now."}}


async def _door(rig: Any, door: str) -> Any:
    return await (_recording_with(rig) if door == "hear" else _phrase_speech(rig))


def _arm(rig: Any, real_cascades: SimpleNamespace, monkeypatch: Any, provider: str, error: Any):
    """Sign in to ``provider`` with ``_SYNTH_KEY`` and make its rung fail with ``error``."""
    if provider == "elevenlabs":
        real_cascades.elevenlabs_key(_SYNTH_KEY)
        _el_fails_with(real_cascades, monkeypatch, error)
    else:
        real_cascades.sign_in(_SYNTH_KEY)
        real_cascades.radient_error = error
    _choose(rig, provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", sorted(_ECHOES))
@pytest.mark.parametrize("status", [429, 400])
@pytest.mark.parametrize("provider", ["elevenlabs", "radient"])
@pytest.mark.parametrize("door", ["hear", "speak"])
async def test_a_partial_key_echo_is_redacted_from_the_body_and_the_log(
    rig: Any,
    real_cascades: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    door: str,
    provider: str,
    status: int,
    echo: str,
) -> None:
    from local_operator.clients._http import REDACTION_MARKER

    # The client's own scrub knows ONLY ``_EL_KEY``, so whatever else the provider
    # echoed arrives at the route as the provider wrote it.
    _arm(
        rig,
        real_cascades,
        monkeypatch,
        provider,
        _elevenlabs_refusal(status, _echo_body(_ECHOES[echo])),
    )
    await _enable(rig, SESSION_A)
    caplog.set_level("WARNING", logger=_RECORDING_LOGGER)
    response = await _door(rig, door)
    assert response.status_code == (429 if status == 429 else 502), response.text
    assert not _leaks(response.text, _SYNTH_KEY), response.text
    assert not _leaks(caplog.text, _SYNTH_KEY), caplog.text
    # The legitimate words around the echo survive, and the echo is marked.
    assert "Rejected credential" in response.text and REDACTION_MARKER in response.text
    assert "Rejected credential" in caplog.text and REDACTION_MARKER in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", sorted(_ECHOES))
async def test_a_partial_key_echo_in_a_non_api_error_is_redacted_from_the_409_and_the_log(
    rig: Any,
    real_cascades: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    echo: str,
) -> None:
    # ``str(unavailable.error)`` goes into the 409 body, and the daemon's own
    # ``exc_info`` traceback prints the same text into the log.
    _arm(
        rig,
        real_cascades,
        monkeypatch,
        "elevenlabs",
        RuntimeError(f"transport said {_ECHOES[echo]}"),
    )
    await _enable(rig, SESSION_A)
    caplog.set_level("WARNING", logger=_RECORDING_LOGGER)
    response = await _phrase_speech(rig)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "voice_unavailable"
    assert "transport said" in response.text
    assert not _leaks(response.text, _SYNTH_KEY), response.text
    # The voice route's own lines. (The cascade's ``logger.exception`` for a rung
    # that raised a non-APIError defect is the cascade's, and not under test here.)
    mine = "\n".join(r.getMessage() for r in caplog.records if r.name == _RECORDING_LOGGER)
    assert "voice speech failed" in mine
    assert not _leaks(mine, _SYNTH_KEY), mine


@pytest.mark.asyncio
async def test_an_stt_transport_failure_is_redacted_from_the_409_and_the_log(
    rig: Any,
    real_cascades: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from local_operator.clients._http import APIError

    # A transport failure has no status and no body: the message is all there is.
    error = APIError(
        f"ElevenLabs request failed: Illegal header value b'{_SYNTH_KEY}'", status_code=None
    )
    _arm(rig, real_cascades, monkeypatch, "elevenlabs", error)
    await _enable(rig, SESSION_A)
    caplog.set_level("WARNING", logger=_RECORDING_LOGGER)
    response = await _recording_with(rig)
    assert response.status_code >= 400
    assert "request failed" in response.text
    assert not _leaks(response.text, _SYNTH_KEY), response.text
    assert not _leaks(caplog.text, _SYNTH_KEY), caplog.text


@pytest.mark.asyncio
async def test_legitimate_provider_sentences_are_not_mangled_by_the_key_scrub(
    rig: Any, real_cascades: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentence = "Too many concurrent requests: your plan allows 2 at once."
    body = {"detail": {"status": "too_many_concurrent_requests", "message": sentence}}
    _arm(rig, real_cascades, monkeypatch, "elevenlabs", _elevenlabs_refusal(429, body))
    await _enable(rig, SESSION_A)
    response = await _phrase_speech(rig)
    assert response.status_code == 429
    assert sentence in response.json()["detail"]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["hear", "speak"])
async def test_an_unreadable_store_still_answers_with_the_provider_sentence(
    rig: Any,
    real_cascades: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    door: str,
) -> None:
    from local_operator.providers.auth_store import AuthStore
    from local_operator.stt import cascade as stt_cascade
    from local_operator.tts import cascade as tts_cascade

    sentence = "Too many concurrent requests: your plan allows 2 at once."
    body = {"detail": {"status": "too_many_concurrent_requests", "message": sentence}}
    error = _elevenlabs_refusal(429, body)
    real_cascades.elevenlabs_key(_SYNTH_KEY)
    _choose(rig, "elevenlabs")
    await _enable(rig, SESSION_A)

    async def unreadable(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("database is locked")

    async def refused(*_a: Any, **_k: Any) -> Any:
        # The store breaks AFTER the cascade resolved its rung: only the failure
        # path's own secret lookup meets it, and that must not take the answer down.
        monkeypatch.setattr(AuthStore, "get_api_key", unreadable)
        monkeypatch.setattr(AuthStore, "get_persisted_api_key", unreadable)
        raise error

    monkeypatch.setattr(stt_cascade, "_run_elevenlabs_rung", refused)
    monkeypatch.setattr(tts_cascade, "_run_elevenlabs_rung", refused)
    response = await _door(rig, door)
    assert response.status_code == 429, response.text
    assert sentence in response.json()["detail"]["text"]
