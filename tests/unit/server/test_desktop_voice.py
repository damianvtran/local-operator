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
        yield SimpleNamespace(client=client, pool=pool, root=tmp_path)


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
async def test_status_reports_the_slot_and_cap_and_nothing_about_speech(rig: Any) -> None:
    result = (await rig.client.get("/v1/desktop/voice")).json()["result"]
    assert result == {"active": None, "settings": {"max_spoken_chars": 800}}


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
    assert set(result) == {"text", "spoken_chars", "truncated", "segments"}
    assert "".join(result["segments"]) == result["text"]
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
        tts=[], stt=[], resolve=[], spend=[], tts_error=None, stt_error=None, text="list files"
    )

    async def synthesize(text: str, **kwargs: Any) -> TtsOutcome:
        calls.tts.append((text, kwargs))
        if calls.tts_error is not None:
            raise calls.tts_error
        return TtsOutcome(
            audio=_wav_bytes(), path=VoicePath.PROVIDER_TTS_GROQ, attempts=(), provider="groq"
        )

    async def resolve(**kwargs: Any) -> VoicePathResolution:
        calls.resolve.append(kwargs)
        rung = RungAvailability(VoicePath.PROVIDER_TTS_GROQ, True, "A Groq API key is stored.")
        return VoicePathResolution(
            path=VoicePath.PROVIDER_TTS_GROQ, reason="ok", rungs=(rung,), servable=True
        )

    async def transcribe(path: Path, **kwargs: Any) -> SttOutcome:
        calls.stt.append((path.read_bytes()[:4], kwargs))
        if calls.stt_error is not None:
            raise calls.stt_error
        return SttOutcome(text=calls.text, path=AudioPath.PROVIDER_STT_GROQ, attempts=())

    monkeypatch.setattr("local_operator.tts.cascade.synthesize_speech", synthesize)
    monkeypatch.setattr("local_operator.tts.cascade.resolve_voice_path", resolve)
    monkeypatch.setattr("local_operator.stt.cascade.transcribe_audio", transcribe)
    monkeypatch.setattr(desktop_voice, "emit_voice_spend", lambda **kw: calls.spend.append(kw))
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
        ({}, "minute", None, None),
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


@pytest.mark.asyncio
async def test_speech_returns_one_segment_as_wav_with_the_cascade_flags(
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
    assert kwargs["response_format"] == "wav"
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
    assert [path.name for path in cached] == ["provider_tts_groq-female-on_screen_first.wav"]

    second = await _speech(rig, source)
    assert second.status_code == 200
    assert second.content == first.content and second.headers["x-voice-cache"] == "hit"
    assert second.headers["content-type"] == "audio/wav"
    assert len(cascades.tts) == 1, "a cache hit reached the cascade"
    assert len(cascades.spend) == 1, "a cache hit was billed"


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
