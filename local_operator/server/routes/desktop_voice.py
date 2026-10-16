"""Per-session voice mode: the desktop surface (docs/design/session-voice.md).

The frozen contract, which both repositories build against:

    GET  /v1/desktop/voice
        200 {active: {session_id, enabled_at} | null,
             stt: {path, available, reason}, tts: {path, available, reason},
             settings: {max_spoken_chars}}
    POST /v1/desktop/voice             body {op: enable|disable, session_id}
        200 {active, replaced}; 404 unknown_session;
        409 voice_unavailable | voice_store_busy
    POST /v1/desktop/voice/utterance   multipart session_id, request_id (dashed UUID),
                                       file | text
        200 {text, path, admission}; 422 invalid_voice_request | empty_transcript;
        409 voice_not_enabled | approval_pending | stt_unavailable
    POST /v1/desktop/voice/speech      body {session_id, source}
        200 audio, X-Voice-Path, X-Voice-Spoken-Chars;
        409 voice_not_enabled; 404 unknown_source

and the ``type: "voice"`` frame on ``GET /v1/desktop/events`` (published by
``server/utils/desktop_feed.py``), ``{active}``, on every slot change.

Refusals use the i18n envelope with a machine ``code``, the shape
``desktop_aida.py`` uses: a client classifies by ``detail.code`` and shows
``detail.text``.

ONE SESSION AT A TIME (V3). Every route but the status read refuses a session
that is not in the slot, so a stale renderer cannot keep talking to the
conversation another window switched away from. A slot whose conversation was
deleted reads as EMPTY: the delete route clears it, and a read that finds the
directory gone (a delete from another surface) treats it as cleared.

THE UTTERANCE IS AN ORDINARY PROMPT (V5). It is admitted through the same door
and the same owner op as ``POST /v1/desktop/sessions/{id}/messages``: the pool's
``session()``, the receipt journal under the same ``session_id:request_id``
key family, and ``bridge.remote.admit_prompt``. It is marked
``input_mode="dictated"``, with ``input_path`` naming the STT rung when audio was
transcribed. The steer-or-prompt choice is ``admit_receipt_request``'s own rule
(``desktop_sessions.py``): steer while the session is streaming, else prompt.

V7, A HARD RULE: THIS MODULE NEVER ANSWERS A GATE. It has no import of, and no
call to, ``answer_gate`` or the ``/answers`` handler. The operator approves on
screen through the existing presence-gated card.

V13: NO UTTERANCE WHILE AN APPROVAL IS PENDING. A steer is not an answer, but it
is not harmless either: an INTERRUPTIBLE tool's card (``bash``) is cancelled by
a steer ("Tool call skipped: interrupted by steering."), so a word heard at the
card would make the card vanish. The utterance route therefore refuses with
409 ``approval_pending`` while the slot session has a pending approval gate,
for the text door and for audio alike, and audio is refused BEFORE STT runs so
nothing is spent. ``tests/unit/server/test_desktop_voice.py`` pins the static
absence; ``tests/unit/voice/test_utterance_during_approval.py`` pins the refusal
against a real runtime with the card still parked.

THE SERVER DECIDES WHAT IS SPOKEN (V8). ``/speech`` takes a source reference,
never text: a message id resolved from the session's own durable journal, or the
pending approval card as the server resolves it (its title and description).
That description is the tool's own approval summary, which can quote
MODEL-WRITTEN arguments (a bash command, a path), so it is not host-authored
prose; it is the same text the card shows on screen. Every source goes through
``voice/speakable.py`` (no code blocks, no URLs, a cap) before synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import TypeAdapter, ValidationError

from local_operator.clients.radient_cost import RadientCost, cost_from_headers
from local_operator.config import ConfigManager
from local_operator.env import EnvConfig
from local_operator.i18n.keys import wire_voice as _wv
from local_operator.providers.auth_store import AuthStore
from local_operator.server.dependencies import (
    get_config_manager,
    get_env_config,
    get_provider_auth_store,
)
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_sessions import MessageAdmission
from local_operator.server.models.desktop_voice import (
    VoiceActive,
    VoiceControl,
    VoiceControlResult,
    VoicePathState,
    VoiceSettings,
    VoiceSpeechRequest,
    VoiceStatus,
    VoiceUtteranceResult,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import (
    Prompt,
    RequestID,
    errors,
    host,
)
from local_operator.server.routes.desktop_sessions import prompt as messages_route
from local_operator.server.routes.desktop_sessions import reply
from local_operator.server.utils.voice_spend import emit_voice_spend
from local_operator.stt import AudioPath
from local_operator.stt import audio as stt_audio
from local_operator.voice import slot as voice_slot
from local_operator.voice import speakable as voice_speakable

logger = logging.getLogger("local_operator.server.routes.desktop_voice")

router = APIRouter(tags=["Voice"], dependencies=[Depends(require_desktop)])

#: The contract's upload ceiling. Larger than the ``/messages`` audio block's
#: (that one rides a JSON body) because this is a multipart upload that never
#: reaches the owner: only the transcript does.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

#: The containers the contract accepts, by sniffed bytes rather than by the
#: client's declared type (the client's label is a hint, never trusted).
ACCEPTED_AUDIO = frozenset({"audio/wav", "audio/webm", "audio/ogg", "audio/mp4"})

#: The STT rungs' vendor labels for the spend record, the ``stt.py`` route's own
#: table; a rung absent from it reports the envelope's provider or nothing.
_STT_PROVIDERS = {
    AudioPath.PROVIDER_STT_ELEVENLABS.value: "elevenlabs",
    AudioPath.PROVIDER_STT_OPENAI.value: "openai",
}

_REQUEST_ID = TypeAdapter(RequestID)


def _refuse(status: int, code: str, message: Any) -> HTTPException:
    """A declared refusal: the i18n envelope with the contract's machine code."""
    return HTTPException(status, message.envelope(code=code))


def _config_dir(request: Request) -> Path:
    return Path(request.app.state.config_manager.config_dir)


def _max_spoken_chars(config_manager: ConfigManager) -> int:
    """The cap as it is on DISK now, not as this process loaded it.

    The section is LIVE, and the write usually happens in another process (the
    TUI's settings page, ``lop config edit``), which never refreshes this
    server's in-memory copy. A fresh read is one small YAML parse per spoken
    reply, the same trade ``GET /v1/settings`` makes. An unreadable file reads
    as the default: voice never fails on this setting.
    """
    from local_operator import settings_io

    try:
        value = settings_io.read_setting(
            ConfigManager(config_manager.config_dir), settings_io.BY_KEY["voice.max_spoken_chars"]
        )
    except Exception:  # noqa: BLE001 - a hand-broken config.yml must not break voice
        value = voice_speakable.DEFAULT_MAX_SPOKEN_CHARS
    return voice_speakable.normalise_max_chars(value)


def _live_slot(config_dir: Path) -> voice_slot.VoiceSlot | None:
    """The slot, or ``None`` when it names a conversation that no longer exists.

    The delete route clears the slot, but a conversation can also be removed by
    another surface (the TUI, ``lop``) that knows nothing about voice. A slot
    left naming it must not keep answering ``active``, so the read treats a
    missing ``sessions/<id>`` as an empty slot. The stale file stays until the
    next enable or disable rewrites it: a read never writes.
    """
    current = voice_slot.read_slot(config_dir)
    if current is None or not _session_exists(config_dir, current.session_id):
        return None
    return current


def _active(config_dir: Path) -> VoiceActive | None:
    current = _live_slot(config_dir)
    return VoiceActive(**current.public()) if current is not None else None


def require_slot(config_dir: Path, session_id: str) -> None:
    """Refuse every route call for a session that is not the voice-enabled one."""
    current = _live_slot(config_dir)
    if current is None or current.session_id != session_id:
        raise _refuse(409, "voice_not_enabled", _wv.not_enabled())


def _session_exists(config_dir: Path, session_id: str) -> bool:
    """Whether ``session_id`` names a LOCAL conversation directory.

    Voice is device-local in v1: the slot, the microphone and the speaker are
    on this machine, so a peer's conversation is refused as unknown rather than
    enabled and then failing on its first utterance. The id rule is the
    catalogue's own (``session_directory_name``), so an id cannot step outside
    ``sessions/``.
    """
    from local_operator.session.catalog import session_directory_name

    if not session_directory_name(session_id):
        return False
    return (config_dir / "sessions" / session_id).is_dir()


async def _resolve_tts(config_manager: ConfigManager, env_config: EnvConfig, store: AuthStore):
    from local_operator.tts.cascade import resolve_voice_path

    return await resolve_voice_path(
        config_dir=config_manager.config_dir,
        base_url=env_config.radient_api_base_url,
        store=store,
        include_local=True,
    )


async def _resolve_stt(config_manager: ConfigManager, env_config: EnvConfig, store: AuthStore):
    from local_operator.stt.cascade import resolve_audio_path

    return await resolve_audio_path(
        config_dir=config_manager.config_dir,
        base_url=env_config.radient_api_base_url,
        store=store,
    )


# -- GET /v1/desktop/voice -----------------------------------------------------


@router.get("/v1/desktop/voice", response_model=CRUDResponse[VoiceStatus])
async def voice_status(
    request: Request,
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
    env_config: Annotated[EnvConfig, Depends(get_env_config)],
    store: Annotated[AuthStore, Depends(get_provider_auth_store)],
) -> CRUDResponse[VoiceStatus]:
    """The slot, both paths and the cap. Never writes anything.

    ``tts.path`` is the rung the voice routes will actually use, including the
    opt-in ``local_tts_system``. ``stt.path`` is ``none`` with
    ``available: false`` on a machine with no STT rung, and the text door still
    works there.
    """
    tts = await _resolve_tts(config_manager, env_config, store)
    stt = await _resolve_stt(config_manager, env_config, store)
    return reply(
        VoiceStatus(
            active=_active(_config_dir(request)),
            stt=VoicePathState(
                path=stt.path.value, available=stt.path != AudioPath.NONE, reason=stt.reason
            ),
            tts=VoicePathState(path=tts.path.value, available=tts.servable, reason=tts.reason),
            settings=VoiceSettings(max_spoken_chars=_max_spoken_chars(config_manager)),
        ),
    )


# -- POST /v1/desktop/voice ----------------------------------------------------


@router.post("/v1/desktop/voice", response_model=CRUDResponse[VoiceControlResult])
async def voice_control(
    body: VoiceControl,
    request: Request,
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
    env_config: Annotated[EnvConfig, Depends(get_env_config)],
    store: Annotated[AuthStore, Depends(get_provider_auth_store)],
) -> CRUDResponse[VoiceControlResult]:
    """Enable (switching the slot to this session) or disable voice.

    ``enable`` refuses when no TTS rung exists at all (``voice_unavailable``):
    a voice mode that can never speak is a promise the UI must not make. STT
    absence is allowed, because the text door still works.

    ``disable`` of a session that does not hold the slot is a no-op that answers
    the slot as it is. A stale window cannot switch off the conversation another
    window just enabled.
    """
    from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable

    config_dir = _config_dir(request)
    if body.op == "enable":
        if not _session_exists(config_dir, body.session_id):
            raise _refuse(404, "unknown_session", _wv.unknown_session(session_id=body.session_id))
        tts = await _resolve_tts(config_manager, env_config, store)
        if not tts.servable:
            raise _refuse(409, "voice_unavailable", _wv.unavailable(reason=tts.reason))
    try:
        if body.op == "enable":
            change = await asyncio.to_thread(
                voice_slot.enable,
                config_dir,
                body.session_id,
                instance_id=str(getattr(request.app.state, "instance_id", "") or ""),
            )
        else:
            change = await asyncio.to_thread(voice_slot.disable, config_dir, body.session_id)
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        logger.info("voice slot write refused: %s", exc)
        raise _refuse(409, "voice_store_busy", _wv.store_busy()) from None
    # A deleted conversation is not a slot holder (B3): neither the slot it left
    # behind nor a "replaced" naming it is news the renderer can act on.
    active = change.active
    if active is not None and not _session_exists(config_dir, active.session_id):
        active = None
    replaced = change.replaced
    if replaced is not None and not _session_exists(config_dir, replaced):
        replaced = None
    return reply(
        VoiceControlResult(
            active=VoiceActive(**active.public()) if active is not None else None,
            replaced=replaced,
        ),
    )


# -- POST /v1/desktop/voice/utterance -------------------------------------------


async def _save_upload(file: UploadFile) -> tuple[str, Path]:
    """Copy the upload to a private temp dir, bounded at ``MAX_UPLOAD_BYTES``.

    Returns ``(sniffed mime, path)``. The temp dir is the caller's to remove.
    The client's filename and content type never steer the path or the format.
    """
    temp_dir = tempfile.mkdtemp(prefix="lop-voice-")
    try:
        head = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(head) > MAX_UPLOAD_BYTES:
            raise _refuse(
                413,
                "voice_upload_too_large",
                _wv.upload_too_large(limit_mib=str(MAX_UPLOAD_BYTES // (1024 * 1024))),
            )
        info = stt_audio.sniff_audio(head)
        if info is None or info.mime_type not in ACCEPTED_AUDIO:
            raise _refuse(422, "invalid_voice_request", _wv.unsupported_audio())
        path = Path(temp_dir) / f"utterance{stt_audio.extension_for_mime(info.mime_type)}"
        path.write_bytes(head)
        return info.mime_type, path
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        await file.close()


async def _transcribe(
    file: UploadFile,
    *,
    session_id: str,
    request_id: str,
    config_manager: ConfigManager,
    env_config: EnvConfig,
    store: AuthStore,
) -> tuple[str, str]:
    """Run the STT cascade over the upload; ``(text, path token)``.

    Refuses with ``stt_unavailable`` (409) when no rung exists, carrying the
    cascade's reason, and with the cascade's own classified status when every
    rung failed. Spend goes through ``emit_voice_spend`` with the session id.
    """
    from local_operator.stt.cascade import SttUnavailable, transcribe_audio

    _mime, path = await _save_upload(file)
    try:
        try:
            outcome = await transcribe_audio(
                path,
                config_dir=config_manager.config_dir,
                base_url=env_config.radient_api_base_url,
                store=store,
                correlation_session_id=session_id,
            )
        except SttUnavailable as exc:
            if not exc.attempts:
                raise _refuse(
                    409, "stt_unavailable", _wv.stt_unavailable(reason=exc.resolution.reason)
                ) from None
            raise _refuse(
                exc.status_code or 502,
                "stt_unavailable",
                _wv.stt_unavailable(reason=exc.detail or str(exc)),
            ) from None
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)
    cost = outcome.cost if isinstance(outcome.cost, RadientCost) else None
    emit_voice_spend(
        config_manager=config_manager,
        channel="stt",
        provider=outcome.provider or _STT_PROVIDERS.get(outcome.path.value, ""),
        units=None,
        unit="seconds",
        cost=cost,
        session_id=session_id,
        request_id=request_id,
    )
    return outcome.text, outcome.path.value


@router.post("/v1/desktop/voice/utterance", response_model=CRUDResponse[VoiceUtteranceResult])
async def voice_utterance(
    request: Request,
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
    env_config: Annotated[EnvConfig, Depends(get_env_config)],
    store: Annotated[AuthStore, Depends(get_provider_auth_store)],
    session_id: str = Form(..., min_length=1, max_length=128),
    request_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None, max_length=200_000),
) -> CRUDResponse[VoiceUtteranceResult]:
    """Transcribe (or take ``text``) and admit it to the slot's session as a prompt.

    The text door has the same authority as ``/messages``: the slot check, the
    pool's door and the receipt journal all apply. Only the STT step is skipped.
    """
    try:
        _REQUEST_ID.validate_python(request_id)
    except ValidationError:
        raise _refuse(422, "invalid_voice_request", _wv.invalid_request_id()) from None
    if (file is None) == (text is None):
        raise _refuse(422, "invalid_voice_request", _wv.invalid_request())
    require_slot(_config_dir(request), session_id)
    # V13, before STT: a refused recording must cost nothing.
    if await _approval_pending(request, session_id):
        raise _refuse(409, "approval_pending", _wv.approval_pending())

    if file is not None:
        spoken, path = await _transcribe(
            file,
            session_id=session_id,
            request_id=request_id,
            config_manager=config_manager,
            env_config=env_config,
            store=store,
        )
        # STT takes seconds, and a card can park inside them. The money is spent
        # by now; the card is not, so check again before the steer can cancel it.
        if await _approval_pending(request, session_id):
            raise _refuse(409, "approval_pending", _wv.approval_pending())
    else:
        spoken, path = text or "", None
    spoken = spoken.strip()
    if not spoken:
        raise _refuse(422, "empty_transcript", _wv.empty_transcript())

    admission = await _admit(request, session_id, request_id, spoken, path)
    return reply(
        VoiceUtteranceResult(
            text=spoken, path=path, admission=MessageAdmission.model_validate(admission)
        ),
    )


async def _approval_pending(request: Request, session_id: str) -> bool:
    """V13: does the session have an approval card parked right now?

    Read through the CONTROL door for ``_streaming``'s reason: a cold read-door
    answer would report a parked card as absent. A facade whose frontend state
    has not synchronized yet has no card to show either, so that reads as
    ``False``; the owner's own steer handling is the backstop for that window.
    """
    async with errors(request):
        async with host(request).session(session_id) as bridge:
            remote = bridge.remote
            try:
                gate = remote.pending_gate if remote is not None else None
            except RuntimeError:
                gate = None
    return gate is not None and getattr(gate, "kind", "") == "approval"


def _validation_reason(exc: ValidationError) -> str:
    """The first validator sentence, without pydantic's ``Value error,`` prefix."""
    errors_ = exc.errors()
    message = str(errors_[0].get("msg", "")) if errors_ else str(exc)
    return message.removeprefix("Value error, ").strip()


async def _streaming(request: Request, session_id: str) -> bool:
    """``admit_receipt_request``'s steer rule: is the session's turn streaming now?

    Read through the CONTROL door (not ``read=True``): ``/messages`` binds the
    runtime anyway, and a read-door answer that came back cold would report a
    busy owner as idle.
    """
    async with errors(request):
        async with host(request).session(session_id) as bridge:
            return bool(getattr(bridge.remote, "is_streaming", False))


async def _admit(
    request: Request, session_id: str, request_id: str, text: str, path: str | None
) -> dict[str, Any]:
    """Admit ``text`` THROUGH the ``/messages`` handler itself (V5).

    Not a copy of it: the handler is called with a ``Prompt`` body, so the
    viewer acknowledgement, the pool's door, the receipt journal (same
    ``session_id:request_id`` key, so a request id reused across the two routes
    with different bodies is refused as on ``/messages``), the slash-command
    refusal and the owner op are all the one implementation.

    ``mode`` is the only input this route decides, and it decides it with
    ``admit_receipt_request``'s rule (steer while streaming), because
    ``/messages`` itself takes the mode from its client.

    The ``Prompt`` is built BEFORE the streaming read, so a body ``/messages``
    would refuse (a transcript that IS a slash command, an oversize frame) is a
    declared 422 ``invalid_voice_request`` naming the validator's own reason,
    and is refused before any owner op (the generic "invalid fields" 422 the
    app's shaper would otherwise publish named nothing the speaker could fix).

    V7: the owner op this reaches is ``prompt``/``steer``. Neither settles a
    gate: gate futures are resolved only by the owner's answer ops, which the
    ``/answers`` route reaches and this module does not.
    """
    try:
        body = Prompt(
            request_id=request_id,
            text=text,
            mode="prompt",
            input_mode="dictated",
            input_path=path,
        )
    except ValidationError as exc:
        raise _refuse(
            422, "invalid_voice_request", _wv.invalid_message(reason=_validation_reason(exc))
        ) from None
    if await _streaming(request, session_id):
        body = body.model_copy(update={"mode": "steer"})
    admitted = await messages_route(session_id, body, request)
    return dict(admitted.result or {})


# -- POST /v1/desktop/voice/speech -----------------------------------------------


async def _message_text(request: Request, session_id: str, message_id: str) -> str | None:
    """The assistant message ``message_id``'s text, from the durable journal.

    The anchored history read (``around_id``, design §D4) with no context rows,
    so the cost is the locator plus one row, through the same bridge reader
    ``/history`` uses (``desktop_sessions.py`` ``history``). Only an ASSISTANT
    message is speakable: a user row, a tool row or a custom row reads as an
    unknown source, so the route cannot be made to read anything aloud except
    words the agent wrote.
    """
    async with errors(request):
        async with host(request).session(session_id, read=True) as bridge:
            page = await bridge.history(around_id=message_id, before=0, after=0, limit=1)
    for entry in page.get("entries") or []:
        if entry.get("id") != message_id or entry.get("type") != "message":
            continue
        payload = entry.get("payload") or {}
        if payload.get("role") != "assistant":
            return None
        blocks = payload.get("content") or []
        return "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type", "text") == "text"
        )
    return None


async def _gate_text(request: Request, session_id: str, gate_request_id: str) -> str | None:
    """The pending APPROVAL card's text as the server resolves it, or ``None``.

    Read-only: the pending gate is read off the facade, never answered. The
    title is the tool name; the detail is the tool's approval summary, which can
    quote model-written arguments (the command, the path). That is the same
    text the card shows on screen, so speaking it reveals nothing new, but it is
    not host-authored prose. Only an approval is spoken in slice 1; an ask is
    slice 2.
    """
    async with errors(request):
        async with host(request).session(session_id, read=True) as bridge:
            remote = bridge.remote
            try:
                gate = remote.pending_gate if remote is not None else None
            except RuntimeError:
                gate = None
    if gate is None or getattr(gate, "request_id", None) != gate_request_id:
        return None
    if getattr(gate, "kind", "") != "approval":
        return None
    title = str(getattr(gate, "title", "") or "").strip()
    detail = str(getattr(gate, "detail", "") or "").strip()
    sentence = f"Approval needed: {title}." if title else "Approval needed."
    return f"{sentence} {detail}".strip() if detail else sentence


def _audio_mime(audio: bytes) -> str:
    info = stt_audio.sniff_audio(audio)
    return info.mime_type if info is not None else "application/octet-stream"


@router.post("/v1/desktop/voice/speech")
async def voice_speech(
    body: VoiceSpeechRequest,
    request: Request,
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
    env_config: Annotated[EnvConfig, Depends(get_env_config)],
    store: Annotated[AuthStore, Depends(get_provider_auth_store)],
) -> Response:
    """Synthesise one SOURCE the server resolves itself; never free text (V8).

    200 carries the audio with ``X-Voice-Path`` (the TTS rung that served) and
    ``X-Voice-Spoken-Chars`` (the characters actually synthesised, after the
    cap). The Content-Type is sniffed from the bytes, because ``say`` answers WAV
    and the provider rungs answer mp3.
    """
    from local_operator.server.routes.speech import _speech_refusal
    from local_operator.tts.adapters import Legacy
    from local_operator.tts.cascade import TtsUnavailable, synthesize_speech
    from local_operator.tts.descriptor import descriptor_from_config

    require_slot(_config_dir(request), body.session_id)
    source = body.source
    if source.kind == "message":
        raw = await _message_text(request, body.session_id, source.message_id)
    elif source.kind == "gate":
        raw = await _gate_text(request, body.session_id, source.request_id)
    else:
        # Asks are spoken in slice 2 (design "Order"); until then the reference
        # resolves to nothing rather than to an unvetted question text.
        raw = None
    if raw is None:
        raise _refuse(404, "unknown_source", _wv.unknown_source())
    spoken = voice_speakable.speakable(raw, max_chars=_max_spoken_chars(config_manager))
    if not spoken.text:
        raise _refuse(404, "unknown_source", _wv.unknown_source())

    # The uuid suffix keeps two replies spoken inside one millisecond (a gate
    # announcement and a reply) from sharing a correlation id on the spend row.
    request_id = f"voice-{body.session_id}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}"
    try:
        outcome = await synthesize_speech(
            spoken.text,
            config_dir=config_manager.config_dir,
            base_url=env_config.radient_api_base_url,
            store=store,
            descriptor=descriptor_from_config(config_manager),
            legacy=Legacy(),
            response_format="mp3",
            correlation_session_id=body.session_id,
            client_request_id=request_id,
            include_local=True,
        )
    except TtsUnavailable as unavailable:
        if unavailable.error is None:
            raise _refuse(
                409, "voice_unavailable", _wv.unavailable(reason=unavailable.resolution.reason)
            ) from None
        from local_operator.clients._http import APIError

        if isinstance(unavailable.error, APIError):
            raise _speech_refusal(unavailable.error, rung=unavailable.failed_path) from None
        logger.error("voice speech failed", exc_info=unavailable.error)
        raise _refuse(
            409, "voice_unavailable", _wv.unavailable(reason=str(unavailable.error))
        ) from None

    emit_voice_spend(
        config_manager=config_manager,
        channel="tts",
        provider=outcome.provider,
        units=float(len(spoken.text)),
        unit="chars",
        cost=cost_from_headers(outcome.cost_headers),
        session_id=body.session_id,
        request_id=request_id,
    )
    return Response(
        content=outcome.audio,
        media_type=_audio_mime(outcome.audio),
        headers={
            "X-Voice-Path": outcome.path.value,
            "X-Voice-Spoken-Chars": str(len(spoken.text)),
            "Cache-Control": "no-store",
        },
    )
