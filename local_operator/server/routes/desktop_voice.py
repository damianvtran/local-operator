"""Per-session voice mode: the desktop surface (docs/design/session-voice.md).

The frozen contract, which both repositories build against:

    GET  /v1/desktop/voice
        200 {active: {session_id, enabled_at} | null,
             stt: {path, available, reason}, tts: {path, available, reason},
             front_ends: [{id: local|gemini_live, available, reason}],
             settings: {max_spoken_chars, front_end}}
    POST /v1/desktop/voice             body {op: enable|disable, session_id, front_end?}
        200 {active, replaced}; 404 unknown_session;
        409 voice_unavailable | voice_store_busy | front_end_unavailable
    POST /v1/desktop/voice/utterance   multipart session_id, request_id (dashed UUID),
                                       file | text, text_origin?, controls?
        200 {text, path, admission | null, control}; 422 invalid_voice_request |
        empty_transcript; 409 voice_not_enabled | approval_pending | stt_unavailable
    POST /v1/desktop/voice/speech      body {session_id, source}
        source {kind: message|gate|ask|phrase}
        200 audio, X-Voice-Path, X-Voice-Spoken-Chars;
        409 voice_not_enabled; 404 unknown_source
    POST /v1/desktop/voice/live-token  (Electron main only) no body
        200 {token, expires_at, new_session_expires_at, model};
        409 live_unavailable; 403 live_main_only; 502 live_failed

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

V13: NO UTTERANCE WHILE A CARD IS PENDING. A steer is not an answer, but it is
not harmless either: an INTERRUPTIBLE tool's card (``bash``'s approval, and a
BLOCKING ask under ``LOP_ASK_NONBLOCKING=0``) is cancelled by a steer ("Tool call
skipped: interrupted by steering."), so a word heard at the card would make it
vanish. The utterance route therefore refuses with 409 ``approval_pending``
while the slot session has ANY pending card, for the text door and for audio
alike, and audio is refused BEFORE STT runs so nothing is spent. A queued ask
(the default) parks no card, so it never refuses.

KNOWN RACE, ACCEPTED: the check reads the facade, and the steer is a separate
owner op. A card that parks between the two (milliseconds; reproduced at 1-4 ms)
is not seen, and the steer cancels it, exactly as a typed steer would today.
Nothing is approved and the tool does not run. Closing it needs an owner-side
"refuse if a card is pending" flag carried through six layers plus a capability
flag (design V13), so it is deferred; the race is pinned, not hidden, by
``tests/unit/voice/test_utterance_during_approval.py``.
``tests/unit/server/test_desktop_voice.py`` pins the static absence of any
answer path.

THE RELAY FRONT END ADDS NO NEW DOOR (R1-R8). A front end (``local``, or
``gemini_live`` in Electron main) only HEARS a take and decides relay or
control. A relay's words still enter through ``/utterance`` and the ``_admit``
door above; ``text_origin`` names where a front end's words came from and is
recorded as ``input_path`` (R3). A control (``controls=1``, the
``voice/controls.py`` grammar, R6) admits nothing and writes no receipt. The
Live token's tool list has no gate or answer tool (``voice/live_config.py``,
R7), so the relay keeps V7 by construction rather than by a check here.

R5: ``/speech`` also speaks ``{kind: "phrase", key}``, a key from the closed
``wire.voice.phrase.*`` catalogue. The server renders the catalogue sentence;
an unknown key is ``unknown_source``, so free text still cannot be spoken.

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
    FrontEndId,
    VoiceActive,
    VoiceControl,
    VoiceControlResult,
    VoiceFrontEnd,
    VoiceLiveToken,
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
from local_operator.voice import controls as voice_controls
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

#: Registry defaults for the relay rows, beside their only reader so
#: ``tests/unit/test_settings_io.py`` can pin the registry literal against them.
DEFAULT_LOCAL_STT_URL = ""
DEFAULT_FRONT_END: FrontEndId = "local"

#: Where a FRONT END's words came from, when it supplied them as text (R3).
#: The transcript is authoritative; the tool argument is the fallback. Recorded
#: as the message's ``input_path``, the slot the STT rungs already write.
TEXT_ORIGINS = frozenset({"gemini_live_transcript", "gemini_live_tool_arg"})

#: The R5 phrase catalogue: the source key -> the catalogue function. CLOSED:
#: ``/speech`` resolves a phrase only through this table, so a key outside it
#: (or any free text in its place) is ``unknown_source``.
PHRASES = {
    "passing_on": _wv.phrase_passing_on,
    "still_working": _wv.phrase_still_working,
    "on_screen_first": _wv.phrase_on_screen_first,
    "didnt_catch": _wv.phrase_didnt_catch,
    "switched": _wv.phrase_switched,
    "nothing_to_repeat": _wv.phrase_nothing_to_repeat,
}


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


def _voice_setting(config_manager: ConfigManager, key: str, default: Any) -> Any:
    """One ``voice.*`` row as it is on DISK now (see :func:`_max_spoken_chars`).

    An unreadable file reads as the default: voice never fails on a setting.
    """
    from local_operator import settings_io

    try:
        return settings_io.read_setting(
            ConfigManager(config_manager.config_dir), settings_io.BY_KEY[key]
        )
    except Exception:  # noqa: BLE001 - a hand-broken config.yml must not break voice
        return default


def _local_stt_url(config_manager: ConfigManager) -> str:
    value = _voice_setting(config_manager, "voice.local_stt_url", DEFAULT_LOCAL_STT_URL)
    return value.strip() if isinstance(value, str) else DEFAULT_LOCAL_STT_URL


def _front_end(config_manager: ConfigManager) -> FrontEndId:
    value = _voice_setting(config_manager, "voice.front_end", DEFAULT_FRONT_END)
    return "gemini_live" if value == "gemini_live" else "local"


def _google_key(config_manager: ConfigManager) -> str | None:
    """The AI Studio key the Live mint would use, by the image rung's rule.

    ``imagegen.availability.google_key``: ``lop login google`` rows, then the
    stored ``GOOGLE_AI_STUDIO_API_KEY`` secret, then the exported variable. One
    rule, so "a Google key is stored" means the same thing on both surfaces.
    """
    from local_operator.imagegen.availability import google_key

    return google_key(Path(str(config_manager.config_dir)))


def _front_ends(
    config_manager: ConfigManager, stt_reason: str, stt_ok: bool
) -> list[VoiceFrontEnd]:
    """Both front ends' availability (R8).

    ``local`` is ALWAYS available: it is the slice-1 pipeline, whose text door
    works with no STT rung at all; its reason is the STT state, so the client
    can say why push-to-talk would be refused. ``gemini_live`` needs a stored
    Google key (the mint's credential); whether Google will mint is a call-time
    fact, like every rung's.
    """
    has_key = bool(_google_key(config_manager))
    live_reason = _wv.front_end_live_available() if has_key else _wv.front_end_live_no_key()
    return [
        VoiceFrontEnd(
            id="local",
            available=True,
            reason=_wv.front_end_local_available().envelope()["text"] if stt_ok else stt_reason,
        ),
        VoiceFrontEnd(id="gemini_live", available=has_key, reason=live_reason.envelope()["text"]),
    ]


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
    """The voice routes' STT resolution: the cascade WITH the loopback rung (R8).

    The voice routes are the only callers that opt in, so ``/v1/stt/paths`` and
    every other surface resolve exactly as before. An empty setting still
    passes ``""``, which lists the rung as unavailable with its reason.
    """
    from local_operator.stt.cascade import resolve_audio_path

    return await resolve_audio_path(
        config_dir=config_manager.config_dir,
        base_url=env_config.radient_api_base_url,
        store=store,
        local_stt_url=_local_stt_url(config_manager),
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
    stt_ok = stt.path != AudioPath.NONE
    front_ends = await asyncio.to_thread(_front_ends, config_manager, stt.reason, stt_ok)
    return reply(
        VoiceStatus(
            active=_active(_config_dir(request)),
            stt=VoicePathState(path=stt.path.value, available=stt_ok, reason=stt.reason),
            tts=VoicePathState(path=tts.path.value, available=tts.servable, reason=tts.reason),
            front_ends=front_ends,
            settings=VoiceSettings(
                max_spoken_chars=_max_spoken_chars(config_manager),
                front_end=_front_end(config_manager),
            ),
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

    ``front_end`` (enable only, R8) is checked BEFORE the slot moves, so an
    unavailable one (``front_end_unavailable``) leaves both the slot and the
    setting as they were; it is persisted to ``voice.front_end`` only once the
    slot write succeeded.
    """
    from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable

    config_dir = _config_dir(request)
    if body.op == "enable":
        if not _session_exists(config_dir, body.session_id):
            raise _refuse(404, "unknown_session", _wv.unknown_session(session_id=body.session_id))
        tts = await _resolve_tts(config_manager, env_config, store)
        if not tts.servable:
            raise _refuse(409, "voice_unavailable", _wv.unavailable(reason=tts.reason))
        if body.front_end is not None:
            stt = await _resolve_stt(config_manager, env_config, store)
            offered = await asyncio.to_thread(
                _front_ends, config_manager, stt.reason, stt.path != AudioPath.NONE
            )
            chosen = next(item for item in offered if item.id == body.front_end)
            if not chosen.available:
                raise _refuse(
                    409, "front_end_unavailable", _wv.front_end_unavailable(reason=chosen.reason)
                )
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
    if body.op == "enable" and body.front_end is not None:
        await asyncio.to_thread(_write_front_end, config_manager, body.front_end)
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


def _write_front_end(config_manager: ConfigManager, front_end: FrontEndId) -> None:
    """Persist the chosen front end through the settings write facade."""
    from local_operator import settings_io

    settings_io.write_setting(
        ConfigManager(config_manager.config_dir), settings_io.BY_KEY["voice.front_end"], front_end
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
                local_stt_url=_local_stt_url(config_manager),
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
    text_origin: Optional[str] = Form(None, max_length=64),
    controls: int = Form(0, ge=0, le=1),
) -> CRUDResponse[VoiceUtteranceResult]:
    """Transcribe (or take ``text``) and admit it to the slot's session as a prompt.

    The text door has the same authority as ``/messages``: the slot check, the
    pool's door and the receipt journal all apply. Only the STT step is skipped.

    ``text_origin`` (R3) names a FRONT END's source for ``text`` and becomes the
    message's ``input_path``; it is refused with a recording, whose path is the
    STT rung's. ``controls=1`` (R6) matches the WHOLE utterance against
    ``voice/controls.py``: a match answers ``control`` with ``admission: null``
    and nothing is admitted or journalled.

    A control take on the TEXT door is matched before the V13 card check,
    because it never reaches the session and a spoken "stop" must still stop
    the announcement of the card. A recorded take cannot be: it is a control
    only once transcribed, and V13 refuses a recording before STT runs.
    """
    try:
        _REQUEST_ID.validate_python(request_id)
    except ValidationError:
        raise _refuse(422, "invalid_voice_request", _wv.invalid_request_id()) from None
    if (file is None) == (text is None):
        raise _refuse(422, "invalid_voice_request", _wv.invalid_request())
    if text_origin is not None and (file is not None or text_origin not in TEXT_ORIGINS):
        raise _refuse(422, "invalid_voice_request", _wv.invalid_text_origin())
    require_slot(_config_dir(request), session_id)
    if controls and text is not None:
        action = voice_controls.match(text)
        if action is not None:
            return reply(
                VoiceUtteranceResult(text=text.strip(), path=None, admission=None, control=action)
            )
    # V13, before STT: a refused recording must cost nothing.
    if await _card_pending(request, session_id):
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
        if await _card_pending(request, session_id):
            raise _refuse(409, "approval_pending", _wv.approval_pending())
        if controls:
            action = voice_controls.match(spoken)
            if action is not None:
                return reply(
                    VoiceUtteranceResult(
                        text=spoken.strip(), path=path, admission=None, control=action
                    )
                )
    else:
        spoken, path = text or "", text_origin
    spoken = spoken.strip()
    if not spoken:
        raise _refuse(422, "empty_transcript", _wv.empty_transcript())

    admission = await _admit(request, session_id, request_id, spoken, path)
    return reply(
        VoiceUtteranceResult(
            text=spoken, path=path, admission=MessageAdmission.model_validate(admission)
        ),
    )


async def _card_pending(request: Request, session_id: str) -> bool:
    """V13: does the session have a card (an approval, or a blocking ask) parked now?

    Every pending card counts, not only approvals: a blocking ask is as
    interruptible as ``bash`` and a steer withdraws it the same way.

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
    return gate is not None


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
    elif source.kind == "phrase":
        phrase = PHRASES.get(source.key)
        raw = phrase().envelope()["text"] if phrase is not None else None
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


# -- POST /v1/desktop/voice/live-token ------------------------------------------


@router.post("/v1/desktop/voice/live-token", response_model=CRUDResponse[VoiceLiveToken])
async def voice_live_token(
    request: Request,
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
) -> CRUDResponse[VoiceLiveToken]:
    """Mint ONE locked, single-use Gemini Live token for Electron main (R2).

    MAIN ONLY. The token is a credential, and R2 keeps it off IPC the way the
    bearer is kept off it, so a browser-shaped request (one carrying the
    unforgeable ``Sec-Fetch-Site``, i.e. the renderer or any page) is refused
    403 ``live_main_only`` even with a valid bearer.

    409 ``live_unavailable`` when no Google AI Studio key is stored; 502
    ``live_failed`` when Google refuses or cannot be reached, with a sentence
    from which both secrets are masked. The key is never logged and the token
    never is either (``voice/live_token.py``).

    No slot check: main mints on the first push-to-talk press, before or
    independent of which session holds the slot, and the token carries no
    session context (R4/R9). The relay itself still goes through
    ``/utterance``, which is slot-gated.
    """
    from datetime import datetime, timezone

    from local_operator.server.desktop import _is_non_browser_caller
    from local_operator.voice import live_token

    if not _is_non_browser_caller(request):
        raise _refuse(403, "live_main_only", _wv.live_main_only())
    api_key = await asyncio.to_thread(_google_key, config_manager)
    if not api_key:
        raise _refuse(
            409,
            "live_unavailable",
            _wv.live_unavailable(reason=_wv.front_end_live_no_key().envelope()["text"]),
        )
    try:
        minted = await live_token.mint(api_key, now=datetime.now(timezone.utc))
    except live_token.LiveTokenError as exc:
        # ``str(exc)`` is built to be safe (no key, no token); see the module.
        logger.info("voice live token refused: %s", exc)
        raise _refuse(502, "live_failed", _wv.live_failed(reason=str(exc))) from None
    return reply(
        VoiceLiveToken(
            token=minted.token,
            expires_at=minted.expires_at,
            new_session_expires_at=minted.new_session_expires_at,
            model=minted.model,
        )
    )
