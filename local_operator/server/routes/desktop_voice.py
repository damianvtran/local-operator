"""Per-session voice mode: the desktop surface (docs/design/qwen-voice.md).

The frozen contract, which both repositories build against:

    GET  /v1/desktop/voice
        200 {active: {session_id, enabled_at} | null, settings: {max_spoken_chars}}
    POST /v1/desktop/voice             body {op: enable|disable, session_id}
        200 {active, replaced}; 404 unknown_session; 409 voice_store_busy
    POST /v1/desktop/voice/utterance   form session_id, request_id (dashed UUID),
                                       EXACTLY ONE of file (a recording) | text,
                                       text_origin? (qwen_s2s | local_moonshine; text door only)
        200 {text, path, admission}; 409 voice_not_enabled | approval_pending |
        stt_unavailable; 413 voice_upload_too_large; 429 voice_rate_limited;
        422 invalid_voice_request | empty_transcript
    POST /v1/desktop/voice/text        body {session_id, source}
        source {kind: message, message_id} | {kind: gate, request_id}
        200 {text, spoken_chars, truncated, segments}; 409 voice_not_enabled;
        404 unknown_source
    POST /v1/desktop/voice/speech      body {session_id, source, segment}
        source {kind: message, message_id} | {kind: gate, request_id} |
               {kind: phrase, key}
        200 audio/wav (X-Voice-Path, X-Voice-Spoken-Chars); 404 unknown_source |
        unknown_segment; 409 voice_not_enabled | voice_unavailable;
        429 voice_rate_limited

and the ``type: "voice"`` frame on ``GET /v1/desktop/events`` (published by
``server/utils/desktop_feed.py``), ``{active}``, on every slot change.

Refusals use the i18n envelope with a machine ``code``, the shape
``desktop_aida.py`` uses: a client classifies by ``detail.code`` and shows
``detail.text``.

NO DUPLICATE SPEECH CODE (session-voice-groq D1, which replaces Q10). Hearing
and speaking are the harness cascades': ``transcribe_audio`` and
``synthesize_speech``, both with ``include_groq=True`` (the Groq rung is opt-in
so the generic ``/v1/stt/paths`` and ``/v1/tts/paths`` stay unchanged). This
module adds no rung, no client and no retry of its own.

ONE SESSION AT A TIME (V3). Every route but the status read refuses a session
that is not in the slot, so a stale renderer cannot keep talking to the
conversation another window switched away from. A slot whose conversation was
deleted reads as EMPTY: the delete route clears it, and a read that finds the
directory gone (a delete from another surface) treats it as cleared.

THE UTTERANCE IS AN ORDINARY PROMPT (V5). It is admitted through the same door
and the same owner op as ``POST /v1/desktop/sessions/{id}/messages``: the pool's
``session()``, the receipt journal under the same ``session_id:request_id``
key family, and ``bridge.remote.admit_prompt``. It is marked
``input_mode="dictated"``, with ``input_path`` naming the ``text_origin`` when
the caller gave one. The steer-or-prompt choice is ``admit_receipt_request``'s
own rule (``desktop_sessions.py``): steer while the session is streaming, else
prompt.

V7, A HARD RULE: THIS MODULE NEVER ANSWERS A GATE. It has no import of, and no
call to, ``answer_gate`` or the ``/answers`` handler. The operator approves on
screen through the existing presence-gated card.

V13: NO UTTERANCE WHILE A CARD IS PENDING. A steer is not an answer, but it is
not harmless either: an INTERRUPTIBLE tool's card (``bash``'s approval, and a
BLOCKING ask under ``LOP_ASK_NONBLOCKING=0``) is cancelled by a steer ("Tool call
skipped: interrupted by steering."), so a word heard at the card would make it
vanish. The utterance route therefore refuses with 409 ``approval_pending``
while the slot session has ANY pending card. A queued ask (the default) parks
no card, so it never refuses.

KNOWN RACE, ACCEPTED: the check reads the facade, and the steer is a separate
owner op. A card that parks between the two (milliseconds; reproduced at 1-4 ms)
is not seen, and the steer cancels it, exactly as a typed steer would today.
Nothing is approved and the tool does not run. Closing it needs an owner-side
"refuse if a card is pending" flag carried through six layers plus a capability
flag (design V13), so it is deferred; the race is pinned, not hidden, by
``tests/unit/voice/test_utterance_during_approval.py``.
``tests/unit/server/test_desktop_voice.py`` pins the static absence of any
answer path.

THE SERVER DECIDES WHAT IS SPOKEN (V8, amended by Q10). ``/voice/text`` takes a
source reference, never text: a message id resolved from the session's own
durable journal (an ASSISTANT row only), or the pending approval card as the
server resolves it (its title and description). That description is the tool's
own approval summary, which can quote MODEL-WRITTEN arguments (a bash command, a
path), so it is not host-authored prose; it is the same text the card shows on
screen. Every source goes through ``voice/speakable.py`` (no code blocks, no
URLs, a cap) and the cleaned text is RETURNED for the rig to speak verbatim.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.routing import APIRoute
from pydantic import TypeAdapter, ValidationError

from local_operator.clients._http import APIError
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
    VoiceSettings,
    VoiceSpeechRequest,
    VoiceStatus,
    VoiceText,
    VoiceTextRequest,
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
from local_operator.voice import segments as voice_segments
from local_operator.voice import slot as voice_slot
from local_operator.voice import speakable as voice_speakable

logger = logging.getLogger("local_operator.server.routes.desktop_voice")


class _UploadCeilingRoute(APIRoute):
    """Refuse an oversized body from ``Content-Length`` before the form is parsed.

    Starlette spools every multipart part to disk while parsing, ahead of any
    handler code, so the post-read check in :func:`_save_upload` fires only after
    the whole upload is on disk. A route class is the one hook that runs first.
    The header can be absent or understated (chunked bodies); the post-read check
    stays as the backstop. The form's text fields add a little to the length, so
    the ceiling here only needs to be a lower bound on the file part.
    """

    def get_route_handler(self) -> Any:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            declared = request.headers.get("content-length", "")
            if (
                request.url.path.endswith("/utterance")
                and declared.isdigit()
                and int(declared) > MAX_UPLOAD_BYTES
            ):
                raise _refuse(
                    413,
                    "voice_upload_too_large",
                    _wv.upload_too_large(limit_mib=str(MAX_UPLOAD_BYTES // (1024 * 1024))),
                )
            return await original(request)

        return handler


router = APIRouter(
    tags=["Voice"], dependencies=[Depends(require_desktop)], route_class=_UploadCeilingRoute
)


_REQUEST_ID = TypeAdapter(RequestID)

#: Where a caller's words came from, when it names that (R3, kept by Q2): the
#: rig's Parakeet final. Recorded as the message's ``input_path``. CLOSED, so an
#: origin is a vocabulary the session's history can be read by, not free text.
TEXT_ORIGINS = frozenset({"qwen_s2s", "local_moonshine"})

#: The recording door's ceiling. Larger than ``/messages``' audio block because
#: this is a multipart upload that never reaches the owner: only the transcript does.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

#: The containers the door accepts, by SNIFFED bytes, never the client's label.
ACCEPTED_AUDIO = frozenset({"audio/wav", "audio/webm", "audio/ogg", "audio/mp4"})

#: The STT rungs' vendor labels for the spend record.
_STT_PROVIDERS = {
    AudioPath.PROVIDER_STT_ELEVENLABS.value: "elevenlabs",
    AudioPath.PROVIDER_STT_OPENAI.value: "openai",
    AudioPath.PROVIDER_STT_GROQ.value: "groq",
}

#: Every voice TTS call asks for WAV: the page decodes and schedules PCM
#: sample-accurately (D3), and the Groq rung always answers WAV. ``TtsOutcome``
#: carries no format, so the route fixes it rather than guessing from the rung.
VOICE_AUDIO_FORMAT = "wav"
VOICE_AUDIO_MIME = "audio/wav"

#: The fixed spoken phrases: key -> catalogue message. CLOSED: ``/speech``
#: resolves a phrase only through this table, so any other key (or free text in
#: its place) is ``unknown_source``. ``truncated`` is ``speakable``'s closing
#: sentence, which the segmenter makes its own final segment so it is served
#: from the phrase cache instead of a paid request.
PHRASES = {
    "on_screen_first": _wv.phrase_on_screen_first,
    "voice_off": _wv.phrase_voice_off,
    "switched": _wv.phrase_switched,
    "failed": _wv.phrase_failed,
    "still_working": _wv.phrase_still_working,
    "truncated": _wv.phrase_truncated,
}

#: A 429 with no ``remaining_requests`` falls back to the size of ``retry-after``
#: (D4): a wait longer than this is a daily limit, not a minute one.
_DAY_SCOPE_RETRY_S = 60.0


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


# -- GET /v1/desktop/voice -----------------------------------------------------


@router.get("/v1/desktop/voice", response_model=CRUDResponse[VoiceStatus])
async def voice_status(
    request: Request,
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
) -> CRUDResponse[VoiceStatus]:
    """The slot and the cap. Never writes anything."""
    return reply(
        VoiceStatus(
            active=_active(_config_dir(request)),
            settings=VoiceSettings(max_spoken_chars=_max_spoken_chars(config_manager)),
        ),
    )


# -- POST /v1/desktop/voice ----------------------------------------------------


@router.post("/v1/desktop/voice", response_model=CRUDResponse[VoiceControlResult])
async def voice_control(
    body: VoiceControl,
    request: Request,
) -> CRUDResponse[VoiceControlResult]:
    """Enable (switching the slot to this session) or disable voice.

    No speech precondition (Q10): hearing and speaking are the desktop rig's,
    so the server has nothing to probe before the slot moves.

    ``disable`` of a session that does not hold the slot is a no-op that answers
    the slot as it is. A stale window cannot switch off the conversation another
    window just enabled.
    """
    from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable

    config_dir = _config_dir(request)
    if body.op == "enable":
        if not _session_exists(config_dir, body.session_id):
            raise _refuse(404, "unknown_session", _wv.unknown_session(session_id=body.session_id))
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


# -- 429 -> voice_rate_limited (D4) ---------------------------------------------


def _rate_limit_scope(details: dict[str, Any]) -> tuple[str, float | None, int | None]:
    """``(scope, retry_after_s, remaining_requests)`` from ``APIError.details``.

    ``day`` when the provider reports ZERO requests left today; with no count the
    size of ``retry-after`` decides. Tolerant of a count that arrives as a string.
    """
    retry: float | None
    try:
        retry = float(details["retry_after_s"])
    except (KeyError, TypeError, ValueError):
        retry = None
    remaining: int | None
    try:
        remaining = int(float(details["remaining_requests"]))
    except (KeyError, TypeError, ValueError):
        remaining = None
    if remaining is not None:
        scope = "day" if remaining <= 0 else "minute"
    else:
        scope = "day" if retry is not None and retry > _DAY_SCOPE_RETRY_S else "minute"
    return scope, retry, remaining


def _rate_limited(error: BaseException | None, rung: Any) -> HTTPException | None:
    """The 429 refusal built from the error's ``details``, or ``None`` for any other error.

    Built BEFORE ``_speech_refusal`` / the STT classifier, because their 429 arm
    is a fixed sentence that carries no ``retry_after_s``, and the app needs the
    number to decide between waiting once and stopping for the day.
    """
    if not isinstance(error, APIError) or error.status_code != 429:
        return None
    scope, retry, remaining = _rate_limit_scope(dict(error.details or {}))
    message = _wv.rate_limited_day() if scope == "day" else _wv.rate_limited_minute()
    detail = message.envelope(code="voice_rate_limited")
    detail.update(
        {
            "scope": scope,
            "retry_after_s": retry,
            "remaining_requests": remaining,
            "rung": getattr(rung, "value", rung),
        }
    )
    headers = {"Retry-After": str(max(1, math.ceil(retry)))} if retry is not None else None
    return HTTPException(429, detail, headers=headers)


# -- POST /v1/desktop/voice/utterance -------------------------------------------


async def _save_upload(file: UploadFile) -> tuple[str, Path]:
    """Copy the upload to a private temp dir, bounded at ``MAX_UPLOAD_BYTES``.

    Returns ``(sniffed mime, path)``; the temp dir is the caller's to remove.
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

    The walk includes the Groq rung (``include_groq=True``). Refuses with
    ``voice_rate_limited`` (429) when the chosen failure is a 429, with
    ``stt_unavailable`` (409) when no rung exists, and with the cascade's own
    classified status when every rung failed. Spend goes through
    ``emit_voice_spend`` with the session id.
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
                include_groq=True,
            )
        except SttUnavailable as exc:
            limited = _rate_limited(exc.error, exc.attempts[-1].path if exc.attempts else None)
            if limited is not None:
                raise limited from None
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
) -> CRUDResponse[VoiceUtteranceResult]:
    """Transcribe a recording (or take ``text``) and admit it to the slot's session.

    Exactly one of ``file`` and ``text``. The text door has the same authority
    as ``/messages``: the slot check, the pool's door and the receipt journal all
    apply; only the STT step is skipped. ``text_origin`` names the text door's
    source and becomes the message's ``input_path``; it is refused with a
    recording, whose ``input_path`` is the STT rung's token.

    V13 is checked BEFORE STT (a refused recording must cost nothing) and again
    after it: STT takes seconds, a card can park inside them, and the money is
    spent by then but the card is not.
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
        if await _card_pending(request, session_id):
            raise _refuse(409, "approval_pending", _wv.approval_pending())
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


# -- POST /v1/desktop/voice/text ----------------------------------------------


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
    not spoken.
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


@router.post("/v1/desktop/voice/text", response_model=CRUDResponse[VoiceText])
async def voice_text(
    body: VoiceTextRequest,
    request: Request,
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
) -> CRUDResponse[VoiceText]:
    """The words to speak for one SOURCE the server resolves itself (V8, Q10).

    Never free text: the body names a message or a pending gate, and the text
    comes from the session's own journal or card. The result is ``speakable``'s
    output, so the rig speaks exactly what this returns and nothing else.

    ``truncated`` is ``speakable``'s own report, forwarded rather than recomputed:
    only it knows whether the cap actually cut this reply at this request's
    ``voice.max_spoken_chars``.
    """
    require_slot(_config_dir(request), body.session_id)
    source = body.source
    if source.kind == "message":
        raw = await _message_text(request, body.session_id, source.message_id)
    else:
        raw = await _gate_text(request, body.session_id, source.request_id)
    if raw is None:
        raise _refuse(404, "unknown_source", _wv.unknown_source())
    spoken = voice_segments.speak_segments(raw, max_chars=_max_spoken_chars(config_manager))
    if not spoken.text:
        raise _refuse(404, "unknown_source", _wv.unknown_source())
    return reply(
        VoiceText(
            text=spoken.text,
            spoken_chars=len(spoken.text),
            truncated=spoken.truncated,
            segments=list(spoken.segments),
        )
    )


# -- POST /v1/desktop/voice/speech -----------------------------------------------


def _phrase_cache_file(config_dir: Path, rung: str, voice: str, key: str) -> Path:
    """``<config>/voice/phrase-cache/<rung>-<voice>-<key>.wav``.

    ``key`` is always a member of :data:`PHRASES` and ``rung``/``voice`` are
    closed vocabularies, so no caller text reaches the path.
    """
    return config_dir / "voice" / "phrase-cache" / f"{rung}-{voice}-{key}.wav"


def _store_phrase(path: Path, audio: bytes) -> None:
    """Write the cache file atomically; a cache that cannot be written is only a miss."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(audio)
        os.replace(tmp, path)
    except OSError:
        logger.warning("voice phrase cache write failed", exc_info=True)


def _audio_headers(rung: str, chars: int, *, cache: str | None = None) -> dict[str, str]:
    headers = {
        "X-Voice-Path": rung,
        "X-Voice-Spoken-Chars": str(chars),
        "Cache-Control": "no-store",
    }
    if cache is not None:
        headers["X-Voice-Cache"] = cache
    return headers


@router.post("/v1/desktop/voice/speech")
async def voice_speech(
    body: VoiceSpeechRequest,
    request: Request,
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
    env_config: Annotated[EnvConfig, Depends(get_env_config)],
    store: Annotated[AuthStore, Depends(get_provider_auth_store)],
) -> Response:
    """Synthesise ONE segment of a source the server resolves itself (V8, D3).

    The server re-resolves the source and re-segments it, so ``segment`` is an
    index and no free text crosses. A phrase source is one segment (index 0). A
    segment that is a fixed phrase (a catalogue phrase, or ``speakable``'s
    closing sentence) is served from the phrase cache and costs no cascade call
    on a hit.

    200 carries WAV (``audio/wav``: the cascade is asked for it, see
    :data:`VOICE_AUDIO_FORMAT`) with ``X-Voice-Path`` (the TTS rung that served)
    and ``X-Voice-Spoken-Chars``. A 429 is ``voice_rate_limited``.
    """
    from local_operator.tts import cascade as tts_cascade
    from local_operator.tts.adapters import Legacy
    from local_operator.tts.descriptor import descriptor_from_config

    config_dir = _config_dir(request)
    require_slot(config_dir, body.session_id)
    source = body.source
    phrase_key: str | None = None
    if source.kind == "phrase":
        phrase = PHRASES.get(source.key)
        if phrase is None or body.segment != 0:
            raise _refuse(404, "unknown_source", _wv.unknown_source())
        phrase_key = source.key
        text = phrase().envelope()["text"]
    else:
        if source.kind == "message":
            raw = await _message_text(request, body.session_id, source.message_id)
        else:
            raw = await _gate_text(request, body.session_id, source.request_id)
        if raw is None:
            raise _refuse(404, "unknown_source", _wv.unknown_source())
        spoken = voice_segments.speak_segments(raw, max_chars=_max_spoken_chars(config_manager))
        if not spoken.text:
            raise _refuse(404, "unknown_source", _wv.unknown_source())
        if body.segment >= len(spoken.segments):
            raise _refuse(404, "unknown_segment", _wv.unknown_segment())
        text = spoken.segments[body.segment]
        if text == voice_speakable.TRUNCATION_TAIL:
            phrase_key = "truncated"

    descriptor = descriptor_from_config(config_manager)
    cache_file: Path | None = None
    if phrase_key is not None:
        # The rung that WOULD serve decides the cache name: a phrase is paid once
        # per rung and voice (D3). A fall-forward to another rung is a miss, then
        # stored under the rung that actually served.
        resolution = await tts_cascade.resolve_voice_path(
            config_dir=config_dir,
            base_url=env_config.radient_api_base_url,
            store=store,
            include_groq=True,
        )
        if resolution.servable:
            cache_file = _phrase_cache_file(
                config_dir, resolution.path.value, descriptor.resolved_gender(), phrase_key
            )
            try:
                cached = cache_file.read_bytes()
            except OSError:
                cached = b""
            if cached and stt_audio.sniff_audio(cached) is None:
                # A truncated or corrupt file must not be served for ever: drop
                # it and fall through to a paid regeneration, which re-stores it.
                cache_file.unlink(missing_ok=True)
                cached = b""
            if cached:
                return Response(
                    content=cached,
                    media_type=VOICE_AUDIO_MIME,
                    headers=_audio_headers(resolution.path.value, len(text), cache="hit"),
                )

    # The uuid suffix keeps two segments spoken inside one millisecond from
    # sharing a correlation id on the spend row.
    request_id = f"voice-{body.session_id}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}"
    try:
        outcome = await tts_cascade.synthesize_speech(
            text,
            config_dir=config_dir,
            base_url=env_config.radient_api_base_url,
            store=store,
            descriptor=descriptor,
            legacy=Legacy(),
            response_format=VOICE_AUDIO_FORMAT,
            correlation_session_id=body.session_id,
            client_request_id=request_id,
            include_groq=True,
        )
    except tts_cascade.TtsUnavailable as unavailable:
        if unavailable.error is None:
            raise _refuse(
                409, "voice_unavailable", _wv.unavailable(reason=unavailable.resolution.reason)
            ) from None
        limited = _rate_limited(unavailable.error, unavailable.failed_path)
        if limited is not None:
            raise limited from None
        from local_operator.server.routes.speech import _speech_refusal

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
        units=float(len(text)),
        unit="chars",
        cost=cost_from_headers(outcome.cost_headers),
        session_id=body.session_id,
        request_id=request_id,
    )
    if cache_file is not None:
        info = stt_audio.sniff_audio(outcome.audio)
        if info is not None and info.mime_type == "audio/wav":
            served = _phrase_cache_file(
                config_dir, outcome.path.value, descriptor.resolved_gender(), phrase_key or ""
            )
            await asyncio.to_thread(_store_phrase, served, outcome.audio)
    return Response(
        content=outcome.audio,
        media_type=VOICE_AUDIO_MIME,
        headers=_audio_headers(outcome.path.value, len(text)),
    )
