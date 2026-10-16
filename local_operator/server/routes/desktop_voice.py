"""Per-session voice mode: the desktop surface (docs/design/qwen-voice.md).

The frozen contract, which both repositories build against:

    GET  /v1/desktop/voice
        200 {active: {session_id, enabled_at} | null, settings: {max_spoken_chars}}
    POST /v1/desktop/voice             body {op: enable|disable, session_id}
        200 {active, replaced}; 404 unknown_session; 409 voice_store_busy
    POST /v1/desktop/voice/utterance   form session_id, request_id (dashed UUID),
                                       text, text_origin? (qwen_s2s)
        200 {text, path, admission}; 409 voice_not_enabled | approval_pending;
        422 invalid_voice_request | empty_transcript
    POST /v1/desktop/voice/text        body {session_id, source}
        source {kind: message, message_id} | {kind: gate, request_id}
        200 {text, spoken_chars}; 409 voice_not_enabled; 404 unknown_source

and the ``type: "voice"`` frame on ``GET /v1/desktop/events`` (published by
``server/utils/desktop_feed.py``), ``{active}``, on every slot change.

Refusals use the i18n envelope with a machine ``code``, the shape
``desktop_aida.py`` uses: a client classifies by ``detail.code`` and shows
``detail.text``.

NO SPEECH CODE HERE (Q10). Hearing and speaking are the Qwen rig's, in the
desktop app: this module neither transcribes nor synthesises, and imports
nothing from ``local_operator.stt`` or ``local_operator.tts``. Words arrive as
text (the rig's transcript) and leave as text (``/voice/text``).

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
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from pydantic import TypeAdapter, ValidationError

from local_operator.config import ConfigManager
from local_operator.i18n.keys import wire_voice as _wv
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_sessions import MessageAdmission
from local_operator.server.models.desktop_voice import (
    VoiceActive,
    VoiceControl,
    VoiceControlResult,
    VoiceSettings,
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
from local_operator.voice import slot as voice_slot
from local_operator.voice import speakable as voice_speakable

logger = logging.getLogger("local_operator.server.routes.desktop_voice")

router = APIRouter(tags=["Voice"], dependencies=[Depends(require_desktop)])

_REQUEST_ID = TypeAdapter(RequestID)

#: Where a caller's words came from, when it names that (R3, kept by Q2): the
#: rig's Parakeet final. Recorded as the message's ``input_path``. CLOSED, so an
#: origin is a vocabulary the session's history can be read by, not free text.
TEXT_ORIGINS = frozenset({"qwen_s2s"})


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


# -- POST /v1/desktop/voice/utterance -------------------------------------------


@router.post("/v1/desktop/voice/utterance", response_model=CRUDResponse[VoiceUtteranceResult])
async def voice_utterance(
    request: Request,
    session_id: str = Form(..., min_length=1, max_length=128),
    request_id: str = Form(...),
    text: Optional[str] = Form(None, max_length=200_000),
    text_origin: Optional[str] = Form(None, max_length=64),
) -> CRUDResponse[VoiceUtteranceResult]:
    """Admit ``text`` to the slot's session as a dictated prompt (or steer).

    The TEXT DOOR ONLY (Q10): there is no recording field, so this route never
    transcribes. It has the same authority as ``/messages``: the slot check, the
    pool's door and the receipt journal all apply.

    ``text_origin`` (R3) names where the words came from and becomes the
    message's ``input_path``.

    ``text`` is optional in the signature only so that its absence is the
    contract's declared ``invalid_voice_request`` rather than FastAPI's generic
    field-missing 422, which names nothing a caller could act on.
    """
    try:
        _REQUEST_ID.validate_python(request_id)
    except ValidationError:
        raise _refuse(422, "invalid_voice_request", _wv.invalid_request_id()) from None
    if text is None:
        raise _refuse(422, "invalid_voice_request", _wv.invalid_request())
    if text_origin is not None and text_origin not in TEXT_ORIGINS:
        raise _refuse(422, "invalid_voice_request", _wv.invalid_text_origin())
    require_slot(_config_dir(request), session_id)
    if await _card_pending(request, session_id):
        raise _refuse(409, "approval_pending", _wv.approval_pending())
    spoken = text.strip()
    if not spoken:
        raise _refuse(422, "empty_transcript", _wv.empty_transcript())

    admission = await _admit(request, session_id, request_id, spoken, text_origin)
    return reply(
        VoiceUtteranceResult(
            text=spoken, path=text_origin, admission=MessageAdmission.model_validate(admission)
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
    """
    require_slot(_config_dir(request), body.session_id)
    source = body.source
    if source.kind == "message":
        raw = await _message_text(request, body.session_id, source.message_id)
    else:
        raw = await _gate_text(request, body.session_id, source.request_id)
    if raw is None:
        raise _refuse(404, "unknown_source", _wv.unknown_source())
    spoken = voice_speakable.speakable(raw, max_chars=_max_spoken_chars(config_manager))
    if not spoken.text:
        raise _refuse(404, "unknown_source", _wv.unknown_source())
    return reply(VoiceText(text=spoken.text, spoken_chars=len(spoken.text)))
