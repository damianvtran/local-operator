"""The desktop's notification-sound surface (``features.notification_sounds``).

- ``GET    /v1/notification-sounds``             every sound, plus the current selection
- ``GET    /v1/notification-sounds/{id}/audio``  the sound as 16-bit WAV at ``?volume=0..100``
- ``POST   /v1/notification-sounds``             add a custom sound (multipart ``file`` + ``name``)
- ``PATCH  /v1/notification-sounds/{id}``        rename a custom sound (``{name}``)
- ``DELETE /v1/notification-sounds/{id}``        remove a custom sound; ``selection_reset`` says
                                                  whether the selection fell back to the default

WHY THE DESKTOP READS SOUNDS FROM HERE, not from files (design §1.3). The
renderer is a browser context: it cannot read ``<config>/sounds`` and must not
learn its paths. The audio route renders on the server with the notification's
own gain curve, so the Settings preview at 70 sounds like a notification at 70
(settings spec: "Do not use setVolume() on the engine").

ONE WRITE FUNNEL. Selection, volume and on/off are not written here: they stay
on ``PATCH /v1/settings/{key}``. The one exception is the reset a DELETE of the
selected sound performs, which ``custom_sounds.remove`` does through the same
``settings_io`` funnel inside the store's lock.

SECURITY (design §3.5). Every route sits behind ``require_desktop`` (bearer +
origin rule), like ``/v1/settings``. No request value reaches the filesystem:
``{id}`` is looked up in the catalog or index (``custom_sounds`` checks the id
regex and resolved containment), and the upload's filename is display text
only. The upload route reads no body byte before ``require_desktop`` has
passed, refuses an oversize ``Content-Length`` unread, and counts the stream
so at most ``MAX_INPUT_BYTES`` plus multipart framing is ever spooled (see
:func:`add_notification_sound`).

REFUSALS use the RFC §2.6 ``{code, params, text}`` envelope with the codes the
settings spec's error table switches on: ``too_large`` 413,
``unsupported_format`` 415 (``params.format`` names opus/m4a/aac),
``too_long`` / ``undecodable`` / ``limit_reached`` / ``invalid_name`` 422,
``not_found`` 404, ``builtin_readonly`` 409, ``store_failed`` 503.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.notifications import custom_sounds
from local_operator.notifications import sound as _sound
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Notification sounds"], dependencies=[Depends(require_desktop)])

# Index writes already serialize on the config dir's writer lock (across
# processes). This lock is about MEMORY: one import holds up to ~30 s of float
# PCM plus its Python-list working copies (tens of MB, see
# `custom_sounds.PRECHECK_DURATION_S`), and a server should hold one of those
# at a time, not one per concurrent upload.
_mutation_lock = asyncio.Lock()


class RenameBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


def _refuse(refusal: custom_sounds.SoundRejected) -> HTTPException:
    return HTTPException(refusal.status, refusal.envelope())


def _row(row: custom_sounds.CustomSound) -> dict[str, Any]:
    return {
        "id": row.id,
        "label": row.name,
        "builtin": False,
        "duration_s": row.duration_s,
        "is_default": False,
        "source_name": row.source_name,
        "normalized": row.normalized,
        "created_at": row.created_at,
    }


def _listing() -> dict[str, Any]:
    from local_operator.settings_io import BY_KEY

    labels = {
        str(choice.value): str(choice.description)
        for choice in BY_KEY["notifications.sound.id"].resolved_choices
    }
    sounds: list[dict[str, Any]] = [
        {
            "id": entry.id,
            # Localized through the registry's keyed choices (follows `language`).
            "label": labels.get(entry.id, entry.id),
            "builtin": True,
            "duration_s": entry.duration_s,
            "is_default": entry.id == _sound.DEFAULT_SOUND_ID,
        }
        for entry in _sound.builtin_sounds()
    ]
    # `system` is a row too (the settings spec lists 11 built-in rows): it has
    # no audio, which `duration_s: null` says, and the audio route 404s it.
    sounds.append(
        {
            "id": _sound.SYSTEM_SOUND_ID,
            "label": labels.get(_sound.SYSTEM_SOUND_ID, _sound.SYSTEM_SOUND_ID),
            "builtin": True,
            "duration_s": None,
            "is_default": False,
        }
    )
    sounds.extend(_row(row) for row in custom_sounds.list_sounds())
    # The same reading `resolve` uses (C1's `configured`), so the list reports
    # exactly what the next notification will obey.
    stored = _sound.configured()
    return {
        "sounds": sounds,
        "selected": stored.sound_id,
        "enabled": stored.enabled,
        "volume": round(stored.effective_volume),
        "limits": {
            "max_sounds": custom_sounds.MAX_SOUNDS,
            "max_bytes": custom_sounds.MAX_INPUT_BYTES,
            "max_duration_s": custom_sounds.MAX_DURATION_S,
            "max_name_chars": custom_sounds.MAX_NAME_CHARS,
        },
    }


def _etag_matches(header: str | None, etag: str) -> bool:
    """RFC 9110 §13.1.2 If-None-Match: ``*``, or a list of tags compared WEAKLY.

    Weak comparison (``W/"x"`` equals ``"x"``) is what the RFC prescribes for
    If-None-Match; the list form is what a browser cache may send.
    """
    if not header:
        return False
    if header.strip() == "*":
        return True
    bare = etag.removeprefix("W/")
    return any(tag.strip().removeprefix("W/") == bare for tag in header.split(","))


@router.get("/v1/notification-sounds", response_model=CRUDResponse[dict[str, Any]])
async def list_notification_sounds() -> CRUDResponse[dict[str, Any]]:
    return CRUDResponse(status=200, message="Notification sounds.", result=_listing())


@router.get("/v1/notification-sounds/{sound_id}/audio")
async def notification_sound_audio(
    sound_id: str,
    request: Request,
    volume: int | None = Query(default=None, ge=0, le=100),
) -> Response:
    """The sound as ``audio/wav`` at ``volume`` (default: the configured volume).

    Cached on disk keyed by content sha + volume, so a rename never serves stale
    bytes and the ETag (sha + volume) lets the renderer revalidate cheaply.
    """
    if volume is None:
        volume = round(_sound.configured().effective_volume)
    try:
        rendered = await asyncio.to_thread(custom_sounds.render, sound_id, volume)
    except OSError:
        logger.exception("notification sound: could not render %s", sound_id)
        raise _refuse(custom_sounds._store_failed()) from None
    if rendered is None:
        raise _refuse(custom_sounds._not_found(sound_id))
    headers = {"ETag": rendered.etag, "X-Content-Type-Options": "nosniff"}
    if _etag_matches(request.headers.get("if-none-match"), rendered.etag):
        return Response(status_code=304, headers=headers)
    # BYTES, not a FileResponse (agent review round 1, R5): FileResponse opens
    # the path at send time, and a concurrent render for another volume (a
    # preview slider being dragged) can prune that file in between. A render is
    # at most ~1.9 MB (10 s of 48 kHz stereo 16-bit), so holding it is cheap.
    return Response(content=rendered.data, media_type="audio/wav", headers=headers)


#: Multipart framing around the one file part (boundaries, part headers and
#: the short ``name`` field). Generous: a real browser upload carries a few
#: hundred bytes of it.
_MULTIPART_SLACK = 64 * 1024


class _BodyTooLarge(Exception):
    """Raised from inside the counted body stream; mapped to 413 by the route."""


def _too_large() -> HTTPException:
    """413 ``too_large`` for a body cut off before its part headers were read.

    The filename is not known yet, so the keyed ``error.too_large`` sentence
    (which names ``{file}``) cannot be used; this is the
    ``/v1/desktop/attachments`` refusal shape instead. The desktop keys its
    copy on ``code`` and already holds the file name it uploaded (settings
    spec §5.2), so ``text`` is only the fallback.
    """
    limit_mb = custom_sounds.MAX_INPUT_BYTES // (1024 * 1024)
    return HTTPException(
        413,
        _wd.desktop_refusal(error=f"the file is larger than {limit_mb} MB").envelope(
            code="too_large"
        ),
    )


@router.post(
    "/v1/notification-sounds", status_code=201, response_model=CRUDResponse[dict[str, Any]]
)
async def add_notification_sound(request: Request) -> CRUDResponse[dict[str, Any]]:
    """Add one custom sound from a multipart ``file`` (and optional ``name``).

    WHY A RAW ``Request`` and not ``UploadFile = File(...)`` (agent review round
    1, R2; QA Q1). FastAPI parses a ``File``/``Form`` body BEFORE it resolves the
    router's dependencies, and Starlette's multipart parser has no size limit,
    so an unauthenticated 1 GB POST (a CORS "simple" request any web page can
    send) was spooled to a temp file in full before ``require_desktop`` said
    401. With no body parameters, the router's ``require_desktop`` runs first
    and nothing reads the body until it has passed. Then:

    1. a declared ``Content-Length`` over the cap is refused without reading;
    2. the body is fed to the multipart parser through a COUNTED stream that
       stops at the cap, so a chunked or lying sender is cut off at
       ``MAX_INPUT_BYTES`` + framing slack; at most that much is ever spooled.
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from starlette.formparsers import MultiPartException, MultiPartParser

    limit = custom_sounds.MAX_INPUT_BYTES + _MULTIPART_SLACK
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise _too_large()
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise RequestValidationError(
            [{"type": "missing", "loc": ("body", "file"), "msg": "Field required"}]
        )

    async def counted():  # type: ignore[no-untyped-def]
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise _BodyTooLarge
            yield chunk

    parser = MultiPartParser(
        request.headers, counted(), max_files=1, max_fields=4, max_part_size=4096
    )
    try:
        form = await parser.parse()
    except _BodyTooLarge:
        raise _too_large() from None
    except MultiPartException:
        raise RequestValidationError(
            [{"type": "value_error", "loc": ("body",), "msg": "Malformed multipart body"}]
        ) from None
    try:
        upload = form.get("file")
        name = form.get("name")
        if not isinstance(upload, StarletteUploadFile):
            raise RequestValidationError(
                [{"type": "missing", "loc": ("body", "file"), "msg": "Field required"}]
            )
        label = upload.filename or "sound"
        raw = await upload.read(custom_sounds.MAX_INPUT_BYTES + 1)
    finally:
        await form.close()
    async with _mutation_lock:
        try:
            row = await asyncio.to_thread(
                custom_sounds.add,
                raw,
                source_name=label,
                name=name if isinstance(name, str) and name else None,
            )
        except custom_sounds.SoundRejected as refusal:
            raise _refuse(refusal) from None
    return CRUDResponse(status=201, message="Sound added.", result=_row(row))


@router.patch("/v1/notification-sounds/{sound_id}", response_model=CRUDResponse[dict[str, Any]])
async def rename_notification_sound(
    sound_id: str, body: RenameBody
) -> CRUDResponse[dict[str, Any]]:
    async with _mutation_lock:
        try:
            row = await asyncio.to_thread(custom_sounds.rename, sound_id, body.name)
        except custom_sounds.SoundRejected as refusal:
            raise _refuse(refusal) from None
    return CRUDResponse(status=200, message="Sound renamed.", result=_row(row))


@router.delete("/v1/notification-sounds/{sound_id}", response_model=CRUDResponse[dict[str, Any]])
async def remove_notification_sound(sound_id: str) -> CRUDResponse[dict[str, Any]]:
    async with _mutation_lock:
        try:
            removal = await asyncio.to_thread(custom_sounds.remove, sound_id)
        except custom_sounds.SoundRejected as refusal:
            raise _refuse(refusal) from None
    result: dict[str, Any] = {"id": removal.sound.id, "selection_reset": removal.selection_reset}
    if removal.selection_reset:
        result["selected"] = _sound.DEFAULT_SOUND_ID
        # A MACHINE code, not the message id: `wire.notification_sound` is a
        # deferred namespace (not in the eager bundle), so a client must never
        # be handed one of its keys to resolve; `text` is already rendered.
        result["message"] = removal.reset_message().envelope(code="selection_reset")
    return CRUDResponse(status=200, message="Sound removed.", result=result)
