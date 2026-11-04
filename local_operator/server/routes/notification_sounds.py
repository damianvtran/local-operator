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
regex and resolved containment), the upload's filename is display text only,
and the upload is read at most ``MAX_INPUT_BYTES + 1`` bytes, so a 1 GB upload
is refused after 10 MB without being held in memory.

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

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict

from local_operator.notifications import custom_sounds
from local_operator.notifications import sound as _sound
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Notification sounds"], dependencies=[Depends(require_desktop)])

# Index writes already serialize on the config dir's writer lock (across
# processes); this keeps one server's concurrent uploads from decoding several
# 10 MB files at once on the event loop's thread pool.
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
    values = _sound._config_values()
    selected = _sound._walk(values, "notifications", "sound", "id")
    enabled = _sound._walk(values, "notifications", "sound", "enabled")
    volume = _sound._walk(values, "notifications", "sound", "volume")
    return {
        "sounds": sounds,
        "selected": selected if isinstance(selected, str) and selected else _sound.DEFAULT_SOUND_ID,
        "enabled": _sound.DEFAULT_ENABLED if enabled is None else enabled is not False,
        "volume": _sound.DEFAULT_VOLUME if not isinstance(volume, int) else volume,
        "limits": {
            "max_sounds": custom_sounds.MAX_SOUNDS,
            "max_bytes": custom_sounds.MAX_INPUT_BYTES,
            "max_duration_s": custom_sounds.MAX_DURATION_S,
            "max_name_chars": custom_sounds.MAX_NAME_CHARS,
        },
    }


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
        configured = _sound._walk(_sound._config_values(), "notifications", "sound", "volume")
        volume = configured if isinstance(configured, int) else _sound.DEFAULT_VOLUME
    try:
        rendered = await asyncio.to_thread(custom_sounds.render, sound_id, volume)
    except OSError:
        logger.exception("notification sound: could not render %s", sound_id)
        raise _refuse(custom_sounds._store_failed()) from None
    if rendered is None:
        raise _refuse(custom_sounds._not_found(sound_id))
    headers = {"ETag": rendered.etag, "X-Content-Type-Options": "nosniff"}
    if request.headers.get("if-none-match") == rendered.etag:
        return Response(status_code=304, headers=headers)
    return FileResponse(rendered.path, media_type="audio/wav", headers=headers)


@router.post(
    "/v1/notification-sounds", status_code=201, response_model=CRUDResponse[dict[str, Any]]
)
async def add_notification_sound(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
) -> CRUDResponse[dict[str, Any]]:
    label = file.filename or "sound"
    # One byte over the cap is enough to answer 413; reading the whole upload
    # first would be the memory the cap exists to bound.
    raw = await file.read(custom_sounds.MAX_INPUT_BYTES + 1)
    async with _mutation_lock:
        try:
            row = await asyncio.to_thread(
                custom_sounds.add, raw, source_name=label, name=name or None
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
        result["message"] = removal.reset_message().envelope()
    return CRUDResponse(status=200, message="Sound removed.", result=result)
