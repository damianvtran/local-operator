"""Desktop settings intents: record a pick at once, apply it in the background.

``POST /v1/desktop/sessions/{id}/settings-intents`` records a model / effort /
fast / cwd pick and answers 202 without touching the runtime;
``GET /v1/desktop/settings-intents`` is every client's baseline (boot and each
feed reconnect); ``DELETE …/settings-intents/{kind}?generation=g`` dismisses a
failed slot or cancels a waiting one. The applier, the store and the retry
policy live in ``server/utils/settings_intents.py``; read its module docstring
for the why.

A MODULE OF ITS OWN rather than more routes in ``desktop_sessions.py``: that
module's error ladder and owner-busy classification are being reworked in
parallel, and these routes deliberately use NEITHER — the POST never reaches an
owner, so there is no owner failure to classify. Helpers from that module are
imported read-only.

Gated for clients by ``features.settings_intents``; a renderer that does not see
it keeps today's ``/commands`` and ``/working-directory`` path byte for byte.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_settings_intents import (
    SettingsIntentIndex,
    SettingsIntentKind,
    SettingsIntentReceipt,
    SettingsIntentRequest,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.utils.settings_intents import (
    ACTIVE_STATES,
    BridgeDriver,
    Current,
    SettingsApplier,
    valid_session_id,
)

# The desktop credential gate, like every desktop router: these routes write
# into a session's directory and drive its runtime in the background.
router = APIRouter(tags=["Desktop settings intents"], dependencies=[Depends(require_desktop)])


def ensure_applier(app: Any) -> SettingsApplier:
    """The process's ONE applier, created lazily like ``desktop_feed``.

    The pool and the feed are resolved PER USE through the desktop routes' own
    lazy accessors (they only read ``request.app``), so the applier neither
    creates a session pool at boot when there is nothing to resume nor holds a
    stale one. Frames are published only when a feed already exists: with no
    feed there is no subscriber, and every client reads ``GET`` on connect.
    """
    applier = getattr(app.state, "settings_applier", None)
    if applier is not None:
        return applier
    from local_operator.server.routes import desktop_sessions

    shim = SimpleNamespace(app=app)

    def publish(session_id: str, kind: str, intent: dict[str, Any] | None) -> None:
        feed = getattr(app.state, "desktop_feed", None)
        if feed is not None:
            feed.publish_settings_intent(session_id, kind, intent)

    applier = SettingsApplier(
        app.state.config_manager.config_dir,
        BridgeDriver(lambda: desktop_sessions.host(shim)),  # type: ignore[arg-type]
        publish=publish,
    )
    app.state.settings_applier = applier
    return applier


def _unknown_session() -> HTTPException:
    return HTTPException(404, _wd.requested_not_found().envelope(code="session_unknown"))


async def _validate_model(provider_model: str, effort: str | None) -> None:
    """Refuse a model (and paired effort) the LOCAL catalogue says cannot run.

    The same boundary check a new draft pays (``_draft_model_spec``: provider
    registry, offline catalogue, the model's effort ladder), so the codes and
    sentences are the existing ones: ``ModelSelectionRefused.code`` and
    ``effort_unsupported``. A catalogue that cannot be enumerated offline does
    not refuse — "we have not looked" is not "it does not exist". Off the event
    loop because it reads catalogue caches.
    """
    from local_operator.server.routes.desktop_sessions import (
        DraftModel,
        _draft_model_spec,
    )

    provider, sep, model_id = provider_model.partition("/")
    if not sep or not provider or not model_id:
        raise HTTPException(
            422,
            _wd.desktop_refusal(error=provider_model).envelope(code="model_unknown"),
        )
    draft = DraftModel(provider=provider, model_id=model_id, reasoning_effort=effort)
    await asyncio.to_thread(_draft_model_spec, draft)


async def _validate_cwd(target: str, base: str | None) -> str:
    from local_operator.tui.move_targets import expand_path, validate_target

    def resolve() -> str:
        # A relative target with no recorded directory resolves against home,
        # the directory a session with no marker starts in.
        return str(validate_target(expand_path(target, cwd=base or Path.home())))

    try:
        return await asyncio.to_thread(resolve)
    except (OSError, ValueError) as error:
        raise HTTPException(
            422, _wd.move_failed(requested=target, error=str(error)).envelope()
        ) from None


@router.post(
    "/v1/desktop/sessions/{session_id}/settings-intents",
    response_model=CRUDResponse[SettingsIntentReceipt],
    status_code=202,
)
async def submit_settings_intent(session_id: str, body: SettingsIntentRequest, request: Request):
    """Record one pick; never call the runtime. 202 new, 200 replayed ``request_id``.

    ``base`` — the value the session had when the pick was made — is read here
    from the session's own durable records (journal selection, desktop marker),
    because it is what the supersede rule compares against later: a reading equal
    to neither ``base`` nor the pick means someone else changed it.
    """
    applier = ensure_applier(request.app)
    if not applier.store.exists(session_id):
        raise _unknown_session()
    try:
        current = await applier.driver.current(session_id)
    except Exception:  # noqa: BLE001 — no reading: validate what we can, no base
        current = Current()

    value: Any = body.value
    if body.kind == "model":
        await _validate_model(str(value), body.effort)
    elif body.kind == "effort":
        # The ladder that matters is the EFFECTIVE target model's: a model pick
        # still in flight wins over the journal's current one.
        existing = (await applier.snapshot(session_id)).get(session_id, {})
        model_slot = existing.get("model")
        target = (
            str(model_slot["value"])
            if model_slot and model_slot.get("state") in ACTIVE_STATES
            else current.model
        )
        if target:
            await _validate_model(target, str(value))
    elif body.kind == "cwd":
        value = await _validate_cwd(str(value), current.cwd)

    created, slot, slots = await applier.submit(
        session_id,
        kind=body.kind,
        value=value,
        effort=body.effort if body.kind == "model" else None,
        request_id=body.request_id,
        base=current.of(body.kind),
    )
    payload = CRUDResponse(
        status=202 if created else 200,
        message="Settings intent recorded." if created else "Settings intent replayed.",
        result=SettingsIntentReceipt.model_validate({"intent": slot, "intents": slots}),
    )
    return JSONResponse(status_code=202 if created else 200, content=jsonable_encoder(payload))


@router.get(
    "/v1/desktop/settings-intents",
    response_model=CRUDResponse[SettingsIntentIndex],
)
async def list_settings_intents(request: Request, session_id: str | None = Query(default=None)):
    """Every session's non-pruned slots: the baseline the feed frames update."""
    applier = ensure_applier(request.app)
    if session_id is not None and not valid_session_id(session_id):
        raise _unknown_session()
    intents = await applier.snapshot(session_id)
    return CRUDResponse(
        status=200,
        message="Settings intents.",
        result=SettingsIntentIndex.model_validate({"intents": intents}),
    )


@router.delete(
    "/v1/desktop/sessions/{session_id}/settings-intents/{kind}",
    status_code=204,
)
async def dismiss_settings_intent(
    session_id: str,
    kind: SettingsIntentKind,
    request: Request,
    generation: int = Query(..., ge=0),
):
    """Dismiss a failed slot or cancel a pending/waiting one.

    An attempt already in flight is NOT recalled — the runtime may already have
    taken it — but its result is discarded because the slot is gone. 409
    ``generation_moved`` when a newer pick replaced the slot the caller saw, so a
    stale Cancel can never throw away the user's latest choice.
    """
    applier = ensure_applier(request.app)
    if not applier.store.exists(session_id):
        raise _unknown_session()
    outcome = await applier.dismiss(session_id, kind, generation)
    if outcome is None:
        raise _unknown_session()
    if outcome is False:
        raise HTTPException(
            409, _wd.settings_intent_generation_moved().envelope(code="generation_moved")
        )
    return Response(status_code=204)
