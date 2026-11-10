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
    RequestReplaced,
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


def _generation_moved() -> HTTPException:
    return HTTPException(
        409, _wd.settings_intent_generation_moved().envelope(code="generation_moved")
    )


async def _validate_model(provider_model: str, effort: str | None) -> None:
    """Refuse a model (and paired effort) the LOCAL catalogue says cannot run.

    The same boundary check a new draft pays (``_draft_model_spec``: provider
    registry, offline catalogue, the model's effort ladder), so the codes and
    sentences are the existing ones: ``ModelSelectionRefused.code`` and
    ``effort_unsupported``. A catalogue that cannot be enumerated offline does
    not refuse — "we have not looked" is not "it does not exist". Off the event
    loop because it reads catalogue caches.
    """
    from local_operator.model.configure import (
        build_model_spec,
        model_unknown_refusal,
        resolve_model_info_paint,
    )
    from local_operator.model.discovery import (
        cached_available_models,
        offered_model_ids,
    )
    from local_operator.providers.registry import (
        get_provider_definition,
        is_decision_only,
        is_media_only,
        is_speech_only,
    )

    def validate() -> None:
        provider, sep, model_id = provider_model.partition("/")
        if (
            not sep
            or not provider
            or not model_id
            or get_provider_definition(provider) is None
            or is_decision_only(provider)
            or is_media_only(provider)
            or is_speech_only(provider)
        ):
            raise HTTPException(
                422, _wd.desktop_refusal(error=provider_model).envelope(code="model_unknown")
            )
        offered = offered_model_ids(provider)
        if offered is not None and model_id not in offered:
            refused = model_unknown_refusal(provider, model_id)
            raise HTTPException(
                422, _wd.desktop_refusal(error=refused.message).envelope(code="model_unknown")
            )
        # Do NOT call validate_model_selection here: its build_model_spec path
        # enriches unlisted metadata over HTTP (up to 13 s). This admission is
        # cache-only; a missing row is not evidence of an invalid pick. The
        # cached listing's stated ladder wins over static metadata, just as it
        # does for the picker that offered this effort level.
        rows, _ = cached_available_models(provider)
        row = next((r for r in rows if r.id == model_id), None)
        ladder = row.reasoning_efforts if row is not None else None
        if ladder is None:
            info, _ = resolve_model_info_paint(provider, model_id)
            ladder = build_model_spec(provider, model_id, info=info).reasoning_efforts
        if effort and effort not in ladder:
            message = (
                _wd.effort_not_in_levels(model=model_id, levels=", ".join(ladder), effort=effort)
                if ladder
                else _wd.effort_levels_missing(model=model_id)
            )
            raise HTTPException(422, message.envelope(code="effort_unsupported"))

    await asyncio.to_thread(validate)


async def _validate_fast(provider_model: str) -> None:
    """Refuse ``fast: true`` for a target model whose route cannot serve fast.

    WHY AT POST (QA Q1): the runtime answers an unsupported ``/fast on`` with an
    ``info`` notice, not an error, so without this the pick sat in "applies at
    the next step" for the whole retry budget and then failed as "owner
    unavailable" — wrong on both counts. The authority is the one the draft
    boundary (``desktop_sessions._draft_model_spec``) and the runtime's own
    ``/fast`` use: ``ModelSpec.supports_fast_mode``, seeded by
    ``build_model_spec`` from the static route+model table in ``model.speed``
    (no network: ``resolve_model_info_paint`` is the cache-only metadata read).
    The code and sentence are that boundary's existing ``fast_mode_unsupported``
    — no new copy. A race this cannot see (a model pick that lands between this
    check and the fast apply) is caught by the applier classifying the runtime's
    refusal as terminal, see ``settings_intents._slash_outcome``.
    """
    from local_operator.model.configure import (
        build_model_spec,
        resolve_model_info_paint,
    )

    def validate() -> None:
        provider, sep, model_id = provider_model.partition("/")
        if not sep or not provider or not model_id:
            return  # no identity to judge; the runtime remains the authority
        info, _ = resolve_model_info_paint(provider, model_id)
        if not build_model_spec(provider, model_id, info=info).supports_fast_mode:
            raise HTTPException(
                422,
                _wd.fast_mode_unsupported(model=model_id).envelope(code="fast_mode_unsupported"),
            )

    await asyncio.to_thread(validate)


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
    existing = (await applier.snapshot(session_id)).get(session_id, {})
    replay = existing.get(body.kind)
    if replay is None or replay["request_id"] != body.request_id:
        # Review m3: a request_id this session already accepted whose slot a
        # NEWER pick has since replaced must not be re-recorded — an HTTP-level
        # retry of the old POST would otherwise take a new generation and
        # re-apply the stale value over the user's latest choice.
        if await applier.seen_request(session_id, body.request_id):
            raise _generation_moved()
    if replay and replay["request_id"] == body.request_id:
        # Replay precedes catalogue/target validation: a cached row can disappear
        # after the original durable admission without invalidating its receipt.
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder(
                CRUDResponse(
                    status=200,
                    message="Desktop session result.",
                    result=SettingsIntentReceipt.model_validate(
                        {"intent": replay, "intents": existing}
                    ),
                )
            ),
        )
    try:
        current = await applier.driver.current(session_id)
    except Exception:  # noqa: BLE001 — no reading: validate what we can, no base
        current = Current()

    value: Any = body.value
    # The model that matters for effort and fast is the EFFECTIVE target: a
    # model pick still in flight wins over the journal's current one.
    model_slot = existing.get("model")
    target = (
        str(model_slot["value"])
        if model_slot and model_slot.get("state") in ACTIVE_STATES
        else current.model
    )
    if body.kind == "model":
        await _validate_model(str(value), body.effort)
    elif body.kind == "effort":
        if target:
            await _validate_model(target, str(value))
    elif body.kind == "fast" and value is True:
        # Only ON is gated: turning fast off is always possible.
        if target:
            await _validate_fast(target)
    elif body.kind == "cwd":
        value = await _validate_cwd(str(value), current.cwd)

    try:
        created, slot, slots = await applier.submit(
            session_id,
            kind=body.kind,
            value=value,
            effort=body.effort if body.kind == "model" else None,
            request_id=body.request_id,
            base=current.of(body.kind),
        )
    except RequestReplaced:
        # The same check as above, re-run under the session lock: a newer pick
        # may have replaced this request's slot while we validated.
        raise _generation_moved() from None
    payload = CRUDResponse(
        status=202 if created else 200,
        message="Desktop session result.",
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
        message="Desktop session result.",
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
    # ``None`` — the session exists but has no slot of this kind (already
    # pruned, or dismissed by another window). The caller's goal, "no slot",
    # already holds, so this is the idempotent 204 (QA Q2): answering
    # ``session_unknown`` made a Dismiss/Cancel after a prune race claim the
    # whole conversation was gone. Generation protection is unaffected: a slot
    # that EXISTS under a newer generation is still the 409 below.
    if outcome is False:
        raise _generation_moved()
    return Response(status_code=204)
