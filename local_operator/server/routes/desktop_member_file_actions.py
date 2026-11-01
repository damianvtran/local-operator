"""UNREGISTERED desktop human-file action router. PRE-REGISTRATION GATE, in order.

1. NOT REGISTERED. No app, route table, generated client or IPC entry point
   imports this module. It is source only until every item below is closed.

2. THE PRODUCTION DEPENDENCY IS INTENTIONALLY ABSENT. `member_file_orchestrator`
   raises 503 and has no permissive fallback, so an accidental wiring fails
   closed instead of trusting a caller. It stays that way until a main-only
   channel actually supplies the trusted window/view provider and credentials;
   the current design has no such channel.

3. TRUSTED-APP-FRAME MODEL, WITH STATED LIMITS. A holder of the desktop bearer
   is treated as the trusted desktop client. This is NOT Electron-main
   authentication, NOT a human-gesture attestation, and NOT protection against a
   compromised admitted renderer (one that already reads the same-user token
   file or uses existing local-file IPC). TWO facts must be PROVEN, not assumed,
   before any registration: (a) the DIRECT-LOCAL gate — main decides from the
   ACTUAL attached backend, so a renderer boolean, configured address or
   discovery name is not that proof; and (b) every bearer-bearing generic relay
   denies this namespace AFTER final-URL normalization, including a
   double-decoding intermediary. The relay policy's single-decode assumption is
   only evidenced for direct local Uvicorn; remote or custom topologies are
   unverified and out of scope until measured.

4. NO RENDERER-SUPPLIED AUTHORITY. Request bodies carry only an opaque
   `window_id`, `context`, `file_id` or `operation`. Account, login incarnation,
   tenant, conversation, message, generation, endpoint/purpose, paths and URLs
   are never accepted from a caller; core derives all of them and the hub stays
   authoritative for current human membership.

Cross-process invalidation is asynchronous: main's revoke has delivery latency
and can never recall a request already dispatched. Registry epoch and TTL bound
that race; they do not eliminate it.
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.server.desktop import require_desktop
from local_operator.team_bridge.client import (
    ACCOUNT_MISMATCH_CODE, BridgeRefusal, BridgeSignInRequired, BridgeUnreachable,
)
from local_operator.team_bridge.file_action_orchestrator import HumanFileActionOrchestrator
from local_operator.team_bridge.file_action_registry import ActionUnavailable
from local_operator.team_bridge.types import ATTACHMENT_ID_PATTERN

router = APIRouter(prefix="/v1/desktop/member-file-actions", tags=["Desktop member files"],
                   dependencies=[Depends(require_desktop)])

# Request bodies are intentionally tiny and streamed into a bounded buffer; an
# invalid body must not be echoed by a validation error (it may contain secrets).
_BODY_LIMIT = 4096
_TOKEN_PATTERN = r"^[A-Za-z0-9_-]{43}$"
_WINDOW_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    window_id: str = Field(pattern=_WINDOW_PATTERN)


class _Commission(_Body):
    pass


class _Context(_Body):
    context: str = Field(pattern=_TOKEN_PATTERN)


class _Begin(_Context):
    file_id: str = Field(pattern=ATTACHMENT_ID_PATTERN)


class _Operation(_Body):
    operation: str = Field(pattern=_TOKEN_PATTERN)


def member_file_orchestrator() -> HumanFileActionOrchestrator:
    """Explicit injection point; no provider or permissive fallback ships here."""
    raise HTTPException(status_code=503, detail="Member file actions are unavailable.")


async def _body(request: Request, model: type[_Body]) -> _Body:
    if request.query_params:
        raise HTTPException(status_code=422, detail="Invalid member file request.")
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(status_code=422, detail="Invalid member file request.")
    length = request.headers.get("content-length")
    if length is not None and (not re.fullmatch(r"[0-9]{1,10}", length) or int(length) > _BODY_LIMIT):
        raise HTTPException(status_code=422, detail="Invalid member file request.")
    data = bytearray()
    async for chunk in request.stream():
        if len(chunk) > _BODY_LIMIT - len(data):
            raise HTTPException(status_code=422, detail="Invalid member file request.")
        data.extend(chunk)
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result
    try:
        return model.model_validate(json.loads(data, object_pairs_hook=unique))
    except (UnicodeDecodeError, ValueError, ValidationError):
        raise HTTPException(status_code=422, detail="Invalid member file request.") from None


def _error(exc: Exception) -> JSONResponse:
    # Never echo upstream details: they can include paths, URLs, account data or
    # secrets. Closed categories permit a UI to distinguish reauth and retry.
    code: Literal["unavailable", "reauthentication_required", "retry", "stale_view"]
    if isinstance(exc, ActionUnavailable):
        status, code = 404, "unavailable"
    elif isinstance(exc, BridgeSignInRequired):
        status, code = 401, "reauthentication_required"
    elif isinstance(exc, BridgeUnreachable):
        status, code = 503, "retry"
    elif isinstance(exc, BridgeRefusal) and exc.code in {"file_auth_refresh_required", "file_reauthentication_required"}:
        status, code = 401, "reauthentication_required"
    elif isinstance(exc, BridgeRefusal) and exc.code == ACCOUNT_MISMATCH_CODE:
        # The action was prepared for another login/account/tenant state: the
        # caller must re-read and re-render rather than retry this action.
        status, code = 409, "stale_view"
    elif isinstance(exc, BridgeRefusal):
        status, code = 404, "unavailable"
    else:
        status, code = 503, "retry"
    return JSONResponse({"error": code}, status_code=status, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


Actions = Annotated[HumanFileActionOrchestrator, Depends(member_file_orchestrator)]


@router.post("/commission")
async def commission(request: Request, actions: Actions) -> Response:
    body = await _body(request, _Commission)
    try:
        context = await actions.commission(body.window_id)
        return JSONResponse({"context": context}, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except Exception as exc:
        return _error(exc)


@router.post("/begin")
async def begin(request: Request, actions: Actions) -> Response:
    body = await _body(request, _Begin)
    assert isinstance(body, _Begin)
    try:
        operation = actions.begin(body.window_id, body.context, body.file_id)
        return JSONResponse({"operation": operation}, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except Exception as exc:
        return _error(exc)


@router.post("/bytes")
async def file_bytes(request: Request, actions: Actions) -> Response:
    body = await _body(request, _Operation)
    assert isinstance(body, _Operation)
    try:
        data = await actions.bytes(body.window_id, body.operation)
        return Response(content=data, media_type="application/octet-stream", headers={
            "Content-Disposition": 'attachment; filename="attachment.bin"',
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })
    except Exception as exc:
        return _error(exc)


@router.post("/verify")
async def verify(request: Request, actions: Actions) -> Response:
    body = await _body(request, _Operation)
    assert isinstance(body, _Operation)
    try:
        actions.verify(body.window_id, body.operation)
        return Response(status_code=204, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except Exception as exc:
        return _error(exc)


@router.post("/cancel")
async def cancel(request: Request, actions: Actions) -> Response:
    body = await _body(request, _Operation)
    assert isinstance(body, _Operation)
    try:
        actions.cancel(body.window_id, body.operation)
        return Response(status_code=204, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except Exception as exc:
        return _error(exc)


@router.post("/revoke")
async def revoke(request: Request, actions: Actions) -> Response:
    body = await _body(request, _Context)
    assert isinstance(body, _Context)
    try:
        actions.revoke(body.window_id, body.context)
        return Response(status_code=204, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except Exception as exc:
        return _error(exc)
