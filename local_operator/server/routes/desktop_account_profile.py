"""Closed human-profile relay; person OAuth never crosses into the renderer.

No Bridge membership is required: this edits the JWT subject, not an agent or
organization. A scope receipt binds an editor to the login that read its draft.
It does not replace desktop authentication and never authorizes another account.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from local_operator.config import ConfigManager
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop

router = APIRouter(tags=["Desktop account profile"], dependencies=[Depends(require_desktop)])


class ProfilePatch(BaseModel):
    """Only supported personal fields; absent differs from explicitly cleared."""

    model_config = ConfigDict(extra="forbid")
    display_name: Annotated[StrictStr, Field(max_length=64)] | None = None
    title: Annotated[StrictStr, Field(max_length=80)] | None = None
    description: Annotated[StrictStr, Field(max_length=500)] | None = None
    timezone: Annotated[StrictStr, Field(max_length=64)] | None = None


def _receipt(request: Request, *, write: bool) -> tuple[str, str | None]:
    keys = {"expected_account_id", "scope"} if write else {"expected_account_id"}
    if request.method == "DELETE":
        keys |= {"mode", "confirmed"}
    query = request.query_params
    if set(query) != keys or any(len(query.getlist(key)) != 1 for key in keys):
        raise HTTPException(422, {"code": "invalid_request"})
    from local_operator.team_bridge.types import is_account_id

    account_id = query["expected_account_id"]
    scope = query.get("scope")
    # The transport uses canonical account ids too. Admit with its ONE grammar
    # here so malformed input gets 422 before client construction or OAuth.
    if not is_account_id(account_id) or (
        write and (scope is None or re.fullmatch(r"[a-f0-9]{64}", scope) is None)
    ):
        raise HTTPException(422, {"code": "invalid_request"})
    return account_id, scope


async def _bounded_body(request: Request, limit: int) -> bytes:
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise HTTPException(413, {"code": "profile_body_too_large"})
        data.extend(chunk)
    return bytes(data)


async def _reply(
    config: ConfigManager,
    operation: Literal["read", "update", "upload", "remove"],
    account_id: str,
    scope: str | None,
    **kwargs: Any,
):
    # Lazy import: the HTTP client stack must not enter every server boot.
    from local_operator.team_bridge.client import (
        BridgeRefusal,
        BridgeSignInRequired,
        TeamBridgeClient,
    )

    try:
        result = await TeamBridgeClient(config.config_dir).account_profile_request(
            operation, expected_account_id=account_id, expected_scope=scope, **kwargs
        )
    except BridgeSignInRequired as error:
        raise HTTPException(409, {"code": "radient_no_credential"}) from error
    except BridgeRefusal as error:
        raise HTTPException(error.http_status or 502, {"code": error.code}) from error
    return JSONResponse({"result": result}, headers={"Cache-Control": "private, no-store"})


@router.get("/v1/desktop/account/profile")
async def read_profile(request: Request, config: ConfigManager = Depends(get_config_manager)):
    account_id, scope = _receipt(request, write=False)
    return await _reply(config, "read", account_id, scope)


@router.patch("/v1/desktop/account/profile")
async def update_profile(request: Request, config: ConfigManager = Depends(get_config_manager)):
    account_id, scope = _receipt(request, write=True)
    from pydantic import ValidationError

    try:
        patch = ProfilePatch.model_validate_json(await _bounded_body(request, 16 * 1024))
    except ValidationError as error:
        raise HTTPException(422, {"code": "profile_invalid"}) from error
    fields = patch.model_dump(exclude_unset=True)
    if not fields or any(value is None for value in fields.values()):
        raise HTTPException(422, {"code": "profile_invalid"})
    # Timezone is an IANA selection, not a free-form personal field. The
    # upstream repeats validation and owns Unicode normalization of all fields.
    if fields.get("timezone"):
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(fields["timezone"])
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise HTTPException(422, {"code": "profile_invalid"}) from error
    if "description" in fields and len(fields["description"].splitlines()) > 8:
        raise HTTPException(422, {"code": "profile_invalid"})
    return await _reply(config, "update", account_id, scope, fields=fields)


@router.put("/v1/desktop/account/avatar")
async def upload_avatar(request: Request, config: ConfigManager = Depends(get_config_manager)):
    account_id, scope = _receipt(request, write=True)
    image = await _bounded_body(request, 1024 * 1024)
    # The upstream fully decodes and re-encodes pixels, rejecting bombs/trailing
    # data. This relay never trusts a filename or a renderer-declared MIME type.
    if not image:
        raise HTTPException(422, {"code": "avatar_invalid"})
    return await _reply(config, "upload", account_id, scope, image=image)


@router.delete("/v1/desktop/account/avatar")
async def remove_avatar(request: Request, config: ConfigManager = Depends(get_config_manager)):
    account_id, scope = _receipt(request, write=True)
    mode = request.query_params["mode"]
    if mode not in {"uploaded", "provider"} or request.query_params["confirmed"] != "true":
        raise HTTPException(422, {"code": "invalid_request"})
    return await _reply(config, "remove", account_id, scope, mode=mode)
