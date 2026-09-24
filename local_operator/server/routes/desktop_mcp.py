"""Sessionless desktop MCP routes: ``/v1/desktop/mcp``.

The session-independent twin of ``/v1/desktop/sessions/{id}/mcp`` (which stays,
unchanged in shape, for desktop builds that predate ``features.mcp_catalog``).
These answer with no conversation and no model configured — see
:mod:`local_operator.server.mcp_host` for why and how.

Every successful POST answers with the whole catalog document so a client
repaints from the response instead of re-reading (the old re-read raced the very
reconnect it had just caused), plus ``operation`` — the op this request started
or named. A REFUSAL is the exception and cannot be otherwise: ``409 {code,
message}`` carries a bounded code and fixed copy, never exception text (config
and connect errors can quote credentials), and no document, so a client that
needs rows after a refusal re-reads ``GET /v1/desktop/mcp``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from local_operator.mcp.catalog import LiveFacts, live_facts_from_snapshot
from local_operator.mcp.config import MCPConfigWriteError
from local_operator.mcp.credentials import MCPCredentials
from local_operator.mcp.desktop import (
    MCPControl,
    MCPRefusal,
    refusal_code,
    refusal_detail,
)
from local_operator.server.desktop import require_desktop
from local_operator.server.mcp_host import mcp_host, resolve_cwd
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.retire import RETIRING_MESSAGE, DaemonRetiring, retiring
from local_operator.server.routes.desktop_sessions import host, reply

router = APIRouter(tags=["Desktop MCP"], dependencies=[Depends(require_desktop)])

#: How long a list waits for a named session's live overlay before answering
#: from config alone. Short on purpose: the overlay is an enrichment, and a
#: silent owner must degrade to the durable answer, never to a 503 (the class
#: of failure ``desktop_lifecycle``'s cold MCP read already fought).
LIVE_OVERLAY_TIMEOUT_S = 2.0


class Result(BaseModel):
    data: dict[str, Any]
    replayed: bool = False


class CatalogControl(MCPControl):
    """``MCPControl`` plus the folder the control applies to."""

    cwd: str | None = Field(default=None, max_length=4096)


class CatalogCredentials(MCPCredentials):
    """``MCPCredentials`` plus the folder whose config names the server.

    ``header`` is the ``add_key`` action's one extra field: the HTTP header the
    key travels in, for a remote server that declares no ``${ID}`` yet. With it,
    ``values`` must hold exactly one id; that id is bound into the server's
    config as ``headers[header] = "${ID}"`` and the value stored. Without it the
    request is the ``set_key`` write it always was.
    """

    cwd: str | None = Field(default=None, max_length=4096)
    header: str | None = Field(default=None, min_length=1, max_length=128)


def _cwd_or_422(raw: str | None) -> str:
    try:
        return resolve_cwd(raw)
    except ValueError as error:
        raise HTTPException(422, {"code": "invalid_cwd", "message": str(error)}) from None


def _refused(error: Exception) -> HTTPException:
    return HTTPException(409, refusal_detail(refusal_code(error)))


def _refuse_if_retiring(request: Request) -> None:
    """A latched daemon starts no new work, the same rule the session door keeps."""
    if retiring(request.app):
        error = DaemonRetiring(RETIRING_MESSAGE)
        raise HTTPException(503, {"code": error.code, "message": str(error)})


def _same_folder(left: str, right: str) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return False


async def _live_overlay(request: Request, session_id: str, cwd: str) -> dict[str, LiveFacts] | None:
    """One ALREADY-WARM session's live statuses for ``cwd``, or ``None``.

    Never starts anything: a cold facade (no runtime attached) answers ``None``
    without ``bind_runtime``. A session in another folder answers ``None`` too
    — its "connected" is about a different config set. Every failure, including
    the bound, degrades to the config answer.
    """

    async def read() -> dict[str, LiveFacts] | None:
        async with host(request).session(session_id, read=True) as bridge:
            remote = bridge.remote
            if remote is None or remote.is_cold:
                return None
            if not _same_folder(str(remote.frontend_state.cwd or ""), cwd):
                return None
            result = await remote.route_shared_slash(
                "desktop_mcp", MCPControl(action="list").model_dump_json()
            )
            data = result.get("data") if isinstance(result, dict) else None
            if result.get("kind") == "error" or not isinstance(data, dict):
                return None
            return live_facts_from_snapshot(data.get("servers"))

    try:
        return await asyncio.wait_for(read(), LIVE_OVERLAY_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — the overlay is optional by contract
        return None


@router.get("/v1/desktop/mcp", response_model=CRUDResponse[Result])
async def mcp_catalog(
    request: Request,
    cwd: str | None = Query(default=None, max_length=4096),
    session_id: str | None = Query(default=None, max_length=64),
):
    """The MCP catalog for ``cwd`` (home when omitted), with no session needed."""
    folder = _cwd_or_422(cwd)
    live = await _live_overlay(request, session_id, folder) if session_id else None
    document = await mcp_host(request.app.state).catalog(folder, live=live, session_id=session_id)
    return reply({"data": document})


@router.post("/v1/desktop/mcp", response_model=CRUDResponse[Result])
async def mcp_catalog_control(body: CatalogControl, request: Request):
    """add / remove / test / login / reauth / logout / status / cancel.

    Answers the catalog document (see the module docstring for the 409
    exception) around the operation this request started or named.
    """
    folder = _cwd_or_422(body.cwd)
    if body.action not in ("status", "cancel", "list"):
        _refuse_if_retiring(request)
    control = MCPControl.model_validate(body.model_dump(exclude={"cwd"}))
    host_ = mcp_host(request.app.state)
    try:
        operation = await host_.execute(control, folder)
    except (MCPRefusal, MCPConfigWriteError) as error:
        raise _refused(error) from None
    document = await host_.catalog(folder)
    return reply({"data": {**document, "operation": operation}})


@router.post("/v1/desktop/mcp/credentials", response_model=CRUDResponse[Result])
async def mcp_catalog_credentials(body: CatalogCredentials, request: Request):
    """Write ``${NAME}`` values into the encrypted store, with no session.

    Same off-record path as the session route: values never enter a slash
    argument, receipt, transcript or child environment.
    """
    folder = _cwd_or_422(body.cwd)
    _refuse_if_retiring(request)
    host_ = mcp_host(request.app.state)
    credentials = MCPCredentials.model_validate(
        {
            "name": body.name,
            "values": body.values,
            "confirmed_replace": body.confirmed_replace,
        }
    )
    result = await host_.store_credentials(credentials, folder, header=body.header)
    return reply({"data": {**result, "catalog": await host_.catalog(folder)}})
