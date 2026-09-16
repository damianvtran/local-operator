"""Closed Radient operations using AuthStore, never renderer tokens or refreshes.

The route census comes from the desktop's existing Radient clients. Google
Workspace consent is deliberately NOT here: effective MCP grants own that flow.
"""

from __future__ import annotations

import asyncio
import re
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field, StrictBool, model_validator

from local_operator.providers.registry import get_provider_definition
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.auth import get_desktop_auth
from local_operator.server.routes.desktop_lifecycle import Result
from local_operator.server.routes.desktop_sessions import (
    Input,
    RequestID,
    errors,
    receipts,
    reply,
)
from local_operator.server.utils.desktop_auth import DesktopAuth

router = APIRouter(tags=["Desktop Radient"], dependencies=[Depends(require_desktop)])
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]

#: How many agent ids one `agents.statuses` request may name.
#:
#: The op exists so a page of cards costs ONE renderer round trip, and its
#: fan-out is bounded by this number rather than by whatever a caller sends: the
#: hub's page size is 12, so 32 leaves room for a larger page without letting a
#: single request become an unbounded burst of upstream traffic.
STATUS_BATCH_LIMIT = 32

#: How many of those upstream reads are in flight at once.
#:
#: Each read is a tiny GET, so the cost of opening all of them at once is a
#: burst against a shared API rather than a local one; six keeps the batch
#: comfortably inside a single page's latency budget while staying a polite
#: client.
STATUS_BATCH_CONCURRENCY = 6

#: The same shape `Identifier` enforces, compiled for use on a comma-joined list.
STATUS_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def status_ids(body: RadientRequest) -> list[str]:
    """The agent ids one `agents.statuses` request names.

    Ids arrive as one comma-joined query value rather than as a JSON list,
    because the closed vocabulary's `query` is a flat string map and widening it
    to carry a typed list would widen every op's surface for one op's sake.

    De-duplicated and kept in the caller's order, so a status report is keyed by
    id and the caller never has to pair a positional list with its input.

    Raises:
        ValueError: no ids, more than ``STATUS_BATCH_LIMIT``, or an id that is
            not the shape the rest of this module accepts as an identifier — see
            the call site in ``validate_request`` for why the last one is a
            refusal rather than a skip.
    """
    raw = str(body.query.get("agent_ids", ""))
    ids = list(dict.fromkeys(part for part in (chunk.strip() for chunk in raw.split(",")) if part))
    if not ids:
        raise ValueError("Choose at least one agent")
    if len(ids) > STATUS_BATCH_LIMIT:
        raise ValueError(f"Read at most {STATUS_BATCH_LIMIT} agent statuses at once")
    if any(STATUS_ID.match(agent_id) is None for agent_id in ids):
        raise ValueError("Invalid agent identifier")
    return ids


class RadientRequest(Input):
    operation: Literal[
        "account",
        "prices",
        "credits",
        "usage",
        "provision",
        "application.create",
        "agents.list",
        "agents.get",
        "agents.create",
        "agents.update",
        "agents.delete",
        "agents.like",
        "agents.unlike",
        "agents.liked",
        "agents.like_count",
        "agents.favourite",
        "agents.unfavourite",
        "agents.favourited",
        "agents.favourite_count",
        "agents.download_count",
        # The viewer's own like/favourite state for a whole page of agents.
        #
        # `agents.liked` and `agents.favourited` answer that question for one
        # agent each, and the desktop hub needs it for every card it paints —
        # which is what made a twelve-card page cost twenty-four requests before
        # it finished painting. This op is the same information for a bounded list
        # of ids in one call, so the renderer asks once. It is ADDITIVE: a UI
        # newer than its server gets a 404 for the unknown operation, which the
        # hub reads as "no viewer state to show" rather than as a failure.
        "agents.statuses",
        "comments.list",
        "comments.create",
        "comments.update",
        "comments.delete",
        "account.agents",
    ]
    request_id: RequestID | None = None
    tenant_id: Identifier | None = None
    account_id: Identifier | None = None
    agent_id: Identifier | None = None
    comment_id: Identifier | None = None
    query: dict[str, str | int] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    confirmed: StrictBool = False

    @model_validator(mode="after")
    def validate_request(self):
        method, _ = endpoint(self)
        if method != "GET" and self.request_id is None:
            raise ValueError("Mutations require a request identifier")
        if method == "DELETE" and not self.confirmed:
            raise ValueError("Confirm this removal")
        query_fields = (
            {
                "page",
                "per_page",
                "categories",
                "tags",
                "account_id",
                "tenant_id",
                "name",
                "description",
                "sort",
                "order",
            }
            if self.operation in {"agents.list", "account.agents"}
            else (
                {"agent_ids"}
                if self.operation == "agents.statuses"
                else (
                    {"page", "per_page"}
                    if self.operation == "comments.list"
                    else (
                        {
                            "start_date",
                            "end_date",
                            "application_id",
                            "usage_type",
                            "provider",
                            "rollup",
                        }
                        if self.operation == "usage"
                        else set()
                    )
                )
            )
        )
        if self.query.keys() - query_fields or any(
            len(str(value)) > 1024 for value in self.query.values()
        ):
            raise ValueError("Unsupported query fields")
        for key in {"page", "per_page"} & self.query.keys():
            value = str(self.query[key])
            if not value.isdecimal() or not 1 <= int(value) <= (
                100 if key == "per_page" else 10000
            ):
                raise ValueError("Invalid pagination")
        if self.operation == "usage" and self.query.get("rollup") not in {
            "daily",
            "monthly",
            "annual",
        }:
            raise ValueError("Choose daily, monthly or annual usage")
        if self.operation == "agents.statuses":
            # Validated here rather than at the fan-out, because these ids become
            # upstream path segments: a malformed one must be refused before a
            # request is built from it, and the bound must be the caller's to
            # read in the error rather than a surprise inside the loop.
            status_ids(self)
        allowed: set[str] = set()
        if self.operation in {"agents.create", "agents.update"}:
            allowed = {
                "name",
                "version",
                "description",
                "model",
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "hosting",
                "security_prompt",
                "current_working_directory",
                "stop",
                "tags",
                "categories",
            }
        elif self.operation in {"comments.create", "comments.update"}:
            allowed = {"text"}
        elif self.operation == "application.create":
            allowed = {"name", "description"}
        if self.payload.keys() - allowed:
            raise ValueError("Unsupported payload fields")
        required = (
            {"name", "version"}
            if self.operation == "agents.create"
            else (
                {"name"}
                if self.operation == "application.create"
                else {"text"} if self.operation in {"comments.create", "comments.update"} else set()
            )
        )
        if any(
            not isinstance(self.payload.get(key), str) or not self.payload[key].strip()
            for key in required
        ):
            raise ValueError("Required text fields are missing")
        if len(self.model_dump_json().encode()) > 200000:
            raise ValueError("The request exceeds the size limit")
        return self


def endpoint(body: RadientRequest) -> tuple[str, str]:
    op = body.operation
    if op == "agents.statuses":
        # The one operation whose real work is several upstream calls, so this
        # pair is NOMINAL: it exists because `validate_request` asks for the
        # method to decide whether a request id is required, and the route
        # dispatches this op before it builds anything from a path. The reads it
        # actually makes are built by `agent_statuses`.
        return "GET", "/agents/statuses"
    if op in {"account", "prices", "provision"}:
        return (
            ("POST", "/provision")
            if op == "provision"
            else ("GET", "/me" if op == "account" else "/prices")
        )
    if op in {"credits", "usage", "application.create"}:
        if not body.tenant_id:
            raise ValueError("Choose a tenant")
        tail = {
            "credits": "billing/credits",
            "usage": "usage/rollup",
            "application.create": "applications",
        }[op]
        return (
            "POST" if op == "application.create" else "GET",
            f"/tenants/{body.tenant_id}/{tail}",
        )
    if op == "account.agents":
        if not body.account_id:
            raise ValueError("Choose an account")
        return "GET", f"/accounts/{body.account_id}/agents"
    if op in {"agents.list", "agents.create"}:
        return ("GET" if op == "agents.list" else "POST"), "/agents"
    if not body.agent_id:
        raise ValueError("Choose an agent")
    path = f"/agents/{body.agent_id}"
    if op.startswith("comments."):
        path += "/comments"
        if op in {"comments.update", "comments.delete"}:
            if not body.comment_id:
                raise ValueError("Choose a comment")
            path += "/" + body.comment_id
        return {
            "comments.list": "GET",
            "comments.create": "POST",
            "comments.update": "PATCH",
            "comments.delete": "DELETE",
        }[op], path
    suffixes = {
        "get": ("GET", ""),
        "update": ("PATCH", ""),
        "delete": ("DELETE", ""),
        "like": ("POST", "/like"),
        "unlike": ("DELETE", "/like"),
        "liked": ("GET", "/like"),
        "like_count": ("GET", "/like/count"),
        "favourite": ("POST", "/favourite"),
        "unfavourite": ("DELETE", "/favourite"),
        "favourited": ("GET", "/favourite"),
        "favourite_count": ("GET", "/favourite/count"),
        "download_count": ("GET", "/download/count"),
    }
    method, tail = suffixes[op.removeprefix("agents.")]
    return method, path + tail


def base_url() -> str:
    provider = get_provider_definition("radient")
    assert provider is not None and provider.base_url
    return provider.base_url.rstrip("/")


def public_data(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: public_data(item, secrets)
            for key, item in value.items()
            if key.lower()
            not in {
                "access_token",
                "refresh_token",
                "id_token",
                "api_key",
                "token",
                "password",
                "secret",
                "authorization",
                "client_secret",
            }
        }
    if isinstance(value, list):
        return [public_data(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
    return value


async def agent_statuses(body: RadientRequest, auth: DesktopAuth) -> dict[str, Any]:
    """The viewer's like and favourite state for a bounded list of agents.

    THE ONE OP THAT MAKES SEVERAL UPSTREAM CALLS, and the reason is the shape of
    what it is asked: `GET /agents/{id}/like` and `/agents/{id}/favourite` answer
    for ONE agent each, and the desktop hub needs the answer for every card it
    paints. Served one id at a time from the renderer, a twelve-card page costs
    twenty-four round trips through this proxy and two more waves of them (after
    first paint, and again on every window focus). Served here, it costs one,
    with the fan-out inside a process that already holds the credential and a
    connection pool to Radient.

    What keeps it from being a general-purpose proxy: the vocabulary is still
    closed, the ids are bounded by ``STATUS_BATCH_LIMIT``, the only upstream
    paths reachable are the two status reads below, and nothing a caller sends
    is interpolated into a URL without matching ``STATUS_ID``.

    The result is keyed by agent id, and every id the caller named is present.

    Raises:
        HTTPException: 409 when no Radient credential is stored, the upstream
            status (401/403/429) when Radient refuses the whole batch, and 502
            when the upstream could not be reached or answered invalidly.
    """
    ids = status_ids(body)
    access = await auth.store.get_oauth_access("radient")
    if access is None:
        raise HTTPException(409, "Sign in to Radient to access your account")
    token = access.access_token
    base = base_url()
    headers = {"Authorization": "Bearer " + token}
    statuses: dict[str, dict[str, bool]] = {
        agent_id: {"liked": False, "favourited": False} for agent_id in ids
    }
    limiter = asyncio.Semaphore(STATUS_BATCH_CONCURRENCY)

    async def read(client: httpx.AsyncClient, agent_id: str, suffix: str, key: str) -> None:
        async with limiter:
            response = await client.get(f"{base}/agents/{agent_id}/{suffix}", headers=headers)
        if response.status_code in {401, 403, 429}:
            # The credential was refused, or this account is being asked to slow
            # down. That is the batch's answer rather than one id's, and the
            # caller's move is to sign in again or to wait - not to render a page
            # of "nothing known".
            raise HTTPException(response.status_code, "Radient could not complete this operation")
        if response.is_redirect:
            # Same classification as the single-op path above: a redirect says the
            # base address is wrong rather than that this agent has no relation,
            # so it is the batch's failure and not an absent answer.
            raise HTTPException(502, "Radient returned an unexpected redirect")
        if response.status_code != 200:
            # A per-agent refusal (an agent delisted while the page was open, a
            # private one) leaves that id at "not liked, not favourited". That is
            # the state a client that knows nothing renders, and the one state
            # whose toggle is still correct: the like and favourite endpoints
            # answer an already-present relation with `already_liked` /
            # `already_favourited` rather than with a conflict.
            return
        # 200 with NO document is this endpoint's "no relation": the handler
        # answers empty content when the account holds no like, so the presence
        # of a body is the flag rather than any field inside it.
        statuses[agent_id][key] = bool(response.content.strip())

    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        outcomes = await asyncio.gather(
            *(
                read(client, agent_id, suffix, key)
                for agent_id in ids
                for suffix, key in (("like", "liked"), ("favourite", "favourited"))
            ),
            # `return_exceptions` so one refused id cannot leave the rest of the
            # batch running unattended: the batch is reported as a whole.
            return_exceptions=True,
        )
    for outcome in outcomes:
        if isinstance(outcome, HTTPException):
            raise outcome
        if isinstance(outcome, BaseException):
            raise HTTPException(
                502, "Radient is unavailable or returned an invalid response"
            ) from None
    return {
        "data": {
            "msg": "Agent statuses read",
            "result": {"statuses": statuses},
        }
    }


@router.post("/v1/desktop/radient", response_model=CRUDResponse[Result])
async def radient(
    body: RadientRequest, request: Request, auth: DesktopAuth = Depends(get_desktop_auth)
):
    if body.operation == "agents.statuses":
        # Dispatched before `endpoint` builds a path, because this op's real work
        # is the bounded fan-out above rather than one upstream call. It is a
        # read, so it takes no receipt.
        async with errors():
            return reply(await agent_statuses(body, auth))

    method, path = endpoint(body)

    async def execute():
        access = await auth.store.get_oauth_access("radient")
        if access is None and body.operation != "prices":
            raise HTTPException(409, "Sign in to Radient to access your account")
        # AuthStore performs the only refresh. Never retry a mutation on a 401:
        # upstream may have accepted it before the connection failed.
        token = access.access_token if access else ""
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                async with client.stream(
                    method,
                    base_url() + path,
                    params=body.query,
                    json=body.payload if method in {"POST", "PATCH"} else None,
                    headers={"Authorization": "Bearer " + token} if token else {},
                ) as response:
                    if response.is_redirect:
                        raise HTTPException(502, "Radient returned an unexpected redirect")
                    if response.status_code >= 400:
                        raise HTTPException(
                            (
                                response.status_code
                                if response.status_code in {400, 401, 403, 404, 409, 422, 429}
                                else 502
                            ),
                            "Radient could not complete this operation",
                        )
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > 2_000_000:
                            raise HTTPException(502, "Radient returned too much data")
                    import json

                    value = json.loads(content) if content else {}
        except (httpx.HTTPError, ValueError):
            raise HTTPException(
                502, "Radient is unavailable or returned an invalid response"
            ) from None
        stored_key = ""
        if body.operation in {"provision", "application.create"}:
            result = value.get("result", {})
            stored_key = result.get("api_key", "")
            if not isinstance(stored_key, str) or not stored_key:
                raise HTTPException(502, "Radient did not return an application credential")
            auth.store.upsert_credential(
                "radient", {"type": "api_key", "source": "login", "key": stored_key}
            )
        return {"data": public_data(value, [token, stored_key])}

    async with errors():
        result = (
            await execute()
            if method == "GET"
            else await receipts(request).run(
                "radient:" + str(body.request_id), body.model_dump(), execute
            )
        )
        return reply(result)
