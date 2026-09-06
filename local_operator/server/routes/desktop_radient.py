"""Closed Radient operations using AuthStore, never renderer tokens or refreshes.

The route census comes from the desktop's existing Radient clients. Google
Workspace consent is deliberately NOT here: effective MCP grants own that flow.
"""

from __future__ import annotations

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
                {"page", "per_page"}
                if self.operation == "comments.list"
                else (
                    {"start_date", "end_date", "application_id", "usage_type", "provider", "rollup"}
                    if self.operation == "usage"
                    else set()
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


@router.post("/v1/desktop/radient", response_model=CRUDResponse[Result])
async def radient(
    body: RadientRequest, request: Request, auth: DesktopAuth = Depends(get_desktop_auth)
):
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
