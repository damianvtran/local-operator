"""Relay web's community door: the daemon's ``/api/community/*`` routes.

RFC v2.4 §7.2: relay web reaches the community through the mobile daemon,
acting as this computer's signed-in Radient account, and "the daemon exposes an
allowlist of member operations only (no staff operations over relay)". The
allowlist is :data:`~local_operator.community.member_ops.MEMBER_OPERATIONS` —
the same table the desktop door registers — so a staff/moderation operation
cannot be reached here without first being added there, where the RFC rule is
written beside it. (The mobile APP does not come through here: it talks to the
hub directly with its own sign-in.)

AUTH is the daemon's existing relay gate (``gate`` from ``build_app``: the
paired-device cookie/bearer plus the cross-origin mutation guard), checked
FIRST on every route so an unauthenticated caller learns nothing — not even
whether the community is switched on.

DARK: ``404 {"error": "community_disabled", "code": "community_disabled"}``
unless ``LOCAL_OPERATOR_COMMUNITY`` is on in the DAEMON's environment.

BODIES. JSON only (images as base64; see ``member_ops`` for why not multipart),
read through a bounded stream: :data:`MAX_BODY_BYTES` is the relay's own 10 MiB
request ceiling (RFC §7.2: "images ≤ 5 MiB, under the relay's 10 MiB body
ceiling"), so the daemon never buffers more than the relay could deliver. A
body that fails validation answers a FIXED ``invalid_request`` — pydantic's
error text quotes the input, and relay responses must never echo a draft.
Nothing here logs a body.

Refusal bodies are the daemon's ``{"error", "code"}`` shape with the community
facts beside them (``category``/``reference``, ``retry_after``/``scope``,
``guidelines_version``, ``idempotency_key``); ``error`` repeats the code —
relay web localizes by code, and the hub's English ``message`` is never echoed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from local_operator.community import community_enabled
from local_operator.community.member_ops import MEMBER_OPERATIONS, MemberOperation

PREFIX = "/api/community"

#: The relay's request-body ceiling (RFC §7.2), applied at the daemon too.
MAX_BODY_BYTES = 10 * 1024 * 1024

_ACCOUNT_PARAM = "expected_account_id"


def _json(
    body: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None
) -> Response:
    response = JSONResponse(body, status_code=status, headers=headers)
    # Member reads carry other people's posts; nothing here may be cached by
    # the relay, a proxy or the browser (the hub's own no-store, preserved).
    response.headers["Cache-Control"] = "no-store"
    return response


def _code(code: str, status: int, **facts: Any) -> Response:
    return _json({**facts, "error": code, "code": code}, status)


def _client(expected_account_id: str | None) -> Any:
    """The shared client against this computer's config root. A test seam."""

    from local_operator.community.client import CommunityClient
    from local_operator.paths import config_dir

    return CommunityClient(config_dir(), expected_account_id=expected_account_id)


async def _bounded_body(request: Request) -> bytes | None:
    """The request body, or None once it passes :data:`MAX_BODY_BYTES`."""

    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _validate(model: type[BaseModel], value: Any) -> BaseModel | None:
    try:
        return model.model_validate(value)
    except ValidationError:
        return None


def _handler(op: MemberOperation, gate: Callable[[Request], Response | None]):
    async def handle(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        if not community_enabled():
            return _code("community_disabled", 404)
        from local_operator.community.errors import CommunityError
        from local_operator.community.member_ops import refusal, run

        params = dict(request.query_params)
        expected = params.pop(_ACCOUNT_PARAM, None) or None
        if expected is not None and len(expected) > 128:
            return _code("invalid_request", 422)
        query = None
        if op.query is not None:
            query = _validate(op.query, params)
            if query is None:
                return _code("invalid_request", 422)
        body = None
        if op.body is not None:
            raw = await _bounded_body(request)
            if raw is None:
                return _code("payload_too_large", 413)
            try:
                decoded = json.loads(raw) if raw else None
            except (ValueError, UnicodeDecodeError):
                return _code("invalid_request", 422)
            body = _validate(op.body, decoded)
            if body is None:
                return _code("invalid_request", 422)
        try:
            status, result = await run(
                op.name,
                _client(expected),
                {k: str(v) for k, v in request.path_params.items()},
                query=query,
                body=body,
            )
        except CommunityError as error:
            code_status, wire = refusal(error)
            headers = None
            retry_after = wire.get("retry_after")
            if code_status == 429 and isinstance(retry_after, int):
                headers = {"Retry-After": str(retry_after)}
            return _json({**wire, "error": wire["code"]}, code_status, headers)
        return _json(result or {}, status)

    handle.__name__ = f"api_community_{op.name}"
    return handle


def _starlette_path(path: str) -> str:
    """``{name}`` → ``{name:str}``: the daemon's route-template spelling."""

    import re

    return re.sub(r"\{(\w+)\}", r"{\1:str}", path)


def community_routes(gate: Callable[[Request], Response | None]) -> list[Route]:
    """Every allowlisted member operation as a daemon route, behind ``gate``."""

    return [
        Route(PREFIX + _starlette_path(op.path), _handler(op, gate), methods=[op.method])
        for op in MEMBER_OPERATIONS
    ]


__all__ = ["MAX_BODY_BYTES", "PREFIX", "community_routes"]
