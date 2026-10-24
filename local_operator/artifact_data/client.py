"""Wire client for the claude.ai artifact data store (``/api/frame/db/agent``).

The protocol is the one Claude Code's ``ArtifactData`` tool speaks (2.1.304):
one ``POST`` per verb with ``{"slug", "verb", "args"}``. It is not a published
API, so every constant below is the observed contract and the reason it is
needed is written beside it — a field that looks decorative usually is not.

Host: ``api.anthropic.com``, never ``claude.ai``. The claude.ai origin sits
behind a Cloudflare browser challenge that a non-browser client cannot pass.
"""

from __future__ import annotations

import dataclasses
import json
import re
import secrets
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

DB_AGENT_URL = "https://api.anthropic.com/api/frame/db/agent"

#: The OAuth beta flag every subscription-token request to api.anthropic.com
#: carries (the same value ``providers/oauth/anthropic.py`` sends).
OAUTH_BETA = "oauth-2025-04-20"

#: The frame-surface headers. Without ALL of them the endpoint answers 404 as
#: if the route did not exist — measured, not assumed. ``surface``/``platform``
#: are the values Claude Code's CLI sends; the server keys the db verbs on them.
_FRAME_HEADERS = {
    "X-Frame-CP": "go",
    "X-Frame-Surface": "code",
    "X-Frame-Platform": "cli",
}

#: A refusal can come back as its own media type instead of JSON, so the
#: Accept header must name it or the server picks a shape we cannot parse.
_ACCEPT = "application/json, application/vnd.ant.frame-refusal+json"

REQUEST_TIMEOUT_S = 30.0

#: The server's own batch ceiling (Claude Code enforces the same 50 client-side).
MAX_BATCH_WRITES = 50

#: Composed document paths beyond this are refused server-side; checking here
#: turns a vague 400 into a message naming the length.
MAX_PATH_BYTES = 1000

MAX_SEGMENTS = 15

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
# Always applied with ``fullmatch``: ``$`` also matches before a trailing
# newline, so a ``.match`` + ``$`` check would let ``"id\n"`` through.
_SHORT_ID_RE = re.compile(r"[1-9A-HJ-NP-Za-km-z]{22}")
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_SEGMENT_RE = re.compile(r"(?!\.\.?\Z)[A-Za-z0-9_\-.~:@+]{1,200}")
_USER_ID_RE = re.compile(r"u_[A-Za-z0-9_-]{1,64}")
_ARTIFACT_HOSTS = frozenset({"claude.ai", "preview.claude.ai"})

#: ``data/users/me`` is a CLIENT-side alias (the server only knows real ids):
#: Claude Code resolves ``me`` through ``whoami`` and substitutes the id.
ME_PREFIX = ("data", "users", "me")

#: 403 bodies that mean "this token class cannot use db verbs" rather than
#: "no access" — Claude Code keys the same three strings.
_CREDENTIAL_REJECTED_BODIES = frozenset(
    {
        "an OAuth access token is required",
        "a directly presented OAuth access token is required",
        "db verbs require a user-scoped OAuth token, not an org service key",
    }
)

#: Server error codes (both spellings the server has used) to one vocabulary.
_SERVER_CODES: Mapping[str, str] = {
    "not_found": "not_found",
    "not-found": "not_found",
    "invalid_argument": "invalid_argument",
    "invalid-argument": "invalid_argument",
    "quota_or_rate": "quota_or_rate",
    "resource-exhausted": "quota_or_rate",
    "resource_exhausted": "quota_or_rate",
    "busy": "busy",
    "upstream_auth": "upstream_auth",
    "store_unavailable": "store_unavailable",
    "data-unavailable": "store_unavailable",
    "unavailable": "store_unavailable",
    "upstream-error": "store_unavailable",
    "not_declared": "not_declared",
    "too_large": "too_large",
    "quota_exceeded": "quota_exceeded",
    "edit_no_match": "edit_no_match",
    "edit_ambiguous": "edit_ambiguous",
    "edit_field_missing": "edit_field_missing",
    "edit_field_not_string": "edit_field_not_string",
    "version_mismatch": "version_mismatch",
}

LOGIN_REQUIRED_MESSAGE = (
    "The artifact data store needs a claude.ai account: no Anthropic subscription "
    "login is stored (an Anthropic API key cannot reach it). Ask the user to run "
    "`/login anthropic` (Claude Pro/Max), then retry."
)


class ArtifactDataError(Exception):
    """A refused or failed call, carrying the code the tool renders from."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        reason: str | None = None,
        version: int | None = None,
        index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.reason = reason
        self.version = version
        self.index = index


# ---------------------------------------------------------------------------
# URL -> slug
# ---------------------------------------------------------------------------


def decode_short_id(short_id: str) -> str:
    """A 22-char base58 artifact id as the UUID slug the server wants.

    The server refuses the short id itself (``invalid_slug``); the link form
    and the slug are two spellings of one 128-bit number.
    """
    if not _SHORT_ID_RE.fullmatch(short_id):
        raise ArtifactDataError("invalid_url", f"{short_id!r} is not a 22-character artifact id")
    number = 0
    for char in short_id:
        number = number * 58 + _BASE58_ALPHABET.index(char)
    if number >= 1 << 128:
        raise ArtifactDataError("invalid_url", f"{short_id!r} is not a valid artifact id")
    digits = f"{number:032x}"
    return f"{digits[:8]}-{digits[8:12]}-{digits[12:16]}-{digits[16:20]}-{digits[20:]}"


def slug_from_url(url: str) -> str:
    """The artifact slug from a claude.ai artifact link.

    Accepts ``https://claude.ai/artifact/<short id>`` and
    ``https://claude.ai/code/artifact/<uuid>`` (and ``preview.claude.ai``).
    """
    text = (url or "").strip()
    parts = urlsplit(text)
    if parts.scheme != "https" or (parts.hostname or "").lower() not in _ARTIFACT_HOSTS:
        raise ArtifactDataError(
            "invalid_url",
            f"{text!r} is not a claude.ai artifact link "
            "(expected https://claude.ai/artifact/<id>)",
        )
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) == 2 and segments[0] == "artifact":
        return decode_short_id(segments[1])
    if (
        len(segments) == 3
        and segments[:2] == ["code", "artifact"]
        and _UUID_RE.fullmatch(segments[2])
    ):
        return segments[2].lower()
    raise ArtifactDataError(
        "invalid_url",
        f"{text!r} is not a claude.ai artifact link (expected https://claude.ai/artifact/<id>)",
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def collection_segments(collection: str) -> list[str]:
    """Validate a collection path and return its segments (odd count, 1-15)."""
    segments = (collection or "").split("/")
    if not collection or any(not _SEGMENT_RE.fullmatch(segment) for segment in segments):
        raise ArtifactDataError(
            "invalid_path",
            f"collection {collection!r} is not a valid path: '/'-separated segments of "
            "letters, digits and _ - . ~ : @ + (1-200 characters each, not '.' or '..')",
        )
    if len(segments) > MAX_SEGMENTS or len(segments) % 2 == 0:
        raise ArtifactDataError(
            "invalid_path",
            f"collection {collection!r} has {len(segments)} segments; a collection has an odd "
            f"number (1-{MAX_SEGMENTS}) — paths alternate collection/document, so "
            f"'{collection}' names a document (pass its last segment as doc_id)",
        )
    return segments


def check_doc_id(doc_id: str) -> str:
    if not doc_id or not _SEGMENT_RE.fullmatch(doc_id):
        raise ArtifactDataError(
            "invalid_path",
            f"doc_id {doc_id!r} must be one path segment: letters, digits and _ - . ~ : @ + "
            "(1-200 characters, not '.' or '..')",
        )
    return doc_id


def uses_me(segments: Sequence[str]) -> bool:
    return tuple(segments[:3]) == ME_PREFIX


def substitute_me(segments: Sequence[str], user_id: str) -> list[str]:
    """``data/users/me/...`` with ``me`` replaced by the caller's user id."""
    if not uses_me(segments):
        return list(segments)
    return [*ME_PREFIX[:2], user_id, *segments[3:]]


def doc_path(collection_segs: Sequence[str], doc_id: str) -> str:
    path = "/".join([*collection_segs, check_doc_id(doc_id)])
    if len(path.encode("utf-8")) > MAX_PATH_BYTES:
        raise ArtifactDataError(
            "invalid_path",
            f"the composed document path is {len(path.encode('utf-8'))} bytes — "
            f"the limit is {MAX_PATH_BYTES}",
        )
    return path


def check_field_name(field: str) -> str:
    """A ``str_replace`` field: one top-level key, never a dotted path."""
    bad = (
        not field
        or len(field) > 200
        or (field.startswith("__") and field.endswith("__"))
        or any(char in '"\\./[]' for char in field)
        or any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in field)
    )
    if bad:
        raise ArtifactDataError(
            "invalid_argument",
            f"field {field!r} must name one top-level key: no dots, slashes, brackets, quotes, "
            "backslashes or control characters, and not a reserved __name__ key",
        )
    return field


# ---------------------------------------------------------------------------
# Wire args (snake_case tool boundary -> the server's shapes)
# ---------------------------------------------------------------------------


def write_args(
    op: str,
    path: str,
    *,
    data: Mapping[str, Any] | None = None,
    if_version: int | None = None,
    field: str | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    replace_all: bool = False,
) -> dict[str, Any]:
    """The ``args`` of one write verb, exactly as Claude Code shapes them.

    ``set`` is the server's ``replace: true`` write (the doc becomes ``data``);
    ``update`` without it is a merge. ``ifVersion`` is omitted, never sent as
    null, when the caller did not pin. ``str_replace`` carries a fresh ``idem``
    (idempotency key) per call, as Claude Code builds it; a reused key could
    let the server treat a second, intended edit as a replay of the first.
    """
    pin: dict[str, Any] = {"ifVersion": if_version} if if_version is not None else {}
    if op == "delete":
        return {"path": path, **pin}
    if op == "set":
        return {"path": path, "data": dict(data or {}), "replace": True, **pin}
    if op == "update":
        return {"path": path, "data": dict(data or {}), **pin}
    if op == "str_replace":
        args: dict[str, Any] = {
            "path": path,
            "field": field,
            "old_str": old_str,
            "new_str": new_str,
        }
        if replace_all:
            args["replace_all"] = True
        args["idem"] = secrets.token_urlsafe(16)
        return {**args, **pin}
    raise ArtifactDataError("invalid_argument", f"unknown write op {op!r}")


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _status_code(status: int) -> str:
    if status in (403, 404):
        return "not_found"
    if status in (400, 422):
        return "invalid_argument"
    if status == 413:
        return "too_large"
    if status == 429:
        return "quota_or_rate"
    if status in (409, 412):
        return "busy"
    if status == 401:
        return "upstream_auth"
    return "store_unavailable"


def error_from_response(status: int, body_text: str) -> ArtifactDataError:
    """Map a non-2xx answer to one :class:`ArtifactDataError`.

    The server's JSON ``error.code`` wins when it is one we know; otherwise the
    HTTP status decides, the same precedence Claude Code uses.
    """
    stripped = body_text.strip()
    if status == 403 and stripped in _CREDENTIAL_REJECTED_BODIES:
        return ArtifactDataError(
            "credential_rejected", stripped, status=status, reason="custody_403"
        )
    payload: Any = None
    try:
        payload = json.loads(body_text) if body_text else None
    except ValueError:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        raw_code = error.get("code")
        code = _SERVER_CODES.get(raw_code) if isinstance(raw_code, str) else None
        reason = error.get("reason") if isinstance(error.get("reason"), str) else None
        message = error.get("message") if isinstance(error.get("message"), str) else ""
        if code is not None:
            if code == "version_mismatch" and error.get("required") is True:
                reason = "version_required"
            index = error.get("index")
            return ArtifactDataError(
                code,
                message or code,
                status=status,
                reason=reason,
                version=_positive_int(error.get("version")),
                index=index if isinstance(index, int) and not isinstance(index, bool) else None,
            )
        code = _status_code(status)
        return ArtifactDataError(code, message or code, status=status, reason=reason)
    code = _status_code(status)
    return ArtifactDataError(code, stripped[:300] or code, status=status)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bearer:
    """One resolved access token, plus how to get a fresher one after a 401."""

    # repr=False: a Bearer that lands in a log line or traceback must not leak it.
    token: str = dataclasses.field(repr=False)
    refresh: Callable[[], Awaitable[str | None]] | None = None
    #: Which login minted the token (account or credential row). Anything
    #: cached per caller — the ``whoami`` id — keys on it, so a different
    #: login mid-session is never answered with the previous account's id.
    identity: str | None = None


TokenSource = Callable[[], Awaitable[Bearer | None]]


def client_version() -> str:
    """The ``X-Frame-Client-Version`` value: this install's version string."""
    try:
        import importlib.metadata

        return importlib.metadata.version("local-operator")
    except Exception:  # noqa: BLE001 - a header value must never fail a call
        return "0"


async def call_verb(
    slug: str,
    verb: str,
    args: Mapping[str, Any],
    *,
    tokens: TokenSource,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """POST one verb; return the decoded JSON body or raise ArtifactDataError.

    A 401 gets exactly one retry with a force-refreshed token: an access token
    that expired between resolve and send is the common cause, and a second 401
    means the grant itself is no good, which retrying cannot fix.
    """
    bearer = await tokens()
    if bearer is None or not bearer.token:
        raise ArtifactDataError("login_required", LOGIN_REQUIRED_MESSAGE)
    body = {"slug": slug, "verb": verb, "args": dict(args)}

    async def send(http: httpx.AsyncClient, token: str) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-beta": OAUTH_BETA,
            "Content-Type": "application/json",
            "Accept": _ACCEPT,
            "X-Frame-Client-Version": client_version(),
            **_FRAME_HEADERS,
        }
        try:
            return await http.post(
                DB_AGENT_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
            )
        except httpx.TimeoutException as exc:
            raise ArtifactDataError(
                "store_unavailable", f"the artifact data store did not answer in time ({exc})"
            ) from exc
        except httpx.HTTPError as exc:
            raise ArtifactDataError(
                "store_unavailable", f"could not reach the artifact data store ({exc})"
            ) from exc

    async def run(http: httpx.AsyncClient) -> httpx.Response:
        response = await send(http, bearer.token)
        if response.status_code == 401 and bearer.refresh is not None:
            fresh = await bearer.refresh()
            if fresh:
                response = await send(http, fresh)
        return response

    if client is None:
        async with httpx.AsyncClient() as owned:
            response = await run(owned)
    else:
        response = await run(client)

    text = response.text
    if response.status_code >= 400:
        raise error_from_response(response.status_code, text)
    try:
        decoded = json.loads(text) if text else {}
    except ValueError as exc:
        raise ArtifactDataError(
            "store_unavailable",
            "the artifact data store answered with something that is not JSON",
            status=response.status_code,
        ) from exc
    if not isinstance(decoded, dict):
        raise ArtifactDataError(
            "store_unavailable",
            "the artifact data store answered with an unexpected shape",
            status=response.status_code,
        )
    return decoded


def user_id_from_whoami(payload: Mapping[str, Any]) -> str:
    user_id = payload.get("id")
    if not isinstance(user_id, str) or not _USER_ID_RE.fullmatch(user_id):
        raise ArtifactDataError(
            "store_unavailable", "the artifact data store's whoami answer carried no user id"
        )
    return user_id
