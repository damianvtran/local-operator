"""The community member client: pinned Radient OAuth, ``/community/v1``.

CONTRACT. ``docs/community-api.yaml`` on agent-server (branch
feat/community-registry). Only MEMBER operations are implemented here — list,
bootstrap, join, leave, channel directory, history, submit, submission
status/cancel, edit, delete, reactions and the events long-poll. Staff and
moderation operations are deliberately absent: no local surface may reach them
(RFC §7.2), and a method that does not exist cannot be proxied by mistake.

ROOT-RELATIVE BASE. The community is served by its own deployment of the
agent-server binary, path-routed on the same host as the private API, at
``/community/v1`` — NOT under the private ``/v1``. The host comes from the same
Radient host configuration Team Bridge reads (``resolve_radient_api_base_url``:
``radient_base_url`` / ``RADIENT_API_BASE_URL`` / the default), and only its
origin is kept: ``https://api.radienthq.com/v1`` → ``https://api.radienthq.com
/community/v1``.

CREDENTIAL PIN, the Team Bridge client's rule adopted wholesale
(``team_bridge/client.py`` ``_pinned_row`` / ``current_access`` /
``_fresh_pinned_access`` / ``_authorized``):

* EXACTLY ONE Radient OAuth row (or the explicit ``credential_id``), resolved
  once and then pinned for this client's lifetime, so a concurrent re-login
  cannot redirect an in-flight operation to a different row;
* the row's ``account_id`` is the owner binding — token and identity come from
  ONE refresh snapshot, and a snapshot carrying another (or no) account is
  refused before anything is sent;
* one FORCED refresh on a 401, then one resend; a second 401 is the sign-in
  state, never a loop;
* an ``expected_account_id`` captured by a surface (the account the user saw
  when they pressed Join or Send) must equal the pinned account, checked
  BEFORE any request leaves: otherwise ``account_mismatch``;
* a non-canonical host needs ``RADIENT_ORG_ALLOW_NONCANONICAL_BASE``, and a
  base carrying URL credentials is refused outright.

NO SHARED STATE WITH PRIVATE TEAM BRIDGE (RFC §7.2). No pooled hub client
(each request opens and closes its own ``httpx.AsyncClient``, cookieless by
construction), no caches, no cursors. The credential store is the one thing
read in common: it is the same signed-in person.

BODIES ARE NEVER LOGGED (RFC §6.5). This module logs nothing but the operation
name and status at DEBUG; request bodies, response bodies and the hub's error
``message`` never reach a logger, an exception string or a local response.
Uploads are streamed from memory (``multipart.StreamedMultipart``); nothing is
written to disk.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import secrets
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from local_operator.community.errors import (
    CommunityAccountMismatch,
    CommunityBadRequest,
    CommunityConflict,
    CommunityDestinationRefused,
    CommunityError,
    CommunityForbidden,
    CommunityGuidelinesChanged,
    CommunityImagesUnavailable,
    CommunityInvalidRequest,
    CommunityInvalidResponse,
    CommunityModerationUnavailable,
    CommunityNotFound,
    CommunityPayloadTooLarge,
    CommunityRateLimited,
    CommunityRefused,
    CommunitySignInRequired,
    CommunityUnavailable,
    CommunityUnreachable,
)
from local_operator.community.limits import (
    MAX_CURSOR_CHARS,
    MAX_EMOJI_CHARS,
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    MAX_TEXT_CHARS,
)
from local_operator.community.models import (
    COMMUNITY_ID_PATTERN,
    IDEMPOTENCY_KEY_PATTERN,
    OPAQUE_ID_PATTERN,
    Bootstrap,
    ChannelPage,
    CommunityList,
    DirectoryFilter,
    EventBatch,
    MessagePage,
    RateLimitScope,
    ReactionSummary,
    Refusal,
    SubmissionOutcome,
)
from local_operator.community.multipart import FilePart, StreamedMultipart, TextPart
from local_operator.env import resolve_radient_api_base_url
from local_operator.providers.auth_store import AuthStoreError, OAuthAccess
from local_operator.providers.radient_credentials import (
    org_oauth_destination_allowed,
    strip_base_credentials,
)

logger = logging.getLogger(__name__)

COMMUNITY_PATH_PREFIX = "/community/v1"

#: Ordinary request deadline: the Team Bridge client's ``REQUEST_TIMEOUT_S``
#: (30 s). A submission's moderation runs inside the request on the hub; if it
#: outlasts this, the outcome is recoverable through the submission status read
#: by its Idempotency-Key, so a longer local wait buys nothing.
REQUEST_TIMEOUT_S = 30.0

#: The contract's long-poll hold ("holds the request open up to 25 s and
#: answers 200 with an empty batch on timeout") plus headroom for the hop, so a
#: healthy empty poll is never cut off as a local timeout.
EVENTS_HOLD_S = 25.0
EVENTS_TIMEOUT_S = EVENTS_HOLD_S + 10.0

#: Hard cap on a JSON response read. NOT MEASURED — the hub's message routes are
#: still `planned`, so no response-size distribution exists. Derived from the
#: contract instead: message text ≤ 4000 characters (≤ 16 KB of UTF-8) and the
#: directory's 25-row pages, so 4 MiB admits ~250 worst-case messages per page,
#: far above any page the contract describes. The invariant is "a hostile or
#: broken hop cannot make this process buffer without bound"; revisit once the
#: history page size is specified.
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

#: The contract's submission limits (``community/limits.py``) are checked here
#: too, so an over-limit draft is refused before a single byte (or a
#: rate-limit slot) is spent, whichever surface built it.

ImageType = Literal["image/png", "image/jpeg", "image/webp"]
_IMAGE_EXT: dict[str, str] = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}

_COMMUNITY_ID = re.compile(COMMUNITY_ID_PATTERN)
_IDEMPOTENCY_KEY = re.compile(IDEMPOTENCY_KEY_PATTERN)
_OPAQUE_ID = re.compile(OPAQUE_ID_PATTERN)
_DIR_FILTERS: frozenset[str] = frozenset(DirectoryFilter.__args__)  # type: ignore[attr-defined]
_RATE_SCOPES: frozenset[str] = frozenset(RateLimitScope.__args__)  # type: ignore[attr-defined]

_M = TypeVar("_M", bound=BaseModel)


@dataclass(frozen=True)
class CommunityImage:
    """One image to submit, already in memory and within the contract's limits."""

    content_type: str
    data: bytes

    @classmethod
    def from_base64(cls, content_type: str, data_base64: str) -> CommunityImage:
        """Decode a surface's base64 image, refusing malformed input as invalid_request."""

        try:
            data = base64.b64decode(data_base64, validate=True)
        except (binascii.Error, ValueError):
            raise CommunityInvalidRequest() from None
        return cls(content_type=content_type, data=data)


def _sniff(data: bytes) -> str | None:
    """The image type the bytes actually are (PNG/JPEG/WebP), or None."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def new_idempotency_key() -> str:
    """A fresh key in the contract grammar (32 URL-safe characters)."""

    return secrets.token_urlsafe(24)


def community_base_url(api_base_url: str) -> str:
    """The ``/community/v1`` root on the same origin as the private API base."""

    parts = urllib.parse.urlsplit(api_base_url)
    if not parts.scheme or not parts.netloc:
        raise CommunityDestinationRefused()
    return f"{parts.scheme}://{parts.netloc}{COMMUNITY_PATH_PREFIX}"


def _require(pattern: re.Pattern[str], value: Any) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise CommunityInvalidRequest()
    return value


def _require_cursor(cursor: Any) -> None:
    """An opaque cursor: a string ≤ 512 characters (``Cursor``/``After``)."""

    if cursor is not None and (not isinstance(cursor, str) or len(cursor) > MAX_CURSOR_CHARS):
        raise CommunityInvalidRequest()


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value) >= 1:
        return int(value)
    return None


def raise_for_response(response: httpx.Response) -> None:
    """Map one non-2xx hub answer onto its typed refusal (see ``errors``).

    The hub's ``message`` is never read into the exception: only the closed
    ``code`` and the contract's structured fields survive.
    """

    status = response.status_code
    if 200 <= status < 300:
        return
    body = _json(response)
    payload = body if isinstance(body, Mapping) else {}
    code = payload.get("code")
    if code == "images_unavailable":
        raise CommunityImagesUnavailable()
    if status == 400:
        raise CommunityBadRequest()
    if status == 401:
        raise CommunitySignInRequired()
    if status == 403:
        raise CommunityForbidden()
    if status == 404:
        raise CommunityNotFound()
    if status == 409:
        if code == "guidelines_changed":
            version = payload.get("guidelines_version")
            raise CommunityGuidelinesChanged(version if isinstance(version, str) else None)
        raise CommunityConflict()
    if status == 413:
        raise CommunityPayloadTooLarge()
    if status == 422:
        try:
            refusal = Refusal.model_validate(payload)
        except ValidationError:
            # A 422 we cannot read still means "not posted"; it must not be
            # rendered with a guessed category, so it is an invalid response.
            raise CommunityInvalidResponse() from None
        raise CommunityRefused(refusal.category, refusal.reference)
    if status == 429:
        retry_after = _positive_int(payload.get("retry_after")) or _positive_int(
            response.headers.get("retry-after")
        )
        scope = payload.get("scope")
        raise CommunityRateLimited(
            retry_after or 1, scope if isinstance(scope, str) and scope in _RATE_SCOPES else None
        )
    if status == 503 and code == "moderation_unavailable":
        raise CommunityModerationUnavailable(_positive_int(payload.get("retry_after")))
    if status >= 500:
        raise CommunityUnavailable()
    raise CommunityInvalidResponse()


def _parse(model: type[_M], response: httpx.Response) -> _M:
    body = _json(response)
    try:
        return model.model_validate(body)
    except ValidationError:
        # ``from None``, here and at every validation failure in this module: a
        # pydantic error's text QUOTES the input, so chaining it would put a
        # response body (message text) into any traceback a caller logs.
        raise CommunityInvalidResponse() from None


class CommunityClient:
    """Authenticated HTTPS for the member half of ``/community/v1``.

    Cheap to construct (nothing opens until a request runs); surfaces build one
    per operation. ``transport`` is a test seam (``httpx.MockTransport``) and is
    never passed by production callers. ``expected_account_id`` is the account
    the surface captured; ``None`` means "whoever the pinned row belongs to".
    """

    def __init__(
        self,
        config_dir: Path | None = None,
        *,
        base_url: str | None = None,
        store: Any = None,
        transport: httpx.AsyncBaseTransport | None = None,
        credential_id: int | None = None,
        expected_account_id: str | None = None,
    ) -> None:
        self._config_dir = config_dir
        api_base, self._base_had_credentials = strip_base_credentials(
            (base_url or resolve_radient_api_base_url()).rstrip("/")
        )
        self._api_base_url = api_base
        self._store = store
        self._transport = transport
        self._pinned_credential_id = credential_id
        self._expected_account_id = expected_account_id

    # -- auth ---------------------------------------------------------------

    def _auth_store(self) -> Any:
        """The PLAIN credential store (never mesh-aware; see Team Bridge H2).

        A bearer lent by a paired device would act as the OWNER account in a
        public space, so only this machine's own Radient login can serve.
        """

        if self._store is None:
            from local_operator.providers.auth_store import shared_auth_store

            db_path = (self._config_dir / "auth.db") if self._config_dir is not None else None
            self._store = shared_auth_store(db_path=db_path, config_dir=self._config_dir)
        return self._store

    def _pinned_row(self) -> Any:
        store = self._auth_store()
        if self._pinned_credential_id is not None:
            row = store.get_credential(self._pinned_credential_id)
            if row is None or row.provider != "radient" or row.credential_type != "oauth":
                return None
            return row
        rows = [r for r in store.list_credentials("radient") if r.credential_type == "oauth"]
        if len(rows) != 1:
            # Zero is "not signed in"; several is "the device cannot pick".
            return None
        # Pin for this client's lifetime: a later re-login that adds or swaps a
        # row must not move an in-flight operation onto another account.
        self._pinned_credential_id = rows[0].id
        return rows[0]

    def _require_destination(self) -> str:
        if self._base_had_credentials:
            raise CommunityDestinationRefused()
        if not org_oauth_destination_allowed(self._api_base_url):
            raise CommunityDestinationRefused()
        return community_base_url(self._api_base_url)

    async def _fresh_access(
        self, credential_id: int, *, expected_account_id: str, force: bool = False
    ) -> OAuthAccess:
        """Refresh-and-read the EXACT pinned row; token and identity from ONE snapshot."""

        try:
            data = await self._auth_store().ensure_oauth_fresh_or_raise(credential_id, force=force)
        except AuthStoreError as failure:
            raise CommunitySignInRequired() from failure
        if not isinstance(data, Mapping) or not data.get("access"):
            raise CommunitySignInRequired()
        account_id = data.get("account_id")
        if not isinstance(account_id, str) or account_id != expected_account_id:
            raise CommunitySignInRequired()
        return OAuthAccess(
            access_token=str(data["access"]),
            credential_id=credential_id,
            account_id=account_id,
            email=data.get("email"),
            org_id=data.get("org_id"),
            api_endpoint=data.get("api_endpoint"),
            kind="oauth",
            raw=dict(data),
        )

    async def current_access(self) -> OAuthAccess:
        """The pinned account's fresh bearer, or the sign-in / mismatch refusal."""

        self._require_destination()
        row = self._pinned_row()
        if row is None:
            raise CommunitySignInRequired()
        raw = dict(row.data) if isinstance(row.data, Mapping) else {}
        owner = raw.get("account_id")
        if not isinstance(owner, str) or not owner:
            raise CommunitySignInRequired()
        access = await self._fresh_access(row.id, expected_account_id=owner)
        if self._expected_account_id is not None and access.account_id != self._expected_account_id:
            raise CommunityAccountMismatch()
        return access

    # -- transport ----------------------------------------------------------

    async def _send_once(
        self,
        method: str,
        url: str,
        token: str,
        *,
        params: Mapping[str, str] | None,
        json_body: Mapping[str, Any] | None,
        multipart: StreamedMultipart | None,
        idempotency_key: str | None,
        timeout: float,
    ) -> httpx.Response:
        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        content: Any = None
        if multipart is not None:
            headers["Content-Type"] = multipart.content_type
            # Sized, so httpx sends no chunked encoding (see multipart module).
            headers["Content-Length"] = str(multipart.content_length)
            content = multipart
        kwargs: dict[str, Any] = {"follow_redirects": False}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        # A fresh, closed-after client per request: nothing pooled or cached is
        # shared with the private Team Bridge client (module docstring).
        async with httpx.AsyncClient(**kwargs) as http:
            async with http.stream(
                method,
                url,
                params=params,
                headers=headers,
                json=json_body,
                content=content,
                timeout=timeout,
            ) as streamed:
                chunks: list[bytes] = []
                size = 0
                async for chunk in streamed.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise CommunityInvalidResponse()
                    chunks.append(chunk)
                rebuilt = httpx.Headers(streamed.headers)
                for name in ("content-encoding", "content-length", "transfer-encoding"):
                    rebuilt.pop(name, None)
                return httpx.Response(
                    streamed.status_code,
                    headers=rebuilt,
                    content=b"".join(chunks),
                    request=streamed.request,
                )

    async def _request(
        self,
        operation: str,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | None] | None = None,
        json_body: Mapping[str, Any] | None = None,
        multipart: StreamedMultipart | None = None,
        idempotency_key: str | None = None,
        timeout: float = REQUEST_TIMEOUT_S,
    ) -> httpx.Response:
        """Resolve, send, refresh-once-on-401, then map any refusal."""

        base = self._require_destination()
        url = base + path
        clean = {k: v for k, v in (params or {}).items() if v is not None} or None
        pinned = await self.current_access()

        async def attempt(token: str) -> httpx.Response:
            try:
                return await self._send_once(
                    method,
                    url,
                    token,
                    params=clean,
                    json_body=json_body,
                    multipart=multipart,
                    idempotency_key=idempotency_key,
                    timeout=timeout,
                )
            except CommunityError:
                raise
            except httpx.HTTPError as failure:
                raise CommunityUnreachable() from failure

        response = await attempt(pinned.access_token)
        if response.status_code == 401:
            if not pinned.account_id:
                raise CommunitySignInRequired()
            refreshed = await self._fresh_access(
                pinned.credential_id, expected_account_id=pinned.account_id, force=True
            )
            response = await attempt(refreshed.access_token)
            if response.status_code == 401:
                # The second failure stops: retrying cannot mint a working bearer.
                raise CommunitySignInRequired()
        logger.debug("community %s -> %s", operation, response.status_code)
        raise_for_response(response)
        return response

    @staticmethod
    def _path(*segments: str) -> str:
        return "/" + "/".join(urllib.parse.quote(s, safe="") for s in segments)

    # -- communities --------------------------------------------------------

    async def list_communities(self) -> CommunityList:
        response = await self._request("list_communities", "GET", "/communities")
        return _parse(CommunityList, response)

    async def bootstrap(self, community_id: str) -> Bootstrap:
        cid = _require(_COMMUNITY_ID, community_id)
        response = await self._request(
            "bootstrap", "GET", self._path("communities", cid, "bootstrap")
        )
        return _parse(Bootstrap, response)

    async def join(
        self, community_id: str, *, guidelines_version: str, accept_guidelines: bool
    ) -> Bootstrap:
        """Join by EXPLICITLY accepting one guidelines version (contract JoinRequest).

        ``accept_guidelines`` must be the literal ``True``: acceptance is the
        user's act on a specific version, so nothing here defaults it.
        """

        cid = _require(_COMMUNITY_ID, community_id)
        if accept_guidelines is not True:
            raise CommunityInvalidRequest()
        if not isinstance(guidelines_version, str) or not guidelines_version:
            raise CommunityInvalidRequest()
        response = await self._request(
            "join",
            "POST",
            self._path("communities", cid, "join"),
            json_body={"guidelines_version": guidelines_version, "accept_guidelines": True},
        )
        return _parse(Bootstrap, response)

    async def leave(self, community_id: str) -> bool:
        """Leave; returns ``already_left``. The contract: a second leave is 404,
        "treat it as already left"."""

        cid = _require(_COMMUNITY_ID, community_id)
        try:
            await self._request("leave", "POST", self._path("communities", cid, "leave"))
        except CommunityNotFound:
            return True
        return False

    # -- channels and history ----------------------------------------------

    async def channels(
        self,
        community_id: str,
        *,
        q: str | None = None,
        filter: str | None = None,
        cursor: str | None = None,
    ) -> ChannelPage:
        cid = _require(_COMMUNITY_ID, community_id)
        # The contract's query bounds (``Q`` ≤ 100, ``DirFilter`` enum, ``Cursor``
        # ≤ 512), refused locally so a malformed browse costs no directory slot.
        if q is not None and len(q) > 100:
            raise CommunityInvalidRequest()
        if filter is not None and filter not in _DIR_FILTERS:
            raise CommunityInvalidRequest()
        _require_cursor(cursor)
        response = await self._request(
            "channels",
            "GET",
            self._path("communities", cid, "directory", "channels"),
            params={"q": q, "filter": filter, "cursor": cursor},
        )
        return _parse(ChannelPage, response)

    async def history(
        self, community_id: str, conversation_id: str, *, cursor: str | None = None
    ) -> MessagePage:
        cid = _require(_COMMUNITY_ID, community_id)
        conv = _require(_OPAQUE_ID, conversation_id)
        _require_cursor(cursor)
        response = await self._request(
            "history",
            "GET",
            self._path("communities", cid, "conversations", conv, "messages"),
            params={"cursor": cursor},
        )
        return _parse(MessagePage, response)

    # -- submissions --------------------------------------------------------

    @staticmethod
    def _submission_body(
        *,
        text: str,
        images: Sequence[CommunityImage],
        thread_root_id: str | None,
        guidelines_version: str | None,
    ) -> StreamedMultipart:
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
            raise CommunityInvalidRequest()
        if len(images) > MAX_IMAGES:
            raise CommunityInvalidRequest()
        parts: list[TextPart | FilePart] = [TextPart("text", text)]
        if thread_root_id is not None:
            parts.append(TextPart("thread_root_id", _require(_OPAQUE_ID, thread_root_id)))
        if guidelines_version is not None:
            if not guidelines_version:
                raise CommunityInvalidRequest()
            parts.append(TextPart("guidelines_version", guidelines_version))
        for index, image in enumerate(images, start=1):
            ext = _IMAGE_EXT.get(image.content_type)
            if ext is None or len(image.data) == 0 or len(image.data) > MAX_IMAGE_BYTES:
                raise CommunityInvalidRequest()
            # The declared type must be what the bytes are: the hub sniffs too,
            # but a mismatch is a client bug better refused before upload.
            if _sniff(image.data) != image.content_type:
                raise CommunityInvalidRequest()
            parts.append(FilePart("images", f"image-{index}.{ext}", image.content_type, image.data))
        return StreamedMultipart(parts)

    async def submit(
        self,
        community_id: str,
        conversation_id: str,
        *,
        text: str,
        images: Sequence[CommunityImage] = (),
        thread_root_id: str | None = None,
        guidelines_version: str | None = None,
        idempotency_key: str | None = None,
    ) -> SubmissionOutcome:
        """Submit for moderation and publication. 200 = published (or replayed).

        ``idempotency_key``: pass the key of an earlier attempt to replay it
        (the hub returns the stored outcome without re-screening or counting it
        twice); ``None`` mints a fresh one. The key is echoed in the outcome.

        A REUSED KEY ALWAYS ANSWERS THE FIRST ATTEMPT'S OUTCOME, whatever the
        new body says (agent-server #414 review R1-M2: the hub stores no body
        digest, RFC §7.4/§8, so it cannot tell a different body apart and
        never answers 409 for one). A key therefore names ONE draft attempt:
        retry the same bytes with it, but an edited draft needs a new key, or
        the edit is silently answered with the old outcome.
        """

        cid = _require(_COMMUNITY_ID, community_id)
        conv = _require(_OPAQUE_ID, conversation_id)
        key = _require(_IDEMPOTENCY_KEY, idempotency_key or new_idempotency_key())
        body = self._submission_body(
            text=text,
            images=images,
            thread_root_id=thread_root_id,
            guidelines_version=guidelines_version,
        )
        response = await self._request(
            "submit",
            "POST",
            self._path("communities", cid, "conversations", conv, "submissions"),
            multipart=body,
            idempotency_key=key,
        )
        return _parse(SubmissionOutcome, response)

    async def submission(self, community_id: str, idempotency_key: str) -> SubmissionOutcome:
        cid = _require(_COMMUNITY_ID, community_id)
        key = _require(_IDEMPOTENCY_KEY, idempotency_key)
        response = await self._request(
            "submission_status", "GET", self._path("communities", cid, "submissions", key)
        )
        return _parse(SubmissionOutcome, response)

    async def cancel_submission(self, community_id: str, idempotency_key: str) -> SubmissionOutcome:
        cid = _require(_COMMUNITY_ID, community_id)
        key = _require(_IDEMPOTENCY_KEY, idempotency_key)
        # The contract's action suffix ``:cancel`` is part of the last segment.
        path = self._path("communities", cid, "submissions") + f"/{key}:cancel"
        response = await self._request("submission_cancel", "POST", path)
        return _parse(SubmissionOutcome, response)

    # -- messages -----------------------------------------------------------

    async def edit(
        self,
        community_id: str,
        message_id: str,
        *,
        text: str,
        images: Sequence[CommunityImage] = (),
        guidelines_version: str | None = None,
        idempotency_key: str | None = None,
    ) -> SubmissionOutcome:
        """Edit = a re-moderated submission (contract editMessage, SubmissionForm)."""

        cid = _require(_COMMUNITY_ID, community_id)
        mid = _require(_OPAQUE_ID, message_id)
        key = _require(_IDEMPOTENCY_KEY, idempotency_key or new_idempotency_key())
        body = self._submission_body(
            text=text, images=images, thread_root_id=None, guidelines_version=guidelines_version
        )
        path = self._path("communities", cid, "messages") + f"/{mid}:edit"
        response = await self._request("edit", "POST", path, multipart=body, idempotency_key=key)
        return _parse(SubmissionOutcome, response)

    async def delete_message(self, community_id: str, message_id: str) -> None:
        cid = _require(_COMMUNITY_ID, community_id)
        mid = _require(_OPAQUE_ID, message_id)
        await self._request("delete", "DELETE", self._path("communities", cid, "messages", mid))

    async def react(
        self, community_id: str, message_id: str, *, emoji: str, remove: bool = False
    ) -> ReactionSummary:
        cid = _require(_COMMUNITY_ID, community_id)
        mid = _require(_OPAQUE_ID, message_id)
        if not isinstance(emoji, str) or not emoji or len(emoji) > MAX_EMOJI_CHARS:
            raise CommunityInvalidRequest()
        response = await self._request(
            "react",
            "POST",
            self._path("communities", cid, "messages", mid, "reactions"),
            json_body={"emoji": emoji, "remove": bool(remove)},
        )
        return _parse(ReactionSummary, response)

    # -- events -------------------------------------------------------------

    async def events(self, community_id: str, *, after: str | None = None) -> EventBatch:
        """One long-poll: content-free invalidations (the caller refetches)."""

        cid = _require(_COMMUNITY_ID, community_id)
        _require_cursor(after)
        response = await self._request(
            "events",
            "GET",
            self._path("communities", cid, "events"),
            params={"after": after},
            timeout=EVENTS_TIMEOUT_S,
        )
        return _parse(EventBatch, response)


__all__ = [
    "COMMUNITY_PATH_PREFIX",
    "EVENTS_TIMEOUT_S",
    "MAX_IMAGES",
    "MAX_IMAGE_BYTES",
    "MAX_RESPONSE_BYTES",
    "MAX_TEXT_CHARS",
    "CommunityClient",
    "CommunityImage",
    "community_base_url",
    "new_idempotency_key",
    "raise_for_response",
]
