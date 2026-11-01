"""Unwired human-file history commissioning; not a route or an authority source.

The caller MUST be core-owned: capture ``HistoryScope`` from an authenticated
native login and a trusted window/view owner, never renderer-provided fence
fields. Bind the scope to the registry epoch at that fresh authorization; after
rotation, reacquire and revalidate scope, never merely replace its old epoch.
Its synchronous reader must reflect invalidation, not echo input. This
module reads the hub itself and commissions only descriptors from that same
human-addressed response. No endpoint/actor/purpose argument is accepted.

The registry serializes its generation watermark with insertion, but only when
the trusted owner publishes invalidation there before reporting it complete.
Observation is not a distributed lock: main and core revocation can race and
cannot recall an HTTP request already sent. The hub must authorize membership at
history and download; the eventual integration must recheck before publication.
This draft is deliberately unregistered pending security review. A sequence is
only a lookup hint: authenticated history must match the fixed selected message
ID, conversation and sequence before any files can be commissioned. BLOCKER:
timeline history omits thread replies; the separate replies API is oldest-first
and no bounded targeted reply lookup is established here. Do not integrate or
claim support for reply attachments until that lookup is securely implemented.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from local_operator.providers.auth_store import GRANT_DEAD_AT_KEY, OAUTH_REFRESH_SKEW_MS
from local_operator.team_bridge.client import (
    ACCOUNT_MISMATCH_CODE,
    ACCOUNT_MISMATCH_SENTENCE,
    REQUEST_TIMEOUT_S,
    SIGN_IN_SENTENCE,
    BridgeRefusal,
    BridgeSignInRequired,
    BridgeUnreachable,
    TeamBridgeClient,
)
from local_operator.team_bridge.file_action_registry import (
    InertActionBinding,
    InertFileActionRegistry,
)
from local_operator.team_bridge.read_context import TENANT_ID_PATTERN
from local_operator.team_bridge.types import (
    ATTACHMENT_ID_PATTERN,
    CONVERSATION_ID_PATTERN,
    is_account_id,
    is_conversation_message_id,
)

_HISTORY_MAX_BYTES = 2 * 1024 * 1024
# Leave room for +1 in JavaScript's exact integer range if a future UI carries
# the hint; the hint never authorizes a message or its attachments.
_HISTORY_MAX_SEQUENCE = (1 << 53) - 2


@dataclass(frozen=True, repr=False)
class HistoryScope:
    credential_id: int
    account_id: str
    login_incarnation: str
    tenant_id: str
    window_id: str
    view_generation: int
    conversation_id: str
    message_id: str
    registry_epoch: str
    message_seq: int


def _invalid() -> BridgeRefusal:
    return BridgeRefusal("bridge_invalid_response", "File history unavailable.", http_status=502)


def _pairs_unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _file_ids(body: bytes, scope: HistoryScope) -> frozenset[str]:
    """Parse only bounded server output, refusing ambiguous or inconsistent pages."""
    try:
        envelope = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_pairs_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant")),
        )
        result = envelope["result"]
        messages = result["messages"]
        if not isinstance(result, dict) or not isinstance(messages, list) or len(messages) != 1:
            raise ValueError("page")
        found = messages[0]
        if (
            not isinstance(found, dict)
            or found.get("id") != scope.message_id
            or found.get("conversation_id") != scope.conversation_id
            or type(found.get("seq")) is not int
            or found.get("seq") != scope.message_seq
        ):
            raise ValueError("message identity")
        files = found.get("files", [])
        if not isinstance(files, list) or len(files) > 16:
            raise ValueError("files")
        ids: set[str] = set()
        observed: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("file")
            file_id = item.get("id")
            if (
                not isinstance(file_id, str)
                or re.fullmatch(ATTACHMENT_ID_PATTERN, file_id) is None
                or file_id in observed
            ):
                raise ValueError("file id")
            observed.add(file_id)
            if (
                item.get("mime") != "application/octet-stream"
                or item.get("download_only") is not True
            ):
                raise ValueError("file type")
            if item.get("state") == "expired":
                continue
            if item.get("state") != "bound" or item.get("class") not in {
                "pdf",
                "text",
                "container",
                "other",
            }:
                raise ValueError("file state")
            if type(item.get("size")) is not int or not 0 <= item["size"] <= 25 * 1024 * 1024:
                raise ValueError("file size")
            expiry = item.get("expires_at")
            if not isinstance(expiry, str) or len(expiry) > 64:
                raise ValueError("file expiry")
            expires = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            if expires.tzinfo is None:
                raise ValueError("file expiry zone")
            if expires > datetime.now(timezone.utc):
                ids.add(file_id)
        if not ids:
            raise ValueError("no files")
        return frozenset(ids)
    except (ValueError, KeyError, TypeError, OverflowError, UnicodeError, RecursionError):
        raise _invalid() from None


def _native_snapshot(
    client: TeamBridgeClient, scope: HistoryScope, current: Callable[[], HistoryScope]
) -> tuple[Any, ...]:
    try:
        same = current() == scope
    except Exception:
        same = False
    if not same:
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
    store = client._auth_store()
    rows = store.native_lifecycle_snapshot(store.db_path, "radient")
    if not isinstance(rows, list) or any(
        not isinstance(row, tuple) or len(row) != 9 for row in rows
    ):
        raise BridgeSignInRequired(SIGN_IN_SENTENCE)
    matches = [row for row in rows if row[0] == scope.credential_id]
    if len(matches) != 1:
        raise BridgeSignInRequired(SIGN_IN_SENTENCE)
    row = matches[0]
    if (
        row[2] != scope.login_incarnation
        or row[3] is not None
        or row[4] != "oauth"
        or row[5] != scope.account_id
        or row[8] is not None
    ):
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
    return row


async def commission_human_history(
    client: TeamBridgeClient,
    registry: InertFileActionRegistry,
    scope: HistoryScope,
    current_scope: Callable[[], HistoryScope],
) -> str:
    """Return only an opaque context; `current_scope` is a TRUSTED integration seam.

    Caller-selected message ID selects only a message in bounded hub history;
    file IDs are derived solely from that message's authenticated response page.
    """
    if (
        not isinstance(scope, HistoryScope)
        or not callable(current_scope)
        or type(scope.credential_id) is not int
        or scope.credential_id <= 0
        or not is_account_id(scope.account_id)
        or not isinstance(scope.login_incarnation, str)
        or re.fullmatch(r"[0-9a-f]{32}", scope.login_incarnation) is None
        or not isinstance(scope.tenant_id, str)
        or re.fullmatch(TENANT_ID_PATTERN, scope.tenant_id) is None
        or not isinstance(scope.conversation_id, str)
        or re.fullmatch(CONVERSATION_ID_PATTERN, scope.conversation_id) is None
        or not is_conversation_message_id(scope.message_id)
        or type(scope.message_seq) is not int
        or not 1 <= scope.message_seq <= _HISTORY_MAX_SEQUENCE
        or not isinstance(scope.window_id, str)
        or not scope.window_id
        or len(scope.window_id) > 256
        or type(scope.view_generation) is not int
        or scope.view_generation < 0
    ):
        raise BridgeRefusal("invalid_request", "Invalid file history request.", http_status=422)
    # A stale scope cannot borrow the registry's post-rotation epoch. The
    # trusted scope owner must mint a fresh scope after reauthorization.
    registry.require_epoch(scope.registry_epoch)
    before = _native_snapshot(client, scope, current_scope)
    try:
        client._require_destination()
    except BridgeRefusal:
        raise BridgeRefusal(
            "bridge_destination_refused", "File service unavailable.", http_status=503
        ) from None
    if (
        client._pinned_credential_id is not None
        and client._pinned_credential_id != scope.credential_id
    ):
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
    row = client._auth_store().get_credential(scope.credential_id)
    if (
        row is None
        or _native_snapshot(client, scope, current_scope) != before
        or row.updated_at != before[1]
    ):
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
    if (
        row.provider != "radient"
        or row.credential_type != "oauth"
        or row.disabled_cause is not None
        or not isinstance(row.data, Mapping)
        or row.data.get("account_id") != scope.account_id
    ):
        raise BridgeSignInRequired(SIGN_IN_SENTENCE)
    token, expires = row.data.get("access"), row.data.get("expires")
    if (
        row.data.get(GRANT_DEAD_AT_KEY) is not None
        or not isinstance(token, str)
        or not token
        or type(expires) is not int
        or expires <= int(time.time() * 1000) + OAUTH_REFRESH_SKEW_MS
    ):
        raise BridgeRefusal(
            "file_auth_refresh_required", "Sign in or refresh before downloading.", http_status=401
        )

    def guard() -> None:
        registry.require_epoch(scope.registry_epoch)
        if _native_snapshot(client, scope, current_scope) != before:
            raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
        # The read must not outlive the same bearer admitted at its start.
        if expires <= int(time.time() * 1000) + OAUTH_REFRESH_SKEW_MS:
            raise BridgeRefusal(
                "file_auth_refresh_required",
                "Sign in or refresh before downloading.",
                http_status=401,
            )

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_S, transport=client._transport, follow_redirects=False
        ) as http:
            guard()
            async with http.stream(
                "GET",
                client.base_url + f"/messaging/conversations/{scope.conversation_id}/messages",
                params={
                    "tenant_id": scope.tenant_id,
                    "limit": 1,
                    "before_seq": scope.message_seq + 1,
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            ) as response:
                guard()
                if response.status_code == 401:
                    raise BridgeRefusal(
                        "file_reauthentication_required",
                        "Sign in again before downloading.",
                        http_status=401,
                    )
                if response.status_code != 200:
                    raise _invalid()
                if (
                    response.headers.get("content-encoding", "identity").strip().lower()
                    != "identity"
                    or response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    != "application/json"
                ):
                    raise _invalid()
                length = response.headers.get("content-length")
                if length is not None and (
                    len(length) > 10
                    or re.fullmatch(r"[0-9]+", length) is None
                    or int(length) > _HISTORY_MAX_BYTES
                ):
                    raise _invalid()
                data = bytearray()
                async for chunk in response.aiter_raw():
                    if len(chunk) > _HISTORY_MAX_BYTES - len(data):
                        raise _invalid()
                    data.extend(chunk)
                guard()
    except httpx.HTTPError:
        raise BridgeUnreachable() from None
    files = _file_ids(bytes(data), scope)
    # This check is not atomic with an independent auth write. View retirement
    # is separately serialized with insertion by the registry's watermark.
    guard()
    return registry.commission(
        InertActionBinding(
            credential_id=scope.credential_id,
            account_id=scope.account_id,
            login_incarnation=scope.login_incarnation,
            tenant_id=scope.tenant_id,
            window_id=scope.window_id,
            view_generation=scope.view_generation,
            conversation_id=scope.conversation_id,
            message_id=scope.message_id,
            file_ids=files,
        ),
        ttl_seconds=30,
        expected_epoch=scope.registry_epoch,
    )
