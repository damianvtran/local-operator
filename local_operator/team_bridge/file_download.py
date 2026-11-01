"""Human-only generic-file transport, intentionally not registered as an API.

This uses the existing client's pinned OAuth and destination guard; its own
contiguous bounded stream does not alter the image/agent method. A
future desktop route must require an explicit click and keep the returned bytes
out of logs, URLs and previews. The hub remains the authority for PERSON and
active HUMAN membership on every request.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx

from local_operator.providers.auth_store import GRANT_DEAD_AT_KEY, OAUTH_REFRESH_SKEW_MS
from local_operator.team_bridge.client import (
    ACCOUNT_MISMATCH_CODE,
    ACCOUNT_MISMATCH_SENTENCE,
    SIGN_IN_SENTENCE,
    REQUEST_TIMEOUT_S,
    BridgeRefusal,
    BridgeSignInRequired,
    BridgeUnreachable,
    TeamBridgeClient,
)
from local_operator.team_bridge.read_context import TENANT_ID_PATTERN
from local_operator.team_bridge.types import (
    ATTACHMENT_ID_PATTERN,
    CONVERSATION_ID_PATTERN,
    is_account_id,
    is_conversation_message_id,
)

# S3a's largest server upload class, not an authority to upload or serve files.
# Refuse content encoding before streaming, and bound total bytes before each
# bytearray extension. This bounds accumulated payload, not socket/RSS peaks.
FILE_DOWNLOAD_MAX_BYTES = 25 * 1024 * 1024
_FILE_MIME = "application/octet-stream"


@dataclass(frozen=True)
class FileActionFence:
    """Trusted action-time snapshot, never a renderer-supplied authority.

    The integration must capture it at the original user action, before any
    await or queued dispatch. A server-owned reader below must reflect changes
    in account, login incarnation, tenant and UI/action generation atomically.
    """

    account_id: str
    credential_id: int
    login_incarnation: str
    tenant_id: str
    generation: int
    conversation_id: str
    message_id: str
    attachment_id: str


def _invalid_response() -> BridgeRefusal:
    return BridgeRefusal("bridge_invalid_response", "Radient returned an invalid response.", http_status=502)


def _authority_snapshot(client: TeamBridgeClient, fence: FileActionFence) -> tuple:
    """Read the committed native lifecycle identity without reading a bearer.

    Unknown/legacy state, disabled or mesh-origin credentials are not authority.
    This observation is not a cross-process lock and cannot retract a request
    already handed to the network.
    """
    store = client._auth_store()
    snapshots = store.native_lifecycle_snapshot(store.db_path, "radient")
    if snapshots is None:
        raise BridgeSignInRequired(SIGN_IN_SENTENCE)
    if not isinstance(snapshots, list) or any(not isinstance(row, tuple) or len(row) != 9 for row in snapshots):
        raise BridgeSignInRequired(SIGN_IN_SENTENCE)
    matches = [row for row in snapshots if row[0] == fence.credential_id]
    if len(matches) != 1:
        raise BridgeSignInRequired(SIGN_IN_SENTENCE)
    identity = matches[0]
    if (identity[2] != fence.login_incarnation or identity[3] is not None
            or identity[4] != "oauth" or identity[5] != fence.account_id or identity[8] is not None):
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
    return identity


def _current_action_snapshot(
    client: TeamBridgeClient, fence: FileActionFence, current_action_fence: Callable[[], FileActionFence]
) -> tuple:
    """Revalidate a trusted, main-owned action before dispatch and final save.

    The eventual integration must supply a server-owned current-action reader;
    accepting a renderer's claimed generation is not this contract. A caller
    must repeat this check after a save dialog and before final file publication.
    """
    try:
        current = current_action_fence()
    except Exception:
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409) from None
    if current != fence:
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
    return _authority_snapshot(client, fence)


def verify_file_action_current(
    client: TeamBridgeClient, fence: FileActionFence, current_action_fence: Callable[[], FileActionFence]
) -> None:
    """Repeat the authority check after a save dialog, before publishing bytes.

    No lifecycle row (which includes private metadata) is returned to a caller.
    """
    _current_action_snapshot(client, fence, current_action_fence)


async def download_human_file(
    client: TeamBridgeClient,
    conversation_id: str,
    message_id: str,
    attachment_id: str,
    *,
    tenant_id: str,
    expected_account_id: str,
    action_fence: FileActionFence,
    current_action_fence: Callable[[], FileActionFence],
) -> bytes:
    """Read a bound file for a captured person; no endpoint or purpose is accepted.

    Zero bytes is a valid file. Unknown status/type/encoding fails closed. The
    raw upstream error body is never exposed: it can contain sensitive content.
    The hub independently verifies current membership and the exact message.
    """
    if (
        not isinstance(conversation_id, str)
        or re.fullmatch(CONVERSATION_ID_PATTERN, conversation_id) is None
        or not is_conversation_message_id(message_id)
        or not isinstance(attachment_id, str)
        or re.fullmatch(ATTACHMENT_ID_PATTERN, attachment_id) is None
        or not isinstance(tenant_id, str)
        or re.fullmatch(TENANT_ID_PATTERN, tenant_id) is None
        or not is_account_id(expected_account_id)
        or not isinstance(action_fence, FileActionFence)
        or action_fence.account_id != expected_account_id
        or action_fence.tenant_id != tenant_id
        or type(action_fence.credential_id) is not int
        or action_fence.credential_id <= 0
        or not isinstance(action_fence.login_incarnation, str)
        or re.fullmatch(r"[0-9a-f]{32}", action_fence.login_incarnation) is None
        or (action_fence.conversation_id, action_fence.message_id, action_fence.attachment_id)
        != (conversation_id, message_id, attachment_id)
        or type(action_fence.generation) is not int
        or action_fence.generation < 0
        or not callable(current_action_fence)
    ):
        raise BridgeRefusal("invalid_request", "Invalid file request.", http_status=422)

    # A delayed action must fail before reading bearer material. No path in
    # this operation calls OAuth freshness/refresh or retries a rejected token.
    before = _current_action_snapshot(client, action_fence, current_action_fence)
    try:
        client._require_destination()
    except BridgeRefusal:
        # The generic destination diagnostic includes a configured URL. This
        # byte path never returns a URL or query to the caller or its logs.
        raise BridgeRefusal("bridge_destination_refused", "File service unavailable.", http_status=503) from None
    if client._pinned_credential_id is not None and client._pinned_credential_id != action_fence.credential_id:
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
    row = client._auth_store().get_credential(action_fence.credential_id)
    after = _current_action_snapshot(client, action_fence, current_action_fence)
    if row is None or before != after or row.updated_at != before[1]:
        raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)
    if (row.provider != "radient" or row.credential_type != "oauth" or row.disabled_cause is not None
            or not isinstance(row.data, Mapping) or row.data.get("account_id") != expected_account_id):
        raise BridgeSignInRequired(SIGN_IN_SENTENCE)
    creds = row.data
    expires = creds.get("expires")
    token = creds.get("access")
    if (creds.get(GRANT_DEAD_AT_KEY) is not None or not isinstance(token, str) or not token
            or type(expires) is not int or expires <= int(time.time() * 1000) + OAUTH_REFRESH_SKEW_MS):
        raise BridgeRefusal("file_auth_refresh_required", "Sign in or refresh before downloading.", http_status=401)

    path = f"/messaging/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}"

    def guard() -> None:
        if _current_action_snapshot(client, action_fence, current_action_fence) != before:
            raise BridgeRefusal(ACCOUNT_MISMATCH_CODE, ACCOUNT_MISMATCH_SENTENCE, http_status=409)

    async def dispatch(token: str) -> tuple[int, bytes]:
        # A bytearray bounds per-chunk object overhead; it may still coexist
        # with the final bytes and the network stack's current chunk. This is a
        # payload limit, not an instantaneous process-RSS ceiling.
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, transport=client._transport, follow_redirects=False) as http:
                guard()  # No await between the last authority check and sending.
                async with http.stream(
                    "GET", client.base_url + path, params={"tenant_id": tenant_id},
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream, application/json", "Accept-Encoding": "identity"},
                ) as response:
                    guard()
                    status = response.status_code
                    if status != 200:
                        return status, b""
                    if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                        raise _invalid_response()
                    if response.headers.get("content-type", "").strip().lower() != _FILE_MIME:
                        raise _invalid_response()
                    length = response.headers.get("content-length")
                    if length is not None:
                        if len(length) > 10 or re.fullmatch(r"[0-9]+", length) is None:
                            raise _invalid_response()
                        if int(length) > FILE_DOWNLOAD_MAX_BYTES:
                            raise BridgeRefusal("attachment_too_large", "File is too large.", http_status=502)
                    data = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(chunk) > FILE_DOWNLOAD_MAX_BYTES - len(data):
                            raise BridgeRefusal("attachment_too_large", "File is too large.", http_status=502)
                        data.extend(chunk)
                    guard()
                    return status, bytes(data)
        except httpx.HTTPError:
            # HTTP exceptions can carry a request URL; do not retain them as causes.
            raise BridgeUnreachable() from None

    status, body = await dispatch(token)
    guard()
    if status == 401:
        # A 401 is terminal for this action. Never silently refresh or retry.
        raise BridgeRefusal("file_reauthentication_required", "Sign in again before downloading.", http_status=401)
    if status in (403, 404):
        raise BridgeRefusal("attachment_not_found", "File is unavailable.", http_status=404)
    if status != 200:
        raise _invalid_response()
    guard()
    return body
