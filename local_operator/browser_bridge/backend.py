"""Session-side client for the browser bridge's authenticated HTTP leg."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.protocol import (
    COMMAND_TIMEOUTS,
    ORIGIN_PROMPT_WINDOW_S,
    ErrorCode,
    ErrorDetail,
    Request,
    Response,
)

#: Slack on top of the daemon's worst-case budget so scheduling jitter and the
#: daemon's own response serialization never push a legitimate typed answer
#: past the client's deadline.
_CLIENT_TIMEOUT_MARGIN_S = 5.0

ERROR_MESSAGES = {
    ErrorCode.EXTENSION_DISCONNECTED: (
        "browser extension not connected: the bridge daemon is running but no browser is "
        "attached. Ask the user to open their browser (the extension reconnects automatically), "
        "or check the extension is enabled."
    ),
    ErrorCode.NOT_PAIRED: (
        "browser bridge not paired: run 'lop browser pair' and enter the code in the "
        "extension popup, then retry."
    ),
    ErrorCode.DEBUGGER_CONFLICT: (
        "cannot drive the tab: DevTools (or another debugger) is attached to it. Ask the "
        "user to close DevTools on that tab."
    ),
    ErrorCode.BUSY: "the browser bridge is busy with another command; retry this action once.",
    ErrorCode.PROTO_MISMATCH: (
        "browser bridge protocol mismatch: update Local Operator and the browser extension, "
        "then restart the bridge daemon."
    ),
}


class BridgeError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


class BridgeUnreachable(RuntimeError):
    pass


def bridge_browser_available(root: Path | None = None) -> bool:
    """File-only createIf probe: no socket, no subprocess, never raises."""
    try:
        return state_store.available(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


def _origin(value: str) -> str:
    parsed = urlsplit(value)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else value


def format_error(error: BridgeError, *, action: str = "", surface: str = "") -> str:
    """Map every wire error to one actionable model-facing diagnostic."""
    if error.code in ERROR_MESSAGES:
        return ERROR_MESSAGES[error.code]
    if error.code == ErrorCode.TAB_CLOSED:
        return (
            f"browser tab {surface or '(unknown)'} is gone; dropped the handle. "
            "Use 'open' with a URL to get a new tab."
        )
    if error.code == ErrorCode.ORIGIN_DENIED:
        origin = str(error.data.get("origin") or _origin(str(error.data.get("url", ""))))
        return (
            f"navigation to {origin or '(unknown origin)'} was denied by the user (or the "
            "permission prompt went unanswered). Do not retry the same origin; ask the user "
            "to allow it from the extension popup if it is needed."
        )
    if error.code == ErrorCode.NAV_TIMEOUT:
        return f"navigation did not complete: {error.message}"
    if error.code == ErrorCode.NAV_FAILED:
        return f"navigation failed: {error.message}"
    if error.code == ErrorCode.ELEMENT_NOT_FOUND:
        return (
            f"element not found: {error.message}. Take a new snapshot and retry with a fresh ref."
        )
    if error.code == ErrorCode.ORIGIN_PROMPT_PENDING:
        return "the extension is waiting for the user to approve this site in its popup."
    if error.code == ErrorCode.INTERNAL and error.data.get("tab_crashed"):
        return (
            f"the browser tab crashed while {action or 'the action'} was running. "
            "'open' the URL again to recover."
        )
    return f"browser bridge error ({error.code.value}): {error.message}"


def client_timeout(method: str) -> float:
    """HTTP budget for one RPC: the daemon's worst case, plus margin.

    The timeout chain (finding A3) is extension deny 60 s < daemon prompt
    window 65 s < this. The daemon deliberately holds a command open for
    base + ORIGIN_PROMPT_WINDOW_S while the extension shows its approval
    popup, so the client must outlive that whole budget or it fabricates an
    "unreachable" failure mid-prompt while the daemon is healthy and about to
    deliver a typed origin_denied/result (the flat 35 s timeout this replaces
    did exactly that; QA transcript 0ee4974ba84a). Unknown methods get the
    most conservative budget — the daemon rejects them quickly anyway.

    Known residual gap: a redirect chain can pause on SEVERAL origins in one
    command (origins.ts keys prompts by origin for this reason) and the daemon
    re-extends its deadline per pause, so two human-prompted hops can hold a
    legitimate wait past this budget. We deliberately do NOT size for N
    prompts: the chain depth is unbounded so any N is arbitrary, and every
    second added here delays reporting a genuinely hung daemon. Instead the
    timeout error's message points at the extension popup, which is the right
    advice in exactly that overrun.
    """
    base = COMMAND_TIMEOUTS.get(method, max(COMMAND_TIMEOUTS.values()))
    return base + ORIGIN_PROMPT_WINDOW_S + _CLIENT_TIMEOUT_MARGIN_S


class BridgeClient:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        current = state_store.read(self.root)
        if current is None:
            raise BridgeUnreachable(
                "browser bridge unreachable: no live daemon state. Run 'lop browser status'; "
                "'lop browser install' starts it."
            )
        request_id = f"r-{secrets.token_hex(6)}"
        request = Request(id=request_id, method=method, params=params)
        timeout = client_timeout(method)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                http_response = await client.post(
                    f"http://127.0.0.1:{current.port}/rpc",
                    headers={"X-Bridge-Key": current.session_key},
                    json=request.model_dump(mode="json"),
                )
        except httpx.ConnectTimeout as exc:
            # Timeout WHILE connecting: nothing was ever reached, so this
            # belongs with the unreachable branch below, not the popup one —
            # httpx.ConnectTimeout subclasses TimeoutException, not
            # ConnectError, so ordering matters here (review finding m1).
            raise BridgeUnreachable(
                f"browser bridge unreachable: the daemon at 127.0.0.1:{current.port} is not "
                "answering. Run 'lop browser status'; 'lop browser install' starts it."
            ) from exc
        except httpx.TimeoutException as exc:
            # Read/write/pool timeout AFTER connecting: the daemon accepted the
            # command and never answered within a budget that already covers
            # every legitimate wait (base + prompt window + margin). Calling
            # this "unreachable" sent a QA session (transcript 0ee4974ba84a)
            # into an hour of restarting a healthy daemon while the extension
            # popup sat waiting on the human, so name the likely cause. We only
            # KNOW the connection was accepted — say that, not "running".
            raise BridgeUnreachable(
                f"the browser bridge accepted '{method}' but did not answer within "
                f"{timeout:.0f}s. The command may be stuck in the browser — e.g. waiting "
                "on a site-permission decision in the extension popup. Ask the user to "
                "check the extension popup before restarting anything."
            ) from exc
        except httpx.RequestError as exc:
            # Everything else — ConnectError (refused), ReadError /
            # RemoteProtocolError (daemon died mid-request), reset — means the
            # daemon is gone or dying, so restart advice is honest here
            # (review finding m2: these must NOT get the popup message).
            raise BridgeUnreachable(
                f"browser bridge unreachable: the daemon at 127.0.0.1:{current.port} is not "
                "answering. Run 'lop browser status'; 'lop browser install' starts it."
            ) from exc
        if http_response.status_code == 401:
            raise BridgeUnreachable(
                "browser bridge rejected its state-file key; restart with 'lop browser restart'."
            )
        try:
            response = Response.model_validate(http_response.json())
        except (ValueError, json.JSONDecodeError) as exc:
            raise BridgeUnreachable(
                f"browser bridge returned an invalid response (HTTP {http_response.status_code})."
            ) from exc
        if not response.ok:
            detail = response.error or ErrorDetail(
                code=ErrorCode.INTERNAL, message="unknown failure"
            )
            raise BridgeError(detail.code, detail.message, detail.data)
        return response.result or {}
