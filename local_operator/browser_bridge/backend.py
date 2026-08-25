"""Session-side client for the browser bridge's authenticated HTTP leg."""

from __future__ import annotations

import asyncio
import json
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.protocol import (
    ErrorCode,
    ErrorDetail,
    Request,
    Response,
)

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
        timeout = 35.0 if method in ("open", "goto") else 30.0
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                http_response = await client.post(
                    f"http://127.0.0.1:{current.port}/rpc",
                    headers={"X-Bridge-Key": current.session_key},
                    json=request.model_dump(mode="json"),
                )
        except (httpx.RequestError, asyncio.TimeoutError) as exc:
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
