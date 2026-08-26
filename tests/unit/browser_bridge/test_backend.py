from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from local_operator.browser_bridge import state
from local_operator.browser_bridge.backend import (
    BridgeClient,
    BridgeError,
    BridgeUnreachable,
    client_timeout,
    format_error,
)
from local_operator.browser_bridge.protocol import (
    COMMAND_TIMEOUTS,
    ORIGIN_PROMPT_TIMEOUT_MS,
    ORIGIN_PROMPT_WINDOW_S,
    PROTO_VERSION,
    ErrorCode,
)


def publish(tmp_path: Path) -> state.BridgeState:
    current = state.BridgeState(
        pid=os.getpid(),
        port=4099,
        session_key="s" * 32,
        proto=PROTO_VERSION,
        extension_connected=True,
        paired=True,
    )
    state.publish(current, tmp_path)
    return current


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        (ErrorCode.EXTENSION_DISCONNECTED, "extension not connected"),
        (ErrorCode.NOT_PAIRED, "lop browser pair"),
        (ErrorCode.TAB_CLOSED, "Use 'open'"),
        (ErrorCode.NAV_FAILED, "navigation failed"),
        (ErrorCode.NAV_TIMEOUT, "navigation did not complete"),
        (ErrorCode.ELEMENT_NOT_FOUND, "new snapshot"),
        (ErrorCode.ORIGIN_DENIED, "Do not retry"),
        (ErrorCode.ORIGIN_PROMPT_PENDING, "waiting for the user"),
        (ErrorCode.DEBUGGER_CONFLICT, "close DevTools"),
        (ErrorCode.BUSY, "busy"),
        (ErrorCode.PROTO_MISMATCH, "protocol mismatch"),
        (ErrorCode.INTERNAL, "browser bridge error"),
    ],
)
def test_error_taxonomy(code: ErrorCode, fragment: str) -> None:
    error = BridgeError(code, "detail", {"origin": "https://example.com"})
    assert fragment in format_error(error, action="goto", surface="bridge:1:n")


@pytest.mark.asyncio
async def test_missing_state_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(BridgeUnreachable, match="lop browser status"):
        await BridgeClient(tmp_path).call("status", {})


def test_client_timeout_outlives_daemon_budget_per_method() -> None:
    # Timeout-chain invariant (finding A3): extension deny < daemon prompt
    # window < client HTTP timeout. The client must survive the daemon's
    # worst case (base + prompt window) for EVERY method, or a command
    # blocked on the extension's approval popup dies client-side as a fake
    # "unreachable" while the daemon is healthy (QA transcript 0ee4974ba84a).
    assert ORIGIN_PROMPT_TIMEOUT_MS / 1000 < ORIGIN_PROMPT_WINDOW_S
    for method, base in COMMAND_TIMEOUTS.items():
        assert client_timeout(method) > base + ORIGIN_PROMPT_WINDOW_S


def test_client_timeout_unknown_method_takes_conservative_max() -> None:
    # The real invariant is >= every known method's budget, not equality with
    # any particular method (which would break silently if a longer one were
    # ever added).
    assert client_timeout("no-such-method") >= max(
        client_timeout(method) for method in COMMAND_TIMEOUTS
    )


def _client_with_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler: object
) -> BridgeClient:
    """Route the client's internal AsyncClient through a mock transport."""
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)  # type: ignore[arg-type]
        return real_async_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    return BridgeClient(tmp_path)


@pytest.mark.parametrize(
    "transport_error",
    [
        # Refused TCP connection: daemon genuinely gone.
        httpx.ConnectError("connection refused"),
        # Timeout WHILE connecting: nothing was reached; must not get the
        # popup message despite being a TimeoutException (finding m1).
        httpx.ConnectTimeout("connect deadline"),
        # Daemon died mid-request after accepting the connection: restart
        # advice is right, popup advice would be a lie (finding m2).
        httpx.ReadError("connection reset"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
    ids=["connect-refused", "connect-timeout", "read-error", "protocol-error"],
)
@pytest.mark.asyncio
async def test_transport_failures_report_daemon_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport_error: Exception
) -> None:
    publish(tmp_path)

    def fail(_request: httpx.Request) -> httpx.Response:
        raise transport_error

    client = _client_with_transport(tmp_path, monkeypatch, fail)
    with pytest.raises(BridgeUnreachable, match="unreachable.*not.*answering") as excinfo:
        await client.call("status", {})
    assert "popup" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_timeout_after_connect_names_the_popup_not_the_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A read timeout AFTER connecting means the daemon accepted the command
    # and is (legitimately) still waiting — most likely on a human origin
    # decision. The old flat 35 s timeout mapped this to "unreachable" and a
    # QA session burned an hour restarting a healthy daemon; the message must
    # point at the extension popup instead.
    publish(tmp_path)

    def hang(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read deadline")

    client = _client_with_transport(tmp_path, monkeypatch, hang)
    with pytest.raises(BridgeUnreachable, match="extension.*popup") as excinfo:
        await client.call("goto", {"url": "https://example.com"})
    assert "unreachable" not in str(excinfo.value)
    # Honesty check: we only know the connection was accepted, so the message
    # must not assert the daemon "is running" (finding m2).
    assert "is running" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_typed_error_still_maps_to_bridge_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The point of outliving the daemon's budget: its typed answer (here an
    # origin_denied) must surface as a BridgeError, never a transport failure.
    publish(tmp_path)

    def denied(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": body["id"],
                "ok": False,
                "error": {
                    "code": "origin_denied",
                    "message": "denied",
                    "data": {"origin": "https://example.com"},
                },
            },
        )

    client = _client_with_transport(tmp_path, monkeypatch, denied)
    with pytest.raises(BridgeError) as excinfo:
        await client.call("goto", {"url": "https://example.com"})
    assert excinfo.value.code == ErrorCode.ORIGIN_DENIED


def test_origin_not_allowed_error_teaches_the_access_flow() -> None:
    # The early-fail error is the agent's ONLY discovery path for the approval
    # dance mid-session, so it must name both follow-up actions and echo the
    # url; the failure it replaces read as a bridge outage (a session burned an
    # hour misdiagnosing it).
    error = BridgeError(
        ErrorCode.ORIGIN_NOT_ALLOWED,
        "site https://example.com is not allowed yet",
        {"origin": "https://example.com", "url": "https://example.com/page"},
    )
    text = format_error(error, action="open")
    assert "site https://example.com is not allowed yet" in text
    assert "action='request_access' url=https://example.com/page" in text
    assert "action='await_access' url=https://example.com/page" in text
    assert "extension popup" in text
    # The agent is the PRIMARY notification channel: Chrome's banner is
    # best-effort on macOS, so the instruction to notify must be explicit.
    assert "NOTIFY THE USER" in text
