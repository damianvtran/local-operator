"""OAuth flow tests: PKCE vectors, callback-server round trip, device-code
polling, JWT identity decoding, endpoint validation. No network."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from typing import Any

import httpx
import pytest

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    CallbackFlowOptions,
    ConfigurationError,
    LoginCallbacks,
    LoginCancelledError,
    OAuthCallbackFlow,
)
from local_operator.providers.oauth.device_code import DevicePollResult, poll_device_code_flow
from local_operator.providers.oauth.openai import (
    decode_jwt_claims,
    identity_from_id_token,
)
from local_operator.providers.oauth.pkce import create_pkce_challenge, create_pkce_pair
from local_operator.providers.oauth.xai import validate_xai_endpoint
from local_operator.providers.oauth.callback_server import LoginError

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------


def test_pkce_challenge_rfc7636_known_vector() -> None:
    """The RFC 7636 appendix B vector must round-trip exactly."""
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert create_pkce_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_pkce_pair_verifier_shape() -> None:
    verifier, challenge = create_pkce_pair()
    # 96 random bytes → 128 base64url chars, no padding.
    raw = base64.urlsafe_b64decode(verifier + "=" * (-len(verifier) % 4))
    assert len(raw) == 96
    assert "=" not in verifier and "+" not in verifier and "/" not in verifier
    digest = hashlib.sha256(verifier.encode()).digest()
    assert challenge == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ---------------------------------------------------------------------------
# Callback server
# ---------------------------------------------------------------------------


class _EchoFlow(OAuthCallbackFlow):
    """Test flow: captures state/redirect_uri, returns the code verbatim."""

    def __init__(self, options: CallbackFlowOptions, callbacks: LoginCallbacks, **kwargs: Any) -> None:
        super().__init__(options, callbacks, open_browser=lambda url: None, **kwargs)
        self.generated: dict[str, str] = {}

    async def generate_auth_url(self, state: str, redirect_uri: str) -> str:
        self.generated = {"state": state, "redirect_uri": redirect_uri}
        return f"https://idp.example/authorize?state={state}&redirect_uri={redirect_uri}"

    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        return {"code": code, "state": state, "redirect_uri": redirect_uri}


async def test_callback_server_captures_code_from_simulated_redirect() -> None:
    flow = _EchoFlow(CallbackFlowOptions(preferred_port=0), LoginCallbacks())
    flow.options.preferred_port = 0

    async def run_and_redirect() -> tuple[dict[str, Any], str]:
        task = asyncio.create_task(flow.run())
        # Wait until the server bound AND the auth URL was generated
        # (the server starts BEFORE the URL so the bound port lands in it).
        for _ in range(400):
            if flow.bound_port is not None and flow.generated:
                break
            await asyncio.sleep(0.01)
        assert flow.bound_port is not None and flow.generated
        state = flow.generated["state"]
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://127.0.0.1:{flow.bound_port}/callback",
                params={"code": "auth-code-123", "state": state},
            )
        assert response.status_code == 200
        return await task, state

    result, state = await asyncio.wait_for(run_and_redirect(), timeout=10)
    assert result["code"] == "auth-code-123"
    assert result["state"] == state
    # redirect_uri carried the ACTUALLY bound port.
    assert f":{flow.bound_port}/callback" in result["redirect_uri"]


async def test_launch_route_redirects_to_auth_url() -> None:
    flow = _EchoFlow(CallbackFlowOptions(preferred_port=0), LoginCallbacks())
    visited: dict[str, Any] = {}

    async def drive() -> None:
        task = asyncio.create_task(flow.run())
        for _ in range(400):
            if flow.bound_port is not None and flow.generated:
                break
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient(follow_redirects=False) as client:
            visited["launch"] = await client.get(f"http://127.0.0.1:{flow.bound_port}/launch")
            # Complete the flow so run() returns.
            await client.get(
                f"http://127.0.0.1:{flow.bound_port}/callback",
                params={"code": "c", "state": flow.generated["state"]},
            )
        await task

    await asyncio.wait_for(drive(), timeout=10)
    assert visited["launch"].status_code == 302
    location = visited["launch"].headers["location"]
    assert location.startswith("https://idp.example/authorize")


async def test_manual_input_only_pastes_code_without_server() -> None:
    pasted = asyncio.Event()

    async def paste() -> str:
        await asyncio.sleep(0.05)
        pasted.set()
        return "manual-code#manual-state"  # fragment splits into code#state

    callbacks = LoginCallbacks(on_manual_code_input=paste)
    flow = _EchoFlow(
        CallbackFlowOptions(preferred_port=0, manual_input_only=True), callbacks
    )
    result = await asyncio.wait_for(flow.run(), timeout=10)
    assert result["code"] == "manual-code"
    assert result["state"] == "manual-state"
    assert flow.bound_port is None  # no server in manual mode


async def test_pinned_port_fails_before_browser_when_busy() -> None:
    """A provider-pinned port that is busy must fail fast — no browser."""
    blocker = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    busy_port = int(blocker.sockets[0].getsockname()[1])
    opened: list[str] = []
    try:
        flow = _EchoFlow(
            CallbackFlowOptions(preferred_port=busy_port, allow_port_fallback=False),
            LoginCallbacks(),
            open_browser=opened.append,
        )
        with pytest.raises(ConfigurationError):
            await flow.run()
        assert opened == []  # browser never opened
    finally:
        blocker.close()
        await blocker.wait_closed()


async def test_port_fallback_when_allowed() -> None:
    blocker = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    busy_port = int(blocker.sockets[0].getsockname()[1])
    try:
        flow = _EchoFlow(CallbackFlowOptions(preferred_port=busy_port, allow_port_fallback=True), LoginCallbacks())

        async def drive() -> dict[str, Any]:
            task = asyncio.create_task(flow.run())
            for _ in range(400):
                if flow.bound_port is not None and flow.generated:
                    break
                await asyncio.sleep(0.01)
            assert flow.bound_port != busy_port  # fell back to an OS port
            assert flow.generated
            async with httpx.AsyncClient() as client:
                await client.get(
                    f"http://127.0.0.1:{flow.bound_port}/callback",
                    params={"code": "c", "state": flow.generated["state"]},
                )
            return await task

        result = await asyncio.wait_for(drive(), timeout=10)
        assert result["code"] == "c"
    finally:
        blocker.close()
        await blocker.wait_closed()


async def test_abort_signal_cancels_login() -> None:
    signal = AbortSignal()
    flow = _EchoFlow(CallbackFlowOptions(preferred_port=0), LoginCallbacks(), signal=signal)

    async def drive() -> None:
        task = asyncio.create_task(flow.run())
        for _ in range(200):
            if flow.bound_port is not None:
                break
            await asyncio.sleep(0.01)
        signal.abort("user hit ctrl-c")
        with pytest.raises(LoginCancelledError):
            await task

    await asyncio.wait_for(drive(), timeout=10)


# ---------------------------------------------------------------------------
# Device code polling
# ---------------------------------------------------------------------------


async def test_device_poll_pending_slow_down_success() -> None:
    outcomes = [
        DevicePollResult.pending(),
        DevicePollResult.slow_down(),
        DevicePollResult.complete({"access_token": "tok"}),
    ]
    calls: list[float] = []
    import time

    async def poll() -> DevicePollResult[dict[str, Any]]:
        calls.append(time.monotonic())
        return outcomes[len(calls) - 1]

    result = await poll_device_code_flow(poll, interval_seconds=1, expires_in_seconds=60)
    assert result == {"access_token": "tok"}
    assert len(calls) == 3
    # First poll immediate; second after ≥1s; third after slow_down (+5s ⇒ ≥6s
    # from poll 2). Use generous bounds: slow_down MUST add 5s.
    assert calls[1] - calls[0] >= 0.9
    assert calls[2] - calls[1] >= 5.5


async def test_device_poll_denied_and_expired() -> None:
    async def denied() -> DevicePollResult[dict[str, Any]]:
        return DevicePollResult.failed("access_denied")

    with pytest.raises(LoginError, match="access_denied"):
        await poll_device_code_flow(denied, interval_seconds=1, expires_in_seconds=60)


async def test_device_poll_expiry_raises_timeout() -> None:
    async def pending_forever() -> DevicePollResult[dict[str, Any]]:
        return DevicePollResult.pending()

    with pytest.raises(Exception) as excinfo:
        await poll_device_code_flow(pending_forever, interval_seconds=1, expires_in_seconds=0.2)
    from local_operator.providers.oauth.callback_server import LoginTimeoutError

    assert isinstance(excinfo.value, LoginTimeoutError)


async def test_device_poll_abort() -> None:
    signal = AbortSignal()

    async def pending_forever() -> DevicePollResult[dict[str, Any]]:
        return DevicePollResult.pending()

    async def abort_soon() -> None:
        await asyncio.sleep(0.1)
        signal.abort("cancelled")

    task = asyncio.create_task(abort_soon())
    with pytest.raises(LoginCancelledError):
        await poll_device_code_flow(pending_forever, interval_seconds=5, expires_in_seconds=60, signal=signal)
    await task


# ---------------------------------------------------------------------------
# JWT identity (no signature verification — never add PyJWT)
# ---------------------------------------------------------------------------


def _jwt(payload: dict[str, Any]) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.sig"


def test_decode_jwt_claims_without_verification() -> None:
    token = _jwt({"sub": "user-1", "email": "user@example.com"})
    claims = decode_jwt_claims(token)
    assert claims["sub"] == "user-1"
    assert claims["email"] == "user@example.com"


def test_openai_identity_from_id_token() -> None:
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acc-42",
                "chatgpt_plan_type": "pro",
            },
            "https://api.openai.com/profile": {"email": "user@example.com"},
        }
    )
    identity = identity_from_id_token(token)
    assert identity["account_id"] == "acc-42"
    assert identity["org_id"] == "acc-42"  # orgId = accountId (required)
    assert identity["org_name"] == "pro"
    assert identity["email"] == "user@example.com"


def test_openai_identity_requires_account_id() -> None:
    token = _jwt({"https://api.openai.com/profile": {"email": "user@example.com"}})
    with pytest.raises(LoginError, match="account id"):
        identity_from_id_token(token)


# ---------------------------------------------------------------------------
# xAI endpoint validation (https-only, host-pinned)
# ---------------------------------------------------------------------------


def test_validate_xai_endpoint_accepts_x_ai_hosts() -> None:
    assert validate_xai_endpoint("https://auth.x.ai/oauth2/token") == "https://auth.x.ai/oauth2/token"
    assert validate_xai_endpoint("https://sso.x.ai/token").startswith("https://")


def test_validate_xai_endpoint_rejects_other_hosts_and_http() -> None:
    with pytest.raises(LoginError):
        validate_xai_endpoint("http://auth.x.ai/oauth2/token")  # not https
    with pytest.raises(LoginError):
        validate_xai_endpoint("https://evil.example/oauth2/token")  # wrong host
    with pytest.raises(LoginError):
        validate_xai_endpoint("https://notx.ai/token")  # suffix trick


# ---------------------------------------------------------------------------
# Registry wiring: login thunks resolve to real flows lazily
# ---------------------------------------------------------------------------


async def test_registry_login_thunks_import_lazily() -> None:
    from local_operator.providers.registry import get_provider_definition

    for provider_id in ("openai", "anthropic", "kimi", "xai-oauth"):
        definition = get_provider_definition(provider_id)
        assert definition is not None and definition.login is not None
