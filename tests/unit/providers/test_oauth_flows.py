"""OAuth flow tests: PKCE vectors, callback-server round trip, device-code
polling, JWT identity decoding, endpoint validation. No network."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import urllib.parse
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

    def __init__(
        self, options: CallbackFlowOptions, callbacks: LoginCallbacks, **kwargs: Any
    ) -> None:
        kwargs.setdefault("open_browser", lambda url: None)
        super().__init__(options, callbacks, **kwargs)
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
    flow = _EchoFlow(CallbackFlowOptions(preferred_port=0, manual_input_only=True), callbacks)
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
        flow = _EchoFlow(
            CallbackFlowOptions(preferred_port=busy_port, allow_port_fallback=True),
            LoginCallbacks(),
        )

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
        await poll_device_code_flow(
            pending_forever, interval_seconds=5, expires_in_seconds=60, signal=signal
        )
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
    assert (
        validate_xai_endpoint("https://auth.x.ai/oauth2/token") == "https://auth.x.ai/oauth2/token"
    )
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


# ---------------------------------------------------------------------------
# PR-26 fidelity pins
# ---------------------------------------------------------------------------


async def test_anthropic_authorize_uses_claude_ai_host() -> None:
    """The authorize URL must be claude.ai (issues inference scopes), never
    platform.claude.com (issues console tokens without user:inference)."""
    from local_operator.providers.oauth.anthropic import AnthropicOAuthFlow

    flow = AnthropicOAuthFlow(LoginCallbacks(), open_browser=lambda url: None)
    url = await flow.generate_auth_url("state-1", "http://localhost:54545/callback")
    assert url.startswith("https://claude.ai/oauth/authorize")
    assert "user%3Ainference" in url  # scope list is url-encoded


async def test_openai_redirect_uri_pinned_to_1455() -> None:
    """OpenAI allowlists the literal http://localhost:1455/auth/callback;
    no port fallback, no 127.0.0.1 rewrite."""
    from local_operator.providers.oauth.openai import OpenAIOAuthFlow, REDIRECT_URI

    flow = OpenAIOAuthFlow(LoginCallbacks(), open_browser=lambda url: None)
    assert REDIRECT_URI == "http://localhost:1455/auth/callback"
    assert flow.redirect_uri() == REDIRECT_URI
    url = await flow.generate_auth_url("state-1", flow.redirect_uri())
    assert "localhost%3A1455%2Fauth%2Fcallback" in url


async def test_callback_server_rejects_state_mismatch() -> None:
    """PR-13: a code returned with the wrong state fails the login."""
    flow = _EchoFlow(CallbackFlowOptions(preferred_port=0), LoginCallbacks())

    async def drive() -> None:
        task = asyncio.create_task(flow.run())
        for _ in range(400):
            if flow.bound_port is not None and flow.generated:
                break
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient() as client:
            await client.get(
                f"http://127.0.0.1:{flow.bound_port}/callback",
                params={"code": "stolen-code", "state": "attacker-state"},
            )
        with pytest.raises(LoginError, match="state mismatch"):
            await task

    await asyncio.wait_for(drive(), timeout=10)


async def test_openai_exchange_sends_state() -> None:
    """PR-13: the token exchange echoes the verified state."""
    from local_operator.providers.oauth.openai import OpenAIOAuthFlow, TOKEN_URL, REDIRECT_URI

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = dict(urllib.parse.parse_qsl(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "access_token": "a",
                "refresh_token": "r",
                "expires_in": 3600,
                "id_token": _jwt(
                    {
                        "https://api.openai.com/auth": {
                            "chatgpt_account_id": "acct-1",
                            "chatgpt_plan_type": "pro",
                        },
                        "https://api.openai.com/profile": {"email": "u@example.com"},
                    }
                ),
            },
        )

    transport = httpx.MockTransport(handler)
    flow = OpenAIOAuthFlow(
        LoginCallbacks(),
        open_browser=lambda url: None,
        http_client=httpx.AsyncClient(transport=transport),
    )
    result = await flow.exchange_token("code-1", "state-xyz", REDIRECT_URI)
    assert captured["body"]["state"] == "state-xyz"
    assert captured["url"] == TOKEN_URL
    assert result["org_id"] == "acct-1"


async def test_callback_missing_code_fails_promptly() -> None:
    """PR-14: a redirect carrying neither code nor error fails the login
    immediately — no 300 s hang."""
    flow = _EchoFlow(CallbackFlowOptions(preferred_port=0, timeout_seconds=120), LoginCallbacks())

    async def drive() -> None:
        task = asyncio.create_task(flow.run())
        for _ in range(400):
            if flow.bound_port is not None and flow.generated:
                break
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient() as client:
            await client.get(f"http://127.0.0.1:{flow.bound_port}/callback", params={})
        with pytest.raises(LoginError, match="neither a code nor an error"):
            await task

    await asyncio.wait_for(drive(), timeout=10)


def test_openai_credentials_apply_five_minute_skew() -> None:
    """PR-08: minted expiry is expires_in minus the 5-minute skew."""
    import time

    from local_operator.providers.oauth.openai import _credentials_from_token, EXPIRY_SKEW_MS

    now = int(time.time() * 1000)
    creds = _credentials_from_token(
        {
            "access_token": "a",
            "refresh_token": "r",
            "expires_in": 3600,
            "id_token": _jwt(
                {
                    "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"},
                }
            ),
        }
    )
    assert creds["expires"] <= now + 3600 * 1000 - EXPIRY_SKEW_MS + 5000
    assert creds["org_id"] == "acct-1"


def test_openai_login_fails_hard_without_identity() -> None:
    """PR-08: no id_token ⇒ no account id ⇒ login fails, no stored row."""
    from local_operator.providers.oauth.openai import _credentials_from_token

    with pytest.raises(LoginError, match="id_token"):
        _credentials_from_token({"access_token": "a", "refresh_token": "r", "expires_in": 3600})


def test_kimi_device_id_created_0600(tmp_path: Any) -> None:
    """PR-11: the persisted device id file is 0600 and stable."""
    import os
    import stat

    from local_operator.providers.oauth.kimi import get_or_create_device_id

    first = get_or_create_device_id(tmp_path)
    second = get_or_create_device_id(tmp_path)
    assert first == second
    mode = stat.S_IMODE(os.stat(tmp_path / "kimi-device-id").st_mode)
    assert mode == 0o600


def test_kimi_common_headers_carry_device_identity() -> None:
    """PR-26: the X-Msh-* fingerprint headers are complete."""
    from local_operator.providers.oauth.kimi import kimi_common_headers

    headers = kimi_common_headers(device_id="dev-id-1")
    for name in (
        "User-Agent",
        "X-Msh-Platform",
        "X-Msh-Version",
        "X-Msh-Device-Name",
        "X-Msh-Device-Model",
        "X-Msh-Os-Version",
        "X-Msh-Device-Id",
    ):
        assert name in headers, name
    assert headers["X-Msh-Device-Id"] == "dev-id-1"
    assert headers["X-Msh-Platform"] == "kimi_cli"


async def test_kimi_slow_down_updates_poller_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """PR-10: a slow_down payload with a larger interval raises the poller's
    interval via the mutable holder."""
    from local_operator.providers.oauth.kimi import (
        DEFAULT_AUTH_HOST,
        DEVICE_AUTHORIZATION_PATH,
        TOKEN_PATH,
        login_kimi,
    )

    polls: list[float] = []
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == DEVICE_AUTHORIZATION_PATH:
            return httpx.Response(
                200,
                json={
                    "device_code": "dc-1",
                    "user_code": "ABCD",
                    "verification_uri": "https://auth.kimi.com/verify",
                    "interval": 1,
                    "expires_in": 120,
                },
            )
        assert request.url.path == TOKEN_PATH
        polls.append(asyncio.get_running_loop().time())
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(400, json={"error": "slow_down", "interval": 2})
        return httpx.Response(
            200, json={"access_token": "a", "refresh_token": "r", "expires_in": 3600}
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", "/tmp/kimi-test-config")
    creds = await login_kimi(
        LoginCallbacks(),
        http_client=httpx.AsyncClient(transport=transport),
        config_dir=None,
    )
    assert creds["access"] == "a"
    assert len(polls) == 2
    # slow_down bumped the interval from 1s to the provider's 2s.
    assert polls[1] - polls[0] >= 1.9


def test_xai_refresh_keeps_old_refresh_token_when_response_omits_one() -> None:
    """PR-09: xai refresh falls back to the stored refresh token."""
    import time

    from local_operator.providers.oauth.xai import _credentials_from_token

    creds = _credentials_from_token(
        {"access_token": "new-access", "expires_in": 3600},
        "https://auth.x.ai/oauth2/token",
        old_refresh="old-refresh",
    )
    assert creds["refresh"] == "old-refresh"
    assert creds["expires"] < int(time.time() * 1000) + 3600_000


async def test_paste_prompt_gated_to_paste_code_flow_providers() -> None:
    """PR-02: the CLI attaches the paste prompt ONLY for paste_code_flow
    providers, and it is async (loop-friendly)."""
    from local_operator.providers.auth_cli import _callbacks_interactive
    from local_operator.providers.registry import get_provider_definition

    anthropic_def = get_provider_definition("anthropic")
    assert anthropic_def is not None and anthropic_def.paste_code_flow
    paste_callbacks = _callbacks_interactive(anthropic_def)
    assert paste_callbacks.on_manual_code_input is not None
    assert asyncio.iscoroutinefunction(paste_callbacks.on_manual_code_input)

    openai_def = get_provider_definition("openai")
    assert openai_def is not None and not openai_def.paste_code_flow
    assert _callbacks_interactive(openai_def).on_manual_code_input is None


async def test_anthropic_login_via_browser_redirect_without_paste(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-02 end-to-end: an anthropic login completes from the simulated
    browser redirect alone, with NO paste callback attached."""
    from local_operator.providers.oauth.anthropic import (
        AUTHORIZE_URL,
        CALLBACK_PORT,
        TOKEN_URL,
        login_anthropic,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == TOKEN_URL
        return httpx.Response(
            200,
            json={
                "access_token": "oauth-token",
                "refresh_token": "refresh-1",
                "expires_in": 3600,
                "account": {"uuid": "acct-1", "email_address": "user@example.com"},
                "organization": {"uuid": "org-1", "name": "Claude"},
            },
        )

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    auth_urls: list[str] = []
    opened: list[str] = []

    # Port 54545 may be busy on CI; allow fallback via a subclass run.
    from local_operator.providers.oauth.anthropic import AnthropicOAuthFlow

    flow = AnthropicOAuthFlow(
        LoginCallbacks(on_auth_url=lambda url, instructions=None: auth_urls.append(url)),
        open_browser=opened.append,
        http_client=http,
    )
    flow.options.preferred_port = 0  # test-local port; redirect_uri carries it

    async def drive() -> Any:
        task = asyncio.create_task(flow.run())
        for _ in range(400):
            if flow.bound_port is not None and auth_urls:
                break
            await asyncio.sleep(0.01)
        state = urllib.parse.parse_qs(urllib.parse.urlsplit(auth_urls[0]).query)["state"][0]
        async with httpx.AsyncClient() as browser:
            response = await browser.get(
                f"http://127.0.0.1:{flow.bound_port}/callback",
                params={"code": "auth-code", "state": state},
            )
        assert response.status_code == 200
        return await task

    creds = await asyncio.wait_for(drive(), timeout=15)
    assert creds["access"] == "oauth-token"
    assert creds["org_id"] == "org-1"
    assert auth_urls[0].startswith(AUTHORIZE_URL)
    await http.aclose()
