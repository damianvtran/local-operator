"""OAuth flow tests: PKCE vectors, callback-server round trip, device-code
polling, JWT identity decoding, endpoint validation. No network."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    CallbackFlowOptions,
    ConfigurationError,
    LoginCallbacks,
    LoginCancelledError,
    LoginError,
    OAuthCallbackFlow,
)
from local_operator.providers.oauth.device_code import (
    DevicePollResult,
    poll_device_code_flow,
)
from local_operator.providers.oauth.openai import (
    decode_jwt_claims,
    identity_from_id_token,
)
from local_operator.providers.oauth.pkce import create_pkce_challenge, create_pkce_pair
from local_operator.providers.oauth.xai import validate_xai_endpoint

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
    from local_operator.providers.oauth.openai import REDIRECT_URI, OpenAIOAuthFlow

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


async def test_openai_exchange_omits_state() -> None:
    """OpenAI's token endpoint rejects unknown parameters outright (400
    ``invalid_request_error`` / ``unknown_parameter: state``), so the browser
    exchange must NOT send ``state``. The device-code flow never did. CSRF
    protection is unaffected: state is verified at the callback, before the code
    is trusted (:func:`test_callback_server_rejects_state_mismatch`)."""
    from local_operator.providers.oauth.openai import (
        REDIRECT_URI,
        TOKEN_URL,
        OpenAIOAuthFlow,
    )

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
    # Drive the real pair so the PKCE verifier exists, as it does in the flow.
    await flow.generate_auth_url("state-xyz", REDIRECT_URI)
    result = await flow.exchange_token("code-1", "state-xyz", REDIRECT_URI)
    assert "state" not in captured["body"]
    assert captured["body"]["code_verifier"]
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

    from local_operator.providers.oauth.openai import (
        EXPIRY_SKEW_MS,
        _credentials_from_token,
    )

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


async def test_paste_prompt_gated_to_providers_that_accept_one() -> None:
    """PR-02: the CLI attaches the paste prompt to providers that ACCEPT one,
    and to no others, and it is async (loop-friendly).

    Three cases, because there are three and reading them as two is the defect:
    a provider whose login IS the paste (``alibaba``), Anthropic's optional
    fallback beside its loopback flow, and a loopback-only provider where a
    prompt would race the HTTP callback and block the terminal.
    """
    from local_operator.providers.auth_cli import _callbacks_interactive
    from local_operator.providers.registry import get_provider_definition

    anthropic_def = get_provider_definition("anthropic")
    assert anthropic_def is not None and anthropic_def.paste_code_flow
    paste_callbacks = _callbacks_interactive(anthropic_def)
    assert paste_callbacks.on_manual_code_input is not None
    assert asyncio.iscoroutinefunction(paste_callbacks.on_manual_code_input)

    # The regression this file previously asserted the wrong way round: a
    # paste-a-key provider has no loopback path at all, so a host that attaches
    # no prompt makes its login fail every single time.
    alibaba_def = get_provider_definition("alibaba")
    assert alibaba_def is not None and alibaba_def.paste_prompt_required
    key_callbacks = _callbacks_interactive(alibaba_def)
    assert key_callbacks.on_manual_code_input is not None
    assert asyncio.iscoroutinefunction(key_callbacks.on_manual_code_input)

    openai_def = get_provider_definition("openai")
    assert openai_def is not None and not openai_def.accepts_paste_prompt
    assert _callbacks_interactive(openai_def).on_manual_code_input is None


async def test_every_paste_key_provider_can_actually_be_logged_into() -> None:
    """The class-level regression: `/login <provider>` must never be a flow
    that can only fail.

    Enumerates every provider offering an interactive login and drives each
    one's registered login callable with the CLI's own callbacks, rather than
    checking the two the bug report happened to name. The reported failure
    ("QwenCloud Token Plan login requires an interactive key prompt") was one
    of EIGHT providers in this state, and a test naming providers individually
    is exactly what let the other seven ship.

    Only the paste-a-key providers are driven to completion here: they are the
    ones whose whole login is local. A loopback/device provider is asserted to
    be offered no prompt, which is the other half of the contract.
    """
    from local_operator.providers.auth_cli import _callbacks_interactive
    from local_operator.providers.registry import PROVIDER_REGISTRY

    drivable = [
        p
        for p in PROVIDER_REGISTRY
        # The QwenCloud Token Plan requires a paste too, but continues into a
        # device flow that would reach the network; it is driven with a mock
        # transport in ``test_qwencloud_token_plan_login_captures_key_and_device_grant``.
        if p.login is not None and p.paste_prompt_required and p.id != "alibaba-token-plan-oauth"
    ]
    # A non-zero control: if the filter ever matches nothing (a renamed field, a
    # registry refactor) the loop below would pass by doing nothing at all.
    assert len(drivable) >= 8, [p.id for p in drivable]

    for definition in drivable:
        callbacks = _callbacks_interactive(definition)
        assert callbacks.on_manual_code_input is not None, definition.id
        # Substitute the terminal read; everything else is the shipped path,
        # including the registry's own login callable.
        callbacks.on_manual_code_input = lambda: "sk-pasted-key"
        login = definition.login
        assert login is not None, definition.id
        assert await login(callbacks) == "sk-pasted-key", definition.id


def test_cli_hides_an_api_key_and_echoes_an_oauth_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 1 F2: an API key must not be echoed into the terminal scrollback.

    ``CredentialManager`` and the web-search CLI already read this same class of
    value through ``getpass``; the login prompt used ``input()``, and making the
    paste-a-key providers work is what made that path reachable for nine real
    provider keys.

    The OAuth code branch deliberately stays echoed: it is single-use, expires
    in minutes, and is a long opaque string the user needs to read back to check
    the paste landed whole. Asserting BOTH is what makes this a discriminating
    test rather than one that would pass on a blanket change either way.
    """
    import getpass as getpass_mod

    from local_operator.providers.auth_cli import _callbacks_interactive
    from local_operator.providers.registry import get_provider_definition

    used: list[str] = []
    monkeypatch.setattr(getpass_mod, "getpass", lambda *a, **k: used.append("getpass") or "sk-x")
    monkeypatch.setattr("builtins.input", lambda *a, **k: used.append("input") or "code#state")

    key_def = get_provider_definition("alibaba")
    assert key_def is not None
    prompt = _callbacks_interactive(key_def).on_manual_code_input
    assert prompt is not None
    assert asyncio.run(_maybe(prompt())) == "sk-x"
    assert used == ["getpass"], used

    used.clear()
    code_def = get_provider_definition("anthropic")
    assert code_def is not None
    prompt = _callbacks_interactive(code_def).on_manual_code_input
    assert prompt is not None
    assert asyncio.run(_maybe(prompt())) == "code#state"
    assert used == ["input"], used


async def _maybe(value: Any) -> Any:
    """Await ``value`` when it is awaitable; the prompts are coroutines."""
    if inspect.isawaitable(value):
        return await value
    return value


async def test_paste_key_login_treats_an_empty_paste_as_a_cancel() -> None:
    """An empty submit must not become a stored blank credential.

    A blank ``api_key`` row is worse than no row: it shadows a working
    environment key in the stream-time cascade, so every later request fails to
    authenticate with nothing on screen to explain why.
    """
    from local_operator.providers.oauth.callback_server import LoginCancelledError
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition("alibaba")
    assert definition is not None and definition.login is not None

    for pasted in ("", "   ", None):
        callbacks = LoginCallbacks(
            on_auth_url=lambda url, instructions=None: None,
            on_manual_code_input=lambda value=pasted: value,
        )
        with pytest.raises(LoginCancelledError):
            await definition.login(callbacks)


async def test_paste_key_login_reports_a_missing_prompt_without_opening_a_browser() -> None:
    """A host that provides no prompt is told which hook it is missing, and the
    user is not first sent to a dashboard to fetch a key that will be refused.

    The ordering is the assertion: before this, the URL was surfaced and only
    then did the flow discover it could not read anything.
    """
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition("alibaba")
    assert definition is not None and definition.login is not None

    urls: list[str] = []
    callbacks = LoginCallbacks(on_auth_url=lambda url, instructions=None: urls.append(url))
    with pytest.raises(ValueError, match="on_manual_code_input"):
        await definition.login(callbacks)
    assert urls == [], "the browser must not be opened for a login that cannot complete"


async def test_anthropic_login_via_browser_redirect_without_paste(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-02 end-to-end: an anthropic login completes from the simulated
    browser redirect alone, with NO paste callback attached."""
    from local_operator.providers.oauth.anthropic import AUTHORIZE_URL, TOKEN_URL

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


# -- the loopback is the real path; no human types a code --------------------


@pytest.mark.asyncio
async def test_the_loopback_completes_a_login_with_no_manual_input() -> None:
    """The whole point of the callback server: authorize in the browser and the
    flow finishes on its own.

    Reported defect: `/login anthropic` printed a URL and then demanded a pasted
    code. Anthropic IS a loopback provider — it redirects to
    `http://localhost:54545/callback` — so a code prompt should never be the
    path. This drives the real flow with a stubbed token endpoint and supplies
    NO manual-input callback at all.
    """
    from local_operator.providers.oauth.anthropic import login_anthropic
    from local_operator.providers.oauth.callback_server import LoginCallbacks

    seen: dict[str, str] = {}
    progress: list[str] = []

    def on_auth_url(url: str, instructions: str | None = None) -> None:
        seen["url"] = url

    async def browser() -> int:
        for _ in range(200):
            if "url" in seen:
                break
            await asyncio.sleep(0.02)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(seen["url"]).query)
        # The redirect target must be a loopback URL the browser can actually
        # reach, not a provider-hosted page that shows a code.
        assert query["redirect_uri"][0].startswith("http://localhost:")
        async with httpx.AsyncClient(timeout=10.0) as http:
            response = await http.get(
                query["redirect_uri"][0],
                params={"code": "auth-code", "state": query["state"][0]},
            )
        return response.status_code

    class StubTokens(httpx.AsyncClient):
        async def post(self, url, **kwargs):  # noqa: ANN001, ANN003
            return httpx.Response(
                200,
                json={
                    "access_token": "at",
                    "refresh_token": "rt",
                    "expires_in": 3600,
                    "account": {"uuid": "u1", "email_address": "you@example.com"},
                },
                request=httpx.Request("POST", url),
            )

        async def get(self, url, **kwargs):  # noqa: ANN001, ANN003
            return httpx.Response(200, json={}, request=httpx.Request("GET", url))

    credentials, status = await asyncio.gather(
        login_anthropic(
            LoginCallbacks(on_auth_url=on_auth_url, on_progress=progress.append),
            open_browser=lambda _url: None,
            http_client=StubTokens(),
        ),
        browser(),
    )

    assert status == 200, "the browser's redirect must be answered by our server"
    assert credentials["access"] == "at"
    assert credentials["refresh"], "a refresh token must be stored or the session dies at expiry"
    assert credentials["email"] == "you@example.com"


def test_only_browser_signin_flows_race_a_pasted_code_against_their_callback() -> None:
    """`paste_code_flow` is a FALLBACK marker, and ONLY that.

    It marks a loopback provider that ALSO accepts a pasted code (for a browser
    on another machine), raced against the callback rather than awaited.
    Attaching such a prompt to any other loopback provider blocks the terminal
    on a line nobody will type.

    Two providers qualify, and both are browser sign-ins whose redirect lands on
    a loopback port: Anthropic, and Z.AI's GLM Coding Plan sign-in (which mirrors
    the Anthropic wiring, as omp's `zai-coding-plan` does). A provider whose
    login is "paste your API key" is NOT one of these -- that is
    ``paste_prompt_required``, asserted in the companion test.

    This test used to assert something wider and false in its name and
    docstring — that no provider REQUIRES a paste — while asserting only the
    narrow set below. Eight providers require one: their whole login is
    "open the dashboard, paste the key", which is a different property with a
    different field (``requires_paste_prompt``), asserted in the companion test.
    Reading the two as one is what shipped `/login alibaba` unable to succeed.
    """
    from local_operator.providers.registry import PROVIDER_REGISTRY

    paste = {p.id for p in PROVIDER_REGISTRY if p.login is not None and p.paste_code_flow}
    assert paste == {"anthropic", "zai-oauth"}, paste


def test_paste_key_providers_declare_that_they_require_a_prompt() -> None:
    """Every paste-a-key login carries the flag hosts gate on.

    Derived from the login callable rather than declared per provider, so this
    also pins the derivation: a new ``create_api_key_login`` provider must show
    up here without anyone remembering to set a field.
    """
    from local_operator.providers.registry import PROVIDER_REGISTRY

    required = {p.id for p in PROVIDER_REGISTRY if p.login is not None and p.paste_prompt_required}
    assert required == {
        "xai",
        "deepseek",
        "google",
        "mistral",
        "openrouter",
        "radient",
        "alibaba",
        "alibaba-token-plan",
        "alibaba-token-plan-oauth",
        "zai",
    }, required

    # And the union a host actually gates on: required plus the browser
    # sign-ins' optional fallback, and nothing else. A loopback-only provider
    # appearing here would mean a prompt racing its HTTP callback.
    accepts = {p.id for p in PROVIDER_REGISTRY if p.login is not None and p.accepts_paste_prompt}
    assert accepts == required | {"anthropic", "zai-oauth"}, accepts


@pytest.mark.asyncio
async def test_qwencloud_token_plan_login_captures_key_and_device_grant() -> None:
    """The one credential row carries BOTH halves: the pasted sk-sp key the
    wire spends, and the device-flow management token only usage spends."""
    import time

    from local_operator.providers.oauth.qwencloud import (
        DEVICE_CODE_PATH,
        DEVICE_TOKEN_PATH,
        login_qwencloud_token_plan,
    )

    auth_urls: list[str] = []
    callbacks = LoginCallbacks(
        on_auth_url=lambda url, instructions=None: auth_urls.append(url),
        on_progress=lambda message: None,
        on_manual_code_input=lambda: "sk-sp-test",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == DEVICE_CODE_PATH:
            # The init URL must carry the S256 challenge (RFC 7636).
            assert request.url.params["code_challenge_method"] == "S256"
            return httpx.Response(
                200,
                json={
                    "Success": True,
                    "Data": {
                        "Token": "device-token",
                        "VerificationUrl": "https://account.qwencloud.com/sso/OIDCAuth",
                        "ExpiresIn": 300,
                        "Interval": 1,
                    },
                },
            )
        assert request.url.path == DEVICE_TOKEN_PATH
        assert request.url.params["token"] == "device-token"
        assert "code_verifier" in request.url.params
        return httpx.Response(
            200,
            json={
                "Success": True,
                "Data": {
                    "Status": "complete",
                    "Credentials": {
                        "AccessToken": "mgmt-access",
                        "RefreshToken": "mgmt-refresh",
                        "ExpireTime": "2030-01-01T00:00:00Z",
                        "User": {"AliyunId": "damian-aliyun", "Email": "damian@example.com"},
                    },
                },
            },
        )

    creds = await login_qwencloud_token_plan(
        callbacks, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert creds["access"] == "mgmt-access"
    assert creds["refresh"] == "mgmt-refresh"
    assert creds["api_key"] == "sk-sp-test"
    assert creds["account_id"] == "damian-aliyun"
    assert creds["email"] == "damian@example.com"
    assert creds["expires"] == int(datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert creds["authorized_at"] <= int(time.time() * 1000)
    assert auth_urls == ["https://account.qwencloud.com/sso/OIDCAuth"]


@pytest.mark.asyncio
async def test_qwencloud_token_plan_login_rejects_non_token_plan_keys() -> None:
    """A key that is not sk-… cannot be a Token Plan key; fail with guidance
    rather than storing a credential the wire will 401 on."""
    from local_operator.providers.oauth.qwencloud import login_qwencloud_token_plan

    callbacks = LoginCallbacks(on_manual_code_input=lambda: "not-a-qwencloud-key")
    with pytest.raises(Exception, match="Token Plan key"):
        await login_qwencloud_token_plan(callbacks)


@pytest.mark.asyncio
async def test_qwencloud_token_plan_login_expired_device_code_is_terminal() -> None:
    from local_operator.providers.oauth.qwencloud import (
        DEVICE_CODE_PATH,
        login_qwencloud_token_plan,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == DEVICE_CODE_PATH:
            return httpx.Response(
                200,
                json={
                    "Success": True,
                    "Data": {
                        "Token": "device-token",
                        "VerificationUrl": "https://account.qwencloud.com/sso/OIDCAuth",
                        "ExpiresIn": 300,
                        "Interval": 1,
                    },
                },
            )
        return httpx.Response(200, json={"Success": True, "Data": {"Status": "expired_token"}})

    callbacks = LoginCallbacks(on_manual_code_input=lambda: "sk-sp-test")
    with pytest.raises(Exception, match="expired"):
        await login_qwencloud_token_plan(
            callbacks, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )


class TestZaiSignInMintsADurableKey:
    """Z.AI's sign-in is only useful if it ends with the key the WIRE accepts.

    The short-lived OAuth token from the token exchange is rejected by the
    inference endpoint, so a flow that stored it would log in successfully and
    then fail every request. These tests pin the whole sequence:
    token exchange -> business login -> org/project -> find-or-create key ->
    read the secret from the copy endpoint.
    """

    @staticmethod
    def _transport(calls: list[str], *, existing_key: bool) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            calls.append(f"{request.method} {path}")
            if path.endswith("/oauth/token"):
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "data": {
                            "zai": {"access_token": "short-lived-oauth"},
                            "user": {"email": "damian@example.com", "id": 4242},
                        },
                    },
                )
            if path.endswith("/api/auth/z/login"):
                assert json.loads(request.content)["token"] == "short-lived-oauth"
                return httpx.Response(200, json={"code": 200, "data": {"access_token": "biz-tok"}})
            if path.endswith("/getCustomerInfo"):
                assert request.headers["Authorization"] == "Bearer biz-tok"
                return httpx.Response(
                    200,
                    json={
                        "code": 200,
                        "data": {
                            "organizations": [
                                {"organizationId": "org-other", "projects": []},
                                {
                                    "organizationId": "org-1",
                                    "isDefault": True,
                                    "projects": [
                                        {"projectId": "proj-other"},
                                        {"projectId": "proj-1", "isDefault": True},
                                    ],
                                },
                            ]
                        },
                    },
                )
            if path.endswith("/api_keys") and request.method == "GET":
                listed = [{"name": "local-operator", "apiKey": "key-id"}] if existing_key else []
                return httpx.Response(200, json={"code": 200, "data": {"list": listed}})
            if path.endswith("/api_keys") and request.method == "POST":
                assert json.loads(request.content)["name"] == "local-operator"
                return httpx.Response(200, json={"code": 200, "data": {"apiKey": "key-id"}})
            if "/api_keys/copy/" in path:
                return httpx.Response(200, json={"code": 200, "data": {"secretKey": "sec-ret"}})
            raise AssertionError(f"unexpected request: {request.method} {path}")

        return httpx.MockTransport(handler)

    async def test_exchange_mints_and_stores_the_durable_key(self) -> None:
        from local_operator.providers.oauth.zai import ZaiOAuthFlow

        calls: list[str] = []
        flow = ZaiOAuthFlow(
            http_client=httpx.AsyncClient(transport=self._transport(calls, existing_key=False))
        )
        creds = await flow.exchange_token(
            "the-code", "the-state", "http://localhost:54548/callback"
        )

        # The MINTED `id.secret` key, never the short-lived OAuth token.
        assert creds["access"] == "key-id.sec-ret"
        assert creds["email"] == "damian@example.com"
        assert creds["account_id"] == "4242"
        # `expires: None` is AuthStore's "static token" marker: no refresh is
        # ever attempted, so the row persists across sessions.
        assert creds["expires"] is None
        # The secret always comes from the copy endpoint -- list entries mask it.
        assert any("/api_keys/copy/" in call for call in calls), calls

    async def test_an_existing_key_is_reused_rather_than_duplicated(self) -> None:
        """Signing in twice must not litter the console with one key per login."""
        from local_operator.providers.oauth.zai import ZaiOAuthFlow

        calls: list[str] = []
        flow = ZaiOAuthFlow(
            http_client=httpx.AsyncClient(transport=self._transport(calls, existing_key=True))
        )
        creds = await flow.exchange_token("c", "s", "http://localhost:54548/callback")

        assert creds["access"] == "key-id.sec-ret"
        assert "POST /api/biz/v1/organization/org-1/projects/proj-1/api_keys" not in calls, calls

    async def test_an_envelope_error_is_raised_not_stored(self) -> None:
        """Z.AI reports failures with HTTP 200 and a `code` field, so a caller
        that trusted the status would persist an error body as a credential."""
        from local_operator.providers.oauth.zai import ZaiOAuthFlow

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"code": 401, "msg": "authorization expired"})

        flow = ZaiOAuthFlow(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with pytest.raises(LoginError, match="authorization expired"):
            await flow.exchange_token("c", "s", "http://localhost:54548/callback")

    async def test_the_authorize_url_carries_no_pkce(self) -> None:
        """The provider's own client sends none and the endpoint rejects the
        extra parameters, so adding PKCE "for safety" breaks the login."""
        from local_operator.providers.oauth.zai import CLIENT_ID, ZaiOAuthFlow

        flow = ZaiOAuthFlow()
        url = await flow.generate_auth_url("st-1", "http://localhost:54548/callback")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)

        assert query["client_id"] == [CLIENT_ID]
        assert query["state"] == ["st-1"]
        assert query["response_type"] == ["code"]
        assert "code_challenge" not in query

    async def test_the_stored_credential_is_readable_by_the_cascade(self, tmp_path: Any) -> None:
        """R21: the join nothing tested -- sign in, then RESOLVE what was stored.

        Two green tests used to sit either side of this gap: one asserted the
        registry's `zai-oauth` fields, the other asserted the login flow's
        return dict. Neither stored a real payload and asked the AuthStore for
        it back, so a credential shape that no cascade tier could read shipped
        as a working sign-in: the login reported success and every subsequent
        request failed without one HTTP call being made.

        This walks the real path end to end -- `ZaiOAuthFlow.exchange_token` ->
        the exact `upsert_credential` call `ProviderController.login` makes ->
        `get_api_key` / `get_oauth_access` -- because each of those three
        components was individually correct while their composition was not.
        """
        from local_operator.providers.auth_store import AuthStore
        from local_operator.providers.registry import get_provider_definition
        from local_operator.providers.oauth.zai import ZaiOAuthFlow

        calls: list[str] = []
        flow = ZaiOAuthFlow(
            http_client=httpx.AsyncClient(transport=self._transport(calls, existing_key=False))
        )
        creds = await flow.exchange_token("c", "s", "http://localhost:54548/callback")

        # `store_credentials_as` is what makes the sign-in and the pasted key
        # share one row, so the test must store under the alias the controller
        # resolves rather than under the provider id it was invoked with.
        definition = get_provider_definition("zai-oauth")
        assert definition is not None
        storage = definition.store_credentials_as or "zai-oauth"
        assert storage == "zai"

        store = AuthStore(db_path=tmp_path / "auth.db")
        creds.setdefault("authorized_at", 1)
        store.upsert_credential(storage, creds)

        # The bearer the wire actually receives. `None` here is the R21 failure:
        # a row present, typed `api_key`, secret under `access`, unreadable.
        assert await store.get_api_key("zai") == "key-id.sec-ret"

        access = await store.get_oauth_access("zai")
        assert access is not None
        assert access.access_token == "key-id.sec-ret"
        # Typed `oauth` so tier 3 can see it, and so the row carries the signed-in
        # identity that `_FETCHERS["zai-oauth"]` needs for quota reporting.
        assert access.kind == "oauth"
        assert access.email == "damian@example.com"
