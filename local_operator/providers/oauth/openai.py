"""OpenAI (ChatGPT Plus/Pro) OAuth — two flows, one credential row.

OpenAI Codex OAuth port:

- **Browser flow** (provider id ``openai``): authorization code + PKCE,
  loopback port **1455** pinned (``/auth/callback``). OpenAI allowlists the
  redirect URI, so there is NO port fallback — a busy port must fail before
  the browser opens.
- **Device flow** (provider id ``openai-device``): OpenAI-private
  ``deviceauth`` endpoints (NOT RFC 8628). Both store under ``openai``.
- Identity comes from **JWT claims decoded without signature verification**
  (no PyJWT — the IdP already validated it): ``https://api.openai.com/auth``
  → ``chatgpt_account_id``/``chatgpt_plan_type``, ``.../profile`` → email.
  ``org_id = account_id`` and login FAILS HARD without an account id.
"""

import base64
import json
import time
import urllib.parse
from typing import Any, Callable

import httpx

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    CallbackFlowOptions,
    LoginCallbacks,
    LoginError,
    OAuthCallbackFlow,
    maybe_await,
)
from local_operator.providers.oauth.device_code import (
    DevicePollResult,
    poll_device_code_flow,
)
from local_operator.providers.oauth.pkce import create_pkce_pair

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEVICE_USERCODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
DEVICE_PAGE_URL = "https://auth.openai.com/codex/device"
DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"

SCOPES = "openid profile email offline_access api.connectors.read api.connectors.invoke"
# Our product's originator for the IdP's telemetry; intentionally not the
# reference value — local-operator identifies itself (PR-23).
ORIGINATOR = "local-operator"

CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"

AUTH_CLAIM_KEY = "https://api.openai.com/auth"
PROFILE_CLAIM_KEY = "https://api.openai.com/profile"

DEVICE_MAX_POLLS = 120
DEVICE_SAFETY_MARGIN_SECONDS = 3


def _b64url_decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT payload WITHOUT signature verification.

    Never add PyJWT here: the token came
    straight from the IdP's token endpoint over TLS, and verification would
    only add a dependency plus a failure mode for rotating key sets.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise LoginError("Malformed JWT: expected three dot-separated segments")
    try:
        return json.loads(_b64url_decode(parts[1]))
    except (ValueError, json.JSONDecodeError) as exc:
        raise LoginError(f"Malformed JWT payload: {exc}") from exc


def identity_from_id_token(id_token: str) -> dict[str, Any]:
    """Extract ``email``/``account_id``/``org_id``/``org_name`` from claims.

    Raises when no account id is present — ChatGPT login is unusable without
    one (org id doubles as account id downstream).
    """
    claims = decode_jwt_claims(id_token)
    auth = claims.get(AUTH_CLAIM_KEY) or {}
    profile = claims.get(PROFILE_CLAIM_KEY) or {}
    account_id = auth.get("chatgpt_account_id")
    if not account_id:
        raise LoginError("OpenAI login returned no account id (chatgpt_account_id claim missing)")
    plan_type = auth.get("chatgpt_plan_type")
    email = profile.get("email") or claims.get("email")
    identity: dict[str, Any] = {"account_id": str(account_id), "org_id": str(account_id)}
    if plan_type:
        identity["org_name"] = str(plan_type)
    if email:
        identity["email"] = email
    return identity


# 5-minute mint skew so a fresh token never dies mid-request (parity with
# the anthropic/kimi/xai flows; PR-08).
EXPIRY_SKEW_MS = 5 * 60 * 1000


def _credentials_from_token(token: dict[str, Any]) -> dict[str, Any]:
    access = token.get("access_token")
    refresh = token.get("refresh_token")
    if not access or not refresh:
        raise LoginError("OpenAI token response is missing access/refresh tokens")
    creds: dict[str, Any] = {
        "refresh": refresh,
        "access": access,
        "expires": int(time.time() * 1000)
        + int(token.get("expires_in", 3600)) * 1000
        - EXPIRY_SKEW_MS,
        "authorized_at": int(time.time() * 1000),
    }
    # Identity: account_id/org_id are REQUIRED — ChatGPT tokens are routed
    # by account id (chatgpt-account-id), so a row without them cannot
    # stream. Fail hard at login even when the IdP omitted the id_token
    # entirely (PR-08).
    id_token = token.get("id_token")
    if id_token:
        identity = identity_from_id_token(id_token)
    else:
        raise LoginError(
            "OpenAI token response carried no id_token; cannot determine the "
            "ChatGPT account required for inference. Try the device-code login."
        )
    creds.update(identity)
    return creds


class OpenAIOAuthFlow(OAuthCallbackFlow):
    """ChatGPT browser login; pinned port 1455, no fallback."""

    def __init__(
        self,
        callbacks: LoginCallbacks | None = None,
        *,
        open_browser: Callable[[str], None] | None = None,
        signal: AbortSignal | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            CallbackFlowOptions(
                preferred_port=CALLBACK_PORT,
                callback_path=CALLBACK_PATH,
                # Pinned: OpenAI allowlists this exact URI.
                redirect_uri=REDIRECT_URI,
                allow_port_fallback=False,
            ),
            callbacks,
            open_browser=open_browser,
            signal=signal,
        )
        self._verifier: str | None = None
        self._http = http_client

    async def generate_auth_url(self, state: str, redirect_uri: str) -> str:
        verifier, challenge = create_pkce_pair()
        self._verifier = verifier
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": ORIGINATOR,
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        # Form-encoded exchange, 15 s timeout (established parity).
        #
        # `state` is NOT sent here. OpenAI's token endpoint rejects unknown
        # parameters outright (400 `invalid_request_error` /
        # `unknown_parameter: state`) — observed live on `/login openai`. The
        # device-code flow below has always omitted it and works, which is the
        # shape this endpoint accepts. Nothing is lost: state is verified at
        # the CALLBACK (callback_server compares it with the value sent, before
        # the code is trusted), which is the point in OAuth where it has
        # meaning; RFC 6749 §4.1.3 does not require echoing it on exchange. The
        # earlier echo (PR-13) predates the endpoint's strict validation.
        payload = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": self._verifier,
        }
        if self._http is not None:
            response = await self._http.post(TOKEN_URL, data=payload)
        else:
            async with httpx.AsyncClient(timeout=15.0) as http:
                response = await http.post(TOKEN_URL, data=payload)
        if response.status_code != 200:
            raise LoginError(
                f"OpenAI token exchange failed ({response.status_code}): {response.text}"
            )
        return _credentials_from_token(response.json())


async def login_openai(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,  # reserved for tests
    open_browser: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the browser flow; returns OAuthCredentials dict."""
    flow = OpenAIOAuthFlow(
        callbacks, open_browser=open_browser, signal=signal, http_client=http_client
    )
    return await flow.run()


async def login_openai_device(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """OpenAI-private device flow (not RFC 8628).

    1. ``POST deviceauth/usercode`` → ``{device_auth_id, user_code, interval}``.
    2. User opens the device page and types the code.
    3. Poll ``POST deviceauth/token`` (≤120 polls, interval + 3 s margin).
    4. Success returns ``{authorization_code, code_verifier}``, exchanged
       against ``redirect_uri = https://auth.openai.com/deviceauth/callback``.
    """
    owns_client = http_client is None
    http = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        start = await http.post(DEVICE_USERCODE_URL, data={"client_id": CLIENT_ID})
        if start.status_code != 200:
            raise LoginError(f"OpenAI device auth start failed ({start.status_code}): {start.text}")
        device = start.json()
        device_auth_id = device.get("device_auth_id")
        user_code = device.get("user_code")
        if not device_auth_id or not user_code:
            raise LoginError(f"OpenAI device auth response malformed: {device}")
        interval = max(1.0, float(device.get("interval", 5)))

        if callbacks.on_auth_url is not None:
            await maybe_await(
                callbacks.on_auth_url(DEVICE_PAGE_URL, instructions=f"Enter code: {user_code}")
            )

        async def _poll() -> DevicePollResult[dict[str, Any]]:
            response = await http.post(
                DEVICE_TOKEN_URL,
                data={"client_id": CLIENT_ID, "device_auth_id": device_auth_id},
            )
            if response.status_code == 200:
                payload = response.json()
                if payload.get("authorization_code"):
                    return DevicePollResult.complete(payload)
                return DevicePollResult.pending()
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            error = str(payload.get("error", "")).lower()
            if error in ("authorization_pending", "pending"):
                return DevicePollResult.pending()
            if error == "slow_down":
                return DevicePollResult.slow_down()
            return DevicePollResult.failed(
                payload.get("error_description")
                or f"Device authorization failed ({response.status_code})"
            )

        expires_in = DEVICE_MAX_POLLS * (interval + DEVICE_SAFETY_MARGIN_SECONDS)
        token = await poll_device_code_flow(
            _poll,
            interval_seconds=interval,
            expires_in_seconds=expires_in,
            signal=signal,
            on_progress=callbacks.on_progress,
        )

        exchange = await http.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": token["authorization_code"],
                "redirect_uri": DEVICE_REDIRECT_URI,
                "code_verifier": token.get("code_verifier", ""),
            },
        )
        if exchange.status_code != 200:
            raise LoginError(
                f"OpenAI device token exchange failed ({exchange.status_code}): {exchange.text}"
            )
        return _credentials_from_token(exchange.json())
    finally:
        if owns_client:
            await http.aclose()


async def refresh_openai_token(
    creds: dict[str, Any], *, http_client: httpx.AsyncClient | None = None
) -> dict[str, Any]:
    """Form-encoded refresh; org fields are never rewritten."""
    owns_client = http_client is None
    http = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        response = await http.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": creds.get("refresh"),
                "client_id": CLIENT_ID,
            },
        )
    finally:
        if owns_client:
            await http.aclose()
    if response.status_code != 200:
        raise LoginError(f"OpenAI refresh failed ({response.status_code}): {response.text}")
    token = response.json()
    if not token.get("access_token"):
        raise LoginError("OpenAI refresh response is missing access_token")
    merged = dict(creds)
    merged["access"] = token["access_token"]
    if token.get("refresh_token"):
        merged["refresh"] = token["refresh_token"]
    merged["expires"] = int(time.time() * 1000) + int(token.get("expires_in", 3600)) * 1000
    return merged
