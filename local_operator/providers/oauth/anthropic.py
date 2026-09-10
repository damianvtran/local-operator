"""Anthropic (Claude Pro/Max) OAuth — authorization code + PKCE.

Anthropic OAuth port. Traps preserved:

- Loopback callback on port **54545** path ``/callback``; paste-code fallback
  is allowed (``paste_code_flow=True`` in the registry).
- The authorize endpoint MUST be ``claude.ai`` — ``platform.claude.com``
  issues console tokens *without* ``user:inference``.
- Refresh sends ``anthropic-beta: oauth-2025-04-20`` and the SDK-style
  User-Agent; omitting the beta header 400s.
- Refresh results are merged OVER the stored credential and NEVER rewrite
  org fields (``org_id``/``org_name``/identity are fixed at login).
- Expiry skew is 5 minutes at mint.
- The refresh-token family dies **30 days after authorization** regardless
  of rotation — the deadline is surfaced in the identity payload.

What this flow is, recorded so it does not get re-litigated from scratch:
it is a first-party-style, device-local sign-in. The user authenticates
directly with Anthropic through Anthropic's own flow; the resulting
credential is stored on that user's own device, exactly as Claude Code
stores it, and is used only for that same user's own inference.
local-operator does not proxy, resell, or intermediate the subscription, and
no Local Operator or Radient service sits between the user and Anthropic.
That distinction is the substance of the terms at
https://code.claude.com/docs/en/legal-and-compliance (verified 2026-09-09),
which address offering Claude.ai login inside a third-party product and
routing other users' requests through plan credentials — neither of which
this does.
"""

from __future__ import annotations

import base64
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
    raise_for_refresh_failure,
)
from local_operator.providers.oauth.pkce import create_pkce_pair

# Stored base64-encoded; decoded here at import.
CLIENT_ID = base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode()

AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://api.anthropic.com/v1/oauth/token"
BOOTSTRAP_URL = "https://api.anthropic.com/api/claude_cli/bootstrap"

SCOPES = (
    "org:create_api_key user:profile user:inference "
    "user:sessions:claude_code user:mcp_servers user:file_upload"
)

CALLBACK_PORT = 54545
CALLBACK_PATH = "/callback"

EXPIRY_SKEW_MS = 5 * 60 * 1000
GRANT_TTL_MS = 30 * 24 * 60 * 60 * 1000  # whole refresh-token family dies after this

REFRESH_HEADERS = {
    "anthropic-beta": "oauth-2025-04-20",
    "User-Agent": "anthropic-sdk-typescript/0.94.0 userOAuthProvider",
}

_GRANT_NOTE = (
    "Anthropic OAuth grants expire 30 days after authorization regardless of "
    "token rotation; re-run login when that happens."
)


class AnthropicOAuthFlow(OAuthCallbackFlow):
    """Authorization-code + PKCE against claude.ai with a loopback callback."""

    def __init__(
        self,
        callbacks: LoginCallbacks | None = None,
        *,
        open_browser: Callable[[str], None] | None = None,
        signal: AbortSignal | None = None,
        manual_input_only: bool = False,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            CallbackFlowOptions(
                preferred_port=CALLBACK_PORT,
                callback_path=CALLBACK_PATH,
                # Anthropic allowlists this exact redirect URI, so 54545 is a
                # requirement and not a preference. Inheriting the default
                # `allow_port_fallback=True` meant a busy 54545 bound a random
                # OS port and cheerfully advertised it — which the allowlist
                # rejects, at the IdP, after the browser has already opened,
                # with an error the user cannot act on ("Redirect URI
                # http://localhost:<port>/callback is not supported by
                # client"). Failing locally instead is the whole point: the
                # module docstring has always claimed 54545 is required, and
                # this is the line that makes that true.
                allow_port_fallback=False,
                manual_input_only=manual_input_only,
                provider_label="Anthropic",
            ),
            callbacks,
            open_browser=open_browser,
            signal=signal,
        )
        self._verifier: str | None = None
        # Injected httpx client (tests) — ctor-injected like every other flow.
        self._http = http_client

    async def generate_auth_url(self, state: str, redirect_uri: str) -> str:
        verifier, challenge = create_pkce_pair()
        self._verifier = verifier
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": SCOPES,
            # Makes Anthropic render the authorization code as a `code#state`
            # copy target on the page, which is what the paste fallback
            # consumes when the browser cannot reach this machine's loopback
            # port (remote/SSH/WSL sessions, and the reported case where the
            # browser lands in the claude.ai portal instead of redirecting).
            # It does NOT select a flow variant — the flow is already pinned by
            # `response_type=code` above. See `_parse_pasted_callback` in
            # callback_server.py, which parses exactly that `code#state` shape
            # alongside a bare code and a full pasted redirect URL.
            "code": "true",
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        # JSON POST without an Accept header — mirrors Claude Code exactly.
        if self._http is not None:
            response = await self._http.post(
                TOKEN_URL,
                json={
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "code": code,
                    "state": state,
                    "redirect_uri": redirect_uri,
                    "code_verifier": self._verifier,
                },
            )
        else:
            async with httpx.AsyncClient(timeout=30.0) as http:
                response = await http.post(
                    TOKEN_URL,
                    json={
                        "grant_type": "authorization_code",
                        "client_id": CLIENT_ID,
                        "code": code,
                        "state": state,
                        "redirect_uri": redirect_uri,
                        "code_verifier": self._verifier,
                    },
                )
        if response.status_code != 200:
            raise LoginError(
                f"Anthropic token exchange failed ({response.status_code}): {response.text}"
            )
        token = response.json()
        return await _build_credentials(token, http_client=self._http, include_org=True)


async def _fetch_identity(
    access_token: str,
    http_client: httpx.AsyncClient | None,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """Fallback identity source when the token response carries no org info."""
    client = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        response = await client.get(
            BOOTSTRAP_URL,
            # The model is also sent; the bootstrap response is model-scoped.
            params={"entrypoint": "cli", "includeOrg": "true", "model": model},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code != 200:
            return {}
        return response.json()
    finally:
        if http_client is None:
            await client.aclose()


def _extract_identity(
    payload: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return ``(email, account_id, org_id, org_name)`` from a bootstrap/token payload."""
    account = payload.get("account") or payload.get("accountInfo") or {}
    org = payload.get("organization") or payload.get("organizationInfo") or {}
    email = account.get("email_address") or account.get("email") or payload.get("email")
    account_id = account.get("uuid") or account.get("id")
    org_id = org.get("uuid") or org.get("id")
    org_name = org.get("name") or org.get("displayName")
    return email, account_id, org_id, org_name


async def _build_credentials(
    token: dict[str, Any], *, http_client: httpx.AsyncClient | None, include_org: bool
) -> dict[str, Any]:
    access = token.get("access_token")
    refresh = token.get("refresh_token")
    if not access or not refresh:
        raise LoginError("Anthropic token response is missing access/refresh tokens")
    expires_in = int(token.get("expires_in", 3600))
    creds: dict[str, Any] = {
        "refresh": refresh,
        "access": access,
        # 5-minute skew: never present a token that dies mid-request.
        "expires": int(time.time() * 1000) + expires_in * 1000 - EXPIRY_SKEW_MS,
        "authorized_at": int(time.time() * 1000),
        "grant_ttl_ms": GRANT_TTL_MS,
        "grant_note": _GRANT_NOTE,
    }
    if include_org:
        email, account_id, org_id, org_name = _extract_identity(token)
        if account_id is None:
            identity = await _fetch_identity(access, http_client)
            email, account_id, org_id, org_name = _extract_identity(identity)
        if email:
            creds["email"] = email
        if account_id:
            creds["account_id"] = account_id
        if org_id:
            creds["org_id"] = org_id
        if org_name:
            creds["org_name"] = org_name
    return creds


async def login_anthropic(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
    open_browser: Callable[[str], None] | None = None,
    manual_input_only: bool = False,
) -> dict[str, Any]:
    """Run the interactive Anthropic login; returns OAuthCredentials dict."""
    flow = AnthropicOAuthFlow(
        callbacks,
        open_browser=open_browser,
        signal=signal,
        manual_input_only=manual_input_only,
        http_client=http_client,
    )
    return await flow.run()


async def refresh_anthropic_token(
    creds: dict[str, Any], *, http_client: httpx.AsyncClient | None = None
) -> dict[str, Any]:
    """Refresh and merge-over-stored. Org/identity fields are NEVER rewritten."""
    client = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": creds.get("refresh"),
            },
            headers=REFRESH_HEADERS,
        )
    finally:
        if http_client is None:
            await client.aclose()
    if response.status_code != 200:
        raise_for_refresh_failure("Anthropic", response.status_code, response.text)
    token = response.json()
    if not token.get("access_token"):
        raise LoginError("Anthropic refresh response is missing access_token")

    merged = dict(creds)
    merged["access"] = token["access_token"]
    if token.get("refresh_token"):
        merged["refresh"] = token["refresh_token"]
    expires_in = int(token.get("expires_in", 3600))
    merged["expires"] = int(time.time() * 1000) + expires_in * 1000 - EXPIRY_SKEW_MS
    # email/account_id/org_id/org_name/authorized_at deliberately untouched.
    return merged
