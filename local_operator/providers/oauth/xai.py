"""xAI / Grok OAuth — RFC 8628 device code via OIDC discovery.

xAI OAuth port (adapted from
NousResearch/hermes-agent, MIT). The token endpoint is discovered from
``https://auth.x.ai/.well-known/openid-configuration`` and HARD-validated
(https only, host ``x.ai`` or ``*.x.ai``) because every future refresh token
is POSTed to it — an unvalidated discovery document would be a token-theft
vector. The validated endpoint is pinned for the credential's lifetime.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    LoginCallbacks,
    LoginError,
    maybe_await,
    raise_for_refresh_failure,
)
from local_operator.providers.oauth.device_code import (
    DevicePollResult,
    poll_device_code_flow,
)

CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
SCOPES = "openid profile email offline_access grok-cli:access api:access"

DISCOVERY_URL = "https://auth.x.ai/.well-known/openid-configuration"
DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"

DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
EXPIRY_SKEW_SECONDS = 5 * 60


def validate_xai_endpoint(url: str) -> str:
    """Validate a discovered endpoint: https only, host ``x.ai`` or ``*.x.ai``.

    Returns the URL unchanged so callers can chain: ``url = validate(...)``.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https":
        raise LoginError(f"xAI OAuth endpoint must be https: {url}")
    if host != "x.ai" and not host.endswith(".x.ai"):
        raise LoginError(f"xAI OAuth endpoint host not trusted: {host}")
    return url


async def discover_token_endpoint(http: httpx.AsyncClient) -> str:
    """Fetch OIDC discovery and validate the token endpoint."""
    response = await http.get(DISCOVERY_URL)
    if response.status_code != 200:
        raise LoginError(f"xAI OIDC discovery failed ({response.status_code}): {response.text}")
    config = response.json()
    token_endpoint = config.get("token_endpoint")
    if not token_endpoint:
        raise LoginError("xAI OIDC discovery returned no token_endpoint")
    return validate_xai_endpoint(str(token_endpoint))


def _credentials_from_token(
    token: dict[str, Any], token_endpoint: str, old_refresh: str | None = None
) -> dict[str, Any]:
    access = token.get("access_token")
    # IdPs that do not rotate refresh tokens omit one on refresh; keep the
    # old one (kimi pattern; PR-09).
    refresh = token.get("refresh_token") or old_refresh
    if not access or not refresh:
        raise LoginError("xAI token response is missing access/refresh tokens")
    expires_in = int(token.get("expires_in", 3600))
    creds: dict[str, Any] = {
        "refresh": refresh,
        "access": access,
        # 5-minute skew so a token never dies mid-request.
        "expires": int(time.time() * 1000) + (expires_in - EXPIRY_SKEW_SECONDS) * 1000,
        "authorized_at": int(time.time() * 1000),
        # Pin the validated endpoint; refreshes must hit the same host.
        "token_endpoint": token_endpoint,
    }
    id_token = token.get("id_token")
    if id_token:
        try:
            from local_operator.providers.oauth.openai import decode_jwt_claims

            claims = decode_jwt_claims(id_token)
            if claims.get("email"):
                creds["email"] = claims["email"]
            if claims.get("sub"):
                creds["account_id"] = str(claims["sub"])
        except LoginError:
            pass  # identity is best-effort; the token itself is still usable
    return creds


async def login_xai(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Run the RFC 8628 device flow with endpoint discovery."""
    owns_client = http_client is None
    http = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        token_endpoint = await discover_token_endpoint(http)

        start = await http.post(
            DEVICE_CODE_URL,
            data={"client_id": CLIENT_ID, "scope": SCOPES},
        )
        if start.status_code != 200:
            raise LoginError(f"xAI device authorization failed ({start.status_code}): {start.text}")
        authz = start.json()
        device_code = authz.get("device_code")
        verification_url = authz.get("verification_uri_complete") or authz.get("verification_uri")
        if not device_code or not verification_url:
            raise LoginError(f"xAI device authorization response malformed: {authz}")

        if callbacks.on_auth_url is not None:
            user_code = authz.get("user_code", "")
            await maybe_await(
                callbacks.on_auth_url(
                    verification_url,
                    instructions=f"Enter code: {user_code}" if user_code else None,
                )
            )

        interval = float(authz.get("interval", 5))
        expires_in = float(authz.get("expires_in", 900))

        async def _poll() -> DevicePollResult[dict[str, Any]]:
            response = await http.post(
                token_endpoint,
                data={
                    "grant_type": DEVICE_GRANT_TYPE,
                    "client_id": CLIENT_ID,
                    "device_code": device_code,
                    # The IdP returns an error payload instead
                    # of a redirect it cannot perform for device flows.
                    "redirect": "error",
                },
            )
            if response.status_code == 200:
                return DevicePollResult.complete(response.json())
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            error = str(payload.get("error", "")).lower()
            if error == "authorization_pending":
                return DevicePollResult.pending()
            if error == "slow_down":
                return DevicePollResult.slow_down()
            if error == "expired_token":
                return DevicePollResult.failed("The xAI login code expired; run login again.")
            if error == "access_denied":
                return DevicePollResult.failed("xAI authorization was denied.")
            return DevicePollResult.failed(
                payload.get("error_description")
                or f"xAI token poll failed ({response.status_code})"
            )

        token = await poll_device_code_flow(
            _poll,
            interval_seconds=interval,
            expires_in_seconds=expires_in,
            signal=signal,
            on_progress=callbacks.on_progress,
        )
        return _credentials_from_token(token, token_endpoint)
    finally:
        if owns_client:
            await http.aclose()


async def refresh_xai_token(
    creds: dict[str, Any], *, http_client: httpx.AsyncClient | None = None
) -> dict[str, Any]:
    """Refresh against the endpoint pinned at login (validated at discovery)."""
    token_endpoint = creds.get("token_endpoint")
    if not token_endpoint:
        raise LoginError("xAI credential has no pinned token endpoint; run login again")
    validate_xai_endpoint(str(token_endpoint))
    owns_client = http_client is None
    http = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        response = await http.post(
            str(token_endpoint),
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": creds.get("refresh"),
            },
        )
    finally:
        if owns_client:
            await http.aclose()
    if response.status_code != 200:
        raise_for_refresh_failure("xAI", response.status_code, response.text)
    merged = dict(creds)
    merged.update(
        _credentials_from_token(
            response.json(), str(token_endpoint), old_refresh=creds.get("refresh")
        )
    )
    merged["authorized_at"] = creds.get("authorized_at", merged["authorized_at"])
    return merged
