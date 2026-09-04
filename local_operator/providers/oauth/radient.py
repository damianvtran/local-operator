"""Radient OAuth 2.0 PKCE flow.

Provides interactive browser-based OAuth authentication for Radient via
console.radienthq.com/oauth/authorize and api.radienthq.com/v1/auth/oauth/token.
Supports rolling refresh tokens to keep the user signed in indefinitely as long
as they remain active.
"""

from __future__ import annotations

import os
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


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


CLIENT_ID = _env("RADIENT_OAUTH_CLIENT_ID", "lop")
AUTHORIZE_URL = _env("RADIENT_OAUTH_AUTHORIZE_URL", "https://console.radienthq.com/oauth/authorize")
TOKEN_URL = _env("RADIENT_OAUTH_TOKEN_URL", "https://api.radienthq.com/v1/auth/oauth/token")

CALLBACK_PORT = 54549
CALLBACK_PATH = "/callback"
EXPIRY_SKEW_MS = 5 * 60 * 1000  # 5 minutes skew


class RadientOAuthFlow(OAuthCallbackFlow):
    """Authorization-code + PKCE against Radient with a loopback callback."""

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
                manual_input_only=manual_input_only,
                provider_label="Radient",
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
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": "openid profile email offline_access",
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        if not self._verifier:
            raise LoginError("PKCE verifier is missing from session state")

        payload = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": self._verifier,
        }

        async def _call(client: httpx.AsyncClient) -> httpx.Response:
            return await client.post(
                TOKEN_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )

        if self._http is not None:
            resp = await _call(self._http)
        else:
            async with httpx.AsyncClient() as client:
                resp = await _call(client)

        if resp.status_code != 200:
            raise LoginError(f"Radient token exchange failed: HTTP {resp.status_code} {resp.text}")

        data = resp.json()
        access_token = data.get("access_token")
        if not access_token:
            raise LoginError("Radient token response missing access_token")

        expires_in = data.get("expires_in", 3600)
        expires_at = int(time.time() * 1000) + int(expires_in * 1000) - EXPIRY_SKEW_MS

        refresh_token = data.get("refresh_token")
        return {
            # Declared type avoids structural guessing in AuthStore.upsert_credential.
            "type": "oauth",
            # AuthStore standard keys.
            "access": access_token,
            "refresh": refresh_token,
            # Retain OAuth token response keys for compatibility.
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires": expires_at,
            "authorized_at": int(time.time() * 1000),
            "token_type": data.get("token_type", "Bearer"),
            "scope": data.get("scope", ""),
        }


async def refresh_radient_token(
    credentials: dict[str, Any],
    *,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Refresh a Radient access token using rolling refresh token."""
    # Check both standard 'refresh' and compatibility 'refresh_token' keys
    refresh_token = credentials.get("refresh") or credentials.get("refresh_token")
    if not refresh_token:
        raise LoginError("No refresh token stored for Radient")

    payload = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    }

    async def _call(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post(
            TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

    if http_client is not None:
        resp = await _call(http_client)
    else:
        async with httpx.AsyncClient() as client:
            resp = await _call(client)

    if resp.status_code != 200:
        # Radient's wording predates the shared helper and is preserved
        # verbatim: an error string is what a user greps for and quotes, so
        # classification must not restyle it.
        raise_for_refresh_failure(
            "Radient",
            resp.status_code,
            resp.text,
            message=f"Radient refresh failed: HTTP {resp.status_code} {resp.text}",
        )

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        raise LoginError("Radient refresh response missing access_token")

    expires_in = data.get("expires_in", 3600)
    expires_at = int(time.time() * 1000) + int(expires_in * 1000) - EXPIRY_SKEW_MS

    # If server rotated the refresh token, save new one; otherwise retain existing
    new_refresh = data.get("refresh_token") or refresh_token

    merged = dict(credentials)
    merged.update(
        {
            "type": "oauth",
            "access": access_token,
            "refresh": new_refresh,
            "access_token": access_token,
            "refresh_token": new_refresh,
            "expires": expires_at,
        }
    )
    # Preserve original authorized_at timestamp if present
    if "authorized_at" in credentials:
        merged["authorized_at"] = credentials["authorized_at"]
    return merged


async def login_radient(
    callbacks: LoginCallbacks | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    signal: AbortSignal | None = None,
    open_browser: Callable[[str], None] | None = None,
    manual_input_only: bool = False,
) -> dict[str, Any]:
    """Interactive Radient login via browser OAuth 2.0 PKCE."""
    flow = RadientOAuthFlow(
        callbacks,
        open_browser=open_browser,
        signal=signal,
        manual_input_only=manual_input_only,
        http_client=http_client,
    )
    return await flow.run()
