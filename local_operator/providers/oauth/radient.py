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
from copy import deepcopy
from typing import Any, Callable, SupportsIndex

import httpx

from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    CallbackFlowOptions,
    LoginCallbacks,
    LoginCancelledError,
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


class RadientLoginDisabled(LoginError):
    """Known local preserved-disable outcome; hosts must not expose raw errors."""

    def __init__(self) -> None:
        # Round 2, D2: "unavailable" belongs to retained/unverified rows. This
        # row is disabled, and the message is the only text a generic host can
        # print, so it names the real outcome and says nothing else moved.
        super().__init__(
            "Sign-in verified. This Radient sign-in stays disabled; "
            "its saved state was preserved."
        )


_LOGIN_ISSUER = object()


class _VerifiedRadientLogin(dict[str, Any]):
    """Ephemeral flow provenance, deliberately lost on dict/JSON copying.

    This is not a persistent proof flag or a boundary against arbitrary Python
    execution. Only a successful canonical exchange + authenticated /me response
    issues it; hosts consume it after their cancellation check, once.
    """

    def __init__(self, payload: dict[str, Any], signal: AbortSignal | None, issuer: object) -> None:
        if issuer is not _LOGIN_ISSUER:
            raise LoginError("Radient login completion was not verified")
        super().__init__(payload)
        self._issued = deepcopy(payload)
        self._signal = signal
        self._consumed = False
        # A host must finish the login promptly, not retain authority for a
        # later unrelated UI action after the account/session context changes.
        self._deadline = time.monotonic() + 60.0

    def __copy__(self) -> dict[str, Any]:
        return dict(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return deepcopy(dict(self), memo)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        # Background transports may copy credentials, not completion authority.
        raise TypeError("A verified Radient completion cannot be serialized")

    def consume(self) -> dict[str, Any]:
        if self._consumed:
            raise LoginError("Radient login completion was already consumed")
        self._consumed = True
        if self._signal is not None and self._signal.aborted:
            raise LoginCancelledError(self._signal.reason or "Login cancelled")
        if time.monotonic() > self._deadline:
            raise LoginError("Radient login completion expired; sign in again")
        if self != self._issued:
            raise LoginError("Radient login completion changed after verification")
        return deepcopy(self._issued)


def consume_verified_login(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, _VerifiedRadientLogin) or type(result) is not _VerifiedRadientLogin:
        raise LoginError("Radient login requires a verified interactive completion")
    return result.consume()


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
        # Capture the actual exchange destination, not a later config/default
        # lookup. Only the published canonical issuer-to-hub contract is known.
        self._token_url = TOKEN_URL
        self._hub = (
            DEFAULT_RADIENT_API_BASE_URL
            if self._token_url == DEFAULT_RADIENT_API_BASE_URL + "/auth/oauth/token"
            else None
        )

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

        if self._signal is not None and self._signal.aborted:
            raise LoginCancelledError(self._signal.reason or "Login cancelled")
        hub = self._hub
        token_url = self._token_url
        if hub is None or token_url != hub + "/auth/oauth/token":
            raise LoginError("Radient token endpoint has no trusted API hub binding")

        payload = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": self._verifier,
        }

        async def _call(client: httpx.AsyncClient) -> httpx.Response:
            return await client.post(
                token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
                follow_redirects=False,
            )

        try:
            if self._http is not None:
                resp = await _call(self._http)
            else:
                async with httpx.AsyncClient() as client:
                    resp = await _call(client)
        except httpx.HTTPError as failure:
            if self._signal is not None and self._signal.aborted:
                raise LoginCancelledError(self._signal.reason or "Login cancelled") from failure
            raise LoginError("Radient token exchange did not complete") from failure

        if self._signal is not None and self._signal.aborted:
            raise LoginCancelledError(self._signal.reason or "Login cancelled")
        if resp.status_code != 200:
            # A provider error body can contain credential material.
            raise LoginError(f"Radient token exchange failed: HTTP {resp.status_code}")

        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("object required")
            lifetime = data.get("expires_in", 3600)
            if isinstance(lifetime, bool) or not isinstance(lifetime, (int, str)):
                raise ValueError("invalid lifetime")
            expires_in = int(lifetime)
            if expires_in <= 0:
                raise ValueError("invalid lifetime")
        except (ValueError, TypeError) as failure:
            raise LoginError("Radient token exchange returned an invalid response") from failure
        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise LoginError("Radient token response missing access_token")

        expires_at = int(time.time() * 1000) + expires_in * 1000 - EXPIRY_SKEW_MS

        # The committed server contract (8c2a35ac, MeHandlerEcho) authenticates
        # this fresh bearer, loads its account, and returns matching IDs. JWT
        # labels are display metadata only, never the account admission proof.
        async def _profile(client: httpx.AsyncClient) -> httpx.Response:
            return await client.get(
                hub + "/me",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
                follow_redirects=False,
            )

        try:
            if self._http is not None:
                profile = await _profile(self._http)
            else:
                async with httpx.AsyncClient() as client:
                    profile = await _profile(client)
            if self._signal is not None and self._signal.aborted:
                raise LoginCancelledError(self._signal.reason or "Login cancelled")
            if profile.status_code != 200:
                raise LoginError(
                    f"Radient identity verification failed: HTTP {profile.status_code}"
                )
            body = profile.json()
            result = body.get("result") if isinstance(body, dict) else None
            identity = result.get("identity") if isinstance(result, dict) else None
            account = result.get("account") if isinstance(result, dict) else None
            account_id = identity.get("account_id") if isinstance(identity, dict) else None
            actual_id = account.get("id") if isinstance(account, dict) else None
            if (
                not isinstance(account_id, str)
                or not account_id.strip()
                or account_id != account_id.strip()
                or account_id != actual_id
            ):
                raise LoginError(
                    "Radient identity verification returned an invalid account binding"
                )
        except (httpx.HTTPError, ValueError) as failure:
            if self._signal is not None and self._signal.aborted:
                raise LoginCancelledError(self._signal.reason or "Login cancelled") from failure
            raise LoginError("Radient identity verification did not complete") from failure

        refresh_token = data.get("refresh_token")
        credentials = {
            **identity_from_token_response(data),
            "account_id": account_id,
            "api_endpoint": hub,
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
        return _VerifiedRadientLogin(credentials, self._signal, _LOGIN_ISSUER)


def identity_from_token_response(data: dict[str, Any]) -> dict[str, Any]:
    """``email``/``name``/``account_id`` from the token response's ``id_token``.

    Radient's token endpoint mints an ``id_token`` on both the
    authorization-code and the refresh grant, carrying ``sub`` (the Radient
    account id), ``email`` and ``name`` (``agent-server``
    ``internal/services/auth_service.go``, the ``idTokenClaims`` maps). The
    login used to drop it, so the install knew WHO had signed in only by
    asking — and Aida's first conversation asked for an email the sign-in had
    just proved (audit A5/A6). Decoded WITHOUT signature verification for the
    reason ``openai.decode_jwt_claims`` documents (straight from the IdP's own
    token endpoint over TLS); the claims are an identity LABEL, never an
    authorization input.

    ``account_id`` also gives the auth store a real dedupe key instead of the
    per-provider ``oauth:radient`` constant, so two Radient accounts on one
    machine stop collapsing onto one row. Best-effort: a missing or malformed
    token answers ``{}`` and the login proceeds exactly as before.
    """
    token = data.get("id_token")
    if not isinstance(token, str) or not token:
        return {}
    try:
        from local_operator.providers.oauth.openai import decode_jwt_claims

        claims = decode_jwt_claims(token)
    except LoginError:
        return {}
    identity: dict[str, Any] = {}
    email = claims.get("email")
    if isinstance(email, str) and email.strip():
        identity["email"] = email.strip()
    name = claims.get("name")
    if isinstance(name, str) and name.strip():
        identity["name"] = name.strip()
    sub = claims.get("sub")
    if isinstance(sub, str) and sub.strip():
        identity["account_id"] = sub.strip()
    return identity


async def refresh_radient_token(
    credentials: dict[str, Any],
    *,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Refresh a Radient access token using rolling refresh token."""
    token_url = TOKEN_URL
    if token_url != DEFAULT_RADIENT_API_BASE_URL + "/auth/oauth/token" or credentials.get(
        "api_endpoint"
    ) not in (None, "", DEFAULT_RADIENT_API_BASE_URL):
        # A background refresh cannot move a grant to another issuer or infer
        # a new hub. Unknown legacy metadata remains unknown until sign-in.
        raise LoginError("Radient refresh endpoint has no trusted API hub binding")
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
            token_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
            follow_redirects=False,
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
    # The refresh grant re-mints the id_token, so a name changed in the
    # console reaches the stored label on the next refresh.
    labels = identity_from_token_response(data)
    # Refresh can update display labels, not switch or newly admit an account.
    # The verified account/hub binding survives until another explicit login.
    labels.pop("account_id", None)
    merged.update(labels)
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
