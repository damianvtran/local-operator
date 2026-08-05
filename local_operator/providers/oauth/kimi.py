"""Kimi / Moonshot OAuth — RFC 8628 device authorization grant.

Ported from omp ``registry/oauth/kimi.ts``. No callback server: the user
opens a verification URL and the CLI polls the token endpoint.

Traps preserved:

- Every OAuth AND usage call carries the ``X-Msh-*`` device fingerprint
  headers; the device id is persisted at ``~/.local-operator/kimi-device-id``
  (mode 0600) so the account stays bound to one "device".
- ``slow_down`` both adds 5 s AND honours a larger returned ``interval``.
- Refresh reuses the old refresh token when the response omits a new one.
"""

from __future__ import annotations

import os
import platform
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import LoginCallbacks, LoginError
from local_operator.providers.oauth.device_code import DevicePollResult, poll_device_code_flow

CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_AUTH_HOST = "https://auth.kimi.com"
DEVICE_AUTHORIZATION_PATH = "/api/oauth/device_authorization"
TOKEN_PATH = "/api/oauth/token"
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_TTL_SECONDS = 15 * 60
EXPIRY_SKEW_MS = 5 * 60 * 1000

KIMI_CLI_VERSION = "1.0.0"

DEVICE_ID_FILENAME = "kimi-device-id"


def auth_host() -> str:
    """OAuth host with omp-compatible env overrides."""
    return os.environ.get("KIMI_CODE_OAUTH_HOST") or os.environ.get("KIMI_OAUTH_HOST") or DEFAULT_AUTH_HOST


def _config_dir() -> Path:
    override = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local-operator"


def get_or_create_device_id(config_dir: Path | None = None) -> str:
    """Persist a stable device id (0600); ephemeral UUID if unwritable.

    The file is created with mode 0600 BEFORE the first write (os.open),
    never written world-readable then chmod'd (PR-11).
    """
    directory = config_dir or _config_dir()
    path = directory / DEVICE_ID_FILENAME
    try:
        if path.exists():
            stored = path.read_text().strip()
            if stored:
                os.chmod(path, 0o600)
                return stored
        device_id = str(uuid.uuid4())
        directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, (device_id + "\n").encode())
        finally:
            os.close(fd)
        return device_id
    except OSError:
        # Best-effort persistence: a per-process ephemeral id still works.
        return str(uuid.uuid4())


def kimi_common_headers(device_id: str | None = None) -> dict[str, str]:
    """The ``X-Msh-*`` fingerprint headers required on every Kimi call."""
    if device_id is None:
        device_id = get_or_create_device_id()
    uname = platform.uname()
    return {
        "User-Agent": f"KimiCLI/{KIMI_CLI_VERSION}",
        "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": KIMI_CLI_VERSION,
        "X-Msh-Device-Name": socket.gethostname(),
        "X-Msh-Device-Model": f"{uname.system} {uname.release} {uname.machine}",
        "X-Msh-Os-Version": uname.version,
        "X-Msh-Device-Id": device_id,
    }


def _credentials_from_token(token: dict[str, Any], old_refresh: str | None) -> dict[str, Any]:
    access = token.get("access_token")
    if not access:
        raise LoginError("Kimi token response is missing access_token")
    refresh = token.get("refresh_token") or old_refresh
    if not refresh:
        raise LoginError("Kimi token response is missing refresh_token")
    expires_in = float(token.get("expires_in", 3600))
    return {
        "refresh": refresh,
        "access": access,
        # 5-minute skew: never present a token that dies mid-request.
        "expires": int(time.time() * 1000) + int(expires_in * 1000) - EXPIRY_SKEW_MS,
        "authorized_at": int(time.time() * 1000),
    }


async def login_kimi(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the RFC 8628 device flow against ``auth.kimi.com``."""
    host = auth_host().rstrip("/")
    headers = kimi_common_headers(get_or_create_device_id(config_dir))
    owns_client = http_client is None
    http = http_client or httpx.AsyncClient(timeout=30.0, headers=headers)
    try:
        start = await http.post(
            f"{host}{DEVICE_AUTHORIZATION_PATH}",
            data={"client_id": CLIENT_ID},
            headers=headers,
        )
        if start.status_code != 200:
            raise LoginError(f"Kimi device authorization failed ({start.status_code}): {start.text}")
        authz = start.json()
        device_code = authz.get("device_code")
        if not device_code:
            raise LoginError(f"Kimi device authorization response malformed: {authz}")

        verification_url = authz.get("verification_uri_complete") or authz.get("verification_uri")
        user_code = authz.get("user_code", "")
        if callbacks.on_auth_url is not None and verification_url:
            result = callbacks.on_auth_url(
                verification_url, instructions=f"Enter code: {user_code}" if user_code else None
            )
            if hasattr(result, "__await__"):
                await result

        interval = float(authz.get("interval", DEFAULT_INTERVAL_SECONDS))
        expires_in = float(authz.get("expires_in", DEFAULT_TTL_SECONDS))

        # Mutable holder lets slow_down communicate a provider-requested
        # interval into the shared poller (PR-10).
        poll_interval_holder: list[float] = []

        async def _poll() -> DevicePollResult[dict[str, Any]]:
            response = await http.post(
                f"{host}{TOKEN_PATH}",
                data={
                    "grant_type": DEVICE_GRANT_TYPE,
                    "client_id": CLIENT_ID,
                    "device_code": device_code,
                },
                headers=headers,
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
                # Honour a larger returned interval too.
                returned = payload.get("interval")
                if isinstance(returned, (int, float)) and returned > interval:
                    poll_interval_holder.append(float(returned))
                return DevicePollResult.slow_down()
            if error == "expired_token":
                return DevicePollResult.failed("The Kimi login code expired; run login again.")
            if error == "access_denied":
                return DevicePollResult.failed("Kimi authorization was denied.")
            return DevicePollResult.failed(
                payload.get("error_description") or f"Kimi token poll failed ({response.status_code})"
            )

        token = await poll_device_code_flow(
            _poll,
            interval_seconds=interval,
            expires_in_seconds=expires_in,
            signal=signal,
            on_progress=callbacks.on_progress,
            interval_holder=poll_interval_holder,
        )
        return _credentials_from_token(token, old_refresh=None)
    finally:
        if owns_client:
            await http.aclose()


async def refresh_kimi_token(
    creds: dict[str, Any], *, http_client: httpx.AsyncClient | None = None
) -> dict[str, Any]:
    """Form-encoded refresh; reuses the old refresh token when none returns."""
    host = auth_host().rstrip("/")
    headers = kimi_common_headers()
    owns_client = http_client is None
    http = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        response = await http.post(
            f"{host}{TOKEN_PATH}",
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": creds.get("refresh"),
            },
            headers=headers,
        )
    finally:
        if owns_client:
            await http.aclose()
    if response.status_code != 200:
        raise LoginError(f"Kimi refresh failed ({response.status_code}): {response.text}")
    merged = dict(creds)
    merged.update(_credentials_from_token(response.json(), old_refresh=creds.get("refresh")))
    # Preserve the original authorization timestamp across refreshes.
    merged["authorized_at"] = creds.get("authorized_at", merged["authorized_at"])
    return merged
