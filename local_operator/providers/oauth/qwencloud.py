"""QwenCloud management OAuth — device flow for Token Plan usage reporting.

The QwenCloud Token Plan authenticates INFERENCE with a dedicated API key
(``sk-sp-…``) that cannot read quota. Consumption tracking lives behind a
separate first-party management surface — the same one the official
``qwencloud-cli`` drives — with an RFC 8628 device flow (PKCE S256, no
registered client id: the server accepts any stable identifier, so one is
minted per login) at ``t.qwencloud.com``. The resulting bearer token carries
``usage:read`` and POSTs flat BSS calls to ``cli.qwencloud.com``.

Login therefore captures BOTH halves in one credential row: the pasted API
key (``api_key``, so the wire keeps sending ``sk-sp-…`` as its bearer — see
``_token_plan_wire_key`` in the registry) and the device-flow grant
(``access``), which only the usage fetcher spends. There is no refresh
endpoint, so an expired grant simply drops out of usage reporting (and the
wire cascade falls through to the API-key tier) until the user logs in again.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Any

import httpx

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    LoginCallbacks,
    LoginCancelledError,
    LoginError,
    maybe_await,
)
from local_operator.providers.oauth.device_code import (
    DevicePollResult,
    poll_device_code_flow,
)

AUTH_ENDPOINT = "https://t.qwencloud.com"
DEVICE_CODE_PATH = "/cli/device/code"
DEVICE_TOKEN_PATH = "/cli/device/token"

DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_TTL_SECONDS = 5 * 60

#: QwenCloud returns Pascal-cased envelopes; the CLI tolerates snake_case too,
#: so both are accepted when picking fields out of a response.
_CREDENTIALS_KEYS = (("AccessToken", "access_token"),)
_USER_KEYS = (("AliyunId", "aliyunId"), ("Email", "email"))
_EXPIRY_KEYS = (("ExpireTime", "expire_time"),)


def _pick(payload: dict[str, Any], keys: tuple[tuple[str, ...], ...]) -> Any:
    """First non-empty value among camel/snake key spellings."""
    for variant in keys:
        for key in variant:
            value = payload.get(key)
            if value:
                return value
    return None


def _pkce_pair() -> tuple[str, str]:
    """``(verifier, s256_challenge)`` — RFC 7636, via the stdlib only."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode()
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    return verifier, challenge


def _expire_time_ms(value: Any) -> int | None:
    """QwenCloud's expiry arrives as RFC 3339 or epoch seconds/ms."""
    if isinstance(value, (int, float)) and value > 0:
        ms = float(value) * 1000 if value < 1_000_000_000_000 else float(value)
        return int(ms)
    if isinstance(value, str) and value:
        try:
            from datetime import datetime

            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(parsed.timestamp() * 1000)
        except ValueError:
            return None
    return None


async def login_qwencloud_token_plan(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Collect the Token Plan API key, then complete the QwenCloud device flow.

    Returns the credential dict stored by ``ProviderController.login``:
    ``access`` (management token), optional ``refresh``/``expires``, account
    identity, and ``api_key`` (the ``sk-sp-…`` inference key).
    """
    if callbacks.on_manual_code_input is None:
        # A contract violation by the host, not a user-facing path: the registry
        # marks this provider ``requires_paste_prompt`` (via ``_lazy_login``) and
        # every shipped host attaches a prompt for such providers, so reaching
        # this means an embedder wired callbacks that cannot complete the flow.
        raise ValueError(
            "QwenCloud Token Plan login reads an API key, but this interface "
            "provided no on_manual_code_input callback to read it with."
        )
    pasted = await maybe_await(callbacks.on_manual_code_input())
    if pasted is None:
        raise LoginCancelledError("QwenCloud Token Plan login cancelled")
    api_key = pasted.strip()
    if not api_key:
        # Empty paste is a cancel, matching ``create_api_key_login``: a blank
        # key would be stored as a credential and shadow a working env key.
        raise LoginCancelledError("QwenCloud Token Plan login cancelled")
    if not api_key.startswith("sk-"):
        raise LoginError(
            "That does not look like a QwenCloud Token Plan key (expected sk-sp-…); "
            "copy it from home.qwencloud.com → API Keys"
        )

    client_id = secrets.token_hex(16)
    verifier, challenge = _pkce_pair()
    owns_client = http_client is None
    http = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        start = await http.post(
            f"{AUTH_ENDPOINT}{DEVICE_CODE_PATH}",
            params={
                "client_id": client_id,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        )
        if start.status_code != 200:
            raise LoginError(
                f"QwenCloud device authorization failed ({start.status_code}): {start.text}"
            )
        init = start.json().get("Data") or {}
        device_token = init.get("Token")
        verification_url = init.get("VerificationUrl")
        if not device_token or not verification_url:
            raise LoginError(f"QwenCloud device authorization response malformed: {init}")

        if callbacks.on_auth_url is not None:
            await maybe_await(
                callbacks.on_auth_url(
                    verification_url,
                    instructions=(
                        "Sign in to QwenCloud and approve the CLI access request; "
                        "login continues automatically once approved."
                    ),
                )
            )

        interval = float(init.get("Interval", DEFAULT_INTERVAL_SECONDS))
        expires_in = float(init.get("ExpiresIn", DEFAULT_TTL_SECONDS))

        async def _poll() -> DevicePollResult[dict[str, Any]]:
            response = await http.post(
                f"{AUTH_ENDPOINT}{DEVICE_TOKEN_PATH}",
                params={
                    "client_id": client_id,
                    "token": device_token,
                    "code_verifier": verifier,
                },
            )
            if response.status_code != 200:
                return DevicePollResult.failed(
                    f"QwenCloud token poll failed ({response.status_code}): {response.text}"
                )
            data = response.json().get("Data") or {}
            status = str(data.get("Status", "authorization_pending")).lower()
            if status == "authorization_pending":
                return DevicePollResult.pending()
            if status == "slow_down":
                return DevicePollResult.slow_down()
            if status == "expired_token":
                return DevicePollResult.failed("The QwenCloud login code expired; run login again.")
            if status != "complete":
                reason = "was denied" if status == "access_denied" else f"failed: {status}"
                return DevicePollResult.failed(f"QwenCloud authorization {reason}.")
            return DevicePollResult.complete(data.get("Credentials") or {})

        credentials = await poll_device_code_flow(
            _poll,
            interval_seconds=interval,
            expires_in_seconds=expires_in,
            signal=signal,
            on_progress=callbacks.on_progress,
        )
        access = _pick(credentials, _CREDENTIALS_KEYS)
        if not isinstance(access, str) or not access:
            raise LoginError("QwenCloud login completed without an access token")
        user = credentials.get("User") or credentials.get("user") or {}
        creds: dict[str, Any] = {
            "access": access,
            "api_key": api_key,
            "authorized_at": int(time.time() * 1000),
        }
        refresh = _pick(credentials, (("RefreshToken", "refresh_token"),))
        if isinstance(refresh, str) and refresh:
            creds["refresh"] = refresh
        expires_ms = _expire_time_ms(_pick(credentials, _EXPIRY_KEYS))
        if expires_ms is not None:
            creds["expires"] = expires_ms
        aliyun_id = _pick(user, _USER_KEYS) if isinstance(user, dict) else None
        if isinstance(aliyun_id, str) and aliyun_id:
            creds["account_id"] = aliyun_id
        email = user.get("Email") or user.get("email") if isinstance(user, dict) else None
        if isinstance(email, str) and email:
            creds["email"] = email
        return creds
    finally:
        if owns_client:
            await http.aclose()
