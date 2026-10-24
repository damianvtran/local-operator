"""Kimi / Moonshot OAuth — RFC 8628 device authorization grant.

Kimi OAuth port. No callback server: the user
opens a verification URL and the CLI polls the token endpoint.

Traps preserved:

- Every OAuth AND usage call carries the ``X-Msh-*`` device fingerprint
  headers; the device id is persisted at ``~/.local-operator/kimi-device-id``
  (mode 0600) so the account stays bound to one "device".
- ``slow_down`` both adds 5 s AND honours a larger returned ``interval``.
- Refresh reuses the old refresh token when the response omits a new one.
- A grant refreshes against the host that ISSUED it (``auth_host`` in the
  payload), never against whatever the login default happens to be today. Rows
  written before the host split carry no host: their explicit env override is
  retained, otherwise the historical .com default applies. Exactly one refresh
  request is made; a sibling host is never guessed from a failed response.
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
from local_operator.paths import config_dir as _app_config_dir
from local_operator.procstate import O_BINARY
from local_operator.providers.oauth.callback_server import (
    LoginCallbacks,
    LoginError,
    maybe_await,
    raise_for_refresh_failure,
    report_flow_details,
)
from local_operator.providers.oauth.device_code import (
    DevicePollResult,
    poll_device_code_flow,
)

CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
#: The LOGIN default: kimi.ai is the international surface. auth.kimi.com is
#: the mainland host and its browser flow routes Western accounts into a WeChat
#: QR they cannot complete, which is why it is no longer what an unqualified
#: `lop login kimi` reaches. Both hosts speak the same device-grant protocol
#: with the same client id; only the host (and the user-facing verification
#: site behind it) differs.
DEFAULT_AUTH_HOST = "https://auth.kimi.ai"
WECHAT_AUTH_HOST = "https://auth.kimi.com"
#: Compatibility default, not an assertion about issuance: hostless rows from
#: before the split normally used .com, but an explicit env override could have
#: selected .ai or a custom host. Preserve such an override; otherwise use .com.
LEGACY_AUTH_HOST = WECHAT_AUTH_HOST
#: The only hosts a STORED payload may name. See :func:`_normalise_known_host`
#: for why the set is closed rather than a URL-syntax check.
KNOWN_AUTH_HOSTS = frozenset({DEFAULT_AUTH_HOST, WECHAT_AUTH_HOST})
#: Payload key recording the issuing host. Refreshes read it so one account is
#: not re-pointed at an IdP that never minted its grant.
AUTH_HOST_KEY = "auth_host"
DEVICE_AUTHORIZATION_PATH = "/api/oauth/device_authorization"
TOKEN_PATH = "/api/oauth/token"
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_TTL_SECONDS = 15 * 60
EXPIRY_SKEW_MS = 5 * 60 * 1000

KIMI_CLI_VERSION = "1.0.0"

DEVICE_ID_FILENAME = "kimi-device-id"


def auth_host() -> str:
    """The LOGIN default host, with the operator's env overrides.

    This is deliberately not what a refresh reads first: a refresh resolves the
    host recorded on the credential so an already-issued grant keeps working
    when the default moves. Env overrides stay the escape hatch for rows that
    predate host recording.
    """
    return (
        os.environ.get("KIMI_CODE_OAUTH_HOST")
        or os.environ.get("KIMI_OAUTH_HOST")
        or DEFAULT_AUTH_HOST
    )


def _normalise_known_host(value: Any) -> str | None:
    """The known host a stored payload names, else ``None``.

    An ALLOWLIST, never a URL-syntax check. ``auth_host`` travels in the same
    payload as the refresh token, which is a bearer secret; accepting whatever
    URL a (hand-edited) payload carried would turn a local file edit into token
    exfiltration. Only the two hosts this module ships can be addressed by a
    stored value. The env override is read separately because it is the
    operator's explicit instruction rather than data from a credential.
    """
    # Accept legacy runtime mappings that predate the typed credential model;
    # absent/malformed values still resolve through env/legacy, never as hosts.
    if not isinstance(value, str):
        return None
    host = value.strip().rstrip("/")
    return host if host in KNOWN_AUTH_HOSTS else None


def _refresh_host(creds: dict[str, Any]) -> str:
    """The ONE host whose token endpoint may receive this refresh grant.

    A valid recorded host wins over env. A hostless row retains an explicit
    override, because old overrides could legitimately select .ai or a custom
    host; absent one, .com is the historical default rather than a claim about
    where every old grant was issued. A rejected response never triggers a
    second-host guess: operators who deliberately used a non-default host must
    keep that override until a new login records its issuer.
    """
    recorded = _normalise_known_host(creds.get(AUTH_HOST_KEY))
    if recorded is not None:
        return recorded
    override = os.environ.get("KIMI_CODE_OAUTH_HOST") or os.environ.get("KIMI_OAUTH_HOST")
    if override:
        return override.rstrip("/")
    return LEGACY_AUTH_HOST


def _config_dir() -> Path:
    return _app_config_dir()


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
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | O_BINARY, 0o600)
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
    host: str | None = None,
) -> dict[str, Any]:
    """Run the RFC 8628 device flow against ``host`` (default :func:`auth_host`).

    ``host`` is an explicit, keyword-only argument rather than another env
    variable: the registry ships two login ENTRIES that differ only in which
    IdP they address ("kimi" and "kimi-wechat"), and an entry must not mutate
    process env to choose its host.

    Both entries are alternative sign-ins for ONE stored Kimi credential: the
    registry stores the row under ``kimi`` (``store_credentials_as``) and
    ``auth_store._identity_key_for`` dedupes OAuth payloads to ``oauth:kimi``,
    exactly as ``openai-device``/``openai`` share one slot. Signing in with the
    other flavour therefore replaces this row rather than adding a second one.
    """
    host = (host or auth_host()).rstrip("/")

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
            raise LoginError(
                f"Kimi device authorization failed ({start.status_code}): {start.text}"
            )
        authz = start.json()
        device_code = authz.get("device_code")
        if not device_code:
            raise LoginError(f"Kimi device authorization response malformed: {authz}")

        verification_url = authz.get("verification_uri_complete") or authz.get("verification_uri")
        user_code = authz.get("user_code", "")
        await report_flow_details(
            callbacks,
            user_code=str(user_code or ""),
            expires_in=float(authz.get("expires_in", DEFAULT_TTL_SECONDS)),
        )
        if callbacks.on_auth_url is not None and verification_url:
            await maybe_await(
                callbacks.on_auth_url(
                    verification_url,
                    instructions=f"Enter code: {user_code}" if user_code else None,
                )
            )

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
                payload.get("error_description")
                or f"Kimi token poll failed ({response.status_code})"
            )

        token = await poll_device_code_flow(
            _poll,
            interval_seconds=interval,
            expires_in_seconds=expires_in,
            signal=signal,
            on_progress=callbacks.on_progress,
            interval_holder=poll_interval_holder,
        )
        creds = _credentials_from_token(token, old_refresh=None)
        # Record the issuing host so refresh keeps talking to it even after the
        # login default moves again (the same durability question as below).
        creds[AUTH_HOST_KEY] = host
        return creds
    finally:
        if owns_client:
            await http.aclose()


async def login_kimi_wechat(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """The mainland/WeChat login entry: :func:`login_kimi` pinned to kimi.com.

    A thin wrapper rather than a second device-flow implementation: the two
    hosts speak the same protocol, and duplicating the polling/error mapping is
    how the ``slow_down`` and terminal-error traps drift apart. Named here so
    the registry can import the callable by name like every other login.
    """
    return await login_kimi(
        callbacks,
        signal=signal,
        http_client=http_client,
        config_dir=config_dir,
        host=WECHAT_AUTH_HOST,
    )


async def refresh_kimi_token(
    creds: dict[str, Any], *, http_client: httpx.AsyncClient | None = None
) -> dict[str, Any]:
    """Refresh against the host that ISSUED the grant, not today's login default.

    The host is recorded on the credential at login/refresh and resolved by
    :func:`_refresh_host`. Exactly ONE token POST is made: a failed response
    must not send the grant to a guessed second IdP or let that sibling's
    transient failure reclassify the original terminal/transient verdict.

    Form-encoded refresh; reuses the old refresh token when none returns.
    """
    host = _refresh_host(creds)
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
        # Preserve the selected host's own terminal/transient classification.
        raise_for_refresh_failure("Kimi", response.status_code, response.text)
    merged = dict(creds)
    merged.update(_credentials_from_token(response.json(), old_refresh=creds.get("refresh")))
    # Preserve the original authorization timestamp across refreshes.
    merged["authorized_at"] = creds.get("authorized_at", merged["authorized_at"])
    # Record the host that answered so later refreshes do not depend on env or
    # the legacy default. A recorded host is echoed back unchanged.
    merged[AUTH_HOST_KEY] = host
    return merged
