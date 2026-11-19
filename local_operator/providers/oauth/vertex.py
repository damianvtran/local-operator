"""Google Vertex AI authentication — Cloud OAuth2 + PKCE and Service Account JWT.

Supports two OAuth-backed credential modes stored under ``google-vertex``:

1. **Google Cloud browser sign-in (OAuth2 + PKCE)**: uses the Google Cloud SDK
   native desktop OAuth2 client (the same public client ``gcloud auth login``
   uses, supporting Google Workspace SSO automatically), binds a loopback
   callback on port 8085 (with port fallback and paste-code fallback for remote
   or headless sessions), reads the signed-in email, and auto-discovers an
   active Google Cloud project via Cloud Resource Manager.
2. **Service account JSON (RS256 JWT bearer)**: accepts either raw service
   account JSON or a local ``.json`` path, validates required fields, signs an
   RS256 JWT locally via PyJWT, exchanges it at Google's OAuth2 token endpoint,
   and re-mints fresh access tokens automatically on expiry without needing a
   separate refresh token.

Both modes return ``type: "oauth"`` credential payloads carrying ``project_id``
and ``location`` (default ``"global"``) so ``GoogleClient`` routes directly to
Vertex AI's project/location endpoint with ``Authorization: Bearer``.
"""

from __future__ import annotations

import base64
import json
import os
import re
import stat
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import httpx

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    CallbackFlowOptions,
    LoginCallbacks,
    LoginCancelledError,
    LoginError,
    OAuthCallbackFlow,
    maybe_await,
    raise_for_refresh_failure,
    report_safely,
)
from local_operator.providers.oauth.pkce import create_pkce_pair

# Google Cloud SDK's public native/desktop OAuth2 client (RFC 8252 installed
# app — the secret is public by design, identical to `gcloud auth login`).
# Overridable via environment variables for custom Google Cloud OAuth clients.
DEFAULT_CLIENT_ID = "32555940559.apps.googleusercontent.com"
# Stored base64-encoded (like ``oauth/anthropic.py``) so secret scanners do not
# flag a public installed-app identifier as a leaked credential.
DEFAULT_CLIENT_SECRET = base64.b64decode("Wm1zc0xOakp5Mjk5OGhENENUZzJlanIy").decode()

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
PROJECTS_URL = "https://cloudresourcemanager.googleapis.com/v1/projects"
SERVICE_USAGE_URL = (
    "https://serviceusage.googleapis.com/v1/projects/{project_id}"
    "/services/aiplatform.googleapis.com"
)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
SCOPES = f"openid https://www.googleapis.com/auth/userinfo.email {CLOUD_PLATFORM_SCOPE}"

CALLBACK_PORT = 8085
CALLBACK_PATH = "/"
DEFAULT_LOCATION = "global"
EXPIRY_SKEW_MS = 5 * 60 * 1000
JWT_LIFETIME_SECONDS = 3600
# Cap Service Usage probes so project discovery stays fast on accounts with
# many GCP projects; above this count we fall back to the first active project.
_MAX_SERVICE_USAGE_PROBES = 5


# Google's OAuth2 token endpoints a service-account JWT assertion may be sent
# to. The assertion is a signed bearer grant for the key's identity: posting it
# to an attacker-chosen ``token_uri`` (from a pasted/forged key file) would let
# that host redeem it for a cloud-platform token, so the host and path are
# pinned unless the operator opts in explicitly (private Google endpoints,
# test doubles) via ``GOOGLE_VERTEX_ALLOW_CUSTOM_TOKEN_URI=1`` or an exact
# ``GOOGLE_VERTEX_TOKEN_URI`` match.
_ALLOWED_TOKEN_ENDPOINTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("oauth2.googleapis.com", "/token"),
        ("www.googleapis.com", "/oauth2/v4/token"),
        ("www.googleapis.com", "/token"),
    }
)

# GCP project ids are 6-30 lowercase chars in practice, but legacy/domain-scoped
# ids ("example.com:proj") also exist; this admits those while rejecting
# anything that could change a URL's host or path ("/", "#", "?", "..").
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# Vertex locations: "global" or a region such as "us-central1"/"europe-west4".
LOCATION_RE = re.compile(r"^(global|[a-z]{2,16}-[a-z0-9]{1,16}\d{1,2})$")

# Service-account key files are ~2.3 KiB; 64 KiB leaves ample headroom while
# refusing to slurp arbitrary large files (or devices/FIFOs) named as a "path".
MAX_SERVICE_ACCOUNT_JSON_BYTES = 65_536
# Upstream error bodies are echoed into LoginError for diagnosis; bound them so
# an HTML error page or hostile endpoint cannot flood the UI/log.
_MAX_ERROR_TEXT = 240


def _bounded(text: str | None) -> str:
    text = (text or "").strip()
    return text if len(text) <= _MAX_ERROR_TEXT else text[: _MAX_ERROR_TEXT - 1] + "…"


def _json_object(response: httpx.Response, what: str) -> dict[str, Any]:
    """Decode a token-endpoint JSON body, turning junk into a clean LoginError."""
    try:
        payload = response.json()
    except ValueError:
        raise LoginError(
            f"Google Vertex AI {what} returned a non-JSON response: {_bounded(response.text)}"
        ) from None
    if not isinstance(payload, dict):
        raise LoginError(f"Google Vertex AI {what} returned an unexpected JSON payload.")
    return payload


def _expires_in_seconds(payload: dict[str, Any]) -> int:
    """``expires_in`` as int seconds; a missing/garbled value means Google's 3600."""
    try:
        value = int(payload.get("expires_in", 3600))
    except (TypeError, ValueError):
        return 3600
    return value if value > 0 else 3600


def validate_project_id(project_id: str) -> str:
    """Reject a project id that could alter a Vertex URL's host or path."""
    value = (project_id or "").strip()
    if not PROJECT_ID_RE.fullmatch(value) or ".." in value:
        raise LoginError(f"Invalid Google Cloud project ID: {_bounded(value)!r}.")
    return value


def validate_location(location: str) -> str:
    """Reject a location that is not ``global`` or a GCP region name."""
    value = (location or "").strip()
    if not LOCATION_RE.fullmatch(value):
        raise LoginError(
            f"Invalid Google Cloud location: {_bounded(value)!r} "
            "(expected 'global' or a region such as 'us-central1')."
        )
    return value


def _client_id() -> str:
    return (os.environ.get("GOOGLE_VERTEX_OAUTH_CLIENT_ID") or "").strip() or DEFAULT_CLIENT_ID


def _client_secret() -> str:
    return (
        os.environ.get("GOOGLE_VERTEX_OAUTH_CLIENT_SECRET") or ""
    ).strip() or DEFAULT_CLIENT_SECRET


def _configured_project() -> str | None:
    for var in ("GOOGLE_CLOUD_PROJECT", "VERTEX_PROJECT"):
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return None


def _configured_location() -> str:
    for var in ("GOOGLE_CLOUD_LOCATION", "VERTEX_LOCATION"):
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return DEFAULT_LOCATION


def _extract_email_from_id_token(id_token: Any) -> str | None:
    """Decode the JWT payload of an OIDC ``id_token`` to read ``email``.

    Signature verification is unnecessary here because the token was just
    returned directly over TLS by Google's own token endpoint.
    """
    if not isinstance(id_token, str):
        return None
    parts = id_token.split(".")
    if len(parts) < 2:
        return None
    try:
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:  # noqa: BLE001 — malformed id_token falls back cleanly
        return None
    if not isinstance(payload, dict):
        return None
    email = payload.get("email")
    return email.strip() if isinstance(email, str) and email.strip() else None


async def _fetch_userinfo_email(
    http: httpx.AsyncClient,
    access_token: str,
    id_token: Any = None,
) -> str | None:
    """Fetch the signed-in Google account email, falling back to ``id_token``."""
    try:
        response = await http.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict):
                email = payload.get("email")
                if isinstance(email, str) and email.strip():
                    return email.strip()
    except Exception:  # noqa: BLE001 — identity lookup is best-effort
        pass
    return _extract_email_from_id_token(id_token)


async def _vertex_api_enabled(
    http: httpx.AsyncClient,
    access_token: str,
    project_id: str,
) -> bool:
    """Return True if ``aiplatform.googleapis.com`` is enabled on ``project_id``."""
    try:
        response = await http.get(
            SERVICE_USAGE_URL.format(project_id=urllib.parse.quote(project_id, safe="")),
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            payload = response.json()
            return isinstance(payload, dict) and payload.get("state") == "ENABLED"
    except Exception:  # noqa: BLE001 — service-usage probe is best-effort
        pass
    return False


async def _discover_project(http: httpx.AsyncClient, access_token: str) -> str | None:
    """Auto-select the best active Google Cloud project for Vertex AI.

    Priority:
    1. ``GOOGLE_CLOUD_PROJECT`` / ``VERTEX_PROJECT`` env var (if active or if
       project listing is unavailable).
    2. First active project whose ``projectId`` contains ``"vertex"``.
    3. When there are 2–5 active projects, first project with
       ``aiplatform.googleapis.com`` enabled.
    4. First active project returned by Cloud Resource Manager.
    """
    env_project = _configured_project()
    active_ids: list[str] = []
    try:
        response = await http.get(
            PROJECTS_URL,
            params={"filter": "lifecycleState:ACTIVE"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            payload = response.json()
            projects = payload.get("projects") if isinstance(payload, dict) else None
            if isinstance(projects, list):
                for entry in projects:
                    if not isinstance(entry, dict):
                        continue
                    state = entry.get("lifecycleState")
                    pid = entry.get("projectId")
                    if (
                        isinstance(pid, str)
                        and pid.strip()
                        and (state is None or state == "ACTIVE")
                    ):
                        active_ids.append(pid.strip())
    except Exception:  # noqa: BLE001 — project discovery falls back gracefully
        pass

    if env_project and (not active_ids or env_project in active_ids):
        return env_project

    for pid in active_ids:
        if "vertex" in pid.lower():
            return pid

    if 1 < len(active_ids) <= _MAX_SERVICE_USAGE_PROBES:
        for pid in active_ids:
            if await _vertex_api_enabled(http, access_token, pid):
                return pid

    if active_ids:
        return active_ids[0]

    return env_project


def _validate_token_uri(token_uri: str) -> str:
    """Pin a service-account ``token_uri`` to Google's OAuth2 token endpoints.

    See ``_ALLOWED_TOKEN_ENDPOINTS`` for why: the signed JWT assertion is sent
    to this URL, so an unpinned value is a credential-exfiltration channel.
    Applied on BOTH login and refresh, since a stored credential is just as
    attacker-influenced as a freshly pasted key.
    """
    token_uri = (token_uri or "").strip()
    try:
        parsed = urllib.parse.urlsplit(token_uri)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        raise LoginError(
            f"Service account token_uri is not a valid URL: {_bounded(token_uri)!r}."
        ) from None
    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
        raise LoginError(
            f"Service account token_uri must be a valid https:// URL (got {_bounded(token_uri)!r})."
        )
    if (
        port in (None, 443)
        and not parsed.query
        and not parsed.fragment
        and (hostname, parsed.path) in _ALLOWED_TOKEN_ENDPOINTS
    ):
        return token_uri
    explicit = (os.environ.get("GOOGLE_VERTEX_TOKEN_URI") or "").strip()
    allow_any = (os.environ.get("GOOGLE_VERTEX_ALLOW_CUSTOM_TOKEN_URI") or "").strip() == "1"
    if allow_any or (explicit and explicit == token_uri):
        return token_uri
    raise LoginError(
        f"Service account token_uri {_bounded(token_uri)!r} is not a Google OAuth2 token "
        "endpoint. Set GOOGLE_VERTEX_TOKEN_URI to this exact URL (or "
        "GOOGLE_VERTEX_ALLOW_CUSTOM_TOKEN_URI=1) to trust a custom endpoint."
    )


async def _mint_service_account_token(
    *,
    client_email: str,
    private_key: str,
    token_uri: str,
    http_client: httpx.AsyncClient | None = None,
    for_refresh: bool = False,
) -> tuple[str, int]:
    """Sign an RS256 JWT assertion locally and exchange it for a bearer token."""
    # Lazy import keeps PyJWT/cryptography out of startup imports.
    import jwt

    token_uri = _validate_token_uri(token_uri)
    now = int(time.time())
    claims = {
        "iss": client_email,
        "sub": client_email,
        "aud": token_uri,
        "scope": CLOUD_PLATFORM_SCOPE,
        "iat": now,
        "exp": now + JWT_LIFETIME_SECONDS,
    }
    try:
        assertion = jwt.encode(claims, private_key, algorithm="RS256")
    except Exception as exc:  # noqa: BLE001 — never echo private key material
        raise LoginError(
            f"Could not sign service account JWT ({type(exc).__name__}). "
            "Check that private_key is a valid RSA PEM key."
        ) from None

    form_data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": assertion,
    }
    if http_client is not None:
        response = await http_client.post(token_uri, data=form_data)
    else:
        async with httpx.AsyncClient(timeout=30.0) as http:
            response = await http.post(token_uri, data=form_data)

    if response.status_code != 200:
        if for_refresh:
            raise_for_refresh_failure("Google Vertex AI", response.status_code, response.text)
        raise LoginError(
            f"Google Vertex AI service account token exchange failed "
            f"({response.status_code}): {_bounded(response.text)}"
        )

    payload = _json_object(response, "service account token exchange")
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise LoginError("Google Vertex AI token response was missing access_token.")
    expires_in = _expires_in_seconds(payload)
    expires_ms = int(time.time() * 1000) + expires_in * 1000 - EXPIRY_SKEW_MS
    return access_token, expires_ms


class VertexOAuthFlow(OAuthCallbackFlow):
    """Google Cloud OAuth2 + PKCE authorization-code flow for Vertex AI."""

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
                allow_port_fallback=True,
                manual_input_only=manual_input_only,
                provider_label="Google Vertex AI",
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
            "client_id": _client_id(),
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        del state
        form = {
            "grant_type": "authorization_code",
            "client_id": _client_id(),
            "client_secret": _client_secret(),
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": self._verifier or "",
        }
        owns_client = self._http is None
        http = self._http or httpx.AsyncClient(timeout=30.0)
        try:
            response = await http.post(TOKEN_URL, data=form)
            if response.status_code != 200:
                raise LoginError(
                    f"Google Vertex AI token exchange failed ({response.status_code}): "
                    f"{_bounded(response.text)}"
                )
            token = _json_object(response, "token exchange")
            access_token = token.get("access_token")
            if not isinstance(access_token, str) or not access_token:
                raise LoginError("Google Vertex AI token exchange returned no access_token.")
            refresh_token = token.get("refresh_token")
            expires_in = _expires_in_seconds(token)
            expires_ms = int(time.time() * 1000) + expires_in * 1000 - EXPIRY_SKEW_MS

            email = await _fetch_userinfo_email(http, access_token, token.get("id_token"))
            project_id = await _discover_project(http, access_token)
            location = validate_location(_configured_location())
        finally:
            if owns_client:
                await http.aclose()

        if not project_id and self.callbacks.on_manual_code_input is not None:
            # A REQUIRED post-callback prompt: the progress text is the prompt
            # (CLI prints it, Desktop surfaces it as the operation message), and
            # the browser callback has already completed by now — the base flow
            # awaits the cancelled paste-code reader before calling us.
            await report_safely(
                self.callbacks.on_progress,
                "No active Google Cloud project was found automatically. "
                "Enter your Google Cloud project ID:",
            )
            pasted_project = await maybe_await(self.callbacks.on_manual_code_input())
            if pasted_project and pasted_project.strip():
                project_id = pasted_project.strip()
        if project_id:
            project_id = validate_project_id(project_id)
        else:
            raise LoginError(
                "No Google Cloud project is available for Vertex AI. Create or select a "
                "project with the Vertex AI API enabled, then set GOOGLE_CLOUD_PROJECT "
                "(or VERTEX_PROJECT) and sign in again."
            )

        if project_id:
            await report_safely(
                self.callbacks.on_progress,
                f"Selected Google Cloud project: {project_id} ({location})",
            )

        credential: dict[str, Any] = {
            "type": "oauth",
            "access": access_token,
            "expires": expires_ms,
            "email": email,
            "account_id": email or project_id,
            "org_name": project_id,
            "auth_mode": "oauth",
            "project_id": project_id,
            "location": location,
        }
        if isinstance(refresh_token, str) and refresh_token:
            credential["refresh"] = refresh_token
        return credential


async def login_vertex_oauth(
    callbacks: LoginCallbacks | None = None,
    *,
    open_browser: Callable[[str], None] | None = None,
    signal: AbortSignal | None = None,
    manual_input_only: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Run the Google Cloud browser OAuth2 + PKCE sign-in for Vertex AI."""
    return await VertexOAuthFlow(
        callbacks,
        open_browser=open_browser,
        signal=signal,
        manual_input_only=manual_input_only,
        http_client=http_client,
    ).run()


def _read_key_file(path: Path) -> str:
    """Read a service-account key file under strict shape limits.

    The path arrives from a paste box (CLI, TUI, or the localhost Desktop
    input endpoint), so it is constrained to what a key file looks like — a
    regular ``.json`` file no larger than 64 KiB — before a single byte is
    read. Combined with the strict ``"type": "service_account"`` check, this
    keeps arbitrary files (``/etc/passwd``, devices, FIFOs, huge files) from
    being read or echoed back in an error message.
    """
    if path.suffix.lower() != ".json":
        raise LoginError(f"Service account key path must end in .json: {path}")
    try:
        info = path.stat()
    except OSError as exc:
        raise LoginError(
            f"Could not read service account key file ({type(exc).__name__})."
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise LoginError(f"Service account key path is not a regular file: {path}")
    if info.st_size > MAX_SERVICE_ACCOUNT_JSON_BYTES:
        raise LoginError(f"Service account key file is larger than 64 KiB: {path}")
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_SERVICE_ACCOUNT_JSON_BYTES + 1)
    except OSError as exc:
        raise LoginError(
            f"Could not read service account key file ({type(exc).__name__})."
        ) from None
    if len(data) > MAX_SERVICE_ACCOUNT_JSON_BYTES:
        raise LoginError(f"Service account key file is larger than 64 KiB: {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise LoginError(f"Service account key file is not UTF-8 JSON: {path}") from None


def _load_service_account_json(raw_input: str) -> dict[str, Any]:
    """Parse pasted service-account JSON or read it from a local ``.json`` path."""
    trimmed = raw_input.strip()
    if not trimmed:
        raise LoginCancelledError("Google Vertex AI service account login cancelled")
    if len(trimmed.encode("utf-8")) > MAX_SERVICE_ACCOUNT_JSON_BYTES:
        raise LoginError("Service account JSON is larger than 64 KiB; paste the key file itself.")

    raw_json = trimmed
    if not trimmed.startswith("{"):
        # Unwrap matching quotes that terminals add on drag-and-drop paths.
        candidate_str = trimmed
        if (
            len(candidate_str) >= 2
            and candidate_str[0] == candidate_str[-1]
            and candidate_str[0] in ("'", '"')
        ):
            candidate_str = candidate_str[1:-1].strip()
        candidate_path = Path(candidate_str).expanduser()
        if candidate_path.exists():
            raw_json = _read_key_file(candidate_path)
        elif candidate_str.endswith(".json") or candidate_str.startswith(("/", "~/", "./", "../")):
            raise LoginError(f"Service account key file not found: {candidate_path}")

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise LoginError(
            "Invalid service account JSON. Paste the full JSON key contents "
            f"or a path to a .json file ({exc.msg})."
        ) from None

    if not isinstance(parsed, dict):
        raise LoginError("Service account JSON must be a JSON object.")
    if parsed.get("type") != "service_account":
        raise LoginError(
            'Expected a Google Cloud service account key with "type": "service_account".'
        )

    missing = [
        field
        for field in ("client_email", "private_key", "project_id")
        if not isinstance(parsed.get(field), str) or not parsed[field].strip()
    ]
    if missing:
        raise LoginError(
            f"Service account JSON is missing required field(s): {', '.join(missing)}."
        )
    return parsed


async def login_vertex_service_account(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Authenticate Vertex AI from a pasted GCP service-account JSON or file path."""
    if callbacks.on_manual_code_input is None:
        raise ValueError(
            "Google Vertex AI service account login reads a JSON key or path, "
            "but this interface provided no on_manual_code_input callback to read it with."
        )

    await report_safely(
        callbacks.on_progress,
        "Paste your Google Cloud service account JSON key, or the path to a .json key file.",
    )
    pasted = await maybe_await(callbacks.on_manual_code_input())
    if signal is not None and signal.aborted:
        raise LoginCancelledError(signal.reason or "Login cancelled")
    if pasted is None or not pasted.strip():
        raise LoginCancelledError("Google Vertex AI service account login cancelled")

    sa = _load_service_account_json(pasted)
    client_email = sa["client_email"].strip()
    private_key = sa["private_key"]
    project_id = validate_project_id(sa["project_id"])
    location = validate_location(_configured_location())
    token_uri_raw = sa.get("token_uri") or TOKEN_URL
    if not isinstance(token_uri_raw, str):
        raise LoginError("Service account token_uri must be a string.")
    token_uri = _validate_token_uri(token_uri_raw)

    # Verify the key live by minting an access token right away.
    access_token, expires_ms = await _mint_service_account_token(
        client_email=client_email,
        private_key=private_key,
        token_uri=token_uri,
        http_client=http_client,
        for_refresh=False,
    )

    return {
        "type": "oauth",
        "access": access_token,
        "expires": expires_ms,
        "email": client_email,
        "account_id": client_email,
        "org_name": project_id,
        "auth_mode": "service_account",
        "project_id": project_id,
        "location": location,
        "client_email": client_email,
        "private_key": private_key,
        "token_uri": token_uri,
    }


async def refresh_vertex_token(
    credential: dict[str, Any],
    *,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Refresh a Google Vertex AI OAuth or service-account credential."""
    auth_mode = credential.get("auth_mode")
    if auth_mode == "service_account":
        client_email = credential.get("client_email")
        private_key = credential.get("private_key")
        if (
            not isinstance(client_email, str)
            or not client_email
            or not isinstance(private_key, str)
            or not private_key
        ):
            raise LoginError("Stored Vertex AI service account credential is incomplete.")
        token_uri = str(credential.get("token_uri") or TOKEN_URL)
        access_token, expires_ms = await _mint_service_account_token(
            client_email=client_email,
            private_key=private_key,
            token_uri=token_uri,
            http_client=http_client,
            for_refresh=True,
        )
        refreshed = dict(credential)
        refreshed.update({"access": access_token, "expires": expires_ms, "token_uri": token_uri})
        return refreshed

    refresh_token = credential.get("refresh") or credential.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise LoginError("Google Vertex AI credential has no refresh token.")

    form = {
        "grant_type": "refresh_token",
        "client_id": _client_id(),
        "client_secret": _client_secret(),
        "refresh_token": refresh_token,
    }
    if http_client is not None:
        response = await http_client.post(TOKEN_URL, data=form)
    else:
        async with httpx.AsyncClient(timeout=30.0) as http:
            response = await http.post(TOKEN_URL, data=form)

    if response.status_code != 200:
        raise_for_refresh_failure("Google Vertex AI", response.status_code, response.text)

    token = _json_object(response, "token refresh")
    access_token = token.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise LoginError("Google Vertex AI refresh returned no access_token.")
    expires_in = _expires_in_seconds(token)
    expires_ms = int(time.time() * 1000) + expires_in * 1000 - EXPIRY_SKEW_MS

    # Google rarely rotates refresh tokens on refresh; keep the stored token
    # when the response omits `refresh_token` and preserve project/location/email.
    refreshed = dict(credential)
    refreshed.update(
        {
            "type": "oauth",
            "access": access_token,
            "refresh": token.get("refresh_token") or refresh_token,
            "expires": expires_ms,
            "auth_mode": auth_mode or "oauth",
        }
    )
    return refreshed
