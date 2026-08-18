"""Z.AI / GLM OAuth — browser sign-in that provisions a durable API key.

Ported from omp's ``registry/oauth/zai.ts``, which in turn mirrors ZCode's
desktop "Individual Plan" sign-in. The shape is unusual enough to be worth
stating up front, because it is not the authorization-code flow the other
providers in this package run:

1. An authorization-code request against ``chat.z.ai`` — **no PKCE**, because
   the provider's own client sends none and the token endpoint rejects the
   extra parameters.
2. A **non-standard** JSON token exchange (no ``grant_type``, no
   ``code_verifier``) that returns a SHORT-LIVED OAuth access token nested at
   ``data.zai.access_token``.
3. A business-API sequence that trades that token for a DURABLE ``id.secret``
   API key: business-login → resolve the default org/project → find-or-create
   this app's named key → read its secret.

Step 3 is the reason this file exists rather than a paste-a-key login. The
short-lived OAuth token is NOT accepted by the inference endpoint, so a flow
that stopped at step 2 would authenticate and then fail every request. The
minted key IS the credential the wire wants, so it is stored in ``access`` and
returned verbatim as the bearer — no dialect change anywhere downstream.

Session persistence: the minted key does not expire, which is expressed as
``expires: None``. :meth:`AuthStore._needs_refresh` reads that as "static
token; never expires" and so never attempts a refresh — correct here, because
there is no refresh token to attempt one with. The row therefore survives
restarts and is reused across sessions exactly like an API key, while still
being an ``oauth`` credential for identity and usage-reporting purposes.
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
    LoginCancelledError,
    LoginError,
    OAuthCallbackFlow,
)


def _env(name: str, default: str) -> str:
    """Env override for an endpoint, falling back to the shipped default.

    Mirrors omp's overridable constants: the CN-facing deployment of the same
    product answers on different hosts, and a user pointed at it must be able
    to log in without a code change.
    """
    value = os.environ.get(name)
    return value if value else default


CLIENT_ID = _env("ZAI_OAUTH_CLIENT_ID", "client_P8X5CMWmlaRO9gyO-KSqtg")
AUTHORIZE_URL = _env("ZAI_OAUTH_AUTHORIZE_URL", "https://chat.z.ai/api/oauth/authorize")
TOKEN_URL = _env("ZAI_OAUTH_TOKEN_URL", "https://zcode.z.ai/api/v1/oauth/token")
BIZ_BASE = _env("ZAI_BIZ_BASE", "https://api.z.ai")
BUSINESS_LOGIN_URL = _env("ZAI_BUSINESS_LOGIN_URL", "https://api.z.ai/api/auth/z/login")

#: The name this app's provisioned key carries in the Z.AI console. Distinct
#: from ZCode's own ``zcode-api-key`` so signing in here never rotates or
#: overwrites the key another tool depends on.
KEY_NAME = "local-operator"

#: Pinned by the provider's OAuth client registration: the redirect URI is on
#: the allowlist for this exact port, so a fallback port would be refused.
CALLBACK_PORT = 54548
CALLBACK_PATH = "/callback"

_TIMEOUT_S = 30.0


def _is_success_code(code: Any) -> bool:
    """Z.AI's two success spellings.

    The OAuth token endpoint signals success with ``code: 0`` while the biz
    endpoints use ``code: 200``. A body with no status field at all is also a
    success — some endpoints return the payload bare.
    """
    if code is None:
        return True
    if isinstance(code, bool):
        return False
    if isinstance(code, (int, float)):
        return int(code) in (0, 200)
    if isinstance(code, str):
        return code in ("0", "200")
    return False


def _unwrap(body: Any, operation: str) -> Any:
    """Unwrap the ``{code, msg, data, success}`` envelope, raising on failure.

    The envelope reports application-level failures with HTTP 200, so a caller
    that only checked the status code would read an error body as a credential.
    """
    if isinstance(body, dict) and ("code" in body or "success" in body):
        if body.get("success") is False or not _is_success_code(body.get("code")):
            detail = body.get("msg") or f"code {body.get('code')}"
            raise LoginError(f"Z.AI {operation} failed: {detail}")
        return body.get("data") if "data" in body else body
    return body


def _trimmed(value: Any) -> str | None:
    """A non-empty trimmed string, or ``None`` for anything else."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _failed(operation: str, response: httpx.Response) -> LoginError:
    """A transport failure, named by OPERATION and status.

    Neither the body nor the URL is echoed. The bodies come from endpoints that
    are handed bearer tokens and quote request context back, and the URLs embed
    the account's organization id, project id and api-key id -- while a login
    error is shown in the TUI and written to the log. The operation name says
    which step failed, which is what a reader needs in order to act, without
    putting account identifiers in front of them.
    """
    return LoginError(
        f"Z.AI {operation} failed ({response.status_code} {response.reason_phrase})".rstrip()
    )


async def _get_json(
    http: httpx.AsyncClient, url: str, headers: dict[str, str], operation: str
) -> Any:
    response = await http.get(url, headers=headers)
    if response.status_code != 200:
        raise _failed(operation, response)
    return response.json() if response.content else None


async def _post_json(
    http: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    operation: str,
    headers: dict[str, str] | None = None,
) -> Any:
    response = await http.post(url, json=body, headers=headers or {})
    if response.status_code != 200:
        raise _failed(operation, response)
    return response.json() if response.content else None


def _key_array(value: Any) -> list[dict[str, Any]]:
    """Coerce an api-keys listing to a list across the shapes Z.AI returns."""
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    if isinstance(value, dict):
        for field in ("list", "keys", "apiKeys", "records"):
            nested = value.get(field)
            if isinstance(nested, list):
                return [entry for entry in nested if isinstance(entry, dict)]
    return []


async def _business_login(http: httpx.AsyncClient, oauth_access_token: str) -> str:
    """Trade the short-lived OAuth token for the biz token the APIs require."""
    data = _unwrap(
        await _post_json(http, BUSINESS_LOGIN_URL, {"token": oauth_access_token}, "business login"),
        "business login",
    )
    token = None
    if isinstance(data, dict):
        token = _trimmed(data.get("access_token")) or _trimmed(data.get("accessToken"))
    if not token:
        raise LoginError("Z.AI business login returned no access token")
    return token


async def mint_zai_api_key(http: httpx.AsyncClient, oauth_access_token: str) -> str:
    """Provision the durable ``id.secret`` key from a short-lived OAuth token.

    business-login → default org/project → find-or-create this app's key →
    read its secret. Exposed (not private) so a test can exercise the
    provisioning sequence without driving a browser flow.
    """
    biz_token = await _business_login(http, oauth_access_token)
    auth = {"Authorization": f"Bearer {biz_token}"}

    customer = _unwrap(
        await _get_json(
            http, f"{BIZ_BASE}/api/biz/customer/getCustomerInfo", auth, "customer lookup"
        ),
        "customer lookup",
    )
    orgs = customer.get("organizations") if isinstance(customer, dict) else None
    orgs = [o for o in orgs if isinstance(o, dict)] if isinstance(orgs, list) else []
    org = next((o for o in orgs if o.get("isDefault")), orgs[0] if orgs else None)
    projects = org.get("projects") if isinstance(org, dict) else None
    projects = [p for p in projects if isinstance(p, dict)] if isinstance(projects, list) else []
    project = next((p for p in projects if p.get("isDefault")), projects[0] if projects else None)

    organization_id = _trimmed(org.get("organizationId")) if org else None
    project_id = _trimmed(project.get("projectId")) if project else None
    if not organization_id or not project_id:
        raise LoginError("Z.AI key provisioning failed: no organization/project on this account")

    keys_url = (
        f"{BIZ_BASE}/api/biz/v1/organization/{organization_id}/projects/{project_id}/api_keys"
    )
    existing = next(
        (
            k
            for k in _key_array(
                _unwrap(await _get_json(http, keys_url, auth, "api key list"), "api key list")
            )
            if k.get("name") == KEY_NAME
        ),
        None,
    )
    record = existing or _unwrap(
        await _post_json(http, keys_url, {"name": KEY_NAME}, "api key create", auth),
        "api key create",
    )
    api_key = _trimmed(record.get("apiKey")) if isinstance(record, dict) else None
    if not api_key:
        raise LoginError("Z.AI key provisioning returned no apiKey")

    # ALWAYS read the secret from the copy endpoint. List entries mask it
    # (`*****abcd`) and the create response's inline secret is not reliable
    # across account states, so anything else can persist a key that cannot
    # authenticate — and it would fail at the first request, not at login.
    copied = _unwrap(
        await _get_json(
            http, f"{keys_url}/copy/{urllib.parse.quote(api_key)}", auth, "api key copy"
        ),
        "api key copy",
    )
    secret_key = _trimmed(copied.get("secretKey")) if isinstance(copied, dict) else None
    if not secret_key:
        raise LoginError("Z.AI key provisioning returned no secretKey")
    return f"{api_key}.{secret_key}"


class ZaiOAuthFlow(OAuthCallbackFlow):
    """Authorization-code sign-in against chat.z.ai (no PKCE), then key minting."""

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
                # The redirect URI is registered against this exact port, so a
                # fallback port would produce a redirect the IdP refuses.
                allow_port_fallback=False,
                manual_input_only=manual_input_only,
            ),
            callbacks,
            open_browser=open_browser,
            signal=signal,
        )
        self._http = http_client

    async def generate_auth_url(self, state: str, redirect_uri: str) -> str:
        params = {
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "client_id": CLIENT_ID,
            "state": state,
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        # Checked BEFORE the exchange, as the reference implementation does.
        # This flow has a side effect no other login here has: it CREATES a
        # named API key in the user's Z.AI console. Minting one for a login the
        # user has already cancelled leaves a credential they never agreed to
        # and did not see created.
        if self._signal is not None and self._signal.aborted:
            raise LoginCancelledError(self._signal.reason or "Z.AI login cancelled")

        owns_client = self._http is None
        http = self._http or httpx.AsyncClient(timeout=_TIMEOUT_S)
        try:
            body = await _post_json(
                http,
                TOKEN_URL,
                # Deliberately NOT an RFC 6749 body: the provider's own client
                # posts these fields and the endpoint rejects grant_type.
                {"provider": "zai", "code": code, "redirect_uri": redirect_uri, "state": state},
                "token exchange",
            )
            data = _unwrap(body, "token exchange")
            zai = data.get("zai") if isinstance(data, dict) else None
            access_token = _trimmed(zai.get("access_token")) if isinstance(zai, dict) else None
            if not access_token:
                raise LoginError("Z.AI token response is missing an access token")

            minted = await mint_zai_api_key(http, access_token)
        finally:
            if owns_client:
                await http.aclose()

        user = data.get("user") if isinstance(data, dict) else None
        creds: dict[str, Any] = {
            # The MINTED key, not the OAuth token: this is what the inference
            # endpoint accepts, and `_oauth_api_key` hands it straight to the wire.
            "access": minted,
            # No refresh token exists and the minted key does not expire.
            # `expires: None` is AuthStore's "static token" marker, so no
            # refresh is ever attempted and the row persists across sessions.
            "expires": None,
            "authorized_at": int(time.time() * 1000),
        }
        if isinstance(user, dict):
            email = _trimmed(user.get("email"))
            if email:
                creds["email"] = email
            account_id = user.get("id")
            if isinstance(account_id, (str, int)) and str(account_id).strip():
                creds["account_id"] = str(account_id).strip()
        return creds


async def login_zai(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
    open_browser: Callable[[str], None] | None = None,
    manual_input_only: bool = False,
) -> dict[str, Any]:
    """Run the interactive Z.AI sign-in; returns the OAuthCredentials dict."""
    flow = ZaiOAuthFlow(
        callbacks,
        open_browser=open_browser,
        signal=signal,
        manual_input_only=manual_input_only,
        http_client=http_client,
    )
    return await flow.run()
