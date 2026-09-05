"""Registry-owned provider capabilities and ephemeral desktop sign-in controls."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, SecretStr

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    credential_provider_id,
    env_key_name,
    get_provider_definition,
    resolve_env_key,
)
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.utils.desktop_auth import DesktopAuth, LoginOperation

router = APIRouter(tags=["Authentication"], dependencies=[Depends(require_desktop)])


class AccountRemoval(BaseModel):
    id: int


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str


class SecretInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: SecretStr


class LoginInput(SecretInput):
    prompt_id: str


async def get_desktop_auth(request: Request) -> DesktopAuth:
    host = getattr(request.app.state, "desktop_auth", None)
    if host is None:
        manager = request.app.state.credential_manager
        root = request.app.state.config_manager.config_dir
        host = DesktopAuth(AuthStore(root / "auth.db", credential_manager=manager), manager)
        request.app.state.desktop_auth = host
    return host


def _reply(result: Any, message: str = "Provider controls retrieved.") -> CRUDResponse[Any]:
    return CRUDResponse(status=200, message=message, result=result)


def _operation(host: DesktopAuth, operation_id: str) -> LoginOperation:
    op = host.operations.get(operation_id)
    if op is None:
        raise HTTPException(404, "This sign-in is no longer available. Start again.")
    return op


def _secret(body: SecretInput) -> str:
    value = body.value.get_secret_value().strip()
    if not value or len(value) > 32768:
        raise HTTPException(422, "Enter a non-empty value of at most 32768 characters.")
    return value


@router.get("/v1/auth/providers", response_model=CRUDResponse)
async def providers(host: DesktopAuth = Depends(get_desktop_auth)):
    controller = host.controller()
    try:
        rows = []
        for provider in PROVIDER_REGISTRY:
            storage_id = credential_provider_id(provider.id)
            if storage_id != provider.id or provider.wire == "mock":
                continue
            storage = get_provider_definition(storage_id) or provider
            methods = [
                {
                    "id": method.id,
                    "label": method.name,
                    "kind": method.login_kind,
                    # Registry-declared flavours already carry distinct ids
                    # (`openai-device`, `zai-oauth`), so their chooser key is
                    # just that id. The field exists so EVERY method has one
                    # stable key, rather than the renderer guessing which
                    # providers happen to need one.
                    "method_id": method.id,
                    "requires_secret_input": method.paste_prompt_required,
                    "paste_fallback": method.paste_code_flow,
                }
                for method in PROVIDER_REGISTRY
                if credential_provider_id(method.id) == storage_id and method.login is not None
            ]
            if storage.env_keys is not None and not any(m["kind"] == "api_key" for m in methods):
                # A SYNTHESIZED method, so it needs a key distinct from the
                # browser/device method it sits beside. Both used to be `id:
                # provider.id`, so anthropic shipped two methods called
                # "anthropic": the React key collided and
                # `find(c => c.id === methodId)` always resolved the FIRST
                # match, leaving the API-key panel unreachable (design D2).
                #
                # `method_id` is the CHOOSER identity; `id` stays the provider
                # the flow acts on. They are genuinely two different things and
                # were only ever equal by coincidence -- `auth.key` and
                # `auth.start` resolve a PROVIDER through
                # `credential_provider_id`, which knows nothing about a
                # per-method suffix, so overloading `id` here would send a saved
                # key to a provider that does not exist.
                methods.append(
                    {
                        "id": provider.id,
                        "method_id": f"{provider.id}:api-key",
                        "label": "API key",
                        "kind": "api_key",
                        "requires_secret_input": True,
                        "paste_fallback": False,
                    }
                )
            rows.append(
                {
                    "id": provider.id,
                    "name": provider.name,
                    "auth_methods": methods,
                    "storage_id": storage_id,
                    "search_aliases": list(provider.search_aliases),
                    "login_kind": provider.login_kind,
                    "accepts_api_key": storage.env_keys is not None,
                    "local": provider.allows_missing_api_key,
                    "credential_name": env_key_name(storage_id),
                    "paste_required": provider.paste_prompt_required,
                    "paste_supported": provider.accepts_paste_prompt,
                    # Configured is not verified: this census never refreshes a
                    # grant or contacts a provider just because Settings opened.
                    "configured": controller.is_usable(provider.id),
                    # `configured` for a LOCAL provider means only "needs no
                    # credential", which is not "reachable" -- nothing here has
                    # contacted the server. Rendering the two as one fact put a
                    # green "Connected" badge on five local providers with
                    # nothing listening (design D1 / UX U1). Callers that want
                    # to state reachability must probe, on an explicit action.
                    "credential_optional": provider.allows_missing_api_key,
                    # A credential the app could actually run on, from the store
                    # OR the environment -- `stored_credentials` counts only the
                    # store, so an env-key provider reads as 0 while being fully
                    # usable, and grouping on that count alone would mislabel it.
                    "has_credential": controller.has_any_credential(provider.id)
                    or bool(resolve_env_key(storage_id)),
                    "stored_credentials": len(host.store.list_credentials(storage_id)),
                    "base_url": provider.base_url,
                }
            )
        return _reply({"providers": rows})
    finally:
        controller.close()


@router.get("/v1/auth/status", response_model=CRUDResponse)
async def account_status(host: DesktopAuth = Depends(get_desktop_auth)):
    accounts = []
    for row in host.store.list_credentials():
        if get_provider_definition(row.provider) is None:
            # MCP registrations/grants have their own server-scoped lifecycle;
            # a DCR-only row is not a signed-in model-provider account.
            continue
        # Select fields explicitly. StoredCredential.data contains full grants;
        # dataclass/asdict serialization here would turn status into a token API.
        label = row.data.get("email") or row.data.get("account_id") or row.data.get("org_name")
        accounts.append(
            {
                "id": row.id,
                "provider": row.provider,
                "type": row.credential_type,
                "identity_label": str(label)[:256] if label else "Stored credential",
                "source": "oauth" if row.credential_type == "oauth" else "api_key",
                "state": (
                    "refresh_due"
                    if isinstance(row.data.get("expires"), (int, float))
                    and row.data["expires"] <= time.time() * 1000
                    else "configured"
                ),
                "expires_at": (
                    row.data.get("expires")
                    if isinstance(row.data.get("expires"), (int, float))
                    else None
                ),
            }
        )
    return _reply({"accounts": accounts})


@router.delete("/v1/auth/accounts/{account_id}", response_model=CRUDResponse[AccountRemoval])
async def remove_account(account_id: int, host: DesktopAuth = Depends(get_desktop_auth)):
    row = host.store.get_credential(account_id)
    if row is None or get_provider_definition(row.provider) is None:
        raise HTTPException(404, "Provider account not found")
    host.store.delete_credential(account_id)
    from local_operator.providers.auth_cli import _invalidate_cached_listing

    _invalidate_cached_listing(row.provider)
    return _reply(
        {"id": account_id}, "Stored account removed. Environment credentials are unchanged."
    )


@router.post("/v1/auth/login", response_model=CRUDResponse)
async def login(body: LoginRequest, host: DesktopAuth = Depends(get_desktop_auth)):
    try:
        op = host.start(body.provider)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from None
    return _reply(op.snapshot(), "Sign-in started.")


@router.get("/v1/auth/operations/{operation_id}", response_model=CRUDResponse)
async def operation(operation_id: str, host: DesktopAuth = Depends(get_desktop_auth)):
    return _reply(_operation(host, operation_id).snapshot())


@router.post("/v1/auth/operations/{operation_id}/input", response_model=CRUDResponse)
async def operation_input(
    operation_id: str, body: LoginInput, host: DesktopAuth = Depends(get_desktop_auth)
):
    op = _operation(host, operation_id)
    pending = op.pending_input
    if pending is None or pending.done() or body.prompt_id != op.prompt_id:
        raise HTTPException(409, "This sign-in is not waiting for input.")
    pending.set_result(_secret(body))
    return _reply(op.snapshot(), "Response submitted.")


@router.delete("/v1/auth/operations/{operation_id}", response_model=CRUDResponse)
async def cancel_operation(operation_id: str, host: DesktopAuth = Depends(get_desktop_auth)):
    op = _operation(host, operation_id)
    await host.cancel(op)
    return _reply(op.snapshot(), "Sign-in cancelled.")


@router.put("/v1/auth/providers/{provider_id}/key", response_model=CRUDResponse)
async def save_key(
    provider_id: str, body: SecretInput, host: DesktopAuth = Depends(get_desktop_auth)
):
    storage_id = credential_provider_id(provider_id)
    definition = get_provider_definition(storage_id)
    if definition is None or definition.env_keys is None:
        raise HTTPException(422, "This provider does not accept an API key.")
    # AuthStore owns alias translation and source precedence. Use the same
    # login tier as the terminal; do not create a renderer-owned secret store.
    host.store.upsert_credential(
        storage_id, {"type": "api_key", "source": "login", "key": _secret(body)}
    )
    from local_operator.providers.auth_cli import _invalidate_cached_listing

    _invalidate_cached_listing(storage_id)
    return _reply({}, "API key saved.")


@router.post("/v1/auth/providers/{provider_id}/probe", response_model=CRUDResponse)
async def probe_provider(provider_id: str, host: DesktopAuth = Depends(get_desktop_auth)):
    """Actually contact a LOCAL provider's server and report what happened.

    Exists because the provider grid must not claim reachability it never
    checked (design D1 / UX U1). Reachability is a fact with a cost -- a
    network round trip that can hang -- so it is an EXPLICIT action rather
    than something a render triggers: "no network call behind first paint" is
    the binding constraint from that review.

    Restricted to `allows_missing_api_key` providers, whose base URL is a
    loopback/LAN server the user runs. The renderer names a provider; it never
    supplies a URL, so this cannot become a general request forwarder.
    """
    import httpx

    definition = get_provider_definition(credential_provider_id(provider_id))
    if definition is None or not definition.allows_missing_api_key:
        raise HTTPException(422, "Only a local provider's server can be tested.")
    base_url = definition.base_url
    if not base_url:
        raise HTTPException(422, "This provider has no server address to test.")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(base_url.rstrip("/") + "/models")
    except httpx.HTTPError:
        # The exception text can carry the resolved endpoint and proxy details.
        # The user needs the verdict and the address they configured, which the
        # grid already shows, not a transport dump.
        return _reply(
            {"reachable": False, "detail": "No server answered at this address."},
            "Provider not reachable.",
        )
    # Any HTTP answer proves something is listening and speaking HTTP, which is
    # the question asked. A 401/404 from a running server is still "running".
    return _reply(
        {"reachable": True, "detail": f"A server answered ({response.status_code})."},
        "Provider reachable.",
    )


@router.delete("/v1/auth/providers/{provider_id}/credentials", response_model=CRUDResponse)
async def logout(provider_id: str, host: DesktopAuth = Depends(get_desktop_auth)):
    controller = host.controller()
    try:
        await controller.logout(provider_id)
    except ValueError:
        raise HTTPException(422, "No stored credentials were found for this provider.")
    finally:
        controller.close()
    return _reply({}, "Stored credentials removed. Environment credentials are unchanged.")
