"""Registry-owned provider capabilities and ephemeral desktop sign-in controls."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, SecretStr

from local_operator.config import ConfigManager
from local_operator.i18n.keys import wire_errors_auth as w
from local_operator.model.defaults import SuggestedModel, suggested_model_for
from local_operator.providers import key_check
from local_operator.providers.auth_store import AuthStore, credential_identity
from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    credential_provider_id,
    env_key_name,
    get_provider_definition,
    provider_brand,
)
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.utils.desktop_auth import (
    LOGIN_READY_TIMEOUT_S,
    DesktopAuth,
    LoginOperation,
    SignInUnavailableError,
    apply_desktop_login_defaults,
)

# Only route-owned error constructors are catalogued here. Lower-layer reasons
# remain opaque params: parsing or rewording them could alter provider verdicts
# or undo their existing redaction. The released desktop reader takes envelope
# text first; English sentences and admission/status decisions stay unchanged.
router = APIRouter(tags=["Authentication"], dependencies=[Depends(require_desktop)])

logger = logging.getLogger(__name__)


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


async def get_desktop_auth(
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
) -> DesktopAuth:
    # The config manager arrives as a DECLARED dependency rather than being read
    # off `app.state` here. FastAPI's `dependency_overrides` only substitutes what
    # a signature declares, so reading state directly made this helper opaque to
    # them -- and it is reached INDIRECTLY (`get_provider_auth_store` calls it,
    # and the models routes depend on that), so mounting one router and
    # overriding its dependencies, the documented way to test a route in
    # isolation, still landed on the un-overridable read.
    #
    # The store's directory comes from the canonical resolver rather than from a
    # manager attribute. `app.py` builds every manager from this one call, so it
    # is the same directory by construction, and it does not make the location
    # of the auth database depend on which manager a caller happened to inject.
    from local_operator.paths import config_dir as resolve_config_dir

    host = getattr(request.app.state, "desktop_auth", None)
    if host is None:
        root = resolve_config_dir()
        # The server's own manager, so a sign-in's defaults are written through
        # the same instance every other config route reads.
        host = DesktopAuth(AuthStore(root / "auth.db", config_dir=root), root, manager)
        # The re-home hook: after the credential lands, repair the bound sessions
        # it stranded (see utils/desktop_rehome). It reads app state lazily and
        # holds the APP, never this request — the host object outlives it. The
        # login provider arrives per call: the re-home is first-login-only and
        # only the call site knows which provider this call added.
        from local_operator.server.utils.desktop_rehome import rehome_after_login

        app = request.app

        def after_login(provider: str):
            # The hook runs only after a VERIFIED sign-in, so it is also where the
            # hub runner learns the login exists (see ``_wake_hub_sync``).
            _wake_hub_sync(app, provider)
            return rehome_after_login(app, provider)

        host.rehome = after_login
        request.app.state.desktop_auth = host
    return host


def _wake_hub_sync(app: Any, provider: str) -> None:
    """Ask the hub update runner for a check now, after a Radient sign-in.

    WHY: checks that ran while signed out persisted ``no-credential`` per linked
    item, and only the next successful check clears it -- up to ~16.5 min away on
    the schedule. Radient only: no other provider's login changes what the hub
    can read. Best effort and never raises (``HubSyncRunner.wake``); a server with
    no runner (embedded host, tests) has nothing to wake.

    Only an in-process sign-in reaches here. ``lop login radient`` in a terminal
    writes the auth store from another process, and the server has no watcher on
    that store to wake from, so a CLI sign-in still waits for the next tick.
    """

    if provider != "radient":
        return
    runner = getattr(getattr(app, "state", None), "hub_sync", None)
    wake = getattr(runner, "wake", None)
    if callable(wake):
        wake()


def _suggestion(hosting: str, *, oauth: bool = False) -> dict[str, str] | None:
    """The wire shape of ``model.defaults.suggested_model_for``.

    ``{"id", "name"}`` or ``None``. The renderer shows this ("Suggested: Claude
    Opus 5.5") before any sign-in, and it is the model a first sign-in will set,
    so the two must come from the same table -- which is why the table lives in
    the backend and the renderer carries no copy.
    """
    suggested: SuggestedModel | None = suggested_model_for(hosting, oauth=oauth)
    return None if suggested is None else {"id": suggested.id, "name": suggested.name}


def _reply(result: Any, message: str = "Provider controls retrieved.") -> CRUDResponse[Any]:
    return CRUDResponse(status=200, message=message, result=result)


def _operation(host: DesktopAuth, operation_id: str) -> LoginOperation:
    op = host.operations.get(operation_id)
    if op is None:
        raise HTTPException(404, w.sign_in_missing().envelope())
    return op


def _secret(body: SecretInput) -> str:
    value = body.value.get_secret_value().strip()
    if not value or len(value) > 32768:
        raise HTTPException(422, w.input_invalid().envelope())
    return value


@router.get("/v1/auth/providers", response_model=CRUDResponse)
async def providers(host: DesktopAuth = Depends(get_desktop_auth)):
    controller = host.controller()
    try:
        # The registry VIEW, assembled once (``ProviderController.view_rows``),
        # rather than derived per field here: `configured`, `has_credential`,
        # `stored_credentials`, `state` and `identity` are one credential read,
        # and the TUI's pickers render the same row -- so the census cannot
        # answer "needs login" where `/login ` says "env key".
        view = controller.view_rows()
        if view is None:
            # No degradation exists for this route: the census IS a credential
            # read, so "cannot tell" has to be an error rather than an empty
            # list that reads as "no credentials anywhere". 503 rather than the
            # 500 this used to raise through the stack: a locked or missing
            # store is an environment fact the client may retry, not a bug it
            # should report.
            raise HTTPException(
                503,
                w.store_unreadable().envelope(),
            )
        rows = []
        for view_row in view:
            provider = view_row.definition
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
                    # Per METHOD, because the route decides the spelling: a Kimi
                    # OAuth grant reaches a host that names K3 `k3`, a Kimi key
                    # one that names it `kimi-k3`.
                    "suggested_model": _suggestion(
                        storage_id, oauth=method.login_kind != "api_key"
                    ),
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
                        "suggested_model": _suggestion(storage_id),
                    }
                )
            rows.append(
                {
                    "id": provider.id,
                    "name": provider.name,
                    # The clean title, server-owned: a client shows `brand`
                    # beside the id without re-deriving it from `name` (the
                    # parenthetical is a login-flavour qualifier, not a brand).
                    "brand": provider_brand(provider),
                    # The registry's WIRE fact, sorted for a stable wire (the
                    # field is a frozenset, whose iteration order is not).
                    "capabilities": sorted(provider.capabilities),
                    # The MACHINE state of the TUI's three-plus-two states (see
                    # `ProviderController.view_rows`): copy stays per-surface,
                    # the fact does not.
                    "state": view_row.state,
                    # The stored account's label, when there is EXACTLY one
                    # credential row; null for multi-account and for anything
                    # unknown -- never invented from the dedupe key.
                    "identity": view_row.identity,
                    # Same count as `stored_credentials`, under the name the
                    # account list reads.
                    "account_count": view_row.account_count,
                    "auth_methods": methods,
                    "storage_id": storage_id,
                    "search_aliases": list(provider.search_aliases),
                    "login_kind": provider.login_kind,
                    "accepts_api_key": storage.accepts_api_key,
                    "local": provider.allows_missing_api_key,
                    "credential_name": env_key_name(storage_id),
                    "paste_required": provider.paste_prompt_required,
                    "paste_supported": provider.accepts_paste_prompt,
                    # Configured is not verified: this census never refreshes a
                    # grant or contacts a provider just because Settings opened.
                    "configured": view_row.usable,
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
                    "has_credential": view_row.has_credential,
                    "stored_credentials": view_row.stored_credentials,
                    "base_url": provider.base_url,
                    # The model a first sign-in here will set as the default (API
                    # key spelling; each method carries its own), or null for a
                    # provider with no suggestion (local runtimes, TypeSafe).
                    "suggested_model": _suggestion(storage_id),
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
        #
        # The label comes from the ONE derivation (``credential_identity``),
        # shared with the CLI's listings and the provider view; the fallback is
        # this route's own copy for a payload carrying no IdP identity at all.
        accounts.append(
            {
                "id": row.id,
                "provider": row.provider,
                "type": row.credential_type,
                "identity_label": credential_identity(row) or "Stored credential",
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
    # The tunnel's login verdict rides here because this is the route the
    # account section already reads, and it is the only thing that makes that
    # section honest about a credential that is no longer accepted: a row can be
    # `configured` with an unexpired token and still be unusable, which is what
    # the incident's 401s with a live access token were. The rest of the tunnel's
    # state is on `GET /v1/desktop/tunnel`; this is deliberately the narrow fact.
    #
    # `reachable=False`: this route is polled beside an interactive login form,
    # so it must not wait on a loopback socket to answer.
    #
    # Imported per request, not at module scope: ``tunnels.report`` pulls the
    # tunnel gateway (PyJWT, httpx, starlette) and this router is imported by
    # ``server.app``, so every ``lop serve`` boot paid for it (backend load
    # report B-F10).
    from local_operator.tunnels import report

    tunnel = await report.local_payload(reachable=False)
    return _reply(
        {"accounts": accounts, "radient_login": tunnel["login"], "tunnel_remedy": tunnel["remedy"]}
    )


@router.delete("/v1/auth/accounts/{account_id}", response_model=CRUDResponse[AccountRemoval])
async def remove_account(account_id: int, host: DesktopAuth = Depends(get_desktop_auth)):
    row = host.store.get_credential(account_id)
    if row is None or get_provider_definition(row.provider) is None:
        raise HTTPException(404, w.account_not_found().envelope())
    host.store.delete_credential(account_id)
    if row.provider == "radient":
        # The same gate close as `lop logout radient` (agent review R2): the org
        # Agent Hub must not stay advertised for an account that was removed.
        host.store.org_hub_signed_out()
    from local_operator.providers.auth_cli import _invalidate_cached_listing

    _invalidate_cached_listing(row.provider)
    return _reply(
        {"id": account_id}, "Stored account removed. Environment credentials are unchanged."
    )


@router.post("/v1/auth/login", response_model=CRUDResponse)
async def login(body: LoginRequest, host: DesktopAuth = Depends(get_desktop_auth)):
    """Start a sign-in and reply once it has something the renderer can act on.

    Starting a sign-in SUPERSEDES any active one (see ``DesktopAuth.start``), so
    this no longer answers 409. The reply waits up to
    ``LOGIN_READY_TIMEOUT_S`` for the flow's URL or prompt, because the renderer
    opens the browser from THIS reply: replying before the flow ran a step --
    as this did -- always sent ``auth_url: null`` and the browser never opened
    until the user pressed "reopen". A flow slower than the bound still works;
    the renderer's poll picks the URL up.
    """
    definition = get_provider_definition(body.provider)
    if definition is not None and definition.login_kind == "aws":
        # Amazon Bedrock's connect needs structured inputs (method, Region,
        # profile, secrets) this single-prompt operation cannot carry; the
        # renderer uses the AWS panel routes below (design §7 route 2).
        raise _bedrock_refusal(422, "bedrock_use_aws_panel", _bedrock_keys().use_aws_panel())
    try:
        op = await host.start(body.provider)
    except SignInUnavailableError as error:
        # 503, not 422: the request was valid and the server is shutting down.
        # The renderer classifies 503 as "backend not answering"
        # (``isServerUnreachable``), which is exactly this state; nothing in the
        # desktop client branches on a 422 from this route.
        raise HTTPException(503, w.auth_refusal(error=str(error)).envelope()) from None
    except ValueError as error:
        raise HTTPException(422, w.auth_refusal(error=str(error)).envelope()) from None
    try:
        await asyncio.wait_for(op.ready.wait(), timeout=LOGIN_READY_TIMEOUT_S)
    except TimeoutError:
        pass
    return _reply(op.snapshot(), "Sign-in started.")


@router.get("/v1/auth/operations/{operation_id}", response_model=CRUDResponse)
async def operation(operation_id: str, host: DesktopAuth = Depends(get_desktop_auth)):
    return _reply(_operation(host, operation_id).snapshot())


@router.post("/v1/auth/operations/{operation_id}/input", response_model=CRUDResponse)
async def operation_input(
    operation_id: str, body: LoginInput, host: DesktopAuth = Depends(get_desktop_auth)
):
    """Answer an operation's open prompt.

    For a login whose prompt reads an API key (``paste_is_api_key``) the answer
    IS the key, and this is the second route that stores one -- so it runs the same
    ``key_check`` as ``PUT /v1/auth/providers/{id}/key`` before the flow sees it
    (QA round 1, Q2). Validating here was chosen over refusing
    ``POST /v1/auth/login`` for these providers: that start-then-paste shape is
    an existing contract (the ``/login`` desktop command advertises it, and older
    renderers may drive it), so closing it would break a client to fix a check.
    A definite rejection answers 422 with the reason and leaves the prompt OPEN,
    so the user re-pastes into the same flow; "could not check" lets the key
    through, exactly as the PUT does.

    Gated on what the PROMPT reads, not on the login flavour: the flavour gate
    (``login_kind == "api_key"``) skipped ``alibaba-token-plan-oauth``, a device
    login whose first prompt reads the ``sk-sp-`` inference key it then stores
    (review round 2, MAJOR 1). An authorization code pasted into Anthropic's
    fallback box is not a key and is never sent to a provider as one.
    """
    op = _operation(host, operation_id)
    pending = op.pending_input
    if pending is None or pending.done() or body.prompt_id != op.prompt_id:
        raise HTTPException(409, w.sign_in_not_waiting().envelope())
    value = _secret(body)
    definition = get_provider_definition(op.provider)
    if definition is not None and definition.paste_is_api_key:
        verdict = await key_check.check_api_key(op.provider, value)
        if verdict.valid is False:
            raise HTTPException(
                422,
                (
                    w.provider_refusal(error=verdict.reason).envelope()
                    if verdict.reason
                    else w.key_rejected().envelope()
                ),
            )
        # The check awaited: the flow may have been cancelled or superseded
        # meanwhile, and a resolved future cannot take a second result.
        if pending.done() or op.pending_input is not pending:
            raise HTTPException(409, w.sign_in_not_waiting().envelope())
    pending.set_result(value)
    return _reply(op.snapshot(), "Response submitted.")


@router.delete("/v1/auth/operations/{operation_id}", response_model=CRUDResponse)
async def cancel_operation(operation_id: str, host: DesktopAuth = Depends(get_desktop_auth)):
    op = _operation(host, operation_id)
    await host.cancel(op)
    return _reply(op.snapshot(), "Sign-in cancelled.")


@router.put("/v1/auth/providers/{provider_id}/key", response_model=CRUDResponse)
async def save_key(
    provider_id: str,
    body: SecretInput,
    request: Request,
    host: DesktopAuth = Depends(get_desktop_auth),
):
    """Check the key with its provider, store it, and apply first-run defaults.

    Validation lives HERE rather than on ``/probe`` so a key is checked on the one
    path that stores it: a separate endpoint the renderer had to remember to call
    afterwards would be a second path, and the reported bug (a fake DeepSeek key
    showing "Signed in") was exactly a save with no check behind it. ``/probe``
    stays the LOCAL-server reachability check it was built as.

    * The provider definitively rejects the key -> 422 with the reason as
      ``detail``; nothing is stored and no defaults are applied.
    * It accepts it -> stored, ``valid: true``.
    * It cannot be checked (timeout, offline, outage) -> stored anyway,
      ``valid: null`` with a reason; an unreachable provider is not evidence
      the key is wrong.

    ``result`` is ``{"valid", "reason", "defaults_applied"}``. The key itself
    never appears in a reply or a log line.

    ``defaults_applied`` gains two ADDITIVE fields when the sign-in moved or
    deferred live sessions — ``rehomed_sessions`` and ``deferred_sessions``
    (see ``utils/desktop_rehome``); both are absent when neither happened, so an
    old renderer sees exactly the old shape.
    """
    storage_id = credential_provider_id(provider_id)
    definition = get_provider_definition(storage_id)
    if definition is None or not definition.accepts_api_key:
        raise HTTPException(422, w.key_not_accepted().envelope())
    secret = _secret(body)
    verdict = await key_check.check_api_key(storage_id, secret)
    if verdict.valid is False:
        raise HTTPException(
            422,
            (
                w.provider_refusal(error=verdict.reason).envelope()
                if verdict.reason
                else w.key_rejected().envelope()
            ),
        )
    # AuthStore owns alias translation and source precedence. Use the same
    # login tier as the terminal; do not create a renderer-owned secret store.
    host.store.upsert_credential(storage_id, {"type": "api_key", "source": "login", "key": secret})
    from local_operator.providers.auth_cli import _invalidate_cached_listing

    _invalidate_cached_listing(storage_id)
    # `oauth=False`: this stored a KEY, even under a provider whose own login is
    # OAuth, and the key's host decides the suggested model's spelling.
    applied = await asyncio.to_thread(
        apply_desktop_login_defaults, host.config_manager, storage_id, oauth=False
    )
    # A saved key strands and un-strands providers exactly as a sign-in does, so
    # the same repair runs here (see utils/desktop_rehome): without it, the user
    # who fixes their setup by pasting a key would keep every open session on the
    # provider the key replaced — the reported bug's second half.
    from local_operator.server.utils.desktop_rehome import (
        rehome_after_login,
        with_rehome_count,
    )

    moved: list[tuple[str, str]] = []
    deferred: list[str] = []
    try:
        # ``storage_id``, not the route's ``provider_id``: the first-login rule
        # compares credential STORAGE ids (``openai-device`` is ``openai``), the
        # same translation the planner and the credential write already use.
        moved, deferred = await rehome_after_login(request.app, storage_id)
    except Exception:  # noqa: BLE001 — the key is stored; the repair is best effort
        logger.warning(
            "could not re-home sessions after saving a key for %s", storage_id, exc_info=True
        )
    applied = with_rehome_count(applied, moved, deferred)
    return _reply(
        {"valid": verdict.valid, "reason": verdict.reason, "defaults_applied": applied},
        "API key saved.",
    )


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
        raise HTTPException(422, w.local_server_only().envelope())
    base_url = definition.base_url
    if not base_url:
        raise HTTPException(422, w.server_address_missing().envelope())
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
        raise HTTPException(422, w.credentials_missing().envelope())
    finally:
        controller.close()
    return _reply({}, "Stored credentials removed. Environment credentials are unchanged.")


# -- Amazon Bedrock (design §7, ``features.bedrock_auth``) ---------------------
#
# Bedrock's connection is a STRUCTURED choice (method, Region, profile, maybe
# two secrets) that the single-prompt sign-in operation cannot carry, so it
# gets four routes of its own on this router (same bearer, same envelope).
# The flow logic is ``providers/bedrock/login.py``, shared with ``lop login
# bedrock`` and the TUI, so every surface validates and stores identically.


def _bedrock_refusal(status: int, code: str, message: Any) -> HTTPException:
    """A Bedrock refusal: the catalogued sentence under the contract's code.

    The wire ``code`` is the machine category the desktop panel switches on
    (``bedrock_sso_expired`` ...), which is not the message id, so it rides
    ``envelope(code=...)`` exactly as the runtime-busy codes do.
    """
    return HTTPException(status, message.envelope(code=code))


def _bedrock_keys() -> Any:
    from local_operator.i18n.keys import wire_errors_bedrock

    return wire_errors_bedrock


class BedrockConnectionBody(BaseModel):
    """``PUT /v1/auth/providers/bedrock/connection`` (contract delta 1: this exact body).

    One flat model with ``extra="forbid"`` rather than a pydantic discriminated
    union, so a value outside the contract's bounds answers OUR 422 envelope
    with a ``bedrock_*`` code the renderer can show, not FastAPI's generic
    validation list. Which fields each ``method`` may carry is enforced in
    :func:`_bedrock_input`. Secrets are ``SecretStr`` so they never appear in a
    repr, a validation error or a log line.
    """

    model_config = ConfigDict(extra="forbid")
    method: str
    region: str | None = None
    profile: str | None = None
    api_key: SecretStr | None = None
    access_key_id: str | None = None
    secret_access_key: SecretStr | None = None
    session_token: SecretStr | None = None


#: The fields each method may carry besides ``method`` (contract shapes).
_BEDROCK_FIELDS: dict[str, frozenset[str]] = {
    "api_key": frozenset({"region", "api_key"}),
    "profile": frozenset({"region", "profile"}),
    "access_key": frozenset({"region", "access_key_id", "secret_access_key", "session_token"}),
}


def _bedrock_input(body: BedrockConnectionBody) -> Any:
    from local_operator.providers.bedrock import login as blogin

    def secret(value: SecretStr | None) -> str | None:
        return value.get_secret_value() if value is not None else None

    allowed = _BEDROCK_FIELDS.get(body.method, frozenset({"region"}))
    extra = (body.model_fields_set - {"method"}) - allowed
    if body.method not in blogin.INPUT_METHODS or extra:
        field_name = "method" if body.method not in blogin.INPUT_METHODS else sorted(extra)[0]
        raise blogin.InputInvalid("bedrock_input_invalid", field_name)
    inp = blogin.ConnectionInput(
        method=body.method,
        region=body.region,
        profile=body.profile,
        api_key=secret(body.api_key),
        access_key_id=body.access_key_id,
        secret_access_key=secret(body.secret_access_key),
        session_token=secret(body.session_token),
    )
    blogin.validate_input(inp)
    return inp


def _invalid_input(error: Any) -> HTTPException:
    keys = _bedrock_keys()
    if error.code == "bedrock_region_invalid":
        return _bedrock_refusal(422, error.code, keys.region_invalid(region=error.value or ""))
    return _bedrock_refusal(422, "bedrock_input_invalid", keys.input_invalid(field=error.field))


def _bedrock_connections(store: AuthStore) -> list[dict[str, Any]]:
    """Stored Bedrock rows for the panel: labels and non-secret facts only."""
    from local_operator.providers.bedrock import connection as bconn
    from local_operator.providers.bedrock.login import WIRE_METHOD

    rows: list[dict[str, Any]] = []
    for row in store.list_credentials("bedrock"):
        key = row.data.get("key")
        raw_meta = row.data.get("bedrock")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        try:
            conn = bconn.parse(key) if isinstance(key, str) else None
        except bconn.DescriptorError:
            conn = None
        if conn is None:
            continue
        rows.append(
            {
                "credential_id": row.id,
                "method": WIRE_METHOD.get(conn.method, conn.method),
                "profile": conn.profile,
                "region": conn.region,
                "label": str(meta.get("label") or credential_identity(row) or conn.method),
            }
        )
    return rows


@router.get("/v1/auth/providers/bedrock/aws", response_model=CRUDResponse)
async def bedrock_aws_context(host: DesktopAuth = Depends(get_desktop_auth)):
    """The AWS context for the panel (design §7 route 3). Never a credential value.

    Profiles are listed from config keys and the on-disk SSO caches only —
    no credential is resolved (``credential_process`` could hang). A config
    botocore cannot parse answers 500 ``bedrock_context_failed`` with the
    parser's own text (contract delta 5): "line 14 is broken" is the fix.
    """
    from local_operator.providers.bedrock import profiles as bprofiles

    def build() -> dict[str, Any]:
        found = bprofiles.list_profiles()
        env_profile = None
        import os

        if os.environ.get("AWS_PROFILE"):
            env_profile = next(
                (p.region for p in found if p.name == os.environ["AWS_PROFILE"]), None
            )
        region, source = bprofiles.default_region(profile_region_value=env_profile)
        enabled, sources = bprofiles.ambient_sources()
        return {
            "regions": bprofiles.regions(),
            "default_region": region,
            "default_region_source": source,
            "profiles": [profile.to_json() for profile in found],
            "aws_cli_available": bprofiles.aws_cli_path() is not None,
            # Built from the descriptor vocabulary so the key and the method
            # name stay one spelling.
            _default_chain_key(): {"enabled": enabled, "sources": sources},
            "connections": _bedrock_connections(host.store),
        }

    try:
        context = await asyncio.to_thread(build)
    except Exception as error:  # noqa: BLE001 - surfaced with its own text, never a credential
        detail = str(error).strip()[:500] or type(error).__name__
        raise _bedrock_refusal(
            500, "bedrock_context_failed", _bedrock_keys().context_failed(detail=detail)
        ) from None
    return _reply(context, "AWS context retrieved.")


def _default_chain_key() -> str:
    from local_operator.providers.bedrock.login import DEFAULT_CHAIN

    return DEFAULT_CHAIN


@router.put("/v1/auth/providers/bedrock/connection", response_model=CRUDResponse)
async def bedrock_connect(
    body: BedrockConnectionBody,
    request: Request,
    host: DesktopAuth = Depends(get_desktop_auth),
):
    """Validate, then store one Bedrock connection (design §7 route 4).

    Same order as ``save_key``: check first, store only a connection that was
    not definitively rejected. A rejection is 422 with its ``bedrock_*`` code;
    an unverified check (``valid: null``) stores and says why in ``reason``,
    with no ``code`` (contract delta 6). Defaults and the session re-home run
    exactly as after a saved key.
    """
    from local_operator.providers.bedrock import login as blogin

    try:
        inp = _bedrock_input(body)
    except blogin.InputInvalid as error:
        raise _invalid_input(error) from None
    region = await asyncio.to_thread(blogin.effective_region, inp)
    verdict = await blogin.check_input(inp, region)
    if verdict.valid is False:
        raise _bedrock_refusal(
            422,
            verdict.code or "bedrock_credentials_rejected",
            w.provider_refusal(error=verdict.reason or ""),
        )
    row = blogin.build_row(inp, region)
    stored = host.store.upsert_credential("bedrock", row)
    from local_operator.providers.auth_cli import _invalidate_cached_listing

    _invalidate_cached_listing("bedrock")
    applied = await asyncio.to_thread(
        apply_desktop_login_defaults, host.config_manager, "bedrock", oauth=False
    )
    from local_operator.server.utils.desktop_rehome import (
        rehome_after_login,
        with_rehome_count,
    )

    moved: list[tuple[str, str]] = []
    deferred: list[str] = []
    try:
        moved, deferred = await rehome_after_login(request.app, "bedrock")
    except Exception:  # noqa: BLE001 — the connection is stored; the repair is best effort
        logger.warning("could not re-home sessions after connecting Bedrock", exc_info=True)
    applied = with_rehome_count(applied, moved, deferred)
    raw_meta = stored.data.get("bedrock")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    return _reply(
        {
            "valid": verdict.valid,
            "reason": verdict.reason,
            "credential_id": stored.id,
            "label": str(meta.get("label") or blogin.label_for(inp, region)),
            # The built-in catalogue is what the picker shows until discovery
            # runs; a live count would cost a second signed call here.
            "models_available": None,
            "defaults_applied": applied,
        },
        "Amazon Bedrock connected.",
    )


class BedrockSsoBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: str


@router.post("/v1/auth/providers/bedrock/sso-login", response_model=CRUDResponse)
async def bedrock_sso_login(body: BedrockSsoBody, host: DesktopAuth = Depends(get_desktop_auth)):
    """Start ``aws sso login`` / ``aws login`` for a profile (design §7 route 5).

    Explicit user action only (a button press). Returns an ordinary
    ``LoginOperation`` snapshot, polled and cancelled through the existing
    operation routes; ``auth_url``/``user_code`` carry the device URL and
    code (contract delta 3). 422 ``aws_cli_missing`` without the CLI, 422
    ``bedrock_profile_not_found`` / ``bedrock_sso_not_applicable`` for a
    profile that cannot sign in, 409 while one runs for the same profile.
    """
    from local_operator.providers.bedrock import profiles as bprofiles
    from local_operator.providers.bedrock import sso_login

    keys = _bedrock_keys()
    if not bprofiles.valid_profile_input(body.profile):
        raise _bedrock_refusal(422, "bedrock_input_invalid", keys.input_invalid(field="profile"))
    try:
        found = await asyncio.to_thread(bprofiles.list_profiles)
    except Exception as error:  # noqa: BLE001
        detail = str(error).strip()[:500] or type(error).__name__
        raise _bedrock_refusal(
            500, "bedrock_context_failed", keys.context_failed(detail=detail)
        ) from None
    profile = next((p for p in found if p.name == body.profile), None)
    if profile is None:
        raise _bedrock_refusal(
            422, "bedrock_profile_not_found", keys.profile_not_found(profile=body.profile)
        )
    command = bprofiles.login_command(profile.name, profile.kind) or ""
    refusal = sso_login.refusal(profile.kind)
    if refusal == "not_sso":
        raise _bedrock_refusal(
            422, "bedrock_sso_not_applicable", keys.sso_not_applicable(profile=profile.name)
        )
    if refusal == "aws_cli_missing":
        raise _bedrock_refusal(422, "aws_cli_missing", keys.aws_cli_missing())
    if refusal == "headless":
        raise _bedrock_refusal(422, "bedrock_sso_headless", keys.sso_headless(command=command))
    if host.active_bedrock_sso(profile.name) is not None:
        raise _bedrock_refusal(409, "bedrock_sso_busy", keys.sso_busy(profile=profile.name))
    try:
        op = await host.start_bedrock_sso(profile.name, profile.kind)
    except SignInUnavailableError as error:
        raise HTTPException(503, w.auth_refusal(error=str(error)).envelope()) from None
    try:
        await asyncio.wait_for(op.ready.wait(), timeout=LOGIN_READY_TIMEOUT_S)
    except TimeoutError:
        pass
    return _reply(op.snapshot(), "AWS sign-in started.")


class BedrockCheckBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credential_id: int


@router.post("/v1/auth/providers/bedrock/check", response_model=CRUDResponse)
async def bedrock_check(body: BedrockCheckBody, host: DesktopAuth = Depends(get_desktop_auth)):
    """Re-validate one stored connection (design §7 route 6): ``{valid, reason, code}``."""
    from local_operator.providers.bedrock import check as bcheck

    row = next(
        (r for r in host.store.list_credentials("bedrock") if r.id == body.credential_id), None
    )
    key = row.data.get("key") if row is not None else None
    if not isinstance(key, str):
        raise _bedrock_refusal(
            404, "bedrock_connection_missing", _bedrock_keys().connection_missing()
        )
    verdict = await bcheck.check_connection(key, 15.0)
    return _reply(
        {
            "valid": verdict.valid,
            "reason": verdict.reason,
            "code": verdict.code if verdict.valid is False else None,
        },
        "Amazon Bedrock connection checked.",
    )
