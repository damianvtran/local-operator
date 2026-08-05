"""ProviderController — the TUI's window into providers, credentials, models
and usage.

One narrow facade over the provider, credential, model and usage layers: it
is exactly what the slash-command UX needs and nothing the TUI must not
reach. The app gets ONE instance injected by ``cli.py``; it owns
no terminal I/O (login is invoked with caller-provided callbacks, the same
interactive-login control flow) and no session state.

Read/credential surfaces are sync and I/O-free. Anything that touches the
network (usage fetches, OAuth refresh during login) is async and belongs in
a ``run_worker`` so the Textual message thread never blocks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Protocol

import httpx

from local_operator.harness.types import ModelSpec
from local_operator.model.configure import build_model_spec  # noqa: F401  (used by callers)
from local_operator.providers.oauth.callback_server import LoginCallbacks
from local_operator.providers.registry import (
    ProviderDefinition,
    get_provider_definition,
    list_login_providers,
)
from local_operator.providers.usage import (
    USAGE_PROVIDERS,
    UsageReport,
    fetch_usage,
    usage_supported,
)

if TYPE_CHECKING:  # auth_store stays off this module's runtime import graph
    from local_operator.credentials import CredentialManager
    from local_operator.providers.auth_store import OAuthAccess, StoredCredential

LoginCallbackFactory = Callable[[ProviderDefinition], LoginCallbacks]


class ControllerAuthStore(Protocol):
    """The credential-store slice this facade uses.

    Structural so a host can supply its own store; every member is called
    exactly as declared here (no extra keywords), which keeps the contract
    small enough for lightweight stand-ins.
    """

    def list_credentials(
        self, provider: str | None = None
    ) -> list["StoredCredential"]: ...  # pragma: no cover

    def upsert_credential(
        self, provider: str, credential: dict[str, Any]
    ) -> "StoredCredential": ...  # pragma: no cover

    def delete_credentials_for_provider(
        self, provider: str, disabled_cause: str = ...
    ) -> int: ...  # pragma: no cover

    async def get_oauth_access(self, provider: str) -> "OAuthAccess | None": ...  # pragma: no cover

    async def get_api_key(self, provider: str) -> str | None: ...  # pragma: no cover


class ProviderController:
    """Provider/model/credential/usage facade for interactive front ends."""

    def __init__(
        self,
        auth_store: ControllerAuthStore,
        credential_manager: "CredentialManager | None" = None,
        *,
        login_callbacks: LoginCallbackFactory | None = None,
    ) -> None:
        self.auth_store = auth_store
        self.credential_manager = credential_manager
        # Terminal-bound login callbacks. The CLI's print/input callbacks are
        # used by default; an embedding host (e.g. a Textual app) injects
        # callbacks that yield the terminal before the flow runs.
        self._login_callbacks = login_callbacks

    # -- discovery ---------------------------------------------------------
    def login_providers(self) -> list[ProviderDefinition]:
        """Providers offering an interactive login, registry order."""
        return list_login_providers()

    def provider(self, provider_id: str) -> ProviderDefinition | None:
        return get_provider_definition(provider_id)

    def credentials(self) -> list["StoredCredential"]:
        """Every active stored credential (StoredCredential rows)."""
        return self.auth_store.list_credentials()

    def has_any_credential(self, provider: str) -> bool:
        """Whether ``provider`` (or its storage id) has a stored credential."""
        definition = get_provider_definition(provider)
        storage = (definition.store_credentials_as or provider) if definition else provider
        return any(c.provider == storage for c in self.auth_store.list_credentials(provider=None))

    def usage_enabled_providers(self) -> list[str]:
        """Provider ids with a live quota endpoint, sorted."""
        return sorted(USAGE_PROVIDERS)

    # -- model -------------------------------------------------------------
    def resolve_model(self, provider: str, model_id: str) -> ModelSpec:
        """Build a ModelSpec for a provider/model pair (raises on unknown
        provider/hosting). Used by ``/model <provider>/<id>``."""
        return build_model_spec(provider, model_id)

    # -- login / logout ----------------------------------------------------
    def _default_callbacks(self, definition: ProviderDefinition) -> LoginCallbacks:
        """Fall back to the CLI's terminal print/input callbacks."""
        from local_operator.providers.auth_cli import _callbacks_interactive

        return _callbacks_interactive(definition)

    async def login(self, provider_id: str) -> str:
        """Run the provider's login flow and report a human summary.

        Must be called with the terminal yielded to the flow (a Textual app
        wraps this in ``App.suspend()``). Returns a message like
        ``Logged in to 'anthropic' (you@example.com).``; raises
        ``ValueError`` / ``LoginError`` on failure.
        """
        definition = get_provider_definition(provider_id)
        if definition is None:
            raise ValueError(f"Unknown provider: {provider_id}")
        if definition.login is None:
            raise ValueError(f"Provider '{provider_id}' has no interactive login.")

        factory = self._login_callbacks or self._default_callbacks
        callbacks = factory(definition)
        result = await definition.login(callbacks)

        storage = definition.store_credentials_as or provider_id
        if isinstance(result, str):
            if result:
                self.auth_store.upsert_credential(
                    storage, {"key": result, "source": "login", "type": "api_key"}
                )
                return f"Stored API key for '{storage}'."
            return f"Login for '{storage}' produced no key; nothing stored."

        result.setdefault("authorized_at", int(time.time() * 1000))
        self.auth_store.upsert_credential(storage, result)
        identity = result.get("email") or result.get("account_id") or result.get("org_name") or ""
        suffix = f" ({identity})" if identity else ""
        msg = f"Logged in to '{storage}'{suffix}."
        if result.get("grant_note"):
            msg += f" Note: {result['grant_note']}"
        return msg

    async def logout(self, provider_id: str) -> str:
        """Remove stored credentials (alias + storage id) for a provider."""
        definition = get_provider_definition(provider_id)
        if definition is None:
            raise ValueError(f"Unknown provider: {provider_id}")
        targets = {provider_id, definition.store_credentials_as or provider_id}
        removed = 0
        for target in sorted(targets):
            removed += self.auth_store.delete_credentials_for_provider(
                target, disabled_cause="logged-out"
            )
        if removed == 0:
            raise ValueError(f"No stored credentials for '{provider_id}'.")
        return f"Removed {removed} credential(s) for '{provider_id}'."

    # -- usage -------------------------------------------------------------
    async def fetch_usage(self, provider_ids: list[str] | None = None) -> list[UsageReport]:
        """Fetch normalized usage reports for the requested (or all
        report-able) providers. Never raises: a provider with no reachable
        credential or endpoint is simply absent from the result, and one
        malformed provider never aborts the others."""
        targets = provider_ids or []
        if not targets:
            targets = [p for p in self.usage_enabled_providers() if self.has_any_credential(p)]
        # De-duplicate aliases that share a storage id (openai vs
        # openai-device; xai vs xai-oauth) so one request/one report per row.
        targets = self._dedupe_targets(targets)
        reports: list[UsageReport] = []
        async with httpx.AsyncClient() as client:
            for provider in targets:
                try:
                    report = await self._fetch_one(client, provider)
                except Exception:  # noqa: BLE001 — isolate one broken provider
                    report = None
                if report is not None:
                    reports.append(report)
        return reports

    def _dedupe_targets(self, targets: list[str]) -> list[str]:
        """Keep one id per storage row so alias providers don't double-fetch."""
        seen: set[str] = set()
        ordered: list[str] = []
        for provider in targets:
            definition = get_provider_definition(provider)
            storage = (definition.store_credentials_as or provider) if definition else provider
            if storage in seen:
                continue
            seen.add(storage)
            ordered.append(provider)
        return ordered

    async def _fetch_one(self, client: httpx.AsyncClient, provider: str) -> UsageReport | None:
        if not usage_supported(provider):
            return None
        access = await self.auth_store.get_oauth_access(provider)
        if access is not None and access.kind == "oauth":
            # An OAuth subscription token — the endpoint wants the access
            # token, not a billing API key.
            return await fetch_usage(
                client,
                provider,
                access_token=access.access_token,
                account_id=access.account_id,
            )
        api_key: str | None = None
        # Fall back to the API-key cascade (covers both a stored/pasted key
        # and the environment tier resolved for api-key providers).
        if access is not None and access.access_token:
            api_key = access.access_token
        else:
            try:
                api_key = await self.auth_store.get_api_key(provider)
            except Exception:  # noqa: BLE001 — a refresh failure is not fatal here
                api_key = None
        if not api_key:
            return None
        return await fetch_usage(client, provider, api_key=api_key)
