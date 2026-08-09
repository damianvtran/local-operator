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

import asyncio
import dataclasses
import time
from typing import TYPE_CHECKING, Any, Callable, Protocol

import httpx

from local_operator.harness.types import ModelSpec
from local_operator.model.configure import (  # noqa: F401  (used by callers)
    build_model_spec,
)
from local_operator.model.discovery import available_models
from local_operator.model.registry import static_models
from local_operator.providers.registry import (
    AGGREGATOR_PROVIDERS,
    PROVIDER_REGISTRY,
    ProviderDefinition,
    get_provider_definition,
    list_login_providers,
    resolve_env_key,
)
from local_operator.providers.usage import (
    USAGE_PROVIDERS,
    UsageReport,
    fetch_usage,
    usage_kinds,
    usage_supported,
)

if TYPE_CHECKING:  # auth_store stays off this module's runtime import graph
    from local_operator.credentials import CredentialManager
    from local_operator.providers.auth_store import OAuthAccess, StoredCredential
    from local_operator.providers.oauth.callback_server import LoginCallbacks

LoginCallbackFactory = Callable[[ProviderDefinition], "LoginCallbacks"]


@dataclasses.dataclass(frozen=True)
class CatalogueEntry:
    """One offerable model, provider-qualified, with the provider's auth state.

    Deliberately a flat record of PRESENTABLE values rather than a `ModelInfo`:
    the TUI's picker needs a display label and two prices and must not have to
    know that a context window of `-1` means unknown while `0` also does. The
    normalization happens once, here, where the registry's conventions are known.
    """

    provider: str
    model_id: str
    label: str
    context_window: int
    input_price: float
    output_price: float
    connected: bool
    #: This provider RESELLS the model rather than serving it. The picker ranks
    #: the direct route first when the same model is reachable both ways.
    aggregated: bool = False

    @property
    def selector(self) -> str:
        """``provider/id`` — what ``/model`` accepts."""
        return f"{self.provider}/{self.model_id}"


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

    def set_login_callbacks(self, factory: "LoginCallbackFactory | None") -> None:
        """Install host-specific login callbacks after construction.

        The CLI builds this controller before it knows whether a TUI will run,
        and a TUI's callbacks are fundamentally different in kind rather than in
        wording: it renders the authorization URL into its own transcript and
        must NEVER read from the real stdin, because doing so either fights the
        app for the terminal or requires suspending it.
        """
        self._login_callbacks = factory

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

    def is_usable(self, provider: str) -> bool:
        """Whether ``provider`` has a credential the session could actually run on.

        Wider than :meth:`has_any_credential`, which only sees the AuthStore. A key
        in the ENVIRONMENT is a working credential by every measure that matters —
        it is the one the stream-time cascade resolves, so a session started that
        way runs fine — and reporting such a provider as "needs a login" was both
        wrong and unactionable, since there is no login to perform.

        ``allows_missing_api_key`` providers (a local Ollama) are usable with no
        credential at all, which is the whole point of running one.
        """
        definition = get_provider_definition(provider)
        if definition is not None and definition.allows_missing_api_key:
            return True
        if self.has_any_credential(provider):
            return True
        return bool(resolve_env_key(provider))

    def usable_providers(self) -> set[str] | None:
        """Every provider id a turn could run on now — or ``None`` when unknowable.

        ``None`` is not "none of them": it means the credential store could not be
        read at all, which is a different answer and the only one that is honest
        when SQLite is locked. A caller that filters on this set has to show
        EVERYTHING in that case, because an empty model list reads as "you own no
        models" — a claim the app cannot make when it read zero rows.

        One store scan for the whole registry, where :meth:`is_usable` costs one
        per provider: the catalogue asks this question about a dozen providers on
        the keystroke that opens the picker.
        """
        try:
            stored = {row.provider for row in self.auth_store.list_credentials(provider=None)}
        except Exception:  # noqa: BLE001 — an unreadable store is reported, not raised
            return None
        usable: set[str] = set()
        for definition in PROVIDER_REGISTRY:
            storage = definition.store_credentials_as or definition.id
            if (
                definition.allows_missing_api_key  # a local Ollama needs no credential
                or storage in stored
                or resolve_env_key(definition.id)
            ):
                usable.add(definition.id)
        return usable

    def usage_enabled_providers(self) -> list[str]:
        """Provider ids with a live quota endpoint, sorted.

        "Has an endpoint", not "can reach it" — see :meth:`can_report_usage` for
        the question every UI surface actually asks.
        """
        return sorted(USAGE_PROVIDERS)

    def can_report_usage(self, provider: str) -> bool:
        """Whether the credentials ON HAND can reach ``provider``'s quota endpoint.

        ONE predicate, because there are three surfaces asking it — `/provider`'s
        "reports quota" list, bare `/usage`'s target list and `/usage <provider>`'s
        up-front warning — and any two of them disagreeing is the defect this
        replaces: with only ``ANTHROPIC_API_KEY`` set, `/provider` advertised
        anthropic, bare `/usage` rendered "no usage data", and `/usage anthropic`
        correctly said it needs a login.

        :meth:`is_usable` alone is too coarse here. It answers "is there any
        credential", including one resolved from the environment, but five of the
        eight usage providers are OAuth-only for usage — an API key cannot reach
        their endpoint at all, so an ``ANTHROPIC_API_KEY`` user holds a credential
        that runs the model and cannot read the quota. The API-key route has to
        exist (``usage_kinds(p)[1]``) before an env/API key counts.
        """
        if not self.is_usable(provider):
            return False
        return self.has_any_credential(provider) or usage_kinds(provider)[1]

    def usage_reportable_providers(self) -> list[str]:
        """Sorted provider ids that both have an endpoint and a way to reach it."""
        return [p for p in self.usage_enabled_providers() if self.can_report_usage(p)]

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
            # The SAME predicate `/provider` and `/usage <provider>` use. An env key
            # counts (it is the tier the stream cascade resolves), but only where an
            # API-key route to the quota endpoint exists — see
            # :meth:`can_report_usage` for why each half is needed.
            targets = self.usage_reportable_providers()
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

    # -- catalogue ---------------------------------------------------------
    def static_catalogue(self) -> list[CatalogueEntry]:
        """Every model the SHIPPED registry knows, with each provider's auth state.

        Synchronous and I/O-free, which is the point: a picker has to paint on the
        keystroke that opened it. This is the first frame; :meth:`live_catalogue`
        replaces it when the network answers.
        """
        entries: list[CatalogueEntry] = []
        # ONE store read for the whole registry, and one that survives a store
        # that cannot be read: `is_usable` per provider would raise straight
        # through a picker that is about to paint. An unknowable state is
        # reported as connected — see :meth:`usable_providers` for why the
        # degradation runs that way and not toward an empty list.
        usable = self.usable_providers()
        for definition in PROVIDER_REGISTRY:
            connected = usable is None or definition.id in usable
            for model_id, info in static_models(definition.id).items():
                entries.append(
                    CatalogueEntry(
                        provider=definition.id,
                        model_id=model_id,
                        label=info.name or model_id,
                        context_window=max(0, info.context_window or 0),
                        input_price=_price(info.input_price, definition),
                        output_price=_price(info.output_price, definition),
                        connected=connected,
                        aggregated=definition.id in AGGREGATOR_PROVIDERS,
                    )
                )
        return entries

    async def live_catalogue(
        self, *, ttl_s: float | None = None
    ) -> tuple[list[CatalogueEntry], dict[str, str]]:
        """The catalogue with each provider's LIVE listing layered over the registry.

        Returns ``(entries, statuses)`` where ``statuses`` maps provider id to one
        of discovery's status strings, so a UI can say "cached" or "login
        required" instead of implying the catalogue is complete.

        Only providers with a credential are fetched. An unconnected provider still
        contributes its STATIC models — the question "what would I get if I logged
        in here" is precisely what a user cannot otherwise answer, and it was the
        reason a newly released model was undiscoverable.

        Each provider is isolated: discovery never raises by contract, but a
        credential resolution can (an OAuth refresh against a dead network), and
        one broken provider must not empty the whole list.
        """
        entries: list[CatalogueEntry] = []
        statuses: dict[str, str] = {}
        usable = self.usable_providers()
        for definition in PROVIDER_REGISTRY:
            connected = usable is None or definition.id in usable
            api_key: str | None = None
            is_oauth = False
            account_id: str | None = None
            if connected:
                try:
                    api_key, is_oauth, account_id = await self._listing_credential(definition.id)
                except Exception:  # noqa: BLE001 — one provider's auth is not fatal
                    api_key, is_oauth, account_id = None, False, None
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "is_oauth": is_oauth,
                "account_id": account_id,
            }
            if ttl_s is not None:
                kwargs["ttl_s"] = ttl_s
            # Off the event loop: discovery is synchronous httpx by design (it is
            # also called from the CLI and the server), and a dozen sequential
            # provider fetches on the loop would freeze a TUI's repaint.
            models, status = await asyncio.to_thread(available_models, definition.id, **kwargs)
            statuses[definition.id] = status
            for model in models:
                entries.append(
                    CatalogueEntry(
                        provider=definition.id,
                        model_id=model.id,
                        label=model.name or model.id,
                        context_window=max(0, model.context_window),
                        input_price=_price(model.input_price, definition),
                        output_price=_price(model.output_price, definition),
                        connected=connected,
                        aggregated=definition.id in AGGREGATOR_PROVIDERS,
                    )
                )
        return entries, statuses

    async def _listing_credential(self, provider: str) -> tuple[str | None, bool, str | None]:
        """``(secret, is_oauth, account_id)`` for a model listing call.

        The account id is part of OpenAI's ChatGPT authorization boundary; the
        current Codex catalogue rejects a subscription token without it. Other
        providers ignore the value, while ``is_oauth`` still selects Anthropic's
        bearer header instead of its API-key header.
        """
        access = await self.auth_store.get_oauth_access(provider)
        if access is not None and access.kind == "oauth" and access.access_token:
            return access.access_token, True, access.account_id or access.org_id
        if access is not None and access.access_token:
            return access.access_token, False, None
        try:
            stored = await self.auth_store.get_api_key(provider)
        except Exception:  # noqa: BLE001 — a refresh failure just means no listing
            stored = None
        # The environment is the last tier of the same cascade the stream uses, so
        # a key set there has to reach the listing too: otherwise the provider a
        # session is ACTUALLY RUNNING ON is the one whose catalogue stays empty.
        return stored or resolve_env_key(provider), False, None

    async def _fetch_one(self, client: httpx.AsyncClient, provider: str) -> UsageReport | None:
        if not usage_supported(provider):
            return None
        access = await self.auth_store.get_oauth_access(provider)
        access_token: str | None = None
        api_key: str | None = None
        account_id: str | None = None
        if access is not None and access.kind == "oauth":
            access_token = access.access_token
            account_id = access.account_id
        elif access is not None and access.access_token:
            api_key = access.access_token
        if api_key is None:
            try:
                api_key = await self.auth_store.get_api_key(provider)
            except Exception:  # noqa: BLE001 — a refresh failure is not fatal here
                api_key = None
        if not access_token and not api_key:
            return None
        # BOTH are handed over, and the dispatcher picks the route each can reach.
        # Passing only one was how the API-key half of a dual-route provider became
        # unreachable: an OAuth token for Kimi went to the coding-plan endpoint, but
        # an API key went nowhere at all because this function had already decided
        # the request was an OAuth one.
        report = await fetch_usage(
            client,
            provider,
            api_key=api_key,
            access_token=access_token,
            account_id=account_id,
        )
        if report is not None and not report.identity and access is not None:
            # Whose account this is. The field existed and no fetcher ever set it, so
            # the TUI's annotation for it was unreachable — and it matters most
            # exactly where usage does: an operator with two accounts on one provider
            # needs to know which one the numbers describe.
            report.identity = getattr(access, "email", None) or access.account_id
        return report


def _price(value: float | None, definition: ProviderDefinition) -> float:
    """A per-million price, with UNKNOWN kept distinct from FREE.

    Discovery and the static registry both use ``0`` for "no price known", and the
    picker renders a genuine pair of zeroes as ``free`` — so passing an unknown
    through as zero advertises a paid model as costing nothing. Anthropic makes
    this immediate rather than theoretical: its listing carries no pricing at all,
    so every model it discovers that we did not already ship would read ``free``.

    ``-1`` is the unknown sentinel the picker blanks. Zero is preserved only for
    providers that need no credential — a local Ollama really is free per token,
    and blanking that would hide the one thing that makes it interesting.
    """
    if value is not None and value > 0:
        return float(value)
    return 0.0 if definition.allows_missing_api_key else -1.0
