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
import logging
import random
import time
from typing import TYPE_CHECKING, Any, Callable, Protocol

import httpx

from local_operator.harness.types import ModelSpec
from local_operator.model.configure import (  # noqa: F401  (used by callers)
    build_model_spec,
)
from local_operator.model.discovery import (
    DiscoveredModel,
    available_models,
    cached_available_models,
    invalidate_listing,
    is_meta_route_id,
)
from local_operator.model.naming import model_label
from local_operator.model.registry import static_models
from local_operator.providers.registry import (
    AGGREGATOR_PROVIDERS,
    PROVIDER_REGISTRY,
    ProviderDefinition,
    credential_provider_id,
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
from local_operator.providers.usage_cache import (
    USAGE_ACCOUNT_MAX_FAILURES,
    USAGE_REPORT_TTL_MS,
    UsageCacheStore,
    account_backoff_ms,
    fingerprint_accounts,
    fingerprint_secret,
    provider_cache_key,
    report_identity_key,
)

if TYPE_CHECKING:  # auth_store stays off this module's runtime import graph
    from local_operator.credentials import CredentialManager
    from local_operator.providers.auth_store import OAuthAccess, StoredCredential
    from local_operator.providers.oauth.callback_server import LoginCallbacks

LoginCallbackFactory = Callable[[ProviderDefinition], "LoginCallbacks"]

logger = logging.getLogger("local_operator.providers.controller")

#: How long an empty refresh keeps deferring to old data before it is believed.
#: The empty-over-data heuristic reads a blank answer over non-empty history as
#: an outage — but a provider that GENUINELY went quota-less (plan lapsed,
#: account emptied) would otherwise be re-fetched on every cool-down forever,
#: because each ``write_failure`` keeps the old row alive. Once the last real
#: data is this old, consecutive empty answers are accepted as the truth and
#: negative-cached at full TTL. Half an hour: long enough to ride out any
#: plausible rate-limit window, short enough that a lapsed plan stops burning
#: a request per warm tick the same day.
EMPTY_OVER_DATA_ACCEPT_MS = 30 * 60_000

#: The listing TTL the ``/model`` picker asks :meth:`live_catalogue` for. The
#: user is asking NOW, and the fetch already runs off-loop behind rows painted
#: from the registry, so a short TTL costs nothing visible. Fifteen minutes is
#: short enough that a release announced during a working session shows on the
#: next open, and long enough that scrolling in and out of the picker does not
#: re-list nine providers. Boot and repaint paths keep the default 24h hard TTL
#: (with an hourly background refresh) because there a request IS visible.
PICKER_TTL_S = 15 * 60


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
    #: This entry is a META-ROUTE: its price depends on the model it dispatches
    #: to, so there is no pair of numbers that describes it.
    #:
    #: WHY this travels when ``free`` deliberately does not. ``free`` stops at
    #: the boundary below because the price pair can already carry it: ``0.0``
    #: means stated-free and ``-1.0`` means unknown, so the flag would be a
    #: second spelling of a fact the floats already hold, free to drift from
    #: them. That argument does not extend here, and the reason is that the
    #: float vocabulary is FULL. A router has no price, so it can only arrive
    #: as ``-1.0``/``-1.0`` — which is already spoken for by "nobody quoted
    #: this", a genuinely different answer that must keep rendering as a blank
    #: cell. Encoding a fourth meaning would mean inventing a fourth sentinel
    #: (``-2.0``) and teaching every reader of these two floats about it, which
    #: is the "smuggle a new meaning into an existing channel" move that
    #: ``format_price_pair`` and ``_price`` both warn against at length.
    #:
    #: So it is carried as its own bit, the way ``DiscoveredModel.routed`` is,
    #: and the prices stay honestly unknown underneath it. Nothing but the
    #: display reads it.
    routed: bool = False

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

    async def list_oauth_accesses(
        self, provider: str
    ) -> list["OAuthAccess"]: ...  # pragma: no cover

    def list_oauth_identities(self, provider: str) -> list["OAuthAccess"]: ...  # pragma: no cover

    async def get_api_key(self, provider: str) -> str | None: ...  # pragma: no cover


class ProviderController:
    """Provider/model/credential/usage facade for interactive front ends."""

    def __init__(
        self,
        auth_store: ControllerAuthStore,
        credential_manager: "CredentialManager | None" = None,
        *,
        login_callbacks: LoginCallbackFactory | None = None,
        usage_cache: UsageCacheStore | None = None,
    ) -> None:
        self.auth_store = auth_store
        self.credential_manager = credential_manager
        # Terminal-bound login callbacks. The CLI's print/input callbacks are
        # used by default; an embedding host (e.g. a Textual app) injects
        # callbacks that yield the terminal before the flow runs.
        self._login_callbacks = login_callbacks
        # Shared on-disk usage cache (see providers.usage_cache). Built lazily
        # so a host that never asks for usage pays nothing, and injectable so
        # tests can aim it at a temp file. Deliberately SHARED across every
        # lop session on this machine: several terminals run at once, and one
        # process's refresh is every process's answer.
        self._usage_cache = usage_cache

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
        storage = credential_provider_id(provider)
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
            stored_rows = self.auth_store.list_credentials(provider=None)
            stored = {row.provider for row in stored_rows}
            oauth_providers = {
                row.provider for row in stored_rows if row.credential_type == "oauth"
            }
        except Exception:  # noqa: BLE001 — an unreadable store is reported, not raised
            return None
        usable: set[str] = set()
        for definition in PROVIDER_REGISTRY:
            storage = credential_provider_id(definition.id)
            # If the user has an active OAuth sign-in under the base provider
            # (e.g. `radient` OAuth), suppress secondary legacy API-key flavours
            # (`radient-key`) from being treated as usable alternatives.
            if definition.store_credentials_as and storage in oauth_providers:
                continue
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
                _invalidate_cached_listing(storage)
                return f"Stored API key for '{storage}'."
            return f"Login for '{storage}' produced no key; nothing stored."

        result.setdefault("authorized_at", int(time.time() * 1000))
        self.auth_store.upsert_credential(storage, result)
        # The new credential may list DIFFERENT models than the one it replaced
        # -- a different account, a different plan, or a catalogue listed
        # anonymously before there was a credential at all. No TTL can observe
        # that, so the login event has to say so itself. Same hook as the CLI's
        # ``run_login``; the TUI's ``/login`` arrives here.
        _invalidate_cached_listing(storage)
        # The cached usage row is wrong for the same reason and one more: a
        # completed login is positive evidence this account's grant is alive,
        # which directly contradicts any ``credential_invalid`` verdict the row
        # still carries. Re-authenticating an ALREADY-stored account keeps the
        # account fingerprint (and so the cache key) identical, so nothing else
        # in the system observes this event -- see ``UsageCacheStore.invalidate``.
        self._invalidate_cached_usage(storage)
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
        # Symmetrical with login: a catalogue fetched under the credential just
        # removed must not decide what the NEXT credential can select. One call
        # per STORAGE id: alias and storage id (``zai-oauth``/``zai``) resolve
        # to the same document set, so iterating both would glob twice.
        for storage_id in sorted({credential_provider_id(t) for t in targets}):
            _invalidate_cached_listing(storage_id)
        return f"Removed {removed} credential(s) for '{provider_id}'."

    # -- usage -------------------------------------------------------------
    async def fetch_usage(
        self,
        provider_ids: list[str] | None = None,
        *,
        force_refresh: bool = False,
    ) -> list[UsageReport]:
        """Fetch normalized usage reports for the requested (or all
        report-able) providers. Never raises: a provider with no reachable
        credential or endpoint is simply absent from the result, and one
        malformed provider never aborts the others.

        Two accelerators sit in front of the network (see
        :mod:`local_operator.providers.usage_cache`):

        - A **shared on-disk cache** keyed per provider + account set, so a
          fresh entry answers with no network at all — and because it is shared
          across every lop session on this machine, one session's refresh is
          every session's answer.
        - **Parallel fan-out** across providers. The old loop awaited each
          provider in turn, so the panel waited for the SUM of the round trips;
          now it waits for the slowest one.

        ``force_refresh`` bypasses the fresh-cache check (the panel's ``r``)
        but still writes its result back, so the next read — in this session or
        any other — is instant.
        """
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
        if not targets:
            return []
        reports: list[UsageReport] = []
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *(
                    self._fetch_provider_cached(client, provider, force_refresh)
                    for provider in targets
                ),
                return_exceptions=True,
            )
        for result in results:
            # Isolate one broken provider: an exception here drops that row
            # rather than aborting the whole report (the old per-provider try).
            if isinstance(result, BaseException):
                continue
            reports.extend(result)
        return reports

    # -- usage cache plumbing ------------------------------------------------

    def close(self) -> None:
        """Release the shared usage cache handle (idempotent, never raises)."""
        if self._usage_cache is not None:
            try:
                self._usage_cache.close()
            except Exception:  # noqa: BLE001 — teardown, never fatal
                pass
            self._usage_cache = None

    def usage_cache_age_ms(self, provider: str) -> int | None:
        """Milliseconds since the cached usage row for ``provider`` was fetched.

        ``None`` when there is no cached row for the provider's CURRENT account
        set (never fetched, or the account set changed since). This is the
        question the TUI's background warmer asks before deciding whether to
        spend a refresh: a warm row means `/usage` will answer from disk, so
        the warmer only fires when the row is missing or going stale.

        Synchronous and cheap (one indexed SQLite read), safe to call from an
        interval callback.
        """
        cache = self._usage_cache_store()
        if cache is None:
            return None
        key = self._usage_cache_key(provider)
        fetched_at = cache.fetched_at_ms(key)
        if fetched_at is None or fetched_at <= 0:
            return None
        now_ms = int(time.time() * 1000)
        return max(0, now_ms - fetched_at)

    def cached_usage_reports(self, provider: str | None = None) -> list[UsageReport]:
        """The cached usage reports for ``provider`` (or all providers), any age.

        The panel's instant-open half: when a row exists, `/usage` can paint it
        immediately (its age stated in the title) while the fetch worker runs in
        the background to confirm or replace it. Reads the shared cache only —
        never crosses the network — so it is safe to call synchronously on the
        keystroke that opens the panel.
        """
        cache = self._usage_cache_store()
        if cache is None:
            return []
        targets = (
            [provider] if provider else self._dedupe_targets(self.usage_reportable_providers())
        )
        reports: list[UsageReport] = []
        for target in targets:
            key = self._usage_cache_key(target)
            try:
                cached = cache.get(key, include_expired=True)
            except Exception:  # noqa: BLE001 — a bad read is an empty open
                cached = None
            if cached:
                reports.extend(cached)
        return reports

    def _invalidate_cached_usage(self, storage_id: str) -> None:
        """Best-effort drop of ``storage_id``'s cached usage row after a login.

        Never raises, for the reason ``_invalidate_cached_listing`` gives: a
        login that actually succeeded must not be reported as failed because a
        cache write went wrong afterwards. The next fetch re-derives the state
        from a live refresh either way; this only removes the stale answer that
        would otherwise be served ahead of it.
        """
        try:
            cache = self._usage_cache_store()
            if cache is not None:
                cache.invalidate(self._usage_cache_key(storage_id))
        except Exception:  # noqa: BLE001 — a stale row is not worth failing a login
            logger.debug("usage cache: post-login invalidate failed", exc_info=True)

    def _usage_cache_store(self) -> UsageCacheStore | None:
        """The shared usage cache, built lazily on first use.

        Lazy so a host that never asks for usage pays nothing. Injectable via
        the constructor so tests can aim it at a temp file instead of the real
        ``~/.local-operator/usage_cache.db``.
        """
        if self._usage_cache is None:
            try:
                self._usage_cache = UsageCacheStore()
            except Exception:  # noqa: BLE001 — no cache = live fetch, never fatal
                return None
        return self._usage_cache

    def _storage_id(self, provider: str) -> str:
        """The credential storage id for ``provider`` (aliases collapse).

        ``openai-device`` logs in under ``openai``, ``xai-oauth`` under ``xai``;
        the registry's ``store_credentials_as`` says so. Cache keys must follow
        the SAME aliasing or the two spellings of one account would hold two
        rows — one of them permanently stale.
        """
        definition = get_provider_definition(provider)
        return (definition.store_credentials_as or provider) if definition else provider

    def _usage_cache_key(self, provider: str) -> str:
        """The shared-cache key for ``provider``'s current account set."""
        return provider_cache_key(self._storage_id(provider), self._account_fingerprint(provider))

    def _account_fingerprint(self, provider: str) -> str:
        """A synchronous fingerprint of WHICH accounts ``provider`` would fetch.

        Built from the stored credential rows (identity keys for OAuth, a hash
        for API keys) plus any env key — no OAuth refresh, no network. Folding
        the account set into the cache key is what makes login/logout
        self-invalidating: the moment the set changes, the key changes and the
        stale row stops matching. See :mod:`usage_cache` for why the key names
        the account rather than the (rotating) access token.
        """
        storage = self._storage_id(provider)
        parts: list[str] = []
        # Cascade tiers 1/2 (runtime `--api-key`, models.yml pointer) can WIN
        # the fetch — `get_api_key` resolves them ahead of every stored row —
        # so they belong in the fingerprint too, or two sessions running on
        # different override keys would share one cache row and read each
        # other's numbers. `override_keys` is AuthStore's public accessor for
        # exactly this question; guarded because the narrow store protocol
        # does not require it, and a store without it has no overrides to name.
        override_keys = getattr(self.auth_store, "override_keys", None)
        if callable(override_keys):
            try:
                secrets = override_keys(provider)
            except Exception:  # noqa: BLE001 — overrides are optional context
                secrets = ()
            if isinstance(secrets, (list, tuple)):
                for secret in secrets:
                    parts.append(fingerprint_secret(str(secret)))
        try:
            rows = self.auth_store.list_credentials(storage)
        except Exception:  # noqa: BLE001 — an unreadable store fingerprints empty
            rows = []
        for row in rows:
            if getattr(row, "credential_type", None) == "oauth":
                identity = getattr(row, "identity_key", None)
                parts.append(identity or f"cred:{getattr(row, 'id', 0)}")
            else:
                data = getattr(row, "data", None) or {}
                key = data.get("key") if isinstance(data, dict) else None
                if key:
                    parts.append(fingerprint_secret(str(key)))
        try:
            env_key = resolve_env_key(storage)
        except Exception:  # noqa: BLE001
            env_key = None
        if env_key:
            parts.append(fingerprint_secret(env_key))
        return fingerprint_accounts(parts)

    def _expected_oauth_identities(self, provider: str) -> list[str]:
        """Stored OAuth identities for ``provider``, including refresh-failed.

        The expected set for ``/usage`` is the logged-in rows, not the subset
        that happened to mint a bearer this cycle. A sibling enumerator
        (:meth:`AuthStore.list_oauth_identities`) is preferred because it
        never calls ``_ensure_oauth_fresh``. An empty return from that
        sibling is authoritative — a runtime/config override short-circuits
        to ``[]`` so stored identity does not apply, and falling through to
        ``list_credentials`` (which does not honour overrides) would paint
        those OAuth emails as last-known stubs and skip the API-key route
        the session is actually using. ``list_credentials`` is only the
        shim for a store protocol that has not grown the sibling.
        """
        storage = self._storage_id(provider)
        try:
            enumerator = self.auth_store.list_oauth_identities
        except AttributeError:
            enumerator = None
        if enumerator is not None:
            try:
                accesses = enumerator(storage)
            except Exception:  # noqa: BLE001 — identities are labels, never fatal
                return []
            identities: list[str] = []
            for access in accesses:
                label = (
                    getattr(access, "email", None)
                    or getattr(access, "account_id", None)
                    or (
                        f"cred:{getattr(access, 'credential_id', 0)}"
                        if getattr(access, "credential_id", 0)
                        else None
                    )
                )
                if label:
                    identities.append(str(label))
            return identities
        try:
            rows = self.auth_store.list_credentials(storage)
        except Exception:  # noqa: BLE001
            return []
        labels: list[str] = []
        for row in rows:
            if getattr(row, "credential_type", None) != "oauth":
                continue
            data = getattr(row, "data", None) or {}
            label = None
            if isinstance(data, dict):
                label = data.get("email") or data.get("account_id")
            label = label or getattr(row, "identity_key", None) or f"cred:{getattr(row, 'id', 0)}"
            if label and not str(label).startswith("oauth:"):
                labels.append(str(label))
            elif getattr(row, "id", 0):
                labels.append(f"cred:{row.id}")
        return labels

    @staticmethod
    def _account_in_backoff(previous: UsageReport | None, now_ms: int, *, force: bool) -> bool:
        """Whether this account should be served from last-good, not re-probed.

        ``r`` always retries. An unavailable account stays dark until then.
        Otherwise the per-account ``next_probe_at_ms`` is the gate — siblings
        that are fresh still refresh on the same provider lease.
        """
        if force or previous is None:
            return False
        if previous.usage_unavailable:
            return True
        # NOTE: ``credential_invalid`` deliberately does NOT gate here, and
        # that is load-bearing rather than an omission.
        #
        # Skipping the cycle for a dead grant looks like it saves the retry
        # budget, and measurably saves nothing: ``list_oauth_accesses`` is
        # awaited before this loop and refreshes EVERY row, so the refresh
        # POST is already spent by the time the gate is consulted, and the
        # usage probe is short-circuited separately by the
        # ``access.credential_invalid`` branch, so no usage request is made
        # either. Measured on three consecutive cycles with and without the
        # skip: 1 refresh POST and 0 usage probes, identically.
        #
        # What the skip did buy was a one-way latch. The only writer that
        # clears the flag is ``_mark_account_success``, which is reached only
        # through a fetch this gate prevented -- so after the user followed
        # the panel's own ``/login`` advice, every automatic poll re-rendered
        # ``sign-in expired`` against a freshly-minted valid grant, and only
        # ``r`` could break the loop. That is this defect with its polarity
        # reversed: a permanent message that the user cannot act their way
        # out of. The verdict must be re-derived from the live refresh each
        # cycle, never remembered as a terminal state.
        # Only a FAILURE cool-down skips the probe. A successful account
        # leaves next_probe_at_ms unset so a sibling's shorter backoff can
        # expire the provider row without freezing the healthy logins.
        if previous.consecutive_failures <= 0:
            return False
        nxt = previous.next_probe_at_ms
        return nxt is not None and nxt > now_ms

    def _reset_account_for_force(self, report: UsageReport) -> UsageReport:
        """``r`` clears the failure streak so a maxed-out account is retried.

        ``credential_invalid`` is deliberately NOT cleared here. The streak
        and the unavailable ceiling are guesses about a provider that may have
        recovered, so ``r`` is right to drop them; a dead grant is a verdict
        the IdP returned, and clearing it optimistically would blank the one
        line telling the user to re-login until the fetch re-derived it.
        ``_mark_account_success`` clears it when a bearer actually works.
        """
        report.consecutive_failures = 0
        report.usage_unavailable = False
        report.next_probe_at_ms = None
        return report

    def _mark_account_success(self, report: UsageReport, now_ms: int) -> UsageReport:
        """A live 200: clear the failure streak and schedule the next probe."""
        report.consecutive_failures = 0
        report.usage_unavailable = False
        report.next_probe_at_ms = None
        # A 200 proves the grant minted a working bearer, so whatever the
        # cached row said about it is stale by definition. Clearing here is
        # what makes the state self-healing after the user re-logs in.
        report.credential_invalid = False
        return report

    def _mark_account_invalid(
        self,
        previous: UsageReport | None,
        *,
        provider: str,
        identity: str,
        now_ms: int,
    ) -> UsageReport:
        """The grant is dead: state it, and stop spending retries on it.

        Deliberately NOT ``_mark_account_failure`` with an extra flag. That
        path exists to decide when a run of transient misses has gone on long
        enough to stop trusting the numbers, and every part of it is wrong
        here: the failure streak measures an outage's length, the exponential
        backoff schedules a retry that cannot succeed, and
        ``usage_unavailable`` renders as ``usage unavailable - last known 2d
        ago``, which tells the user to wait when they need to act.

        So the streak is left untouched and ``next_probe_at_ms`` stays None.
        The account is not in a cool-down -- there is simply nothing to
        re-probe until the credential is replaced, and
        ``_account_in_backoff`` skips it on that flag alone. Last-known limits
        stay on the report: they remain the last true reading of a login the
        user still owns.
        """
        report = (
            previous
            if previous is not None
            else UsageReport(provider=provider, fetched_at=now_ms, identity=identity)
        )
        report.credential_invalid = True
        report.usage_unavailable = False
        report.next_probe_at_ms = None
        return report

    def _mark_account_failure(
        self,
        previous: UsageReport | None,
        *,
        provider: str,
        identity: str,
        now_ms: int,
    ) -> UsageReport:
        """Keep last-good (if any) and bump this account's consecutive misses.

        After :data:`USAGE_ACCOUNT_MAX_FAILURES` the account stays on the
        panel as usage-unavailable. Last-good numbers, when they exist, stay
        on the report so the operator can still see they are logged in and
        what the last successful check said.

        A transient miss also CLEARS any dead-grant verdict, which is not the
        contradiction it first looks like. Reaching here means the store
        handed back a usable bearer and the usage endpoint was what failed --
        so the grant refreshed, which is direct evidence it is alive. Leaving
        the flag set let one endpoint blip re-latch a healthy credential as
        ``sign-in expired``, telling the user to re-authenticate when nothing
        was wrong with their login. A cycle that genuinely still sees
        ``invalid_grant`` sets it again on the spot.
        """
        failures = (previous.consecutive_failures if previous is not None else 0) + 1
        unavailable = failures >= USAGE_ACCOUNT_MAX_FAILURES
        if previous is not None:
            report = previous
        else:
            report = UsageReport(provider=provider, fetched_at=now_ms, identity=identity)
        report.consecutive_failures = failures
        report.usage_unavailable = unavailable
        report.credential_invalid = False
        report.next_probe_at_ms = None if unavailable else now_ms + account_backoff_ms(failures)
        return report

    def _payload_expires_at_ms(self, reports: list[UsageReport], now_ms: int) -> int:
        """When the shared row should go stale so the next due account is probed.

        A mixed payload used to inherit the full 5-minute success TTL, which
        swallowed the 10 s / 20 s / … per-account backoff: the failed login
        sat un-retried until the whole set expired. Expiry is the soonest
        ``next_probe_at_ms`` still in the future; if every account is
        unavailable (only ``r`` retries) the full jittered TTL keeps the
        warmer from spinning.
        """
        soonest: int | None = None
        for report in reports:
            nxt = report.next_probe_at_ms
            if nxt is None or nxt <= now_ms:
                continue
            if soonest is None or nxt < soonest:
                soonest = nxt
        return soonest if soonest is not None else now_ms + self._jittered_ttl_ms()

    def _merge_account_reports(
        self,
        *,
        provider: str,
        expected: list[str],
        live: dict[str, UsageReport],
        previous: dict[str, UsageReport],
        now_ms: int,
        force: bool,
    ) -> list[UsageReport]:
        """Union of this fetch's successes and last-good for everyone else.

        A provider-level ``cache.set`` of only the accounts that succeeded
        this round is the #277 leftover that dropped a 429'd login from
        ``/usage``: the next warm read served a 3-account payload over a
        4-account login set. Expected order is the stored-row order so the
        panel does not reshuffle under the reader.
        """
        merged: list[UsageReport] = []
        seen: set[str] = set()
        for identity in expected:
            seen.add(identity)
            if identity in live:
                merged.append(self._settle_live_report(live[identity], now_ms))
                continue
            prior = previous.get(identity)
            if self._account_in_backoff(prior, now_ms, force=force) and prior is not None:
                # Still inside this account's cool-down (or already
                # unavailable): serve last-good without incrementing. A
                # force refresh never takes this branch.
                merged.append(prior)
                continue
            merged.append(
                self._mark_account_failure(
                    prior, provider=provider, identity=identity, now_ms=now_ms
                )
            )
        # An API-key report (or an identity the store no longer names) still
        # belongs on the panel if it succeeded this fetch; do not invent
        # stubs for leftovers that are no longer logged in.
        for identity, report in live.items():
            if identity in seen:
                continue
            merged.append(self._settle_live_report(report, now_ms))
        return merged

    def _settle_live_report(self, report: UsageReport, now_ms: int) -> UsageReport:
        """Finish a report this cycle produced, honouring a dead-grant verdict.

        Everything in ``live`` used to be a 200 by construction, so the merge
        could call ``_mark_account_success`` on all of it. A dead-grant entry
        is also produced by this cycle (it is a fresh verdict, not last-good)
        but it is the opposite of a success, and passing it through the
        success path would clear the very flag it was created to carry.
        """
        if report.credential_invalid:
            return report
        return self._mark_account_success(report, now_ms)

    @staticmethod
    def _jittered_ttl_ms() -> int:
        """Base TTL spread ±25%, so several accounts/providers do not all expire
        into the same refresh window (the per-IP burst that earns a 429)."""
        jitter = USAGE_REPORT_TTL_MS * (random.random() * 0.5 - 0.25)
        return int(USAGE_REPORT_TTL_MS + jitter)

    async def _fetch_provider_cached(
        self,
        client: httpx.AsyncClient,
        provider: str,
        force_refresh: bool,
    ) -> list[UsageReport]:
        """One provider's reports, cache-first.

        Fast path: a fresh cache entry returns with no network at all. The
        slow path delegates to :meth:`_refresh_provider_usage`, where the
        cross-process lease ensures only one session on the machine actually
        crosses the network for a stale row — every other session serves the
        stale value while that one refreshes. That lease is the coordination;
        no in-process future map is needed on top of it.
        """
        cache = self._usage_cache_store()
        key = ""
        if cache is not None:
            key = self._usage_cache_key(provider)
            if not force_refresh:
                fresh = cache.get(key)
                if fresh is not None:
                    return fresh
        return await self._refresh_provider_usage(client, provider, key, cache, force_refresh)

    async def _refresh_provider_usage(
        self,
        client: httpx.AsyncClient,
        provider: str,
        key: str,
        cache: UsageCacheStore | None,
        force_refresh: bool,
    ) -> list[UsageReport]:
        """Actually cross the network for ``provider``, then settle the cache.

        Cross-process coordination lives here: when the cached row is stale,
        ONE session wins a lease and refreshes while the others serve the stale
        row rather than joining a synchronized fan-out (Anthropic/OpenAI
        rate-limit the usage endpoint per source IP). On failure the last good
        value is served with a short cool-down, so a blip never blanks the
        report.

        **An empty result never overwrites non-empty last-good data.** The
        fetchers signal transport/HTTP failure by returning ``None`` —
        ``_get_json`` swallows ``httpx.HTTPError``, non-200s (including 429)
        and bad JSON — so by the time a result reaches this function, "the
        endpoint is down" and "the account has no quota to report" are the
        same empty list. The one disambiguating fact on hand is history: a
        provider that HAD data a moment ago and reports none now is far more
        likely rate-limited than suddenly quota-less, so the empty answer is
        treated as a failure (last-good kept servable under a short cool-down,
        retried on the next poll). A provider with no history of data — or
        whose last answer was also empty — negative-caches the empty list at
        the full TTL, which is what stops the warmer from re-hitting endpoints
        that legitimately report nothing.
        """
        stale: list[UsageReport] | None = None
        lease_held = False
        #: The empty answer was BELIEVED (no history, or history too old), so
        #: the stale row must not be served over it — see the acceptance branch.
        accepted_empty = False
        if cache is not None and key:
            stale = cache.get(key, include_expired=True)
            # The lease only has something to protect when a stale value exists:
            # the loser serves it while the winner refreshes. With nothing on
            # hand every session must fetch anyway (the pre-cache behaviour).
            if not force_refresh and stale is not None:
                if not cache.try_lease(key):
                    # A peer session owns this refresh; its result lands in the
                    # same shared row. Serve what we have instead of doubling
                    # the fan-out.
                    return stale
                # Held ONLY when try_lease actually granted it: the force path
                # never takes the lease, and releasing one it does not hold
                # would free a concurrent warmer's lease (holder identity is
                # per-process, not per-coroutine).
                lease_held = True
        try:
            try:
                reports = await self._fetch_provider(
                    client, provider, previous=stale or [], force_refresh=force_refresh
                )
            except Exception:  # noqa: BLE001 — isolate a broken provider
                reports = []
            if cache is not None and key:
                if reports:
                    now_ms = int(time.time() * 1000)
                    # The merged payload already carries last-good for
                    # accounts that failed this round. Writing only the live
                    # successes is the #277 leftover that shrank a 4-account
                    # snapshot to 3 the next time one token 429'd.
                    cache.set(
                        key,
                        provider,
                        reports,
                        expires_at_ms=self._payload_expires_at_ms(reports, now_ms),
                    )
                elif stale:
                    # Truthiness, not `is not None`: only a NON-EMPTY history
                    # marks this empty answer as a probable outage. Keep the
                    # last good value servable through a short cool-down —
                    # unless the data is old enough that the "outage" reading
                    # has expired (EMPTY_OVER_DATA_ACCEPT_MS), in which case
                    # the empty answer is accepted and negative-cached so a
                    # genuinely quota-less provider stops being re-fetched on
                    # every cool-down forever.
                    now_ms = int(time.time() * 1000)
                    newest = max((int(r.fetched_at or 0) for r in stale), default=0)
                    if newest and now_ms - newest > EMPTY_OVER_DATA_ACCEPT_MS:
                        cache.set(key, provider, [], expires_at_ms=now_ms + self._jittered_ttl_ms())
                        accepted_empty = True
                    else:
                        cache.write_failure(key, provider)
                else:
                    # No history of data (or an empty one): negative-cache so
                    # the warmer stops re-hitting an endpoint that reports
                    # nothing. `r` (force_refresh) still bypasses this row.
                    now_ms = int(time.time() * 1000)
                    cache.set(key, provider, [], expires_at_ms=now_ms + self._jittered_ttl_ms())
                    accepted_empty = True
        finally:
            if lease_held and cache is not None and key:
                cache.release_lease(key)
        if reports:
            return reports
        if stale is not None and not accepted_empty:
            # A forced refresh that failed still shows the last good numbers
            # (their age is stated in the panel) rather than an empty card. An
            # ACCEPTED empty answer is not papered over with old data, though —
            # the cache just recorded "this provider reports nothing" and the
            # caller should say the same.
            return stale
        return []

    async def _fetch_provider(
        self,
        client: httpx.AsyncClient,
        provider: str,
        *,
        previous: list[UsageReport] | None = None,
        force_refresh: bool = False,
    ) -> list[UsageReport]:
        """Every logged-in account's usage for one provider.

        Quota is per ACCOUNT, so a provider with two logins has two answers.
        Asking :meth:`AuthStore.get_oauth_access` produced one of them, chosen
        by a round-robin that also moved between refreshes: a user with two
        Anthropic accounts saw a single block and could not tell which login it
        described, or that a second one was missing entirely.

        The expected set is the *stored* OAuth rows (blocked included;
        refresh-failed included). A live 200 replaces that account's last-good;
        a 429 / ``None`` / exception keeps the previous numbers and increments
        that account's failure count. The written cache payload is the union,
        never "whoever answered this round" — that shrink is how one 429
        dropped a fourth Anthropic login from ``/usage``.

        The API-key route stays a single report, and is only reached when no
        OAuth identity is stored. An API key is not an identity — the
        cascade's env/config tiers resolve one secret per provider — so fanning
        out there would report the same numbers twice.
        """
        if not usage_supported(provider):
            return []
        now_ms = int(time.time() * 1000)
        expected = self._expected_oauth_identities(provider)
        previous_by_id = {
            report_identity_key(report): report
            for report in (previous or [])
            if report_identity_key(report)
        }
        if force_refresh:
            # ``r`` retries every expected account, including ones that had
            # already hit the unavailable ceiling. Reset first so the
            # backoff gate does not skip them.
            previous_by_id = {
                key: self._reset_account_for_force(report) for key, report in previous_by_id.items()
            }
        live: dict[str, UsageReport] = {}
        if expected:
            # Bearer mint is still list_oauth_accesses (routing contract
            # unchanged). Identities with no bearer this cycle stay in
            # ``expected`` and fall through to last-good / unavailable.
            accesses_by_id: dict[str, Any] = {}
            try:
                accesses = await self.auth_store.list_oauth_accesses(provider)
            except Exception:  # noqa: BLE001 — no bearer is a per-account miss
                accesses = []
            for access in accesses:
                label = (
                    getattr(access, "email", None)
                    or getattr(access, "account_id", None)
                    or (
                        f"cred:{getattr(access, 'credential_id', 0)}"
                        if getattr(access, "credential_id", 0)
                        else None
                    )
                )
                if label:
                    accesses_by_id[str(label)] = access
            for identity in expected:
                prior = previous_by_id.get(identity)
                if self._account_in_backoff(prior, now_ms, force=force_refresh):
                    continue
                access = accesses_by_id.get(identity)
                if access is None:
                    continue
                if getattr(access, "credential_invalid", False):
                    # The store minted no bearer and said why: the grant is
                    # dead. Record that verdict instead of probing with an
                    # empty token, which would fail as a generic 401 and land
                    # this account back in the transient-failure path.
                    live[identity] = self._mark_account_invalid(
                        prior, provider=provider, identity=identity, now_ms=now_ms
                    )
                    continue
                try:
                    report = await self._fetch_one(client, provider, access=access)
                except Exception:  # noqa: BLE001 — one bad account, not the provider
                    continue
                if report is None:
                    continue
                if not report.identity:
                    report.identity = identity
                live[report_identity_key(report) or identity] = report
            return self._merge_account_reports(
                provider=provider,
                expected=expected,
                live=live,
                previous=previous_by_id,
                now_ms=now_ms,
                force=force_refresh,
            )
        try:
            report = await self._fetch_one(client, provider, access=None)
        except Exception:  # noqa: BLE001
            return []
        if report is None:
            return []
        return [self._mark_account_success(report, now_ms)]

    def _dedupe_targets(self, targets: list[str]) -> list[str]:
        """Keep one id per storage row so alias providers don't double-fetch."""
        seen: set[str] = set()
        ordered: list[str] = []
        for provider in targets:
            storage = credential_provider_id(provider)
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
                        label=model_label(definition.id, model_id, info.name or "").full,
                        context_window=max(0, info.context_window or 0),
                        input_price=_price(info.input_price, definition),
                        output_price=_price(info.output_price, definition),
                        connected=connected,
                        aggregated=definition.id in AGGREGATOR_PROVIDERS,
                    )
                )
        return entries

    def initial_catalogue(self, *, cache_dir: Any = None) -> list[CatalogueEntry]:
        """First frame catalogue: shipped models layered with cached aggregator listings.

        Synchronous, non-blocking, and network-free. While direct providers have
        stable shipped static models in the registry, aggregator providers
        (OpenRouter, Radient) have no hardcoded registry models and rely on their
        dynamic catalogues. When a previous live listing exists on disk, reading it
        via :func:`cached_available_models` allows hundreds of available models to
        paint on the very first frame rather than appearing only after a network
        round trip.
        """
        entries: list[CatalogueEntry] = []
        usable = self.usable_providers()
        for definition in PROVIDER_REGISTRY:
            connected = usable is None or definition.id in usable
            if definition.id in AGGREGATOR_PROVIDERS:
                models, _status = cached_available_models(definition.id, cache_dir=cache_dir)
                for model in models:
                    entries.append(
                        CatalogueEntry(
                            provider=definition.id,
                            model_id=model.id,
                            label=model_label(definition.id, model.id, model.name or "").full,
                            context_window=max(0, model.context_window),
                            input_price=_price(model.input_price, definition, free=model.free),
                            output_price=_price(model.output_price, definition, free=model.free),
                            connected=connected,
                            aggregated=True,
                            routed=model.routed,
                        )
                    )
            else:
                for model_id, info in static_models(definition.id).items():
                    entries.append(
                        CatalogueEntry(
                            provider=definition.id,
                            model_id=model_id,
                            label=model_label(definition.id, model_id, info.name or "").full,
                            context_window=max(0, info.context_window or 0),
                            input_price=_price(info.input_price, definition),
                            output_price=_price(info.output_price, definition),
                            connected=connected,
                            aggregated=False,
                        )
                    )
        return entries

    def entry_for(
        self,
        provider: str,
        model_id: str,
        *,
        spec: ModelSpec | None = None,
    ) -> CatalogueEntry | None:
        """One entry for ``provider``/``model_id``, or ``None`` if unknown here.

        Exists for the model a session is ALREADY RUNNING. A picker must offer
        it whatever the catalogue says, and after an authoritative listing it
        may not be in the catalogue at all: the account's live list is allowed
        to prune bundled ids, so a session started on one of those ids had its
        own model disappear from the list while the status band still named it,
        and typing the id answered "no matching models".

        ``spec`` is the session's already-resolved active model. It matters for
        aggregators: they deliberately have no ENUMERABLE static catalogue, so
        ``static_models("openrouter")`` is empty even for a model the session
        is running. Re-listing here would put synchronous network/cache work
        back on the TUI thread; the spec already carries the name and context
        that session startup resolved. Prices are unknown on ``ModelSpec`` and
        stay unknown rather than being invented.

        Built here rather than in the caller because the normalization is this
        module's job (see :class:`CatalogueEntry`): a caller reaching into the
        registry itself would have to know that a context window of ``-1`` and
        ``0`` both mean unknown, and would spell the price rules a second time.
        ``None`` means neither the registry nor the active spec describes this
        pair, which is a real answer for a model an operator configured by hand.
        """
        definition = get_provider_definition(provider)
        if definition is None:
            return None
        info = static_models(definition.id).get(model_id)
        if info is not None:
            name = info.name or ""
            context_window = max(0, info.context_window or 0)
            input_price = _price(info.input_price, definition)
            output_price = _price(info.output_price, definition)
        elif spec is not None and spec.provider == definition.id and spec.model_id == model_id:
            name = spec.display_name
            context_window = max(0, spec.context_window)
            input_price = -1.0
            output_price = -1.0
        else:
            return None
        usable = self.usable_providers()
        return CatalogueEntry(
            provider=definition.id,
            model_id=model_id,
            label=model_label(definition.id, model_id, name).full,
            context_window=context_window,
            input_price=input_price,
            output_price=output_price,
            connected=usable is None or definition.id in usable,
            aggregated=definition.id in AGGREGATOR_PROVIDERS,
            # From the id alone, because this branch has no listing row to read
            # a price off: it is the fallback for when the listing could not be
            # had at all. That makes it the path a user running `radient/auto`
            # on a cold cache actually takes, so leaving it False here would
            # blank the label on exactly the surface that reported the bug.
            # Provider-scoped: `ollama/auto` reaches this same branch (R1).
            routed=is_meta_route_id(model_id, definition.id),
        )

    async def live_catalogue(
        self, *, ttl_s: float | None = None
    ) -> tuple[list[CatalogueEntry], dict[str, str]]:
        """The catalogue with each provider's LIVE listing layered over the registry.

        Returns ``(entries, statuses)`` where ``statuses`` maps provider id to one
        of discovery's status strings, so a UI can say "stale" or "login
        required" instead of implying the catalogue is complete.

        ``ttl_s`` is the hard TTL passed to discovery; the picker passes
        :data:`PICKER_TTL_S`, and ``None`` keeps discovery's default.

        Only providers with a credential are fetched. An unconnected provider still
        contributes its STATIC models — the question "what would I get if I logged
        in here" is precisely what a user cannot otherwise answer, and it was the
        reason a newly released model was undiscoverable.

        Each provider is isolated: discovery never raises by contract, but a
        credential resolution can (an OAuth refresh against a dead network), and
        one broken provider must not empty the whole list. Provider listings are
        fetched concurrently via :func:`asyncio.gather` so overall latency bounds
        to the single slowest provider rather than the sum of all provider round
        trips.
        """
        entries: list[CatalogueEntry] = []
        statuses: dict[str, str] = {}
        usable = self.usable_providers()

        async def _fetch_provider(
            definition: ProviderDefinition,
        ) -> tuple[ProviderDefinition, bool, list[DiscoveredModel], str]:
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
            # also called from the CLI and the server), and fetching on the loop
            # would freeze a TUI's repaint. Providers run concurrently.
            models, status = await asyncio.to_thread(available_models, definition.id, **kwargs)
            return definition, connected, models, status

        results = await asyncio.gather(*[_fetch_provider(defn) for defn in PROVIDER_REGISTRY])
        listed: list[tuple[ProviderDefinition, bool, list[DiscoveredModel]]] = []
        for definition, connected, models, status in results:
            statuses[definition.id] = status
            listed.append((definition, connected, models))

        # Prices for the rows no listing priced, from the same keyless chain the
        # status band resolves through (see :func:`_enrich_prices`). After the
        # listings rather than per provider so the two documents are read ONCE
        # for the whole catalogue, and off-loop for the same reason the listings
        # are: the OpenRouter document is ~120 KB of JSON.
        rows_by_provider = await asyncio.to_thread(
            _enrich_prices, [(definition, models) for definition, _connected, models in listed]
        )
        for definition, connected, _models in listed:
            for model in rows_by_provider[definition.id]:
                entries.append(
                    CatalogueEntry(
                        provider=definition.id,
                        model_id=model.id,
                        label=model_label(definition.id, model.id, model.name or "").full,
                        context_window=max(0, model.context_window),
                        # ``free`` is consumed HERE and goes no further: a
                        # stated zero survives ``_price`` as ``0.0``, which is
                        # already the entry's way of saying free (an unknown is
                        # ``-1.0``). Carrying the flag onto the entry as well
                        # would be a second spelling of one fact, free to drift.
                        input_price=_price(model.input_price, definition, free=model.free),
                        output_price=_price(model.output_price, definition, free=model.free),
                        connected=connected,
                        aggregated=definition.id in AGGREGATOR_PROVIDERS,
                        routed=model.routed,
                    )
                )
        return entries, statuses

    async def _listing_credential(self, provider: str) -> tuple[str | None, bool, str | None]:
        """``(secret, is_oauth, account_id)`` for a model listing call.

        The account id is part of OpenAI's ChatGPT authorization boundary; the
        current Codex catalogue rejects a subscription token without it. Other
        providers ignore the value, while ``is_oauth`` still selects Anthropic's
        bearer header instead of its API-key header.

        The lookup follows ``store_credentials_as``, because the credential a
        login flavour needs is not stored under its own id. ``openai-device``
        and ``xai-oauth`` are login flavours of ``openai`` and ``xai`` -- the
        registry says so, and ``discovery._static_rows`` already follows it for
        the bundled rows -- and the login writes ONE row, under the aliased
        name. Asking ``AuthStore`` for the literal id (its ``WHERE provider = ?``
        is exact) therefore found nothing, so the flavour listed anonymously:
        for ``openai-device`` that meant no OAuth, no account scope, no
        account-scoped catalogue, and the picker offering that logged-in
        ChatGPT account the bundled ``gpt-4o``/``o3`` rows under a second
        prefix -- the very ids this listing exists to stop presenting as
        current.
        """
        credential_id = credential_provider_id(provider)
        access = await self.auth_store.get_oauth_access(credential_id)
        if access is not None and access.kind == "oauth" and access.access_token:
            return access.access_token, True, access.account_id or access.org_id
        if access is not None and access.access_token:
            return access.access_token, False, None
        try:
            stored = await self.auth_store.get_api_key(credential_id)
        except Exception:  # noqa: BLE001 — a refresh failure just means no listing
            stored = None
        # The environment is the last tier of the same cascade the stream uses, so
        # a key set there has to reach the listing too: otherwise the provider a
        # session is ACTUALLY RUNNING ON is the one whose catalogue stays empty.
        return stored or resolve_env_key(credential_id), False, None

    async def _fetch_one(
        self,
        client: httpx.AsyncClient,
        provider: str,
        *,
        access: OAuthAccess | None,
    ) -> UsageReport | None:
        """One report for one account.

        ``access`` is supplied by the caller rather than resolved here: which
        account this report is FOR is the caller's decision now that there can
        be several (see :meth:`_fetch_provider`). ``None`` means "no OAuth
        account" and selects the API-key route.
        """
        if not usage_supported(provider):
            return None
        access_token: str | None = None
        api_key: str | None = None
        account_id: str | None = None
        if access is not None and access.kind == "oauth":
            access_token = access.access_token
            account_id = access.account_id
        elif access is not None and access.access_token:
            api_key = access.access_token
        if access_token is None and api_key is None:
            # Only when no OAuth account was handed in: with one, falling back
            # to the cascade's key would silently report a DIFFERENT account's
            # numbers under this account's identity.
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
            # The raw row lets split-token providers (QwenCloud Token Plan:
            # sk-sp inference key vs. OAuth usage token) spend the right one,
            # where access_token is already the wire-mapped key.
            oauth_creds=access.raw if access is not None and access.kind == "oauth" else None,
        )
        if report is not None and not report.identity and access is not None:
            # Whose account this is. The field existed and no fetcher ever set it, so
            # the TUI's annotation for it was unreachable — and it matters most
            # exactly where usage does: an operator with two accounts on one provider
            # needs to know which one the numbers describe.
            report.identity = getattr(access, "email", None) or access.account_id
        return report


def _invalidate_cached_listing(storage_id: str) -> None:
    """Best-effort listing drop after a credential change; never raises.

    An exception here would fail a login that actually succeeded, which is far
    worse than a stale list that the picker's TTL clears within the quarter
    hour anyway. The in-process resolver memo is dropped too: a status-band
    resolution that degraded BEFORE the credential arrived (no key →
    registry-only limits/price) is memoised per TTL bucket and would otherwise
    stay pinned for the rest of the bucket in a long-lived TUI. Same pairing
    the server's credential route performs for exactly this event.
    """
    try:
        invalidate_listing(storage_id)
    except Exception:  # noqa: BLE001 - never fail a successful login over a cache
        logger.debug("listing invalidation failed for %s", storage_id, exc_info=True)
    try:
        from local_operator.model.configure import invalidate_model_info_cache

        invalidate_model_info_cache()
    except Exception:  # noqa: BLE001 - same rule as the listing drop above
        logger.debug("model-info invalidation failed for %s", storage_id, exc_info=True)


def _enrich_prices(
    listed: list[tuple[ProviderDefinition, list[DiscoveredModel]]],
) -> dict[str, list[DiscoveredModel]]:
    """Each provider's rows with price/limit HOLES filled from the keyless chain.

    WHY: the picker used to price a row from ``merge_models(registry, listing)``
    alone, while the status band priced the same model through the resolver's
    models.dev/OpenRouter leg. A direct-provider model the shipped registry did
    not carry therefore showed a blank price in the picker (``_price``'s unknown
    sentinel) and ``$10/50`` in the band the moment it was selected — the
    operator's ``claude-fable-5-1`` screenshot. Both surfaces now go through
    ``prices.price_row`` so they cannot drift again.

    CONSTRAINTS. (1) Disk only, one read per document: the models.dev projection
    is ~141 KB and the OpenRouter document ~120 KB; parsing either per row would
    turn 400 OpenRouter rows into seconds, and ``resolve_model_info`` per row is
    a three-leg memoised resolution that may fetch. The OpenRouter rows come
    straight from the ``openrouter`` provider's own listing, which this same
    ``live_catalogue`` call has just read under the picker's TTL — so no second
    document, no second request. (2) Only rows whose listing quoted NO money are
    touched, and only the money and the limits the listing left at zero: a price
    the provider's own listing stated is authoritative and never overridden.
    (3) Aggregator rows are never enriched — their listing IS the priced source —
    and a provider the chain does not map (``ollama``, ``radient``) is left as
    is, so a keyless provider's genuine ``free`` stays free (``_price``). An
    aggregator's ``:free`` routes therefore take their ``free`` flag straight
    from ``discovery._row_from_openai_entry``, the parser that saw the explicit
    ``0`` on the wire, and never pass through here at all.
    """
    from local_operator.model.prices import models_dev_providers, price_row

    models_dev = models_dev_providers()
    openrouter: list[DiscoveredModel] = next(
        (rows for definition, rows in listed if definition.id == "openrouter"), []
    )
    result: dict[str, list[DiscoveredModel]] = {}
    for definition, rows in listed:
        if definition.id in AGGREGATOR_PROVIDERS or (models_dev is None and not openrouter):
            result[definition.id] = rows
            continue
        # The chain is keyed on the canonical provider, the same translation the
        # resolver applies: ``openai-device`` prices as ``openai``.
        canonical = credential_provider_id(definition.id)
        enriched: list[DiscoveredModel] = []
        for row in rows:
            if row.input_price > 0 or row.output_price > 0 or row.free:
                # ``free`` counts as priced: the listing already ANSWERED the
                # money question with a quoted zero, and re-asking the chain
                # could only replace that answer with a third party's rate.
                enriched.append(row)
                continue
            found = price_row(canonical, row.id, models_dev=models_dev, openrouter=openrouter)
            if found is None:
                enriched.append(row)
                continue
            if found.free:
                # A stated zero fills the HOLE without filling the prices: the
                # numbers stay 0.0 and the flag is what the picker reads. Kept
                # ahead of the positive-price test below because a free row has
                # no positive leg and would otherwise be dropped as unanswered.
                enriched.append(dataclasses.replace(row, free=True))
                continue
            if not (found.input_price > 0 or found.output_price > 0):
                enriched.append(row)
                continue
            enriched.append(
                dataclasses.replace(
                    row,
                    input_price=found.input_price,
                    output_price=found.output_price,
                    cache_read_price=row.cache_read_price or found.cache_read_price,
                    cache_write_price=row.cache_write_price or found.cache_write_price,
                    # Limits only where the listing gave none: the provider's
                    # own window is the right number for its endpoint.
                    context_window=row.context_window or found.context_window,
                    max_tokens=row.max_tokens or found.max_tokens,
                )
            )
        result[definition.id] = enriched
    return result


def _price(value: float | None, definition: ProviderDefinition, *, free: bool = False) -> float:
    """A per-million price, with UNKNOWN kept distinct from FREE.

    Discovery and the static registry both use ``0`` for "no price known", and the
    picker renders a genuine pair of zeroes as ``free`` — so passing an unknown
    through as zero advertises a paid model as costing nothing. Anthropic makes
    this immediate rather than theoretical: its listing carries no pricing at all,
    so every model it discovers that we did not already ship would read ``free``.

    ``-1`` is the unknown sentinel the picker blanks. Zero — and therefore the
    word ``free`` — survives in exactly two cases:

    * ``allows_missing_api_key``: a local Ollama really is free per token, and
      blanking that would hide the one thing that makes it interesting.
    * ``free``: a SOURCE stated the zero. That is a quoted price, not a silence,
      and repeating a quoted zero fabricates nothing. This is what makes the
      picker's ``free`` label reachable for the 18 ``:free`` OpenRouter routes,
      every one of which the listing prices at an explicit ``0``; before it, a
      stated zero collapsed into the unknown sentinel here and rendered as the
      same blank cell as a model nobody had priced.

    ``free`` is never derived from ``value`` — it arrives from the parser that
    read the wire (:attr:`DiscoveredModel.free`) — which is what keeps two rows
    that both reach here as ``0.0`` apart: a plan-billed row whose real cost is
    unknowable stays blank, because the plan catalogues do not set it (see
    ``prices._PLAN_BILLED_KEYS``), and so does a row nobody quoted at all.

    NOT the same ``-1.0`` as ``model.prices._STATED_ZERO``, which means the
    opposite — "models.dev stated this price and it is zero". That marker is
    module-private to ``prices`` and stripped back to ``0.0`` before any row
    reaches here, so the two never meet; they would collide silently if either
    one's reach were widened, hence the note on both.
    """
    if value is not None and value > 0:
        return float(value)
    return 0.0 if (free or definition.allows_missing_api_key) else -1.0
