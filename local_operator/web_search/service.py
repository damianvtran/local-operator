"""Web-search configuration, status, and load-balanced execution."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx
from pydantic import ValidationError

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.web_search.io import WebReadIO
from local_operator.web_search.models import (
    DEFAULT_WEB_SEARCH_CONFIG,
    PROVIDER_IDS,
    SearchProviderId,
    SearchResponse,
    SearchStrategy,
    WebSearchSettings,
)
from local_operator.web_search.providers import (
    PROVIDERS,
    free_pool,
    provider_available,
    provider_refusal,
    resolve_provider_bands,
    resolve_providers,
)

_ROUND_ROBIN_LOCK = threading.Lock()
_ROUND_ROBIN_OFFSET = 0


def coerce_search_settings(raw: object) -> WebSearchSettings:
    """Validate loose YAML while preserving safe defaults for malformed fields."""
    merged = dict(DEFAULT_WEB_SEARCH_CONFIG)
    if isinstance(raw, Mapping):
        merged.update(raw)

    providers = merged.get("providers")
    if not isinstance(providers, list):
        merged["providers"] = list(DEFAULT_WEB_SEARCH_CONFIG["providers"])  # type: ignore[arg-type]
    else:
        # Stable de-duplication also drops stale provider ids from older/future
        # configs. One typo must not prevent every other provider from loading.
        merged["providers"] = list(
            dict.fromkeys(value for value in providers if value in PROVIDER_IDS)
        )

    # Same de-dupe and unknown-id drop as the priority list, with one deliberate
    # difference: a malformed value (a string, a mapping, None) reads as "nothing
    # excluded" rather than raising. The absent key and its empty list mean the
    # same thing -- no exclusions -- so there is no migration to write and no
    # config that a hand-edited typo can turn into a disabled chain.
    excluded = merged.get("excluded_providers")
    if not isinstance(excluded, list):
        merged["excluded_providers"] = []
    else:
        merged["excluded_providers"] = list(
            dict.fromkeys(value for value in excluded if value in PROVIDER_IDS)
        )

    try:
        settings = WebSearchSettings.model_validate(merged)
    except ValidationError:
        settings = WebSearchSettings.model_validate(DEFAULT_WEB_SEARCH_CONFIG)
    settings.timeout_seconds = min(max(settings.timeout_seconds, 1.0), 120.0)
    return settings


def load_search_settings(manager: ConfigManager) -> WebSearchSettings:
    """Read the current search mapping from a configuration manager."""
    return coerce_search_settings(manager.get_config_value("web_search", None))


def save_search_settings(manager: ConfigManager, settings: WebSearchSettings) -> None:
    """Persist only the stable public search fields under ``values.web_search``."""
    manager.set_config_value("web_search", settings.model_dump(mode="json"))


def set_search_enabled(manager: ConfigManager, enabled: bool) -> WebSearchSettings:
    settings = load_search_settings(manager)
    settings.enabled = enabled
    save_search_settings(manager, settings)
    return settings


def set_provider_enabled(
    manager: ConfigManager,
    provider_id: SearchProviderId,
    enabled: bool,
) -> WebSearchSettings:
    """Enable = clear the exclusion; disable = commit one.

    ONE writer per fact. ``enable`` deliberately does NOT append to
    ``web_search.providers`` any more: that append would promote the provider into
    the priority prefix -- for a metered provider (deepseek, or a keyed exa) that
    is a spend decision the user did not make. ``set_provider_order`` is the verb
    for priority, and `search enable` now only says "not excluded".

    ``disable`` does not remove the id from ``providers`` either: removal could
    empty the prefix into a value the settings registry rejects, and exclusion
    already beats listing at resolve time -- so the stored list stays readable and
    both status surfaces print the row as ``excluded``.
    """
    settings = load_search_settings(manager)
    if enabled:
        settings.excluded_providers = [
            value for value in settings.excluded_providers if value != provider_id
        ]
    elif provider_id not in settings.excluded_providers:
        settings.excluded_providers = [*settings.excluded_providers, provider_id]
    save_search_settings(manager, settings)
    return settings


def set_search_strategy(
    manager: ConfigManager,
    strategy: SearchStrategy,
) -> WebSearchSettings:
    settings = load_search_settings(manager)
    settings.strategy = strategy
    save_search_settings(manager, settings)
    return settings


def set_provider_order(
    manager: ConfigManager,
    providers: list[SearchProviderId],
) -> WebSearchSettings:
    """Replace the priority prefix; callers validate ids before this point.

    Naming a provider here also CLEARS its exclusion: it is an explicit request to
    use it, and honouring an older "never" over the user's most recent instruction
    is the same class of lie this ordering model exists to fix.
    """
    settings = load_search_settings(manager)
    ordered = list(dict.fromkeys(providers))
    settings.providers = ordered
    settings.excluded_providers = [
        value for value in settings.excluded_providers if value not in ordered
    ]
    save_search_settings(manager, settings)
    return settings


def set_searxng_endpoint(manager: ConfigManager, endpoint: str) -> WebSearchSettings:
    settings = load_search_settings(manager)
    settings.searxng_endpoint = endpoint.rstrip("/")
    save_search_settings(manager, settings)
    return settings


def _next_offset(size: int) -> int:
    """Return one process-wide fair starting offset for a provider set."""
    global _ROUND_ROBIN_OFFSET
    if size <= 1:
        return 0
    with _ROUND_ROBIN_LOCK:
        offset = _ROUND_ROBIN_OFFSET % size
        _ROUND_ROBIN_OFFSET += 1
    return offset


def reset_round_robin_for_tests() -> None:
    """Reset deterministic state. Kept explicit so tests never reach into globals."""
    global _ROUND_ROBIN_OFFSET
    with _ROUND_ROBIN_LOCK:
        _ROUND_ROBIN_OFFSET = 0


TavilyOAuthSearch = Callable[[str, int], Awaitable[SearchResponse]]


class WebSearchService:
    """Resolve this install's provider chain and execute one search with fallback.

    The chain is ``prefix ++ rotating band ++ free-fallback band ++ metered band``
    (see ``providers.resolve_provider_bands``). ``round_robin`` rotates the start
    of the ROTATING band only -- never a band boundary, so a metered leg can never
    be rotated in front of a free one -- which spreads successful traffic across
    the free pool without sacrificing availability when a free tier is rate-limited
    or a scraper is challenged. ``ordered`` always walks the bands in declared
    order.
    """

    def __init__(
        self,
        settings: WebSearchSettings,
        credentials: CredentialManager,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        tavily_oauth_search: TavilyOAuthSearch | None = None,
        io: WebReadIO | None = None,
    ) -> None:
        self.settings = settings
        self.credentials = credentials
        self.transport = transport
        self.tavily_oauth_search = tavily_oauth_search
        self.io = io

    def resolve(self) -> list[SearchProviderId]:
        """The chain as this session will walk it, before rotation.

        NON-rotating on purpose: the singleflight key and every status surface
        need a value that is stable within a call (the rotation offset moves).
        """
        return resolve_providers(self.settings, self.credentials)

    def candidates(self, forced_provider: SearchProviderId | None = None) -> list[SearchProviderId]:
        if not self.settings.enabled:
            raise RuntimeError("Web search is disabled. Run `local-operator search on`.")
        bands = resolve_provider_bands(self.settings, self.credentials)
        if forced_provider is not None:
            # Allowed iff it is in the resolved chain, so an EXCLUDED provider is
            # still refused (the forced path is also web_read's pin, and a pin must
            # not override an explicit "never"), while an available provider that
            # auto-joined the metered band is allowed. The refusal says WHY and
            # names the command that can actually fix it: the old copy always said
            # `search enable`, which stopped being able to help once `enable` meant
            # "clear an exclusion" (round-1 U3).
            if forced_provider not in (
                *bands.prefix,
                *bands.rotate,
                *bands.fallback,
                *bands.metered,
            ):
                raise RuntimeError(
                    f"Search provider "
                    f"{provider_refusal(forced_provider, self.settings, self.credentials)}"
                )
            return [forced_provider]
        # ONE rotating free pool: the listed free legs in their listed order, then
        # the auto-joined free band. Rotating only the auto band pinned the first
        # attempt to providers[0] on every install (round-1 M1/Q1/D2/U1).
        pool = free_pool(self.settings, self.credentials)
        if self.settings.strategy == "round_robin" and len(pool) > 1:
            offset = _next_offset(len(pool))
            pool = pool[offset:] + pool[:offset]
        # THE BAND INVARIANT, strengthened: nothing that spends can rotate, and
        # nothing that spends can precede a free leg -- the paid band is appended
        # last here and a listed paid leg is resolved INTO that band rather than
        # left in the prefix.
        return [*pool, *bands.fallback, *bands.metered]

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        forced_provider: SearchProviderId | None = None,
    ) -> SearchResponse:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Search query must not be empty")
        limit = min(max(limit, 1), 20)
        failures: list[str] = []
        candidates = self.candidates(forced_provider)
        if not candidates:
            raise RuntimeError(
                "No web search providers are in this session's chain. Run "
                "`local-operator search enable duckduckgo`, or `local-operator search "
                "list` to see what is excluded."
            )

        timeout = httpx.Timeout(self.settings.timeout_seconds)
        owner = self.io or WebReadIO()
        try:
            return await self._search_with_client(
                owner, timeout, candidates, clean_query, limit, failures
            )
        finally:
            if self.io is None:
                await owner.aclose()

    async def _search_with_client(
        self,
        owner: WebReadIO,
        timeout: httpx.Timeout,
        candidates: list[SearchProviderId],
        clean_query: str,
        limit: int,
        failures: list[str],
    ) -> SearchResponse:
        async with owner.client(
            ("search", self.settings.timeout_seconds, id(self.transport)),
            timeout=timeout,
            follow_redirects=True,
            transport=self.transport,
        ) as client:
            for provider_id in candidates:
                if provider_id == "tavily" and self.tavily_oauth_search is not None:
                    try:
                        oauth_response = await self.tavily_oauth_search(clean_query, limit)
                    except Exception as error:
                        # OAuth MCP is an optional higher-trust transport. Its
                        # failure must not suppress Tavily's keyless/API path.
                        failures.append(f"tavily OAuth MCP: {error}")
                    else:
                        if oauth_response.sources or (oauth_response.answer or "").strip():
                            oauth_response.failures = list(failures)
                            return oauth_response
                        failures.append("tavily OAuth MCP: returned no results")
                if not provider_available(provider_id, self.credentials, self.settings):
                    # Kept INSIDE the loop on purpose: a provider the user LISTED but
                    # that has no usable credential stays in the chain and reports
                    # `not configured` here, rather than being silently dropped from
                    # the chain the status surfaces describe.
                    failures.append(f"{provider_id}: not configured")
                    continue
                try:
                    response = await PROVIDERS[provider_id].search(
                        client,
                        self.credentials,
                        self.settings,
                        clean_query,
                        limit,
                    )
                except Exception as error:  # one provider failure is the fallback trigger
                    # `str(error) or type(error).__name__`: a transport error whose
                    # message is empty -- httpx.ReadTimeout is the one that happens --
                    # used to render as a bare `exa: ` in the digest and, once the
                    # lead moved to the last leg, as an empty LEAD sentence in the
                    # cropped card (round-2 N4/Q2-2, round-1 Q3).
                    failures.append(f"{provider_id}: {str(error) or type(error).__name__}")
                    continue
                if not response.sources and not (response.answer or "").strip():
                    failures.append(f"{provider_id}: returned no results")
                    continue
                # Price the search in ONE place for every provider: the
                # transports report usage, the price table and the token
                # arithmetic live in web_search.cost, and a transport that
                # already knows its own cost (it recorded usage mid-flight)
                # keeps it.
                if response.cost is None:
                    from local_operator.web_search.cost import estimate_search_cost

                    response.cost = estimate_search_cost(provider_id, response.usage)
                # The provider's OWN failures are preserved alongside the
                # chain's. Replacing them hid a provider's partial degradation
                # (a search that succeeded but whose optional enrichment pass
                # failed) at exactly the moment the caller needed to know why
                # the result looked thinner than expected.
                response.failures = [*failures, *response.failures]
                return response

        summary = "; ".join(failures) or "no candidates"
        # Name only what was TRIED, and put the COUNT first: a six-leg chain makes
        # the whole-prefix summary a ~1200-cell string, and the collapsed tool card
        # paints about 25 cells of it, so anything after the opening clause is always
        # cropped. Two rounds of measurement got here -- round 1 dropped the false
        # "All configured …" preamble, round 2 measured that the count sitting at
        # cell ~57 could never appear (`D2-1`), so the shape is now
        # `6/6 providers failed: <reason> (<tried list>)`: count, cause, and the
        # list for the expansion.
        #
        # One candidate covers both the forced call and the one-provider install,
        # and this function cannot tell them apart -- so the message says what is
        # true of either: this provider failed and no other was tried.
        if len(candidates) == 1:
            only = candidates[0]
            detail = summary
            redundant = f"{only}: "
            if detail.startswith(redundant):
                detail = detail[len(redundant) :]
            raise RuntimeError(
                f"Web search failed: {detail} ({only!r} was the only provider tried)"
            )
        # Split each entry back into its provider and its reason. The reasons are in
        # the order the legs were TRIED, which is the order a reader wants them in.
        tried = [entry.partition(": ")[::2] for entry in failures]
        # Lead with the first reason that says something. `not configured` is a
        # placeholder (the leg was never going to run) and an empty reason is a
        # formatting artefact -- leading with either is how the cropped card came to
        # show `Web search failed: not configured` while the leg that actually failed
        # on the wire was named later (round-2 N4/Q2-2).
        lead = next(
            (reason for _provider, reason in tried if reason and reason != "not configured"),
            tried[-1][1] if tried else "no candidates",
        )
        # COUNT FIRST, because the collapsed tool card paints ~25 cells of this
        # string and the trailing list is always cropped: `6/6 providers failed: …`
        # puts the count, the fact and the start of the cause inside that budget,
        # and the tried list stays for the expansion (round-2 D2-1).
        # ...and every leg's own reason stays after it: the expansion is where a
        # diagnostic belongs, and round 2 asked for the per-leg breakdown back
        # (`tried:` names the order the legs were attempted in, which under
        # round-robin is this call's order and not a stable chain order).
        raise RuntimeError(
            f"{len(candidates)}/{len(candidates)} providers failed: {lead} (tried: {summary})"
        )


def search_settings_dict(settings: WebSearchSettings) -> dict[str, Any]:
    """JSON-friendly settings view for server/CLI callers."""
    return settings.model_dump(mode="json")
