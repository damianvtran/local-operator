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
from local_operator.web_search.providers import PROVIDERS, provider_available

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
    """Enable/disable a provider without disturbing the chosen priority order."""
    settings = load_search_settings(manager)
    if enabled and provider_id not in settings.providers:
        settings.providers.append(provider_id)
    elif not enabled:
        settings.providers = [value for value in settings.providers if value != provider_id]
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
    """Replace the enabled provider order; callers validate ids before this point."""
    settings = load_search_settings(manager)
    settings.providers = list(dict.fromkeys(providers))
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
    """Resolve configured providers and execute one search with fallback.

    ``round_robin`` rotates the first attempt per call, then walks the remaining
    providers as a fallback chain. This spreads successful traffic without
    sacrificing availability when a free tier is rate-limited or a scraper is
    challenged. ``ordered`` always starts from the configured first provider.
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

    def candidates(self, forced_provider: SearchProviderId | None = None) -> list[SearchProviderId]:
        if not self.settings.enabled:
            raise RuntimeError("Web search is disabled. Run `local-operator search on`.")
        configured = list(self.settings.providers)
        if forced_provider is not None:
            if forced_provider not in configured:
                raise RuntimeError(
                    f"Search provider {forced_provider!r} is disabled. "
                    f"Run `local-operator search enable {forced_provider}`."
                )
            return [forced_provider]
        if self.settings.strategy == "round_robin" and len(configured) > 1:
            offset = _next_offset(len(configured))
            configured = configured[offset:] + configured[:offset]
        return configured

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
                "No web search providers are enabled. Run "
                "`local-operator search enable duckduckgo`."
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
                    failures.append(f"{provider_id}: {error}")
                    continue
                if not response.sources and not (response.answer or "").strip():
                    failures.append(f"{provider_id}: returned no results")
                    continue
                response.failures = list(failures)
                return response

        summary = "; ".join(failures) or "no candidates"
        raise RuntimeError(f"All configured web search providers failed: {summary}")


def search_settings_dict(settings: WebSearchSettings) -> dict[str, Any]:
    """JSON-friendly settings view for server/CLI callers."""
    return settings.model_dump(mode="json")
