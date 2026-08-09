"""Shared web-search configuration and result models.

The built-in providers intentionally expose one small contract.  Provider-specific
payloads stop at this boundary so the model-facing tool, CLI status view, and TUI
cannot drift into seven subtly different result formats.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SearchProviderId = Literal[
    "duckduckgo",
    "tavily",
    "perplexity",
    "brave",
    "exa",
    "serpapi",
    "searxng",
]
SearchStrategy = Literal["round_robin", "ordered"]

PROVIDER_IDS: tuple[SearchProviderId, ...] = (
    "duckduckgo",
    "tavily",
    "perplexity",
    "brave",
    "exa",
    "serpapi",
    "searxng",
)


class SearchSource(BaseModel):
    """One normalized result from any search provider."""

    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    snippet: str | None = None
    published_date: str | None = None


class SearchResponse(BaseModel):
    """Normalized response returned by the load-balancing service."""

    provider: SearchProviderId
    auth_mode: str
    sources: list[SearchSource] = Field(default_factory=list)
    answer: str | None = None
    request_id: str | None = None
    failures: list[str] = Field(default_factory=list)


class ProviderStatus(BaseModel):
    """Readiness row shared by ``search list`` and the TUI's ``/search`` view."""

    id: SearchProviderId
    label: str
    enabled: bool
    available: bool
    access: str
    detail: str


class WebSearchSettings(BaseModel):
    """Validated view of the loose ``values.web_search`` YAML mapping."""

    enabled: bool = True
    strategy: SearchStrategy = "round_robin"
    providers: list[SearchProviderId] = Field(default_factory=lambda: ["duckduckgo", "tavily"])
    timeout_seconds: float = 20.0
    searxng_endpoint: str = ""


DEFAULT_WEB_SEARCH_CONFIG: dict[str, object] = {
    "enabled": True,
    "strategy": "round_robin",
    # Two credential-free transports make load balancing useful on first run.
    # Tavily's official keyless mode is rate-limited; DDG remains the durable
    # no-account fallback when that budget is exhausted.
    "providers": ["duckduckgo", "tavily"],
    "timeout_seconds": 20.0,
    "searxng_endpoint": "",
}
