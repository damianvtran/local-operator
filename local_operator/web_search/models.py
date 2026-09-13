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
    "deepseek",
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
    "deepseek",
    "perplexity",
    "brave",
    "exa",
    "serpapi",
    "searxng",
)


class SearchSource(BaseModel):
    """One normalized result from any search provider.

    ``relevance`` is populated only by providers that return a per-page
    judgement (currently DeepSeek's evidence pass). It is a 0-100 hint for
    ordering WHICH source to fetch next, never a filter: a provider that does
    not supply it leaves it ``None`` and the renderer omits it.
    """

    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    snippet: str | None = None
    published_date: str | None = None
    relevance: int | None = None


class SearchUsage(BaseModel):
    """Provider-reported usage for one search, when the provider reports any.

    Every field is optional because the transports differ: a scraped HTML search
    reports nothing, a token-billed one reports tokens, and a credit-billed one
    reports credits. ``keyless`` is how a keyed provider's free tier is
    distinguished from its paid path, since both return normally.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    credits: float | None = None
    server_searches: int | None = None
    keyless: bool = False

    def merge(self, other: "SearchUsage | None") -> "SearchUsage":
        """Fold another leg's usage into this one (a search plus its evidence pass)."""
        if other is None:
            return self
        return SearchUsage(
            input_tokens=(self.input_tokens or 0) + (other.input_tokens or 0) or None,
            output_tokens=(self.output_tokens or 0) + (other.output_tokens or 0) or None,
            cache_read_tokens=(self.cache_read_tokens or 0) + (other.cache_read_tokens or 0)
            or None,
            credits=(self.credits or 0) + (other.credits or 0) or None,
            server_searches=(self.server_searches or 0) + (other.server_searches or 0) or None,
            keyless=self.keyless and other.keyless,
        )


class SearchCost(BaseModel):
    """What one search cost, and how that number was arrived at.

    ``usd=None`` means "no published rate", which is not the same as ``0.0``
    ("free, and known to be"); ``basis`` states the provenance so a list-price
    estimate is never mistaken for a bill.
    """

    usd: float | None = None
    basis: str = ""
    priced_from_usage: bool = False


class SearchResponse(BaseModel):
    """Normalized response returned by the load-balancing service."""

    provider: SearchProviderId
    auth_mode: str
    sources: list[SearchSource] = Field(default_factory=list)
    answer: str | None = None
    request_id: str | None = None
    failures: list[str] = Field(default_factory=list)
    usage: SearchUsage | None = None
    cost: SearchCost | None = None
    #: Handle to the page context this search captured, when the provider's
    #: payload can be replayed to read those pages without fetching them (see
    #: :mod:`local_operator.web_search.pages`). ``None`` for providers that only
    #: return links.
    page_context_id: str | None = None


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
    #: Run the DeepSeek per-page evidence pass after a native search, so every
    #: source carries a verbatim quote and a relevance score. Off by default:
    #: it is a second model turn (~4-11s measured) on top of the search.
    deepseek_evidence: bool = False
    #: Offer the ``web_read`` tool, which answers questions from pages a previous
    #: search already retrieved instead of fetching them again. On by default
    #: because it costs nothing until it is used, and it degrades to an explicit
    #: "no pages captured, use web_fetch" rather than a silent fetch.
    read_enabled: bool = True


DEFAULT_WEB_SEARCH_CONFIG: dict[str, object] = {
    "enabled": True,
    "strategy": "round_robin",
    # Two credential-free transports make load balancing useful on first run.
    # Tavily's official keyless mode is rate-limited; DDG remains the durable
    # no-account fallback when that budget is exhausted.
    "providers": ["duckduckgo", "tavily"],
    "timeout_seconds": 20.0,
    "searxng_endpoint": "",
    "deepseek_evidence": False,
    "read_enabled": True,
}
