"""What a web search costs, and the ledger that remembers it.

Search spend is real money that the model-token accounting never saw: DeepSeek's
native search bills a whole model turn, Tavily bills credits, and the paid
engines bill per query. A session's displayed cost that omits them understates
what the operator actually spent, and can understate it by a lot on a
search-heavy session -- so this module owns both halves of the problem:

* :func:`estimate_search_cost` prices ONE search from whatever the provider
  reported (tokens for DeepSeek/Perplexity, a published per-query rate for the
  rest, nothing for the free transports); and
* :class:`SearchSpendLedger` accumulates per-session and per-provider totals so
  ``/session`` and ``/analytics`` can show search spend as its own line rather
  than folding it into model cost.

Two conventions matter for honesty:

* ``None`` means "no published rate for this provider", which is NOT ``0.0``
  ("this search was free, and we know it"). The free transports of a keyed
  provider -- Tavily keyless, Perplexity anonymous, DuckDuckGo, SearXNG -- are
  genuinely ``0.0`` with a basis that says which free tier served them.
* Every cost carries a :class:`SearchCostBasis`. A token-derived figure is an
  ESTIMATE from list prices: measured against the DeepSeek account balance, the
  ledger moved more than the token arithmetic predicts, and that gap is not
  resolved (see the note on the provider's balance probe). Labelling the number
  is how the display can stay honest about that.

Prices are the vendors' published list rates as of 2026-09 and are inputs to an
estimate, not a bill.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from local_operator.web_search.models import SearchCost, SearchUsage

__all__ = [
    "DEEPSEEK_PEAK_INPUT_USD_PER_TOKEN",
    "DEEPSEEK_PEAK_OUTPUT_USD_PER_TOKEN",
    "DEEPSEEK_PEAK_CACHE_HIT_USD_PER_TOKEN",
    "PROVIDER_PRICE_NOTES",
    "PROVIDER_USD_PER_SEARCH",
    "SEARCH_SPEND",
    "SearchSpendLedger",
    "deepseek_is_peak_hour",
    "estimate_search_cost",
]

#: deepseek-flash list price during PEAK hours, USD per token (i.e. per-1M rates
#: divided by 1e6). Peak is 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday;
#: every other hour is half price. Source: api-docs.deepseek.com/quick_start/pricing.
DEEPSEEK_PEAK_INPUT_USD_PER_TOKEN = 0.30 / 1_000_000
DEEPSEEK_PEAK_OUTPUT_USD_PER_TOKEN = 1.20 / 1_000_000
DEEPSEEK_PEAK_CACHE_HIT_USD_PER_TOKEN = 0.006 / 1_000_000

#: Published per-query rate, USD, for providers that bill per request rather than
#: per token. ``None`` for the free transports and for providers whose real cost
#: depends on their response usage.
PROVIDER_USD_PER_SEARCH: dict[str, float | None] = {
    "duckduckgo": 0.0,  # credential-free HTML search
    "tavily": 0.008,  # pay-as-you-go, 1 credit per basic search
    "deepseek": None,  # token-priced: see estimate_search_cost
    "perplexity": None,  # token-priced on Sonar; free when anonymous
    "brave": 0.004,  # $4 per 1,000 queries (Brave's own pricing page, 2026-09)
    "exa": 0.005,  # ≈$5 per 1,000 searches (list, 2026-09)
    "serpapi": 0.015,  # ≈$15 per 1,000 at the 5,000-search plan
    "searxng": 0.0,  # self-hosted
}

#: Why a provider's number looks the way it does. Rendered next to the estimate
#: so an operator can tell a list price from a measurement.
PROVIDER_PRICE_NOTES: dict[str, str] = {
    "duckduckgo": "credential-free; costs no money",
    "tavily": (
        "$0.008 per credit (1 per basic search); keyless mode is free within its monthly tier"
    ),
    "deepseek": "one model turn billed as tokens at deepseek-flash list price",
    "perplexity": "anonymous mode is free; Sonar bills tokens plus a per-request fee",
    "brave": "$4 per 1,000 queries, subscription",
    "exa": "≈$5 per 1,000 searches, subscription",
    "serpapi": "subscription plans; ≈$15 per 1,000 searches",
    "searxng": "self-hosted; costs no money",
}

#: Basis strings, kept as constants so the display and the tests agree.
BASIS_FREE = "free"
BASIS_PER_SEARCH = "published per-search rate"
BASIS_TOKENS = "tokens at list price (estimate)"
BASIS_UNKNOWN = "no published rate"


def deepseek_is_peak_hour(moment: datetime | None = None) -> bool:
    """Whether DeepSeek charges full price at ``moment`` (UTC, Mon-Fri).

    Peak is 01:00-04:00 and 06:00-10:00 UTC on weekdays; the rates halve outside
    it. The window is evaluated in UTC because that is how DeepSeek states it --
    pricing in local time would silently mis-price every off-hours session.
    """
    when = moment or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)
    if when.weekday() >= 5:  # Saturday/Sunday are off-peak all day
        return False
    hour = when.hour
    return 1 <= hour < 4 or 6 <= hour < 10


def _deepseek_cost(usage: SearchUsage, moment: datetime | None) -> SearchCost:
    peak = deepseek_is_peak_hour(moment)
    scale = 1.0 if peak else 0.5
    input_tokens = usage.input_tokens or 0
    cache_reads = usage.cache_read_tokens or 0
    # Cache reads are billed at their own (much cheaper) rate, so they must come
    # out of the input count rather than being charged twice.
    miss_tokens = max(input_tokens - cache_reads, 0)
    usd = (
        miss_tokens * DEEPSEEK_PEAK_INPUT_USD_PER_TOKEN
        + cache_reads * DEEPSEEK_PEAK_CACHE_HIT_USD_PER_TOKEN
        + (usage.output_tokens or 0) * DEEPSEEK_PEAK_OUTPUT_USD_PER_TOKEN
    ) * scale
    window = "peak" if peak else "off-peak"
    return SearchCost(
        usd=round(usd, 6),
        basis=f"{BASIS_TOKENS}, {window}",
        priced_from_usage=True,
    )


def estimate_search_cost(
    provider_id: str,
    usage: SearchUsage | None = None,
    *,
    moment: datetime | None = None,
) -> SearchCost:
    """Price one search on ``provider_id`` from its published rate or its usage.

    ``usage`` wins wherever a provider reports tokens, because a billed token
    count beats any per-query approximation. Providers whose transport has no
    usage to report fall back to the published per-query rate, and a provider
    with neither is reported as unpriced (``usd=None``) rather than as free.
    """
    if provider_id == "deepseek":
        if usage is not None and (usage.input_tokens or usage.output_tokens):
            return _deepseek_cost(usage, moment)
        # No usage captured: fall back to the measured median for one DeepSeek
        # search so the ledger is not silently empty. Labelled as a fallback.
        return SearchCost(
            usd=None,
            basis=f"{BASIS_TOKENS}; usage not captured",
            priced_from_usage=False,
        )

    if provider_id == "perplexity":
        if usage is not None and (usage.input_tokens or usage.output_tokens):
            # Sonar rates are not published in one table we can pin; the request
            # fee alone is 5-14 per 1,000 depending on context size.
            return SearchCost(
                usd=0.005,
                basis="anonymous is free; Sonar ≈$5/1,000 requests plus tokens",
                priced_from_usage=False,
            )
        return SearchCost(usd=0.0, basis=f"{BASIS_FREE} (anonymous)", priced_from_usage=False)

    if provider_id == "tavily" and usage is not None and usage.keyless:
        return SearchCost(usd=0.0, basis=f"{BASIS_FREE} (keyless tier)", priced_from_usage=False)

    rate = PROVIDER_USD_PER_SEARCH.get(provider_id)
    if rate is None:
        return SearchCost(usd=None, basis=BASIS_UNKNOWN, priced_from_usage=False)
    if rate == 0.0:
        return SearchCost(usd=0.0, basis=BASIS_FREE, priced_from_usage=False)
    credits = usage.credits if usage is not None and usage.credits else 1
    return SearchCost(
        usd=round(rate * credits, 6),
        basis=f"{BASIS_PER_SEARCH} ({PROVIDER_PRICE_NOTES.get(provider_id, '')})".strip(),
        priced_from_usage=False,
    )


@dataclass
class ProviderSearchSpend:
    """One provider's contribution to a search-spend total."""

    provider: str
    searches: int = 0
    usd: float = 0.0
    #: Searches whose price is not known. Counted separately so a total that
    #: covers only some of the work says so instead of reading as complete.
    unpriced_searches: int = 0
    bases: set[str] = field(default_factory=set)

    def add(self, usd: float | None, basis: str) -> None:
        self.searches += 1
        if usd is None:
            self.unpriced_searches += 1
        else:
            self.usd += usd
        if basis:
            self.bases.add(basis)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "searches": self.searches,
            "usd": round(self.usd, 6),
            "unpriced_searches": self.unpriced_searches,
            "basis": "; ".join(sorted(self.bases)),
        }


@dataclass
class SearchSpendTotals:
    """Search spend across one session, or across the whole process."""

    searches: int = 0
    usd: float = 0.0
    unpriced_searches: int = 0
    by_provider: dict[str, ProviderSearchSpend] = field(default_factory=dict)

    @property
    def priced_searches(self) -> int:
        return self.searches - self.unpriced_searches

    def as_dict(self) -> dict[str, Any]:
        return {
            "searches": self.searches,
            "usd": round(self.usd, 6),
            "unpriced_searches": self.unpriced_searches,
            "by_provider": [entry.as_dict() for entry in self.by_provider.values()],
        }


class SearchSpendLedger:
    """Process-wide, session-keyed search spend.

    Deliberately process-wide rather than passed down: the tool that spends the
    money and the panels that display it are constructed far apart (the tool has
    a ``ToolContext``; the panels have the app), and threading a ledger through
    the whole harness for a display total would touch every layer in between.
    ``session_id`` is what keeps it honest -- a total is always asked for BY
    session, and ``overall()`` exists for the cross-session view.

    Not persisted here: a resumed session re-reads its spend from the transcript
    (the tool result carries the cost) exactly as it recovers model cost. This
    ledger is the live-session view.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SearchSpendTotals] = {}
        self._lock = threading.Lock()

    def record(
        self,
        session_id: str,
        provider: str,
        cost: SearchCost | None,
        *,
        searches: int = 1,
    ) -> ProviderSearchSpend:
        key = session_id or "unattributed"
        usd = cost.usd if cost is not None else None
        basis = cost.basis if cost is not None else ""
        with self._lock:
            totals = self._sessions.setdefault(key, SearchSpendTotals())
            entry = totals.by_provider.setdefault(provider, ProviderSearchSpend(provider=provider))
            for _ in range(max(searches, 1)):
                totals.searches += 1
                if usd is None:
                    totals.unpriced_searches += 1
                else:
                    totals.usd += usd
                entry.add(usd, basis)
            return entry

    def session(self, session_id: str) -> SearchSpendTotals:
        with self._lock:
            return self._sessions.get(session_id or "unattributed", SearchSpendTotals())

    def overall(self) -> SearchSpendTotals:
        with self._lock:
            merged = SearchSpendTotals()
            for totals in self._sessions.values():
                merged.searches += totals.searches
                merged.usd += totals.usd
                merged.unpriced_searches += totals.unpriced_searches
                for provider, entry in totals.by_provider.items():
                    target = merged.by_provider.setdefault(
                        provider, ProviderSearchSpend(provider=provider)
                    )
                    target.searches += entry.searches
                    target.usd += entry.usd
                    target.unpriced_searches += entry.unpriced_searches
                    target.bases |= entry.bases
            return merged

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id or "unattributed", None)

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()


#: The process-wide ledger the tool writes and the panels read.
SEARCH_SPEND = SearchSpendLedger()
