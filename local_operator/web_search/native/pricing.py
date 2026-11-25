"""What one native search cost: the vendor's per-search fee plus the host turn's tokens.

A native search is a MODEL TURN with a server tool attached, so the vendor bills
two things: a flat fee per server-side search (published per vendor) and the
host model's tokens, which are dominated by the retrieved page text the search
appends to the prompt. Pricing only one of them understates the spend, so both
are estimated here from list prices and the usage the vendor reported.

Token rates come from the shipped model registry (``model.registry.
get_model_info``), the same table chat-turn costs read, so a price change lands
in one place. A host model the registry does not price leaves the token half
UNPRICED: the fee is still charged and the figure is a floor, which
``priced_from_usage=False`` and the basis both say (the ``cost.py`` rule: never
present a guess as a measurement).

Plan legs (a ChatGPT subscription) move no cash: ``usd=None`` and a basis that
names the plan, so the ledger counts the search without inventing a charge.
"""

from __future__ import annotations

from typing import Literal

from local_operator.web_search.models import SearchCost, SearchUsage

#: How a wire counts cache reads. ``inside``: Responses-shaped, ``input_tokens``
#: already INCLUDES the cached part. ``apart``: Anthropic-shaped, cache reads are
#: a separate bucket added on top (``cost._deepseek_cost`` explains the defect
#: that mixing these up caused).
CacheSemantics = Literal["inside", "apart"]

BASIS_NATIVE = "per-search fee + tokens at list price (estimate)"


def plan_cost(plan: str) -> SearchCost:
    """A subscription-funded search: counted, never priced as cash."""
    return SearchCost(usd=None, basis=f"included in {plan}", priced_from_usage=False)


def _token_rates(hosting: str, model: str) -> tuple[float, float, float] | None:
    """(input, output, cache-read) USD per token, or ``None`` when unpriced."""
    try:
        from local_operator.model.registry import get_model_info, unknown_model_info

        info = get_model_info(hosting, model)
    except Exception:  # noqa: BLE001 - an unknown hosting is simply "unpriced"
        return None
    if info is unknown_model_info or not (info.input_price or info.output_price):
        return None
    cache = info.cache_reads_price if info.cache_reads_price else info.input_price
    return info.input_price / 1e6, info.output_price / 1e6, cache / 1e6


def native_cost(
    *,
    fee_usd: float,
    usage: SearchUsage,
    hosting: str,
    model: str,
    cache: CacheSemantics,
    basis_note: str = "",
) -> SearchCost:
    """Fee x searches + host-model tokens, as an estimate from list prices.

    The fee is charged per server-side search the vendor reported, and at least
    once: a successful native result means the search ran even when the usage
    block omits the count. ``basis_note`` qualifies the basis (e.g. "API
    equivalent; drawn from Claude extra usage" for the subscription leg).
    """
    searches = max(1, usage.server_searches or 0)
    fee = fee_usd * searches
    suffix = f"; {basis_note}" if basis_note else ""
    rates = _token_rates(hosting, model)
    tokens_seen = bool(usage.input_tokens or usage.output_tokens)
    if rates is None or not tokens_seen:
        return SearchCost(
            usd=round(fee, 6),
            basis=f"per-search fee; tokens unpriced (floor){suffix}",
            priced_from_usage=False,
        )
    input_rate, output_rate, cache_rate = rates
    # Clamped at zero, the Perplexity NIT-1 lesson: a negative or inconsistent
    # count must never price a search below its own fee.
    input_tokens = max(0, usage.input_tokens or 0)
    cached = max(0, usage.cache_read_tokens or 0)
    output_tokens = max(0, usage.output_tokens or 0)
    if cache == "inside":
        cached = min(cached, input_tokens)
        fresh = input_tokens - cached
    else:
        fresh = input_tokens
    usd = fee + fresh * input_rate + cached * cache_rate + output_tokens * output_rate
    return SearchCost(usd=round(usd, 6), basis=f"{BASIS_NATIVE}{suffix}", priced_from_usage=True)
