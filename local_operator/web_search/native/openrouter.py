"""OpenRouter native search: the ``openrouter:web_search`` server tool on Responses.

OpenRouter runs the search itself (engine ``exa`` here) and hands the results
to whichever model the request names, so one cheap host model is enough. The
request rides OpenRouter's Responses endpoint, whose payload is the same
``output[]`` / ``url_citation`` shape :mod:`.responses_wire` already parses;
the call item is spelled ``openrouter:web_search`` (live capture 2026-11-24,
``fixtures/native/openrouter_responses_live.json``).

PAID CLASS, AFTER RADIENT. An OpenRouter key spends prepaid credits per call,
so it is a model-key leg (``providers.NATIVE_MODEL_KEY_BAND``). Low marginal
value (design §4): the default engine is the same Exa index the free band
already queries keylessly; it is here so a user whose only search-capable
credential is OpenRouter still has a native leg.

COST IS BILLED, NOT ESTIMATED: OpenRouter returns the charged amount at
``usage.cost`` (engine fee plus tokens, measured $0.0072 for one Exa search on
``gpt-4.1-nano``), so the response records that figure as a measurement. A
response without it falls back to the list-price estimate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from local_operator.i18n.keys import web_search as web_search_copy
from local_operator.web_search.models import (
    SearchCost,
    SearchResponse,
    WebSearchSettings,
)
from local_operator.web_search.native import credentials, pricing, responses_wire
from local_operator.web_search.native.base import (
    NativeClass,
    NativeSearchVendor,
    leg_mode,
)

LABEL = "OpenRouter"
NAMESPACE = "openrouter"
ENV_KEY = "OPENROUTER_API_KEY"
RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
#: A cheap tool-capable host: the search result text dominates the tokens, and
#: OpenRouter runs the search server-side whatever model answers.
HOST_MODEL = "openai/gpt-4.1-nano"
#: Exa is pinned (not ``auto``): ``auto`` may switch to the host's NATIVE search,
#: whose fee is the host vendor's ($10/1k for OpenAI), not the $0.007 used to
#: order this leg.
ENGINE = "exa"
MAX_OUTPUT_TOKENS = 512
#: Exa engine fee on OpenRouter: $0.007 per request up to 10 results
#: (openrouter.ai/docs server-tools/web-search, fetched 2026-11-24). Used to
#: order the leg and as the fallback estimate when ``usage.cost`` is missing.
FEE_USD = 0.007
BASIS_BILLED = "billed by OpenRouter"

KEY_MODE = "api-key"


def _modes(config_dir: Path | None, _settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    if credentials.has_api_key(NAMESPACE, config_dir, ENV_KEY):
        return [(KEY_MODE, "key")]
    return []


def _billed_cost(payload: dict[str, Any]) -> SearchCost | None:
    usage = payload.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    # bool is an int subclass; a ``true`` here is not a price.
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        return SearchCost(usd=round(float(cost), 6), basis=BASIS_BILLED, priced_from_usage=True)
    return None


async def search(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    from local_operator.i18n.text import tr

    leg_mode(LABEL, _modes(config_dir, settings))
    key = await credentials.api_key(NAMESPACE, config_dir, ENV_KEY)
    if not key:
        raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
    payload, request_id = await responses_wire.post_responses(
        client,
        url=RESPONSES_URL,
        bearer=key,
        body={
            "model": HOST_MODEL,
            "input": responses_wire.search_prompt(query),
            "tools": [
                {
                    "type": "openrouter:web_search",
                    # One search per call: the chain wants one result list, and
                    # every extra search is another engine fee.
                    "parameters": {"engine": ENGINE, "max_results": limit, "max_uses": 1},
                }
            ],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        label=LABEL,
    )
    sources, answer = responses_wire.parse_responses_search(payload, limit, LABEL)
    usage = responses_wire.responses_usage(payload)
    cost = _billed_cost(payload) or pricing.native_cost(
        fee_usd=FEE_USD, usage=usage, hosting="openai", model="gpt-4.1-nano", cache="inside"
    )
    return SearchResponse(
        provider="openrouter",
        auth_mode=KEY_MODE,
        sources=sources,
        answer=answer,
        # OpenRouter sends no ``x-request-id``; the response id is its trace handle.
        request_id=request_id or (str(payload.get("id") or "").strip() or None),
        usage=usage,
        cost=cost,
    )


VENDOR = NativeSearchVendor(
    id="openrouter",
    label=LABEL,
    access=web_search_copy.native_access_key(),
    detail=web_search_copy.native_detail_key(vendor=LABEL),
    modes=_modes,
    search=search,
    fee_usd=FEE_USD,
    credential_keys=(ENV_KEY,),
)
