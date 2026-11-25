"""Alibaba Cloud (Qwen) native search: the Responses ``web_search`` tool on DashScope intl.

Only the Responses route is usable: the OpenAI-compatible chat
``enable_search`` flag searches but returns NO sources (design §4), and a
sourceless answer is exactly what the chain refuses. Sources come from
``web_search_call.action.sources``; the billed count from
``tool_usage.web_search.num_requests`` (Alibaba: output items are not a billing
count), both read by ``responses_wire``.

Key class only: an ``api_key`` row under ``alibaba`` (``lop login alibaba``) or
``ALIBABA_CLOUD_API_KEY``. The Token Plan rows (``alibaba-token-plan*``) are a
different namespace and are NOT read: search on that plan is undocumented.
Singapore pricing, $10 per 1,000 searches plus tokens.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from local_operator.i18n.keys import web_search as web_search_copy
from local_operator.web_search.models import SearchResponse, WebSearchSettings
from local_operator.web_search.native import credentials, pricing, responses_wire
from local_operator.web_search.native.base import (
    NativeClass,
    NativeSearchVendor,
    leg_mode,
)

LABEL = "Alibaba Cloud"
NAMESPACE = "alibaba"
ENV_KEY = "ALIBABA_CLOUD_API_KEY"
#: The registry row's base URL (``providers/registry.py`` ``alibaba``) + ``/responses``.
RESPONSES_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses"
#: The cheapest model Alibaba lists for Responses ``web_search``. The shipped
#: model registry does not price it, so token cost is reported as a floor
#: (``pricing.native_cost``) rather than guessed.
HOST_MODEL = "qwen3.8-flash"
MAX_OUTPUT_TOKENS = 512
#: Singapore ``agent`` policy fee: $10 / 1,000 calls (alibabacloud web-search billing).
FEE_USD = 0.01
KEY_MODE = "api-key"


def _modes(config_dir: Path | None, _settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    if credentials.has_api_key(NAMESPACE, config_dir, ENV_KEY):
        return [(KEY_MODE, "key")]
    return []


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
            "tools": [{"type": "web_search"}],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        label=LABEL,
    )
    sources, answer = responses_wire.parse_responses_search(payload, limit, LABEL)
    usage = responses_wire.responses_usage(payload)
    return SearchResponse(
        provider="alibaba",
        auth_mode=KEY_MODE,
        sources=sources,
        answer=answer,
        request_id=request_id,
        usage=usage,
        cost=pricing.native_cost(
            fee_usd=FEE_USD, usage=usage, hosting="alibaba", model=HOST_MODEL, cache="inside"
        ),
    )


VENDOR = NativeSearchVendor(
    id="alibaba",
    label=LABEL,
    access=web_search_copy.native_access_key(),
    detail=web_search_copy.native_detail_key(vendor=LABEL),
    modes=_modes,
    search=search,
    fee_usd=FEE_USD,
    credential_keys=(ENV_KEY,),
)
