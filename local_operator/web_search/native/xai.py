"""xAI native search: the Responses ``web_search`` tool on ``api.x.ai``.

xAI removed Live Search; ``web_search`` on ``/v1/responses`` is the documented
replacement (docs.x.ai "Web Search", fetched 2026-11-24). Both credential
classes the imagegen xAI rung already runs on this host are accepted here: an
``api_key`` row (or ``XAI_API_KEY``) and the Grok sign-in grant stored under
``xai`` by ``xai-oauth``. Both are ordered as PAID legs after Radient: the key
is metered cash at $5 per 1,000 searches plus tokens, and whether xAI meters
the grant against a plan or bills it is not published, so the conservative
placement is the one that never spends it ahead of Radient. The grant's cost
is reported as an API-equivalent estimate for the same reason.

The key wins over the grant (the imagegen ``xai_call_credential`` order): the
key path is the reliable one, xAI tiers its OAuth surface.
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

LABEL = "xAI"
NAMESPACE = "xai"
ENV_KEY = "XAI_API_KEY"
RESPONSES_URL = "https://api.x.ai/v1/responses"
#: The cheapest catalogue Grok with tool support; page tokens dominate the cost.
HOST_MODEL = "grok-4.3"
MAX_OUTPUT_TOKENS = 512
#: Published ``web_search`` tool fee: $5 / 1,000 calls (docs.x.ai pricing).
FEE_USD = 0.005

KEY_MODE = "api-key"
SIGNIN_MODE = "oauth"
SIGNIN_NOTE = "API equivalent; Grok sign-in billing is unpublished"


def _modes(config_dir: Path | None, _settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    modes: list[tuple[str, NativeClass]] = []
    if credentials.has_api_key(NAMESPACE, config_dir, ENV_KEY):
        modes.append((KEY_MODE, "key"))
    if credentials.has_oauth_row(NAMESPACE):
        modes.append((SIGNIN_MODE, "key"))
    return modes


async def search(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    from local_operator.i18n.text import tr

    mode = leg_mode(LABEL, _modes(config_dir, settings))
    if mode == SIGNIN_MODE:
        access = await credentials.oauth_access(NAMESPACE)
        bearer = access.access_token if access is not None else ""
    else:
        bearer = await credentials.api_key(NAMESPACE, config_dir, ENV_KEY)
    if not bearer:
        raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
    payload, request_id = await responses_wire.post_responses(
        client,
        url=RESPONSES_URL,
        bearer=bearer,
        body={
            "model": HOST_MODEL,
            "input": [{"role": "user", "content": responses_wire.search_prompt(query)}],
            "tools": [{"type": "web_search"}],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
        },
        label=LABEL,
    )
    sources, answer = responses_wire.parse_responses_search(payload, limit, LABEL)
    usage = responses_wire.responses_usage(payload)
    return SearchResponse(
        provider="xai",
        auth_mode=mode,
        sources=sources,
        answer=answer,
        request_id=request_id,
        usage=usage,
        cost=pricing.native_cost(
            fee_usd=FEE_USD,
            usage=usage,
            hosting="xai",
            model=HOST_MODEL,
            cache="inside",
            basis_note=SIGNIN_NOTE if mode == SIGNIN_MODE else "",
        ),
    )


VENDOR = NativeSearchVendor(
    id="xai",
    label=LABEL,
    access=web_search_copy.native_access_key_signin(),
    detail=web_search_copy.native_detail_key(vendor=LABEL),
    modes=_modes,
    search=search,
    fee_usd=FEE_USD,
    credential_keys=(ENV_KEY,),
)
