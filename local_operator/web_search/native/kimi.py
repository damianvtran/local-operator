"""Kimi native search: Moonshot's standalone search API, plus a probe-gated Kimi Code plan leg.

Unlike every other vendor in this family, Kimi's search is NOT a model turn: it
is a plain search endpoint that returns ``search_results[]{title,url,snippet,
date}``, so there are no tokens to price and no prose answer.

TWO LEGS, TWO CREDENTIAL CLASSES, NEVER MIXED:

* ``api-key`` -- a Moonshot platform key (``api_key`` rows under ``kimi``, then
  ``KIMI_API_KEY``). ``POST {host}/v1/tools/search`` ("Web Search Basic"),
  $0.002 per successful NON-EMPTY call; empty or failed calls are free
  (platform.kimi.ai pricing, fetched 2026-11-24). Model-key class: after Radient.
* ``plan`` -- the Kimi Code sign-in (``lop login kimi`` / ``kimi-wechat``, an
  OAuth grant under ``kimi``). ``POST https://api.kimi.com/coding/v1/search``,
  the route kimi-cli's ``SearchWeb`` tool calls, with the ``X-Msh-*`` device
  headers every Kimi OAuth call must carry (``providers/oauth/kimi.py``).
  Plan quota, no cash: the NATIVE band, before Radient -- once probed.

HOSTS (probed 2026-11-24 without a credential): ``/v1/tools/search`` answers
401 on BOTH ``api.moonshot.cn`` and ``api.moonshot.ai`` while an unknown path
answers 403, so the route exists on both platforms. A key belongs to exactly
one of them (the registry row's ``base_url`` note: a mainland key 401s on the
other platform), so the key leg tries the registry's mainland host first and
retries the international host ONLY on 401 -- a refused call is free, and no
other status is retried.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from local_operator.i18n.keys import web_search as web_search_copy
from local_operator.web_search.models import (
    SearchCost,
    SearchResponse,
    SearchSource,
    SearchUsage,
    WebSearchSettings,
)
from local_operator.web_search.native import credentials, pricing
from local_operator.web_search.native.base import (
    NativeClass,
    NativeSearchVendor,
    error_text,
    leg_mode,
)

LABEL = "Kimi"
NAMESPACE = "kimi"
ENV_KEY = "KIMI_API_KEY"
#: Mainland first: it is the registry row's ``base_url`` for ``KIMI_API_KEY``.
KEY_SEARCH_URLS = (
    "https://api.moonshot.cn/v1/tools/search",
    "https://api.moonshot.ai/v1/tools/search",
)
PLAN_SEARCH_URL = "https://api.kimi.com/coding/v1/search"
#: Server-side budget kimi-cli sends; the service caps it at 30 s itself.
TIMEOUT_SECONDS = 30
#: $0.002 per non-empty search (Web Search Basic).
FEE_USD = 0.002
PLAN_NAME = "Kimi Code plan"

#: Whether the Kimi Code plan leg is registered. FALSE until a live probe
#: passes: on 2026-11-24 the only stored grant was dead (``grant_dead_at`` set,
#: access token expired), so the route was confirmed only as "exists, wants
#: auth" (401 vs 404 for an unknown path). Its wire shape comes from kimi-cli,
#: not formal docs, and a leg that has never answered must not sit ahead of
#: Radient. Flip once a probe returns results; transport and tests are here.
PLAN_LEG_PROBED = False

PLAN_MODE = "plan"
KEY_MODE = "api-key"


def _modes(config_dir: Path | None, _settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    modes: list[tuple[str, NativeClass]] = []
    if PLAN_LEG_PROBED and credentials.has_oauth_row(NAMESPACE):
        modes.append((PLAN_MODE, "plan"))
    if credentials.has_api_key(NAMESPACE, config_dir, ENV_KEY):
        modes.append((KEY_MODE, "key"))
    return modes


def parse_kimi_search(payload: object, limit: int) -> list[SearchSource]:
    """Map ``search_results[]`` onto sources; raises when there are none."""
    from local_operator.web_search.providers import _source

    if not isinstance(payload, dict):
        raise RuntimeError(f"{LABEL} returned a non-object response")
    sources: list[SearchSource] = []
    for item in payload.get("search_results") or []:
        if not isinstance(item, dict):
            continue
        source = _source(
            title=item.get("title"),
            url=item.get("url"),
            snippet=item.get("snippet"),
            published_date=item.get("date") or None,
        )
        if source is None:
            continue
        sources.append(source)
        if len(sources) >= limit:
            break
    if not sources:
        raise RuntimeError(f"{LABEL} search returned no results")
    return sources


async def _post(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], query: str, limit: int
) -> httpx.Response:
    return await client.post(
        url,
        headers={"Content-Type": "application/json", **headers},
        json={
            "text_query": query,
            "limit": limit,
            # Page crawling multiplies the payload; the chain wants a result list.
            "enable_page_crawling": False,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
    )


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as error:
        raise RuntimeError(f"{LABEL} returned a non-JSON response") from error


async def search(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    from local_operator.i18n.text import tr

    mode = leg_mode(LABEL, _modes(config_dir, settings))
    if mode == PLAN_MODE:
        from local_operator.providers.oauth.kimi import kimi_common_headers

        access = await credentials.oauth_access(NAMESPACE)
        if access is None:
            raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
        response = await _post(
            client,
            PLAN_SEARCH_URL,
            {"Authorization": f"Bearer {access.access_token}", **kimi_common_headers()},
            query,
            limit,
        )
        cost = pricing.plan_cost(PLAN_NAME)
    else:
        key = await credentials.api_key(NAMESPACE, config_dir, ENV_KEY)
        if not key:
            raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
        response = await _post(
            client, KEY_SEARCH_URLS[0], {"Authorization": f"Bearer {key}"}, query, limit
        )
        if response.status_code == 401:
            response = await _post(
                client, KEY_SEARCH_URLS[1], {"Authorization": f"Bearer {key}"}, query, limit
            )
        # Charged only because a non-empty result is about to be returned: an
        # empty one raises in the parser below and costs nothing.
        cost = SearchCost(usd=FEE_USD, basis="per-search fee (list price)", priced_from_usage=False)
    if not response.is_success:
        raise error_text(LABEL, response)
    sources = parse_kimi_search(_json(response), limit)
    request_id = (
        str(
            response.headers.get("x-request-id") or response.headers.get("msh-request-id") or ""
        ).strip()
        or None
    )
    return SearchResponse(
        provider="kimi",
        auth_mode=mode,
        sources=sources,
        request_id=request_id,
        usage=SearchUsage(server_searches=1),
        cost=cost,
    )


VENDOR = NativeSearchVendor(
    id="kimi",
    label=LABEL,
    access=web_search_copy.native_access_key(),
    detail=web_search_copy.native_detail_search_api(vendor=LABEL),
    modes=_modes,
    search=search,
    fee_usd=FEE_USD,
    credential_keys=(ENV_KEY,),
    # ``login kimi`` is the Kimi Code sign-in; the search API bills the platform key.
    setup_command="`local-operator credential update KIMI_API_KEY`",
)
