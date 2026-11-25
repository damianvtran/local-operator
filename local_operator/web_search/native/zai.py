"""Z.AI native search: the GLM Coding Plan's ``webSearchPrime`` remote MCP tool.

Z.AI ships search to Coding Plan subscribers as a remote MCP server
(``https://api.z.ai/api/mcp/web_search_prime/mcp``, Bearer auth; docs.z.ai
"Web Search MCP Server", fetched 2026-11-24). It is a search tool, not a model
turn: one JSON-RPC ``tools/call`` returns result rows, so there are no tokens.

PLAN CLASS, BEFORE RADIENT. Both credentials lop stores under ``zai`` are
Coding Plan credentials: the pasted key from the plan page (``lop login zai``,
an ``api_key`` row or ``ZAI_API_KEY``) and the browser sign-in (``zai-oauth``,
an ``oauth`` row whose ``access`` is a MINTED plan key, ``providers/oauth/
zai.py``). The search draws on the plan's MCP quota and moves no cash, so the
one leg is ``plan`` whichever credential serves it. The OAuth grant is
preferred (it is the plan's own identity); the key is the fallback.

TRANSPORT reuses ``providers._mcp_call`` / ``_mcp_text`` (the Exa/Parallel MCP
client: both framings, error envelopes raised). Z.AI reports a refused call as
HTTP 200 with a ``{"code", "msg", "success": false}`` body instead of a
JSON-RPC envelope (reproduced 2026-11-24 without a credential: code 1001), which
``_mcp_payload`` would pass through as an empty result; it is raised here.

NO-CRED BUILD: no Z.AI credential was stored on the probe machine, so the
``webSearchPrime`` arguments and result rows follow the documented REST twin
(``/paas/v4/web_search``: ``search_query`` in; ``title``/``link``/``content``/
``publish_date`` out). The parser accepts the rows as a JSON list, a doubly
encoded JSON string, or ``{"search_result": [...]}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from local_operator.i18n.keys import web_search as web_search_copy
from local_operator.web_search.models import (
    SearchResponse,
    SearchSource,
    SearchUsage,
    WebSearchSettings,
)
from local_operator.web_search.native import credentials, pricing
from local_operator.web_search.native.base import (
    NativeClass,
    NativeSearchVendor,
    leg_mode,
)

LABEL = "Z.AI"
NAMESPACE = "zai"
ENV_KEY = "ZAI_API_KEY"
MCP_URL = "https://api.z.ai/api/mcp/web_search_prime/mcp"
TOOL = "webSearchPrime"
PLAN_NAME = "GLM Coding Plan"
PLAN_MODE = "plan"


def _modes(config_dir: Path | None, _settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    if credentials.has_oauth_row(NAMESPACE) or credentials.has_api_key(
        NAMESPACE, config_dir, ENV_KEY
    ):
        return [(PLAN_MODE, "plan")]
    return []


async def _bearer(config_dir: Path | None) -> str:
    access = await credentials.oauth_access(NAMESPACE)
    if access is not None:
        return access.access_token
    return await credentials.api_key(NAMESPACE, config_dir, ENV_KEY)


def _rows(text: str) -> list[Any]:
    try:
        decoded: Any = json.loads(text)
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
    except ValueError as error:
        raise RuntimeError(f"{LABEL} search returned an unparseable result") from error
    if isinstance(decoded, dict):
        decoded = decoded.get("search_result") or decoded.get("results") or []
    return decoded if isinstance(decoded, list) else []


def parse_zai_search(text: str, limit: int) -> list[SearchSource]:
    """Map the MCP text payload onto sources; raises when there are none."""
    from local_operator.web_search.providers import _source

    sources: list[SearchSource] = []
    for row in _rows(text):
        if not isinstance(row, dict):
            continue
        source = _source(
            title=row.get("title"),
            url=row.get("link") or row.get("url"),
            snippet=row.get("content"),
            published_date=row.get("publish_date") or None,
        )
        if source is None:
            continue
        sources.append(source)
        if len(sources) >= limit:
            break
    if not sources:
        raise RuntimeError(f"{LABEL} search returned no results")
    return sources


def _refusal(payload: dict[str, Any]) -> str | None:
    """Z.AI's non-JSON-RPC refusal body (``success: false``), as a message."""
    if payload.get("success") is False or ("code" in payload and "result" not in payload):
        return f"{payload.get('code')}: {str(payload.get('msg') or 'request refused')[:300]}"
    return None


async def search(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    from local_operator.i18n.text import tr
    from local_operator.web_search.providers import _mcp_call, _mcp_text

    leg_mode(LABEL, _modes(config_dir, settings))
    bearer = await _bearer(config_dir)
    if not bearer:
        raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
    payload = await _mcp_call(
        client,
        MCP_URL,
        TOOL,
        # ``content_size: medium`` is the documented default summary length;
        # ``high`` returns page crops the chain would only truncate.
        {"search_query": query, "content_size": "medium"},
        label=LABEL,
        headers={"Authorization": f"Bearer {bearer}"},
    )
    refusal = _refusal(payload)
    if refusal is not None:
        raise RuntimeError(f"{LABEL} search refused: {refusal}")
    sources = parse_zai_search(_mcp_text(payload), limit)
    return SearchResponse(
        provider="zai",
        auth_mode=PLAN_MODE,
        sources=sources,
        usage=SearchUsage(server_searches=1),
        cost=pricing.plan_cost(PLAN_NAME),
    )


VENDOR = NativeSearchVendor(
    id="zai",
    label=LABEL,
    access=web_search_copy.native_access_plan(),
    detail=web_search_copy.native_detail_plan(vendor=LABEL, plan=PLAN_NAME),
    modes=_modes,
    search=search,
    fee_usd=None,
    credential_keys=(ENV_KEY,),
)
