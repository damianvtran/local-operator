"""Google native search: Gemini ``generateContent`` with the ``google_search`` grounding tool.

A grounded Gemini turn returns its sources on ``candidates[0].groundingMetadata``:
``groundingChunks[].web{uri,title}`` lists the pages, ``groundingSupports[]``
ties answer segments to chunk indices (used here as snippets), and
``webSearchQueries[]`` lists the queries Google ran, which is what Google bills
per (ai.google.dev "Grounding with Google Search" and pricing, 2026-11-24).

PAID CLASS, AFTER RADIENT. An AI Studio key is a pay-as-you-go model key. Gemini
3+ includes 5,000 grounded queries a month free, then $14 per 1,000; the free
allowance is not visible from the client, so every search is priced as if it
were billed (an UPPER estimate, the cost note says so).

SOURCE URLS ARE REDIRECTS: Google returns ``vertexaisearch.cloud.google.com/
grounding-api-redirect/...`` links with the site's domain as the title (design
§8 risk 6). They resolve through ``web_fetch`` like any link; they are kept
as-is rather than resolved here, which would cost one request per source.

Credential: the ``google`` namespace (``lop login google``) and the
``GOOGLE_AI_STUDIO_API_KEY`` name the registry row declares, the same pair
``imagegen.availability.google_key`` reads. Vertex is deferred (design §4).
"""

from __future__ import annotations

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
from local_operator.web_search.native import credentials, pricing, responses_wire
from local_operator.web_search.native.base import (
    NativeClass,
    NativeSearchVendor,
    error_text,
    leg_mode,
)

LABEL = "Google"
NAMESPACE = "google"
ENV_KEY = "GOOGLE_AI_STUDIO_API_KEY"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
#: The cheapest Flash the shipped model registry prices, so the token half of
#: the estimate is priced rather than left as a floor. Grounded page text is
#: what the turn mostly consumes.
HOST_MODEL = "gemini-3.8-flash"
MAX_OUTPUT_TOKENS = 512
#: $14 per 1,000 grounded search queries after the monthly free allowance.
FEE_USD = 0.014
BASIS_NOTE = "monthly free allowance not visible; upper estimate"

KEY_MODE = "api-key"


def _modes(config_dir: Path | None, _settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    if credentials.has_api_key(NAMESPACE, config_dir, ENV_KEY):
        return [(KEY_MODE, "key")]
    return []


def _candidate(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        return candidates[0]
    return {}


def _answer(candidate: dict[str, Any]) -> str | None:
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    texts = [
        str(part.get("text") or "").strip()
        for part in parts or []
        if isinstance(part, dict) and not part.get("thought")
    ]
    answer = "\n\n".join(text for text in texts if text).strip()
    return answer or None


def parse_grounded_search(payload: object, limit: int) -> tuple[list[SearchSource], str | None]:
    """``(sources, answer)`` from one grounded ``generateContent`` payload; raises when sourceless.

    Snippets come from ``groundingSupports``: the answer segment a chunk backs is
    the only page-derived text Google returns, and it is labelled as the
    answer's wording by being the segment verbatim (never invented text).
    """
    from local_operator.web_search.providers import _source

    if not isinstance(payload, dict):
        raise RuntimeError(f"{LABEL} returned a non-object response")
    error = payload.get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else error
        raise RuntimeError(f"{LABEL} search failed: {str(message)[:500]}")
    candidate = _candidate(payload)
    metadata = candidate.get("groundingMetadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    chunks = [chunk for chunk in metadata.get("groundingChunks") or [] if isinstance(chunk, dict)]

    snippets: dict[int, str] = {}
    for support in metadata.get("groundingSupports") or []:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment")
        text = str(segment.get("text") or "").strip() if isinstance(segment, dict) else ""
        for index in support.get("groundingChunkIndices") or []:
            if isinstance(index, int) and text and index not in snippets:
                snippets[index] = text

    sources: list[SearchSource] = []
    seen: set[str] = set()
    for index, chunk in enumerate(chunks):
        web = chunk.get("web")
        if not isinstance(web, dict):
            continue
        url = str(web.get("uri") or "").strip()
        if not url or url in seen:
            continue
        source = _source(title=web.get("title"), url=url, snippet=snippets.get(index))
        if source is None:
            continue
        seen.add(url)
        sources.append(source)
        if len(sources) >= limit:
            break
    if not sources:
        raise RuntimeError(
            f"{LABEL} returned no grounding sources; the request may not have triggered search"
        )
    return sources, _answer(candidate)


def grounded_usage(payload: object) -> SearchUsage:
    """Tokens and queries as Gemini reported them.

    ``promptTokenCount`` INCLUDES ``cachedContentTokenCount`` (priced ``inside``).
    The grounding tool's own prompt (``toolUsePromptTokenCount``) is billed as
    input and thinking tokens (``thoughtsTokenCount``) as output, so both are
    folded in: leaving them out would understate a search's token half.
    """
    if not isinstance(payload, dict):
        return SearchUsage()
    metadata = _candidate(payload).get("groundingMetadata")
    queries = metadata.get("webSearchQueries") if isinstance(metadata, dict) else None
    searches = len(queries) if isinstance(queries, list) and queries else None
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return SearchUsage(server_searches=searches)

    def count(name: str) -> int:
        value = usage.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    prompt = count("promptTokenCount") + count("toolUsePromptTokenCount")
    output = count("candidatesTokenCount") + count("thoughtsTokenCount")
    return SearchUsage(
        input_tokens=prompt or None,
        output_tokens=output or None,
        cache_read_tokens=count("cachedContentTokenCount") or None,
        server_searches=searches,
    )


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
    response = await client.post(
        f"{API_BASE}/{HOST_MODEL}:generateContent",
        # The header form, not ``?key=``: a key in the URL would land in any
        # logged request line or error that echoes the URL.
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "contents": [
                {"role": "user", "parts": [{"text": responses_wire.search_prompt(query)}]}
            ],
            "tools": [{"google_search": {}}],
            "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS},
        },
    )
    if not response.is_success:
        raise error_text(LABEL, response)
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(f"{LABEL} returned a non-JSON response") from error
    sources, answer = parse_grounded_search(payload, limit)
    usage = grounded_usage(payload)
    response_id = str(payload.get("responseId") or "").strip() or None
    return SearchResponse(
        provider="google",
        auth_mode=KEY_MODE,
        sources=sources,
        answer=answer,
        request_id=response_id,
        usage=usage,
        cost=pricing.native_cost(
            fee_usd=FEE_USD,
            usage=usage,
            hosting="google",
            model=HOST_MODEL,
            cache="inside",
            basis_note=BASIS_NOTE,
        ),
    )


VENDOR = NativeSearchVendor(
    id="google",
    label=LABEL,
    access=web_search_copy.native_access_key(),
    detail=web_search_copy.native_detail_key(vendor=LABEL),
    modes=_modes,
    search=search,
    fee_usd=FEE_USD,
    credential_keys=(ENV_KEY,),
)
