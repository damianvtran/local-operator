"""Built-in web-search provider transports.

This is deliberately a curated subset of Oh My Pi's much larger provider list:
the credential-free defaults, the established search APIs Local Operator already
documented, two prominent independent/AI-native APIs, and self-hosted SearXNG.
Each transport stays dependency-free beyond the project's existing ``httpx``.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from local_operator.credentials import CredentialManager
from local_operator.web_search.models import (
    PROVIDER_IDS,
    ProviderStatus,
    SearchProviderId,
    SearchResponse,
    SearchSource,
    WebSearchSettings,
)

ProviderSearch = Callable[
    [httpx.AsyncClient, CredentialManager, WebSearchSettings, str, int],
    Awaitable[SearchResponse],
]


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    """Static provider metadata plus its normalized transport."""

    id: SearchProviderId
    label: str
    access: str
    detail: str
    credential_keys: tuple[str, ...]
    search: ProviderSearch


def _credential(manager: CredentialManager, *keys: str) -> str:
    """Resolve the first non-empty stored/environment credential without logging it."""
    for key in keys:
        value = manager.get_credential(key).get_secret_value().strip()
        if value:
            return value
    return ""


def _http_error(provider: str, response: httpx.Response) -> RuntimeError:
    body = response.text.strip()
    if len(body) > 500:
        body = body[:500] + "…"
    suffix = f": {body}" if body else ""
    return RuntimeError(f"{provider} returned HTTP {response.status_code}{suffix}")


def _ensure_success(provider: str, response: httpx.Response) -> None:
    if not response.is_success:
        raise _http_error(provider, response)


def _bounded_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _source(
    *,
    title: object,
    url: object,
    snippet: object = None,
    published_date: object = None,
) -> SearchSource | None:
    target = str(url or "").strip()
    if len(target) > 4_096:
        return None
    if not target.startswith(("http://", "https://")):
        return None
    shown_title = _bounded_text(title or target, 500) or target
    shown_snippet = _bounded_text(snippet, 2_000) or None
    shown_date = _bounded_text(published_date, 100) or None
    return SearchSource(
        title=shown_title,
        url=target,
        snippet=shown_snippet,
        published_date=shown_date,
    )


def _clean_html(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _unwrap_duckduckgo_url(href: str) -> str:
    decoded = html.unescape(href)
    parsed = urlparse(decoded if "://" in decoded else f"https:{decoded}")
    wrapped = parse_qs(parsed.query).get("uddg")
    if wrapped:
        return unquote(wrapped[0])
    if decoded.startswith("//"):
        return "https:" + decoded
    return decoded


def parse_duckduckgo_html(page: str, limit: int) -> list[SearchSource]:
    """Parse DDG's no-JavaScript result rows without adding an HTML dependency."""
    rows: list[SearchSource] = []
    block_pattern = re.compile(
        r'<div\b[^>]*class="[^"]*\bresult\b[^"]*"[^>]*>([\s\S]*?)'
        r'(?=<div\b[^>]*class="[^"]*\bresult\b|<div\b[^>]*class="[^"]*\bnav-link\b|$)',
        re.IGNORECASE,
    )
    title_pattern = re.compile(
        r'<a\b[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]+)"[^>]*>' r"([\s\S]*?)</a>",
        re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<(?:a|div|span)\b[^>]*class="[^"]*\bresult__snippet\b[^"]*"[^>]*>'
        r"([\s\S]*?)</(?:a|div|span)>",
        re.IGNORECASE,
    )
    for block_match in block_pattern.finditer(page):
        block = block_match.group(1)
        title_match = title_pattern.search(block)
        if title_match is None:
            continue
        snippet_match = snippet_pattern.search(block)
        source = _source(
            title=_clean_html(title_match.group(2)),
            url=_unwrap_duckduckgo_url(title_match.group(1)),
            snippet=_clean_html(snippet_match.group(1)) if snippet_match else None,
        )
        if source is not None:
            rows.append(source)
        if len(rows) >= limit:
            break
    return rows


async def _search_duckduckgo(
    client: httpx.AsyncClient,
    _credentials: CredentialManager,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    response = await client.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": (
                "Mozilla/5.0 (compatible; LocalOperator/0.16; "
                "+https://github.com/damianvtran/local-operator)"
            ),
        },
    )
    _ensure_success("DuckDuckGo", response)
    if "anomaly-modal" in response.text or "anomaly.js" in response.text:
        raise RuntimeError("DuckDuckGo returned a bot challenge")
    return SearchResponse(
        provider="duckduckgo",
        auth_mode="credential-free",
        sources=parse_duckduckgo_html(response.text, limit),
    )


def tavily_response_from_payload(
    payload: dict[str, Any],
    *,
    auth_mode: str,
    limit: int,
) -> SearchResponse:
    """Normalize the identical direct-API and remote-MCP Tavily schemas."""
    sources = [
        source
        for item in payload.get("results", [])
        if isinstance(item, dict)
        and (
            source := _source(
                title=item.get("title"),
                url=item.get("url"),
                snippet=item.get("content"),
                published_date=item.get("published_date"),
            )
        )
        is not None
    ]
    return SearchResponse(
        provider="tavily",
        auth_mode=auth_mode,
        sources=sources[:limit],
        answer=str(payload.get("answer") or "").strip() or None,
        request_id=str(payload.get("request_id") or "").strip() or None,
    )


async def _search_tavily(
    client: httpx.AsyncClient,
    credentials: CredentialManager,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _credential(credentials, "TAVILY_API_KEY")
    headers = {"Content-Type": "application/json"}
    auth_mode = "api-key"
    if key:
        headers["Authorization"] = f"Bearer {key}"
    else:
        # Tavily documents this as its zero-account, rate-limited mode. Keeping
        # the wire shape identical lets a later key upgrade change no callers.
        headers["X-Tavily-Access-Mode"] = "keyless"
        auth_mode = "keyless"
    response = await client.post(
        "https://api.tavily.com/search",
        headers=headers,
        json={
            "query": query,
            "search_depth": "basic",
            "max_results": limit,
            "include_answer": "basic",
            "include_raw_content": False,
        },
    )
    _ensure_success("Tavily", response)
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Tavily returned a non-object response")
    return tavily_response_from_payload(payload, auth_mode=auth_mode, limit=limit)


def _perplexity_sources(payload: dict[str, Any], limit: int) -> list[SearchSource]:
    rows: list[SearchSource] = []
    candidates = payload.get("search_results") or payload.get("sources_list") or []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        source = _source(
            title=item.get("title") or item.get("name"),
            url=item.get("url"),
            snippet=item.get("snippet"),
            published_date=item.get("date") or item.get("timestamp"),
        )
        if source is not None:
            rows.append(source)
        if len(rows) >= limit:
            break
    if rows:
        return rows
    for citation in payload.get("citations", []):
        source = _source(title=citation, url=citation)
        if source is not None:
            rows.append(source)
        if len(rows) >= limit:
            break
    return rows


def _perplexity_answer(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            answer = str(message.get("content") or "").strip()
            if answer:
                return answer
    answer = str(payload.get("text") or "").strip()
    return answer or None


def _parse_perplexity_sse(body: str) -> dict[str, Any]:
    """Fold Perplexity's partial SSE blocks without losing earlier sources.

    The stream sends web results and answer markdown in different events. A
    plain ``dict.update`` keeps whichever block arrived last, which produced a
    cited answer with an empty source list on the live anonymous endpoint.
    """
    merged: dict[str, Any] = {}
    sources_by_url: dict[str, dict[str, Any]] = {}
    answer = ""
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        merged.update({key: value for key, value in event.items() if key != "blocks"})
        event_sources = event.get("sources_list")
        if isinstance(event_sources, list):
            for item in event_sources:
                if isinstance(item, dict) and item.get("url"):
                    sources_by_url[str(item["url"])] = item
        for block in event.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            web_results = (block.get("web_result_block") or {}).get("web_results") or []
            for item in web_results:
                if isinstance(item, dict) and item.get("url"):
                    sources_by_url[str(item["url"])] = item
            markdown = block.get("markdown_block")
            if not isinstance(markdown, dict):
                continue
            chunks = markdown.get("chunks")
            if isinstance(chunks, list) and chunks:
                answer = "".join(str(chunk) for chunk in chunks)
            elif markdown.get("answer"):
                answer = str(markdown["answer"])
        if event.get("text"):
            answer = str(event["text"])
    if sources_by_url:
        merged["sources_list"] = list(sources_by_url.values())
    if answer:
        merged["text"] = answer
    return merged


async def _search_perplexity(
    client: httpx.AsyncClient,
    credentials: CredentialManager,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _credential(credentials, "PERPLEXITY_API_KEY")
    if key:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": query}],
                "return_citations": True,
                "return_related_questions": False,
            },
        )
        _ensure_success("Perplexity", response)
        payload = response.json()
        return SearchResponse(
            provider="perplexity",
            auth_mode="api-key",
            sources=_perplexity_sources(payload, limit),
            answer=_perplexity_answer(payload),
            request_id=str(payload.get("id") or "").strip() or None,
        )

    request_id = str(uuid.uuid4())
    response = await client.post(
        "https://www.perplexity.ai/rest/sse/perplexity_ask",
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Origin": "https://www.perplexity.ai",
            "Referer": "https://www.perplexity.ai/",
            "User-Agent": "Mozilla/5.0 (compatible; LocalOperator/0.16)",
            "X-Request-ID": request_id,
        },
        json={
            "query_str": query,
            "params": {
                "query_str": query,
                "search_focus": "internet",
                "mode": "copilot",
                "sources": ["web"],
                "attachments": [],
                "frontend_uuid": str(uuid.uuid4()),
                "frontend_context_uuid": str(uuid.uuid4()),
                "language": "en-US",
                "is_incognito": True,
                "use_schematized_api": True,
                "skip_search_enabled": False,
                "always_search_override": True,
                "send_back_text_in_streaming_api": True,
            },
        },
    )
    _ensure_success("Perplexity", response)
    payload = _parse_perplexity_sse(response.text)
    return SearchResponse(
        provider="perplexity",
        auth_mode="anonymous",
        sources=_perplexity_sources(payload, limit),
        answer=_perplexity_answer(payload),
        request_id=str(payload.get("uuid") or request_id),
    )


async def _search_brave(
    client: httpx.AsyncClient,
    credentials: CredentialManager,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _credential(credentials, "BRAVE_API_KEY")
    response = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": key},
        params={"q": query, "count": limit, "extra_snippets": "true"},
    )
    _ensure_success("Brave", response)
    payload = response.json()
    sources: list[SearchSource] = []
    for item in payload.get("web", {}).get("results", []):
        if not isinstance(item, dict):
            continue
        snippets = [str(item.get("description") or "").strip()]
        snippets.extend(str(value).strip() for value in item.get("extra_snippets") or [])
        source = _source(
            title=item.get("title"),
            url=item.get("url"),
            snippet="\n".join(value for value in dict.fromkeys(snippets) if value),
            published_date=item.get("age"),
        )
        if source is not None:
            sources.append(source)
    return SearchResponse(
        provider="brave",
        auth_mode="api-key",
        sources=sources[:limit],
        request_id=response.headers.get("x-request-id"),
    )


async def _search_exa(
    client: httpx.AsyncClient,
    credentials: CredentialManager,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _credential(credentials, "EXA_API_KEY")
    response = await client.post(
        "https://api.exa.ai/search",
        headers={"Content-Type": "application/json", "x-api-key": key},
        json={
            "query": query,
            "numResults": limit,
            "type": "auto",
            # Exa can return whole page text here, but downloading it only to
            # truncate it wastes latency and context. Query-grounded summaries
            # are the provider-native short snippet contract the UI needs.
            "contents": {"summary": {"query": query}},
        },
    )
    _ensure_success("Exa", response)
    payload = response.json()
    sources = [
        source
        for item in payload.get("results", [])
        if isinstance(item, dict)
        and (
            source := _source(
                title=item.get("title"),
                url=item.get("url"),
                snippet=item.get("summary"),
                published_date=item.get("publishedDate"),
            )
        )
        is not None
    ]
    return SearchResponse(provider="exa", auth_mode="api-key", sources=sources[:limit])


async def _search_serpapi(
    client: httpx.AsyncClient,
    credentials: CredentialManager,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _credential(credentials, "SERPAPI_API_KEY", "SERP_API_KEY")
    response = await client.get(
        "https://serpapi.com/search.json",
        params={"q": query, "engine": "google", "api_key": key, "num": limit},
    )
    _ensure_success("SerpApi", response)
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"SerpApi error: {payload['error']}")
    sources = [
        source
        for item in payload.get("organic_results", [])
        if isinstance(item, dict)
        and (
            source := _source(
                title=item.get("title"),
                url=item.get("link"),
                snippet=item.get("snippet"),
                published_date=item.get("date"),
            )
        )
        is not None
    ]
    metadata = payload.get("search_metadata") or {}
    return SearchResponse(
        provider="serpapi",
        auth_mode="api-key",
        sources=sources[:limit],
        request_id=str(metadata.get("id") or "").strip() or None,
    )


async def _search_searxng(
    client: httpx.AsyncClient,
    _credentials: CredentialManager,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    endpoint = settings.searxng_endpoint.rstrip("/")
    response = await client.get(
        f"{endpoint}/search",
        params={"q": query, "format": "json", "categories": "general"},
    )
    _ensure_success("SearXNG", response)
    payload = response.json()
    sources = [
        source
        for item in payload.get("results", [])
        if isinstance(item, dict)
        and (
            source := _source(
                title=item.get("title"),
                url=item.get("url"),
                snippet=item.get("content"),
                published_date=item.get("publishedDate"),
            )
        )
        is not None
    ]
    return SearchResponse(provider="searxng", auth_mode="self-hosted", sources=sources[:limit])


PROVIDERS: dict[SearchProviderId, ProviderDefinition] = {
    "duckduckgo": ProviderDefinition(
        "duckduckgo",
        "DuckDuckGo",
        "free",
        "Credential-free HTML search",
        (),
        _search_duckduckgo,
    ),
    "tavily": ProviderDefinition(
        "tavily",
        "Tavily",
        "free / key / OAuth MCP",
        "Official keyless access; TAVILY_API_KEY raises limits",
        ("TAVILY_API_KEY",),
        _search_tavily,
    ),
    "perplexity": ProviderDefinition(
        "perplexity",
        "Perplexity",
        "anonymous / key",
        "Best-effort anonymous search; PERPLEXITY_API_KEY uses Sonar",
        ("PERPLEXITY_API_KEY",),
        _search_perplexity,
    ),
    "brave": ProviderDefinition(
        "brave",
        "Brave",
        "API key",
        "Independent search index; requires BRAVE_API_KEY",
        ("BRAVE_API_KEY",),
        _search_brave,
    ),
    "exa": ProviderDefinition(
        "exa",
        "Exa",
        "API key",
        "AI-native semantic search; requires EXA_API_KEY",
        ("EXA_API_KEY",),
        _search_exa,
    ),
    "serpapi": ProviderDefinition(
        "serpapi",
        "SerpApi",
        "API key",
        "Google-backed results; requires SERPAPI_API_KEY",
        ("SERPAPI_API_KEY", "SERP_API_KEY"),
        _search_serpapi,
    ),
    "searxng": ProviderDefinition(
        "searxng",
        "SearXNG",
        "self-hosted",
        "Private metasearch; requires a SearXNG endpoint",
        (),
        _search_searxng,
    ),
}


def provider_available(
    provider_id: SearchProviderId,
    credentials: CredentialManager,
    settings: WebSearchSettings,
) -> bool:
    """Whether the provider can make a request with current local configuration."""
    if provider_id in ("duckduckgo", "tavily", "perplexity"):
        return True
    if provider_id == "searxng":
        return settings.searxng_endpoint.startswith(("http://", "https://"))
    definition = PROVIDERS[provider_id]
    return bool(_credential(credentials, *definition.credential_keys))


def provider_statuses(
    settings: WebSearchSettings,
    credentials: CredentialManager,
) -> list[ProviderStatus]:
    """Return every provider in the stable, user-facing catalogue order."""
    enabled = set(settings.providers)
    return [
        ProviderStatus(
            id=provider_id,
            label=PROVIDERS[provider_id].label,
            enabled=provider_id in enabled,
            available=provider_available(provider_id, credentials, settings),
            access=PROVIDERS[provider_id].access,
            detail=PROVIDERS[provider_id].detail,
        )
        for provider_id in PROVIDER_IDS
    ]
