"""Built-in web-search provider transports.

This is deliberately a curated subset of Oh My Pi's much larger provider list:
the credential-free defaults, the established search APIs Local Operator already
documented, two prominent independent/AI-native APIs, and self-hosted SearXNG.
Each transport stays dependency-free beyond the project's existing ``httpx``.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, urlparse

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
        return wrapped[0]
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


# ---------------------------------------------------------------------------
# DeepSeek native search (Anthropic-format Messages API)
# ---------------------------------------------------------------------------
#
# DeepSeek exposes no dedicated search endpoint: the only server-side search it
# serves is the Anthropic `web_search_20250305` server tool on the
# Anthropic-compatible Messages route. This transport mirrors the reference
# implementation in DeepSeek's own harness
# (`@deepseek-ai/dsh-web-search-deepseek`) so both agree on the wire shape and on
# what counts as a usable result:
#
#   * ONE auxiliary model turn per search, carrying exactly
#     "Perform a web search for the query: <query>" as its user text;
#   * `max_uses: 1` -- across 31 measured live searches DeepSeek issued exactly
#     one server-side search (`usage.server_tool_use.web_search_requests == 1`
#     every time), so a higher cap only risks paying for a second one;
#   * `max_tokens: 1024` -- observed completions were 507-1444 output tokens and
#     the sources arrive in `web_search_tool_result` blocks independently of how
#     much prose follows, so a small cap cannot lose a source;
#   * sources come from `web_search_tool_result` items only. The turn's prose is
#     a synthesized answer, NOT a source of results, so it is returned as
#     `answer` and never parsed for URLs;
#   * a response with no result block, or with no usable item, RAISES. Returning
#     the prose as a successful answer would repeat the anonymity-wall defect the
#     Perplexity transport has, where a non-empty string stops the fallback chain
#     with nothing the model can act on.
#
# Two measured limitations belong here rather than in a ticket:
#
#   * `web_search_result` items carry `page_age` in the schema but it arrived
#     EMPTY on all 310 items observed, and text blocks carried no `citations`
#     array at all (thinking and non-thinking alike), so DeepSeek's own harness
#     note holds: "Uncited results carry no `snippet`". Sources are therefore
#     title+URL on this path, weaker than DuckDuckGo/Tavily/Brave, and the
#     synthesized `answer` is the only descriptive text the model gets.
#   * Cost is a full model turn: median 14.5k input + 0.9k output tokens per
#     search (cached page content dominates the input side), i.e. ~$0.0024
#     off-peak / ~$0.0048 peak at deepseek-flash list price (2026-09). That is
#     more than the free transports and than Tavily's monthly plans, so this
#     provider is credential-gated and belongs in an `ordered` chain behind the
#     free paths rather than in the default rotation.

DEEPSEEK_SEARCH_ENDPOINT = "https://api.deepseek.com/anthropic/v1/messages"
DEEPSEEK_BALANCE_ENDPOINT = "https://api.deepseek.com/user/balance"
#: Anthropic-format model name. `deepseek-v4-flash` is an accepted alias of the
#: current Flash model, kept because the DeepSeek harness pins it.
DEEPSEEK_SEARCH_MODEL = "deepseek-v4-flash"
DEEPSEEK_API_VERSION = "2023-06-01"
DEEPSEEK_SEARCH_MAX_TOKENS = 1_024
#: The search turn only has to produce the result blocks and a short answer; the
#: evidence pass does the rest of the work. Measured, trimming the search turn
#: to 256 tokens still returned all 10 sources and cut its latency to 3.1-3.4s.
DEEPSEEK_SEARCH_ANSWER_MAX_TOKENS = 256
DEEPSEEK_SEARCH_MAX_USES = 1
#: Balance below which a search is not attempted. One search bills a full model
#: turn, so an account with cents left must be skipped in favour of the free
#: transports instead of failing mid-chain.
DEEPSEEK_MIN_BALANCE_USD = 0.50
#: How long a balance verdict is trusted. The probe is one keyless GET, but a
#: burst of searches must not pay it per call; a minute is short enough that a
#: topped-up or drained account flips on the next real search.
DEEPSEEK_BALANCE_TTL_SECONDS = 60.0

# ---------------------------------------------------------------------------
# DeepSeek per-page evidence pass (`web_search.deepseek_evidence`)
# ---------------------------------------------------------------------------
#
# Native search returns `web_search_result` items carrying only url/title (no
# snippet: measured, `page_age` is always empty and text blocks carry no
# `citations`), so the model gets nothing to judge WHICH page is worth fetching.
# The page text is nevertheless reachable: returning the assistant's content
# blocks verbatim in a follow-up Messages request restores them in the model's
# context -- DeepSeek honours Anthropic's `encrypted_content` contract, and the
# restored pages arrive as CACHE READS (input_tokens ~200, cache_read 9-19k), so
# re-asking about them is cheap. A third turn can keep asking about the same
# pages with no new search billed.
#
# Measured over three queries, replay + triage of the top 5:
#   * 4.1-7.4s added, $0.0013-0.0021 peak on top of the search turn;
#   * 5/5 rows returned a url, summary and verbatim quote, 0 malformed lines;
#   * quotes verified verbatim against the live pages in 4/5-5/5 cases.
# Asking for all 10 sources in one turn instead truncates at any sane
# `max_tokens` (measured: JSONL overran 3072 tokens, malformed rows, and the
# quotes degraded), and asking the SAME turn to emit the evidence was slower
# (9.1-13.0s) and 2-3x the cost of search-then-triage -- which is why the
# evidence pass is a separate, bounded, opt-in turn.

#: Sources the evidence pass covers. Bounded because the payload is generated
#: text: past ~5 rows it overruns the token cap and the later rows degrade.
DEEPSEEK_EVIDENCE_TOP_N = 5
#: Sized from measurement, not taste: top-5 JSONL payloads ran 1000-1729 output
#: tokens across live runs, and a cap below that truncates mid-line, which loses
#: rows AND makes the retry look like a silent no-op. 2048 leaves headroom.
DEEPSEEK_EVIDENCE_MAX_TOKENS = 2_048
DEEPSEEK_EVIDENCE_INSTRUCTION = (
    "Using the pages already retrieved above, produce per-page evidence.\n"
    "Reply with JSONL ONLY: one JSON object per line, no array, no prose, no code fence. "
    'Each line: {"url": "...", "on_topic": true|false, "relevance": 0-100, '
    '"summary": "<=15 words", "quote": "<=25 words verbatim from that page"}. '
    f"One line per search result, best first, exactly {DEEPSEEK_EVIDENCE_TOP_N} lines. "
    "Fields must be valid JSON strings on a single line."
)

_DEEPSEEK_BALANCE_LOCK = threading.Lock()
_DEEPSEEK_BALANCE: tuple[float, bool] | None = None
#: In-flight balance refreshes. Held so a background probe is never garbage
#: collected mid-request and so its exception is always retrievable.
_DEEPSEEK_BALANCE_TASKS: set[asyncio.Task[None]] = set()


def _deepseek_citation_snippets(blocks: list[Any]) -> dict[str, str]:
    """Excerpts DeepSeek attaches to its answer text, keyed by source URL.

    Empty in practice (see the module note above) but kept because it is the
    only mechanism that can ever give this provider a snippet, and dropping it
    would silently discard the field if DeepSeek starts emitting citations.
    """
    snippets: dict[str, str] = {}
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        for citation in block.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            url = str(citation.get("url") or "")
            text = str(citation.get("cited_text") or "").strip()
            if url and text and url not in snippets:
                snippets[url] = text
    return snippets


def _deepseek_answer(blocks: list[Any]) -> str | None:
    """The turn's prose, which is an answer summary and never a result list."""
    parts = [
        str(block.get("text") or "").strip()
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    answer = "\n\n".join(part for part in parts if part).strip()
    return answer or None


def parse_deepseek_search(
    payload: object, limit: int
) -> tuple[list[SearchSource], str | None]:
    """Normalize one Anthropic-format Messages response into sources + answer.

    Dedupes by URL because a `max_uses > 1` request can surface the same page
    from more than one server-side search; the DeepSeek harness normalizer does
    the same for the same reason.
    """
    if not isinstance(payload, dict):
        raise RuntimeError("DeepSeek returned a non-object response")
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise RuntimeError("DeepSeek returned no content blocks")

    snippets = _deepseek_citation_snippets(blocks)
    sources: list[SearchSource] = []
    seen: set[str] = set()
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "web_search_tool_result":
            continue
        items = block.get("content")
        if not isinstance(items, list):
            # Error-shaped tool results (a dict instead of a list) carry no
            # sources; the transport's no-result check below turns a wholly
            # failed search into the fallback trigger.
            continue
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "web_search_result":
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in seen:
                continue
            source = _source(
                title=item.get("title"),
                url=url,
                snippet=snippets.get(url),
                published_date=item.get("page_age"),
            )
            if source is None:
                continue
            seen.add(url)
            sources.append(source)
            if len(sources) >= limit:
                break
        if len(sources) >= limit:
            break

    if not sources:
        raise RuntimeError(
            "DeepSeek returned no web_search results; the request may not have "
            "triggered native search"
        )
    return sources, _deepseek_answer(blocks)


def _deepseek_login_present() -> bool:
    """Whether a DeepSeek API key was stored by ``lop login deepseek``.

    The model key and the search key are deliberately the same credential, which
    is why this provider needs no search-specific setup step. Read through the
    auth store because that is where ``login`` writes it; the env/credentials.env
    tier is checked first by the caller, so this stays a pure store probe.
    """
    try:
        from local_operator.providers.auth_store import AuthStore

        return bool(AuthStore().list_credentials("deepseek"))
    except Exception:  # noqa: BLE001 -- an unreadable store is simply "unknown"
        return False


async def _resolve_deepseek_key(credentials: CredentialManager) -> str:
    """The key the search will bill: env/credentials.env first, then the login.

    ``read_only=True`` because a search must never decide model routing: it must
    not consume an OAuth rotation slot, clear session stickiness, or flip the
    auth tier the conversation's next request will resolve from.
    """
    key = _credential(credentials, "DEEPSEEK_API_KEY")
    if key:
        return key
    try:
        from local_operator.providers.auth_store import AuthStore

        return await AuthStore().get_api_key("deepseek", read_only=True) or ""
    except Exception:  # noqa: BLE001 -- resolution failure is "no key"
        return ""


def _cached_deepseek_balance_verdict() -> bool | None:
    """The last balance verdict within its TTL, or None when there is none."""
    global _DEEPSEEK_BALANCE
    with _DEEPSEEK_BALANCE_LOCK:
        if _DEEPSEEK_BALANCE is None:
            return None
        checked_at, allowed = _DEEPSEEK_BALANCE
    if time.monotonic() - checked_at > DEEPSEEK_BALANCE_TTL_SECONDS:
        return None
    return allowed


def _remember_deepseek_balance(allowed: bool) -> None:
    global _DEEPSEEK_BALANCE
    with _DEEPSEEK_BALANCE_LOCK:
        _DEEPSEEK_BALANCE = (time.monotonic(), allowed)


def reset_deepseek_balance_cache_for_tests() -> None:
    """Deterministic state for tests, mirroring the round-robin reset."""
    global _DEEPSEEK_BALANCE
    with _DEEPSEEK_BALANCE_LOCK:
        _DEEPSEEK_BALANCE = None
    _DEEPSEEK_BALANCE_TASKS.clear()


async def _deepseek_balance_ok(client: httpx.AsyncClient, key: str) -> bool:
    """Whether the account can still pay for a search.

    Deliberately a live probe rather than a read of the cached usage report: the
    shared usage cache key is derived from the account fingerprint, and a second
    derivation in this module would drift from the controller's the first time
    the tier list changes. A failed probe is treated as ``ok`` -- a balance
    endpoint outage must not silently remove a paid transport from the chain,
    and the search itself will surface a real funding problem as its own error.
    """
    try:
        response = await client.get(
            DEEPSEEK_BALANCE_ENDPOINT,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        )
        if not response.is_success:
            return True
        payload = response.json()
    except Exception:  # noqa: BLE001 -- see docstring: unknown is permissive
        return True
    if not isinstance(payload, dict):
        return True
    if payload.get("is_available") is False:
        return False
    infos = payload.get("balance_infos")
    if not isinstance(infos, list):
        return True
    total = 0.0
    saw_usd = False
    for item in infos:
        if not isinstance(item, dict):
            continue
        if str(item.get("currency") or "").upper() != "USD":
            # A non-USD balance cannot be compared to a USD floor; leave the
            # decision to the search itself rather than guessing an FX rate.
            continue
        try:
            total += float(item.get("total_balance") or 0)
        except (TypeError, ValueError):
            continue
        saw_usd = True
    if not saw_usd:
        return True
    return total >= DEEPSEEK_MIN_BALANCE_USD


async def _refresh_deepseek_balance_verdict(key: str) -> None:
    """Refresh the cached balance verdict for the NEXT search, off the hot path.

    Uses its own short-lived client rather than the session's pooled one: this
    task can outlive the search that spawned it (the tool's pooled client is
    owned by the session, and a one-shot CLI caller closes its own), so borrowing
    a client here would race the owner's ``aclose``.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            allowed = await _deepseek_balance_ok(client, key)
    except Exception:  # noqa: BLE001 -- a refresh is best-effort by construction
        return
    _remember_deepseek_balance(allowed)


def _spawn_deepseek_balance_refresh(key: str) -> None:
    """Start the background balance refresh, keeping a handle on the task."""
    task = asyncio.create_task(_refresh_deepseek_balance_verdict(key))
    _DEEPSEEK_BALANCE_TASKS.add(task)
    task.add_done_callback(_DEEPSEEK_BALANCE_TASKS.discard)


def parse_deepseek_evidence(text: str) -> dict[str, dict[str, Any]]:
    """Parse the evidence pass's JSONL into ``url -> row``, skipping bad lines.

    Tolerant by design: this is generated text on a token budget, so a line that
    is not valid JSON is dropped rather than failing the whole search. The
    sources themselves are already in hand; evidence only ever ENRICHES them.
    """
    rows: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped.startswith("{"):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if url and url not in rows:
            rows[url] = row
    return rows


async def _deepseek_evidence_pass(
    client: httpx.AsyncClient,
    key: str,
    assistant_blocks: list[Any],
    prompt: str,
) -> dict[str, dict[str, Any]]:
    """Replay the search turn and ask for per-page evidence, in one extra turn.

    The assistant blocks are sent back EXACTLY as received, including each
    result's opaque ``encrypted_content``: that is what makes DeepSeek restore
    the page text into context. Sending the items stripped of it does not, so
    this must not "clean" the blocks.
    """
    response = await client.post(
        DEEPSEEK_SEARCH_ENDPOINT,
        headers={
            "x-api-key": key,
            "authorization": f"Bearer {key}",
            "anthropic-version": DEEPSEEK_API_VERSION,
            "content-type": "application/json",
            "accept": "application/json",
        },
        json={
            "model": DEEPSEEK_SEARCH_MODEL,
            "max_tokens": DEEPSEEK_EVIDENCE_MAX_TOKENS,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": assistant_blocks},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": DEEPSEEK_EVIDENCE_INSTRUCTION}],
                },
            ],
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": DEEPSEEK_SEARCH_MAX_USES,
                }
            ],
        },
    )
    _ensure_success("DeepSeek evidence", response)
    payload = response.json()
    text = "\n".join(
        str(block.get("text") or "")
        for block in (payload.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return parse_deepseek_evidence(text)


async def _search_deepseek(
    client: httpx.AsyncClient,
    credentials: CredentialManager,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = await _resolve_deepseek_key(credentials)
    if not key:
        raise RuntimeError(
            "DeepSeek search needs an API key; run `local-operator login deepseek`"
        )

    # The balance gate is a CACHED verdict, never a probe on this call's path.
    # Measured, the probe costs 300-580 ms -- 7-13% of a ~4.5 s search -- and it
    # is the only part of this provider's latency that is ours rather than
    # DeepSeek's. A cached verdict still skips an account that is out of money on
    # every search after the first, and the search itself fails loudly (and so
    # triggers the fallback chain) if the account cannot pay after all.
    if _cached_deepseek_balance_verdict() is False:
        raise RuntimeError(
            f"DeepSeek balance is below ${DEEPSEEK_MIN_BALANCE_USD:.2f}; "
            "top up or disable the deepseek search provider"
        )
    if _cached_deepseek_balance_verdict() is None:
        _spawn_deepseek_balance_refresh(key)

    # One string, reused verbatim by the evidence pass: the replayed conversation
    # must be exactly what produced these blocks, or the restored pages and the
    # follow-up question would describe two different searches.
    prompt = f"Perform a web search for the query: {query}"

    response = await client.post(
        DEEPSEEK_SEARCH_ENDPOINT,
        headers={
            # Official DeepSeek accepts `x-api-key`; an Anthropic-compatible proxy
            # may expect `Authorization: Bearer`. Sending both is the DeepSeek
            # harness's own choice so either deployment resolves.
            "x-api-key": key,
            "authorization": f"Bearer {key}",
            "anthropic-version": DEEPSEEK_API_VERSION,
            "content-type": "application/json",
            "accept": "application/json",
        },
        json={
            "model": DEEPSEEK_SEARCH_MODEL,
            # With evidence on, the search turn is only a results carrier: the
            # triage turn supplies the descriptive text, so the answer budget
            # drops to the minimum that still emits every source.
            "max_tokens": (
                DEEPSEEK_SEARCH_ANSWER_MAX_TOKENS
                if settings.deepseek_evidence
                else DEEPSEEK_SEARCH_MAX_TOKENS
            ),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": DEEPSEEK_SEARCH_MAX_USES,
                }
            ],
        },
    )
    _ensure_success("DeepSeek", response)
    payload = response.json()
    sources, answer = parse_deepseek_search(payload, limit)
    if settings.deepseek_evidence:
        # Enrichment only. A failed, truncated or unparseable evidence pass must
        # leave the sources exactly as the search returned them, because the
        # model can still fetch them; losing the search to a triage failure would
        # trade a usable answer for nothing. The reason is REPORTED rather than
        # swallowed -- an unenriched result that says why is diagnosable, and one
        # that silently looks like "this provider has no snippets" is not.
        try:
            evidence = await _deepseek_evidence_pass(
                client, key, payload.get("content") or [], prompt
            )
        except Exception as error:  # noqa: BLE001 -- see comment above
            evidence = {}
            evidence_failure = f"deepseek evidence pass: {error}"
        else:
            evidence_failure = (
                None if evidence else "deepseek evidence pass: returned no usable rows"
            )
        sources = _apply_deepseek_evidence(sources, evidence)
    else:
        evidence_failure = None
    return SearchResponse(
        provider="deepseek",
        auth_mode="api-key",
        sources=sources,
        answer=answer,
        request_id=str(response.headers.get("x-request-id") or "").strip() or None,
        failures=[evidence_failure] if evidence_failure else [],
    )


def _apply_deepseek_evidence(
    sources: list[SearchSource], evidence: dict[str, dict[str, Any]]
) -> list[SearchSource]:
    """Merge the evidence rows onto the sources, then rank them by relevance.

    Ordering is the point of the pass: DeepSeek's own result order is opaque,
    while a judged relevance lets the model fetch the two or three pages that
    matter instead of reading all ten. Sources the pass did not cover keep their
    relative order at the END rather than being dropped -- the pass is a top-N
    view, not a verdict on the rest of the page set.
    """
    scored: list[SearchSource] = []
    unscored: list[SearchSource] = []
    for source in sources:
        row = evidence.get(source.url)
        if not row:
            unscored.append(source)
            continue
        quote = str(row.get("quote") or "").strip()
        summary = str(row.get("summary") or "").strip()
        relevance = row.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
            relevance = None
        scored.append(
            source.model_copy(
                update={
                    # A verbatim quote is a real snippet; the summary is the
                    # fallback when a page could not be quoted.
                    "snippet": quote or summary or source.snippet,
                    "relevance": int(relevance) if relevance is not None else None,
                }
            )
        )
    scored.sort(
        key=lambda item: item.relevance if item.relevance is not None else -1, reverse=True
    )
    return [*scored, *unscored]


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
        "Keyless by default; API key raises limits; OAuth setup available",
        ("TAVILY_API_KEY",),
        _search_tavily,
    ),
    "deepseek": ProviderDefinition(
        "deepseek",
        "DeepSeek",
        "model API key",
        (
            "Native search via the Anthropic-format Messages API; reuses the "
            "DeepSeek model key, bills one model turn per search"
        ),
        ("DEEPSEEK_API_KEY",),
        _search_deepseek,
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
    if provider_id == "deepseek":
        # The search key IS the model key, and ``login`` writes it to the auth
        # store rather than to ``credentials.env``. Both tiers are checked, or
        # `search list` would report "setup needed" for a provider whose calls
        # would in fact succeed -- and, worse, the reverse for a keyless branch
        # that has no key at all.
        return bool(_credential(credentials, "DEEPSEEK_API_KEY")) or _deepseek_login_present()
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
