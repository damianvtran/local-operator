"""Built-in web-search provider transports.

This is deliberately a curated subset of Oh My Pi's much larger provider list:
the credential-free defaults, the established search APIs Local Operator already
documented, two prominent independent/AI-native APIs, and self-hosted SearXNG.
Each transport stays dependency-free beyond the project's existing ``httpx``.
"""

from __future__ import annotations

import asyncio
import contextvars
import html
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, NamedTuple
from urllib.parse import parse_qs, urlparse

import httpx

from local_operator.i18n.keys import web_search as web_search_copy
from local_operator.i18n.messages import Msg
from local_operator.web_search.cost import estimate_search_cost
from local_operator.web_search.models import (
    PROVIDER_IDS,
    ChainLegStatus,
    ProviderStatus,
    ProviderTier,
    SearchProviderId,
    SearchResponse,
    SearchSource,
    SearchUsage,
    WebSearchSettings,
)
from local_operator.web_search.pages import PAGE_CONTEXTS
from local_operator.web_search.radient import (
    radient_credit_refusal_remaining,
    radient_login_present,
    search_radient,
)

ProviderSearch = Callable[
    [httpx.AsyncClient, Path | None, WebSearchSettings, str, int],
    Awaitable[SearchResponse],
]

#: The mode the chain leg being walked asked for, or ``None`` outside a walk.
#:
#: WHY A CONTEXT VARIABLE rather than a sixth ``ProviderSearch`` argument: the
#: transport signature is a public seam -- tests and harness wrappers substitute
#: ``replace(PROVIDERS[id], search=fake)`` with five-argument callables -- and only
#: the four dual-mode transports care about the mode. The service sets this around
#: each leg (``leg_mode``); a dual-mode transport reads it to serve its keyless
#: tier even when a key is stored, and falls back to the old key-decides rule when
#: it is unset (a direct call outside the chain).
LEG_MODE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "web_search_leg_mode", default=None
)


def _keyed_leg(config_dir: Path | None, key_name: str) -> str:
    """The key to use for THIS call of a dual-mode transport, or "" for its free tier.

    A walked leg decides: its free mode means keyless even with a stored key, and
    its keyed mode means the key. Outside a walk (``LEG_MODE`` unset) the stored
    key decides, as it always did.
    """
    mode = LEG_MODE.get()
    if mode is not None and mode not in METERED_AUTH_MODES:
        return ""
    return _credential(config_dir, key_name)


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    """Static provider metadata plus its normalized transport."""

    id: SearchProviderId
    label: str
    #: ``Msg`` for copy that is keyed (newer rows); plain ``str`` rows predate the
    #: catalogue. ``provider_copy`` renders either at display time.
    access: str | Msg
    detail: str | Msg
    credential_keys: tuple[str, ...]
    search: ProviderSearch


def provider_copy(value: str | Msg) -> str:
    """A provider row's access/detail text in the active locale."""
    if isinstance(value, Msg):
        from local_operator.i18n.text import tr

        # Every keyed provider row lives in the ``web_search`` namespace.
        return tr("web_search")(value)
    return value


def _credential(config_dir: Path | None, *keys: str) -> str:
    """First configured value across the provider store, then the environment.

    Store-first: a ``LOP_PROVIDER_<key>`` row the operator saved (via ``lop
    search setup`` or ``lop credential update``) outranks an ambient export,
    which is the order every other provider-key reader in the repo uses. The
    legacy ``credentials.env`` leg is GONE (PR2a): a key the store does not hold
    and the environment does not export resolves to nothing. The value is never
    logged.
    """
    from local_operator.providers.registry import provider_secret_value

    for key in keys:
        stored = provider_secret_value(key, base=config_dir)
        if stored:
            return stored
    for key in keys:
        exported = os.environ.get(key, "").strip()
        if exported:
            return exported
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


#: Longest url a source may carry, in characters, measured on the stripped url.
#:
#: ONE constant for two callers that must agree: ``_source`` drops a longer url,
#: and ``_parse_perplexity_sse``'s served-content predicate refuses to count such
#: a row as served content. Spelled twice they drift, and the drift is silent in
#: the worst direction -- the predicate accepts the row, the extractor builds no
#: source from it, and the refusal sentence is handed back as the answer because
#: the wall was suppressed for a payload with zero sources (#1070, the #1061
#: shape one rule further out).
_MAX_URL_CHARS = 4_096


def _source(
    *,
    title: object,
    url: object,
    snippet: object = None,
    published_date: object = None,
) -> SearchSource | None:
    target = str(url or "").strip()
    if len(target) > _MAX_URL_CHARS:
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
    _config_dir: Path | None,
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
        # Keyless is a FREE TIER, not a missing price: without this flag the
        # ledger would charge the paid per-credit rate for searches that the
        # free tier served.
        usage=SearchUsage(keyless=auth_mode == "keyless"),
    )


async def _search_tavily(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _keyed_leg(config_dir, "TAVILY_API_KEY")
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


#: The block shapes the anonymous endpoint uses to SERVE results. The wall test
#: asks whether any of these appeared, rather than whether the source extractor
#: recognised one: a shopping, hotels, maps or media answer is served content
#: that ``_perplexity_sources`` has no rows for, and treating its absence as "the
#: endpoint served nothing" would discard a real answer that happens to carry a
#: sign-in nudge (raised by the round-1 code review; the shapes are Perplexity's
#: documented SSE block union).
_RESULT_BLOCK_KEYS: tuple[str, ...] = (
    "web_result_block",
    "shopping_block",
    "hotels_mode_block",
    "maps_mode_block",
    "media_block",
)


def _parse_perplexity_sse(body: str) -> dict[str, Any]:
    """Fold Perplexity's partial SSE blocks without losing earlier sources.

    The stream sends web results and answer markdown in different events. A
    plain ``dict.update`` keeps whichever block arrived last, which produced a
    cited answer with an empty source list on the live anonymous endpoint.
    """
    merged: dict[str, Any] = {}
    sources_by_url: dict[str, dict[str, Any]] = {}
    #: Result-bearing blocks seen anywhere in the stream, so the wall test can
    #: ask what the endpoint SERVED instead of what the extractor recognised.
    served_blocks: set[str] = set()
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
        # A top-level source list is SERVED CONTENT even when the extractor
        # rejects every entry (an entry with no URL scheme reaches here but not
        # into ``sources_by_url``). Round-2 review MINOR-1: without this, a real
        # answer plus a top-level source list, carrying the soft sign-in marker,
        # was raised away -- the call site ANDs the verdict with ``not sources``,
        # which protects only the shape the extractor ACCEPTED.
        # A row counts as SERVED only when it carries a URL, which is what makes
        # it a search result at all: ``_perplexity_sources`` builds nothing from a
        # row without one, so a name-only or empty dict row beside a wall would
        # leave the wall's own sentence as the response -- the original bug,
        # through this door (round-1 review MINOR-1, whose own repro shows a
        # name-only row being served). A row WITHOUT a scheme still counts: it is
        # a result the extractor rejects, which is the shape this check exists for.
        for key in ("sources_list", "search_results"):
            rows = event.get(key)
            if isinstance(rows, list) and any(
                # ``isinstance`` AND a non-blank string, not truthiness: ``_source``
                # strips the url and requires a scheme, so ``{"url": 5}``,
                # ``{"url": True}``, ``{"url": {}}`` and ``{"url": "   "}`` each
                # produce zero sources while suppressing the wall -- the same hole
                # one type-check narrower (round-2 review MINOR-1). A blank url is
                # no url; a non-string is not one either.
                #
                # ``_MAX_URL_CHARS`` caps it for the same reason: ``_source`` drops
                # a url past the cap, so counting the row as served would suppress
                # the wall for a payload the extractor builds nothing from -- #1070,
                # where the refusal sentence went back as the answer. The length is
                # taken on the STRIPPED url, as ``_source`` takes it: a url that is
                # over the cap only because of surrounding whitespace is one the
                # extractor accepts, so it is one this must accept too.
                isinstance(row, dict)
                and isinstance(row.get("url"), str)
                and 0 < len(row["url"].strip()) <= _MAX_URL_CHARS
                for row in rows
            ):
                served_blocks.add(f"top_level_{key}")
                break
        for block in event.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            served_blocks.update(key for key in _RESULT_BLOCK_KEYS if block.get(key) is not None)
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
    # Written UNCONDITIONALLY, and namespaced. Round-2 review NIT-1: a payload
    # that arrives carrying our own key -- the endpoint echoing a field name, or
    # anything else on the wire -- used to survive ``merged.update`` and suppress
    # the wall, which is the original bug reached from the other direction. The
    # parser owns this key, so the parser always sets it, empty included.
    #
    # It is not a source list: it is the answer to "did the endpoint serve
    # anything at all", which is what separates a refusal from a thin result.
    merged["_lo_served_blocks"] = sorted(served_blocks)
    if answer:
        merged["text"] = answer
    return merged


def _perplexity_authwall(payload: dict[str, Any]) -> str | None:
    """The sign-in wall this payload is, or ``None`` when it is not one.

    The anonymous endpoint refuses with a NORMALLY COMPLETED stream: HTTP 200,
    ``status: COMPLETED``, ``final: true``, ``text`` carrying a short invitation
    to sign in. Nothing in the transport says "refused", so the payload's own
    ``upsell_information`` is the only marker (observed live:
    ``{'name': 'fraud_authwall_upsell', 'upsell_type': 'LOGIN'}``).

    It matters because the refusal is otherwise indistinguishable from a RESULT:
    the chain accepts a response with an empty source list when its answer is
    non-empty -- correct in general, since a provider may legitimately answer
    without citations -- so this sentence was served to the model as the search
    it asked for. Reported from a real session: five searches in one
    conversation came back with no sources and ``Sign up and repeat your
    request.`` as the answer, after DuckDuckGo matched nothing and Tavily's
    keyless tier hit its daily cap. The model had to notice the sentence was not
    an answer and fall back to fetching pages itself.

    Returns the reason to record, or ``None``. Structural only: no wording is
    matched, so a change to the sentence cannot silently un-fix this.
    """
    if payload.get("_lo_served_blocks"):
        # Something WAS served -- a shopping, hotels, maps, media or web-result
        # block, or a top-level source list -- so this is a result carrying a
        # sign-in nudge, not a refusal. Only "the endpoint served nothing" is a
        # wall.
        return None
    upsell = payload.get("upsell_information")
    # Unwrap however many JSON-string layers the stream used. The SSE carries it
    # as a nested object in some responses and a JSON string in others, and a
    # DOUBLE-encoded string was observed to escape detection entirely -- the wall
    # then went back to being served as a search result, which is this fix's own
    # bug. Bounded rather than ``while isinstance(...)``: a malformed payload
    # must not spin.
    for _ in range(3):
        if not isinstance(upsell, str):
            break
        try:
            upsell = json.loads(upsell)
        except json.JSONDecodeError:
            upsell = None
            break
    if not isinstance(upsell, dict):
        return None
    name = str(upsell.get("name") or upsell.get("upsell_type") or "").strip()
    if not name:
        return None
    kind = str(upsell.get("upsell_type") or "").strip()
    detail = f"{name}/{kind}" if kind else name
    # Action FIRST, diagnostics last. The operator's transcript renders this on
    # one line and never wraps it, so at the collapsed budget (~width//3) only
    # the head survives: with the cause first, the reader saw "All configured
    # web sear…" and never reached "fetch a page directly", the one move the
    # model reading it can take unaided (design review D1, reproduced from saved
    # frames at 80-200 columns).
    return (
        f"Fetch a page directly, or set PERPLEXITY_API_KEY for keyed Sonar: the "
        f"anonymous tier refused this search (wall {detail})"
    )


async def _search_perplexity(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _keyed_leg(config_dir, "PERPLEXITY_API_KEY")
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
        # ``keyless=False`` is the whole point of this object: without a usage
        # report the ledger priced a BILLED Sonar call as ``free (anonymous)``
        # (round-1 review MAJOR-1), and the free/paid counters then printed that
        # as an explicit claim. The tokens come from the API when it sends them;
        # the request fee stands on its own when it does not.
        api_usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return SearchResponse(
            provider="perplexity",
            auth_mode="api-key",
            sources=_perplexity_sources(payload, limit),
            answer=_perplexity_answer(payload),
            request_id=str(payload.get("id") or "").strip() or None,
            usage=SearchUsage(
                input_tokens=api_usage.get("prompt_tokens"),
                output_tokens=api_usage.get("completion_tokens"),
                keyless=False,
            ),
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
    sources = _perplexity_sources(payload, limit)
    # A wall with sources alongside it is still a result: never discard pages
    # that were actually returned. A wall with nothing is a REFUSAL, and it is
    # raised rather than returned empty so the chain records this reason and
    # moves on -- and so a search whose every provider came back empty fails
    # loudly instead of handing the model a sentence to disbelieve.
    wall = _perplexity_authwall(payload)
    if wall is not None and not sources:
        raise RuntimeError(wall)
    return SearchResponse(
        provider="perplexity",
        auth_mode="anonymous",
        sources=sources,
        answer=_perplexity_answer(payload),
        request_id=str(payload.get("uuid") or request_id),
        # ``keyless=True`` is what makes this the free tier, and it is the only
        # thing that does: the keyed path above reports its tokens and its
        # ``keyless=False``, so the estimator routes each to its own rate. (This
        # comment used to say the keyed path left ``usage`` unset; it has reported
        # usage since round-1 review MAJOR-1, and the stale wording described the
        # opposite of the code -- round-2 review NIT-3.)
        usage=SearchUsage(keyless=True),
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


def parse_deepseek_search(payload: object, limit: int) -> tuple[list[SearchSource], str | None]:
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
    auth store because that is where ``login`` writes it; the store/env tier is
    checked first by the caller — the plaintext ``credentials.env`` leg is GONE
    (PR2a) — so this stays a pure store probe.

    The PROCESS-level store, not a fresh one. This is reached per search-key
    CHECK — the login probe and the settings surface both call it, and it ran on
    a path measured at seven connections to one ``auth.db`` per boot. It reads
    ``list_credentials``, which is a bare SELECT, and it never closes what it is
    handed (``shared_auth_store`` owns that; see its CLOSING note).
    """
    try:
        from local_operator.providers.auth_store import shared_auth_store

        return bool(shared_auth_store().list_credentials("deepseek"))
    except Exception:  # noqa: BLE001 -- an unreadable store is simply "unknown"
        return False


async def _resolve_deepseek_key(config_dir: Path | None) -> str:
    """The key the search will bill: the store/env tier first, then the login.

    ``read_only=True`` because a search must never decide model routing: it must
    not consume an OAuth rotation slot, clear session stickiness, or flip the
    auth tier the conversation's next request will resolve from.

    Resolved through the process-level store (see :func:`_deepseek_login_present`)
    with the same ``read_only`` contract, so this is a read of shared state
    rather than a connection opened to make one call.
    """
    key = _credential(config_dir, "DEEPSEEK_API_KEY")
    if key:
        return key
    try:
        from local_operator.providers.auth_store import shared_auth_store

        return await shared_auth_store().get_api_key("deepseek", read_only=True) or ""
    except Exception:  # noqa: BLE001 -- resolution failure is "no key"
        return ""


def _cached_deepseek_balance_verdict() -> bool | None:
    """The last balance verdict within its TTL, or None when there is none.

    Read-only: no ``global`` declaration, because this function never ASSIGNS the
    module global and flake8's F824 (rightly) treats a declaration that only
    reads as dead code.
    """
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
) -> tuple[dict[str, dict[str, Any]], SearchUsage, bool]:
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
    usage = _deepseek_usage(payload)
    # A pass cut off at the token cap is NOT the same result as a complete one:
    # the rows it lost are the later, lower-ranked pages, which is precisely the
    # part of the set the pass exists to triage. ``stop_reason`` is in the same
    # payload and was unused, so the truncation used to be invisible whenever at
    # least one row parsed -- a silent partial enrichment. Reported through the
    # same channel as a hard failure so the caller can say it out loud.
    truncated = str(payload.get("stop_reason") or "").lower() in {"max_tokens", "length"}
    text = "\n".join(
        str(block.get("text") or "")
        for block in (payload.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return parse_deepseek_evidence(text), usage, truncated


async def resolve_deepseek_key(config_dir: Path | None) -> str:
    """Public entry point for the DeepSeek key, for the page reader.

    A thin wrapper rather than a rename so the private resolver keeps its
    monkeypatchable name in tests while the reader gets a documented entry point.
    """
    return await _resolve_deepseek_key(config_dir)


def _deepseek_usage(payload: object) -> SearchUsage:
    """DeepSeek's billed tokens for one Messages call.

    Token counts are what the vendor reports, which makes the DeepSeek cost an
    estimate from a real measurement rather than a guess at a per-query rate.
    The server-side search count comes from `server_tool_use` because that is the
    work the turn actually did.
    """
    if not isinstance(payload, dict):
        return SearchUsage()
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return SearchUsage()
    server = usage.get("server_tool_use")
    server_searches = None
    if isinstance(server, dict):
        value = server.get("web_search_requests")
        if isinstance(value, int):
            server_searches = value

    def as_int(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) else None

    return SearchUsage(
        input_tokens=as_int("input_tokens"),
        output_tokens=as_int("output_tokens"),
        cache_read_tokens=as_int("cache_read_input_tokens"),
        server_searches=server_searches,
    )


async def _search_deepseek(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = await _resolve_deepseek_key(config_dir)
    if not key:
        raise RuntimeError("DeepSeek search needs an API key; run `local-operator login deepseek`")

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
    usage = _deepseek_usage(payload)
    #: Evidence rows keyed by URL. Bound BEFORE the try, not inside it: pyright
    #: reads a name assigned only in a ``try``/``except`` pair as possibly
    #: unbound at the ``_apply_deepseek_evidence`` call below, and the type
    #: checker is right that the invariant is non-obvious here even though both
    #: arms assign it. An explicit empty default states it once.
    evidence: dict[str, dict[str, Any]] = {}
    evidence_applied = False
    if settings.deepseek_evidence:
        # Enrichment only. A failed, truncated or unparseable evidence pass must
        # leave the sources exactly as the search returned them, because the
        # model can still fetch them; losing the search to a triage failure would
        # trade a usable answer for nothing. The reason is REPORTED rather than
        # swallowed -- an unenriched result that says why is diagnosable, and one
        # that silently looks like "this provider has no snippets" is not.
        try:
            evidence, evidence_usage, evidence_truncated = await _deepseek_evidence_pass(
                client, key, payload.get("content") or [], prompt
            )
        except Exception as error:  # noqa: BLE001 -- see comment above
            evidence = {}
            evidence_failure = f"deepseek evidence pass: {error}"
        else:
            # Both legs of an enriched search are billed tokens on the same
            # account, so one search's cost is the sum of the two calls.
            usage = usage.merge(evidence_usage)
            if not evidence:
                evidence_failure = "deepseek evidence pass: returned no usable rows"
            elif evidence_truncated:
                # Partial, and says so: the pages it did not reach are the ones
                # the pass was supposed to triage, so calling this complete would
                # overstate the enrichment the caller is looking at.
                evidence_failure = "deepseek evidence pass: hit the token cap, later pages unscored"
            else:
                evidence_failure = None
        sources, evidence_applied = _apply_deepseek_evidence(sources, evidence)
    else:
        # No pass ran, so nothing on screen came from one. (Stated rather than
        # defaulted: the footer reads this, and a pass that DID run may still
        # have applied nothing -- see ``_apply_deepseek_evidence``.)
        evidence_failure = None

    # Capture the page context whatever else happened: the blocks are what make
    # "read the pages this search found" possible without a fetch, and that is
    # independent of the evidence pass. A capture failure must not fail the
    # search -- the links are still usable.
    try:
        context = PAGE_CONTEXTS.store(
            provider="deepseek",
            query=query,
            blocks=payload.get("content") or [],
            sources=[source.model_dump(mode="json") for source in sources],
            enriched=bool(settings.deepseek_evidence and evidence),
        )
        page_context_id = context.context_id if context is not None else None
    except Exception:  # noqa: BLE001 -- capture is an optimization, not the result
        page_context_id = None

    return SearchResponse(
        provider="deepseek",
        auth_mode="api-key",
        sources=sources,
        answer=answer,
        evidence_applied=evidence_applied,
        request_id=str(response.headers.get("x-request-id") or "").strip() or None,
        failures=[evidence_failure] if evidence_failure else [],
        usage=usage,
        cost=estimate_search_cost("deepseek", usage),
        page_context_id=page_context_id,
    )


def _apply_deepseek_evidence(
    sources: list[SearchSource], evidence: dict[str, dict[str, Any]]
) -> tuple[list[SearchSource], bool]:
    """Merge the evidence rows onto the sources, then rank them by relevance.

    Ordering is the point of the pass: DeepSeek's own result order is opaque,
    while a judged relevance lets the model fetch the two or three pages that
    matter instead of reading all ten. Sources the pass did not cover keep their
    relative order at the END rather than being dropped -- the pass is a top-N
    view, not a verdict on the rest of the page set.
    """
    scored: list[SearchSource] = []
    unscored: list[SearchSource] = []
    applied = False
    for source in sources:
        row = evidence.get(source.url)
        if not row:
            unscored.append(source)
            continue
        quote = str(row.get("quote") or "").strip()
        summary = str(row.get("summary") or "").strip()
        if quote or summary:
            applied = True
        relevance = row.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
            relevance = None
        scored.append(
            source.model_copy(
                update={
                    # A verbatim quote is a real snippet; the summary is the
                    # fallback when a page could not be quoted.
                    "snippet": quote or summary or source.snippet,
                    # Clamped: ``relevance`` is whatever the model wrote, and a
                    # "500" would otherwise be rendered into the model's own
                    # context as ``[relevance 500/100]``.
                    "relevance": (
                        max(0, min(100, int(relevance))) if relevance is not None else None
                    ),
                }
            )
        )
    scored.sort(key=lambda item: item.relevance if item.relevance is not None else -1, reverse=True)
    # ``applied`` is tracked while merging rather than derived from the sources
    # afterwards: a snippet and a relevance are chosen independently, so
    # "some relevance is set" is not the same question as "some snippet came from
    # the pass", and the footer has to follow the snippet.
    return [*scored, *unscored], applied


async def _search_brave(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _credential(config_dir, "BRAVE_API_KEY")
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


#: Credential-free MCP search endpoints. Both answer a bare JSON-RPC
#: ``tools/call`` POST -- no MCP session handshake and no new dependency -- and
#: both document anonymous use as a free tier:
#:   * Exa: exa.ai/docs/get-started/exa-mcp ("No API key is required to get
#:     started") and exa-labs/exa-mcp-server, whose hosted server is described as
#:     "rate-limited free-tier access for users without a key". A 429 means that
#:     bucket is spent; the remedy the docs give (OAuth or an API key) is this
#:     repo's existing keyed REST path, which is why a stored key keeps using it.
#:   * Parallel: docs.parallel.ai/integrations/mcp/quickstart and
#:     parallel.ai/blog/free-web-search-mcp -- "The Search MCP is free to use --
#:     no API key required". A key only raises the limits.
#: A bare call rather than a session (``initialize`` + ``Mcp-Session-Id``) per
#: search: the handshake is cost with no benefit for a one-shot query, and a
#: future session requirement degrades LOUDLY here (HTTP 4xx, a JSON-RPC error
#: envelope, or ``result.isError`` all raise) instead of silently returning
#: nothing.
EXA_MCP_ENDPOINT = "https://mcp.exa.ai/mcp"
PARALLEL_MCP_ENDPOINT = "https://search.parallel.ai/mcp"

#: Both media types are REQUIRED by Exa: a JSON-only ``Accept`` is refused with
#: 406 "Client must accept both application/json and text/event-stream"
#: (reproduced 2026-09-18). Parallel accepts the same header.
_MCP_ACCEPT = "application/json, text/event-stream"
_MCP_USER_AGENT = "local-operator (+https://github.com/damianvtran/local-operator)"


async def _mcp_call(
    client: httpx.AsyncClient,
    endpoint: str,
    tool: str,
    arguments: dict[str, Any],
    *,
    label: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One JSON-RPC ``tools/call`` against an MCP search server."""
    response = await client.post(
        endpoint,
        headers={
            "Accept": _MCP_ACCEPT,
            "Content-Type": "application/json",
            "User-Agent": _MCP_USER_AGENT,
            **(headers or {}),
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        },
    )
    _ensure_success(label, response)
    return _mcp_payload(response, label)


def _mcp_payload(response: httpx.Response, label: str) -> dict[str, Any]:
    """The JSON-RPC envelope, from either framing these servers use.

    Exa answers SSE (``event: message`` + ``data: {...}``) and Parallel answers
    plain JSON -- both reproduced 2026-09-18 -- so a parser for only one of them
    would report the other as unparseable. An envelope that parses but carries an
    ``error``, or a ``result.isError``, is raised HERE: no parsing path may
    fabricate results out of an error envelope, and the service treats the raise
    as the fallback trigger.
    """
    payload: Any = None
    try:
        payload = json.loads(response.text)
    except ValueError:
        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                candidate = json.loads(raw)
            except ValueError:
                continue
            if isinstance(candidate, dict) and ("result" in candidate or "error" in candidate):
                payload = candidate
                break
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} MCP returned an unparseable response")
    error = payload.get("error")
    if isinstance(error, dict):
        raise RuntimeError(f"{label} MCP error {error.get('code')}: {error.get('message')}")
    result = payload.get("result")
    if isinstance(result, dict) and result.get("isError"):
        raise RuntimeError(f"{label} MCP error: {_mcp_text(payload) or 'unknown error'}")
    return payload


def _mcp_text(payload: dict[str, Any]) -> str:
    """The first text block of an MCP result -- both servers' payload carrier."""
    result = payload.get("result")
    content = result.get("content") if isinstance(result, dict) else None
    if not isinstance(content, list):
        return ""
    for block in content:
        if isinstance(block, dict) and str(block.get("text") or "").strip():
            return str(block["text"])
    return ""


#: Exa's MCP answer is a rendered TEXT blob rather than structured fields:
#: repeated ``Title:`` blocks (shape reproduced 2026-09-18). Splitting on the
#: block header is what makes the blob map onto the same ``SearchSource`` every
#: other provider builds, so the url/title/snippet/date caps apply here too.
_EXA_BLOCK_HEADER = re.compile(r"(?m)^Title:[ \t]*")


def parse_exa_mcp_text(text: str, limit: int) -> list[SearchSource]:
    """Map Exa's rendered text blob onto normalized sources."""
    sources: list[SearchSource] = []
    for block in _EXA_BLOCK_HEADER.split(text)[1:]:
        lines = block.splitlines()
        title = lines[0].strip() if lines else ""
        url = ""
        published: str | None = None
        snippet_lines: list[str] = []
        in_highlights = False
        for line in lines[1:]:
            if in_highlights:
                snippet_lines.append(line.rstrip())
                continue
            stripped = line.strip()
            if stripped.startswith("URL:"):
                # First URL in the block wins; later lines are body text.
                url = url or stripped[len("URL:") :].strip()
            elif stripped.startswith("Published:"):
                value = stripped[len("Published:") :].strip()
                published = None if value in ("", "N/A") else value
            elif stripped.startswith("Highlights:"):
                in_highlights = True
        source = _source(
            title=title,
            url=url,
            snippet="\n".join(value for value in snippet_lines if value.strip()),
            published_date=published,
        )
        if source is None:
            # No usable URL, or one past the shared cap: dropped by ``_source``,
            # the same rule every other provider's rows go through.
            continue
        sources.append(source)
        if len(sources) >= limit:
            break
    return sources


def parse_parallel_mcp_text(text: str, limit: int) -> list[SearchSource]:
    """Map Parallel's ``content[0].text``, a JSON *string*, onto sources."""
    try:
        payload = json.loads(text)
    except ValueError as error:
        raise RuntimeError("Parallel MCP returned an unparseable response") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Parallel MCP returned an unparseable response")
    sources: list[SearchSource] = []
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        excerpts = [
            str(value).strip() for value in item.get("excerpts") or [] if str(value).strip()
        ]
        source = _source(
            title=item.get("title"),
            url=item.get("url"),
            # The first two excerpts, joined: Parallel returns full page crops, and
            # ``_source`` caps what survives (2 000 chars) like every other row.
            snippet="\n".join(excerpts[:2]) or None,
            published_date=item.get("publish_date"),
        )
        if source is None:
            continue
        sources.append(source)
        # The service returns ~10 results regardless of ``limit`` (reproduced), so
        # the slice is ours to apply.
        if len(sources) >= limit:
            break
    return sources


async def _search_exa(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    # A stored key keeps the REST transport deliberately: it returns a
    # query-grounded ``summary`` as structured JSON, where MCP returns a text blob.
    # Routing keyed traffic to MCP would regress the keyed experience for nothing.
    leg_mode = LEG_MODE.get()
    keyed = (
        leg_mode in METERED_AUTH_MODES
        if leg_mode is not None
        else provider_auth_mode("exa", config_dir, settings) == "api-key"
    )
    if not keyed:
        payload = await _mcp_call(
            client,
            EXA_MCP_ENDPOINT,
            "web_search_exa",
            {"query": query, "type": "auto", "numResults": limit, "livecrawl": "fallback"},
            label="Exa",
        )
        return SearchResponse(
            provider="exa",
            auth_mode="keyless-mcp",
            sources=parse_exa_mcp_text(_mcp_text(payload), limit),
            # Keyless is a FREE TIER, not a missing price: without this flag the
            # ledger would charge the keyed $5/1,000 rate for an anonymous search.
            usage=SearchUsage(keyless=True),
        )
    key = _credential(config_dir, "EXA_API_KEY")
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


async def _search_parallel(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    """Keyless-first Parallel MCP search.

    COST HONESTY. Parallel's payload carries its own meter --
    ``_meta."parallel/usage" = [{"name": "sku_search", "count": 1,
    "cost_usd": 0.001}]`` -- and this transport deliberately does NOT book it as
    spend. That figure is Parallel's internal cost of goods at an account nobody
    has: there is no key, so there is nothing to bill, and the endpoint is
    documented as free (docs.parallel.ai/integrations/mcp/quickstart;
    parallel.ai/blog/free-web-search-mcp). Booking $0.001 would report money the
    operator never paid, which is the same class of error as inferring "free":
    the row is marked ``keyless`` and priced as the free tier it is. If a reviewer
    ever wants the vendor's meter surfaced, it belongs in ``SearchUsage`` -- never
    in ``usd``.
    """
    key = _keyed_leg(config_dir, "PARALLEL_API_KEY")
    payload = await _mcp_call(
        client,
        PARALLEL_MCP_ENDPOINT,
        "web_search",
        # ``session_id`` is omitted: it is optional (reproduced: 200 without it)
        # and this stateless transport has no session handle to pass.
        {"objective": query, "search_queries": [query]},
        label="Parallel",
        headers={"Authorization": f"Bearer {key}"} if key else None,
    )
    return SearchResponse(
        provider="parallel",
        auth_mode="api-key" if key else "keyless-mcp",
        sources=parse_parallel_mcp_text(_mcp_text(payload), limit),
        # Keyless is a free tier, and the keyed Search API has no rate we can
        # verify -- see ``cost.PROVIDER_USD_PER_SEARCH``, which marks it unpriced
        # rather than free.
        usage=SearchUsage(keyless=not key),
    )


async def _search_serpapi(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    key = _credential(config_dir, "SERPAPI_API_KEY", "SERP_API_KEY")
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
    _config_dir: Path | None,
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
    "radient": ProviderDefinition(
        "radient",
        "Radient",
        web_search_copy.radient_access(),
        web_search_copy.radient_detail(),
        # No search-specific secret: the leg uses the Radient sign-in
        # (`local-operator login radient`), never an environment variable.
        (),
        search_radient,
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
        "free keyless MCP / API key",
        (
            "Keyless MCP search (free, rate-limited); EXA_API_KEY switches to the "
            "REST API with query summaries"
        ),
        ("EXA_API_KEY",),
        _search_exa,
    ),
    "parallel": ProviderDefinition(
        "parallel",
        "Parallel",
        "free keyless MCP / API key",
        "Keyless MCP search (free, rate-limited); PARALLEL_API_KEY raises the limits",
        ("PARALLEL_API_KEY",),
        _search_parallel,
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


#: Which automatic band each provider joins when the user has NOT named it in
#: ``web_search.providers``. DECLARATION ORDER is the within-band order, so this
#: table is the single source of truth for the automatic half of the chain -- the
#: resolver, `search list` and `/search` all read it, and a provider cannot be
#: load-balanced by one surface while another describes it as off.
#:
#: ``rotate`` is the credential-free pool that round-robin balances; ``fallback``
#: is credential-free but best-effort (perplexity's anonymous tier is documented
#: "best-effort" and refused live on this machine during the investigation), so it
#: is never a first attempt; ``radient`` is Radient-hosted search billed to the
#: signed-in account's credits; ``metered`` (the API_KEY band) spends a key the
#: user stored. The dual-mode providers (``DUAL_MODE_FREE_MODES``) are listed by
#: their FREE leg's band: a stored key ADDS a second, keyed leg in ``metered``
#: rather than moving the provider (see :func:`provider_legs`).
#:
#: Native model search (DeepSeek today) is not in this table: its band depends on
#: the credential class, which :data:`NATIVE_VENDORS` and
#: :data:`NATIVE_MODEL_KEY_BAND` decide.
AUTO_PROVIDER_TIERS: dict[SearchProviderId, ProviderTier] = {
    "duckduckgo": "rotate",
    "tavily": "rotate",
    "exa": "rotate",
    "parallel": "rotate",
    "searxng": "rotate",
    "perplexity": "fallback",
    "radient": "radient",
    "brave": "metered",
    "serpapi": "metered",
}

#: Auth modes that spend a credential -- money, or a model turn billed to one. A
#: provider whose EFFECTIVE mode is one of these has a leg outside the free pool:
#: a keyed exa or tavily the user never listed must not be rotated into the free
#: pool, because that would spend on their behalf. ``radient-credits`` spends the
#: user's Radient balance.
METERED_AUTH_MODES = frozenset({"api-key", "login", "radient-credits"})

#: The free mode each dual-mode provider serves WITHOUT a key. A stored key never
#: removes this leg: "first use free providers where available" means the keyless
#: tier is tried in the FREE band and the key is spent only in the API_KEY band,
#: after native search and Radient. (Before the leg split a stored key moved the
#: WHOLE provider to the paid tail, so its free tier was never tried at all.)
DUAL_MODE_FREE_MODES: dict[SearchProviderId, str] = {
    "tavily": "keyless",
    "exa": "keyless-mcp",
    "parallel": "keyless-mcp",
    "perplexity": "anonymous",
}

#: Where native search through a PAY-AS-YOU-GO MODEL API KEY sits in the chain.
#:
#: POLICY, DECIDED BY THE OPERATOR 2026-11-24: after Radient. The request reads
#: "free providers -> native search on each provider -> Radient if credits are
#: available -> any API keys"; native search billed to a model API key (DeepSeek,
#: an Anthropic/OpenAI key) counts as an "API key" leg and sits in the
#: ``metered`` (API_KEY) band AFTER Radient, because it spends per search and can
#: cost more than Radient's flat $0.01. Flip it to ``"native"`` to put model-key
#: native legs before Radient; nothing else changes (the resolver and the matrix
#: tests read this constant). Plan/subscription native legs ALWAYS sit in
#: ``native``. DeepSeek is model-key class in both its modes: ``api-key`` and
#: ``login`` both store a pay-per-use DeepSeek key.
NATIVE_MODEL_KEY_BAND: ProviderTier = "metered"

#: Credential class of a native-search leg: ``plan`` is included in a
#: subscription the user already pays for (no marginal cash), ``key`` bills a
#: pay-as-you-go model API key per search.
NativeClass = Literal["plan", "key"]


@dataclass(frozen=True, slots=True)
class NativeVendor:
    """A model vendor whose own API has built-in web search.

    ``modes`` returns the legs this vendor can serve HERE as ``(mode, class)``
    pairs -- empty when no credential is stored. ``fee_usd`` is the approximate
    per-search price used only to order paid legs cheapest-first inside a band.
    The registry is the seam the native-search family (one module per vendor)
    plugs into; the band code reads nothing vendor-specific.
    """

    id: str
    modes: Callable[[Path | None, WebSearchSettings], list[tuple[str, NativeClass]]]
    fee_usd: float | None = None


def _deepseek_native_modes(
    config_dir: Path | None, settings: WebSearchSettings
) -> list[tuple[str, NativeClass]]:
    # Read through the module attribute so tests that pin ``provider_auth_mode``
    # classify DeepSeek from the same pinned state as every other provider.
    mode = provider_auth_mode("deepseek", config_dir, settings)
    return [(mode, "key")] if mode else []


#: Native-search vendors, in registration order. DeepSeek is the only one in this
#: release; the band code is vendor-agnostic so more slot in by registration.
#: ``fee_usd`` for DeepSeek is the measured median model turn (~$0.0024
#: off-peak, see the DeepSeek block above).
NATIVE_VENDORS: dict[str, NativeVendor] = {
    "deepseek": NativeVendor("deepseek", _deepseek_native_modes, fee_usd=0.0024),
}


@dataclass(frozen=True, slots=True)
class ChainLeg:
    """One attempt in the chain: a provider, the mode it serves in, and its band.

    WHY LEGS. The chain used to be a list of provider ids and each transport chose
    its own mode from whether a key was stored, which made "try Tavily's free tier
    first, and spend the Tavily key only after Radient" unexpressible: one id,
    one position. A leg pins the mode, so one provider can appear twice in
    different bands. ``mode == ""`` marks a LISTED provider that cannot serve; it
    is walked and reports ``not configured`` (see ``WebSearchService``).
    """

    provider: SearchProviderId
    mode: str
    band: ProviderTier
    #: ``free`` / ``best-effort`` / ``plan`` / ``paid`` / ``credits`` / ``setup``.
    cost_class: str = "free"

    @property
    def label(self) -> str:
        """How the failure digest names this leg: ``tavily`` or ``tavily (key)``.

        Only a dual-mode provider's KEYED leg is qualified, because only there
        can one id name two different attempts in one digest.
        """
        if self.provider in DUAL_MODE_FREE_MODES and self.mode in METERED_AUTH_MODES:
            return f"{self.provider} (key)"
        return self.provider


def _leg(provider_id: SearchProviderId, mode: str, band: ProviderTier) -> ChainLeg:
    if not mode:
        cost_class = "setup"
    elif band == "radient":
        cost_class = "credits"
    elif band == "metered":
        cost_class = "paid"
    elif band == "fallback":
        cost_class = "best-effort"
    else:
        cost_class = "free"
    return ChainLeg(provider_id, mode, band, cost_class)


def provider_legs(
    provider_id: SearchProviderId,
    config_dir: Path | None,
    settings: WebSearchSettings,
) -> list[ChainLeg]:
    """Every leg ``provider_id`` contributes on this install, in band order.

    Empty when it cannot serve at all. The order of the returned legs is the
    order of their bands, so a caller that keeps them in sequence never puts a
    provider's keyed leg ahead of its free one.
    """
    vendor = NATIVE_VENDORS.get(provider_id)
    if vendor is not None:
        legs: list[ChainLeg] = []
        for mode, native_class in vendor.modes(config_dir, settings):
            if native_class == "plan":
                legs.append(ChainLeg(provider_id, mode, "native", "plan"))
            else:
                band = NATIVE_MODEL_KEY_BAND
                legs.append(ChainLeg(provider_id, mode, band, "paid"))
        return legs
    mode = provider_auth_mode(provider_id, config_dir, settings)
    if not mode:
        return []
    free_mode = DUAL_MODE_FREE_MODES.get(provider_id)
    if free_mode is not None:
        free = _leg(provider_id, free_mode, AUTO_PROVIDER_TIERS[provider_id])
        if mode in METERED_AUTH_MODES:
            return [free, _leg(provider_id, mode, "metered")]
        return [free]
    if mode in METERED_AUTH_MODES and AUTO_PROVIDER_TIERS[provider_id] != "radient":
        return [_leg(provider_id, mode, "metered")]
    return [_leg(provider_id, mode, AUTO_PROVIDER_TIERS[provider_id])]


class ProviderBands(NamedTuple):
    """The chain split into its bands, in the order a search walks them.

    Every field is a list of :class:`ChainLeg`. ``prefix`` + ``rotate`` is the
    FREE pool round-robin rotates; nothing after it ever rotates.
    """

    prefix: list[ChainLeg]
    rotate: list[ChainLeg]
    fallback: list[ChainLeg]
    native: list[ChainLeg]
    radient: list[ChainLeg]
    metered: list[ChainLeg]

    def legs(self) -> list[ChainLeg]:
        return [
            *self.prefix,
            *self.rotate,
            *self.fallback,
            *self.native,
            *self.radient,
            *self.metered,
        ]


def provider_auth_mode(
    provider_id: SearchProviderId,
    config_dir: Path | None,
    settings: WebSearchSettings,
) -> str:
    """The STRONGEST transport this provider would use HERE, or "" when it cannot serve.

    One function answers two questions that used to be answered separately --
    "can this provider serve" (:func:`provider_available`) and "what would it use"
    -- so the chain and the status view cannot disagree about WHY a provider is or
    is not in play. For a dual-mode provider with a key this is ``api-key``; its
    free leg still exists (:func:`provider_legs` splits them).

    N3 (round 1) asked whether Tavily's OAuth MCP transport belongs here. It does
    not, deliberately: the delegate that serves it is injected by the harness
    (``WebSearchService.tavily_oauth_search``) and the MCP server entry lives in
    ``mcp.json``, so neither is visible to a function that reads config_dir and
    search settings -- and reading ``mcp.json`` per call would put a filesystem
    probe inside the status loop. It also cannot change the tier: Tavily's OAuth
    MCP is the user's own free tier, so it classifies ``rotate`` either way. If a
    future change makes an OAuth transport cost money, the mode has to be added
    here and the mcp config read once per session, not per call.
    """
    if provider_id == "duckduckgo":
        return "credential-free"
    if provider_id == "tavily":
        return "api-key" if _credential(config_dir, "TAVILY_API_KEY") else "keyless"
    if provider_id == "perplexity":
        return "api-key" if _credential(config_dir, "PERPLEXITY_API_KEY") else "anonymous"
    if provider_id == "exa":
        # Both modes exist and BOTH serve without a key, so this is a mode choice
        # rather than an availability one.
        return "api-key" if _credential(config_dir, "EXA_API_KEY") else "keyless-mcp"
    if provider_id == "parallel":
        return "api-key" if _credential(config_dir, "PARALLEL_API_KEY") else "keyless-mcp"
    if provider_id == "deepseek":
        # The search key IS the model key, and ``login`` writes it to the auth
        # store rather than to ``credentials.env``. Both tiers are checked, or
        # `search list` would report "setup needed" for a provider whose calls
        # would in fact succeed -- and, worse, the reverse for a keyless branch
        # that has no key at all.
        if _credential(config_dir, "DEEPSEEK_API_KEY"):
            return "api-key"
        return "login" if _deepseek_login_present() else ""
    if provider_id == "radient":
        # A PERSISTED login only: env never makes this paid leg appear.
        return "radient-credits" if radient_login_present(config_dir) else ""
    if provider_id == "searxng":
        endpoint = settings.searxng_endpoint
        return "self-hosted" if endpoint.startswith(("http://", "https://")) else ""
    definition = PROVIDERS[provider_id]
    return "api-key" if _credential(config_dir, *definition.credential_keys) else ""


def provider_available(
    provider_id: SearchProviderId,
    config_dir: Path | None,
    settings: WebSearchSettings,
) -> bool:
    """Whether the provider can make a request with current local configuration."""
    if provider_id in NATIVE_VENDORS:
        return bool(NATIVE_VENDORS[provider_id].modes(config_dir, settings))
    return provider_auth_mode(provider_id, config_dir, settings) != ""


def provider_tier(
    provider_id: SearchProviderId,
    config_dir: Path | None,
    settings: WebSearchSettings,
) -> ProviderTier | None:
    """The band of ``provider_id``'s FIRST leg on this install, or None.

    A keyed tavily/exa/parallel/perplexity answers with its FREE band: its first
    attempt is the keyless tier, and the keyed leg follows in ``metered``.
    """
    legs = provider_legs(provider_id, config_dir, settings)
    return legs[0].band if legs else None


def _native_order(legs: list[ChainLeg]) -> list[ChainLeg]:
    """Plan legs first (no marginal cash), then paid legs cheapest-first.

    Stable, so vendors with the same fee keep registration order.
    """

    def fee(leg: ChainLeg) -> float:
        vendor = NATIVE_VENDORS.get(leg.provider)
        return vendor.fee_usd if vendor is not None and vendor.fee_usd is not None else 1.0

    plan = [leg for leg in legs if leg.cost_class == "plan"]
    paid = sorted((leg for leg in legs if leg.cost_class != "plan"), key=fee)
    return [*plan, *paid]


def resolve_provider_bands(
    settings: WebSearchSettings,
    config_dir: Path | None,
) -> ProviderBands:
    """Split this install's provider chain into its bands, as legs.

    THE BAND INVARIANT: band order is fixed and no configuration crosses it --
    FREE (prefix + rotate, then best-effort fallback) < NATIVE < RADIENT <
    API_KEY (``metered``). That is the operator's "free first, then native search,
    then Radient if credits are available, then any API keys", and it generalises
    the older rule "no free leg is ever skipped for a paid one" (round-1
    M2/Q2/D1/U2).

    ``web_search.providers`` is a PRIORITY PREFIX whose authority is over ORDER
    WITHIN A BAND: a listed provider's legs go to the head of each band they land
    in. Its free and best-effort legs form the ``prefix`` (the head of the
    rotating pool, as before); a listed keyed leg heads ``metered``, a listed
    Radient heads ``radient``, a listed native leg heads its band. A listed
    provider that cannot serve stays in the prefix: trying it costs nothing, and
    the search loop reports `not configured` for it rather than silently dropping
    what the user asked for. ``excluded_providers`` removes EVERY leg of a
    provider and beats listing.

    Within API_KEY, dedicated search keys come first (in declaration order) and
    native model-key legs follow them: a search key is what the user bought for
    search; a model key serves search only incidentally.
    """
    excluded: set[str] = set(settings.excluded_providers)
    prefix: list[ChainLeg] = []
    listed_heads: dict[str, list[ChainLeg]] = {"native": [], "radient": [], "metered": []}
    for provider_id in settings.providers:
        if provider_id in excluded:
            continue
        legs = provider_legs(provider_id, config_dir, settings)
        if not legs:
            prefix.append(_leg(provider_id, "", AUTO_PROVIDER_TIERS.get(provider_id, "rotate")))
            continue
        for leg in legs:
            if leg.band in ("rotate", "fallback"):
                prefix.append(leg)
            else:
                listed_heads[leg.band].append(leg)
    auto: dict[str, list[ChainLeg]] = {
        "rotate": [],
        "fallback": [],
        "native": [],
        "radient": [],
        "metered": [],
    }
    native_key_legs: list[ChainLeg] = []
    # `settings.providers`, not `prefix`: a listed leg already placed in a later
    # band must not also be auto-joined there.
    skip = set(settings.providers) | excluded
    for provider_id in [*AUTO_PROVIDER_TIERS, *NATIVE_VENDORS]:
        if provider_id in skip:
            continue
        for leg in provider_legs(provider_id, config_dir, settings):  # type: ignore[arg-type]
            if provider_id in NATIVE_VENDORS and leg.band == "metered":
                native_key_legs.append(leg)
            else:
                auto[leg.band].append(leg)
    return ProviderBands(
        prefix,
        auto["rotate"],
        auto["fallback"],
        [*listed_heads["native"], *_native_order(auto["native"])],
        [*listed_heads["radient"], *auto["radient"]],
        [*listed_heads["metered"], *auto["metered"], *_native_order(native_key_legs)],
    )


def _unique(provider_ids: list[SearchProviderId]) -> list[SearchProviderId]:
    return list(dict.fromkeys(provider_ids))


def resolve_legs(
    settings: WebSearchSettings,
    config_dir: Path | None,
) -> list[ChainLeg]:
    """Every leg as a search will walk it, before rotation."""
    return resolve_provider_bands(settings, config_dir).legs()


def free_pool(
    settings: WebSearchSettings,
    config_dir: Path | None,
) -> list[SearchProviderId]:
    """The provider ids `round_robin` spreads its first attempt across: prefix + rotate.

    ONE pool, not two: the listed free legs in their listed order, then the
    auto-joined free band in declaration order. Rotating only the auto band -- which
    sits behind the whole prefix -- pinned the first attempt to ``providers[0]`` on
    every install, withdrawing the released strategy's whole purpose (round-1
    M1/Q1/D2/U1). Native, Radient, metered and best-effort legs are never in the
    pool: they do not rotate.
    """
    bands = resolve_provider_bands(settings, config_dir)
    return _unique([leg.provider for leg in (*bands.prefix, *bands.rotate)])


def resolve_providers(
    settings: WebSearchSettings,
    config_dir: Path | None,
) -> list[SearchProviderId]:
    """The chain's provider ids in first-attempt order, de-duplicated.

    For callers that need a provider set (bridge egress, the singleflight key,
    status rows); the walk itself uses :func:`resolve_legs`.
    """
    return _unique([leg.provider for leg in resolve_legs(settings, config_dir)])


def provider_state_label(status: ProviderStatus) -> str:
    """The state word for ``status``, rendered for display.

    The older words ARE their codes (English literals that predate the
    catalogue); the words this release adds are keyed and rendered in the active
    locale. Surfaces that look a word up in a table (the TUI ink map, the legend)
    use :func:`provider_state_code` instead, so a translated word never misses.
    """
    code = provider_state_code(status)
    keyed = _KEYED_STATE_WORDS.get(code)
    if keyed is None:
        return code
    from local_operator.i18n.text import tr

    return tr("web_search")(keyed())


def provider_state_code(status: ProviderStatus) -> str:
    """The ONE state vocabulary `search list` and `/search` print, as stable codes.

    Every word answers "what does the chain do with this provider": ``enabled``
    (listed and free) / ``enabled (paid)`` (listed, and therefore in the paid band)
    / ``auto free`` / ``auto best-effort`` / ``auto paid`` / ``excluded`` (the user
    said never) / ``needs setup`` (it cannot serve, so no chain would use it).
    Shared so the two surfaces cannot drift into describing one provider
    differently; the meanings live beside the words in ``STATE_MEANINGS``, which is
    also what the legend prints.
    """
    if status.excluded:
        return "excluded"
    if not status.available:
        # Nothing can use a provider with no credential or endpoint, so "cannot
        # serve" is a state of its own rather than a second column repeating it.
        # (`off` used to say this, which also read as the master switch and as the
        # old `disabled`; round-1 D6.)
        return "needs setup"
    if status.tier == "radient" and radient_credit_refusal_remaining() is not None:
        # Out of credits is a state of its own, listed or not: the leg is still in
        # the chain but is skipped locally until the remembered 402 lapses, and
        # that is the one fact the user has to act on.
        return "radient refused"
    if status.listed:
        # A listed leg whose transport spends money or a model turn is in the PAID
        # band, and plain `enabled` is what made the operator's own frame read as
        # "nothing paid is involved" (round-1 D1/U2). A listed BEST-EFFORT leg keeps
        # its tier visible for the same reason in the other direction: `perplexity`
        # is listed on the operator's config and is the tier the product documents
        # as walled in practice, so plain `enabled` would hide a fact the auto row
        # prints for the same provider on an install that did not list it
        # (round-2 U2-6).
        if status.tier == "radient":
            # Radient spends CREDITS, and the chain marks this leg `(credits)`:
            # `enabled (paid)` gave one leg two words and borrowed the API-key
            # band's vocabulary (PR #4553 design round 1, D2).
            return "enabled (credits)"
        if status.tier == "metered" or any(
            leg.cost_class == "paid" and leg.band == "native" for leg in status.legs
        ):
            return "enabled (paid)"
        if any(leg.band == "metered" for leg in status.legs):
            # A LISTED keyed dual-mode provider (Tavily is in the default order)
            # is tried free first and spends its key later in the API-key band;
            # plain `enabled` hid that second, paid leg. Listed twin of
            # `auto free, then key` (PR #4553 design round 1, D1).
            return "enabled, then key"
        return "enabled (best-effort)" if status.tier == "fallback" else "enabled"
    if status.tier == "radient":
        return "auto radient"
    if status.tier == "native":
        if status.legs and all(leg.cost_class == "plan" for leg in status.legs):
            return "auto native plan"
        return "auto native paid"
    if status.tier in ("rotate", "fallback") and any(leg.band == "metered" for leg in status.legs):
        # A stored key no longer replaces the free tier: the provider is tried
        # free first and its key later, and the row has to say both.
        return "auto free, then key"
    if status.tier == "rotate":
        return "auto free"
    if status.tier == "fallback":
        return "auto best-effort"
    return "auto paid"


#: Codes added with the leg split, rendered through the catalogue. The codes
#: themselves are stable identifiers (table keys), never shown.
_KEYED_STATE_WORDS: dict[str, Callable[[], Msg]] = {
    "auto radient": web_search_copy.state_auto_radient,
    "radient refused": web_search_copy.state_radient_refused,
    "auto native plan": web_search_copy.state_auto_native_plan,
    "auto native paid": web_search_copy.state_auto_native_paid,
    "auto free, then key": web_search_copy.state_auto_free_then_key,
    "enabled, then key": web_search_copy.state_enabled_then_key,
    "enabled (credits)": web_search_copy.state_enabled_credits,
}


#: What each state label MEANS for the user's next search. It lives beside the
#: labels themselves (not in the CLI and the TUI) for the same reason the labels
#: do: a surface that prints `auto paid` and a surface that explains it must not be
#: able to drift into saying different things. Declaration order is the legend's
#: order.
STATE_MEANINGS: dict[str, str | Msg] = {
    "enabled": "in your priority order",
    # "and Radient": the paid (API-key) band runs after the Radient band, the
    # operator's free -> native -> Radient -> API-key order (design round 1, N2).
    "enabled (paid)": "listed, and tried in the paid band after every free leg and Radient",
    # NOT "tried after the free pool": only a METERED listed leg is hoisted, so a
    # listed best-effort leg stays in the prefix -- inside the rotating pool, ahead
    # of the auto-joined free band this entry would be describing (round-3 D3-1,
    # measured on the operator's own config, where the listed perplexity is itself a
    # distinct first attempt). The auto-joined tier really does run after the pool,
    # and it has its own word.
    "enabled (best-effort)": "listed; the tier that gets walled most often, tried in the pool",
    "auto free": "with the free providers",
    "auto best-effort": "joined automatically; tried after the free pool",
    "auto paid": "tried after the free providers and Radient, never before a free leg",
    "excluded": "never used, whatever else is configured",
    # NOT "not in any chain": a LISTED leg with no credential is deliberately kept
    # in the chain and walked, where it fails locally in the availability check --
    # so the old wording was a claim the `chain:` line above it disproved
    # (round-1 N5, round-2 Q2-1/D2-2/U2-3).
    "needs setup": "cannot serve yet; a listed one is still tried and reports `not configured`",
    # Keyed meanings for the codes the leg split added (rendered by
    # ``state_meaning``; the older English entries predate the catalogue).
    "auto free, then key": web_search_copy.meaning_auto_free_then_key(),
    "enabled, then key": web_search_copy.meaning_enabled_then_key(),
    # A listed Radient leg sits in the same band as an auto-joined one, so it
    # means the same thing (design round 1, D2).
    "enabled (credits)": web_search_copy.meaning_auto_radient(),
    "auto native plan": web_search_copy.meaning_auto_native_plan(),
    "auto native paid": web_search_copy.meaning_auto_native_paid(),
    "auto radient": web_search_copy.meaning_auto_radient(),
    "radient refused": web_search_copy.meaning_radient_refused(),
}


def state_meaning(code: str) -> str:
    """What a state code means, in the active locale where it is keyed."""
    meaning = STATE_MEANINGS[code]
    if isinstance(meaning, Msg):
        from local_operator.i18n.text import tr

        return tr("web_search")(meaning)
    return meaning


def state_legend(statuses: list[ProviderStatus]) -> str:
    """One line defining the state words THIS listing paints, from the table above.

    It prints the MEANINGS, not just the words: the reader who lands on `/search`
    cold needs to know what `auto best-effort` or `(paid)` costs them, and the
    wording already exists in this module -- so the line carries it rather than
    making them infer it from the row above (round-2 U2-5, D2-4).

    Scoped to the words on screen, because the full table costs 526 cells on one
    header line while a listing paints only a few of its words, most of the table
    defining states the install does not have -- and the header rows of the same
    listing are what a narrow terminal folds away first (round-3 D3-3). Order is
    first appearance, so the line reads in the order the rows above it do.
    """
    codes: list[str] = []
    labels: dict[str, str] = {}
    for status in statuses:
        code = provider_state_code(status)
        if code not in codes:
            codes.append(code)
            labels[code] = provider_state_label(status)
    return "States: " + " · ".join(f"{labels[code]} = {state_meaning(code)}" for code in codes)


def provider_setup_hint(provider_id: SearchProviderId) -> str:
    """The command that makes ``provider_id`` servable, for copy that must say it."""
    if provider_id == "searxng":
        return f"`local-operator search setup {provider_id} --endpoint <url>`"
    if provider_id == "radient":
        # Radient search rides the Radient sign-in; there is nothing to set up
        # under `search setup`.
        return "`local-operator login radient`"
    if provider_id == "deepseek":
        # DeepSeek search has no search-specific secret: it bills the model key, and
        # `login` is where that key lives. `search setup` says so too.
        return f"`local-operator login deepseek` (or `local-operator search setup {provider_id}`)"
    # `.get`, not `PROVIDERS[...]`: this runs on the REFUSAL path, and an id that is
    # not in the catalogue must produce a sentence rather than a KeyError inside the
    # error message (round-2 N6 -- the validation that keeps bad ids out lives at the
    # tool/CLI boundary, which is not where a copy helper should depend on it).
    definition = PROVIDERS.get(provider_id)
    keys = definition.credential_keys if definition is not None else ()
    suffix = f" ({', '.join(keys)})" if keys else ""
    return f"`local-operator search setup {provider_id}`{suffix}"


def provider_refusal(
    provider_id: SearchProviderId,
    settings: WebSearchSettings,
    config_dir: Path | None,
) -> str:
    """Why this session will not use ``provider_id``, and the command that fixes it.

    The old advice was one sentence naming `search enable`, which stopped working
    the moment `enable` came to mean "clear an exclusion" rather than "append to
    the chain": a provider with no credential was answered with a command that
    could not help, twice (round-1 U3). The reason is already known here, so the
    copy branches on it instead of naming the wrong verb.
    """
    if provider_id in settings.excluded_providers:
        return (
            f"{provider_id!r} is excluded, so it will not be used by any search. "
            f"Run `local-operator search enable {provider_id}` to allow it again."
        )
    return (
        f"{provider_id!r} cannot serve yet on this install, so it is not in the "
        f"chain. Run {provider_setup_hint(provider_id)}."
    )


def provider_landing_line(provider_id: SearchProviderId, status: ProviderStatus) -> str:
    """The one sentence both surfaces print after `search enable <pid>`.

    Returned WITHOUT a closing period: the two surfaces end it differently (the CLI
    full stop, the TUI `; applies now`), and a second copy for the tail is how the
    two sentences drifted apart before (round-1 U6).
    """
    code = provider_state_code(status)
    state = provider_state_label(status)
    if code == "needs setup":
        # "enabled (off; not usable yet)" told the user their action had both
        # worked and not worked and named no next step (round-1 U4). The honest
        # reply says the provider is allowed and names what it still needs.
        return (
            f"{provider_id} is allowed, but no search can use it yet: run "
            f"{provider_setup_hint(provider_id)}"
        )
    meaning = state_meaning(code)
    if code.startswith("enabled"):
        # The state word IS the verb here, so printing it again stutters:
        # "brave enabled (enabled; in your priority order)" (round-1 D3/U10).
        return f"{provider_id} enabled ({meaning})"
    return f"{provider_id} enabled ({state}; {meaning})"


def chain_leg_marker(status: ProviderStatus) -> str:
    """The parenthetical for one chain leg, or "" when the leg needs no warning.

    ONE table for the two surfaces: the CLI prints the marker as text and the TUI
    paints it in the ink its fact deserves, and both read this function, so a
    marker cannot appear on one surface and not the other (round-2 D2-3).

    - `(paid)` -- the effective transport spends money or a model turn.
    - `(setup needed)` -- the leg cannot serve yet, so a reader who sees it in the
      chain (a LISTED leg is walked and fails locally, which is deliberate) is told
      why the same provider's row says `needs setup` (round-2 D2-2/Q2-1/U2-3).
    - `(best-effort)` -- the best-effort tier, walled often in practice. Printed for
      a listed leg as well as an auto-joined one: the operator lists `perplexity`,
      and hiding its tier was the disclosure gap U2-6 found.
    """
    if status.legs:
        return _leg_marker(status.legs[0].cost_class)[0]
    if status.tier == "metered":
        return "(paid)"
    if not status.available:
        return "(setup needed)"
    if status.tier == "fallback":
        return "(best-effort)"
    return ""


def _leg_marker(cost_class: str) -> tuple[str, str]:
    """``(marker text, ink token)`` for one leg's cost class.

    `(plan)` and `(credits)` are keyed (added with the leg split); the older
    markers are the English literals the surfaces always printed.
    """
    if cost_class == "paid":
        return "(paid)", "warning"
    if cost_class == "setup":
        return "(setup needed)", "muted"
    if cost_class == "best-effort":
        return "(best-effort)", "muted"
    if cost_class in ("plan", "credits"):
        from local_operator.i18n.text import tr

        keyed = (
            web_search_copy.marker_plan()
            if cost_class == "plan"
            else web_search_copy.marker_credits()
        )
        # `(credits)` spends money, so it takes the same amber as `(paid)`.
        return tr("web_search")(keyed), "muted" if cost_class == "plan" else "warning"
    return "", ""


def chain_leg_entries(statuses: list[ProviderStatus]) -> list[tuple[str, str, str]]:
    """Every chain LEG in try order as ``(label, marker, ink token)``.

    Built from legs, not providers: a keyed Tavily is two entries -- `Tavily`
    in the free pool and `Tavily (paid)` in the API_KEY band -- which is the
    chain the search actually walks. The CLI prints label+marker; the TUI paints
    the marker in its token, so both surfaces read this one function.
    """
    legs = [
        (leg.position, status.label, leg.cost_class)
        for status in statuses
        if status.enabled
        for leg in status.legs
    ]
    refused = radient_credit_refusal_remaining() is not None
    entries: list[tuple[str, str, str]] = []
    for _position, label, cost_class in sorted(legs):
        if cost_class == "credits" and refused:
            # A remembered 402 makes the transport refuse at once, so an amber
            # `(credits)` would promise a leg that will not be tried; the dim
            # `(no credits)` says it is skipped (PR #4553 design round 1, D5).
            from local_operator.i18n.text import tr

            entries.append((label, tr("web_search")(web_search_copy.marker_no_credits()), "muted"))
            continue
        marker, token = _leg_marker(cost_class)
        entries.append((label, marker, token))
    return entries


#: Ink token for each marker, so the chain row emphasises what the row below it
#: emphasises: `(paid)` is the same amber as `enabled (paid)`, and the other two
#: carry the same weight as the state words they echo (round-2 D2-3).
CHAIN_MARKER_TOKENS: dict[str, str] = {
    "(paid)": "warning",
    "(setup needed)": "muted",
    "(best-effort)": "muted",
}


def chain_label(statuses: list[ProviderStatus]) -> str:
    """The whole chain in try order: `DuckDuckGo → Exa → DeepSeek (paid)`.

    It spans EVERY leg in the chain, listed ones included, because the summary it
    replaces reported only the auto-joined bands -- so on an install that listed a
    paid provider it printed `paid: (none)` while a model-turn leg sat in the
    chain (round-1 D1). See :func:`chain_leg_marker` for what a leg is marked with
    and why.
    """
    entries = chain_leg_entries(statuses)
    if not entries:
        return "(none)"
    return " → ".join(f"{label} {marker}".strip() for label, marker, _token in entries)


def provider_order_note(
    provider_ids: list[SearchProviderId],
    settings: WebSearchSettings,
    config_dir: Path | None,
) -> str:
    """The tail of the `search order` receipt, derived from where each id LANDS.

    `search order` writes the priority prefix, and "tried first" was true of that
    verb until the paid legs were hoisted behind every free one: naming a paid
    provider now produces a chain that reaches the free legs first, so a receipt
    promising otherwise contradicts the chain the command just wrote -- printed by
    the same CLI that prints the truth two commands later (round-2 U2-1, carried by
    the code reviewer). The sentence is therefore read off the resolver, and it
    keeps both halves: the ids that DO lead, and the ones that are held back.
    """
    bands = resolve_provider_bands(settings, config_dir)
    # A provider LEADS when one of its legs landed in the prefix; every other
    # named provider landed in a later band (native, Radient or API keys).
    in_prefix = {leg.provider for leg in bands.prefix}
    later = {leg.provider for leg in (*bands.native, *bands.radient, *bands.metered)}
    leads = [provider_id for provider_id in provider_ids if provider_id in in_prefix]
    held = [
        provider_id
        for provider_id in provider_ids
        if provider_id not in in_prefix and provider_id in later
    ]
    clauses: list[str] = []
    if leads:
        clauses.append("tried first")
    if held:
        joined = ", ".join(held)
        clauses.append(
            f"{joined} is paid and runs after the free legs"
            if len(held) == 1
            else f"{joined} are paid and run after the free legs"
        )
    clauses.append("any exclusion named here was cleared")
    return "(" + "; ".join(clauses) + ")"


def _provider_detail(
    provider_id: SearchProviderId, config_dir: Path | None, settings: WebSearchSettings
) -> str | Msg:
    """The row's detail text, which for Radient depends on the sign-in state.

    A signed-in Radient row that still said "sign in with `local-operator login
    radient`" read as "you are not signed in" beside its amber state word, so the
    sign-in clause is printed only while the row needs it (PR #4553 design round
    1, D4).
    """
    if provider_id == "radient" and provider_available(provider_id, config_dir, settings):
        return web_search_copy.radient_detail_signed_in()
    return PROVIDERS[provider_id].detail


def provider_statuses(
    settings: WebSearchSettings,
    config_dir: Path | None,
) -> list[ProviderStatus]:
    """Every provider: the chain in TRY order first, then the rest by catalogue.

    ``enabled`` reports membership of THIS session's resolved chain (listed, or
    auto-joined), which is what a reader of `search list` is asking; ``listed`` and
    ``excluded`` keep the stored state visible beside it.

    The rows follow the chain rather than ``PROVIDER_IDS`` so they corroborate the
    chain line above them: in catalogue order the DeepSeek row printed above the
    Perplexity row while the chain put it last (round-1 D1). A provider outside the
    chain keeps its catalogue position, which is the only order left for it.
    """
    chain = resolve_legs(settings, config_dir)
    position: dict[str, int] = {}
    leg_rows: dict[str, list[ChainLegStatus]] = {}
    for index, leg in enumerate(chain):
        position.setdefault(leg.provider, index)
        leg_rows.setdefault(leg.provider, []).append(
            ChainLegStatus(mode=leg.mode, band=leg.band, position=index, cost_class=leg.cost_class)
        )
    statuses = [
        ProviderStatus(
            id=provider_id,
            label=PROVIDERS[provider_id].label,
            enabled=provider_id in position,
            available=provider_available(provider_id, config_dir, settings),
            access=provider_copy(PROVIDERS[provider_id].access),
            detail=provider_copy(_provider_detail(provider_id, config_dir, settings)),
            listed=provider_id in settings.providers,
            excluded=provider_id in settings.excluded_providers,
            # The band the provider is FIRST tried in; a listed leg sits where
            # the resolver put it, which is what the chain row shows.
            tier=(
                leg_rows[provider_id][0].band
                if provider_id in leg_rows and leg_rows[provider_id][0].mode
                else provider_tier(provider_id, config_dir, settings) or ""
            ),
            mode=provider_auth_mode(provider_id, config_dir, settings),
            legs=leg_rows.get(provider_id, []),
        )
        for provider_id in PROVIDER_IDS
    ]
    return sorted(
        statuses,
        key=lambda status: (
            (0, position[status.id])
            if status.id in position
            else (1, PROVIDER_IDS.index(status.id))
        ),
    )
