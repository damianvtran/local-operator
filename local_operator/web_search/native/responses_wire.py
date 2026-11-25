"""OpenAI-Responses-style ``web_search`` parsing, shared by openai, xai and alibaba.

Three vendors serve native search through the same Responses shape
(``POST .../responses`` with ``tools: [{"type": "web_search"}]``):

* ``output[]`` items of ``type: "web_search_call"`` carry the search; with
  ``include: ["web_search_call.action.sources"]`` (OpenAI) -- or by default
  (Alibaba) -- ``action.sources[]`` lists every page the search consulted;
* the ``message`` item's ``output_text`` parts carry ``url_citation``
  annotations (``url``, ``title``) for the pages the answer actually cites;
* xAI additionally returns a top-level ``citations`` list of URLs.

SOURCE ORDER: cited pages first (they carry a title and are the ones the
answer leaned on), then the remaining consulted pages in search order, deduped
by URL. The answer prose is returned as ``answer`` and NEVER mined for URLs --
the DeepSeek transport's rule: prose is a summary, not a result list, and a
response with no source at all raises so the chain falls through instead of
handing the model a sourceless paragraph.

USAGE SEMANTICS (load-bearing for pricing): Responses ``input_tokens``
INCLUDES ``input_tokens_details.cached_tokens`` (OpenAI-shaped), unlike the
Anthropic wire where cache reads are counted apart. :func:`responses_usage`
records both as reported and :mod:`.pricing` is told which wire it is pricing.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from local_operator.web_search.models import SearchSource, SearchUsage

#: The ``include`` entry that makes OpenAI return every consulted source on the
#: ``web_search_call`` item (without it only cited pages are visible).
SOURCES_INCLUDE = "web_search_call.action.sources"


def _source(title: object, url: object) -> SearchSource | None:
    from local_operator.web_search.providers import _source as build

    return build(title=title, url=url)


def _items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = payload.get("output")
    return [item for item in output if isinstance(item, dict)] if isinstance(output, list) else []


def responses_answer(payload: dict[str, Any]) -> str | None:
    """The assistant's ``output_text`` prose, joined; ``None`` when there is none."""
    parts: list[str] = []
    for item in _items(payload):
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = str(part.get("text") or "").strip()
                if text:
                    parts.append(text)
    answer = "\n\n".join(parts).strip()
    return answer or None


def search_call_count(payload: dict[str, Any]) -> int:
    """How many server-side searches ran: the vendor's own count, else the call items.

    Alibaba reports the BILLED count at ``tool_usage.web_search.num_requests``
    (its docs: the number of output items is not a billing count); xAI reports
    ``usage.server_side_tool_usage_details.web_search_calls``. Otherwise each
    ``web_search_call`` item is one search.
    """
    tool_usage = payload.get("tool_usage")
    if isinstance(tool_usage, dict):
        web = tool_usage.get("web_search")
        if isinstance(web, dict) and isinstance(web.get("num_requests"), int):
            return int(web["num_requests"])
    usage = payload.get("usage")
    if isinstance(usage, dict):
        details = usage.get("server_side_tool_usage_details")
        if isinstance(details, dict) and isinstance(details.get("web_search_calls"), int):
            return int(details["web_search_calls"])
    return sum(1 for item in _items(payload) if item.get("type") == "web_search_call")


def parse_responses_search(
    payload: object, limit: int, label: str
) -> tuple[list[SearchSource], str | None]:
    """Normalize one Responses payload into ``(sources, answer)``; raises when sourceless.

    ``label`` is the vendor's brand name for the error text.
    """
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned a non-object response")
    error = payload.get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else error
        raise RuntimeError(f"{label} search failed: {str(message)[:500]}")

    cited: list[tuple[object, object]] = []
    consulted: list[tuple[object, object]] = []
    for item in _items(payload):
        kind = item.get("type")
        if kind == "web_search_call":
            action = item.get("action")
            sources = action.get("sources") if isinstance(action, dict) else None
            for entry in sources or []:
                if isinstance(entry, dict):
                    consulted.append((entry.get("title"), entry.get("url")))
        elif kind == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                for note in part.get("annotations") or []:
                    if isinstance(note, dict) and note.get("type") == "url_citation":
                        cited.append((note.get("title"), note.get("url")))
    for url in payload.get("citations") or []:
        # xAI's flat list: URLs only, so they rank with the consulted pages.
        if isinstance(url, str):
            consulted.append((None, url))

    # A consulted entry may carry the title a later citation lacks (or the
    # reverse), so titles are merged per URL before the order is decided.
    titles: dict[str, object] = {}
    for title, url in (*cited, *consulted):
        key = str(url or "").strip()
        if key and title and key not in titles:
            titles[key] = title

    result: list[SearchSource] = []
    seen: set[str] = set()
    for _title, url in (*cited, *consulted):
        key = str(url or "").strip()
        if not key or key in seen:
            continue
        source = _source(titles.get(key), key)
        if source is None:
            continue
        seen.add(key)
        result.append(source)
        if len(result) >= limit:
            break

    if not result:
        raise RuntimeError(
            f"{label} returned no web_search results; the request may not have "
            "triggered native search"
        )
    return result, responses_answer(payload)


def responses_usage(payload: object) -> SearchUsage:
    """Tokens and searches as the vendor reported them (``input`` includes cached)."""
    if not isinstance(payload, dict):
        return SearchUsage()
    usage = payload.get("usage")
    searches = search_call_count(payload) or None
    if not isinstance(usage, dict):
        return SearchUsage(server_searches=searches)

    def as_int(value: object) -> int | None:
        return value if isinstance(value, int) else None

    details = usage.get("input_tokens_details")
    cached = as_int(details.get("cached_tokens")) if isinstance(details, dict) else None
    return SearchUsage(
        input_tokens=as_int(usage.get("input_tokens")),
        output_tokens=as_int(usage.get("output_tokens")),
        cache_read_tokens=cached or None,
        server_searches=searches,
    )


async def collect_sse_response(response: httpx.Response, label: str) -> dict[str, Any]:
    """Fold a streamed Responses call (the ChatGPT Codex backend) into one payload.

    The Codex backend only streams. Its terminal ``response.completed`` event
    carries the final response object; with ``store: false`` its ``output`` can
    arrive empty, so every ``response.output_item.done`` item is kept as well
    and used when the terminal object has none. ``response.failed`` and
    ``error`` events raise the vendor's own sentence. A stream that ends with no
    terminal event raises too -- a truncated stream is not a result.
    """
    items: list[dict[str, Any]] = []
    async for data in _iter_sse_json(response):
        kind = data.get("type")
        if kind == "response.output_item.done" and isinstance(data.get("item"), dict):
            items.append(data["item"])
        elif kind in ("response.completed", "response.incomplete"):
            final = data.get("response")
            final = dict(final) if isinstance(final, dict) else {}
            if not _items(final):
                final["output"] = items
            return final
        elif kind in ("response.failed", "error"):
            body = data.get("response") if kind == "response.failed" else data
            error = body.get("error") if isinstance(body, dict) else None
            message = (
                error.get("message")
                if isinstance(error, dict)
                else (body.get("message") if isinstance(body, dict) else None)
            )
            raise RuntimeError(f"{label} search failed: {str(message or kind)[:500]}")
    raise RuntimeError(f"{label} search stream ended without a completed response")


async def _iter_sse_json(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """``data:`` payloads as JSON objects; keep-alives and partial lines skipped.

    The ``imagegen/rungs_openai_sub.py`` reader's tolerance, for the same route:
    a parser that died on an unknown line would fail searches the backend
    considers fine; the terminal state is carried by the events themselves.
    """
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        try:
            data = json.loads(body)
        except ValueError:
            continue
        if isinstance(data, dict):
            yield data
