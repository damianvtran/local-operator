"""Anthropic Messages ``web_search_20250305`` parsing, shared by DeepSeek and Anthropic.

MOVED, NOT REWRITTEN. This is the parser DeepSeek's native search has used
since it shipped (``providers.parse_deepseek_search``), lifted here with the
vendor name as a parameter so the Anthropic transport reads the same wire with
the same rules. ``providers.parse_deepseek_search`` is now a wrapper that passes
``"DeepSeek"``, which renders every error string byte-for-byte as before -- the
existing DeepSeek tests are the behaviour-neutrality proof.

The rules (see the DeepSeek block in ``providers.py`` for the measurements):

* sources come from ``web_search_tool_result`` items only; the turn's prose is
  a synthesized answer, returned as ``answer`` and never parsed for URLs;
* an error-shaped tool result (a dict instead of a list) carries no sources;
* a response with no usable source RAISES so the chain falls through;
* dedupe by URL, because ``max_uses > 1`` can surface one page twice.

Usage is Anthropic-shaped: ``input_tokens`` EXCLUDES ``cache_read_input_tokens``
(the two are added when pricing; ``cost._deepseek_cost`` spells out why).
"""

from __future__ import annotations

from typing import Any

from local_operator.web_search.models import SearchSource, SearchUsage

#: The server tool, as both vendors name it on this wire.
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"


def citation_snippets(blocks: list[Any]) -> dict[str, str]:
    """Excerpts the answer text cites, keyed by source URL.

    Empty in practice on DeepSeek (no ``citations`` arrive); Anthropic does
    attach ``cited_text`` to its answer blocks, which is the only snippet this
    wire can ever give a source.
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


def messages_answer(blocks: list[Any]) -> str | None:
    """The turn's prose, which is an answer summary and never a result list."""
    parts = [
        str(block.get("text") or "").strip()
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    answer = "\n\n".join(part for part in parts if part).strip()
    return answer or None


def parse_messages_search(
    payload: object, limit: int, label: str
) -> tuple[list[SearchSource], str | None]:
    """Normalize one Messages response into ``(sources, answer)``; raises when sourceless."""
    from local_operator.web_search.providers import _source

    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned a non-object response")
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise RuntimeError(f"{label} returned no content blocks")

    snippets = citation_snippets(blocks)
    sources: list[SearchSource] = []
    seen: set[str] = set()
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "web_search_tool_result":
            continue
        items = block.get("content")
        if not isinstance(items, list):
            # Error-shaped tool results (a dict instead of a list) carry no
            # sources; the no-result check below turns a wholly failed search
            # into the fallback trigger.
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
            f"{label} returned no web_search results; the request may not have "
            "triggered native search"
        )
    return sources, messages_answer(blocks)


def tool_result_error(payload: object) -> str | None:
    """The ``error_code`` of an error-shaped ``web_search_tool_result``, if any.

    Anthropic reports a refused search (``max_uses_exceeded``, ``unavailable``,
    ``too_many_requests``...) as a tool-result OBJECT instead of a list. The
    parser already skips it; surfacing the code lets the failure digest say why
    a leg returned nothing rather than "no web_search results".
    """
    if not isinstance(payload, dict):
        return None
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "web_search_tool_result":
            content = block.get("content")
            if isinstance(content, dict) and content.get("error_code"):
                return str(content["error_code"])
    return None


def messages_usage(payload: object) -> SearchUsage:
    """Billed tokens and server-side searches for one Messages call.

    The search count comes from ``server_tool_use.web_search_requests`` because
    that is the work the turn actually did (and what Anthropic bills the fee on).
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
