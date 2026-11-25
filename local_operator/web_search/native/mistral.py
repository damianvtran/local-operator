"""Mistral native search: the Conversations API with the built-in ``web_search`` tool.

``web_search`` exists only on ``POST /v1/conversations`` (and Agents), not on
Chat Completions, because only Conversations returns the search references
(docs.mistral.ai "Websearch", fetched 2026-11-24). Sources arrive as
``tool_reference`` chunks (``title``, ``url``) inside the ``message.output``
entry's ``content``, interleaved with ``text`` chunks that form the answer.

PAID CLASS, AFTER RADIENT: $30 per 1,000 calls plus tokens (Mistral's published
tool price, via third-party summaries; re-verify), the most expensive leg in
the family, so ``_native_order`` sorts it last among paid legs.

USAGE: ``prompt_tokens`` + ``completion_tokens`` exclude ``connector_tokens``
(the documented example: 188 + 55 + 7,112 = ``total_tokens`` 7,355). The
connector tokens are the retrieved page text fed to the model and are counted
as input here; leaving them out would understate the token half ~40x.

NO-CRED BUILD: no Mistral credential was stored on the probe machine, so the
request and parser follow the documented example (``fixtures/native/
mistral_conversations_documented.json``). ``store: false`` keeps the search out
of the account's conversation history.
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
    leg_mode,
)

LABEL = "Mistral"
NAMESPACE = "mistral"
ENV_KEY = "MISTRAL_API_KEY"
CONVERSATIONS_URL = "https://api.mistral.ai/v1/conversations"
#: The documented host for the tool and Mistral's cheapest general model. The
#: shipped registry does not price the ``-latest`` alias, so the token half is
#: reported as a floor (``pricing.native_cost``) rather than guessed.
HOST_MODEL = "mistral-small-latest"
MAX_OUTPUT_TOKENS = 512
FEE_USD = 0.03

KEY_MODE = "api-key"


def _modes(config_dir: Path | None, _settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    if credentials.has_api_key(NAMESPACE, config_dir, ENV_KEY):
        return [(KEY_MODE, "key")]
    return []


def _outputs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = payload.get("outputs")
    return (
        [entry for entry in outputs if isinstance(entry, dict)] if isinstance(outputs, list) else []
    )


def parse_conversation_search(payload: object, limit: int) -> tuple[list[SearchSource], str | None]:
    """``(sources, answer)`` from one Conversations response; raises when sourceless."""
    from local_operator.web_search.providers import _source

    if not isinstance(payload, dict):
        raise RuntimeError(f"{LABEL} returned a non-object response")
    sources: list[SearchSource] = []
    seen: set[str] = set()
    texts: list[str] = []
    for entry in _outputs(payload):
        if entry.get("type") != "message.output":
            continue
        content = entry.get("content")
        if isinstance(content, str):
            # A plain-string content is an answer with no references.
            texts.append(content)
            continue
        for chunk in content or []:
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") == "text":
                texts.append(str(chunk.get("text") or ""))
            elif chunk.get("type") == "tool_reference" and len(sources) < limit:
                url = str(chunk.get("url") or "").strip()
                if not url or url in seen:
                    continue
                source = _source(title=chunk.get("title"), url=url)
                if source is not None:
                    seen.add(url)
                    sources.append(source)
    if not sources:
        raise RuntimeError(
            f"{LABEL} returned no web_search references; the request may not have "
            "triggered native search"
        )
    answer = "".join(texts).strip()
    return sources, answer or None


def conversation_usage(payload: object) -> SearchUsage:
    if not isinstance(payload, dict):
        return SearchUsage()
    usage = payload.get("usage")
    executions = sum(
        1
        for entry in _outputs(payload)
        if entry.get("type") == "tool.execution" and entry.get("name") == "web_search"
    )
    if not isinstance(usage, dict):
        return SearchUsage(server_searches=executions or None)

    def count(value: object) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    connectors = usage.get("connectors")
    billed = count(connectors.get("web_search")) if isinstance(connectors, dict) else 0
    prompt = count(usage.get("prompt_tokens")) + count(usage.get("connector_tokens"))
    return SearchUsage(
        input_tokens=prompt or None,
        output_tokens=count(usage.get("completion_tokens")) or None,
        server_searches=billed or executions or None,
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
    # Not ``post_responses``'s wire, but the same auth/error/request-id contract.
    payload, request_id = await responses_wire.post_responses(
        client,
        url=CONVERSATIONS_URL,
        bearer=key,
        body={
            "model": HOST_MODEL,
            "inputs": [{"role": "user", "content": responses_wire.search_prompt(query)}],
            "tools": [{"type": "web_search"}],
            "completion_args": {"max_tokens": MAX_OUTPUT_TOKENS},
            "store": False,
        },
        label=LABEL,
    )
    sources, answer = parse_conversation_search(payload, limit)
    usage = conversation_usage(payload)
    return SearchResponse(
        provider="mistral",
        auth_mode=KEY_MODE,
        sources=sources,
        answer=answer,
        request_id=request_id or (str(payload.get("conversation_id") or "").strip() or None),
        usage=usage,
        cost=pricing.native_cost(
            fee_usd=FEE_USD, usage=usage, hosting="mistral", model=HOST_MODEL, cache="inside"
        ),
    )


VENDOR = NativeSearchVendor(
    id="mistral",
    label=LABEL,
    access=web_search_copy.native_access_key(),
    detail=web_search_copy.native_detail_key(vendor=LABEL),
    modes=_modes,
    search=search,
    fee_usd=FEE_USD,
    credential_keys=(ENV_KEY,),
)
