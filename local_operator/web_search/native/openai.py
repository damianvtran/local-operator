"""OpenAI native search: a ChatGPT plan leg (Codex backend) and an API-key leg (Responses).

TWO LEGS, TWO CREDENTIAL CLASSES, NEVER MIXED (design §4):

* ``plan`` -- the ChatGPT sign-in (``lop login openai`` / ``openai-device``, an
  OAuth grant stored under ``openai``). Served by the Codex backend that the
  ChatGPT subscription funds, so it sits in the NATIVE band before Radient and
  is recorded as "included in ChatGPT plan" (no cash). Headers and the
  streamed shape are the ones ``imagegen/rungs_openai_sub.py`` already ships on
  the same route.
* ``api-key`` -- a platform key (``lop login openai-api-key`` stores an
  ``api_key`` row under ``openai``; ``OPENAI_API_KEY`` also counts). Served by
  the public Responses API, billed $10 per 1,000 searches plus tokens, so it
  sits after Radient (``providers.NATIVE_MODEL_KEY_BAND``).

The class gate is load-bearing: ``get_oauth_access`` can rotate to a sibling
``api_key`` row, and a platform key must never reach chatgpt.com (the imagegen
reviewer-round-1 F1 rule) -- ``credentials.oauth_access`` refuses that.
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
    error_text,
    leg_mode,
)

LABEL = "OpenAI"
NAMESPACE = "openai"
ENV_KEY = "OPENAI_API_KEY"

#: The Codex responses route the ChatGPT plan funds. Same spelling as
#: ``imagegen.rungs_openai_sub.CODEX_RESPONSES_URL`` (a test pins them equal;
#: importing that module would drag the image stack into every search).
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_BETA_HEADER = "responses=experimental"
#: The backend refuses a request without ``instructions`` (the chat client
#: always sends one); a search has no system prompt, so this is a fixed line.
CODEX_INSTRUCTIONS = "Search the web for the user's query and cite your sources."

RESPONSES_URL = "https://api.openai.com/v1/responses"
#: The cheapest catalogue model OpenAI documents for the ``web_search`` tool.
#: The search's cost is dominated by retrieved page tokens, so a small host
#: model is the whole lever on the key leg's price.
API_KEY_MODEL = "gpt-4.1-mini"
#: Enough for a short cited answer; the sources ride on the call items, not
#: on the prose, so a bigger budget buys nothing the chain uses.
MAX_OUTPUT_TOKENS = 512
#: Published Responses ``web_search`` tool fee: $10 / 1,000 calls
#: (developers.openai.com pricing, fetched 2026-11-24).
FEE_USD = 0.01

#: Whether the ChatGPT plan leg is registered. FALSE until a live probe passes:
#: the 2026-11-24 probe reached the Codex backend with both stored grants and
#: both answered HTTP 429 ``usage_limit_reached`` (weekly window, plan ``pro``),
#: so the request shape -- in particular whether this backend accepts
#: ``include: ["web_search_call.action.sources"]`` -- is still unproven. A leg
#: that has never answered once must not sit in the NATIVE band ahead of
#: Radient. Flip to ``True`` once a probe returns sources; the transport and its
#: tests are already here.
PLAN_LEG_PROBED = False

PLAN_MODE = "plan"
KEY_MODE = "api-key"
PLAN_NAME = "ChatGPT plan"


def _modes(config_dir: Path | None, _settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    modes: list[tuple[str, NativeClass]] = []
    if PLAN_LEG_PROBED and credentials.has_oauth_row(NAMESPACE):
        modes.append((PLAN_MODE, "plan"))
    if credentials.has_api_key(NAMESPACE, config_dir, ENV_KEY):
        modes.append((KEY_MODE, "key"))
    return modes


def _host_model() -> str:
    """The account's current Codex model: the imagegen rung's suggestion-table rule."""
    from local_operator.model.defaults import default_model_for

    return default_model_for("openai") or "gpt-6-astra"


async def _search_plan(client: httpx.AsyncClient, query: str, limit: int) -> SearchResponse:
    from local_operator.i18n.text import tr

    access = await credentials.oauth_access(NAMESPACE)
    if access is None:
        raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
    headers = {
        "Authorization": f"Bearer {access.access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": CODEX_BETA_HEADER,
        "originator": "local-operator",
        "User-Agent": "local-operator",
    }
    account_id = access.org_id or access.account_id
    if account_id:
        headers["chatgpt-account-id"] = account_id
    body = {
        "model": _host_model(),
        "instructions": CODEX_INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": responses_wire.search_prompt(query)}],
            }
        ],
        "tools": [{"type": "web_search"}],
        # `required`: a turn that answers from memory spends plan quota and
        # returns no sources, which the parser would reject anyway.
        "tool_choice": "required",
        "include": [responses_wire.SOURCES_INCLUDE],
        "stream": True,
        "store": False,
    }
    async with client.stream("POST", CODEX_RESPONSES_URL, headers=headers, json=body) as response:
        if not response.is_success:
            await response.aread()
            raise error_text(f"{LABEL} (ChatGPT plan)", response)
        payload = await responses_wire.collect_sse_response(response, LABEL)
        request_id = str(response.headers.get("x-request-id") or "").strip() or None
    sources, answer = responses_wire.parse_responses_search(payload, limit, LABEL)
    return SearchResponse(
        provider="openai",
        auth_mode=PLAN_MODE,
        sources=sources,
        answer=answer,
        request_id=request_id,
        usage=responses_wire.responses_usage(payload),
        cost=pricing.plan_cost(PLAN_NAME),
    )


async def _search_key(
    client: httpx.AsyncClient, config_dir: Path | None, query: str, limit: int
) -> SearchResponse:
    from local_operator.i18n.text import tr

    key = await credentials.api_key(NAMESPACE, config_dir, ENV_KEY)
    if not key:
        raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
    payload, request_id = await responses_wire.post_responses(
        client,
        url=RESPONSES_URL,
        bearer=key,
        body={
            "model": API_KEY_MODEL,
            "input": responses_wire.search_prompt(query),
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "include": [responses_wire.SOURCES_INCLUDE],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
        },
        label=LABEL,
    )
    sources, answer = responses_wire.parse_responses_search(payload, limit, LABEL)
    usage = responses_wire.responses_usage(payload)
    return SearchResponse(
        provider="openai",
        auth_mode=KEY_MODE,
        sources=sources,
        answer=answer,
        request_id=request_id,
        usage=usage,
        cost=pricing.native_cost(
            fee_usd=FEE_USD, usage=usage, hosting="openai", model=API_KEY_MODEL, cache="inside"
        ),
    )


async def search(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    mode = leg_mode(LABEL, _modes(config_dir, settings))
    if mode == PLAN_MODE:
        return await _search_plan(client, query, limit)
    return await _search_key(client, config_dir, query, limit)


VENDOR = NativeSearchVendor(
    id="openai",
    label=LABEL,
    access=web_search_copy.native_access_plan_key(),
    detail=web_search_copy.native_detail_plan_key(vendor=LABEL, plan="ChatGPT"),
    modes=_modes,
    search=search,
    fee_usd=FEE_USD,
    credential_keys=(ENV_KEY,),
)
