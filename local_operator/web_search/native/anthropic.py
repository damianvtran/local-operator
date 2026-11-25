"""Anthropic native search: the Messages ``web_search_20250305`` server tool.

TWO LEGS, BOTH PAID-CLASS (design §4; operator decisions 2026-11-24):

* ``api-key`` -- an Anthropic platform key (``lop login anthropic-key`` stores an
  ``api_key`` row under ``anthropic``; ``ANTHROPIC_API_KEY`` also counts). Billed
  $10 per 1,000 searches plus tokens, so it sits after Radient
  (``providers.NATIVE_MODEL_KEY_BAND``).
* ``subscription`` -- the Claude sign-in (an OAuth grant under ``anthropic``).
  OFF BY DEFAULT behind ``web_search.claude_subscription_search``: Anthropic
  draws third-party-harness use of a plan from EXTRA USAGE (Apr 2026), so it is
  money the user did not obviously agree to spend on search, and the policy
  risk is the operator's to accept. When enabled it is still ordered as a PAID
  leg (never in the plan band ahead of Radient) and priced as an
  API-equivalent estimate, because no list price exists for it.

The OAuth request has to look like the chat client's on the same credential:
Bearer + ``anthropic-beta: oauth-2025-04-20`` and the Claude Code identity as
the FIRST system block -- without it the endpoint answers an opaque HTTP 429
(measured, ``providers/clients.py`` ``AnthropicClient.CLAUDE_CODE_IDENTITY``).
An API-key request never carries that identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from local_operator.i18n.keys import web_search as web_search_copy
from local_operator.web_search.models import SearchResponse, WebSearchSettings
from local_operator.web_search.native import anthropic_wire, credentials, pricing
from local_operator.web_search.native.base import (
    NativeClass,
    NativeSearchVendor,
    error_text,
    leg_mode,
)
from local_operator.web_search.native.responses_wire import search_prompt

LABEL = "Anthropic"
NAMESPACE = "anthropic"
ENV_KEY = "ANTHROPIC_API_KEY"

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
OAUTH_BETA = "oauth-2025-04-20"
#: Spelled here, pinned equal to ``AnthropicClient.CLAUDE_CODE_IDENTITY`` by a
#: test: importing the chat client would load the whole provider stack per search.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
#: The cheapest catalogue model; the search's cost is mostly retrieved page
#: tokens, so the host model is the only lever on price.
HOST_MODEL = "claude-haiku-5-5"
#: Sources ride on the tool-result blocks; the prose budget only has to let
#: the turn finish (DeepSeek's 1,024 with its single search).
MAX_TOKENS = 1_024
#: One server-side search per call: every extra one is another $0.01.
MAX_USES = 1
#: Published web search fee: $10 / 1,000 searches (platform.claude.com pricing).
FEE_USD = 0.01

KEY_MODE = "api-key"
SUBSCRIPTION_MODE = "subscription"
SUBSCRIPTION_NOTE = "API equivalent; drawn from Claude extra usage"


def _modes(config_dir: Path | None, settings: WebSearchSettings) -> list[tuple[str, NativeClass]]:
    modes: list[tuple[str, NativeClass]] = []
    if credentials.has_api_key(NAMESPACE, config_dir, ENV_KEY):
        modes.append((KEY_MODE, "key"))
    # The opt-in is checked BEFORE the store, so a default install never even
    # probes for the grant; `key` class keeps it after Radient (see module note).
    if settings.claude_subscription_search and credentials.has_oauth_row(NAMESPACE):
        modes.append((SUBSCRIPTION_MODE, "key"))
    return modes


def _request_body(query: str, *, subscription: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": HOST_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": [{"type": "text", "text": search_prompt(query)}]}],
        "tools": [
            {
                "type": anthropic_wire.WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
                "max_uses": MAX_USES,
            }
        ],
    }
    if subscription:
        body["system"] = [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]
    return body


async def search(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    from local_operator.i18n.text import tr

    mode = leg_mode(LABEL, _modes(config_dir, settings))
    headers = {
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
        "accept": "application/json",
    }
    if mode == SUBSCRIPTION_MODE:
        access = await credentials.oauth_access(NAMESPACE)
        if access is None:
            raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
        headers["Authorization"] = f"Bearer {access.access_token}"
        headers["anthropic-beta"] = OAUTH_BETA
    else:
        key = await credentials.api_key(NAMESPACE, config_dir, ENV_KEY)
        if not key:
            raise RuntimeError(tr("web_search")(web_search_copy.native_no_credential(vendor=LABEL)))
        headers["x-api-key"] = key
    subscription = mode == SUBSCRIPTION_MODE
    response = await client.post(
        MESSAGES_URL, headers=headers, json=_request_body(query, subscription=subscription)
    )
    if not response.is_success:
        raise error_text(LABEL, response)
    payload = response.json()
    refused = anthropic_wire.tool_result_error(payload)
    try:
        sources, answer = anthropic_wire.parse_messages_search(payload, limit, LABEL)
    except RuntimeError as error:
        # A refused search (`max_uses_exceeded`, `too_many_requests`...) is a
        # tool-result object, not a list; name the code so the digest says why.
        if refused:
            raise RuntimeError(f"{LABEL} web_search refused: {refused}") from error
        raise
    usage = anthropic_wire.messages_usage(payload)
    return SearchResponse(
        provider="anthropic",
        auth_mode=mode,
        sources=sources,
        answer=answer,
        request_id=str(response.headers.get("request-id") or "").strip() or None,
        usage=usage,
        cost=pricing.native_cost(
            fee_usd=FEE_USD,
            usage=usage,
            hosting="anthropic",
            model=HOST_MODEL,
            cache="apart",
            basis_note=SUBSCRIPTION_NOTE if subscription else "",
        ),
    )


VENDOR = NativeSearchVendor(
    id="anthropic",
    label=LABEL,
    access=web_search_copy.native_access_key_signin(),
    detail=web_search_copy.native_detail_key(vendor=LABEL),
    modes=_modes,
    search=search,
    fee_usd=FEE_USD,
    credential_keys=(ENV_KEY,),
)
