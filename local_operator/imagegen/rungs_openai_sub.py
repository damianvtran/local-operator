"""Rung: the ChatGPT subscription's built-in image tool (Codex backend, SSE).

**Why this rung exists**: the harness already holds ChatGPT OAuth grants for
chat (the Codex Responses surface), and that backend serves a built-in
``image_generation`` tool — so a subscription can fund images WITHOUT a
platform key. It is the deliberate INVERSE of the ``openai-key`` rule: the
grant that is invalid at ``/v1/images`` is exactly the credential this rung
spends. The manager's sign-off (design §14.5, v1 KEY-FIRST): this rung runs
when the earlier rungs refuse or are absent — an existing user's path must not
change, and the key rung supports seeds this one cannot.

**Funding and honesty**: quota-funded — no cash charge exists, so
``cost_source="subscription"`` and the guide states the 3-5x quota burn. The
operator's wave-2 cost rule still wants a comparable number, so ``cost_usd``
carries the API-EQUIVALENT price (see :data:`OPENAI_SUB_API_EQUIVALENT_USD`)
labelled ``billing_basis="subscription-api-equivalent"`` with its provenance on
``cost_provenance`` - never a charge, never a rate table (design D8 stands:
one provenance-labelled constant for THIS rung, not a general price lookup).
NOT available on the Free plan (the backend refuses it; that surfaces as a
rung failure and fails forward).

**Wire** (tiered citations in ``docs/design/image-providers.md``; NO live
probe was run in the implementing wave, by operator decision — the
documented contract is implemented strictly):

- POST the Codex responses URL (the same ChatGPT backend the chat path
  streams from; ``providers/clients.py`` spells the same URL) with
  ``tools:[{"type":"image_generation"}]``, ``tool_choice`` pinning that tool,
  and ``stream: true`` + ``store: false`` (both mandatory on this route).
- Headers mirror the chat client's OAuth request: Bearer token,
  ``chatgpt-account-id`` from the stored grant, ``Accept: text/event-stream``
  and the responses beta header.
- The image arrives base64-encoded inside the SSE stream as an
  ``image_generation_call`` output item's ``result`` field, on the
  ``response.completed`` event (and/or a standalone ``response.output_item.done``
  event). Partial-image preview events are deliberately NOT used as the
  artifact — only a completed result is.
- The host model is the Codex catalogue's current model for the account
  (the harness's own suggestion table; the caller's ``model`` overrides). No
  ``seed`` parameter exists on this route: a pinned seed is dropped with the
  same rule as OpenAI's (a field the wire cannot carry cannot be honoured).
- **No provider-side cancel** (the OpenAI-key precedent): abort discards the
  wait, the backend may still complete and burn quota. The cancel handle is
  never filled, so best-effort cancel reports "none — nothing to cancel".

IMG2IMG is SKIPPED with a recorded reason in v1 (the backend's refs/edits
support is documented but not wired in this wave), and ``num_images > 1`` is
skipped too: the built-in tool renders ONE image per request (v1 scope), and
silently delivering fewer images than the approval stated would be the wrong
kind of quiet.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from local_operator.artifacts.progress import ProgressFn
from local_operator.artifacts.rung import RungResult, RungSkipped
from local_operator.artifacts.walk import PauseFn
from local_operator.clients._http import APIError
from local_operator.imagegen import ImageRoute, MediaAsset
from local_operator.imagegen.errors import api_error_from_httpx_response

__all__ = ["run_openai_sub"]

#: The Codex responses endpoint. Spelled here rather than imported from
#: ``providers/clients.py`` because importing the chat-client stack would drag
#: the failover machinery into an image rung; a test pins the two spellings
#: equal so they cannot drift apart silently.
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
#: The beta header the chat path sends on the same route.
CODEX_BETA_HEADER = "responses=experimental"

#: The rung's internal bound for the whole streamed call. The WALK's rung
#: budget (240 s) is the effective bound — its ``wait_for`` cancels this read
#: — and this mirrors it so a stalled stream dies at the same place when the
#: rung is used as a library.
OPENAI_SUB_TIMEOUT_S = 240.0
#: Minimum gap between progress emissions while the stream runs: progress is
#: eyes, not telemetry, and a chatty stream must not flood the surface.
OPENAI_SUB_EMIT_INTERVAL_S = 2.0
#: Fallback host model when the suggestion table has no openai entry. The
#: table does carry one (pinned by its own tests); this keeps the call total.
OPENAI_SUB_FALLBACK_MODEL = "gpt-6-astra"

#: The top-level ``instructions`` the request carries. The chat client on this
#: same backend always sends ``instructions`` (its request's system blocks
#: joined with blank lines, ``clients.py`` ``_build_responses_body``), so the
#: harness sends the field here too rather than relying on the backend's
#: default; an image request has no system prompt, so it is a fixed one-liner.
#: TIER: open risk — no published document covers this field for the Codex
#: image tool (reviewer round 1, F2; see docs/design/image-providers.md
#: § Evidence tiers); the live probe settles it.
OPENAI_SUB_INSTRUCTIONS = "Generate the image the user asks for."

#: API-EQUIVALENT price of one subscription-funded image, USD. NOT a charge:
#: the plan quota funds the call and no cash moves; this is what the same
#: render would list at on OpenAI's API, so subscription and keyed spend are
#: comparable (the convention inference cost follows: list price, labelled).
#:
#: Source: OpenAI "Image generation" guide, per-image output table for
#: ``gpt-image-2`` at 1024x1024 / ``medium`` = $0.053
#: (https://developers.openai.com/api/docs/guides/image-generation.md),
#: cross-checked against https://developers.openai.com/api/docs/pricing.md
#: (``gpt-image-2`` image output $30.00/1M tokens), both fetched 2026-10-09.
#: ``gpt-image-2`` is the model the Codex docs name for the built-in tool
#: (docs/design/image-providers.md row 4); the host model in the request body
#: is the chat model, not the image model.
#:
#: ASSUMPTION (the call underdetermines the price): this rung sends
#: ``{"type": "image_generation"}`` with NO ``size``/``quality``, so the
#: backend picks them (documented default ``auto``, which "depends on the
#: generated image" and has no fixed price). The figure is therefore the
#: documented medium-quality square price, a mid-range point (low $0.006 ..
#: high $0.211 at 1024x1024); it excludes the host model's own token usage
#: that a Responses-API call also bills. Treat it as an order-of-magnitude
#: equivalent, not a per-image measurement. Revisit if the rung starts pinning
#: size/quality or the backend returns usage.
OPENAI_SUB_API_EQUIVALENT_USD = 0.053
OPENAI_SUB_API_EQUIVALENT_PROVENANCE = (
    "API-equivalent, not billed: OpenAI published gpt-image-2 price at 1024x1024 "
    "medium (developers.openai.com/api/docs/guides/image-generation, fetched "
    "2026-10-09); size/quality are not pinned on this route so the medium square "
    "price is assumed; excludes host-model tokens"
)


def _default_host_model() -> str:
    """The current Codex model for the account — a suggestion, never a pin.

    Read at call time from the harness's own suggestion table
    (``local_operator.model.defaults``), the same source the session resolver
    uses, so a catalogue move updates both. The caller's ``model`` param
    overrides it.
    """
    from local_operator.model.defaults import default_model_for

    return default_model_for("openai") or OPENAI_SUB_FALLBACK_MODEL


async def _iter_sse_json(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Parse an SSE stream's ``data:`` payloads as JSON objects.

    Lines that are not a complete JSON document (comments, keep-alives, a
    multi-line payload) are SKIPPED, not fatal: this route emits one compact
    JSON document per ``data:`` line, and a parser that died on an unknown
    line would fail generations the provider considers fine. The terminal
    state is carried by the events themselves (see :func:`_scan_event`).
    """
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if isinstance(data, dict):
            yield data


def _error_message(value: Any, fallback: str) -> str:
    """A provider error's sentence from ``str`` or ``{"message": ...}``."""
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(value, str) and value:
        return value
    return fallback


def _scan_event(data: dict[str, Any]) -> list[str]:
    """The image payload(s) in one SSE event; raises on a terminal failure.

    Accepted shapes (docs + community implementations, see the matrix):
    ``response.output_item.done`` with an ``image_generation_call`` item, and
    ``response.completed`` whose ``response.output`` carries the same item.
    ``response.failed`` — and an item whose status is ``failed`` — raise the
    provider's own sentence as an ``upstream`` failure so the walk records it
    and fails forward. Image items WITHOUT a ``result`` yet (in progress) are
    ignored: only a completed render is an artifact.
    """
    kind = data.get("type")
    if kind == "response.failed":
        response = data.get("response")
        error = response.get("error") if isinstance(response, dict) else None
        raise APIError(
            _error_message(error, "OpenAI reported the subscription request as failed."),
            status_code=None,
            code="upstream",
        )
    items: list[Any] = []
    if kind == "response.output_item.done":
        items.append(data.get("item"))
    elif kind == "response.completed":
        response = data.get("response")
        output = response.get("output") if isinstance(response, dict) else None
        if isinstance(output, list):
            items.extend(output)
    results: list[str] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "image_generation_call":
            continue
        if item.get("status") == "failed":
            raise APIError(
                _error_message(
                    item.get("error"), "OpenAI reported the image generation as failed."
                ),
                status_code=None,
                code="upstream",
            )
        result = item.get("result")
        if isinstance(result, str) and result:
            results.append(result)
    return results


async def run_openai_sub(
    *,
    prompt: str,
    access_token: str,
    account_id: str | None,
    num_images: int,
    image_size: str,
    source_url: str | None,
    seed: int | None,
    model: str | None,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Run one subscription-funded image generation over the Codex backend.

    See the module docstring for the wire, the funding posture and the v1
    scope boundaries. ``image_size`` and ``seed`` have no wire parameter on
    this route and are dropped silently (the OpenAI rung's rule).
    """
    # Observe a pre-aborted signal before anything else happens (QA round 2,
    # Q9): without this the request was SENT before the abort was seen — the
    # first SSE event observed it — spending plan quota the other rungs now
    # avoid. Aligns with google/xai/openrouter; a no-op without a signal.
    if pause is not None:
        await pause(0.0)
    if source_url is not None:
        raise RungSkipped(
            "The subscription rung has no image-to-image route in v1.",
            reason_class="unsupported",
        )
    if num_images > 1:
        raise RungSkipped(
            "The subscription rung generates one image per call in v1.",
            reason_class="unsupported",
        )
    model_id = (model or _default_host_model()).strip()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": CODEX_BETA_HEADER,
        "originator": "local-operator",
        "User-Agent": "local-operator",
    }
    if account_id:
        headers["chatgpt-account-id"] = account_id
    body = {
        "model": model_id,
        "instructions": OPENAI_SUB_INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "tools": [{"type": "image_generation"}],
        "tool_choice": {"type": "image_generation"},
        "stream": True,
        "store": False,
    }

    # Call-time import on purpose: ``imagegen/rungs.py`` re-exports this
    # executor at ITS import time, so a module-scope import here would close
    # an import cycle for a caller that imports this module first. The
    # monkeypatch seam (``image_rungs._client_scope``) is preserved because
    # this reads the module attribute at call time.
    from local_operator.imagegen import rungs as image_rungs

    # One initial frame before the stream opens so a surface shows the rung
    # running from the first moment (the walk emits nothing itself between
    # rungs; reviewer round 1 nit — this rung is the slowest at 1-4 min, so
    # silence until the first SSE event was the longest gap in the lane).
    image_rungs.emit_progress(
        emit,
        f"Generating via ChatGPT plan ({model_id}): running — 0s",
        **image_rungs.progress_details(
            stage="in_progress",
            provider=str(ImageRoute.OPENAI_SUB),
            model=model_id,
            elapsed_s=0,
            num_images=1,
        ),
    )

    async with image_rungs._client_scope(client) as http:
        started = time.monotonic()
        last_emit = 0.0
        results: list[str] = []
        seen: set[str] = set()
        async with http.stream(
            "POST",
            CODEX_RESPONSES_URL,
            headers=headers,
            json=body,
            timeout=OPENAI_SUB_TIMEOUT_S,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise api_error_from_httpx_response(
                    response,
                    fallback_message="OpenAI subscription image request failed",
                    secrets=(access_token,),
                )
            async for data in _iter_sse_json(response):
                if pause is not None:
                    await pause(0.0)
                for payload in _scan_event(data):
                    if payload not in seen:
                        seen.add(payload)
                        results.append(payload)
                elapsed = int(time.monotonic() - started)
                now = time.monotonic()
                if now - last_emit >= OPENAI_SUB_EMIT_INTERVAL_S:
                    last_emit = now
                    image_rungs.emit_progress(
                        emit,
                        f"Generating via ChatGPT plan ({model_id}): running — {elapsed}s",
                        **image_rungs.progress_details(
                            stage="in_progress",
                            provider=str(ImageRoute.OPENAI_SUB),
                            model=model_id,
                            elapsed_s=elapsed,
                            num_images=1,
                        ),
                    )

    if not results:
        raise APIError(
            "The subscription image stream completed without an image.",
            status_code=None,
            code="invalid_response",
        )
    assets: list[MediaAsset] = []
    for payload in results:
        try:
            data = base64.b64decode(payload, validate=False)
        except (ValueError, TypeError) as exc:
            raise APIError(
                "The subscription stream returned image data that is not valid base64.",
                status_code=None,
                code="invalid_response",
            ) from exc
        # The backend renders PNG bytes; no content type or dimensions ride
        # the payload (cache_media sniffs dims), matching the images API's
        # b64 form.
        assets.append(MediaAsset(data=data, content_type="image/png", source_url=""))
    return RungResult(
        assets=assets,
        model=model_id,
        # ``cost_source`` keeps its meaning (quota-funded); the amount is the
        # labelled API-equivalent, per image delivered (the route requests one;
        # n > 1 is skipped above, so this is 1x unless the stream yields more).
        cost_usd=OPENAI_SUB_API_EQUIVALENT_USD * len(assets),
        cost_source="subscription",
        billing_basis="subscription-api-equivalent",
        cost_provenance=OPENAI_SUB_API_EQUIVALENT_PROVENANCE,
    )
