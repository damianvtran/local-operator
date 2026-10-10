"""Rung: xAI's Grok Imagine image generation (OpenAI-images-shaped).

**Wire** (xAI's own OpenAPI spec, ``api.x.ai/api-docs/openapi.json``, fetched
2026-10-09; citations tiered in ``docs/design/image-providers.md``; NO live
probe ran this wave by operator decision):

- ``POST https://api.x.ai/v1/images/generations`` with a bearer — the spec's
  ``security`` is ``bearerAuth`` and xAI documents it works with API keys AND
  OAuth tokens; this rung's credential resolution accepts either class (both
  store under the ``xai`` namespace) and reuses the harness's own store
  resolution, so refresh/backoff behave as a chat turn's credential would.
- Request schema (``GenerateImageRequest``): ``prompt``, ``model``, ``n``
  1..10, ``response_format`` ``"url"`` (default) or ``"b64_json"``,
  ``aspect_ratio`` (enum incl. 1:1, 3:4, 4:3, 9:16, 16:9), ``resolution``,
  ``deferred``. This rung asks for ``b64_json`` so the artifact never depends
  on the provider's URL lifetime, maps the tool's size enum onto the
  documented ratios, and passes ``n`` straight through (the tool's cap of 4
  is inside the schema's 1..10; the schema has NO seed parameter, so a pinned
  seed is dropped silently — the OpenAI rule).
- **Cost is REPORTED, not a rate table.** The spec's response schema carries
  ``usage`` as ``oneOf [null, MediaUsage]``, and WHEN PRESENT
  ``cost_in_usd_ticks`` is required on it ("the cost of this request"; one
  USD cent = 100,000,000 ticks, one USD = 10,000,000,000), so this rung emits
  ``cost_usd`` from it with ``cost_source="reported"`` — the design's D8
  permits only reported figures on ``cost_usd``, and this is one. (The
  wave's sign-off had pencilled xAI as ``rate_table`` on the research note
  that "nothing per request" is returned; the first-party schema read at
  implement time says otherwise — reworded at reviewer round 1, F5, for the
  nullable ``usage`` — so the reported figure wins, when it arrives. The
  pricing page's $0.04/image flat rate stays the guide's cross-check.)
- ``GeneratedImage.b64_json`` is documented as the b64 form WITHOUT a
  data-URI prefix; ``mime_type`` rides each item and is carried through when
  present. If a ``url`` item appears anyway (a proxy deployment ignoring
  ``response_format``), the bounded downloader fetches it — tolerant, never
  the primary path.

**Editing** (wired media wave-2 edit lane, 2026-10-10): ``POST
/v1/images/edits`` with the single-image shape — ``"image": {"url": <data
URI>, "type": "image_url"}`` — and the same ``data[]``/``usage`` parse as
generations. Edits bill input AND output (the pricing page says so
explicitly), so the generations flat rate must never be reused as an edit
cost: the reported ``usage.cost_in_usd_ticks`` stays the only figure, and
nothing replaces it when absent. Up to 5 references are documented for
multi-image editing; that shape arrives with the multi-source wire, and
``num_images > 1`` on the single-image shape is a recorded skip.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from local_operator.artifacts import BillingBasis
from local_operator.artifacts.progress import ProgressFn
from local_operator.artifacts.rung import RungResult, RungSkipped
from local_operator.artifacts.walk import PauseFn
from local_operator.clients._http import APIError
from local_operator.imagegen import MediaAsset
from local_operator.imagegen.media import download_asset

__all__ = ["run_xai"]

XAI_IMAGE_BASE_URL = "https://api.x.ai/v1"
XAI_IMAGES_PATH = "/images/generations"
#: The edit endpoint (wired media wave-2 edit lane): the single-image shape,
#: ``image`` object with a data URI in ``url``.
XAI_IMAGES_EDITS_PATH = "/images/edits"

#: The current Grok Imagine image model (the docs' own examples). A DEFAULT,
#: never a pin: the caller's ``model`` overrides it.
XAI_DEFAULT_IMAGE_MODEL = "grok-imagine-image-2.0"

#: One synchronous request's bound (the OpenAI rung's class: single call).
XAI_IMAGE_TIMEOUT_S = 120.0

#: One US dollar in xAI's cost ticks (spec: one USD CENT = 100,000,000 ticks).
XAI_USD_TICKS_PER_USD = 10_000_000_000

#: The tool's FAL-shaped size enum onto the spec's documented aspect ratios.
XAI_ASPECT_RATIOS = {
    "square_hd": "1:1",
    "square": "1:1",
    "portrait_4_3": "3:4",
    "portrait_16_9": "9:16",
    "landscape_4_3": "4:3",
    "landscape_16_9": "16:9",
}


async def run_xai(
    *,
    prompt: str,
    key: str,
    num_images: int,
    image_size: str,
    source_url: str | None,
    seed: int | None,
    model: str | None,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    client: httpx.AsyncClient | None = None,
    credential_kind: str = "api_key",
) -> RungResult:
    """Run one xAI image generation. See the module docstring.

    ``credential_kind`` is the class of ``key`` (``"api_key"`` or ``"oauth"``,
    from ``availability.xai_call_credential``); it decides only the result's
    ``billing_basis``. Defaulting to ``api_key`` keeps every existing caller's
    behaviour (a bare key is metered cash).
    """
    # Observe a pre-aborted signal before anything else happens (reviewer
    # round 1 Q6): a single-request rung has no poll loop for the abort to
    # land in, so this zero-length wait is the only place the user's stop can
    # take effect before the spend. A no-op without a signal.
    if pause is not None:
        await pause(0.0)
    model_id = (model or XAI_DEFAULT_IMAGE_MODEL).strip()
    body: dict[str, Any] = {
        "model": model_id,
        "prompt": prompt,
        "response_format": "b64_json",
    }
    path = XAI_IMAGES_PATH
    if source_url is not None:
        # The wired edit shape is the single-image one (docs fetched
        # 2026-10-10): an ``image`` object whose ``url`` takes the data URI;
        # ``n`` is a generations-only parameter, so an edit carries no count
        # (the multi-image page's 1..5 references arrive with the
        # multi-source wire). A larger request is a recorded SKIP rather
        # than silently delivering one image (review round 1, R2).
        if num_images > 1:
            raise RungSkipped(
                "xAI's edit endpoint documents no image count; a multi-image "
                "request is skipped rather than silently delivering fewer.",
                reason_class="unsupported",
            )
        body["image"] = {"url": source_url, "type": "image_url"}
        path = XAI_IMAGES_EDITS_PATH
    else:
        body["n"] = num_images
    aspect = XAI_ASPECT_RATIOS.get(image_size)
    if aspect is not None:
        body["aspect_ratio"] = aspect

    # Call-time import on purpose: ``imagegen/rungs.py`` re-exports this
    # executor at ITS import time, so a module-scope import here would close
    # an import cycle for a caller that imports this module first. The
    # monkeypatch seam (``image_rungs._client_scope``) is preserved because
    # this reads the module attribute at call time.
    from local_operator.imagegen import rungs as image_rungs

    async with image_rungs._client_scope(client) as http:
        payload = await image_rungs._request_json(
            http,
            "POST",
            f"{XAI_IMAGE_BASE_URL}{path}",
            label="xAI images",
            timeout_s=XAI_IMAGE_TIMEOUT_S,
            secrets=(key,),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
        )
        items = payload.get("data")
        if not isinstance(items, list) or not items:
            raise APIError("xAI returned no image data.", status_code=None, code="invalid_response")
        assets: list[MediaAsset] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            mime = item.get("mime_type")
            content_type = mime if isinstance(mime, str) and mime else None
            b64 = item.get("b64_json")
            if isinstance(b64, str) and b64:
                try:
                    data = base64.b64decode(b64, validate=False)
                except (ValueError, TypeError) as exc:
                    raise APIError(
                        "xAI returned image data that is not valid base64.",
                        status_code=None,
                        code="invalid_response",
                    ) from exc
                assets.append(
                    MediaAsset(
                        data=data,
                        content_type=content_type or "image/png",
                        source_url="",
                    )
                )
                continue
            url = item.get("url")
            if isinstance(url, str) and url:
                # Tolerant fallback: a proxy that ignores ``response_format``
                # answers with URLs; the shared bounded downloader fetches it.
                assets.append(
                    await download_asset(url, client=http, fallback_content_type=content_type)
                )
        if not assets:
            raise APIError(
                "xAI returned no usable image entries.",
                status_code=None,
                code="invalid_response",
            )
        cost_usd: float | None = None
        usage = payload.get("usage")
        if isinstance(usage, dict):
            ticks = usage.get("cost_in_usd_ticks")
            if isinstance(ticks, int) and not isinstance(ticks, bool):
                cost_usd = ticks / XAI_USD_TICKS_PER_USD
        # BILLING BASIS by credential class. The amount is the same field on
        # both paths - xAI's own per-request ``cost_in_usd_ticks`` - so
        # ``cost_source`` stays ``reported`` either way. What differs is who
        # pays: an API key is metered against the account's prepaid credits
        # (``billed``); a Grok sign-in grant draws on the SUBSCRIPTION's
        # allotment, so the provider-reported price is what that usage WOULD
        # cost at API rates, not a charge - exactly the meaning of
        # ``subscription-api-equivalent`` (``estimated`` would wrongly say
        # the figure is modelled; ``billed`` wrongly says cash moved). Caveat,
        # recorded because no live OAuth call has run (see the module
        # docstring's 403 note): whether ``usage`` is populated on the OAuth
        # path is unverified - when absent there is no figure and no basis,
        # never a synthesized one.
        basis: BillingBasis | None = None
        provenance: str | None = None
        if cost_usd is not None:
            if credential_kind == "oauth":
                basis = "subscription-api-equivalent"
                provenance = (
                    "xAI response usage.cost_in_usd_ticks on a Grok sign-in (subscription): "
                    "API-equivalent per-request cost, not billed"
                )
            else:
                basis = "billed"
                provenance = "xAI response usage.cost_in_usd_ticks (API key, metered)"
        return RungResult(
            assets=assets,
            model=model_id,
            cost_usd=cost_usd,
            cost_source="reported" if cost_usd is not None else None,
            billing_basis=basis,
            cost_provenance=provenance,
        )
