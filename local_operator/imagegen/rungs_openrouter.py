"""Rung: OpenRouter's dedicated Images API (one key, the whole catalog).

**Wire** (OpenRouter's own docs, fetched 2026-10-09 — citations tiered in
``docs/design/image-providers.md``; NO live probe ran this wave by operator
decision):

- ``POST https://openrouter.ai/api/v1/images`` with a bearer.
- Request parameters (docs' table): ``model`` and ``prompt`` required;
  ``n`` (1..10; "providers may return fewer, and single-image providers
  reject n > 1"), ``aspect_ratio``, ``resolution``, ``seed`` ("where
  supported"), ``input_references`` (image-to-image), ``stream``. This rung
  sends ``n`` only when the caller asks for more than one (leaving a
  single-image provider's default alone), maps the tool's size enum onto the
  documented ratios, and passes a pinned ``seed`` through (a model that
  cannot honour it fails forward rather than silently ignoring it).
- Response: ``data[].b64_json`` plus ``media_type`` ("present whenever the
  format is identifiable") and ``usage.cost`` — the docs' own settlement
  example. ``cost`` is used as-is: a figure a provider REPORTED
  (``cost_source="reported"``, design D8); absent ``usage`` → no figure and
  no label.
- Billing is all-or-nothing (failed generations return 502 and are not
  billed) — irrelevant to the rung's behaviour, relevant to the guide's cost
  note.

**Model default**: ``bytedance-seed/seedream-4.5`` — a default, never a pin
(the caller's ``model`` overrides it). It is the catalog's multi-image
workhorse (its endpoint record documents ``n`` up to 10 and seed support);
the guide names the cheaper ``black-forest-labs/flux.2-klein-4b`` and the
frontier ``openai/gpt-image-2.5`` / ``google/gemini-nano-banana-2.1`` classes
as alternatives, and per-model availability/pricing moves, so nothing is
pinned in code.

**Editing** (wired media wave-2 edit lane, 2026-10-10): the source rides
``input_references`` — the docs' own shape, ``[{"type": "image_url",
"image_url": {"url": <data URL>}}]`` — but only after ``GET
/api/v1/images/models`` confirms the pinned model's
``architecture.input_modalities`` accepts ``"image"``: a model that ignores a
reference would bill a plain text-to-image, the silent class this lane exists
to kill. A deterministic negative (model unlisted, or no image modality) is a
recorded SKIP that fails forward; a models read that cannot answer RAISES
(the rung failed — its real class rides the attempt record, and no
capability is ever guessed). ``usage.cost`` settles the figure as before; the
per-endpoint billables include the reference, so no reuse or invention is
needed here.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from local_operator.artifacts.progress import ProgressFn
from local_operator.artifacts.rung import RungResult, RungSkipped
from local_operator.artifacts.walk import PauseFn
from local_operator.clients._http import APIError
from local_operator.imagegen import MediaAsset

__all__ = ["run_openrouter"]

OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"
#: The discovery read the edit path's capability check uses.
OPENROUTER_IMAGES_MODELS_URL = "https://openrouter.ai/api/v1/images/models"

#: A default, not a pin — see the module docstring.
OPENROUTER_DEFAULT_IMAGE_MODEL = "bytedance-seed/seedream-4.5"

#: One synchronous request's bound (the OpenAI rung's class: single call).
OPENROUTER_IMAGE_TIMEOUT_S = 120.0

#: The tool's FAL-shaped size enum onto the docs' documented ratios.
OPENROUTER_ASPECT_RATIOS = {
    "square_hd": "1:1",
    "square": "1:1",
    "portrait_4_3": "3:4",
    "portrait_16_9": "9:16",
    "landscape_4_3": "4:3",
    "landscape_16_9": "16:9",
}


async def run_openrouter(
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
) -> RungResult:
    """Run one OpenRouter image generation. See the module docstring."""
    # Observe a pre-aborted signal before anything else happens (reviewer
    # round 1 Q6): a single-request rung has no poll loop for the abort to
    # land in, so this zero-length wait is the only place the user's stop can
    # take effect before the spend. A no-op without a signal.
    if pause is not None:
        await pause(0.0)
    model_id = (model or OPENROUTER_DEFAULT_IMAGE_MODEL).strip()
    body: dict[str, Any] = {"model": model_id, "prompt": prompt}
    aspect = OPENROUTER_ASPECT_RATIOS.get(image_size)
    if aspect is not None:
        body["aspect_ratio"] = aspect
    if num_images > 1:
        body["n"] = num_images
    if seed is not None:
        body["seed"] = seed

    # Call-time import on purpose: ``imagegen/rungs.py`` re-exports this
    # executor at ITS import time, so a module-scope import here would close
    # an import cycle for a caller that imports this module first. The
    # monkeypatch seam (``image_rungs._client_scope``) is preserved because
    # this reads the module attribute at call time.
    from local_operator.imagegen import rungs as image_rungs

    async with image_rungs._client_scope(client) as http:
        if source_url is not None:
            # Capability check BEFORE sending (audit §C.iv) — see the module
            # docstring for the skip-vs-raise split.
            models_payload = await image_rungs._request_json(
                http,
                "GET",
                OPENROUTER_IMAGES_MODELS_URL,
                label="OpenRouter images",
                timeout_s=OPENROUTER_IMAGE_TIMEOUT_S,
                secrets=(key,),
                headers={"Authorization": f"Bearer {key}"},
            )
            rows = models_payload.get("data")
            rows = rows if isinstance(rows, list) else []
            row = next(
                (item for item in rows if isinstance(item, dict) and item.get("id") == model_id),
                None,
            )
            if row is None:
                raise RungSkipped(
                    f"OpenRouter does not list image model {model_id!r}; edit skipped.",
                    reason_class="unsupported",
                )
            architecture = row.get("architecture")
            modalities = (
                architecture.get("input_modalities") if isinstance(architecture, dict) else None
            )
            supports_image = isinstance(modalities, list) and "image" in [
                str(value).lower() for value in modalities
            ]
            if not supports_image:
                raise RungSkipped(
                    f"OpenRouter model {model_id} does not declare image input "
                    f"(input_modalities: {modalities!r}); edit skipped.",
                    reason_class="unsupported",
                )
            body["input_references"] = [{"type": "image_url", "image_url": {"url": source_url}}]
        payload = await image_rungs._request_json(
            http,
            "POST",
            OPENROUTER_IMAGES_URL,
            label="OpenRouter images",
            timeout_s=OPENROUTER_IMAGE_TIMEOUT_S,
            secrets=(key,),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
        )
        items = payload.get("data")
        if not isinstance(items, list) or not items:
            raise APIError(
                "OpenRouter returned no image data.",
                status_code=None,
                code="invalid_response",
            )
        assets: list[MediaAsset] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            b64 = item.get("b64_json")
            if not isinstance(b64, str) or not b64:
                continue
            try:
                data = base64.b64decode(b64, validate=False)
            except (ValueError, TypeError) as exc:
                raise APIError(
                    "OpenRouter returned image data that is not valid base64.",
                    status_code=None,
                    code="invalid_response",
                ) from exc
            media_type = item.get("media_type")
            content_type = media_type if isinstance(media_type, str) and media_type else "image/png"
            assets.append(MediaAsset(data=data, content_type=content_type, source_url=""))
        if not assets:
            raise APIError(
                "OpenRouter returned no usable image entries.",
                status_code=None,
                code="invalid_response",
            )
        cost_usd: float | None = None
        usage = payload.get("usage")
        if isinstance(usage, dict):
            cost = usage.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                cost_usd = float(cost)
        return RungResult(
            assets=assets,
            model=model_id,
            cost_usd=cost_usd,
            cost_source="reported" if cost_usd is not None else None,
            # ``usage.cost`` is the settled per-request cost of an
            # all-or-nothing billed call (a failure is a 502, unbilled).
            billing_basis="billed" if cost_usd is not None else None,
            cost_provenance=(
                "OpenRouter /images response usage.cost" if cost_usd is not None else None
            ),
        )
