"""Rung: Google's Gemini API image generation (Nano Banana, Interactions API).

**Wire** (official docs, fetched 2026-10-09 — citations tiered in
``docs/design/image-providers.md``; NO live probe ran in this wave by operator
decision):

- ``POST https://generativelanguage.googleapis.com/v1beta/interactions`` with
  ``x-goog-api-key``. The REST example documents the body as
  ``{"model": ..., "input": [{"type": "text", "text": ...}]}`` — the same
  content-block array form the SDK examples use.
- ``response_format`` selects image-only output (the default returns text AND
  image); our aspect-ratio mapping rides it (``3:4``/``9:16``/``4:3``/``16:9``/
  ``1:1`` are all documented ratios). The resolution tier (1K/2K/4K) is NOT
  sent: the tool's size enum does not carry one, and inventing a tier would be
  a fabricated cost knob.
- The response is an Interaction object; image bytes arrive base64 in
  ``steps[]``' ``model_output`` content blocks of type ``image``
  (``data``/``mime_type``), and the SDK-side convenience property
  ``output_image`` is documented as "the last generated image block". This
  rung reads ALL image blocks from the steps shape, falling back to
  ``output_image`` — both documented, so an envelope change that keeps either
  still works.

**Verified at implement time (2026-10-09, docs only) and labelled:** there is
NO documented ``seed`` parameter and NO documented image-count (``n``)
parameter on this endpoint (a text search of the fetched image-generation page
finds neither; the page's own limitation note says the model "won't always
follow the exact number of image outputs" — count is prompt-driven and not
guaranteed). So: a pinned ``seed`` is dropped silently (the OpenAI rule — a
field the wire cannot carry cannot be honoured) and ``num_images > 1`` is a
recorded SKIP rather than a silent single-image delivery. Both labels are
carried into the guide.

**Editing** (wired media wave-2 edit lane, 2026-10-10): the source travels as
an image content block appended to ``input`` — ``{"type": "image", "data":
<b64>, "mime_type": ...}``, the docs' own block shape — with up to 14
references per the model family. This rung sends ONE source (the
single-source v1 wire) and splits the tool's data URI into the block's two
fields. No API mask exists (the docs describe prompt-driven "semantic
masking"), and the endpoint has no count parameter, so ``num_images > 1``
stays a recorded skip.

Imagen is shut down in the Gemini API (docs, 2026-10-09): this rung never
targets it.
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
from local_operator.imagegen.errors import api_error_from_httpx_response

__all__ = ["run_google"]

GOOGLE_IMAGE_BASE_URL = "https://generativelanguage.googleapis.com"
GOOGLE_INTERACTIONS_PATH = "/v1beta/interactions"

#: The documented workhorse image model (the Nano Banana 2.1 family). A
#: DEFAULT, never a pin: the caller's ``model`` overrides it and the guide
#: names the family.
GOOGLE_DEFAULT_IMAGE_MODEL = "gemini-nano-banana-2.1"

#: One synchronous request's bound (the OpenAI precedent: generation is a
#: single call, seconds-to-tens-of-seconds).
GOOGLE_IMAGE_TIMEOUT_S = 120.0

#: The tool's FAL-shaped size enum onto documented aspect ratios (see the
#: module docstring; every value here appears in the endpoint's ratio tables).
GOOGLE_ASPECT_RATIOS = {
    "square_hd": "1:1",
    "square": "1:1",
    "portrait_4_3": "3:4",
    "portrait_16_9": "9:16",
    "landscape_4_3": "4:3",
    "landscape_16_9": "16:9",
}


def _image_blocks_from_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Every image content block in the Interaction's ``model_output`` steps."""
    blocks: list[dict[str, Any]] = []
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return blocks
    for step in steps:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        content = step.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                blocks.append(block)
    return blocks


def _image_blocks_from_output_image(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The documented convenience property: the LAST generated image block."""
    block = payload.get("output_image")
    if isinstance(block, dict) and block.get("type") in (None, "image"):
        return [block]
    return []


async def run_google(
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
    base_url: str = GOOGLE_IMAGE_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Run one Google image generation. See the module docstring."""
    # Observe a pre-aborted signal before anything else happens (reviewer
    # round 1 Q6): a single-request rung has no poll loop for the abort to
    # land in, so this zero-length wait is the only place the user's stop can
    # take effect before the spend. A no-op without a signal.
    if pause is not None:
        await pause(0.0)
    if num_images > 1:
        raise RungSkipped(
            "Google's interactions rung generates one image per call "
            "(no count parameter is documented on the endpoint).",
            reason_class="unsupported",
        )
    model_id = (model or GOOGLE_DEFAULT_IMAGE_MODEL).strip()
    response_format: dict[str, str] = {"type": "image"}
    aspect = GOOGLE_ASPECT_RATIOS.get(image_size)
    if aspect is not None:
        response_format["aspect_ratio"] = aspect

    # Call-time import on purpose: ``imagegen/rungs.py`` re-exports this
    # executor at ITS import time, so a module-scope import here would close
    # an import cycle for a caller that imports this module first. The
    # monkeypatch seam (``image_rungs._client_scope``) is preserved because
    # this reads the module attribute at call time.
    from local_operator.imagegen import rungs as image_rungs

    # An EDIT request appends the source as an image content block; the
    # data-URI splitter is shared with the other rungs' edit paths.
    input_blocks: list[dict[str, str]] = [{"type": "text", "text": prompt}]
    if source_url is not None:
        parts = image_rungs._data_uri_parts(source_url)
        if parts is None:
            raise APIError(
                "Google edits require a base64 data-URI source.",
                status_code=None,
                code="invalid_response",
            )
        mime_type, data_b64 = parts
        input_blocks.append({"type": "image", "data": data_b64, "mime_type": mime_type})
    body = {
        "model": model_id,
        "input": input_blocks,
        "response_format": response_format,
    }

    async with image_rungs._client_scope(client) as http:
        try:
            response = await http.request(
                "POST",
                f"{base_url.rstrip('/')}{GOOGLE_INTERACTIONS_PATH}",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=body,
                timeout=GOOGLE_IMAGE_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise APIError(
                f"Google image request failed: {exc}", status_code=None, code="network"
            ) from exc
        if response.status_code >= 400:
            raise api_error_from_httpx_response(
                response,
                fallback_message="Google image request refused",
                secrets=(key,),
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise APIError(
                "Google returned a response that is not JSON.",
                status_code=None,
                code="invalid_response",
            ) from exc
        if not isinstance(payload, dict):
            raise APIError(
                "Google returned an unexpected response shape.",
                status_code=None,
                code="invalid_response",
            )
        blocks = _image_blocks_from_steps(payload) or _image_blocks_from_output_image(payload)
        assets: list[MediaAsset] = []
        for block in blocks:
            data_b64 = block.get("data")
            if not isinstance(data_b64, str) or not data_b64:
                continue
            try:
                data = base64.b64decode(data_b64, validate=False)
            except (ValueError, TypeError) as exc:
                raise APIError(
                    "Google returned image data that is not valid base64.",
                    status_code=None,
                    code="invalid_response",
                ) from exc
            mime = block.get("mime_type")
            content_type = mime if isinstance(mime, str) and mime else "image/png"
            assets.append(MediaAsset(data=data, content_type=content_type, source_url=""))
        if not assets:
            raise APIError(
                "Google returned no image data.",
                status_code=None,
                code="invalid_response",
            )
        # No progress emission on this rung: a single synchronous request has
        # nothing observable to report until it returns (the OpenAI rung's own
        # b64 path is silent the same way), and the tool emits its terminal
        # ``completed`` update after the walk. ``emit``/``pause`` stay in the
        # signature because the rung contract is uniform across transports.
        return RungResult(assets=assets, model=model_id)
