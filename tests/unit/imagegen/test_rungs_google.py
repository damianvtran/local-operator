"""The Google rung's wire, against ``httpx.MockTransport`` — no network.

What is pinned: the Interactions-API call shape (endpoint, ``x-goog-api-key``,
the documented content-block input, the image-only ``response_format`` and the
aspect-ratio mapping), the response parse across BOTH documented shapes
(``steps[].model_output`` content blocks; the ``output_image`` convenience
property), the wired edit path (the source as an image content block),
the recorded skip for >1 image, and the failure classes
that fail forward. NO live probe ran in this wave (operator decision) — the
tiered citations live in ``docs/design/image-providers.md``.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable

import httpx
import pytest

from local_operator.artifacts.rung import RungSkipped
from local_operator.clients._http import APIError
from local_operator.imagegen import rungs_google as rung_mod

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)
PNG_B64 = base64.b64encode(PNG_1X1).decode("ascii")


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[Any] = []

    def handler(self, response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
        def _handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            try:
                self.bodies.append(json.loads(request.content) if request.content else None)
            except ValueError:
                self.bodies.append(None)
            return response

        return _handle


def _steps_response(*blocks: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "interaction_1",
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": list(blocks)},
            ],
        },
    )


async def _run(recorder: _Recorder, **kwargs: Any):
    defaults: dict[str, Any] = dict(
        prompt="a cat",
        key="gk-1",
        num_images=1,
        image_size="square_hd",
        source_url=None,
        seed=None,
        model=None,
        emit=None,
        pause=None,
    )
    defaults.update(kwargs)
    return await rung_mod.run_google(**defaults)


@pytest.mark.asyncio
async def test_the_call_shape_and_the_steps_parse() -> None:
    recorder = _Recorder()
    response = _steps_response({"type": "image", "data": PNG_B64, "mime_type": "image/png"})
    http = _client(recorder.handler(response))

    result = await _run(recorder, client=http)

    assert result.model == rung_mod.GOOGLE_DEFAULT_IMAGE_MODEL
    assert result.cost_usd is None
    assert len(result.assets) == 1
    assert result.assets[0].data == PNG_1X1
    assert result.assets[0].content_type == "image/png"

    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == (rung_mod.GOOGLE_IMAGE_BASE_URL + rung_mod.GOOGLE_INTERACTIONS_PATH)
    assert request.headers["x-goog-api-key"] == "gk-1"
    body = recorder.bodies[0]
    assert body["model"] == rung_mod.GOOGLE_DEFAULT_IMAGE_MODEL
    assert body["input"] == [{"type": "text", "text": "a cat"}]
    # Image-only output (the endpoint defaults to text AND image), plus the
    # documented aspect ratio for the FAL-shaped size enum.
    assert body["response_format"] == {"type": "image", "aspect_ratio": "1:1"}
    # No seed / count the wire cannot carry.
    assert "seed" not in body


@pytest.mark.asyncio
async def test_the_output_image_convenience_property_is_the_fallback() -> None:
    recorder = _Recorder()
    response = httpx.Response(
        200,
        json={
            "id": "interaction_1",
            "status": "completed",
            "output_image": {"type": "image", "data": PNG_B64, "mime_type": "image/webp"},
        },
    )
    http = _client(recorder.handler(response))

    result = await _run(recorder, client=http)

    assert len(result.assets) == 1
    # The block's own mime type is carried through when present.
    assert result.assets[0].content_type == "image/webp"


@pytest.mark.asyncio
async def test_every_model_output_image_block_is_collected() -> None:
    recorder = _Recorder()
    response = _steps_response(
        {"type": "text", "text": "here are two"},
        {"type": "image", "data": PNG_B64, "mime_type": "image/png"},
        {"type": "image", "data": PNG_B64, "mime_type": "image/png"},
    )
    http = _client(recorder.handler(response))

    result = await _run(recorder, client=http)

    assert len(result.assets) == 2


@pytest.mark.asyncio
async def test_the_aspect_ratio_mapping_is_documented_values_only() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_steps_response({"type": "image", "data": PNG_B64})))

    await _run(recorder, image_size="portrait_16_9", client=http)

    assert recorder.bodies[0]["response_format"]["aspect_ratio"] == "9:16"
    # Every mapped value is one of the endpoint's documented ratio tables.
    assert set(rung_mod.GOOGLE_ASPECT_RATIOS.values()) <= {
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "4:5",
        "5:4",
        "9:16",
        "16:9",
        "21:9",
    }


@pytest.mark.asyncio
async def test_a_pinned_model_overrides_the_default() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_steps_response({"type": "image", "data": PNG_B64})))

    result = await _run(recorder, model="gemini-3.1-flash-lite-image", client=http)

    assert recorder.bodies[0]["model"] == "gemini-3.1-flash-lite-image"
    assert result.model == "gemini-3.1-flash-lite-image"


@pytest.mark.asyncio
async def test_an_edit_appends_the_source_as_an_image_content_block() -> None:
    recorder = _Recorder()
    response = _steps_response({"type": "image", "data": PNG_B64, "mime_type": "image/png"})
    http = _client(recorder.handler(response))

    result = await _run(recorder, source_url=f"data:image/png;base64,{PNG_B64}", client=http)

    body = recorder.bodies[0]
    # The docs' own block shape: base64 WITHOUT the data-URI prefix, the mime
    # split into its own field, appended after the text block.
    assert body["input"][0] == {"type": "text", "text": "a cat"}
    assert body["input"][1] == {"type": "image", "data": PNG_B64, "mime_type": "image/png"}
    assert result.assets[0].data == PNG_1X1


@pytest.mark.asyncio
async def test_an_edit_with_a_non_data_uri_source_is_refused_before_the_wire() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(httpx.Response(200, json={})))

    with pytest.raises(APIError):
        await _run(recorder, source_url="https://example.com/x.png", client=http)
    assert recorder.requests == []


@pytest.mark.asyncio
async def test_multi_image_is_still_a_recorded_skip() -> None:
    http = _client(lambda request: httpx.Response(500))

    with pytest.raises(RungSkipped) as caught:
        await _run(_Recorder(), num_images=2, client=http)
    assert caught.value.reason_class == "unsupported"
    assert "one image per call" in str(caught.value)


@pytest.mark.asyncio
async def test_an_http_failure_maps_to_the_lane_error_shape() -> None:
    recorder = _Recorder()
    response = httpx.Response(403, json={"error": {"message": "key not allowed"}})
    http = _client(recorder.handler(response))

    with pytest.raises(APIError) as caught:
        await _run(recorder, client=http)

    assert caught.value.status_code == 403
    assert "gk-1" not in (caught.value.body or "")


@pytest.mark.asyncio
async def test_a_response_without_image_data_is_invalid_response() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(httpx.Response(200, json={"id": "i", "steps": []})))

    with pytest.raises(APIError) as caught:
        await _run(recorder, client=http)

    assert caught.value.code == "invalid_response"


@pytest.mark.asyncio
async def test_a_pre_aborted_signal_stops_before_the_request() -> None:
    # Reviewer round 1 Q6: a single-request rung has no poll loop, so the
    # zero-length pause is the abort check that must run before the spend.
    from local_operator.imagegen import cascade as image_cascade

    calls: list[float] = []

    async def pause(seconds: float) -> None:
        calls.append(seconds)
        raise image_cascade.ImageGenerationCancelled()

    recorder = _Recorder()
    http = _client(recorder.handler(_steps_response({"type": "image", "data": PNG_B64})))
    with pytest.raises(image_cascade.ImageGenerationCancelled):
        await _run(recorder, pause=pause, client=http)

    assert calls == [0.0]
    assert recorder.requests == []


def test_the_spec_declares_no_cancel_and_a_rate_table() -> None:
    from local_operator.artifacts.rung import CancelSupport, SourceSupport
    from local_operator.imagegen import ImageRoute, cascade

    spec = cascade.RUNG_SPECS[ImageRoute.GOOGLE]
    assert spec.label == "Google"
    assert spec.cancel_support == CancelSupport.NONE
    assert spec.cost == "rate_table"
    assert spec.capabilities == frozenset({"t2i", "i2i"}), "the edit path is wired"
    assert spec.sources == SourceSupport.MULTI
    assert spec.max_sources == 14
