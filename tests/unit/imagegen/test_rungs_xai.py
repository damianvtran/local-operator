"""The xAI rung's wire, against ``httpx.MockTransport`` — no network.

What is pinned: the OpenAI-images-shaped call (endpoint, bearer, body with
``n`` and ``response_format: "b64_json"``, the documented aspect-ratio
mapping), the REPORTED cost from ``usage.cost_in_usd_ticks`` (the official
OpenAPI schema requires it; 1 USD = 10,000,000,000 ticks), the b64 parse and
the url fallback, the wired edit path (``/v1/images/edits``, single-image
shape, reported ticks kept), and the failure classes. NO live
probe ran this wave (operator decision) — tiered citations are in
``docs/design/image-providers.md``.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable

import httpx
import pytest

from local_operator.clients._http import APIError
from local_operator.imagegen import rungs_xai as rung_mod

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


def _ok_response(
    *,
    ticks: int | None = 40_000_000_000,
    items: list[dict[str, Any]] | None = None,
) -> httpx.Response:
    payload: dict[str, Any] = {
        "data": items if items is not None else [{"b64_json": PNG_B64, "mime_type": "image/png"}],
    }
    if ticks is not None:
        payload["usage"] = {"cost_in_usd_ticks": ticks}
    return httpx.Response(200, json=payload)


async def _run(recorder: _Recorder, **kwargs: Any):
    defaults: dict[str, Any] = dict(
        prompt="a cat",
        key="xk-1",
        num_images=1,
        image_size="square_hd",
        source_url=None,
        seed=None,
        model=None,
        emit=None,
        pause=None,
    )
    defaults.update(kwargs)
    return await rung_mod.run_xai(**defaults)


@pytest.mark.asyncio
async def test_the_call_shape_and_the_reported_cost() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_ok_response()))

    result = await _run(recorder, client=http)

    assert result.model == rung_mod.XAI_DEFAULT_IMAGE_MODEL
    assert result.assets[0].data == PNG_1X1
    assert result.assets[0].content_type == "image/png"
    # 40,000,000,000 ticks = $4.00 — a REPORTED figure (the schema requires
    # usage.cost_in_usd_ticks), so it may ride cost_usd under design D8.
    assert result.cost_usd == pytest.approx(4.0)
    assert result.cost_source == "reported"
    # Default credential class is an API key: metered cash.
    assert result.billing_basis == "billed"
    assert result.cost_provenance is not None and "API key" in result.cost_provenance

    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == rung_mod.XAI_IMAGE_BASE_URL + rung_mod.XAI_IMAGES_PATH
    assert request.headers["authorization"] == "Bearer xk-1"
    body = recorder.bodies[0]
    assert body["model"] == rung_mod.XAI_DEFAULT_IMAGE_MODEL
    assert body["prompt"] == "a cat"
    assert body["n"] == 1
    assert body["response_format"] == "b64_json"
    assert body["aspect_ratio"] == "1:1"
    # The schema has no seed parameter: it is dropped, never smuggled.
    assert "seed" not in body


@pytest.mark.asyncio
async def test_a_grok_sign_in_labels_the_same_figure_subscription_api_equivalent() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_ok_response()))

    result = await _run(recorder, credential_kind="oauth", client=http)

    # Same reported amount and source as the key path; only the money meaning
    # differs (a subscription draws allotment, it is not billed).
    assert result.cost_usd == pytest.approx(4.0)
    assert result.cost_source == "reported"
    assert result.billing_basis == "subscription-api-equivalent"
    assert result.cost_provenance is not None and "not billed" in result.cost_provenance


@pytest.mark.asyncio
async def test_a_grok_sign_in_without_usage_has_no_figure_and_no_basis() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_ok_response(ticks=None)))

    result = await _run(recorder, credential_kind="oauth", client=http)

    assert result.cost_usd is None
    assert result.billing_basis is None


@pytest.mark.asyncio
async def test_n_passes_through_and_every_item_is_collected() -> None:
    recorder = _Recorder()
    http = _client(
        recorder.handler(
            _ok_response(
                ticks=50_000_000_000,
                items=[
                    {"b64_json": PNG_B64, "mime_type": "image/png"},
                    {"b64_json": PNG_B64, "mime_type": "image/webp"},
                    {"b64_json": PNG_B64, "mime_type": "image/png"},
                ],
            )
        )
    )

    result = await _run(recorder, num_images=3, image_size="landscape_16_9", client=http)

    assert recorder.bodies[0]["n"] == 3
    assert recorder.bodies[0]["aspect_ratio"] == "16:9"
    assert len(result.assets) == 3
    assert result.assets[1].content_type == "image/webp"


@pytest.mark.asyncio
async def test_a_missing_usage_leaves_the_cost_unset() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_ok_response(ticks=None)))

    result = await _run(recorder, client=http)

    assert result.cost_usd is None
    assert result.cost_source is None
    assert result.billing_basis is None
    assert result.cost_provenance is None


@pytest.mark.asyncio
async def test_a_url_item_falls_back_to_the_bounded_downloader() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _ok_response(items=[{"url": "https://img.xai.test/one.png"}])
        assert request.method == "GET"
        assert str(request.url) == "https://img.xai.test/one.png"
        return httpx.Response(
            200,
            content=PNG_1X1,
            headers={"content-type": "image/png"},
        )

    result = await _run(_Recorder(), client=_client(handler))

    assert len(result.assets) == 1
    assert result.assets[0].data == PNG_1X1
    assert result.assets[0].source_url == "https://img.xai.test/one.png"


@pytest.mark.asyncio
async def test_an_edit_uses_the_single_image_shape_and_keeps_reported_ticks() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_ok_response()))

    result = await _run(recorder, source_url=f"data:image/png;base64,{PNG_B64}", client=http)

    assert recorder.requests[0].url.path == "/v1/images/edits"
    body = recorder.bodies[0]
    assert body["image"] == {"url": f"data:image/png;base64,{PNG_B64}", "type": "image_url"}
    assert "n" not in body, "n is a generations-only parameter on this endpoint"
    assert body["response_format"] == "b64_json"
    # Edits bill input AND output (pricing page); the REPORTED ticks stay the
    # only figure, and the generations flat rate is never reused.
    assert result.cost_usd == pytest.approx(4.0)
    assert result.cost_source == "reported"


@pytest.mark.asyncio
async def test_an_http_failure_maps_to_the_lane_error_shape() -> None:
    recorder = _Recorder()
    response = httpx.Response(403, json={"error": {"message": "OAuth not allowlisted"}})
    http = _client(recorder.handler(response))

    with pytest.raises(APIError) as caught:
        await _run(recorder, client=http)

    assert caught.value.status_code == 403
    assert "xk-1" not in (caught.value.body or "")


@pytest.mark.asyncio
async def test_a_response_without_image_data_is_invalid_response() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(httpx.Response(200, json={"data": []})))

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
    http = _client(recorder.handler(_ok_response()))
    with pytest.raises(image_cascade.ImageGenerationCancelled):
        await _run(recorder, pause=pause, client=http)

    assert calls == [0.0]
    assert recorder.requests == []


def test_the_spec_declares_reported_cost_and_no_cancel() -> None:
    from local_operator.artifacts.rung import CancelSupport, SourceSupport
    from local_operator.imagegen import ImageRoute, cascade

    spec = cascade.RUNG_SPECS[ImageRoute.XAI]
    assert spec.label == "xAI"
    assert spec.cancel_support == CancelSupport.NONE
    # Amended from the wave's initial rate_table pencil: the official schema
    # requires usage.cost_in_usd_ticks, so the figure is REPORTED (design D8).
    assert spec.cost == "reported"
    assert spec.capabilities == frozenset({"t2i", "i2i"}), "the edit path is wired"
    assert spec.sources == SourceSupport.MULTI
    assert spec.max_sources == 5
